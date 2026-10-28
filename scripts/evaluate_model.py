import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys

import joblib

from utils.constants import (
    CANDIDATE_MANIFEST_PATH,
    CANDIDATE_MODEL_PATH,
    MODEL_MINIMUM_R2,
    PRODUCTION_MODEL_PATH,
    TRAINING_DATA_PATH,
)
from utils.model_retraining import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    calculate_metrics,
    dependency_versions,
    evaluate_quality_gate,
    load_labeled_dataset,
    sha256_file,
    split_labeled_dataset,
    validate_model_contract,
    validate_training_runtime,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a candidate against the production baseline."
    )
    parser.add_argument("--data", default=TRAINING_DATA_PATH)
    parser.add_argument("--candidate", default=CANDIDATE_MODEL_PATH)
    parser.add_argument("--manifest", default=CANDIDATE_MANIFEST_PATH)
    parser.add_argument("--baseline", default=PRODUCTION_MODEL_PATH)
    parser.add_argument(
        "--report-path",
        help="Optional path to save the JSON evaluation report.",
    )
    return parser.parse_args()


def validate_candidate_manifest(
    manifest_path,
    candidate_path,
    baseline_path,
    data_path,
    data_hash,
    split,
    training_rows,
    evaluation_rows,
):
    manifest = json.loads(
        Path(manifest_path).read_text(encoding="utf-8")
    )

    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported candidate manifest schema version.")

    if Path(manifest["candidate_artifact"]).resolve() != Path(candidate_path).resolve():
        raise ValueError("Manifest candidate path does not match the artifact.")

    if manifest.get("candidate_sha256") != sha256_file(candidate_path):
        raise ValueError("Candidate artifact SHA-256 does not match its manifest.")

    training_data = manifest.get("training_data", {})
    if Path(training_data.get("path", "")).resolve() != Path(data_path).resolve():
        raise ValueError("Manifest training data path does not match.")
    if training_data.get("sha256") != data_hash:
        raise ValueError("Training data SHA-256 does not match the manifest.")

    if manifest.get("feature_columns") != list(FEATURE_COLUMNS):
        raise ValueError("Manifest feature schema does not match.")
    if manifest.get("target_column") != TARGET_COLUMN:
        raise ValueError("Manifest target column does not match.")

    model_metadata = manifest.get("model", {})
    if model_metadata.get("pipeline") != [
        "sklearn.preprocessing.StandardScaler",
        "sklearn.ensemble.GradientBoostingRegressor",
    ]:
        raise ValueError("Manifest model pipeline does not match the supported model.")

    manifest_split = manifest.get("split", {})
    if manifest_split.get("evaluation_identity") != split["evaluation_identity"]:
        raise ValueError("Manifest evaluation split does not match.")
    if manifest_split.get("evaluation_row_indices") != split[
        "evaluation_row_indices"
    ]:
        raise ValueError("Manifest evaluation rows do not match.")
    if manifest_split.get("training_row_indices") != split[
        "training_row_indices"
    ]:
        raise ValueError("Manifest training rows do not match.")

    row_counts = manifest.get("row_counts", {})
    if row_counts.get("training") != training_rows:
        raise ValueError("Manifest training row count does not match.")
    if row_counts.get("evaluation") != evaluation_rows:
        raise ValueError("Manifest evaluation row count does not match.")
    if training_data.get("row_count") != training_rows + evaluation_rows:
        raise ValueError("Manifest dataset row count does not match.")

    baseline_metadata = manifest.get("evaluation", {}).get("baseline", {})
    if baseline_metadata.get("artifact_sha256") != sha256_file(baseline_path):
        raise ValueError(
            "Production baseline artifact differs from the one used for training."
        )

    candidate_versions = manifest.get("dependency_versions", {})
    current_versions = dependency_versions()
    if candidate_versions.get("scikit-learn") != current_versions["scikit-learn"]:
        raise ValueError(
            "Candidate scikit-learn version does not match the evaluation "
            "runtime."
        )
    if candidate_versions.get("python", "").split(".")[:2] != (
        current_versions["python"].split(".")[:2]
    ):
        raise ValueError(
            "Candidate Python major/minor version does not match the runtime."
        )

    for package in ("numpy", "pandas", "joblib"):
        if candidate_versions.get(package) != current_versions[package]:
            raise ValueError(
                f"Candidate {package} version does not match the evaluation runtime."
            )

    return manifest


def evaluate(args):
    validate_training_runtime(
        Path(__file__).resolve().parents[1] / "requirements.txt"
    )

    features, target, data_hash = load_labeled_dataset(args.data)
    (
        train_features,
        evaluation_features,
        _,
        evaluation_target,
        split,
    ) = split_labeled_dataset(features, target)

    manifest = validate_candidate_manifest(
        args.manifest,
        args.candidate,
        args.baseline,
        args.data,
        data_hash,
        split,
        len(train_features),
        len(evaluation_features),
    )

    candidate = joblib.load(args.candidate)
    baseline = joblib.load(args.baseline)
    validate_model_contract(candidate, "Candidate model")
    validate_model_contract(baseline, "Production baseline")
    candidate_metrics = calculate_metrics(
        candidate,
        evaluation_features,
        evaluation_target,
    )
    baseline_metrics = calculate_metrics(
        baseline,
        evaluation_features,
        evaluation_target,
    )
    quality_gate = evaluate_quality_gate(candidate_metrics)

    return {
        "schema_version": 1,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data": {
            "path": str(Path(args.data)),
            "sha256": data_hash,
            "training_rows": len(train_features),
            "evaluation_rows": len(evaluation_features),
            "evaluation_identity": split["evaluation_identity"],
        },
        "candidate": {
            "artifact_path": manifest["candidate_artifact"],
            "sha256": sha256_file(args.candidate),
            "metrics": candidate_metrics,
        },
        "production_baseline": {
            "artifact_path": str(Path(args.baseline)),
            "sha256": sha256_file(args.baseline),
            "metrics": baseline_metrics,
        },
        "comparison": {
            "rmse_delta_candidate_minus_baseline": (
                candidate_metrics["rmse"] - baseline_metrics["rmse"]
            ),
            "mae_delta_candidate_minus_baseline": (
                candidate_metrics["mae"] - baseline_metrics["mae"]
            ),
            "r2_delta_candidate_minus_baseline": (
                candidate_metrics["r2"] - baseline_metrics["r2"]
            ),
            "rmse_mae_limits_configured": False,
            "human_review_required": True,
        },
        "quality_gate": quality_gate,
    }


def main():
    args = parse_args()
    try:
        report = evaluate(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        failure = {
            "quality_gate": {
                "status": "FAIL",
                "reason": str(exc),
                "r2_minimum_exclusive": MODEL_MINIMUM_R2,
            }
        }
        if args.report_path:
            Path(args.report_path).write_text(
                json.dumps(failure, indent=2) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1

    if args.report_path:
        Path(args.report_path).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["quality_gate"]["r2_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
