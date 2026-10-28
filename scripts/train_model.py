import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

import joblib

from utils.constants import (
    CANDIDATE_MANIFEST_PATH,
    CANDIDATE_MODEL_PATH,
    PRODUCTION_MODEL_PATH,
    TRAINING_DATA_PATH,
)
from utils.model_retraining import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_model,
    calculate_metrics,
    dependency_versions,
    evaluate_quality_gate,
    get_git_commit,
    load_labeled_dataset,
    sha256_file,
    split_labeled_dataset,
    validate_model_contract,
    validate_training_runtime,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a local candidate model; production is never changed."
    )
    parser.add_argument("--data", default=TRAINING_DATA_PATH)
    parser.add_argument("--candidate", default=CANDIDATE_MODEL_PATH)
    parser.add_argument("--manifest", default=CANDIDATE_MANIFEST_PATH)
    return parser.parse_args()


def _atomic_save_model(model, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            suffix=".joblib",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)

        joblib.dump(model, temp_path, compress=3)
        os.replace(temp_path, destination)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _atomic_write_json(value, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            suffix=".json",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(value, temp_file, indent=2, sort_keys=True)
            temp_file.write("\n")

        os.replace(temp_path, destination)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def main():
    args = parse_args()
    candidate_path = Path(args.candidate).resolve()
    production_path = Path(PRODUCTION_MODEL_PATH).resolve()
    if candidate_path == production_path:
        raise ValueError(
            "Candidate output path must not overwrite the production model."
        )

    manifest_path = Path(args.manifest).resolve()
    if manifest_path in (candidate_path, production_path):
        raise ValueError("Candidate manifest must have its own output path.")

    validate_training_runtime(
        Path(__file__).resolve().parents[1] / "requirements.txt"
    )

    features, target, data_hash = load_labeled_dataset(args.data)
    (
        train_features,
        evaluation_features,
        train_target,
        evaluation_target,
        split,
    ) = split_labeled_dataset(features, target)

    model = build_model()
    model.fit(train_features, train_target)
    validate_model_contract(model, "Candidate model")
    candidate_metrics = calculate_metrics(
        model,
        evaluation_features,
        evaluation_target,
    )

    baseline = joblib.load(PRODUCTION_MODEL_PATH)
    validate_model_contract(baseline, "Production baseline")
    baseline_metrics = calculate_metrics(
        baseline,
        evaluation_features,
        evaluation_target,
    )

    _atomic_save_model(model, candidate_path)
    candidate_hash = sha256_file(candidate_path)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_artifact": str(Path(args.candidate)),
        "candidate_sha256": candidate_hash,
        "training_data": {
            "path": str(Path(args.data)),
            "sha256": data_hash,
            "row_count": len(features),
        },
        "training_code_commit": get_git_commit(),
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "model": {
            "pipeline": [
                "sklearn.preprocessing.StandardScaler",
                "sklearn.ensemble.GradientBoostingRegressor",
            ],
            "hyperparameters": model.named_steps["model"].get_params(
                deep=True
            ),
        },
        "split": split,
        "row_counts": {
            "training": len(train_features),
            "evaluation": len(evaluation_features),
        },
        "dependency_versions": dependency_versions(),
        "evaluation": {
            "baseline": {
                "artifact_path": PRODUCTION_MODEL_PATH,
                "artifact_sha256": sha256_file(PRODUCTION_MODEL_PATH),
                "metrics": baseline_metrics,
            },
            "candidate_metrics": candidate_metrics,
            "quality_gate": evaluate_quality_gate(candidate_metrics),
            "rmse_mae_limits_configured": False,
            "human_review_required": True,
        },
    }

    _atomic_write_json(manifest, manifest_path)
    print(
        json.dumps(
            {
                "candidate_artifact": str(candidate_path),
                "candidate_manifest": str(manifest_path),
                "candidate_sha256": candidate_hash,
                "candidate_metrics": candidate_metrics,
                "baseline_metrics": baseline_metrics,
                "quality_gate": manifest["evaluation"]["quality_gate"],
                "human_review_required": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
