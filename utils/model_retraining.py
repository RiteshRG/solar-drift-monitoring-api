import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.constants import (
    MODEL_EVALUATION_TEST_SIZE,
    MODEL_MINIMUM_R2,
    MODEL_RANDOM_STATE,
)


FEATURE_COLUMNS = (
    "IRRADIATION",
    "AMBIENT_TEMPERATURE",
    "HOUR_SIN",
    "HOUR_COS",
)
TARGET_COLUMN = "AC_POWER"
EXPECTED_COLUMNS = (TARGET_COLUMN, *FEATURE_COLUMNS)
MINIMUM_DATASET_ROWS = 10
SUPPORTED_PYTHON_VERSION = (3, 11)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value):
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def load_labeled_dataset(path):
    path = Path(path)
    dataset_hash = sha256_file(path)
    data = pd.read_csv(path)

    if tuple(data.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "Training CSV must have columns in this exact order: "
            + ", ".join(EXPECTED_COLUMNS)
        )

    if len(data) < MINIMUM_DATASET_ROWS:
        raise ValueError(
            f"Training CSV requires at least {MINIMUM_DATASET_ROWS} rows; "
            f"found {len(data)}."
        )

    numeric = data.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("Training CSV contains missing or non-numeric values.")

    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Training CSV contains non-finite values.")

    features = numeric.loc[:, FEATURE_COLUMNS].copy()
    target = numeric.loc[:, TARGET_COLUMN].copy()
    return features, target, dataset_hash


def split_labeled_dataset(features, target):
    if len(features) != len(target):
        raise ValueError("Feature and target row counts do not match.")

    train_features, evaluation_features, train_target, evaluation_target = (
        train_test_split(
            features,
            target,
            test_size=MODEL_EVALUATION_TEST_SIZE,
            random_state=MODEL_RANDOM_STATE,
            shuffle=True,
        )
    )

    if len(train_features) < 2 or len(evaluation_features) < 2:
        raise ValueError(
            "The reproducible split must contain at least two training and "
            "two evaluation rows."
        )

    split = {
        "method": "sklearn.model_selection.train_test_split",
        "test_size": MODEL_EVALUATION_TEST_SIZE,
        "random_state": MODEL_RANDOM_STATE,
        "shuffle": True,
        "training_row_indices": [int(index) for index in train_features.index],
        "evaluation_row_indices": [
            int(index) for index in evaluation_features.index
        ],
    }
    split["evaluation_identity"] = sha256_json(
        {
            "training_row_indices": split["training_row_indices"],
            "evaluation_row_indices": split["evaluation_row_indices"],
        }
    )

    return (
        train_features,
        evaluation_features,
        train_target,
        evaluation_target,
        split,
    )


def build_model():
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                GradientBoostingRegressor(
                    random_state=MODEL_RANDOM_STATE
                ),
            ),
        ]
    )


def validate_model_contract(model, name):
    if not callable(getattr(model, "predict", None)):
        raise ValueError(f"{name} does not expose a callable predict method.")

    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None or list(feature_names) != list(FEATURE_COLUMNS):
        raise ValueError(
            f"{name} feature schema must be exactly "
            f"{list(FEATURE_COLUMNS)} in that order."
        )

    feature_count = getattr(model, "n_features_in_", None)
    if feature_count != len(FEATURE_COLUMNS):
        raise ValueError(
            f"{name} must accept exactly {len(FEATURE_COLUMNS)} input features."
        )


def calculate_metrics(model, features, target):
    predictions = np.asarray(model.predict(features))
    expected = np.asarray(target)

    if predictions.shape != expected.shape:
        raise ValueError(
            "Model predictions do not match the evaluation target shape."
        )

    if not np.isfinite(predictions).all():
        raise ValueError("Model produced non-finite predictions.")

    if not np.isfinite(expected).all():
        raise ValueError("Evaluation target contains non-finite values.")

    metrics = {
        "r2": float(r2_score(expected, predictions)),
        "rmse": float(math.sqrt(mean_squared_error(expected, predictions))),
        "mae": float(mean_absolute_error(expected, predictions)),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Model evaluation produced non-finite metrics.")
    return metrics


def evaluate_quality_gate(metrics):
    r2 = metrics.get("r2")
    passed = (
        isinstance(r2, (float, int))
        and math.isfinite(r2)
        and r2 > MODEL_MINIMUM_R2
    )
    return {
        "r2_minimum_exclusive": MODEL_MINIMUM_R2,
        "r2_passed": passed,
        "rmse_mae_limits_configured": False,
        "status": (
            "PASS_R2_REQUIRES_HUMAN_REVIEW"
            if passed
            else "FAIL"
        ),
    }


def validate_training_runtime(requirements_path):
    if sys.version_info[:2] != SUPPORTED_PYTHON_VERSION:
        raise RuntimeError(
            "Model training must use Python "
            f"{SUPPORTED_PYTHON_VERSION[0]}.{SUPPORTED_PYTHON_VERSION[1]} "
            "to match the production container. Create an environment from "
            "requirements.txt and rerun the command."
        )

    requirement_text = Path(requirements_path).read_text(encoding="utf-8")
    match = re.search(
        r"(?m)^scikit-learn==([^\s;]+)\s*$",
        requirement_text,
    )
    if match is None:
        raise RuntimeError(
            "requirements.txt must pin scikit-learn to an exact version."
        )

    expected_version = match.group(1)
    actual_version = importlib.metadata.version("scikit-learn")
    if actual_version != expected_version:
        raise RuntimeError(
            "Installed scikit-learn version "
            f"{actual_version} does not match the production pin "
            f"{expected_version}. Install requirements.txt in the active "
            "Python 3.11 environment and rerun the command."
        )


def get_git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None

    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


def dependency_versions():
    packages = (
        ("scikit-learn", "scikit-learn"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("joblib", "joblib"),
    )
    versions = {
        package: importlib.metadata.version(distribution)
        for package, distribution in packages
    }
    versions["python"] = ".".join(str(part) for part in sys.version_info[:3])
    return versions
