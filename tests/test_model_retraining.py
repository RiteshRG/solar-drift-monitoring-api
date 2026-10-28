import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from utils.model_retraining import (
    EXPECTED_COLUMNS,
    evaluate_quality_gate,
    load_labeled_dataset,
    repository_relative_path,
    sha256_text_file_normalized,
    split_labeled_dataset,
    calculate_metrics,
    build_model,
    validate_model_contract,
)


class ModelRetrainingTests(unittest.TestCase):
    def _write_dataset(self, directory, data):
        path = Path(directory) / "train.csv"
        data.to_csv(path, index=False)
        return path

    def _valid_data(self):
        rows = 20
        return pd.DataFrame(
            {
                "AC_POWER": np.arange(rows, dtype=float) * 10,
                "IRRADIATION": np.linspace(0.1, 1.0, rows),
                "AMBIENT_TEMPERATURE": np.linspace(15, 35, rows),
                "HOUR_SIN": np.sin(np.arange(rows)),
                "HOUR_COS": np.cos(np.arange(rows)),
            },
            columns=EXPECTED_COLUMNS,
        )

    def test_loads_valid_dataset_and_creates_reproducible_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_dataset(directory, self._valid_data())
            features, target, data_hash = load_labeled_dataset(path)

            first_split = split_labeled_dataset(features, target)
            second_split = split_labeled_dataset(features, target)

        self.assertEqual(len(features.columns), 4)
        self.assertEqual(len(target), 20)
        self.assertEqual(len(data_hash), 64)
        self.assertEqual(
            first_split[4]["evaluation_identity"],
            second_split[4]["evaluation_identity"],
        )
        self.assertEqual(
            first_split[4]["evaluation_row_indices"],
            second_split[4]["evaluation_row_indices"],
        )

    def test_rejects_unexpected_column_order(self):
        data = self._valid_data().loc[
            :, ["IRRADIATION", "AC_POWER", *EXPECTED_COLUMNS[2:]]
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_dataset(directory, data)
            with self.assertRaisesRegex(ValueError, "exact order"):
                load_labeled_dataset(path)

    def test_rejects_missing_and_non_finite_values(self):
        for invalid_value in (np.nan, np.inf):
            with self.subTest(value=invalid_value):
                data = self._valid_data()
                data.loc[0, "IRRADIATION"] = invalid_value
                with tempfile.TemporaryDirectory() as directory:
                    path = self._write_dataset(directory, data)
                    with self.assertRaisesRegex(ValueError, "non-finite|missing"):
                        load_labeled_dataset(path)

    def test_r2_gate_is_strictly_greater_than_point_eight(self):
        self.assertFalse(evaluate_quality_gate({"r2": 0.80})["r2_passed"])
        self.assertTrue(evaluate_quality_gate({"r2": 0.800001})["r2_passed"])
        self.assertFalse(evaluate_quality_gate({"r2": float("nan")})["r2_passed"])

    def test_manifest_paths_use_repository_relative_posix_format(self):
        self.assertEqual(
            repository_relative_path("models/candidate_model_pipeline.joblib"),
            "models/candidate_model_pipeline.joblib",
        )
        with self.assertRaisesRegex(ValueError, "inside the repository"):
            repository_relative_path("../outside.csv")

    def test_training_data_hash_ignores_windows_newlines(self):
        with tempfile.TemporaryDirectory() as directory:
            lf_path = Path(directory) / "lf.csv"
            crlf_path = Path(directory) / "crlf.csv"
            lf_path.write_bytes(b"AC_POWER,IRRADIATION\n1,0.5\n")
            crlf_path.write_bytes(b"AC_POWER,IRRADIATION\r\n1,0.5\r\n")

            self.assertEqual(
                sha256_text_file_normalized(lf_path),
                sha256_text_file_normalized(crlf_path),
            )

    def test_metrics_reject_wrong_prediction_shape(self):
        model = DummyRegressor(strategy="mean").fit(
            [[0], [1], [2]],
            [0, 1, 2],
        )
        with self.assertRaisesRegex(ValueError, "shape"):
            calculate_metrics(model, [[0], [1]], [0])

    def test_pipeline_preserves_serving_feature_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_dataset(directory, self._valid_data())
            features, target, _ = load_labeled_dataset(path)
            model = build_model().fit(features, target)

        validate_model_contract(model, "Candidate model")


if __name__ == "__main__":
    unittest.main()
