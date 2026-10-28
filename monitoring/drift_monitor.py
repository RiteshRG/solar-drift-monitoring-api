import pandas as pd
from scipy.stats import ks_2samp
from datetime import datetime, timezone
from google.api_core.exceptions import PreconditionFailed
from google.cloud import storage
from io import StringIO

from utils import load_data
from utils.constants import (
    BUCKET_NAME,
    GCP_PROJECT_ID,
    MINIMUM_CURRENT_RECORDS,
    REPORT_FILE,
    RUN_REPORT_PREFIX,
    TRAINING_DATA_PATH,
)

import os

os.makedirs("reports", exist_ok=True)


def check_drift(reference, current, feature, alpha=0.05):

    statistic, p_value = ks_2samp(
        reference[feature],
        current[feature]
    )

    drift_detected = p_value < alpha

    return {
        "feature": feature,
        "test": "KS Test",
        "ks_statistic": statistic,
        "p_value": p_value,
        "alpha": alpha,
        "drift_detected": drift_detected
    }


def run_drift_monitoring(reference, current):

    # Not enough production data
    if len(current) < MINIMUM_CURRENT_RECORDS:

        print(
            f"Not enough production data for drift monitoring. "
            f"Current records: {len(current)}, "
            f"Required: {MINIMUM_CURRENT_RECORDS}"
        )

        return None

    monitoring_time = datetime.now(timezone.utc).isoformat()

    results = []

    for feature in [
        "IRRADIATION",
        "AMBIENT_TEMPERATURE"
    ]:

        result = check_drift(
            reference,
            current,
            feature
        )

        result["monitoring_time"] = monitoring_time

        results.append(result)

    report = pd.DataFrame(results)

    return report


def save_report(report, run_id=None):

    if report is None:
        return False

    report_to_save = report.copy()
    if run_id is not None:
        report_to_save["monitoring_run_id"] = run_id

    client = storage.Client(
        project=GCP_PROJECT_ID
    )

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(REPORT_FILE)

    # -----------------------------------
    run_report = report_to_save
    run_blob = None

    if run_id is not None:
        run_blob = bucket.blob(f"{RUN_REPORT_PREFIX}{run_id}.csv")

        if run_blob.exists():
            run_report = pd.read_csv(
                StringIO(run_blob.download_as_text())
            )
        else:
            run_blob.upload_from_string(
                report_to_save.to_csv(index=False),
                content_type="text/csv",
                if_generation_match=0,
            )

    for _ in range(5):
        if blob.exists():
            blob.reload()
            generation = blob.generation
            existing_data = blob.download_as_text()
            existing_report = pd.read_csv(StringIO(existing_data))

            if run_id is not None and (
                "monitoring_run_id" in existing_report.columns
                and run_id in existing_report["monitoring_run_id"].values
            ):
                return True

            combined_report = pd.concat(
                [existing_report, run_report],
                ignore_index=True
            )
        else:
            generation = 0
            combined_report = run_report

        try:
            blob.upload_from_string(
                combined_report.to_csv(index=False),
                content_type="text/csv",
                if_generation_match=generation,
            )
            break
        except PreconditionFailed:
            continue
    else:
        raise RuntimeError(
            "Could not append drift results after repeated report updates."
        )

    print(
        f"Drift report saved to gs://"
        f"{BUCKET_NAME}/{REPORT_FILE}"
    )
    return True


def get_latest_status():

    client = storage.Client(
        project=GCP_PROJECT_ID
    )

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(REPORT_FILE)

    # Report does not exist
    if not blob.exists():
        return "NO REPORT"

    # Read report from GCS
    report_data = blob.download_as_text()

    report = pd.read_csv(
        StringIO(report_data)
    )

    # Get latest monitoring timestamp
    latest_time = report[
        "monitoring_time"
    ].max()

    latest_report = report[
        report["monitoring_time"] == latest_time
    ]

    # Check whether any feature has drift
    overall_drift = latest_report[
        "drift_detected"
    ].any()

    if overall_drift:
        return "DRIFT DETECTED"

    return "NO DRIFT"


if __name__ == "__main__":

    reference, current = load_data(
        TRAINING_DATA_PATH
    )

    report = run_drift_monitoring(
        reference,
        current
    )

    if report is None:

        print("Overall Result: NOT ENOUGH DATA")

    else:

        save_report(report)

        status = get_latest_status()

        print(
            "Overall Result:",
            status
        )