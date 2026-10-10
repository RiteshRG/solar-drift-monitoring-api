import pandas as pd
from scipy.stats import ks_2samp
from datetime import datetime
from google.cloud import storage
from io import StringIO

from utils import load_data

import os

os.makedirs("reports", exist_ok=True)

BUCKET_NAME = "solar-monitoring-511206"
REPORT_FILE = "reports/drift_report.csv"



MINIMUM_CURRENT_RECORDS = 50


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

    monitoring_time = datetime.now().isoformat()

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


def save_report(report):

    if report is None:
        return

    client = storage.Client(
        project="solar-power-monitoring-511206"
    )

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(REPORT_FILE)

    # -----------------------------------
    # Check if report already exists
    # -----------------------------------

    if blob.exists():

        existing_data = blob.download_as_text()

        existing_report = pd.read_csv(
            StringIO(existing_data)
        )

        combined_report = pd.concat(
            [existing_report, report],
            ignore_index=True
        )

    else:

        # First monitoring run
        combined_report = report

    # -----------------------------------
    # Convert DataFrame to CSV
    # -----------------------------------

    csv_data = combined_report.to_csv(
        index=False
    )

    # -----------------------------------
    # Upload to GCS
    # -----------------------------------

    blob.upload_from_string(
        csv_data,
        content_type="text/csv"
    )

    print(
        f"Drift report saved to gs://"
        f"{BUCKET_NAME}/{REPORT_FILE}"
    )


def get_latest_status():

    client = storage.Client(
        project="solar-power-monitoring-511206"
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

    REFERENCE_FILE = "data/train.csv"
    REPORT_FILE = "reports/drift_report.csv"

    reference, current = load_data(
        REFERENCE_FILE
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