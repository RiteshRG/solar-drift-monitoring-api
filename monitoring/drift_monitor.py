import pandas as pd
from scipy.stats import ks_2samp
from datetime import datetime

from utils import load_data

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

def save_report(report, report_file):
    report.to_csv(
        report_file,
        mode="a",
        header=not os.path.exists(report_file),
        index=False
    )


def get_latest_status(report_file):

    report = pd.read_csv(report_file)

    latest_time = report["monitoring_time"].max()

    latest_report = report[
        report["monitoring_time"] == latest_time
    ]

    overall_drift = latest_report["drift_detected"].any()

    if overall_drift:
        return "DRIFT DETECTED"
    else:
        return "NO DRIFT"

if __name__ == "__main__":

    REFERENCE_FILE = "data/train.csv"
    CURRENT_FILE = "data/validation_drift.csv"
    REPORT_FILE = "reports/drift_report.csv"

    alpha = 0.05

    reference, current = load_data(
        REFERENCE_FILE,
        CURRENT_FILE
    )

    report = run_drift_monitoring(
        reference,
        current
    )

    save_report(report, REPORT_FILE)


    status = get_latest_status(REPORT_FILE)

    print("Overall Result:", status)
