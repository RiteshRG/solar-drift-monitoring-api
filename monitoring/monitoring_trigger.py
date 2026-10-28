from datetime import datetime, timedelta, timezone
import json
import logging

from google.api_core.exceptions import NotFound, PreconditionFailed
from google.cloud import storage

from monitoring.drift_monitor import run_drift_monitoring, save_report
from utils import load_data
from utils.constants import (
    BUCKET_NAME,
    GCP_PROJECT_ID,
    MINIMUM_CURRENT_RECORDS,
    MONITORING_LOCK_FILE,
    STATE_FILE,
    TRAINING_DATA_PATH,
    TRIGGER_SIZE,
)


LOCK_STALE_AFTER = timedelta(minutes=30)


def _bucket():
    client = storage.Client(project=GCP_PROJECT_ID)
    return client.bucket(BUCKET_NAME)


def get_production_count():
    return sum(
        1
        for blob in _bucket().list_blobs(prefix="production/")
        if blob.name.endswith(".json")
    )


def get_last_monitored_count():
    blob = _bucket().blob(STATE_FILE)

    if not blob.exists():
        return 0

    data = json.loads(blob.download_as_text())
    count = data.get("last_monitored_count", 0)
    if not isinstance(count, int) or count < 0:
        raise ValueError("Monitoring state has an invalid record count.")
    return count


def save_last_monitored_count(count):
    blob = _bucket().blob(STATE_FILE)
    blob.upload_from_string(
        json.dumps({"last_monitored_count": count}),
        content_type="application/json",
    )


def _acquire_lock():
    lock = _bucket().blob(MONITORING_LOCK_FILE)
    payload = json.dumps({"created_at": datetime.now(timezone.utc).isoformat()})

    try:
        lock.upload_from_string(
            payload,
            content_type="application/json",
            if_generation_match=0,
        )
    except PreconditionFailed:
        lock.reload()
        lock_time = lock.updated
        if lock_time is None or (
            datetime.now(timezone.utc) - lock_time
        ) <= LOCK_STALE_AFTER:
            return None

        try:
            lock.delete(if_generation_match=lock.generation)
        except PreconditionFailed:
            return None

        try:
            lock.upload_from_string(
                payload,
                content_type="application/json",
                if_generation_match=0,
            )
        except PreconditionFailed:
            return None

    lock.reload()
    return lock


def _release_lock(lock):
    try:
        lock.delete(if_generation_match=lock.generation)
    except NotFound:
        logging.warning("Monitoring lock was already removed.")
    except PreconditionFailed:
        logging.warning("Monitoring lock ownership changed before release.")


def run_scheduled_monitoring():
    lock = _acquire_lock()
    if lock is None:
        return {"status": "MONITORING_ALREADY_RUNNING"}

    try:
        production_count = get_production_count()
        last_monitored_count = get_last_monitored_count()
        new_records = production_count - last_monitored_count

        if new_records < TRIGGER_SIZE:
            return {
                "status": "NOT_DUE",
                "production_records": production_count,
                "new_records": new_records,
                "required_new_records": TRIGGER_SIZE,
            }

        reference, current = load_data(TRAINING_DATA_PATH)
        if len(current) < MINIMUM_CURRENT_RECORDS:
            return {
                "status": "NOT_ENOUGH_DATA",
                "current_records": len(current),
                "required_records": MINIMUM_CURRENT_RECORDS,
            }

        run_id = f"records-{last_monitored_count}-{production_count}"
        report = run_drift_monitoring(reference, current)
        if report is None:
            return {
                "status": "NOT_ENOUGH_DATA",
                "current_records": len(current),
                "required_records": MINIMUM_CURRENT_RECORDS,
            }

        results = report.to_dict(orient="records")
        for result in results:
            result["monitoring_run_id"] = run_id

        save_report(report, run_id=run_id)
        save_last_monitored_count(production_count)

        drift_detected = any(
            bool(result["drift_detected"])
            for result in results
        )

        return {
            "status": "DRIFT DETECTED" if drift_detected else "NO DRIFT",
            "run_id": run_id,
            "production_records": production_count,
            "new_records": new_records,
            "results": results,
        }
    finally:
        _release_lock(lock)


def should_run_monitoring():
    production_count = get_production_count()
    last_monitored_count = get_last_monitored_count()
    return (
        production_count - last_monitored_count >= TRIGGER_SIZE,
        production_count,
    )


if __name__ == "__main__":
    should_run, production_count = should_run_monitoring()
    if should_run:
        print(run_scheduled_monitoring())
    else:
        print(
            "Not enough new records. "
            f"Production records: {production_count}"
        )
