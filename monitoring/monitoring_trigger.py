from google.cloud import storage
import json


BUCKET_NAME = "solar-monitoring-511206"

STATE_FILE = "monitoring/monitoring_state.json"

TRIGGER_SIZE = 50


def get_production_count():

    client = storage.Client(
        project="solar-power-monitoring-511206"
    )

    bucket = client.bucket(BUCKET_NAME)

    count = 0

    blobs = bucket.list_blobs(
        prefix="production/"
    )

    for blob in blobs:

        if blob.name.endswith(".json"):
            count += 1

    return count


def get_last_monitored_count():

    client = storage.Client(
        project="solar-power-monitoring-511206"
    )

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(STATE_FILE)

    if not blob.exists():
        return 0

    data = json.loads(
        blob.download_as_text()
    )

    return data.get(
        "last_monitored_count",
        0
    )


def save_last_monitored_count(count):

    client = storage.Client(
        project="solar-power-monitoring-511206"
    )

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(STATE_FILE)

    data = {
        "last_monitored_count": count
    }

    blob.upload_from_string(
        json.dumps(data),
        content_type="application/json"
    )


def should_run_monitoring():

    production_count = get_production_count()

    last_monitored_count = get_last_monitored_count()

    new_records = (
        production_count - last_monitored_count
    )

    print(
        f"Production records: {production_count}"
    )

    print(
        f"Last monitored count: "
        f"{last_monitored_count}"
    )

    print(
        f"New records: {new_records}"
    )

    if new_records >= TRIGGER_SIZE:

        return True, production_count

    return False, production_count


if __name__ == "__main__":

    should_run, production_count = (
        should_run_monitoring()
    )

    if should_run:

        print(
            "50 or more new records found."
        )

        print(
            "Drift monitoring should run."
        )

        save_last_monitored_count(
            production_count
        )

    else:

        print(
            "Not enough new records."
        )

        print(
            "Drift monitoring should not run."
        )