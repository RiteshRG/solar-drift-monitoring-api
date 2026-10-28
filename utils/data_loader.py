import pandas as pd
import json

from google.cloud import storage
from utils.constants import (
    BUCKET_NAME,
    GCP_PROJECT_ID,
    MINIMUM_CURRENT_RECORDS,
)


PRODUCTION_PREFIX = "production/"


def load_data(reference_file):

    # -----------------------------------
    # 1. Load reference/training data
    # -----------------------------------

    reference = pd.read_csv(reference_file)

    # -----------------------------------
    # 2. Connect to Google Cloud Storage
    # -----------------------------------

    client = storage.Client(
        project=GCP_PROJECT_ID
    )

    bucket = client.bucket(BUCKET_NAME)

    # -----------------------------------
    # 3. Read production JSON files
    # -----------------------------------

    records = []

    blobs = bucket.list_blobs(
        prefix=PRODUCTION_PREFIX
    )

    for blob in blobs:

        if not blob.name.endswith(".json"):
            continue

        data = json.loads(
            blob.download_as_text()
        )

        records.append(data)

    # -----------------------------------
    # 4. Convert production data to DataFrame
    # -----------------------------------

    current = pd.DataFrame(records)

    # -----------------------------------
    # 5. Check minimum production records
    # -----------------------------------

    if len(current) < MINIMUM_CURRENT_RECORDS:

        return reference, current

    # -----------------------------------
    # 6. Convert timestamp to datetime
    # -----------------------------------

    current["timestamp"] = pd.to_datetime(
        current["timestamp"]
    )

    # -----------------------------------
    # 7. Sort newest records first
    # -----------------------------------

    current = current.sort_values(
        "timestamp",
        ascending=False
    )

    # -----------------------------------
    # 8. Dynamic current window
    # -----------------------------------

    current_window_size = min(
        len(reference),
        len(current)
    )

    # -----------------------------------
    # 9. Keep latest records
    # -----------------------------------

    current = current.head(
        current_window_size
    )

    print(
        f"Reference records: {len(reference)}"
    )

    print(
        f"Production records available: "
        f"{len(records)}"
    )

    print(
        f"Current monitoring window: "
        f"{len(current)}"
    )

    return reference, current