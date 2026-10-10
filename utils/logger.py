from google.cloud import storage
import csv
import io
from datetime import datetime

BUCKET_NAME = "YOUR_BUCKET_NAME"
FILE_NAME = "production/production_predictions.csv"

def log_prediction(
    irradiation,
    ambient_temperature,
    hour_sin,
    hour_cos,
    prediction
):
    client = storage.Client()

    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(FILE_NAME)

    # Read existing file if it exists
    if blob.exists():
        existing_data = blob.download_as_text()
    else:
        existing_data = ""

    output = io.StringIO()

    writer = csv.writer(output)

    if not existing_data:
        writer.writerow([
            "timestamp",
            "IRRADIATION",
            "AMBIENT_TEMPERATURE",
            "HOUR_SIN",
            "HOUR_COS",
            "predicted_AC_POWER"
        ])

    writer.writerow([
        datetime.now().isoformat(),
        irradiation,
        ambient_temperature,
        hour_sin,
        hour_cos,
        prediction
    ])

    blob.upload_from_string(
        existing_data + output.getvalue(),
        content_type="text/csv"
    )