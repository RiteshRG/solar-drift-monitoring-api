from google.cloud import storage
from datetime import datetime
import json
import uuid

from utils.constants import BUCKET_NAME, GCP_PROJECT_ID


def log_prediction(
    irradiation,
    ambient_temperature,
    hour_sin,
    hour_cos,
    prediction
):
    client = storage.Client(
        project=GCP_PROJECT_ID
    )

    bucket = client.bucket(BUCKET_NAME)

    timestamp = datetime.now().isoformat()
    unique_id = uuid.uuid4().hex

    file_name = f"production/prediction_{unique_id}.json"

    prediction_data = {
        "timestamp": timestamp,
        "IRRADIATION": irradiation,
        "AMBIENT_TEMPERATURE": ambient_temperature,
        "HOUR_SIN": hour_sin,
        "HOUR_COS": hour_cos,
        "predicted_AC_POWER": prediction
    }

    blob = bucket.blob(file_name)

    blob.upload_from_string(
        json.dumps(prediction_data),
        content_type="application/json"
    )