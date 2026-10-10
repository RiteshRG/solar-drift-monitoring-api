from fastapi import FastAPI, HTTPException
import joblib
import logging
from datetime import datetime

import pandas as pd

from utils import create_time_features, log_prediction
from schema import PredictionRequest, PredictionResponse

from utils.data_loader import load_data
from monitoring.drift_monitor import (
    run_drift_monitoring,
    save_report
)


app = FastAPI()


# -----------------------------------
# Load model once when API starts
# -----------------------------------

model = joblib.load("models/model_pipeline.joblib")

MODEL_STATUS_MESSAGE = (
    "model loaded successfully"
    if model
    else "model not loaded"
)

print(MODEL_STATUS_MESSAGE)


# -----------------------------------
# Home
# -----------------------------------

@app.get("/")
def home():

    return {
        "message": "Solar power application api"
    }


# -----------------------------------
# Model status
# -----------------------------------

@app.get("/model-status")
def get_model_status():

    return {
        "status": "healthy" if model else "unavailable",
        "model_loaded": bool(model),
        "message": MODEL_STATUS_MESSAGE,
    }


# -----------------------------------
# Prediction endpoint
# -----------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: PredictionRequest):

    try:

        date_time = datetime.strptime(
            request.DATE_TIME,
            "%d-%m-%Y %H:%M"
        )

    except ValueError:

        raise HTTPException(
            status_code=400,
            detail="DATE_TIME must be in format DD-MM-YYYY HH:MM"
        )

    try:

        # Create time features internally
        hour_sin, hour_cos = create_time_features(
            date_time
        )

        # Create model input
        input_data = pd.DataFrame(
            [[
                request.IRRADIATION,
                request.AMBIENT_TEMPERATURE,
                hour_sin,
                hour_cos
            ]],
            columns=[
                "IRRADIATION",
                "AMBIENT_TEMPERATURE",
                "HOUR_SIN",
                "HOUR_COS"
            ]
        )

        # Make prediction
        prediction = model.predict(
            input_data
        )[0]

        # Store production record in GCS
        log_prediction(
            request.IRRADIATION,
            request.AMBIENT_TEMPERATURE,
            hour_sin,
            hour_cos,
            float(prediction)
        )

        return {
            "predicted_AC_POWER": float(prediction)
        }

    except Exception as exc:

        logging.exception(
            "Prediction endpoint failed"
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        ) from exc


# -----------------------------------
# Drift monitoring endpoint
# -----------------------------------


@app.get("/drift")
def drift_monitor():

    try:

        # Reference = training data
        # Current = production data from GCS
        reference, current = load_data(
            "data/train.csv"
        )

        # Run drift monitoring
        report = run_drift_monitoring(
            reference,
            current
        )

        # Not enough production data
        if report is None:

            return {
                "status": "NOT ENOUGH DATA",
                "current_records": len(current),
                "required_records": 50
            }

        # Save drift report to Google Cloud Storage
        save_report(report)

        # Convert report to JSON
        results = report.to_dict(
            orient="records"
        )

        # Overall drift status
        drift_detected = report[
            "drift_detected"
        ].any()

        status = (
            "DRIFT DETECTED"
            if drift_detected
            else "NO DRIFT"
        )

        return {
            "status": status,
            "current_records": len(current),
            "reference_records": len(reference),
            "results": results
        }

    except Exception as exc:

        logging.exception(
            "Drift monitoring failed"
        )

        raise HTTPException(
            status_code=500,
            detail="Drift monitoring failed"
        ) from exc



@app.get("/drift-report")
def get_drift_report():

    try:

        from google.cloud import storage
        from io import StringIO

        BUCKET_NAME = "solar-monitoring-511206"
        REPORT_FILE = "reports/drift_report.csv"

        client = storage.Client(
            project="solar-power-monitoring-511206"
        )

        bucket = client.bucket(
            BUCKET_NAME
        )

        blob = bucket.blob(
            REPORT_FILE
        )

        # -----------------------------------
        # Check whether report exists
        # -----------------------------------

        if not blob.exists():

            return {
                "status": "NO REPORT",
                "message": "Drift report does not exist yet."
            }

        # -----------------------------------
        # Read report from GCS
        # -----------------------------------

        report_data = blob.download_as_text()

        report = pd.read_csv(
            StringIO(report_data)
        )

        # -----------------------------------
        # Return report
        # -----------------------------------

        return {
            "status": "success",
            "records": len(report),
            "report": report.to_dict(
                orient="records"
            )
        }

    except Exception as exc:

        logging.exception(
            "Failed to read drift report"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to read drift report"
        ) from exc


@app.get("/production-data")
def get_production_data():

    try:

        from google.cloud import storage
        import json

        BUCKET_NAME = "solar-monitoring-511206"
        PRODUCTION_PREFIX = "production/"

        client = storage.Client(
            project="solar-power-monitoring-511206"
        )

        bucket = client.bucket(
            BUCKET_NAME
        )

        records = []

        # -----------------------------------
        # Read production JSON files
        # -----------------------------------

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
        # Return production records
        # -----------------------------------

        return {
            "status": "success",
            "records": len(records),
            "production_data": records
        }

    except Exception as exc:

        logging.exception(
            "Failed to read production data"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to read production data"
        ) from exc