from fastapi import FastAPI, HTTPException
from typing import Annotated
import joblib
from datetime import datetime
from utils import create_time_features, log_prediction
from schema import PredictionRequest,PredictionResponse


import pandas as pd
from monitoring.drift_monitor import check_drift

app = FastAPI()

model = joblib.load("models/model_pipeline.joblib")

MODEL_STATUS_MESSAGE = "model loaded successfully" if model else "model not loaded"
print(MODEL_STATUS_MESSAGE)


@app.get("/")
def home():
    return {"message":"Solar power application api"}


@app.get("/model-status")
def get_model_status():
    return {
        "status": "healthy" if model else "unavailable",
        "model_loaded": bool(model),
        "message": MODEL_STATUS_MESSAGE,
    }

@app.post("/predict",response_model=PredictionResponse)
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
        hour_sin, hour_cos = create_time_features(date_time)

        input_data = [[
            request.IRRADIATION,
            request.AMBIENT_TEMPERATURE,
            hour_sin,
            hour_cos
        ]]

        prediction = model.predict(input_data)[0]

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

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )


current_file = pd.read_csv("data/train.csv")
referrence_file = pd.read_csv("data/validation_drift.csv")
check_drift(referrence_file,current_file,"AMBIENT_TEMPERATURE")