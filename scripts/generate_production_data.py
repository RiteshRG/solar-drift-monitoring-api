import pandas as pd
import requests

from datetime import datetime, timedelta


API_URL = "http://127.0.0.1:8000/predict"


# -----------------------------------
# Load validation data
# -----------------------------------

validation = pd.read_csv(
    "data/validation.csv"
)


# -----------------------------------
# Take 60 validation records
# -----------------------------------

production_data = validation.head(
    60
).copy()


# -----------------------------------
# Starting production timestamp
# -----------------------------------

start_time = datetime(
    2026,
    10,
    1,
    8,
    0
)


# -----------------------------------
# Send production requests
# -----------------------------------

for index, row in production_data.iterrows():

    current_time = (
        start_time
        + timedelta(hours=index)
    )

    payload = {

        "IRRADIATION": float(
            row["IRRADIATION"]
        ),

        "AMBIENT_TEMPERATURE": float(
            row["AMBIENT_TEMPERATURE"]
        ),

        "DATE_TIME": current_time.strftime(
            "%d-%m-%Y %H:%M"
        )
    }

    response = requests.post(
        API_URL,
        json=payload
    )

    print(
        f"Request {index + 1}/60 "
        f"→ {response.status_code}"
    )

    if response.status_code != 200:
        print(response.text)


print(
    "Production data generation completed."
)

