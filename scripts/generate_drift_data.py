import pandas as pd

validation = pd.read_csv("data/validation.csv")

validation_drift = validation.copy()

validation_drift["IRRADIATION"] = (
    validation_drift["IRRADIATION"] * 1.3
)

validation_drift.to_csv(
    "data/validation_drift.csv",
    index=False
)

print("Drift dataset created.")