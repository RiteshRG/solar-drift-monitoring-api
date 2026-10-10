from datetime import datetime
import numpy as np
import pandas as pd

def create_time_features(date_time):
    hour = date_time.hour

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    return hour_sin, hour_cos