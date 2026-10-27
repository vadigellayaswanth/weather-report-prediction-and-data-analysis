from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "rainfall_model.pkl"


def validate_prediction_input(avg_temp, min_temp, max_temp, wind_speed, air_pressure, elevation, latitude, longitude):
    numeric_values = {
        "avg_temp": avg_temp,
        "min_temp": min_temp,
        "max_temp": max_temp,
        "wind_speed": wind_speed,
        "air_pressure": air_pressure,
        "elevation": elevation,
        "latitude": latitude,
        "longitude": longitude,
    }

    # 1. Numerical checks
    for name, value in numeric_values.items():
        if value is None or value == "":
            raise ValueError(f"'{name}' is required and cannot be empty.")
        try:
            numeric_values[name] = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"'{name}' must be a valid numerical value. Received: {value}") from None

    # 2. Temperature consistency
    t_min = numeric_values["min_temp"]
    t_max = numeric_values["max_temp"]
    t_avg = numeric_values["avg_temp"]

    if t_min > t_max:
        raise ValueError(
            f"Minimum temperature ({t_min}°C) cannot exceed maximum temperature ({t_max}°C)."
        )

    if not (t_min <= t_avg <= t_max):
        raise ValueError(
            f"Average temperature ({t_avg}°C) must be between minimum temperature ({t_min}°C) and maximum temperature ({t_max}°C)."
        )

    # 3. Geographic ranges (General bounds & Indian subcontinent dataset bounds)
    lat = numeric_values["latitude"]
    lon = numeric_values["longitude"]

    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"Latitude ({lat}°N) must be between -90.0 and 90.0 degrees.")

    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"Longitude ({lon}°E) must be between -180.0 and 180.0 degrees.")

    # 4. Realistic bounds supported by historical meteorological observations
    if not (-20.0 <= t_min <= 45.0):
        raise ValueError(f"Minimum temperature ({t_min}°C) is outside realistic historical limits (-20°C to 45°C).")

    if not (-10.0 <= t_max <= 60.0):
        raise ValueError(f"Maximum temperature ({t_max}°C) is outside realistic historical limits (-10°C to 60°C).")

    if not (0.0 <= numeric_values["wind_speed"] <= 150.0):
        raise ValueError(f"Wind speed ({numeric_values['wind_speed']} km/h) must be between 0.0 and 150.0 km/h.")

    if not (850.0 <= numeric_values["air_pressure"] <= 1060.0):
        raise ValueError(
            f"Air pressure ({numeric_values['air_pressure']} hPa) must be between 850.0 and 1060.0 hPa."
        )

    if not (-100.0 <= numeric_values["elevation"] <= 6000.0):
        raise ValueError(
            f"Elevation ({numeric_values['elevation']} m) must be between -100.0 and 6000.0 meters."
        )

    return numeric_values


def predict_rainfall(avg_temp, min_temp, max_temp, wind_speed, air_pressure, elevation, latitude, longitude):
    model_path = MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError("The trained rainfall model was not found. Please train the model first.")

    validated = validate_prediction_input(avg_temp, min_temp, max_temp, wind_speed, air_pressure, elevation, latitude, longitude)
    input_df = pd.DataFrame([
        {
            "avg_temp": validated["avg_temp"],
            "min_temp": validated["min_temp"],
            "max_temp": validated["max_temp"],
            "wind_speed": validated["wind_speed"],
            "air_pressure": validated["air_pressure"],
            "elevation": validated["elevation"],
            "latitude": validated["latitude"],
            "longitude": validated["longitude"],
        }
    ])

    model = joblib.load(model_path)
    prediction = model.predict(input_df)[0]
    return float(prediction)
