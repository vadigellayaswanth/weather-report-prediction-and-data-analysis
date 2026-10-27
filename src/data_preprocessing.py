from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import List

import kagglehub
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_FILE = DATA_DIR / "india_weather_rainfall_data.xlsx"
TARGET_COLUMN = "rainfall"
DATE_COLUMN = "date_of_record"
FEATURE_COLUMNS = [
    "avg_temp",
    "min_temp",
    "max_temp",
    "wind_speed",
    "air_pressure",
    "elevation",
    "latitude",
    "longitude",
]


def ensure_dataset_file() -> Path:
    """Ensure the dataset exists in the local project data folder."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_FILE.exists():
        return DATA_FILE

    print("Dataset not found locally. Downloading from Kaggle...")
    dataset_path = kagglehub.dataset_download("ameydilipmorye/indian-rainfall-and-weather-data")
    downloaded_file = Path(dataset_path) / DATA_FILE.name

    if not downloaded_file.exists():
        raise FileNotFoundError(f"Could not find dataset file at {downloaded_file}.")

    shutil.copy2(downloaded_file, DATA_FILE)
    return DATA_FILE


def load_raw_dataset() -> pd.DataFrame:
    """Load the Indian rainfall dataset from the local Excel file, using a local parquet cache for high performance if available."""
    dataset_path = ensure_dataset_file()
    cache_path = DATA_DIR / "india_weather_rainfall_data.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    df = pd.read_excel(dataset_path)
    try:
        df.to_parquet(cache_path, index=False)
    except Exception:
        pass
    return df


def clean_weather_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the data while preserving the project's original weather features."""
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    if DATE_COLUMN in cleaned.columns:
        cleaned[DATE_COLUMN] = pd.to_datetime(cleaned[DATE_COLUMN], errors="coerce")
        cleaned = cleaned.dropna(subset=[DATE_COLUMN]).reset_index(drop=True)

    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    cleaned = cleaned[cleaned[TARGET_COLUMN] >= 0].reset_index(drop=True)

    if "min_temp" in cleaned.columns and "max_temp" in cleaned.columns:
        cleaned = cleaned[cleaned["min_temp"] <= cleaned["max_temp"]].reset_index(drop=True)

    return cleaned


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    available = [column for column in FEATURE_COLUMNS if column in df.columns]
    if not available:
        raise ValueError("No valid prediction features were found in the dataset.")
    return available


def prepare_features_target(df: pd.DataFrame):
    cleaned = clean_weather_dataset(df)
    feature_columns = get_feature_columns(cleaned)
    X = cleaned[feature_columns].copy()
    y = cleaned[TARGET_COLUMN].copy()
    return X, y, cleaned
