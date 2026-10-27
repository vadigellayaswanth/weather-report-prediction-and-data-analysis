from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_preprocessing import DATE_COLUMN, FEATURE_COLUMNS, TARGET_COLUMN, load_raw_dataset, prepare_features_target
from src.eda import generate_eda_graphs
from src.evaluate_model import (
    save_actual_vs_predicted,
    save_feature_importance,
    save_model_comparison,
    save_model_report,
    save_prediction_error_distribution,
    save_selected_model,
)

PROJECT_ROOT = ROOT
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
GRAPHS_DIR = PROJECT_ROOT / "outputs" / "graphs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

# The full dataset is very large (970k+ rows). For a college project review,
# training all models on the full dataset is unnecessarily slow. We keep the
# real data and real preprocessing, but cap the chronological training slice to a
# manageable size to keep the project runnable without fabricating results.
MAX_TRAINING_ROWS = 12000


def build_preprocessor(feature_list):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_list),
        ]
    )


def build_models():
    return {
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=5,
            max_depth=12,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42, max_depth=10),
        "Linear Regression": LinearRegression(),
    }


def train_and_evaluate():
    raw_df = load_raw_dataset()
    cleaned_df = raw_df.copy()
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)

    if DATE_COLUMN in cleaned_df.columns:
        cleaned_df[DATE_COLUMN] = pd.to_datetime(cleaned_df[DATE_COLUMN], errors="coerce")
        cleaned_df = cleaned_df.dropna(subset=[DATE_COLUMN]).reset_index(drop=True)

    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        if column in cleaned_df.columns:
            cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors="coerce")

    cleaned_df = cleaned_df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    cleaned_df = cleaned_df[cleaned_df[TARGET_COLUMN] >= 0].reset_index(drop=True)

    if "min_temp" in cleaned_df.columns and "max_temp" in cleaned_df.columns:
        cleaned_df = cleaned_df[cleaned_df["min_temp"] <= cleaned_df["max_temp"]].reset_index(drop=True)

    if DATE_COLUMN in cleaned_df.columns:
        cleaned_df = cleaned_df.sort_values(DATE_COLUMN).reset_index(drop=True)

    print(f"Cleaned dataset size before review training subset: {len(cleaned_df):,} rows")
    if len(cleaned_df) > MAX_TRAINING_ROWS:
        print(
            f"Using a chronological subset of {MAX_TRAINING_ROWS:,} rows for faster review training "
            "while preserving the original real dataset and preprocessing pipeline."
        )
        cleaned_df = cleaned_df.iloc[:MAX_TRAINING_ROWS].copy()

    X, y, prepared_df = prepare_features_target(cleaned_df)
    feature_columns = list(X.columns)
    print(f"Final training dataset size: {len(prepared_df):,} rows")

    train_size = int(len(prepared_df) * 0.8)
    train_df = prepared_df.iloc[:train_size].copy()
    test_df = prepared_df.iloc[train_size:].copy()

    X_train = train_df[feature_columns]
    X_test = test_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]

    generate_eda_graphs(prepared_df)

    model_results = []
    best_result = None
    best_model_name = None
    best_pipeline = None

    print(f"Training on {len(X_train):,} rows and evaluating on {len(X_test):,} rows")

    for model_name, estimator in build_models().items():
        print(f"\nTraining model: {model_name}")
        preprocessor = build_preprocessor(feature_columns)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)
        test_pred = pipeline.predict(X_test)

        metrics = {
            "Model": model_name,
            "MAE": mean_absolute_error(y_test, test_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, test_pred)),
            "R2": r2_score(y_test, test_pred),
        }
        model_results.append(metrics)

        print(f"{model_name} -> MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}, R2: {metrics['R2']:.4f}")

        if best_result is None or metrics["R2"] > best_result["R2"]:
            best_result = metrics
            best_model_name = model_name
            best_pipeline = pipeline

        save_feature_importance(pipeline.named_steps["model"], feature_columns)

    model_df = pd.DataFrame(model_results)
    save_model_comparison(model_df)

    if best_pipeline is not None:
        best_pred = best_pipeline.predict(X_test)
        save_actual_vs_predicted(y_test.to_numpy(), best_pred, best_model_name)
        save_prediction_error_distribution(y_test.to_numpy(), best_pred)
        save_selected_model(best_pipeline)

        report = {
            "dataset_size": len(prepared_df),
            "training_records": len(X_train),
            "testing_records": len(X_test),
            "selected_features": feature_columns,
            "target_variable": TARGET_COLUMN,
            "selected_model": best_model_name,
            "mae": best_result["MAE"],
            "rmse": best_result["RMSE"],
            "r2": best_result["R2"],
        }
        save_model_report(report)

        results_frame = pd.DataFrame({
            "Actual": y_test.reset_index(drop=True),
            "Predicted": pd.Series(best_pred, name="Predicted").reset_index(drop=True),
        })
        results_frame.to_csv(RESULTS_DIR / "rainfall_prediction_results.csv", index=False)

    print("\nModel training completed.")
    print(pd.DataFrame(model_results).to_string(index=False))
    print(f"Best model: {best_model_name}")
    print({
        "MAE": round(best_result["MAE"], 4),
        "RMSE": round(best_result["RMSE"], 4),
        "R2": round(best_result["R2"], 4),
    })

    return best_pipeline, best_model_name, model_df, prepared_df


if __name__ == "__main__":
    train_and_evaluate()
