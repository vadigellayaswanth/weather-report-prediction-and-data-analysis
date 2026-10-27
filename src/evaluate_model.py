from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
GRAPHS_DIR = PROJECT_ROOT / "outputs" / "graphs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


def compute_metrics(y_true, y_pred) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse,
        "R2": r2_score(y_true, y_pred),
    }


def save_model_comparison(model_results: pd.DataFrame):
    model_results.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=model_results, x="Model", y="RMSE", palette="Blues_d", ax=ax)
    ax.set_title("Model Performance Comparison (RMSE)")
    ax.set_ylabel("RMSE")
    ax.set_xlabel("Model")
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / "08_model_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_actual_vs_predicted(y_true, y_pred, model_name: str):
    comparison = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
    comparison.to_csv(RESULTS_DIR / "actual_vs_predicted.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    if len(y_true) > 10000:
        idx = np.random.choice(len(y_true), 10000, replace=False)
        ax.scatter(y_true[idx], y_pred[idx], alpha=0.3, color="#2563eb", s=15)
    else:
        ax.scatter(y_true, y_pred, alpha=0.4, color="#2563eb", s=15)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="red", linewidth=1.5, label="Actual = Predicted")
    ax.set_title(f"Actual vs Predicted Rainfall ({model_name})")
    ax.set_xlabel("Actual Rainfall")
    ax.set_ylabel("Predicted Rainfall")
    ax.legend()
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / "09_actual_vs_predicted.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_prediction_error_distribution(y_true, y_pred):
    errors = pd.Series(y_true - y_pred, name="Prediction Error")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(errors, bins=40, kde=True, color="#0f766e", ax=ax)
    ax.set_title("Prediction Error Distribution")
    ax.set_xlabel("Error (Actual - Predicted)")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / "10_prediction_error_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=importance_df, x="Importance", y="Feature", palette="Blues_d", ax=ax)
        ax.set_title("Feature Importance for Rainfall Prediction")
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Feature")
        fig.tight_layout()
        fig.savefig(GRAPHS_DIR / "11_feature_importance.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def save_model_report(report: dict):
    pd.DataFrame([report]).to_csv(RESULTS_DIR / "evaluation_report.csv", index=False)


def save_selected_model(model):
    joblib.dump(model, PROJECT_ROOT / "models" / "rainfall_model.pkl")
