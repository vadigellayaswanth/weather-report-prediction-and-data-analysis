from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_preprocessing import FEATURE_COLUMNS, TARGET_COLUMN

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAPH_DIR = PROJECT_ROOT / "outputs" / "graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


def save_histograms(df: pd.DataFrame):
    hist_features = [column for column in FEATURE_COLUMNS if column in df.columns]
    fig, axes = plt.subplots(len(hist_features), 1, figsize=(10, 18))
    if len(hist_features) == 1:
        axes = [axes]
    for ax, column in zip(axes, hist_features):
        sns.histplot(df[column].dropna(), bins=30, kde=True, ax=ax, color="#3b82f6")
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(GRAPH_DIR / "01_numerical_feature_histograms.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_correlation_heatmap(df: pd.DataFrame):
    numerical = df[[column for column in FEATURE_COLUMNS if column in df.columns]].copy()
    corr = numerical.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Weather Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(GRAPH_DIR / "02_correlation_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_rainfall_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[TARGET_COLUMN].dropna(), bins=40, kde=True, color="#2563eb", ax=ax)
    ax.set_title("Rainfall Distribution")
    ax.set_xlabel("Rainfall")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(GRAPH_DIR / "03_rainfall_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_avg_temp_distribution(df: pd.DataFrame):
    if "avg_temp" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["avg_temp"].dropna(), bins=30, kde=True, color="#0ea5e9", ax=ax)
    ax.set_title("Average Temperature Distribution")
    ax.set_xlabel("Average Temperature")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(GRAPH_DIR / "04_avg_temp_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_temp_vs_rainfall(df: pd.DataFrame):
    if "avg_temp" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="avg_temp", y=TARGET_COLUMN, alpha=0.5, color="#2563eb", ax=ax)
    ax.set_title("Average Temperature vs Rainfall")
    ax.set_xlabel("Average Temperature")
    ax.set_ylabel("Rainfall")
    fig.tight_layout()
    fig.savefig(GRAPH_DIR / "05_avg_temp_vs_rainfall.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_wind_vs_rainfall(df: pd.DataFrame):
    if "wind_speed" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="wind_speed", y=TARGET_COLUMN, alpha=0.5, color="#0284c7", ax=ax)
    ax.set_title("Wind Speed vs Rainfall")
    ax.set_xlabel("Wind Speed")
    ax.set_ylabel("Rainfall")
    fig.tight_layout()
    fig.savefig(GRAPH_DIR / "06_wind_speed_vs_rainfall.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_eda_graphs(df: pd.DataFrame):
    plot_df = df.sample(min(15000, len(df)), random_state=42) if len(df) > 15000 else df
    save_histograms(plot_df)
    save_correlation_heatmap(plot_df)
    save_rainfall_distribution(plot_df)
    save_avg_temp_distribution(plot_df)
    save_temp_vs_rainfall(plot_df)
    save_wind_vs_rainfall(plot_df)

    if "date_of_record" in df.columns:
        trend = df.groupby("date_of_record")[TARGET_COLUMN].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(trend["date_of_record"], trend[TARGET_COLUMN], color="#1d4ed8", linewidth=1.5)
        ax.set_title("Daily Rainfall Trend")
        ax.set_xlabel("Date")
        ax.set_ylabel("Average Rainfall")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(GRAPH_DIR / "07_rainfall_trend.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    return [
        "01_numerical_feature_histograms.png",
        "02_correlation_heatmap.png",
        "03_rainfall_distribution.png",
        "04_avg_temp_distribution.png",
        "05_avg_temp_vs_rainfall.png",
        "06_wind_speed_vs_rainfall.png",
    ]
