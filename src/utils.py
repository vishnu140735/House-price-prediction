from __future__ import annotations
import os
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "visualizations")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def load_california_housing() -> pd.DataFrame:
    """Fetch the California Housing dataset and return as a DataFrame with target column 'target'."""
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    df.rename(columns={"MedHouseVal": "target"}, inplace=True)
    return df

def train_test_split_df(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
                        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def plot_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, filename: str = "predicted_vs_actual.png"):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Predicted vs Actual House Prices")
    lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    plt.plot(lims, lims)
    out_path = os.path.join(PLOT_DIR, filename)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, filename: str = "residual_plot.png"):
    residuals = y_true - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residual Plot")
    out_path = os.path.join(PLOT_DIR, filename)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path
