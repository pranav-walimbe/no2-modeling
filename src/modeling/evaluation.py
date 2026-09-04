"""Compute and persist model evaluation metrics."""

import json
import os

import numpy as np
import pandas as pd


def plant_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-plant RMSE, MAE, MAPE, and mean true emissions."""
    return (
        df.groupby(["facilityId", "facilityName", "lon", "lat"])
        .apply(
            lambda group: pd.Series(
                {
                    "rmse": np.sqrt(((group["y_pred"] - group["y_true"]) ** 2).mean()),
                    "mae": (group["y_pred"] - group["y_true"]).abs().mean(),
                    "mape": ((group["y_pred"] - group["y_true"]).abs() / group["y_true"].replace(0, np.nan)).mean()
                    * 100,
                    "mean_y_true": group["y_true"].mean(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )


def save_results(train_df: pd.DataFrame, test_df: pd.DataFrame, val_df: pd.DataFrame, run_dir: str) -> None:
    """Save aggregate and test-tertile evaluation metrics as JSON."""
    results = {}
    for split, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        y_true, y_pred = df["y_true"].values, df["y_pred"].values
        mae = float(np.abs(y_true - y_pred).mean())
        mape = float(np.nanmean(np.abs(y_true - y_pred) / np.where(y_true == 0, np.nan, y_true)) * 100)
        results[split] = {"mae": mae, "mape": mape}

    lower_threshold, upper_threshold = np.percentile(test_df["y_true"], [33, 66])
    tertiles = {
        "low": test_df[test_df["y_true"] < lower_threshold],
        "mid": test_df[(test_df["y_true"] >= lower_threshold) & (test_df["y_true"] < upper_threshold)],
        "high": test_df[test_df["y_true"] >= upper_threshold],
    }
    results["test_tertiles"] = {}
    for name, subset in tertiles.items():
        y_true, y_pred = subset["y_true"].values, subset["y_pred"].values
        results["test_tertiles"][name] = {
            "mae": float(np.abs(y_true - y_pred).mean()),
            "mape": float(np.nanmean(np.abs(y_true - y_pred) / np.where(y_true == 0, np.nan, y_true)) * 100),
            "n": len(subset),
            "label_range": [float(y_true.min()), float(y_true.max())],
        }

    with open(os.path.join(run_dir, "results.json"), "w") as file:
        json.dump(results, file, indent=2)
