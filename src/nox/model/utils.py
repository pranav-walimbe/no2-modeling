"""
Define utility functions (normalization, plotting) for ML pipeline
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import seaborn as sns
import json
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *

def compute_stats(split, batch_size=512):
    """Compute mean/std normalization stats from training images and tabular features"""
    images = np.load(os.path.join(IMAGES_DIR, f"{split}_tempo.npy"))
    df = pd.read_csv(os.path.join(DATASET_DF, f"{split}_df.csv"))
    num_adj = df["num_adj_units"].values.astype(np.float32)
    prev_qtr_mass = df["prev_qtr_mass"].values.astype(np.float32)
    u10 = df["u10"].values.astype(np.float32)
    v10 = df["v10"].values.astype(np.float32)
    n, n_channels = images.shape[0], images.shape[1]

    # first pass: compute per-channel mean
    total, sums = 0, np.zeros(n_channels, dtype=np.float64)
    for i in range(0, n, batch_size):
        batch = images[i:i+batch_size].astype(np.float64)
        batch[:, 0] = np.clip(batch[:, 0], None, MAX_IMG_VAL)
        total += batch.shape[0] * batch.shape[2] * batch.shape[3]
        sums += batch.sum(axis=(0, 2, 3))
    means = sums / total

    # second pass: compute per-channel std
    sum_sq_diffs = np.zeros(n_channels, dtype=np.float64)
    for i in range(0, n, batch_size):
        batch = images[i:i+batch_size].astype(np.float64)
        batch[:, 0] = np.clip(batch[:, 0], None, MAX_IMG_VAL)
        sum_sq_diffs += ((batch - means[np.newaxis, :, np.newaxis, np.newaxis]) ** 2).sum(axis=(0, 2, 3))
    stds = np.sqrt(sum_sq_diffs / total)

    return {
        "image_mean": torch.tensor(means, dtype=torch.float32).view(n_channels, 1, 1),
        "image_std": torch.tensor(stds, dtype=torch.float32).view(n_channels, 1, 1),
        "num_adj_mean": float(num_adj.mean()),
        "num_adj_std": float(num_adj.std()),
        "prev_qtr_mass_mean": float(prev_qtr_mass.mean()),
        "prev_qtr_mass_std": float(prev_qtr_mass.std()),
        "u10_mean": float(u10.mean()),
        "u10_std": float(u10.std()),
        "v10_mean": float(v10.mean()),
        "v10_std": float(v10.std()),
    }

def _save(fig, run_dir, plot_name):
    """Save figure to run directory and close it"""
    path = os.path.join(run_dir, f"{plot_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def _plant_metrics(df):
    """Compute per-plant RMSE, MAE, MAPE, and mean true emissions"""
    return (
        df.groupby(["facilityId", "facilityName", "lon", "lat"])
        .apply(lambda g: pd.Series({
            "rmse": np.sqrt(((g["y_pred"] - g["y_true"]) ** 2).mean()),
            "mae": (g["y_pred"] - g["y_true"]).abs().mean(),
            "mape": ((g["y_pred"] - g["y_true"]).abs() / g["y_true"].replace(0, np.nan)).mean() * 100,
            "mean_y_true": g["y_true"].mean(),
        })).reset_index()
    )

def plot_loss_curve(train_losses, val_losses, run_dir):
    """Plot training and validation loss across epochs"""
    run_name = os.path.basename(run_dir)
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train loss", color="#4C9BE8", linewidth=2)
    ax.plot(epochs, val_losses, label="Val loss", color="#E85D5D", linewidth=2)
    ax.set_xlabel("Epoch", labelpad=8)
    ax.set_ylabel("Loss", labelpad=8)
    ax.set_title(f"Training and Validation Loss\n({run_name})", fontweight="bold", pad=14)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(fontsize=10, framealpha=0.9)
    plt.tight_layout()
    _save(fig, run_dir, "loss_curve")

def plot_pred_vs_true(train_df, test_df, val_df, run_dir):
    """Scatter plot of predicted vs true NOx emissions for all splits"""
    run_name = os.path.basename(run_dir)
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    for ax, (df, split) in zip(axes, [(train_df, "train"), (test_df, "test"), (val_df, "val")]):
        ax.scatter(df["y_true"], df["y_pred"], alpha=0.4, s=8, linewidth=0, color="#4C9BE8")
        lims = [min(df["y_true"].min(), df["y_pred"].min()),
                max(df["y_true"].max(), df["y_pred"].max())]
        ax.plot(lims, lims, color="#222222", linewidth=1.2, linestyle="--", label="Perfect prediction")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"y_true ({LABEL_COL})", labelpad=8)
        ax.set_ylabel(f"y_pred ({LABEL_COL})", labelpad=8)
        ax.set_title(f"{split} set", fontweight="bold", pad=14)
        ax.legend(fontsize=10, framealpha=0.9)
    fig.suptitle(f"Predicted vs True Emissions - {run_name}", fontweight="bold", fontsize=14)
    plt.tight_layout()
    _save(fig, run_dir, "pred_vs_true")

def plot_residuals(train_df, test_df, val_df, run_dir):
    """Scatter plot of absolute residuals vs true NOx emissions for all splits"""
    run_name = os.path.basename(run_dir)
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(1, 3, figsize=(24, 5))
    for ax, (df, split) in zip(axes, [(train_df, "train"), (test_df, "test"), (val_df, "val")]):
        df = df.copy()
        df["abs_residual"] = (df["y_pred"] - df["y_true"]).abs()
        ax.scatter(df["y_true"], df["abs_residual"], alpha=0.4, s=8, linewidth=0, color="#4C9BE8")
        ax.set_xlabel(f"y_true  ({LABEL_COL})", labelpad=8)
        ax.set_ylabel(f"|y_pred - y_true|  ({LABEL_COL})", labelpad=8)
        ax.set_title(f"{split} set", fontweight="bold", pad=14)
    fig.suptitle(f"Absolute Residuals vs True Emissions - {run_name}", fontweight="bold", fontsize=14)
    plt.tight_layout()
    _save(fig, run_dir, "residuals")

def plot_spatial_error(train_df, test_df, val_df, run_dir):
    """Heatmap of per-plant MAE overlaid on a US map for all splits"""
    run_name = os.path.basename(run_dir)
    us = gpd.read_file(COUNTRIES_URL)
    us = us[us.NAME == "United States of America"]
    sns.set_theme(style="white", font_scale=1.1)
    fig, axes = plt.subplots(3, 1, figsize=(12, 21))
    for ax, (df, split) in zip(axes, [(train_df, "train"), (test_df, "test"), (val_df, "val")]):
        metrics = _plant_metrics(df)
        us.plot(ax=ax, color="lightgray", edgecolor="black")
        sc = ax.scatter(
            metrics["lon"], metrics["lat"],
            c=metrics["mae"], cmap="YlOrRd", s=40, alpha=0.85,
            edgecolors="black", linewidths=0.3,
            vmin=metrics["mae"].min(), vmax=metrics["mae"].max()
        )
        plt.colorbar(sc, ax=ax, label=f"MAE  ({LABEL_COL})")
        ax.set_xlim(-130, -65)
        ax.set_ylim(24, 50)
        ax.set_xlabel("Longitude", labelpad=8)
        ax.set_ylabel("Latitude", labelpad=8)
        ax.set_title(f"{split} set", fontweight="bold", pad=14)
    fig.suptitle(f"Spatial Error Distribution - {run_name}", fontweight="bold", fontsize=14)
    plt.tight_layout()
    _save(fig, run_dir, "spatial_error")

def save_results(train_df, test_df, val_df, run_dir):
    results = {}

    for split, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        y_true, y_pred = df["y_true"].values, df["y_pred"].values
        mae = float(np.abs(y_true - y_pred).mean())
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        r2 = float(1 - ss_res / ss_tot)
        results[split] = {"mae": mae, "r2": r2}

    t33, t66 = np.percentile(test_df["y_true"], [33, 66])
    low  = test_df[test_df["y_true"] <  t33]
    mid  = test_df[(test_df["y_true"] >= t33) & (test_df["y_true"] < t66)]
    high = test_df[test_df["y_true"] >= t66]

    results["test_tertiles"] = {}
    for name, subset in [("low", low), ("mid", mid), ("high", high)]:
        y_true, y_pred = subset["y_true"].values, subset["y_pred"].values
        mae = float(np.abs(y_true - y_pred).mean())
        results["test_tertiles"][name] = {
            "mae": mae,
            "n": len(subset),
            "label_range": [float(y_true.min()), float(y_true.max())],
        }

    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

def generate_eval_plots(train_df, test_df, val_df, run_dir):
    """Generate all inference evaluation plots for a completed training run"""
    plot_pred_vs_true(train_df, test_df, val_df, run_dir)
    plot_residuals(train_df, test_df, val_df, run_dir)
    plot_spatial_error(train_df, test_df, val_df, run_dir)
    save_results(train_df, test_df, val_df, run_dir)