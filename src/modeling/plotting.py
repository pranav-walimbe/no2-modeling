"""Create training and evaluation visualizations."""

import os

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import COUNTRIES_URL, LABEL_COL
from modeling.evaluation import plant_metrics


def _save(fig, run_dir: str, plot_name: str) -> None:
    # Save a figure to the run directory and close it
    path = os.path.join(run_dir, f"{plot_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(train_losses: list[float], val_losses: list[float], run_dir: str) -> None:
    """Plot training and validation loss across epochs."""
    run_name = os.path.basename(run_dir)
    sns.set_theme(style="whitegrid", font_scale=1.4)
    fig, axis = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    axis.plot(epochs, train_losses, label="Train loss", color="#4C9BE8", linewidth=2)
    axis.plot(epochs, val_losses, label="Val loss", color="#E85D5D", linewidth=2)
    axis.set_xlabel("Epoch", labelpad=8)
    axis.set_ylabel("Loss (MAE)", labelpad=8)
    axis.set_title(f"Training and Validation Loss\n({run_name})", fontweight="bold", pad=14)
    axis.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axis.legend(fontsize=10, framealpha=0.9)
    plt.tight_layout()
    _save(fig, run_dir, "loss_curve")


def plot_pred_vs_true(train_df: pd.DataFrame, test_df: pd.DataFrame, val_df: pd.DataFrame, run_dir: str) -> None:
    """Plot predicted versus true NOx emissions for all splits."""
    run_name = os.path.basename(run_dir)
    sns.set_theme(style="whitegrid", font_scale=1.4)
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    for axis, (df, split) in zip(axes, [(train_df, "train"), (test_df, "test"), (val_df, "val")]):
        axis.scatter(df["y_true"], df["y_pred"], alpha=0.4, s=8, linewidth=0, color="#4C9BE8")
        limits = [min(df["y_true"].min(), df["y_pred"].min()), max(df["y_true"].max(), df["y_pred"].max())]
        axis.plot(limits, limits, color="#222222", linewidth=1.2, linestyle="--")
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_aspect("equal")
        axis.set_xlabel("True NOx Emissions (lb/hr)", labelpad=8)
        axis.set_ylabel("Predicted NOx Emissions (lb/hr)", labelpad=8)
        axis.set_title(f"{split} set", fontweight="bold", pad=14)
    fig.suptitle(f"Predicted vs True Emissions - {run_name}", fontweight="bold", fontsize=14)
    plt.tight_layout()
    _save(fig, run_dir, "pred_vs_true")


def plot_residuals(train_df: pd.DataFrame, test_df: pd.DataFrame, val_df: pd.DataFrame, run_dir: str) -> None:
    """Plot absolute residuals versus true NOx emissions for all splits."""
    run_name = os.path.basename(run_dir)
    sns.set_theme(style="whitegrid", font_scale=1.4)
    fig, axes = plt.subplots(1, 3, figsize=(24, 5))
    for axis, (df, split) in zip(axes, [(train_df, "train"), (test_df, "test"), (val_df, "val")]):
        residuals = (df["y_pred"] - df["y_true"]).abs()
        axis.scatter(df["y_true"], residuals, alpha=0.4, s=8, linewidth=0, color="#4C9BE8")
        axis.set_xlabel(f"y_true  ({LABEL_COL})", labelpad=8)
        axis.set_ylabel(f"|y_pred - y_true|  ({LABEL_COL})", labelpad=8)
        axis.set_title(f"{split} set", fontweight="bold", pad=14)
    fig.suptitle(f"Absolute Residuals vs True Emissions - {run_name}", fontweight="bold", fontsize=14)
    plt.tight_layout()
    _save(fig, run_dir, "residuals")


def plot_spatial_error(train_df: pd.DataFrame, test_df: pd.DataFrame, val_df: pd.DataFrame, run_dir: str) -> None:
    """Plot per-plant MAE on a map of the contiguous United States."""
    run_name = os.path.basename(run_dir)
    countries = gpd.read_file(COUNTRIES_URL)
    us = countries[countries.NAME == "United States of America"]
    sns.set_theme(style="white", font_scale=1.4)
    fig, axes = plt.subplots(3, 1, figsize=(12, 21))
    for axis, (df, split) in zip(axes, [(train_df, "train"), (test_df, "test"), (val_df, "val")]):
        metrics = plant_metrics(df)
        us.plot(ax=axis, color="lightgray", edgecolor="black")
        scatter = axis.scatter(
            metrics["lon"],
            metrics["lat"],
            c=metrics["mae"],
            cmap="YlOrRd",
            s=40,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.3,
            vmin=metrics["mae"].min(),
            vmax=metrics["mae"].max(),
        )
        plt.colorbar(scatter, ax=axis, label=f"MAE  ({LABEL_COL})")
        axis.set_xlim(-130, -65)
        axis.set_ylim(24, 50)
        axis.set_xlabel("Longitude", labelpad=8)
        axis.set_ylabel("Latitude", labelpad=8)
        axis.set_title(f"{split} set", fontweight="bold", pad=14)
    fig.suptitle(f"Spatial Error Distribution - {run_name}", fontweight="bold", fontsize=14)
    plt.tight_layout()
    _save(fig, run_dir, "spatial_error")
