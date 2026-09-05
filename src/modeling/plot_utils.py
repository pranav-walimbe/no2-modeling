"""Plotting utilities for training and signed-target evaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from modeling.eval_utils import (
    MASS_PRED_COL,
    MASS_TRUE_COL,
    NORMALIZED_PRED_COL,
    NORMALIZED_TRUE_COL,
    plant_metrics,
)

SPLIT_ORDER = ("train", "val", "test")


def _save(figure: plt.Figure, run_dir: str | Path, plot_name: str) -> None:
    figure.savefig(Path(run_dir) / f"{plot_name}.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_loss_curve(train_losses: list[float], val_losses: list[float], run_dir: str | Path) -> None:
    """Plot standardized-target Huber loss across epochs."""
    sns.set_theme(style="whitegrid", font_scale=1.2)
    figure, axis = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    axis.plot(epochs, train_losses, label="Train", linewidth=2)
    axis.plot(epochs, val_losses, label="Validation", linewidth=2)
    axis.set(xlabel="Epoch", ylabel="Huber loss", title="Training and validation loss")
    axis.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axis.legend()
    figure.tight_layout()
    _save(figure, run_dir, "loss_curve")


def _shared_limits(frames: dict[str, pd.DataFrame], true_col: str, pred_col: str) -> tuple[float, float]:
    minimum = min(min(frame[true_col].min(), frame[pred_col].min()) for frame in frames.values())
    maximum = max(max(frame[true_col].max(), frame[pred_col].max()) for frame in frames.values())
    padding = max((maximum - minimum) * 0.03, 1e-6)
    return minimum - padding, maximum + padding


def plot_pred_vs_true(split_frames: dict[str, pd.DataFrame], run_dir: str | Path) -> None:
    """Plot normalized and physical predictions without assuming positivity."""
    sns.set_theme(style="whitegrid", font_scale=1.0)
    figure, axes = plt.subplots(2, 3, figsize=(17, 10))
    panels = (
        (NORMALIZED_TRUE_COL, NORMALIZED_PRED_COL, "Normalized NOx change"),
        (MASS_TRUE_COL, MASS_PRED_COL, "NOx mass change"),
    )
    for row, (true_col, pred_col, label) in enumerate(panels):
        limits = _shared_limits(split_frames, true_col, pred_col)
        for axis, split in zip(axes[row], SPLIT_ORDER):
            frame = split_frames[split]
            axis.scatter(frame[true_col], frame[pred_col], alpha=0.35, s=7, linewidth=0)
            axis.plot(limits, limits, color="#222222", linewidth=1.1, linestyle="--")
            axis.set(xlim=limits, ylim=limits, xlabel=f"True {label}", ylabel=f"Predicted {label}", title=split)
    figure.tight_layout()
    _save(figure, run_dir, "pred_vs_true")


def plot_residuals(split_frames: dict[str, pd.DataFrame], run_dir: str | Path) -> None:
    """Plot signed physical residuals against true NOx mass change."""
    sns.set_theme(style="whitegrid", font_scale=1.0)
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    for axis, split in zip(axes, SPLIT_ORDER):
        frame = split_frames[split]
        residual = frame[MASS_PRED_COL] - frame[MASS_TRUE_COL]
        axis.scatter(frame[MASS_TRUE_COL], residual, alpha=0.35, s=7, linewidth=0)
        axis.axhline(0.0, color="#222222", linewidth=1.1, linestyle="--")
        axis.set(xlabel="True NOx mass change", ylabel="Prediction residual", title=split)
    figure.tight_layout()
    _save(figure, run_dir, "residuals")


def plot_spatial_error(split_frames: dict[str, pd.DataFrame], run_dir: str | Path) -> None:
    """Map held-out AOI error without a runtime network dependency."""
    sns.set_theme(style="white", font_scale=1.0)
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
    for axis, split in zip(axes, ("val", "test")):
        metrics = plant_metrics(split_frames[split])
        points = axis.scatter(
            metrics["lon"],
            metrics["lat"],
            c=metrics["mass_change_mae"],
            cmap="viridis",
            s=30,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.2,
        )
        figure.colorbar(points, ax=axis, label="NOx mass-change MAE")
        axis.set(xlabel="Longitude", ylabel="Latitude", title=f"{split} AOIs")
    figure.tight_layout()
    _save(figure, run_dir, "spatial_error")
