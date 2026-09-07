"""Plotting utilities for binary NOx-change classification."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from modeling.eval_utils import POSITIVE_PROBABILITY_COL, TRUE_CLASS_COL, plant_metrics

SPLIT_ORDER = ("train", "val", "test")


def _save(figure: plt.Figure, run_dir: str | Path, plot_name: str) -> None:
    # Persist and close one completed plot
    figure.savefig(Path(run_dir) / f"{plot_name}.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_loss_curve(train_losses: list[float], val_losses: list[float], run_dir: str | Path) -> None:
    """Plot binary cross-entropy loss across epochs.

    Args:
        train_losses: Mean training loss for each epoch.
        val_losses: Mean validation loss for each epoch.
        run_dir: Model-run output directory.
    """
    sns.set_theme(style="whitegrid", font_scale=1.2)
    figure, axis = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    axis.plot(epochs, train_losses, label="Train", linewidth=2)
    axis.plot(epochs, val_losses, label="Validation", linewidth=2)
    axis.set(xlabel="Epoch", ylabel="Binary cross-entropy", title="Training and validation loss")
    axis.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axis.legend()
    figure.tight_layout()
    _save(figure, run_dir, "loss_curve")


def plot_class_probabilities(split_frames: dict[str, pd.DataFrame], run_dir: str | Path) -> None:
    """Plot positive-class probability by true class for every split.

    Args:
        split_frames: Row-level predictions for each data split.
        run_dir: Model-run output directory.
    """
    sns.set_theme(style="whitegrid", font_scale=1.0)
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), sharex=True, sharey=True)
    for axis, split in zip(axes, SPLIT_ORDER, strict=True):
        frame = split_frames[split]
        for label, color in ((0, "#4c72b0"), (1, "#dd8452")):
            values = frame.loc[frame[TRUE_CLASS_COL] == label, POSITIVE_PROBABILITY_COL]
            axis.hist(values, bins=20, range=(0, 1), alpha=0.55, color=color, label=f"Class {label}")
        axis.axvline(0.5, color="#222222", linewidth=1.1, linestyle="--")
        axis.set(xlabel="Predicted probability of class 1", ylabel="Records", title=split)
        axis.legend()
    figure.tight_layout()
    _save(figure, run_dir, "class_probabilities")


def plot_spatial_accuracy(split_frames: dict[str, pd.DataFrame], run_dir: str | Path) -> None:
    """Map held-out AOI accuracy without a runtime network dependency.

    Args:
        split_frames: Row-level predictions for each data split.
        run_dir: Model-run output directory.
    """
    sns.set_theme(style="white", font_scale=1.0)
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
    for axis, split in zip(axes, ("val", "test"), strict=True):
        metrics = plant_metrics(split_frames[split])
        points = axis.scatter(
            metrics["lon"],
            metrics["lat"],
            c=metrics["accuracy"],
            cmap="viridis",
            vmin=0,
            vmax=1,
            s=30,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.2,
        )
        figure.colorbar(points, ax=axis, label="Classification accuracy")
        axis.set(xlabel="Longitude", ylabel="Latitude", title=f"{split} AOIs")
    figure.tight_layout()
    _save(figure, run_dir, "spatial_accuracy")
