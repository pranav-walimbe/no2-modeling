"""Plotting utilities for binary NOx-change classification."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from modeling.eval_utils import (
    POSITIVE_PROBABILITY_COL,
    TRUE_CLASS_COL,
    classification_metrics,
    plant_metrics,
)

SPLIT_ORDER = ("train", "val", "test")
MODEL_DISPLAY_NAMES = {"deep_learning": "Deep learning", "xgboost": "XGBoost"}
COMPARISON_METRIC_NAMES = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "f1": "F1",
    "roc_auc": "ROC AUC",
}


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


def plot_model_comparison(model_frames: dict[str, dict[str, pd.DataFrame]], run_dir: str | Path) -> None:
    """Compare deep-learning and XGBoost metrics on every frozen split.

    Args:
        model_frames: Row-level predictions by model and data split.
        run_dir: Model-run output directory.
    """
    rows = []
    for model_name, split_frames in model_frames.items():
        for split, frame in split_frames.items():
            metrics = classification_metrics(
                frame[TRUE_CLASS_COL].to_numpy(),
                frame[POSITIVE_PROBABILITY_COL].to_numpy(),
            )
            for metric, display_name in COMPARISON_METRIC_NAMES.items():
                rows.append(
                    {
                        "model": MODEL_DISPLAY_NAMES.get(model_name, model_name),
                        "split": split,
                        "metric": display_name,
                        "score": metrics[metric],
                    }
                )

    sns.set_theme(style="whitegrid", font_scale=1.0)
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    comparison = pd.DataFrame(rows)
    for axis, split in zip(axes, SPLIT_ORDER, strict=True):
        subset = comparison.loc[comparison["split"] == split]
        sns.barplot(data=subset, x="metric", y="score", hue="model", ax=axis)
        axis.set(xlabel="Metric", ylabel="Score", title=split, ylim=(0, 1))
        axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    _save(figure, run_dir, "model_comparison")
