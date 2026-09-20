"""Plotting utilities for emissions-change classification."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from modeling.eval_utils import COMPARISON_METRICS, classification_metrics

from config import MODEL_CLASS_NAMES

MODEL_DISPLAY_NAMES = {
    "raster_partial_convgru": "Partial-convolution ConvGRU",
    "mlp": "Tabular MLP",
}
METRIC_DISPLAY_NAMES = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "macro_f1": "Macro F1",
}


def _save(figure: plt.Figure, run_dir: str | Path, plot_name: str) -> None:
    # Persist and close one completed plot
    figure.savefig(Path(run_dir) / f"{plot_name}.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_loss_curve(
    train_losses: list[float],
    val_losses: list[float],
    run_dir: str | Path,
    *,
    plot_name: str = "loss_curve",
    title: str = "Training and validation loss",
) -> None:
    """Plot cross-entropy loss across epochs."""
    sns.set_theme(style="whitegrid", font_scale=1.2)
    figure, axis = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    axis.plot(epochs, train_losses, label="Train", linewidth=2)
    axis.plot(epochs, val_losses, label="Validation", linewidth=2)
    axis.set(xlabel="Epoch", ylabel="Cross-entropy", title=title)
    axis.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axis.legend()
    figure.tight_layout()
    _save(figure, run_dir, plot_name)


def plot_model_comparison(model_frames: dict[str, dict[str, pd.DataFrame]], run_dir: str | Path) -> None:
    """Compare classification metrics for each model and split."""
    rows = []
    for model_name, split_frames in model_frames.items():
        for split, frame in split_frames.items():
            metrics = classification_metrics(frame)
            for metric in COMPARISON_METRICS:
                rows.append(
                    {
                        "model": MODEL_DISPLAY_NAMES.get(model_name, model_name),
                        "split": split,
                        "metric": METRIC_DISPLAY_NAMES[metric],
                        "score": metrics[metric],
                    }
                )

    sns.set_theme(style="whitegrid", font_scale=1.0)
    figure, axes = plt.subplots(1, len(COMPARISON_METRICS), figsize=(15, 5), sharey=True)
    axes = np.atleast_1d(axes)
    comparison = pd.DataFrame(rows)
    for axis, metric in zip(axes, COMPARISON_METRICS, strict=True):
        display_name = METRIC_DISPLAY_NAMES[metric]
        subset = comparison.loc[comparison["metric"] == display_name]
        sns.barplot(data=subset, x="split", y="score", hue="model", ax=axis)
        axis.set(xlabel="Split", ylabel=display_name, title=display_name, ylim=(0, 1))
        axis.legend().set_visible(axis is axes[-1])
    figure.tight_layout()
    _save(figure, run_dir, "model_comparison")


def plot_confusion_matrices(model_frames: dict[str, dict[str, pd.DataFrame]], run_dir: str | Path) -> None:
    """Plot row-normalized test confusion matrices for both models."""
    sns.set_theme(style="whitegrid", font_scale=1.0)
    figure, axes = plt.subplots(1, len(model_frames), figsize=(6.5 * len(model_frames), 5.5))
    for axis, (model_name, split_frames) in zip(np.atleast_1d(axes), model_frames.items(), strict=True):
        metrics = classification_metrics(split_frames["test"])
        confusion = np.asarray(metrics["confusion_matrix"], dtype=np.float64)
        normalized = np.divide(
            confusion,
            confusion.sum(axis=1, keepdims=True),
            out=np.zeros_like(confusion),
            where=confusion.sum(axis=1, keepdims=True) > 0,
        )
        sns.heatmap(
            normalized,
            annot=True,
            fmt=".1%",
            cmap="Blues",
            vmin=0,
            vmax=1,
            xticklabels=MODEL_CLASS_NAMES,
            yticklabels=MODEL_CLASS_NAMES,
            ax=axis,
        )
        axis.set(
            xlabel="Predicted class",
            ylabel="True class",
            title=f"{MODEL_DISPLAY_NAMES.get(model_name, model_name)}\nBalanced accuracy {metrics['balanced_accuracy']:.3f}",
        )
    figure.tight_layout()
    _save(figure, run_dir, "classification_confusion")
