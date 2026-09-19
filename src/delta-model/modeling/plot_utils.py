"""Plotting utilities for effective NOx-change regression."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from modeling.eval_utils import PREDICTION_COL, TRUE_TARGET_COL, regression_metrics

SPLIT_ORDER = ("train", "val", "test")
MODEL_DISPLAY_NAMES = {"raster_convgru": "Raster ConvGRU", "mlp": "MLP"}
COMPARISON_METRIC_NAMES = {"mae": "MAE", "rmse": "RMSE", "r2": "R2", "pearson_r": "Pearson r"}


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
    """Plot weighted Huber loss across epochs.

    Args:
        train_losses: Mean training loss for each epoch.
        val_losses: Mean validation loss for each epoch.
        run_dir: Model-run output directory.
        plot_name: Output filename without an extension.
        title: Plot title.
    """
    sns.set_theme(style="whitegrid", font_scale=1.2)
    figure, axis = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    axis.plot(epochs, train_losses, label="Train", linewidth=2)
    axis.plot(epochs, val_losses, label="Validation", linewidth=2)
    axis.set(xlabel="Epoch", ylabel="Weighted Huber loss", title=title)
    axis.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axis.legend()
    figure.tight_layout()
    _save(figure, run_dir, plot_name)


def plot_regression_predictions(split_frames: dict[str, pd.DataFrame], run_dir: str | Path) -> None:
    """Plot predicted against observed targets for every split.

    Args:
        split_frames: Row-level predictions for each data split.
        run_dir: Model-run output directory.
    """
    sns.set_theme(style="whitegrid", font_scale=1.0)
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), sharex=True, sharey=True)
    combined = pd.concat(split_frames.values(), ignore_index=True)
    limit = float(combined[[TRUE_TARGET_COL, PREDICTION_COL]].abs().quantile(0.995).max())
    for axis, split in zip(axes, SPLIT_ORDER, strict=True):
        frame = split_frames[split]
        axis.hexbin(frame[TRUE_TARGET_COL], frame[PREDICTION_COL], gridsize=35, mincnt=1, cmap="viridis")
        axis.plot((-limit, limit), (-limit, limit), color="#222222", linewidth=1.1, linestyle="--")
        axis.set(
            xlim=(-limit, limit),
            ylim=(-limit, limit),
            xlabel="Observed target",
            ylabel="Predicted target",
            title=split,
        )
    figure.tight_layout()
    _save(figure, run_dir, "regression_predictions")


def plot_model_comparison(model_frames: dict[str, dict[str, pd.DataFrame]], run_dir: str | Path) -> None:
    """Compare regression metrics for each model and split.

    Args:
        model_frames: Row-level predictions by model and data split.
        run_dir: Model-run output directory.
    """
    rows = []
    for model_name, split_frames in model_frames.items():
        for split, frame in split_frames.items():
            metrics = regression_metrics(frame[TRUE_TARGET_COL].to_numpy(), frame[PREDICTION_COL].to_numpy())
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
    figure, axes = plt.subplots(1, 4, figsize=(20, 5))
    comparison = pd.DataFrame(rows)
    for axis, (metric, display_name) in zip(axes, COMPARISON_METRIC_NAMES.items(), strict=True):
        subset = comparison.loc[comparison["metric"] == display_name]
        sns.barplot(data=subset, x="split", y="score", hue="model", ax=axis)
        axis.set(xlabel="Split", ylabel=display_name, title=display_name)
        if metric in {"r2", "pearson_r"}:
            axis.axhline(0, color="#222222", linewidth=0.8)
        axis.legend().set_visible(axis is axes[-1])
    figure.tight_layout()
    _save(figure, run_dir, "model_comparison")
