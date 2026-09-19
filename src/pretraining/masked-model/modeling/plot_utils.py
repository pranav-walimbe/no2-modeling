"""Plotting utilities for masked-pretraining runs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curve(train_losses: list[float], validation_losses: list[float], run_dir: str | Path) -> None:
    """Save the masked L1 learning curve.

    Args:
        train_losses: Training loss by epoch.
        validation_losses: Validation loss by epoch.
        run_dir: Model-run output directory.
    """
    figure, axis = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    axis.plot(epochs, train_losses, label="train")
    axis.plot(epochs, validation_losses, label="validation")
    axis.set(xlabel="Epoch", ylabel="Masked normalized L1", title="Masked NO2 reconstruction loss")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(Path(run_dir) / "loss_curve.png", dpi=180)
    plt.close(figure)


def plot_results(
    train_losses: list[float],
    validation_losses: list[float],
    results: dict[str, object],
    run_dir: str | Path,
) -> None:
    """Plot training performance and interpolation comparisons.

    Args:
        train_losses: Training loss by epoch.
        validation_losses: Validation loss by epoch.
        results: Reconstruction metrics grouped by evaluation split.
        run_dir: Model-run output directory.
    """
    split_results = results["splits"]
    split_names = tuple(split_results)
    method_names = ("masked_autoencoder", "bilinear_interpolation")
    method_labels = ("Masked model", "Bilinear interpolation")
    colors = ("#3274A1", "#E1812C")
    positions = np.arange(len(split_names))
    width = 0.36
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))

    epochs = range(1, len(train_losses) + 1)
    axes[0, 0].plot(epochs, train_losses, label="Train", color=colors[0])
    axes[0, 0].plot(epochs, validation_losses, label="Validation", color=colors[1])
    axes[0, 0].set(title="Masked L1 learning curve", xlabel="Epoch", ylabel="Normalized L1")
    axes[0, 0].legend()

    for axis, metric, title in (
        (axes[0, 1], "normalized_l1_loss", "Masked-pixel L1"),
        (axes[1, 0], "normalized_rmse", "Masked-pixel RMSE"),
    ):
        for method_index, (method, label, color) in enumerate(zip(method_names, method_labels, colors, strict=True)):
            values = [split_results[split]["models"][method][metric] for split in split_names]
            offset = (method_index - 0.5) * width
            bars = axis.bar(positions + offset, values, width, label=label, color=color)
            axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        axis.set(title=title, ylabel="Normalized error", xticks=positions, xticklabels=split_names)
        axis.legend()

    relative_improvement = [
        100.0 * split_results[split]["comparison"]["normalized_l1_relative_improvement"] for split in split_names
    ]
    improvement_colors = [colors[0] if value >= 0 else "#C44E52" for value in relative_improvement]
    bars = axes[1, 1].bar(positions, relative_improvement, color=improvement_colors, width=0.55)
    axes[1, 1].bar_label(bars, fmt="%.1f%%", padding=3)
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set(
        title="L1 improvement over bilinear interpolation",
        ylabel="Relative improvement (%)",
        xticks=positions,
        xticklabels=split_names,
    )

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Masked NO2 reconstruction results", fontsize=15)
    figure.tight_layout()
    figure.savefig(Path(run_dir) / "results.png", dpi=180)
    plt.close(figure)
