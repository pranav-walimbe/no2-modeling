"""Plotting utilities for masked-pretraining runs."""

from pathlib import Path

import matplotlib.pyplot as plt


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
