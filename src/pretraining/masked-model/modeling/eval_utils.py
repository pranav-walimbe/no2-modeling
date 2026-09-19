"""Evaluation utilities for masked NO2 reconstruction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import MODEL_IMAGE_CHANNELS

from .dataset import MaskedNormalizationStats


def bilinear_interpolation_baseline(image: torch.Tensor) -> torch.Tensor:
    """Fill missing NO2 by separable linear interpolation on the raster grid.

    Args:
        image: CPU tensor containing normalized image channels and visible mask.

    Returns:
        Complete normalized NO2 raster.
    """
    no2 = image[:, 0].numpy()
    visible = image[:, MODEL_IMAGE_CHANNELS].numpy().astype(bool)
    predictions = no2.copy()
    rows = np.arange(no2.shape[1])
    columns = np.arange(no2.shape[2])

    for sample in range(no2.shape[0]):
        fallback = float(no2[sample][visible[sample]].mean())
        horizontal = np.empty_like(no2[sample])
        vertical = np.empty_like(no2[sample])
        for row in rows:
            support = visible[sample, row]
            horizontal[row] = (
                np.interp(columns, columns[support], no2[sample, row, support]) if support.any() else fallback
            )
        for column in columns:
            support = visible[sample, :, column]
            vertical[:, column] = (
                np.interp(rows, rows[support], no2[sample, support, column]) if support.any() else fallback
            )
        missing = ~visible[sample]
        predictions[sample, missing] = 0.5 * (horizontal[missing] + vertical[missing])
    return torch.from_numpy(predictions[:, None])


def evaluate_reconstruction(
    model: nn.Module,
    loader: DataLoader,
    stats: MaskedNormalizationStats,
    device: torch.device,
) -> dict[str, object]:
    """Compare model reconstruction with naive bilinear interpolation.

    Args:
        model: Trained reconstruction model.
        loader: Evaluation data loader.
        stats: Train-split normalization state.
        device: Inference device.

    Returns:
        Normalized and physical-unit metrics for both methods.
    """
    totals = {
        name: {"absolute": 0.0, "squared": 0.0, "signed": 0.0}
        for name in ("masked_autoencoder", "bilinear_interpolation")
    }
    masked_pixels = 0
    model.eval()
    amp_enabled = device.type == "cuda"
    with torch.inference_mode():
        for image, target, loss_mask, _ in loader:
            baseline = bilinear_interpolation_baseline(image)
            image_device = image.to(device, non_blocking=amp_enabled)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                prediction = model(image_device)
            prediction = prediction.float().cpu()
            for name, values in (
                ("masked_autoencoder", prediction),
                ("bilinear_interpolation", baseline),
            ):
                error = (values - target) * loss_mask
                totals[name]["absolute"] += error.abs().sum().item()
                totals[name]["squared"] += error.square().sum().item()
                totals[name]["signed"] += error.sum().item()
            masked_pixels += int(loss_mask.sum().item())

    no2_scale = stats.image_scale[0]
    metrics: dict[str, object] = {"masked_pixels": masked_pixels, "models": {}}
    for name, values in totals.items():
        normalized_mae = values["absolute"] / masked_pixels
        normalized_rmse = (values["squared"] / masked_pixels) ** 0.5
        normalized_bias = values["signed"] / masked_pixels
        metrics["models"][name] = {
            "normalized_l1_loss": normalized_mae,
            "normalized_mae": normalized_mae,
            "normalized_rmse": normalized_rmse,
            "normalized_bias": normalized_bias,
            "physical_mae": normalized_mae * no2_scale,
            "physical_rmse": normalized_rmse * no2_scale,
            "physical_bias": normalized_bias * no2_scale,
        }

    model_loss = metrics["models"]["masked_autoencoder"]["normalized_l1_loss"]
    baseline_loss = metrics["models"]["bilinear_interpolation"]["normalized_l1_loss"]
    metrics["comparison"] = {
        "normalized_l1_improvement": baseline_loss - model_loss,
        "normalized_l1_relative_improvement": (baseline_loss - model_loss) / baseline_loss,
    }
    return metrics


def save_results(results: dict[str, object], run_dir: str | Path) -> None:
    """Save reconstruction evaluation results.

    Args:
        results: Metrics grouped by data split.
        run_dir: Model-run output directory.
    """
    with (Path(run_dir) / "results.json").open("w") as destination:
        json.dump(results, destination, indent=2)
