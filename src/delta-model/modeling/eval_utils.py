"""Evaluation utilities for effective NOx-change regression."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.nn import functional as F

TRUE_TARGET_COL = "y_true"
PREDICTION_COL = "y_pred"
RESIDUAL_COL = "residual"
ABSOLUTE_ERROR_COL = "absolute_error"
MODEL_COMPARISON_METRICS = ("mse", "mae", "rmse", "r2", "pearson_r", "spearman_r")
DEFAULT_LDS_BINS = 101
DEFAULT_LDS_SIGMA = 2.0
DEFAULT_MAX_WEIGHT = 5.0
DEFAULT_HUBER_DELTA = 0.1


@dataclass(frozen=True)
class LossWeightStats:
    """Serializable description of training-label loss weights."""

    weighting: str
    bin_count: int
    gaussian_sigma_bins: float
    maximum_weight: float
    huber_delta: float
    target_minimum: float
    target_maximum: float
    sample_weight_minimum: float
    sample_weight_median: float
    sample_weight_mean: float
    sample_weight_maximum: float

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe loss-weight metadata."""
        return asdict(self)


class WeightedHuberLoss(nn.Module):
    """Huber loss with train-fitted label-distribution weights."""

    def __init__(
        self,
        training_targets: np.ndarray,
        *,
        weighting: str = "lds_sqrt_inverse",
        bin_count: int = DEFAULT_LDS_BINS,
        gaussian_sigma_bins: float = DEFAULT_LDS_SIGMA,
        maximum_weight: float = DEFAULT_MAX_WEIGHT,
        huber_delta: float = DEFAULT_HUBER_DELTA,
    ) -> None:
        super().__init__()
        targets = np.asarray(training_targets, dtype=np.float64)
        if weighting == "none":
            edges = np.array([-np.inf, np.inf], dtype=np.float64)
            bin_weights = np.ones(1, dtype=np.float64)
        else:
            counts, edges = np.histogram(targets, bins=bin_count)
            effective_counts = gaussian_filter1d(
                counts.astype(np.float64),
                sigma=gaussian_sigma_bins,
                mode="nearest",
            )
            bin_weights = 1.0 / np.sqrt(np.maximum(effective_counts, 1.0))
            target_bins = np.clip(np.digitize(targets, edges[1:-1]), 0, len(bin_weights) - 1)
            bin_weights /= bin_weights[target_bins].mean()
            bin_weights = np.minimum(bin_weights, maximum_weight)

        sample_bins = np.clip(np.digitize(targets, edges[1:-1]), 0, len(bin_weights) - 1)
        sample_weights = bin_weights[sample_bins]
        self.weighting = weighting
        self.huber_delta = huber_delta
        self.register_buffer("internal_edges", torch.tensor(edges[1:-1], dtype=torch.float32))
        self.register_buffer("bin_weights", torch.tensor(bin_weights, dtype=torch.float32))
        self.stats = LossWeightStats(
            weighting=weighting,
            bin_count=len(bin_weights),
            gaussian_sigma_bins=gaussian_sigma_bins,
            maximum_weight=maximum_weight,
            huber_delta=huber_delta,
            target_minimum=float(targets.min()),
            target_maximum=float(targets.max()),
            sample_weight_minimum=float(sample_weights.min()),
            sample_weight_median=float(np.median(sample_weights)),
            sample_weight_mean=float(sample_weights.mean()),
            sample_weight_maximum=float(sample_weights.max()),
        )

    def weights_for(self, targets: torch.Tensor) -> torch.Tensor:
        """Return training-derived loss weights for continuous targets.

        Args:
            targets: Effective emissions-change targets.

        Returns:
            Per-example loss weights.
        """
        indices = torch.bucketize(targets.detach(), self.internal_edges)
        return self.bin_weights[indices]

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return the normalized weighted Huber loss."""
        weights = self.weights_for(targets)
        losses = F.huber_loss(predictions, targets, reduction="none", delta=self.huber_delta)
        return torch.sum(weights * losses) / torch.sum(weights)


def regression_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float | int | None]:
    """Calculate regression errors and associations.

    Args:
        y_true: Ground-truth continuous targets.
        prediction: Continuous model predictions.

    Returns:
        Sample count, error metrics, bias, and correlations.
    """
    truth = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(prediction, dtype=np.float64)
    residual = predicted - truth
    mse = float(mean_squared_error(truth, predicted))
    has_variation = truth.size > 1 and np.std(truth) > 0 and np.std(predicted) > 0
    pearson = float(np.corrcoef(truth, predicted)[0, 1]) if has_variation else None
    spearman = float(pd.Series(truth).corr(pd.Series(predicted), method="spearman")) if has_variation else None
    return {
        "n": int(truth.size),
        "mse": mse,
        "mae": float(mean_absolute_error(truth, predicted)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(truth, predicted)) if truth.size > 1 else None,
        "pearson_r": pearson,
        "spearman_r": spearman,
        "mean_bias": float(residual.mean()),
    }


def plant_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate regression metrics for each held-out AOI.

    Args:
        frame: Row-level regression predictions with AOI coordinates.

    Returns:
        One regression summary row per AOI.
    """
    rows = []
    for (aoi_id, lon, lat), group in frame.groupby(["aoi_id", "lon", "lat"], sort=True):
        metrics = regression_metrics(group[TRUE_TARGET_COL].to_numpy(), group[PREDICTION_COL].to_numpy())
        rows.append({"aoi_id": aoi_id, "lon": lon, "lat": lat, **metrics})
    return pd.DataFrame(rows)


def _model_results(split_frames: dict[str, pd.DataFrame]) -> dict[str, object]:
    # Summarize natural splits and fixed equal-count test magnitude slices
    results: dict[str, object] = {
        "splits": {
            name: regression_metrics(frame[TRUE_TARGET_COL].to_numpy(), frame[PREDICTION_COL].to_numpy())
            for name, frame in split_frames.items()
        }
    }
    test = split_frames["test"]
    ordered_indices = np.argsort(np.abs(test[TRUE_TARGET_COL].to_numpy(dtype=np.float64)), kind="stable")
    slices = {
        name: test.iloc[indices]
        for name, indices in zip(("low", "mid", "high"), np.array_split(ordered_indices, 3), strict=True)
    }
    results["test_absolute_target_tertiles"] = {
        name: regression_metrics(subset[TRUE_TARGET_COL].to_numpy(), subset[PREDICTION_COL].to_numpy())
        for name, subset in slices.items()
    }
    return results


def _model_comparison(
    model_results: dict[str, dict[str, object]],
    primary_model_name: str,
    comparison_model_name: str,
) -> dict[str, object]:
    # Place both scores and signed primary-minus-comparison differences together
    primary = model_results[primary_model_name]["splits"]
    comparison = model_results[comparison_model_name]["splits"]

    def metric_values(split: str, metric: str) -> dict[str, float | None]:
        primary_value = primary[split][metric]
        comparison_value = comparison[split][metric]
        difference = None if primary_value is None or comparison_value is None else primary_value - comparison_value
        return {
            primary_model_name: primary_value,
            comparison_model_name: comparison_value,
            f"{primary_model_name}_minus_{comparison_model_name}": difference,
        }

    return {split: {metric: metric_values(split, metric) for metric in MODEL_COMPARISON_METRICS} for split in primary}


def save_results(
    model_frames: dict[str, dict[str, pd.DataFrame]],
    run_dir: str | Path,
    *,
    primary_model_name: str,
    comparison_model_name: str | None = None,
) -> None:
    """Save regression metrics and row-level predictions.

    Args:
        model_frames: Row-level predictions by model and data split.
        run_dir: Model-run output directory.
        primary_model_name: Model copied into the top-level result summary.
        comparison_model_name: Baseline model used to calculate differences.
    """
    model_results = {name: _model_results(split_frames) for name, split_frames in model_frames.items()}
    primary_results = model_results[primary_model_name]
    results: dict[str, object] = {
        "primary_model": primary_model_name,
        "splits": primary_results["splits"],
        "test_absolute_target_tertiles": primary_results["test_absolute_target_tertiles"],
        "models": model_results,
    }
    if comparison_model_name is not None:
        results["comparison"] = _model_comparison(model_results, primary_model_name, comparison_model_name)

    output_dir = Path(run_dir)
    with (output_dir / "results.json").open("w") as destination:
        json.dump(results, destination, indent=2)
    for model_name, split_frames in model_frames.items():
        for split, frame in split_frames.items():
            filename = (
                f"{split}_predictions.csv"
                if model_name == primary_model_name
                else f"{model_name}_{split}_predictions.csv"
            )
            frame.to_csv(output_dir / filename, index=False)
