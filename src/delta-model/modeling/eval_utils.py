"""Evaluation utilities for effective NOx-change regression."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from modeling.convgru import HurdleOutput
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from torch import nn
from torch.nn import functional as F

TRUE_TARGET_COL = "y_true"
PREDICTION_COL = "y_pred"
RESIDUAL_COL = "residual"
ABSOLUTE_ERROR_COL = "absolute_error"
TRUE_CLASS_COL = "hurdle_class_true"
PREDICTED_CLASS_COL = "hurdle_class_predicted"
DECREASE_PROBABILITY_COL = "probability_decrease"
STEADY_PROBABILITY_COL = "probability_steady"
INCREASE_PROBABILITY_COL = "probability_increase"
DECREASE_MAGNITUDE_COL = "predicted_decrease_magnitude"
INCREASE_MAGNITUDE_COL = "predicted_increase_magnitude"
HURDLE_CLASS_NAMES = ("decrease", "steady", "increase")
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


class HurdleLoss(nn.Module):
    """Class-balanced gate loss with LDS-weighted conditional magnitude losses."""

    def __init__(
        self,
        training_targets: np.ndarray,
        *,
        steady_threshold: float,
        classification_weight: float,
        regression_weight: float,
        weighting: str,
        bin_count: int,
        gaussian_sigma_bins: float,
        maximum_weight: float,
        huber_delta: float,
    ) -> None:
        super().__init__()
        targets = np.asarray(training_targets, dtype=np.float64)
        classes = hurdle_classes(targets, steady_threshold)
        class_counts = np.bincount(classes, minlength=len(HURDLE_CLASS_NAMES))
        class_weights = 1.0 / np.sqrt(class_counts.astype(np.float64))
        class_weights /= np.average(class_weights, weights=class_counts)
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        self.steady_threshold = steady_threshold
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        self.regression_scale = 1.0 / (huber_delta * huber_delta)
        loss_options = {
            "weighting": weighting,
            "bin_count": bin_count,
            "gaussian_sigma_bins": gaussian_sigma_bins,
            "maximum_weight": maximum_weight,
            "huber_delta": huber_delta,
        }
        self.decrease_loss = WeightedHuberLoss(np.abs(targets[classes == 0]), **loss_options)
        self.increase_loss = WeightedHuberLoss(targets[classes == 2], **loss_options)
        self.stats = {
            "steady_threshold": steady_threshold,
            "classification_weight": classification_weight,
            "regression_weight": regression_weight,
            "regression_scale": self.regression_scale,
            "class_names": list(HURDLE_CLASS_NAMES),
            "class_counts": class_counts.tolist(),
            "class_weights": class_weights.tolist(),
            "decrease_magnitude_loss": self.decrease_loss.stats.to_dict(),
            "increase_magnitude_loss": self.increase_loss.stats.to_dict(),
        }

    def weights_for(self, targets: torch.Tensor) -> torch.Tensor:
        """Return one aggregation weight per training record."""
        return torch.ones_like(targets)

    def forward(self, output: HurdleOutput, targets: torch.Tensor) -> torch.Tensor:
        """Return the joint gate and conditional magnitude objective."""
        classes = torch.where(
            targets < -self.steady_threshold,
            torch.zeros_like(targets, dtype=torch.long),
            torch.where(
                targets > self.steady_threshold,
                torch.full_like(targets, 2, dtype=torch.long),
                torch.ones_like(targets, dtype=torch.long),
            ),
        )
        classification_loss = F.cross_entropy(output.class_logits, classes, weight=self.class_weights)
        directional_losses = []
        decrease = classes == 0
        increase = classes == 2
        if decrease.any():
            directional_losses.append(self.decrease_loss(output.magnitudes[decrease, 0], targets[decrease].abs()))
        if increase.any():
            directional_losses.append(self.increase_loss(output.magnitudes[increase, 1], targets[increase]))
        regression_loss = (
            torch.stack(directional_losses).mean() if directional_losses else output.magnitudes.sum() * 0.0
        )
        return (
            self.classification_weight * classification_loss
            + self.regression_weight * self.regression_scale * regression_loss
        )


def hurdle_classes(targets: np.ndarray, steady_threshold: float) -> np.ndarray:
    """Map signed targets to decrease, steady, and increase class indices."""
    values = np.asarray(targets)
    return np.where(values < -steady_threshold, 0, np.where(values > steady_threshold, 2, 1)).astype(np.int64)


def hurdle_metrics(frame: pd.DataFrame) -> dict[str, object]:
    """Calculate gate metrics from a hurdle prediction frame."""
    truth = frame[TRUE_CLASS_COL].to_numpy(dtype=np.int64)
    predicted = frame[PREDICTED_CLASS_COL].to_numpy(dtype=np.int64)
    return {
        "accuracy": float(np.mean(truth == predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(
            truth,
            predicted,
            labels=np.arange(len(HURDLE_CLASS_NAMES)),
        ).tolist(),
    }


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
    if {TRUE_CLASS_COL, PREDICTED_CLASS_COL}.issubset(test.columns):
        results["hurdle_gate"] = {name: hurdle_metrics(frame) for name, frame in split_frames.items()}
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
