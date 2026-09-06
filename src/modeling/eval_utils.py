"""Loss weighting and evaluation utilities for signed NOx-mass changes."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

NORMALIZED_TRUE_COL = "y_true"
NORMALIZED_PRED_COL = "y_pred"
MASS_TRUE_COL = "delta_nox_mass_true"
MASS_PRED_COL = "delta_nox_mass_pred"


@dataclass(frozen=True)
class HistogramWeightConfig:
    """Serializable histogram edges, frequencies, and loss weights."""

    label: str
    units: str
    bin_edges: tuple[float, ...]
    bin_counts: tuple[int, ...]
    bin_weights: tuple[float, ...]
    weight_cap: float
    training_records: int

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe loss-weighting state."""
        return asdict(self)


def build_histogram_weight_config(
    labels: np.ndarray,
    bin_count: int,
    weight_cap: float,
) -> HistogramWeightConfig:
    """Estimate capped inverse-frequency weights from signed training labels.

    Args:
        labels: One-dimensional training labels in their stored units.
        bin_count: Total number of signed histogram bins.
        weight_cap: Maximum per-record loss multiplier.

    Returns:
        Frozen bin edges, counts, and inverse-frequency weights.
    """
    values = np.asarray(labels, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Training labels must be a nonempty finite vector")
    if bin_count < 2 or bin_count % 2:
        raise ValueError("Signed label bin count must be an even integer of at least two")
    if not np.isfinite(weight_cap) or weight_cap <= 0:
        raise ValueError("Label weight cap must be finite and positive")
    if values.min() >= 0 or values.max() <= 0:
        raise ValueError("Signed label weighting requires both negative and positive training labels")

    bins_per_sign = bin_count // 2
    negative_edges = np.linspace(values.min(), 0.0, bins_per_sign + 1)
    positive_edges = np.linspace(0.0, values.max(), bins_per_sign + 1)
    edges = np.concatenate((negative_edges, positive_edges[1:]))
    counts, _ = np.histogram(values, bins=edges)
    occupied = counts > 0
    weights = np.zeros(bin_count, dtype=np.float64)
    weights[occupied] = values.size / (occupied.sum() * counts[occupied])
    np.minimum(weights, weight_cap, out=weights)
    return HistogramWeightConfig(
        label="delta_nox_norm",
        units="delta_nox_norm",
        bin_edges=tuple(float(value) for value in edges),
        bin_counts=tuple(int(value) for value in counts),
        bin_weights=tuple(float(value) for value in weights),
        weight_cap=float(weight_cap),
        training_records=int(values.size),
    )


class HistogramWeightedHuberLoss(nn.Module):
    """Apply frozen histogram weights to per-record Huber losses."""

    def __init__(
        self,
        config: HistogramWeightConfig,
        delta: float,
        *,
        target_mean: float = 0.0,
        target_std: float = 1.0,
    ) -> None:
        super().__init__()
        if delta <= 0:
            raise ValueError("Huber delta must be positive")
        if not np.isfinite(target_mean) or not np.isfinite(target_std) or target_std <= 0:
            raise ValueError("Target normalization must be finite with a positive standard deviation")
        self.delta = delta
        standardized_edges = (np.asarray(config.bin_edges) - target_mean) / target_std
        self.register_buffer("boundaries", torch.tensor(standardized_edges[1:-1], dtype=torch.float32))
        self.register_buffer("weights", torch.tensor(config.bin_weights, dtype=torch.float32))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return the mean weighted Huber loss for a batch."""
        bin_indices = torch.bucketize(target.detach(), self.boundaries, right=True)
        losses = F.huber_loss(prediction, target, delta=self.delta, reduction="none")
        return (losses * self.weights[bin_indices]).mean()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int | None]:
    """Return stable metrics that remain meaningful for signed targets."""
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    if truth.shape != prediction.shape or truth.ndim != 1 or not truth.size:
        raise ValueError("Regression metric inputs must be non-empty one-dimensional arrays with equal shape")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Regression metric inputs must be finite")

    residual = prediction - truth
    sum_squared_error = float(np.square(residual).sum())
    total_sum_squares = float(np.square(truth - truth.mean()).sum())
    correlation = float(np.corrcoef(truth, prediction)[0, 1]) if truth.std() > 0 and prediction.std() > 0 else None
    return {
        "n": int(truth.size),
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(sum_squared_error / truth.size)),
        "bias": float(residual.mean()),
        "r2": float(1.0 - sum_squared_error / total_sum_squares) if total_sum_squares > 0 else None,
        "pearson_r": correlation,
    }


def add_mass_change_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Invert the normalized target using each record's historical scale."""
    required = {"delta_nox_scale", "delta_nox_mass", NORMALIZED_TRUE_COL, NORMALIZED_PRED_COL}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction frame is missing inverse-transform columns: {', '.join(sorted(missing))}")
    output = frame.copy()
    scale = pd.to_numeric(output["delta_nox_scale"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("delta_nox_scale must be finite and positive")
    output[MASS_TRUE_COL] = pd.to_numeric(output["delta_nox_mass"], errors="raise").to_numpy(dtype=np.float64)
    output[MASS_PRED_COL] = scale * np.sinh(output[NORMALIZED_PRED_COL].to_numpy(dtype=np.float64))
    if not np.isfinite(output[[MASS_TRUE_COL, MASS_PRED_COL]].to_numpy()).all():
        raise ValueError("Inverse target transformation produced non-finite NOx mass changes")
    return output


def plant_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute normalized and physical errors for each held-out AOI."""
    required = {"aoi_id", "lon", "lat", NORMALIZED_TRUE_COL, NORMALIZED_PRED_COL, MASS_TRUE_COL, MASS_PRED_COL}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction frame is missing plant-metric columns: {', '.join(sorted(missing))}")

    rows = []
    for (aoi_id, lon, lat), group in frame.groupby(["aoi_id", "lon", "lat"], sort=True):
        normalized = regression_metrics(group[NORMALIZED_TRUE_COL].to_numpy(), group[NORMALIZED_PRED_COL].to_numpy())
        physical = regression_metrics(group[MASS_TRUE_COL].to_numpy(), group[MASS_PRED_COL].to_numpy())
        rows.append(
            {
                "aoi_id": aoi_id,
                "lon": lon,
                "lat": lat,
                "n": len(group),
                "normalized_mae": normalized["mae"],
                "mass_change_mae": physical["mae"],
                "mass_change_bias": physical["bias"],
            }
        )
    return pd.DataFrame(rows)


def _split_metrics(frame: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    return {
        "normalized_target": regression_metrics(
            frame[NORMALIZED_TRUE_COL].to_numpy(), frame[NORMALIZED_PRED_COL].to_numpy()
        ),
        "nox_mass_change": regression_metrics(frame[MASS_TRUE_COL].to_numpy(), frame[MASS_PRED_COL].to_numpy()),
    }


def save_results(
    split_frames: dict[str, pd.DataFrame],
    run_dir: str | Path,
    *,
    train_target_mean: float,
) -> None:
    """Save metrics, magnitude slices, baselines, and row-level predictions."""
    results: dict[str, object] = {"splits": {name: _split_metrics(frame) for name, frame in split_frames.items()}}
    baselines: dict[str, object] = {}
    for split, frame in split_frames.items():
        truth = frame[NORMALIZED_TRUE_COL].to_numpy(dtype=np.float64)
        mass_truth = frame[MASS_TRUE_COL].to_numpy(dtype=np.float64)
        scale = frame["delta_nox_scale"].to_numpy(dtype=np.float64)
        baselines[split] = {}
        for name, normalized_prediction in {
            "zero_change": np.zeros_like(truth),
            "train_mean": np.full_like(truth, train_target_mean),
        }.items():
            baselines[split][name] = {
                "normalized_target": regression_metrics(truth, normalized_prediction),
                "nox_mass_change": regression_metrics(
                    mass_truth,
                    scale * np.sinh(normalized_prediction),
                ),
            }
    results["baselines"] = baselines

    test = split_frames["test"]
    magnitude = np.abs(test[NORMALIZED_TRUE_COL].to_numpy(dtype=np.float64))
    if len(test) < 3:
        raise ValueError("At least three test records are required for magnitude-sliced evaluation")
    ordered_indices = np.argsort(magnitude, kind="stable")
    slices = {
        name: test.iloc[indices] for name, indices in zip(("low", "mid", "high"), np.array_split(ordered_indices, 3))
    }
    results["test_absolute_magnitude_tertiles"] = {
        name: {
            **_split_metrics(subset),
            "absolute_normalized_range": [
                float(np.abs(subset[NORMALIZED_TRUE_COL]).min()),
                float(np.abs(subset[NORMALIZED_TRUE_COL]).max()),
            ],
        }
        for name, subset in slices.items()
    }

    output_dir = Path(run_dir)
    with (output_dir / "results.json").open("w") as destination:
        json.dump(results, destination, indent=2)
    for split, frame in split_frames.items():
        frame.to_csv(output_dir / f"{split}_predictions.csv", index=False)
