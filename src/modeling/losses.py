"""Training-loss weighting for imbalanced signed regression labels."""

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


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
