"""Loss weighting for skewed continuous emissions targets."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from torch import nn
from torch.nn import functional as F

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
