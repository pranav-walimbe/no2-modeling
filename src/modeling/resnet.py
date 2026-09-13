"""Compact residual network for emissions-change classification."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from config import MODEL_IMAGE_CHANNELS, MODEL_MASK_KEYS

DEFAULT_HEAD_DIM = 128
DEFAULT_DROPOUT = 0.30
DEFAULT_AMPLITUDE_HIDDEN_DIM = 32
DEFAULT_AMPLITUDE_DIM = 16
AMPLITUDE_TAIL_FRACTION = 0.01
AMPLITUDE_STATISTIC_NAMES = (
    "masked_mean",
    "robust_scale",
    "root_mean_square",
    "upper_tail_mean",
    "lower_tail_mean",
)


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            _group_norm(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _group_norm(out_channels),
        )
        self.residual = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                _group_norm(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.block(inputs) + self.residual(inputs))


class PartialConv2d(nn.Module):
    """Convolve valid values and renormalize for local mask support."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.padding = kernel_size // 2
        self.kernel_area = kernel_size * kernel_size
        self.convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=False,
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer("mask_kernel", torch.ones(1, 1, kernel_size, kernel_size))

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply a mask-aware convolution.

        Args:
            inputs: Dense feature tensor with invalid positions set to zero.
            mask: One-channel binary validity tensor.

        Returns:
            Renormalized features and the propagated binary mask.
        """
        masked = inputs * mask
        features = self.convolution(masked)
        with torch.no_grad():
            support = F.conv2d(mask, self.mask_kernel, padding=self.padding)
            next_mask = (support > 0).to(inputs.dtype)
            scale = self.kernel_area / support.clamp_min(1.0)
        features = (features * scale + self.bias[None, :, None, None]) * next_mask
        return features, next_mask


class MaskedNO2Stem(nn.Module):
    """Extract one NO2 channel without treating missing cells as observations."""

    def __init__(self, out_channels: int = 12) -> None:
        super().__init__()
        self.first = PartialConv2d(1, out_channels, kernel_size=5)
        self.first_norm = _group_norm(out_channels)
        self.second = PartialConv2d(out_channels, out_channels, kernel_size=3)
        self.second_norm = _group_norm(out_channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode one value-mask pair and propagate local validity."""
        features, mask = self.first(values, mask)
        features = self.activation(self.first_norm(features)) * mask
        features, mask = self.second(features, mask)
        return self.activation(self.second_norm(features)) * mask


def _masked_amplitude_statistics(values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    # Summarize physical amplitude before sample-wise activation normalization
    valid = masks > 0
    flattened = values.flatten(2)
    flattened_valid = valid.flatten(2)
    counts = flattened_valid.sum(dim=2)
    safe_counts = counts.clamp_min(1)
    masked_values = torch.where(flattened_valid, flattened, torch.zeros_like(flattened))
    mean = masked_values.sum(dim=2) / safe_counts
    root_mean_square = torch.sqrt(masked_values.square().sum(dim=2) / safe_counts)

    nan_masked = flattened.masked_fill(~flattened_valid, torch.nan)
    median = torch.nanmedian(nan_masked, dim=2).values
    absolute_deviation = torch.abs(flattened - median.unsqueeze(2)).masked_fill(~flattened_valid, torch.nan)
    robust_scale = 1.4826 * torch.nanmedian(absolute_deviation, dim=2).values

    pixel_count = flattened.shape[2]
    tail_count = max(1, math.ceil(AMPLITUDE_TAIL_FRACTION * pixel_count))
    upper_values = torch.topk(flattened.masked_fill(~flattened_valid, -torch.inf), tail_count, dim=2).values
    lower_values = torch.topk(flattened.masked_fill(~flattened_valid, torch.inf), tail_count, dim=2, largest=False).values

    def finite_mean(selected: torch.Tensor) -> torch.Tensor:
        finite = torch.isfinite(selected)
        return torch.where(finite, selected, torch.zeros_like(selected)).sum(dim=2) / finite.sum(dim=2).clamp_min(1)

    statistics = torch.stack(
        (mean, robust_scale, root_mean_square, finite_mean(upper_values), finite_mean(lower_values)),
        dim=2,
    )
    return torch.where(counts.unsqueeze(2) > 0, statistics, torch.zeros_like(statistics)).flatten(1)


class MaskedAmplitudeEncoder(nn.Module):
    """Preserve scene-level NO2 amplitude outside the GroupNorm path."""

    def __init__(
        self,
        channels: int,
        hidden_dim: int = DEFAULT_AMPLITUDE_HIDDEN_DIM,
        output_dim: int = DEFAULT_AMPLITUDE_DIM,
    ) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(channels * len(AMPLITUDE_STATISTIC_NAMES), hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Encode masked pre-GroupNorm amplitude summaries.

        Args:
            values: Train-normalized NO2 rasters before activation normalization.
            masks: Binary validity masks aligned with the NO2 rasters.

        Returns:
            Learned scene-amplitude embedding.
        """
        return self.projection(_masked_amplitude_statistics(values, masks))


class NOxModel(nn.Module):
    """Fuse TEMPO and wind rasters with leakage-safe scalar features."""

    def __init__(
        self,
        n_tabular_features: int,
        *,
        use_image: bool = True,
        use_tabular: bool = True,
        head_dim: int = DEFAULT_HEAD_DIM,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.use_image = use_image
        self.use_tabular = use_tabular

        if use_image:
            self.no2_stems = nn.ModuleList(MaskedNO2Stem() for _ in MODEL_MASK_KEYS)
            self.amplitude_encoder = MaskedAmplitudeEncoder(len(MODEL_MASK_KEYS))
            self.wind_stem = nn.Sequential(
                nn.Conv2d(MODEL_IMAGE_CHANNELS - len(MODEL_MASK_KEYS), 16, kernel_size=5, padding=2, bias=False),
                _group_norm(16),
                nn.SiLU(inplace=True),
            )
            self.stem_fusion = nn.Sequential(
                nn.Conv2d(12 * len(MODEL_MASK_KEYS) + 16, 32, kernel_size=1, bias=False),
                _group_norm(32),
                nn.SiLU(inplace=True),
            )
            self.encoder = nn.Sequential(
                ResBlock(32, 32),
                ResBlock(32, 64, stride=2),
                ResBlock(64, 64),
                ResBlock(64, 128, stride=2),
                ResBlock(128, 128),
                ResBlock(128, 192, stride=2),
                ResBlock(192, 192),
            )

            # A 3x3 average summary retains coarse plume position; a global
            # maximum preserves a localized enhancement an average can dilute.
            self.spatial_pool = nn.AdaptiveAvgPool2d((3, 3))
            self.peak_pool = nn.AdaptiveMaxPool2d((1, 1))
            self.image_projection = nn.Sequential(
                nn.Linear(192 * 10, 256),
                nn.LayerNorm(256),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
            )
        if use_tabular:
            self.tabular_projection = nn.Sequential(
                nn.Linear(n_tabular_features, 64),
                nn.LayerNorm(64),
                nn.SiLU(inplace=True),
                nn.Linear(64, 64),
                nn.SiLU(inplace=True),
            )
        fusion_features = 256 * int(use_image) + 64 * int(use_tabular)
        self.head_projection = nn.Sequential(
            nn.Linear(fusion_features, head_dim),
            nn.LayerNorm(head_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        classifier_features = head_dim + DEFAULT_AMPLITUDE_DIM * int(use_image)
        self.classifier = nn.Linear(classifier_features, 1)

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        features = []
        amplitude = None
        if self.use_image:
            masks = image[:, MODEL_IMAGE_CHANNELS:]
            amplitude = self.amplitude_encoder(image[:, : len(MODEL_MASK_KEYS)], masks)
            no2_features = [
                stem(image[:, channel : channel + 1], masks[:, channel : channel + 1])
                for channel, stem in enumerate(self.no2_stems)
            ]
            wind_features = self.wind_stem(image[:, len(MODEL_MASK_KEYS) : MODEL_IMAGE_CHANNELS])
            encoded = self.encoder(self.stem_fusion(torch.cat((*no2_features, wind_features), dim=1)))
            spatial = self.spatial_pool(encoded).flatten(1)
            peak = self.peak_pool(encoded).flatten(1)
            features.append(self.image_projection(torch.cat((spatial, peak), dim=1)))
        if self.use_tabular:
            features.append(self.tabular_projection(tabular))
        hidden = self.head_projection(torch.cat(features, dim=1))
        if amplitude is not None:
            hidden = torch.cat((hidden, amplitude), dim=1)
        return self.classifier(hidden).squeeze(1)

    def num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
