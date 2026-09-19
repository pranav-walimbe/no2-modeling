"""Mask-aware ConvGRU for emissions-change prediction."""

import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from config import MODEL_IMAGE_CHANNELS

DEFAULT_HEAD_DIM = 128
DEFAULT_DROPOUT = 0.20
VISION_EMBEDDING_DIM = 128
CONVGRU_HIDDEN_CHANNELS = 96
HURDLE_CLASS_COUNT = 3
HURDLE_MAGNITUDE_COUNT = 2


class HurdleOutput(NamedTuple):
    """Outputs from the directional hurdle heads."""

    class_logits: torch.Tensor
    class_probabilities: torch.Tensor
    magnitudes: torch.Tensor
    expected_value: torch.Tensor


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualBlock(nn.Module):
    """Residual spatial block with optional downsampling."""

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
        masked = inputs * mask
        features = self.convolution(masked)
        with torch.no_grad():
            support = F.conv2d(mask, self.mask_kernel, padding=self.padding)
            next_mask = (support > 0).to(inputs.dtype)
            scale = self.kernel_area / support.clamp_min(1.0)
        features = (features * scale + self.bias[None, :, None, None]) * next_mask
        return features, next_mask


class MaskedNO2Stem(nn.Module):
    """Encode NO2 without interpreting missing pixels as physical zeros."""

    def __init__(self, out_channels: int = 16) -> None:
        super().__init__()
        self.first = PartialConv2d(1, out_channels, kernel_size=5)
        self.first_norm = _group_norm(out_channels)
        self.second = PartialConv2d(out_channels, out_channels, kernel_size=3)
        self.second_norm = _group_norm(out_channels)
        self.activation = nn.SiLU()

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        features, mask = self.first(values, mask)
        features = self.activation(self.first_norm(features)) * mask
        features, mask = self.second(features, mask)
        return self.activation(self.second_norm(features)) * mask


class ConvGRUCell(nn.Module):
    """Spatial gated recurrent unit that preserves a feature map over time."""

    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        combined_channels = input_channels + hidden_channels
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(combined_channels, 2 * hidden_channels, kernel_size=3, padding=1)
        self.candidate = nn.Conv2d(combined_channels, hidden_channels, kernel_size=3, padding=1)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        reset, update = self.gates(torch.cat((inputs, hidden), dim=1)).sigmoid().chunk(2, dim=1)
        candidate = self.candidate(torch.cat((inputs, reset * hidden), dim=1)).tanh()
        return (1.0 - update) * hidden + update * candidate


class RasterConvGRU(nn.Module):
    """Regress effective emissions changes from raster sequences alone."""

    def __init__(
        self,
        *,
        head_dim: int = DEFAULT_HEAD_DIM,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.no2_stem = MaskedNO2Stem(out_channels=16)
        self.weather_stem = nn.Sequential(
            nn.Conv2d(MODEL_IMAGE_CHANNELS - 1, 16, kernel_size=5, padding=2, bias=False),
            _group_norm(16),
            nn.SiLU(inplace=True),
        )
        self.stem_fusion = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, bias=False),
            _group_norm(32),
            nn.SiLU(inplace=True),
        )
        self.spatial_encoder = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 48, stride=2),
            ResidualBlock(48, 48),
            ResidualBlock(48, 64, stride=2),
            ResidualBlock(64, 64),
        )
        self.temporal_encoder = ConvGRUCell(64, CONVGRU_HIDDEN_CHANNELS)
        self.vision_projection = nn.Sequential(
            nn.Linear(2 * CONVGRU_HIDDEN_CHANNELS, VISION_EMBEDDING_DIM),
            nn.LayerNorm(VISION_EMBEDDING_DIM),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.regressor = nn.Sequential(
            nn.Linear(VISION_EMBEDDING_DIM, head_dim),
            nn.LayerNorm(head_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )

    def _encode_sequence(self, image: torch.Tensor) -> torch.Tensor:
        batch_size, timesteps, _, height, width = image.shape
        frames = image.reshape(batch_size * timesteps, image.shape[2], height, width)
        no2 = frames[:, :1]
        weather = frames[:, 1:MODEL_IMAGE_CHANNELS]
        mask = frames[:, MODEL_IMAGE_CHANNELS : MODEL_IMAGE_CHANNELS + 1]
        no2_features = self.no2_stem(no2, mask)
        weather_features = self.weather_stem(weather)
        encoded = self.spatial_encoder(self.stem_fusion(torch.cat((no2_features, weather_features), dim=1)))
        encoded = encoded.reshape(batch_size, timesteps, *encoded.shape[1:])

        hidden = encoded.new_zeros(
            batch_size,
            CONVGRU_HIDDEN_CHANNELS,
            encoded.shape[-2],
            encoded.shape[-1],
        )
        for timestep in range(timesteps):
            hidden = self.temporal_encoder(encoded[:, timestep], hidden)
        average = F.adaptive_avg_pool2d(hidden, 1).flatten(1)
        peak = F.adaptive_max_pool2d(hidden, 1).flatten(1)
        return self.vision_projection(torch.cat((average, peak), dim=1))

    def forward(
        self,
        image: torch.Tensor,
        tabular: torch.Tensor,
        elapsed_hours: torch.Tensor,
    ) -> torch.Tensor:
        del tabular, elapsed_hours
        return self.regressor(self._encode_sequence(image)).squeeze(1)

    def num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class RasterHurdleConvGRU(RasterConvGRU):
    """Predict change occurrence and conditional directional magnitudes."""

    def __init__(
        self,
        *,
        steady_threshold: float,
        class_weights: torch.Tensor | None = None,
        head_dim: int = DEFAULT_HEAD_DIM,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__(head_dim=head_dim, dropout=dropout)
        self.steady_threshold = steady_threshold
        self.regressor = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(VISION_EMBEDDING_DIM, head_dim),
            nn.LayerNorm(head_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(head_dim, HURDLE_CLASS_COUNT)
        self.magnitude_regressor = nn.Linear(head_dim, HURDLE_MAGNITUDE_COUNT)
        probability_adjustment = (
            torch.zeros(HURDLE_CLASS_COUNT) if class_weights is None else class_weights.detach().float().log()
        )
        self.register_buffer("probability_logit_adjustment", probability_adjustment)
        initial_magnitude_bias = math.log(math.expm1(steady_threshold))
        nn.init.constant_(self.magnitude_regressor.bias, initial_magnitude_bias)

    def forward(
        self,
        image: torch.Tensor,
        tabular: torch.Tensor,
        elapsed_hours: torch.Tensor,
    ) -> HurdleOutput:
        """Predict class probabilities and positive conditional magnitudes."""
        del tabular, elapsed_hours
        features = self.head(self._encode_sequence(image))
        class_logits = self.classifier(features)
        magnitudes = self.steady_threshold + F.softplus(self.magnitude_regressor(features))
        probabilities = (class_logits - self.probability_logit_adjustment).softmax(dim=1)
        expected_value = probabilities[:, 2] * magnitudes[:, 1] - probabilities[:, 0] * magnitudes[:, 0]
        return HurdleOutput(class_logits, probabilities, magnitudes, expected_value)
