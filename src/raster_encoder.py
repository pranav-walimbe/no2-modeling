"""Shared mask-aware single-frame raster encoder."""

import torch
from torch import nn
from torch.nn import functional as F

from config import MODEL_IMAGE_CHANNELS

ENCODER_OUTPUT_CHANNELS = 64


def group_norm(channels: int) -> nn.GroupNorm:
    """Build group normalization with a divisor of the channel count.

    Args:
        channels: Feature channels to normalize.

    Returns:
        Configured group-normalization layer.
    """
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
            group_norm(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            group_norm(out_channels),
        )
        self.residual = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                group_norm(out_channels),
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
        self.first_norm = group_norm(out_channels)
        self.second = PartialConv2d(out_channels, out_channels, kernel_size=3)
        self.second_norm = group_norm(out_channels)
        self.activation = nn.SiLU()

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        features, mask = self.first(values, mask)
        features = self.activation(self.first_norm(features)) * mask
        features, mask = self.second(features, mask)
        return self.activation(self.second_norm(features)) * mask


class RasterFrameEncoder(nn.Module):
    """Encode one NO2 and weather frame into a spatial latent map."""

    def __init__(self) -> None:
        super().__init__()
        self.no2_stem = MaskedNO2Stem(out_channels=16)
        self.weather_stem = nn.Sequential(
            nn.Conv2d(MODEL_IMAGE_CHANNELS - 1, 16, kernel_size=5, padding=2, bias=False),
            group_norm(16),
            nn.SiLU(inplace=True),
        )
        self.stem_fusion = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, bias=False),
            group_norm(32),
            nn.SiLU(inplace=True),
        )
        self.spatial_encoder = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 48, stride=2),
            ResidualBlock(48, 48),
            ResidualBlock(48, ENCODER_OUTPUT_CHANNELS, stride=2),
            ResidualBlock(ENCODER_OUTPUT_CHANNELS, ENCODER_OUTPUT_CHANNELS),
        )

    def forward(
        self,
        no2: torch.Tensor,
        weather: torch.Tensor,
        visible_mask: torch.Tensor,
    ) -> torch.Tensor:
        no2_features = self.no2_stem(no2, visible_mask)
        weather_features = self.weather_stem(weather)
        fused = self.stem_fusion(torch.cat((no2_features, weather_features), dim=1))
        return self.spatial_encoder(fused)

    def num_params(self) -> int:
        """Return the number of trainable encoder parameters."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
