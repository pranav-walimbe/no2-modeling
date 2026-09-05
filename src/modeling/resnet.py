"""Compact residual network for signed emissions-change regression."""

import torch
from torch import nn

from config import MODEL_IMAGE_CHANNELS

DEFAULT_HEAD_DIM = 128
DEFAULT_DROPOUT = 0.30


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


class NOxModel(nn.Module):
    """Fuse a mask-aware delta-NO2 encoder with leakage-safe scalar features."""

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
        if n_tabular_features < 1:
            raise ValueError("n_tabular_features must be positive")
        if not use_image and not use_tabular:
            raise ValueError("At least one model input branch must be enabled")
        if head_dim < 1:
            raise ValueError("head_dim must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.use_image = use_image
        self.use_tabular = use_tabular

        if use_image:
            self.stem = nn.Sequential(
                nn.Conv2d(MODEL_IMAGE_CHANNELS, 32, kernel_size=5, padding=2, bias=False),
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
        self.head = nn.Sequential(
            nn.Linear(fusion_features, head_dim),
            nn.LayerNorm(head_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        features = []
        if self.use_image:
            encoded = self.encoder(self.stem(image))
            spatial = self.spatial_pool(encoded).flatten(1)
            peak = self.peak_pool(encoded).flatten(1)
            features.append(self.image_projection(torch.cat((spatial, peak), dim=1)))
        if self.use_tabular:
            features.append(self.tabular_projection(tabular))
        return self.head(torch.cat(features, dim=1)).squeeze(1)

    def num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
