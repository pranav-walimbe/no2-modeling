"""Convolutional recurrent network for emissions-change classification."""

import torch
from modeling.mlp import TabularMLP
from torch import nn
from torch.nn import functional as F

from config import MODEL_CLASS_NAMES, MODEL_IMAGE_CHANNELS

DEFAULT_HEAD_DIM = 128
DEFAULT_DROPOUT = 0.20
VISION_EMBEDDING_DIM = 128
CONVGRU_HIDDEN_CHANNELS = 96
ENCODER_ARCHITECTURE_NAME = "convolutional_raster_frame_encoder_v1"
ENCODER_OUTPUT_CHANNELS = 64


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


class NO2Stem(nn.Module):
    """Encode a completed NO2 raster with ordinary convolutions."""

    def __init__(self, out_channels: int = 16) -> None:
        super().__init__()
        self.first = nn.Conv2d(1, out_channels, kernel_size=5, padding=2)
        self.first_norm = _group_norm(out_channels)
        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.second_norm = _group_norm(out_channels)
        self.activation = nn.SiLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        features = self.activation(self.first_norm(self.first(values)))
        return self.activation(self.second_norm(self.second(features)))


class RasterFrameEncoder(nn.Module):
    """Encode one completed NO2 and weather frame with ordinary convolutions."""

    def __init__(self) -> None:
        super().__init__()
        self.no2_stem = NO2Stem(out_channels=16)
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
            ResidualBlock(48, ENCODER_OUTPUT_CHANNELS, stride=2),
            ResidualBlock(ENCODER_OUTPUT_CHANNELS, ENCODER_OUTPUT_CHANNELS),
        )

    def forward(self, no2: torch.Tensor, weather: torch.Tensor) -> torch.Tensor:
        no2_features = self.no2_stem(no2)
        weather_features = self.weather_stem(weather)
        fused = self.stem_fusion(torch.cat((no2_features, weather_features), dim=1))
        return self.spatial_encoder(fused)


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


class RasterConvGRUClassifier(nn.Module):
    """Classify emissions changes from completed raster sequences."""

    def __init__(
        self,
        *,
        head_dim: int = DEFAULT_HEAD_DIM,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.frame_encoder = RasterFrameEncoder()
        self.temporal_encoder = ConvGRUCell(ENCODER_OUTPUT_CHANNELS, CONVGRU_HIDDEN_CHANNELS)
        self.vision_projection = nn.Sequential(
            nn.Linear(2 * CONVGRU_HIDDEN_CHANNELS, VISION_EMBEDDING_DIM),
            nn.LayerNorm(VISION_EMBEDDING_DIM),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(VISION_EMBEDDING_DIM, head_dim),
            nn.LayerNorm(head_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_dim, len(MODEL_CLASS_NAMES)),
        )

    def _encode_sequence(self, image: torch.Tensor) -> torch.Tensor:
        batch_size, timesteps, channels, height, width = image.shape
        if channels != MODEL_IMAGE_CHANNELS:
            raise ValueError(f"Expected {MODEL_IMAGE_CHANNELS} completed raster channels, received {channels}")
        frames = image.reshape(batch_size * timesteps, channels, height, width)
        no2 = frames[:, :1]
        weather = frames[:, 1:MODEL_IMAGE_CHANNELS]
        encoded = self.frame_encoder(no2, weather)
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
        return self.classifier(self._encode_sequence(image))

    def num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class RasterTabularFusionClassifier(RasterConvGRUClassifier):
    """Add raster evidence to a frozen tabular classifier's logits."""

    def __init__(
        self,
        n_features: int,
        tabular_state_dict: dict[str, torch.Tensor],
        *,
        head_dim: int = DEFAULT_HEAD_DIM,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__(head_dim=head_dim, dropout=dropout)
        self.tabular_model = TabularMLP(n_features)
        self.tabular_model.load_state_dict(tabular_state_dict)
        self.tabular_model.requires_grad_(False)

    def forward(
        self,
        image: torch.Tensor,
        tabular: torch.Tensor,
        elapsed_hours: torch.Tensor,
    ) -> torch.Tensor:
        raster_logits = super().forward(image, tabular, elapsed_hours)
        with torch.no_grad():
            tabular_logits = self.tabular_model(image, tabular, elapsed_hours)
        return raster_logits + tabular_logits
