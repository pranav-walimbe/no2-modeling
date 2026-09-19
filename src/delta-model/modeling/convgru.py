"""Mask-aware ConvGRU for emissions-change prediction."""

import torch
from torch import nn
from torch.nn import functional as F

from config import MODEL_IMAGE_CHANNELS
from raster_encoder import ENCODER_OUTPUT_CHANNELS, RasterFrameEncoder

DEFAULT_HEAD_DIM = 128
DEFAULT_DROPOUT = 0.20
VISION_EMBEDDING_DIM = 128
CONVGRU_HIDDEN_CHANNELS = 96


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
    """Classify emissions changes from raster sequences alone."""

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
            nn.Linear(head_dim, 1),
        )

    def _encode_sequence(self, image: torch.Tensor) -> torch.Tensor:
        batch_size, timesteps, _, height, width = image.shape
        frames = image.reshape(batch_size * timesteps, image.shape[2], height, width)
        no2 = frames[:, :1]
        weather = frames[:, 1:MODEL_IMAGE_CHANNELS]
        mask = frames[:, MODEL_IMAGE_CHANNELS : MODEL_IMAGE_CHANNELS + 1]
        encoded = self.frame_encoder(no2, weather, mask)
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
        return self.classifier(self._encode_sequence(image)).squeeze(1)

    def num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
