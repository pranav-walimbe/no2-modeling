"""Mask-aware convolutional autoencoder for NO2 reconstruction."""

import torch
from torch import nn
from torch.nn import functional as F

from config import MODEL_IMAGE_CHANNELS
from raster_encoder import ENCODER_OUTPUT_CHANNELS, RasterFrameEncoder, ResidualBlock

ARCHITECTURE_NAME = "masked_no2_convolutional_autoencoder"
LATENT_SPATIAL_SIZE = 6


class MaskedNO2Autoencoder(nn.Module):
    """Reconstruct a complete NO2 raster from visible NO2 and weather."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = RasterFrameEncoder()
        self.bottleneck = ResidualBlock(ENCODER_OUTPUT_CHANNELS, ENCODER_OUTPUT_CHANNELS)
        self.decoder_12 = ResidualBlock(ENCODER_OUTPUT_CHANNELS, 48)
        self.decoder_24 = ResidualBlock(48, 32)
        self.output = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        no2 = image[:, :1]
        weather = image[:, 1:MODEL_IMAGE_CHANNELS]
        visible_mask = image[:, MODEL_IMAGE_CHANNELS : MODEL_IMAGE_CHANNELS + 1]
        latent = self.encoder(no2, weather, visible_mask)
        decoded = self.bottleneck(latent)
        decoded = F.interpolate(decoded, scale_factor=2, mode="bilinear", align_corners=False)
        decoded = self.decoder_12(decoded)
        decoded = F.interpolate(decoded, scale_factor=2, mode="bilinear", align_corners=False)
        return self.output(self.decoder_24(decoded))

    def fill_missing(self, image: torch.Tensor) -> torch.Tensor:
        """Preserve visible NO2 and insert predictions at missing pixels.

        Args:
            image: Normalized image channels followed by the visible-pixel mask.

        Returns:
            Completed normalized NO2 raster.
        """
        visible_mask = image[:, MODEL_IMAGE_CHANNELS : MODEL_IMAGE_CHANNELS + 1]
        return image[:, :1] * visible_mask + self(image) * (1.0 - visible_mask)

    def num_params(self) -> int:
        """Return the number of trainable model parameters."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def masked_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Calculate mean absolute error over artificially hidden valid pixels.

    Args:
        prediction: Complete predicted normalized NO2 raster.
        target: Complete observed normalized NO2 raster.
        loss_mask: One where an observed target was artificially hidden.

    Returns:
        Scalar masked mean absolute error.
    """
    return ((prediction - target).abs() * loss_mask).sum() / loss_mask.sum()
