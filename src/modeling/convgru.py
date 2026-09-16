"""Physics-guided mask-aware ConvGRU for emissions-change prediction."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from config import IMG_RANGE, IMG_SIZE, MODEL_IMAGE_CHANNELS, MODEL_IMAGE_CLIP_ABS, MODEL_IMAGE_KEYS
from modeling.mlp import TabularMLP

DEFAULT_HEAD_DIM = 128
DEFAULT_DROPOUT = 0.20
VISION_EMBEDDING_DIM = 128
CONVGRU_HIDDEN_CHANNELS = 96
SECONDS_PER_HOUR = 3_600.0
METRES_PER_KM = 1_000.0
MIN_LIFETIME_HOURS = 0.5
MAX_LIFETIME_HOURS = 12.0
INITIAL_LIFETIME_HOURS = 4.0
GRID_CELL_SIZE_KM = IMG_RANGE / IMG_SIZE


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
        self.residual_convolution = nn.Conv2d(1, out_channels, kernel_size=5, padding=2, bias=False)
        self.first_norm = _group_norm(out_channels)
        self.second = PartialConv2d(out_channels, out_channels, kernel_size=3)
        self.second_norm = _group_norm(out_channels)
        self.activation = nn.SiLU()

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        transport_residual: torch.Tensor,
        residual_mask: torch.Tensor,
    ) -> torch.Tensor:
        features, mask = self.first(values, mask)
        features = features + self.residual_convolution(transport_residual)
        mask = torch.maximum(mask, residual_mask)
        features = self.activation(self.first_norm(features)) * mask
        features, mask = self.second(features, mask)
        return self.activation(self.second_norm(features)) * mask


class AdvectionDecayResidual(nn.Module):
    """Build NO2 innovations with differentiable advection and global decay."""

    def __init__(self, image_center: tuple[float, ...], image_scale: tuple[float, ...]) -> None:
        super().__init__()
        no2_channel = MODEL_IMAGE_KEYS.index("no2")
        wind_u_channel = MODEL_IMAGE_KEYS.index("wind_u_80m_mps")
        wind_v_channel = MODEL_IMAGE_KEYS.index("wind_v_80m_mps")
        self.register_buffer("no2_center", torch.tensor(float(image_center[no2_channel])))
        self.register_buffer("no2_scale", torch.tensor(float(image_scale[no2_channel])))
        self.register_buffer("wind_u_center", torch.tensor(float(image_center[wind_u_channel])))
        self.register_buffer("wind_u_scale", torch.tensor(float(image_scale[wind_u_channel])))
        self.register_buffer("wind_v_center", torch.tensor(float(image_center[wind_v_channel])))
        self.register_buffer("wind_v_scale", torch.tensor(float(image_scale[wind_v_channel])))

        lifetime_fraction = (INITIAL_LIFETIME_HOURS - MIN_LIFETIME_HOURS) / (
            MAX_LIFETIME_HOURS - MIN_LIFETIME_HOURS
        )
        initial_logit = math.log(lifetime_fraction / (1.0 - lifetime_fraction))
        self.lifetime_logit = nn.Parameter(torch.tensor(initial_logit))
        coordinates = (2.0 * (torch.arange(IMG_SIZE, dtype=torch.float32) + 0.5) / IMG_SIZE) - 1.0
        grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        self.register_buffer("base_grid", torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0))

    @property
    def lifetime_hours(self) -> torch.Tensor:
        """Return the bounded positive global NO2 lifetime."""
        fraction = self.lifetime_logit.sigmoid()
        return MIN_LIFETIME_HOURS + (MAX_LIFETIME_HOURS - MIN_LIFETIME_HOURS) * fraction

    @staticmethod
    def _linear_fill(values: torch.Tensor, valid: torch.Tensor, dimension: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Interpolate from nearest valid neighbours along one spatial axis
        axis_values = values.movedim(dimension, -1)
        axis_valid = valid.movedim(dimension, -1)
        length = axis_values.shape[-1]
        positions = torch.arange(length, device=values.device).reshape((1,) * (values.ndim - 1) + (length,))

        left_indices = torch.where(axis_valid, positions, -1).cummax(dim=-1).values
        right_indices = torch.where(axis_valid, positions, length).flip(-1).cummin(dim=-1).values.flip(-1)
        has_left = left_indices >= 0
        has_right = right_indices < length
        left_values = torch.gather(axis_values, -1, left_indices.clamp_min(0))
        right_values = torch.gather(axis_values, -1, right_indices.clamp_max(length - 1))

        span = (right_indices - left_indices).clamp_min(1)
        fraction = (positions - left_indices).to(values.dtype) / span.to(values.dtype)
        interpolated = left_values + fraction * (right_values - left_values)
        interpolated = torch.where(has_left & has_right, interpolated, torch.where(has_left, left_values, right_values))
        fillable = ~axis_valid & (has_left | has_right)
        filled = torch.where(fillable, interpolated, axis_values)
        return filled.movedim(-1, dimension), (axis_valid | fillable).movedim(-1, dimension)

    def _fill_missing(self, values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        # Apply separable mask-aware interpolation while preserving observations
        filled, valid = self._linear_fill(values * masks, masks > 0, -1)
        filled, _ = self._linear_fill(filled, valid, -2)
        return filled

    def _sampling_grid(
        self,
        wind_u_mps: torch.Tensor,
        wind_v_mps: torch.Tensor,
        elapsed_hours: torch.Tensor,
    ) -> torch.Tensor:
        # Backtrace arrivals on the north-up equal-area AOI grid
        batch_size, _, height, width = wind_u_mps.shape
        base_grid = self.base_grid.to(wind_u_mps).expand(batch_size, -1, -1, -1)
        seconds = elapsed_hours[:, None, None, None] * SECONDS_PER_HOUR
        column_displacement = wind_u_mps * seconds / (GRID_CELL_SIZE_KM * METRES_PER_KM)
        northward_displacement = wind_v_mps * seconds / (GRID_CELL_SIZE_KM * METRES_PER_KM)
        source_x = base_grid[..., 0] - 2.0 * column_displacement[:, 0] / width
        source_y = base_grid[..., 1] + 2.0 * northward_displacement[:, 0] / height
        return torch.stack((source_x, source_y), dim=-1)

    def forward(
        self,
        no2: torch.Tensor,
        masks: torch.Tensor,
        weather: torch.Tensor,
        elapsed_hours: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate normalized current-minus-transported NO2 residuals.

        Args:
            no2: Normalized NO2 sequence shaped batch by time by one by height by width.
            masks: Binary NO2 validity masks aligned with the sequence.
            weather: Normalized temperature and wind sequence.
            elapsed_hours: Hours between adjacent sequence timesteps.

        Returns:
            Train-scale NO2 residuals aligned with the input sequence.
        """
        batch_size, timesteps, _, height, width = no2.shape
        flattened_no2 = no2.reshape(batch_size * timesteps, 1, height, width)
        flattened_masks = masks.reshape(batch_size * timesteps, 1, height, width)
        filled = self._fill_missing(flattened_no2, flattened_masks).reshape_as(no2)

        prior = filled[:, :-1].reshape(batch_size * (timesteps - 1), 1, height, width)
        current_weather = weather[:, 1:]
        wind_u = current_weather[:, :, 1:2].reshape_as(prior) * self.wind_u_scale + self.wind_u_center
        wind_v = current_weather[:, :, 2:3].reshape_as(prior) * self.wind_v_scale + self.wind_v_center
        interval_hours = elapsed_hours.reshape(-1)
        grid = self._sampling_grid(wind_u, wind_v, interval_hours)
        advected = F.grid_sample(
            prior,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

        decay = torch.exp(-interval_hours[:, None, None, None] / self.lifetime_hours)
        prior_physical = advected * self.no2_scale + self.no2_center
        current_physical = filled[:, 1:].reshape_as(prior) * self.no2_scale + self.no2_center
        residual = ((current_physical - decay * prior_physical) / self.no2_scale).reshape(
            batch_size,
            timesteps - 1,
            1,
            height,
            width,
        )
        first = residual.new_zeros(batch_size, 1, 1, height, width)
        return torch.cat((first, residual), dim=1).clamp(-MODEL_IMAGE_CLIP_ABS, MODEL_IMAGE_CLIP_ABS)


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


class NOxModel(nn.Module):
    """Encode raster sequences and fuse a frozen tabular embedding."""

    def __init__(
        self,
        *,
        tabular_model: TabularMLP,
        image_center: tuple[float, ...],
        image_scale: tuple[float, ...],
        head_dim: int = DEFAULT_HEAD_DIM,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.advection_decay = AdvectionDecayResidual(image_center, image_scale)
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

        self.tabular_model = tabular_model
        self.tabular_model.requires_grad_(False)
        self.tabular_model.eval()
        fusion_dim = VISION_EMBEDDING_DIM + tabular_model.embedding_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, head_dim),
            nn.LayerNorm(head_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )

    def train(self, mode: bool = True) -> "NOxModel":
        super().train(mode)
        self.tabular_model.eval()
        return self

    def _encode_sequence(self, image: torch.Tensor, elapsed_hours: torch.Tensor) -> torch.Tensor:
        batch_size, timesteps, _, height, width = image.shape
        no2_sequence = image[:, :, :1]
        weather_sequence = image[:, :, 1:MODEL_IMAGE_CHANNELS]
        mask_sequence = image[:, :, MODEL_IMAGE_CHANNELS : MODEL_IMAGE_CHANNELS + 1]
        transport_residual = self.advection_decay(no2_sequence, mask_sequence, weather_sequence, elapsed_hours)
        residual_mask = torch.cat((mask_sequence[:, :1], torch.ones_like(mask_sequence[:, 1:])), dim=1)
        frames = image.reshape(batch_size * timesteps, image.shape[2], height, width)
        no2 = frames[:, :1]
        weather = frames[:, 1:MODEL_IMAGE_CHANNELS]
        mask = frames[:, MODEL_IMAGE_CHANNELS : MODEL_IMAGE_CHANNELS + 1]
        no2_features = self.no2_stem(
            no2,
            mask,
            transport_residual.flatten(0, 1),
            residual_mask.flatten(0, 1),
        )
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
        vision_features = self._encode_sequence(image, elapsed_hours)
        with torch.no_grad():
            tabular_features = self.tabular_model.encode(tabular)
        return self.head(torch.cat((vision_features, tabular_features), dim=1)).squeeze(1)

    def num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
