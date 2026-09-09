"""Dataset loading and train-only normalization for TEMPO raster modeling."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import (
    DATASET_DF,
    DATASET_DIR,
    DELTA_THRESHOLD,
    LABEL_COL,
    MODEL_CYCLIC_FEATURES,
    MODEL_IMAGE_CLIP_ABS,
    MODEL_IMAGE_KEYS,
    MODEL_RAW_FEATURES,
    MODEL_ROBUST_IMAGE_KEYS,
)

RASTER_PATH_COL = "delta_no2_path"
LABEL_MODE_COL = "label_mode"
MIN_SCALE = 1e-12
ROBUST_STD_NORMALIZER = 1.349
ROBUST_IMAGE_CHANNELS = tuple(MODEL_IMAGE_KEYS.index(name) for name in MODEL_ROBUST_IMAGE_KEYS)
STANDARD_IMAGE_CHANNELS = tuple(
    channel for channel in range(len(MODEL_IMAGE_KEYS)) if channel not in ROBUST_IMAGE_CHANNELS
)


def _model_feature_names() -> tuple[str, ...]:
    # Expand raw and cyclic inputs into their model column names
    names = list(MODEL_RAW_FEATURES)
    for name in MODEL_CYCLIC_FEATURES:
        names.extend((f"{name}_sin", f"{name}_cos"))
    return tuple(names)


MODEL_FEATURE_NAMES = _model_feature_names()


@dataclass(frozen=True)
class NormalizationStats:
    """JSON-safe train-split preprocessing state used by every data split."""

    image_keys: tuple[str, ...]
    image_center: tuple[float, ...]
    image_scale: tuple[float, ...]
    image_valid_pixels: tuple[int, ...]
    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_std: tuple[float, ...]
    delta_threshold: float
    training_records: int

    def to_dict(self) -> dict[str, object]:
        """Return the normalization state as JSON-safe values.

        Returns:
            Serialized normalization fields.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "NormalizationStats":
        """Build normalization state from JSON-safe values.

        Args:
            values: Serialized normalization fields.

        Returns:
            Parsed normalization state.
        """
        return cls(
            image_keys=tuple(str(name) for name in values["image_keys"]),
            image_center=tuple(float(value) for value in values["image_center"]),
            image_scale=tuple(float(value) for value in values["image_scale"]),
            image_valid_pixels=tuple(int(value) for value in values["image_valid_pixels"]),
            feature_names=tuple(str(name) for name in values["feature_names"]),
            feature_mean=tuple(float(value) for value in values["feature_mean"]),
            feature_std=tuple(float(value) for value in values["feature_std"]),
            delta_threshold=float(values["delta_threshold"]),
            training_records=int(values["training_records"]),
        )


def _read_split_frame(split: str, dataframe_dir: Path) -> pd.DataFrame:
    # Load the split produced by dataset generation
    path = dataframe_dir / f"{split}_df.csv"
    return pd.read_csv(path)


def _feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    # Create leakage-safe numeric features in their documented order
    columns: list[np.ndarray] = [
        pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64) for name in MODEL_RAW_FEATURES
    ]

    for cyclic_feature in MODEL_CYCLIC_FEATURES:
        if cyclic_feature == "hour":
            values = pd.to_numeric(frame["hour"], errors="coerce").to_numpy(dtype=np.float64)
            angle = 2 * np.pi * values / 24.0
        elif cyclic_feature == "day_of_year":
            dates = pd.to_datetime(frame["date"], errors="coerce")
            values = dates.dt.dayofyear.to_numpy(dtype=np.float64)
            angle = 2 * np.pi * (values - 1.0) / 365.25
        columns.extend((np.sin(angle), np.cos(angle)))

    return np.column_stack(columns)


def _raster_path(serialized_path: object, dataset_dir: Path) -> Path:
    # Resolve a stored raster path against the dataset root
    path = Path(str(serialized_path))
    return path if path.is_absolute() else dataset_dir / path


def _load_raster_bundle(path: Path) -> np.ndarray:
    # Load numeric model rasters in their configured order
    with np.load(path, allow_pickle=False) as bundle:
        rasters = np.stack(
            [np.asarray(bundle[name], dtype=np.float32) for name in MODEL_IMAGE_KEYS],
            axis=0,
        )
    return rasters


def _safe_scale(values: np.ndarray) -> np.ndarray:
    # Replace unusable normalization scales with one
    return np.where(np.isfinite(values) & (values > MIN_SCALE), values, 1.0)


def _channel_valid_mask(rasters: np.ndarray, channel: int) -> np.ndarray:
    # Identify valid pixels for one numeric image channel
    return np.isfinite(rasters[channel])


def _valid_channel_values(rasters: np.ndarray, channel: int) -> np.ndarray:
    # Select valid values for one numeric image channel
    return rasters[channel, _channel_valid_mask(rasters, channel)]


def _update_moments(
    values: np.ndarray,
    count: int,
    mean: float,
    sum_squared_deviation: float,
) -> tuple[float, float]:
    # Combine one pixel batch with running population moments accumulated over count pixels
    batch_count = int(values.size)
    if batch_count == 0:
        return mean, sum_squared_deviation
    batch_mean = float(values.mean())
    batch_squared_deviation = float(np.square(values - batch_mean).sum())
    combined_count = count + batch_count
    delta = batch_mean - mean
    combined_mean = mean + delta * batch_count / combined_count
    combined_deviation = (
        sum_squared_deviation + batch_squared_deviation + delta * delta * count * batch_count / combined_count
    )
    return combined_mean, combined_deviation


def _fit_image_stats(raster_paths: np.ndarray, root: Path, progress_interval: int) -> tuple[np.ndarray, ...]:
    # Count valid pixels everywhere and accumulate moments for the standardized channels
    channel_count = len(MODEL_IMAGE_KEYS)
    count = np.zeros(channel_count, dtype=np.int64)
    mean = np.zeros(channel_count, dtype=np.float64)
    sum_squared_deviation = np.zeros(channel_count, dtype=np.float64)
    for index, serialized_path in enumerate(raster_paths, start=1):
        rasters = _load_raster_bundle(_raster_path(serialized_path, root))
        for channel in range(channel_count):
            values = _valid_channel_values(rasters, channel)
            if channel in STANDARD_IMAGE_CHANNELS:
                mean[channel], sum_squared_deviation[channel] = _update_moments(
                    values.astype(np.float64, copy=False),
                    int(count[channel]),
                    float(mean[channel]),
                    float(sum_squared_deviation[channel]),
                )
            count[channel] += values.size
        if progress_interval > 0 and (index % progress_interval == 0 or index == len(raster_paths)):
            print(f"Image-statistics scan: {index:,}/{len(raster_paths):,} rasters")

    center = mean.copy()
    scale = np.ones(channel_count, dtype=np.float64)
    for channel in STANDARD_IMAGE_CHANNELS:
        if count[channel] > 0:
            scale[channel] = np.sqrt(sum_squared_deviation[channel] / count[channel])
    for channel, (channel_center, channel_scale) in _fit_robust_image_stats(raster_paths, root, count).items():
        center[channel] = channel_center
        scale[channel] = channel_scale
    return center, _safe_scale(scale), count


def _fit_robust_image_stats(raster_paths: np.ndarray, root: Path, count: np.ndarray) -> dict[int, tuple[float, float]]:
    # Pool valid NO2 pixels to derive a median and robust standard deviation per channel
    channels = tuple(channel for channel in ROBUST_IMAGE_CHANNELS if count[channel] > 0)
    if not channels:
        return {}
    with tempfile.TemporaryDirectory(prefix=".no2-quantiles-") as temporary_dir:
        pooled = {
            channel: np.memmap(
                Path(temporary_dir) / f"channel-{channel}.bin",
                dtype=np.float32,
                mode="w+",
                shape=(int(count[channel]),),
            )
            for channel in channels
        }
        offsets = dict.fromkeys(channels, 0)
        for serialized_path in raster_paths:
            rasters = _load_raster_bundle(_raster_path(serialized_path, root))
            for channel, destination in pooled.items():
                values = _valid_channel_values(rasters, channel)
                stop = offsets[channel] + values.size
                destination[offsets[channel] : stop] = values
                offsets[channel] = stop
        robust = {}
        for channel, values in pooled.items():
            lower, median, upper = np.percentile(values, (25, 50, 75), overwrite_input=True)
            robust[channel] = (float(median), float(upper - lower) / ROBUST_STD_NORMALIZER)
        del pooled
    return robust


def compute_stats(
    split: str = "train",
    *,
    dataset_dir: str | Path = DATASET_DIR,
    dataframe_dir: str | Path = DATASET_DF,
    progress_interval: int = 1_000,
) -> NormalizationStats:
    """Compute memory-bounded normalization statistics from one split.

    Current-NO2 and delta-NO2 use pooled valid-pixel medians and robust standard
    deviations. Wind channels use pooled finite-pixel means and standard
    deviations.

    Args:
        split: Dataset split used to estimate statistics.
        dataset_dir: Root containing raster bundles.
        dataframe_dir: Directory containing split CSV files.
        progress_interval: Records between progress messages.

    Returns:
        Frozen image and tabular normalization statistics.
    """
    root = Path(dataset_dir)
    frame = _read_split_frame(split, Path(dataframe_dir))
    features = _feature_matrix(frame)
    raster_paths = frame[RASTER_PATH_COL].to_numpy(dtype=str)
    image_center, image_scale, image_valid_pixels = _fit_image_stats(raster_paths, root, progress_interval)
    feature_std = _safe_scale(features.std(axis=0))
    return NormalizationStats(
        image_keys=MODEL_IMAGE_KEYS,
        image_center=tuple(float(value) for value in image_center),
        image_scale=tuple(float(value) for value in image_scale),
        image_valid_pixels=tuple(int(value) for value in image_valid_pixels),
        feature_names=MODEL_FEATURE_NAMES,
        feature_mean=tuple(float(value) for value in features.mean(axis=0)),
        feature_std=tuple(float(value) for value in feature_std),
        delta_threshold=DELTA_THRESHOLD,
        training_records=len(frame),
    )


def clipped_pixel_fractions(
    split: str,
    stats: NormalizationStats,
    *,
    dataset_dir: str | Path = DATASET_DIR,
    dataframe_dir: str | Path = DATASET_DF,
) -> dict[str, float]:
    """Calculate the clipped share of valid pixels for one split.

    Args:
        split: Dataset split to inspect.
        stats: Frozen training normalization statistics.
        dataset_dir: Root containing raster bundles.
        dataframe_dir: Directory containing split CSV files.

    Returns:
        Clipped valid-pixel fraction keyed by numeric channel.
    """
    root = Path(dataset_dir)
    frame = _read_split_frame(split, Path(dataframe_dir))
    clipped = np.zeros(len(MODEL_IMAGE_KEYS), dtype=np.int64)
    valid = np.zeros(len(MODEL_IMAGE_KEYS), dtype=np.int64)
    center = np.asarray(stats.image_center)
    scale = np.asarray(stats.image_scale)
    for serialized_path in frame[RASTER_PATH_COL].to_numpy(dtype=str):
        rasters = _load_raster_bundle(_raster_path(serialized_path, root))
        for channel in range(len(MODEL_IMAGE_KEYS)):
            values = _valid_channel_values(rasters, channel)
            normalized = (values - center[channel]) / scale[channel]
            clipped[channel] += np.count_nonzero(np.abs(normalized) > MODEL_IMAGE_CLIP_ABS)
            valid[channel] += values.size
    return {
        name: float(channel_clipped / channel_valid) if channel_valid else 0.0
        for name, channel_clipped, channel_valid in zip(MODEL_IMAGE_KEYS, clipped, valid, strict=True)
    }


def save_stats(stats: NormalizationStats, path: str | Path) -> None:
    """Atomically persist preprocessing state beside a model checkpoint.

    Args:
        stats: Normalization state to save.
        path: Destination JSON path.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(stats.to_dict(), temporary, indent=2)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


class NOxDataset(Dataset):
    """Lazy per-record TEMPO raster and tabular dataset."""

    def __init__(
        self,
        split: str,
        stats: NormalizationStats | dict[str, object],
        *,
        dataset_dir: str | Path = DATASET_DIR,
        dataframe_dir: str | Path = DATASET_DF,
        load_images: bool = True,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.load_images = load_images
        self.frame = _read_split_frame(split, Path(dataframe_dir))
        self.stats = stats if isinstance(stats, NormalizationStats) else NormalizationStats.from_dict(stats)

        raw_features = _feature_matrix(self.frame)
        feature_mean = np.asarray(self.stats.feature_mean, dtype=np.float64)
        feature_std = np.asarray(self.stats.feature_std, dtype=np.float64)
        self.features = ((raw_features - feature_mean) / feature_std).astype(np.float32)
        labels = pd.to_numeric(self.frame[LABEL_COL], errors="raise").to_numpy(dtype=np.float64)
        self.labels = labels.astype(np.float32)
        self.raster_paths = self.frame[RASTER_PATH_COL].to_numpy(dtype=str)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if self.load_images:
            rasters = _load_raster_bundle(_raster_path(self.raster_paths[index], self.dataset_dir))
            normalized = np.zeros_like(rasters, dtype=np.float32)
            for channel in range(len(MODEL_IMAGE_KEYS)):
                channel_valid = _channel_valid_mask(rasters, channel)
                channel_values = rasters[channel, channel_valid]
                normalized[channel, channel_valid] = (
                    channel_values - self.stats.image_center[channel]
                ) / self.stats.image_scale[channel]
            np.clip(normalized, -MODEL_IMAGE_CLIP_ABS, MODEL_IMAGE_CLIP_ABS, out=normalized)
            image = torch.from_numpy(normalized)
        else:
            image = torch.empty(0, dtype=torch.float32)
        return (
            image,
            torch.from_numpy(self.features[index]),
            torch.tensor(self.labels[index]),
            index,
        )
