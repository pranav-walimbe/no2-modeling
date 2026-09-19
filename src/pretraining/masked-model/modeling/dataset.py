"""Dataset loading and train-only normalization for masked NO2 modeling."""

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
    MASKED_PRETRAINING_ARTIFICIAL_MASK_KEY,
    MASKED_PRETRAINING_BASE_DIR,
    MASKED_PRETRAINING_DF_DIR,
    MASKED_PRETRAINING_IMAGE_KEYS,
    MASKED_PRETRAINING_ORIGINAL_MASK_KEY,
    MODEL_IMAGE_CLIP_ABS,
    MODEL_ROBUST_IMAGE_KEYS,
)

RASTER_PATH_COL = "raster_bundle_path"
MIN_SCALE = 1e-12
ROBUST_STD_NORMALIZER = 1.349


@dataclass(frozen=True)
class MaskedNormalizationStats:
    """JSON-safe image normalization fitted on the pretraining train split."""

    image_keys: tuple[str, ...]
    image_center: tuple[float, ...]
    image_scale: tuple[float, ...]
    image_valid_pixels: tuple[int, ...]
    training_records: int

    def to_dict(self) -> dict[str, object]:
        """Return normalization state as JSON-safe values."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "MaskedNormalizationStats":
        """Build normalization state from serialized values.

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
            training_records=int(values["training_records"]),
        )


def compute_stats(
    *,
    dataset_dir: str | Path = MASKED_PRETRAINING_BASE_DIR,
    dataframe_dir: str | Path = MASKED_PRETRAINING_DF_DIR,
    progress_interval: int = 1_000,
) -> MaskedNormalizationStats:
    """Compute image normalization statistics from the training split.

    Args:
        dataset_dir: Root used to resolve relative raster paths.
        dataframe_dir: Directory containing split CSV files.
        progress_interval: Records between progress messages.

    Returns:
        Frozen train-split normalization statistics.
    """
    frame = pd.read_csv(Path(dataframe_dir) / "train_df.csv")
    paths = frame[RASTER_PATH_COL].to_numpy(dtype=str)
    root = Path(dataset_dir)
    channel_count = len(MASKED_PRETRAINING_IMAGE_KEYS)
    robust_channels = tuple(MASKED_PRETRAINING_IMAGE_KEYS.index(name) for name in MODEL_ROBUST_IMAGE_KEYS)
    counts = np.zeros(channel_count, dtype=np.int64)
    sums = np.zeros(channel_count, dtype=np.float64)
    squared_sums = np.zeros(channel_count, dtype=np.float64)

    for index, serialized_path in enumerate(paths, start=1):
        path = Path(serialized_path)
        path = path if path.is_absolute() else root / path
        with np.load(path, allow_pickle=False) as bundle:
            for channel, name in enumerate(MASKED_PRETRAINING_IMAGE_KEYS):
                values = np.asarray(bundle[name], dtype=np.float64)
                valid = values[np.isfinite(values)]
                counts[channel] += valid.size
                if channel not in robust_channels:
                    sums[channel] += valid.sum()
                    squared_sums[channel] += np.square(valid).sum()
        if progress_interval > 0 and (index % progress_interval == 0 or index == len(paths)):
            print(f"Image-statistics scan: {index:,}/{len(paths):,} rasters")

    center = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    variance = np.divide(squared_sums, counts, out=np.zeros_like(sums), where=counts > 0) - np.square(center)
    scale = np.sqrt(np.maximum(variance, 0.0))
    with tempfile.TemporaryDirectory(prefix=".masked-no2-quantiles-") as temporary_dir:
        for channel in robust_channels:
            values = np.memmap(
                Path(temporary_dir) / f"channel-{channel}.bin",
                dtype=np.float32,
                mode="w+",
                shape=(int(counts[channel]),),
            )
            offset = 0
            for serialized_path in paths:
                path = Path(serialized_path)
                path = path if path.is_absolute() else root / path
                with np.load(path, allow_pickle=False) as bundle:
                    raster = np.asarray(bundle[MASKED_PRETRAINING_IMAGE_KEYS[channel]], dtype=np.float32)
                valid = raster[np.isfinite(raster)]
                values[offset : offset + valid.size] = valid
                offset += valid.size
            lower, median, upper = np.percentile(values, (25, 50, 75), overwrite_input=True)
            center[channel] = median
            scale[channel] = (upper - lower) / ROBUST_STD_NORMALIZER
            del values

    scale = np.where(np.isfinite(scale) & (scale > MIN_SCALE), scale, 1.0)
    return MaskedNormalizationStats(
        image_keys=MASKED_PRETRAINING_IMAGE_KEYS,
        image_center=tuple(float(value) for value in center),
        image_scale=tuple(float(value) for value in scale),
        image_valid_pixels=tuple(int(value) for value in counts),
        training_records=len(frame),
    )


def save_stats(stats: MaskedNormalizationStats, path: str | Path) -> None:
    """Atomically persist preprocessing state beside the model checkpoint.

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
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


class MaskedNO2Dataset(Dataset):
    """Lazy masked-raster reconstruction dataset."""

    def __init__(
        self,
        split: str,
        stats: MaskedNormalizationStats | dict[str, object],
        *,
        dataset_dir: str | Path = MASKED_PRETRAINING_BASE_DIR,
        dataframe_dir: str | Path = MASKED_PRETRAINING_DF_DIR,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.frame = pd.read_csv(Path(dataframe_dir) / f"{split}_df.csv")
        self.stats = stats if isinstance(stats, MaskedNormalizationStats) else MaskedNormalizationStats.from_dict(stats)
        self.raster_paths = self.frame[RASTER_PATH_COL].to_numpy(dtype=str)

    def __len__(self) -> int:
        return len(self.raster_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        path = Path(self.raster_paths[index])
        path = path if path.is_absolute() else self.dataset_dir / path
        with np.load(path, allow_pickle=False) as bundle:
            rasters = np.stack([np.asarray(bundle[name], dtype=np.float32) for name in MASKED_PRETRAINING_IMAGE_KEYS])
            original_valid = np.asarray(bundle[MASKED_PRETRAINING_ORIGINAL_MASK_KEY], dtype=bool)
            artificial_visible = np.asarray(bundle[MASKED_PRETRAINING_ARTIFICIAL_MASK_KEY], dtype=bool)

        center = np.asarray(self.stats.image_center, dtype=np.float32)[:, None, None]
        scale = np.asarray(self.stats.image_scale, dtype=np.float32)[:, None, None]
        normalized = (rasters - center) / scale
        np.clip(normalized, -MODEL_IMAGE_CLIP_ABS, MODEL_IMAGE_CLIP_ABS, out=normalized)
        visible = original_valid & artificial_visible
        target = normalized[:1].copy()
        normalized[0] *= visible
        image = np.concatenate((normalized, visible[None].astype(np.float32)), axis=0)
        loss_mask = (original_valid & ~artificial_visible)[None].astype(np.float32)
        return torch.from_numpy(image), torch.from_numpy(target), torch.from_numpy(loss_mask), index
