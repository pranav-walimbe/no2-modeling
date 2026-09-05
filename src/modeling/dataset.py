"""Dataset loading and train-only normalization for delta-NO2 modeling."""

from __future__ import annotations

import json
import math
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
    IMG_SIZE,
    LABEL_COL,
    MODEL_CYCLIC_FEATURES,
    MODEL_IMAGE_CLIP_Z,
    MODEL_IMAGE_KEY,
    MODEL_LOG1P_FEATURES,
    MODEL_RAW_FEATURES,
)

RASTER_PATH_COL = "delta_no2_path"
STATS_VERSION = 1
MIN_SCALE = 1e-12


def _model_feature_names() -> tuple[str, ...]:
    names = [f"log1p_{name}" if name in MODEL_LOG1P_FEATURES else name for name in MODEL_RAW_FEATURES]
    for name in MODEL_CYCLIC_FEATURES:
        names.extend((f"{name}_sin", f"{name}_cos"))
    return tuple(names)


MODEL_FEATURE_NAMES = _model_feature_names()


@dataclass(frozen=True)
class NormalizationStats:
    """JSON-safe train-split preprocessing state used by every data split."""

    version: int
    image_mean: float
    image_std: float
    image_finite_pixels: int
    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_std: tuple[float, ...]
    target_mean: float
    target_std: float
    training_records: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "NormalizationStats":
        if int(values.get("version", -1)) != STATS_VERSION:
            raise ValueError(f"Unsupported normalization-statistics version: {values.get('version')}")
        return cls(
            version=int(values["version"]),
            image_mean=float(values["image_mean"]),
            image_std=float(values["image_std"]),
            image_finite_pixels=int(values["image_finite_pixels"]),
            feature_names=tuple(str(name) for name in values["feature_names"]),
            feature_mean=tuple(float(value) for value in values["feature_mean"]),
            feature_std=tuple(float(value) for value in values["feature_std"]),
            target_mean=float(values["target_mean"]),
            target_std=float(values["target_std"]),
            training_records=int(values["training_records"]),
        )


def _read_split_frame(split: str, dataframe_dir: Path) -> pd.DataFrame:
    path = dataframe_dir / f"{split}_df.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {split} dataframe: {path}")
    frame = pd.read_csv(path)
    required = {RASTER_PATH_COL, LABEL_COL, "date", "hour", *MODEL_RAW_FEATURES}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{split} dataframe is missing model columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError(f"{split} dataframe is empty")
    return frame


def _feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Create leakage-safe numeric features in a stable, documented order."""
    columns: list[np.ndarray] = []
    for name in MODEL_RAW_FEATURES:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
        if name in MODEL_LOG1P_FEATURES:
            if np.any(values < 0):
                raise ValueError(f"Model feature {name} must be nonnegative before log1p")
            values = np.log1p(values)
        columns.append(values)

    for cyclic_feature in MODEL_CYCLIC_FEATURES:
        if cyclic_feature == "hour":
            values = pd.to_numeric(frame["hour"], errors="coerce").to_numpy(dtype=np.float64)
            angle = 2 * np.pi * values / 24.0
        elif cyclic_feature == "day_of_year":
            dates = pd.to_datetime(frame["date"], errors="coerce")
            values = dates.dt.dayofyear.to_numpy(dtype=np.float64)
            angle = 2 * np.pi * (values - 1.0) / 365.25
        else:
            raise ValueError(f"Unsupported cyclic model feature: {cyclic_feature}")
        columns.extend((np.sin(angle), np.cos(angle)))

    matrix = np.column_stack(columns)
    if not np.isfinite(matrix).all():
        bad_columns = [MODEL_FEATURE_NAMES[index] for index in np.flatnonzero(~np.isfinite(matrix).all(axis=0))]
        raise ValueError(f"Model features contain non-finite values: {', '.join(bad_columns)}")
    return matrix


def _raster_path(serialized_path: object, dataset_dir: Path) -> Path:
    if not isinstance(serialized_path, str) or not serialized_path:
        raise ValueError("delta_no2_path must be a non-empty string")
    path = Path(serialized_path)
    return path if path.is_absolute() else dataset_dir / path


def _load_delta_raster(path: Path) -> np.ndarray:
    try:
        with np.load(path, allow_pickle=False) as bundle:
            raster = np.asarray(bundle[MODEL_IMAGE_KEY], dtype=np.float32)
    except KeyError as error:
        raise ValueError(f"Raster bundle is missing {MODEL_IMAGE_KEY}: {path}") from error
    if raster.shape != (IMG_SIZE, IMG_SIZE):
        raise ValueError(f"Expected {(IMG_SIZE, IMG_SIZE)} raster at {path}, found {raster.shape}")
    return raster


def _safe_scale(values: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(values) & (values > MIN_SCALE), values, 1.0)


def compute_stats(
    split: str = "train",
    *,
    dataset_dir: str | Path = DATASET_DIR,
    dataframe_dir: str | Path = DATASET_DF,
    progress_interval: int = 1_000,
) -> NormalizationStats:
    """Compute constant-memory normalization statistics from one split.

    Raster mean and variance use a numerically stable, batch-combined Welford
    update over finite pixels only. Images are opened one at a time, so peak
    memory is independent of dataset size.
    """
    root = Path(dataset_dir)
    frame = _read_split_frame(split, Path(dataframe_dir))
    features = _feature_matrix(frame)
    labels = pd.to_numeric(frame[LABEL_COL], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(labels).all():
        raise ValueError(f"{LABEL_COL} contains non-finite values in {split}")

    count = 0
    mean = 0.0
    sum_squared_deviation = 0.0
    for index, serialized_path in enumerate(frame[RASTER_PATH_COL], start=1):
        raster = _load_delta_raster(_raster_path(serialized_path, root))
        values = raster[np.isfinite(raster)].astype(np.float64, copy=False)
        if not values.size:
            raise ValueError(f"Raster has no finite {MODEL_IMAGE_KEY} values: {serialized_path}")
        batch_count = int(values.size)
        batch_mean = float(values.mean())
        batch_squared_deviation = float(np.square(values - batch_mean).sum())
        combined_count = count + batch_count
        delta = batch_mean - mean
        mean += delta * batch_count / combined_count
        sum_squared_deviation += batch_squared_deviation + delta * delta * count * batch_count / combined_count
        count = combined_count
        if progress_interval > 0 and (index % progress_interval == 0 or index == len(frame)):
            print(f"Normalization scan: {index:,}/{len(frame):,} rasters")

    image_std = math.sqrt(sum_squared_deviation / count)
    if not math.isfinite(image_std) or image_std <= MIN_SCALE:
        raise ValueError("Training raster standard deviation is zero or non-finite")
    feature_std = _safe_scale(features.std(axis=0))
    target_std = float(_safe_scale(np.asarray([labels.std()]))[0])
    return NormalizationStats(
        version=STATS_VERSION,
        image_mean=mean,
        image_std=image_std,
        image_finite_pixels=count,
        feature_names=MODEL_FEATURE_NAMES,
        feature_mean=tuple(float(value) for value in features.mean(axis=0)),
        feature_std=tuple(float(value) for value in feature_std),
        target_mean=float(labels.mean()),
        target_std=target_std,
        training_records=len(frame),
    )


def save_stats(stats: NormalizationStats, path: str | Path) -> None:
    """Atomically persist preprocessing state beside a model checkpoint."""
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


def load_stats(path: str | Path) -> NormalizationStats:
    with Path(path).open() as source:
        values = json.load(source)
    if not isinstance(values, dict):
        raise ValueError("Normalization statistics must be a JSON object")
    return NormalizationStats.from_dict(values)


def denormalize_target(values: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    """Return predictions in delta_nox_norm units."""
    return np.asarray(values) * stats.target_std + stats.target_mean


class NOxDataset(Dataset):
    """Lazy per-record delta-NO2 raster and tabular dataset."""

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
        if self.stats.feature_names != MODEL_FEATURE_NAMES:
            raise ValueError("Normalization feature order does not match the configured model features")

        raw_features = _feature_matrix(self.frame)
        feature_mean = np.asarray(self.stats.feature_mean, dtype=np.float64)
        feature_std = np.asarray(self.stats.feature_std, dtype=np.float64)
        self.features = ((raw_features - feature_mean) / feature_std).astype(np.float32)
        labels = pd.to_numeric(self.frame[LABEL_COL], errors="raise").to_numpy(dtype=np.float64)
        self.labels = ((labels - self.stats.target_mean) / self.stats.target_std).astype(np.float32)
        self.raster_paths = self.frame[RASTER_PATH_COL].to_numpy(dtype=str)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if self.load_images:
            raster = _load_delta_raster(_raster_path(self.raster_paths[index], self.dataset_dir))
            finite = np.isfinite(raster)
            normalized = np.zeros_like(raster, dtype=np.float32)
            normalized[finite] = (raster[finite] - self.stats.image_mean) / self.stats.image_std
            np.clip(normalized, -MODEL_IMAGE_CLIP_Z, MODEL_IMAGE_CLIP_Z, out=normalized)
            image = torch.from_numpy(np.stack((normalized, finite.astype(np.float32, copy=False)), axis=0))
        else:
            image = torch.empty(0, dtype=torch.float32)
        return (
            image,
            torch.from_numpy(self.features[index]),
            torch.tensor(self.labels[index]),
            index,
        )
