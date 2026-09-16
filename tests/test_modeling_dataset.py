"""Regression tests for modeling dataset loading."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from config import LABEL_COL, MODEL_IMAGE_KEYS, SEQUENCE_TIMESTEPS
from modeling.dataset import MODEL_FEATURE_NAMES, NormalizationStats, NOxDataset


class NOxDatasetTest(unittest.TestCase):
    """Test timestamp handling in the modeling dataset."""

    def test_converts_utc_timestamps_to_elapsed_hours(self) -> None:
        """UTC timestamp columns produce numeric elapsed-hour tensors."""
        timestep_times = pd.to_datetime(
            [
                "2023-10-03T16:23:00.072211Z",
                "2023-10-03T17:22:57.608076Z",
                "2023-10-03T18:22:57.155630Z",
                "2023-10-03T19:22:54.545799Z",
                "2023-10-03T20:22:53.517316Z",
            ],
            utc=True,
        )
        record = {
            LABEL_COL: 1,
            "raster_bundle_path": "unused.npz",
            "date": "2023-10-03",
            "hour": 20,
            "lon": -95.0,
            "num_coal_units": 1,
            "num_ng_units": 2,
            "total_nameplate_capacity_mw": 500.0,
            "avg_heat_input": 100.0,
            "avg_pwr_gen": 50.0,
        }
        record.update({f"timestep_time_t{index}": value.isoformat() for index, value in enumerate(timestep_times)})
        stats = NormalizationStats(
            image_keys=MODEL_IMAGE_KEYS,
            image_center=(0.0,) * len(MODEL_IMAGE_KEYS),
            image_scale=(1.0,) * len(MODEL_IMAGE_KEYS),
            image_valid_pixels=(0,) * len(MODEL_IMAGE_KEYS),
            feature_names=MODEL_FEATURE_NAMES,
            feature_mean=(0.0,) * len(MODEL_FEATURE_NAMES),
            feature_std=(1.0,) * len(MODEL_FEATURE_NAMES),
            delta_threshold=100.0,
            training_records=1,
        )

        with tempfile.TemporaryDirectory() as temporary_dir:
            dataframe_dir = Path(temporary_dir)
            pd.DataFrame([record]).to_csv(dataframe_dir / "train_df.csv", index=False)
            dataset = NOxDataset(
                "train",
                stats,
                dataset_dir=dataframe_dir,
                dataframe_dir=dataframe_dir,
                load_images=False,
            )

        expected = np.diff(timestep_times.asi8) / 3_600_000_000_000
        self.assertEqual(dataset.elapsed_hours.dtype, np.float32)
        np.testing.assert_allclose(dataset.elapsed_hours[0], expected, rtol=1e-6)
        self.assertEqual(dataset.elapsed_hours.shape, (1, SEQUENCE_TIMESTEPS - 1))
