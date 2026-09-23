"""Regression tests for masked-pretraining cache inventories."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

MASKED_PREPROCESSING_DIR = (
    Path(__file__).resolve().parents[1] / "src" / "pretraining" / "masked-model" / "preprocessing"
)
sys.path.insert(0, str(MASKED_PREPROCESSING_DIR))

import dataset_generation_utils as generation_utils  # noqa: E402
from preprocessing.generate_dataset_utils import ScanTask, WeatherTask  # noqa: E402

from config import IMG_SIZE  # noqa: E402


class CacheInventoryTest(unittest.TestCase):
    def test_inventory_hits_skip_filesystem_processing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            tempo_path = root / "tempo-key.npz"
            weather_path = root / "weather-key.npz"
            np.savez_compressed(tempo_path, no2=np.ones((IMG_SIZE, IMG_SIZE), dtype=np.float32))
            np.savez_compressed(weather_path, placeholder=np.ones(1, dtype=np.float32))
            scan = ScanTask("tempo-key", 1, 0.0, 0.0, ("granule.nc",), str(tempo_path))
            weather = WeatherTask("weather-key", 1, 0.0, 0.0, "weather.grib2", "weather.grib2", str(weather_path))
            mapped_tasks: list[list[object]] = []

            def empty_parallel_map(function: object, tasks: object, workers: int) -> iter:
                del function, workers
                mapped_tasks.append(list(tasks))
                return iter(())

            with (
                patch.object(generation_utils, "candidate_cache_tasks", return_value=(scan, weather)),
                patch.object(generation_utils, "bounded_parallel_map", side_effect=empty_parallel_map),
            ):
                outcomes = generation_utils.discover_candidate_batch(
                    [{"candidate_index": 0, "tempo_cached": True, "weather_cached": True}],
                    tempo_root=root,
                    tempo_cache_dir=root,
                    hrrr_root=root,
                    weather_cache_dir=root,
                    workers=1,
                    tempo_cache_additions=set(),
                    weather_cache_additions=set(),
                )

        self.assertEqual([tasks for tasks in mapped_tasks], [[], []])
        self.assertEqual(outcomes[0].status, generation_utils.VALID_STATUS)
        self.assertEqual(outcomes[0].tempo_cache_path, str(tempo_path))
        self.assertEqual(outcomes[0].weather_cache_path, str(weather_path))


if __name__ == "__main__":
    unittest.main()
