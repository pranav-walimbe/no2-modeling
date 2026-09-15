import tempfile
import unittest
from pathlib import Path

import numpy as np

from config import IMG_SIZE, SEQUENCE_TIMESTEPS
from preprocessing.generate_dataset_utils import (
    HOTSPOT_COLUMN_COL,
    HOTSPOT_FINITE_FRACTION_COLUMNS,
    HOTSPOT_ROW_COL,
    TEMPERATURE_RASTER_NAME,
    WIND_U_RASTER_NAME,
    WIND_V_RASTER_NAME,
    derive_raster_features,
    hotspot_finite_fraction,
    select_hotspot_cell,
)


class HotspotCoverageTest(unittest.TestCase):
    """Verify source-cluster selection and complete-window filtering."""

    def test_combines_unit_counts_within_one_raster_cell(self) -> None:
        """Facilities sharing a cell form one source cluster."""
        selected = select_hotspot_cell(
            (0.0, 0.5, 10.0),
            (-0.5, -1.0, -0.5),
            (2, 3, 4),
        )

        self.assertEqual(selected, (12, 12))

    def test_breaks_equal_unit_count_by_distance_to_aoi_centre(self) -> None:
        """The nearer cluster wins when unit counts are equal."""
        selected = select_hotspot_cell(
            (-10.0, 5.0),
            (-0.5, -0.5),
            (4, 4),
        )

        self.assertEqual(selected, (12, 13))

    def test_hotspot_fraction_uses_all_nine_cells(self) -> None:
        """One missing cell reduces 3 by 3 coverage below one."""
        valid = np.ones((IMG_SIZE, IMG_SIZE), dtype=bool)
        valid[11, 11] = False

        self.assertEqual(hotspot_finite_fraction(valid, 12, 12), 8 / 9)

    def test_rejects_hotspot_without_a_complete_window(self) -> None:
        """A boundary source cannot silently use a clipped neighborhood."""
        valid = np.ones((IMG_SIZE, IMG_SIZE), dtype=bool)

        with self.assertRaisesRegex(ValueError, "too close to the AOI boundary"):
            hotspot_finite_fraction(valid, 0, 12)

    def test_rejects_missing_hotspot_cell_despite_global_coverage(self) -> None:
        """High whole-raster coverage cannot override hotspot missingness."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            scan_path = root / "scan.npz"
            weather_path = root / "weather.npz"
            shape = (IMG_SIZE, IMG_SIZE)
            no2 = np.ones(shape, dtype=np.float32)
            no2[12, 12] = np.nan
            np.savez(
                scan_path,
                no2=no2,
                weighted_cloud_fraction=np.zeros(shape, dtype=np.float32),
                good_quality_fraction=np.ones(shape, dtype=np.float32),
                retrieval_uncertainty=np.ones(shape, dtype=np.float32),
            )
            np.savez(
                weather_path,
                **{
                    TEMPERATURE_RASTER_NAME: np.ones(shape, dtype=np.float32),
                    WIND_U_RASTER_NAME: np.zeros(shape, dtype=np.float32),
                    WIND_V_RASTER_NAME: np.zeros(shape, dtype=np.float32),
                },
            )

            with self.assertRaisesRegex(ValueError, "hotspot NO2 coverage must be at least 100%"):
                derive_raster_features(
                    (str(scan_path),) * SEQUENCE_TIMESTEPS,
                    (str(weather_path),) * SEQUENCE_TIMESTEPS,
                    hotspot_row=12,
                    hotspot_column=12,
                )

    def test_retains_hotspot_audit_fields(self) -> None:
        """Successful records expose the selected cell and timestep coverage."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            scan_path = root / "scan.npz"
            weather_path = root / "weather.npz"
            shape = (IMG_SIZE, IMG_SIZE)
            np.savez(
                scan_path,
                no2=np.ones(shape, dtype=np.float32),
                weighted_cloud_fraction=np.zeros(shape, dtype=np.float32),
                good_quality_fraction=np.ones(shape, dtype=np.float32),
                retrieval_uncertainty=np.ones(shape, dtype=np.float32),
            )
            np.savez(
                weather_path,
                **{
                    TEMPERATURE_RASTER_NAME: np.ones(shape, dtype=np.float32),
                    WIND_U_RASTER_NAME: np.zeros(shape, dtype=np.float32),
                    WIND_V_RASTER_NAME: np.zeros(shape, dtype=np.float32),
                },
            )

            _, features = derive_raster_features(
                (str(scan_path),) * SEQUENCE_TIMESTEPS,
                (str(weather_path),) * SEQUENCE_TIMESTEPS,
                hotspot_row=12,
                hotspot_column=13,
            )

        self.assertEqual(features[HOTSPOT_ROW_COL], 12)
        self.assertEqual(features[HOTSPOT_COLUMN_COL], 13)
        self.assertTrue(all(features[column] == 1.0 for column in HOTSPOT_FINITE_FRACTION_COLUMNS))


if __name__ == "__main__":
    unittest.main()
