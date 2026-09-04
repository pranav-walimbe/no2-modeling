"""Tests for TEMPO Level 2 oversampling onto a fixed AOI grid."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import IMG_RANGE, IMG_SIZE  # noqa: E402
from eda.popy_reference import DEFAULT_RESPONSE_CUTOFF  # noqa: E402
from eda.popy_reference import regrid as popy_regrid  # noqa: E402
from preprocessing.regrid import (  # noqa: E402
    OVERSAMPLE_FACTOR,
    AoiGrid,
    GranulePixels,
    aggregate_fine_raster,
    apply_cell_mask,
    concatenate_pixels,
    empty_pixels,
    oversample,
    regrid_aoi_raster,
)
from preprocessing.stratify_utils import CONUS_TO_WGS84, WGS84_TO_CONUS  # noqa: E402

AOI_LON = -82.13
AOI_LAT = 33.62
PIXEL_XTRACK_M = 2100.0
PIXEL_STEP_M = 4700.0


def _pixel_at(x_m: float, y_m: float, value: float) -> GranulePixels:
    # Build one axis-aligned native pixel centred on a projected coordinate
    half_a = PIXEL_XTRACK_M / 2
    half_b = PIXEL_STEP_M / 2
    corner_x = np.array([[x_m + half_b, x_m - half_b, x_m - half_b, x_m + half_b]])
    corner_y = np.array([[y_m + half_a, y_m + half_a, y_m - half_a, y_m - half_a]])
    corner_lon, corner_lat = CONUS_TO_WGS84.transform(corner_x, corner_y)
    lon, lat = CONUS_TO_WGS84.transform(x_m, y_m)
    return GranulePixels(
        values=np.array([value], dtype=np.float64),
        longitudes=np.array([float(lon)]),
        latitudes=np.array([float(lat)]),
        corner_longitudes=np.asarray(corner_lon),
        corner_latitudes=np.asarray(corner_lat),
    )


def _skewed_pixel_at(x_m: float, y_m: float, value: float) -> GranulePixels:
    # Build a non-parallelogram footprint in the documented SW, SE, NE, NW order
    offsets = np.array([[-2400.0, -1100.0], [2400.0, -1300.0], [2100.0, 1100.0], [-1900.0, 900.0]])
    corner_x = (x_m + offsets[:, 0])[None, :]
    corner_y = (y_m + offsets[:, 1])[None, :]
    corner_lon, corner_lat = CONUS_TO_WGS84.transform(corner_x, corner_y)
    lon, lat = CONUS_TO_WGS84.transform(x_m, y_m)
    return GranulePixels(
        values=np.array([value], dtype=np.float64),
        longitudes=np.array([float(lon)]),
        latitudes=np.array([float(lat)]),
        corner_longitudes=np.asarray(corner_lon),
        corner_latitudes=np.asarray(corner_lat),
    )


def _tiled_pixels(grid: AoiGrid, value: float) -> GranulePixels:
    # Tile the AOI and a margin with identical pixels
    steps_x = np.arange(-9, 10) * PIXEL_STEP_M
    steps_y = np.arange(-19, 20) * PIXEL_XTRACK_M
    parts = [_pixel_at(grid.x_m + dx, grid.y_m + dy, value) for dx in steps_x for dy in steps_y]
    return concatenate_pixels(parts)


class AoiGridTest(unittest.TestCase):
    def test_grid_spacing_matches_configured_extent_and_size(self) -> None:
        grid = AoiGrid.from_lon_lat(1, AOI_LON, AOI_LAT)

        self.assertEqual(grid.size, IMG_SIZE)
        self.assertAlmostEqual(grid.cell_size_m, IMG_RANGE * 1000 / IMG_SIZE)
        self.assertAlmostEqual(grid.cell_size_m, 1500.0)

    def test_cell_centres_span_the_extent_and_run_north_up(self) -> None:
        grid = AoiGrid.from_lon_lat(1, AOI_LON, AOI_LAT)

        grid_x, grid_y = grid.cell_centres()

        self.assertEqual(grid_x.shape, (IMG_SIZE, IMG_SIZE))
        self.assertGreater(grid_y[0, 0], grid_y[-1, 0])
        span = grid_x.max() - grid_x.min() + grid.cell_size_m
        self.assertAlmostEqual(span, IMG_RANGE * 1000)

    def test_bounds_widen_with_the_requested_margin(self) -> None:
        grid = AoiGrid.from_lon_lat(1, AOI_LON, AOI_LAT)

        tight = grid.bounds_wgs84()
        wide = grid.bounds_wgs84(margin_km=8.0)

        self.assertLess(wide[0], tight[0])
        self.assertGreater(wide[2], tight[2])


class OversampleTest(unittest.TestCase):
    def test_uniform_field_reproduces_its_value_everywhere(self) -> None:
        grid = AoiGrid.from_lon_lat(7, AOI_LON, AOI_LAT)

        raster = oversample(_tiled_pixels(grid, 5.0e15), grid)

        self.assertEqual(raster.no2.shape, (IMG_SIZE, IMG_SIZE))
        np.testing.assert_allclose(raster.no2, 5.0e15, rtol=1e-9)
        self.assertTrue((raster.sum_weight > 0).all())
        self.assertTrue((raster.count > 0).all())

    def test_empty_pixels_give_an_unobserved_raster(self) -> None:
        grid = AoiGrid.from_lon_lat(7, AOI_LON, AOI_LAT)

        raster = oversample(empty_pixels(), grid)

        self.assertTrue(np.isnan(raster.no2).all())
        self.assertEqual(raster.sum_weight.sum(), 0.0)
        self.assertEqual(raster.native_pixel_count, 0)

    def test_single_pixel_peaks_at_its_own_centre(self) -> None:
        grid = AoiGrid.from_lon_lat(7, AOI_LON, AOI_LAT)
        offset = 12_000.0

        raster = oversample(_pixel_at(grid.x_m + offset, grid.y_m, 3.0e15), grid)

        peak_row, peak_col = np.unravel_index(np.argmax(raster.sum_weight), raster.sum_weight.shape)
        grid_x, grid_y = grid.cell_centres()
        self.assertLess(abs(grid_x[peak_row, peak_col] - (grid.x_m + offset)), grid.cell_size_m)
        self.assertLess(abs(grid_y[peak_row, peak_col] - grid.y_m), grid.cell_size_m)

    def test_weighted_mean_lies_between_the_contributing_values(self) -> None:
        grid = AoiGrid.from_lon_lat(7, AOI_LON, AOI_LAT)
        low, high = 1.0e15, 9.0e15
        pixels = concatenate_pixels(
            [
                _pixel_at(grid.x_m - PIXEL_STEP_M, grid.y_m, low),
                _pixel_at(grid.x_m + PIXEL_STEP_M, grid.y_m, high),
            ]
        )

        raster = oversample(pixels, grid)

        observed = raster.no2[np.isfinite(raster.no2)]
        self.assertTrue((observed >= low - 1).all())
        self.assertTrue((observed <= high + 1).all())

    def test_response_widens_when_pixels_are_inflated(self) -> None:
        grid = AoiGrid.from_lon_lat(7, AOI_LON, AOI_LAT)
        pixels = _pixel_at(grid.x_m, grid.y_m, 4.0e15)

        tight = oversample(pixels, grid, inflate=1.0)
        loose = oversample(pixels, grid, inflate=2.0)

        self.assertGreater(loose.count.sum(), tight.count.sum())

    def test_both_scans_of_a_pair_share_one_grid(self) -> None:
        grid = AoiGrid.from_lon_lat(7, AOI_LON, AOI_LAT)
        shifted = AoiGrid.from_lon_lat(7, AOI_LON, AOI_LAT)

        first = oversample(_tiled_pixels(grid, 4.0e15), grid)
        second = oversample(_tiled_pixels(shifted, 6.0e15), shifted)

        np.testing.assert_allclose(second.no2 - first.no2, 2.0e15, rtol=1e-9)


class RegridRasterTest(unittest.TestCase):
    def test_aggregated_uniform_field_keeps_its_value(self) -> None:
        grid = AoiGrid.from_lon_lat(7, AOI_LON, AOI_LAT)

        raster = regrid_aoi_raster(_tiled_pixels(grid, 5.0e15), grid)

        self.assertEqual(raster.no2.shape, (IMG_SIZE, IMG_SIZE))
        np.testing.assert_allclose(raster.no2, 5.0e15, rtol=1e-9)

    def test_aggregation_keeps_weight_on_the_direct_scale(self) -> None:
        grid = AoiGrid.from_lon_lat(7, AOI_LON, AOI_LAT)
        pixels = _tiled_pixels(grid, 5.0e15)

        direct = oversample(pixels, grid)
        aggregated = regrid_aoi_raster(pixels, grid, weight_floor=0.0)

        self.assertAlmostEqual(float(np.median(aggregated.sum_weight)), float(np.median(direct.sum_weight)), delta=0.15)

    def test_factor_one_skips_aggregation(self) -> None:
        grid = AoiGrid.from_lon_lat(7, AOI_LON, AOI_LAT)
        pixels = _tiled_pixels(grid, 3.0e15)

        direct = oversample(pixels, grid)
        through = regrid_aoi_raster(pixels, grid, factor=1, weight_floor=0.0)

        np.testing.assert_allclose(through.no2, direct.no2, rtol=1e-12)

    def test_aggregate_reduces_the_fine_grid_to_the_output_grid(self) -> None:
        grid = AoiGrid.from_lon_lat(7, AOI_LON, AOI_LAT)
        fine_grid = AoiGrid(
            aoi_id=7,
            x_m=grid.x_m,
            y_m=grid.y_m,
            size=grid.size * OVERSAMPLE_FACTOR,
            extent_km=grid.extent_km,
        )

        fine = oversample(_tiled_pixels(grid, 2.0e15), fine_grid)
        reduced = aggregate_fine_raster(fine, grid, OVERSAMPLE_FACTOR)

        self.assertEqual(fine.no2.shape, (IMG_SIZE * OVERSAMPLE_FACTOR,) * 2)
        self.assertEqual(reduced.no2.shape, (IMG_SIZE, IMG_SIZE))
        self.assertLessEqual(int(reduced.count.max()), int(fine.count.max()))

    def test_weight_floor_masks_only_cells_below_it(self) -> None:
        grid = AoiGrid.from_lon_lat(7, AOI_LON, AOI_LAT)
        raster = oversample(_pixel_at(grid.x_m, grid.y_m, 4.0e15), grid)
        floor = 0.5

        masked = apply_cell_mask(raster, floor)

        self.assertTrue(np.isnan(masked.no2[raster.sum_weight < floor]).all())
        below = raster.sum_weight < floor
        self.assertTrue(below.any())
        np.testing.assert_allclose(masked.no2[~below], raster.no2[~below], rtol=1e-12)
        np.testing.assert_allclose(masked.sum_weight, raster.sum_weight, rtol=1e-12)

    def test_zero_floor_leaves_the_raster_untouched(self) -> None:
        grid = AoiGrid.from_lon_lat(7, AOI_LON, AOI_LAT)
        raster = oversample(_tiled_pixels(grid, 1.0e15), grid)

        self.assertIs(apply_cell_mask(raster, 0.0), raster)

    def test_factor_below_one_is_rejected(self) -> None:
        grid = AoiGrid.from_lon_lat(7, AOI_LON, AOI_LAT)

        with self.assertRaises(ValueError):
            regrid_aoi_raster(empty_pixels(), grid, factor=0)


class ProjectiveGeometryTest(unittest.TestCase):
    def test_matches_the_pinned_popy_reference_on_skewed_footprints(self) -> None:
        grid = AoiGrid(aoi_id=3, x_m=1_000_000.0, y_m=2_000_000.0, size=16, extent_km=24)
        pixels = concatenate_pixels(
            [
                _skewed_pixel_at(grid.x_m - 1800.0, grid.y_m + 900.0, 4.0e15),
                _skewed_pixel_at(grid.x_m + 2100.0, grid.y_m - 1400.0, 7.0e15),
            ]
        )

        ours = oversample(pixels, grid, response_cutoff=DEFAULT_RESPONSE_CUTOFF)
        reference = popy_regrid(pixels, grid, area_weight=False, response_cutoff=DEFAULT_RESPONSE_CUTOFF)

        observed = np.isfinite(reference.no2)
        self.assertTrue(observed.any())
        np.testing.assert_allclose(ours.no2[observed], reference.no2[observed], rtol=1e-9)
        np.testing.assert_allclose(ours.sum_weight, reference.sum_weight, rtol=1e-9, atol=0.0)
        np.testing.assert_allclose(
            ours.effective_sample_size[observed], reference.effective_sample_size[observed], rtol=1e-9
        )

    def test_constant_field_survives_a_non_parallelogram_footprint(self) -> None:
        grid = AoiGrid(aoi_id=3, x_m=1_000_000.0, y_m=2_000_000.0, size=12, extent_km=18)
        parts = [
            _skewed_pixel_at(grid.x_m + column * 4200.0, grid.y_m + row * 2100.0, 5.0e15)
            for column in range(-4, 5)
            for row in range(-8, 9)
        ]

        raster = oversample(concatenate_pixels(parts), grid)

        observed = np.isfinite(raster.no2)
        self.assertTrue(observed.all())
        np.testing.assert_allclose(raster.no2[observed], 5.0e15, rtol=1e-9)

    def test_cutoff_keeps_distant_cells_empty(self) -> None:
        grid = AoiGrid(aoi_id=3, x_m=1_000_000.0, y_m=2_000_000.0, size=32, extent_km=120)

        raster = oversample(_pixel_at(grid.x_m, grid.y_m, 4.0e15), grid)

        self.assertEqual(raster.sum_weight[0, 0], 0.0)
        self.assertTrue(np.isnan(raster.no2[0, 0]))
        self.assertGreater(raster.sum_weight[grid.size // 2, grid.size // 2], 0.0)

    def test_local_windows_touch_far_fewer_cells_than_the_full_grid(self) -> None:
        grid = AoiGrid(aoi_id=3, x_m=1_000_000.0, y_m=2_000_000.0, size=64, extent_km=240)

        raster = oversample(_pixel_at(grid.x_m, grid.y_m, 4.0e15), grid)

        reached = int((raster.sum_weight > 0).sum())
        self.assertGreater(reached, 0)
        self.assertLess(reached, grid.size * grid.size / 10)

    def test_zero_cutoff_reaches_every_cell_the_response_can_represent(self) -> None:
        grid = AoiGrid(aoi_id=3, x_m=1_000_000.0, y_m=2_000_000.0, size=16, extent_km=18)

        cut = oversample(_pixel_at(grid.x_m, grid.y_m, 4.0e15), grid)
        full = oversample(_pixel_at(grid.x_m, grid.y_m, 4.0e15), grid, response_cutoff=0.0)

        self.assertTrue((full.sum_weight > 0).all())
        self.assertGreater(int((full.sum_weight > 0).sum()), int((cut.sum_weight > 0).sum()))

    def test_cutoff_outside_the_unit_interval_is_rejected(self) -> None:
        grid = AoiGrid.from_lon_lat(7, AOI_LON, AOI_LAT)

        with self.assertRaises(ValueError):
            oversample(_pixel_at(grid.x_m, grid.y_m, 1.0e15), grid, response_cutoff=1.0)


class EffectiveSampleTest(unittest.TestCase):
    def test_one_isolated_pixel_is_worth_one_sample(self) -> None:
        grid = AoiGrid.from_lon_lat(7, AOI_LON, AOI_LAT)

        raster = oversample(_pixel_at(grid.x_m, grid.y_m, 4.0e15), grid)

        observed = raster.sum_weight > 0
        np.testing.assert_allclose(raster.effective_sample_size[observed], 1.0, rtol=1e-9)

    def test_colocated_pixels_count_as_two_samples(self) -> None:
        grid = AoiGrid.from_lon_lat(7, AOI_LON, AOI_LAT)
        pixels = concatenate_pixels(
            [
                _pixel_at(grid.x_m, grid.y_m, 4.0e15),
                _pixel_at(grid.x_m, grid.y_m, 4.0e15),
            ]
        )

        raster = oversample(pixels, grid)

        observed = raster.sum_weight > 0
        np.testing.assert_allclose(raster.effective_sample_size[observed], 2.0, rtol=1e-9)

    def test_effective_samples_never_exceed_the_pixel_count(self) -> None:
        grid = AoiGrid.from_lon_lat(7, AOI_LON, AOI_LAT)
        pixels = _tiled_pixels(grid, 5.0e15)

        raster = oversample(pixels, grid)

        self.assertLessEqual(float(raster.effective_sample_size.max()), float(len(pixels)))
        self.assertGreaterEqual(float(raster.effective_sample_size[raster.sum_weight > 0].min()), 1.0 - 1e-9)


class CellMaskTest(unittest.TestCase):
    def test_effective_sample_floor_masks_thinly_supported_cells(self) -> None:
        grid = AoiGrid.from_lon_lat(7, AOI_LON, AOI_LAT)
        raster = oversample(_tiled_pixels(grid, 5.0e15), grid)
        floor = float(np.median(raster.effective_sample_size))

        masked = apply_cell_mask(raster, 0.0, effective_sample_floor=floor)

        below = raster.effective_sample_size < floor
        self.assertTrue(below.any())
        self.assertTrue(np.isnan(masked.no2[below]).all())
        np.testing.assert_allclose(masked.effective_sample_size, raster.effective_sample_size, rtol=1e-12)

    def test_both_floors_apply_together(self) -> None:
        grid = AoiGrid.from_lon_lat(7, AOI_LON, AOI_LAT)
        raster = oversample(_tiled_pixels(grid, 5.0e15), grid)

        weight_only = apply_cell_mask(raster, 0.02)
        both = apply_cell_mask(raster, 0.02, effective_sample_floor=2.0)

        self.assertLessEqual(int(np.isfinite(both.no2).sum()), int(np.isfinite(weight_only.no2).sum()))


class GranulePixelsTest(unittest.TestCase):
    def test_select_bounds_keeps_only_centres_inside_the_box(self) -> None:
        grid = AoiGrid.from_lon_lat(7, AOI_LON, AOI_LAT)
        inside_lon, inside_lat = CONUS_TO_WGS84.transform(grid.x_m, grid.y_m)
        far_x, far_y = WGS84_TO_CONUS.transform(inside_lon, inside_lat)
        pixels = concatenate_pixels(
            [
                _pixel_at(float(far_x), float(far_y), 1.0e15),
                _pixel_at(float(far_x) + 400_000.0, float(far_y), 2.0e15),
            ]
        )

        selected = pixels.select_bounds(*grid.bounds_wgs84())

        self.assertEqual(len(selected), 1)
        self.assertAlmostEqual(selected.values[0], 1.0e15)

    def test_concatenating_no_pixels_gives_an_empty_set(self) -> None:
        self.assertEqual(len(concatenate_pixels([empty_pixels(), empty_pixels()])), 0)


if __name__ == "__main__":
    unittest.main()
