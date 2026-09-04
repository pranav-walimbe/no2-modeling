"""Tessellate TEMPO Level 2 footprints onto fixed equal-area AOI grids."""

import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import netCDF4 as nc
import numpy as np
import shapely

from config import (
    IMG_RANGE,
    IMG_SIZE,
    MIN_PIXEL_CLOUD,
    TEMPO_CELL_OVERLAP_FLOOR_KM2,
    TEMPO_EFFECTIVE_SAMPLE_FLOOR,
    TEMPO_GOOD_QUALITY_FLAG,
)
from preprocessing.stratify_utils import CONUS_TO_WGS84, WGS84_TO_CONUS

METRES_PER_KM = 1000.0
CORNER_COUNT = 4
QUALITY_FILL_VALUE = -1
SAVED_RASTER_NAMES = (
    "no2",
    "weighted_cloud_fraction",
    "good_quality_fraction",
    "retrieval_uncertainty",
    "sum_weight",
)


@dataclass(frozen=True)
class AoiGrid:
    """Fixed equal-area grid covering one AOI in EPSG:5070 metres."""

    aoi_id: int
    x_m: float
    y_m: float
    size: int = IMG_SIZE
    extent_km: float = IMG_RANGE

    @classmethod
    def from_lon_lat(
        cls,
        aoi_id: int,
        lon: float,
        lat: float,
        size: int = IMG_SIZE,
        extent_km: float = IMG_RANGE,
    ) -> "AoiGrid":
        """Build the grid from an AOI centroid given in WGS84 degrees."""
        x_m, y_m = WGS84_TO_CONUS.transform(lon, lat)
        return cls(aoi_id=aoi_id, x_m=float(x_m), y_m=float(y_m), size=size, extent_km=extent_km)

    @property
    def cell_size_m(self) -> float:
        """Return grid spacing in metres."""
        return self.extent_km * METRES_PER_KM / self.size

    @property
    def cell_area_km2(self) -> float:
        """Return one output cell's area in square kilometres."""
        return (self.cell_size_m / METRES_PER_KM) ** 2

    def cell_centres(self) -> tuple[np.ndarray, np.ndarray]:
        """Return projected cell-centre easting and northing, ordered north-up."""
        offsets = (np.arange(self.size, dtype=np.float64) - (self.size - 1) / 2) * self.cell_size_m
        return np.meshgrid(self.x_m + offsets, self.y_m - offsets)

    def bounds_wgs84(self) -> tuple[float, float, float, float]:
        """Return the lon and lat box covering the output grid."""
        half_m = self.extent_km * METRES_PER_KM / 2
        corner_x = [self.x_m - half_m, self.x_m - half_m, self.x_m + half_m, self.x_m + half_m]
        corner_y = [self.y_m - half_m, self.y_m + half_m, self.y_m - half_m, self.y_m + half_m]
        lons, lats = CONUS_TO_WGS84.transform(corner_x, corner_y)
        return float(np.min(lons)), float(np.min(lats)), float(np.max(lons)), float(np.max(lats))


@dataclass(frozen=True)
class GranulePixels:
    """Native pixels carrying values, diagnostics, and footprint corners."""

    values: np.ndarray
    longitudes: np.ndarray
    latitudes: np.ndarray
    corner_longitudes: np.ndarray
    corner_latitudes: np.ndarray
    cloud_fractions: np.ndarray
    quality_flags: np.ndarray
    uncertainties: np.ndarray

    def __len__(self) -> int:
        return int(self.values.size)

    def select_bounds(self, lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> "GranulePixels":
        """Keep footprints whose corner bounding box intersects a lon-lat box."""
        keep = (
            (np.min(self.corner_longitudes, axis=1) <= lon_max)
            & (np.max(self.corner_longitudes, axis=1) >= lon_min)
            & (np.min(self.corner_latitudes, axis=1) <= lat_max)
            & (np.max(self.corner_latitudes, axis=1) >= lat_min)
        )
        return _select_pixels(self, keep)

    def select_grid(self, grid: AoiGrid) -> "GranulePixels":
        """Keep footprints whose projected bounds intersect an AOI grid."""
        corner_x, corner_y = WGS84_TO_CONUS.transform(self.corner_longitudes, self.corner_latitudes)
        half_extent = grid.extent_km * METRES_PER_KM / 2
        keep = (
            (np.min(corner_x, axis=1) <= grid.x_m + half_extent)
            & (np.max(corner_x, axis=1) >= grid.x_m - half_extent)
            & (np.min(corner_y, axis=1) <= grid.y_m + half_extent)
            & (np.max(corner_y, axis=1) >= grid.y_m - half_extent)
        )
        return _select_pixels(self, keep)


@dataclass(frozen=True)
class RegriddedRaster:
    """One NO2 raster with independently reusable cell diagnostics."""

    aoi_id: int
    no2: np.ndarray
    sum_weight: np.ndarray
    sum_weight_squared: np.ndarray
    effective_sample_size: np.ndarray
    total_overlap_area: np.ndarray
    native_pixel_count: np.ndarray
    accepted_pixel_count: np.ndarray
    weighted_cloud_fraction: np.ndarray
    good_quality_fraction: np.ndarray
    main_data_quality_flag: np.ndarray
    retrieval_uncertainty: np.ndarray
    input_pixel_count: int

    @property
    def count(self) -> np.ndarray:
        """Return accepted contributor count as a compatibility alias."""
        return self.accepted_pixel_count


def _select_pixels(pixels: GranulePixels, keep: np.ndarray) -> GranulePixels:
    # Apply one row mask without allowing fields to drift out of alignment
    return GranulePixels(
        values=pixels.values[keep],
        longitudes=pixels.longitudes[keep],
        latitudes=pixels.latitudes[keep],
        corner_longitudes=pixels.corner_longitudes[keep],
        corner_latitudes=pixels.corner_latitudes[keep],
        cloud_fractions=pixels.cloud_fractions[keep],
        quality_flags=pixels.quality_flags[keep],
        uncertainties=pixels.uncertainties[keep],
    )


def empty_pixels() -> GranulePixels:
    """Return a pixel set with no rows and the correct array shapes."""
    return GranulePixels(
        values=np.empty(0, dtype=np.float64),
        longitudes=np.empty(0, dtype=np.float64),
        latitudes=np.empty(0, dtype=np.float64),
        corner_longitudes=np.empty((0, CORNER_COUNT), dtype=np.float64),
        corner_latitudes=np.empty((0, CORNER_COUNT), dtype=np.float64),
        cloud_fractions=np.empty(0, dtype=np.float64),
        quality_flags=np.empty(0, dtype=np.int16),
        uncertainties=np.empty(0, dtype=np.float64),
    )


def concatenate_pixels(parts: list[GranulePixels]) -> GranulePixels:
    """Join pixels from the granules of one scan that share an AOI seam."""
    usable = [part for part in parts if len(part) > 0]
    if not usable:
        return empty_pixels()
    return GranulePixels(
        values=np.concatenate([part.values for part in usable]),
        longitudes=np.concatenate([part.longitudes for part in usable]),
        latitudes=np.concatenate([part.latitudes for part in usable]),
        corner_longitudes=np.concatenate([part.corner_longitudes for part in usable]),
        corner_latitudes=np.concatenate([part.corner_latitudes for part in usable]),
        cloud_fractions=np.concatenate([part.cloud_fractions for part in usable]),
        quality_flags=np.concatenate([part.quality_flags for part in usable]),
        uncertainties=np.concatenate([part.uncertainties for part in usable]),
    )


def _filled(variable: nc.Variable) -> np.ndarray:
    # Widen before filling so integer flag variables can represent missing data
    return np.ma.filled(np.ma.asarray(variable[:]).astype(np.float64), np.nan)


def read_granule_pixels(
    path: str,
    *,
    max_solar_zenith: float | None = None,
    max_snow_ice_fraction: float | None = None,
) -> GranulePixels:
    """Read geometrically valid Level 2 pixels without cloud or quality filtering.

    Args:
        path: Path to a TEMPO NO2 V04 Level 2 granule.
        max_solar_zenith: Optional solar zenith limit in degrees.
        max_snow_ice_fraction: Optional snow and ice fraction limit.

    Returns:
        Flattened pixels with diagnostics and four footprint corners.
    """
    with nc.Dataset(path) as dataset:
        product = dataset.groups["product"]
        geolocation = dataset.groups["geolocation"]
        support = dataset.groups["support_data"]
        values = _filled(product.variables["vertical_column_troposphere"])
        uncertainties = _filled(product.variables["vertical_column_troposphere_uncertainty"])
        quality = _filled(product.variables["main_data_quality_flag"])
        cloud = _filled(support.variables["eff_cloud_fraction"])
        latitudes = _filled(geolocation.variables["latitude"])
        longitudes = _filled(geolocation.variables["longitude"])
        corner_latitudes = _filled(geolocation.variables["latitude_bounds"])
        corner_longitudes = _filled(geolocation.variables["longitude_bounds"])
        solar_zenith = _filled(geolocation.variables["solar_zenith_angle"]) if max_solar_zenith is not None else None
        snow_ice = _filled(support.variables["snow_ice_fraction"]) if max_snow_ice_fraction is not None else None

    keep = (
        np.isfinite(latitudes)
        & np.isfinite(longitudes)
        & np.isfinite(cloud)
        & np.isfinite(quality)
        & np.isfinite(corner_latitudes).all(axis=2)
        & np.isfinite(corner_longitudes).all(axis=2)
    )
    if solar_zenith is not None:
        keep &= solar_zenith <= max_solar_zenith
    if snow_ice is not None:
        keep &= snow_ice <= max_snow_ice_fraction

    return GranulePixels(
        values=values[keep],
        longitudes=longitudes[keep],
        latitudes=latitudes[keep],
        corner_longitudes=corner_longitudes[keep],
        corner_latitudes=corner_latitudes[keep],
        cloud_fractions=cloud[keep],
        quality_flags=quality[keep].astype(np.int16),
        uncertainties=uncertainties[keep],
    )


def _unobserved_raster(grid: AoiGrid, input_pixel_count: int = 0) -> RegriddedRaster:
    # Keep missing values distinct from real zeros in continuous diagnostics
    shape = (grid.size, grid.size)
    return RegriddedRaster(
        aoi_id=grid.aoi_id,
        no2=np.full(shape, np.nan),
        sum_weight=np.zeros(shape),
        sum_weight_squared=np.zeros(shape),
        effective_sample_size=np.zeros(shape),
        total_overlap_area=np.zeros(shape),
        native_pixel_count=np.zeros(shape, dtype=np.int32),
        accepted_pixel_count=np.zeros(shape, dtype=np.int32),
        weighted_cloud_fraction=np.full(shape, np.nan),
        good_quality_fraction=np.full(shape, np.nan),
        main_data_quality_flag=np.full(shape, QUALITY_FILL_VALUE, dtype=np.int16),
        retrieval_uncertainty=np.full(shape, np.nan),
        input_pixel_count=input_pixel_count,
    )


def _grid_cells(grid: AoiGrid) -> np.ndarray:
    # Build the fixed destination cells once for the vectorized overlap query
    half_extent = grid.extent_km * METRES_PER_KM / 2
    left = grid.x_m - half_extent
    top = grid.y_m + half_extent
    columns = np.arange(grid.size)
    rows = np.arange(grid.size)
    min_x, min_y = np.meshgrid(left + columns * grid.cell_size_m, top - (rows + 1) * grid.cell_size_m)
    return shapely.box(min_x, min_y, min_x + grid.cell_size_m, min_y + grid.cell_size_m).ravel()


def tessellate(
    pixels: GranulePixels,
    grid: AoiGrid,
    *,
    max_cloud_fraction: float = MIN_PIXEL_CLOUD,
    good_quality_flag: int = TEMPO_GOOD_QUALITY_FLAG,
) -> RegriddedRaster:
    """Area-weight native footprint intersections onto an AOI grid.

    Args:
        pixels: Geometrically valid native pixels near the AOI.
        grid: Fixed AOI grid that paired scans must share.
        max_cloud_fraction: Largest cloud fraction accepted into the NO2 mean.
        good_quality_flag: Native quality flag accepted into the NO2 mean.

    Returns:
        Area-weighted NO2 and independent cell diagnostic rasters.
    """
    if not 0 <= max_cloud_fraction <= 1:
        raise ValueError("max_cloud_fraction must be in [0, 1]")
    if len(pixels) == 0:
        return _unobserved_raster(grid)

    shape = (grid.size, grid.size)
    sum_weight = np.zeros(shape)
    sum_weight_squared = np.zeros(shape)
    weighted_no2 = np.zeros(shape)
    native_count = np.zeros(shape, dtype=np.int32)
    accepted_count = np.zeros(shape, dtype=np.int32)
    all_weight = np.zeros(shape)
    cloud_weight = np.zeros(shape)
    weighted_cloud = np.zeros(shape)
    good_quality_weight = np.zeros(shape)
    uncertainty_weight = np.zeros(shape)
    weighted_uncertainty = np.zeros(shape)
    worst_quality = np.full(shape, QUALITY_FILL_VALUE, dtype=np.int16)

    corner_x, corner_y = WGS84_TO_CONUS.transform(pixels.corner_longitudes, pixels.corner_latitudes)
    coordinates = np.stack([corner_x, corner_y], axis=2)
    polygons = shapely.polygons(coordinates)
    usable_indices = np.flatnonzero(shapely.is_valid(polygons) & (shapely.area(polygons) > 0))
    cells = _grid_cells(grid)
    if usable_indices.size:
        pairs = shapely.STRtree(cells).query(polygons[usable_indices], predicate="intersects")
        pixel_indices = usable_indices[pairs[0]]
        cell_indices = pairs[1]
        overlap_km2 = shapely.area(shapely.intersection(polygons[pixel_indices], cells[cell_indices]))
        overlap_km2 = np.asarray(overlap_km2) / METRES_PER_KM**2
        positive = overlap_km2 > 0
        pixel_indices = pixel_indices[positive]
        cell_indices = cell_indices[positive]
        overlap_km2 = overlap_km2[positive]
    else:
        pixel_indices = np.empty(0, dtype=np.int64)
        cell_indices = np.empty(0, dtype=np.int64)
        overlap_km2 = np.empty(0)

    # Accumulate only actual footprint-cell intersections, never a dense pixel-cell matrix
    flat_all_weight = all_weight.ravel()
    np.add.at(flat_all_weight, cell_indices, overlap_km2)
    np.add.at(native_count.ravel(), cell_indices, 1)
    np.add.at(cloud_weight.ravel(), cell_indices, overlap_km2)
    np.add.at(
        weighted_cloud.ravel(),
        cell_indices,
        overlap_km2 * pixels.cloud_fractions[pixel_indices],
    )
    quality = pixels.quality_flags[pixel_indices]
    np.maximum.at(worst_quality.ravel(), cell_indices, quality)
    good = quality == good_quality_flag
    np.add.at(good_quality_weight.ravel(), cell_indices[good], overlap_km2[good])

    accepted = (
        good & (pixels.cloud_fractions[pixel_indices] <= max_cloud_fraction) & np.isfinite(pixels.values[pixel_indices])
    )
    accepted_pixels = pixel_indices[accepted]
    accepted_cells = cell_indices[accepted]
    accepted_overlap = overlap_km2[accepted]
    np.add.at(sum_weight.ravel(), accepted_cells, accepted_overlap)
    np.add.at(sum_weight_squared.ravel(), accepted_cells, np.square(accepted_overlap))
    np.add.at(weighted_no2.ravel(), accepted_cells, accepted_overlap * pixels.values[accepted_pixels])
    np.add.at(accepted_count.ravel(), accepted_cells, 1)
    has_uncertainty = np.isfinite(pixels.uncertainties[accepted_pixels])
    uncertainty_pixels = accepted_pixels[has_uncertainty]
    uncertainty_cells = accepted_cells[has_uncertainty]
    uncertainty_overlap = accepted_overlap[has_uncertainty]
    np.add.at(uncertainty_weight.ravel(), uncertainty_cells, uncertainty_overlap)
    np.add.at(
        weighted_uncertainty.ravel(),
        uncertainty_cells,
        uncertainty_overlap * pixels.uncertainties[uncertainty_pixels],
    )

    no2 = np.full(shape, np.nan)
    observed = sum_weight > 0
    no2[observed] = weighted_no2[observed] / sum_weight[observed]
    effective = np.zeros(shape)
    effective[observed] = np.square(sum_weight[observed]) / sum_weight_squared[observed]
    cloud = np.full(shape, np.nan)
    diagnosed = cloud_weight > 0
    cloud[diagnosed] = weighted_cloud[diagnosed] / cloud_weight[diagnosed]
    good_fraction = np.full(shape, np.nan)
    covered = all_weight > 0
    good_fraction[covered] = good_quality_weight[covered] / all_weight[covered]
    uncertainty = np.full(shape, np.nan)
    has_uncertainty = uncertainty_weight > 0
    uncertainty[has_uncertainty] = weighted_uncertainty[has_uncertainty] / uncertainty_weight[has_uncertainty]
    return RegriddedRaster(
        aoi_id=grid.aoi_id,
        no2=no2,
        sum_weight=sum_weight,
        sum_weight_squared=sum_weight_squared,
        effective_sample_size=effective,
        total_overlap_area=all_weight,
        native_pixel_count=native_count,
        accepted_pixel_count=accepted_count,
        weighted_cloud_fraction=cloud,
        good_quality_fraction=good_fraction,
        main_data_quality_flag=worst_quality,
        retrieval_uncertainty=uncertainty,
        input_pixel_count=len(pixels),
    )


def apply_cell_mask(
    raster: RegriddedRaster,
    overlap_floor_km2: float = TEMPO_CELL_OVERLAP_FLOOR_KM2,
    effective_sample_floor: float = TEMPO_EFFECTIVE_SAMPLE_FLOOR,
) -> RegriddedRaster:
    """Mask NO2 below support floors while retaining every diagnostic raster.

    Args:
        raster: Raster carrying raw cell diagnostics.
        overlap_floor_km2: Accepted overlap area required per output cell.
        effective_sample_floor: Effective accepted sample count required per cell.

    Returns:
        Raster with only the NO2 values masked by the support rule.
    """
    if overlap_floor_km2 <= 0 and effective_sample_floor <= 0:
        return raster
    keep = raster.sum_weight >= overlap_floor_km2
    if effective_sample_floor > 0:
        keep &= raster.effective_sample_size >= effective_sample_floor
    return replace(raster, no2=np.where(keep, raster.no2, np.nan))


def write_raster_npz(raster: RegriddedRaster, destination: str | Path) -> None:
    """Atomically save the five modeling rasters as compressed float32 arrays.

    Args:
        raster: Regridded AOI scan with its internal diagnostics.
        destination: Output `.npz` path.
    """
    output_path = Path(destination)
    if output_path.suffix != ".npz":
        raise ValueError("destination must end in .npz")
    arrays = {name: np.asarray(getattr(raster, name), dtype=np.float32) for name in SAVED_RASTER_NAMES}
    expected_shape = raster.no2.shape
    if expected_shape != (IMG_SIZE, IMG_SIZE):
        raise ValueError(f"raster shape must be {(IMG_SIZE, IMG_SIZE)}, found {expected_shape}")
    if any(array.shape != expected_shape for array in arrays.values()):
        raise ValueError("all saved rasters must share the NO2 shape")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(temporary, **arrays)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def regrid_aoi_raster(
    pixels: GranulePixels,
    grid: AoiGrid,
    *,
    max_cloud_fraction: float = MIN_PIXEL_CLOUD,
    overlap_floor_km2: float = TEMPO_CELL_OVERLAP_FLOOR_KM2,
    effective_sample_floor: float = TEMPO_EFFECTIVE_SAMPLE_FLOOR,
) -> RegriddedRaster:
    """Build and mask one fixed-grid AOI raster.

    Args:
        pixels: Native pixels near the AOI.
        grid: Fixed output grid that paired scans must share.
        max_cloud_fraction: Largest cloud fraction accepted into the NO2 mean.
        overlap_floor_km2: Accepted overlap area required per output cell.
        effective_sample_floor: Effective accepted samples required per cell.

    Returns:
        Tessellated NO2 and its unmasked diagnostic rasters.
    """
    raster = tessellate(pixels, grid, max_cloud_fraction=max_cloud_fraction)
    return apply_cell_mask(raster, overlap_floor_km2, effective_sample_floor)


def regrid_aoi_scan(
    granule_paths: list[str],
    grid: AoiGrid,
    *,
    max_cloud_fraction: float = MIN_PIXEL_CLOUD,
    **regrid_options: float,
) -> RegriddedRaster:
    """Build one AOI raster from all granules in a single scan.

    Args:
        granule_paths: Granules of one scan that cover the AOI.
        grid: Fixed AOI grid.
        max_cloud_fraction: Largest cloud fraction accepted into the NO2 mean.
        regrid_options: Additional options forwarded to `regrid_aoi_raster`.

    Returns:
        Tessellated raster for that AOI and scan.
    """
    parts = [read_granule_pixels(path).select_grid(grid) for path in granule_paths]
    return regrid_aoi_raster(
        concatenate_pixels(parts),
        grid,
        max_cloud_fraction=max_cloud_fraction,
        **regrid_options,
    )
