"""Oversample TEMPO Level 2 native pixels onto one fixed AOI grid.

Adapts the physical oversampling core of POPY v0.4 (MIT licensed) to one raster
per AOI per scan, following Sun et al. 2018 (doi:10.5194/amt-11-6679-2018).
"""

from dataclasses import dataclass

import netCDF4 as nc
import numpy as np

from config import (
    IMG_RANGE,
    IMG_SIZE,
    MIN_PIXEL_CLOUD,
    TEMPO_CELL_WEIGHT_FLOOR,
    TEMPO_OVERSAMPLE_FACTOR,
    TEMPO_SELECTION_MARGIN_KM,
    TEMPO_SRF_EXPONENT_OUTER,
    TEMPO_SRF_EXPONENT_STEP,
    TEMPO_SRF_EXPONENT_XTRACK,
    TEMPO_SRF_INFLATE,
    TEMPO_SRF_MIN_WEIGHT,
)
from preprocessing.stratify_utils import CONUS_TO_WGS84, WGS84_TO_CONUS

METRES_PER_KM = 1000.0
CORNER_COUNT = 4
QUALITY_FLAG_GOOD = 0


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
        """Grid spacing in metres."""
        return self.extent_km * METRES_PER_KM / self.size

    def cell_centres(self) -> tuple[np.ndarray, np.ndarray]:
        """Return projected cell-centre easting and northing, ordered north-up."""
        offsets = (np.arange(self.size, dtype=np.float64) - (self.size - 1) / 2) * self.cell_size_m
        return np.meshgrid(self.x_m + offsets, self.y_m - offsets)

    def bounds_wgs84(self, margin_km: float = 0.0) -> tuple[float, float, float, float]:
        """Return the lon and lat box covering the grid plus a margin in km."""
        half_m = self.extent_km * METRES_PER_KM / 2 + margin_km * METRES_PER_KM
        corner_x = [self.x_m - half_m, self.x_m - half_m, self.x_m + half_m, self.x_m + half_m]
        corner_y = [self.y_m - half_m, self.y_m + half_m, self.y_m - half_m, self.y_m + half_m]
        lons, lats = CONUS_TO_WGS84.transform(corner_x, corner_y)
        return float(np.min(lons)), float(np.min(lats)), float(np.max(lons)), float(np.max(lats))


@dataclass(frozen=True)
class GranulePixels:
    """Quality-filtered native pixels carrying their footprint corners."""

    values: np.ndarray
    longitudes: np.ndarray
    latitudes: np.ndarray
    corner_longitudes: np.ndarray
    corner_latitudes: np.ndarray

    def __len__(self) -> int:
        return int(self.values.size)

    def select_bounds(self, lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> "GranulePixels":
        """Keep pixels whose centre falls inside a lon and lat box."""
        keep = (
            (self.longitudes >= lon_min)
            & (self.longitudes <= lon_max)
            & (self.latitudes >= lat_min)
            & (self.latitudes <= lat_max)
        )
        return GranulePixels(
            values=self.values[keep],
            longitudes=self.longitudes[keep],
            latitudes=self.latitudes[keep],
            corner_longitudes=self.corner_longitudes[keep],
            corner_latitudes=self.corner_latitudes[keep],
        )


@dataclass(frozen=True)
class RegriddedRaster:
    """One AOI raster with the per-cell bookkeeping the mask rule needs."""

    aoi_id: int
    no2: np.ndarray
    weight: np.ndarray
    count: np.ndarray
    native_pixel_count: int


@dataclass(frozen=True)
class _PixelResponse:
    # Projected super-Gaussian parameters for every contributing pixel
    centre: np.ndarray
    unit_xtrack: np.ndarray
    unit_step: np.ndarray
    width_xtrack: np.ndarray
    width_step: np.ndarray
    values: np.ndarray


def empty_pixels() -> GranulePixels:
    """Return a pixel set with no rows and the correct array shapes."""
    return GranulePixels(
        values=np.empty(0, dtype=np.float64),
        longitudes=np.empty(0, dtype=np.float64),
        latitudes=np.empty(0, dtype=np.float64),
        corner_longitudes=np.empty((0, CORNER_COUNT), dtype=np.float64),
        corner_latitudes=np.empty((0, CORNER_COUNT), dtype=np.float64),
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
    )


def _filled(variable: nc.Variable) -> np.ndarray:
    # Widen to float before filling so integer flag variables accept NaN
    return np.ma.filled(np.ma.asarray(variable[:]).astype(np.float64), np.nan)


def read_granule_pixels(
    path: str,
    *,
    max_cloud_fraction: float = MIN_PIXEL_CLOUD,
    max_solar_zenith: float | None = None,
    max_snow_ice_fraction: float | None = None,
) -> GranulePixels:
    """Read one granule and keep the native pixels that pass the science filters.

    A granule covers many AOIs, so call this once per file and reuse the result
    across every AOI it covers rather than reopening it per AOI.

    Args:
        path: Path to a TEMPO NO2 V04 Level 2 granule.
        max_cloud_fraction: Upper limit on effective cloud fraction.
        max_solar_zenith: Optional upper limit on solar zenith angle in degrees.
        max_snow_ice_fraction: Optional upper limit on snow and ice fraction.

    Returns:
        Pixels flattened to one dimension with their four footprint corners.
    """
    with nc.Dataset(path) as dataset:
        product = dataset.groups["product"]
        geolocation = dataset.groups["geolocation"]
        support = dataset.groups["support_data"]
        values = _filled(product.variables["vertical_column_troposphere"])
        quality = _filled(product.variables["main_data_quality_flag"])
        cloud = _filled(support.variables["eff_cloud_fraction"])
        latitudes = _filled(geolocation.variables["latitude"])
        longitudes = _filled(geolocation.variables["longitude"])
        corner_latitudes = _filled(geolocation.variables["latitude_bounds"])
        corner_longitudes = _filled(geolocation.variables["longitude_bounds"])
        solar_zenith = _filled(geolocation.variables["solar_zenith_angle"])
        snow_ice = _filled(support.variables["snow_ice_fraction"])

    keep = (
        (quality == QUALITY_FLAG_GOOD)
        & (cloud <= max_cloud_fraction)
        & np.isfinite(values)
        & np.isfinite(latitudes)
        & np.isfinite(longitudes)
        & np.isfinite(corner_latitudes).all(axis=2)
        & np.isfinite(corner_longitudes).all(axis=2)
    )
    if max_solar_zenith is not None:
        keep &= solar_zenith <= max_solar_zenith
    if max_snow_ice_fraction is not None:
        keep &= snow_ice <= max_snow_ice_fraction

    return GranulePixels(
        values=values[keep],
        longitudes=longitudes[keep],
        latitudes=latitudes[keep],
        corner_longitudes=corner_longitudes[keep],
        corner_latitudes=corner_latitudes[keep],
    )


def _edge_midpoints(corner_x: np.ndarray, corner_y: np.ndarray, first: int, second: int) -> np.ndarray:
    # Take the midpoint of one footprint edge for every pixel
    return np.stack([corner_x[:, first] + corner_x[:, second], corner_y[:, first] + corner_y[:, second]], axis=1) / 2


def _super_gaussian_width(fwhm: np.ndarray, exponent: float, outer_exponent: float, inflate: float) -> np.ndarray:
    # Convert a full width at half maximum into the super-Gaussian scale parameter
    return inflate * fwhm / (2 * np.log(2) ** (1 / (exponent * outer_exponent)))


def _build_pixel_response(
    pixels: GranulePixels,
    exponent_xtrack: float,
    exponent_step: float,
    exponent_outer: float,
    inflate: float,
) -> _PixelResponse:
    # Project footprint corners into response axes and super-Gaussian widths
    corner_x, corner_y = WGS84_TO_CONUS.transform(pixels.corner_longitudes, pixels.corner_latitudes)
    corner_x, corner_y = np.asarray(corner_x), np.asarray(corner_y)
    north_edge = _edge_midpoints(corner_x, corner_y, 0, 1)
    south_edge = _edge_midpoints(corner_x, corner_y, 2, 3)
    west_edge = _edge_midpoints(corner_x, corner_y, 1, 2)
    east_edge = _edge_midpoints(corner_x, corner_y, 3, 0)
    axis_xtrack = south_edge - north_edge
    axis_step = east_edge - west_edge
    fwhm_xtrack = np.linalg.norm(axis_xtrack, axis=1)
    fwhm_step = np.linalg.norm(axis_step, axis=1)

    usable = (fwhm_xtrack > 0) & (fwhm_step > 0)
    fwhm_xtrack, fwhm_step = fwhm_xtrack[usable], fwhm_step[usable]
    return _PixelResponse(
        centre=((north_edge + south_edge) / 2)[usable],
        unit_xtrack=axis_xtrack[usable] / fwhm_xtrack[:, None],
        unit_step=axis_step[usable] / fwhm_step[:, None],
        width_xtrack=_super_gaussian_width(fwhm_xtrack, exponent_xtrack, exponent_outer, inflate),
        width_step=_super_gaussian_width(fwhm_step, exponent_step, exponent_outer, inflate),
        values=pixels.values[usable],
    )


def _response_weights(
    response: _PixelResponse,
    grid: AoiGrid,
    exponent_xtrack: float,
    exponent_step: float,
    exponent_outer: float,
) -> np.ndarray:
    # Evaluate every pixel response on every grid cell
    grid_x, grid_y = grid.cell_centres()
    offset_x = grid_x.ravel()[None, :] - response.centre[:, 0, None]
    offset_y = grid_y.ravel()[None, :] - response.centre[:, 1, None]
    local_xtrack = offset_x * response.unit_xtrack[:, 0, None] + offset_y * response.unit_xtrack[:, 1, None]
    local_step = offset_x * response.unit_step[:, 0, None] + offset_y * response.unit_step[:, 1, None]
    radial = np.abs(local_xtrack / response.width_xtrack[:, None]) ** exponent_xtrack
    radial += np.abs(local_step / response.width_step[:, None]) ** exponent_step
    return np.exp(-(radial**exponent_outer))


def _unobserved_raster(grid: AoiGrid) -> RegriddedRaster:
    # Return the raster for an AOI scan that no usable pixel reached
    shape = (grid.size, grid.size)
    return RegriddedRaster(
        aoi_id=grid.aoi_id,
        no2=np.full(shape, np.nan),
        weight=np.zeros(shape),
        count=np.zeros(shape, dtype=np.int64),
        native_pixel_count=0,
    )


def oversample(
    pixels: GranulePixels,
    grid: AoiGrid,
    *,
    exponent_xtrack: float = TEMPO_SRF_EXPONENT_XTRACK,
    exponent_step: float = TEMPO_SRF_EXPONENT_STEP,
    exponent_outer: float = TEMPO_SRF_EXPONENT_OUTER,
    inflate: float = TEMPO_SRF_INFLATE,
    min_weight: float = TEMPO_SRF_MIN_WEIGHT,
) -> RegriddedRaster:
    """Integrate every pixel's spatial response over the AOI grid.

    Each native pixel is represented by a 2-D super-Gaussian rather than a
    uniform polygon, so a cell receives a share of the pixel's value in
    proportion to the instrument sensitivity that falls in it.

    Args:
        pixels: Quality-filtered pixels, already restricted to the AOI.
        grid: Fixed AOI grid that both scans of a pair must share.
        exponent_xtrack: Shape exponent along the north-south detector array.
        exponent_step: Shape exponent along the east-west mirror-step axis.
        exponent_outer: Outer exponent applied to the summed radial term.
        inflate: Multiplier stretching each response beyond its footprint.
        min_weight: Response below which a pixel does not count toward a cell.

    Returns:
        The weighted-mean NO2 raster with per-cell weight and sample count.
    """
    if len(pixels) == 0:
        return _unobserved_raster(grid)
    response = _build_pixel_response(pixels, exponent_xtrack, exponent_step, exponent_outer, inflate)
    if response.values.size == 0:
        return _unobserved_raster(grid)

    weights = _response_weights(response, grid, exponent_xtrack, exponent_step, exponent_outer)
    weight_sum = weights.sum(axis=0)
    observed = weight_sum > 0
    no2 = np.full(weight_sum.shape, np.nan)
    no2[observed] = (weights.T @ response.values)[observed] / weight_sum[observed]

    shape = (grid.size, grid.size)
    return RegriddedRaster(
        aoi_id=grid.aoi_id,
        no2=no2.reshape(shape),
        weight=weight_sum.reshape(shape),
        count=(weights >= min_weight).sum(axis=0).reshape(shape).astype(np.int64),
        native_pixel_count=int(response.values.size),
    )


def aggregate_fine_raster(raster: RegriddedRaster, grid: AoiGrid, factor: int) -> RegriddedRaster:
    """Reduce a raster oversampled at a finer grid onto the output AOI grid.

    Fine cells are combined with their own sample weights, so a well-sampled
    corner of an output cell carries more of that cell's value than a sparse one.

    Args:
        raster: Raster built on a grid `factor` times finer than the output.
        grid: The output AOI grid.
        factor: Fine cells per output cell along each axis.

    Returns:
        The raster on the output grid, with mean weight and peak sample count.
    """
    size = grid.size
    blocks = (size, factor, size, factor)
    filled = np.where(np.isfinite(raster.no2), raster.no2, 0.0)
    weighted = (filled * raster.weight).reshape(blocks).sum(axis=(1, 3))
    total = raster.weight.reshape(blocks).sum(axis=(1, 3))
    observed = total > 0
    no2 = np.full((size, size), np.nan)
    no2[observed] = weighted[observed] / total[observed]
    return RegriddedRaster(
        aoi_id=raster.aoi_id,
        no2=no2,
        # Mean keeps weight on the same scale as a directly sampled cell
        weight=raster.weight.reshape(blocks).mean(axis=(1, 3)),
        # Peak rather than sum, because one pixel reaches many fine cells
        count=raster.count.reshape(blocks).max(axis=(1, 3)),
        native_pixel_count=raster.native_pixel_count,
    )


def apply_weight_floor(raster: RegriddedRaster, weight_floor: float) -> RegriddedRaster:
    """Mark cells whose total response weight falls below the floor as unobserved."""
    if weight_floor <= 0:
        return raster
    masked = np.where(raster.weight >= weight_floor, raster.no2, np.nan)
    return RegriddedRaster(
        aoi_id=raster.aoi_id,
        no2=masked,
        weight=raster.weight,
        count=raster.count,
        native_pixel_count=raster.native_pixel_count,
    )


def regrid_aoi_raster(
    pixels: GranulePixels,
    grid: AoiGrid,
    *,
    factor: int = TEMPO_OVERSAMPLE_FACTOR,
    weight_floor: float = TEMPO_CELL_WEIGHT_FLOOR,
    **oversample_options: float,
) -> RegriddedRaster:
    """Build one AOI raster the way the pilot settled on.

    Oversampling to a finer grid and aggregating beat the direct grid against
    NASA Level 3 in 19 of 20 AOI scans, so that is the default path. Cells whose
    total weight falls below the floor are returned as unobserved.

    Args:
        pixels: Quality-filtered pixels, already restricted to the AOI.
        grid: Fixed output grid that both scans of a pair must share.
        factor: Fine cells per output cell along each axis, 1 to sample directly.
        weight_floor: Total weight below which an output cell is unobserved.
        oversample_options: Overrides forwarded to `oversample`.

    Returns:
        The finished raster on the output grid.
    """
    if factor < 1:
        raise ValueError("factor must be at least 1")
    if factor == 1:
        return apply_weight_floor(oversample(pixels, grid, **oversample_options), weight_floor)
    fine_grid = AoiGrid(
        aoi_id=grid.aoi_id, x_m=grid.x_m, y_m=grid.y_m, size=grid.size * factor, extent_km=grid.extent_km
    )
    fine = oversample(pixels, fine_grid, **oversample_options)
    return apply_weight_floor(aggregate_fine_raster(fine, grid, factor), weight_floor)


def regrid_aoi_scan(
    granule_paths: list[str],
    grid: AoiGrid,
    *,
    margin_km: float = TEMPO_SELECTION_MARGIN_KM,
    max_cloud_fraction: float = MIN_PIXEL_CLOUD,
    **regrid_options: float,
) -> RegriddedRaster:
    """Build one AOI raster from the granules of a single scan.

    This reopens each granule, so callers regridding many AOIs from the same
    granule should call `read_granule_pixels` themselves and reuse the result.

    Args:
        granule_paths: Granules of one scan that cover the AOI.
        grid: Fixed AOI grid.
        margin_km: Distance outside the grid within which pixel centres are kept.
        max_cloud_fraction: Upper limit on effective cloud fraction.
        regrid_options: Overrides forwarded to `regrid_aoi_raster`.

    Returns:
        The regridded raster for that AOI and scan.
    """
    bounds = grid.bounds_wgs84(margin_km)
    parts = [
        read_granule_pixels(path, max_cloud_fraction=max_cloud_fraction).select_bounds(*bounds)
        for path in granule_paths
    ]
    return regrid_aoi_raster(concatenate_pixels(parts), grid, **regrid_options)
