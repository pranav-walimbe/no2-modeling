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
    TEMPO_EFFECTIVE_SAMPLE_FLOOR,
    TEMPO_RESPONSE_CUTOFF,
    TEMPO_SRF_EXPONENT_OUTER,
    TEMPO_SRF_EXPONENT_STEP,
    TEMPO_SRF_EXPONENT_XTRACK,
    TEMPO_SRF_INFLATE,
)
from preprocessing.stratify_utils import CONUS_TO_WGS84, WGS84_TO_CONUS

METRES_PER_KM = 1000.0
CORNER_COUNT = 4
QUALITY_FLAG_GOOD = 0
SRF_MIN_WEIGHT = 1e-3  # response below this does not count a pixel toward a cell
SELECTION_MARGIN_KM = 8.0  # keep pixels centred this far outside the AOI grid
OVERSAMPLE_FACTOR = 3  # fine cells per output cell along each axis


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
    uncertainties: np.ndarray | None = None

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
            uncertainties=None if self.uncertainties is None else self.uncertainties[keep],
        )


@dataclass(frozen=True)
class RegriddedRaster:
    """One AOI raster with the per-cell bookkeeping the mask rule needs."""

    aoi_id: int
    no2: np.ndarray
    sum_weight: np.ndarray
    sum_weight_squared: np.ndarray
    effective_sample_size: np.ndarray
    count: np.ndarray
    native_pixel_count: int


@dataclass(frozen=True)
class _PixelResponse:
    # Projective transform and super-Gaussian widths for every contributing pixel
    centre: np.ndarray
    transform: np.ndarray
    width_xtrack: np.ndarray
    width_step: np.ndarray
    support_radius: np.ndarray
    values: np.ndarray


def empty_pixels() -> GranulePixels:
    """Return a pixel set with no rows and the correct array shapes."""
    return GranulePixels(
        values=np.empty(0, dtype=np.float64),
        longitudes=np.empty(0, dtype=np.float64),
        latitudes=np.empty(0, dtype=np.float64),
        corner_longitudes=np.empty((0, CORNER_COUNT), dtype=np.float64),
        corner_latitudes=np.empty((0, CORNER_COUNT), dtype=np.float64),
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
        uncertainties=(
            np.concatenate([part.uncertainties for part in usable])
            if all(part.uncertainties is not None for part in usable)
            else None
        ),
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
        uncertainties = _filled(product.variables["vertical_column_troposphere_uncertainty"])
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
        uncertainties=uncertainties[keep],
    )


def _homography(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    # Fit one projective map per pixel with the lower-right coefficient fixed to one
    count = source.shape[0]
    source_u, source_v = source[:, :, 0], source[:, :, 1]
    target_x, target_y = destination[:, :, 0], destination[:, :, 1]
    matrix = np.zeros((count, 8, 8), dtype=np.float64)
    matrix[:, 0::2, 0] = source_u
    matrix[:, 0::2, 1] = source_v
    matrix[:, 0::2, 2] = 1.0
    matrix[:, 0::2, 6] = -target_x * source_u
    matrix[:, 0::2, 7] = -target_x * source_v
    matrix[:, 1::2, 3] = source_u
    matrix[:, 1::2, 4] = source_v
    matrix[:, 1::2, 5] = 1.0
    matrix[:, 1::2, 6] = -target_y * source_u
    matrix[:, 1::2, 7] = -target_y * source_v
    coefficients = np.linalg.solve(matrix, destination.reshape(count, 8))
    return np.concatenate([coefficients, np.ones((count, 1))], axis=1).reshape(count, 3, 3)


def _width_fraction(exponent: float, outer_exponent: float, inflate: float) -> float:
    # Super-Gaussian scale parameter as a fraction of the full width at half maximum
    return inflate / (2 * np.log(2) ** (1 / (exponent * outer_exponent)))


def _build_pixel_response(
    pixels: GranulePixels,
    exponent_xtrack: float,
    exponent_step: float,
    exponent_outer: float,
    inflate: float,
    response_cutoff: float,
) -> _PixelResponse:
    # Map each footprint quadrilateral onto its own response rectangle
    corner_x, corner_y = WGS84_TO_CONUS.transform(pixels.corner_longitudes, pixels.corner_latitudes)
    centre_x, centre_y = WGS84_TO_CONUS.transform(pixels.longitudes, pixels.latitudes)
    corners = np.stack([np.asarray(corner_x), np.asarray(corner_y)], axis=2)
    centre = np.column_stack([np.asarray(centre_x), np.asarray(centre_y)])
    offsets = corners - centre[:, None, :]
    fwhm_xtrack = np.linalg.norm(offsets[:, 2:4].mean(axis=1) - offsets[:, 0:2].mean(axis=1), axis=1)
    fwhm_step = np.linalg.norm(offsets[:, 1:3].mean(axis=1) - offsets[:, [0, 3]].mean(axis=1), axis=1)

    usable = np.isfinite(offsets).all(axis=(1, 2)) & (fwhm_xtrack > 0) & (fwhm_step > 0)
    offsets, centre = offsets[usable], centre[usable]
    fwhm_xtrack, fwhm_step = fwhm_xtrack[usable], fwhm_step[usable]
    rectangle = np.stack([np.array([-1.0, -1.0, 1.0, 1.0]), np.array([-1.0, 1.0, 1.0, -1.0])], axis=1)
    destination = rectangle[None, :, :] * np.column_stack([fwhm_xtrack, fwhm_step])[:, None, :] / 2

    fraction_xtrack = _width_fraction(exponent_xtrack, exponent_outer, inflate)
    fraction_step = _width_fraction(exponent_step, exponent_outer, inflate)
    return _PixelResponse(
        centre=centre,
        transform=_homography(offsets, destination),
        width_xtrack=fwhm_xtrack * fraction_xtrack,
        width_step=fwhm_step * fraction_step,
        support_radius=_support_radius(
            offsets, exponent_xtrack, exponent_step, exponent_outer, inflate, response_cutoff
        ),
        values=pixels.values[usable],
    )


def _support_radius(
    offsets: np.ndarray,
    exponent_xtrack: float,
    exponent_step: float,
    exponent_outer: float,
    inflate: float,
    response_cutoff: float,
) -> np.ndarray:
    # Bound the distance at which a footprint can still clear the response cutoff
    if response_cutoff <= 0:
        return np.full(offsets.shape[0], np.inf)
    decay = -np.log(response_cutoff)
    limit_xtrack = _width_fraction(exponent_xtrack, exponent_outer, inflate) * decay ** (
        1 / (exponent_xtrack * exponent_outer)
    )
    limit_step = _width_fraction(exponent_step, exponent_outer, inflate) * decay ** (
        1 / (exponent_step * exponent_outer)
    )
    footprint_radius = np.linalg.norm(offsets, axis=2).max(axis=1)
    return footprint_radius * 2 * max(limit_xtrack, limit_step)


def _cell_window(grid: AoiGrid, centre: np.ndarray, radius: float) -> tuple[slice, slice]:
    # Index the smallest block of cells covering one pixel's support disc
    if not np.isfinite(radius):
        return slice(0, grid.size), slice(0, grid.size)
    origin = (grid.size - 1) / 2
    first_column = np.floor((centre[0] - radius - grid.x_m) / grid.cell_size_m + origin)
    last_column = np.ceil((centre[0] + radius - grid.x_m) / grid.cell_size_m + origin)
    first_row = np.floor((grid.y_m - centre[1] - radius) / grid.cell_size_m + origin)
    last_row = np.ceil((grid.y_m - centre[1] + radius) / grid.cell_size_m + origin)
    columns = slice(max(0, int(first_column)), min(grid.size, int(last_column) + 1))
    rows = slice(max(0, int(first_row)), min(grid.size, int(last_row) + 1))
    return rows, columns


def _unobserved_raster(grid: AoiGrid) -> RegriddedRaster:
    # Return the raster for an AOI scan that no usable pixel reached
    shape = (grid.size, grid.size)
    return RegriddedRaster(
        aoi_id=grid.aoi_id,
        no2=np.full(shape, np.nan),
        sum_weight=np.zeros(shape),
        sum_weight_squared=np.zeros(shape),
        effective_sample_size=np.zeros(shape),
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
    min_weight: float = SRF_MIN_WEIGHT,
    response_cutoff: float = TEMPO_RESPONSE_CUTOFF,
) -> RegriddedRaster:
    """Integrate every pixel's spatial response over the AOI grid.

    Args:
        pixels: Quality-filtered pixels, already restricted to the AOI.
        grid: Fixed AOI grid that both scans of a pair must share.
        exponent_xtrack: Shape exponent along the north-south detector array.
        exponent_step: Shape exponent along the east-west mirror-step axis.
        exponent_outer: Outer exponent applied to the summed radial term.
        inflate: Multiplier stretching each response beyond its footprint.
        min_weight: Response below which a pixel does not count toward a cell.
        response_cutoff: Response below which a pixel does not reach a cell at all.

    Returns:
        The weighted-mean NO2 raster with per-cell weight sums and sample count.
    """
    if not 0 <= response_cutoff < 1:
        raise ValueError("response_cutoff must be in [0, 1)")
    if len(pixels) == 0:
        return _unobserved_raster(grid)
    response = _build_pixel_response(
        pixels, exponent_xtrack, exponent_step, exponent_outer, inflate, response_cutoff
    )
    if response.values.size == 0:
        return _unobserved_raster(grid)

    shape = (grid.size, grid.size)
    grid_x, grid_y = grid.cell_centres()
    sum_weight = np.zeros(shape)
    sum_weight_squared = np.zeros(shape)
    sum_weighted_value = np.zeros(shape)
    count = np.zeros(shape, dtype=np.int64)
    for index in range(response.values.size):
        centre = response.centre[index]
        radius = response.support_radius[index]
        rows, columns = _cell_window(grid, centre, radius)
        if rows.start >= rows.stop or columns.start >= columns.stop:
            continue
        offset_x = grid_x[rows, columns] - centre[0]
        offset_y = grid_y[rows, columns] - centre[1]
        reached = (
            np.ones(offset_x.shape, dtype=bool)
            if not np.isfinite(radius)
            else np.hypot(offset_x, offset_y) <= radius
        )
        if not reached.any():
            continue
        weights = _window_weights(
            response, index, offset_x[reached], offset_y[reached], exponent_xtrack, exponent_step, exponent_outer
        )
        weights[weights < response_cutoff] = 0.0
        # These slices are views into the full rasters
        sum_weight[rows, columns][reached] += weights
        sum_weight_squared[rows, columns][reached] += np.square(weights)
        sum_weighted_value[rows, columns][reached] += weights * response.values[index]
        count[rows, columns][reached] += weights >= min_weight

    observed = sum_weight > 0
    no2 = np.full(shape, np.nan)
    no2[observed] = sum_weighted_value[observed] / sum_weight[observed]
    effective = np.zeros(shape)
    effective[observed] = np.square(sum_weight[observed]) / sum_weight_squared[observed]
    return RegriddedRaster(
        aoi_id=grid.aoi_id,
        no2=no2,
        sum_weight=sum_weight,
        sum_weight_squared=sum_weight_squared,
        effective_sample_size=effective,
        count=count,
        native_pixel_count=int(response.values.size),
    )


def _window_weights(
    response: _PixelResponse,
    index: int,
    offset_x: np.ndarray,
    offset_y: np.ndarray,
    exponent_xtrack: float,
    exponent_step: float,
    exponent_outer: float,
) -> np.ndarray:
    # Evaluate one pixel's super-Gaussian response on the cells inside its window
    transform = response.transform[index]
    # POPY v0.4 skips the homogeneous divide whose horizon can flip the sign outside the footprint
    local = np.column_stack([offset_x, offset_y, np.ones(offset_x.size)]) @ transform[:2].T
    radial = np.abs(local[:, 0] / response.width_xtrack[index]) ** exponent_xtrack
    radial += np.abs(local[:, 1] / response.width_step[index]) ** exponent_step
    return np.exp(-(radial**exponent_outer))


def aggregate_fine_raster(raster: RegriddedRaster, grid: AoiGrid, factor: int) -> RegriddedRaster:
    """Reduce a raster oversampled at a finer grid onto the output AOI grid.

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
    weighted = (filled * raster.sum_weight).reshape(blocks).sum(axis=(1, 3))
    total = raster.sum_weight.reshape(blocks).sum(axis=(1, 3))
    observed = total > 0
    no2 = np.full((size, size), np.nan)
    no2[observed] = weighted[observed] / total[observed]
    # Mean keeps both sums on the same scale as a directly sampled cell
    sum_weight = raster.sum_weight.reshape(blocks).mean(axis=(1, 3))
    sum_weight_squared = raster.sum_weight_squared.reshape(blocks).mean(axis=(1, 3))
    effective = np.zeros((size, size))
    # Recomputed from the aggregated sums rather than averaged
    effective[observed] = np.square(sum_weight[observed]) / sum_weight_squared[observed]
    return RegriddedRaster(
        aoi_id=raster.aoi_id,
        no2=no2,
        sum_weight=sum_weight,
        sum_weight_squared=sum_weight_squared,
        effective_sample_size=effective,
        # Peak rather than sum because one pixel reaches many fine cells
        count=raster.count.reshape(blocks).max(axis=(1, 3)),
        native_pixel_count=raster.native_pixel_count,
    )


def apply_cell_mask(
    raster: RegriddedRaster,
    weight_floor: float,
    effective_sample_floor: float = TEMPO_EFFECTIVE_SAMPLE_FLOOR,
) -> RegriddedRaster:
    """Mark cells below the total-weight or effective-sample floor as unobserved.

    Args:
        raster: Raster carrying its weight sums and effective sample sizes.
        weight_floor: Total response weight below which a cell is unobserved.
        effective_sample_floor: Effective sample size below which a cell is unobserved.

    Returns:
        The raster with masked cells set to NaN and every diagnostic retained.
    """
    if weight_floor <= 0 and effective_sample_floor <= 0:
        return raster
    keep = raster.sum_weight >= weight_floor
    if effective_sample_floor > 0:
        keep &= raster.effective_sample_size >= effective_sample_floor
    return RegriddedRaster(
        aoi_id=raster.aoi_id,
        no2=np.where(keep, raster.no2, np.nan),
        sum_weight=raster.sum_weight,
        sum_weight_squared=raster.sum_weight_squared,
        effective_sample_size=raster.effective_sample_size,
        count=raster.count,
        native_pixel_count=raster.native_pixel_count,
    )


def regrid_aoi_raster(
    pixels: GranulePixels,
    grid: AoiGrid,
    *,
    factor: int = OVERSAMPLE_FACTOR,
    weight_floor: float = TEMPO_CELL_WEIGHT_FLOOR,
    **oversample_options: float,
) -> RegriddedRaster:
    """Build one AOI raster the way the pilot settled on.

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
        return apply_cell_mask(oversample(pixels, grid, **oversample_options), weight_floor)
    fine_grid = AoiGrid(
        aoi_id=grid.aoi_id, x_m=grid.x_m, y_m=grid.y_m, size=grid.size * factor, extent_km=grid.extent_km
    )
    fine = oversample(pixels, fine_grid, **oversample_options)
    return apply_cell_mask(aggregate_fine_raster(fine, grid, factor), weight_floor)


def regrid_aoi_scan(
    granule_paths: list[str],
    grid: AoiGrid,
    *,
    margin_km: float = SELECTION_MARGIN_KM,
    max_cloud_fraction: float = MIN_PIXEL_CLOUD,
    **regrid_options: float,
) -> RegriddedRaster:
    """Build one AOI raster from the granules of a single scan.

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
