"""Source-relative upwind normalization for paired smoothed NO2 rasters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from config import IMG_RANGE, IMG_SIZE

CELL_SIZE_KM = IMG_RANGE / IMG_SIZE
SOURCE_GROUP_DISTANCE_KM = CELL_SIZE_KM
UPWIND_NEAR_KM = 24.0
UPWIND_FAR_KM = 36.0
UPWIND_HALF_WIDTH_KM = 6.0
DOWNWIND_LENGTH_KM = 24.0
DOWNWIND_HALF_WIDTH_KM = 7.5
MIN_BACKGROUND_PIXELS = 12
MIN_LOCAL_WIND_SPEED_MPS = 0.5
HUBER_TUNING = 1.345
HUBER_ITERATIONS = 12


@dataclass(frozen=True)
class UpwindNormalizationResult:
    """Normalized paired rasters and the applied scan backgrounds."""

    current_no2: np.ndarray
    previous_no2: np.ndarray
    current_background: float | None
    previous_background: float | None

    @property
    def applied(self) -> bool:
        """Return whether both scan backgrounds were applied."""
        return self.current_background is not None and self.previous_background is not None


def _validate_inputs(
    current_no2: np.ndarray,
    previous_no2: np.ndarray,
    current_wind_u: np.ndarray,
    current_wind_v: np.ndarray,
    previous_wind_u: np.ndarray,
    previous_wind_v: np.ndarray,
    source_east_km: tuple[float, ...],
    source_north_km: tuple[float, ...],
) -> None:
    # Reject inconsistent raster and source geometry
    arrays = (current_no2, previous_no2, current_wind_u, current_wind_v, previous_wind_u, previous_wind_v)
    if current_no2.ndim != 2 or any(array.shape != current_no2.shape for array in arrays):
        raise ValueError("NO2 and wind arrays must share one two-dimensional shape")
    if len(source_east_km) != len(source_north_km) or not source_east_km:
        raise ValueError("Source east and north offsets must describe at least one common location")
    source_coordinates = np.column_stack((source_east_km, source_north_km))
    if not np.all(np.isfinite(source_coordinates)):
        raise ValueError("Source offsets must be finite")


def _group_sources(
    source_east_km: tuple[float, ...],
    source_north_km: tuple[float, ...],
) -> tuple[tuple[float, float], ...]:
    # Merge connected source locations within one raster cell
    coordinates = np.column_stack((source_east_km, source_north_km))
    parents = np.arange(len(coordinates))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = int(parents[index])
        return index

    for left in range(len(coordinates)):
        for right in range(left + 1, len(coordinates)):
            if np.linalg.norm(coordinates[left] - coordinates[right]) <= SOURCE_GROUP_DISTANCE_KM:
                left_root = find(left)
                right_root = find(right)
                parents[right_root] = left_root
    groups: dict[int, list[np.ndarray]] = {}
    for index, coordinate in enumerate(coordinates):
        groups.setdefault(find(index), []).append(coordinate)
    return tuple(tuple(np.mean(group, axis=0)) for group in groups.values())


def _grid_coordinates(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    # Return east and north offsets from the AOI centre
    row_offsets = (np.arange(shape[0], dtype=np.float64) - (shape[0] - 1) / 2) * CELL_SIZE_KM
    column_offsets = (np.arange(shape[1], dtype=np.float64) - (shape[1] - 1) / 2) * CELL_SIZE_KM
    east_km, south_km = np.meshgrid(column_offsets, row_offsets)
    return east_km, -south_km


def _source_wind(
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    source_east_km: float,
    source_north_km: float,
) -> tuple[float, float] | None:
    # Interpolate source wind and fall back to the finite raster median
    row_center = (wind_u.shape[0] - 1) / 2
    column_center = (wind_u.shape[1] - 1) / 2
    coordinates = np.array(
        [
            [row_center - source_north_km / CELL_SIZE_KM],
            [column_center + source_east_km / CELL_SIZE_KM],
        ]
    )
    u_value = float(ndimage.map_coordinates(wind_u, coordinates, order=1, mode="nearest")[0])
    v_value = float(ndimage.map_coordinates(wind_v, coordinates, order=1, mode="nearest")[0])
    if np.isfinite(u_value) and np.isfinite(v_value) and np.hypot(u_value, v_value) >= MIN_LOCAL_WIND_SPEED_MPS:
        return u_value, v_value
    valid = np.isfinite(wind_u) & np.isfinite(wind_v)
    if not np.any(valid):
        return None
    fallback = float(np.median(wind_u[valid])), float(np.median(wind_v[valid]))
    if not np.all(np.isfinite(fallback)) or np.hypot(*fallback) <= 0:
        return None
    return fallback


def _background_mask(
    shape: tuple[int, int],
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    sources: tuple[tuple[float, float], ...],
) -> np.ndarray:
    # Build a shared upwind mask excluding every source's downwind corridor
    east_km, north_km = _grid_coordinates(shape)
    upwind_union = np.zeros(shape, dtype=bool)
    downwind_union = np.zeros(shape, dtype=bool)
    for source_east_km, source_north_km in sources:
        wind = _source_wind(wind_u, wind_v, source_east_km, source_north_km)
        if wind is None:
            continue
        u_value, v_value = wind
        speed = float(np.hypot(u_value, v_value))
        downwind_east = u_value / speed
        downwind_north = v_value / speed
        relative_east = east_km - source_east_km
        relative_north = north_km - source_north_km
        along_km = relative_east * downwind_east + relative_north * downwind_north
        cross_km = -relative_east * downwind_north + relative_north * downwind_east
        upwind_union |= (
            (along_km >= -UPWIND_FAR_KM) & (along_km <= -UPWIND_NEAR_KM) & (np.abs(cross_km) <= UPWIND_HALF_WIDTH_KM)
        )
        downwind_union |= (
            (along_km > 0) & (along_km <= DOWNWIND_LENGTH_KM) & (np.abs(cross_km) <= DOWNWIND_HALF_WIDTH_KM)
        )
    return upwind_union & ~downwind_union


def _mad(values: np.ndarray) -> float:
    # Return the Gaussian-consistent median absolute deviation
    median = float(np.median(values))
    return 1.4826 * float(np.median(np.abs(values - median)))


def _huber_location(values: np.ndarray) -> float:
    # Fit a robust scalar location with fixed-scale IRLS
    location = float(np.median(values))
    scale = _mad(values)
    if not np.isfinite(scale) or scale <= np.finfo(np.float64).eps:
        return location
    for _ in range(HUBER_ITERATIONS):
        standardized = np.abs(values - location) / scale
        weights = np.ones_like(values)
        outside = standardized > HUBER_TUNING
        weights[outside] = HUBER_TUNING / standardized[outside]
        updated = float(np.average(values, weights=weights))
        tolerance = np.finfo(np.float64).eps * max(abs(location), 1.0)
        if abs(updated - location) <= tolerance:
            break
        location = updated
    return location


def _background(no2: np.ndarray, mask: np.ndarray) -> float | None:
    # Estimate the scan background after enforcing finite support
    values = no2[mask & np.isfinite(no2)]
    if values.size < MIN_BACKGROUND_PIXELS:
        return None
    return _huber_location(values)


def normalize_smoothed_pair(
    current_no2: np.ndarray,
    previous_no2: np.ndarray,
    current_wind_u: np.ndarray,
    current_wind_v: np.ndarray,
    previous_wind_u: np.ndarray,
    previous_wind_v: np.ndarray,
    source_east_km: tuple[float, ...] = (0.0,),
    source_north_km: tuple[float, ...] = (0.0,),
) -> UpwindNormalizationResult:
    """Subtract scan-specific upwind backgrounds from a smoothed pair.

    Args:
        current_no2: Smoothed current NO2 raster.
        previous_no2: Smoothed previous NO2 raster.
        current_wind_u: Current eastward wind raster.
        current_wind_v: Current northward wind raster.
        previous_wind_u: Previous eastward wind raster.
        previous_wind_v: Previous northward wind raster.
        source_east_km: Facility offsets east of the AOI centre.
        source_north_km: Facility offsets north of the AOI centre.

    Returns:
        Paired normalized rasters and background levels, unchanged when either scan lacks sufficient support.
    """
    _validate_inputs(
        current_no2,
        previous_no2,
        current_wind_u,
        current_wind_v,
        previous_wind_u,
        previous_wind_v,
        source_east_km,
        source_north_km,
    )
    paired = np.isfinite(current_no2) & np.isfinite(previous_no2)
    paired_current = np.where(paired, current_no2, np.nan)
    paired_previous = np.where(paired, previous_no2, np.nan)
    sources = _group_sources(source_east_km, source_north_km)
    current_mask = _background_mask(current_no2.shape, current_wind_u, current_wind_v, sources)
    previous_mask = _background_mask(previous_no2.shape, previous_wind_u, previous_wind_v, sources)
    current_background = _background(paired_current, current_mask)
    previous_background = _background(paired_previous, previous_mask)
    if current_background is None or previous_background is None:
        return UpwindNormalizationResult(
            current_no2=current_no2.copy(),
            previous_no2=previous_no2.copy(),
            current_background=None,
            previous_background=None,
        )
    return UpwindNormalizationResult(
        current_no2=current_no2 - current_background,
        previous_no2=previous_no2 - previous_background,
        current_background=current_background,
        previous_background=previous_background,
    )
