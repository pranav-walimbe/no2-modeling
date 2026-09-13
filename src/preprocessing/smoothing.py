"""Wind- and uncertainty-aware smoothing for regridded TEMPO NO2."""

from __future__ import annotations

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve

from config import IMG_RANGE, IMG_SIZE

REGULARIZATION_STRENGTH = 1.1
CROSSWIND_SMOOTHING_RATIO = 0.15
EDGE_SIGNIFICANCE_THRESHOLD = 2.5
DETAIL_REINJECTION_STRENGTH = 0.70
MIN_DATA_PRECISION = 0.25
MAX_DATA_PRECISION = 4.0
CALM_WIND_THRESHOLD_MPS = 0.5
CELL_SIZE_KM = IMG_RANGE / IMG_SIZE
SOURCE_GROUP_DISTANCE_KM = CELL_SIZE_KM
UPWIND_NEAR_KM = 24.0
UPWIND_FAR_KM = 36.0
UPWIND_HALF_WIDTH_KM = 6.0
DOWNWIND_LENGTH_KM = 24.0
DOWNWIND_HALF_WIDTH_KM = 7.5
MIN_BACKGROUND_PIXELS = 12
HUBER_TUNING = 1.345
HUBER_ITERATIONS = 12


class InsufficientUpwindBackgroundError(ValueError):
    """Signal that a smoothed pair cannot support upwind normalization."""


def _validate_normalization_inputs(
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
    if np.isfinite(u_value) and np.isfinite(v_value) and np.hypot(u_value, v_value) >= CALM_WIND_THRESHOLD_MPS:
        return u_value, v_value
    valid = np.isfinite(wind_u) & np.isfinite(wind_v)
    if not np.any(valid):
        return None
    fallback = float(np.median(wind_u[valid])), float(np.median(wind_v[valid]))
    if not np.all(np.isfinite(fallback)) or np.hypot(*fallback) <= 0:
        return None
    return fallback


def _upwind_background_mask(
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


def _upwind_background(no2: np.ndarray, mask: np.ndarray, scan_name: str) -> float:
    # Estimate the scan background after enforcing finite support
    values = no2[mask & np.isfinite(no2)]
    if values.size < MIN_BACKGROUND_PIXELS:
        raise InsufficientUpwindBackgroundError(
            f"{scan_name} upwind background requires {MIN_BACKGROUND_PIXELS} pixels; got {values.size}"
        )
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
) -> tuple[np.ndarray, np.ndarray]:
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
        Current and previous rasters with their own upwind backgrounds removed.
    """
    _validate_normalization_inputs(
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
    current_mask = _upwind_background_mask(current_no2.shape, current_wind_u, current_wind_v, sources)
    previous_mask = _upwind_background_mask(previous_no2.shape, previous_wind_u, previous_wind_v, sources)
    current_background = _upwind_background(paired_current, current_mask, "Current scan")
    previous_background = _upwind_background(paired_previous, previous_mask, "Previous scan")
    return current_no2 - current_background, previous_no2 - previous_background


def _valid_uncertainty(no2: np.ndarray, uncertainty: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    # Fill missing uncertainty without changing NO2 support
    valid = np.isfinite(no2)
    observed = uncertainty[valid & np.isfinite(uncertainty) & (uncertainty > 0)]
    if not observed.size:
        raise ValueError("No positive retrieval uncertainties are available")
    median_uncertainty = float(np.median(observed))
    filled = np.where(np.isfinite(uncertainty) & (uncertainty > 0), uncertainty, median_uncertainty)
    return valid, filled, median_uncertainty


def _directional_edge_weight(
    row_a: np.ndarray,
    column_a: np.ndarray,
    row_b: np.ndarray,
    column_b: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
) -> np.ndarray:
    # Favor along-wind smoothing and use isotropic coupling in calm air
    edge_u = 0.5 * (wind_u[row_a, column_a] + wind_u[row_b, column_b])
    edge_v = 0.5 * (wind_v[row_a, column_a] + wind_v[row_b, column_b])
    speed = np.hypot(edge_u, edge_v)
    east = (column_b - column_a).astype(np.float64)
    north = -(row_b - row_a).astype(np.float64)
    distance = np.hypot(east, north)
    alignment = np.ones_like(speed)
    directional = np.isfinite(speed) & (speed >= CALM_WIND_THRESHOLD_MPS)
    alignment[directional] = np.abs(
        (east[directional] * edge_u[directional] + north[directional] * edge_v[directional])
        / (distance[directional] * speed[directional])
    )
    return CROSSWIND_SMOOTHING_RATIO + (1 - CROSSWIND_SMOOTHING_RATIO) * alignment**2


def _normalized_gaussian(array: np.ndarray, valid: np.ndarray) -> np.ndarray:
    # Smooth without allowing missing pixels to dilute nearby values
    numerator = ndimage.gaussian_filter(np.where(valid, array, 0.0), sigma=1.0, mode="nearest")
    support = ndimage.gaussian_filter(valid.astype(np.float64), sigma=1.0, mode="nearest")
    return np.divide(numerator, support, out=np.zeros_like(numerator), where=support > 1e-6)


def _coherent_detail_confidence(
    no2: np.ndarray,
    uncertainty: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    # Detect uncertainty-significant ridges whose tangent follows local wind
    smoothed = _normalized_gaussian(no2, valid)
    gradient_east = ndimage.sobel(smoothed, axis=1, mode="nearest") / 8
    gradient_north = -ndimage.sobel(smoothed, axis=0, mode="nearest") / 8
    tensor_east = ndimage.gaussian_filter(np.square(gradient_east), sigma=1.0)
    tensor_north = ndimage.gaussian_filter(np.square(gradient_north), sigma=1.0)
    tensor_cross = ndimage.gaussian_filter(gradient_east * gradient_north, sigma=1.0)
    trace = tensor_east + tensor_north
    eigenvalue_gap = np.sqrt(np.square(tensor_east - tensor_north) + 4 * np.square(tensor_cross))
    coherence = np.divide(eigenvalue_gap, trace, out=np.zeros_like(trace), where=trace > 0)
    gradient_magnitude = np.hypot(gradient_east, gradient_north)
    significance = gradient_magnitude / uncertainty
    significance_confidence = significance / (significance + EDGE_SIGNIFICANCE_THRESHOLD)
    wind_speed = np.hypot(wind_u, wind_v)
    wind_normal_alignment = np.divide(
        np.abs(gradient_east * wind_u + gradient_north * wind_v),
        gradient_magnitude * wind_speed,
        out=np.zeros_like(wind_speed),
        where=(gradient_magnitude > 0) & (wind_speed >= CALM_WIND_THRESHOLD_MPS),
    )
    ridge_alignment = 1 - np.square(wind_normal_alignment)
    confidence = coherence * significance_confidence * (0.25 + 0.75 * ridge_alignment)
    return np.where(valid, np.clip(confidence, 0, 1), 0.0)


def smooth_no2(
    no2: np.ndarray,
    uncertainty: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
) -> np.ndarray:
    """Apply the selected uncertainty- and wind-aware smoothing kernel.

    Args:
        no2: North-up NO2 raster.
        uncertainty: Pixel retrieval uncertainty in the same units as NO2.
        wind_u: Eastward wind component at every raster cell.
        wind_v: Northward wind component at every raster cell.

    Returns:
        Smoothed NO2 with the original finite-pixel support.
    """
    arrays = (no2, uncertainty, wind_u, wind_v)
    if no2.ndim != 2 or any(array.shape != no2.shape for array in arrays):
        raise ValueError("NO2, uncertainty, and wind arrays must share one two-dimensional shape")
    valid, sigma, median_sigma = _valid_uncertainty(no2, uncertainty)
    flat_indices = np.full(no2.shape, -1, dtype=np.int64)
    flat_indices[valid] = np.arange(np.count_nonzero(valid))
    observed = no2[valid]
    observed_sigma = sigma[valid]
    precision = np.clip(np.square(median_sigma / observed_sigma), MIN_DATA_PRECISION, MAX_DATA_PRECISION)

    graph_rows: list[np.ndarray] = []
    graph_columns: list[np.ndarray] = []
    graph_weights: list[np.ndarray] = []
    rows, columns = np.indices(no2.shape)
    for row_offset, column_offset in ((0, 1), (1, 0), (1, 1), (1, -1)):
        neighbor_rows = rows + row_offset
        neighbor_columns = columns + column_offset
        inside = (
            (neighbor_rows >= 0)
            & (neighbor_rows < no2.shape[0])
            & (neighbor_columns >= 0)
            & (neighbor_columns < no2.shape[1])
        )
        row_a = rows[inside]
        column_a = columns[inside]
        row_b = neighbor_rows[inside]
        column_b = neighbor_columns[inside]
        paired = valid[row_a, column_a] & valid[row_b, column_b]
        row_a, column_a = row_a[paired], column_a[paired]
        row_b, column_b = row_b[paired], column_b[paired]
        combined_sigma = np.hypot(sigma[row_a, column_a], sigma[row_b, column_b])
        significance = np.abs(no2[row_a, column_a] - no2[row_b, column_b]) / combined_sigma
        edge_preservation = 1 / np.sqrt(1 + (significance / EDGE_SIGNIFICANCE_THRESHOLD) ** 4)
        distance_squared = row_offset**2 + column_offset**2
        graph_rows.append(flat_indices[row_a, column_a])
        graph_columns.append(flat_indices[row_b, column_b])
        graph_weights.append(
            _directional_edge_weight(row_a, column_a, row_b, column_b, wind_u, wind_v)
            * edge_preservation
            / distance_squared
        )

    edge_rows = np.concatenate(graph_rows)
    edge_columns = np.concatenate(graph_columns)
    edge_weights = np.concatenate(graph_weights)
    node_count = observed.size
    degree = np.bincount(
        np.concatenate((edge_rows, edge_columns)),
        weights=np.concatenate((edge_weights, edge_weights)),
        minlength=node_count,
    )
    laplacian = sparse.coo_matrix(
        (
            np.concatenate((degree, -edge_weights, -edge_weights)),
            (
                np.concatenate((np.arange(node_count), edge_rows, edge_columns)),
                np.concatenate((np.arange(node_count), edge_columns, edge_rows)),
            ),
        ),
        shape=(node_count, node_count),
    ).tocsr()
    system = sparse.diags(precision) + REGULARIZATION_STRENGTH * laplacian
    solved = spsolve(system, precision * observed)
    confidence = _coherent_detail_confidence(no2, sigma, wind_u, wind_v, valid)
    solved += DETAIL_REINJECTION_STRENGTH * confidence[valid] * (observed - solved)
    smoothed = np.full(no2.shape, np.nan, dtype=np.float64)
    smoothed[valid] = solved
    return smoothed
