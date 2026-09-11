"""Wind- and uncertainty-aware smoothing for regridded TEMPO NO2."""

from __future__ import annotations

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve

REGULARIZATION_STRENGTH = 1.1
CROSSWIND_SMOOTHING_RATIO = 0.15
EDGE_SIGNIFICANCE_THRESHOLD = 2.5
DETAIL_REINJECTION_STRENGTH = 0.70
MIN_DATA_PRECISION = 0.25
MAX_DATA_PRECISION = 4.0
CALM_WIND_THRESHOLD_MPS = 0.5


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
