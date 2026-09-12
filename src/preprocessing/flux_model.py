"""Aggregate integrated-mass NOx flux estimation from TEMPO NO2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

AVOGADRO_MOLECULES_PER_MOL = 6.02214076e23
NO2_KG_PER_MOL = 0.0460055
MOLECULES_CM2_TO_KG_M2 = 10_000 * NO2_KG_PER_MOL / AVOGADRO_MOLECULES_PER_MOL
KG_S_TO_LB_HOUR = 7_936.6414387
MIN_WIND_SPEED_MPS = 1.0
NOX_NO2_INITIAL_EXCESS = 1.6
NOX_NO2_CONVERSION_RATE_SECONDS = 1_638.0
NOX_NO2_EQUILIBRIUM_RATIO = 1.31
NOX_LIFETIME_SECONDS = 1.5 * 60 * 60
PLUME_HALF_WIDTH_KM = 4.5
PLUME_LENGTH_KM = 12.0
UPWIND_MIN_DISTANCE_KM = 7.5
UPWIND_MAX_DISTANCE_KM = 30.0
UPWIND_HALF_WIDTH_KM = 25.0
# Fitted on 700 shard samples with five-fold cross-validation
FLUX_CALIBRATION_FACTOR = 1.4158915687682765
MIN_BACKGROUND_PIXELS = 8
MIN_UPWIND_PIXELS = 16
MIN_PLUME_PIXELS = 2
CONFIDENCE_SNR_SCALE = 3.0


@dataclass(frozen=True)
class FluxEstimate:
    """Aggregate NOx flux and a unitless reliability score."""

    flux_nox: float
    confidence: float


def _grid_coordinates(shape: tuple[int, int], cell_size_m: float) -> tuple[np.ndarray, np.ndarray]:
    # Return east and north offsets from the AOI centre
    row_offsets = (np.arange(shape[0], dtype=np.float64) - (shape[0] - 1) / 2) * cell_size_m
    column_offsets = (np.arange(shape[1], dtype=np.float64) - (shape[1] - 1) / 2) * cell_size_m
    east_m, south_m = np.meshgrid(column_offsets, row_offsets)
    return east_m, -south_m


def _robust_background(
    no2: np.ndarray,
    uncertainty: np.ndarray,
    cross_km: np.ndarray,
    background: np.ndarray,
) -> tuple[np.ndarray, float]:
    # Fit a crosswind linear background with uncertainty weights and MAD clipping
    x = cross_km[background]
    y = no2[background]
    sigma = uncertainty[background]
    finite_sigma = sigma[np.isfinite(sigma) & (sigma > 0)]
    fallback_sigma = float(np.median(finite_sigma)) if finite_sigma.size else 1.0
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, fallback_sigma)
    weights = np.square(fallback_sigma / sigma)
    design = np.column_stack((np.ones(x.size), x))
    keep = np.ones(x.size, dtype=bool)
    coefficients = np.array([float(np.median(y)), 0.0])
    for _ in range(3):
        weighted_design = design[keep] * np.sqrt(weights[keep, None])
        weighted_values = y[keep] * np.sqrt(weights[keep])
        coefficients = np.linalg.lstsq(weighted_design, weighted_values, rcond=None)[0]
        residuals = y - design @ coefficients
        residual_median = float(np.median(residuals[keep]))
        mad = float(np.median(np.abs(residuals[keep] - residual_median)))
        if mad <= np.finfo(np.float64).eps:
            break
        keep = np.abs(residuals - residual_median) <= 3 * 1.4826 * mad
        if np.count_nonzero(keep) < MIN_BACKGROUND_PIXELS:
            break
    return coefficients[0] + coefficients[1] * cross_km, float(np.mean(keep))


def _conversion_factor(age_seconds: np.ndarray) -> np.ndarray:
    # Convert observed NO2 to emitted NOx and reverse first-order NOx loss
    nox_to_no2 = (
        NOX_NO2_INITIAL_EXCESS * np.exp(-age_seconds / NOX_NO2_CONVERSION_RATE_SECONDS) + NOX_NO2_EQUILIBRIUM_RATIO
    )
    return nox_to_no2 * np.exp(age_seconds / NOX_LIFETIME_SECONDS)


def _source_relative_coordinates(
    along_km: np.ndarray,
    cross_km: np.ndarray,
    source_along_km: np.ndarray,
    source_cross_km: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Project every raster cell relative to every source
    relative_along = along_km[..., None] - source_along_km[None, None, :]
    relative_cross = cross_km[..., None] - source_cross_km[None, None, :]
    return relative_along, relative_cross


def _upwind_background(
    no2: np.ndarray,
    valid: np.ndarray,
    relative_along: np.ndarray,
    relative_cross: np.ndarray,
) -> tuple[float, float] | None:
    # Estimate a common background from source-relative upwind corridors
    corridor = np.any(
        (relative_along <= -UPWIND_MIN_DISTANCE_KM)
        & (relative_along >= -UPWIND_MAX_DISTANCE_KM)
        & (np.abs(relative_cross) <= UPWIND_HALF_WIDTH_KM),
        axis=2,
    )
    background = corridor & valid
    if np.count_nonzero(background) < MIN_UPWIND_PIXELS:
        return None
    coverage = float(np.mean(valid[corridor]))
    return float(np.median(no2[background])), coverage


def _plume_mask_and_ages(
    valid: np.ndarray,
    relative_along: np.ndarray,
    relative_cross: np.ndarray,
    transport_speed_mps: float,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    # Select the compact downwind union and calculate nearest-source ages
    eligible = (
        (relative_along > 0) & (relative_along <= PLUME_LENGTH_KM) & (np.abs(relative_cross) <= PLUME_HALF_WIDTH_KM)
    )
    corridor = np.any(eligible, axis=2)
    plume = corridor & valid
    if np.count_nonzero(plume) < MIN_PLUME_PIXELS:
        return None
    nearest_distance_km = np.min(np.where(eligible, relative_along, np.inf), axis=2)
    ages_seconds = nearest_distance_km[plume] * 1_000 / transport_speed_mps
    coverage = float(np.mean(valid[corridor]))
    return plume, ages_seconds, coverage


def _flux_confidence(
    enhancement: np.ndarray,
    uncertainty: np.ndarray,
    observed_speed_mps: float,
    plume_coverage: float,
    background_coverage: float,
) -> float:
    # Combine plume signal, wind strength and spatial coverage
    finite_uncertainty = uncertainty[np.isfinite(uncertainty) & (uncertainty > 0)]
    noise = float(np.sqrt(np.square(finite_uncertainty).sum())) if finite_uncertainty.size else float("inf")
    signal = float(np.sum(enhancement))
    signal_to_noise = signal / noise if noise > 0 else 0.0
    signal_confidence = signal_to_noise / (signal_to_noise + CONFIDENCE_SNR_SCALE)
    wind_confidence = min(observed_speed_mps / 3.0, 1.0)
    confidence = signal_confidence * wind_confidence * plume_coverage * background_coverage
    return float(np.clip(confidence, 0.0, 1.0))


def estimate_aggregate_flux(
    no2: np.ndarray,
    uncertainty: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    source_east_km: tuple[float, ...] = (0.0,),
    source_north_km: tuple[float, ...] = (0.0,),
    cell_size_m: float = 1_500.0,
) -> FluxEstimate:
    """Estimate aggregate AOI NOx with integrated downwind plume mass.

    Args:
        no2: Smoothed tropospheric NO2 column in molecules per square centimetre.
        uncertainty: Retrieval uncertainty in the same units as NO2.
        wind_u: Eastward wind in metres per second.
        wind_v: Northward wind in metres per second.
        source_east_km: Facility offsets east of the AOI centre.
        source_north_km: Facility offsets north of the AOI centre.
        cell_size_m: Square raster-cell width in metres.

    Returns:
        Aggregate NOx in pounds per hour and a confidence from zero to one.
    """
    arrays = (no2, uncertainty, wind_u, wind_v)
    if no2.ndim != 2 or any(array.shape != no2.shape for array in arrays):
        raise ValueError("NO2, uncertainty, and wind arrays must share one two-dimensional shape")
    if len(source_east_km) != len(source_north_km) or not source_east_km:
        raise ValueError("Source east and north offsets must describe at least one common location")

    valid = np.isfinite(no2) & np.isfinite(wind_u) & np.isfinite(wind_v)
    if not np.any(valid):
        return FluxEstimate(0.0, 0.0)
    reference_u = float(np.median(wind_u[valid]))
    reference_v = float(np.median(wind_v[valid]))
    observed_speed = float(np.hypot(reference_u, reference_v))
    if not np.isfinite(observed_speed):
        return FluxEstimate(0.0, 0.0)
    transport_speed = max(observed_speed, MIN_WIND_SPEED_MPS)
    if observed_speed > 0:
        downwind_east = reference_u / observed_speed
        downwind_north = reference_v / observed_speed
    else:
        downwind_east, downwind_north = 1.0, 0.0

    east_m, north_m = _grid_coordinates(no2.shape, cell_size_m)
    along_km = (east_m * downwind_east + north_m * downwind_north) / 1_000
    cross_km = (-east_m * downwind_north + north_m * downwind_east) / 1_000
    source_east = np.asarray(source_east_km, dtype=np.float64)
    source_north = np.asarray(source_north_km, dtype=np.float64)
    source_along = source_east * downwind_east + source_north * downwind_north
    source_cross = -source_east * downwind_north + source_north * downwind_east
    finite_sources = np.isfinite(source_along) & np.isfinite(source_cross)
    source_along = source_along[finite_sources]
    source_cross = source_cross[finite_sources]
    if not source_along.size:
        return FluxEstimate(0.0, 0.0)

    relative_along, relative_cross = _source_relative_coordinates(
        along_km,
        cross_km,
        source_along,
        source_cross,
    )
    background_result = _upwind_background(no2, valid, relative_along, relative_cross)
    plume_result = _plume_mask_and_ages(valid, relative_along, relative_cross, transport_speed)
    if background_result is None or plume_result is None:
        return FluxEstimate(0.0, 0.0)

    background_level, background_coverage = background_result
    plume, ages_seconds, plume_coverage = plume_result
    enhancement = np.maximum(no2[plume] - background_level, 0.0)
    if not np.any(enhancement > 0):
        return FluxEstimate(0.0, 0.0)
    projected_speed = np.maximum(
        wind_u * downwind_east + wind_v * downwind_north,
        MIN_WIND_SPEED_MPS,
    )
    corrected_mass = enhancement * MOLECULES_CM2_TO_KG_M2 * _conversion_factor(ages_seconds)
    flux_kg_s = float(np.sum(corrected_mass * projected_speed[plume] * cell_size_m**2) / (PLUME_LENGTH_KM * 1_000))
    flux_nox = max(flux_kg_s * KG_S_TO_LB_HOUR * FLUX_CALIBRATION_FACTOR, 0.0)
    confidence = _flux_confidence(
        enhancement,
        uncertainty[plume],
        observed_speed,
        plume_coverage,
        background_coverage,
    )
    return FluxEstimate(flux_nox, float(np.clip(confidence, 0.0, 1.0)))
