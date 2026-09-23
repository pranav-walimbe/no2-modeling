"""Deterministic synthetic masks for NO2 reconstruction pretraining."""

from __future__ import annotations

import hashlib

import numpy as np

MIN_MASK_FRACTION = 0.01
MAX_MASK_FRACTION = 0.15
FRONTIER_SELECTION_PROBABILITY = 0.94
EDGE_DECAY_PIXELS = 2.5
EDGE_WEIGHT_FLOOR = 0.06


def _record_rng(identifier: str, seed: int) -> np.random.Generator:
    # Derive a platform-stable random stream from the record and run seed
    digest = hashlib.sha256(f"{seed}:{identifier}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], byteorder="little"))


def _edge_weights(shape: tuple[int, int]) -> np.ndarray:
    # Assign the highest sampling weight at the raster boundary
    rows, columns = np.indices(shape)
    edge_distance = np.minimum.reduce((rows, columns, shape[0] - 1 - rows, shape[1] - 1 - columns))
    return np.exp(-edge_distance / EDGE_DECAY_PIXELS).ravel() + EDGE_WEIGHT_FLOOR


def _weighted_choice(
    candidates: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> int:
    # Draw one candidate after normalizing its positive weights
    candidate_weights = weights[candidates]
    probabilities = candidate_weights / candidate_weights.sum()
    return int(candidates[rng.choice(len(candidates), p=probabilities)])


def _neighbors(pixel: int, shape: tuple[int, int]) -> tuple[int, ...]:
    # Return valid eight-connected neighbors in stable order
    row, column = divmod(pixel, shape[1])
    return tuple(
        neighbor_row * shape[1] + neighbor_column
        for neighbor_row in range(max(0, row - 1), min(shape[0], row + 2))
        for neighbor_column in range(max(0, column - 1), min(shape[1], column + 2))
        if neighbor_row != row or neighbor_column != column
    )


def synthetic_visible_mask(
    shape: tuple[int, int],
    identifier: str,
    seed: int,
) -> np.ndarray:
    """Create one deterministic edge-biased clustered visibility mask.

    Args:
        shape: Raster height and width.
        identifier: Stable record identifier used for hash seeding.
        seed: Run seed mixed into the record hash.

    Returns:
        Boolean mask where true marks a visible pixel.
    """
    rng = _record_rng(identifier, seed)
    missing_fraction = rng.uniform(MIN_MASK_FRACTION, MAX_MASK_FRACTION)
    target_count = max(1, round(missing_fraction * shape[0] * shape[1]))
    edge_weights = _edge_weights(shape)
    missing = np.zeros(shape[0] * shape[1], dtype=bool)
    frontier: dict[int, int] = {}
    missing_count = 0

    while missing_count < target_count:
        if frontier and rng.random() < FRONTIER_SELECTION_PROBABILITY:
            candidates = np.fromiter(frontier, dtype=np.int64)
            neighbor_counts = np.fromiter(frontier.values(), dtype=np.float64)
            weights = edge_weights.copy()
            weights[candidates] *= np.square(1.0 + neighbor_counts)
        else:
            candidates = np.flatnonzero(~missing)
            weights = edge_weights

        selected = _weighted_choice(candidates, weights, rng)
        missing[selected] = True
        missing_count += 1
        frontier.pop(selected, None)
        for neighbor in _neighbors(selected, shape):
            if not missing[neighbor]:
                frontier[neighbor] = frontier.get(neighbor, 0) + 1

    return (~missing).reshape(shape)
