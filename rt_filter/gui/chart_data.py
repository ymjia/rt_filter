from __future__ import annotations

import numpy as np


def neighbor_mean_deviation(values: np.ndarray, window: int) -> np.ndarray:
    """Return XYZ deviation from the centered neighboring-frame mean.

    The neighborhood uses up to ``window`` samples before and after each frame,
    excludes the current frame, and clips naturally at trajectory boundaries.
    """

    positions = np.asarray(values, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"values must have shape (N, 3), got {positions.shape}")
    if window < 1:
        raise ValueError("window must be >= 1")

    count = positions.shape[0]
    deviations = np.zeros_like(positions)
    if count <= 1:
        return deviations

    prefix = np.vstack([np.zeros((1, 3), dtype=float), np.cumsum(positions, axis=0)])
    for index in range(count):
        start = max(0, index - window)
        stop = min(count, index + window + 1)
        neighbor_count = stop - start - 1
        if neighbor_count <= 0:
            continue
        neighbor_sum = prefix[stop] - prefix[start] - positions[index]
        deviations[index] = positions[index] - neighbor_sum / neighbor_count
    return deviations


def complete_neighbor_slice(count: int, window: int) -> slice:
    """Return the sample range where both sides have a full neighbor window."""

    if window < 1:
        raise ValueError("window must be >= 1")
    if count <= window * 2:
        return slice(0, 0)
    return slice(window, count - window)
