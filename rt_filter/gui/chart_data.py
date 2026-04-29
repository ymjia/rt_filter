from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExpectedPathModel:
    """Sample-aligned expected path fitted from a raw trajectory."""

    kind: str
    expected: np.ndarray
    tangent: np.ndarray
    details: dict[str, float | int | str]


@dataclass(frozen=True)
class PathDeviation:
    """Deviation from a fitted expected path."""

    along: np.ndarray
    cross: np.ndarray
    norm: np.ndarray


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


def fit_expected_path(values: np.ndarray) -> ExpectedPathModel:
    """Fit a sample-aligned expected path from raw XYZ positions.

    The returned model has one expected point and one local tangent per input
    sample. Comparison series should use the same sample index/time basis. This
    intentionally makes lag visible as an along-track error instead of
    projecting it away with a nearest-point distance.
    """

    positions = _positions(values)
    count = positions.shape[0]
    if count == 1:
        return ExpectedPathModel(
            kind="static",
            expected=positions.copy(),
            tangent=np.tile(np.array([[1.0, 0.0, 0.0]]), (1, 1)),
            details={"samples": 1},
        )

    static_line = _static_endpoint_line(positions)
    if static_line is not None:
        start, end, start_count, end_count = static_line
        expected, tangent = _line_expected(positions, start, end)
        return ExpectedPathModel(
            kind="line-static",
            expected=expected,
            tangent=tangent,
            details={
                "start_static_samples": start_count,
                "end_static_samples": end_count,
                "length": float(np.linalg.norm(end - start)),
            },
        )

    pca_line = _pca_line(positions)
    if pca_line is not None:
        start, end, linearity = pca_line
        expected, tangent = _line_expected(positions, start, end)
        return ExpectedPathModel(
            kind="line-pca",
            expected=expected,
            tangent=tangent,
            details={
                "linearity": linearity,
                "length": float(np.linalg.norm(end - start)),
            },
        )

    ellipse = _ellipse_expected(positions)
    if ellipse is not None:
        expected, tangent, residual = ellipse
        return ExpectedPathModel(
            kind="ellipse",
            expected=expected,
            tangent=tangent,
            details={"median_radial_residual": residual},
        )

    expected, tangent, vertex_count = _polyline_expected(positions)
    return ExpectedPathModel(
        kind="polyline",
        expected=expected,
        tangent=tangent,
        details={"vertices": vertex_count},
    )


def path_deviation(values: np.ndarray, model: ExpectedPathModel) -> PathDeviation:
    """Return along-track, cross-track, and total deviations from a model."""

    positions = _positions(values)
    count = min(positions.shape[0], model.expected.shape[0], model.tangent.shape[0])
    if count == 0:
        return PathDeviation(
            along=np.zeros(0, dtype=float),
            cross=np.zeros(0, dtype=float),
            norm=np.zeros(0, dtype=float),
        )

    delta = positions[:count] - model.expected[:count]
    tangent = _unit_rows(model.tangent[:count], fallback=np.array([1.0, 0.0, 0.0]))
    along = np.sum(delta * tangent, axis=1)
    cross_vec = delta - along[:, None] * tangent
    cross = np.linalg.norm(cross_vec, axis=1)
    norm = np.linalg.norm(delta, axis=1)
    return PathDeviation(along=along, cross=cross, norm=norm)


def _positions(values: np.ndarray) -> np.ndarray:
    positions = np.asarray(values, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"values must have shape (N, 3), got {positions.shape}")
    if positions.shape[0] == 0:
        raise ValueError("values must contain at least one sample")
    return positions


def _static_endpoint_line(
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    count = positions.shape[0]
    if count < 8:
        return None

    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    moving_scale = float(np.percentile(steps, 90))
    if moving_scale <= 0.0:
        return None
    threshold = max(float(np.percentile(steps, 20)) * 2.0, moving_scale * 0.08)

    start_steps = 0
    for step in steps:
        if step > threshold:
            break
        start_steps += 1

    end_steps = 0
    for step in steps[::-1]:
        if step > threshold:
            break
        end_steps += 1

    start_count = start_steps + 1
    end_count = end_steps + 1
    min_static = max(3, min(8, int(np.ceil(count * 0.03))))
    if start_count < min_static or end_count < min_static:
        return None
    if start_count + end_count >= count:
        return None

    start = np.mean(positions[:start_count], axis=0)
    end = np.mean(positions[count - end_count :], axis=0)
    span = float(np.linalg.norm(np.ptp(positions, axis=0)))
    length = float(np.linalg.norm(end - start))
    if length <= max(span * 0.15, moving_scale * 5.0, 1e-12):
        return None
    return start, end, start_count, end_count


def _pca_line(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    centered = positions - np.mean(positions, axis=0)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    total = float(np.sum(singular_values**2))
    if total <= 0.0:
        return None
    linearity = float(singular_values[0] ** 2 / total)
    if linearity < 0.97:
        return None

    direction = vt[0]
    projection = centered @ direction
    if projection[-1] < projection[0]:
        direction = -direction
        projection = -projection
    low = float(np.percentile(projection, 1))
    high = float(np.percentile(projection, 99))
    if high <= low:
        return None
    origin = np.mean(positions, axis=0)
    return origin + low * direction, origin + high * direction, linearity


def _line_expected(
    positions: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    vector = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    length = float(np.linalg.norm(vector))
    if length <= 0.0:
        direction = np.array([1.0, 0.0, 0.0], dtype=float)
        expected = np.tile(np.asarray(start, dtype=float), (positions.shape[0], 1))
    else:
        direction = vector / length
        progress = (positions - start) @ direction
        progress = np.clip(progress, 0.0, length)
        expected = start + progress[:, None] * direction
    tangent = np.tile(direction, (positions.shape[0], 1))
    return expected, tangent


def _ellipse_expected(
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    count = positions.shape[0]
    if count < 12:
        return None

    center = np.mean(positions, axis=0)
    centered = positions - center
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    total = float(np.sum(singular_values**2))
    if total <= 0.0:
        return None
    planarity = float(singular_values[2] ** 2 / total) if singular_values.shape[0] >= 3 else 0.0
    linearity = float(singular_values[0] ** 2 / total)
    if planarity > 0.08 or linearity > 0.93:
        return None

    axis_u = vt[0]
    axis_v = vt[1]
    coords_u = centered @ axis_u
    coords_v = centered @ axis_v
    radius_u = _robust_radius(coords_u)
    radius_v = _robust_radius(coords_v)
    if min(radius_u, radius_v) <= 0.0:
        return None

    theta = np.unwrap(np.arctan2(coords_v / radius_v, coords_u / radius_u))
    span = float(np.max(theta) - np.min(theta))
    path_length = _path_length(positions)
    closedness = float(np.linalg.norm(positions[0] - positions[-1]) / path_length) if path_length else 1.0
    if span < np.pi * 1.2 and closedness > 0.25:
        return None

    radial = np.sqrt((coords_u / radius_u) ** 2 + (coords_v / radius_v) ** 2)
    residual = float(np.median(np.abs(radial - 1.0)))
    if residual > 0.25:
        return None

    expected = (
        center
        + radius_u * np.cos(theta)[:, None] * axis_u
        + radius_v * np.sin(theta)[:, None] * axis_v
    )
    derivative = (
        -radius_u * np.sin(theta)[:, None] * axis_u
        + radius_v * np.cos(theta)[:, None] * axis_v
    )
    if theta[-1] < theta[0]:
        derivative = -derivative
    tangent = _unit_rows(derivative, fallback=axis_u)
    return expected, tangent, residual


def _polyline_expected(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    count = positions.shape[0]
    if count == 1:
        return positions.copy(), np.array([[1.0, 0.0, 0.0]], dtype=float), 1

    span = float(np.linalg.norm(np.ptp(positions, axis=0)))
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    nonzero_steps = steps[steps > 0.0]
    median_step = float(np.median(nonzero_steps)) if nonzero_steps.size else 0.0
    tolerance = max(span * 0.015, median_step * 2.0, 1e-12)
    indices = _rdp_indices(positions, tolerance)
    if len(indices) < 2:
        indices = [0, count - 1]
    polyline = positions[indices]
    expected, tangent = _project_to_polyline(positions, polyline)
    return expected, tangent, len(indices)


def _rdp_indices(points: np.ndarray, tolerance: float) -> list[int]:
    count = points.shape[0]
    if count <= 2:
        return list(range(count))

    keep = np.zeros(count, dtype=bool)
    keep[0] = True
    keep[-1] = True
    stack: list[tuple[int, int]] = [(0, count - 1)]
    while stack:
        start, stop = stack.pop()
        if stop <= start + 1:
            continue
        segment = points[stop] - points[start]
        segment_length = float(np.linalg.norm(segment))
        if segment_length <= 0.0:
            distances = np.linalg.norm(points[start + 1 : stop] - points[start], axis=1)
        else:
            unit = segment / segment_length
            rel = points[start + 1 : stop] - points[start]
            projected = np.sum(rel * unit, axis=1)[:, None] * unit
            distances = np.linalg.norm(rel - projected, axis=1)
        if distances.size == 0:
            continue
        local_index = int(np.argmax(distances))
        if float(distances[local_index]) <= tolerance:
            continue
        split = start + 1 + local_index
        keep[split] = True
        stack.append((start, split))
        stack.append((split, stop))

    return np.nonzero(keep)[0].astype(int).tolist()


def _project_to_polyline(
    positions: np.ndarray,
    polyline: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    segments_start = polyline[:-1]
    segments_end = polyline[1:]
    vectors = segments_end - segments_start
    lengths = np.linalg.norm(vectors, axis=1)
    valid = lengths > 0.0
    if not np.any(valid):
        expected = np.tile(polyline[0], (positions.shape[0], 1))
        tangent = np.tile(np.array([1.0, 0.0, 0.0], dtype=float), (positions.shape[0], 1))
        return expected, tangent

    segments_start = segments_start[valid]
    vectors = vectors[valid]
    lengths = lengths[valid]
    directions = vectors / lengths[:, None]

    expected = np.empty_like(positions)
    tangent = np.empty_like(positions)
    for index, point in enumerate(positions):
        rel = point - segments_start
        progress = np.sum(rel * directions, axis=1)
        progress = np.clip(progress, 0.0, lengths)
        candidates = segments_start + progress[:, None] * directions
        distances = np.sum((candidates - point) ** 2, axis=1)
        best = int(np.argmin(distances))
        expected[index] = candidates[best]
        tangent[index] = directions[best]
    return expected, tangent


def _robust_radius(values: np.ndarray) -> float:
    radius = 0.5 * (float(np.percentile(values, 95)) - float(np.percentile(values, 5)))
    if radius <= 0.0:
        radius = float(np.max(np.abs(values)))
    return radius


def _unit_rows(values: np.ndarray, *, fallback: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    norm = np.linalg.norm(arr, axis=1)
    output = np.empty_like(arr)
    valid = norm > 0.0
    output[valid] = arr[valid] / norm[valid, None]
    fallback_unit = np.asarray(fallback, dtype=float)
    fallback_norm = float(np.linalg.norm(fallback_unit))
    if fallback_norm <= 0.0:
        fallback_unit = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        fallback_unit = fallback_unit / fallback_norm
    output[~valid] = fallback_unit
    return output


def _path_length(positions: np.ndarray) -> float:
    if positions.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
