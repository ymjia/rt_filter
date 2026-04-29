from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.signal import savgol_filter


EXPECTED_PATH_CACHE_VERSION = 1
DEFAULT_EXPECTED_PATH_MAX_DEVIATION_MM = 10.0
DEFAULT_EXPECTED_PATH_CACHE_DIR = Path(__file__).resolve().parents[2] / "outputs" / "expected_path_cache"


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


def fit_expected_path(
    values: np.ndarray,
    *,
    max_deviation_mm: float = DEFAULT_EXPECTED_PATH_MAX_DEVIATION_MM,
) -> ExpectedPathModel:
    """Fit a sample-aligned expected path from raw XYZ positions.

    The returned model has one expected point and one local tangent per input
    sample. Comparison series should use the same sample index/time basis. This
    intentionally makes lag visible as an along-track error instead of
    projecting it away with a nearest-point distance.
    """

    positions = _positions(values)
    if max_deviation_mm <= 0.0:
        raise ValueError("max_deviation_mm must be positive")
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
        model = ExpectedPathModel(
            kind="line-static",
            expected=expected,
            tangent=tangent,
            details={
                "start_static_samples": start_count,
                "end_static_samples": end_count,
                "length": float(np.linalg.norm(end - start)),
            },
        )
        if _model_is_acceptable(positions, model, max_deviation_mm):
            return model

    pca_line = _pca_line(positions)
    if pca_line is not None:
        start, end, linearity = pca_line
        expected, tangent = _line_expected(positions, start, end)
        model = ExpectedPathModel(
            kind="line-pca",
            expected=expected,
            tangent=tangent,
            details={
                "linearity": linearity,
                "length": float(np.linalg.norm(end - start)),
            },
        )
        if _model_is_acceptable(positions, model, max_deviation_mm):
            return model

    ellipse = _ellipse_expected(positions)
    if ellipse is not None:
        expected, tangent, residual = ellipse
        model = ExpectedPathModel(
            kind="ellipse",
            expected=expected,
            tangent=tangent,
            details={"median_radial_residual": residual},
        )
        if _model_is_acceptable(positions, model, max_deviation_mm):
            return model

    expected, tangent, vertex_count = _polyline_expected(positions)
    model = ExpectedPathModel(
        kind="polyline",
        expected=expected,
        tangent=tangent,
        details={"vertices": vertex_count},
    )
    if _model_is_acceptable(positions, model, max_deviation_mm):
        return model

    smooth_curve = _savgol_curve_expected(positions, max_deviation_mm)
    if smooth_curve is not None:
        expected, tangent, window, polyorder, max_error = smooth_curve
        return ExpectedPathModel(
            kind="savgol-curve",
            expected=expected,
            tangent=tangent,
            details={
                "window": window,
                "polyorder": polyorder,
                "max_deviation_mm": max_error,
            },
        )

    smooth_curve = _savgol_curve_expected(positions, max_deviation_mm, allow_tight=True)
    if smooth_curve is not None:
        expected, tangent, window, polyorder, max_error = smooth_curve
        return ExpectedPathModel(
            kind="savgol-curve",
            expected=expected,
            tangent=tangent,
            details={
                "window": window,
                "polyorder": polyorder,
                "max_deviation_mm": max_error,
                "tight_fallback": 1,
            },
        )
    return model


def fit_expected_path_cached(
    values: np.ndarray,
    *,
    source_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    max_deviation_mm: float = DEFAULT_EXPECTED_PATH_MAX_DEVIATION_MM,
) -> ExpectedPathModel:
    """Fit an expected path and persist successful fits for later GUI runs."""

    positions = _positions(values)
    if source_path is None:
        return fit_expected_path(positions, max_deviation_mm=max_deviation_mm)

    directory = Path(cache_dir) if cache_dir is not None else DEFAULT_EXPECTED_PATH_CACHE_DIR
    cache_path = _expected_path_cache_path(
        positions,
        source_path=Path(source_path),
        cache_dir=directory,
        max_deviation_mm=max_deviation_mm,
    )
    cached = _load_expected_path_cache(cache_path, positions, max_deviation_mm)
    if cached is not None:
        return cached

    model = fit_expected_path(positions, max_deviation_mm=max_deviation_mm)
    if _model_is_acceptable(positions, model, max_deviation_mm):
        _save_expected_path_cache(cache_path, model, source_path=Path(source_path), max_deviation_mm=max_deviation_mm)
    return model


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


def _model_is_acceptable(
    positions: np.ndarray,
    model: ExpectedPathModel,
    max_deviation_mm: float,
) -> bool:
    stats = _deviation_stats(positions, model)
    return stats["max_middle"] <= max_deviation_mm


def _deviation_stats(positions: np.ndarray, model: ExpectedPathModel) -> dict[str, float]:
    deviation = path_deviation(positions, model).norm
    sample_slice = _validation_slice(positions.shape[0], model)
    middle = deviation[sample_slice]
    if middle.size == 0:
        middle = deviation
    return {
        "max": float(np.max(deviation)) if deviation.size else 0.0,
        "p95": float(np.percentile(deviation, 95)) if deviation.size else 0.0,
        "max_middle": float(np.max(middle)) if middle.size else 0.0,
        "p95_middle": float(np.percentile(middle, 95)) if middle.size else 0.0,
    }


def _validation_slice(count: int, model: ExpectedPathModel) -> slice:
    if count <= 2:
        return slice(0, count)
    if model.kind == "line-static":
        start = int(model.details.get("start_static_samples", 0))
        end_count = int(model.details.get("end_static_samples", 0))
        stop = count - end_count
        if stop > start:
            return slice(start, stop)
    trim = min(max(1, count // 20), count // 4)
    return slice(trim, count - trim)


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


def _savgol_curve_expected(
    positions: np.ndarray,
    max_deviation_mm: float,
    *,
    allow_tight: bool = False,
) -> tuple[np.ndarray, np.ndarray, int, int, float] | None:
    count = positions.shape[0]
    if count < 7:
        return None

    polyorders = (4, 5, 3) if count > 6 else (3,)
    max_window = min(101, count if count % 2 == 1 else count - 1)
    min_window = 7 if allow_tight else 21
    if max_window < min_window:
        min_window = 7

    candidate_windows = [
        window
        for window in range(max_window, min_window - 1, -2)
        if window >= 5
    ]
    best: tuple[np.ndarray, np.ndarray, int, int, float] | None = None
    for polyorder in polyorders:
        for window in candidate_windows:
            if window <= polyorder:
                continue
            expected = np.column_stack(
                [
                    savgol_filter(positions[:, axis], window, polyorder, mode="interp")
                    for axis in range(3)
                ]
            )
            deviation = np.linalg.norm(positions - expected, axis=1)
            max_error = float(np.max(deviation)) if deviation.size else 0.0
            tangent = _curve_tangent(expected)
            if best is None or (window > best[2] and max_error <= max_deviation_mm):
                best = (expected, tangent, window, polyorder, max_error)
            if max_error <= max_deviation_mm:
                return expected, tangent, window, polyorder, max_error
            if best is None or max_error < best[4]:
                best = (expected, tangent, window, polyorder, max_error)

    if allow_tight and best is not None and best[4] <= max_deviation_mm:
        return best
    return None


def _curve_tangent(expected: np.ndarray) -> np.ndarray:
    count = expected.shape[0]
    if count <= 1:
        return np.tile(np.array([1.0, 0.0, 0.0], dtype=float), (count, 1))
    tangent = np.empty_like(expected)
    tangent[0] = expected[1] - expected[0]
    tangent[-1] = expected[-1] - expected[-2]
    if count > 2:
        tangent[1:-1] = expected[2:] - expected[:-2]
    return _unit_rows(tangent, fallback=np.array([1.0, 0.0, 0.0]))


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


def _expected_path_cache_path(
    positions: np.ndarray,
    *,
    source_path: Path,
    cache_dir: Path,
    max_deviation_mm: float,
) -> Path:
    source_text = str(source_path.expanduser().resolve())
    payload = np.ascontiguousarray(positions, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(f"v{EXPECTED_PATH_CACHE_VERSION}|{source_text}|{max_deviation_mm:.12g}|".encode("utf-8"))
    digest.update(str(payload.shape).encode("utf-8"))
    digest.update(payload.tobytes())
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_path.stem).strip("_") or "trajectory"
    readable = readable[:80]
    return cache_dir / f"{readable}__{digest.hexdigest()[:16]}.npz"


def _load_expected_path_cache(
    path: Path,
    positions: np.ndarray,
    max_deviation_mm: float,
) -> ExpectedPathModel | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            if int(metadata.get("version", -1)) != EXPECTED_PATH_CACHE_VERSION:
                return None
            if int(metadata.get("sample_count", -1)) != int(positions.shape[0]):
                return None
            if float(metadata.get("max_deviation_mm", -1.0)) != float(max_deviation_mm):
                return None
            expected = np.asarray(data["expected"], dtype=float)
            tangent = np.asarray(data["tangent"], dtype=float)
            if expected.shape != positions.shape or tangent.shape != positions.shape:
                return None
            details = metadata.get("details", {})
            if not isinstance(details, dict):
                details = {}
            model = ExpectedPathModel(
                kind=str(metadata.get("kind", "cached")),
                expected=expected,
                tangent=tangent,
                details=details,
            )
            if not _model_is_acceptable(positions, model, max_deviation_mm):
                return None
            return model
    except Exception:
        return None


def _save_expected_path_cache(
    path: Path,
    model: ExpectedPathModel,
    *,
    source_path: Path,
    max_deviation_mm: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "version": EXPECTED_PATH_CACHE_VERSION,
        "kind": model.kind,
        "details": model.details,
        "sample_count": int(model.expected.shape[0]),
        "source_path": str(source_path),
        "max_deviation_mm": float(max_deviation_mm),
    }
    np.savez_compressed(
        path,
        expected=np.asarray(model.expected, dtype=np.float64),
        tangent=np.asarray(model.tangent, dtype=np.float64),
        metadata=np.array(json.dumps(metadata, ensure_ascii=False)),
    )
