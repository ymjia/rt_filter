from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rt_filter.se3 import interpolate_trajectory, rotation_angle, translation_norm
from rt_filter.trajectory import Trajectory


def trajectory_metrics(traj: Trajectory) -> dict[str, float | int]:
    positions = traj.positions
    metrics: dict[str, float | int] = {
        "sample_count": traj.count,
        "path_length": _path_length(positions),
    }
    if traj.duration is not None:
        metrics["duration"] = traj.duration
        metrics["sample_rate_mean"] = (traj.count - 1) / traj.duration if traj.duration > 0 else 0.0

    span = positions.max(axis=0) - positions.min(axis=0)
    metrics.update(
        {
            "range_x": float(span[0]),
            "range_y": float(span[1]),
            "range_z": float(span[2]),
            "velocity_rms": _derivative_rms(positions, traj.timestamps, order=1),
            "acceleration_rms": _derivative_rms(positions, traj.timestamps, order=2),
            "jerk_rms": _derivative_rms(positions, traj.timestamps, order=3),
            "rotation_step_rms_deg": _rotation_step_rms_deg(traj),
        }
    )
    return metrics


def delta_metrics(
    traj: Trajectory,
    reference: Trajectory,
    *,
    prefix: str = "to_reference",
) -> dict[str, float | int]:
    aligned_traj, aligned_ref = _align(traj, reference)
    trans = translation_norm(aligned_traj.poses, aligned_ref.poses)
    rot_deg = np.rad2deg(rotation_angle(aligned_traj.poses, aligned_ref.poses))
    return {
        f"{prefix}_sample_count": aligned_traj.count,
        f"{prefix}_translation_mean": float(np.mean(trans)),
        f"{prefix}_translation_rmse": float(np.sqrt(np.mean(trans**2))),
        f"{prefix}_translation_max": float(np.max(trans)),
        f"{prefix}_translation_p95": float(np.percentile(trans, 95)),
        f"{prefix}_rotation_mean_deg": float(np.mean(rot_deg)),
        f"{prefix}_rotation_rmse_deg": float(np.sqrt(np.mean(rot_deg**2))),
        f"{prefix}_rotation_max_deg": float(np.max(rot_deg)),
    }


def compare_filter_result(
    raw: Trajectory,
    filtered: Trajectory,
    *,
    reference: Trajectory | None = None,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    metrics.update({f"filtered_{k}": v for k, v in trajectory_metrics(filtered).items()})
    metrics.update(delta_metrics(filtered, raw, prefix="to_raw"))

    raw_acc = float(trajectory_metrics(raw)["acceleration_rms"])
    raw_jerk = float(trajectory_metrics(raw)["jerk_rms"])
    filtered_acc = float(metrics["filtered_acceleration_rms"])
    filtered_jerk = float(metrics["filtered_jerk_rms"])
    metrics["acceleration_rms_ratio"] = _safe_ratio(filtered_acc, raw_acc)
    metrics["jerk_rms_ratio"] = _safe_ratio(filtered_jerk, raw_jerk)

    if reference is not None:
        raw_delta = delta_metrics(raw, reference, prefix="raw_to_reference")
        filtered_delta = delta_metrics(filtered, reference, prefix="to_reference")
        metrics.update(raw_delta)
        metrics.update(filtered_delta)
        metrics["reference_rmse_improvement"] = float(
            raw_delta["raw_to_reference_translation_rmse"]
            - filtered_delta["to_reference_translation_rmse"]
        )
        metrics["reference_max_improvement"] = float(
            raw_delta["raw_to_reference_translation_max"]
            - filtered_delta["to_reference_translation_max"]
        )
    return metrics


def write_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def _align(left: Trajectory, right: Trajectory) -> tuple[Trajectory, Trajectory]:
    if left.timestamps is not None and right.timestamps is not None:
        start = max(left.timestamps[0], right.timestamps[0])
        stop = min(left.timestamps[-1], right.timestamps[-1])
        mask = (left.timestamps >= start) & (left.timestamps <= stop)
        if not np.any(mask):
            raise ValueError("trajectories do not overlap in time")
        left_subset = left.copy_with(poses=left.poses[mask], timestamps=left.timestamps[mask])
        right_poses = interpolate_trajectory(left_subset.timestamps, right.timestamps, right.poses)
        right_subset = right.copy_with(poses=right_poses, timestamps=left_subset.timestamps)
        return left_subset, right_subset

    count = min(left.count, right.count)
    return (
        left.copy_with(poses=left.poses[:count], timestamps=_head(left.timestamps, count)),
        right.copy_with(poses=right.poses[:count], timestamps=_head(right.timestamps, count)),
    )


def _head(values: np.ndarray | None, count: int) -> np.ndarray | None:
    if values is None:
        return None
    return values[:count]


def _path_length(positions: np.ndarray) -> float:
    if len(positions) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))


def _derivative_rms(
    values: np.ndarray,
    timestamps: np.ndarray | None,
    *,
    order: int,
) -> float:
    arr = np.asarray(values, dtype=float)
    if len(arr) <= order:
        return 0.0
    current = arr
    current_t = None if timestamps is None else np.asarray(timestamps, dtype=float)
    for _ in range(order):
        if current_t is None:
            dt = 1.0
            current = np.diff(current, axis=0) / dt
        else:
            dt_arr = np.diff(current_t)
            current = np.diff(current, axis=0) / dt_arr[:, None]
            current_t = current_t[1:]
    return float(np.sqrt(np.mean(np.sum(current**2, axis=1))))


def _rotation_step_rms_deg(traj: Trajectory) -> float:
    if traj.count < 2:
        return 0.0
    rotations = traj.rotations
    step = (rotations[:-1].inv() * rotations[1:]).magnitude()
    return float(np.rad2deg(np.sqrt(np.mean(step**2))))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return float(numerator / denominator)
