from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rt_filter.gui.chart_data import complete_neighbor_slice, neighbor_mean_deviation
from rt_filter.io import read_trajectory, write_trajectory
from rt_filter.se3 import make_poses
from rt_filter.trajectory import Trajectory


SOURCE_ROOT = Path("ref_data/sn")
INPUT_ROOT = Path("input/sn")
SOURCE_FOLDER_NAME = "\u672c\u5730\u5b9e\u9a8c\u5ba4\u6570\u636e"
SOURCE_FILE_NAME = "track_data_03.txt"
CASE12_ID = "case_12_local_lab_trajectory"
CASE12_TRAJECTORY = f"{CASE12_ID}_01_track_data_03"
SEGMENT_CASES = [
    {
        "case_id": "case_13_local_lab_trajectory_segment_01",
        "case_label": "local_lab_trajectory_segment",
        "segment_index": 1,
        "variant_id": "segment_01",
    },
    {
        "case_id": "case_14_local_lab_trajectory_segment_02",
        "case_label": "local_lab_trajectory_segment",
        "segment_index": 2,
        "variant_id": "segment_02",
    },
    {
        "case_id": "case_15_local_lab_trajectory_segment_03",
        "case_label": "local_lab_trajectory_segment",
        "segment_index": 3,
        "variant_id": "segment_03",
    },
]
LOCAL_LAB_COLUMNS = ["x", "y", "z", "xr", "yr", "zr", "timestamp", "rate"]
TIMESTAMP_TICKS_PER_SECOND = 10000.0
MAX_CONTINUOUS_STEP_S = 0.05
DRAG_WARMUP_FRAMES = 20


def main() -> None:
    rows = [ensure_sn_case12(), *ensure_sn_cases13_to15()]
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def ensure_sn_case12() -> dict[str, Any]:
    source_path = _find_source_path()
    trajectory, raw_row_count, frame, metadata = _read_local_lab_track(
        source_path,
        name=CASE12_TRAJECTORY,
        case_id=CASE12_ID,
        trajectory_name=CASE12_TRAJECTORY,
    )

    case_dir = INPUT_ROOT / CASE12_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    input_csv = case_dir / f"{CASE12_TRAJECTORY}.csv"
    metadata_json = case_dir / f"{CASE12_TRAJECTORY}_metadata.json"
    write_trajectory(trajectory, input_csv)
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    row = {
        "trajectory": CASE12_TRAJECTORY,
        "case_id": CASE12_ID,
        "case_label": "local_lab_trajectory",
        "source_folder": SOURCE_FOLDER_NAME,
        "source_file": source_path.name,
        "source_path": str(source_path),
        "input_csv": str(input_csv),
        "source_workbook": "",
        "base": "local_lab",
        "target": "robot_arm_motion",
        "calibration": "unknown_calibration",
        "exposure": "default_exposure",
        "added_points": False,
        "variant_id": "track_data_03",
        "tracker_smoothing": "unknown",
        "variant_note": "local lab motion trajectory",
        "sample_count": trajectory.count,
        "raw_row_count": raw_row_count,
        "position_unit": "mm",
        "rotation_source": "xr,yr,zr",
        "rotation_assumption": "XYZ Euler angles in degrees",
        "timestamp_source": "normalized from source timestamp ticks / 10000",
        "sample_rate_hz": _median_sample_rate(trajectory.timestamps),
        "reference_csv": "",
        "metadata_json": str(metadata_json),
    }
    _upsert_manifest_row(INPUT_ROOT / "manifest.csv", row, key="trajectory")
    return row


def ensure_sn_cases13_to15() -> list[dict[str, Any]]:
    source_path = _find_source_path()
    _, raw_row_count, frame, _ = _read_local_lab_track(
        source_path,
        name=CASE12_TRAJECTORY,
        case_id=CASE12_ID,
        trajectory_name=CASE12_TRAJECTORY,
    )
    segment_frames = _split_frame_by_gaps(frame)
    if len(segment_frames) != len(SEGMENT_CASES):
        raise ValueError(
            f"expected {len(SEGMENT_CASES)} local lab segments, got {len(segment_frames)}"
        )

    rows: list[dict[str, Any]] = []
    for segment, segment_frame in zip(SEGMENT_CASES, segment_frames, strict=True):
        case_id = str(segment["case_id"])
        segment_index = int(segment["segment_index"])
        trajectory_name = f"{case_id}_01_track_data_03"
        trajectory = _trajectory_from_frame(
            segment_frame,
            name=trajectory_name,
            source_path=source_path,
            timestamp_origin="segment",
        )
        metadata = _metadata(
            source_path,
            len(segment_frame),
            segment_frame,
            trajectory,
            case_id=case_id,
            trajectory_name=trajectory_name,
        )
        metadata.update(
            {
                "source_full_raw_row_count": raw_row_count,
                "segment_index": segment_index,
                "source_start_row": int(segment_frame.index[0]),
                "source_stop_row_exclusive": int(segment_frame.index[-1] + 1),
            }
        )

        case_dir = INPUT_ROOT / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        input_csv = case_dir / f"{trajectory_name}.csv"
        metadata_json = case_dir / f"{trajectory_name}_metadata.json"
        write_trajectory(trajectory, input_csv)
        metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        row = {
            "trajectory": trajectory_name,
            "case_id": case_id,
            "case_label": segment["case_label"],
            "source_folder": SOURCE_FOLDER_NAME,
            "source_file": source_path.name,
            "source_path": str(source_path),
            "input_csv": str(input_csv),
            "source_workbook": "",
            "base": "local_lab",
            "target": "robot_arm_motion",
            "calibration": "unknown_calibration",
            "exposure": "default_exposure",
            "added_points": False,
            "variant_id": segment["variant_id"],
            "tracker_smoothing": "unknown",
            "variant_note": f"local lab motion trajectory segment {segment_index}",
            "sample_count": trajectory.count,
            "raw_row_count": len(segment_frame),
            "position_unit": "mm",
            "rotation_source": "xr,yr,zr",
            "rotation_assumption": "XYZ Euler angles in degrees",
            "timestamp_source": "segment-normalized source timestamp ticks / 10000",
            "sample_rate_hz": _median_sample_rate(trajectory.timestamps),
            "reference_csv": "",
            "metadata_json": str(metadata_json),
        }
        _upsert_manifest_row(INPUT_ROOT / "manifest.csv", row, key="trajectory")
        rows.append(row)
    return rows


def write_case12_motion_report(run_dir: Path, manifest: pd.DataFrame) -> Path | None:
    case_manifest = manifest.loc[manifest["case_id"].astype(str).eq(CASE12_ID)].copy()
    if case_manifest.empty:
        return None
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        return None

    summary = pd.read_csv(summary_path)
    trajectories = set(case_manifest["trajectory"].astype(str))
    summary = summary.loc[summary["trajectory"].astype(str).isin(trajectories)].copy()
    if summary.empty:
        return None

    manifest_row = case_manifest.iloc[0]
    raw = read_trajectory(manifest_row["input_csv"])
    raw_noise_10 = _local_deviation_stats(raw, window=10, prefix="raw_neighbor10")
    raw_noise_20 = _local_deviation_stats(raw, window=20, prefix="raw_neighbor20")
    raw_motion = _motion_stats(raw)
    rows: list[dict[str, Any]] = []
    for _, result in summary.iterrows():
        filtered = read_trajectory(result["trajectory_path"])
        filtered_noise_10 = _local_deviation_stats(filtered, window=10, prefix="filtered_neighbor10")
        filtered_noise_20 = _local_deviation_stats(filtered, window=20, prefix="filtered_neighbor20")
        drag = _drag_metrics(raw, filtered)
        row = {
            "trajectory": result["trajectory"],
            "algorithm": result["algorithm"],
            "params": result["params"],
            "trajectory_path": result["trajectory_path"],
            "vtk_path": result.get("vtk_path", ""),
            **raw_noise_10,
            **raw_noise_20,
            **filtered_noise_10,
            **filtered_noise_20,
            "neighbor10_z_rms_ratio": _safe_ratio(
                filtered_noise_10["filtered_neighbor10_z_rms"],
                raw_noise_10["raw_neighbor10_z_rms"],
            ),
            "neighbor20_z_rms_ratio": _safe_ratio(
                filtered_noise_20["filtered_neighbor20_z_rms"],
                raw_noise_20["raw_neighbor20_z_rms"],
            ),
            "neighbor10_norm_rms_ratio": _safe_ratio(
                filtered_noise_10["filtered_neighbor10_norm_rms"],
                raw_noise_10["raw_neighbor10_norm_rms"],
            ),
            "neighbor20_norm_rms_ratio": _safe_ratio(
                filtered_noise_20["filtered_neighbor20_norm_rms"],
                raw_noise_20["raw_neighbor20_norm_rms"],
            ),
            "to_raw_translation_rmse": result.get("to_raw_translation_rmse", np.nan),
            "to_raw_translation_p95": result.get("to_raw_translation_p95", np.nan),
            "to_raw_translation_max": result.get("to_raw_translation_max", np.nan),
            "acceleration_rms_ratio": result.get("acceleration_rms_ratio", np.nan),
            "jerk_rms_ratio": result.get("jerk_rms_ratio", np.nan),
            **drag,
        }
        rows.append(row)

    metrics_path = run_dir / "case12_motion_metrics.csv"
    _write_csv(metrics_path, rows)
    report_path = run_dir / "case12_motion_report.md"
    _write_motion_report(
        report_path,
        manifest_row=manifest_row,
        metrics=pd.DataFrame(rows),
        metadata_path=Path(str(manifest_row["metadata_json"])),
        raw_motion=raw_motion,
    )
    return report_path


def _find_source_path() -> Path:
    preferred = SOURCE_ROOT / SOURCE_FOLDER_NAME / SOURCE_FILE_NAME
    if preferred.exists():
        return preferred
    matches = sorted(SOURCE_ROOT.glob(f"*/{SOURCE_FILE_NAME}"))
    if len(matches) == 1:
        return matches[0]
    for match in matches:
        if match.parent.name == SOURCE_FOLDER_NAME:
            return match
    if not matches:
        raise FileNotFoundError(f"source file not found under {SOURCE_ROOT}: {SOURCE_FILE_NAME}")
    raise FileNotFoundError(
        f"found multiple {SOURCE_FILE_NAME} files; expected one in {SOURCE_FOLDER_NAME}"
    )


def _read_local_lab_track(
    source_path: Path,
    *,
    name: str,
    case_id: str,
    trajectory_name: str,
) -> tuple[Trajectory, int, pd.DataFrame, dict[str, Any]]:
    raw_frame = pd.read_csv(
        source_path,
        sep=r"\s+",
        header=None,
        names=LOCAL_LAB_COLUMNS,
        engine="python",
    )
    raw_row_count = len(raw_frame)
    frame = raw_frame[LOCAL_LAB_COLUMNS].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{source_path} contains no valid numeric trajectory rows")

    trajectory = _trajectory_from_frame(
        frame,
        name=name,
        source_path=source_path,
        timestamp_origin="full",
    )
    metadata = _metadata(
        source_path,
        raw_row_count,
        frame,
        trajectory,
        case_id=case_id,
        trajectory_name=trajectory_name,
    )
    return trajectory, raw_row_count, frame, metadata


def _trajectory_from_frame(
    frame: pd.DataFrame,
    *,
    name: str,
    source_path: Path,
    timestamp_origin: str,
) -> Trajectory:
    local_frame = frame.reset_index(drop=True)
    positions = local_frame[["x", "y", "z"]].to_numpy(dtype=float)
    rotations = Rotation.from_euler(
        "xyz",
        local_frame[["xr", "yr", "zr"]].to_numpy(dtype=float),
        degrees=True,
    )
    timestamps = _timestamps_from_source(local_frame)
    return Trajectory(
        make_poses(positions, rotations),
        timestamps=timestamps,
        name=name,
        metadata={
            "source": str(source_path),
            "position_unit": "mm",
            "rotation_assumption": "XYZ Euler angles in degrees",
            "timestamp_source": f"{timestamp_origin}-normalized source timestamp ticks",
        },
    )


def _split_frame_by_gaps(frame: pd.DataFrame) -> list[pd.DataFrame]:
    timestamps = _timestamps_from_source(frame.reset_index(drop=True))
    gap_indices = np.where(np.diff(timestamps) > MAX_CONTINUOUS_STEP_S)[0]
    starts = [0, *[int(index + 1) for index in gap_indices]]
    stops = [*[int(index + 1) for index in gap_indices], len(frame)]
    return [frame.iloc[start:stop].copy() for start, stop in zip(starts, stops, strict=True)]


def _timestamps_from_source(frame: pd.DataFrame) -> np.ndarray:
    source_timestamps = frame["timestamp"].to_numpy(dtype=float)
    timestamps = (source_timestamps - source_timestamps[0]) / TIMESTAMP_TICKS_PER_SECOND
    if len(timestamps) > 1 and np.all(np.diff(timestamps) > 0):
        return timestamps

    rate = float(frame["rate"].replace(0, np.nan).dropna().median())
    if not np.isfinite(rate) or rate <= 0:
        rate = 100.0
    return np.arange(len(frame), dtype=float) / rate


def _metadata(
    source_path: Path,
    raw_row_count: int,
    frame: pd.DataFrame,
    trajectory: Trajectory,
    *,
    case_id: str,
    trajectory_name: str,
) -> dict[str, Any]:
    positions = trajectory.positions
    timestamps = trajectory.timestamps
    assert timestamps is not None
    dt = np.diff(timestamps)
    valid_dt = dt[dt > 0]
    gaps = np.where(dt > MAX_CONTINUOUS_STEP_S)[0]
    speed_stats = _step_speed_stats(positions, timestamps)
    euler = frame[["xr", "yr", "zr"]].to_numpy(dtype=float)
    return {
        "case_id": case_id,
        "trajectory": trajectory_name,
        "source_path": str(source_path),
        "source_columns": LOCAL_LAB_COLUMNS,
        "raw_row_count": raw_row_count,
        "sample_count": trajectory.count,
        "position_unit": "mm",
        "rotation_assumption": "XYZ Euler angles in degrees",
        "timestamp_ticks_per_second": TIMESTAMP_TICKS_PER_SECOND,
        "duration_s": float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0,
        "sample_rate_hz_median": _median_sample_rate(timestamps),
        "source_rate_unique": sorted(float(value) for value in frame["rate"].dropna().unique()),
        "dt_s_min": float(valid_dt.min()) if valid_dt.size else 0.0,
        "dt_s_median": float(np.median(valid_dt)) if valid_dt.size else 0.0,
        "dt_s_max": float(valid_dt.max()) if valid_dt.size else 0.0,
        "large_gap_threshold_s": MAX_CONTINUOUS_STEP_S,
        "large_gap_count": int(gaps.size),
        "large_gaps": [
            {
                "after_frame": int(index),
                "before_frame": int(index),
                "next_frame": int(index + 1),
                "dt_s": float(dt[index]),
            }
            for index in gaps[:20]
        ],
        "position_min_mm": _axis_dict(positions.min(axis=0)),
        "position_max_mm": _axis_dict(positions.max(axis=0)),
        "position_range_mm": _axis_dict(np.ptp(positions, axis=0)),
        "position_std_mm": _axis_dict(positions.std(axis=0, ddof=0)),
        "euler_range_deg": _axis_dict(np.ptp(euler, axis=0), axes=("xr", "yr", "zr")),
        "continuous_step_speed_mm_s": speed_stats,
    }


def _motion_stats(traj: Trajectory) -> dict[str, float]:
    positions = traj.positions
    timestamps = traj.timestamps
    assert timestamps is not None
    return {
        "range_x": float(np.ptp(positions[:, 0])),
        "range_y": float(np.ptp(positions[:, 1])),
        "range_z": float(np.ptp(positions[:, 2])),
        **{f"speed_{key}": value for key, value in _step_speed_stats(positions, timestamps).items()},
    }


def _step_speed_stats(positions: np.ndarray, timestamps: np.ndarray) -> dict[str, float]:
    dt = np.diff(timestamps)
    dp = np.diff(positions, axis=0)
    valid = (dt > 0.0) & (dt <= MAX_CONTINUOUS_STEP_S)
    if not np.any(valid):
        return {"median": 0.0, "p95": 0.0, "max": 0.0}
    speed = np.linalg.norm(dp[valid], axis=1) / dt[valid]
    return {
        "median": float(np.median(speed)),
        "p95": float(np.percentile(speed, 95)),
        "max": float(np.max(speed)),
    }


def _local_deviation_stats(traj: Trajectory, *, window: int, prefix: str) -> dict[str, float | int]:
    deviations = neighbor_mean_deviation(traj.positions, window)
    sample_slice = complete_neighbor_slice(traj.count, window)
    values = deviations[sample_slice]
    valid = _continuous_window_mask(traj, window)[sample_slice]
    values = values[valid]
    if values.size == 0:
        return {
            f"{prefix}_sample_count": 0,
            f"{prefix}_x_rms": 0.0,
            f"{prefix}_y_rms": 0.0,
            f"{prefix}_z_rms": 0.0,
            f"{prefix}_norm_rms": 0.0,
            f"{prefix}_norm_p95": 0.0,
        }
    norm = np.linalg.norm(values, axis=1)
    rms = np.sqrt(np.mean(values**2, axis=0))
    return {
        f"{prefix}_sample_count": int(values.shape[0]),
        f"{prefix}_x_rms": float(rms[0]),
        f"{prefix}_y_rms": float(rms[1]),
        f"{prefix}_z_rms": float(rms[2]),
        f"{prefix}_norm_rms": float(np.sqrt(np.mean(norm**2))),
        f"{prefix}_norm_p95": float(np.percentile(norm, 95)),
    }


def _continuous_window_mask(traj: Trajectory, window: int) -> np.ndarray:
    valid = np.zeros(traj.count, dtype=bool)
    if traj.count <= window * 2:
        return valid
    if traj.timestamps is None:
        valid[window : traj.count - window] = True
        return valid

    dt = np.diff(traj.timestamps)
    bad = (dt <= 0.0) | (dt > MAX_CONTINUOUS_STEP_S)
    bad_prefix = np.concatenate([[0], np.cumsum(bad.astype(int))])
    centers = np.arange(window, traj.count - window)
    bad_count = bad_prefix[centers + window] - bad_prefix[centers - window]
    valid[centers] = bad_count == 0
    return valid


def _drag_metrics(raw: Trajectory, filtered: Trajectory) -> dict[str, float | int]:
    count = min(raw.count, filtered.count)
    positions = raw.positions[:count]
    filtered_positions = filtered.positions[:count]
    timestamps = raw.timestamps[:count] if raw.timestamps is not None else None
    velocity, valid = _sample_velocity(positions, timestamps)
    speed = np.linalg.norm(velocity, axis=1)
    valid &= speed > 10.0
    valid &= _drag_warmup_mask(raw, count)

    if not np.any(valid):
        return {
            "drag_sample_count": 0,
            "continuous_speed_median_mm_s": 0.0,
            "along_lag_mm_mean_positive": 0.0,
            "along_lag_mm_median_positive": 0.0,
            "along_lag_mm_p95_positive": 0.0,
            "along_lag_ms_p95_positive": 0.0,
            "signed_along_lag_mm_median": 0.0,
            "signed_along_lag_mm_p95": 0.0,
            "positive_lag_fraction": 0.0,
            "cross_track_error_p95_mm": 0.0,
        }

    tangent = velocity[valid] / speed[valid, None]
    error = filtered_positions[valid] - positions[valid]
    along_error = np.einsum("ij,ij->i", error, tangent)
    signed_lag = -along_error
    positive_lag = np.maximum(signed_lag, 0.0)
    cross_track = np.linalg.norm(error - along_error[:, None] * tangent, axis=1)
    lag_ms = positive_lag / np.maximum(speed[valid], 1e-12) * 1000.0
    return {
        "drag_sample_count": int(np.count_nonzero(valid)),
        "continuous_speed_median_mm_s": float(np.median(speed[valid])),
        "along_lag_mm_mean_positive": float(np.mean(positive_lag)),
        "along_lag_mm_median_positive": float(np.median(positive_lag)),
        "along_lag_mm_p95_positive": float(np.percentile(positive_lag, 95)),
        "along_lag_ms_p95_positive": float(np.percentile(lag_ms, 95)),
        "signed_along_lag_mm_median": float(np.median(signed_lag)),
        "signed_along_lag_mm_p95": float(np.percentile(signed_lag, 95)),
        "positive_lag_fraction": float(np.mean(signed_lag > 0.0)),
        "cross_track_error_p95_mm": float(np.percentile(cross_track, 95)),
    }


def _sample_velocity(
    positions: np.ndarray,
    timestamps: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    count = positions.shape[0]
    velocity = np.zeros_like(positions)
    valid = np.zeros(count, dtype=bool)
    if count < 2:
        return velocity, valid

    if timestamps is None:
        dt = np.ones(count - 1, dtype=float)
    else:
        dt = np.diff(timestamps)
    step_valid = (dt > 0.0) & (dt <= MAX_CONTINUOUS_STEP_S)
    step_velocity = np.zeros((count - 1, 3), dtype=float)
    step_velocity[step_valid] = np.diff(positions, axis=0)[step_valid] / dt[step_valid, None]

    for index in range(count):
        candidates = []
        if index > 0 and step_valid[index - 1]:
            candidates.append(step_velocity[index - 1])
        if index < count - 1 and step_valid[index]:
            candidates.append(step_velocity[index])
        if candidates:
            velocity[index] = np.mean(candidates, axis=0)
            valid[index] = True
    return velocity, valid


def _drag_warmup_mask(raw: Trajectory, count: int) -> np.ndarray:
    mask = np.ones(count, dtype=bool)
    if raw.timestamps is None or count < 2:
        return mask
    dt = np.diff(raw.timestamps[:count])
    segment_starts = [0, *[int(index + 1) for index in np.where(dt > MAX_CONTINUOUS_STEP_S)[0]]]
    for start in segment_starts:
        stop = min(count, start + DRAG_WARMUP_FRAMES)
        mask[start:stop] = False
    return mask


def _write_motion_report(
    path: Path,
    *,
    manifest_row: pd.Series,
    metrics: pd.DataFrame,
    metadata_path: Path,
    raw_motion: dict[str, float],
) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    best_noise_10 = metrics.sort_values(["neighbor10_z_rms_ratio", "along_lag_ms_p95_positive"]).head(8)
    best_noise_20 = metrics.sort_values(["neighbor20_z_rms_ratio", "along_lag_ms_p95_positive"]).head(8)
    best_drag = metrics.sort_values(["along_lag_ms_p95_positive", "neighbor10_z_rms_ratio"]).head(8)
    one_euro = metrics.loc[metrics["algorithm"].astype(str).eq("one_euro_z")].copy()
    one_euro = one_euro.sort_values(["neighbor10_z_rms_ratio", "along_lag_ms_p95_positive"]).head(8)

    lines = [
        "# Case 12 Local Lab Motion Report",
        "",
        "## Data",
        "",
        f"- Input: `{manifest_row['input_csv']}`",
        f"- Source: `{manifest_row['source_path']}`",
        f"- Samples: {int(manifest_row['sample_count'])}",
        f"- Duration: {metadata.get('duration_s', 0.0):.6g} s",
        f"- Median sample rate: {metadata.get('sample_rate_hz_median', 0.0):.6g} Hz",
        f"- Large timestamp gaps (> {MAX_CONTINUOUS_STEP_S:.3g} s): {metadata.get('large_gap_count', 0)}",
        f"- Position ranges: X {raw_motion['range_x']:.6g} mm, "
        f"Y {raw_motion['range_y']:.6g} mm, Z {raw_motion['range_z']:.6g} mm",
        f"- Continuous speed median/p95/max: {raw_motion['speed_median']:.6g} / "
        f"{raw_motion['speed_p95']:.6g} / {raw_motion['speed_max']:.6g} mm/s",
        "",
        "## Interpretation",
        "",
        "- No ground-truth reference was provided for this trajectory.",
        "- Noise is estimated with XYZ deviation from a centered neighbor mean, excluding head/tail samples and windows crossing timestamp gaps.",
        "- Drag is estimated as positive along-track lag against the raw local motion tangent, excluding long timestamp gaps and the first 20 frames of each continuous segment.",
        "- Centered filters such as moving average and Savitzky-Golay use future samples in this batch test; their lag numbers are not directly usable as realtime latency.",
        "",
        "## Best Z Noise Reduction, Neighbor Window 10",
        "",
        *_metrics_table(best_noise_10),
        "",
        "## Best Z Noise Reduction, Neighbor Window 20",
        "",
        *_metrics_table(best_noise_20),
        "",
        "## Lowest Along-Track Drag Proxy",
        "",
        *_metrics_table(best_drag),
    ]
    if not one_euro.empty:
        lines.extend(
            [
                "",
                "## One Euro Z Candidates",
                "",
                *_metrics_table(one_euro),
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _metrics_table(frame: pd.DataFrame) -> list[str]:
    columns = [
        "algorithm",
        "params",
        "neighbor10_z_rms_ratio",
        "neighbor20_z_rms_ratio",
        "along_lag_ms_p95_positive",
        "along_lag_mm_p95_positive",
        "to_raw_translation_rmse",
        "jerk_rms_ratio",
    ]
    lines = [
        "| algorithm | params | z ratio w10 | z ratio w20 | lag p95 ms | lag p95 mm | to raw RMSE | jerk ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(f"`{value}`" if column == "params" else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _upsert_manifest_row(path: Path, row: dict[str, Any], *, key: str) -> None:
    if path.exists():
        frame = pd.read_csv(path)
        frame = frame.loc[frame[key].astype(str) != str(row[key])].copy()
        for column in row:
            if column not in frame.columns:
                frame[column] = ""
        for column in frame.columns:
            if column not in row:
                row[column] = ""
        frame = pd.concat([frame, pd.DataFrame([row], columns=frame.columns)], ignore_index=True)
    else:
        frame = pd.DataFrame([row])
    frame.to_csv(path, index=False)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _axis_dict(values: np.ndarray, axes: tuple[str, str, str] = ("x", "y", "z")) -> dict[str, float]:
    return {axis: float(value) for axis, value in zip(axes, values, strict=True)}


def _median_sample_rate(timestamps: np.ndarray | None) -> float:
    if timestamps is None or len(timestamps) < 2:
        return 0.0
    dt = np.diff(timestamps)
    dt = dt[(dt > 0.0) & (dt <= MAX_CONTINUOUS_STEP_S)]
    if dt.size == 0:
        return 0.0
    return float(1.0 / np.median(dt))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0 if numerator == 0 else float("inf")
    return float(numerator / denominator)


if __name__ == "__main__":
    main()
