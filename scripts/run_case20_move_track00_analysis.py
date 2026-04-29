from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in (ROOT, SCRIPT_DIR):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

from rt_filter.batch import parameter_slug
from rt_filter.filters import run_filter
from rt_filter.io import write_trajectory
from rt_filter.se3 import make_poses
from rt_filter.trajectory import Trajectory

import run_case16_hotel_analysis as static_base
import run_local_lab_case12 as motion_base


SOURCE_PATH = Path("ref_data/move/track_data_00.txt")
INPUT_ROOT = Path("input/sn")
OUTPUT_ROOT = Path("outputs/case20_track_data_00_segments")
SOURCE_COLUMNS = ["x", "y", "z", "xr", "yr", "zr", "time", "rate"]
TIMESTAMP_TICKS_PER_SECOND = 10000.0

CASE_START_INDEX = 20
MOTION_SPEED_THRESHOLD_MM_S = 20.0
MOTION_SMOOTH_WINDOW_FRAMES = 50
MIN_MOTION_DURATION_S = 0.8
STATIC_BUFFER_S = 3.0
NEIGHBOR_WINDOW_STATIC = 10
NEIGHBOR_WINDOW_MOTION = 5

FILTER_SPECS = [
    {"algorithm": "raw", "params": {}},
    {"algorithm": "butterworth_z", "params": {"cutoff_hz": 4.0, "order": 2}},
    {"algorithm": "butterworth_z", "params": {"cutoff_hz": 8.0, "order": 2}},
    {"algorithm": "butterworth_z", "params": {"cutoff_hz": 12.0, "order": 2}},
    {
        "algorithm": "one_euro_z",
        "params": {
            "min_cutoff": 2.0,
            "beta": 1.0,
            "d_cutoff": 4.0,
            "derivative_deadband": 0.0,
        },
    },
    {
        "algorithm": "one_euro_z",
        "params": {
            "min_cutoff": 5.0,
            "beta": 0.5,
            "d_cutoff": 4.0,
            "derivative_deadband": 0.0,
        },
    },
    {"algorithm": "savgol", "params": {"window": 9, "polyorder": 2}},
    {
        "algorithm": "adaptive_kalman_z",
        "params": {
            "process_noise": 1e-8,
            "measurement_noise": 1e-4,
            "initial_covariance": 1.0,
            "motion_process_gain": 0.1,
            "velocity_deadband": 5.0,
            "innovation_scale": 4.0,
            "innovation_gate": 3.0,
            "max_measurement_scale": 20.0,
        },
    },
]


def main() -> None:
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run() -> dict[str, Any]:
    frame = _read_source_frame()
    segments = _detect_segments(frame)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    case_rows: list[dict[str, Any]] = []
    overall_rows: list[dict[str, Any]] = []
    segment_manifest: list[dict[str, Any]] = []
    reports: list[str] = []

    for index, segment in enumerate(segments):
        case_number = CASE_START_INDEX + index
        case_result = _write_segment_case(case_number, index + 1, segment)
        case_rows.append(case_result["manifest_row"])
        segment_manifest.append(case_result["segment_manifest"])
        overall_rows.extend(case_result["filter_rows"])
        reports.append(case_result["report_md"])

    segment_manifest_path = OUTPUT_ROOT / "segment_manifest.json"
    segment_manifest_path.write_text(json.dumps(segment_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    overall_csv = OUTPUT_ROOT / "overall_filter_comparison.csv"
    motion_base._write_csv(overall_csv, overall_rows)

    summary_rows = _summarize_overall_filters(pd.DataFrame(overall_rows))
    overall_summary_csv = OUTPUT_ROOT / "overall_filter_summary.csv"
    motion_base._write_csv(overall_summary_csv, summary_rows)

    overall_report_md = OUTPUT_ROOT / "overall_report.md"
    _write_overall_report(overall_report_md, pd.DataFrame(segment_manifest), pd.DataFrame(summary_rows))

    return {
        "segment_count": len(segment_manifest),
        "cases": [row["case_id"] for row in case_rows],
        "segment_manifest_json": str(segment_manifest_path),
        "overall_csv": str(overall_csv),
        "overall_summary_csv": str(overall_summary_csv),
        "overall_report_md": str(overall_report_md),
        "reports": reports,
    }


def _read_source_frame() -> pd.DataFrame:
    raw_frame = pd.read_csv(SOURCE_PATH, sep="\t")
    frame = raw_frame[SOURCE_COLUMNS].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{SOURCE_PATH} contains no valid numeric trajectory rows")
    return frame


def _detect_segments(frame: pd.DataFrame) -> list[dict[str, Any]]:
    timestamps = _timestamps_from_frame(frame)
    positions = frame[["x", "y", "z"]].to_numpy(dtype=float)
    step_speed = _step_speed_mm_s(positions, timestamps)
    smooth_speed = _smooth_step_speed(step_speed, MOTION_SMOOTH_WINDOW_FRAMES)
    moving_steps = smooth_speed > MOTION_SPEED_THRESHOLD_MM_S
    step_runs = _boolean_runs(moving_steps)
    sample_rate_hz = motion_base._median_sample_rate(timestamps)
    buffer_frames = max(1, int(round(STATIC_BUFFER_S * sample_rate_hz)))
    min_motion_frames = max(1, int(round(MIN_MOTION_DURATION_S * sample_rate_hz)))

    segments: list[dict[str, Any]] = []
    for is_moving, start_step, stop_step_exclusive in step_runs:
        if not is_moving:
            continue
        motion_frame_count = stop_step_exclusive - start_step
        if motion_frame_count < min_motion_frames:
            continue
        motion_start = int(start_step)
        motion_stop_exclusive = int(min(len(frame), stop_step_exclusive + 1))
        segment_start = max(0, motion_start - buffer_frames)
        segment_stop_exclusive = min(len(frame), motion_stop_exclusive + buffer_frames)
        segment_frame = frame.iloc[segment_start:segment_stop_exclusive].reset_index(drop=True)
        relative_motion_start = motion_start - segment_start
        relative_motion_stop_exclusive = motion_stop_exclusive - segment_start

        segments.append(
            {
                "source_start_row": segment_start,
                "source_stop_row_exclusive": segment_stop_exclusive,
                "motion_start_row": motion_start,
                "motion_stop_row_exclusive": motion_stop_exclusive,
                "motion_start_relative": relative_motion_start,
                "motion_stop_relative": relative_motion_stop_exclusive,
                "segment_frame": segment_frame,
                "detection": {
                    "speed_threshold_mm_s": MOTION_SPEED_THRESHOLD_MM_S,
                    "smooth_window_frames": MOTION_SMOOTH_WINDOW_FRAMES,
                    "min_motion_duration_s": MIN_MOTION_DURATION_S,
                    "static_buffer_s": STATIC_BUFFER_S,
                },
            }
        )
    return segments


def _write_segment_case(case_number: int, segment_index: int, segment: dict[str, Any]) -> dict[str, Any]:
    case_id = f"case_{case_number:02d}_dynamic_robot_arm_track_data_00_segment_{segment_index:02d}"
    trajectory_name = f"{case_id}_01_track_data_00"
    case_dir = INPUT_ROOT / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    trajectory = _trajectory_from_frame(
        segment["segment_frame"],
        name=trajectory_name,
        timestamp_origin="segment",
    )
    metadata = _segment_metadata(case_id, trajectory_name, segment, trajectory)
    metadata_path = case_dir / f"{trajectory_name}_metadata.json"
    input_csv = case_dir / f"{trajectory_name}.csv"
    write_trajectory(trajectory, input_csv)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_row = {
        "trajectory": trajectory_name,
        "case_id": case_id,
        "case_label": "dynamic_robot_arm_segment",
        "source_folder": "ref_data/move",
        "source_file": SOURCE_PATH.name,
        "source_path": str(SOURCE_PATH),
        "input_csv": str(input_csv),
        "source_workbook": "",
        "base": "move",
        "target": "robot_arm_motion",
        "calibration": "unknown_calibration",
        "exposure": "default_exposure",
        "added_points": False,
        "variant_id": f"segment_{segment_index:02d}",
        "tracker_smoothing": "unknown",
        "variant_note": f"track_data_00 motion segment {segment_index} with {STATIC_BUFFER_S:.1f}s static context",
        "sample_count": trajectory.count,
        "raw_row_count": len(segment["segment_frame"]),
        "position_unit": "mm",
        "rotation_source": "xr,yr,zr",
        "rotation_assumption": "XYZ Euler angles in degrees",
        "timestamp_source": "segment-normalized source timestamp ticks / 10000 or generated from rate",
        "sample_rate_hz": motion_base._median_sample_rate(trajectory.timestamps),
        "reference_csv": "",
        "metadata_json": str(metadata_path),
    }
    motion_base._upsert_manifest_row(INPUT_ROOT / "manifest.csv", manifest_row, key="trajectory")

    output_dir = OUTPUT_ROOT / case_id
    output_dir.mkdir(parents=True, exist_ok=True)
    filter_rows = _run_filter_analysis(output_dir, trajectory, metadata)
    comparison_csv = output_dir / f"{case_id}_filter_comparison.csv"
    motion_base._write_csv(comparison_csv, filter_rows)
    report_md = output_dir / f"{case_id}_report.md"
    _write_segment_report(report_md, metadata, pd.DataFrame(filter_rows))

    return {
        "manifest_row": manifest_row,
        "segment_manifest": {
            "case_id": case_id,
            "trajectory": trajectory_name,
            "input_csv": str(input_csv),
            "metadata_json": str(metadata_path),
            "comparison_csv": str(comparison_csv),
            "report_md": str(report_md),
            "source_start_row": segment["source_start_row"],
            "source_stop_row_exclusive": segment["source_stop_row_exclusive"],
            "motion_start_row": segment["motion_start_row"],
            "motion_stop_row_exclusive": segment["motion_stop_row_exclusive"],
            "duration_s": metadata["duration_s"],
            "motion_duration_s": metadata["motion_duration_s"],
        },
        "filter_rows": filter_rows,
        "report_md": str(report_md),
    }


def _trajectory_from_frame(
    frame: pd.DataFrame,
    *,
    name: str,
    timestamp_origin: str,
) -> Trajectory:
    positions = frame[["x", "y", "z"]].to_numpy(dtype=float)
    rotations = Rotation.from_euler("xyz", frame[["xr", "yr", "zr"]].to_numpy(dtype=float), degrees=True)
    timestamps = _timestamps_from_frame(frame)
    return Trajectory(
        make_poses(positions, rotations),
        timestamps=timestamps,
        name=name,
        metadata={
            "source": str(SOURCE_PATH),
            "position_unit": "mm",
            "rotation_assumption": "XYZ Euler angles in degrees",
            "timestamp_source": f"{timestamp_origin}-normalized source timestamp ticks",
        },
    )


def _timestamps_from_frame(frame: pd.DataFrame) -> np.ndarray:
    source_timestamps = frame["time"].to_numpy(dtype=float)
    timestamps = (source_timestamps - source_timestamps[0]) / TIMESTAMP_TICKS_PER_SECOND
    if len(timestamps) > 1 and np.all(np.diff(timestamps) > 0):
        return timestamps

    rate = float(frame["rate"].replace(0, np.nan).dropna().median())
    if not np.isfinite(rate) or rate <= 0:
        rate = 100.0
    return np.arange(len(frame), dtype=float) / rate


def _step_speed_mm_s(positions: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    dt = np.diff(timestamps)
    dp = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    valid = dt > 0.0
    speed = np.zeros_like(dp)
    speed[valid] = dp[valid] / dt[valid]
    return speed


def _smooth_step_speed(step_speed: np.ndarray, window: int) -> np.ndarray:
    if step_speed.size == 0:
        return step_speed.copy()
    window = max(1, min(int(window), step_speed.size))
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(step_speed, kernel, mode="same")


def _boolean_runs(values: np.ndarray) -> list[tuple[bool, int, int]]:
    if values.size == 0:
        return []
    runs: list[tuple[bool, int, int]] = []
    start = 0
    state = bool(values[0])
    for index in range(1, values.size):
        current = bool(values[index])
        if current != state:
            runs.append((state, start, index))
            start = index
            state = current
    runs.append((state, start, values.size))
    return runs


def _segment_metadata(
    case_id: str,
    trajectory_name: str,
    segment: dict[str, Any],
    trajectory: Trajectory,
) -> dict[str, Any]:
    positions = trajectory.positions
    timestamps = trajectory.timestamps
    assert timestamps is not None
    motion_start = int(segment["motion_start_relative"])
    motion_stop = int(segment["motion_stop_relative"])
    pre_summary = _window_summary(positions[:motion_start], timestamps[:motion_start], NEIGHBOR_WINDOW_STATIC)
    motion_summary = _window_summary(
        positions[motion_start:motion_stop],
        timestamps[motion_start:motion_stop],
        NEIGHBOR_WINDOW_MOTION,
    )
    post_summary = _window_summary(positions[motion_stop:], timestamps[motion_stop:], NEIGHBOR_WINDOW_STATIC)
    dt = np.diff(timestamps)
    valid_dt = dt[dt > 0]

    return {
        "case_id": case_id,
        "trajectory": trajectory_name,
        "source_path": str(SOURCE_PATH),
        "source_columns": SOURCE_COLUMNS,
        "raw_row_count": len(segment["segment_frame"]),
        "sample_count": trajectory.count,
        "position_unit": "mm",
        "rotation_assumption": "XYZ Euler angles in degrees",
        "timestamp_ticks_per_second": TIMESTAMP_TICKS_PER_SECOND,
        "duration_s": float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0,
        "sample_rate_hz_median": motion_base._median_sample_rate(timestamps),
        "source_rate_unique": sorted(float(value) for value in segment["segment_frame"]["rate"].dropna().unique()),
        "dt_s_min": float(valid_dt.min()) if valid_dt.size else 0.0,
        "dt_s_median": float(np.median(valid_dt)) if valid_dt.size else 0.0,
        "dt_s_max": float(valid_dt.max()) if valid_dt.size else 0.0,
        "source_start_row": int(segment["source_start_row"]),
        "source_stop_row_exclusive": int(segment["source_stop_row_exclusive"]),
        "motion_start_row": int(segment["motion_start_row"]),
        "motion_stop_row_exclusive": int(segment["motion_stop_row_exclusive"]),
        "motion_start_relative": motion_start,
        "motion_stop_relative": motion_stop,
        "pre_static_duration_s": float(motion_start / max(motion_base._median_sample_rate(timestamps), 1e-12)),
        "motion_duration_s": float(max(motion_stop - motion_start - 1, 0) / max(motion_base._median_sample_rate(timestamps), 1e-12)),
        "post_static_duration_s": float(max(trajectory.count - motion_stop, 0) / max(motion_base._median_sample_rate(timestamps), 1e-12)),
        "position_min_mm": motion_base._axis_dict(positions.min(axis=0)),
        "position_max_mm": motion_base._axis_dict(positions.max(axis=0)),
        "position_range_mm": motion_base._axis_dict(np.ptp(positions, axis=0)),
        "position_std_mm": motion_base._axis_dict(positions.std(axis=0, ddof=0)),
        "continuous_step_speed_mm_s": motion_base._step_speed_stats(positions, timestamps),
        "segment_detection": segment["detection"],
        "pre_static_summary": pre_summary,
        "motion_summary": motion_summary,
        "post_static_summary": post_summary,
    }


def _window_summary(positions: np.ndarray, timestamps: np.ndarray, neighbor_window: int) -> dict[str, float | int]:
    if positions.shape[0] == 0:
        return {
            "sample_count": 0,
            "duration_s": 0.0,
            "x_std_mm": 0.0,
            "y_std_mm": 0.0,
            "z_std_mm": 0.0,
            "xy_mean_std_mm": 0.0,
            "z_over_xy_mean_std_ratio": 0.0,
            "neighbor_z_rms_mm": 0.0,
            "neighbor_norm_rms_mm": 0.0,
            "z_range_mm": 0.0,
            "dominant_frequency_hz": 0.0,
            "spectral_centroid_hz": 0.0,
            "power_fraction_above_5hz": 0.0,
            "power_fraction_above_10hz": 0.0,
            "peak_count": 0,
            "peaks_per_second": 0.0,
        }
    std = np.std(positions, axis=0, ddof=0)
    xy_mean_std = float(np.mean(std[:2]))
    z_values = positions[:, 2]
    centered_z = z_values - float(np.mean(z_values))
    sample_rate_hz = motion_base._median_sample_rate(timestamps)
    spectrum = static_base._spectrum_summary(centered_z, sample_rate_hz=sample_rate_hz or 100.0)
    residual = static_base._centered_neighbor_residual(positions, neighbor_window)
    residual_norm = np.linalg.norm(residual, axis=1)

    if len(z_values) >= 3 and sample_rate_hz > 0.0:
        min_peak_distance = max(1, int(sample_rate_hz / 12.0))
        prominence = max(float(np.std(z_values, ddof=0)) * 0.05, 1e-6)
        peaks, _ = find_peaks(z_values, distance=min_peak_distance, prominence=prominence)
    else:
        peaks = np.zeros(0, dtype=int)
    duration_s = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
    return {
        "sample_count": int(positions.shape[0]),
        "duration_s": duration_s,
        "x_std_mm": float(std[0]),
        "y_std_mm": float(std[1]),
        "z_std_mm": float(std[2]),
        "xy_mean_std_mm": xy_mean_std,
        "z_over_xy_mean_std_ratio": motion_base._safe_ratio(float(std[2]), xy_mean_std),
        "neighbor_z_rms_mm": float(np.sqrt(np.mean(residual[:, 2] ** 2))) if residual.size else 0.0,
        "neighbor_norm_rms_mm": float(np.sqrt(np.mean(residual_norm**2))) if residual_norm.size else 0.0,
        "z_range_mm": float(np.ptp(z_values)),
        "dominant_frequency_hz": float(spectrum["dominant_frequency_hz"]),
        "spectral_centroid_hz": float(spectrum["spectral_centroid_hz"]),
        "power_fraction_above_5hz": float(spectrum["power_fraction_above_5hz"]),
        "power_fraction_above_10hz": float(spectrum["power_fraction_above_10hz"]),
        "peak_count": int(peaks.size),
        "peaks_per_second": float(peaks.size / duration_s) if duration_s > 0 else 0.0,
    }


def _run_filter_analysis(output_dir: Path, raw: Trajectory, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    motion_start = int(metadata["motion_start_relative"])
    motion_stop = int(metadata["motion_stop_relative"])
    raw_pre = metadata["pre_static_summary"]
    raw_motion = metadata["motion_summary"]
    raw_post = metadata["post_static_summary"]

    rows: list[dict[str, Any]] = []
    for spec in FILTER_SPECS:
        algorithm = str(spec["algorithm"])
        params = dict(spec["params"])
        filtered = raw if algorithm == "raw" else run_filter(algorithm, raw, params)

        slug = parameter_slug(params)
        filtered_csv = output_dir / f"{algorithm}__{slug}.csv"
        write_trajectory(filtered, filtered_csv)

        positions = filtered.positions
        timestamps = filtered.timestamps
        assert timestamps is not None
        pre_summary = _window_summary(positions[:motion_start], timestamps[:motion_start], NEIGHBOR_WINDOW_STATIC)
        motion_summary = _window_summary(
            positions[motion_start:motion_stop],
            timestamps[motion_start:motion_stop],
            NEIGHBOR_WINDOW_MOTION,
        )
        post_summary = _window_summary(positions[motion_stop:], timestamps[motion_stop:], NEIGHBOR_WINDOW_STATIC)
        drag = motion_base._drag_metrics(raw, filtered)

        static_z_std_ratio = np.mean(
            [
                motion_base._safe_ratio(pre_summary["z_std_mm"], raw_pre["z_std_mm"]),
                motion_base._safe_ratio(post_summary["z_std_mm"], raw_post["z_std_mm"]),
            ]
        )
        static_neighbor_ratio = np.mean(
            [
                motion_base._safe_ratio(pre_summary["neighbor_z_rms_mm"], raw_pre["neighbor_z_rms_mm"]),
                motion_base._safe_ratio(post_summary["neighbor_z_rms_mm"], raw_post["neighbor_z_rms_mm"]),
            ]
        )
        range_ratio = motion_base._safe_ratio(motion_summary["z_range_mm"], raw_motion["z_range_mm"])
        motion_neighbor_ratio = motion_base._safe_ratio(
            motion_summary["neighbor_z_rms_mm"],
            raw_motion["neighbor_z_rms_mm"],
        )
        motion_power5_ratio = motion_base._safe_ratio(
            motion_summary["power_fraction_above_5hz"],
            raw_motion["power_fraction_above_5hz"],
        )
        motion_power10_ratio = motion_base._safe_ratio(
            motion_summary["power_fraction_above_10hz"],
            raw_motion["power_fraction_above_10hz"],
        )
        motion_dom_shift = abs(motion_summary["dominant_frequency_hz"] - raw_motion["dominant_frequency_hz"])
        raw_motion_z = raw.positions[motion_start:motion_stop, 2]
        filtered_motion_z = filtered.positions[motion_start:motion_stop, 2]
        corr = _centered_correlation(filtered_motion_z, raw_motion_z)
        motion_z_rmse = float(np.sqrt(np.mean((filtered_motion_z - raw_motion_z) ** 2)))
        score = (
            0.30 * static_z_std_ratio
            + 0.20 * static_neighbor_ratio
            + 0.20 * motion_neighbor_ratio
            + 0.15 * abs(range_ratio - 1.0)
            + 0.10 * min(float(drag["along_lag_ms_p95_positive"]) / 20.0, 3.0)
            + 0.05 * motion_base._safe_ratio(motion_dom_shift, max(raw_motion["dominant_frequency_hz"], 1.0))
        )

        rows.append(
            {
                "case_id": str(metadata["case_id"]),
                "trajectory": str(metadata["trajectory"]),
                "algorithm": algorithm,
                "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
                "filtered_csv": str(filtered_csv),
                "static_pre_z_std_ratio": motion_base._safe_ratio(pre_summary["z_std_mm"], raw_pre["z_std_mm"]),
                "static_post_z_std_ratio": motion_base._safe_ratio(post_summary["z_std_mm"], raw_post["z_std_mm"]),
                "static_mean_z_std_ratio": float(static_z_std_ratio),
                "static_mean_neighbor_z_ratio": float(static_neighbor_ratio),
                "motion_z_range_ratio_to_raw": float(range_ratio),
                "motion_neighbor_z_ratio_to_raw": float(motion_neighbor_ratio),
                "motion_power_above_5hz_ratio_to_raw": float(motion_power5_ratio),
                "motion_power_above_10hz_ratio_to_raw": float(motion_power10_ratio),
                "motion_z_corr_to_raw": float(corr),
                "motion_dominant_frequency_hz": float(motion_summary["dominant_frequency_hz"]),
                "motion_dominant_frequency_shift_hz": float(motion_dom_shift),
                "motion_peaks_per_second": float(motion_summary["peaks_per_second"]),
                "motion_z_rmse_to_raw_mm": float(motion_z_rmse),
                "to_raw_translation_rmse_mm": float(
                    np.sqrt(np.mean(np.sum((filtered.positions - raw.positions) ** 2, axis=1)))
                ),
                "along_lag_ms_p95_positive": float(drag["along_lag_ms_p95_positive"]),
                "along_lag_mm_p95_positive": float(drag["along_lag_mm_p95_positive"]),
                "cross_track_error_p95_mm": float(drag["cross_track_error_p95_mm"]),
                "score": float(score),
            }
        )
    rows.sort(key=lambda row: (row["score"], row["along_lag_ms_p95_positive"]))
    return rows


def _centered_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0 or left.size != right.size:
        return 0.0
    left_centered = np.asarray(left, dtype=float) - float(np.mean(left))
    right_centered = np.asarray(right, dtype=float) - float(np.mean(right))
    left_norm = float(np.linalg.norm(left_centered))
    right_norm = float(np.linalg.norm(right_centered))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return float(np.dot(left_centered, right_centered) / (left_norm * right_norm))


def _summarize_overall_filters(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    grouped = (
        frame.groupby(["algorithm", "params"], dropna=False)
        .agg(
            segment_count=("case_id", "count"),
            mean_score=("score", "mean"),
            worst_score=("score", "max"),
            mean_static_z_std_ratio=("static_mean_z_std_ratio", "mean"),
            worst_static_z_std_ratio=("static_mean_z_std_ratio", "max"),
            mean_motion_neighbor_ratio=("motion_neighbor_z_ratio_to_raw", "mean"),
            mean_motion_power10_ratio=("motion_power_above_10hz_ratio_to_raw", "mean"),
            mean_motion_range_ratio=("motion_z_range_ratio_to_raw", "mean"),
            mean_motion_corr=("motion_z_corr_to_raw", "mean"),
            mean_lag_p95_ms=("along_lag_ms_p95_positive", "mean"),
            worst_lag_p95_ms=("along_lag_ms_p95_positive", "max"),
        )
        .reset_index()
    )
    grouped["robust_score"] = 0.6 * grouped["mean_score"] + 0.4 * grouped["worst_score"]
    grouped = grouped.sort_values(["robust_score", "worst_score", "mean_lag_p95_ms"])
    return grouped.to_dict(orient="records")


def _write_segment_report(path: Path, metadata: dict[str, Any], metrics: pd.DataFrame) -> None:
    best_compromise = metrics.head(8)
    best_static = metrics.sort_values(["static_mean_z_std_ratio", "along_lag_ms_p95_positive"]).head(8)
    best_shape = metrics.sort_values(
        ["motion_z_corr_to_raw", "motion_z_range_ratio_to_raw", "along_lag_ms_p95_positive"],
        ascending=[False, False, True],
    ).head(8)

    motion = metadata["motion_summary"]
    pre = metadata["pre_static_summary"]
    post = metadata["post_static_summary"]
    lines = [
        f"# {metadata['case_id']} Report",
        "",
        "## Segment",
        "",
        f"- Source: `{metadata['source_path']}`",
        f"- Source rows: {metadata['source_start_row']} .. {metadata['source_stop_row_exclusive'] - 1}",
        f"- Samples: {metadata['sample_count']}",
        f"- Duration: {metadata['duration_s']:.6g} s",
        f"- Median sample rate: {metadata['sample_rate_hz_median']:.6g} Hz",
        f"- Motion relative rows: {metadata['motion_start_relative']} .. {metadata['motion_stop_relative'] - 1}",
        f"- Pre static / motion / post static: "
        f"{metadata['pre_static_duration_s']:.2f} / {metadata['motion_duration_s']:.2f} / {metadata['post_static_duration_s']:.2f} s",
        "",
        "## Raw Fluctuation Summary",
        "",
        f"- Pre static Z std: {pre['z_std_mm']:.6g} mm; XY mean std: {pre['xy_mean_std_mm']:.6g} mm; "
        f"Z/XY: {pre['z_over_xy_mean_std_ratio']:.6g}",
        f"- Post static Z std: {post['z_std_mm']:.6g} mm; XY mean std: {post['xy_mean_std_mm']:.6g} mm; "
        f"Z/XY: {post['z_over_xy_mean_std_ratio']:.6g}",
        f"- Motion dominant Z frequency: {motion['dominant_frequency_hz']:.6g} Hz; "
        f"spectral centroid: {motion['spectral_centroid_hz']:.6g} Hz",
        f"- Motion power fraction above 5 / 10 Hz: {motion['power_fraction_above_5hz']:.6g} / "
        f"{motion['power_fraction_above_10hz']:.6g}",
        f"- Motion peaks per second (raw Z, descriptive only): {motion['peaks_per_second']:.6g}",
        "",
        "## Best Compromise",
        "",
        *_metrics_table(best_compromise),
        "",
        "## Strongest Static-Window Z Denoise",
        "",
        *_metrics_table(best_static),
        "",
        "## Best Motion Shape Preservation",
        "",
        *_metrics_table(best_shape),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_overall_report(path: Path, segments: pd.DataFrame, summary: pd.DataFrame) -> None:
    if segments.empty or summary.empty:
        path.write_text("# Case 20-2x Overall Report\n\nNo segments or no metrics generated.\n", encoding="utf-8")
        return
    lines = [
        "# Case 20-2x Overall Report",
        "",
        "## Segments",
        "",
        f"- Segment count: {len(segments)}",
        f"- Cases: {', '.join(segments['case_id'].astype(str))}",
        "",
        "## Motion Segment Summary",
        "",
        "| case | duration s | motion s | source rows |",
        "| --- | ---: | ---: | --- |",
    ]
    for _, row in segments.iterrows():
        lines.append(
            f"| {row['case_id']} | {row['duration_s']:.6g} | {row['motion_duration_s']:.6g} | "
            f"{int(row['source_start_row'])}..{int(row['source_stop_row_exclusive']) - 1} |"
        )

    lines.extend(
        [
            "",
            "## Best Overall Filter Candidates",
            "",
            "| algorithm | params | robust | mean static z ratio | mean motion neighbor z ratio | mean motion corr | mean lag p95 ms |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in summary.head(12).iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["algorithm"]),
                    f"`{row['params']}`",
                    f"{row['robust_score']:.6g}",
                    f"{row['mean_static_z_std_ratio']:.6g}",
                    f"{row['mean_motion_neighbor_ratio']:.6g}",
                    f"{row['mean_motion_corr']:.6g}",
                    f"{row['mean_lag_p95_ms']:.6g}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _metrics_table(frame: pd.DataFrame) -> list[str]:
    lines = [
        "| algorithm | params | static z ratio | static neighbor z ratio | motion neighbor z ratio | motion range ratio | motion corr | lag p95 ms | score |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in frame.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["algorithm"]),
                    f"`{row['params']}`",
                    f"{row['static_mean_z_std_ratio']:.6g}",
                    f"{row['static_mean_neighbor_z_ratio']:.6g}",
                    f"{row['motion_neighbor_z_ratio_to_raw']:.6g}",
                    f"{row['motion_z_range_ratio_to_raw']:.6g}",
                    f"{row['motion_z_corr_to_raw']:.6g}",
                    f"{row['along_lag_ms_p95_positive']:.6g}",
                    f"{row['score']:.6g}",
                ]
            )
            + " |"
        )
    return lines


if __name__ == "__main__":
    main()
