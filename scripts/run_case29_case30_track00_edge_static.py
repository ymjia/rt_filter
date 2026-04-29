from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in (ROOT, SCRIPT_DIR):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

from rt_filter.io import write_trajectory
from rt_filter.trajectory import Trajectory

import run_case16_hotel_analysis as static_base
import run_case20_move_track00_analysis as motion_base
import run_local_lab_case12 as manifest_base


SOURCE_PATH = Path("ref_data/move/track_data_00.txt")
INPUT_ROOT = Path("input/sn")
OUTPUT_ROOT = Path("outputs/case29_case30_track_data_00_edge_static")

EDGE_TRIM_FRAMES = 15


def main() -> None:
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run() -> dict[str, Any]:
    frame = motion_base._read_source_frame()
    motion_segments = motion_base._detect_segments(frame)
    if len(motion_segments) < 2:
        raise ValueError(f"expected multiple motion segments in {SOURCE_PATH}")

    holds = _detect_edge_static_holds(frame, motion_segments)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for hold in holds:
        result = _write_hold_case(hold)
        manifest_rows.append(result["manifest_row"])
        summary_rows.append(result["summary_row"])

    summary_csv = OUTPUT_ROOT / "edge_static_summary.csv"
    report_md = OUTPUT_ROOT / "edge_static_report.md"
    manifest_json = OUTPUT_ROOT / "edge_static_manifest.json"
    manifest_base._write_csv(summary_csv, summary_rows)
    manifest_json.write_text(json.dumps(manifest_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(report_md, pd.DataFrame(summary_rows))

    return {
        "case_count": len(manifest_rows),
        "cases": [row["case_id"] for row in manifest_rows],
        "summary_csv": str(summary_csv),
        "report_md": str(report_md),
        "manifest_json": str(manifest_json),
    }


def _detect_edge_static_holds(
    frame: pd.DataFrame,
    motion_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    first_motion = motion_segments[0]
    last_motion = motion_segments[-1]

    case30_raw_start = 0
    case30_raw_stop = int(first_motion["motion_start_row"])
    case30_start = case30_raw_start
    case30_stop = case30_raw_stop - EDGE_TRIM_FRAMES

    case29_raw_start = int(last_motion["motion_stop_row_exclusive"])
    case29_raw_stop = len(frame)
    case29_start = case29_raw_start + EDGE_TRIM_FRAMES
    case29_stop = case29_raw_stop

    holds: list[dict[str, Any]] = []
    if case29_stop - case29_start >= 5:
        holds.append(
            {
                "case_number": 29,
                "case_id": "case_29_static_robot_arm_track_data_00_after_motion_05",
                "trajectory_name": "case_29_static_robot_arm_track_data_00_after_motion_05_01_track_data_00",
                "variant_id": "after_motion_05",
                "variant_note": (
                    f"track_data_00 static hold after motion segment 5, trimmed "
                    f"{EDGE_TRIM_FRAMES} frames at the motion-adjacent head"
                ),
                "edge_role": "after_motion_05",
                "nearest_motion_segment": 5,
                "source_start_row": case29_start,
                "source_stop_row_exclusive": case29_stop,
                "source_start_row_untrimmed": case29_raw_start,
                "source_stop_row_exclusive_untrimmed": case29_raw_stop,
                "trimmed_head_frames": EDGE_TRIM_FRAMES,
                "trimmed_tail_frames": 0,
                "raw_interval_frame_count": case29_raw_stop - case29_raw_start,
                "trimmed_frame_count": case29_stop - case29_start,
                "frame": frame.iloc[case29_start:case29_stop].reset_index(drop=True),
            }
        )

    if case30_stop - case30_start >= 5:
        holds.append(
            {
                "case_number": 30,
                "case_id": "case_30_static_robot_arm_track_data_00_before_motion_01",
                "trajectory_name": "case_30_static_robot_arm_track_data_00_before_motion_01_01_track_data_00",
                "variant_id": "before_motion_01",
                "variant_note": (
                    f"track_data_00 static hold before motion segment 1, trimmed "
                    f"{EDGE_TRIM_FRAMES} frames at the motion-adjacent tail"
                ),
                "edge_role": "before_motion_01",
                "nearest_motion_segment": 1,
                "source_start_row": case30_start,
                "source_stop_row_exclusive": case30_stop,
                "source_start_row_untrimmed": case30_raw_start,
                "source_stop_row_exclusive_untrimmed": case30_raw_stop,
                "trimmed_head_frames": 0,
                "trimmed_tail_frames": EDGE_TRIM_FRAMES,
                "raw_interval_frame_count": case30_raw_stop - case30_raw_start,
                "trimmed_frame_count": case30_stop - case30_start,
                "frame": frame.iloc[case30_start:case30_stop].reset_index(drop=True),
            }
        )

    return holds


def _write_hold_case(hold: dict[str, Any]) -> dict[str, Any]:
    case_id = str(hold["case_id"])
    trajectory_name = str(hold["trajectory_name"])
    case_dir = INPUT_ROOT / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    trajectory = motion_base._trajectory_from_frame(
        hold["frame"],
        name=trajectory_name,
        timestamp_origin="edge_hold",
    )
    metadata = _metadata(case_id=case_id, trajectory_name=trajectory_name, hold=hold, trajectory=trajectory)

    input_csv = case_dir / f"{trajectory_name}.csv"
    metadata_json = case_dir / f"{trajectory_name}_metadata.json"
    write_trajectory(trajectory, input_csv)
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_row = {
        "trajectory": trajectory_name,
        "case_id": case_id,
        "case_label": "static_robot_arm",
        "source_folder": "ref_data/move",
        "source_file": SOURCE_PATH.name,
        "source_path": str(SOURCE_PATH),
        "input_csv": str(input_csv),
        "source_workbook": "",
        "base": "move",
        "target": "robot_arm",
        "calibration": "unknown_calibration",
        "exposure": "default_exposure",
        "added_points": False,
        "variant_id": str(hold["variant_id"]),
        "tracker_smoothing": "unknown",
        "variant_note": str(hold["variant_note"]),
        "sample_count": trajectory.count,
        "raw_row_count": int(hold["trimmed_frame_count"]),
        "position_unit": "mm",
        "rotation_source": "xr,yr,zr",
        "rotation_assumption": "XYZ Euler angles in degrees",
        "timestamp_source": "edge-hold-normalized source timestamp ticks / 10000 or generated from rate",
        "sample_rate_hz": static_base._median_sample_rate(trajectory.timestamps),
        "reference_csv": "",
        "metadata_json": str(metadata_json),
    }
    manifest_base._upsert_manifest_row(INPUT_ROOT / "manifest.csv", manifest_row, key="trajectory")

    summary = metadata["static_noise_summary"]
    neighbor10 = summary["neighbor_mean_windows"]["10"]
    summary_row = {
        "case_id": case_id,
        "trajectory": trajectory_name,
        "edge_role": hold["edge_role"],
        "nearest_motion_segment": int(hold["nearest_motion_segment"]),
        "source_rows": f"{hold['source_start_row']}..{hold['source_stop_row_exclusive'] - 1}",
        "sample_count": trajectory.count,
        "duration_s": metadata["duration_s"],
        "x_std_mm": metadata["position_std_mm"]["x"],
        "y_std_mm": metadata["position_std_mm"]["y"],
        "z_std_mm": metadata["position_std_mm"]["z"],
        "x_range_mm": metadata["position_range_mm"]["x"],
        "y_range_mm": metadata["position_range_mm"]["y"],
        "z_range_mm": metadata["position_range_mm"]["z"],
        "xy_mean_std_mm": float(
            np.mean([metadata["position_std_mm"]["x"], metadata["position_std_mm"]["y"]])
        ),
        "z_over_xy_mean_std_ratio": summary["z_over_xy_mean_std_ratio"],
        "neighbor10_x_rms_mm": neighbor10["x_rms_mm"],
        "neighbor10_y_rms_mm": neighbor10["y_rms_mm"],
        "neighbor10_z_rms_mm": neighbor10["z_rms_mm"],
        "neighbor10_z_over_xy_mean_rms": neighbor10["z_over_xy_mean_rms"],
        "dominant_frequency_hz": summary["z_spectrum"]["dominant_frequency_hz"],
        "spectral_centroid_hz": summary["z_spectrum"]["spectral_centroid_hz"],
        "power_fraction_above_5hz": summary["z_spectrum"]["power_fraction_above_5hz"],
        "power_fraction_above_10hz": summary["z_spectrum"]["power_fraction_above_10hz"],
    }
    return {
        "manifest_row": manifest_row,
        "summary_row": summary_row,
    }


def _metadata(
    *,
    case_id: str,
    trajectory_name: str,
    hold: dict[str, Any],
    trajectory: Trajectory,
) -> dict[str, Any]:
    positions = trajectory.positions
    timestamps = trajectory.timestamps
    assert timestamps is not None

    frame = hold["frame"]
    dt = np.diff(timestamps)
    valid_dt = dt[dt > 0.0]
    euler = frame[["xr", "yr", "zr"]].to_numpy(dtype=float)

    return {
        "case_id": case_id,
        "trajectory": trajectory_name,
        "source_path": str(SOURCE_PATH),
        "source_columns": list(frame.columns),
        "raw_row_count": int(hold["trimmed_frame_count"]),
        "sample_count": trajectory.count,
        "position_unit": "mm",
        "rotation_assumption": "XYZ Euler angles in degrees",
        "timestamp_ticks_per_second": motion_base.TIMESTAMP_TICKS_PER_SECOND,
        "duration_s": float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0,
        "sample_rate_hz_median": static_base._median_sample_rate(timestamps),
        "source_rate_unique": sorted(float(value) for value in frame["rate"].dropna().unique()),
        "dt_s_min": float(valid_dt.min()) if valid_dt.size else 0.0,
        "dt_s_median": float(np.median(valid_dt)) if valid_dt.size else 0.0,
        "dt_s_max": float(valid_dt.max()) if valid_dt.size else 0.0,
        "position_min_mm": static_base._axis_dict(positions.min(axis=0)),
        "position_max_mm": static_base._axis_dict(positions.max(axis=0)),
        "position_range_mm": static_base._axis_dict(np.ptp(positions, axis=0)),
        "position_std_mm": static_base._axis_dict(positions.std(axis=0, ddof=0)),
        "euler_range_deg": static_base._axis_dict(np.ptp(euler, axis=0), axes=("xr", "yr", "zr")),
        "continuous_step_speed_mm_s": static_base._step_speed_stats(positions, timestamps),
        "static_noise_summary": static_base._static_noise_summary(positions),
        "edge_role": str(hold["edge_role"]),
        "nearest_motion_segment": int(hold["nearest_motion_segment"]),
        "source_start_row": int(hold["source_start_row"]),
        "source_stop_row_exclusive": int(hold["source_stop_row_exclusive"]),
        "source_start_row_untrimmed": int(hold["source_start_row_untrimmed"]),
        "source_stop_row_exclusive_untrimmed": int(hold["source_stop_row_exclusive_untrimmed"]),
        "trimmed_head_frames": int(hold["trimmed_head_frames"]),
        "trimmed_tail_frames": int(hold["trimmed_tail_frames"]),
        "raw_interval_frame_count": int(hold["raw_interval_frame_count"]),
        "trimmed_frame_count": int(hold["trimmed_frame_count"]),
    }


def _write_report(path: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# Case 29-30 Edge Static Holds Report",
        "",
        f"- Source: `{SOURCE_PATH}`",
        f"- Motion-adjacent trim: {EDGE_TRIM_FRAMES} frames",
        f"- Hold count: {len(summary)}",
        "",
        "| case | role | rows | duration s | x std mm | y std mm | z std mm | x range mm | y range mm | z range mm | z/xy std |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['case_id']} | {row['edge_role']} | {row['source_rows']} | {row['duration_s']:.6g} | "
            f"{row['x_std_mm']:.6g} | {row['y_std_mm']:.6g} | {row['z_std_mm']:.6g} | "
            f"{row['x_range_mm']:.6g} | {row['y_range_mm']:.6g} | {row['z_range_mm']:.6g} | "
            f"{row['z_over_xy_mean_std_ratio']:.6g} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
