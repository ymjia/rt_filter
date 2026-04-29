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

from rt_filter.batch import parameter_slug
from rt_filter.filters import run_filter
from rt_filter.io import write_trajectory
from rt_filter.trajectory import Trajectory

import run_case20_move_track00_analysis as segment_base
import run_local_lab_case12 as motion_base


SOURCE_PATH = Path("ref_data/move/track_data_00.txt")
INPUT_ROOT = Path("input/sn")
OUTPUT_ROOT = Path("outputs/case19_track_data_00_full")

CASE_ID = "case_19_dynamic_robot_arm_track_data_00_full"
CASE_LABEL = "dynamic_robot_arm_full"
TRAJECTORY = f"{CASE_ID}_01_track_data_00"
CASE_VARIANT_ID = "track_data_00_full"
CASE_VARIANT_NOTE = "full track_data_00 robot arm motion trajectory with static intervals"
REPORT_TITLE = "Case 19 Full Track Data 00 Motion Report"

FILTER_SPECS = list(segment_base.FILTER_SPECS)


def main() -> None:
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run() -> dict[str, Any]:
    frame = segment_base._read_source_frame()
    segments = segment_base._detect_segments(frame)
    if not segments:
        raise ValueError(f"no motion segments detected in {SOURCE_PATH}")

    trajectory = segment_base._trajectory_from_frame(
        frame,
        name=TRAJECTORY,
        timestamp_origin="full",
    )
    metadata = _metadata(trajectory, segments)

    case_dir = INPUT_ROOT / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    input_csv = case_dir / f"{TRAJECTORY}.csv"
    metadata_json = case_dir / f"{TRAJECTORY}_metadata.json"
    write_trajectory(trajectory, input_csv)
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_row = {
        "trajectory": TRAJECTORY,
        "case_id": CASE_ID,
        "case_label": CASE_LABEL,
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
        "variant_id": CASE_VARIANT_ID,
        "tracker_smoothing": "unknown",
        "variant_note": CASE_VARIANT_NOTE,
        "sample_count": trajectory.count,
        "raw_row_count": int(len(frame)),
        "position_unit": "mm",
        "rotation_source": "xr,yr,zr",
        "rotation_assumption": "XYZ Euler angles in degrees",
        "timestamp_source": "full-normalized source timestamp ticks / 10000 or generated from rate",
        "sample_rate_hz": motion_base._median_sample_rate(trajectory.timestamps),
        "reference_csv": "",
        "metadata_json": str(metadata_json),
    }
    motion_base._upsert_manifest_row(INPUT_ROOT / "manifest.csv", manifest_row, key="trajectory")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    segment_metrics, summary_rows = _run_filter_analysis(trajectory, segments)

    segment_metrics_csv = OUTPUT_ROOT / f"{CASE_ID}_segment_filter_metrics.csv"
    summary_csv = OUTPUT_ROOT / f"{CASE_ID}_filter_summary.csv"
    report_md = OUTPUT_ROOT / f"{CASE_ID}_report.md"
    motion_base._write_csv(segment_metrics_csv, segment_metrics)
    motion_base._write_csv(summary_csv, summary_rows)
    _write_report(
        report_md,
        metadata=metadata,
        segment_metrics=pd.DataFrame(segment_metrics),
        filter_summary=pd.DataFrame(summary_rows),
    )

    return {
        "input_csv": str(input_csv),
        "metadata_json": str(metadata_json),
        "segment_metrics_csv": str(segment_metrics_csv),
        "filter_summary_csv": str(summary_csv),
        "report_md": str(report_md),
        "motion_segment_count": int(metadata["motion_segment_count"]),
        "filter_count": len(FILTER_SPECS),
    }


def _metadata(trajectory: Trajectory, segments: list[dict[str, Any]]) -> dict[str, Any]:
    positions = trajectory.positions
    timestamps = trajectory.timestamps
    assert timestamps is not None

    dt = np.diff(timestamps)
    valid_dt = dt[dt > 0.0]
    sample_rate_hz = motion_base._median_sample_rate(timestamps)

    segment_rows: list[dict[str, Any]] = []
    motion_ranges: list[float] = []
    motion_frequencies: list[float] = []
    motion_peak_rates: list[float] = []
    static_z_std_values: list[float] = []
    static_xy_std_values: list[float] = []
    static_z_xy_ratios: list[float] = []
    motion_mask = np.zeros(trajectory.count, dtype=bool)

    for segment_index, segment in enumerate(segments, start=1):
        segment_start = int(segment["source_start_row"])
        segment_stop = int(segment["source_stop_row_exclusive"])
        motion_start = int(segment["motion_start_relative"])
        motion_stop = int(segment["motion_stop_relative"])
        motion_mask[int(segment["motion_start_row"]) : int(segment["motion_stop_row_exclusive"])] = True

        segment_positions = positions[segment_start:segment_stop]
        segment_timestamps = timestamps[segment_start:segment_stop]
        pre_summary = segment_base._window_summary(
            segment_positions[:motion_start],
            segment_timestamps[:motion_start],
            segment_base.NEIGHBOR_WINDOW_STATIC,
        )
        motion_summary = segment_base._window_summary(
            segment_positions[motion_start:motion_stop],
            segment_timestamps[motion_start:motion_stop],
            segment_base.NEIGHBOR_WINDOW_MOTION,
        )
        post_summary = segment_base._window_summary(
            segment_positions[motion_stop:],
            segment_timestamps[motion_stop:],
            segment_base.NEIGHBOR_WINDOW_STATIC,
        )

        static_z_std_values.extend([pre_summary["z_std_mm"], post_summary["z_std_mm"]])
        static_xy_std_values.extend(
            [pre_summary["xy_mean_std_mm"], post_summary["xy_mean_std_mm"]]
        )
        static_z_xy_ratios.extend(
            [
                pre_summary["z_over_xy_mean_std_ratio"],
                post_summary["z_over_xy_mean_std_ratio"],
            ]
        )
        motion_ranges.append(motion_summary["z_range_mm"])
        motion_frequencies.append(motion_summary["dominant_frequency_hz"])
        motion_peak_rates.append(motion_summary["peaks_per_second"])

        segment_rows.append(
            {
                "segment_index": segment_index,
                "source_start_row": segment_start,
                "source_stop_row_exclusive": segment_stop,
                "motion_start_row": int(segment["motion_start_row"]),
                "motion_stop_row_exclusive": int(segment["motion_stop_row_exclusive"]),
                "motion_start_relative": motion_start,
                "motion_stop_relative": motion_stop,
                "duration_s": float(segment_timestamps[-1] - segment_timestamps[0])
                if len(segment_timestamps) > 1
                else 0.0,
                "pre_static_duration_s": float(motion_start / max(sample_rate_hz, 1e-12)),
                "motion_duration_s": float(
                    max(motion_stop - motion_start - 1, 0) / max(sample_rate_hz, 1e-12)
                ),
                "post_static_duration_s": float(
                    max(segment_stop - segment_start - motion_stop, 0) / max(sample_rate_hz, 1e-12)
                ),
                "pre_static_summary": pre_summary,
                "motion_summary": motion_summary,
                "post_static_summary": post_summary,
            }
        )

    total_motion_duration_s = float(
        sum(float(row["motion_duration_s"]) for row in segment_rows)
    )
    duration_s = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0

    return {
        "case_id": CASE_ID,
        "trajectory": TRAJECTORY,
        "source_path": str(SOURCE_PATH),
        "raw_row_count": int(trajectory.count),
        "sample_count": trajectory.count,
        "position_unit": "mm",
        "rotation_assumption": "XYZ Euler angles in degrees",
        "timestamp_ticks_per_second": segment_base.TIMESTAMP_TICKS_PER_SECOND,
        "duration_s": duration_s,
        "sample_rate_hz_median": sample_rate_hz,
        "dt_s_min": float(valid_dt.min()) if valid_dt.size else 0.0,
        "dt_s_median": float(np.median(valid_dt)) if valid_dt.size else 0.0,
        "dt_s_max": float(valid_dt.max()) if valid_dt.size else 0.0,
        "position_min_mm": motion_base._axis_dict(positions.min(axis=0)),
        "position_max_mm": motion_base._axis_dict(positions.max(axis=0)),
        "position_range_mm": motion_base._axis_dict(np.ptp(positions, axis=0)),
        "position_std_mm": motion_base._axis_dict(positions.std(axis=0, ddof=0)),
        "continuous_step_speed_mm_s": motion_base._step_speed_stats(positions, timestamps),
        "motion_segment_count": len(segment_rows),
        "motion_segment_rows": segment_rows,
        "motion_detection": {
            "speed_threshold_mm_s": segment_base.MOTION_SPEED_THRESHOLD_MM_S,
            "smooth_window_frames": segment_base.MOTION_SMOOTH_WINDOW_FRAMES,
            "min_motion_duration_s": segment_base.MIN_MOTION_DURATION_S,
            "static_buffer_s": segment_base.STATIC_BUFFER_S,
        },
        "motion_total_duration_s": total_motion_duration_s,
        "motion_duration_fraction": motion_base._safe_ratio(total_motion_duration_s, duration_s),
        "motion_sample_count": int(np.count_nonzero(motion_mask)),
        "static_sample_count": int(trajectory.count - np.count_nonzero(motion_mask)),
        "raw_segment_aggregate": {
            "mean_static_z_std_mm": float(np.mean(static_z_std_values)) if static_z_std_values else 0.0,
            "mean_static_xy_mean_std_mm": float(np.mean(static_xy_std_values))
            if static_xy_std_values
            else 0.0,
            "mean_static_z_over_xy_ratio": float(np.mean(static_z_xy_ratios))
            if static_z_xy_ratios
            else 0.0,
            "mean_motion_z_range_mm": float(np.mean(motion_ranges)) if motion_ranges else 0.0,
            "mean_motion_dominant_frequency_hz": float(np.mean(motion_frequencies))
            if motion_frequencies
            else 0.0,
            "mean_motion_peaks_per_second": float(np.mean(motion_peak_rates))
            if motion_peak_rates
            else 0.0,
        },
    }


def _run_filter_analysis(
    raw: Trajectory,
    segments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    segment_rows: list[dict[str, Any]] = []

    for spec in FILTER_SPECS:
        algorithm = str(spec["algorithm"])
        params = dict(spec["params"])
        filtered = raw if algorithm == "raw" else run_filter(algorithm, raw, params)

        slug = parameter_slug(params)
        filtered_csv = OUTPUT_ROOT / f"{algorithm}__{slug}.csv"
        write_trajectory(filtered, filtered_csv)

        for segment_index, segment in enumerate(segments, start=1):
            segment_rows.append(
                _evaluate_filter_segment(
                    raw=raw,
                    filtered=filtered,
                    segment=segment,
                    segment_index=segment_index,
                    algorithm=algorithm,
                    params=params,
                    filtered_csv=filtered_csv,
                )
            )

    segment_rows.sort(
        key=lambda row: (
            row["algorithm"],
            row["params"],
            row["segment_index"],
        )
    )
    summary_rows = _summarize_filters(pd.DataFrame(segment_rows))
    return segment_rows, summary_rows


def _evaluate_filter_segment(
    *,
    raw: Trajectory,
    filtered: Trajectory,
    segment: dict[str, Any],
    segment_index: int,
    algorithm: str,
    params: dict[str, Any],
    filtered_csv: Path,
) -> dict[str, Any]:
    segment_start = int(segment["source_start_row"])
    segment_stop = int(segment["source_stop_row_exclusive"])
    motion_start = int(segment["motion_start_relative"])
    motion_stop = int(segment["motion_stop_relative"])

    raw_segment = _slice_trajectory(
        raw,
        start=segment_start,
        stop=segment_stop,
        name=f"{raw.name}_segment_{segment_index:02d}",
    )
    filtered_segment = _slice_trajectory(
        filtered,
        start=segment_start,
        stop=segment_stop,
        name=f"{filtered.name}_segment_{segment_index:02d}",
    )
    raw_positions = raw_segment.positions
    filtered_positions = filtered_segment.positions
    timestamps = raw_segment.timestamps
    assert timestamps is not None

    raw_pre = segment_base._window_summary(
        raw_positions[:motion_start],
        timestamps[:motion_start],
        segment_base.NEIGHBOR_WINDOW_STATIC,
    )
    raw_motion = segment_base._window_summary(
        raw_positions[motion_start:motion_stop],
        timestamps[motion_start:motion_stop],
        segment_base.NEIGHBOR_WINDOW_MOTION,
    )
    raw_post = segment_base._window_summary(
        raw_positions[motion_stop:],
        timestamps[motion_stop:],
        segment_base.NEIGHBOR_WINDOW_STATIC,
    )
    filtered_pre = segment_base._window_summary(
        filtered_positions[:motion_start],
        timestamps[:motion_start],
        segment_base.NEIGHBOR_WINDOW_STATIC,
    )
    filtered_motion = segment_base._window_summary(
        filtered_positions[motion_start:motion_stop],
        timestamps[motion_start:motion_stop],
        segment_base.NEIGHBOR_WINDOW_MOTION,
    )
    filtered_post = segment_base._window_summary(
        filtered_positions[motion_stop:],
        timestamps[motion_stop:],
        segment_base.NEIGHBOR_WINDOW_STATIC,
    )
    drag = motion_base._drag_metrics(raw_segment, filtered_segment)

    static_z_std_ratio = np.mean(
        [
            motion_base._safe_ratio(filtered_pre["z_std_mm"], raw_pre["z_std_mm"]),
            motion_base._safe_ratio(filtered_post["z_std_mm"], raw_post["z_std_mm"]),
        ]
    )
    static_neighbor_ratio = np.mean(
        [
            motion_base._safe_ratio(
                filtered_pre["neighbor_z_rms_mm"],
                raw_pre["neighbor_z_rms_mm"],
            ),
            motion_base._safe_ratio(
                filtered_post["neighbor_z_rms_mm"],
                raw_post["neighbor_z_rms_mm"],
            ),
        ]
    )
    range_ratio = motion_base._safe_ratio(
        filtered_motion["z_range_mm"],
        raw_motion["z_range_mm"],
    )
    motion_neighbor_ratio = motion_base._safe_ratio(
        filtered_motion["neighbor_z_rms_mm"],
        raw_motion["neighbor_z_rms_mm"],
    )
    motion_power5_ratio = motion_base._safe_ratio(
        filtered_motion["power_fraction_above_5hz"],
        raw_motion["power_fraction_above_5hz"],
    )
    motion_power10_ratio = motion_base._safe_ratio(
        filtered_motion["power_fraction_above_10hz"],
        raw_motion["power_fraction_above_10hz"],
    )
    motion_peak_ratio = motion_base._safe_ratio(
        filtered_motion["peaks_per_second"],
        raw_motion["peaks_per_second"],
    )
    motion_dom_shift = abs(
        filtered_motion["dominant_frequency_hz"] - raw_motion["dominant_frequency_hz"]
    )
    raw_motion_z = raw_positions[motion_start:motion_stop, 2]
    filtered_motion_z = filtered_positions[motion_start:motion_stop, 2]
    corr = segment_base._centered_correlation(filtered_motion_z, raw_motion_z)
    motion_z_rmse = float(np.sqrt(np.mean((filtered_motion_z - raw_motion_z) ** 2)))
    score = (
        0.30 * static_z_std_ratio
        + 0.20 * static_neighbor_ratio
        + 0.20 * motion_neighbor_ratio
        + 0.15 * abs(range_ratio - 1.0)
        + 0.10 * min(float(drag["along_lag_ms_p95_positive"]) / 20.0, 3.0)
        + 0.05
        * motion_base._safe_ratio(
            motion_dom_shift,
            max(raw_motion["dominant_frequency_hz"], 1.0),
        )
    )

    return {
        "case_id": CASE_ID,
        "trajectory": TRAJECTORY,
        "segment_index": segment_index,
        "source_start_row": segment_start,
        "source_stop_row_exclusive": segment_stop,
        "motion_start_row": int(segment["motion_start_row"]),
        "motion_stop_row_exclusive": int(segment["motion_stop_row_exclusive"]),
        "algorithm": algorithm,
        "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
        "filtered_csv": str(filtered_csv),
        "static_pre_z_std_ratio": motion_base._safe_ratio(
            filtered_pre["z_std_mm"], raw_pre["z_std_mm"]
        ),
        "static_post_z_std_ratio": motion_base._safe_ratio(
            filtered_post["z_std_mm"], raw_post["z_std_mm"]
        ),
        "static_mean_z_std_ratio": float(static_z_std_ratio),
        "static_mean_neighbor_z_ratio": float(static_neighbor_ratio),
        "motion_z_range_ratio_to_raw": float(range_ratio),
        "motion_neighbor_z_ratio_to_raw": float(motion_neighbor_ratio),
        "motion_power_above_5hz_ratio_to_raw": float(motion_power5_ratio),
        "motion_power_above_10hz_ratio_to_raw": float(motion_power10_ratio),
        "motion_peaks_per_second_ratio_to_raw": float(motion_peak_ratio),
        "motion_z_corr_to_raw": float(corr),
        "motion_dominant_frequency_hz": float(filtered_motion["dominant_frequency_hz"]),
        "raw_motion_dominant_frequency_hz": float(raw_motion["dominant_frequency_hz"]),
        "motion_dominant_frequency_shift_hz": float(motion_dom_shift),
        "motion_peaks_per_second": float(filtered_motion["peaks_per_second"]),
        "raw_motion_peaks_per_second": float(raw_motion["peaks_per_second"]),
        "motion_z_rmse_to_raw_mm": float(motion_z_rmse),
        "to_raw_translation_rmse_mm": float(
            np.sqrt(np.mean(np.sum((filtered_positions - raw_positions) ** 2, axis=1)))
        ),
        "along_lag_ms_p95_positive": float(drag["along_lag_ms_p95_positive"]),
        "along_lag_mm_p95_positive": float(drag["along_lag_mm_p95_positive"]),
        "cross_track_error_p95_mm": float(drag["cross_track_error_p95_mm"]),
        "score": float(score),
    }


def _slice_trajectory(
    trajectory: Trajectory,
    *,
    start: int,
    stop: int,
    name: str,
) -> Trajectory:
    timestamps = None
    if trajectory.timestamps is not None:
        timestamps = trajectory.timestamps[start:stop].copy()
    return Trajectory(
        poses=trajectory.poses[start:stop].copy(),
        timestamps=timestamps,
        name=name,
        metadata=dict(trajectory.metadata),
    )


def _summarize_filters(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []

    grouped = (
        frame.groupby(["algorithm", "params", "filtered_csv"], dropna=False)
        .agg(
            segment_count=("segment_index", "count"),
            mean_score=("score", "mean"),
            worst_score=("score", "max"),
            mean_static_z_std_ratio=("static_mean_z_std_ratio", "mean"),
            worst_static_z_std_ratio=("static_mean_z_std_ratio", "max"),
            mean_static_neighbor_z_ratio=("static_mean_neighbor_z_ratio", "mean"),
            mean_motion_neighbor_ratio=("motion_neighbor_z_ratio_to_raw", "mean"),
            mean_motion_power10_ratio=("motion_power_above_10hz_ratio_to_raw", "mean"),
            mean_motion_range_ratio=("motion_z_range_ratio_to_raw", "mean"),
            mean_motion_peak_ratio=("motion_peaks_per_second_ratio_to_raw", "mean"),
            mean_motion_corr=("motion_z_corr_to_raw", "mean"),
            mean_motion_dom_shift_hz=("motion_dominant_frequency_shift_hz", "mean"),
            worst_motion_dom_shift_hz=("motion_dominant_frequency_shift_hz", "max"),
            mean_lag_p95_ms=("along_lag_ms_p95_positive", "mean"),
            worst_lag_p95_ms=("along_lag_ms_p95_positive", "max"),
            mean_cross_track_p95_mm=("cross_track_error_p95_mm", "mean"),
            mean_translation_rmse_mm=("to_raw_translation_rmse_mm", "mean"),
        )
        .reset_index()
    )
    grouped["robust_score"] = 0.6 * grouped["mean_score"] + 0.4 * grouped["worst_score"]
    grouped = grouped.sort_values(["robust_score", "worst_score", "mean_lag_p95_ms"])
    return grouped.to_dict(orient="records")


def _write_report(
    path: Path,
    *,
    metadata: dict[str, Any],
    segment_metrics: pd.DataFrame,
    filter_summary: pd.DataFrame,
) -> None:
    raw_aggregate = metadata["raw_segment_aggregate"]
    top_filters = filter_summary.head(12) if not filter_summary.empty else filter_summary

    lines = [
        f"# {REPORT_TITLE}",
        "",
        "## Data",
        "",
        f"- Input: `input/sn/{CASE_ID}/{TRAJECTORY}.csv`",
        f"- Source: `{metadata['source_path']}`",
        f"- Samples: {metadata['sample_count']}",
        f"- Duration: {metadata['duration_s']:.6g} s",
        f"- Median sample rate: {metadata['sample_rate_hz_median']:.6g} Hz",
        f"- Detected motion segments: {metadata['motion_segment_count']}",
        f"- Total motion duration: {metadata['motion_total_duration_s']:.6g} s "
        f"({metadata['motion_duration_fraction']:.3%} of full trajectory)",
        "",
        "## Raw Motion/Static Baseline",
        "",
        f"- Mean static Z std across segment context windows: "
        f"{raw_aggregate['mean_static_z_std_mm']:.6g} mm",
        f"- Mean static XY std across segment context windows: "
        f"{raw_aggregate['mean_static_xy_mean_std_mm']:.6g} mm",
        f"- Mean static Z/XY ratio: {raw_aggregate['mean_static_z_over_xy_ratio']:.6g}",
        f"- Mean motion Z range per segment: {raw_aggregate['mean_motion_z_range_mm']:.6g} mm",
        f"- Mean motion dominant Z frequency: "
        f"{raw_aggregate['mean_motion_dominant_frequency_hz']:.6g} Hz",
        f"- Mean motion peaks per second: "
        f"{raw_aggregate['mean_motion_peaks_per_second']:.6g}",
        "",
        "## Detected Motion Segments",
        "",
        "| segment | source rows | motion rows | duration s | motion s | pre z std mm | motion dom hz | motion peaks/s | post z std mm |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in metadata["motion_segment_rows"]:
        pre = row["pre_static_summary"]
        motion = row["motion_summary"]
        post = row["post_static_summary"]
        lines.append(
            f"| {row['segment_index']} | "
            f"{row['source_start_row']}..{row['source_stop_row_exclusive'] - 1} | "
            f"{row['motion_start_row']}..{row['motion_stop_row_exclusive'] - 1} | "
            f"{row['duration_s']:.6g} | {row['motion_duration_s']:.6g} | "
            f"{pre['z_std_mm']:.6g} | {motion['dominant_frequency_hz']:.6g} | "
            f"{motion['peaks_per_second']:.6g} | {post['z_std_mm']:.6g} |"
        )

    lines.extend(
        [
            "",
            "## Best Filter Candidates",
            "",
            "| algorithm | params | robust score | mean static z ratio | mean motion neighbor ratio | mean motion corr | mean lag p95 ms | filtered csv |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )

    if top_filters.empty:
        lines.append("| - | - | - | - | - | - | - | - |")
    else:
        for _, row in top_filters.iterrows():
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
                        f"`{row['filtered_csv']}`",
                    ]
                )
                + " |"
            )

    if not segment_metrics.empty:
        best_rows = segment_metrics.sort_values(["score", "along_lag_ms_p95_positive"]).head(10)
        lines.extend(
            [
                "",
                "## Best Segment-Level Results",
                "",
                "| segment | algorithm | params | score | static z ratio | motion range ratio | motion corr | lag p95 ms |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in best_rows.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(int(row["segment_index"])),
                        str(row["algorithm"]),
                        f"`{row['params']}`",
                        f"{row['score']:.6g}",
                        f"{row['static_mean_z_std_ratio']:.6g}",
                        f"{row['motion_z_range_ratio_to_raw']:.6g}",
                        f"{row['motion_z_corr_to_raw']:.6g}",
                        f"{row['along_lag_ms_p95_positive']:.6g}",
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Filtering is run on the full trajectory, then evaluated on the same five motion windows used in case20-case24.",
            "- Static suppression is judged from the buffered static context before/after each motion segment.",
            "- Motion preservation is judged by segment-local Z shape metrics such as range ratio, dominant-frequency shift, correlation, and along-track lag proxy.",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
