from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in (ROOT, SCRIPT_DIR):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

from rt_filter.batch import parameter_slug
from rt_filter.filters import run_filter
from rt_filter.io import read_trajectory

import run_case16_hotel_analysis as static_base


INPUT_ROOT = Path("input/sn")
OUTPUT_ROOT = Path("outputs/case25_30_xyz_peak_preservation")
AXES = ("x", "y", "z")
TARGET_PEAK_RATE_RANGE_HZ = (10.0, 15.0)
PEAK_DISTANCE_CAP_HZ = TARGET_PEAK_RATE_RANGE_HZ[1]
PEAK_PROMINENCE_STD_RATIO = 0.05

FILTER_SPECS = [
    {"algorithm": "raw", "params": {}},
    {"algorithm": "butterworth", "params": {"cutoff_hz": 12.0, "order": 2}},
    {"algorithm": "butterworth", "params": {"cutoff_hz": 15.0, "order": 2}},
    {"algorithm": "butterworth", "params": {"cutoff_hz": 18.0, "order": 2}},
    {"algorithm": "butterworth", "params": {"cutoff_hz": 20.0, "order": 2}},
    {"algorithm": "butterworth", "params": {"cutoff_hz": 25.0, "order": 2}},
    {"algorithm": "butterworth", "params": {"cutoff_hz": 30.0, "order": 2}},
    {
        "algorithm": "one_euro",
        "params": {
            "min_cutoff": 8.0,
            "beta": 0.0,
            "d_cutoff": 2.0,
            "derivative_deadband": 0.0,
        },
    },
    {
        "algorithm": "one_euro",
        "params": {
            "min_cutoff": 10.0,
            "beta": 0.0,
            "d_cutoff": 2.0,
            "derivative_deadband": 0.0,
        },
    },
    {
        "algorithm": "one_euro",
        "params": {
            "min_cutoff": 12.0,
            "beta": 0.0,
            "d_cutoff": 2.0,
            "derivative_deadband": 0.0,
        },
    },
    {
        "algorithm": "one_euro",
        "params": {
            "min_cutoff": 15.0,
            "beta": 0.0,
            "d_cutoff": 2.0,
            "derivative_deadband": 0.0,
        },
    },
    {
        "algorithm": "one_euro",
        "params": {
            "min_cutoff": 18.0,
            "beta": 0.0,
            "d_cutoff": 2.0,
            "derivative_deadband": 0.0,
        },
    },
    {
        "algorithm": "one_euro",
        "params": {
            "min_cutoff": 20.0,
            "beta": 0.0,
            "d_cutoff": 2.0,
            "derivative_deadband": 0.0,
        },
    },
    {
        "algorithm": "one_euro",
        "params": {
            "min_cutoff": 25.0,
            "beta": 0.0,
            "d_cutoff": 2.0,
            "derivative_deadband": 0.0,
        },
    },
]


def main() -> None:
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run() -> dict[str, Any]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cases = _case_paths()
    if not cases:
        raise ValueError("no case25~30 trajectories found under input/sn")

    rows: list[dict[str, Any]] = []
    for case_id, path in cases:
        raw = read_trajectory(path)
        raw_summary = _trajectory_peak_summary(raw)
        for spec in FILTER_SPECS:
            algorithm = str(spec["algorithm"])
            params = dict(spec["params"])
            filtered = raw if algorithm == "raw" else run_filter(algorithm, raw, params)
            filtered_summary = _trajectory_peak_summary(filtered)
            rows.append(
                _comparison_row(
                    case_id=case_id,
                    algorithm=algorithm,
                    params=params,
                    raw_summary=raw_summary,
                    filtered_summary=filtered_summary,
                )
            )

    detail = pd.DataFrame(rows)
    aggregate = _aggregate(detail)

    detail_csv = OUTPUT_ROOT / "per_case_peak_summary.csv"
    aggregate_csv = OUTPUT_ROOT / "aggregate_peak_summary.csv"
    report_md = OUTPUT_ROOT / "peak_preservation_report.md"
    detail.to_csv(detail_csv, index=False)
    aggregate.to_csv(aggregate_csv, index=False)
    _write_report(report_md, detail, aggregate)

    return {
        "case_count": len(cases),
        "detail_csv": str(detail_csv),
        "aggregate_csv": str(aggregate_csv),
        "report_md": str(report_md),
    }


def _case_paths() -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    for case_number in range(25, 31):
        pattern = f"case_{case_number:02d}_*/*.csv"
        for path in sorted(INPUT_ROOT.glob(pattern)):
            paths.append((path.parent.name, path))
    return paths


def _trajectory_peak_summary(traj) -> dict[str, Any]:
    timestamps = traj.timestamps
    if timestamps is None:
        raise ValueError(f"{traj.name} has no timestamps")
    sample_rate_hz = static_base._median_sample_rate(timestamps)
    duration_s = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
    positions = traj.positions

    axes: dict[str, dict[str, float | int | bool]] = {}
    for axis_name, axis_index in zip(AXES, range(3), strict=True):
        axes[axis_name] = _axis_peak_summary(
            positions[:, axis_index],
            sample_rate_hz=sample_rate_hz,
            duration_s=duration_s,
        )
    return {
        "sample_rate_hz": float(sample_rate_hz),
        "duration_s": duration_s,
        "axes": axes,
    }


def _axis_peak_summary(
    values: np.ndarray,
    *,
    sample_rate_hz: float,
    duration_s: float,
) -> dict[str, float | int | bool]:
    arr = np.asarray(values, dtype=float)
    centered = arr - float(np.mean(arr))
    spectrum = static_base._spectrum_summary(centered, sample_rate_hz=sample_rate_hz or 100.0)
    min_peak_distance = max(1, int((sample_rate_hz or 100.0) / PEAK_DISTANCE_CAP_HZ))
    prominence = max(float(np.std(arr, ddof=0)) * PEAK_PROMINENCE_STD_RATIO, 1e-6)
    if arr.size >= 3:
        peaks, _ = find_peaks(arr, distance=min_peak_distance, prominence=prominence)
    else:
        peaks = np.zeros(0, dtype=int)
    peaks_per_second = float(peaks.size / duration_s) if duration_s > 0.0 else 0.0
    in_target_band = TARGET_PEAK_RATE_RANGE_HZ[0] <= peaks_per_second <= TARGET_PEAK_RATE_RANGE_HZ[1]
    mean_peak_spacing_frames = float(np.mean(np.diff(peaks))) if peaks.size >= 2 else 0.0
    return {
        "std_mm": float(np.std(arr, ddof=0)),
        "peak_count": int(peaks.size),
        "peaks_per_second": peaks_per_second,
        "mean_peak_spacing_frames": mean_peak_spacing_frames,
        "dominant_frequency_hz": float(spectrum["dominant_frequency_hz"]),
        "spectral_centroid_hz": float(spectrum["spectral_centroid_hz"]),
        "in_target_band": bool(in_target_band),
    }


def _comparison_row(
    *,
    case_id: str,
    algorithm: str,
    params: dict[str, Any],
    raw_summary: dict[str, Any],
    filtered_summary: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": case_id,
        "algorithm": algorithm,
        "params_slug": parameter_slug(params),
        "params_json": json.dumps(params, sort_keys=True, ensure_ascii=False),
        "sample_rate_hz": float(filtered_summary["sample_rate_hz"]),
        "duration_s": float(filtered_summary["duration_s"]),
    }
    axes_in_target = 0
    abs_peak_shifts: list[float] = []
    std_ratios: list[float] = []
    for axis_name in AXES:
        raw_axis = raw_summary["axes"][axis_name]
        filtered_axis = filtered_summary["axes"][axis_name]
        raw_peak_rate = float(raw_axis["peaks_per_second"])
        filtered_peak_rate = float(filtered_axis["peaks_per_second"])
        raw_std = float(raw_axis["std_mm"])
        filtered_std = float(filtered_axis["std_mm"])
        peak_shift = abs(filtered_peak_rate - raw_peak_rate)
        std_ratio = filtered_std / max(raw_std, 1e-12)
        axes_in_target += int(bool(filtered_axis["in_target_band"]))
        abs_peak_shifts.append(peak_shift)
        std_ratios.append(std_ratio)

        row[f"raw_{axis_name}_std_mm"] = raw_std
        row[f"filtered_{axis_name}_std_mm"] = filtered_std
        row[f"{axis_name}_std_ratio_to_raw"] = std_ratio
        row[f"raw_{axis_name}_peaks_per_second"] = raw_peak_rate
        row[f"filtered_{axis_name}_peaks_per_second"] = filtered_peak_rate
        row[f"{axis_name}_abs_peak_rate_shift"] = peak_shift
        row[f"raw_{axis_name}_dominant_frequency_hz"] = float(raw_axis["dominant_frequency_hz"])
        row[f"filtered_{axis_name}_dominant_frequency_hz"] = float(filtered_axis["dominant_frequency_hz"])
        row[f"filtered_{axis_name}_in_target_band"] = bool(filtered_axis["in_target_band"])

    row["axes_in_target_count"] = axes_in_target
    row["all_axes_in_target"] = axes_in_target == len(AXES)
    row["mean_abs_peak_rate_shift_to_raw"] = float(np.mean(abs_peak_shifts))
    row["mean_std_ratio_to_raw"] = float(np.mean(std_ratios))
    return row


def _aggregate(detail: pd.DataFrame) -> pd.DataFrame:
    aggregate = (
        detail.groupby(["algorithm", "params_slug", "params_json"], as_index=False)
        .agg(
            case_count=("case_id", "nunique"),
            mean_axes_in_target=("axes_in_target_count", "mean"),
            min_axes_in_target=("axes_in_target_count", "min"),
            all_cases_all_axes_in_target=("all_axes_in_target", "min"),
            mean_abs_peak_rate_shift_to_raw=("mean_abs_peak_rate_shift_to_raw", "mean"),
            worst_abs_peak_rate_shift_to_raw=("mean_abs_peak_rate_shift_to_raw", "max"),
            mean_std_ratio_to_raw=("mean_std_ratio_to_raw", "mean"),
        )
        .sort_values(
            [
                "all_cases_all_axes_in_target",
                "min_axes_in_target",
                "mean_axes_in_target",
                "mean_abs_peak_rate_shift_to_raw",
                "mean_std_ratio_to_raw",
            ],
            ascending=[False, False, False, True, True],
            ignore_index=True,
        )
    )
    return aggregate


def _write_report(path: Path, detail: pd.DataFrame, aggregate: pd.DataFrame) -> None:
    lines = [
        "# Case 25-30 XYZ Peak Preservation Report",
        "",
        f"- Input root: `{INPUT_ROOT}`",
        f"- Output root: `{OUTPUT_ROOT}`",
        f"- Target peak-rate band: {TARGET_PEAK_RATE_RANGE_HZ[0]:.0f}-{TARGET_PEAK_RATE_RANGE_HZ[1]:.0f} peaks/s",
        (
            "- Peak detection rule: `scipy.signal.find_peaks`, "
            f"`distance=floor(sample_rate/{PEAK_DISTANCE_CAP_HZ:.0f})`, "
            f"`prominence=max(std*{PEAK_PROMINENCE_STD_RATIO:.2f}, 1e-6)`"
        ),
        "",
        "## Aggregate Ranking",
        "",
        "| algorithm | params | mean target axes | min target axes | all cases all axes in target | mean abs peak-rate shift | mean std ratio |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for _, row in aggregate.iterrows():
        lines.append(
            "| "
            f"{row['algorithm']} | `{row['params_slug']}` | "
            f"{row['mean_axes_in_target']:.3f} | {int(row['min_axes_in_target'])} | "
            f"{bool(row['all_cases_all_axes_in_target'])} | "
            f"{row['mean_abs_peak_rate_shift_to_raw']:.3f} | {row['mean_std_ratio_to_raw']:.3f} |"
        )

    best_by_case = (
        detail.loc[detail["algorithm"].ne("raw")]
        .sort_values(
            ["all_axes_in_target", "axes_in_target_count", "mean_abs_peak_rate_shift_to_raw", "mean_std_ratio_to_raw"],
            ascending=[False, False, True, True],
        )
        .groupby("case_id", as_index=False, sort=False)
        .head(1)
    )
    lines.extend(
        [
            "",
            "## Best Per Case",
            "",
            "| case | algorithm | params | axes in target | mean abs peak-rate shift | x peaks/s | y peaks/s | z peaks/s |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in best_by_case.iterrows():
        lines.append(
            "| "
            f"{row['case_id']} | {row['algorithm']} | `{row['params_slug']}` | "
            f"{int(row['axes_in_target_count'])} | {row['mean_abs_peak_rate_shift_to_raw']:.3f} | "
            f"{row['filtered_x_peaks_per_second']:.3f} | {row['filtered_y_peaks_per_second']:.3f} | "
            f"{row['filtered_z_peaks_per_second']:.3f} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
