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

from rt_filter.filters import run_filter
from rt_filter.io import write_trajectory
from rt_filter.se3 import make_poses
from rt_filter.trajectory import Trajectory


SOURCE_PATH = Path("ref_data/track_data_0酒店.csv")
INPUT_ROOT = Path("input/sn")
OUTPUT_ROOT = Path("outputs/case16_hotel_static")

CASE_ID = "case_16_static_robot_arm_hotel"
CASE_LABEL = "static_robot_arm"
TRAJECTORY = f"{CASE_ID}_01_track_data_0_hotel"
CASE_BASE = "hotel"
CASE_TARGET = "robot_arm"
CASE_VARIANT_ID = "track_data_0_hotel"
CASE_VARIANT_NOTE = "hotel static robot arm trajectory"
REPORT_TITLE = "Case 16 Hotel Static Robot Arm Report"
SOURCE_COLUMNS = ["x", "y", "z", "xr", "yr", "zr", "time", "rate"]
TIMESTAMP_TICKS_PER_SECOND = 10000.0
MAX_CONTINUOUS_STEP_S = 0.05

STATIC_FILTERS = [
    {"algorithm": "raw", "params": {}},
    {"algorithm": "moving_average", "params": {"window": 9}},
    {"algorithm": "savgol", "params": {"window": 21, "polyorder": 2}},
    {"algorithm": "exponential", "params": {"alpha": 0.35}},
    {
        "algorithm": "kalman_cv",
        "params": {"process_noise": 1e-2, "measurement_noise": 5e-3, "initial_covariance": 1.0},
    },
    {
        "algorithm": "ukf",
        "params": {
            "motion_model": "constant_velocity",
            "process_noise": 100.0,
            "measurement_noise": 0.001,
            "initial_covariance": 1.0,
            "initial_linear_velocity": [0.0, 0.0, 0.0],
            "initial_angular_velocity": [0.0, 0.0, 0.0],
        },
    },
    {
        "algorithm": "one_euro_z",
        "params": {
            "min_cutoff": 0.02,
            "beta": 6.0,
            "d_cutoff": 2.0,
            "derivative_deadband": 1.0,
        },
    },
    {
        "algorithm": "one_euro_z",
        "params": {
            "min_cutoff": 0.10,
            "beta": 3.0,
            "d_cutoff": 2.0,
            "derivative_deadband": 0.2,
        },
    },
    {
        "algorithm": "one_euro_z",
        "params": {
            "min_cutoff": 0.80,
            "beta": 2.0,
            "d_cutoff": 2.0,
            "derivative_deadband": 0.0,
        },
    },
    {
        "algorithm": "adaptive_kalman_z",
        "params": {
            "process_noise": 1e-12,
            "measurement_noise": 1e-5,
            "initial_covariance": 1.0,
            "motion_process_gain": 0.0,
            "velocity_deadband": 1.0,
            "innovation_scale": 20.0,
            "innovation_gate": 2.5,
            "max_measurement_scale": 100.0,
        },
    },
    {
        "algorithm": "adaptive_kalman_z",
        "params": {
            "process_noise": 3e-12,
            "measurement_noise": 1e-5,
            "initial_covariance": 1.0,
            "motion_process_gain": 0.0,
            "velocity_deadband": 1.0,
            "innovation_scale": 10.0,
            "innovation_gate": 2.5,
            "max_measurement_scale": 50.0,
        },
    },
]

DYNAMIC_SCENARIOS = [
    {"name": "mm_per_frame_0p2", "z_mm_per_frame": 0.2},
    {"name": "mm_per_frame_0p5", "z_mm_per_frame": 0.5},
    {"name": "mm_per_frame_1p0", "z_mm_per_frame": 1.0},
]

DYNAMIC_FILTERS = [
    {"algorithm": "raw", "params": {}},
    {"algorithm": "exponential", "params": {"alpha": 0.35}},
    {
        "algorithm": "kalman_cv",
        "params": {"process_noise": 1e-2, "measurement_noise": 5e-3, "initial_covariance": 1.0},
    },
    {
        "algorithm": "one_euro_z",
        "params": {
            "min_cutoff": 0.02,
            "beta": 6.0,
            "d_cutoff": 2.0,
            "derivative_deadband": 1.0,
        },
    },
    {
        "algorithm": "one_euro_z",
        "params": {
            "min_cutoff": 0.10,
            "beta": 3.0,
            "d_cutoff": 2.0,
            "derivative_deadband": 0.2,
        },
    },
    {
        "algorithm": "one_euro_z",
        "params": {
            "min_cutoff": 0.80,
            "beta": 2.0,
            "d_cutoff": 2.0,
            "derivative_deadband": 0.0,
        },
    },
    {"algorithm": "savgol", "params": {"window": 21, "polyorder": 2}},
]


def main() -> None:
    output = run()
    print(json.dumps(output, ensure_ascii=False, indent=2))


def _output_stem() -> str:
    return CASE_ID


def run() -> dict[str, str]:
    trajectory, raw_row_count, frame, metadata = _read_source()
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
        "source_folder": "ref_data",
        "source_file": SOURCE_PATH.name,
        "source_path": str(SOURCE_PATH),
        "input_csv": str(input_csv),
        "source_workbook": "",
        "base": CASE_BASE,
        "target": CASE_TARGET,
        "calibration": "unknown_calibration",
        "exposure": "default_exposure",
        "added_points": False,
        "variant_id": CASE_VARIANT_ID,
        "tracker_smoothing": "unknown",
        "variant_note": CASE_VARIANT_NOTE,
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
    _upsert_manifest_row(INPUT_ROOT / "manifest.csv", manifest_row, key="trajectory")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    static_rows = _run_static_filter_sweep(trajectory)
    dynamic_rows = _run_dynamic_scenarios(trajectory)
    output_stem = _output_stem()
    static_csv = OUTPUT_ROOT / f"{output_stem}_static_filter_comparison.csv"
    dynamic_csv = OUTPUT_ROOT / f"{output_stem}_dynamic_filter_comparison.csv"
    static_json = OUTPUT_ROOT / f"{output_stem}_static_noise_summary.json"
    report_md = OUTPUT_ROOT / f"{output_stem}_report.md"

    _write_csv(static_csv, static_rows)
    _write_csv(dynamic_csv, dynamic_rows)
    static_json.write_text(
        json.dumps(
            {
                "case_id": CASE_ID,
                "trajectory": TRAJECTORY,
                "source_path": str(SOURCE_PATH),
                "summary": metadata["static_noise_summary"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_report(report_md, metadata, static_rows, dynamic_rows)

    return {
        "input_csv": str(input_csv),
        "metadata_json": str(metadata_json),
        "static_csv": str(static_csv),
        "dynamic_csv": str(dynamic_csv),
        "report_md": str(report_md),
    }


def _read_source() -> tuple[Trajectory, int, pd.DataFrame, dict[str, Any]]:
    raw_frame = pd.read_csv(SOURCE_PATH, sep="\t")
    raw_row_count = len(raw_frame)
    frame = raw_frame[SOURCE_COLUMNS].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{SOURCE_PATH} contains no valid numeric trajectory rows")

    trajectory = _trajectory_from_frame(frame)
    metadata = _metadata(raw_row_count, frame, trajectory)
    return trajectory, raw_row_count, frame, metadata


def _trajectory_from_frame(frame: pd.DataFrame) -> Trajectory:
    positions = frame[["x", "y", "z"]].to_numpy(dtype=float)
    rotations = Rotation.from_euler("xyz", frame[["xr", "yr", "zr"]].to_numpy(dtype=float), degrees=True)
    timestamps = _timestamps_from_source(frame)
    return Trajectory(
        make_poses(positions, rotations),
        timestamps=timestamps,
        name=TRAJECTORY,
        metadata={
            "source": str(SOURCE_PATH),
            "position_unit": "mm",
            "rotation_assumption": "XYZ Euler angles in degrees",
            "timestamp_source": "full-normalized source timestamp ticks",
        },
    )


def _timestamps_from_source(frame: pd.DataFrame) -> np.ndarray:
    source_timestamps = frame["time"].to_numpy(dtype=float)
    timestamps = (source_timestamps - source_timestamps[0]) / TIMESTAMP_TICKS_PER_SECOND
    if len(timestamps) > 1 and np.all(np.diff(timestamps) > 0):
        return timestamps

    rate = float(frame["rate"].replace(0, np.nan).dropna().median())
    if not np.isfinite(rate) or rate <= 0:
        rate = 100.0
    return np.arange(len(frame), dtype=float) / rate


def _metadata(raw_row_count: int, frame: pd.DataFrame, trajectory: Trajectory) -> dict[str, Any]:
    positions = trajectory.positions
    timestamps = trajectory.timestamps
    assert timestamps is not None
    dt = np.diff(timestamps)
    valid_dt = dt[dt > 0]
    speed_stats = _step_speed_stats(positions, timestamps)
    static_noise = _static_noise_summary(positions)
    euler = frame[["xr", "yr", "zr"]].to_numpy(dtype=float)
    return {
        "case_id": CASE_ID,
        "trajectory": TRAJECTORY,
        "source_path": str(SOURCE_PATH),
        "source_columns": SOURCE_COLUMNS,
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
        "position_min_mm": _axis_dict(positions.min(axis=0)),
        "position_max_mm": _axis_dict(positions.max(axis=0)),
        "position_range_mm": _axis_dict(np.ptp(positions, axis=0)),
        "position_std_mm": _axis_dict(positions.std(axis=0, ddof=0)),
        "euler_range_deg": _axis_dict(np.ptp(euler, axis=0), axes=("xr", "yr", "zr")),
        "continuous_step_speed_mm_s": speed_stats,
        "static_noise_summary": static_noise,
    }


def _static_noise_summary(positions: np.ndarray) -> dict[str, Any]:
    centered = positions - np.mean(positions, axis=0, keepdims=True)
    std = positions.std(axis=0, ddof=0)
    ratio = float(std[2] / max((std[0] + std[1]) * 0.5, 1e-12))

    windows = {}
    for window in (5, 10, 20):
        residual = _centered_neighbor_residual(positions, window)
        rms = np.sqrt(np.mean(residual**2, axis=0))
        norm = np.linalg.norm(residual, axis=1)
        windows[str(window)] = {
            "sample_count": int(residual.shape[0]),
            "x_rms_mm": float(rms[0]),
            "y_rms_mm": float(rms[1]),
            "z_rms_mm": float(rms[2]),
            "norm_rms_mm": float(np.sqrt(np.mean(norm**2))),
            "norm_p95_mm": float(np.percentile(norm, 95)),
            "z_over_xy_mean_rms": float(rms[2] / max((rms[0] + rms[1]) * 0.5, 1e-12)),
        }

    spectrum = _spectrum_summary(centered[:, 2], sample_rate_hz=100.0)
    return {
        "centered_std_mm": _axis_dict(centered.std(axis=0, ddof=0)),
        "centered_p95_abs_mm": _axis_dict(np.percentile(np.abs(centered), 95, axis=0)),
        "z_over_xy_mean_std_ratio": ratio,
        "neighbor_mean_windows": windows,
        "z_spectrum": spectrum,
    }


def _centered_neighbor_residual(positions: np.ndarray, window: int) -> np.ndarray:
    if positions.shape[0] <= 2 * window:
        return positions - positions.mean(axis=0, keepdims=True)
    residuals = []
    for index in range(window, positions.shape[0] - window):
        start = index - window
        stop = index + window + 1
        neighborhood = positions[start:stop]
        mean = (np.sum(neighborhood, axis=0) - positions[index]) / (neighborhood.shape[0] - 1)
        residuals.append(positions[index] - mean)
    return np.asarray(residuals, dtype=float)


def _spectrum_summary(values: np.ndarray, sample_rate_hz: float) -> dict[str, float]:
    centered = np.asarray(values, dtype=float) - float(np.mean(values))
    if centered.size < 4:
        return {
            "dominant_frequency_hz": 0.0,
            "spectral_centroid_hz": 0.0,
            "power_fraction_above_5hz": 0.0,
            "power_fraction_above_10hz": 0.0,
        }
    freqs = np.fft.rfftfreq(centered.size, d=1.0 / sample_rate_hz)
    power = np.abs(np.fft.rfft(centered)) ** 2
    if power.shape[0] > 0:
        power[0] = 0.0
    total_power = float(power.sum())
    if total_power <= 0.0:
        return {
            "dominant_frequency_hz": 0.0,
            "spectral_centroid_hz": 0.0,
            "power_fraction_above_5hz": 0.0,
            "power_fraction_above_10hz": 0.0,
        }
    dominant_index = int(np.argmax(power))
    return {
        "dominant_frequency_hz": float(freqs[dominant_index]),
        "spectral_centroid_hz": float(np.sum(freqs * power) / total_power),
        "power_fraction_above_5hz": float(power[freqs >= 5.0].sum() / total_power),
        "power_fraction_above_10hz": float(power[freqs >= 10.0].sum() / total_power),
    }


def _run_static_filter_sweep(raw: Trajectory) -> list[dict[str, Any]]:
    centered = raw.positions[:, 2] - np.mean(raw.positions[:, 2])
    raw_z_std = float(np.std(centered, ddof=0))
    raw_xy_std = float(np.mean(np.std(raw.positions[:, :2], axis=0, ddof=0)))
    rows = []
    for spec in STATIC_FILTERS:
        algorithm = str(spec["algorithm"])
        params = dict(spec["params"])
        filtered = raw if algorithm == "raw" else run_filter(algorithm, raw, params)
        z_centered = filtered.positions[:, 2] - np.mean(filtered.positions[:, 2])
        xy_std = float(np.mean(np.std(filtered.positions[:, :2], axis=0, ddof=0)))
        z_std = float(np.std(z_centered, ddof=0))
        row = {
            "algorithm": algorithm,
            "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
            "x_std_mm": float(np.std(filtered.positions[:, 0], ddof=0)),
            "y_std_mm": float(np.std(filtered.positions[:, 1], ddof=0)),
            "z_std_mm": z_std,
            "xy_mean_std_mm": xy_std,
            "z_std_ratio_to_raw": _safe_ratio(z_std, raw_z_std),
            "xy_std_ratio_to_raw": _safe_ratio(xy_std, raw_xy_std),
            "radial_p95_mm": _radial_p95(filtered.positions),
            "to_raw_translation_rmse_mm": float(
                np.sqrt(np.mean(np.sum((filtered.positions - raw.positions) ** 2, axis=1)))
            ),
        }
        rows.append(row)
    return rows


def _run_dynamic_scenarios(raw_static: Trajectory) -> list[dict[str, Any]]:
    rows = []
    noise = raw_static.positions - np.mean(raw_static.positions, axis=0, keepdims=True)
    timestamps = raw_static.timestamps
    assert timestamps is not None
    rotations = raw_static.rotations
    base_rotation = Rotation.concatenate([rotations[0]] * raw_static.count)
    for scenario in DYNAMIC_SCENARIOS:
        clean_positions, hold_mask, moving_mask = _build_dynamic_positions(
            raw_static.count,
            z_mm_per_frame=float(scenario["z_mm_per_frame"]),
        )
        noisy_positions = clean_positions + noise
        noisy = Trajectory(
            make_poses(noisy_positions, base_rotation),
            timestamps=timestamps.copy(),
            name=f"{TRAJECTORY}__{scenario['name']}__raw",
            metadata={"scenario": scenario["name"]},
        )
        for spec in DYNAMIC_FILTERS:
            algorithm = str(spec["algorithm"])
            params = dict(spec["params"])
            filtered = noisy if algorithm == "raw" else run_filter(algorithm, noisy, params)
            metrics = _dynamic_metrics(
                clean_positions=clean_positions,
                noisy_positions=noisy_positions,
                filtered_positions=filtered.positions,
                timestamps=timestamps,
                hold_mask=hold_mask,
                moving_mask=moving_mask,
            )
            rows.append(
                {
                    "scenario": scenario["name"],
                    "z_mm_per_frame": float(scenario["z_mm_per_frame"]),
                    "algorithm": algorithm,
                    "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
                    **metrics,
                }
            )
    return rows


def _build_dynamic_positions(count: int, *, z_mm_per_frame: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if count < 600:
        raise ValueError("dynamic scenario requires at least 600 samples")
    positions = np.zeros((count, 3), dtype=float)
    positions[:, 0] = np.linspace(0.0, 0.6 * z_mm_per_frame * (count - 1), count)
    positions[:, 1] = 8.0 * np.sin(np.linspace(0.0, 4.0 * np.pi, count))

    hold_a = 150
    ramp_a = 250
    hold_b = 150
    ramp_b = 250
    remaining = count - (hold_a + ramp_a + hold_b + ramp_b)
    hold_c = max(0, remaining)

    z = np.zeros(count, dtype=float)
    start = hold_a
    stop = start + ramp_a
    z[start:stop] = z_mm_per_frame * np.arange(ramp_a, dtype=float)
    z[stop : stop + hold_b] = z[stop - 1]
    start = stop + hold_b
    stop = start + ramp_b
    ramp_down_start = float(z[start - 1])
    z[start:stop] = ramp_down_start - z_mm_per_frame * np.arange(1, ramp_b + 1, dtype=float)
    if hold_c > 0:
        z[stop:] = z[stop - 1]
    positions[:, 2] = z

    hold_mask = np.zeros(count, dtype=bool)
    hold_mask[:hold_a] = True
    hold_mask[hold_a + ramp_a : hold_a + ramp_a + hold_b] = True
    if hold_c > 0:
        hold_mask[-hold_c:] = True
    moving_mask = ~hold_mask
    return positions, hold_mask, moving_mask


def _dynamic_metrics(
    *,
    clean_positions: np.ndarray,
    noisy_positions: np.ndarray,
    filtered_positions: np.ndarray,
    timestamps: np.ndarray,
    hold_mask: np.ndarray,
    moving_mask: np.ndarray,
) -> dict[str, float]:
    raw_z_hold_std = float(np.std(noisy_positions[hold_mask, 2] - clean_positions[hold_mask, 2], ddof=0))
    filtered_z_hold_std = float(
        np.std(filtered_positions[hold_mask, 2] - clean_positions[hold_mask, 2], ddof=0)
    )

    z_error = filtered_positions[:, 2] - clean_positions[:, 2]
    xy_error = filtered_positions[:, :2] - clean_positions[:, :2]
    raw_z_error = noisy_positions[:, 2] - clean_positions[:, 2]
    velocity_z = np.gradient(clean_positions[:, 2], timestamps)
    moving_velocity = np.abs(velocity_z[moving_mask])
    signed_motion = np.sign(velocity_z[moving_mask])
    lag_mm = np.maximum(-z_error[moving_mask] * signed_motion, 0.0)
    lag_ms = lag_mm / np.maximum(moving_velocity, 1e-12) * 1000.0
    raw_lag_mm = np.maximum(-raw_z_error[moving_mask] * signed_motion, 0.0)
    raw_lag_ms = raw_lag_mm / np.maximum(moving_velocity, 1e-12) * 1000.0

    return {
        "z_hold_noise_std_mm": filtered_z_hold_std,
        "z_hold_noise_ratio_to_raw": _safe_ratio(filtered_z_hold_std, raw_z_hold_std),
        "z_rmse_mm": float(np.sqrt(np.mean(z_error**2))),
        "xy_rmse_mm": float(np.sqrt(np.mean(np.sum(xy_error**2, axis=1)))),
        "z_rmse_on_hold_mm": float(np.sqrt(np.mean(z_error[hold_mask] ** 2))),
        "z_rmse_on_motion_mm": float(np.sqrt(np.mean(z_error[moving_mask] ** 2))),
        "z_lag_p95_ms": float(np.percentile(lag_ms, 95)),
        "z_lag_p95_mm": float(np.percentile(lag_mm, 95)),
        "raw_z_lag_p95_ms": float(np.percentile(raw_lag_ms, 95)),
        "raw_z_hold_noise_std_mm": raw_z_hold_std,
    }


def _write_report(
    path: Path,
    metadata: dict[str, Any],
    static_rows: list[dict[str, Any]],
    dynamic_rows: list[dict[str, Any]],
) -> None:
    static_frame = pd.DataFrame(static_rows)
    dynamic_frame = pd.DataFrame(dynamic_rows)

    best_static = static_frame.sort_values(["z_std_ratio_to_raw", "to_raw_translation_rmse_mm"]).head(5)
    lines = [
        f"# {REPORT_TITLE}",
        "",
        "## Data Summary",
        "",
        f"- Source: `{SOURCE_PATH}`",
        f"- Output case: `{CASE_ID}`",
        f"- Samples: {metadata['sample_count']}",
        f"- Duration: {metadata['duration_s']:.6g} s",
        f"- Median sample rate: {metadata['sample_rate_hz_median']:.6g} Hz",
        f"- Position std: X {metadata['position_std_mm']['x']:.6g} mm, "
        f"Y {metadata['position_std_mm']['y']:.6g} mm, "
        f"Z {metadata['position_std_mm']['z']:.6g} mm",
        f"- Z / mean(XY) std ratio: {metadata['static_noise_summary']['z_over_xy_mean_std_ratio']:.6g}",
        f"- Z spectral centroid: {metadata['static_noise_summary']['z_spectrum']['spectral_centroid_hz']:.6g} Hz",
        "",
        "## Static Filter Sweep",
        "",
        *_markdown_table(best_static),
        "",
        "## Dynamic Synthetic Scenarios",
        "",
        "The synthetic scenarios reuse the measured static XYZ noise from this case and add it to clean motion with "
        "Z motion speeds of 0.2, 0.5, and 1.0 mm per frame at 100 Hz.",
        "",
    ]

    for scenario, group in dynamic_frame.groupby("scenario", sort=True):
        best_low_noise = group.sort_values(["z_hold_noise_ratio_to_raw", "z_lag_p95_ms"]).head(3)
        best_balanced = group.assign(
            balance_score=group["z_hold_noise_ratio_to_raw"] + 0.02 * group["z_lag_p95_ms"]
        ).sort_values("balance_score").head(3)
        lines.extend(
            [
                f"### {scenario}",
                "",
                "Best pure denoise:",
                "",
                *_markdown_table(best_low_noise),
                "",
                "Best denoise/lag balance:",
                "",
                *_markdown_table(best_balanced.drop(columns=["balance_score"])),
                "",
            ]
        )

    lines.extend(
        [
            "## Engineering Takeaways",
            "",
            "- This dataset is truly close to static, and the Z-axis jitter is several times larger than XY, "
            "so axis-selective filtering is justified.",
            "- If you need realtime filtering, use a Z-only causal filter first; it preserves XY and avoids "
            "unnecessary distortion of the better channels.",
            "- If you can afford future frames offline, Savitzky-Golay can be used as an upper-bound reference "
            "for how much denoise is achievable without strict realtime constraints.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


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


def _median_sample_rate(timestamps: np.ndarray | None) -> float:
    if timestamps is None or len(timestamps) < 2:
        return 0.0
    dt = np.diff(timestamps)
    dt = dt[(dt > 0.0) & (dt <= MAX_CONTINUOUS_STEP_S)]
    if dt.size == 0:
        return 0.0
    return float(1.0 / np.median(dt))


def _axis_dict(values: np.ndarray, axes: tuple[str, str, str] = ("x", "y", "z")) -> dict[str, float]:
    return {axis: float(value) for axis, value in zip(axes, values, strict=True)}


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0 if numerator == 0 else float("inf")
    return float(numerator / denominator)


def _radial_p95(positions: np.ndarray) -> float:
    centered = positions - np.mean(positions, axis=0, keepdims=True)
    return float(np.percentile(np.linalg.norm(centered, axis=1), 95))


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


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    display = frame.copy()
    columns = []
    preferred = [
        "algorithm",
        "params",
        "z_std_ratio_to_raw",
        "xy_std_ratio_to_raw",
        "z_hold_noise_ratio_to_raw",
        "z_lag_p95_ms",
        "z_rmse_mm",
        "xy_rmse_mm",
        "to_raw_translation_rmse_mm",
    ]
    for column in preferred:
        if column in display.columns:
            columns.append(column)
    display = display[columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in display.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


if __name__ == "__main__":
    main()
