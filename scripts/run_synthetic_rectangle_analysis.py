from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rt_filter.batch import run_batch
from rt_filter.io import read_trajectory, write_trajectory
from rt_filter.se3 import make_poses
from rt_filter.stats import create_report
from rt_filter.trajectory import Trajectory


CASE10_ID = "case_10_static_robot_arm_high_base_before_or_unspecified_calibration_default_exposure"
CASE11_ID = "case_11_synthetic_rectangle_high_base_case10_noise_speed_100mms"
INPUT_ROOT = Path("input/synthetic")
SN_INPUT_ROOT = Path("input/sn")
OUTPUT_ROOT = Path("outputs/synthetic_rectangle")

FILTERS = [
    {"name": "moving_average", "params": {"window": [5, 9, 15]}},
    {"name": "savgol", "params": {"window": [9, 15, 31], "polyorder": [2]}},
    {"name": "exponential", "params": {"alpha": [0.25, 0.4, 0.6]}},
    {"name": "kalman_cv", "params": {"process_noise": [1e-4, 1e-3], "measurement_noise": [1e-2]}},
    {"name": "one_euro_z", "params": {"min_cutoff": [0.5, 0.7], "beta": [2.0, 4.0], "d_cutoff": [1.0]}},
]


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    clean, noisy, metadata = build_case(
        speed_mm_s=args.speed,
        sample_rate_hz=args.sample_rate,
        loops=args.loops,
        seed=args.seed,
        corner_radius=args.corner_radius,
        z_center=args.z_center,
        z_range=args.z_range,
        noise_variant=args.noise_variant,
    )

    speed_tag = _number_tag(args.speed)
    stem = f"rectangle_case10_noise_{speed_tag}mms"
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    noisy_path = INPUT_ROOT / f"{stem}.csv"
    reference_path = INPUT_ROOT / f"{stem}_reference.csv"
    metadata_path = INPUT_ROOT / f"{stem}_metadata.json"
    write_trajectory(noisy, noisy_path)
    write_trajectory(clean, reference_path)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    run_name = args.run_name or f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = run_batch(
        [noisy_path],
        FILTERS,
        output_dir=OUTPUT_ROOT,
        references={noisy_path.stem: reference_path},
        run_name=run_name,
        visualization=True,
    )
    report_dir = create_report(run_dir / "summary.csv", metric="to_reference_translation_rmse")
    _write_case_report(run_dir, noisy_path, reference_path, metadata_path, metadata)

    print(f"input: {noisy_path}")
    print(f"reference: {reference_path}")
    print(f"metadata: {metadata_path}")
    print(f"run_dir: {run_dir}")
    print(f"report_dir: {report_dir}")
    _print_top_results(run_dir / "summary.csv")


def ensure_sn_case11(
    *,
    speed_mm_s: float = 100.0,
    sample_rate_hz: float = 100.0,
    loops: int = 10,
    seed: int = 20260426,
    corner_radius: float = 15.0,
    z_center: float = 20.0,
    z_range: float = 20.0,
    noise_variant: str = "median",
) -> dict[str, Any]:
    clean, noisy, metadata = build_case(
        speed_mm_s=speed_mm_s,
        sample_rate_hz=sample_rate_hz,
        loops=loops,
        seed=seed,
        corner_radius=corner_radius,
        z_center=z_center,
        z_range=z_range,
        noise_variant=noise_variant,
    )
    case_dir = SN_INPUT_ROOT / CASE11_ID
    case_dir.mkdir(parents=True, exist_ok=True)

    speed_tag = _number_tag(speed_mm_s)
    trajectory = f"{CASE11_ID}_01_rectangle_{speed_tag}mms"
    input_csv = case_dir / f"{trajectory}.csv"
    reference_csv = case_dir / f"{trajectory}_reference.csv"
    metadata_path = case_dir / f"{trajectory}_metadata.json"
    write_trajectory(noisy.copy_with(name=trajectory), input_csv)
    write_trajectory(clean.copy_with(name=f"{trajectory}_reference"), reference_csv)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    row = {
        "trajectory": trajectory,
        "case_id": CASE11_ID,
        "case_label": "synthetic_rectangle",
        "source_folder": "synthetic_rectangle_case10_noise",
        "source_file": input_csv.name,
        "source_path": str(input_csv),
        "input_csv": str(input_csv),
        "source_workbook": "",
        "base": "high_base",
        "target": "synthetic_rectangle",
        "calibration": "before_or_unspecified_calibration",
        "exposure": "default_exposure",
        "added_points": False,
        "variant_id": f"speed_{speed_tag}mms",
        "tracker_smoothing": "synthetic",
        "variant_note": "rounded rectangle with case10 median noise",
        "sample_count": noisy.count,
        "raw_row_count": noisy.count,
        "position_unit": "mm",
        "rotation_source": "generated yaw plus case10 rotation noise",
        "rotation_assumption": "Generated yaw around Z with small case10 rotation noise",
        "timestamp_source": "generated from sample_rate parameter",
        "sample_rate_hz": sample_rate_hz,
        "reference_csv": str(reference_csv),
        "metadata_json": str(metadata_path),
    }
    _upsert_manifest_row(SN_INPUT_ROOT / "manifest.csv", row, key="trajectory")
    return row


def build_case(
    *,
    speed_mm_s: float,
    sample_rate_hz: float,
    loops: int,
    seed: int,
    corner_radius: float,
    z_center: float,
    z_range: float,
    noise_variant: str,
) -> tuple[Trajectory, Trajectory, dict[str, Any]]:
    if speed_mm_s <= 0 or sample_rate_hz <= 0 or loops < 1:
        raise ValueError("speed, sample_rate must be positive and loops must be >= 1")
    noise = case10_noise_level(noise_variant)
    xy, tangents, path_length = rounded_rectangle_xy(
        speed_mm_s=speed_mm_s,
        sample_rate_hz=sample_rate_hz,
        loops=loops,
        x_min=300.0,
        x_max=500.0,
        y_min=600.0,
        y_max=700.0,
        radius=corner_radius,
    )
    count = xy.shape[0]
    timestamps = np.arange(count, dtype=float) / sample_rate_hz
    phase = np.linspace(0.0, float(loops), count, endpoint=True)
    z = z_center + 0.5 * z_range * _continuous_unit_wave(phase)
    positions = np.column_stack([xy, z])

    yaw = np.arctan2(tangents[:, 1], tangents[:, 0])
    rotations = Rotation.from_euler("xyz", np.column_stack([np.zeros(count), np.zeros(count), yaw]))
    clean = Trajectory(
        make_poses(positions, rotations),
        timestamps=timestamps,
        name=f"rectangle_case10_noise_{_number_tag(speed_mm_s)}mms_reference",
        metadata={
            "kind": "synthetic_rectangle_reference",
            "speed_mm_s": speed_mm_s,
            "sample_rate_hz": sample_rate_hz,
            "loops": loops,
            "z_center": z_center,
            "z_range": z_range,
        },
    )

    rng = np.random.default_rng(seed)
    noisy_positions = positions + rng.normal(
        scale=[noise["std_x"], noise["std_y"], noise["std_z"]],
        size=positions.shape,
    )
    rot_noise = Rotation.from_rotvec(
        rng.normal(
            scale=np.deg2rad(
                [noise["rot_std_rx_deg"], noise["rot_std_ry_deg"], noise["rot_std_rz_deg"]]
            ),
            size=(count, 3),
        )
    )
    noisy_rotations = rotations * rot_noise
    noisy = Trajectory(
        make_poses(noisy_positions, noisy_rotations),
        timestamps=timestamps,
        name=f"rectangle_case10_noise_{_number_tag(speed_mm_s)}mms",
        metadata={
            "kind": "synthetic_rectangle_noisy",
            "noise_source_case": CASE10_ID,
            "noise_variant": noise_variant,
            "seed": seed,
        },
    )

    metadata = {
        "case": noisy.name,
        "reference": clean.name,
        "speed_mm_s": speed_mm_s,
        "sample_rate_hz": sample_rate_hz,
        "loops": loops,
        "sample_count": count,
        "duration_s": float(timestamps[-1] - timestamps[0]) if count > 1 else 0.0,
        "path_length_mm": path_length * loops,
        "actual_mean_speed_mm_s": float(path_length * loops / timestamps[-1]) if count > 1 else 0.0,
        "x_range": [300.0, 500.0],
        "y_range": [600.0, 700.0],
        "z_center": z_center,
        "z_range_requested": z_range,
        "z_range_clean": float(np.ptp(z)),
        "corner_radius": corner_radius,
        "noise": noise,
    }
    return clean, noisy, metadata


def case10_noise_level(variant: str) -> dict[str, float]:
    manifest = pd.read_csv("input/sn/manifest.csv")
    rows = manifest[manifest["case_id"].eq(CASE10_ID)]
    if rows.empty:
        raise ValueError(f"case not found in input/sn/manifest.csv: {CASE10_ID}")

    stats: list[dict[str, float]] = []
    labels: list[str] = []
    for _, row in rows.iterrows():
        label = str(row["tracker_smoothing"])
        if variant != "median" and variant != label:
            continue
        traj = read_trajectory(row["input_csv"])
        pos = traj.positions
        mean_rot = traj.rotations.mean()
        rotvec = (mean_rot.inv() * traj.rotations).as_rotvec()
        stats.append(
            {
                "std_x": float(pos[:, 0].std(ddof=0)),
                "std_y": float(pos[:, 1].std(ddof=0)),
                "std_z": float(pos[:, 2].std(ddof=0)),
                "rot_std_rx_deg": float(np.rad2deg(rotvec[:, 0].std(ddof=0))),
                "rot_std_ry_deg": float(np.rad2deg(rotvec[:, 1].std(ddof=0))),
                "rot_std_rz_deg": float(np.rad2deg(rotvec[:, 2].std(ddof=0))),
            }
        )
        labels.append(label)
    if not stats:
        raise ValueError("noise_variant must be one of: median, off, on")
    frame = pd.DataFrame(stats)
    result = frame.median().to_dict() if variant == "median" else frame.iloc[0].to_dict()
    return {key: float(value) for key, value in result.items()} | {"source_count": float(len(stats))}


def rounded_rectangle_xy(
    *,
    speed_mm_s: float,
    sample_rate_hz: float,
    loops: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    width = x_max - x_min
    height = y_max - y_min
    max_radius = min(width, height) / 2.0
    if not 0.0 < radius < max_radius:
        raise ValueError(f"corner radius must be in (0, {max_radius})")
    line_x = width - 2.0 * radius
    line_y = height - 2.0 * radius
    segments = [
        ("line", line_x, np.array([x_min + radius, y_min]), np.array([1.0, 0.0]), None, None),
        ("arc", radius * np.pi / 2.0, None, None, np.array([x_max - radius, y_min + radius]), -np.pi / 2.0),
        ("line", line_y, np.array([x_max, y_min + radius]), np.array([0.0, 1.0]), None, None),
        ("arc", radius * np.pi / 2.0, None, None, np.array([x_max - radius, y_max - radius]), 0.0),
        ("line", line_x, np.array([x_max - radius, y_max]), np.array([-1.0, 0.0]), None, None),
        ("arc", radius * np.pi / 2.0, None, None, np.array([x_min + radius, y_max - radius]), np.pi / 2.0),
        ("line", line_y, np.array([x_min, y_max - radius]), np.array([0.0, -1.0]), None, None),
        ("arc", radius * np.pi / 2.0, None, None, np.array([x_min + radius, y_min + radius]), np.pi),
    ]
    path_length = float(sum(segment[1] for segment in segments))
    duration = path_length * loops / speed_mm_s
    count = int(round(duration * sample_rate_hz)) + 1
    distances = np.linspace(0.0, path_length * loops, count)
    points = np.empty((count, 2), dtype=float)
    tangents = np.empty((count, 2), dtype=float)
    for idx, distance in enumerate(distances):
        local = distance % path_length
        if idx == count - 1:
            local = 0.0
        points[idx], tangents[idx] = _rounded_rectangle_sample(segments, radius, local)
    return points, tangents, path_length


def _rounded_rectangle_sample(
    segments: list[tuple[str, float, Any, Any, Any, Any]],
    radius: float,
    distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    remaining = distance
    for kind, length, start, direction, center, theta0 in segments:
        if remaining <= length:
            if kind == "line":
                tangent = np.asarray(direction, dtype=float)
                return np.asarray(start, dtype=float) + tangent * remaining, tangent
            theta = float(theta0) + remaining / radius
            point = np.asarray(center, dtype=float) + radius * np.array([np.cos(theta), np.sin(theta)])
            tangent = np.array([-np.sin(theta), np.cos(theta)])
            return point, tangent
        remaining -= length
    return _rounded_rectangle_sample(segments, radius, 0.0)


def _continuous_unit_wave(phase: np.ndarray) -> np.ndarray:
    wave = np.sin(2.0 * np.pi * phase) + 0.35 * np.sin(4.0 * np.pi * phase + 0.6)
    center = 0.5 * (float(np.max(wave)) + float(np.min(wave)))
    half_range = 0.5 * float(np.ptp(wave))
    return (wave - center) / half_range if half_range > 0 else wave


def _write_case_report(
    run_dir: Path,
    noisy_path: Path,
    reference_path: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
) -> None:
    lines = [
        "# Synthetic Rectangle Analysis",
        "",
        f"- Input: `{noisy_path}`",
        f"- Reference: `{reference_path}`",
        f"- Metadata: `{metadata_path}`",
        f"- Speed: {metadata['speed_mm_s']:.6g} mm/s",
        f"- Sample rate: {metadata['sample_rate_hz']:.6g} Hz",
        f"- Samples: {metadata['sample_count']}",
        f"- Duration: {metadata['duration_s']:.6g} s",
        f"- Clean path length: {metadata['path_length_mm']:.6g} mm",
        f"- Clean Z range: {metadata['z_range_clean']:.6g} mm",
        "",
        "## Case 10 Noise",
        "",
    ]
    for key, value in metadata["noise"].items():
        lines.append(f"- {key}: {value:.6g}")
    lines.extend(["", "## Top Results", ""])
    summary = pd.read_csv(run_dir / "summary.csv")
    top = summary.sort_values("to_reference_translation_rmse").head(10)
    lines.append("| algorithm | params | to_reference_translation_rmse | to_raw_translation_rmse | jerk_rms_ratio |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for _, row in top.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["algorithm"]),
                    f"`{row['params']}`",
                    f"{row['to_reference_translation_rmse']:.6g}",
                    f"{row['to_raw_translation_rmse']:.6g}",
                    f"{row['jerk_rms_ratio']:.6g}",
                ]
            )
            + " |"
        )
    (run_dir / "synthetic_rectangle_report.md").write_text("\n".join(lines), encoding="utf-8")


def _print_top_results(summary_csv: Path) -> None:
    summary = pd.read_csv(summary_csv)
    cols = [
        "algorithm",
        "params",
        "to_reference_translation_rmse",
        "to_raw_translation_rmse",
        "jerk_rms_ratio",
    ]
    top = summary.sort_values("to_reference_translation_rmse")[cols].head(8)
    print("top results by to_reference_translation_rmse:")
    print(top.to_string(index=False))


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


def _number_tag(value: float) -> str:
    text = f"{value:g}"
    return text.replace(".", "p").replace("-", "m")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and analyze a noisy synthetic rectangle trajectory.")
    parser.add_argument("--speed", type=float, default=100.0, help="Nominal XY path speed in mm/s.")
    parser.add_argument("--sample-rate", type=float, default=100.0, help="Sample rate in Hz.")
    parser.add_argument("--loops", type=int, default=10, help="Number of rectangle loops.")
    parser.add_argument("--corner-radius", type=float, default=15.0, help="Rounded corner radius in mm.")
    parser.add_argument("--z-center", type=float, default=20.0, help="Mean clean Z value in mm.")
    parser.add_argument("--z-range", type=float, default=20.0, help="Approximate clean Z peak-to-peak range in mm.")
    parser.add_argument("--noise-variant", choices=["median", "off", "on"], default="median")
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument("--run-name")
    return parser


if __name__ == "__main__":
    main()
