from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rt_filter.batch import run_batch
from rt_filter.io import read_trajectory
from rt_filter.stats import create_report

from scripts.prepare_sn_ref_data import INPUT_ROOT, prepare_sn_data
from scripts.run_local_lab_case12 import (
    ensure_sn_case12,
    ensure_sn_cases13_to15,
    write_case12_motion_report,
)
from scripts.run_synthetic_rectangle_analysis import ensure_sn_case11


OUTPUT_ROOT = Path("outputs/sn")


FILTERS = [
    {"name": "moving_average", "params": {"window": [5, 15, 31]}},
    {"name": "savgol", "params": {"window": [11, 31], "polyorder": [2]}},
    {"name": "exponential", "params": {"alpha": [0.15, 0.3, 0.5]}},
    {"name": "kalman_cv", "params": {"process_noise": [1e-4], "measurement_noise": [1e-2]}},
    {
        "name": "ukf",
        "params": [
            {
                "motion_model": "constant_velocity",
                "process_noise": 1000.0,
                "measurement_noise": 0.001,
                "initial_linear_velocity": [0.0, 0.0, 0.0],
                "initial_angular_velocity": [0.0, 0.0, 0.0],
            },
            {
                "motion_model": "constant_acceleration",
                "process_noise": 10000.0,
                "measurement_noise": 0.001,
                "initial_linear_velocity": [0.0, 0.0, 0.0],
                "initial_angular_velocity": [0.0, 0.0, 0.0],
            },
            {
                "motion_model": "constant_velocity",
                "process_noise": 100.0,
                "measurement_noise": 0.001,
                "initial_linear_velocity": [0.0, 0.0, 0.0],
                "initial_angular_velocity": [0.0, 0.0, 0.0],
            },
        ],
    },
    {
        "name": "one_euro_z",
        "params": [
            {"min_cutoff": 0.02, "beta": 6.0, "d_cutoff": 2.0, "derivative_deadband": 1.0},
            {"min_cutoff": 0.02, "beta": 4.0, "d_cutoff": 2.0, "derivative_deadband": 1.0},
            {"min_cutoff": 0.7, "beta": 4.0, "d_cutoff": 1.0, "derivative_deadband": 0.0},
        ],
    },
]


def main() -> None:
    prepare_sn_data()
    ensure_sn_case11()
    ensure_sn_case12()
    ensure_sn_cases13_to15()
    manifest = pd.read_csv(INPUT_ROOT / "manifest.csv")
    inputs = [Path(path) for path in manifest["input_csv"]]
    references = _reference_map(manifest)
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = run_batch(
        inputs,
        FILTERS,
        output_dir=OUTPUT_ROOT,
        references=references,
        run_name=run_name,
        visualization=True,
    )
    create_report(run_dir / "summary.csv", metric="filtered_range_z")
    stability_rows = build_stability_summary(run_dir, manifest)
    stability_csv = run_dir / "sn_filter_stability.csv"
    _write_csv(stability_csv, stability_rows)
    write_markdown_report(run_dir, manifest, stability_rows)
    write_case12_motion_report(run_dir, manifest)
    write_stability_plots(run_dir, pd.DataFrame(stability_rows))
    print(f"SN analysis output: {run_dir}")


def _reference_map(manifest: pd.DataFrame) -> dict[str, Path]:
    if "reference_csv" not in manifest.columns:
        return {}
    references: dict[str, Path] = {}
    for _, row in manifest.iterrows():
        reference = row.get("reference_csv", "")
        if pd.isna(reference) or not str(reference).strip():
            continue
        input_path = Path(row["input_csv"])
        references[input_path.stem] = Path(str(reference))
    return references


def build_stability_summary(run_dir: Path, manifest: pd.DataFrame) -> list[dict[str, Any]]:
    summary = pd.read_csv(run_dir / "summary.csv")
    manifest_by_trajectory = manifest.set_index("trajectory").to_dict(orient="index")
    raw_stats = {}
    for _, row in manifest.iterrows():
        traj = read_trajectory(row["input_csv"])
        raw_stats[row["trajectory"]] = _stability_stats(traj, prefix="raw")

    rows: list[dict[str, Any]] = []
    for _, result in summary.iterrows():
        trajectory = result["trajectory"]
        meta = manifest_by_trajectory[trajectory]
        filtered = read_trajectory(result["trajectory_path"])
        filtered_stats = _stability_stats(filtered, prefix="filtered")
        raw = raw_stats[trajectory]
        row = {
            "trajectory": trajectory,
            "case_id": meta["case_id"],
            "case_label": meta["case_label"],
            "source_folder": meta["source_folder"],
            "source_file": meta["source_file"],
            "base": meta["base"],
            "target": meta["target"],
            "calibration": meta["calibration"],
            "exposure": meta["exposure"],
            "added_points": meta["added_points"],
            "tracker_smoothing": meta["tracker_smoothing"],
            "variant_id": meta["variant_id"],
            "algorithm": result["algorithm"],
            "params": result["params"],
            "trajectory_path": result["trajectory_path"],
            "vtk_path": result["vtk_path"],
            **raw,
            **filtered_stats,
            "range_norm_ratio": _safe_ratio(filtered_stats["filtered_range_norm"], raw["raw_range_norm"]),
            "std_norm_ratio": _safe_ratio(filtered_stats["filtered_std_norm"], raw["raw_std_norm"]),
            "z_range_ratio": _safe_ratio(filtered_stats["filtered_range_z"], raw["raw_range_z"]),
            "z_std_ratio": _safe_ratio(filtered_stats["filtered_std_z"], raw["raw_std_z"]),
            "to_raw_translation_rmse": result["to_raw_translation_rmse"],
            "to_raw_translation_max": result["to_raw_translation_max"],
            "acceleration_rms_ratio": result["acceleration_rms_ratio"],
            "jerk_rms_ratio": result["jerk_rms_ratio"],
        }
        rows.append(row)
    return rows


def write_markdown_report(run_dir: Path, manifest: pd.DataFrame, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    lines = [
        "# SN Filtering Report",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## Data",
        "",
        f"- Source folders: {manifest['source_folder'].nunique()}",
        f"- Source trajectories: {len(manifest)}",
        f"- Filtered results: {len(frame)}",
        "- Position unit: mm",
        "- Rotation assumption: XYZ Euler angles in degrees from `xr,yr,zr`",
        "- Timestamps: see `timestamp_source` in `input/sn/manifest.csv`",
        "",
        "## Overall Best Results",
        "",
    ]
    for key, label in [
        ("filtered_range_norm", "smallest filtered position range norm"),
        ("std_norm_ratio", "best std reduction ratio"),
        ("z_std_ratio", "best Z std reduction ratio"),
        ("jerk_rms_ratio", "best jerk reduction ratio"),
    ]:
        row = frame.sort_values(key).iloc[0]
        lines.append(
            f"- {label}: `{row['trajectory']}` with `{row['algorithm']}` {row['params']} "
            f"({key}={row[key]:.6g}, to_raw_rmse={row['to_raw_translation_rmse']:.6g})"
        )

    lines.extend(["", "## Best Result Per Case", ""])
    for case_id, group in frame.groupby("case_id", sort=True):
        best = group.sort_values(["std_norm_ratio", "to_raw_translation_rmse"]).iloc[0]
        raw = group.iloc[0]
        lines.extend(
            [
                f"### {case_id}",
                "",
                f"- Source folder: `{best['source_folder']}`",
                f"- Raw range norm: {raw['raw_range_norm']:.6g} mm",
                f"- Raw std norm: {raw['raw_std_norm']:.6g} mm",
                f"- Best filter: `{best['algorithm']}` {best['params']}",
                f"- Filtered range norm: {best['filtered_range_norm']:.6g} mm",
                f"- Filtered std norm ratio: {best['std_norm_ratio']:.6g}",
                f"- Z std ratio: {best['z_std_ratio']:.6g}",
                f"- VTK: `{best['vtk_path']}`",
                "",
            ]
        )

    lines.extend(["", "## Device Smoothing Raw Comparison", ""])
    raw_compare = (
        manifest.groupby(["case_id", "tracker_smoothing"])["trajectory"]
        .count()
        .unstack(fill_value=0)
        .reset_index()
    )
    lines.extend(_markdown_table(raw_compare))
    lines.append("")
    (run_dir / "sn_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_stability_plots(run_dir: Path, frame: pd.DataFrame) -> None:
    report_dir = run_dir / "sn_report_assets"
    report_dir.mkdir(parents=True, exist_ok=True)
    best = frame.sort_values(["case_id", "std_norm_ratio"]).groupby("case_id", as_index=False).first()

    plt.figure(figsize=(12, 5))
    plt.bar(best["case_id"], best["std_norm_ratio"])
    plt.xticks(rotation=75, ha="right", fontsize=8)
    plt.ylabel("std_norm_ratio")
    plt.title("Best filtered std ratio per SN case")
    plt.tight_layout()
    plt.savefig(report_dir / "best_std_ratio_by_case.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.bar(best["case_id"], best["z_std_ratio"])
    plt.xticks(rotation=75, ha="right", fontsize=8)
    plt.ylabel("z_std_ratio")
    plt.title("Best filtered Z std ratio per SN case")
    plt.tight_layout()
    plt.savefig(report_dir / "best_z_std_ratio_by_case.png", dpi=160)
    plt.close()


def _stability_stats(traj, *, prefix: str) -> dict[str, float]:
    positions = traj.positions
    ranges = positions.max(axis=0) - positions.min(axis=0)
    std = positions.std(axis=0, ddof=0)
    centered = positions - positions.mean(axis=0, keepdims=True)
    radial = np.linalg.norm(centered, axis=1)
    return {
        f"{prefix}_range_x": float(ranges[0]),
        f"{prefix}_range_y": float(ranges[1]),
        f"{prefix}_range_z": float(ranges[2]),
        f"{prefix}_range_norm": float(np.linalg.norm(ranges)),
        f"{prefix}_std_x": float(std[0]),
        f"{prefix}_std_y": float(std[1]),
        f"{prefix}_std_z": float(std[2]),
        f"{prefix}_std_norm": float(np.linalg.norm(std)),
        f"{prefix}_radial_p95": float(np.percentile(radial, 95)),
        f"{prefix}_radial_max": float(np.max(radial)),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0 if numerator == 0 else float("inf")
    return float(numerator / denominator)


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
    columns = [str(col) for col in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in frame.columns) + " |")
    return lines


if __name__ == "__main__":
    main()
