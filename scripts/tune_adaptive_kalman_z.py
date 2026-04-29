from __future__ import annotations

import csv
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rt_filter.filters import run_filter
from rt_filter.io import read_trajectory
from rt_filter.se3 import make_poses
from rt_filter.trajectory import Trajectory

import run_case16_hotel_analysis as analysis_base


OUTPUT_ROOT = Path("outputs/adaptive_kalman_z_tuning")
CASES = [
    {
        "case_id": "case_16_static_robot_arm_hotel",
        "trajectory_path": Path(
            "input/sn/case_16_static_robot_arm_hotel/"
            "case_16_static_robot_arm_hotel_01_track_data_0_hotel.csv"
        ),
    },
    {
        "case_id": "case_17_static_robot_arm_poor_environment",
        "trajectory_path": Path(
            "input/sn/case_17_static_robot_arm_poor_environment/"
            "case_17_static_robot_arm_poor_environment_01_track_data_500.csv"
        ),
    },
]

BASELINE_FILTERS = [
    (
        "one_euro_z",
        {
            "min_cutoff": 0.02,
            "beta": 6.0,
            "d_cutoff": 2.0,
            "derivative_deadband": 1.0,
        },
    ),
    (
        "one_euro_z",
        {
            "min_cutoff": 0.1,
            "beta": 3.0,
            "d_cutoff": 2.0,
            "derivative_deadband": 0.2,
        },
    ),
]

PARAM_GRID = {
    "process_noise": [0.3, 1.0, 3.0],
    "measurement_noise": [1.0e-4, 2.5e-4, 5.0e-4],
    "motion_process_gain": [0.04, 0.08, 0.12],
    "velocity_deadband": [0.5, 1.0],
    "innovation_scale": [1.0, 2.0, 4.0],
    "innovation_gate": [2.5, 3.5],
}


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    static_rows, dynamic_rows, summary_rows = tune()
    static_csv = OUTPUT_ROOT / "adaptive_kalman_z_static_rows.csv"
    dynamic_csv = OUTPUT_ROOT / "adaptive_kalman_z_dynamic_rows.csv"
    summary_csv = OUTPUT_ROOT / "adaptive_kalman_z_summary.csv"
    report_md = OUTPUT_ROOT / "adaptive_kalman_z_report.md"
    _write_csv(static_csv, static_rows)
    _write_csv(dynamic_csv, dynamic_rows)
    _write_csv(summary_csv, summary_rows)
    _write_report(report_md, pd.DataFrame(summary_rows))
    print(
        json.dumps(
            {
                "static_csv": str(static_csv),
                "dynamic_csv": str(dynamic_csv),
                "summary_csv": str(summary_csv),
                "report_md": str(report_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def tune() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    static_rows: list[dict[str, Any]] = []
    dynamic_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for case in CASES:
        traj = read_trajectory(case["trajectory_path"])
        static_rows.extend(_baseline_static_rows(case["case_id"], traj))
        dynamic_rows.extend(_baseline_dynamic_rows(case["case_id"], traj))

    keys = list(PARAM_GRID.keys())
    for values in product(*(PARAM_GRID[key] for key in keys)):
        params = dict(zip(keys, values, strict=True))
        params["initial_covariance"] = 1.0
        case_static_rows = []
        case_dynamic_rows = []
        for case in CASES:
            traj = read_trajectory(case["trajectory_path"])
            case_static_rows.append(_static_eval(case["case_id"], traj, params))
            case_dynamic_rows.extend(_dynamic_eval(case["case_id"], traj, params))
        static_rows.extend(case_static_rows)
        dynamic_rows.extend(case_dynamic_rows)
        summary_rows.append(_summarize_params(params, case_static_rows, case_dynamic_rows))

    summary_rows.sort(key=lambda row: (row["robust_score"], row["worst_dynamic_score"]))
    return static_rows, dynamic_rows, summary_rows


def _baseline_static_rows(case_id: str, traj: Trajectory) -> list[dict[str, Any]]:
    rows = []
    raw_z_std = float(np.std(traj.positions[:, 2], ddof=0))
    raw_xy_std = float(np.mean(np.std(traj.positions[:, :2], axis=0, ddof=0)))
    rows.append(
        {
            "case_id": case_id,
            "algorithm": "raw",
            "params": "{}",
            "x_std_mm": float(np.std(traj.positions[:, 0], ddof=0)),
            "y_std_mm": float(np.std(traj.positions[:, 1], ddof=0)),
            "z_std_mm": raw_z_std,
            "xy_mean_std_mm": raw_xy_std,
            "z_std_ratio_to_raw": 1.0,
            "xy_std_ratio_to_raw": 1.0,
            "to_raw_translation_rmse_mm": 0.0,
            "static_score": 1.0,
        }
    )
    for algorithm, params in BASELINE_FILTERS:
        rows.append(_static_eval(case_id, traj, params, algorithm=algorithm))
    return rows


def _baseline_dynamic_rows(case_id: str, traj: Trajectory) -> list[dict[str, Any]]:
    rows = []
    for scenario in analysis_base.DYNAMIC_SCENARIOS:
        rows.append(_dynamic_eval_one(case_id, traj, {}, "raw", scenario))
        for algorithm, params in BASELINE_FILTERS:
            rows.append(_dynamic_eval_one(case_id, traj, params, algorithm, scenario))
    return rows


def _static_eval(
    case_id: str,
    raw: Trajectory,
    params: dict[str, Any],
    *,
    algorithm: str = "adaptive_kalman_z",
) -> dict[str, Any]:
    filtered = run_filter(algorithm, raw, params)
    raw_z_std = float(np.std(raw.positions[:, 2], ddof=0))
    raw_xy_std = float(np.mean(np.std(raw.positions[:, :2], axis=0, ddof=0)))
    z_std = float(np.std(filtered.positions[:, 2], ddof=0))
    xy_std = float(np.mean(np.std(filtered.positions[:, :2], axis=0, ddof=0)))
    to_raw = float(np.sqrt(np.mean(np.sum((filtered.positions - raw.positions) ** 2, axis=1))))
    static_score = z_std / max(raw_z_std, 1e-12) + 10.0 * to_raw
    return {
        "case_id": case_id,
        "algorithm": algorithm,
        "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
        "x_std_mm": float(np.std(filtered.positions[:, 0], ddof=0)),
        "y_std_mm": float(np.std(filtered.positions[:, 1], ddof=0)),
        "z_std_mm": z_std,
        "xy_mean_std_mm": xy_std,
        "z_std_ratio_to_raw": z_std / max(raw_z_std, 1e-12),
        "xy_std_ratio_to_raw": xy_std / max(raw_xy_std, 1e-12),
        "to_raw_translation_rmse_mm": to_raw,
        "static_score": static_score,
    }


def _dynamic_eval(
    case_id: str,
    raw_static: Trajectory,
    params: dict[str, Any],
    *,
    algorithm: str = "adaptive_kalman_z",
) -> list[dict[str, Any]]:
    return [_dynamic_eval_one(case_id, raw_static, params, algorithm, scenario) for scenario in analysis_base.DYNAMIC_SCENARIOS]


def _dynamic_eval_one(
    case_id: str,
    raw_static: Trajectory,
    params: dict[str, Any],
    algorithm: str,
    scenario: dict[str, Any],
) -> dict[str, Any]:
    noise = raw_static.positions - np.mean(raw_static.positions, axis=0, keepdims=True)
    timestamps = raw_static.timestamps
    assert timestamps is not None
    clean_positions, hold_mask, moving_mask = analysis_base._build_dynamic_positions(
        raw_static.count,
        z_mm_per_frame=float(scenario["z_mm_per_frame"]),
    )
    rotations = Rotation.concatenate([raw_static.rotations[0]] * raw_static.count)
    noisy_positions = clean_positions + noise
    noisy = Trajectory(
        make_poses(noisy_positions, rotations),
        timestamps=timestamps.copy(),
        name=f"{case_id}__{scenario['name']}__raw",
    )
    filtered = noisy if algorithm == "raw" else run_filter(algorithm, noisy, params)
    metrics = analysis_base._dynamic_metrics(
        clean_positions=clean_positions,
        noisy_positions=noisy_positions,
        filtered_positions=filtered.positions,
        timestamps=timestamps,
        hold_mask=hold_mask,
        moving_mask=moving_mask,
    )
    dynamic_score = (
        metrics["z_hold_noise_ratio_to_raw"]
        + 0.03 * metrics["z_lag_p95_ms"]
        + 5.0 * metrics["z_rmse_mm"]
    )
    return {
        "case_id": case_id,
        "scenario": scenario["name"],
        "z_mm_per_frame": float(scenario["z_mm_per_frame"]),
        "algorithm": algorithm,
        "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
        **metrics,
        "dynamic_score": float(dynamic_score),
    }


def _summarize_params(
    params: dict[str, Any],
    static_rows: list[dict[str, Any]],
    dynamic_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    static_scores = np.asarray([row["static_score"] for row in static_rows], dtype=float)
    dynamic_scores = np.asarray([row["dynamic_score"] for row in dynamic_rows], dtype=float)
    static_z_ratios = np.asarray([row["z_std_ratio_to_raw"] for row in static_rows], dtype=float)
    dynamic_noise_ratios = np.asarray([row["z_hold_noise_ratio_to_raw"] for row in dynamic_rows], dtype=float)
    dynamic_lag_ms = np.asarray([row["z_lag_p95_ms"] for row in dynamic_rows], dtype=float)
    robust_score = float(0.6 * np.mean(dynamic_scores) + 0.4 * np.max(dynamic_scores) + 0.4 * np.mean(static_scores))
    return {
        "algorithm": "adaptive_kalman_z",
        "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
        "robust_score": robust_score,
        "mean_static_score": float(np.mean(static_scores)),
        "max_static_score": float(np.max(static_scores)),
        "mean_dynamic_score": float(np.mean(dynamic_scores)),
        "worst_dynamic_score": float(np.max(dynamic_scores)),
        "mean_static_z_ratio": float(np.mean(static_z_ratios)),
        "worst_static_z_ratio": float(np.max(static_z_ratios)),
        "mean_dynamic_noise_ratio": float(np.mean(dynamic_noise_ratios)),
        "worst_dynamic_noise_ratio": float(np.max(dynamic_noise_ratios)),
        "mean_dynamic_lag_ms": float(np.mean(dynamic_lag_ms)),
        "worst_dynamic_lag_ms": float(np.max(dynamic_lag_ms)),
    }


def _write_report(path: Path, summary: pd.DataFrame) -> None:
    best = summary.sort_values(["robust_score", "worst_dynamic_score"]).head(10)
    lines = [
        "# Adaptive Kalman Z Tuning Report",
        "",
        f"- Cases: {', '.join(case['case_id'] for case in CASES)}",
        f"- Grid size: {len(summary)}",
        "",
        "## Best Robust Parameter Sets",
        "",
        "| params | robust | mean static | mean dynamic | worst dynamic | mean dyn noise | worst dyn lag ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in best.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["params"]),
                    f"{row['robust_score']:.6g}",
                    f"{row['mean_static_score']:.6g}",
                    f"{row['mean_dynamic_score']:.6g}",
                    f"{row['worst_dynamic_score']:.6g}",
                    f"{row['mean_dynamic_noise_ratio']:.6g}",
                    f"{row['worst_dynamic_lag_ms']:.6g}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


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


if __name__ == "__main__":
    main()
