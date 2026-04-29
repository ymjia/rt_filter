from __future__ import annotations

import csv
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rt_filter.filters import run_filter
from rt_filter.io import read_trajectory

import run_case16_hotel_analysis as analysis_base


OUTPUT_ROOT = Path("outputs/adaptive_kalman_z_static_tuning")
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
    {
        "case_id": "case_18_static_robot_arm_track_noise_0_1",
        "trajectory_path": Path(
            "input/sn/case_18_static_robot_arm_track_noise_0_1/"
            "case_18_static_robot_arm_track_noise_0_1_01_track_noise_0_1.csv"
        ),
    },
]
BASELINES = [
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
    "process_noise": [1e-12, 3e-12, 1e-11, 3e-11, 1e-10],
    "measurement_noise": [1e-5, 3e-5, 1e-4],
    "innovation_scale": [4.0, 10.0, 20.0],
    "innovation_gate": [2.5, 3.0, 3.5],
    "max_measurement_scale": [20.0, 50.0, 100.0],
}
FIXED_PARAMS = {
    "initial_covariance": 1.0,
    "motion_process_gain": 0.0,
    "velocity_deadband": 1.0,
}


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows, summary_rows = tune()
    rows_csv = OUTPUT_ROOT / "adaptive_kalman_z_static_rows.csv"
    summary_csv = OUTPUT_ROOT / "adaptive_kalman_z_static_summary.csv"
    report_md = OUTPUT_ROOT / "adaptive_kalman_z_static_report.md"
    _write_csv(rows_csv, rows)
    _write_csv(summary_csv, summary_rows)
    _write_report(report_md, pd.DataFrame(summary_rows), pd.DataFrame(rows))
    print(
        json.dumps(
            {
                "rows_csv": str(rows_csv),
                "summary_csv": str(summary_csv),
                "report_md": str(report_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def tune() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    case_cache = {
        case["case_id"]: read_trajectory(case["trajectory_path"])
        for case in CASES
    }

    for case in CASES:
        traj = case_cache[case["case_id"]]
        rows.append(_eval_filter(case["case_id"], traj, {}, "raw"))
        for algorithm, params in BASELINES:
            rows.append(_eval_filter(case["case_id"], traj, params, algorithm))

    keys = list(PARAM_GRID.keys())
    for values in product(*(PARAM_GRID[key] for key in keys)):
        params = {**FIXED_PARAMS, **dict(zip(keys, values, strict=True))}
        current_rows = []
        for case in CASES:
            traj = case_cache[case["case_id"]]
            current_rows.append(_eval_filter(case["case_id"], traj, params, "adaptive_kalman_z"))
        rows.extend(current_rows)
        summary_rows.append(_summarize_params(params, current_rows))

    summary_rows.sort(key=lambda row: (row["robust_score"], row["worst_case_score"]))
    return rows, summary_rows


def _eval_filter(case_id: str, raw, params: dict[str, Any], algorithm: str) -> dict[str, Any]:
    filtered = raw if algorithm == "raw" else run_filter(algorithm, raw, params)
    raw_stats = _static_metrics(raw)
    filtered_stats = _static_metrics(filtered)
    to_raw_rmse = float(np.sqrt(np.mean(np.sum((filtered.positions - raw.positions) ** 2, axis=1))))
    score = (
        0.45 * _safe_ratio(filtered_stats["z_std_mm"], raw_stats["z_std_mm"])
        + 0.35 * _safe_ratio(filtered_stats["neighbor10_z_rms_mm"], raw_stats["neighbor10_z_rms_mm"])
        + 0.20 * _safe_ratio(filtered_stats["neighbor20_z_rms_mm"], raw_stats["neighbor20_z_rms_mm"])
        + 3.0 * to_raw_rmse
    )
    return {
        "case_id": case_id,
        "algorithm": algorithm,
        "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
        **filtered_stats,
        "to_raw_translation_rmse_mm": to_raw_rmse,
        "z_std_ratio_to_raw": _safe_ratio(filtered_stats["z_std_mm"], raw_stats["z_std_mm"]),
        "neighbor10_z_ratio_to_raw": _safe_ratio(
            filtered_stats["neighbor10_z_rms_mm"], raw_stats["neighbor10_z_rms_mm"]
        ),
        "neighbor20_z_ratio_to_raw": _safe_ratio(
            filtered_stats["neighbor20_z_rms_mm"], raw_stats["neighbor20_z_rms_mm"]
        ),
        "score": float(score),
    }


def _static_metrics(traj) -> dict[str, float]:
    positions = traj.positions
    centered = positions - np.mean(positions, axis=0, keepdims=True)
    residual10 = analysis_base._centered_neighbor_residual(positions, 10)
    residual20 = analysis_base._centered_neighbor_residual(positions, 20)
    return {
        "x_std_mm": float(np.std(positions[:, 0], ddof=0)),
        "y_std_mm": float(np.std(positions[:, 1], ddof=0)),
        "z_std_mm": float(np.std(positions[:, 2], ddof=0)),
        "radial_p95_mm": float(np.percentile(np.linalg.norm(centered, axis=1), 95)),
        "neighbor10_z_rms_mm": float(np.sqrt(np.mean(residual10[:, 2] ** 2))),
        "neighbor20_z_rms_mm": float(np.sqrt(np.mean(residual20[:, 2] ** 2))),
    }


def _summarize_params(params: dict[str, Any], case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = np.asarray([row["score"] for row in case_rows], dtype=float)
    z_ratios = np.asarray([row["z_std_ratio_to_raw"] for row in case_rows], dtype=float)
    neighbor10 = np.asarray([row["neighbor10_z_ratio_to_raw"] for row in case_rows], dtype=float)
    neighbor20 = np.asarray([row["neighbor20_z_ratio_to_raw"] for row in case_rows], dtype=float)
    radial = np.asarray([row["radial_p95_mm"] for row in case_rows], dtype=float)
    return {
        "algorithm": "adaptive_kalman_z",
        "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
        "robust_score": float(0.6 * np.mean(scores) + 0.4 * np.max(scores)),
        "mean_score": float(np.mean(scores)),
        "worst_case_score": float(np.max(scores)),
        "mean_z_std_ratio": float(np.mean(z_ratios)),
        "worst_z_std_ratio": float(np.max(z_ratios)),
        "mean_neighbor10_z_ratio": float(np.mean(neighbor10)),
        "worst_neighbor10_z_ratio": float(np.max(neighbor10)),
        "mean_neighbor20_z_ratio": float(np.mean(neighbor20)),
        "worst_neighbor20_z_ratio": float(np.max(neighbor20)),
        "max_radial_p95_mm": float(np.max(radial)),
    }


def _write_report(path: Path, summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    best = summary.sort_values(["robust_score", "worst_case_score"]).head(10)
    baseline = rows.loc[rows["algorithm"].astype(str).ne("adaptive_kalman_z")].copy()
    lines = [
        "# Adaptive Kalman Z Static Tuning Report",
        "",
        f"- Cases: {', '.join(case['case_id'] for case in CASES)}",
        f"- Grid size: {len(summary)}",
        "",
        "## Baselines",
        "",
        "| case | algorithm | params | z std ratio | neighbor10 z ratio | neighbor20 z ratio |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for _, row in baseline.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["case_id"]),
                    str(row["algorithm"]),
                    str(row["params"]),
                    f"{row['z_std_ratio_to_raw']:.6g}",
                    f"{row['neighbor10_z_ratio_to_raw']:.6g}",
                    f"{row['neighbor20_z_ratio_to_raw']:.6g}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Best Adaptive Parameter Sets",
            "",
            "| params | robust | mean z std | worst z std | mean n10 | mean n20 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in best.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["params"]),
                    f"{row['robust_score']:.6g}",
                    f"{row['mean_z_std_ratio']:.6g}",
                    f"{row['worst_z_std_ratio']:.6g}",
                    f"{row['mean_neighbor10_z_ratio']:.6g}",
                    f"{row['mean_neighbor20_z_ratio']:.6g}",
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


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return float(numerator / denominator)


if __name__ == "__main__":
    main()
