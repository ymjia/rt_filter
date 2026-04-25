from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from rt_filter.batch import expand_parameter_grid, parameter_slug
from rt_filter.evaluation import compare_filter_result, trajectory_metrics
from rt_filter.filters import run_filter
from rt_filter.io import write_trajectory
from rt_filter.se3 import rotation_angle
from rt_filter.trajectory import Trajectory
from rt_filter.vtk_export import write_vtk_unstructured_grid


@dataclass(frozen=True)
class FilterSpec:
    algorithm: str
    params: dict[str, Any]


@dataclass
class FilterAnalysisResult:
    spec: FilterSpec
    trajectory: Trajectory
    metrics: dict[str, float | int]
    dimension_metrics: dict[str, float | int]
    output_dir: Path | None = None
    trajectory_path: Path | None = None
    vtk_path: Path | None = None

    @property
    def label(self) -> str:
        slug = parameter_slug(self.spec.params)
        return f"{self.spec.algorithm}/{slug}"

    def table_row(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "algorithm": self.spec.algorithm,
            "params": json.dumps(self.spec.params, ensure_ascii=False, sort_keys=True),
            **self.metrics,
            **self.dimension_metrics,
            "trajectory_path": "" if self.trajectory_path is None else str(self.trajectory_path),
            "vtk_path": "" if self.vtk_path is None else str(self.vtk_path),
        }


def parse_filter_specs(rows: list[dict[str, Any]]) -> list[FilterSpec]:
    specs: list[FilterSpec] = []
    for row in rows:
        enabled = bool(row.get("enabled", True))
        if not enabled:
            continue
        algorithm = str(row["algorithm"])
        params_payload = row.get("params", {})
        if isinstance(params_payload, str):
            params_payload = json.loads(params_payload) if params_payload.strip() else {}
        for params in expand_parameter_grid(params_payload):
            specs.append(FilterSpec(algorithm=algorithm, params=params))
    return specs


def analyze_filters(
    raw: Trajectory,
    specs: list[FilterSpec],
    *,
    reference: Trajectory | None = None,
    output_root: str | Path | None = None,
    run_name: str | None = None,
    write_outputs: bool = True,
    write_vtk: bool = True,
) -> tuple[Path | None, list[FilterAnalysisResult]]:
    if not specs:
        raise ValueError("at least one filter spec is required")

    run_dir: Path | None = None
    if write_outputs:
        root = Path(output_root or "outputs/gui")
        run_dir = root / (run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S"))
        run_dir.mkdir(parents=True, exist_ok=True)
        write_trajectory(raw, run_dir / "raw.csv")
        if write_vtk:
            write_vtk_unstructured_grid(raw, run_dir / "raw.vtu")

    results: list[FilterAnalysisResult] = []
    for spec in specs:
        filtered = run_filter(spec.algorithm, raw, spec.params)
        metrics = compare_filter_result(raw, filtered, reference=reference)
        dimension_metrics = compare_dimensions(raw, filtered, reference=reference)
        result = FilterAnalysisResult(spec, filtered, metrics, dimension_metrics)
        if run_dir is not None:
            result_dir = run_dir / spec.algorithm / parameter_slug(spec.params)
            result_dir.mkdir(parents=True, exist_ok=True)
            result.output_dir = result_dir
            result.trajectory_path = result_dir / "trajectory.csv"
            write_trajectory(filtered, result.trajectory_path)
            if write_vtk:
                result.vtk_path = result_dir / "trajectory.vtu"
                write_vtk_unstructured_grid(filtered, result.vtk_path)
            (result_dir / "metrics.json").write_text(
                json.dumps(result.table_row(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        results.append(result)

    if run_dir is not None:
        _write_summary(run_dir / "summary.csv", [result.table_row() for result in results])
    return run_dir, results


def compare_dimensions(
    raw: Trajectory,
    filtered: Trajectory,
    *,
    reference: Trajectory | None = None,
) -> dict[str, float | int]:
    count = min(raw.count, filtered.count)
    raw_pos = raw.positions[:count]
    filtered_pos = filtered.positions[:count]
    delta = filtered_pos - raw_pos
    metrics: dict[str, float | int] = {}

    for axis_index, axis in enumerate("xyz"):
        axis_delta = delta[:, axis_index]
        raw_axis = raw_pos[:, axis_index]
        filtered_axis = filtered_pos[:, axis_index]
        raw_acc = _axis_derivative_rms(raw_axis, raw.timestamps, order=2, count=count)
        filtered_acc = _axis_derivative_rms(filtered_axis, filtered.timestamps, order=2, count=count)
        raw_jerk = _axis_derivative_rms(raw_axis, raw.timestamps, order=3, count=count)
        filtered_jerk = _axis_derivative_rms(filtered_axis, filtered.timestamps, order=3, count=count)
        metrics[f"to_raw_{axis}_mean"] = float(np.mean(axis_delta))
        metrics[f"to_raw_{axis}_rmse"] = float(np.sqrt(np.mean(axis_delta**2)))
        metrics[f"to_raw_{axis}_max_abs"] = float(np.max(np.abs(axis_delta)))
        metrics[f"{axis}_acceleration_rms_ratio"] = _safe_ratio(filtered_acc, raw_acc)
        metrics[f"{axis}_jerk_rms_ratio"] = _safe_ratio(filtered_jerk, raw_jerk)

    rot_delta = np.rad2deg(rotation_angle(filtered.poses[:count], raw.poses[:count]))
    metrics["to_raw_rotation_p95_deg"] = float(np.percentile(rot_delta, 95))

    if reference is not None:
        ref_count = min(reference.count, filtered.count)
        ref_pos = reference.positions[:ref_count]
        filt_ref_delta = filtered.positions[:ref_count] - ref_pos
        raw_ref_delta = raw.positions[:ref_count] - ref_pos
        for axis_index, axis in enumerate("xyz"):
            metrics[f"to_reference_{axis}_rmse"] = float(
                np.sqrt(np.mean(filt_ref_delta[:, axis_index] ** 2))
            )
            metrics[f"raw_to_reference_{axis}_rmse"] = float(
                np.sqrt(np.mean(raw_ref_delta[:, axis_index] ** 2))
            )
    return metrics


def analysis_conclusions(results: list[FilterAnalysisResult]) -> list[str]:
    if not results:
        return ["尚未运行滤波。"]

    lines: list[str] = []
    if all("to_reference_translation_rmse" in item.metrics for item in results):
        best_ref = min(results, key=lambda item: float(item.metrics["to_reference_translation_rmse"]))
        improvement = float(best_ref.metrics.get("reference_rmse_improvement", 0.0))
        lines.append(
            "参考轨迹指标最优："
            f"{best_ref.label}，translation RMSE={best_ref.metrics['to_reference_translation_rmse']:.6g}，"
            f"相对原始改善={improvement:.6g}。"
        )

    best_smooth = min(results, key=lambda item: float(item.metrics["jerk_rms_ratio"]))
    lines.append(
        "平滑强度最强："
        f"{best_smooth.label}，jerk RMS ratio={best_smooth.metrics['jerk_rms_ratio']:.4g}。"
    )

    least_offset = min(results, key=lambda item: float(item.metrics["to_raw_translation_rmse"]))
    lines.append(
        "相对原始轨迹改动最小："
        f"{least_offset.label}，to_raw translation RMSE={least_offset.metrics['to_raw_translation_rmse']:.6g}。"
    )

    for axis in "xyz":
        best_axis = min(results, key=lambda item: float(item.dimension_metrics[f"{axis}_jerk_rms_ratio"]))
        worst_offset = max(results, key=lambda item: float(item.dimension_metrics[f"to_raw_{axis}_rmse"]))
        lines.append(
            f"{axis.upper()} 维：{best_axis.label} 对 jerk 抑制最强 "
            f"({best_axis.dimension_metrics[f'{axis}_jerk_rms_ratio']:.4g})；"
            f"{worst_offset.label} 的该维偏移最大 "
            f"({worst_offset.dimension_metrics[f'to_raw_{axis}_rmse']:.6g})。"
        )

    high_offset = [
        item
        for item in results
        if float(item.metrics["to_raw_translation_rmse"])
        > max(float(item.metrics["filtered_path_length"]) * 0.05, 1e-12)
    ]
    if high_offset:
        labels = ", ".join(item.label for item in high_offset[:3])
        lines.append(f"注意：{labels} 相对原始轨迹偏移超过路径长度的 5%，可能存在过度平滑或滞后。")
    return lines


def result_metric_matrix(results: list[FilterAnalysisResult]) -> list[dict[str, Any]]:
    return [result.table_row() for result in results]


def _axis_derivative_rms(
    values: np.ndarray,
    timestamps: np.ndarray | None,
    *,
    order: int,
    count: int,
) -> float:
    arr = np.asarray(values[:count], dtype=float)
    if len(arr) <= order:
        return 0.0
    time = None if timestamps is None else np.asarray(timestamps[:count], dtype=float)
    current = arr
    current_t = time
    for _ in range(order):
        if current_t is None:
            current = np.diff(current)
        else:
            dt = np.diff(current_t)
            current = np.diff(current) / dt
            current_t = current_t[1:]
    return float(np.sqrt(np.mean(current**2)))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return float(numerator / denominator)


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
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
