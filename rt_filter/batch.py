from __future__ import annotations

import csv
import glob
import itertools
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from rt_filter.evaluation import compare_filter_result, write_metrics
from rt_filter.filters import run_filter
from rt_filter.io import read_trajectory, write_trajectory
from rt_filter.trajectory import Trajectory
from rt_filter.vtk_export import write_vtk_unstructured_grid


def run_batch_config(config_path: str | Path) -> Path:
    path = Path(config_path)
    config = _load_config(path)
    base_dir = path.parent
    inputs = _expand_paths(config.get("inputs", []), base_dir)
    references = _reference_map(config.get("references"), base_dir)
    output_dir = Path(config.get("output_dir", "outputs"))
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    algorithms = config.get("algorithms", [])
    visualization = _visualization_options(config.get("visualization", True))
    return run_batch(
        inputs,
        algorithms,
        output_dir=output_dir,
        references=references,
        run_name=config.get("run_name"),
        visualization=visualization,
    )


def run_batch(
    inputs: list[str | Path],
    algorithms: list[dict[str, Any]],
    *,
    output_dir: str | Path,
    references: dict[str, str | Path] | None = None,
    run_name: str | None = None,
    visualization: bool | dict[str, Any] | None = None,
) -> Path:
    if not inputs:
        raise ValueError("batch input list is empty")
    if not algorithms:
        raise ValueError("batch algorithm list is empty")

    run_id = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "inputs": [str(Path(p)) for p in inputs],
        "algorithms": algorithms,
        "visualization": _visualization_options(visualization),
        "input_artifacts": [],
        "outputs": [],
    }

    visualization_options = manifest["visualization"]
    references = {} if references is None else references
    for input_path in inputs:
        input_file = Path(input_path)
        raw = read_trajectory(input_file)
        reference = _load_reference(input_file, references)
        input_artifacts = _write_input_artifacts(
            raw,
            run_dir / input_file.stem / "input",
            visualization_options=visualization_options,
        )
        input_artifacts["input"] = str(input_file)
        manifest["input_artifacts"].append(input_artifacts)

        for algorithm in algorithms:
            name = algorithm["name"]
            for params in expand_parameter_grid(algorithm.get("params", {})):
                filtered = run_filter(name, raw, params)
                slug = parameter_slug(params)
                result_dir = run_dir / input_file.stem / name / slug
                result_dir.mkdir(parents=True, exist_ok=True)

                trajectory_path = result_dir / "trajectory.csv"
                metrics_path = result_dir / "metrics.json"
                vtk_path = _vtk_path(result_dir, visualization_options)
                write_trajectory(filtered, trajectory_path)
                if vtk_path is not None:
                    _write_vtk(filtered, vtk_path, visualization_options)
                metrics = compare_filter_result(raw, filtered, reference=reference)
                write_metrics(metrics, metrics_path)

                row = {
                    "input": str(input_file),
                    "trajectory": raw.name,
                    "algorithm": name,
                    "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
                    "result_dir": str(result_dir),
                    "input_trajectory_path": input_artifacts["trajectory_path"],
                    "input_vtk_path": input_artifacts.get("vtk_path", ""),
                    "trajectory_path": str(trajectory_path),
                    "vtk_path": "" if vtk_path is None else str(vtk_path),
                    **metrics,
                }
                summary_rows.append(row)
                manifest["outputs"].append(row)

    _write_summary(run_dir / "summary.csv", summary_rows)
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return run_dir


def expand_parameter_grid(params: Any) -> list[dict[str, Any]]:
    if params is None:
        return [{}]
    if isinstance(params, list):
        return [dict(item) for item in params]
    if not isinstance(params, dict):
        raise ValueError("algorithm params must be a dict or a list of dicts")
    if not params:
        return [{}]
    keys = list(params.keys())
    value_lists = [_parameter_values(key, params[key]) for key in keys]
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*value_lists)]


def _parameter_values(key: str, value: Any) -> list[Any]:
    if isinstance(value, list) and _is_vector_parameter(key, value):
        return [value]
    return value if isinstance(value, list) else [value]


def _is_vector_parameter(key: str, value: list[Any]) -> bool:
    vector_lengths = {
        "initial_linear_velocity": 3,
        "initial_angular_velocity": 3,
        "initial_velocity": 6,
    }
    expected = vector_lengths.get(key)
    if expected is None or len(value) != expected:
        return False
    return all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)


def parameter_slug(params: dict[str, Any]) -> str:
    if not params:
        return "default"
    parts = []
    for key in sorted(params):
        value = params[key]
        if _is_zero_vector_parameter(key, value):
            continue
        parts.append(f"{_slug_key(key)}-{_slug_value(value)}")
    return "_".join(parts) if parts else "default"


def _is_zero_vector_parameter(key: str, value: Any) -> bool:
    if not isinstance(value, list) or not _is_vector_parameter(key, value):
        return False
    return all(float(item) == 0.0 for item in value)


def _slug_key(key: str) -> str:
    return {
        "initial_linear_velocity": "ilv",
        "initial_angular_velocity": "iav",
        "initial_velocity": "iv",
    }.get(key, key)


def _slug_value(value: Any) -> str:
    if isinstance(value, list):
        return "-".join(_slug_scalar(item) for item in value)
    return _slug_scalar(value)


def _slug_scalar(value: Any) -> str:
    return str(value).replace(" ", "").replace("/", "_").replace("\\", "_")


def _visualization_options(payload: bool | dict[str, Any] | None) -> dict[str, Any]:
    defaults = {
        "enabled": True,
        "format": "vtu",
        "normal_axis": "z",
        "include_axes": True,
        "include_path_distance": True,
    }
    if payload is None:
        return defaults
    if isinstance(payload, bool):
        return {**defaults, "enabled": payload}
    if not isinstance(payload, dict):
        raise ValueError("visualization must be a bool or mapping")
    options = {**defaults, **payload}
    fmt = str(options["format"]).lower().lstrip(".")
    if fmt not in {"vtu", "vtk"}:
        raise ValueError("visualization.format must be vtu or vtk")
    options["format"] = fmt
    options["enabled"] = bool(options["enabled"])
    options["include_axes"] = bool(options["include_axes"])
    options["include_path_distance"] = bool(options["include_path_distance"])
    return options


def _write_input_artifacts(
    traj: Trajectory,
    input_dir: Path,
    *,
    visualization_options: dict[str, Any],
) -> dict[str, str]:
    input_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = input_dir / "trajectory.csv"
    write_trajectory(traj, trajectory_path)
    artifacts = {"trajectory_path": str(trajectory_path)}
    vtk_path = _vtk_path(input_dir, visualization_options)
    if vtk_path is not None:
        _write_vtk(traj, vtk_path, visualization_options)
        artifacts["vtk_path"] = str(vtk_path)
    return artifacts


def _vtk_path(directory: Path, visualization_options: dict[str, Any]) -> Path | None:
    if not visualization_options["enabled"]:
        return None
    return directory / f"trajectory.{visualization_options['format']}"


def _write_vtk(
    traj: Trajectory,
    path: Path,
    visualization_options: dict[str, Any],
) -> None:
    write_vtk_unstructured_grid(
        traj,
        path,
        normal_axis=visualization_options["normal_axis"],
        include_axes=visualization_options["include_axes"],
        include_path_distance=visualization_options["include_path_distance"],
    )


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("batch config must be a mapping")
    return payload


def _expand_paths(patterns: list[str], base_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        pattern_path = Path(pattern)
        search_pattern = str(pattern_path if pattern_path.is_absolute() else base_dir / pattern_path)
        matches = [Path(match) for match in glob.glob(search_pattern, recursive="**" in search_pattern)]
        if matches:
            paths.extend(sorted(matches))
        else:
            paths.append(Path(search_pattern))
    return paths


def _reference_map(payload: Any, base_dir: Path) -> dict[str, Path]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("references must be a mapping from input stem/path to reference path")
    refs = {}
    for key, value in payload.items():
        path = Path(value)
        refs[str(key)] = path if path.is_absolute() else base_dir / path
    return refs


def _load_reference(input_file: Path, references: dict[str, str | Path]) -> Trajectory | None:
    for key in (str(input_file), input_file.name, input_file.stem):
        if key in references:
            return read_trajectory(references[key])
    return None


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
