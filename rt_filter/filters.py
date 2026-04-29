from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import butter, savgol_filter, sosfiltfilt
from scipy.spatial.transform import Rotation

from rt_filter.se3 import (
    make_poses,
    relative_rotvecs,
    rotations_from_relative_rotvecs,
)
from rt_filter.io import read_trajectory, write_trajectory
from rt_filter.trajectory import Trajectory


ArrayF = NDArray[np.float64]
ArrayI = NDArray[np.int64]
FilterFunc = Callable[..., Trajectory]


@dataclass(frozen=True)
class FilterInfo:
    name: str
    description: str
    defaults: dict[str, Any]


@dataclass(frozen=True)
class TimedFilterRun:
    trajectory: Trajectory
    per_pose_time_ns: ArrayI
    total_time_ns: int


REPO_ROOT = Path(__file__).resolve().parents[1]
CPP_DEMO_ENV_VAR = "RT_FILTER_CPP_DEMO_EXE"
CPP_FILTER_SPECS: dict[str, dict[str, Any]] = {
    "butterworth_cpp": {
        "alias": "butterworth-cpp",
        "cpp_algorithm": "butterworth",
        "result_suffix": "butterworth_cpp",
        "description": (
            "Run the standalone C++ realtime Butterworth XYZ executable and import "
            "its trajectory and timing outputs for GUI analysis."
        ),
        "defaults": {
            "cutoff_hz": 20.0,
            "order": 2,
            "sample_rate_hz": 100.0,
        },
    },
    "butterworth_z_cpp": {
        "alias": "butterworth_z-cpp",
        "cpp_algorithm": "butterworth_z",
        "result_suffix": "butterworth_z_cpp",
        "description": (
            "Run the standalone C++ realtime Butterworth Z executable and import "
            "its trajectory and timing outputs for GUI analysis."
        ),
        "defaults": {
            "cutoff_hz": 20.0,
            "order": 2,
            "sample_rate_hz": 100.0,
        },
    },
    "one_euro_z_cpp": {
        "alias": "one_euro_z-cpp",
        "cpp_algorithm": "one_euro_z",
        "result_suffix": "one_euro_z_cpp",
        "description": (
            "Run the standalone C++ One Euro Z executable and import its trajectory "
            "and timing outputs for GUI analysis."
        ),
        "defaults": {
            "min_cutoff": 0.02,
            "beta": 6.0,
            "d_cutoff": 2.0,
            "derivative_deadband": 1.0,
            "sample_rate_hz": 80.0,
        },
    },
    "ukf_cpp": {
        "alias": "ukf-cpp",
        "cpp_algorithm": "ukf",
        "result_suffix": "ukf_cpp",
        "description": (
            "Run the standalone C++ UKF executable and import its trajectory and "
            "timing outputs for GUI analysis."
        ),
        "defaults": {
            "motion_model": "constant_velocity",
            "process_noise": 1000.0,
            "measurement_noise": 0.001,
            "initial_covariance": 1.0,
            "initial_linear_velocity": [0.0, 0.0, 0.0],
            "initial_angular_velocity": [0.0, 0.0, 0.0],
        },
    },
}


def available_filters() -> dict[str, FilterInfo]:
    filters = {
        "moving_average": FilterInfo(
            name="moving_average",
            description="Sliding-window translation average and SO(3) mean.",
            defaults={"window": 5},
        ),
        "savgol": FilterInfo(
            name="savgol",
            description="Savitzky-Golay smoothing on translation and relative rotation vectors.",
            defaults={"window": 9, "polyorder": 2},
        ),
        "exponential": FilterInfo(
            name="exponential",
            description="Causal exponential smoothing on SE(3).",
            defaults={"alpha": 0.25},
        ),
        "kalman_cv": FilterInfo(
            name="kalman_cv",
            description="Constant-velocity Kalman filtering for translation and relative rotation vectors.",
            defaults={"process_noise": 1e-4, "measurement_noise": 1e-2},
        ),
        "ukf": FilterInfo(
            name="ukf",
            description=(
                "Unscented Kalman filtering for translation and relative rotation "
                "vectors with constant-velocity or constant-acceleration motion."
            ),
            defaults={
                "motion_model": "constant_velocity",
                "process_noise": 1000.0,
                "measurement_noise": 0.001,
                "initial_covariance": 1.0,
                "initial_linear_velocity": [0.0, 0.0, 0.0],
                "initial_angular_velocity": [0.0, 0.0, 0.0],
            },
        ),
        "one_euro": FilterInfo(
            name="one_euro",
            description="Causal One Euro adaptive low-pass filtering on X/Y/Z translation.",
            defaults={
                "min_cutoff": 0.02,
                "beta": 6.0,
                "d_cutoff": 2.0,
                "derivative_deadband": 1.0,
                "sample_rate_hz": 80.0,
            },
        ),
        "one_euro_z": FilterInfo(
            name="one_euro_z",
            description="Causal One Euro adaptive low-pass filtering on Z translation only.",
            defaults={
                "min_cutoff": 0.02,
                "beta": 6.0,
                "d_cutoff": 2.0,
                "derivative_deadband": 1.0,
                "sample_rate_hz": 80.0,
            },
        ),
        "butterworth": FilterInfo(
            name="butterworth",
            description=(
                "Offline zero-phase Butterworth low-pass filtering on X/Y/Z translation "
                "to preserve the main waveform while suppressing higher-frequency noise."
            ),
            defaults={
                "cutoff_hz": 20.0,
                "order": 2,
                "sample_rate_hz": 100.0,
            },
        ),
        "butterworth_z": FilterInfo(
            name="butterworth_z",
            description=(
                "Offline zero-phase Butterworth low-pass filtering on Z translation only "
                "to preserve the main waveform while suppressing higher-frequency noise."
            ),
            defaults={
                "cutoff_hz": 20.0,
                "order": 2,
                "sample_rate_hz": 100.0,
            },
        ),
        "adaptive_kalman_z": FilterInfo(
            name="adaptive_kalman_z",
            description=(
                "Causal Z-only adaptive scalar Kalman filtering with robust innovation "
                "gating and optional motion-aware process-noise scaling."
            ),
            defaults={
                "process_noise": 1e-12,
                "measurement_noise": 1e-5,
                "initial_covariance": 1.0,
                "motion_process_gain": 0.0,
                "velocity_deadband": 1.0,
                "innovation_scale": 20.0,
                "innovation_gate": 2.5,
                "max_measurement_scale": 100.0,
                "sample_rate_hz": 80.0,
            },
        ),
    }
    filters.update(
        {
            spec["alias"]: FilterInfo(
                name=spec["alias"],
                description=str(spec["description"]),
                defaults=dict(spec["defaults"]),
            )
            for spec in CPP_FILTER_SPECS.values()
        }
    )
    return filters


def run_filter(
    name: str,
    trajectory: Trajectory,
    params: dict[str, Any] | None = None,
) -> Trajectory:
    normalized = name.lower().replace("-", "_")
    params = {} if params is None else dict(params)
    if _cpp_filter_spec(normalized) is not None:
        return _run_cpp_filter_timed(normalized, trajectory, params).trajectory
    registry: dict[str, FilterFunc] = {
        "moving_average": moving_average_filter,
        "savgol": savgol_filter_trajectory,
        "savitzky_golay": savgol_filter_trajectory,
        "exponential": exponential_filter,
        "kalman_cv": kalman_cv_filter,
        "ukf": ukf_filter,
        "ukf_cv": ukf_filter,
        "one_euro": one_euro_filter,
        "one_euro_xyz": one_euro_filter,
        "euro": one_euro_filter,
        "one_euro_z": one_euro_z_filter,
        "butterworth": butterworth_filter,
        "butterworth_xyz": butterworth_filter,
        "butterworth_z": butterworth_z_filter,
        "adaptive_kalman_z": adaptive_kalman_z_filter,
    }
    if normalized not in registry:
        known = ", ".join(sorted(available_filters()))
        raise ValueError(f"unknown filter '{name}', available filters: {known}")
    return registry[normalized](trajectory, **params)


def run_filter_timed(
    name: str,
    trajectory: Trajectory,
    params: dict[str, Any] | None = None,
) -> TimedFilterRun:
    normalized = name.lower().replace("-", "_")
    params = {} if params is None else dict(params)
    if _cpp_filter_spec(normalized) is not None:
        return _run_cpp_filter_timed(normalized, trajectory, params)
    registry: dict[str, Callable[..., TimedFilterRun]] = {
        "moving_average": _run_moving_average_timed,
        "savgol": _run_savgol_timed,
        "savitzky_golay": _run_savgol_timed,
        "exponential": _run_exponential_timed,
        "kalman_cv": _run_kalman_cv_timed,
        "ukf": _run_ukf_timed,
        "ukf_cv": _run_ukf_timed,
        "one_euro": _run_one_euro_timed,
        "one_euro_xyz": _run_one_euro_timed,
        "euro": _run_one_euro_timed,
        "one_euro_z": _run_one_euro_z_timed,
        "butterworth": _run_butterworth_timed,
        "butterworth_xyz": _run_butterworth_timed,
        "butterworth_z": _run_butterworth_z_timed,
        "adaptive_kalman_z": _run_adaptive_kalman_z_timed,
    }
    if normalized not in registry:
        known = ", ".join(sorted(available_filters()))
        raise ValueError(f"unknown filter '{name}', available filters: {known}")
    return registry[normalized](trajectory, **params)


def _cpp_filter_spec(name: str) -> dict[str, Any] | None:
    normalized = name.lower().replace("-", "_")
    return CPP_FILTER_SPECS.get(normalized)


def _run_cpp_filter_timed(
    name: str,
    traj: Trajectory,
    params: dict[str, Any],
) -> TimedFilterRun:
    spec = _cpp_filter_spec(name)
    if spec is None:
        raise ValueError(f"unknown C++ filter '{name}'")

    executable = _find_cpp_demo_executable()
    effective_params = dict(spec["defaults"])
    effective_params.update(params)

    with tempfile.TemporaryDirectory(prefix="rt_filter_cpp_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        input_path = tmp_dir / "input.csv"
        output_path = tmp_dir / "output.csv"
        write_trajectory(traj, input_path)

        command = [
            str(executable),
            "--algorithm",
            str(spec["cpp_algorithm"]),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
        command.extend(_cpp_filter_cli_args(str(spec["cpp_algorithm"]), effective_params))
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f"{spec['alias']} failed via {executable.name}: {detail or f'exit code {completed.returncode}'}"
            )

        timing_path = output_path.with_suffix(".timing.csv")
        metrics_path = output_path.with_suffix(".metrics.json")
        if not output_path.exists():
            raise RuntimeError(f"{spec['alias']} did not produce output: {output_path}")
        if not timing_path.exists():
            raise RuntimeError(f"{spec['alias']} did not produce timing output: {timing_path}")

        filtered = read_trajectory(output_path)
        if filtered.count != traj.count:
            raise RuntimeError(
                f"{spec['alias']} returned {filtered.count} poses for {traj.count} input poses"
            )
        per_pose_time_ns = _read_cpp_timing_series(timing_path, expected_count=filtered.count)
        metrics = _read_cpp_metrics(metrics_path)
        total_time_ns = int(metrics.get("compute_total_ns", per_pose_time_ns.sum(dtype=np.int64)))

        result = traj.copy_with(
            poses=filtered.poses,
            timestamps=filtered.timestamps if filtered.timestamps is not None else traj.timestamps,
            name=f"{traj.name}__{spec['result_suffix']}",
            metadata={
                "filter": spec["alias"],
                "params": effective_params,
                "backend": "cpp_demo",
                "cpp_demo_executable": str(executable),
                "cpp_timing_summary": _cpp_timing_summary(metrics),
            },
        )
        return TimedFilterRun(result, per_pose_time_ns, total_time_ns)


def _find_cpp_demo_executable() -> Path:
    searched: list[Path] = []
    candidates: list[Path] = []
    executable_name = "rt_filter_cpp_demo.exe" if sys.platform.startswith("win") else "rt_filter_cpp_demo"

    override = os.environ.get(CPP_DEMO_ENV_VAR)
    if override:
        candidates.append(Path(override).expanduser())

    if getattr(sys, "frozen", False):
        bundled_root = getattr(sys, "_MEIPASS", None)
        if bundled_root:
            candidates.append(Path(bundled_root) / executable_name)
        executable_path = Path(sys.executable).resolve()
        candidates.append(executable_path.parent / executable_name)
        if sys.platform == "darwin" and executable_path.parent.name == "MacOS":
            candidates.append(executable_path.parent.parent / "Resources" / executable_name)

    if sys.platform.startswith("win"):
        candidates.append(REPO_ROOT / "build" / "cpp_demo" / "windows-vs2022" / "Release" / executable_name)
    elif sys.platform == "darwin":
        candidates.append(REPO_ROOT / "build" / "cpp_demo" / "macos-xcode" / "Release" / executable_name)

    search_root = REPO_ROOT / "build" / "cpp_demo"
    if search_root.exists():
        candidates.extend(sorted(search_root.rglob(executable_name)))
        if executable_name != "rt_filter_cpp_demo":
            candidates.extend(sorted(search_root.rglob("rt_filter_cpp_demo")))
        if executable_name != "rt_filter_cpp_demo.exe":
            candidates.extend(sorted(search_root.rglob("rt_filter_cpp_demo.exe")))

    on_path = shutil.which("rt_filter_cpp_demo")
    if on_path:
        candidates.append(Path(on_path))

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        searched.append(resolved)
        if resolved.is_file():
            return resolved

    build_hint = (
        "py -3 scripts\\build_cpp_demo.py"
        if sys.platform.startswith("win")
        else "python3 scripts/build_cpp_demo.py"
    )
    searched_text = ", ".join(str(path) for path in searched) or "<none>"
    raise FileNotFoundError(
        "C++ filter executable `rt_filter_cpp_demo` was not found. "
        f"Searched: {searched_text}. Build it with `{build_hint}` or set `{CPP_DEMO_ENV_VAR}`."
    )


def cpp_demo_available() -> bool:
    try:
        _find_cpp_demo_executable()
    except FileNotFoundError:
        return False
    return True


def _cpp_filter_cli_args(cpp_algorithm: str, params: dict[str, Any]) -> list[str]:
    command: list[str] = []
    if "sample_rate_hz" in params:
        command.extend(["--sample-rate-hz", _format_cli_scalar(params["sample_rate_hz"])])
    if bool(params.get("strict_timestamps", False)):
        command.append("--strict-timestamps")

    if cpp_algorithm in {"butterworth", "butterworth_z"}:
        command.extend(
            [
                "--cutoff-hz",
                _format_cli_scalar(params.get("cutoff_hz", 20.0)),
                "--order",
                _format_cli_scalar(int(params.get("order", 2))),
            ]
        )
        return command

    if cpp_algorithm == "one_euro_z":
        command.extend(
            [
                "--min-cutoff",
                _format_cli_scalar(params.get("min_cutoff", 0.02)),
                "--beta",
                _format_cli_scalar(params.get("beta", 6.0)),
                "--d-cutoff",
                _format_cli_scalar(params.get("d_cutoff", 2.0)),
                "--derivative-deadband",
                _format_cli_scalar(params.get("derivative_deadband", 1.0)),
            ]
        )
        return command

    motion_model = str(params.get("motion_model", "constant_velocity")).lower().replace("-", "_")
    if motion_model in {"cv", "constant_velocity"}:
        canonical_motion_model = "constant_velocity"
    elif motion_model in {"ca", "constant_acceleration"}:
        canonical_motion_model = "constant_acceleration"
    else:
        raise ValueError("motion_model must be constant_velocity or constant_acceleration")
    command.extend(
        [
            "--motion-model",
            canonical_motion_model,
            "--process-noise",
            _format_cli_scalar(params.get("process_noise", 1000.0)),
            "--measurement-noise",
            _format_cli_scalar(params.get("measurement_noise", 0.001)),
            "--initial-covariance",
            _format_cli_scalar(params.get("initial_covariance", 1.0)),
        ]
    )

    for flag, key in (("--alpha", "alpha"), ("--ukf-beta", "beta"), ("--kappa", "kappa")):
        if key in params:
            command.extend([flag, _format_cli_scalar(params[key])])

    if params.get("initial_velocity") is not None:
        command.extend(
            [
                "--initial-velocity",
                _format_cli_vector(_validate_vector(params["initial_velocity"], 6, "initial_velocity")),
            ]
        )
        return command

    linear_velocity = params.get("initial_linear_velocity")
    angular_velocity = params.get("initial_angular_velocity")
    if linear_velocity is not None:
        command.extend(
            [
                "--initial-linear-velocity",
                _format_cli_vector(
                    _validate_vector(linear_velocity, 3, "initial_linear_velocity")
                ),
            ]
        )
    if angular_velocity is not None:
        command.extend(
            [
                "--initial-angular-velocity",
                _format_cli_vector(
                    _validate_vector(angular_velocity, 3, "initial_angular_velocity")
                ),
            ]
        )
    return command


def _read_cpp_timing_series(path: Path, *, expected_count: int) -> ArrayI:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        values = [int(row["compute_time_ns"]) for row in reader]
    per_pose_time_ns = np.asarray(values, dtype=np.int64)
    if per_pose_time_ns.shape != (expected_count,):
        raise RuntimeError(
            f"unexpected timing series length from {path}: got {per_pose_time_ns.shape[0]}, "
            f"expected {expected_count}"
        )
    return per_pose_time_ns


def _read_cpp_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _cpp_timing_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    summary_keys = (
        "frames",
        "has_timestamps",
        "compute_total_ns",
        "compute_total_ms",
        "compute_mean_us",
        "compute_p95_us",
        "compute_max_us",
    )
    return {key: metrics[key] for key in summary_keys if key in metrics}


def _format_cli_scalar(value: Any) -> str:
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.17g}"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    return str(value)


def _format_cli_vector(values: ArrayLike) -> str:
    vector = np.asarray(values, dtype=float).reshape(-1)
    return ",".join(f"{float(value):.17g}" for value in vector)


def moving_average_filter(traj: Trajectory, window: int = 5) -> Trajectory:
    result, _, _ = _moving_average_filter_impl(traj, window=window, collect_timing=False)
    return result


def _run_moving_average_timed(traj: Trajectory, window: int = 5) -> TimedFilterRun:
    result, per_pose_time_ns, total_time_ns = _moving_average_filter_impl(
        traj,
        window=window,
        collect_timing=True,
    )
    assert per_pose_time_ns is not None
    return TimedFilterRun(result, per_pose_time_ns, total_time_ns)


def _moving_average_filter_impl(
    traj: Trajectory,
    *,
    window: int = 5,
    collect_timing: bool,
) -> tuple[Trajectory, ArrayI | None, int]:
    window = _validate_window(window, traj.count, odd=False)
    total_start_ns = perf_counter_ns() if collect_timing else 0
    per_pose_time_ns = np.zeros(traj.count, dtype=np.int64) if collect_timing else None
    positions = _moving_average(traj.positions, window)
    rotations = traj.rotations
    smoothed_rotations = []
    half = window // 2
    for idx in range(traj.count):
        step_start_ns = perf_counter_ns() if per_pose_time_ns is not None else 0
        start = max(0, idx - half)
        stop = min(traj.count, idx + half + 1)
        smoothed_rotations.append(rotations[start:stop].mean())
        if per_pose_time_ns is not None:
            per_pose_time_ns[idx] += perf_counter_ns() - step_start_ns
    result = traj.copy_with(
        poses=make_poses(positions, Rotation.concatenate(smoothed_rotations)),
        name=f"{traj.name}__moving_average",
        metadata={"filter": "moving_average", "params": {"window": window}},
    )
    if per_pose_time_ns is None:
        return result, None, 0
    total_time_ns = perf_counter_ns() - total_start_ns
    return result, _finalize_time_series(per_pose_time_ns, total_time_ns), total_time_ns


def savgol_filter_trajectory(
    traj: Trajectory,
    window: int = 9,
    polyorder: int = 2,
    mode: str = "interp",
) -> Trajectory:
    result, _, _ = _savgol_filter_trajectory_impl(
        traj,
        window=window,
        polyorder=polyorder,
        mode=mode,
        collect_timing=False,
    )
    return result


def _run_savgol_timed(
    traj: Trajectory,
    window: int = 9,
    polyorder: int = 2,
    mode: str = "interp",
) -> TimedFilterRun:
    result, per_pose_time_ns, total_time_ns = _savgol_filter_trajectory_impl(
        traj,
        window=window,
        polyorder=polyorder,
        mode=mode,
        collect_timing=True,
    )
    assert per_pose_time_ns is not None
    return TimedFilterRun(result, per_pose_time_ns, total_time_ns)


def _savgol_filter_trajectory_impl(
    traj: Trajectory,
    *,
    window: int = 9,
    polyorder: int = 2,
    mode: str = "interp",
    collect_timing: bool,
) -> tuple[Trajectory, ArrayI | None, int]:
    total_start_ns = perf_counter_ns() if collect_timing else 0
    if traj.count <= polyorder + 1:
        result = traj.copy_with(
            name=f"{traj.name}__savgol",
            metadata={
                "filter": "savgol",
                "params": {"window": window, "polyorder": polyorder, "mode": mode},
                "warning": "trajectory too short; returned unchanged",
            },
        )
        if not collect_timing:
            return result, None, 0
        total_time_ns = perf_counter_ns() - total_start_ns
        return result, _uniform_time_series(traj.count, total_time_ns), total_time_ns
    window = _validate_window(window, traj.count, odd=True)
    if polyorder >= window:
        raise ValueError("polyorder must be smaller than window")

    positions = savgol_filter(traj.positions, window, polyorder, axis=0, mode=mode)
    reference = traj.rotations[0]
    rotvecs = relative_rotvecs(traj.rotations, reference=reference)
    smoothed_rotvecs = savgol_filter(rotvecs, window, polyorder, axis=0, mode=mode)
    rotations = rotations_from_relative_rotvecs(smoothed_rotvecs, reference)
    result = traj.copy_with(
        poses=make_poses(positions, rotations),
        name=f"{traj.name}__savgol",
        metadata={
            "filter": "savgol",
            "params": {"window": window, "polyorder": polyorder, "mode": mode},
        },
    )
    if not collect_timing:
        return result, None, 0
    total_time_ns = perf_counter_ns() - total_start_ns
    return result, _uniform_time_series(traj.count, total_time_ns), total_time_ns


def exponential_filter(traj: Trajectory, alpha: float = 0.25) -> Trajectory:
    result, _, _ = _exponential_filter_impl(traj, alpha=alpha, collect_timing=False)
    return result


def _run_exponential_timed(traj: Trajectory, alpha: float = 0.25) -> TimedFilterRun:
    result, per_pose_time_ns, total_time_ns = _exponential_filter_impl(
        traj,
        alpha=alpha,
        collect_timing=True,
    )
    assert per_pose_time_ns is not None
    return TimedFilterRun(result, per_pose_time_ns, total_time_ns)


def _exponential_filter_impl(
    traj: Trajectory,
    *,
    alpha: float = 0.25,
    collect_timing: bool,
) -> tuple[Trajectory, ArrayI | None, int]:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    total_start_ns = perf_counter_ns() if collect_timing else 0
    per_pose_time_ns = np.zeros(traj.count, dtype=np.int64) if collect_timing else None
    init_start_ns = perf_counter_ns() if per_pose_time_ns is not None else 0
    positions = np.empty_like(traj.positions)
    positions[0] = traj.positions[0]
    input_rotations = traj.rotations
    output_rotations = [input_rotations[0]]
    if per_pose_time_ns is not None:
        per_pose_time_ns[0] += perf_counter_ns() - init_start_ns
    for idx in range(1, traj.count):
        step_start_ns = perf_counter_ns() if per_pose_time_ns is not None else 0
        positions[idx] = (1.0 - alpha) * positions[idx - 1] + alpha * traj.positions[idx]
        delta = output_rotations[-1].inv() * input_rotations[idx]
        output_rotations.append(output_rotations[-1] * Rotation.from_rotvec(alpha * delta.as_rotvec()))
        if per_pose_time_ns is not None:
            per_pose_time_ns[idx] += perf_counter_ns() - step_start_ns
    result = traj.copy_with(
        poses=make_poses(positions, Rotation.concatenate(output_rotations)),
        name=f"{traj.name}__exponential",
        metadata={"filter": "exponential", "params": {"alpha": alpha}},
    )
    if per_pose_time_ns is None:
        return result, None, 0
    total_time_ns = perf_counter_ns() - total_start_ns
    return result, _finalize_time_series(per_pose_time_ns, total_time_ns), total_time_ns


def kalman_cv_filter(
    traj: Trajectory,
    process_noise: float = 1e-4,
    measurement_noise: float = 1e-2,
    initial_covariance: float = 1.0,
) -> Trajectory:
    result, _, _ = _kalman_cv_filter_impl(
        traj,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        initial_covariance=initial_covariance,
        collect_timing=False,
    )
    return result


def _run_kalman_cv_timed(
    traj: Trajectory,
    process_noise: float = 1e-4,
    measurement_noise: float = 1e-2,
    initial_covariance: float = 1.0,
) -> TimedFilterRun:
    result, per_pose_time_ns, total_time_ns = _kalman_cv_filter_impl(
        traj,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        initial_covariance=initial_covariance,
        collect_timing=True,
    )
    assert per_pose_time_ns is not None
    return TimedFilterRun(result, per_pose_time_ns, total_time_ns)


def _kalman_cv_filter_impl(
    traj: Trajectory,
    *,
    process_noise: float = 1e-4,
    measurement_noise: float = 1e-2,
    initial_covariance: float = 1.0,
    collect_timing: bool,
) -> tuple[Trajectory, ArrayI | None, int]:
    if process_noise <= 0 or measurement_noise <= 0 or initial_covariance <= 0:
        raise ValueError("noise and covariance parameters must be positive")
    total_start_ns = perf_counter_ns() if collect_timing else 0
    position_result = _kalman_constant_velocity_impl(
        traj.positions,
        timestamps=traj.timestamps,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        initial_covariance=initial_covariance,
        collect_timing=collect_timing,
    )
    if collect_timing:
        positions, position_time_ns, _ = position_result
    else:
        positions = position_result
    reference = traj.rotations[0]
    rotvecs = relative_rotvecs(traj.rotations, reference=reference)
    rotation_result = _kalman_constant_velocity_impl(
        rotvecs,
        timestamps=traj.timestamps,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        initial_covariance=initial_covariance,
        collect_timing=collect_timing,
    )
    if collect_timing:
        smoothed_rotvecs, rotation_time_ns, _ = rotation_result
    else:
        smoothed_rotvecs = rotation_result
    rotations = rotations_from_relative_rotvecs(smoothed_rotvecs, reference)
    result = traj.copy_with(
        poses=make_poses(positions, rotations),
        name=f"{traj.name}__kalman_cv",
        metadata={
            "filter": "kalman_cv",
            "params": {
                "process_noise": process_noise,
                "measurement_noise": measurement_noise,
                "initial_covariance": initial_covariance,
            },
        },
    )
    if not collect_timing:
        return result, None, 0
    assert position_time_ns is not None
    assert rotation_time_ns is not None
    total_time_ns = perf_counter_ns() - total_start_ns
    return (
        result,
        _finalize_time_series(position_time_ns + rotation_time_ns, total_time_ns),
        total_time_ns,
    )


def ukf_filter(
    traj: Trajectory,
    motion_model: str = "constant_velocity",
    process_noise: float = 1000.0,
    measurement_noise: float = 0.001,
    initial_covariance: float = 1.0,
    initial_velocity: ArrayLike | None = None,
    initial_linear_velocity: ArrayLike | None = None,
    initial_angular_velocity: ArrayLike | None = None,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
    sample_rate_hz: float = 100.0,
) -> Trajectory:
    result, _, _ = _ukf_filter_impl(
        traj,
        motion_model=motion_model,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        initial_covariance=initial_covariance,
        initial_velocity=initial_velocity,
        initial_linear_velocity=initial_linear_velocity,
        initial_angular_velocity=initial_angular_velocity,
        alpha=alpha,
        beta=beta,
        kappa=kappa,
        sample_rate_hz=sample_rate_hz,
        collect_timing=False,
    )
    return result


def _run_ukf_timed(
    traj: Trajectory,
    motion_model: str = "constant_velocity",
    process_noise: float = 1000.0,
    measurement_noise: float = 0.001,
    initial_covariance: float = 1.0,
    initial_velocity: ArrayLike | None = None,
    initial_linear_velocity: ArrayLike | None = None,
    initial_angular_velocity: ArrayLike | None = None,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
    sample_rate_hz: float = 100.0,
) -> TimedFilterRun:
    result, per_pose_time_ns, total_time_ns = _ukf_filter_impl(
        traj,
        motion_model=motion_model,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        initial_covariance=initial_covariance,
        initial_velocity=initial_velocity,
        initial_linear_velocity=initial_linear_velocity,
        initial_angular_velocity=initial_angular_velocity,
        alpha=alpha,
        beta=beta,
        kappa=kappa,
        sample_rate_hz=sample_rate_hz,
        collect_timing=True,
    )
    assert per_pose_time_ns is not None
    return TimedFilterRun(result, per_pose_time_ns, total_time_ns)


def _ukf_filter_impl(
    traj: Trajectory,
    *,
    motion_model: str = "constant_velocity",
    process_noise: float = 1000.0,
    measurement_noise: float = 0.001,
    initial_covariance: float = 1.0,
    initial_velocity: ArrayLike | None = None,
    initial_linear_velocity: ArrayLike | None = None,
    initial_angular_velocity: ArrayLike | None = None,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
    sample_rate_hz: float = 100.0,
    collect_timing: bool,
) -> tuple[Trajectory, ArrayI | None, int]:
    """Unscented Kalman filter over translation and relative rotation vectors."""

    model = motion_model.lower().replace("-", "_")
    if model not in {"constant_velocity", "cv", "constant_acceleration", "ca"}:
        raise ValueError("motion_model must be constant_velocity or constant_acceleration")
    if (
        process_noise <= 0
        or measurement_noise <= 0
        or initial_covariance <= 0
        or alpha <= 0
        or sample_rate_hz <= 0
    ):
        raise ValueError(
            "process_noise, measurement_noise, initial_covariance, alpha, "
            "and sample_rate_hz must be positive"
        )
    total_start_ns = perf_counter_ns() if collect_timing else 0
    initial_motion = _ukf_initial_motion_vector(
        initial_velocity,
        initial_linear_velocity,
        initial_angular_velocity,
    )

    reference = traj.rotations[0]
    measurements = np.column_stack(
        [traj.positions, relative_rotvecs(traj.rotations, reference=reference)]
    )
    filtered_result = _ukf_filter_measurements(
        measurements,
        timestamps=traj.timestamps,
        motion_model=model,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        initial_covariance=initial_covariance,
        alpha=alpha,
        beta=beta,
        kappa=kappa,
        sample_rate_hz=sample_rate_hz,
        initial_velocity=initial_motion,
        collect_timing=collect_timing,
    )
    if collect_timing:
        filtered, per_pose_time_ns, _ = filtered_result
    else:
        filtered = filtered_result
    positions = filtered[:, :3]
    rotations = rotations_from_relative_rotvecs(filtered[:, 3:6], reference)
    canonical_model = (
        "constant_acceleration" if model in {"constant_acceleration", "ca"} else "constant_velocity"
    )
    result = traj.copy_with(
        poses=make_poses(positions, rotations),
        name=f"{traj.name}__ukf",
        metadata={
            "filter": "ukf",
            "params": {
                "motion_model": canonical_model,
                "process_noise": process_noise,
                "measurement_noise": measurement_noise,
                "initial_covariance": initial_covariance,
                "initial_velocity": initial_motion.tolist(),
                "alpha": alpha,
                "beta": beta,
                "kappa": kappa,
                "sample_rate_hz": sample_rate_hz,
            },
        },
    )
    if not collect_timing:
        return result, None, 0
    assert per_pose_time_ns is not None
    total_time_ns = perf_counter_ns() - total_start_ns
    return result, _finalize_time_series(per_pose_time_ns, total_time_ns), total_time_ns


def one_euro_filter(
    traj: Trajectory,
    min_cutoff: float = 0.02,
    beta: float = 6.0,
    d_cutoff: float = 2.0,
    derivative_deadband: float = 1.0,
    sample_rate_hz: float = 80.0,
) -> Trajectory:
    result, _, _ = _one_euro_translation_filter_impl(
        traj,
        axis_indices=(0, 1, 2),
        filter_name="one_euro",
        name_suffix="one_euro",
        min_cutoff=min_cutoff,
        beta=beta,
        d_cutoff=d_cutoff,
        derivative_deadband=derivative_deadband,
        sample_rate_hz=sample_rate_hz,
        collect_timing=False,
    )
    return result


def one_euro_z_filter(
    traj: Trajectory,
    min_cutoff: float = 0.02,
    beta: float = 6.0,
    d_cutoff: float = 2.0,
    derivative_deadband: float = 1.0,
    sample_rate_hz: float = 80.0,
) -> Trajectory:
    result, _, _ = _one_euro_translation_filter_impl(
        traj,
        axis_indices=(2,),
        filter_name="one_euro_z",
        name_suffix="one_euro_z",
        min_cutoff=min_cutoff,
        beta=beta,
        d_cutoff=d_cutoff,
        derivative_deadband=derivative_deadband,
        sample_rate_hz=sample_rate_hz,
        collect_timing=False,
    )
    return result


def _run_one_euro_timed(
    traj: Trajectory,
    min_cutoff: float = 0.02,
    beta: float = 6.0,
    d_cutoff: float = 2.0,
    derivative_deadband: float = 1.0,
    sample_rate_hz: float = 80.0,
) -> TimedFilterRun:
    result, per_pose_time_ns, total_time_ns = _one_euro_translation_filter_impl(
        traj,
        axis_indices=(0, 1, 2),
        filter_name="one_euro",
        name_suffix="one_euro",
        min_cutoff=min_cutoff,
        beta=beta,
        d_cutoff=d_cutoff,
        derivative_deadband=derivative_deadband,
        sample_rate_hz=sample_rate_hz,
        collect_timing=True,
    )
    assert per_pose_time_ns is not None
    return TimedFilterRun(result, per_pose_time_ns, total_time_ns)


def _run_one_euro_z_timed(
    traj: Trajectory,
    min_cutoff: float = 0.02,
    beta: float = 6.0,
    d_cutoff: float = 2.0,
    derivative_deadband: float = 1.0,
    sample_rate_hz: float = 80.0,
) -> TimedFilterRun:
    result, per_pose_time_ns, total_time_ns = _one_euro_translation_filter_impl(
        traj,
        axis_indices=(2,),
        filter_name="one_euro_z",
        name_suffix="one_euro_z",
        min_cutoff=min_cutoff,
        beta=beta,
        d_cutoff=d_cutoff,
        derivative_deadband=derivative_deadband,
        sample_rate_hz=sample_rate_hz,
        collect_timing=True,
    )
    assert per_pose_time_ns is not None
    return TimedFilterRun(result, per_pose_time_ns, total_time_ns)


def _one_euro_translation_filter_impl(
    traj: Trajectory,
    *,
    axis_indices: tuple[int, ...],
    filter_name: str,
    name_suffix: str,
    min_cutoff: float = 0.02,
    beta: float = 6.0,
    d_cutoff: float = 2.0,
    derivative_deadband: float = 1.0,
    sample_rate_hz: float = 80.0,
    collect_timing: bool,
) -> tuple[Trajectory, ArrayI | None, int]:
    """Causal adaptive low-pass filter for selected translation channels."""

    if (
        min_cutoff <= 0
        or beta < 0
        or d_cutoff <= 0
        or derivative_deadband < 0
        or sample_rate_hz <= 0
    ):
        raise ValueError(
            "min_cutoff, d_cutoff, and sample_rate_hz must be positive; "
            "beta and derivative_deadband must be >= 0"
        )

    total_start_ns = perf_counter_ns() if collect_timing else 0
    positions = traj.positions.copy()
    per_pose_time_ns: ArrayI | None = None
    for axis in axis_indices:
        axis_result = _one_euro_filter_1d(
            positions[:, axis],
            timestamps=traj.timestamps,
            min_cutoff=min_cutoff,
            beta=beta,
            d_cutoff=d_cutoff,
            derivative_deadband=derivative_deadband,
            sample_rate_hz=sample_rate_hz,
            collect_timing=collect_timing,
        )
        if collect_timing:
            filtered_axis, axis_time_ns, _ = axis_result
            per_pose_time_ns = (
                axis_time_ns.copy()
                if per_pose_time_ns is None
                else per_pose_time_ns + axis_time_ns
            )
        else:
            filtered_axis = axis_result
        positions[:, axis] = filtered_axis
    result = traj.copy_with(
        poses=make_poses(positions, traj.rotations),
        name=f"{traj.name}__{name_suffix}",
        metadata={
            "filter": filter_name,
            "params": {
                "min_cutoff": min_cutoff,
                "beta": beta,
                "d_cutoff": d_cutoff,
                "derivative_deadband": derivative_deadband,
                "sample_rate_hz": sample_rate_hz,
                "filtered_axes": _translation_axis_labels(axis_indices),
            },
        },
    )
    if not collect_timing:
        return result, None, 0
    assert per_pose_time_ns is not None
    total_time_ns = perf_counter_ns() - total_start_ns
    return result, _finalize_time_series(per_pose_time_ns, total_time_ns), total_time_ns


def adaptive_kalman_z_filter(
    traj: Trajectory,
    process_noise: float = 1e-12,
    measurement_noise: float = 1e-5,
    initial_covariance: float = 1.0,
    motion_process_gain: float = 0.0,
    velocity_deadband: float = 1.0,
    innovation_scale: float = 20.0,
    innovation_gate: float = 2.5,
    max_measurement_scale: float = 100.0,
    sample_rate_hz: float = 80.0,
) -> Trajectory:
    result, _, _ = _adaptive_kalman_z_filter_impl(
        traj,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        initial_covariance=initial_covariance,
        motion_process_gain=motion_process_gain,
        velocity_deadband=velocity_deadband,
        innovation_scale=innovation_scale,
        innovation_gate=innovation_gate,
        max_measurement_scale=max_measurement_scale,
        sample_rate_hz=sample_rate_hz,
        collect_timing=False,
    )
    return result


def butterworth_filter(
    traj: Trajectory,
    cutoff_hz: float = 20.0,
    order: int = 2,
    sample_rate_hz: float = 100.0,
) -> Trajectory:
    result, _, _ = _butterworth_translation_filter_impl(
        traj,
        axis_indices=(0, 1, 2),
        filter_name="butterworth",
        name_suffix="butterworth",
        cutoff_hz=cutoff_hz,
        order=order,
        sample_rate_hz=sample_rate_hz,
        collect_timing=False,
    )
    return result


def butterworth_z_filter(
    traj: Trajectory,
    cutoff_hz: float = 20.0,
    order: int = 2,
    sample_rate_hz: float = 100.0,
) -> Trajectory:
    result, _, _ = _butterworth_translation_filter_impl(
        traj,
        axis_indices=(2,),
        filter_name="butterworth_z",
        name_suffix="butterworth_z",
        cutoff_hz=cutoff_hz,
        order=order,
        sample_rate_hz=sample_rate_hz,
        collect_timing=False,
    )
    return result


def _run_butterworth_timed(
    traj: Trajectory,
    cutoff_hz: float = 20.0,
    order: int = 2,
    sample_rate_hz: float = 100.0,
) -> TimedFilterRun:
    result, per_pose_time_ns, total_time_ns = _butterworth_translation_filter_impl(
        traj,
        axis_indices=(0, 1, 2),
        filter_name="butterworth",
        name_suffix="butterworth",
        cutoff_hz=cutoff_hz,
        order=order,
        sample_rate_hz=sample_rate_hz,
        collect_timing=True,
    )
    assert per_pose_time_ns is not None
    return TimedFilterRun(result, per_pose_time_ns, total_time_ns)


def _run_butterworth_z_timed(
    traj: Trajectory,
    cutoff_hz: float = 20.0,
    order: int = 2,
    sample_rate_hz: float = 100.0,
) -> TimedFilterRun:
    result, per_pose_time_ns, total_time_ns = _butterworth_translation_filter_impl(
        traj,
        axis_indices=(2,),
        filter_name="butterworth_z",
        name_suffix="butterworth_z",
        cutoff_hz=cutoff_hz,
        order=order,
        sample_rate_hz=sample_rate_hz,
        collect_timing=True,
    )
    assert per_pose_time_ns is not None
    return TimedFilterRun(result, per_pose_time_ns, total_time_ns)


def _butterworth_translation_filter_impl(
    traj: Trajectory,
    *,
    axis_indices: tuple[int, ...],
    filter_name: str,
    name_suffix: str,
    cutoff_hz: float = 20.0,
    order: int = 2,
    sample_rate_hz: float = 100.0,
    collect_timing: bool,
) -> tuple[Trajectory, ArrayI | None, int]:
    """Offline zero-phase Butterworth low-pass filter for selected translation channels."""

    order = int(order)
    if cutoff_hz <= 0.0 or sample_rate_hz <= 0.0 or order < 1:
        raise ValueError("cutoff_hz and sample_rate_hz must be positive; order must be >= 1")

    total_start_ns = perf_counter_ns() if collect_timing else 0
    positions = traj.positions.copy()
    effective_sample_rate_hz = _effective_sample_rate_hz(
        traj.timestamps,
        positions.shape[0],
        sample_rate_hz,
    )
    for axis in axis_indices:
        positions[:, axis] = _zero_phase_butterworth_filter_1d(
            positions[:, axis],
            cutoff_hz=cutoff_hz,
            order=order,
            sample_rate_hz=effective_sample_rate_hz,
        )
    result = traj.copy_with(
        poses=make_poses(positions, traj.rotations),
        name=f"{traj.name}__{name_suffix}",
        metadata={
            "filter": filter_name,
            "params": {
                "cutoff_hz": cutoff_hz,
                "order": order,
                "sample_rate_hz": sample_rate_hz,
                "effective_sample_rate_hz": effective_sample_rate_hz,
                "filtered_axes": _translation_axis_labels(axis_indices),
            },
        },
    )
    if not collect_timing:
        return result, None, 0
    total_time_ns = perf_counter_ns() - total_start_ns
    return result, _uniform_time_series(traj.count, total_time_ns), total_time_ns


def _run_adaptive_kalman_z_timed(
    traj: Trajectory,
    process_noise: float = 1e-12,
    measurement_noise: float = 1e-5,
    initial_covariance: float = 1.0,
    motion_process_gain: float = 0.0,
    velocity_deadband: float = 1.0,
    innovation_scale: float = 20.0,
    innovation_gate: float = 2.5,
    max_measurement_scale: float = 100.0,
    sample_rate_hz: float = 80.0,
) -> TimedFilterRun:
    result, per_pose_time_ns, total_time_ns = _adaptive_kalman_z_filter_impl(
        traj,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        initial_covariance=initial_covariance,
        motion_process_gain=motion_process_gain,
        velocity_deadband=velocity_deadband,
        innovation_scale=innovation_scale,
        innovation_gate=innovation_gate,
        max_measurement_scale=max_measurement_scale,
        sample_rate_hz=sample_rate_hz,
        collect_timing=True,
    )
    assert per_pose_time_ns is not None
    return TimedFilterRun(result, per_pose_time_ns, total_time_ns)


def _adaptive_kalman_z_filter_impl(
    traj: Trajectory,
    *,
    process_noise: float = 1e-12,
    measurement_noise: float = 1e-5,
    initial_covariance: float = 1.0,
    motion_process_gain: float = 0.0,
    velocity_deadband: float = 1.0,
    innovation_scale: float = 20.0,
    innovation_gate: float = 2.5,
    max_measurement_scale: float = 100.0,
    sample_rate_hz: float = 80.0,
    collect_timing: bool,
) -> tuple[Trajectory, ArrayI | None, int]:
    """Causal adaptive scalar Kalman filter for the Z translation channel only.

    The filter keeps X/Y translation and orientation untouched, and focuses on
    making Z more robust in the presence of slow drift, outliers, and strongly
    anisotropic depth noise. The current default is tuned for static scenes and
    behaves like a robust random-walk Kalman smoother on Z.
    """

    if (
        process_noise <= 0.0
        or measurement_noise <= 0.0
        or initial_covariance <= 0.0
        or motion_process_gain < 0.0
        or velocity_deadband < 0.0
        or innovation_scale < 0.0
        or innovation_gate <= 0.0
        or max_measurement_scale < 1.0
        or sample_rate_hz <= 0.0
    ):
        raise ValueError(
            "process_noise, measurement_noise, initial_covariance, innovation_gate, "
            "max_measurement_scale, and sample_rate_hz must be positive; "
            "motion_process_gain, velocity_deadband, and innovation_scale must be >= 0"
        )

    total_start_ns = perf_counter_ns() if collect_timing else 0
    positions = traj.positions.copy()
    z_result = _adaptive_kalman_filter_1d(
        positions[:, 2],
        timestamps=traj.timestamps,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        initial_covariance=initial_covariance,
        motion_process_gain=motion_process_gain,
        velocity_deadband=velocity_deadband,
        innovation_scale=innovation_scale,
        innovation_gate=innovation_gate,
        max_measurement_scale=max_measurement_scale,
        sample_rate_hz=sample_rate_hz,
        collect_timing=collect_timing,
    )
    if collect_timing:
        filtered_z, per_pose_time_ns, _ = z_result
    else:
        filtered_z = z_result
    positions[:, 2] = filtered_z
    result = traj.copy_with(
        poses=make_poses(positions, traj.rotations),
        name=f"{traj.name}__adaptive_kalman_z",
        metadata={
            "filter": "adaptive_kalman_z",
            "params": {
                "process_noise": process_noise,
                "measurement_noise": measurement_noise,
                "initial_covariance": initial_covariance,
                "motion_process_gain": motion_process_gain,
                "velocity_deadband": velocity_deadband,
                "innovation_scale": innovation_scale,
                "innovation_gate": innovation_gate,
                "max_measurement_scale": max_measurement_scale,
                "sample_rate_hz": sample_rate_hz,
            },
        },
    )
    if not collect_timing:
        return result, None, 0
    assert per_pose_time_ns is not None
    total_time_ns = perf_counter_ns() - total_start_ns
    return result, _finalize_time_series(per_pose_time_ns, total_time_ns), total_time_ns


def _moving_average(values: ArrayLike, window: int) -> ArrayF:
    arr = np.asarray(values, dtype=float)
    if window <= 1:
        return arr.copy()
    left = window // 2
    right = window - 1 - left
    padded = np.pad(arr, ((left, right), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.vstack(
        [np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(arr.shape[1])]
    ).T


def _validate_window(window: int, count: int, *, odd: bool) -> int:
    window = int(window)
    if window < 1:
        raise ValueError("window must be positive")
    if window > count:
        window = count
    if odd and window % 2 == 0:
        window = max(1, window - 1)
    return window


def _translation_axis_labels(axis_indices: tuple[int, ...]) -> list[str]:
    labels = ("x", "y", "z")
    return [labels[index] for index in axis_indices]


def _one_euro_filter_1d(
    values: ArrayLike,
    *,
    timestamps: ArrayLike | None,
    min_cutoff: float,
    beta: float,
    d_cutoff: float,
    derivative_deadband: float,
    sample_rate_hz: float,
    collect_timing: bool = False,
) -> ArrayF | tuple[ArrayF, ArrayI, int]:
    total_start_ns = perf_counter_ns() if collect_timing else 0
    measurements = np.asarray(values, dtype=float)
    if measurements.ndim != 1:
        raise ValueError("one euro filter values must be one-dimensional")
    if measurements.shape[0] <= 1:
        result = measurements.copy()
        if not collect_timing:
            return result
        total_time_ns = perf_counter_ns() - total_start_ns
        return result, _uniform_time_series(result.shape[0], total_time_ns), total_time_ns

    result = np.empty_like(measurements)
    per_pose_time_ns = np.zeros(measurements.shape[0], dtype=np.int64) if collect_timing else None
    init_start_ns = perf_counter_ns() if per_pose_time_ns is not None else 0
    result[0] = measurements[0]
    derivative_hat = 0.0
    if timestamps is not None:
        time_values = np.asarray(timestamps, dtype=float)
        if time_values.ndim != 1 or time_values.shape[0] != measurements.shape[0]:
            raise ValueError("timestamps must be a 1-D array with the same length as values")
        dt_values = np.diff(time_values)
        if np.any(dt_values <= 0):
            raise ValueError("timestamps must be strictly increasing")
    else:
        dt_values = np.full(measurements.shape[0] - 1, 1.0 / sample_rate_hz, dtype=float)
    if per_pose_time_ns is not None:
        per_pose_time_ns[0] += perf_counter_ns() - init_start_ns

    for idx, dt in enumerate(dt_values, start=1):
        step_start_ns = perf_counter_ns() if per_pose_time_ns is not None else 0
        derivative = (measurements[idx] - measurements[idx - 1]) / dt
        derivative_alpha = _lowpass_alpha(d_cutoff, dt)
        derivative_hat = derivative_alpha * derivative + (1.0 - derivative_alpha) * derivative_hat
        effective_derivative = max(abs(derivative_hat) - derivative_deadband, 0.0)
        cutoff = min_cutoff + beta * effective_derivative
        value_alpha = _lowpass_alpha(cutoff, dt)
        result[idx] = value_alpha * measurements[idx] + (1.0 - value_alpha) * result[idx - 1]
        if per_pose_time_ns is not None:
            per_pose_time_ns[idx] += perf_counter_ns() - step_start_ns
    if per_pose_time_ns is None:
        return result
    total_time_ns = perf_counter_ns() - total_start_ns
    return result, _finalize_time_series(per_pose_time_ns, total_time_ns), total_time_ns


def _adaptive_kalman_filter_1d(
    values: ArrayLike,
    *,
    timestamps: ArrayLike | None,
    process_noise: float,
    measurement_noise: float,
    initial_covariance: float,
    motion_process_gain: float,
    velocity_deadband: float,
    innovation_scale: float,
    innovation_gate: float,
    max_measurement_scale: float,
    sample_rate_hz: float,
    collect_timing: bool = False,
) -> ArrayF | tuple[ArrayF, ArrayI, int]:
    total_start_ns = perf_counter_ns() if collect_timing else 0
    measurements = np.asarray(values, dtype=float)
    if measurements.ndim != 1:
        raise ValueError("adaptive Kalman filter values must be one-dimensional")
    count = measurements.shape[0]
    if count <= 1:
        result = measurements.copy()
        if not collect_timing:
            return result
        total_time_ns = perf_counter_ns() - total_start_ns
        return result, _uniform_time_series(count, total_time_ns), total_time_ns

    dt_values = _dt_values(timestamps, count, sample_rate_hz)
    result = np.empty_like(measurements)
    per_pose_time_ns = np.zeros(count, dtype=np.int64) if collect_timing else None
    init_start_ns = perf_counter_ns() if per_pose_time_ns is not None else 0
    result[0] = measurements[0]
    state = float(measurements[0])
    covariance = float(initial_covariance)
    if per_pose_time_ns is not None:
        per_pose_time_ns[0] += perf_counter_ns() - init_start_ns

    for idx, dt in enumerate(dt_values, start=1):
        step_start_ns = perf_counter_ns() if per_pose_time_ns is not None else 0
        measurement_speed = abs(measurements[idx] - measurements[idx - 1]) / dt
        motion_scale = 1.0 + motion_process_gain * max(measurement_speed - velocity_deadband, 0.0)

        covariance = covariance + process_noise * motion_scale

        innovation = float(measurements[idx] - state)
        base_variance = covariance + measurement_noise
        sigma = float(np.sqrt(max(base_variance, 1e-12)))
        normalized_innovation = abs(innovation) / sigma
        measurement_scale = 1.0 + innovation_scale * max(normalized_innovation - innovation_gate, 0.0) ** 2
        measurement_scale = min(measurement_scale, max_measurement_scale)
        r = measurement_noise * measurement_scale
        s = covariance + r

        innovation_limit = innovation_gate * np.sqrt(max(s, 1e-12))
        clipped_innovation = float(np.clip(innovation, -innovation_limit, innovation_limit))
        gain = covariance / s
        state = state + gain * clipped_innovation
        covariance = (1.0 - gain) * covariance
        result[idx] = state
        if per_pose_time_ns is not None:
            per_pose_time_ns[idx] += perf_counter_ns() - step_start_ns
    if not collect_timing:
        return result
    assert per_pose_time_ns is not None
    total_time_ns = perf_counter_ns() - total_start_ns
    return result, _finalize_time_series(per_pose_time_ns, total_time_ns), total_time_ns


def _zero_phase_butterworth_filter_1d(
    values: ArrayLike,
    *,
    cutoff_hz: float,
    order: int,
    sample_rate_hz: float,
) -> ArrayF:
    measurements = np.asarray(values, dtype=float)
    if measurements.ndim != 1:
        raise ValueError("Butterworth filter values must be one-dimensional")
    count = measurements.shape[0]
    if count <= 2:
        return measurements.copy()

    nyquist_hz = 0.5 * float(sample_rate_hz)
    if cutoff_hz >= nyquist_hz:
        raise ValueError(f"cutoff_hz must be smaller than Nyquist frequency ({nyquist_hz:.6g} Hz)")

    sos = butter(int(order), cutoff_hz, btype="lowpass", output="sos", fs=sample_rate_hz)
    padlen = min(_sosfiltfilt_padlen(sos), count - 1)
    if padlen <= 0:
        return measurements.copy()
    return np.asarray(
        sosfiltfilt(sos, measurements, axis=0, padtype="odd", padlen=padlen),
        dtype=float,
    )


def _lowpass_alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return float(dt / (dt + tau))


def _kalman_constant_velocity(
    values: ArrayLike,
    *,
    timestamps: ArrayLike | None,
    process_noise: float,
    measurement_noise: float,
    initial_covariance: float,
) -> ArrayF:
    return _kalman_constant_velocity_impl(
        values,
        timestamps=timestamps,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        initial_covariance=initial_covariance,
        collect_timing=False,
    )


def _kalman_constant_velocity_impl(
    values: ArrayLike,
    *,
    timestamps: ArrayLike | None,
    process_noise: float,
    measurement_noise: float,
    initial_covariance: float,
    collect_timing: bool,
) -> ArrayF | tuple[ArrayF, ArrayI, int]:
    total_start_ns = perf_counter_ns() if collect_timing else 0
    measurements = np.asarray(values, dtype=float)
    n, dims = measurements.shape
    if n == 1:
        result = measurements.copy()
        if not collect_timing:
            return result
        total_time_ns = perf_counter_ns() - total_start_ns
        return result, _uniform_time_series(n, total_time_ns), total_time_ns

    result = np.empty_like(measurements)
    per_pose_time_ns = np.zeros(n, dtype=np.int64) if collect_timing else None
    init_start_ns = perf_counter_ns() if per_pose_time_ns is not None else 0
    state = np.zeros(dims * 2, dtype=float)
    state[:dims] = measurements[0]
    covariance = np.eye(dims * 2, dtype=float) * initial_covariance
    h = np.zeros((dims, dims * 2), dtype=float)
    h[:, :dims] = np.eye(dims)
    r = np.eye(dims, dtype=float) * measurement_noise
    result[0] = measurements[0]
    if per_pose_time_ns is not None:
        per_pose_time_ns[0] += perf_counter_ns() - init_start_ns

    if timestamps is None:
        dt_values = np.ones(n - 1, dtype=float)
    else:
        dt_values = np.diff(np.asarray(timestamps, dtype=float))
        if np.any(dt_values <= 0):
            raise ValueError("timestamps must be strictly increasing")

    identity = np.eye(dims * 2, dtype=float)
    for idx, dt in enumerate(dt_values, start=1):
        step_start_ns = perf_counter_ns() if per_pose_time_ns is not None else 0
        f = np.eye(dims * 2, dtype=float)
        f[:dims, dims:] = np.eye(dims) * dt
        q_1d = np.array(
            [[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]],
            dtype=float,
        ) * process_noise
        q = np.kron(q_1d, np.eye(dims, dtype=float))

        state = f @ state
        covariance = f @ covariance @ f.T + q

        innovation = measurements[idx] - h @ state
        s = h @ covariance @ h.T + r
        gain = covariance @ h.T @ np.linalg.inv(s)
        state = state + gain @ innovation
        covariance = (identity - gain @ h) @ covariance
        result[idx] = state[:dims]
        if per_pose_time_ns is not None:
            per_pose_time_ns[idx] += perf_counter_ns() - step_start_ns
    if per_pose_time_ns is None:
        return result
    total_time_ns = perf_counter_ns() - total_start_ns
    return result, _finalize_time_series(per_pose_time_ns, total_time_ns), total_time_ns


def _ukf_initial_motion_vector(
    initial_velocity: ArrayLike | None,
    initial_linear_velocity: ArrayLike | None,
    initial_angular_velocity: ArrayLike | None,
) -> ArrayF:
    if initial_velocity is not None:
        if initial_linear_velocity is not None or initial_angular_velocity is not None:
            raise ValueError(
                "initial_velocity cannot be combined with initial_linear_velocity "
                "or initial_angular_velocity"
            )
        return _validate_vector(initial_velocity, 6, "initial_velocity")

    motion = np.zeros(6, dtype=float)
    if initial_linear_velocity is not None:
        motion[:3] = _validate_vector(initial_linear_velocity, 3, "initial_linear_velocity")
    if initial_angular_velocity is not None:
        motion[3:] = _validate_vector(initial_angular_velocity, 3, "initial_angular_velocity")
    return motion


def _validate_vector(values: ArrayLike | None, length: int, name: str) -> ArrayF:
    if values is None:
        return np.zeros(length, dtype=float)
    vector = np.asarray(values, dtype=float)
    if vector.shape != (length,):
        raise ValueError(f"{name} must be a {length}-element vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.astype(float, copy=False)


def _ukf_filter_measurements(
    measurements: ArrayLike,
    *,
    timestamps: ArrayLike | None,
    motion_model: str,
    process_noise: float,
    measurement_noise: float,
    initial_covariance: float,
    alpha: float,
    beta: float,
    kappa: float,
    sample_rate_hz: float,
    initial_velocity: ArrayLike | None = None,
    collect_timing: bool = False,
) -> ArrayF | tuple[ArrayF, ArrayI, int]:
    total_start_ns = perf_counter_ns() if collect_timing else 0
    z_values = np.asarray(measurements, dtype=float)
    if z_values.ndim != 2:
        raise ValueError("UKF measurements must have shape (N, D)")
    count, dims = z_values.shape
    if count <= 1:
        result = z_values.copy()
        if not collect_timing:
            return result
        total_time_ns = perf_counter_ns() - total_start_ns
        return result, _uniform_time_series(count, total_time_ns), total_time_ns

    order = 3 if motion_model in {"constant_acceleration", "ca"} else 2
    state_dim = dims * order
    lam = alpha**2 * (state_dim + kappa) - state_dim
    scale = state_dim + lam
    if scale <= 0:
        raise ValueError("invalid UKF scaling; increase alpha or kappa")

    weights_mean = np.full(2 * state_dim + 1, 0.5 / scale, dtype=float)
    weights_cov = weights_mean.copy()
    weights_mean[0] = lam / scale
    weights_cov[0] = lam / scale + (1.0 - alpha**2 + beta)

    state = np.zeros(state_dim, dtype=float)
    state[:dims] = z_values[0]
    state[dims : 2 * dims] = _validate_vector(initial_velocity, dims, "initial_velocity")
    covariance = np.eye(state_dim, dtype=float) * initial_covariance
    measurement_covariance = np.eye(dims, dtype=float) * measurement_noise

    dt_values = _dt_values(timestamps, count, sample_rate_hz)
    result = np.empty_like(z_values)
    per_pose_time_ns = np.zeros(count, dtype=np.int64) if collect_timing else None
    init_start_ns = perf_counter_ns() if per_pose_time_ns is not None else 0
    result[0] = z_values[0]
    if per_pose_time_ns is not None:
        per_pose_time_ns[0] += perf_counter_ns() - init_start_ns

    for idx, dt in enumerate(dt_values, start=1):
        step_start_ns = perf_counter_ns() if per_pose_time_ns is not None else 0
        sigma_points = _ukf_sigma_points(state, covariance, scale)
        predicted_sigma = np.array(
            [_ukf_predict_sigma(point, dims, dt, motion_model) for point in sigma_points],
            dtype=float,
        )
        state_pred = weights_mean @ predicted_sigma
        state_diff = predicted_sigma - state_pred
        covariance_pred = (
            state_diff.T @ (weights_cov[:, None] * state_diff)
            + _ukf_process_covariance(dims, order, dt, process_noise)
        )
        covariance_pred = _symmetrize(covariance_pred)

        measurement_sigma = predicted_sigma[:, :dims]
        measurement_pred = weights_mean @ measurement_sigma
        measurement_diff = measurement_sigma - measurement_pred
        innovation_covariance = (
            measurement_diff.T @ (weights_cov[:, None] * measurement_diff)
            + measurement_covariance
        )
        innovation_covariance = _symmetrize(innovation_covariance)
        cross_covariance = state_diff.T @ (weights_cov[:, None] * measurement_diff)

        gain = np.linalg.solve(innovation_covariance.T, cross_covariance.T).T
        innovation = z_values[idx] - measurement_pred
        state = state_pred + gain @ innovation
        covariance = _symmetrize(covariance_pred - gain @ innovation_covariance @ gain.T)
        result[idx] = state[:dims]
        if per_pose_time_ns is not None:
            per_pose_time_ns[idx] += perf_counter_ns() - step_start_ns
    if per_pose_time_ns is None:
        return result
    total_time_ns = perf_counter_ns() - total_start_ns
    return result, _finalize_time_series(per_pose_time_ns, total_time_ns), total_time_ns


def _uniform_time_series(count: int, total_time_ns: int) -> ArrayI:
    if count <= 0:
        return np.zeros(0, dtype=np.int64)
    base, remainder = divmod(max(int(total_time_ns), 0), count)
    values = np.full(count, base, dtype=np.int64)
    if remainder:
        values[:remainder] += 1
    return values


def _finalize_time_series(partial_time_ns: ArrayI, total_time_ns: int) -> ArrayI:
    finalized = np.asarray(partial_time_ns, dtype=np.int64).copy()
    measured_time_ns = int(finalized.sum(dtype=np.int64))
    remaining_time_ns = max(int(total_time_ns) - measured_time_ns, 0)
    if remaining_time_ns and finalized.size:
        finalized += _uniform_time_series(finalized.size, remaining_time_ns)
    return finalized


def _dt_values(timestamps: ArrayLike | None, count: int, sample_rate_hz: float) -> ArrayF:
    if timestamps is None:
        return np.full(count - 1, 1.0 / sample_rate_hz, dtype=float)
    values = np.asarray(timestamps, dtype=float)
    if values.ndim != 1 or values.shape[0] != count:
        raise ValueError("timestamps must be a 1-D array with the same length as measurements")
    dt = np.diff(values)
    if np.any(dt <= 0):
        raise ValueError("timestamps must be strictly increasing")
    return dt.astype(float)


def _effective_sample_rate_hz(
    timestamps: ArrayLike | None,
    count: int,
    sample_rate_hz: float,
) -> float:
    if timestamps is None or count <= 1:
        return float(sample_rate_hz)
    dt = _dt_values(timestamps, count, sample_rate_hz)
    median_dt = float(np.median(dt))
    if median_dt <= 0.0:
        raise ValueError("timestamps must produce a positive median dt")
    return float(1.0 / median_dt)


def _sosfiltfilt_padlen(sos: ArrayF) -> int:
    zero_count_num = int(np.sum(np.isclose(sos[:, 2], 0.0)))
    zero_count_den = int(np.sum(np.isclose(sos[:, 5], 0.0)))
    return int(3 * (2 * sos.shape[0] + 1 - min(zero_count_num, zero_count_den)))


def _ukf_sigma_points(state: ArrayF, covariance: ArrayF, scale: float) -> ArrayF:
    covariance = _symmetrize(covariance)
    jitter = 1e-12
    identity = np.eye(state.shape[0], dtype=float)
    for _ in range(8):
        try:
            root = np.linalg.cholesky(scale * (covariance + jitter * identity))
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    else:
        root = np.linalg.cholesky(scale * (covariance + jitter * identity))

    points = np.empty((2 * state.shape[0] + 1, state.shape[0]), dtype=float)
    points[0] = state
    for idx in range(state.shape[0]):
        points[idx + 1] = state + root[:, idx]
        points[idx + 1 + state.shape[0]] = state - root[:, idx]
    return points


def _ukf_predict_sigma(point: ArrayF, dims: int, dt: float, motion_model: str) -> ArrayF:
    predicted = point.copy()
    if motion_model in {"constant_acceleration", "ca"}:
        predicted[:dims] = point[:dims] + point[dims : 2 * dims] * dt + 0.5 * point[2 * dims :] * dt**2
        predicted[dims : 2 * dims] = point[dims : 2 * dims] + point[2 * dims :] * dt
    else:
        predicted[:dims] = point[:dims] + point[dims:] * dt
    return predicted


def _ukf_process_covariance(dims: int, order: int, dt: float, process_noise: float) -> ArrayF:
    if order == 3:
        q_1d = np.array(
            [
                [dt**6 / 36.0, dt**5 / 12.0, dt**4 / 6.0],
                [dt**5 / 12.0, dt**4 / 4.0, dt**3 / 2.0],
                [dt**4 / 6.0, dt**3 / 2.0, dt**2],
            ],
            dtype=float,
        )
    else:
        q_1d = np.array(
            [[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]],
            dtype=float,
        )
    return np.kron(q_1d, np.eye(dims, dtype=float)) * process_noise


def _symmetrize(matrix: ArrayF) -> ArrayF:
    return 0.5 * (matrix + matrix.T)
