from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from rt_filter.se3 import (
    make_poses,
    relative_rotvecs,
    rotations_from_relative_rotvecs,
)
from rt_filter.trajectory import Trajectory


ArrayF = NDArray[np.float64]
FilterFunc = Callable[..., Trajectory]


@dataclass(frozen=True)
class FilterInfo:
    name: str
    description: str
    defaults: dict[str, Any]


def available_filters() -> dict[str, FilterInfo]:
    return {
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
        "one_euro_z": FilterInfo(
            name="one_euro_z",
            description="Causal One Euro adaptive low-pass filtering on Z translation only.",
            defaults={
                "min_cutoff": 0.7,
                "beta": 4.0,
                "d_cutoff": 1.0,
                "sample_rate_hz": 80.0,
            },
        ),
    }


def run_filter(
    name: str,
    trajectory: Trajectory,
    params: dict[str, Any] | None = None,
) -> Trajectory:
    normalized = name.lower().replace("-", "_")
    params = {} if params is None else dict(params)
    registry: dict[str, FilterFunc] = {
        "moving_average": moving_average_filter,
        "savgol": savgol_filter_trajectory,
        "savitzky_golay": savgol_filter_trajectory,
        "exponential": exponential_filter,
        "kalman_cv": kalman_cv_filter,
        "one_euro_z": one_euro_z_filter,
    }
    if normalized not in registry:
        known = ", ".join(sorted(available_filters()))
        raise ValueError(f"unknown filter '{name}', available filters: {known}")
    return registry[normalized](trajectory, **params)


def moving_average_filter(traj: Trajectory, window: int = 5) -> Trajectory:
    window = _validate_window(window, traj.count, odd=False)
    positions = _moving_average(traj.positions, window)
    rotations = traj.rotations
    smoothed_rotations = []
    half = window // 2
    for idx in range(traj.count):
        start = max(0, idx - half)
        stop = min(traj.count, idx + half + 1)
        smoothed_rotations.append(rotations[start:stop].mean())
    result = traj.copy_with(
        poses=make_poses(positions, Rotation.concatenate(smoothed_rotations)),
        name=f"{traj.name}__moving_average",
        metadata={"filter": "moving_average", "params": {"window": window}},
    )
    return result


def savgol_filter_trajectory(
    traj: Trajectory,
    window: int = 9,
    polyorder: int = 2,
    mode: str = "interp",
) -> Trajectory:
    if traj.count <= polyorder + 1:
        return traj.copy_with(
            name=f"{traj.name}__savgol",
            metadata={
                "filter": "savgol",
                "params": {"window": window, "polyorder": polyorder, "mode": mode},
                "warning": "trajectory too short; returned unchanged",
            },
        )
    window = _validate_window(window, traj.count, odd=True)
    if polyorder >= window:
        raise ValueError("polyorder must be smaller than window")

    positions = savgol_filter(traj.positions, window, polyorder, axis=0, mode=mode)
    reference = traj.rotations[0]
    rotvecs = relative_rotvecs(traj.rotations, reference=reference)
    smoothed_rotvecs = savgol_filter(rotvecs, window, polyorder, axis=0, mode=mode)
    rotations = rotations_from_relative_rotvecs(smoothed_rotvecs, reference)
    return traj.copy_with(
        poses=make_poses(positions, rotations),
        name=f"{traj.name}__savgol",
        metadata={
            "filter": "savgol",
            "params": {"window": window, "polyorder": polyorder, "mode": mode},
        },
    )


def exponential_filter(traj: Trajectory, alpha: float = 0.25) -> Trajectory:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    positions = np.empty_like(traj.positions)
    positions[0] = traj.positions[0]
    input_rotations = traj.rotations
    output_rotations = [input_rotations[0]]
    for idx in range(1, traj.count):
        positions[idx] = (1.0 - alpha) * positions[idx - 1] + alpha * traj.positions[idx]
        delta = output_rotations[-1].inv() * input_rotations[idx]
        output_rotations.append(output_rotations[-1] * Rotation.from_rotvec(alpha * delta.as_rotvec()))
    return traj.copy_with(
        poses=make_poses(positions, Rotation.concatenate(output_rotations)),
        name=f"{traj.name}__exponential",
        metadata={"filter": "exponential", "params": {"alpha": alpha}},
    )


def kalman_cv_filter(
    traj: Trajectory,
    process_noise: float = 1e-4,
    measurement_noise: float = 1e-2,
    initial_covariance: float = 1.0,
) -> Trajectory:
    if process_noise <= 0 or measurement_noise <= 0 or initial_covariance <= 0:
        raise ValueError("noise and covariance parameters must be positive")
    positions = _kalman_constant_velocity(
        traj.positions,
        timestamps=traj.timestamps,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        initial_covariance=initial_covariance,
    )
    reference = traj.rotations[0]
    rotvecs = relative_rotvecs(traj.rotations, reference=reference)
    smoothed_rotvecs = _kalman_constant_velocity(
        rotvecs,
        timestamps=traj.timestamps,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        initial_covariance=initial_covariance,
    )
    rotations = rotations_from_relative_rotvecs(smoothed_rotvecs, reference)
    return traj.copy_with(
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


def one_euro_z_filter(
    traj: Trajectory,
    min_cutoff: float = 0.7,
    beta: float = 4.0,
    d_cutoff: float = 1.0,
    sample_rate_hz: float = 80.0,
) -> Trajectory:
    """Causal adaptive low-pass filter for the Z translation channel.

    X/Y translation and orientation are intentionally preserved. This is useful
    when depth noise is much larger than lateral noise, and motion lag must stay
    low during slow turns or continuous motion.
    """

    if min_cutoff <= 0 or beta < 0 or d_cutoff <= 0 or sample_rate_hz <= 0:
        raise ValueError(
            "min_cutoff, d_cutoff, and sample_rate_hz must be positive; beta must be >= 0"
        )

    positions = traj.positions.copy()
    positions[:, 2] = _one_euro_filter_1d(
        positions[:, 2],
        timestamps=traj.timestamps,
        min_cutoff=min_cutoff,
        beta=beta,
        d_cutoff=d_cutoff,
        sample_rate_hz=sample_rate_hz,
    )
    return traj.copy_with(
        poses=make_poses(positions, traj.rotations),
        name=f"{traj.name}__one_euro_z",
        metadata={
            "filter": "one_euro_z",
            "params": {
                "min_cutoff": min_cutoff,
                "beta": beta,
                "d_cutoff": d_cutoff,
                "sample_rate_hz": sample_rate_hz,
            },
        },
    )


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


def _one_euro_filter_1d(
    values: ArrayLike,
    *,
    timestamps: ArrayLike | None,
    min_cutoff: float,
    beta: float,
    d_cutoff: float,
    sample_rate_hz: float,
) -> ArrayF:
    measurements = np.asarray(values, dtype=float)
    if measurements.ndim != 1:
        raise ValueError("one euro filter values must be one-dimensional")
    if measurements.shape[0] <= 1:
        return measurements.copy()

    result = np.empty_like(measurements)
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

    for idx, dt in enumerate(dt_values, start=1):
        derivative = (measurements[idx] - measurements[idx - 1]) / dt
        derivative_alpha = _lowpass_alpha(d_cutoff, dt)
        derivative_hat = derivative_alpha * derivative + (1.0 - derivative_alpha) * derivative_hat
        cutoff = min_cutoff + beta * abs(derivative_hat)
        value_alpha = _lowpass_alpha(cutoff, dt)
        result[idx] = value_alpha * measurements[idx] + (1.0 - value_alpha) * result[idx - 1]
    return result


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
    measurements = np.asarray(values, dtype=float)
    n, dims = measurements.shape
    if n == 1:
        return measurements.copy()

    result = np.empty_like(measurements)
    state = np.zeros(dims * 2, dtype=float)
    state[:dims] = measurements[0]
    covariance = np.eye(dims * 2, dtype=float) * initial_covariance
    h = np.zeros((dims, dims * 2), dtype=float)
    h[:, :dims] = np.eye(dims)
    r = np.eye(dims, dtype=float) * measurement_noise
    result[0] = measurements[0]

    if timestamps is None:
        dt_values = np.ones(n - 1, dtype=float)
    else:
        dt_values = np.diff(np.asarray(timestamps, dtype=float))
        if np.any(dt_values <= 0):
            raise ValueError("timestamps must be strictly increasing")

    identity = np.eye(dims * 2, dtype=float)
    for idx, dt in enumerate(dt_values, start=1):
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
    return result
