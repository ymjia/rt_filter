from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation


PoseArray = NDArray[np.float64]
Vector = NDArray[np.float64]


@dataclass
class UkfParameters:
    """Runtime parameters for SE(3) Unscented Kalman filtering.

    The measurement vector is ``[x, y, z, rx, ry, rz]``. The rotation vector is
    relative to the first pose after reset. Linear velocity uses the same
    distance unit as the input poses per second, and angular velocity uses
    radians per second.
    """

    motion_model: str = "constant_velocity"
    process_noise: float = 1000.0
    measurement_noise: float = 0.001
    initial_covariance: float = 1.0
    initial_velocity: ArrayLike | None = None
    initial_linear_velocity: ArrayLike | None = None
    initial_angular_velocity: ArrayLike | None = None
    alpha: float = 1e-3
    beta: float = 2.0
    kappa: float = 0.0
    sample_rate_hz: float = 100.0
    history_size: int = 0
    strict_timestamps: bool = False

    def validate(self) -> None:
        _canonical_motion_model(self.motion_model)
        if (
            self.process_noise <= 0
            or self.measurement_noise <= 0
            or self.initial_covariance <= 0
            or self.alpha <= 0
            or self.sample_rate_hz <= 0
        ):
            raise ValueError(
                "process_noise, measurement_noise, initial_covariance, alpha, "
                "and sample_rate_hz must be positive"
            )
        if self.history_size < 0:
            raise ValueError("history_size must be >= 0")
        _initial_motion_vector(
            self.initial_velocity,
            self.initial_linear_velocity,
            self.initial_angular_velocity,
        )


class UkfRealtimeFilter:
    """Realtime UKF for 4x4 SE(3) pose trajectories."""

    def __init__(self, params: UkfParameters | None = None) -> None:
        self.params = UkfParameters() if params is None else params
        self.params.validate()
        self.reset()

    def reset(self) -> None:
        self._initialized = False
        self._reference_rotation: Rotation | None = None
        self._state: Vector | None = None
        self._covariance: Vector | None = None
        self._weights_mean: Vector | None = None
        self._weights_cov: Vector | None = None
        self._scale = 0.0
        self._order = 2
        self._last_timestamp: float | None = None
        self._history: deque[PoseArray] = deque(maxlen=self.params.history_size or None)

    def set_parameters(self, params: UkfParameters, *, reset: bool = False) -> None:
        params.validate()
        self.params = params
        if reset:
            self.reset()
        else:
            self._history = deque(self._history, maxlen=self.params.history_size or None)

    @property
    def history(self) -> list[PoseArray]:
        return [pose.copy() for pose in self._history]

    def update(self, pose: ArrayLike, timestamp: float | None = None) -> PoseArray:
        """Filter one incoming pose and return one display/output pose."""

        pose_arr = _as_pose(pose)
        if not self._initialized:
            return self._initialize(pose_arr, timestamp)

        assert self._state is not None
        assert self._covariance is not None
        assert self._weights_mean is not None
        assert self._weights_cov is not None
        assert self._reference_rotation is not None

        measurement = _measurement_from_pose(pose_arr, self._reference_rotation)
        dt = self._dt(timestamp)
        sigma_points = _sigma_points(self._state, self._covariance, self._scale)
        predicted_sigma = np.array(
            [
                _predict_sigma(point, 6, dt, self.params.motion_model)
                for point in sigma_points
            ],
            dtype=float,
        )

        state_pred = self._weights_mean @ predicted_sigma
        state_diff = predicted_sigma - state_pred
        covariance_pred = (
            state_diff.T @ (self._weights_cov[:, None] * state_diff)
            + _process_covariance(6, self._order, dt, self.params.process_noise)
        )
        covariance_pred = _symmetrize(covariance_pred)

        measurement_sigma = predicted_sigma[:, :6]
        measurement_pred = self._weights_mean @ measurement_sigma
        measurement_diff = measurement_sigma - measurement_pred
        innovation_covariance = (
            measurement_diff.T @ (self._weights_cov[:, None] * measurement_diff)
            + np.eye(6, dtype=float) * self.params.measurement_noise
        )
        innovation_covariance = _symmetrize(innovation_covariance)
        cross_covariance = state_diff.T @ (self._weights_cov[:, None] * measurement_diff)

        gain = np.linalg.solve(innovation_covariance.T, cross_covariance.T).T
        innovation = measurement - measurement_pred
        self._state = state_pred + gain @ innovation
        self._covariance = _symmetrize(
            covariance_pred - gain @ innovation_covariance @ gain.T
        )
        if timestamp is not None:
            self._last_timestamp = float(timestamp)

        filtered = _pose_from_measurement(self._state[:6], self._reference_rotation)
        self._history.append(filtered.copy())
        return filtered

    def filter_trajectory(
        self,
        poses: ArrayLike,
        timestamps: Iterable[float] | None = None,
        *,
        reset: bool = True,
    ) -> PoseArray:
        """Filter every input frame and return a filtered ``(N, 4, 4)`` array."""

        pose_arr = _as_pose_sequence(poses)
        time_values = _timestamps(timestamps, pose_arr.shape[0])
        if reset:
            self.reset()
        output = np.empty_like(pose_arr)
        for idx, pose in enumerate(pose_arr):
            ts = None if time_values is None else float(time_values[idx])
            output[idx] = self.update(pose, ts)
        return output

    def filter_latest_from_history(
        self,
        poses: ArrayLike,
        timestamps: Iterable[float] | None = None,
    ) -> PoseArray:
        """Filter a bounded history and return only the latest filtered pose."""

        filtered = self.filter_trajectory(poses, timestamps, reset=True)
        if filtered.shape[0] == 0:
            raise ValueError("poses must contain at least one frame")
        return filtered[-1]

    def _initialize(self, pose: PoseArray, timestamp: float | None) -> PoseArray:
        model = _canonical_motion_model(self.params.motion_model)
        self._order = 3 if model == "constant_acceleration" else 2
        state_dim = 6 * self._order
        lam = self.params.alpha**2 * (state_dim + self.params.kappa) - state_dim
        self._scale = state_dim + lam
        if self._scale <= 0:
            raise ValueError("invalid UKF scaling; increase alpha or kappa")

        self._weights_mean = np.full(2 * state_dim + 1, 0.5 / self._scale, dtype=float)
        self._weights_cov = self._weights_mean.copy()
        self._weights_mean[0] = lam / self._scale
        self._weights_cov[0] = lam / self._scale + (1.0 - self.params.alpha**2 + self.params.beta)

        self._reference_rotation = Rotation.from_matrix(pose[:3, :3])
        measurement = _measurement_from_pose(pose, self._reference_rotation)
        self._state = np.zeros(state_dim, dtype=float)
        self._state[:6] = measurement
        self._state[6:12] = _initial_motion_vector(
            self.params.initial_velocity,
            self.params.initial_linear_velocity,
            self.params.initial_angular_velocity,
        )
        self._covariance = np.eye(state_dim, dtype=float) * self.params.initial_covariance
        self._last_timestamp = None if timestamp is None else float(timestamp)
        self._initialized = True
        filtered = pose.copy()
        self._history.append(filtered.copy())
        return filtered

    def _dt(self, timestamp: float | None) -> float:
        nominal = 1.0 / self.params.sample_rate_hz
        if timestamp is None or self._last_timestamp is None:
            return nominal
        dt = float(timestamp) - self._last_timestamp
        if dt > 0:
            return dt
        if self.params.strict_timestamps:
            raise ValueError("timestamps must be strictly increasing")
        return nominal


def filter_trajectory(
    poses: ArrayLike,
    timestamps: Iterable[float] | None = None,
    params: UkfParameters | None = None,
) -> PoseArray:
    """Stateless helper for batch-style per-frame output."""

    return UkfRealtimeFilter(params).filter_trajectory(poses, timestamps)


def filter_latest_from_history(
    poses: ArrayLike,
    timestamps: Iterable[float] | None = None,
    params: UkfParameters | None = None,
) -> PoseArray:
    """Stateless helper for UI display from a recent pose history."""

    return UkfRealtimeFilter(params).filter_latest_from_history(poses, timestamps)


def _as_pose(pose: ArrayLike) -> PoseArray:
    pose_arr = np.asarray(pose, dtype=float)
    if pose_arr.shape != (4, 4):
        raise ValueError(f"pose must have shape (4, 4), got {pose_arr.shape}")
    return pose_arr


def _as_pose_sequence(poses: ArrayLike) -> PoseArray:
    pose_arr = np.asarray(poses, dtype=float)
    if pose_arr.ndim != 3 or pose_arr.shape[1:] != (4, 4):
        raise ValueError(f"poses must have shape (N, 4, 4), got {pose_arr.shape}")
    if pose_arr.shape[0] == 0:
        raise ValueError("poses must contain at least one frame")
    return pose_arr


def _timestamps(timestamps: Iterable[float] | None, count: int) -> Vector | None:
    if timestamps is None:
        return None
    values = np.asarray(list(timestamps), dtype=float)
    if values.shape != (count,):
        raise ValueError(f"timestamps must have shape ({count},), got {values.shape}")
    return values


def _canonical_motion_model(value: str) -> str:
    model = value.lower().replace("-", "_")
    if model in {"constant_velocity", "cv"}:
        return "constant_velocity"
    if model in {"constant_acceleration", "ca"}:
        return "constant_acceleration"
    raise ValueError("motion_model must be constant_velocity or constant_acceleration")


def _initial_motion_vector(
    initial_velocity: ArrayLike | None,
    initial_linear_velocity: ArrayLike | None,
    initial_angular_velocity: ArrayLike | None,
) -> Vector:
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


def _validate_vector(values: ArrayLike, length: int, name: str) -> Vector:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (length,):
        raise ValueError(f"{name} must be a {length}-element vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.astype(float, copy=False)


def _measurement_from_pose(pose: PoseArray, reference: Rotation) -> Vector:
    rotation = Rotation.from_matrix(pose[:3, :3])
    rotvec = (reference.inv() * rotation).as_rotvec()
    return np.concatenate([pose[:3, 3], rotvec])


def _pose_from_measurement(measurement: Vector, reference: Rotation) -> PoseArray:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = measurement[:3]
    pose[:3, :3] = (reference * Rotation.from_rotvec(measurement[3:6])).as_matrix()
    return pose


def _sigma_points(state: Vector, covariance: Vector, scale: float) -> Vector:
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


def _predict_sigma(point: Vector, dims: int, dt: float, motion_model: str) -> Vector:
    model = _canonical_motion_model(motion_model)
    predicted = point.copy()
    if model == "constant_acceleration":
        predicted[:dims] = point[:dims] + point[dims : 2 * dims] * dt + 0.5 * point[2 * dims :] * dt**2
        predicted[dims : 2 * dims] = point[dims : 2 * dims] + point[2 * dims :] * dt
    else:
        predicted[:dims] = point[:dims] + point[dims:] * dt
    return predicted


def _process_covariance(dims: int, order: int, dt: float, process_noise: float) -> Vector:
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


def _symmetrize(matrix: Vector) -> Vector:
    return 0.5 * (matrix + matrix.T)
