from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


PoseArray = NDArray[np.float64]


@dataclass
class OneEuroZParameters:
    """Runtime parameters for Z-only One Euro filtering.

    Lower ``min_cutoff`` gives stronger static denoising and more lag.
    Higher ``beta`` raises the cutoff during motion, reducing lag at the cost
    of letting more motion-time noise through.
    ``derivative_deadband`` ignores small filtered Z velocity before adapting
    the cutoff, which keeps static depth noise from weakening the denoising.
    """

    min_cutoff: float = 1.0
    beta: float = 10.0
    d_cutoff: float = 8.0
    derivative_deadband: float = 0.02
    sample_rate_hz: float = 100.0
    history_size: int = 0
    delay_frames: int = 0
    strict_timestamps: bool = False

    def validate(self) -> None:
        if self.min_cutoff <= 0:
            raise ValueError("min_cutoff must be positive")
        if self.beta < 0:
            raise ValueError("beta must be >= 0")
        if self.d_cutoff <= 0:
            raise ValueError("d_cutoff must be positive")
        if self.derivative_deadband < 0:
            raise ValueError("derivative_deadband must be >= 0")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.history_size < 0:
            raise ValueError("history_size must be >= 0")
        if self.delay_frames < 0:
            raise ValueError("delay_frames must be >= 0")


class OneEuroZRealtimeFilter:
    """Realtime Z-only One Euro filter for 4x4 trajectory poses.

    Input and output poses have shape ``(4, 4)``. Only ``pose[2, 3]`` is
    filtered; X/Y translation and all rotation matrix entries are preserved.
    """

    def __init__(self, params: OneEuroZParameters | None = None) -> None:
        self.params = OneEuroZParameters() if params is None else params
        self.params.validate()
        self.reset()

    def reset(self) -> None:
        self._initialized = False
        self._last_raw_z = 0.0
        self._filtered_z = 0.0
        self._derivative_hat = 0.0
        self._last_timestamp: float | None = None
        self._history: deque[PoseArray] = deque(maxlen=self.params.history_size or None)
        self._raw_buffer: deque[PoseArray] = deque(maxlen=_delay_window_size(self.params.delay_frames))
        self._timestamp_buffer: deque[float | None] = deque(
            maxlen=_delay_window_size(self.params.delay_frames)
        )

    def set_parameters(self, params: OneEuroZParameters, *, reset: bool = False) -> None:
        params.validate()
        self.params = params
        if reset:
            self.reset()
        else:
            self._history = deque(self._history, maxlen=self.params.history_size or None)
            self._raw_buffer = deque(
                self._raw_buffer,
                maxlen=_delay_window_size(self.params.delay_frames),
            )
            self._timestamp_buffer = deque(
                self._timestamp_buffer,
                maxlen=_delay_window_size(self.params.delay_frames),
            )

    @property
    def history(self) -> list[PoseArray]:
        return [pose.copy() for pose in self._history]

    def update(self, pose: ArrayLike, timestamp: float | None = None) -> PoseArray:
        """Filter one incoming pose and return one display/output pose."""

        pose_arr = _as_pose(pose)
        filtered = pose_arr.copy()
        raw_z = float(pose_arr[2, 3])

        if self.params.delay_frames > 0:
            self._push_delay_frame(pose_arr, timestamp)
            target_index = _delayed_target_index(len(self._raw_buffer), self.params.delay_frames)
            filtered = self._raw_buffer[target_index].copy()
            filtered[2, 3] = self._smoothed_delayed_z(target_index)
            if timestamp is not None:
                self._last_timestamp = float(timestamp)
            self._initialized = True
            self._history.append(filtered.copy())
            return filtered

        if not self._initialized:
            self._initialized = True
            self._last_raw_z = raw_z
            self._filtered_z = raw_z
            self._derivative_hat = 0.0
            self._last_timestamp = None if timestamp is None else float(timestamp)
            filtered[2, 3] = raw_z
            self._history.append(filtered.copy())
            return filtered

        dt = self._dt(timestamp)
        derivative = (raw_z - self._last_raw_z) / dt
        derivative_alpha = _lowpass_alpha(self.params.d_cutoff, dt)
        self._derivative_hat = (
            derivative_alpha * derivative + (1.0 - derivative_alpha) * self._derivative_hat
        )

        effective_derivative = max(abs(self._derivative_hat) - self.params.derivative_deadband, 0.0)
        cutoff = self.params.min_cutoff + self.params.beta * effective_derivative
        value_alpha = _lowpass_alpha(cutoff, dt)
        self._filtered_z = value_alpha * raw_z + (1.0 - value_alpha) * self._filtered_z
        self._last_raw_z = raw_z
        if timestamp is not None:
            self._last_timestamp = float(timestamp)

        filtered[2, 3] = self._filtered_z
        self._history.append(filtered.copy())
        return filtered

    def _push_delay_frame(self, pose: PoseArray, timestamp: float | None) -> None:
        if timestamp is not None and self._last_timestamp is not None:
            dt = float(timestamp) - self._last_timestamp
            if dt <= 0 and self.params.strict_timestamps:
                raise ValueError("timestamps must be strictly increasing")
        self._raw_buffer.append(pose.copy())
        self._timestamp_buffer.append(None if timestamp is None else float(timestamp))

    def _smoothed_delayed_z(self, target_index: int) -> float:
        if len(self._raw_buffer) <= 1:
            return float(self._raw_buffer[target_index][2, 3])

        dt = self._buffer_mean_dt()
        derivative_hat = self._estimate_window_derivative(target_index)
        effective_derivative = max(abs(derivative_hat) - self.params.derivative_deadband, 0.0)
        cutoff = self.params.min_cutoff + self.params.beta * effective_derivative
        value_alpha = _lowpass_alpha(cutoff, dt)
        decay = 1.0 - value_alpha

        weighted_sum = 0.0
        weight_sum = 0.0
        for index, pose in enumerate(self._raw_buffer):
            weight = decay ** abs(index - target_index)
            weighted_sum += weight * float(pose[2, 3])
            weight_sum += weight
        if weight_sum <= 0.0:
            return float(self._raw_buffer[target_index][2, 3])
        return float(weighted_sum / weight_sum)

    def _buffer_dt(self, previous_index: int, index: int) -> float:
        nominal = 1.0 / self.params.sample_rate_hz
        try:
            previous = self._timestamp_buffer[previous_index]
            current = self._timestamp_buffer[index]
        except IndexError:
            return nominal
        if previous is None or current is None:
            return nominal
        dt = float(current) - float(previous)
        if dt > 0:
            return dt
        if self.params.strict_timestamps:
            raise ValueError("timestamps must be strictly increasing")
        return nominal

    def _buffer_mean_dt(self) -> float:
        if len(self._raw_buffer) <= 1:
            return 1.0 / self.params.sample_rate_hz
        values = [self._buffer_dt(index - 1, index) for index in range(1, len(self._raw_buffer))]
        valid = [value for value in values if value > 0.0]
        if not valid:
            return 1.0 / self.params.sample_rate_hz
        return float(np.mean(valid))

    def _estimate_window_derivative(self, target_index: int) -> float:
        if len(self._raw_buffer) <= 1:
            return 0.0
        derivative_alpha = _lowpass_alpha(self.params.d_cutoff, self._buffer_mean_dt())
        derivative_decay = 1.0 - derivative_alpha
        weighted_sum = 0.0
        weight_sum = 0.0
        for index in range(1, len(self._raw_buffer)):
            dt = self._buffer_dt(index - 1, index)
            previous_z = float(self._raw_buffer[index - 1][2, 3])
            current_z = float(self._raw_buffer[index][2, 3])
            derivative = (current_z - previous_z) / dt
            step_center = float(index) - 0.5
            weight = derivative_decay ** abs(step_center - float(target_index))
            weighted_sum += weight * derivative
            weight_sum += weight
        if weight_sum <= 0.0:
            return 0.0
        return float(weighted_sum / weight_sum)

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
    params: OneEuroZParameters | None = None,
) -> PoseArray:
    """Stateless helper for batch-style per-frame output."""

    return OneEuroZRealtimeFilter(params).filter_trajectory(poses, timestamps)


def filter_latest_from_history(
    poses: ArrayLike,
    timestamps: Iterable[float] | None = None,
    params: OneEuroZParameters | None = None,
) -> PoseArray:
    """Stateless helper for UI display from a recent pose history."""

    return OneEuroZRealtimeFilter(params).filter_latest_from_history(poses, timestamps)


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


def _timestamps(timestamps: Iterable[float] | None, count: int) -> PoseArray | None:
    if timestamps is None:
        return None
    values = np.asarray(list(timestamps), dtype=float)
    if values.shape != (count,):
        raise ValueError(f"timestamps must have shape ({count},), got {values.shape}")
    return values


def _lowpass_alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return float(dt / (dt + tau))


def _delay_window_size(delay_frames: int) -> int | None:
    if delay_frames <= 0:
        return None
    return 2 * int(delay_frames) + 1


def _delayed_target_index(count: int, delay_frames: int) -> int:
    if count <= 0:
        return 0
    if count > delay_frames:
        return count - 1 - delay_frames
    return 0
