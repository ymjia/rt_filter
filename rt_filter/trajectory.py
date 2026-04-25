from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


ArrayF = NDArray[np.float64]


@dataclass(frozen=True)
class Trajectory:
    """A sampled SE(3) trajectory represented by homogeneous matrices."""

    poses: ArrayF
    timestamps: ArrayF | None = None
    name: str = "trajectory"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        poses = np.asarray(self.poses, dtype=float)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError(f"poses must have shape (N, 4, 4), got {poses.shape}")
        if poses.shape[0] == 0:
            raise ValueError("trajectory must contain at least one pose")

        object.__setattr__(self, "poses", poses)

        if self.timestamps is not None:
            timestamps = np.asarray(self.timestamps, dtype=float)
            if timestamps.ndim != 1 or timestamps.shape[0] != poses.shape[0]:
                raise ValueError(
                    "timestamps must be a 1-D array with the same length as poses"
                )
            object.__setattr__(self, "timestamps", timestamps)

    @property
    def count(self) -> int:
        return int(self.poses.shape[0])

    @property
    def positions(self) -> ArrayF:
        return self.poses[:, :3, 3]

    @property
    def rotations(self) -> Rotation:
        return Rotation.from_matrix(self.poses[:, :3, :3])

    @property
    def duration(self) -> float | None:
        if self.timestamps is None or self.count < 2:
            return None
        return float(self.timestamps[-1] - self.timestamps[0])

    def copy_with(
        self,
        *,
        poses: ArrayF | None = None,
        timestamps: ArrayF | None = None,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "Trajectory":
        merged_metadata = dict(self.metadata)
        if metadata:
            merged_metadata.update(metadata)
        return Trajectory(
            poses=self.poses.copy() if poses is None else poses,
            timestamps=self.timestamps.copy()
            if timestamps is None and self.timestamps is not None
            else timestamps,
            name=self.name if name is None else name,
            metadata=merged_metadata,
        )

    def finite_difference_dt(self) -> ArrayF:
        if self.count < 2:
            return np.ones(0, dtype=float)
        if self.timestamps is None:
            return np.ones(self.count - 1, dtype=float)
        dt = np.diff(self.timestamps)
        if np.any(dt <= 0):
            raise ValueError("timestamps must be strictly increasing")
        return dt.astype(float)
