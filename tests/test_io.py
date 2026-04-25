from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from rt_filter.io import read_trajectory, write_trajectory
from rt_filter.se3 import make_poses
from rt_filter.trajectory import Trajectory


def test_csv_roundtrip(tmp_path):
    positions = np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    rotations = Rotation.from_rotvec(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2]])
    )
    timestamps = np.array([0.0, 0.01, 0.02])
    traj = Trajectory(make_poses(positions, rotations), timestamps=timestamps, name="sample")

    path = tmp_path / "sample.csv"
    write_trajectory(traj, path)
    loaded = read_trajectory(path)

    assert loaded.count == traj.count
    np.testing.assert_allclose(loaded.positions, positions)
    np.testing.assert_allclose(loaded.timestamps, timestamps)
