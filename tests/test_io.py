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


def test_read_sn_track_csv_with_euler_columns(tmp_path):
    path = tmp_path / "sn_track.csv"
    path.write_text(
        "x\ty\tz\txr\tyr\tzr\ttime\trate\n"
        "1\t2\t3\t10\t20\t30\t1000\t50\n"
        "4\t5\t6\t40\t50\t60\t1100\t50\n",
        encoding="utf-8",
    )

    loaded = read_trajectory(path)

    np.testing.assert_allclose(loaded.positions, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    np.testing.assert_allclose(loaded.timestamps, np.array([0.0, 0.02]))
    expected = Rotation.from_euler("xyz", np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]), degrees=True)
    np.testing.assert_allclose(loaded.rotations.as_matrix(), expected.as_matrix())


def test_read_sn_track_txt_without_header(tmp_path):
    path = tmp_path / "sn_track.txt"
    path.write_text(
        "1\t2\t3\t10\t20\t30\t1000\t25\n"
        "4\t5\t6\t40\t50\t60\t1100\t25\n",
        encoding="utf-8",
    )

    loaded = read_trajectory(path)

    np.testing.assert_allclose(loaded.positions, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    np.testing.assert_allclose(loaded.timestamps, np.array([0.0, 0.04]))
    expected = Rotation.from_euler("xyz", np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]), degrees=True)
    np.testing.assert_allclose(loaded.rotations.as_matrix(), expected.as_matrix())
