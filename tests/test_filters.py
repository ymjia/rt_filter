from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from rt_filter.evaluation import trajectory_metrics
from rt_filter.filters import run_filter
from rt_filter.se3 import make_poses
from rt_filter.trajectory import Trajectory


def _noisy_static(seed: int = 7) -> Trajectory:
    rng = np.random.default_rng(seed)
    count = 120
    timestamps = np.arange(count, dtype=float) / 100.0
    positions = np.tile(np.array([[100.0, 200.0, 300.0]]), (count, 1))
    positions += rng.normal(scale=[0.03, 0.03, 0.12], size=positions.shape)
    rotations = Rotation.from_rotvec(rng.normal(scale=np.deg2rad(0.02), size=(count, 3)))
    return Trajectory(make_poses(positions, rotations), timestamps=timestamps, name="static")


def test_moving_average_reduces_static_acceleration():
    traj = _noisy_static()
    filtered = run_filter("moving_average", traj, {"window": 9})

    raw_acc = trajectory_metrics(traj)["acceleration_rms"]
    filtered_acc = trajectory_metrics(filtered)["acceleration_rms"]

    assert filtered_acc < raw_acc


def test_all_filters_keep_shape():
    traj = _noisy_static()
    specs = [
        ("moving_average", {"window": 5}),
        ("savgol", {"window": 9, "polyorder": 2}),
        ("exponential", {"alpha": 0.3}),
        ("kalman_cv", {"process_noise": 1e-4, "measurement_noise": 1e-2}),
        ("one_euro_z", {"min_cutoff": 0.7, "beta": 4.0, "d_cutoff": 1.0}),
    ]
    for name, params in specs:
        filtered = run_filter(name, traj, params)
        assert filtered.poses.shape == traj.poses.shape
        assert filtered.timestamps is not None


def test_one_euro_z_reduces_z_noise_without_changing_xy_or_rotation():
    traj = _noisy_static()
    filtered = run_filter("one_euro_z", traj, {"min_cutoff": 0.7, "beta": 4.0, "d_cutoff": 1.0})

    np.testing.assert_allclose(filtered.positions[:, :2], traj.positions[:, :2])
    np.testing.assert_allclose(filtered.rotations.as_matrix(), traj.rotations.as_matrix())
    assert filtered.positions[:, 2].std() < traj.positions[:, 2].std()
