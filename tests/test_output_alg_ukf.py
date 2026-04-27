from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from output_alg.ukf import (
    UkfParameters,
    UkfRealtimeFilter,
    filter_latest_from_history,
    filter_trajectory,
)
from rt_filter.filters import run_filter
from rt_filter.se3 import make_poses
from rt_filter.trajectory import Trajectory


def _trajectory() -> Trajectory:
    count = 80
    timestamps = np.arange(count, dtype=float) / 100.0
    positions = np.column_stack(
        [
            np.linspace(0.0, 10.0, count),
            2.0 * np.sin(np.linspace(0.0, 1.0, count)),
            100.0 + 0.2 * np.cos(np.linspace(0.0, 2.0, count)),
        ]
    )
    rotations = Rotation.from_rotvec(
        np.column_stack(
            [
                np.zeros(count),
                np.zeros(count),
                np.linspace(0.0, 0.1, count),
            ]
        )
    )
    return Trajectory(make_poses(positions, rotations), timestamps=timestamps, name="moving")


def test_standalone_ukf_matches_framework_filter():
    traj = _trajectory()
    params = UkfParameters(
        motion_model="constant_velocity",
        process_noise=1000.0,
        measurement_noise=0.001,
        initial_linear_velocity=[1.0, 0.0, 0.0],
        initial_angular_velocity=[0.0, 0.0, 0.01],
    )

    standalone = filter_trajectory(traj.poses, traj.timestamps, params)
    framework = run_filter(
        "ukf",
        traj,
        {
            "motion_model": "constant_velocity",
            "process_noise": 1000.0,
            "measurement_noise": 0.001,
            "initial_linear_velocity": [1.0, 0.0, 0.0],
            "initial_angular_velocity": [0.0, 0.0, 0.01],
        },
    )

    np.testing.assert_allclose(standalone, framework.poses, atol=1e-10)


def test_standalone_ukf_latest_from_history_matches_last_frame():
    traj = _trajectory()
    params = UkfParameters(process_noise=1000.0, measurement_noise=0.001)

    latest = filter_latest_from_history(traj.poses, traj.timestamps, params)
    filtered = filter_trajectory(traj.poses, traj.timestamps, params)

    np.testing.assert_allclose(latest, filtered[-1], atol=1e-12)


def test_standalone_ukf_stateful_update_matches_batch():
    traj = _trajectory()
    params = UkfParameters(process_noise=1000.0, measurement_noise=0.001)
    filter_ukf = UkfRealtimeFilter(params)

    updated = np.stack(
        [
            filter_ukf.update(pose, float(timestamp))
            for pose, timestamp in zip(traj.poses, traj.timestamps, strict=True)
        ]
    )
    batched = filter_trajectory(traj.poses, traj.timestamps, params)

    np.testing.assert_allclose(updated, batched, atol=1e-12)


def test_standalone_ukf_rejects_mixed_initial_velocity_forms():
    with pytest.raises(ValueError, match="initial_velocity"):
        UkfParameters(
            initial_velocity=[1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
            initial_linear_velocity=[1.0, 2.0, 3.0],
        ).validate()
