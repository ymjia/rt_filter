from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rt_filter.evaluation import trajectory_metrics
from rt_filter.filters import available_filters, cpp_demo_available, run_filter, run_filter_timed
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
        ("ukf", {"motion_model": "constant_velocity", "process_noise": 1000.0, "measurement_noise": 0.001}),
        (
            "ukf",
            {
                "motion_model": "constant_acceleration",
                "process_noise": 10000.0,
                "measurement_noise": 0.001,
            },
        ),
        ("one_euro_z", {"min_cutoff": 0.7, "beta": 4.0, "d_cutoff": 1.0}),
    ]
    for name, params in specs:
        filtered = run_filter(name, traj, params)
        assert filtered.poses.shape == traj.poses.shape
        assert filtered.timestamps is not None


def test_cpp_filters_are_listed():
    filters = available_filters()
    assert "one_euro_z-cpp" in filters
    assert "ukf-cpp" in filters


def test_one_euro_z_reduces_z_noise_without_changing_xy_or_rotation():
    traj = _noisy_static()
    filtered = run_filter("one_euro_z", traj, {"min_cutoff": 0.7, "beta": 4.0, "d_cutoff": 1.0})

    np.testing.assert_allclose(filtered.positions[:, :2], traj.positions[:, :2])
    np.testing.assert_allclose(filtered.rotations.as_matrix(), traj.rotations.as_matrix())
    assert filtered.positions[:, 2].std() < traj.positions[:, 2].std()


def test_one_euro_z_derivative_deadband_improves_static_denoising():
    traj = _noisy_static()

    without_deadband = run_filter(
        "one_euro_z",
        traj,
        {"min_cutoff": 0.02, "beta": 6.0, "d_cutoff": 2.0, "derivative_deadband": 0.0},
    )
    with_deadband = run_filter(
        "one_euro_z",
        traj,
        {"min_cutoff": 0.02, "beta": 6.0, "d_cutoff": 2.0, "derivative_deadband": 10.0},
    )

    assert with_deadband.positions[:, 2].std() < without_deadband.positions[:, 2].std()


def test_ukf_reduces_static_acceleration():
    traj = _noisy_static()

    filtered = run_filter(
        "ukf",
        traj,
        {"motion_model": "constant_velocity", "process_noise": 1000.0, "measurement_noise": 0.001},
    )

    raw_acc = trajectory_metrics(traj)["acceleration_rms"]
    filtered_acc = trajectory_metrics(filtered)["acceleration_rms"]
    assert filtered_acc < raw_acc


def test_ukf_accepts_initial_linear_and_angular_velocity():
    traj = _noisy_static()

    filtered = run_filter(
        "ukf",
        traj,
        {
            "motion_model": "constant_velocity",
            "process_noise": 1000.0,
            "measurement_noise": 0.001,
            "initial_linear_velocity": [1.0, 2.0, 3.0],
            "initial_angular_velocity": [0.01, 0.02, 0.03],
        },
    )

    assert filtered.poses.shape == traj.poses.shape
    assert filtered.metadata["params"]["initial_velocity"] == [1.0, 2.0, 3.0, 0.01, 0.02, 0.03]


def test_ukf_rejects_invalid_initial_velocity_shape():
    traj = _noisy_static()

    with pytest.raises(ValueError, match="initial_linear_velocity"):
        run_filter(
            "ukf",
            traj,
            {
                "motion_model": "constant_velocity",
                "initial_linear_velocity": [1.0, 2.0],
            },
        )


@pytest.mark.skipif(not cpp_demo_available(), reason="C++ demo executable is not built")
@pytest.mark.parametrize(
    ("name", "params"),
    [
        (
            "one_euro_z-cpp",
            {"min_cutoff": 0.02, "beta": 6.0, "d_cutoff": 2.0, "derivative_deadband": 1.0},
        ),
        (
            "ukf_cpp",
            {
                "motion_model": "constant_velocity",
                "process_noise": 1000.0,
                "measurement_noise": 0.001,
                "initial_linear_velocity": [0.0, 0.0, 0.0],
                "initial_angular_velocity": [0.0, 0.0, 0.0],
            },
        ),
    ],
)
def test_cpp_filters_can_be_run_via_python_adapter(name: str, params: dict[str, object]):
    traj = _noisy_static()

    timed = run_filter_timed(name, traj, params)

    assert timed.trajectory.poses.shape == traj.poses.shape
    assert timed.per_pose_time_ns.shape == (traj.count,)
    assert timed.total_time_ns >= 0
    assert timed.trajectory.metadata["backend"] == "cpp_demo"
