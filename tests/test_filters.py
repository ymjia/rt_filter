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


def _noisy_dynamic_z_wave(seed: int = 11) -> tuple[Trajectory, np.ndarray]:
    rng = np.random.default_rng(seed)
    sample_rate_hz = 100.0
    count = 500
    timestamps = np.arange(count, dtype=float) / sample_rate_hz
    positions = np.empty((count, 3), dtype=float)
    positions[:, 0] = 100.0 + 0.15 * np.sin(2.0 * np.pi * 0.6 * timestamps) + rng.normal(scale=0.01, size=count)
    positions[:, 1] = 200.0 + 0.12 * np.cos(2.0 * np.pi * 0.4 * timestamps) + rng.normal(scale=0.01, size=count)
    clean_z = (
        300.0
        + 0.45 * np.sin(2.0 * np.pi * 12.0 * timestamps)
        + 0.12 * np.sin(2.0 * np.pi * 6.0 * timestamps + 0.4)
    )
    high_freq_noise = 0.16 * np.sin(2.0 * np.pi * 34.0 * timestamps + 0.2)
    random_noise = rng.normal(scale=0.035, size=count)
    positions[:, 2] = clean_z + high_freq_noise + random_noise
    rotations = Rotation.from_rotvec(rng.normal(scale=np.deg2rad(0.02), size=(count, 3)))
    return (
        Trajectory(make_poses(positions, rotations), timestamps=timestamps, name="dynamic_z_wave"),
        clean_z,
    )


def _noisy_dynamic_xyz_wave(seed: int = 13) -> tuple[Trajectory, np.ndarray]:
    rng = np.random.default_rng(seed)
    sample_rate_hz = 100.0
    count = 500
    timestamps = np.arange(count, dtype=float) / sample_rate_hz
    clean_positions = np.empty((count, 3), dtype=float)
    clean_positions[:, 0] = 100.0 + 0.18 * np.sin(2.0 * np.pi * 10.5 * timestamps + 0.1)
    clean_positions[:, 1] = 200.0 + 0.16 * np.sin(2.0 * np.pi * 12.0 * timestamps + 0.5)
    clean_positions[:, 2] = (
        300.0
        + 0.45 * np.sin(2.0 * np.pi * 12.0 * timestamps)
        + 0.12 * np.sin(2.0 * np.pi * 6.0 * timestamps + 0.4)
    )
    high_freq_noise = np.column_stack(
        [
            0.07 * np.sin(2.0 * np.pi * 34.0 * timestamps + 0.2),
            0.06 * np.sin(2.0 * np.pi * 31.0 * timestamps + 0.6),
            0.16 * np.sin(2.0 * np.pi * 34.0 * timestamps + 0.2),
        ]
    )
    random_noise = rng.normal(scale=[0.015, 0.015, 0.035], size=(count, 3))
    positions = clean_positions + high_freq_noise + random_noise
    rotations = Rotation.from_rotvec(rng.normal(scale=np.deg2rad(0.02), size=(count, 3)))
    return (
        Trajectory(make_poses(positions, rotations), timestamps=timestamps, name="dynamic_xyz_wave"),
        clean_positions,
    )


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
        ("one_euro", {"min_cutoff": 0.7, "beta": 4.0, "d_cutoff": 1.0}),
        ("one_euro_z", {"min_cutoff": 0.7, "beta": 4.0, "d_cutoff": 1.0}),
        ("butterworth", {"cutoff_hz": 20.0, "order": 2}),
        ("butterworth_z", {"cutoff_hz": 20.0, "order": 2}),
        (
            "adaptive_kalman_z",
            {
                "process_noise": 1e-12,
                "measurement_noise": 1e-5,
                "motion_process_gain": 0.0,
                "velocity_deadband": 1.0,
                "innovation_scale": 20.0,
                "innovation_gate": 2.5,
                "max_measurement_scale": 100.0,
            },
        ),
    ]
    for name, params in specs:
        filtered = run_filter(name, traj, params)
        assert filtered.poses.shape == traj.poses.shape
        assert filtered.timestamps is not None


def test_cpp_filters_are_listed():
    filters = available_filters()
    assert "one_euro" in filters
    assert "butterworth" in filters
    assert "butterworth-cpp" in filters
    assert "butterworth_z-cpp" in filters
    assert "one_euro_z-cpp" in filters
    assert "ukf-cpp" in filters


def test_one_euro_reduces_xyz_noise_without_changing_rotation():
    traj = _noisy_static()
    filtered = run_filter("one_euro", traj, {"min_cutoff": 0.7, "beta": 4.0, "d_cutoff": 1.0})

    np.testing.assert_allclose(filtered.rotations.as_matrix(), traj.rotations.as_matrix())
    assert np.all(filtered.positions.std(axis=0, ddof=0) < traj.positions.std(axis=0, ddof=0))


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


def test_butterworth_z_preserves_xy_and_main_dynamic_shape():
    traj, clean_z = _noisy_dynamic_z_wave()

    filtered = run_filter("butterworth_z", traj, {"cutoff_hz": 20.0, "order": 2})

    np.testing.assert_allclose(filtered.positions[:, :2], traj.positions[:, :2])
    np.testing.assert_allclose(filtered.rotations.as_matrix(), traj.rotations.as_matrix())

    raw_rmse = float(np.sqrt(np.mean((traj.positions[:, 2] - clean_z) ** 2)))
    filtered_rmse = float(np.sqrt(np.mean((filtered.positions[:, 2] - clean_z) ** 2)))
    assert filtered_rmse < raw_rmse * 0.55

    clean_centered = clean_z - np.mean(clean_z)
    raw_corr = float(np.corrcoef(clean_centered, traj.positions[:, 2] - np.mean(traj.positions[:, 2]))[0, 1])
    filtered_corr = float(
        np.corrcoef(clean_centered, filtered.positions[:, 2] - np.mean(filtered.positions[:, 2]))[0, 1]
    )
    assert filtered_corr > raw_corr
    assert filtered_corr > 0.97


def test_butterworth_preserves_xyz_main_dynamic_shape():
    traj, clean_positions = _noisy_dynamic_xyz_wave()

    filtered = run_filter("butterworth", traj, {"cutoff_hz": 20.0, "order": 2})

    np.testing.assert_allclose(filtered.rotations.as_matrix(), traj.rotations.as_matrix())

    raw_rmse = float(np.sqrt(np.mean((traj.positions - clean_positions) ** 2)))
    filtered_rmse = float(np.sqrt(np.mean((filtered.positions - clean_positions) ** 2)))
    assert filtered_rmse < raw_rmse * 0.6

    for axis in range(3):
        clean_centered = clean_positions[:, axis] - np.mean(clean_positions[:, axis])
        raw_centered = traj.positions[:, axis] - np.mean(traj.positions[:, axis])
        filtered_centered = filtered.positions[:, axis] - np.mean(filtered.positions[:, axis])
        raw_corr = float(np.corrcoef(clean_centered, raw_centered)[0, 1])
        filtered_corr = float(np.corrcoef(clean_centered, filtered_centered)[0, 1])
        assert filtered_corr > raw_corr
        assert filtered_corr > 0.96


def test_butterworth_z_rejects_cutoff_above_nyquist():
    traj = _noisy_static()
    with pytest.raises(ValueError, match="Nyquist"):
        run_filter("butterworth_z", traj, {"cutoff_hz": 60.0, "order": 2, "sample_rate_hz": 100.0})


def test_butterworth_rejects_cutoff_above_nyquist():
    traj = _noisy_static()
    with pytest.raises(ValueError, match="Nyquist"):
        run_filter("butterworth", traj, {"cutoff_hz": 60.0, "order": 2, "sample_rate_hz": 100.0})


def test_adaptive_kalman_z_reduces_z_noise_without_changing_xy_or_rotation():
    traj = _noisy_static()
    filtered = run_filter(
        "adaptive_kalman_z",
        traj,
        {
            "process_noise": 1e-12,
            "measurement_noise": 1e-5,
            "motion_process_gain": 0.0,
            "velocity_deadband": 1.0,
            "innovation_scale": 20.0,
            "innovation_gate": 2.5,
            "max_measurement_scale": 100.0,
        },
    )

    np.testing.assert_allclose(filtered.positions[:, :2], traj.positions[:, :2])
    np.testing.assert_allclose(filtered.rotations.as_matrix(), traj.rotations.as_matrix())
    assert filtered.positions[:, 2].std() < traj.positions[:, 2].std()


def test_adaptive_kalman_z_is_robust_to_single_z_spike():
    traj = _noisy_static()
    spiked = traj.copy_with(poses=traj.poses.copy())
    spiked.poses[60, 2, 3] += 2.0

    filtered = run_filter(
        "adaptive_kalman_z",
        spiked,
        {
            "process_noise": 1e-12,
            "measurement_noise": 1e-5,
            "motion_process_gain": 0.0,
            "velocity_deadband": 1.0,
            "innovation_scale": 20.0,
            "innovation_gate": 2.5,
            "max_measurement_scale": 100.0,
        },
    )

    spike_raw = abs(spiked.positions[60, 2] - np.median(spiked.positions[:, 2]))
    spike_filtered = abs(filtered.positions[60, 2] - np.median(filtered.positions[:, 2]))
    assert spike_filtered < spike_raw * 0.2


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
            "butterworth-cpp",
            {"cutoff_hz": 20.0, "order": 2},
        ),
        (
            "butterworth_z-cpp",
            {"cutoff_hz": 20.0, "order": 2},
        ),
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
