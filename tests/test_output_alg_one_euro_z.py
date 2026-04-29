from __future__ import annotations

import numpy as np

from output_alg.one_euro_z import (
    OneEuroZParameters,
    OneEuroZRealtimeFilter,
    filter_latest_from_history,
    filter_trajectory,
)


def _poses(z_values: np.ndarray) -> np.ndarray:
    poses = np.repeat(np.eye(4, dtype=float)[None, :, :], len(z_values), axis=0)
    poses[:, 0, 3] = np.linspace(10.0, 20.0, len(z_values))
    poses[:, 1, 3] = -3.0
    poses[:, 2, 3] = z_values
    poses[:, 0, 0] = 0.0
    poses[:, 0, 1] = -1.0
    poses[:, 1, 0] = 1.0
    poses[:, 1, 1] = 0.0
    return poses


def test_realtime_filter_only_changes_z_translation():
    z = np.array([1.0, 1.2, 0.9, 1.1])
    poses = _poses(z)
    params = OneEuroZParameters(min_cutoff=0.7, beta=4.0, sample_rate_hz=100.0)

    filtered = filter_trajectory(poses, params=params)

    np.testing.assert_allclose(filtered[:, :, :3], poses[:, :, :3])
    np.testing.assert_allclose(filtered[:, 0, 3], poses[:, 0, 3])
    np.testing.assert_allclose(filtered[:, 1, 3], poses[:, 1, 3])
    assert not np.allclose(filtered[:, 2, 3], poses[:, 2, 3])


def test_realtime_filter_reduces_static_z_noise():
    rng = np.random.default_rng(42)
    z = 100.0 + rng.normal(scale=0.1, size=300)
    poses = _poses(z)
    params = OneEuroZParameters(
        min_cutoff=0.02,
        beta=6.0,
        d_cutoff=2.0,
        derivative_deadband=1.0,
        sample_rate_hz=100.0,
    )

    filtered = filter_trajectory(poses, params=params)

    assert filtered[:, 2, 3].std() < poses[:, 2, 3].std()


def test_derivative_deadband_increases_static_denoising():
    rng = np.random.default_rng(123)
    z = 100.0 + rng.normal(scale=0.1, size=300)
    poses = _poses(z)
    base = OneEuroZParameters(
        min_cutoff=0.02,
        beta=6.0,
        d_cutoff=2.0,
        derivative_deadband=0.0,
        sample_rate_hz=100.0,
    )
    stronger = OneEuroZParameters(
        min_cutoff=0.02,
        beta=6.0,
        d_cutoff=2.0,
        derivative_deadband=10.0,
        sample_rate_hz=100.0,
    )

    base_filtered = filter_trajectory(poses, params=base)
    stronger_filtered = filter_trajectory(poses, params=stronger)

    assert stronger_filtered[:, 2, 3].std() < base_filtered[:, 2, 3].std()


def test_latest_from_history_matches_last_trajectory_frame():
    z = np.sin(np.linspace(0.0, 1.0, 40))
    poses = _poses(z)
    timestamps = np.arange(len(poses), dtype=float) / 100.0
    params = OneEuroZParameters(min_cutoff=0.7, beta=4.0, sample_rate_hz=100.0)

    latest = filter_latest_from_history(poses, timestamps, params)
    trajectory = filter_trajectory(poses, timestamps, params)

    np.testing.assert_allclose(latest, trajectory[-1])


def test_stateful_update_matches_batch_filtering():
    z = np.cos(np.linspace(0.0, 1.0, 60))
    poses = _poses(z)
    timestamps = np.arange(len(poses), dtype=float) / 100.0
    params = OneEuroZParameters(min_cutoff=0.7, beta=4.0, sample_rate_hz=100.0)
    filter_z = OneEuroZRealtimeFilter(params)

    updated = np.stack([filter_z.update(pose, float(ts)) for pose, ts in zip(poses, timestamps, strict=True)])
    batched = filter_trajectory(poses, timestamps, params)

    np.testing.assert_allclose(updated, batched)


def test_delay_frames_returns_delayed_pose_matrix():
    z = np.linspace(0.0, 9.0, 10)
    poses = _poses(z)
    timestamps = np.arange(len(poses), dtype=float) / 100.0
    params = OneEuroZParameters(
        min_cutoff=1000.0,
        beta=0.0,
        d_cutoff=2.0,
        derivative_deadband=0.0,
        sample_rate_hz=100.0,
        delay_frames=3,
    )
    filter_z = OneEuroZRealtimeFilter(params)

    updated = np.stack([filter_z.update(pose, float(ts)) for pose, ts in zip(poses, timestamps, strict=True)])

    np.testing.assert_allclose(updated[6, 0, 3], poses[3, 0, 3])
    np.testing.assert_allclose(updated[6, 1, 3], poses[3, 1, 3])
    np.testing.assert_allclose(updated[6, :3, :3], poses[3, :3, :3])


def test_delay_frames_uses_future_frames_for_center_z():
    z = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
    poses = _poses(z)
    timestamps = np.arange(len(poses), dtype=float) / 100.0
    params = OneEuroZParameters(
        min_cutoff=0.02,
        beta=0.0,
        d_cutoff=2.0,
        derivative_deadband=0.0,
        sample_rate_hz=100.0,
        delay_frames=3,
    )
    filter_z = OneEuroZRealtimeFilter(params)

    updated = np.stack([filter_z.update(pose, float(ts)) for pose, ts in zip(poses, timestamps, strict=True)])

    assert updated[6, 2, 3] < poses[3, 2, 3]
    assert updated[6, 2, 3] > 0.0
