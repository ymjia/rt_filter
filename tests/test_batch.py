from __future__ import annotations

import csv
import json

import numpy as np
from scipy.spatial.transform import Rotation

from rt_filter.batch import expand_parameter_grid, parameter_slug, run_batch
from rt_filter.io import write_trajectory
from rt_filter.se3 import make_poses
from rt_filter.trajectory import Trajectory


def test_run_batch_writes_summary(tmp_path):
    count = 40
    timestamps = np.arange(count, dtype=float) / 50.0
    positions = np.column_stack([np.linspace(0.0, 10.0, count), np.zeros(count), np.zeros(count)])
    rotations = Rotation.identity(count)
    traj = Trajectory(make_poses(positions, rotations), timestamps=timestamps, name="line")
    input_path = tmp_path / "line.csv"
    write_trajectory(traj, input_path)

    run_dir = run_batch(
        [input_path],
        [{"name": "moving_average", "params": {"window": [3, 5]}}],
        output_dir=tmp_path / "outputs",
        run_name="test_run",
    )

    assert (run_dir / "summary.csv").exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "line" / "input" / "trajectory.csv").exists()
    assert (run_dir / "line" / "input" / "trajectory.vtu").exists()
    assert (run_dir / "line" / "moving_average" / "window-3" / "trajectory.vtu").exists()
    assert (run_dir / "line" / "moving_average" / "window-5" / "trajectory.vtu").exists()

    with (run_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert rows[0]["input_vtk_path"].endswith("trajectory.vtu")
    assert rows[0]["vtk_path"].endswith("trajectory.vtu")

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["visualization"]["enabled"] is True
    assert manifest["input_artifacts"][0]["vtk_path"].endswith("trajectory.vtu")


def test_expand_parameter_grid_preserves_ukf_velocity_vectors():
    grid = expand_parameter_grid(
        {
            "motion_model": ["constant_velocity", "constant_acceleration"],
            "initial_linear_velocity": [1.0, 2.0, 3.0],
            "initial_angular_velocity": [0.01, 0.02, 0.03],
        }
    )

    assert len(grid) == 2
    assert grid[0]["initial_linear_velocity"] == [1.0, 2.0, 3.0]
    assert grid[0]["initial_angular_velocity"] == [0.01, 0.02, 0.03]


def test_parameter_slug_skips_default_zero_ukf_velocity_vectors():
    slug = parameter_slug(
        {
            "motion_model": "constant_velocity",
            "initial_linear_velocity": [0.0, 0.0, 0.0],
            "initial_angular_velocity": [0.0, 0.0, 0.0],
        }
    )

    assert slug == "motion_model-constant_velocity"
