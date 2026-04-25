from __future__ import annotations

import csv

import numpy as np
from scipy.spatial.transform import Rotation

from rt_filter.analysis import analyze_filters, parse_filter_specs
from rt_filter.paraview_export import write_paraview_comparison_script
from rt_filter.se3 import make_poses
from rt_filter.trajectory import Trajectory


def _trajectory() -> Trajectory:
    count = 16
    positions = np.column_stack(
        [
            np.linspace(0.0, 1.0, count),
            0.1 * np.sin(np.linspace(0.0, np.pi, count)),
            np.zeros(count),
        ]
    )
    rotations = Rotation.identity(count)
    return Trajectory(make_poses(positions, rotations), name="sample")


def test_parse_filter_specs_expands_parameter_grid():
    specs = parse_filter_specs(
        [
            {"enabled": True, "algorithm": "moving_average", "params": '{"window": [3, 5]}'},
            {"enabled": False, "algorithm": "exponential", "params": '{"alpha": 0.4}'},
        ]
    )

    assert [spec.params for spec in specs] == [{"window": 3}, {"window": 5}]


def test_analyze_filters_writes_outputs(tmp_path):
    raw = _trajectory()
    specs = parse_filter_specs(
        [{"enabled": True, "algorithm": "moving_average", "params": {"window": [3, 5]}}]
    )

    run_dir, results = analyze_filters(raw, specs, output_root=tmp_path, run_name="gui_test")

    assert run_dir == tmp_path / "gui_test"
    assert len(results) == 2
    assert (run_dir / "raw.csv").exists()
    assert (run_dir / "raw.vtu").exists()
    assert (run_dir / "moving_average" / "window-3" / "trajectory.csv").exists()
    assert (run_dir / "moving_average" / "window-3" / "trajectory.vtu").exists()
    assert (run_dir / "summary.csv").exists()

    with (run_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert "to_raw_x_rmse" in rows[0]


def test_write_paraview_comparison_script(tmp_path):
    raw = tmp_path / "raw.vtu"
    filtered = tmp_path / "filtered.vtu"
    raw.write_text("<VTKFile />", encoding="utf-8")
    filtered.write_text("<VTKFile />", encoding="utf-8")

    script = write_paraview_comparison_script(
        [raw, filtered],
        ["raw", "moving_average/window-3"],
        tmp_path / "compare.py",
    )

    text = script.read_text(encoding="utf-8")
    assert "XMLUnstructuredGridReader" in text
    assert "moving_average/window-3" in text
    assert "Glyph" in text
    assert "Normals" in text
