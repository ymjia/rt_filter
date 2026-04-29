from __future__ import annotations

import json
import os

import numpy as np
import pytest


def test_gui_algorithm_change_resets_params_to_filter_defaults():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")

    from PySide6.QtWidgets import QApplication, QComboBox

    from rt_filter.gui.app import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        combo = window.filter_table.cellWidget(0, 1)
        assert isinstance(combo, QComboBox)

        combo.setCurrentText("kalman_cv")

        params = json.loads(window.filter_table.item(0, 2).text())
        assert params == {"process_noise": 1e-4, "measurement_noise": 1e-2}
    finally:
        window.close()
        app.processEvents()


def test_curve_visibility_toggle_preserves_current_zoom():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")

    from PySide6.QtWidgets import QApplication
    from scipy.spatial.transform import Rotation

    from rt_filter.analysis import FilterAnalysisResult, FilterSpec
    from rt_filter.gui.app import MainWindow
    from rt_filter.se3 import make_poses
    from rt_filter.trajectory import Trajectory

    count = 80
    timestamps = np.arange(count, dtype=float) / 100.0
    raw_positions = np.column_stack(
        [
            100.0 + 0.02 * np.sin(2.0 * np.pi * 1.1 * timestamps),
            200.0 + 0.02 * np.cos(2.0 * np.pi * 0.9 * timestamps),
            300.0 + 0.05 * np.sin(2.0 * np.pi * 1.5 * timestamps),
        ]
    )
    filtered_positions = raw_positions.copy()
    filtered_positions[:, 2] = 300.0 + 0.03 * np.sin(2.0 * np.pi * 1.5 * timestamps + 0.2)
    rotations = Rotation.from_rotvec(np.zeros((count, 3)))
    raw = Trajectory(make_poses(raw_positions, rotations), timestamps=timestamps, name="raw")
    filtered = Trajectory(make_poses(filtered_positions, rotations), timestamps=timestamps, name="filtered")
    result = FilterAnalysisResult(
        FilterSpec("moving_average", {"window": 5}),
        filtered,
        np.zeros(count, dtype=np.int64),
        {
            "range_x": 0.04,
            "range_y": 0.04,
            "range_z": 0.1,
            "filtered_range_x": 0.04,
            "filtered_range_y": 0.04,
                "filtered_range_z": 0.06,
                "filtered_path_length": 0.1,
                "to_raw_translation_rmse": 0.01,
                "to_raw_translation_max": 0.02,
            "jerk_rms_ratio": 0.8,
            "acceleration_rms_ratio": 0.85,
            "compute_mean_us": 1.0,
            "compute_p95_us": 1.2,
            "compute_max_us": 1.5,
        },
        {
            "x_jerk_rms_ratio": 1.0,
            "y_jerk_rms_ratio": 1.0,
            "z_jerk_rms_ratio": 0.8,
            "to_raw_x_rmse": 0.0,
            "to_raw_y_rmse": 0.0,
            "to_raw_z_rmse": 0.01,
        },
    )

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        window.raw = raw
        window.results = [result]
        window.update_results()

        axes = [ax for ax in window.canvas.figure.axes if ax.axison]
        assert len(axes) == 3

        for ax in axes:
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            x_center = (x0 + x1) / 2.0
            y_center = (y0 + y1) / 2.0
            x_half = (x1 - x0) / 6.0
            y_half = (y1 - y0) / 6.0
            new_x = (x_center - x_half, x_center + x_half)
            new_y = (y_center - y_half, y_center + y_half)
            ax.set_xlim(new_x)
            ax.set_ylim(new_y)
        window.canvas.draw_idle()
        app.processEvents()
        expected_limits = window.canvas.capture_view_limits()

        window._set_curve_visible(result.label, False)
        app.processEvents()

        actual_limits = window.canvas.capture_view_limits()
        assert len(actual_limits) == len(expected_limits)
        for actual, expected in zip(actual_limits, expected_limits, strict=True):
            np.testing.assert_allclose(actual[0], expected[0])
            np.testing.assert_allclose(actual[1], expected[1])
    finally:
        window.close()
        app.processEvents()


def test_expected_path_deviation_chart_renders():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")

    from PySide6.QtWidgets import QApplication
    from scipy.spatial.transform import Rotation

    from rt_filter.analysis import FilterAnalysisResult, FilterSpec
    from rt_filter.gui.app import MainWindow
    from rt_filter.se3 import make_poses
    from rt_filter.trajectory import Trajectory

    start_static = np.zeros((6, 3), dtype=float)
    moving_x = np.linspace(0.0, 10.0, 31)
    moving = np.column_stack([moving_x, np.zeros_like(moving_x), np.zeros_like(moving_x)])
    end_static = np.tile(np.array([10.0, 0.0, 0.0]), (6, 1))
    raw_positions = np.vstack([start_static, moving, end_static])
    filtered_positions = raw_positions.copy()
    filtered_positions[8:35, 0] -= 0.5
    count = raw_positions.shape[0]
    rotations = Rotation.from_rotvec(np.zeros((count, 3)))
    raw = Trajectory(make_poses(raw_positions, rotations), name="raw")
    filtered = Trajectory(make_poses(filtered_positions, rotations), name="filtered")
    result = FilterAnalysisResult(
        FilterSpec("one_euro_z", {"delay_frames": 3}),
        filtered,
        np.zeros(count, dtype=np.int64),
        {
            "filtered_path_length": 10.0,
            "to_raw_translation_rmse": 0.1,
            "to_raw_translation_max": 0.5,
            "jerk_rms_ratio": 0.8,
            "acceleration_rms_ratio": 0.9,
            "compute_mean_us": 1.0,
            "compute_p95_us": 1.2,
            "compute_max_us": 1.5,
        },
        {
            "x_jerk_rms_ratio": 0.8,
            "y_jerk_rms_ratio": 1.0,
            "z_jerk_rms_ratio": 1.0,
            "to_raw_x_rmse": 0.1,
            "to_raw_y_rmse": 0.0,
            "to_raw_z_rmse": 0.0,
        },
    )

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        window.raw = raw
        window.results = [result]
        window.chart_combo.setCurrentText("Expected Path Deviation")
        window.update_results()

        axes = [ax for ax in window.canvas.figure.axes if ax.axison]
        assert len(axes) == 3
        assert "Expected path deviation" in axes[0].get_title()
        labels = window._current_curve_labels()
        assert labels == ["raw", result.label]
    finally:
        window.close()
        app.processEvents()


def test_gui_one_euro_z_algorithm_change_uses_tuned_defaults():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")

    from PySide6.QtWidgets import QApplication, QComboBox

    from rt_filter.gui.app import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        combo = window.filter_table.cellWidget(0, 1)
        assert isinstance(combo, QComboBox)

        combo.setCurrentText("one_euro_z")

        params = json.loads(window.filter_table.item(0, 2).text())
        assert params == {
            "min_cutoff": 1.0,
            "beta": 10.0,
            "d_cutoff": 8.0,
            "derivative_deadband": 0.02,
            "sample_rate_hz": 100.0,
        }
    finally:
        window.close()
        app.processEvents()


def test_gui_cpp_status_and_presets_include_butterworth_cpp(monkeypatch: pytest.MonkeyPatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")

    from rt_filter.gui.app import MainWindow

    monkeypatch.setattr("rt_filter.gui.app.cpp_demo_available", lambda: True)

    dummy = object()
    presets = MainWindow._default_filter_presets(dummy)
    preset_names = [name for name, _params in presets]
    status_text = MainWindow._cpp_filter_status_text(dummy)

    assert "butterworth-cpp" in preset_names
    assert "butterworth_z-cpp" in preset_names
    assert "ukf-cpp" in preset_names
    assert "one_euro_z-cpp" in preset_names
    assert "butterworth-cpp" in status_text
    assert "butterworth_z-cpp" in status_text


def test_gui_one_euro_z_cpp_preset_uses_tuned_defaults(monkeypatch: pytest.MonkeyPatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")

    from rt_filter.gui.app import MainWindow

    monkeypatch.setattr("rt_filter.gui.app.cpp_demo_available", lambda: True)

    dummy = object()
    presets = MainWindow._default_filter_presets(dummy)
    cpp_presets = [params for name, params in presets if name == "one_euro_z-cpp"]

    assert cpp_presets == [
        {
            "min_cutoff": 1.0,
            "beta": 10.0,
            "d_cutoff": 8.0,
            "derivative_deadband": 0.02,
        }
    ]


def test_gui_one_euro_z_presets_include_butterworth_like_candidates():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")

    from rt_filter.gui.app import MainWindow

    dummy = object()
    presets = MainWindow._default_filter_presets(dummy)
    one_euro_presets = [params for name, params in presets if name == "one_euro_z"]

    assert len(one_euro_presets) == 1
    preset_list = one_euro_presets[0]
    assert isinstance(preset_list, list)
    assert {
        "min_cutoff": 1.0,
        "beta": 10.0,
        "d_cutoff": 8.0,
        "derivative_deadband": 0.02,
    } in preset_list
    assert {
        "min_cutoff": 2.0,
        "beta": 10.0,
        "d_cutoff": 8.0,
        "derivative_deadband": 0.05,
    } in preset_list
