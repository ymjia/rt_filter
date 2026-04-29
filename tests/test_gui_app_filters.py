from __future__ import annotations

import json
import os

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
