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
