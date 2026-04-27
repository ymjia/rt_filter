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
