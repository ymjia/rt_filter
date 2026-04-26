from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from rt_filter.analysis import (
    FilterAnalysisResult,
    analysis_conclusions,
    analyze_filters,
    parse_filter_specs,
)
from rt_filter.evaluation import trajectory_metrics
from rt_filter.filters import available_filters
from rt_filter.gui.chart_data import complete_neighbor_slice, neighbor_mean_deviation
from rt_filter.io import read_trajectory
from rt_filter.paraview_export import write_paraview_comparison_script
from rt_filter.trajectory import Trajectory


try:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import (
        QApplication,
        QAbstractItemView,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSizePolicy,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without GUI dependency.
    raise SystemExit(
        "PySide6 is required for the GUI. Install it with: python -m pip install -e .[gui]"
    ) from exc

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


METRIC_COLUMNS = [
    "label",
    "to_raw_translation_rmse",
    "to_raw_translation_max",
    "acceleration_rms_ratio",
    "jerk_rms_ratio",
    "to_raw_x_rmse",
    "to_raw_y_rmse",
    "to_raw_z_rmse",
    "x_jerk_rms_ratio",
    "y_jerk_rms_ratio",
    "z_jerk_rms_ratio",
    "to_reference_translation_rmse",
    "reference_rmse_improvement",
]


class PlotCanvas(FigureCanvas):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(7.5, 5.2), tight_layout=True)
        super().__init__(self.figure)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def plot_empty(self, message: str) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        self.draw()

    def plot_positions(
        self,
        raw: Trajectory,
        results: list[FilterAnalysisResult],
        visible_labels: set[str] | None = None,
    ) -> None:
        self.figure.clear()
        x_axis = _sample_axis(raw)
        for idx, axis in enumerate("xyz", start=1):
            ax = self.figure.add_subplot(3, 1, idx)
            x_values: list[np.ndarray] = []
            y_values: list[np.ndarray] = []
            if _is_visible("raw", visible_labels):
                y = raw.positions[:, idx - 1]
                ax.plot(x_axis, y, label="raw", linewidth=1.2)
                x_values.append(x_axis)
                y_values.append(y)
            for result in results:
                if not _is_visible(result.label, visible_labels):
                    continue
                result_axis = _sample_axis(result.trajectory)
                y = result.trajectory.positions[:, idx - 1]
                ax.plot(
                    result_axis,
                    y,
                    label=result.label,
                    linewidth=1.0,
                    alpha=0.85,
                )
                x_values.append(result_axis)
                y_values.append(y)
            ax.set_ylabel(axis)
            ax.grid(True, alpha=0.25)
            if not _set_data_limits(ax, x_values, y_values):
                _annotate_no_visible_data(ax)
            if idx == 1:
                _legend_if_needed(ax, fontsize=8, ncols=2)
        self.figure.axes[-1].set_xlabel("timestamp / sample")
        self.draw()

    def plot_delta(
        self,
        raw: Trajectory,
        results: list[FilterAnalysisResult],
        visible_labels: set[str] | None = None,
    ) -> None:
        self.figure.clear()
        raw_axis = _sample_axis(raw)
        for idx, axis in enumerate("xyz", start=1):
            ax = self.figure.add_subplot(3, 1, idx)
            x_values: list[np.ndarray] = []
            y_values: list[np.ndarray] = []
            for result in results:
                if not _is_visible(result.label, visible_labels):
                    continue
                count = min(raw.count, result.trajectory.count)
                delta = result.trajectory.positions[:count, idx - 1] - raw.positions[:count, idx - 1]
                ax.plot(raw_axis[:count], delta, label=result.label, linewidth=1.0)
                x_values.append(raw_axis[:count])
                y_values.append(delta)
            if x_values:
                ax.axhline(0.0, color="0.4", linewidth=0.8)
                _set_data_limits(ax, x_values, y_values)
            else:
                _annotate_no_visible_data(ax)
            ax.set_ylabel(f"d{axis}")
            ax.grid(True, alpha=0.25)
            if idx == 1:
                _legend_if_needed(ax, fontsize=8, ncols=2)
        self.figure.axes[-1].set_xlabel("timestamp / sample")
        self.draw()

    def plot_neighbor_mean_deviation(
        self,
        raw: Trajectory,
        results: list[FilterAnalysisResult],
        visible_labels: set[str] | None = None,
        windows: tuple[int, ...] = (10, 20),
    ) -> None:
        self.figure.clear()
        series = [("raw", raw)] + [(result.label, result.trajectory) for result in results]
        colors = [
            "#1F1F1F",
            "#4C78A8",
            "#F58518",
            "#54A24B",
            "#B279A2",
            "#E45756",
            "#72B7B2",
            "#FF9DA6",
            "#9D755D",
            "#BAB0AC",
        ]
        styles = ["-", "--", ":"]

        for idx, axis in enumerate("xyz", start=1):
            ax = self.figure.add_subplot(3, 1, idx)
            x_values: list[np.ndarray] = []
            y_values: list[np.ndarray] = []
            for series_idx, (label, trajectory) in enumerate(series):
                x_axis = _sample_axis(trajectory)
                color = colors[series_idx % len(colors)]
                for window_idx, window in enumerate(windows):
                    curve_label = f"{label} w{window}"
                    if not _is_visible(curve_label, visible_labels):
                        continue
                    deviations = neighbor_mean_deviation(trajectory.positions, window)
                    sample_slice = complete_neighbor_slice(trajectory.count, window)
                    x = x_axis[sample_slice]
                    y = deviations[sample_slice, idx - 1]
                    if y.size == 0:
                        continue
                    ax.plot(
                        x,
                        y,
                        label=curve_label,
                        linewidth=1.15 if label == "raw" else 0.95,
                        linestyle=styles[window_idx % len(styles)],
                        color=color,
                        alpha=0.95 if label == "raw" else 0.78,
                    )
                    x_values.append(x)
                    y_values.append(y)
            if x_values:
                ax.axhline(0.0, color="0.45", linewidth=0.8)
                _set_data_limits(ax, x_values, y_values)
            else:
                _annotate_no_visible_data(ax)
            ax.set_ylabel(f"{axis} local dev")
            ax.grid(True, alpha=0.25)
            if idx == 1:
                ax.set_title("XYZ deviation from neighboring-frame mean")
                _legend_if_needed(ax, fontsize=7, ncols=2)
        self.figure.axes[-1].set_xlabel("timestamp / sample")
        self.draw()

    def plot_metric_bars(
        self,
        results: list[FilterAnalysisResult],
        visible_labels: set[str] | None = None,
    ) -> None:
        self.figure.clear()
        visible_results = [result for result in results if _is_visible(result.label, visible_labels)]
        if not visible_results:
            self.plot_empty("No visible curves")
            return
        labels = [result.label for result in visible_results]
        metrics = [
            ("to_raw_translation_rmse", "offset RMSE"),
            ("acceleration_rms_ratio", "acc ratio"),
            ("jerk_rms_ratio", "jerk ratio"),
        ]
        for idx, (key, title) in enumerate(metrics, start=1):
            ax = self.figure.add_subplot(1, 3, idx)
            values = [float(result.metrics.get(key, np.nan)) for result in visible_results]
            ax.bar(range(len(values)), values, color="#4C78A8")
            ax.set_title(title)
            ax.set_xticks(range(len(labels)), labels, rotation=75, ha="right", fontsize=7)
            ax.grid(True, axis="y", alpha=0.25)
        self.draw()

    def plot_dimension_ratios(
        self,
        results: list[FilterAnalysisResult],
        visible_labels: set[str] | None = None,
    ) -> None:
        self.figure.clear()
        labels = [result.label for result in results]
        axes = [
            ("x", "X jerk ratio", "#4C78A8"),
            ("y", "Y jerk ratio", "#F58518"),
            ("z", "Z jerk ratio", "#54A24B"),
        ]
        visible_axes = [item for item in axes if _is_visible(item[1], visible_labels)]
        if not visible_axes:
            self.plot_empty("No visible curves")
            return
        width = min(0.8 / len(visible_axes), 0.25)
        x = np.arange(len(labels))
        ax = self.figure.add_subplot(111)
        for idx, (axis, label, color) in enumerate(visible_axes):
            values = [float(result.dimension_metrics[f"{axis}_jerk_rms_ratio"]) for result in results]
            offset = (idx - (len(visible_axes) - 1) / 2.0) * width
            ax.bar(x + offset, values, width=width, label=label, color=color)
        ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("ratio")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RT Filter Workbench")
        self.resize(1480, 880)
        self.input_paths: list[Path] = []
        self.raw: Trajectory | None = None
        self.reference: Trajectory | None = None
        self.results: list[FilterAnalysisResult] = []
        self.run_dir: Path | None = None
        self.curve_visibility: dict[str, bool] = {}
        self._updating_curve_table = False

        self._build_actions()
        self._build_ui()
        self._load_default_filters()

    def _build_actions(self) -> None:
        open_action = QAction("Add Files", self)
        open_action.triggered.connect(self.add_files)
        self.addAction(open_action)

    def _build_ui(self) -> None:
        root = QSplitter(Qt.Horizontal)
        root.addWidget(self._left_panel())
        root.addWidget(self._center_panel())
        root.addWidget(self._right_panel())
        root.setStretchFactor(0, 0)
        root.setStretchFactor(1, 1)
        root.setStretchFactor(2, 2)
        self.setCentralWidget(root)
        self.statusBar().showMessage("Ready")

    def _left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(self._data_group())
        layout.addWidget(self._filter_group())
        layout.addStretch(1)
        return panel

    def _data_group(self) -> QGroupBox:
        group = QGroupBox("Data")
        layout = QVBoxLayout(group)
        button_row = QHBoxLayout()
        for text, slot in [
            ("Add Files", self.add_files),
            ("Add Dir", self.add_directory),
            ("Clear", self.clear_inputs),
        ]:
            button = QPushButton(text)
            button.clicked.connect(slot)
            button_row.addWidget(button)
        layout.addLayout(button_row)

        self.input_table = QTableWidget(0, 3)
        self.input_table.setHorizontalHeaderLabels(["File", "Samples", "Path length"])
        self.input_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.input_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.input_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.input_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.input_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.input_table.itemSelectionChanged.connect(self._load_selected_input)
        layout.addWidget(self.input_table, 1)

        form = QFormLayout()
        self.reference_edit = QLineEdit()
        ref_button = QPushButton("Browse")
        ref_button.clicked.connect(self.choose_reference)
        ref_row = QHBoxLayout()
        ref_row.addWidget(self.reference_edit)
        ref_row.addWidget(ref_button)
        form.addRow("Reference", ref_row)

        self.output_edit = QLineEdit(str(_output_dir()))
        out_button = QPushButton("Browse")
        out_button.clicked.connect(self.choose_output_dir)
        out_row = QHBoxLayout()
        out_row.addWidget(self.output_edit)
        out_row.addWidget(out_button)
        form.addRow("Output", out_row)
        layout.addLayout(form)
        return group

    def _filter_group(self) -> QGroupBox:
        group = QGroupBox("Filters")
        layout = QVBoxLayout(group)
        self.filter_table = QTableWidget(0, 3)
        self.filter_table.setHorizontalHeaderLabels(["Use", "Algorithm", "Params JSON"])
        self.filter_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.filter_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.filter_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        layout.addWidget(self.filter_table, 1)

        button_row = QHBoxLayout()
        for text, slot in [
            ("Add Row", self.add_filter_row),
            ("Defaults", self._load_default_filters),
            ("Remove", self.remove_filter_rows),
        ]:
            button = QPushButton(text)
            button.clicked.connect(slot)
            button_row.addWidget(button)
        layout.addLayout(button_row)

        run_button = QPushButton("Run Analysis")
        run_button.clicked.connect(self.run_analysis)
        layout.addWidget(run_button)
        return group

    def _center_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.tabs = QTabWidget()

        self.metric_table = QTableWidget(0, len(METRIC_COLUMNS))
        self.metric_table.setHorizontalHeaderLabels(METRIC_COLUMNS)
        self.metric_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.metric_table.horizontalHeader().setStretchLastSection(True)
        self.metric_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tabs.addTab(self.metric_table, "Metrics")

        self.conclusion_text = QTextEdit()
        self.conclusion_text.setReadOnly(True)
        self.tabs.addTab(self.conclusion_text, "Conclusions")

        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.tabs.addTab(self.detail_text, "Run Files")
        layout.addWidget(self.tabs, 1)

        export_row = QHBoxLayout()
        self.paraview_button = QPushButton("Generate ParaView Script")
        self.paraview_button.clicked.connect(self.generate_paraview_script)
        export_row.addWidget(self.paraview_button)
        open_button = QPushButton("Open Output Dir")
        open_button.clicked.connect(self.open_output_dir)
        export_row.addWidget(open_button)
        layout.addLayout(export_row)
        return panel

    def _right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        chart_row = QHBoxLayout()
        chart_row.addWidget(QLabel("Chart"))
        self.chart_combo = QComboBox()
        self.chart_combo.addItems(
            [
                "Position XYZ",
                "Filtered - Raw XYZ",
                "XYZ Neighbor Mean Deviation",
                "Metric Bars",
                "Dimension Jerk Ratios",
            ]
        )
        self.chart_combo.currentIndexChanged.connect(self._on_chart_changed)
        chart_row.addWidget(self.chart_combo)
        chart_row.addStretch(1)
        layout.addLayout(chart_row)

        curve_group = QGroupBox("Curves")
        curve_layout = QVBoxLayout(curve_group)
        curve_buttons = QHBoxLayout()
        show_all_button = QPushButton("Show All")
        show_all_button.clicked.connect(self.show_all_curves)
        curve_buttons.addWidget(show_all_button)
        hide_all_button = QPushButton("Hide All")
        hide_all_button.clicked.connect(self.hide_all_curves)
        curve_buttons.addWidget(hide_all_button)
        curve_buttons.addStretch(1)
        curve_layout.addLayout(curve_buttons)
        self.curve_table = QTableWidget(0, 2)
        self.curve_table.setHorizontalHeaderLabels(["Show", "Curve"])
        self.curve_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.curve_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.curve_table.verticalHeader().setVisible(False)
        self.curve_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.curve_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.curve_table.setMaximumHeight(170)
        curve_layout.addWidget(self.curve_table)
        layout.addWidget(curve_group)

        self.canvas = PlotCanvas()
        self.canvas.plot_empty("No results")
        layout.addWidget(self.canvas, 1)
        return panel

    def add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select trajectories",
            str(_input_dir()),
            "Trajectories (*.csv *.json *.npy *.npz);;All files (*.*)",
        )
        self._append_inputs([Path(file) for file in files])

    def add_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select trajectory directory", str(_input_dir()))
        if not directory:
            return
        paths: list[Path] = []
        for pattern in ("*.csv", "*.json", "*.npy", "*.npz"):
            paths.extend(Path(directory).glob(pattern))
        paths = [path for path in sorted(paths) if path.name.lower() != "manifest.csv"]
        self._append_inputs(paths)

    def clear_inputs(self) -> None:
        self.input_paths.clear()
        self.input_table.setRowCount(0)
        self.raw = None
        self.results.clear()
        self.update_results()

    def choose_reference(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select reference trajectory",
            str(_input_dir()),
            "Trajectories (*.csv *.json *.npy *.npz);;All files (*.*)",
        )
        if path:
            self.reference_edit.setText(path)

    def choose_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select output directory", self.output_edit.text())
        if directory:
            self.output_edit.setText(directory)

    def add_filter_row(self) -> None:
        self._add_filter_row("moving_average", {"window": 5}, enabled=True)

    def remove_filter_rows(self) -> None:
        rows = sorted({index.row() for index in self.filter_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.filter_table.removeRow(row)

    def run_analysis(self) -> None:
        if self.raw is None:
            self._load_selected_input()
        if self.raw is None:
            QMessageBox.warning(self, "Missing Input", "Select an input trajectory first.")
            return

        try:
            specs = parse_filter_specs(self._filter_rows())
            reference = None
            if self.reference_edit.text().strip():
                reference = read_trajectory(self.reference_edit.text().strip())
            output_root = Path(self.output_edit.text().strip() or "outputs/gui")
            self.statusBar().showMessage("Running filters...")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.run_dir, self.results = analyze_filters(
                self.raw,
                specs,
                reference=reference,
                output_root=output_root,
                write_outputs=True,
                write_vtk=True,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Analysis Failed", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()

        self.update_results()
        self.statusBar().showMessage(f"Analysis complete: {self.run_dir}")

    def generate_paraview_script(self) -> None:
        if not self.results or self.run_dir is None:
            QMessageBox.warning(self, "No Results", "Run analysis before generating a ParaView script.")
            return
        vtk_files = [self.run_dir / "raw.vtu"]
        labels = ["raw"]
        for result in self.results:
            if result.vtk_path is not None:
                vtk_files.append(result.vtk_path)
                labels.append(result.label)
        script_path = write_paraview_comparison_script(
            vtk_files,
            labels,
            self.run_dir / "paraview_compare.py",
            normal_scale=self._normal_scale(),
        )
        QMessageBox.information(self, "ParaView Script", f"Wrote {script_path}")
        self._update_detail_text()

    def open_output_dir(self) -> None:
        path = self.run_dir or Path(self.output_edit.text().strip() or ".")
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            os.system(f"open {json.dumps(str(path))}")
        else:
            os.system(f"xdg-open {json.dumps(str(path))}")

    def update_results(self) -> None:
        self.metric_table.setRowCount(0)
        for result in self.results:
            row = self.metric_table.rowCount()
            self.metric_table.insertRow(row)
            table_row = result.table_row()
            for col, key in enumerate(METRIC_COLUMNS):
                value = table_row.get(key, "")
                item = QTableWidgetItem(_format_cell(value))
                item.setData(Qt.UserRole, value)
                self.metric_table.setItem(row, col, item)
        self.conclusion_text.setPlainText("\n".join(analysis_conclusions(self.results)))
        self._update_detail_text()
        self._refresh_curve_table()
        self.update_plot()

    def update_plot(self) -> None:
        if self.raw is None or not self.results:
            self.canvas.plot_empty("No results")
            self._refresh_curve_table()
            return
        mode = self.chart_combo.currentText()
        visible_labels = self._visible_curve_labels()
        if mode == "Position XYZ":
            self.canvas.plot_positions(self.raw, self.results, visible_labels)
        elif mode == "Filtered - Raw XYZ":
            self.canvas.plot_delta(self.raw, self.results, visible_labels)
        elif mode == "XYZ Neighbor Mean Deviation":
            self.canvas.plot_neighbor_mean_deviation(self.raw, self.results, visible_labels)
        elif mode == "Metric Bars":
            self.canvas.plot_metric_bars(self.results, visible_labels)
        elif mode == "Dimension Jerk Ratios":
            self.canvas.plot_dimension_ratios(self.results, visible_labels)
        else:
            self.canvas.plot_empty("Unknown chart")

    def _on_chart_changed(self) -> None:
        self._refresh_curve_table()
        self.update_plot()

    def _current_curve_labels(self) -> list[str]:
        if not self.results:
            return []
        mode = self.chart_combo.currentText()
        if mode == "Position XYZ":
            return ["raw"] + [result.label for result in self.results]
        if mode == "Filtered - Raw XYZ":
            return [result.label for result in self.results]
        if mode == "XYZ Neighbor Mean Deviation":
            labels: list[str] = []
            for base_label in ["raw"] + [result.label for result in self.results]:
                labels.extend([f"{base_label} w10", f"{base_label} w20"])
            return labels
        if mode == "Metric Bars":
            return [result.label for result in self.results]
        if mode == "Dimension Jerk Ratios":
            return ["X jerk ratio", "Y jerk ratio", "Z jerk ratio"]
        return []

    def _visible_curve_labels(self) -> set[str]:
        return {
            label
            for label in self._current_curve_labels()
            if self.curve_visibility.get(label, True)
        }

    def _refresh_curve_table(self) -> None:
        if not hasattr(self, "curve_table"):
            return
        labels = self._current_curve_labels()
        self._updating_curve_table = True
        self.curve_table.setRowCount(0)
        for label in labels:
            row = self.curve_table.rowCount()
            self.curve_table.insertRow(row)

            check = QCheckBox()
            check.setChecked(self.curve_visibility.get(label, True))
            check.toggled.connect(lambda checked, curve_label=label: self._set_curve_visible(curve_label, checked))
            check_frame = QFrame()
            check_layout = QHBoxLayout(check_frame)
            check_layout.setContentsMargins(0, 0, 0, 0)
            check_layout.setAlignment(Qt.AlignCenter)
            check_layout.addWidget(check)
            self.curve_table.setCellWidget(row, 0, check_frame)

            item = QTableWidgetItem(label)
            item.setToolTip(label)
            self.curve_table.setItem(row, 1, item)
        self.curve_table.setEnabled(bool(labels))
        self._updating_curve_table = False

    def _set_curve_visible(self, label: str, visible: bool) -> None:
        self.curve_visibility[label] = visible
        if not self._updating_curve_table:
            self.update_plot()

    def show_all_curves(self) -> None:
        for label in self._current_curve_labels():
            self.curve_visibility[label] = True
        self._refresh_curve_table()
        self.update_plot()

    def hide_all_curves(self) -> None:
        for label in self._current_curve_labels():
            self.curve_visibility[label] = False
        self._refresh_curve_table()
        self.update_plot()

    def _append_inputs(self, paths: list[Path]) -> None:
        for path in paths:
            if path in self.input_paths:
                continue
            try:
                traj = read_trajectory(path)
                metrics = trajectory_metrics(traj)
            except Exception as exc:
                QMessageBox.warning(self, "Skipped Input", f"{path}\n{exc}")
                continue
            self.input_paths.append(path)
            row = self.input_table.rowCount()
            self.input_table.insertRow(row)
            name_item = QTableWidgetItem(path.name)
            name_item.setData(Qt.UserRole, str(path))
            self.input_table.setItem(row, 0, name_item)
            self.input_table.setItem(row, 1, QTableWidgetItem(str(traj.count)))
            self.input_table.setItem(row, 2, QTableWidgetItem(f"{metrics['path_length']:.6g}"))
        if self.input_table.rowCount() and not self.input_table.selectedItems():
            self.input_table.selectRow(0)

    def _load_selected_input(self) -> None:
        row = self.input_table.currentRow()
        if row < 0 or row >= len(self.input_paths):
            return
        path = self.input_paths[row]
        try:
            self.raw = read_trajectory(path)
            metrics = trajectory_metrics(self.raw)
        except Exception as exc:
            QMessageBox.critical(self, "Load Failed", str(exc))
            self.raw = None
            return
        self.detail_text.setPlainText(
            "\n".join(
                [
                    f"Input: {path}",
                    f"Samples: {self.raw.count}",
                    f"Path length: {metrics['path_length']:.6g}",
                    f"Range: x={metrics['range_x']:.6g}, y={metrics['range_y']:.6g}, z={metrics['range_z']:.6g}",
                ]
            )
        )

    def _load_default_filters(self) -> None:
        self.filter_table.setRowCount(0)
        presets = [
            ("moving_average", {"window": [3, 5, 9]}),
            ("savgol", {"window": [5, 9], "polyorder": 2}),
            ("exponential", {"alpha": [0.25, 0.4]}),
            ("kalman_cv", {"process_noise": 1e-4, "measurement_noise": 1e-2}),
            (
                "ukf",
                [
                    {
                        "motion_model": "constant_velocity",
                        "process_noise": 1000.0,
                        "measurement_noise": 0.001,
                        "initial_linear_velocity": [0.0, 0.0, 0.0],
                        "initial_angular_velocity": [0.0, 0.0, 0.0],
                    },
                    {
                        "motion_model": "constant_acceleration",
                        "process_noise": 10000.0,
                        "measurement_noise": 0.001,
                        "initial_linear_velocity": [0.0, 0.0, 0.0],
                        "initial_angular_velocity": [0.0, 0.0, 0.0],
                    },
                    {
                        "motion_model": "constant_velocity",
                        "process_noise": 100.0,
                        "measurement_noise": 0.001,
                        "initial_linear_velocity": [0.0, 0.0, 0.0],
                        "initial_angular_velocity": [0.0, 0.0, 0.0],
                    },
                ],
            ),
            (
                "one_euro_z",
                [
                    {
                        "min_cutoff": 0.02,
                        "beta": 6.0,
                        "d_cutoff": 2.0,
                        "derivative_deadband": 1.0,
                    },
                    {
                        "min_cutoff": 0.7,
                        "beta": 4.0,
                        "d_cutoff": 1.0,
                        "derivative_deadband": 0.0,
                    },
                ],
            ),
        ]
        for algorithm, params in presets:
            self._add_filter_row(algorithm, params, enabled=True)

    def _add_filter_row(self, algorithm: str, params: Any, *, enabled: bool) -> None:
        row = self.filter_table.rowCount()
        self.filter_table.insertRow(row)
        check = QCheckBox()
        check.setChecked(enabled)
        check_frame = QFrame()
        check_layout = QHBoxLayout(check_frame)
        check_layout.setContentsMargins(0, 0, 0, 0)
        check_layout.setAlignment(Qt.AlignCenter)
        check_layout.addWidget(check)
        self.filter_table.setCellWidget(row, 0, check_frame)

        combo = QComboBox()
        combo.addItems(list(available_filters().keys()))
        combo.setCurrentText(algorithm)
        self.filter_table.setCellWidget(row, 1, combo)
        self.filter_table.setItem(row, 2, QTableWidgetItem(json.dumps(params, ensure_ascii=False)))
        combo.currentTextChanged.connect(
            lambda text, source=combo: self._set_filter_default_params(source, text)
        )

    def _filter_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in range(self.filter_table.rowCount()):
            check_frame = self.filter_table.cellWidget(row, 0)
            check = check_frame.findChild(QCheckBox) if check_frame is not None else None
            combo = self.filter_table.cellWidget(row, 1)
            item = self.filter_table.item(row, 2)
            rows.append(
                {
                    "enabled": True if check is None else check.isChecked(),
                    "algorithm": combo.currentText() if isinstance(combo, QComboBox) else "",
                    "params": item.text() if item is not None else "{}",
                }
            )
        return rows

    def _set_filter_default_params(self, combo: QComboBox, algorithm: str) -> None:
        info = available_filters().get(algorithm)
        if info is None:
            return
        for row in range(self.filter_table.rowCount()):
            if self.filter_table.cellWidget(row, 1) is combo:
                self.filter_table.setItem(
                    row,
                    2,
                    QTableWidgetItem(json.dumps(info.defaults, ensure_ascii=False)),
                )
                return

    def _update_detail_text(self) -> None:
        lines: list[str] = []
        if self.run_dir is not None:
            lines.append(f"Run dir: {self.run_dir}")
            lines.append(f"Summary: {self.run_dir / 'summary.csv'}")
            lines.append(f"Raw VTK: {self.run_dir / 'raw.vtu'}")
            script = self.run_dir / "paraview_compare.py"
            if script.exists():
                lines.append(f"ParaView script: {script}")
        for result in self.results:
            lines.append("")
            lines.append(result.label)
            if result.trajectory_path:
                lines.append(f"  CSV: {result.trajectory_path}")
            if result.vtk_path:
                lines.append(f"  VTK: {result.vtk_path}")
        if lines:
            self.detail_text.setPlainText("\n".join(lines))

    def _normal_scale(self) -> float:
        if self.raw is None:
            return 0.03
        metrics = trajectory_metrics(self.raw)
        return max(float(metrics["path_length"]) / max(self.raw.count, 1) * 0.6, 1e-4)


def _sample_axis(traj: Trajectory) -> np.ndarray:
    if traj.timestamps is not None:
        return traj.timestamps
    return np.arange(traj.count, dtype=float)


def _is_visible(label: str, visible_labels: set[str] | None) -> bool:
    return visible_labels is None or label in visible_labels


def _set_data_limits(
    ax: Any,
    x_values: list[np.ndarray],
    y_values: list[np.ndarray],
) -> bool:
    finite_x: list[np.ndarray] = []
    finite_y: list[np.ndarray] = []
    for x, y in zip(x_values, y_values, strict=False):
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        if np.any(mask):
            finite_x.append(x_arr[mask])
            finite_y.append(y_arr[mask])
    if not finite_x:
        return False
    ax.set_xlim(_padded_bounds(np.concatenate(finite_x)))
    ax.set_ylim(_padded_bounds(np.concatenate(finite_y)))
    return True


def _padded_bounds(values: np.ndarray) -> tuple[float, float]:
    low = float(np.min(values))
    high = float(np.max(values))
    if low == high:
        margin = max(abs(low) * 0.05, 1e-9)
    else:
        margin = (high - low) * 0.05
    return low - margin, high + margin


def _annotate_no_visible_data(ax: Any) -> None:
    ax.text(0.5, 0.5, "No visible curves", ha="center", va="center", transform=ax.transAxes)


def _legend_if_needed(ax: Any, **kwargs: Any) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(**kwargs)


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        if np.isfinite(value):
            return f"{value:.6g}"
        return str(value)
    return str(value)


def _project_root() -> Path:
    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent.parent)
    candidates.extend([Path.cwd(), Path(__file__).resolve().parents[2]])
    for candidate in candidates:
        if (candidate / "input").exists() or (candidate / ".git").exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _input_dir() -> Path:
    return (_project_root() / "input").resolve()


def _output_dir() -> Path:
    return (_project_root() / "outputs" / "gui").resolve()


def _default_input_files() -> list[Path]:
    root = _input_dir()
    if not root.exists():
        return []
    files: list[Path] = []
    for pattern in ("*.csv", "sn/case_*/*.csv"):
        files.extend(root.glob(pattern))
    return sorted({path for path in files if path.name.lower() != "manifest.csv"})


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if any(arg in {"-h", "--help"} for arg in args):
        print("usage: rt-filter-gui")
        print("")
        print("Launch the Qt trajectory filtering workbench.")
        print("Install GUI dependencies with: python -m pip install -e .[gui]")
        return 0
    if "--smoke-test" in args:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QApplication([sys.argv[0]])
        window = MainWindow()
        window._append_inputs(_default_input_files())
        print(f"smoke ok: inputs={window.input_table.rowCount()} filters={window.filter_table.rowCount()}")
        window.close()
        app.quit()
        return 0
    app = QApplication(sys.argv)
    window = MainWindow()
    window._append_inputs(_default_input_files())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
