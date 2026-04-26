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

    def plot_positions(self, raw: Trajectory, results: list[FilterAnalysisResult]) -> None:
        self.figure.clear()
        x_axis = _sample_axis(raw)
        for idx, axis in enumerate("xyz", start=1):
            ax = self.figure.add_subplot(3, 1, idx)
            ax.plot(x_axis, raw.positions[:, idx - 1], label="raw", linewidth=1.2)
            for result in results:
                ax.plot(
                    _sample_axis(result.trajectory),
                    result.trajectory.positions[:, idx - 1],
                    label=result.label,
                    linewidth=1.0,
                    alpha=0.85,
                )
            ax.set_ylabel(axis)
            ax.grid(True, alpha=0.25)
            if idx == 1:
                ax.legend(fontsize=8, ncols=2)
        self.figure.axes[-1].set_xlabel("timestamp / sample")
        self.draw()

    def plot_delta(self, raw: Trajectory, results: list[FilterAnalysisResult]) -> None:
        self.figure.clear()
        raw_axis = _sample_axis(raw)
        for idx, axis in enumerate("xyz", start=1):
            ax = self.figure.add_subplot(3, 1, idx)
            for result in results:
                count = min(raw.count, result.trajectory.count)
                delta = result.trajectory.positions[:count, idx - 1] - raw.positions[:count, idx - 1]
                ax.plot(raw_axis[:count], delta, label=result.label, linewidth=1.0)
            ax.axhline(0.0, color="0.4", linewidth=0.8)
            ax.set_ylabel(f"d{axis}")
            ax.grid(True, alpha=0.25)
            if idx == 1:
                ax.legend(fontsize=8, ncols=2)
        self.figure.axes[-1].set_xlabel("timestamp / sample")
        self.draw()

    def plot_metric_bars(self, results: list[FilterAnalysisResult]) -> None:
        self.figure.clear()
        labels = [result.label for result in results]
        metrics = [
            ("to_raw_translation_rmse", "offset RMSE"),
            ("acceleration_rms_ratio", "acc ratio"),
            ("jerk_rms_ratio", "jerk ratio"),
        ]
        for idx, (key, title) in enumerate(metrics, start=1):
            ax = self.figure.add_subplot(1, 3, idx)
            values = [float(result.metrics.get(key, np.nan)) for result in results]
            ax.bar(range(len(values)), values, color="#4C78A8")
            ax.set_title(title)
            ax.set_xticks(range(len(labels)), labels, rotation=75, ha="right", fontsize=7)
            ax.grid(True, axis="y", alpha=0.25)
        self.draw()

    def plot_dimension_ratios(self, results: list[FilterAnalysisResult]) -> None:
        self.figure.clear()
        labels = [result.label for result in results]
        axes = ["x", "y", "z"]
        width = 0.25
        x = np.arange(len(labels))
        ax = self.figure.add_subplot(111)
        colors = ["#4C78A8", "#F58518", "#54A24B"]
        for idx, axis in enumerate(axes):
            values = [float(result.dimension_metrics[f"{axis}_jerk_rms_ratio"]) for result in results]
            ax.bar(x + (idx - 1) * width, values, width=width, label=f"{axis.upper()} jerk ratio", color=colors[idx])
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
                "Metric Bars",
                "Dimension Jerk Ratios",
            ]
        )
        self.chart_combo.currentIndexChanged.connect(self.update_plot)
        chart_row.addWidget(self.chart_combo)
        chart_row.addStretch(1)
        layout.addLayout(chart_row)
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
        self.update_plot()

    def update_plot(self) -> None:
        if self.raw is None or not self.results:
            self.canvas.plot_empty("No results")
            return
        mode = self.chart_combo.currentText()
        if mode == "Position XYZ":
            self.canvas.plot_positions(self.raw, self.results)
        elif mode == "Filtered - Raw XYZ":
            self.canvas.plot_delta(self.raw, self.results)
        elif mode == "Metric Bars":
            self.canvas.plot_metric_bars(self.results)
        else:
            self.canvas.plot_dimension_ratios(self.results)

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
        ]
        for algorithm, params in presets:
            self._add_filter_row(algorithm, params, enabled=True)

    def _add_filter_row(self, algorithm: str, params: dict[str, Any], *, enabled: bool) -> None:
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
