"""
plotter2d.py - Comprehensive 2D Plotter Widget for PyQt5 Scientific Suite

Provides an Origin/MATLAB-like plotting environment with embedded Matplotlib canvas,
multiple plot types, data entry, axis/style controls, multi-series support, and export.
"""

import os
import io
import csv
import json
import glob
import traceback
import tempfile

import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QPushButton,
    QComboBox, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QCheckBox, QColorDialog, QFileDialog, QGroupBox, QFormLayout,
    QMessageBox, QToolBar, QAction, QMenu, QSizePolicy, QGridLayout,
    QTextEdit, QAbstractItemView, QDialog, QDialogButtonBox,
    QProgressBar, QInputDialog, QScrollArea, QApplication
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QColor, QIcon, QFont

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

try:
    from mpl_style import style_figure, style_axes
except ImportError:
    def style_figure(f): pass
    def style_axes(a): pass


PLOT_TYPES = [
    "Line", "Scatter", "Bar", "Histogram", "Contour",
    "Polar", "Error Bars", "Box Plot", "Violin Plot", "Heatmap",
    "Error Envelope", "Waterfall"
]

MARKER_STYLES = [
    "None", "o", "s", "^", "v", "D", "x", "+", "*", ".", "p", "h"
]

LINE_STYLES = ["-", "--", "-.", ":", "None"]

COLORMAPS = ["viridis", "plasma", "inferno", "magma", "cividis",
             "hot", "cool", "coolwarm", "RdYlBu", "jet", "gray"]


# ---------------------------------------------------------------------------
# Publication style templates
# ---------------------------------------------------------------------------
PUBLICATION_TEMPLATES = {
    "Nature": {
        "font.family": "Arial",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "figure.figsize": (3.5, 2.625),
    },
    "Science": {
        "font.family": "Helvetica",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "figure.figsize": (3.4, 2.55),
    },
    "APS (Phys Rev)": {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "lines.linewidth": 1.2,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "figure.figsize": (3.375, 2.5),
    },
    "IEEE": {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "figure.figsize": (3.5, 2.625),
    },
    "Presentation": {
        "font.family": "sans-serif",
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "lines.linewidth": 2.5,
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "figure.figsize": (10, 7),
    },
}


class DataSeriesModel:
    """Holds data and style information for a single plot series."""

    def __init__(self, name="Series 1"):
        self.name = name
        self.x_data = np.array([])
        self.y_data = np.array([])
        self.z_data = np.array([])
        self.y_err = np.array([])
        self.plot_type = "Line"
        self.color = "#1f77b4"
        self.line_width = 2.0
        self.line_style = "-"
        self.marker = "None"
        self.marker_size = 6.0
        self.alpha = 1.0
        self.label = ""
        self.hist_bins = 30
        self.bar_width = 0.8
        self.colormap = "viridis"


class DataTableWidget(QWidget):
    """Table for manual data entry with X, Y, Y_err columns."""

    data_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        toolbar = QHBoxLayout()
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, 100000)
        self.rows_spin.setValue(50)
        self.rows_spin.setPrefix("Rows: ")
        toolbar.addWidget(self.rows_spin)

        btn_set_rows = QPushButton("Set Rows")
        btn_set_rows.clicked.connect(self._set_rows)
        toolbar.addWidget(btn_set_rows)

        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._clear_table)
        toolbar.addWidget(btn_clear)

        toolbar.addStretch()
        layout.addLayout(toolbar)

        self.table = QTableWidget(50, 4)
        self.table.setHorizontalHeaderLabels(["X", "Y", "Y Error", "Z"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
        layout.addWidget(self.table)

        self.table.cellChanged.connect(lambda: self.data_changed.emit())

    def _set_rows(self):
        count = self.rows_spin.value()
        self.table.setRowCount(count)

    def _clear_table(self):
        self.table.clearContents()
        self.data_changed.emit()

    def get_column_data(self, col):
        """Extract numeric data from a table column, skipping empty/invalid cells."""
        values = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, col)
            if item and item.text().strip():
                try:
                    values.append(float(item.text()))
                except ValueError:
                    continue
        return np.array(values)

    def set_column_data(self, col, data):
        """Populate a table column from an array."""
        if len(data) > self.table.rowCount():
            self.table.setRowCount(len(data))
            self.rows_spin.setValue(len(data))
        for i, val in enumerate(data):
            self.table.setItem(i, col, QTableWidgetItem(str(val)))

    def load_dataframe(self, df):
        """Load a pandas DataFrame into the table."""
        self.table.setRowCount(len(df))
        self.rows_spin.setValue(len(df))
        cols = list(df.columns)
        for ci, col_name in enumerate(cols[:4]):
            for ri, val in enumerate(df[col_name]):
                self.table.setItem(ri, ci, QTableWidgetItem(str(val)))


class FormulaWidget(QWidget):
    """Generate data from mathematical formulae."""

    formula_applied = pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QFormLayout(self)

        self.x_start = QDoubleSpinBox()
        self.x_start.setRange(-1e9, 1e9)
        self.x_start.setValue(0)
        self.x_start.setDecimals(4)
        layout.addRow("X Start:", self.x_start)

        self.x_end = QDoubleSpinBox()
        self.x_end.setRange(-1e9, 1e9)
        self.x_end.setValue(10)
        self.x_end.setDecimals(4)
        layout.addRow("X End:", self.x_end)

        self.num_points = QSpinBox()
        self.num_points.setRange(2, 1000000)
        self.num_points.setValue(200)
        layout.addRow("Points:", self.num_points)

        self.formula_edit = QLineEdit("np.sin(x)")
        self.formula_edit.setPlaceholderText("e.g. np.sin(x) * np.exp(-x/5)")
        layout.addRow("Y = f(x):", self.formula_edit)

        self.formula_hint = QLabel(
            "Available: np.sin, np.cos, np.exp, np.log, np.sqrt,\n"
            "np.abs, np.pi, np.e, np.tan, np.arctan, etc."
        )
        self.formula_hint.setStyleSheet("color: #888; font-size: 11px;")
        layout.addRow(self.formula_hint)

        btn_apply = QPushButton("Generate Data")
        btn_apply.clicked.connect(self._apply_formula)
        layout.addRow(btn_apply)

    def _apply_formula(self):
        try:
            x = np.linspace(
                self.x_start.value(),
                self.x_end.value(),
                self.num_points.value()
            )
            y = eval(self.formula_edit.text(), {"np": np, "x": x, "pi": np.pi, "e": np.e})
            y = np.asarray(y, dtype=float)
            if y.shape != x.shape:
                raise ValueError("Formula must produce an array the same size as x.")
            self.formula_applied.emit(x, y)
        except Exception as exc:
            QMessageBox.warning(self, "Formula Error", str(exc))


class SeriesStylePanel(QWidget):
    """Controls for styling the currently selected data series."""

    style_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QFormLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.series_combo = QComboBox()
        layout.addRow("Series:", self.series_combo)

        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(PLOT_TYPES)
        self.plot_type_combo.currentIndexChanged.connect(lambda: self.style_changed.emit())
        layout.addRow("Plot Type:", self.plot_type_combo)

        self.color_btn = QPushButton("  ")
        self.color_btn.setStyleSheet("background-color: #1f77b4; border: 1px solid #aaa;")
        self.color_btn.setFixedSize(60, 24)
        self.color_btn.clicked.connect(self._pick_color)
        self._current_color = "#1f77b4"
        layout.addRow("Color:", self.color_btn)

        self.line_width_spin = QDoubleSpinBox()
        self.line_width_spin.setRange(0.1, 20.0)
        self.line_width_spin.setValue(2.0)
        self.line_width_spin.setSingleStep(0.5)
        self.line_width_spin.valueChanged.connect(lambda: self.style_changed.emit())
        layout.addRow("Line Width:", self.line_width_spin)

        self.line_style_combo = QComboBox()
        self.line_style_combo.addItems(LINE_STYLES)
        self.line_style_combo.currentIndexChanged.connect(lambda: self.style_changed.emit())
        layout.addRow("Line Style:", self.line_style_combo)

        self.marker_combo = QComboBox()
        self.marker_combo.addItems(MARKER_STYLES)
        self.marker_combo.currentIndexChanged.connect(lambda: self.style_changed.emit())
        layout.addRow("Marker:", self.marker_combo)

        self.marker_size_spin = QDoubleSpinBox()
        self.marker_size_spin.setRange(1.0, 50.0)
        self.marker_size_spin.setValue(6.0)
        self.marker_size_spin.valueChanged.connect(lambda: self.style_changed.emit())
        layout.addRow("Marker Size:", self.marker_size_spin)

        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setValue(1.0)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setDecimals(2)
        self.alpha_spin.valueChanged.connect(lambda: self.style_changed.emit())
        layout.addRow("Opacity:", self.alpha_spin)

        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("Legend label")
        self.label_edit.textChanged.connect(lambda: self.style_changed.emit())
        layout.addRow("Label:", self.label_edit)

        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(1, 1000)
        self.bins_spin.setValue(30)
        self.bins_spin.valueChanged.connect(lambda: self.style_changed.emit())
        layout.addRow("Hist Bins:", self.bins_spin)

        self.bar_width_spin = QDoubleSpinBox()
        self.bar_width_spin.setRange(0.01, 10.0)
        self.bar_width_spin.setValue(0.8)
        self.bar_width_spin.setSingleStep(0.1)
        self.bar_width_spin.valueChanged.connect(lambda: self.style_changed.emit())
        layout.addRow("Bar Width:", self.bar_width_spin)

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(COLORMAPS)
        self.cmap_combo.currentIndexChanged.connect(lambda: self.style_changed.emit())
        layout.addRow("Colormap:", self.cmap_combo)

    def _pick_color(self):
        color = QColorDialog.getColor(QColor(self._current_color), self, "Pick Series Color")
        if color.isValid():
            self._current_color = color.name()
            self.color_btn.setStyleSheet(
                f"background-color: {self._current_color}; border: 1px solid #aaa;"
            )
            self.style_changed.emit()

    def get_style(self):
        return {
            "plot_type": self.plot_type_combo.currentText(),
            "color": self._current_color,
            "line_width": self.line_width_spin.value(),
            "line_style": self.line_style_combo.currentText(),
            "marker": self.marker_combo.currentText(),
            "marker_size": self.marker_size_spin.value(),
            "alpha": self.alpha_spin.value(),
            "label": self.label_edit.text(),
            "hist_bins": self.bins_spin.value(),
            "bar_width": self.bar_width_spin.value(),
            "colormap": self.cmap_combo.currentText(),
        }

    def set_style(self, series):
        self.plot_type_combo.setCurrentText(series.plot_type)
        self._current_color = series.color
        self.color_btn.setStyleSheet(
            f"background-color: {series.color}; border: 1px solid #aaa;"
        )
        self.line_width_spin.setValue(series.line_width)
        self.line_style_combo.setCurrentText(series.line_style)
        self.marker_combo.setCurrentText(series.marker)
        self.marker_size_spin.setValue(series.marker_size)
        self.alpha_spin.setValue(series.alpha)
        self.label_edit.setText(series.label)
        self.bins_spin.setValue(series.hist_bins)
        self.bar_width_spin.setValue(series.bar_width)
        self.cmap_combo.setCurrentText(series.colormap)


class AxisControlPanel(QWidget):
    """Controls for axis labels, title, limits, log scale, and grid."""

    axis_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QFormLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("Plot Title")
        self.title_edit.textChanged.connect(lambda: self.axis_changed.emit())
        layout.addRow("Title:", self.title_edit)

        self.xlabel_edit = QLineEdit()
        self.xlabel_edit.setPlaceholderText("X Axis Label")
        self.xlabel_edit.textChanged.connect(lambda: self.axis_changed.emit())
        layout.addRow("X Label:", self.xlabel_edit)

        self.ylabel_edit = QLineEdit()
        self.ylabel_edit.setPlaceholderText("Y Axis Label")
        self.ylabel_edit.textChanged.connect(lambda: self.axis_changed.emit())
        layout.addRow("Y Label:", self.ylabel_edit)

        self.auto_limits_cb = QCheckBox("Auto")
        self.auto_limits_cb.setChecked(True)
        self.auto_limits_cb.stateChanged.connect(lambda: self.axis_changed.emit())
        layout.addRow("Axis Limits:", self.auto_limits_cb)

        limits_grid = QGridLayout()
        self.xmin_spin = QDoubleSpinBox(); self.xmin_spin.setRange(-1e15, 1e15); self.xmin_spin.setDecimals(4)
        self.xmax_spin = QDoubleSpinBox(); self.xmax_spin.setRange(-1e15, 1e15); self.xmax_spin.setValue(10); self.xmax_spin.setDecimals(4)
        self.ymin_spin = QDoubleSpinBox(); self.ymin_spin.setRange(-1e15, 1e15); self.ymin_spin.setDecimals(4)
        self.ymax_spin = QDoubleSpinBox(); self.ymax_spin.setRange(-1e15, 1e15); self.ymax_spin.setValue(10); self.ymax_spin.setDecimals(4)
        for s in (self.xmin_spin, self.xmax_spin, self.ymin_spin, self.ymax_spin):
            s.valueChanged.connect(lambda: self.axis_changed.emit())
        limits_grid.addWidget(QLabel("X Min:"), 0, 0)
        limits_grid.addWidget(self.xmin_spin, 0, 1)
        limits_grid.addWidget(QLabel("X Max:"), 0, 2)
        limits_grid.addWidget(self.xmax_spin, 0, 3)
        limits_grid.addWidget(QLabel("Y Min:"), 1, 0)
        limits_grid.addWidget(self.ymin_spin, 1, 1)
        limits_grid.addWidget(QLabel("Y Max:"), 1, 2)
        limits_grid.addWidget(self.ymax_spin, 1, 3)
        layout.addRow(limits_grid)

        self.xlog_cb = QCheckBox("X Log Scale")
        self.xlog_cb.stateChanged.connect(lambda: self.axis_changed.emit())
        layout.addRow(self.xlog_cb)

        self.ylog_cb = QCheckBox("Y Log Scale")
        self.ylog_cb.stateChanged.connect(lambda: self.axis_changed.emit())
        layout.addRow(self.ylog_cb)

        self.grid_cb = QCheckBox("Show Grid")
        self.grid_cb.setChecked(True)
        self.grid_cb.stateChanged.connect(lambda: self.axis_changed.emit())
        layout.addRow(self.grid_cb)

        self.legend_cb = QCheckBox("Show Legend")
        self.legend_cb.setChecked(True)
        self.legend_cb.stateChanged.connect(lambda: self.axis_changed.emit())
        layout.addRow(self.legend_cb)

        self.tight_layout_cb = QCheckBox("Tight Layout")
        self.tight_layout_cb.setChecked(True)
        self.tight_layout_cb.stateChanged.connect(lambda: self.axis_changed.emit())
        layout.addRow(self.tight_layout_cb)

        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(6, 36)
        self.font_size_spin.setValue(12)
        self.font_size_spin.valueChanged.connect(lambda: self.axis_changed.emit())
        layout.addRow("Font Size:", self.font_size_spin)

    def get_axis_config(self):
        return {
            "title": self.title_edit.text(),
            "xlabel": self.xlabel_edit.text(),
            "ylabel": self.ylabel_edit.text(),
            "auto_limits": self.auto_limits_cb.isChecked(),
            "xmin": self.xmin_spin.value(),
            "xmax": self.xmax_spin.value(),
            "ymin": self.ymin_spin.value(),
            "ymax": self.ymax_spin.value(),
            "xlog": self.xlog_cb.isChecked(),
            "ylog": self.ylog_cb.isChecked(),
            "grid": self.grid_cb.isChecked(),
            "legend": self.legend_cb.isChecked(),
            "tight_layout": self.tight_layout_cb.isChecked(),
            "font_size": self.font_size_spin.value(),
        }


class Plotter2DWidget(QWidget):
    """
    Full-featured 2D plotting widget for a PyQt5 scientific application.

    Supports Line, Scatter, Bar, Histogram, Contour, Polar, Error Bars,
    Box Plot, Violin Plot, and Heatmap. Provides table-based and formula-based
    data input, axis controls, style controls, multi-series support, and export.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._logger = None
        self._series_list: list[DataSeriesModel] = []
        self._current_series_idx = 0
        self._build_ui()
        self._add_series()  # start with one series
        self._connect_signals()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_logger(self, fn):
        """Set an external logging callback: fn(message: str)."""
        self._logger = fn

    def load_file(self, path: str):
        """Load CSV/TSV data from *path* into the current series table."""
        self._log(f"Loading file: {path}")
        try:
            if path.endswith(".tsv") or path.endswith(".tab"):
                df = pd.read_csv(path, sep="\t")
            else:
                df = pd.read_csv(path)
            self.data_table.load_dataframe(df)
            self._sync_table_to_series()
            self._refresh_plot()
            self._log(f"Loaded {len(df)} rows from {os.path.basename(path)}")
        except Exception as exc:
            self._log(f"Error loading file: {exc}")
            QMessageBox.warning(self, "Load Error", str(exc))

    def export(self, path: str = None, fmt: str = "png", dpi: int = 150):
        """
        Export the current figure.

        Parameters
        ----------
        path : str, optional
            Destination file path. If None, a save dialog is shown.
        fmt : str
            Format: 'png', 'svg', 'pdf'.
        dpi : int
            Resolution for raster formats.
        """
        if path is None:
            filters = "PNG (*.png);;SVG (*.svg);;PDF (*.pdf);;All Files (*)"
            path, _ = QFileDialog.getSaveFileName(self, "Export Plot", "", filters)
            if not path:
                return
        try:
            self.figure.savefig(path, dpi=dpi, bbox_inches="tight")
            self._log(f"Exported plot to {path}")
        except Exception as exc:
            self._log(f"Export error: {exc}")
            QMessageBox.warning(self, "Export Error", str(exc))

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(2, 2, 2, 2)

        # Top toolbar
        top_bar = QHBoxLayout()
        btn_add_series = QPushButton("+ Series")
        btn_add_series.setToolTip("Add a new data series")
        btn_add_series.clicked.connect(self._add_series)
        top_bar.addWidget(btn_add_series)

        btn_remove_series = QPushButton("- Series")
        btn_remove_series.setToolTip("Remove the current data series")
        btn_remove_series.clicked.connect(self._remove_series)
        top_bar.addWidget(btn_remove_series)

        top_bar.addSpacing(20)

        btn_load = QPushButton("Load CSV")
        btn_load.clicked.connect(self._on_load_csv)
        top_bar.addWidget(btn_load)

        btn_export = QPushButton("Export")
        btn_export.clicked.connect(lambda: self.export())
        top_bar.addWidget(btn_export)

        btn_copy_plot = QPushButton("Copy Plot")
        btn_copy_plot.setToolTip("Copy current plot to clipboard as image")
        btn_copy_plot.clicked.connect(self._copy_plot_to_clipboard)
        top_bar.addWidget(btn_copy_plot)

        btn_plot = QPushButton("Plot")
        btn_plot.setStyleSheet("font-weight: bold;")
        btn_plot.clicked.connect(self._on_plot_clicked)
        top_bar.addWidget(btn_plot)

        top_bar.addSpacing(20)

        btn_pub = QPushButton("Publication Figure")
        btn_pub.setToolTip("Generate publication-quality figure (300+ DPI)")
        btn_pub.clicked.connect(self._generate_publication_figure)
        top_bar.addWidget(btn_pub)

        btn_multi = QPushButton("Multi-Panel")
        btn_multi.setToolTip("Create multi-panel subplot grids")
        btn_multi.clicked.connect(self._create_multi_panel)
        top_bar.addWidget(btn_multi)

        btn_annotate = QPushButton("Annotate")
        btn_annotate.setToolTip("Add annotations to current plot")
        btn_annotate.clicked.connect(self._add_annotation)
        top_bar.addWidget(btn_annotate)

        btn_template = QPushButton("Templates")
        btn_template.setToolTip("Apply or save plot style templates")
        btn_template.clicked.connect(self._show_template_menu)
        top_bar.addWidget(btn_template)

        btn_batch = QPushButton("Batch Plot")
        btn_batch.setToolTip("Generate multiple plots from CSV columns")
        btn_batch.clicked.connect(self._batch_plot)
        top_bar.addWidget(btn_batch)

        btn_anim = QPushButton("Animation")
        btn_anim.setToolTip("Create animated plots (GIF export)")
        btn_anim.clicked.connect(self._generate_animation)
        top_bar.addWidget(btn_anim)

        btn_inset = QPushButton("Inset Plot")
        btn_inset.setToolTip("Add a zoomed-in inset to the current plot")
        btn_inset.clicked.connect(self._add_inset_plot)
        top_bar.addWidget(btn_inset)

        btn_waterfall = QPushButton("Waterfall")
        btn_waterfall.setToolTip("Create waterfall plot from all series")
        btn_waterfall.clicked.connect(self._create_waterfall_plot)
        top_bar.addWidget(btn_waterfall)

        top_bar.addStretch()
        main_layout.addLayout(top_bar)

        # Splitter: left controls | right canvas
        splitter = QSplitter(Qt.Horizontal)

        # Left panel: tabs for data, style, axes
        left_tabs = QTabWidget()
        left_tabs.setMinimumWidth(280)
        left_tabs.setMaximumWidth(420)

        # Data tab
        data_widget = QWidget()
        data_layout = QVBoxLayout(data_widget)
        data_layout.setContentsMargins(2, 2, 2, 2)

        self.data_table = DataTableWidget()
        data_layout.addWidget(self.data_table)

        self.formula_widget = FormulaWidget()
        formula_group = QGroupBox("Generate from Formula")
        formula_group.setCheckable(True)
        formula_group.setChecked(False)
        fg_layout = QVBoxLayout(formula_group)
        fg_layout.addWidget(self.formula_widget)
        data_layout.addWidget(formula_group)

        left_tabs.addTab(data_widget, "Data")

        # Style tab
        self.style_panel = SeriesStylePanel()
        left_tabs.addTab(self.style_panel, "Style")

        # Axis tab
        self.axis_panel = AxisControlPanel()
        left_tabs.addTab(self.axis_panel, "Axes")

        splitter.addWidget(left_tabs)

        # Right panel: matplotlib canvas + toolbar
        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(8, 6), dpi=100)
        style_figure(self.figure)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.nav_toolbar = NavigationToolbar(self.canvas, self)
        canvas_layout.addWidget(self.nav_toolbar)
        canvas_layout.addWidget(self.canvas)

        splitter.addWidget(canvas_container)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter, stretch=1)

        # Status / log bar
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #555; font-size: 11px; padding: 2px;")
        main_layout.addWidget(self.status_label)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self.style_panel.series_combo.currentIndexChanged.connect(self._on_series_changed)
        self.style_panel.style_changed.connect(self._on_style_changed)
        self.axis_panel.axis_changed.connect(self._refresh_plot)
        self.data_table.data_changed.connect(self._on_table_data_changed)
        self.formula_widget.formula_applied.connect(self._on_formula_applied)

    # ------------------------------------------------------------------
    # Series management
    # ------------------------------------------------------------------

    def _add_series(self):
        idx = len(self._series_list) + 1
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                   "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        series = DataSeriesModel(name=f"Series {idx}")
        series.color = colors[(idx - 1) % len(colors)]
        series.label = series.name
        self._series_list.append(series)
        self.style_panel.series_combo.addItem(series.name)
        self.style_panel.series_combo.setCurrentIndex(len(self._series_list) - 1)
        self._log(f"Added {series.name}")

    def _remove_series(self):
        if len(self._series_list) <= 1:
            QMessageBox.information(self, "Info", "Cannot remove the last series.")
            return
        idx = self.style_panel.series_combo.currentIndex()
        if 0 <= idx < len(self._series_list):
            name = self._series_list[idx].name
            del self._series_list[idx]
            self.style_panel.series_combo.removeItem(idx)
            self._log(f"Removed {name}")
            self._refresh_plot()

    def _current_series(self) -> DataSeriesModel:
        idx = self.style_panel.series_combo.currentIndex()
        if 0 <= idx < len(self._series_list):
            return self._series_list[idx]
        return self._series_list[0] if self._series_list else None

    def _on_series_changed(self, idx):
        if 0 <= idx < len(self._series_list):
            self._current_series_idx = idx
            series = self._series_list[idx]
            self.style_panel.set_style(series)
            # Populate table with this series' data
            self.data_table.table.blockSignals(True)
            self.data_table._clear_table()
            if len(series.x_data) > 0:
                self.data_table.set_column_data(0, series.x_data)
            if len(series.y_data) > 0:
                self.data_table.set_column_data(1, series.y_data)
            if len(series.y_err) > 0:
                self.data_table.set_column_data(2, series.y_err)
            if len(series.z_data) > 0:
                self.data_table.set_column_data(3, series.z_data)
            self.data_table.table.blockSignals(False)

    def _on_style_changed(self):
        series = self._current_series()
        if series is None:
            return
        style = self.style_panel.get_style()
        series.plot_type = style["plot_type"]
        series.color = style["color"]
        series.line_width = style["line_width"]
        series.line_style = style["line_style"]
        series.marker = style["marker"]
        series.marker_size = style["marker_size"]
        series.alpha = style["alpha"]
        series.label = style["label"]
        series.hist_bins = style["hist_bins"]
        series.bar_width = style["bar_width"]
        series.colormap = style["colormap"]
        self._refresh_plot()

    # ------------------------------------------------------------------
    # Data handling
    # ------------------------------------------------------------------

    def _on_table_data_changed(self):
        self._sync_table_to_series()

    def _sync_table_to_series(self):
        series = self._current_series()
        if series is None:
            return
        series.x_data = self.data_table.get_column_data(0)
        series.y_data = self.data_table.get_column_data(1)
        series.y_err = self.data_table.get_column_data(2)
        series.z_data = self.data_table.get_column_data(3)

    def _on_formula_applied(self, x, y):
        series = self._current_series()
        if series is None:
            return
        series.x_data = x
        series.y_data = y
        self.data_table.table.blockSignals(True)
        self.data_table._clear_table()
        self.data_table.set_column_data(0, x)
        self.data_table.set_column_data(1, y)
        self.data_table.table.blockSignals(False)
        self._refresh_plot()
        self._log(f"Formula applied: {len(x)} points generated")

    def _on_load_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "",
            "CSV Files (*.csv);;TSV Files (*.tsv *.tab);;All Files (*)"
        )
        if path:
            self.load_file(path)

    def _on_plot_clicked(self):
        self._sync_table_to_series()
        self._refresh_plot()

    # ------------------------------------------------------------------
    # Plotting engine
    # ------------------------------------------------------------------

    def _refresh_plot(self):
        self.figure.clear()
        cfg = self.axis_panel.get_axis_config()

        # Determine if any series needs polar axes
        needs_polar = any(s.plot_type == "Polar" for s in self._series_list
                         if len(s.x_data) > 0 or len(s.y_data) > 0)

        if needs_polar:
            ax = self.figure.add_subplot(111, projection="polar")
        else:
            ax = self.figure.add_subplot(111)

        has_data = False
        for series in self._series_list:
            try:
                drawn = self._draw_series(ax, series, cfg)
                if drawn:
                    has_data = True
            except Exception as exc:
                self._log(f"Plot error ({series.name}): {exc}")

        # Apply axis configuration
        fs = cfg["font_size"]
        if cfg["title"]:
            ax.set_title(cfg["title"], fontsize=fs + 2)
        if cfg["xlabel"]:
            ax.set_xlabel(cfg["xlabel"], fontsize=fs)
        if cfg["ylabel"]:
            ax.set_ylabel(cfg["ylabel"], fontsize=fs)

        if not cfg["auto_limits"] and not needs_polar:
            ax.set_xlim(cfg["xmin"], cfg["xmax"])
            ax.set_ylim(cfg["ymin"], cfg["ymax"])

        if not needs_polar:
            if cfg["xlog"]:
                ax.set_xscale("log")
            if cfg["ylog"]:
                ax.set_yscale("log")

        ax.grid(cfg["grid"], alpha=0.4)

        if cfg["legend"] and has_data:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=max(fs - 2, 6), loc="best",
                          framealpha=0.85, edgecolor="#cccccc",
                          fancybox=True, shadow=False)

        if cfg["tight_layout"]:
            try:
                self.figure.tight_layout()
            except Exception:
                pass

        self.canvas.draw_idle()

    def _draw_series(self, ax, series: DataSeriesModel, cfg: dict) -> bool:
        """Draw one data series on the axes. Returns True if something was drawn."""
        pt = series.plot_type
        x = series.x_data
        y = series.y_data
        mk = series.marker if series.marker != "None" else None
        ls = series.line_style if series.line_style != "None" else "None"
        lbl = series.label or series.name

        if pt == "Line":
            if len(x) == 0 or len(y) == 0:
                return False
            n = min(len(x), len(y))
            ax.plot(x[:n], y[:n], color=series.color, linewidth=series.line_width,
                    linestyle=ls, marker=mk, markersize=series.marker_size,
                    alpha=series.alpha, label=lbl)
            return True

        elif pt == "Scatter":
            if len(x) == 0 or len(y) == 0:
                return False
            n = min(len(x), len(y))
            ax.scatter(x[:n], y[:n], c=series.color, s=series.marker_size ** 2,
                       marker=mk or "o", alpha=series.alpha, label=lbl,
                       edgecolors="face")
            return True

        elif pt == "Bar":
            if len(x) == 0 or len(y) == 0:
                return False
            n = min(len(x), len(y))
            ax.bar(x[:n], y[:n], width=series.bar_width, color=series.color,
                   alpha=series.alpha, label=lbl, edgecolor="white", linewidth=0.5)
            return True

        elif pt == "Histogram":
            data = y if len(y) > 0 else x
            if len(data) == 0:
                return False
            ax.hist(data, bins=series.hist_bins, color=series.color,
                    alpha=series.alpha, label=lbl, edgecolor="white", linewidth=0.5)
            return True

        elif pt == "Contour":
            if len(x) == 0 or len(y) == 0:
                return False
            z = series.z_data
            # If Z data provided and forms a grid, use it; otherwise generate
            side = int(np.sqrt(len(x)))
            if side * side == len(x) and side * side == len(y):
                if len(z) == side * side:
                    X = x.reshape(side, side)
                    Y = y.reshape(side, side)
                    Z = z.reshape(side, side)
                else:
                    X = x.reshape(side, side)
                    Y = y.reshape(side, side)
                    Z = np.sin(X) * np.cos(Y)
                cs = ax.contourf(X, Y, Z, levels=20, cmap=series.colormap, alpha=series.alpha)
                self.figure.colorbar(cs, ax=ax, shrink=0.8)
                return True
            else:
                self._log("Contour requires grid data (N*N points).")
                return False

        elif pt == "Polar":
            if len(x) == 0 or len(y) == 0:
                return False
            n = min(len(x), len(y))
            ax.plot(x[:n], y[:n], color=series.color, linewidth=series.line_width,
                    linestyle=ls, marker=mk, markersize=series.marker_size,
                    alpha=series.alpha, label=lbl)
            return True

        elif pt == "Error Bars":
            if len(x) == 0 or len(y) == 0:
                return False
            n = min(len(x), len(y))
            yerr = series.y_err[:n] if len(series.y_err) >= n else None
            ax.errorbar(x[:n], y[:n], yerr=yerr, color=series.color,
                        linewidth=series.line_width, linestyle=ls, marker=mk or "o",
                        markersize=series.marker_size, alpha=series.alpha,
                        label=lbl, capsize=3, elinewidth=1)
            return True

        elif pt == "Box Plot":
            data = y if len(y) > 0 else x
            if len(data) == 0:
                return False
            bp = ax.boxplot([data], patch_artist=True, labels=[lbl])
            for patch in bp["boxes"]:
                patch.set_facecolor(series.color)
                patch.set_alpha(series.alpha)
            return True

        elif pt == "Violin Plot":
            data = y if len(y) > 0 else x
            if len(data) == 0:
                return False
            parts = ax.violinplot([data], showmeans=True, showmedians=True)
            for pc in parts.get("bodies", []):
                pc.set_facecolor(series.color)
                pc.set_alpha(series.alpha)
            return True

        elif pt == "Heatmap":
            if len(x) == 0 or len(y) == 0:
                return False
            z = series.z_data
            side = int(np.sqrt(len(z))) if len(z) > 0 else 0
            if side > 0 and side * side == len(z):
                Z = z.reshape(side, side)
            elif len(y) > 1 and len(x) > 1:
                # Create a heatmap from x, y as bin edges
                n = min(len(x), len(y))
                Z, xedges, yedges = np.histogram2d(x[:n], y[:n], bins=series.hist_bins)
                Z = Z.T
            else:
                return False
            im = ax.imshow(Z, aspect="auto", cmap=series.colormap, alpha=series.alpha,
                           origin="lower")
            self.figure.colorbar(im, ax=ax, shrink=0.8)
            return True

        elif pt == "Error Envelope":
            if len(x) == 0 or len(y) == 0:
                return False
            n = min(len(x), len(y))
            xd, yd = x[:n], y[:n]
            yerr = series.y_err[:n] if len(series.y_err) >= n else np.std(yd) * 0.1 * np.ones(n)
            ax.plot(xd, yd, color=series.color, linewidth=series.line_width,
                    linestyle=ls, marker=mk, markersize=series.marker_size,
                    alpha=series.alpha, label=lbl)
            ax.fill_between(xd, yd - yerr, yd + yerr, color=series.color,
                            alpha=series.alpha * 0.25, label=f"{lbl} (envelope)")
            return True

        elif pt == "Waterfall":
            # Waterfall is handled per-series: offset each series vertically by its index
            if len(x) == 0 or len(y) == 0:
                return False
            n = min(len(x), len(y))
            idx = self._series_list.index(series) if series in self._series_list else 0
            offset = idx * np.ptp(y[:n]) * 0.5 if np.ptp(y[:n]) > 0 else idx
            ax.plot(x[:n], y[:n] + offset, color=series.color,
                    linewidth=series.line_width, linestyle=ls,
                    alpha=series.alpha, label=lbl)
            ax.fill_between(x[:n], offset, y[:n] + offset,
                            color=series.color, alpha=series.alpha * 0.15)
            return True

        return False

    # ------------------------------------------------------------------
    # Publication Figure Generation
    # ------------------------------------------------------------------

    def _generate_publication_figure(self):
        """Generate a publication-quality figure with journal-ready formatting."""
        templates = list(PUBLICATION_TEMPLATES.keys())
        template, ok = QInputDialog.getItem(
            self, "Publication Figure", "Select journal style:", templates, editable=False
        )
        if not ok:
            return

        style = PUBLICATION_TEMPLATES[template]
        dpi_val = style.get("savefig.dpi", 300)
        figsize = style.get("figure.figsize", (3.5, 2.625))

        # Save current rcParams and apply publication style
        old_params = {k: plt.rcParams.get(k) for k in style if k in plt.rcParams}
        for k, v in style.items():
            if k in plt.rcParams:
                plt.rcParams[k] = v

        try:
            # Create a high-DPI figure
            pub_fig = Figure(figsize=figsize, dpi=dpi_val)
            style_figure(pub_fig)
            pub_fig.set_tight_layout(True)

            cfg = self.axis_panel.get_axis_config()
            needs_polar = any(s.plot_type == "Polar" for s in self._series_list
                              if len(s.x_data) > 0 or len(s.y_data) > 0)
            if needs_polar:
                ax = pub_fig.add_subplot(111, projection="polar")
            else:
                ax = pub_fig.add_subplot(111)

            for series in self._series_list:
                self._draw_series(ax, series, cfg)

            fs = style.get("font.size", 10)
            if cfg["title"]:
                ax.set_title(cfg["title"], fontsize=style.get("axes.titlesize", fs + 2), fontweight="bold")
            if cfg["xlabel"]:
                ax.set_xlabel(cfg["xlabel"], fontsize=style.get("axes.labelsize", fs))
            if cfg["ylabel"]:
                ax.set_ylabel(cfg["ylabel"], fontsize=style.get("axes.labelsize", fs))

            ax.tick_params(axis="both", which="major", direction="in", top=True, right=True)
            if cfg["grid"]:
                ax.grid(True, alpha=0.3, linewidth=0.5)
            if cfg["legend"]:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(fontsize=style.get("legend.fontsize", 7),
                              frameon=True, fancybox=False, edgecolor="black",
                              framealpha=0.9)

            pub_fig.tight_layout(pad=0.3)

            # Save dialog
            path, _ = QFileDialog.getSaveFileName(
                self, f"Export Publication Figure ({template})", f"figure_{template.lower().replace(' ', '_')}.png",
                "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;EPS (*.eps);;TIFF (*.tiff);;All Files (*)"
            )
            if path:
                pub_fig.savefig(path, dpi=dpi_val, bbox_inches="tight",
                                pad_inches=0.02, facecolor="white", edgecolor="none")
                self._log(f"Publication figure saved: {path} ({template}, {dpi_val} DPI)")
        except Exception as exc:
            self._log(f"Publication figure error: {exc}")
            QMessageBox.warning(self, "Publication Figure Error", str(exc))
        finally:
            # Restore rcParams
            for k, v in old_params.items():
                if v is not None:
                    plt.rcParams[k] = v

    # ------------------------------------------------------------------
    # Multi-Panel Figure Support
    # ------------------------------------------------------------------

    def _create_multi_panel(self):
        """Create a multi-panel subplot grid from existing series."""
        if len(self._series_list) < 1:
            QMessageBox.information(self, "Multi-Panel", "Add at least one data series first.")
            return

        presets = ["1x2 (side by side)", "2x1 (stacked)", "2x2 (grid)", "3x1 (triple stack)", "Custom NxM"]
        choice, ok = QInputDialog.getItem(
            self, "Multi-Panel Layout", "Select layout:", presets, editable=False
        )
        if not ok:
            return

        if choice == "Custom NxM":
            nrows, ok1 = QInputDialog.getInt(self, "Rows", "Number of rows:", 2, 1, 10)
            if not ok1:
                return
            ncols, ok2 = QInputDialog.getInt(self, "Columns", "Number of columns:", 2, 1, 10)
            if not ok2:
                return
        else:
            layout_map = {
                "1x2 (side by side)": (1, 2),
                "2x1 (stacked)": (2, 1),
                "2x2 (grid)": (2, 2),
                "3x1 (triple stack)": (3, 1),
            }
            nrows, ncols = layout_map[choice]

        total_panels = nrows * ncols
        cfg = self.axis_panel.get_axis_config()

        self.figure.clear()
        gs = GridSpec(nrows, ncols, figure=self.figure, hspace=0.35, wspace=0.3)

        # Distribute series across panels (round-robin if fewer panels than series)
        for i in range(min(total_panels, len(self._series_list))):
            row_idx = i // ncols
            col_idx = i % ncols
            series = self._series_list[i]

            needs_polar = series.plot_type == "Polar"
            if needs_polar:
                ax = self.figure.add_subplot(gs[row_idx, col_idx], projection="polar")
            else:
                ax = self.figure.add_subplot(gs[row_idx, col_idx])

            try:
                self._draw_series(ax, series, cfg)
            except Exception as exc:
                ax.text(0.5, 0.5, f"Error: {exc}", transform=ax.transAxes,
                        ha="center", va="center", color="red", fontsize=8)

            panel_label = chr(ord('a') + i)
            ax.set_title(f"({panel_label}) {series.label or series.name}", fontsize=cfg["font_size"])
            ax.tick_params(axis="both", which="major", direction="in")
            if cfg["grid"]:
                ax.grid(True, alpha=0.3)

        # Fill empty panels
        for i in range(len(self._series_list), total_panels):
            row_idx = i // ncols
            col_idx = i % ncols
            ax = self.figure.add_subplot(gs[row_idx, col_idx])
            ax.text(0.5, 0.5, "(empty)", transform=ax.transAxes,
                    ha="center", va="center", color="#ccc", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

        try:
            self.figure.tight_layout()
        except Exception:
            pass
        self.canvas.draw_idle()
        self._log(f"Multi-panel figure: {nrows}x{ncols} ({len(self._series_list)} series)")

    # ------------------------------------------------------------------
    # Annotation Tools
    # ------------------------------------------------------------------

    def _add_annotation(self):
        """Add annotations (text, arrows, boxes, highlights) to the current plot."""
        ann_types = ["Text Label", "Arrow with Text", "Rectangle Region", "Vertical Span", "Horizontal Span"]
        ann_type, ok = QInputDialog.getItem(
            self, "Annotation", "Annotation type:", ann_types, editable=False
        )
        if not ok:
            return

        # Get current axes
        axes = self.figure.get_axes()
        if not axes:
            QMessageBox.warning(self, "Annotation", "Plot something first.")
            return
        ax = axes[0]

        try:
            if ann_type == "Text Label":
                text, ok = QInputDialog.getText(self, "Text", "Text to add:")
                if not ok or not text:
                    return
                x_pos, ok = QInputDialog.getDouble(self, "Position", "X position:", 0.5)
                if not ok:
                    return
                y_pos, ok = QInputDialog.getDouble(self, "Position", "Y position:", 0.5)
                if not ok:
                    return
                fontsize, ok = QInputDialog.getInt(self, "Font", "Font size:", 12, 4, 72)
                if not ok:
                    return
                ax.text(x_pos, y_pos, text, fontsize=fontsize,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7),
                        transform=ax.transData)

            elif ann_type == "Arrow with Text":
                text, ok = QInputDialog.getText(self, "Text", "Annotation text:")
                if not ok or not text:
                    return
                x_point, ok = QInputDialog.getDouble(self, "Arrow Target", "Target X:")
                if not ok:
                    return
                y_point, ok = QInputDialog.getDouble(self, "Arrow Target", "Target Y:")
                if not ok:
                    return
                x_text, ok = QInputDialog.getDouble(self, "Text Position", "Text X:", x_point + 1)
                if not ok:
                    return
                y_text, ok = QInputDialog.getDouble(self, "Text Position", "Text Y:", y_point + 1)
                if not ok:
                    return
                ax.annotate(text, xy=(x_point, y_point), xytext=(x_text, y_text),
                            fontsize=10,
                            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

            elif ann_type == "Rectangle Region":
                x1, ok = QInputDialog.getDouble(self, "Region", "X start:")
                if not ok:
                    return
                x2, ok = QInputDialog.getDouble(self, "Region", "X end:", x1 + 1)
                if not ok:
                    return
                y1, ok = QInputDialog.getDouble(self, "Region", "Y start:")
                if not ok:
                    return
                y2, ok = QInputDialog.getDouble(self, "Region", "Y end:", y1 + 1)
                if not ok:
                    return
                rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                           linewidth=1.5, edgecolor="red",
                                           facecolor="red", alpha=0.15)
                ax.add_patch(rect)

            elif ann_type == "Vertical Span":
                x1, ok = QInputDialog.getDouble(self, "Span", "X start:")
                if not ok:
                    return
                x2, ok = QInputDialog.getDouble(self, "Span", "X end:", x1 + 1)
                if not ok:
                    return
                ax.axvspan(x1, x2, alpha=0.2, color="blue")

            elif ann_type == "Horizontal Span":
                y1, ok = QInputDialog.getDouble(self, "Span", "Y start:")
                if not ok:
                    return
                y2, ok = QInputDialog.getDouble(self, "Span", "Y end:", y1 + 1)
                if not ok:
                    return
                ax.axhspan(y1, y2, alpha=0.2, color="green")

            self.canvas.draw_idle()
            self._log(f"Added annotation: {ann_type}")
        except Exception as exc:
            self._log(f"Annotation error: {exc}")
            QMessageBox.warning(self, "Annotation Error", str(exc))

    # ------------------------------------------------------------------
    # Template System
    # ------------------------------------------------------------------

    def _show_template_menu(self):
        """Show a menu with template options: apply built-in, save custom, load custom."""
        menu = QMenu(self)

        # Built-in templates
        builtin_menu = menu.addMenu("Apply Built-in Template")
        for name in PUBLICATION_TEMPLATES:
            action = builtin_menu.addAction(name)
            action.triggered.connect(lambda checked, n=name: self._apply_template(n))

        menu.addSeparator()
        save_action = menu.addAction("Save Current Style as Template...")
        save_action.triggered.connect(self._save_template)

        load_action = menu.addAction("Load Template from File...")
        load_action.triggered.connect(self._load_template)

        # Show at button position
        btn = self.sender()
        if btn:
            menu.exec_(btn.mapToGlobal(btn.rect().bottomLeft()))
        else:
            menu.exec_(self.mapToGlobal(self.rect().center()))

    def _apply_template(self, template_name: str):
        """Apply a built-in publication template to the current plot."""
        if template_name not in PUBLICATION_TEMPLATES:
            return
        style = PUBLICATION_TEMPLATES[template_name]
        for k, v in style.items():
            if k in plt.rcParams:
                plt.rcParams[k] = v
        self._refresh_plot()
        self._log(f"Applied template: {template_name}")

    def _save_template(self):
        """Save the current plot style as a JSON template file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Template", "my_template.json", "JSON Files (*.json)"
        )
        if not path:
            return
        cfg = self.axis_panel.get_axis_config()
        series_styles = []
        for s in self._series_list:
            series_styles.append({
                "plot_type": s.plot_type, "color": s.color,
                "line_width": s.line_width, "line_style": s.line_style,
                "marker": s.marker, "marker_size": s.marker_size,
                "alpha": s.alpha, "label": s.label,
                "hist_bins": s.hist_bins, "bar_width": s.bar_width,
                "colormap": s.colormap,
            })
        template = {
            "axis_config": cfg,
            "series_styles": series_styles,
            "rcParams": {
                "font.size": plt.rcParams.get("font.size", 12),
                "lines.linewidth": plt.rcParams.get("lines.linewidth", 1.5),
                "axes.linewidth": plt.rcParams.get("axes.linewidth", 0.8),
            },
        }
        try:
            with open(path, "w") as f:
                json.dump(template, f, indent=2)
            self._log(f"Template saved: {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Save Error", str(exc))

    def _load_template(self):
        """Load a custom template from a JSON file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Template", "", "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        try:
            with open(path, "r") as f:
                template = json.load(f)

            # Apply rcParams
            for k, v in template.get("rcParams", {}).items():
                if k in plt.rcParams:
                    plt.rcParams[k] = v

            # Apply axis config
            acfg = template.get("axis_config", {})
            if acfg.get("title"):
                self.axis_panel.title_edit.setText(acfg["title"])
            if acfg.get("xlabel"):
                self.axis_panel.xlabel_edit.setText(acfg["xlabel"])
            if acfg.get("ylabel"):
                self.axis_panel.ylabel_edit.setText(acfg["ylabel"])
            if "font_size" in acfg:
                self.axis_panel.font_size_spin.setValue(acfg["font_size"])
            if "grid" in acfg:
                self.axis_panel.grid_cb.setChecked(acfg["grid"])

            # Apply series styles
            for i, ss in enumerate(template.get("series_styles", [])):
                if i >= len(self._series_list):
                    self._add_series()
                s = self._series_list[i]
                for key in ("plot_type", "color", "line_width", "line_style",
                            "marker", "marker_size", "alpha", "label",
                            "hist_bins", "bar_width", "colormap"):
                    if key in ss:
                        setattr(s, key, ss[key])

            self._refresh_plot()
            self._log(f"Template loaded: {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Load Error", str(exc))

    # ------------------------------------------------------------------
    # Batch Plot from CSV
    # ------------------------------------------------------------------

    def _batch_plot(self):
        """Generate multiple plots from CSV data columns automatically."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV for Batch Plot", "",
            "CSV Files (*.csv);;TSV Files (*.tsv);;All Files (*)"
        )
        if not path:
            return

        try:
            df = pd.read_csv(path)
        except Exception as exc:
            QMessageBox.warning(self, "Batch Plot Error", f"Cannot read file:\n{exc}")
            return

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            QMessageBox.warning(self, "Batch Plot", "Need at least 2 numeric columns.")
            return

        # Ask for X column
        x_choices = ["(index)"] + num_cols
        x_col, ok = QInputDialog.getItem(
            self, "Batch Plot", "Select X column:", x_choices, editable=False
        )
        if not ok:
            return

        # Ask for output directory
        out_dir = QFileDialog.getExistingDirectory(self, "Output Directory for Batch Plots")
        if not out_dir:
            return

        # Ask for plot type
        ptypes = ["Line", "Scatter", "Bar", "Histogram"]
        ptype, ok = QInputDialog.getItem(
            self, "Batch Plot", "Plot type:", ptypes, editable=False
        )
        if not ok:
            return

        y_cols = [c for c in num_cols if c != x_col]
        x_data = df.index.values if x_col == "(index)" else df[x_col].values
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(y_cols), 1)))

        generated = 0
        for i, col in enumerate(y_cols):
            try:
                fig = Figure(figsize=(8, 5), dpi=150)
                style_figure(fig)
                ax = fig.add_subplot(111)
                y_data = df[col].dropna().values
                x_plot = x_data[:len(y_data)]

                if ptype == "Line":
                    ax.plot(x_plot, y_data, color=colors[i % len(colors)], linewidth=1.5)
                elif ptype == "Scatter":
                    ax.scatter(x_plot, y_data, color=colors[i % len(colors)], s=15, alpha=0.7)
                elif ptype == "Bar":
                    ax.bar(range(len(y_data)), y_data, color=colors[i % len(colors)], alpha=0.8)
                elif ptype == "Histogram":
                    ax.hist(y_data, bins="auto", color=colors[i % len(colors)],
                            alpha=0.8, edgecolor="white")

                ax.set_title(col, fontsize=12)
                ax.set_xlabel(x_col if x_col != "(index)" else "Index")
                ax.set_ylabel(col)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()

                out_path = os.path.join(out_dir, f"batch_{col.replace(' ', '_')}.png")
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                generated += 1
            except Exception as exc:
                self._log(f"Batch plot error for '{col}': {exc}")

        self._log(f"Batch plot complete: {generated}/{len(y_cols)} figures saved to {out_dir}")
        QMessageBox.information(self, "Batch Plot",
                                f"Generated {generated} plots in:\n{out_dir}")

    # ------------------------------------------------------------------
    # Animation Generation
    # ------------------------------------------------------------------

    def _generate_animation(self):
        """Create animated plots and export as GIF."""
        anim_types = [
            "Parameter Sweep (vary coefficient)",
            "Time Series (cumulative reveal)",
            "Phase Animation (shift wave)",
        ]
        anim_type, ok = QInputDialog.getItem(
            self, "Animation", "Animation type:", anim_types, editable=False
        )
        if not ok:
            return

        # Common settings
        n_frames, ok = QInputDialog.getInt(self, "Frames", "Number of frames:", 30, 5, 200)
        if not ok:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Animation", "animation.gif", "GIF (*.gif);;All Files (*)"
        )
        if not path:
            return

        try:
            import imageio
        except ImportError:
            # Fall back to matplotlib save with pillow
            pass

        cfg = self.axis_panel.get_axis_config()
        series = self._current_series()
        if series is None or (len(series.x_data) == 0 and len(series.y_data) == 0):
            QMessageBox.warning(self, "Animation", "Current series has no data. Generate or load data first.")
            return

        x = series.x_data.copy()
        y = series.y_data.copy()

        try:
            frames = []
            for frame_i in range(n_frames):
                fig = Figure(figsize=(8, 5), dpi=100)
                style_figure(fig)
                ax = fig.add_subplot(111)
                t = frame_i / max(n_frames - 1, 1)

                if "Parameter Sweep" in anim_type:
                    coeff = 0.5 + 2.0 * t
                    y_mod = y * coeff
                    ax.plot(x, y_mod, color=series.color, linewidth=series.line_width)
                    ax.set_title(f"Parameter = {coeff:.2f}", fontsize=12)

                elif "Time Series" in anim_type:
                    n_show = max(1, int(len(x) * (t * 0.9 + 0.1)))
                    ax.plot(x[:n_show], y[:n_show], color=series.color, linewidth=series.line_width)
                    ax.set_xlim(x[0], x[-1])
                    ax.set_ylim(np.nanmin(y) * 1.1, np.nanmax(y) * 1.1)
                    ax.set_title(f"Frame {frame_i + 1}/{n_frames}", fontsize=12)

                elif "Phase Animation" in anim_type:
                    shift = 2 * np.pi * t
                    y_shifted = np.roll(y, int(len(y) * t) % len(y))
                    ax.plot(x, y_shifted, color=series.color, linewidth=series.line_width)
                    ax.set_title(f"Phase shift = {shift:.2f} rad", fontsize=12)

                if cfg["xlabel"]:
                    ax.set_xlabel(cfg["xlabel"])
                if cfg["ylabel"]:
                    ax.set_ylabel(cfg["ylabel"])
                ax.grid(True, alpha=0.3)
                fig.tight_layout()

                # Render to image buffer
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=100)
                buf.seek(0)
                plt.close(fig)
                frames.append(buf)

            # Assemble GIF using PIL
            from PIL import Image
            images = [Image.open(buf) for buf in frames]
            images[0].save(
                path, save_all=True, append_images=images[1:],
                duration=100, loop=0, optimize=True
            )
            for buf in frames:
                buf.close()

            self._log(f"Animation saved: {path} ({n_frames} frames)")
            QMessageBox.information(self, "Animation",
                                    f"GIF saved to:\n{path}\n({n_frames} frames)")
        except ImportError:
            QMessageBox.warning(self, "Animation",
                                "PIL (Pillow) is required for GIF export.\nInstall with: pip install Pillow")
        except Exception as exc:
            self._log(f"Animation error: {exc}")
            QMessageBox.warning(self, "Animation Error", str(exc))

    # ------------------------------------------------------------------
    # Inset Plot
    # ------------------------------------------------------------------

    def _add_inset_plot(self):
        """Add a zoomed-in inset to the current plot."""
        if not self._series_list:
            self._log("No series to create inset from.")
            return
        # Find first series with data
        series = None
        for s in self._series_list:
            if len(s.x_data) > 0 and len(s.y_data) > 0:
                series = s
                break
        if series is None:
            self._log("No data available for inset plot.")
            return

        x = series.x_data
        y = series.y_data
        n = min(len(x), len(y))
        if n < 4:
            self._log("Need at least 4 data points for inset.")
            return

        # Determine zoom region: middle 25% of data range
        x_min, x_max = np.min(x[:n]), np.max(x[:n])
        x_center = (x_min + x_max) / 2
        x_span = (x_max - x_min) * 0.25
        x1, x2 = x_center - x_span / 2, x_center + x_span / 2

        mask = (x[:n] >= x1) & (x[:n] <= x2)
        if np.sum(mask) < 2:
            self._log("Not enough points in inset region.")
            return

        # Get the current axes and add inset
        axes_list = self.figure.get_axes()
        if not axes_list:
            self._log("No axes available. Plot data first.")
            return
        ax = axes_list[0]

        # Create inset axes
        inset_ax = ax.inset_axes([0.55, 0.55, 0.4, 0.4])  # [x, y, width, height] in axes fraction
        inset_ax.plot(x[:n][mask], y[:n][mask], color=series.color,
                      linewidth=series.line_width * 0.8,
                      linestyle=series.line_style if series.line_style != "None" else "-")
        inset_ax.set_xlim(x1, x2)
        y_masked = y[:n][mask]
        y_pad = (np.max(y_masked) - np.min(y_masked)) * 0.1 if np.ptp(y_masked) > 0 else 0.5
        inset_ax.set_ylim(np.min(y_masked) - y_pad, np.max(y_masked) + y_pad)
        inset_ax.set_title("Zoom", fontsize=7)
        inset_ax.tick_params(labelsize=6)
        inset_ax.patch.set_alpha(0.9)

        try:
            ax.indicate_inset_zoom(inset_ax, edgecolor="#888888", linewidth=0.8)
        except Exception:
            pass  # indicate_inset_zoom may not be available in older matplotlib

        self.canvas.draw_idle()
        self._log("Inset plot added (middle 25% of x-range).")

    # ------------------------------------------------------------------
    # Waterfall Plot (standalone multi-series)
    # ------------------------------------------------------------------

    def _create_waterfall_plot(self):
        """Create a waterfall plot from all series with vertical offsets."""
        active_series = [s for s in self._series_list
                         if len(s.x_data) > 0 and len(s.y_data) > 0]
        if len(active_series) < 2:
            self._log("Waterfall plot requires at least 2 series with data.")
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        cmap = cm.get_cmap("viridis", len(active_series))

        max_range = max(np.ptp(s.y_data[:min(len(s.x_data), len(s.y_data))])
                        for s in active_series)
        offset_step = max_range * 0.6 if max_range > 0 else 1.0

        for idx, series in enumerate(active_series):
            n = min(len(series.x_data), len(series.y_data))
            offset = idx * offset_step
            color = cmap(idx / max(len(active_series) - 1, 1))
            ax.plot(series.x_data[:n], series.y_data[:n] + offset,
                    color=color, linewidth=series.line_width,
                    label=series.label or series.name)
            ax.fill_between(series.x_data[:n], offset,
                            series.y_data[:n] + offset,
                            color=color, alpha=0.12)

        cfg = self.axis_panel.get_axis_config()
        ax.set_title(cfg["title"] or "Waterfall Plot", fontsize=cfg["font_size"] + 2)
        ax.set_xlabel(cfg["xlabel"] or "X", fontsize=cfg["font_size"])
        ax.set_ylabel("Intensity (offset)", fontsize=cfg["font_size"])
        ax.legend(fontsize=max(cfg["font_size"] - 2, 6), loc="best",
                  framealpha=0.85, fancybox=True)
        ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self._log(f"Waterfall plot created with {len(active_series)} series.")

    # ------------------------------------------------------------------
    # Clipboard
    # ------------------------------------------------------------------

    def _copy_plot_to_clipboard(self):
        """Copy current plot to clipboard as image."""
        import io
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtGui import QImage
        buf = io.BytesIO()
        self.figure.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                            facecolor=self.figure.get_facecolor())
        buf.seek(0)
        img = QImage()
        img.loadFromData(buf.read())
        QApplication.clipboard().setImage(img)
        self._log("Plot copied to clipboard.")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _log(self, msg: str):
        self.status_label.setText(msg)
        if self._logger:
            try:
                self._logger(msg)
            except Exception:
                pass
