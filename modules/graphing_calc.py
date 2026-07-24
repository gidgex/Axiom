"""
graphing_calc.py - Graphing Calculator Widget for Axiom Scientific Suite

A powerful Desmos-like desktop graphing calculator supporting explicit y=f(x),
implicit equations f(x,y)=0, parametric curves, polar curves, inequalities,
parameter sliders, intersection finding, derivative/integral display, and more.
"""

import os
import traceback
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QPushButton,
    QComboBox, QLabel, QLineEdit, QDoubleSpinBox, QCheckBox,
    QColorDialog, QFileDialog, QGroupBox, QFormLayout, QMessageBox,
    QSizePolicy, QGridLayout, QScrollArea, QApplication, QSlider,
    QFrame, QToolButton, QSpinBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QColor, QFont, QPalette, QIcon

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    from mpl_style import style_figure, style_axes
except ImportError:
    def style_figure(f): pass
    def style_axes(a): pass


# ---------------------------------------------------------------------------
# Safe math namespace for expression evaluation
# ---------------------------------------------------------------------------

_SAFE_NAMESPACE = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan, "atan2": np.arctan2,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "log": np.log, "log2": np.log2, "log10": np.log10,
    "sqrt": np.sqrt, "abs": np.abs, "sign": np.sign,
    "floor": np.floor, "ceil": np.ceil,
    "pi": np.pi, "e": np.e,
    "inf": np.inf, "nan": np.nan,
    "maximum": np.maximum, "minimum": np.minimum,
    "clip": np.clip,
    "np": np,
}

FUNC_TYPES = ["y = f(x)", "Implicit", "Parametric", "Polar", "Inequality"]
FUNC_TYPE_HINTS = {
    "y = f(x)": "e.g. sin(x)/x, x**2 - 3*x + 1",
    "Implicit": "e.g. x**2 + y**2 - 1  (=0)",
    "Parametric": "e.g. cos(t), sin(t)",
    "Polar": "e.g. 1 + cos(theta)",
    "Inequality": "e.g. y > sin(x)  or  y < x**2",
}

DEFAULT_COLORS = [
    "#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0",
    "#00BCD4", "#E91E63", "#8BC34A", "#FF5722", "#3F51B5",
]

PRESETS = {
    "Parabola":     ("y = f(x)", "x**2"),
    "Circle":       ("Implicit", "x**2 + y**2 - 1"),
    "Sine":         ("y = f(x)", "sin(x)"),
    "Lissajous":    ("Parametric", "cos(3*t), sin(2*t)"),
    "Cardioid":     ("Polar", "1 + cos(theta)"),
    "Rose Curve":   ("Polar", "cos(3*theta)"),
    "Spirograph":   ("Parametric", "(5-1)*cos(t)+1*cos((5-1)*t), (5-1)*sin(t)-1*sin((5-1)*t)"),
    "Hyperbola":    ("Implicit", "x**2/4 - y**2/9 - 1"),
    "Gaussian":     ("y = f(x)", "exp(-x**2)"),
}

MAX_FUNCTIONS = 10


# ---------------------------------------------------------------------------
# FunctionRow widget for function list panel
# ---------------------------------------------------------------------------

class FunctionRow(QFrame):
    """Single row in the function list representing one equation."""
    changed = pyqtSignal()
    delete_requested = pyqtSignal(object)

    def __init__(self, color="#2196F3", parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self._color = QColor(color)
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        self.type_combo = QComboBox()
        self.type_combo.addItems(FUNC_TYPES)
        self.type_combo.setFixedWidth(100)
        self.type_combo.currentIndexChanged.connect(self._on_type_changed)
        layout.addWidget(self.type_combo)

        self.expr_edit = QLineEdit()
        self.expr_edit.setPlaceholderText(FUNC_TYPE_HINTS["y = f(x)"])
        self.expr_edit.setMinimumWidth(200)
        self.expr_edit.textChanged.connect(lambda: self.changed.emit())
        layout.addWidget(self.expr_edit, stretch=1)

        self.color_btn = QToolButton()
        self.color_btn.setFixedSize(24, 24)
        self._update_color_btn()
        self.color_btn.clicked.connect(self._pick_color)
        layout.addWidget(self.color_btn)

        self.visible_cb = QCheckBox()
        self.visible_cb.setChecked(True)
        self.visible_cb.setToolTip("Show / Hide")
        self.visible_cb.stateChanged.connect(lambda: self.changed.emit())
        layout.addWidget(self.visible_cb)

        self.del_btn = QToolButton()
        self.del_btn.setText("\u00d7")
        self.del_btn.setFixedSize(24, 24)
        self.del_btn.clicked.connect(lambda: self.delete_requested.emit(self))
        layout.addWidget(self.del_btn)

    def _on_type_changed(self, idx):
        ftype = FUNC_TYPES[idx]
        self.expr_edit.setPlaceholderText(FUNC_TYPE_HINTS.get(ftype, ""))
        self.changed.emit()

    def _pick_color(self):
        c = QColorDialog.getColor(self._color, self, "Choose curve color")
        if c.isValid():
            self._color = c
            self._update_color_btn()
            self.changed.emit()

    def _update_color_btn(self):
        self.color_btn.setStyleSheet(
            f"background-color: {self._color.name()}; border: 1px solid #555; border-radius: 3px;"
        )

    # --- public accessors ---
    @property
    def func_type(self):
        return FUNC_TYPES[self.type_combo.currentIndex()]

    @func_type.setter
    def func_type(self, value):
        if value in FUNC_TYPES:
            self.type_combo.setCurrentIndex(FUNC_TYPES.index(value))

    @property
    def expression(self):
        return self.expr_edit.text().strip()

    @expression.setter
    def expression(self, value):
        self.expr_edit.setText(value)

    @property
    def color(self):
        return self._color.name()

    @color.setter
    def color(self, value):
        self._color = QColor(value)
        self._update_color_btn()

    @property
    def visible(self):
        return self.visible_cb.isChecked()


# ---------------------------------------------------------------------------
# ParameterSlider widget
# ---------------------------------------------------------------------------

class ParameterSlider(QFrame):
    """Slider for a named parameter (a, b, c, d) with configurable range."""
    value_changed = pyqtSignal()

    def __init__(self, name="a", min_val=-10.0, max_val=10.0, default=1.0, steps=200, parent=None):
        super().__init__(parent)
        self.name = name
        self.min_val = min_val
        self.max_val = max_val
        self.steps = steps
        self._build_ui(default)

    def _build_ui(self, default):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(4)

        self.label = QLabel(f"<b>{self.name}</b> =")
        self.label.setFixedWidth(30)
        layout.addWidget(self.label)

        self.val_label = QLabel(f"{default:.2f}")
        self.val_label.setFixedWidth(50)
        self.val_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.val_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.steps)
        self.slider.setValue(self._val_to_tick(default))
        self.slider.valueChanged.connect(self._on_slider)
        layout.addWidget(self.slider, stretch=1)

        self.min_spin = QDoubleSpinBox()
        self.min_spin.setRange(-1000, 1000)
        self.min_spin.setValue(self.min_val)
        self.min_spin.setPrefix("min ")
        self.min_spin.setFixedWidth(80)
        self.min_spin.valueChanged.connect(self._on_range_changed)
        layout.addWidget(self.min_spin)

        self.max_spin = QDoubleSpinBox()
        self.max_spin.setRange(-1000, 1000)
        self.max_spin.setValue(self.max_val)
        self.max_spin.setPrefix("max ")
        self.max_spin.setFixedWidth(80)
        self.max_spin.valueChanged.connect(self._on_range_changed)
        layout.addWidget(self.max_spin)

    def _val_to_tick(self, v):
        frac = (v - self.min_val) / max(self.max_val - self.min_val, 1e-12)
        return int(np.clip(frac, 0, 1) * self.steps)

    def _tick_to_val(self, t):
        return self.min_val + (t / self.steps) * (self.max_val - self.min_val)

    def _on_slider(self, t):
        v = self._tick_to_val(t)
        self.val_label.setText(f"{v:.2f}")
        self.value_changed.emit()

    def _on_range_changed(self):
        self.min_val = self.min_spin.value()
        self.max_val = self.max_spin.value()
        self.value_changed.emit()

    @property
    def value(self):
        return self._tick_to_val(self.slider.value())

    @value.setter
    def value(self, v):
        self.slider.setValue(self._val_to_tick(v))


# ---------------------------------------------------------------------------
# Main Widget
# ---------------------------------------------------------------------------

class GraphingCalcWidget(QWidget):
    """
    A powerful graphing calculator widget for the Axiom Scientific Suite.

    Supports explicit, implicit, parametric, polar, and inequality plotting,
    parameter sliders, derivative/integral display, intersections, zoom/pan,
    coordinate readout, trace mode, and presets.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._logger = None
        self._func_rows = []
        self._param_sliders = {}
        self._mouse_coord = None
        self._trace_mode = False
        self._trace_info = None
        self._integral_x0 = None
        self._integral_x1 = None

        # Default axis range
        self._xmin, self._xmax = -10.0, 10.0
        self._ymin, self._ymax = -10.0, 10.0

        self._build_ui()
        self._connect_signals()

        # Deferred first plot
        QTimer.singleShot(100, self._replot)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_logger(self, fn):
        """Set a logging callback fn(message: str)."""
        self._logger = fn

    def run(self):
        """Trigger a replot (called externally by the suite)."""
        self._replot()

    def export(self):
        """Export current plot to an image file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Graph", "graph.png",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf);;All Files (*)"
        )
        if path:
            try:
                self.fig.savefig(path, dpi=200, bbox_inches="tight",
                                 facecolor=self.fig.get_facecolor())
                self._log(f"Exported graph to {path}")
            except Exception as exc:
                QMessageBox.warning(self, "Export Error", str(exc))

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # --- Left panel: function list ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(2, 2, 2, 2)

        lbl = QLabel("<b>Functions</b>")
        lbl.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(lbl)

        # Scrollable function list
        self._func_scroll = QScrollArea()
        self._func_scroll.setWidgetResizable(True)
        self._func_container = QWidget()
        self._func_list_layout = QVBoxLayout(self._func_container)
        self._func_list_layout.setContentsMargins(0, 0, 0, 0)
        self._func_list_layout.setSpacing(2)
        self._func_list_layout.addStretch()
        self._func_scroll.setWidget(self._func_container)
        left_layout.addWidget(self._func_scroll, stretch=1)

        # Add / Preset buttons
        btn_row = QHBoxLayout()
        self._add_btn = QPushButton("+ Add Function")
        self._add_btn.clicked.connect(self._add_function_row)
        btn_row.addWidget(self._add_btn)

        self._preset_combo = QComboBox()
        self._preset_combo.addItem("Presets...")
        self._preset_combo.addItems(list(PRESETS.keys()))
        self._preset_combo.currentIndexChanged.connect(self._load_preset)
        btn_row.addWidget(self._preset_combo)
        left_layout.addLayout(btn_row)

        # Parameter sliders
        slider_group = QGroupBox("Parameters")
        slider_layout = QVBoxLayout(slider_group)
        slider_layout.setContentsMargins(4, 4, 4, 4)
        slider_layout.setSpacing(2)
        for name, default in [("a", 1.0), ("b", 1.0), ("c", 0.0), ("d", 0.0)]:
            ps = ParameterSlider(name=name, default=default)
            ps.value_changed.connect(self._on_param_changed)
            slider_layout.addWidget(ps)
            self._param_sliders[name] = ps
        left_layout.addWidget(slider_group)

        splitter.addWidget(left_panel)

        # --- Center: matplotlib canvas ---
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(facecolor="#1e1e1e")
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = NavigationToolbar(self.canvas, self)
        center_layout.addWidget(self.toolbar)
        center_layout.addWidget(self.canvas, stretch=1)

        # Bottom bar: coordinates and intersection info
        bottom_bar = QHBoxLayout()
        self._coord_label = QLabel("Cursor: (-, -)")
        self._coord_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        bottom_bar.addWidget(self._coord_label)
        bottom_bar.addStretch()

        self._intersection_label = QLabel("")
        self._intersection_label.setStyleSheet("font-family: monospace; font-size: 11px; color: #FFD600;")
        bottom_bar.addWidget(self._intersection_label)
        center_layout.addLayout(bottom_bar)

        splitter.addWidget(center_widget)

        # --- Right panel: controls ---
        right_panel = QWidget()
        right_panel.setFixedWidth(220)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)

        # Axis controls
        axis_group = QGroupBox("Axis Range")
        axis_form = QFormLayout(axis_group)
        self._xmin_spin = self._make_range_spin(self._xmin)
        self._xmax_spin = self._make_range_spin(self._xmax)
        self._ymin_spin = self._make_range_spin(self._ymin)
        self._ymax_spin = self._make_range_spin(self._ymax)
        axis_form.addRow("x min", self._xmin_spin)
        axis_form.addRow("x max", self._xmax_spin)
        axis_form.addRow("y min", self._ymin_spin)
        axis_form.addRow("y max", self._ymax_spin)
        self._apply_range_btn = QPushButton("Apply Range")
        self._apply_range_btn.clicked.connect(self._apply_axis_range)
        axis_form.addRow(self._apply_range_btn)
        right_layout.addWidget(axis_group)

        # Display toggles
        disp_group = QGroupBox("Display")
        disp_layout = QVBoxLayout(disp_group)

        self._grid_cb = QCheckBox("Show Grid")
        self._grid_cb.setChecked(True)
        self._grid_cb.stateChanged.connect(self._replot)
        disp_layout.addWidget(self._grid_cb)

        self._equal_aspect_cb = QCheckBox("Equal Aspect Ratio")
        self._equal_aspect_cb.stateChanged.connect(self._replot)
        disp_layout.addWidget(self._equal_aspect_cb)

        self._derivative_cb = QCheckBox("Show Derivatives")
        self._derivative_cb.stateChanged.connect(self._replot)
        disp_layout.addWidget(self._derivative_cb)

        self._trace_cb = QCheckBox("Trace Mode")
        self._trace_cb.stateChanged.connect(self._toggle_trace)
        disp_layout.addWidget(self._trace_cb)

        right_layout.addWidget(disp_group)

        # Integral controls
        int_group = QGroupBox("Integral (area under curve)")
        int_form = QFormLayout(int_group)
        self._int_enable_cb = QCheckBox("Show integral")
        self._int_enable_cb.stateChanged.connect(self._replot)
        int_form.addRow(self._int_enable_cb)

        self._int_func_spin = QSpinBox()
        self._int_func_spin.setMinimum(1)
        self._int_func_spin.setMaximum(MAX_FUNCTIONS)
        self._int_func_spin.setValue(1)
        self._int_func_spin.setToolTip("Function # (1-based)")
        int_form.addRow("Func #", self._int_func_spin)

        self._int_x0_spin = QDoubleSpinBox()
        self._int_x0_spin.setRange(-1000, 1000)
        self._int_x0_spin.setValue(-1.0)
        self._int_x0_spin.setDecimals(3)
        int_form.addRow("x from", self._int_x0_spin)

        self._int_x1_spin = QDoubleSpinBox()
        self._int_x1_spin.setRange(-1000, 1000)
        self._int_x1_spin.setValue(1.0)
        self._int_x1_spin.setDecimals(3)
        int_form.addRow("x to", self._int_x1_spin)

        self._int_result_label = QLabel("Area: --")
        self._int_result_label.setStyleSheet("font-weight: bold;")
        int_form.addRow(self._int_result_label)

        self._int_enable_cb.stateChanged.connect(self._replot)
        self._int_x0_spin.valueChanged.connect(self._replot)
        self._int_x1_spin.valueChanged.connect(self._replot)
        self._int_func_spin.valueChanged.connect(self._replot)

        right_layout.addWidget(int_group)

        # Intersection finder
        isect_group = QGroupBox("Intersections")
        isect_layout = QVBoxLayout(isect_group)
        isect_hint = QLabel("Between functions #1 and #2")
        isect_hint.setWordWrap(True)
        isect_hint.setStyleSheet("font-size: 10px;")
        isect_layout.addWidget(isect_hint)

        isect_row = QHBoxLayout()
        self._isect_a_spin = QSpinBox()
        self._isect_a_spin.setMinimum(1)
        self._isect_a_spin.setMaximum(MAX_FUNCTIONS)
        self._isect_a_spin.setValue(1)
        isect_row.addWidget(QLabel("#"))
        isect_row.addWidget(self._isect_a_spin)
        isect_row.addWidget(QLabel("and #"))
        self._isect_b_spin = QSpinBox()
        self._isect_b_spin.setMinimum(1)
        self._isect_b_spin.setMaximum(MAX_FUNCTIONS)
        self._isect_b_spin.setValue(2)
        isect_row.addWidget(self._isect_b_spin)
        isect_layout.addLayout(isect_row)

        self._find_isect_btn = QPushButton("Find Intersections")
        self._find_isect_btn.clicked.connect(self._find_intersections)
        isect_layout.addWidget(self._find_isect_btn)
        right_layout.addWidget(isect_group)

        right_layout.addStretch()

        # Reset view button
        self._reset_btn = QPushButton("Reset View")
        self._reset_btn.clicked.connect(self._reset_view)
        right_layout.addWidget(self._reset_btn)

        splitter.addWidget(right_panel)

        # Splitter proportions
        splitter.setSizes([280, 600, 220])

        # Add one default function row
        self._add_function_row()

    # ------------------------------------------------------------------
    # Signal connections
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self._on_mouse_release)

        self._pan_active = False
        self._pan_start = None

    # ------------------------------------------------------------------
    # Function list management
    # ------------------------------------------------------------------

    def _add_function_row(self, func_type=None, expression=None, color=None):
        if len(self._func_rows) >= MAX_FUNCTIONS:
            QMessageBox.information(self, "Limit", f"Maximum {MAX_FUNCTIONS} functions.")
            return
        idx = len(self._func_rows)
        c = color or DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
        row = FunctionRow(color=c, parent=self._func_container)
        if func_type:
            row.func_type = func_type
        if expression:
            row.expression = expression
        row.changed.connect(self._on_func_changed)
        row.delete_requested.connect(self._remove_function_row)
        # Insert before the stretch
        self._func_list_layout.insertWidget(self._func_list_layout.count() - 1, row)
        self._func_rows.append(row)
        self._replot()
        return row

    def _remove_function_row(self, row):
        if row in self._func_rows:
            self._func_rows.remove(row)
            self._func_list_layout.removeWidget(row)
            row.deleteLater()
            self._replot()

    def _on_func_changed(self):
        self._replot()

    def _on_param_changed(self):
        self._replot()

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    def _load_preset(self, idx):
        if idx <= 0:
            return
        name = self._preset_combo.currentText()
        if name in PRESETS:
            ftype, expr = PRESETS[name]
            self._add_function_row(func_type=ftype, expression=expr)
            self._log(f"Loaded preset: {name}")
        self._preset_combo.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # Axis / view helpers
    # ------------------------------------------------------------------

    def _make_range_spin(self, val):
        s = QDoubleSpinBox()
        s.setRange(-10000, 10000)
        s.setDecimals(2)
        s.setValue(val)
        s.setSingleStep(1.0)
        return s

    def _apply_axis_range(self):
        self._xmin = self._xmin_spin.value()
        self._xmax = self._xmax_spin.value()
        self._ymin = self._ymin_spin.value()
        self._ymax = self._ymax_spin.value()
        self._replot()

    def _reset_view(self):
        self._xmin, self._xmax = -10.0, 10.0
        self._ymin, self._ymax = -10.0, 10.0
        self._xmin_spin.setValue(self._xmin)
        self._xmax_spin.setValue(self._xmax)
        self._ymin_spin.setValue(self._ymin)
        self._ymax_spin.setValue(self._ymax)
        self._replot()

    # ------------------------------------------------------------------
    # Zoom / Pan via mouse
    # ------------------------------------------------------------------

    def _on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        factor = 0.8 if event.button == "up" else 1.25
        cx, cy = event.xdata, event.ydata
        dx = (self._xmax - self._xmin) * factor / 2
        dy = (self._ymax - self._ymin) * factor / 2
        self._xmin = cx - dx
        self._xmax = cx + dx
        self._ymin = cy - dy
        self._ymax = cy + dy
        self._sync_range_spins()
        self._replot()

    def _on_mouse_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 2:  # middle button pan
            self._pan_active = True
            self._pan_start = (event.xdata, event.ydata)
        elif event.button == 1 and self._trace_mode:
            self._do_trace(event.xdata, event.ydata)

    def _on_mouse_release(self, event):
        if event.button == 2:
            self._pan_active = False
            self._pan_start = None

    def _on_mouse_move(self, event):
        if event.inaxes == self.ax and event.xdata is not None:
            self._coord_label.setText(f"Cursor: ({event.xdata:.4f}, {event.ydata:.4f})")
            if self._pan_active and self._pan_start:
                dx = self._pan_start[0] - event.xdata
                dy = self._pan_start[1] - event.ydata
                self._xmin += dx
                self._xmax += dx
                self._ymin += dy
                self._ymax += dy
                self._sync_range_spins()
                self._replot()
        else:
            self._coord_label.setText("Cursor: (-, -)")

    def _sync_range_spins(self):
        self._xmin_spin.blockSignals(True)
        self._xmax_spin.blockSignals(True)
        self._ymin_spin.blockSignals(True)
        self._ymax_spin.blockSignals(True)
        self._xmin_spin.setValue(self._xmin)
        self._xmax_spin.setValue(self._xmax)
        self._ymin_spin.setValue(self._ymin)
        self._ymax_spin.setValue(self._ymax)
        self._xmin_spin.blockSignals(False)
        self._xmax_spin.blockSignals(False)
        self._ymin_spin.blockSignals(False)
        self._ymax_spin.blockSignals(False)

    # ------------------------------------------------------------------
    # Trace mode
    # ------------------------------------------------------------------

    def _toggle_trace(self, state):
        self._trace_mode = bool(state)
        if not self._trace_mode:
            self._trace_info = None
            self._replot()

    def _do_trace(self, mx, my):
        """Find nearest point on any visible y=f(x) curve and display it."""
        best_dist = float("inf")
        best_point = None
        best_idx = -1
        ns = self._build_namespace()
        x_arr = np.linspace(self._xmin, self._xmax, 1000)

        for i, row in enumerate(self._func_rows):
            if not row.visible or row.func_type != "y = f(x)":
                continue
            try:
                y_arr = self._eval_explicit(row.expression, x_arr, ns)
                valid = np.isfinite(y_arr)
                for j in np.where(valid)[0]:
                    d = (x_arr[j] - mx) ** 2 + (y_arr[j] - my) ** 2
                    if d < best_dist:
                        best_dist = d
                        best_point = (x_arr[j], y_arr[j])
                        best_idx = i
            except Exception:
                continue

        if best_point is not None:
            self._trace_info = (best_idx, best_point)
            self._replot()

    # ------------------------------------------------------------------
    # Expression evaluation helpers
    # ------------------------------------------------------------------

    def _build_namespace(self):
        ns = dict(_SAFE_NAMESPACE)
        for name, slider in self._param_sliders.items():
            ns[name] = slider.value
        return ns

    def _eval_explicit(self, expr, x, ns):
        """Evaluate y = f(x) expression. Returns y array."""
        local = dict(ns)
        local["x"] = x
        y = eval(compile(expr, "<expr>", "eval"), {"__builtins__": {}}, local)
        return np.broadcast_to(np.asarray(y, dtype=float), x.shape).copy()

    def _eval_implicit(self, expr, xg, yg, ns):
        """Evaluate f(x,y) on a grid. Returns 2D array."""
        local = dict(ns)
        local["x"] = xg
        local["y"] = yg
        z = eval(compile(expr, "<expr>", "eval"), {"__builtins__": {}}, local)
        return np.asarray(z, dtype=float)

    def _eval_parametric(self, expr_x, expr_y, t, ns):
        """Evaluate parametric x(t), y(t). Returns (x_arr, y_arr)."""
        local = dict(ns)
        local["t"] = t
        x = eval(compile(expr_x, "<expr>", "eval"), {"__builtins__": {}}, local)
        y = eval(compile(expr_y, "<expr>", "eval"), {"__builtins__": {}}, local)
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    def _eval_polar(self, expr, theta, ns):
        """Evaluate r = f(theta). Returns r array."""
        local = dict(ns)
        local["theta"] = theta
        local["t"] = theta  # allow 't' as alias
        r = eval(compile(expr, "<expr>", "eval"), {"__builtins__": {}}, local)
        return np.broadcast_to(np.asarray(r, dtype=float), theta.shape).copy()

    def _numerical_derivative(self, expr, x, ns, h=1e-7):
        """Central difference derivative of y=f(x)."""
        y1 = self._eval_explicit(expr, x + h, ns)
        y0 = self._eval_explicit(expr, x - h, ns)
        return (y1 - y0) / (2 * h)

    def _numerical_integral(self, expr, x0, x1, ns, n=2000):
        """Trapezoidal numerical integration of y=f(x) from x0 to x1."""
        x = np.linspace(x0, x1, n)
        y = self._eval_explicit(expr, x, ns)
        valid = np.isfinite(y)
        if not np.any(valid):
            return 0.0, x, y
        y_clean = np.where(valid, y, 0.0)
        area = np.trapz(y_clean, x)
        return area, x, y

    # ------------------------------------------------------------------
    # Intersection finder
    # ------------------------------------------------------------------

    def _find_intersections(self):
        """Find approximate intersection points between two y=f(x) curves."""
        a_idx = self._isect_a_spin.value() - 1
        b_idx = self._isect_b_spin.value() - 1
        if a_idx >= len(self._func_rows) or b_idx >= len(self._func_rows):
            self._intersection_label.setText("Invalid function indices.")
            return
        row_a = self._func_rows[a_idx]
        row_b = self._func_rows[b_idx]
        if row_a.func_type != "y = f(x)" or row_b.func_type != "y = f(x)":
            self._intersection_label.setText("Intersections require y=f(x) type.")
            return

        ns = self._build_namespace()
        x = np.linspace(self._xmin, self._xmax, 5000)
        try:
            ya = self._eval_explicit(row_a.expression, x, ns)
            yb = self._eval_explicit(row_b.expression, x, ns)
        except Exception as exc:
            self._intersection_label.setText(f"Error: {exc}")
            return

        diff = ya - yb
        sign_changes = np.where(np.diff(np.sign(diff)))[0]

        points = []
        for idx in sign_changes:
            # linear interpolation for better accuracy
            x0, x1 = x[idx], x[idx + 1]
            d0, d1 = diff[idx], diff[idx + 1]
            if abs(d1 - d0) < 1e-15:
                xi = (x0 + x1) / 2
            else:
                xi = x0 - d0 * (x1 - x0) / (d1 - d0)
            try:
                yi = self._eval_explicit(row_a.expression, np.array([xi]), ns)[0]
                points.append((xi, yi))
            except Exception:
                continue

        if points:
            pts_str = "; ".join(f"({p[0]:.4f}, {p[1]:.4f})" for p in points[:8])
            self._intersection_label.setText(f"Intersections: {pts_str}")
            # Mark on plot
            self._replot()
            for px, py in points:
                self.ax.plot(px, py, "o", color="#FFD600", markersize=8, zorder=10)
            self.canvas.draw_idle()
            self._log(f"Found {len(points)} intersection(s)")
        else:
            self._intersection_label.setText("No intersections found in view.")

    # ------------------------------------------------------------------
    # Core plotting
    # ------------------------------------------------------------------

    def _replot(self, *_args):
        """Recompute and redraw all visible functions."""
        self.ax.clear()
        self._style_axes()
        ns = self._build_namespace()
        n_pts = 1500

        x_arr = np.linspace(self._xmin, self._xmax, n_pts)

        for idx, row in enumerate(self._func_rows):
            if not row.visible or not row.expression:
                continue
            color = row.color
            ftype = row.func_type
            expr = row.expression

            try:
                if ftype == "y = f(x)":
                    self._plot_explicit(expr, x_arr, ns, color, idx)
                elif ftype == "Implicit":
                    self._plot_implicit(expr, ns, color)
                elif ftype == "Parametric":
                    self._plot_parametric(expr, ns, color)
                elif ftype == "Polar":
                    self._plot_polar(expr, ns, color)
                elif ftype == "Inequality":
                    self._plot_inequality(expr, x_arr, ns, color)
            except Exception as exc:
                self._log(f"[Func {idx+1}] Error: {exc}")

        # Integral display
        if self._int_enable_cb.isChecked():
            self._draw_integral(ns)

        # Trace info overlay
        if self._trace_info is not None:
            ti_idx, (tx, ty) = self._trace_info
            self.ax.plot(tx, ty, "o", color="#FFFFFF", markersize=7, zorder=15)
            self.ax.annotate(
                f"({tx:.3f}, {ty:.3f})",
                (tx, ty), textcoords="offset points", xytext=(10, 10),
                fontsize=9, color="#FFFFFF",
                bbox=dict(boxstyle="round,pad=0.3", fc="#333333", ec="#888888", alpha=0.9),
                zorder=15,
            )

        self.ax.set_xlim(self._xmin, self._xmax)
        self.ax.set_ylim(self._ymin, self._ymax)

        if self._equal_aspect_cb.isChecked():
            self.ax.set_aspect("equal", adjustable="datalim")
        else:
            self.ax.set_aspect("auto")

        self.canvas.draw_idle()

    def _style_axes(self):
        """Apply dark theme styling to axes."""
        self.ax.set_facecolor("#1e1e1e")
        self.ax.tick_params(colors="#aaaaaa", which="both")
        self.ax.xaxis.label.set_color("#cccccc")
        self.ax.yaxis.label.set_color("#cccccc")
        for spine in self.ax.spines.values():
            spine.set_color("#444444")
        if self._grid_cb.isChecked():
            self.ax.grid(True, color="#333333", linewidth=0.5, alpha=0.7)
        else:
            self.ax.grid(False)
        # Draw axes lines through origin
        self.ax.axhline(0, color="#555555", linewidth=0.8, zorder=0)
        self.ax.axvline(0, color="#555555", linewidth=0.8, zorder=0)

    # --- Explicit y=f(x) ---
    def _plot_explicit(self, expr, x, ns, color, func_idx):
        y = self._eval_explicit(expr, x, ns)
        # Mask large jumps (discontinuities)
        y_masked = y.copy()
        dy = np.abs(np.diff(y))
        threshold = (self._ymax - self._ymin) * 5
        jumps = np.where(dy > threshold)[0]
        for j in jumps:
            y_masked[j + 1] = np.nan
        self.ax.plot(x, y_masked, color=color, linewidth=2, zorder=5,
                     label=f"f{func_idx+1}: {expr[:30]}")

        # Derivative overlay
        if self._derivative_cb.isChecked():
            dy = self._numerical_derivative(expr, x, ns)
            dy_masked = dy.copy()
            jumps_d = np.where(np.abs(np.diff(dy)) > threshold)[0]
            for j in jumps_d:
                dy_masked[j + 1] = np.nan
            self.ax.plot(x, dy_masked, color=color, linewidth=1, linestyle="--",
                         alpha=0.6, zorder=4, label=f"f{func_idx+1}' (deriv)")

    # --- Implicit f(x,y) = 0 ---
    def _plot_implicit(self, expr, ns, color):
        nx, ny = 400, 400
        xg = np.linspace(self._xmin, self._xmax, nx)
        yg = np.linspace(self._ymin, self._ymax, ny)
        X, Y = np.meshgrid(xg, yg)
        Z = self._eval_implicit(expr, X, Y, ns)
        self.ax.contour(X, Y, Z, levels=[0], colors=[color], linewidths=2, zorder=5)

    # --- Parametric x(t), y(t) ---
    def _plot_parametric(self, expr, ns, color):
        parts = [p.strip() for p in expr.split(",")]
        if len(parts) != 2:
            self._log("Parametric needs 'x(t), y(t)' format.")
            return
        t = np.linspace(0, 2 * np.pi, 2000)
        x, y = self._eval_parametric(parts[0], parts[1], t, ns)
        self.ax.plot(x, y, color=color, linewidth=2, zorder=5)

    # --- Polar r = f(theta) ---
    def _plot_polar(self, expr, ns, color):
        theta = np.linspace(0, 4 * np.pi, 3000)
        r = self._eval_polar(expr, theta, ns)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        self.ax.plot(x, y, color=color, linewidth=2, zorder=5)

    # --- Inequality y > f(x) or y < f(x) ---
    def _plot_inequality(self, expr, x, ns, color):
        # Parse "y > ..." or "y < ..." or "y >= ..." or "y <= ..."
        ineq_expr = expr.strip()
        direction = None
        rhs_expr = None

        for op in (">=", "<=", ">", "<"):
            if op in ineq_expr:
                parts = ineq_expr.split(op, 1)
                lhs = parts[0].strip()
                rhs = parts[1].strip()
                if lhs == "y":
                    rhs_expr = rhs
                    direction = "above" if ">" in op else "below"
                elif rhs == "y":
                    rhs_expr = lhs
                    direction = "below" if ">" in op else "above"
                break

        if rhs_expr is None or direction is None:
            self._log("Inequality must have form 'y > expr' or 'y < expr'.")
            return

        y_boundary = self._eval_explicit(rhs_expr, x, ns)
        self.ax.plot(x, y_boundary, color=color, linewidth=1.5, zorder=5)

        if direction == "above":
            self.ax.fill_between(x, y_boundary, self._ymax,
                                 color=color, alpha=0.15, zorder=2)
        else:
            self.ax.fill_between(x, self._ymin, y_boundary,
                                 color=color, alpha=0.15, zorder=2)

    # --- Integral shading ---
    def _draw_integral(self, ns):
        func_idx = self._int_func_spin.value() - 1
        if func_idx < 0 or func_idx >= len(self._func_rows):
            return
        row = self._func_rows[func_idx]
        if row.func_type != "y = f(x)" or not row.expression:
            self._int_result_label.setText("Area: (need y=f(x))")
            return
        x0 = self._int_x0_spin.value()
        x1 = self._int_x1_spin.value()
        try:
            area, xi, yi = self._numerical_integral(row.expression, x0, x1, ns)
            self.ax.fill_between(xi, 0, yi, color=row.color, alpha=0.25, zorder=3)
            self.ax.axvline(x0, color="#FFD600", linewidth=0.8, linestyle=":", zorder=4)
            self.ax.axvline(x1, color="#FFD600", linewidth=0.8, linestyle=":", zorder=4)
            self._int_result_label.setText(f"Area: {area:.6f}")
        except Exception as exc:
            self._int_result_label.setText(f"Area: Error")
            self._log(f"Integral error: {exc}")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, msg):
        if self._logger:
            self._logger(msg)


# ---------------------------------------------------------------------------
# Standalone entry point for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, QColor(204, 204, 204))
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(35, 35, 35))
    palette.setColor(QPalette.ToolTipBase, QColor(50, 50, 50))
    palette.setColor(QPalette.ToolTipText, QColor(204, 204, 204))
    palette.setColor(QPalette.Text, QColor(204, 204, 204))
    palette.setColor(QPalette.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ButtonText, QColor(204, 204, 204))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)

    win = GraphingCalcWidget()
    win.set_logger(print)
    win.setWindowTitle("Axiom Scientific Suite - Graphing Calculator")
    win.resize(1200, 750)
    win.show()
    sys.exit(app.exec_())
