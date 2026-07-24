"""
plotter3d.py - 3D Plotter Widget for PyQt5 Scientific Suite

Provides an interactive 3D plotting widget with support for surface plots,
wireframes, scatter plots, contour plots, parametric surfaces, and vector fields.
Embeds matplotlib 3D figures in PyQt5 with full interactivity.
"""

import sys
import io
import os
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QLineEdit, QPushButton, QComboBox, QSlider, QSpinBox,
    QDoubleSpinBox, QGroupBox, QTabWidget, QFileDialog,
    QMessageBox, QCheckBox, QSplitter, QApplication, QFrame,
    QInputDialog, QProgressBar
)
from PyQt5.QtCore import Qt, QTimer

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
from matplotlib.gridspec import GridSpec

try:
    from mpl_style import style_figure, style_axes
except ImportError:
    def style_figure(f): pass
    def style_axes(a): pass
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Preset parametric / explicit functions
# ---------------------------------------------------------------------------
PRESETS = {
    "Sinc": "np.sinc(np.sqrt(x**2 + y**2))",
    "Saddle": "x**2 - y**2",
    "Ripple": "np.sin(np.sqrt(x**2 + y**2))",
    "Gaussian": "np.exp(-(x**2 + y**2))",
    "Egg Carton": "np.sin(x) * np.cos(y)",
    "Paraboloid": "x**2 + y**2",
    "Monkey Saddle": "x**3 - 3*x*y**2",
    "Wave": "np.sin(x) * np.sin(y)",
}

PARAMETRIC_PRESETS = {
    "Sphere": {
        "x_expr": "np.sin(u) * np.cos(v)",
        "y_expr": "np.sin(u) * np.sin(v)",
        "z_expr": "np.cos(u)",
        "u_range": (0, np.pi),
        "v_range": (0, 2 * np.pi),
    },
    "Torus": {
        "x_expr": "(2 + 0.8*np.cos(v)) * np.cos(u)",
        "y_expr": "(2 + 0.8*np.cos(v)) * np.sin(u)",
        "z_expr": "0.8 * np.sin(v)",
        "u_range": (0, 2 * np.pi),
        "v_range": (0, 2 * np.pi),
    },
    "Mobius Strip": {
        "x_expr": "(1 + 0.5*v*np.cos(u/2)) * np.cos(u)",
        "y_expr": "(1 + 0.5*v*np.cos(u/2)) * np.sin(u)",
        "z_expr": "0.5 * v * np.sin(u/2)",
        "u_range": (0, 2 * np.pi),
        "v_range": (-1, 1),
    },
    "Klein Bottle (immersed)": {
        "x_expr": "(2 + np.cos(u/2)*np.sin(v) - np.sin(u/2)*np.sin(2*v)) * np.cos(u)",
        "y_expr": "(2 + np.cos(u/2)*np.sin(v) - np.sin(u/2)*np.sin(2*v)) * np.sin(u)",
        "z_expr": "np.sin(u/2)*np.sin(v) + np.cos(u/2)*np.sin(2*v)",
        "u_range": (0, 2 * np.pi),
        "v_range": (0, 2 * np.pi),
    },
    "Helicoid": {
        "x_expr": "v * np.cos(u)",
        "y_expr": "v * np.sin(u)",
        "z_expr": "u",
        "u_range": (0, 4 * np.pi),
        "v_range": (-1, 1),
    },
    "Enneper Surface": {
        "x_expr": "u - u**3/3 + u*v**2",
        "y_expr": "v - v**3/3 + v*u**2",
        "z_expr": "u**2 - v**2",
        "u_range": (-1.5, 1.5),
        "v_range": (-1.5, 1.5),
    },
}

COLORMAPS = [
    "viridis", "plasma", "inferno", "magma", "cividis",
    "coolwarm", "RdYlBu", "Spectral", "jet", "turbo",
    "ocean", "terrain", "rainbow", "gnuplot", "cubehelix",
]

PLOT_TYPES = [
    "Surface", "Wireframe", "Scatter 3D",
    "Contour 3D", "Parametric Surface", "Vector Field",
]


class Plotter3DWidget(QWidget):
    """Interactive 3D plotter widget for embedding in a PyQt5 application."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._logger = None
        self._init_ui()
        self._log("Plotter3DWidget initialised.")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def set_logger(self, fn):
        """Set an external logging callback ``fn(message: str)``."""
        self._logger = fn

    def _log(self, msg: str):
        if self._logger:
            self._logger(msg)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _init_ui(self):
        root = QHBoxLayout(self)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # --- Left: controls panel ---
        ctrl_frame = QFrame()
        ctrl_layout = QVBoxLayout(ctrl_frame)
        ctrl_layout.setContentsMargins(4, 4, 4, 4)

        self._build_plot_type_group(ctrl_layout)
        self._build_function_group(ctrl_layout)
        self._build_range_group(ctrl_layout)
        self._build_view_group(ctrl_layout)
        self._build_style_group(ctrl_layout)
        self._build_action_buttons(ctrl_layout)

        ctrl_layout.addStretch()
        splitter.addWidget(ctrl_frame)

        # --- Right: matplotlib canvas ---
        canvas_frame = QFrame()
        canvas_layout = QVBoxLayout(canvas_frame)
        canvas_layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(7, 6), dpi=100)
        style_figure(self.figure)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        canvas_layout.addWidget(self.toolbar)
        canvas_layout.addWidget(self.canvas)
        splitter.addWidget(canvas_frame)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 700])

        self.setWindowTitle("3D Plotter")
        self.resize(1100, 700)

    # --- Plot type ---
    def _build_plot_type_group(self, parent_layout):
        grp = QGroupBox("Plot Type")
        lay = QVBoxLayout(grp)

        self.combo_plot_type = QComboBox()
        self.combo_plot_type.addItems(PLOT_TYPES)
        self.combo_plot_type.currentTextChanged.connect(self._on_plot_type_changed)
        lay.addWidget(self.combo_plot_type)

        parent_layout.addWidget(grp)

    # --- Function input ---
    def _build_function_group(self, parent_layout):
        grp = QGroupBox("Function / Expression")
        lay = QVBoxLayout(grp)

        # z = f(x, y)
        self.lbl_func = QLabel("z = f(x, y):")
        lay.addWidget(self.lbl_func)
        self.input_func = QLineEdit("np.sin(np.sqrt(x**2 + y**2))")
        lay.addWidget(self.input_func)

        # Parametric expressions
        self.lbl_px = QLabel("x(u,v):")
        self.input_px = QLineEdit("np.sin(u) * np.cos(v)")
        self.lbl_py = QLabel("y(u,v):")
        self.input_py = QLineEdit("np.sin(u) * np.sin(v)")
        self.lbl_pz = QLabel("z(u,v):")
        self.input_pz = QLineEdit("np.cos(u)")
        for w in (self.lbl_px, self.input_px, self.lbl_py, self.input_py, self.lbl_pz, self.input_pz):
            lay.addWidget(w)
            w.setVisible(False)

        # Vector field components
        self.lbl_vx = QLabel("Fx(x,y,z):")
        self.input_vx = QLineEdit("y")
        self.lbl_vy = QLabel("Fy(x,y,z):")
        self.input_vy = QLineEdit("-x")
        self.lbl_vz = QLabel("Fz(x,y,z):")
        self.input_vz = QLineEdit("z*0")
        for w in (self.lbl_vx, self.input_vx, self.lbl_vy, self.input_vy, self.lbl_vz, self.input_vz):
            lay.addWidget(w)
            w.setVisible(False)

        # Presets
        h = QHBoxLayout()
        h.addWidget(QLabel("Preset:"))
        self.combo_preset = QComboBox()
        self.combo_preset.addItem("-- select --")
        self.combo_preset.addItems(PRESETS.keys())
        self.combo_preset.currentTextChanged.connect(self._apply_preset)
        h.addWidget(self.combo_preset)
        lay.addLayout(h)

        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Parametric:"))
        self.combo_parametric_preset = QComboBox()
        self.combo_parametric_preset.addItem("-- select --")
        self.combo_parametric_preset.addItems(PARAMETRIC_PRESETS.keys())
        self.combo_parametric_preset.currentTextChanged.connect(self._apply_parametric_preset)
        h2.addWidget(self.combo_parametric_preset)
        lay.addLayout(h2)

        parent_layout.addWidget(grp)

    # --- Range controls ---
    def _build_range_group(self, parent_layout):
        grp = QGroupBox("Data Range")
        grid = QGridLayout(grp)

        labels = ["x min", "x max", "y min", "y max"]
        defaults = [-5.0, 5.0, -5.0, 5.0]
        self.range_spins = []
        for i, (lbl, val) in enumerate(zip(labels, defaults)):
            grid.addWidget(QLabel(lbl + ":"), i, 0)
            sb = QDoubleSpinBox()
            sb.setRange(-1000, 1000)
            sb.setDecimals(2)
            sb.setValue(val)
            grid.addWidget(sb, i, 1)
            self.range_spins.append(sb)

        grid.addWidget(QLabel("Resolution:"), 4, 0)
        self.spin_res = QSpinBox()
        self.spin_res.setRange(10, 500)
        self.spin_res.setValue(80)
        grid.addWidget(self.spin_res, 4, 1)

        parent_layout.addWidget(grp)

    # --- Viewing angle ---
    def _build_view_group(self, parent_layout):
        grp = QGroupBox("View Angle")
        grid = QGridLayout(grp)

        grid.addWidget(QLabel("Elevation:"), 0, 0)
        self.slider_elev = QSlider(Qt.Horizontal)
        self.slider_elev.setRange(-90, 90)
        self.slider_elev.setValue(30)
        self.lbl_elev_val = QLabel("30")
        self.slider_elev.valueChanged.connect(lambda v: self.lbl_elev_val.setText(str(v)))
        self.slider_elev.valueChanged.connect(self._update_view_angle)
        grid.addWidget(self.slider_elev, 0, 1)
        grid.addWidget(self.lbl_elev_val, 0, 2)

        grid.addWidget(QLabel("Azimuth:"), 1, 0)
        self.slider_azim = QSlider(Qt.Horizontal)
        self.slider_azim.setRange(0, 360)
        self.slider_azim.setValue(45)
        self.lbl_azim_val = QLabel("45")
        self.slider_azim.valueChanged.connect(lambda v: self.lbl_azim_val.setText(str(v)))
        self.slider_azim.valueChanged.connect(self._update_view_angle)
        grid.addWidget(self.slider_azim, 1, 1)
        grid.addWidget(self.lbl_azim_val, 1, 2)

        parent_layout.addWidget(grp)

    # --- Style / appearance ---
    def _build_style_group(self, parent_layout):
        grp = QGroupBox("Style")
        grid = QGridLayout(grp)

        grid.addWidget(QLabel("Colormap:"), 0, 0)
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(COLORMAPS)
        grid.addWidget(self.combo_cmap, 0, 1)

        grid.addWidget(QLabel("Alpha:"), 1, 0)
        self.slider_alpha = QSlider(Qt.Horizontal)
        self.slider_alpha.setRange(10, 100)
        self.slider_alpha.setValue(90)
        self.lbl_alpha_val = QLabel("0.90")
        self.slider_alpha.valueChanged.connect(
            lambda v: self.lbl_alpha_val.setText(f"{v / 100:.2f}")
        )
        grid.addWidget(self.slider_alpha, 1, 1)
        grid.addWidget(self.lbl_alpha_val, 1, 2)

        self.chk_colorbar = QCheckBox("Show colour bar")
        self.chk_colorbar.setChecked(True)
        grid.addWidget(self.chk_colorbar, 2, 0, 1, 2)

        self.chk_grid = QCheckBox("Show grid")
        self.chk_grid.setChecked(True)
        grid.addWidget(self.chk_grid, 3, 0, 1, 2)

        # Axis labels
        grid.addWidget(QLabel("X label:"), 4, 0)
        self.input_xlabel = QLineEdit("X")
        grid.addWidget(self.input_xlabel, 4, 1)

        grid.addWidget(QLabel("Y label:"), 5, 0)
        self.input_ylabel = QLineEdit("Y")
        grid.addWidget(self.input_ylabel, 5, 1)

        grid.addWidget(QLabel("Z label:"), 6, 0)
        self.input_zlabel = QLineEdit("Z")
        grid.addWidget(self.input_zlabel, 6, 1)

        grid.addWidget(QLabel("Title:"), 7, 0)
        self.input_title = QLineEdit("")
        grid.addWidget(self.input_title, 7, 1)

        parent_layout.addWidget(grp)

    # --- Action buttons ---
    def _build_action_buttons(self, parent_layout):
        h = QHBoxLayout()

        btn_plot = QPushButton("Plot")
        btn_plot.clicked.connect(self._do_plot)
        h.addWidget(btn_plot)

        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._clear_plot)
        h.addWidget(btn_clear)

        btn_export = QPushButton("Export...")
        btn_export.clicked.connect(self.export)
        h.addWidget(btn_export)

        btn_copy_plot = QPushButton("Copy Plot")
        btn_copy_plot.setToolTip("Copy current plot to clipboard as image")
        btn_copy_plot.clicked.connect(self._copy_plot_to_clipboard)
        h.addWidget(btn_copy_plot)

        parent_layout.addLayout(h)

        # Advanced features row
        h2 = QHBoxLayout()

        btn_animate = QPushButton("Animate")
        btn_animate.setToolTip("Create rotation / parameter sweep animation (GIF)")
        btn_animate.clicked.connect(self._create_animation)
        h2.addWidget(btn_animate)

        btn_isosurface = QPushButton("Isosurface")
        btn_isosurface.setToolTip("Render isosurface from volumetric data")
        btn_isosurface.clicked.connect(self._render_isosurface)
        h2.addWidget(btn_isosurface)

        btn_streamline = QPushButton("Streamlines")
        btn_streamline.setToolTip("Visualize vector field with streamlines")
        btn_streamline.clicked.connect(self._plot_streamlines)
        h2.addWidget(btn_streamline)

        parent_layout.addLayout(h2)

        h3 = QHBoxLayout()

        btn_multi = QPushButton("Multi-Plot")
        btn_multi.setToolTip("Create 3D subplot grid")
        btn_multi.clicked.connect(self._create_multi_plot)
        h3.addWidget(btn_multi)

        btn_mesh = QPushButton("Load Mesh")
        btn_mesh.setToolTip("Load and render STL/OBJ 3D mesh")
        btn_mesh.clicked.connect(self._load_mesh)
        h3.addWidget(btn_mesh)

        parent_layout.addLayout(h3)

    # ------------------------------------------------------------------
    # Slot: plot-type changed  -> toggle input fields
    # ------------------------------------------------------------------
    def _on_plot_type_changed(self, text: str):
        is_parametric = text == "Parametric Surface"
        is_vector = text == "Vector Field"
        is_standard = not is_parametric and not is_vector

        self.lbl_func.setVisible(is_standard)
        self.input_func.setVisible(is_standard)

        for w in (self.lbl_px, self.input_px, self.lbl_py, self.input_py,
                  self.lbl_pz, self.input_pz):
            w.setVisible(is_parametric)

        for w in (self.lbl_vx, self.input_vx, self.lbl_vy, self.input_vy,
                  self.lbl_vz, self.input_vz):
            w.setVisible(is_vector)

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------
    def _apply_preset(self, name: str):
        if name in PRESETS:
            self.input_func.setText(PRESETS[name])
            self.combo_plot_type.setCurrentText("Surface")
            self._log(f"Applied preset: {name}")

    def _apply_parametric_preset(self, name: str):
        if name not in PARAMETRIC_PRESETS:
            return
        p = PARAMETRIC_PRESETS[name]
        self.input_px.setText(p["x_expr"])
        self.input_py.setText(p["y_expr"])
        self.input_pz.setText(p["z_expr"])
        self.range_spins[0].setValue(p["u_range"][0])
        self.range_spins[1].setValue(p["u_range"][1])
        self.range_spins[2].setValue(p["v_range"][0])
        self.range_spins[3].setValue(p["v_range"][1])
        self.combo_plot_type.setCurrentText("Parametric Surface")
        self._log(f"Applied parametric preset: {name}")

    # ------------------------------------------------------------------
    # View angle live update
    # ------------------------------------------------------------------
    def _update_view_angle(self):
        if not hasattr(self, "_ax"):
            return
        try:
            self._ax.view_init(elev=self.slider_elev.value(),
                               azim=self.slider_azim.value())
            self.canvas.draw_idle()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Core plotting
    # ------------------------------------------------------------------
    def _do_plot(self):
        plot_type = self.combo_plot_type.currentText()
        try:
            if plot_type == "Parametric Surface":
                self._plot_parametric()
            elif plot_type == "Vector Field":
                self._plot_vector_field()
            else:
                self._plot_standard(plot_type)
            self._log(f"Plotted: {plot_type}")
        except Exception as exc:
            self._log(f"Plot error: {exc}")
            QMessageBox.warning(self, "Plot Error", str(exc))

    def _prepare_axes(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection="3d")
        ax.set_xlabel(self.input_xlabel.text())
        ax.set_ylabel(self.input_ylabel.text())
        ax.set_zlabel(self.input_zlabel.text())
        title = self.input_title.text().strip()
        if title:
            ax.set_title(title)
        ax.view_init(elev=self.slider_elev.value(),
                     azim=self.slider_azim.value())
        if not self.chk_grid.isChecked():
            ax.grid(False)
        self._ax = ax
        return ax

    def _get_cmap(self):
        return self.combo_cmap.currentText()

    def _get_alpha(self):
        return self.slider_alpha.value() / 100.0

    def _get_ranges(self):
        xmin = self.range_spins[0].value()
        xmax = self.range_spins[1].value()
        ymin = self.range_spins[2].value()
        ymax = self.range_spins[3].value()
        res = self.spin_res.value()
        return xmin, xmax, ymin, ymax, res

    # --- Standard z=f(x,y) plots ---
    def _plot_standard(self, plot_type: str):
        ax = self._prepare_axes()
        xmin, xmax, ymin, ymax, res = self._get_ranges()
        x = np.linspace(xmin, xmax, res)
        y = np.linspace(ymin, ymax, res)
        X, Y = np.meshgrid(x, y)

        expr = self.input_func.text().strip()
        # Evaluate expression with x, y available
        local_ns = {"x": X, "y": Y, "np": np, "pi": np.pi}
        Z = eval(expr, {"__builtins__": {}}, local_ns)  # noqa: S307
        if np.isscalar(Z):
            Z = np.full_like(X, Z)

        cmap_name = self._get_cmap()
        alpha = self._get_alpha()

        if plot_type == "Surface":
            surf = ax.plot_surface(X, Y, Z, cmap=cmap_name, alpha=alpha,
                                   edgecolor="none", antialiased=True)
            if self.chk_colorbar.isChecked():
                self.figure.colorbar(surf, ax=ax, shrink=0.5, pad=0.08)

        elif plot_type == "Wireframe":
            ax.plot_wireframe(X, Y, Z, color="steelblue", alpha=alpha,
                              rstride=max(1, res // 20),
                              cstride=max(1, res // 20))

        elif plot_type == "Scatter 3D":
            step = max(1, res // 25)
            xs = X[::step, ::step].ravel()
            ys = Y[::step, ::step].ravel()
            zs = Z[::step, ::step].ravel()
            sc = ax.scatter(xs, ys, zs, c=zs, cmap=cmap_name, alpha=alpha,
                            s=12, depthshade=True)
            if self.chk_colorbar.isChecked():
                self.figure.colorbar(sc, ax=ax, shrink=0.5, pad=0.08)

        elif plot_type == "Contour 3D":
            cset = ax.contour3D(X, Y, Z, 40, cmap=cmap_name, alpha=alpha)
            if self.chk_colorbar.isChecked():
                self.figure.colorbar(cset, ax=ax, shrink=0.5, pad=0.08)

        self.figure.tight_layout()
        self.canvas.draw()

    # --- Parametric surface ---
    def _plot_parametric(self):
        ax = self._prepare_axes()
        xmin, xmax, ymin, ymax, res = self._get_ranges()

        u = np.linspace(xmin, xmax, res)
        v = np.linspace(ymin, ymax, res)
        U, V = np.meshgrid(u, v)

        ns = {"u": U, "v": V, "np": np, "pi": np.pi}
        X = eval(self.input_px.text().strip(), {"__builtins__": {}}, ns)  # noqa: S307
        Y = eval(self.input_py.text().strip(), {"__builtins__": {}}, ns)  # noqa: S307
        Z = eval(self.input_pz.text().strip(), {"__builtins__": {}}, ns)  # noqa: S307

        for arr_name, arr in [("X", X), ("Y", Y), ("Z", Z)]:
            if np.isscalar(arr):
                raise ValueError(f"Parametric expression for {arr_name} returned a scalar.")

        cmap_name = self._get_cmap()
        alpha = self._get_alpha()

        surf = ax.plot_surface(X, Y, Z, cmap=cmap_name, alpha=alpha,
                               edgecolor="none", antialiased=True)
        if self.chk_colorbar.isChecked():
            self.figure.colorbar(surf, ax=ax, shrink=0.5, pad=0.08)

        self.figure.tight_layout()
        self.canvas.draw()

    # --- Vector field ---
    def _plot_vector_field(self):
        ax = self._prepare_axes()
        xmin, xmax, ymin, ymax, _ = self._get_ranges()

        pts = 8  # grid density for quiver
        x = np.linspace(xmin, xmax, pts)
        y = np.linspace(ymin, ymax, pts)
        z = np.linspace(xmin, xmax, pts)
        X, Y, Z = np.meshgrid(x, y, z)

        ns = {"x": X, "y": Y, "z": Z, "np": np, "pi": np.pi}
        Fx = eval(self.input_vx.text().strip(), {"__builtins__": {}}, ns)  # noqa: S307
        Fy = eval(self.input_vy.text().strip(), {"__builtins__": {}}, ns)  # noqa: S307
        Fz = eval(self.input_vz.text().strip(), {"__builtins__": {}}, ns)  # noqa: S307

        for name, arr in [("Fx", Fx), ("Fy", Fy), ("Fz", Fz)]:
            if np.isscalar(arr):
                if name == "Fx":
                    Fx = np.full_like(X, Fx)
                elif name == "Fy":
                    Fy = np.full_like(X, Fy)
                else:
                    Fz = np.full_like(X, Fz)

        alpha = self._get_alpha()
        ax.quiver(X, Y, Z, Fx, Fy, Fz, length=0.4, normalize=True,
                  alpha=alpha, color="steelblue", arrow_length_ratio=0.3)

        self.figure.tight_layout()
        self.canvas.draw()

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------
    def _clear_plot(self):
        self.figure.clear()
        self.canvas.draw()
        self._log("Plot cleared.")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export(self):
        """Open a file dialog and export the current figure to PNG, SVG, or PDF."""
        path, filt = QFileDialog.getSaveFileName(
            self, "Export Plot",
            "plot.png",
            "PNG Image (*.png);;SVG Image (*.svg);;PDF Document (*.pdf);;All Files (*)"
        )
        if not path:
            return
        try:
            self.figure.savefig(path, dpi=150, bbox_inches="tight")
            self._log(f"Exported plot to {path}")
            QMessageBox.information(self, "Export", f"Saved to:\n{path}")
        except Exception as exc:
            self._log(f"Export error: {exc}")
            QMessageBox.warning(self, "Export Error", str(exc))

    # ------------------------------------------------------------------
    # Animation: rotation and parameter sweep with GIF export
    # ------------------------------------------------------------------

    def _create_animation(self):
        """Create an animated 3D plot (rotation or parameter sweep) and export as GIF."""
        anim_types = ["Rotate View (360 degrees)", "Parameter Sweep", "Elevation Sweep"]
        anim_type, ok = QInputDialog.getItem(
            self, "3D Animation", "Animation type:", anim_types, editable=False
        )
        if not ok:
            return

        n_frames, ok = QInputDialog.getInt(self, "Frames", "Number of frames:", 36, 5, 200)
        if not ok:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Animation", "animation_3d.gif", "GIF (*.gif);;All Files (*)"
        )
        if not path:
            return

        try:
            frames = []
            plot_type = self.combo_plot_type.currentText()

            for i in range(n_frames):
                t = i / max(n_frames - 1, 1)
                fig = Figure(figsize=(7, 6), dpi=80)
                style_figure(fig)
                ax = fig.add_subplot(111, projection="3d")

                # Set view based on animation type
                if "Rotate View" in anim_type:
                    azim = int(360 * t)
                    elev = self.slider_elev.value()
                    ax.view_init(elev=elev, azim=azim)
                elif "Elevation Sweep" in anim_type:
                    azim = self.slider_azim.value()
                    elev = -30 + 120 * t
                    ax.view_init(elev=elev, azim=azim)
                else:  # Parameter Sweep
                    azim = self.slider_azim.value()
                    elev = self.slider_elev.value()
                    ax.view_init(elev=elev, azim=azim)

                # Generate data with parameter variation for sweep
                xmin, xmax, ymin, ymax, res = self._get_ranges()
                x = np.linspace(xmin, xmax, res)
                y = np.linspace(ymin, ymax, res)
                X, Y = np.meshgrid(x, y)

                if "Parameter Sweep" in anim_type:
                    # Modify expression with time-varying parameter
                    param = 0.5 + 3.0 * t
                    expr = self.input_func.text().strip()
                    local_ns = {"x": X, "y": Y, "np": np, "pi": np.pi, "t": param}
                    try:
                        Z = eval(expr, {"__builtins__": {}}, local_ns)
                    except Exception:
                        Z = np.sin(np.sqrt(X**2 + Y**2) * param)
                    if np.isscalar(Z):
                        Z = np.full_like(X, Z)
                else:
                    expr = self.input_func.text().strip()
                    local_ns = {"x": X, "y": Y, "np": np, "pi": np.pi}
                    Z = eval(expr, {"__builtins__": {}}, local_ns)
                    if np.isscalar(Z):
                        Z = np.full_like(X, Z)

                cmap_name = self._get_cmap()
                alpha = self._get_alpha()
                ax.plot_surface(X, Y, Z, cmap=cmap_name, alpha=alpha,
                                edgecolor="none", antialiased=True)
                ax.set_xlabel(self.input_xlabel.text())
                ax.set_ylabel(self.input_ylabel.text())
                ax.set_zlabel(self.input_zlabel.text())
                title = self.input_title.text().strip()
                if title:
                    ax.set_title(title)
                elif "Parameter Sweep" in anim_type:
                    ax.set_title(f"t = {0.5 + 3.0 * t:.2f}")
                fig.tight_layout()

                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=80)
                buf.seek(0)
                plt.close(fig)
                frames.append(buf)

            # Assemble GIF
            from PIL import Image
            images = [Image.open(buf) for buf in frames]
            images[0].save(
                path, save_all=True, append_images=images[1:],
                duration=100, loop=0, optimize=True
            )
            for buf in frames:
                buf.close()

            self._log(f"3D animation saved: {path} ({n_frames} frames)")
            QMessageBox.information(self, "Animation", f"GIF saved:\n{path}")
        except ImportError:
            QMessageBox.warning(self, "Animation",
                                "Pillow is required for GIF export.\nInstall with: pip install Pillow")
        except Exception as exc:
            self._log(f"Animation error: {exc}")
            QMessageBox.warning(self, "Animation Error", str(exc))

    # ------------------------------------------------------------------
    # Isosurface rendering for volumetric data
    # ------------------------------------------------------------------

    def _render_isosurface(self):
        """Render an isosurface from a volumetric scalar field."""
        expr, ok = QInputDialog.getText(
            self, "Isosurface", "Scalar field f(x,y,z):",
            text="x**2 + y**2 + z**2"
        )
        if not ok or not expr.strip():
            return

        iso_val, ok = QInputDialog.getDouble(
            self, "Isosurface", "Iso-value (level):", 1.0, -1e6, 1e6, 4
        )
        if not ok:
            return

        try:
            xmin, xmax, ymin, ymax, res = self._get_ranges()
            grid_res = min(res, 40)  # Keep manageable for marching cubes

            x = np.linspace(xmin, xmax, grid_res)
            y = np.linspace(ymin, ymax, grid_res)
            z = np.linspace(xmin, xmax, grid_res)
            X, Y, Z = np.meshgrid(x, y, z)

            ns = {"x": X, "y": Y, "z": Z, "np": np, "pi": np.pi}
            F = eval(expr, {"__builtins__": {}}, ns)
            if np.isscalar(F):
                F = np.full_like(X, F)

            # Simple marching cubes approximation using contour-based triangulation
            self.figure.clear()
            ax = self.figure.add_subplot(111, projection="3d")

            # Extract isosurface using simple threshold-based visualization
            mask = np.abs(F - iso_val) < (F.max() - F.min()) * 0.05
            if not mask.any():
                # Widen the threshold
                mask = np.abs(F - iso_val) < (F.max() - F.min()) * 0.15

            xs = X[mask].ravel()
            ys = Y[mask].ravel()
            zs = Z[mask].ravel()

            if len(xs) == 0:
                QMessageBox.warning(self, "Isosurface",
                                    f"No points found near iso-value {iso_val}.")
                return

            # Subsample if too many points
            if len(xs) > 5000:
                idx = np.random.choice(len(xs), 5000, replace=False)
                xs, ys, zs = xs[idx], ys[idx], zs[idx]

            ax.scatter(xs, ys, zs, c=zs, cmap=self._get_cmap(),
                       alpha=self._get_alpha() * 0.6, s=4, depthshade=True)

            ax.set_xlabel(self.input_xlabel.text())
            ax.set_ylabel(self.input_ylabel.text())
            ax.set_zlabel(self.input_zlabel.text())
            ax.set_title(f"Isosurface: {expr} = {iso_val}")
            ax.view_init(elev=self.slider_elev.value(), azim=self.slider_azim.value())
            self._ax = ax
            self.figure.tight_layout()
            self.canvas.draw()
            self._log(f"Isosurface rendered: {expr} = {iso_val} ({len(xs)} points)")

            # Try to use scikit-image marching cubes if available
            try:
                from skimage.measure import marching_cubes
                verts, faces, normals, values = marching_cubes(F, level=iso_val)
                # Scale vertices to data coordinates
                verts[:, 0] = verts[:, 0] / grid_res * (xmax - xmin) + xmin
                verts[:, 1] = verts[:, 1] / grid_res * (ymax - ymin) + ymin
                verts[:, 2] = verts[:, 2] / grid_res * (xmax - xmin) + xmin

                self.figure.clear()
                ax = self.figure.add_subplot(111, projection="3d")
                mesh = Poly3DCollection(verts[faces], alpha=self._get_alpha() * 0.7)
                mesh.set_facecolor(cm.get_cmap(self._get_cmap())(0.5))
                mesh.set_edgecolor("none")
                ax.add_collection3d(mesh)
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.set_zlim(xmin, xmax)
                ax.set_xlabel(self.input_xlabel.text())
                ax.set_ylabel(self.input_ylabel.text())
                ax.set_zlabel(self.input_zlabel.text())
                ax.set_title(f"Isosurface: {expr} = {iso_val}")
                ax.view_init(elev=self.slider_elev.value(), azim=self.slider_azim.value())
                self._ax = ax
                self.figure.tight_layout()
                self.canvas.draw()
                self._log(f"Isosurface (marching cubes): {len(faces)} triangles")
            except ImportError:
                pass  # Use scatter fallback above

        except Exception as exc:
            self._log(f"Isosurface error: {exc}")
            QMessageBox.warning(self, "Isosurface Error", str(exc))

    # ------------------------------------------------------------------
    # Vector field visualization with streamlines and quiver
    # ------------------------------------------------------------------

    def _plot_streamlines(self):
        """Visualize a 3D vector field with quiver plots and 2D streamline slices."""
        # Use existing vector field inputs
        self.combo_plot_type.setCurrentText("Vector Field")

        modes = ["Quiver (3D arrows)", "2D Streamline Slices", "Combined (quiver + slices)"]
        mode, ok = QInputDialog.getItem(
            self, "Vector Field Visualization", "Mode:", modes, editable=False
        )
        if not ok:
            return

        try:
            xmin, xmax, ymin, ymax, _ = self._get_ranges()
            pts = 10

            if "Quiver" in mode or "Combined" in mode:
                self.figure.clear()
                ax = self.figure.add_subplot(111, projection="3d")

                x = np.linspace(xmin, xmax, pts)
                y = np.linspace(ymin, ymax, pts)
                z = np.linspace(xmin, xmax, pts)
                X, Y, Z = np.meshgrid(x, y, z)

                ns = {"x": X, "y": Y, "z": Z, "np": np, "pi": np.pi}
                Fx = eval(self.input_vx.text().strip(), {"__builtins__": {}}, ns)
                Fy = eval(self.input_vy.text().strip(), {"__builtins__": {}}, ns)
                Fz = eval(self.input_vz.text().strip(), {"__builtins__": {}}, ns)

                for name, arr in [("Fx", Fx), ("Fy", Fy), ("Fz", Fz)]:
                    if np.isscalar(arr):
                        if name == "Fx": Fx = np.full_like(X, Fx)
                        elif name == "Fy": Fy = np.full_like(X, Fy)
                        else: Fz = np.full_like(X, Fz)

                # Color by magnitude
                magnitude = np.sqrt(Fx**2 + Fy**2 + Fz**2)
                magnitude[magnitude == 0] = 1e-10

                alpha = self._get_alpha()
                ax.quiver(X, Y, Z, Fx, Fy, Fz, length=0.4, normalize=True,
                          alpha=alpha, color="steelblue", arrow_length_ratio=0.3)

                ax.set_xlabel(self.input_xlabel.text())
                ax.set_ylabel(self.input_ylabel.text())
                ax.set_zlabel(self.input_zlabel.text())
                ax.set_title("Vector Field (3D Quiver)")
                ax.view_init(elev=self.slider_elev.value(), azim=self.slider_azim.value())
                self._ax = ax

            if "Streamline" in mode or "Combined" in mode:
                # Add 2D streamline slices at z-midpoint
                if "Combined" not in mode:
                    self.figure.clear()

                pts_2d = 30
                x2 = np.linspace(xmin, xmax, pts_2d)
                y2 = np.linspace(ymin, ymax, pts_2d)
                X2, Y2 = np.meshgrid(x2, y2)
                z_mid = (xmin + xmax) / 2
                Z2 = np.full_like(X2, z_mid)

                ns2 = {"x": X2, "y": Y2, "z": Z2, "np": np, "pi": np.pi}
                Fx2 = eval(self.input_vx.text().strip(), {"__builtins__": {}}, ns2)
                Fy2 = eval(self.input_vy.text().strip(), {"__builtins__": {}}, ns2)

                if np.isscalar(Fx2):
                    Fx2 = np.full_like(X2, Fx2)
                if np.isscalar(Fy2):
                    Fy2 = np.full_like(X2, Fy2)

                if "Combined" not in mode:
                    ax_2d = self.figure.add_subplot(111)
                else:
                    # Add a small inset for 2D streamlines
                    ax_2d = self.figure.add_axes([0.65, 0.05, 0.33, 0.33])

                speed = np.sqrt(Fx2**2 + Fy2**2)
                ax_2d.streamplot(x2, y2, Fx2, Fy2, color=speed,
                                 cmap=self._get_cmap(), density=1.5, linewidth=1)
                ax_2d.set_xlabel("X")
                ax_2d.set_ylabel("Y")
                ax_2d.set_title(f"Streamlines (z={z_mid:.1f})", fontsize=9)
                ax_2d.set_aspect("equal")

            self.figure.tight_layout()
            self.canvas.draw()
            self._log(f"Vector field visualized: {mode}")
        except Exception as exc:
            self._log(f"Streamlines error: {exc}")
            QMessageBox.warning(self, "Streamlines Error", str(exc))

    # ------------------------------------------------------------------
    # Multi-plot support (3D subplots)
    # ------------------------------------------------------------------

    def _create_multi_plot(self):
        """Create a grid of 3D subplots with different functions or view angles."""
        modes = [
            "Multiple functions (2x2)",
            "Multiple view angles (2x2)",
            "Function comparison (1x2)",
            "Custom grid with presets",
        ]
        mode, ok = QInputDialog.getItem(
            self, "3D Multi-Plot", "Mode:", modes, editable=False
        )
        if not ok:
            return

        try:
            xmin, xmax, ymin, ymax, res = self._get_ranges()
            x = np.linspace(xmin, xmax, min(res, 60))
            y = np.linspace(ymin, ymax, min(res, 60))
            X, Y = np.meshgrid(x, y)
            cmap_name = self._get_cmap()
            alpha = self._get_alpha()

            self.figure.clear()

            if "Multiple functions" in mode:
                funcs = [
                    ("Sinc", "np.sinc(np.sqrt(x**2 + y**2))"),
                    ("Saddle", "x**2 - y**2"),
                    ("Gaussian", "np.exp(-(x**2 + y**2))"),
                    ("Ripple", "np.sin(np.sqrt(x**2 + y**2))"),
                ]
                for i, (name, expr) in enumerate(funcs):
                    ax = self.figure.add_subplot(2, 2, i + 1, projection="3d")
                    ns = {"x": X, "y": Y, "np": np, "pi": np.pi}
                    Z = eval(expr, {"__builtins__": {}}, ns)
                    ax.plot_surface(X, Y, Z, cmap=cmap_name, alpha=alpha, edgecolor="none")
                    ax.set_title(name, fontsize=9)
                    ax.view_init(elev=self.slider_elev.value(), azim=self.slider_azim.value())

            elif "Multiple view angles" in mode:
                expr = self.input_func.text().strip()
                ns = {"x": X, "y": Y, "np": np, "pi": np.pi}
                Z = eval(expr, {"__builtins__": {}}, ns)
                if np.isscalar(Z):
                    Z = np.full_like(X, Z)
                angles = [(30, 45), (30, 135), (60, 45), (0, 0)]
                labels = ["Front", "Side", "Top-angle", "Top-down"]
                for i, ((elev, azim), label) in enumerate(zip(angles, labels)):
                    ax = self.figure.add_subplot(2, 2, i + 1, projection="3d")
                    ax.plot_surface(X, Y, Z, cmap=cmap_name, alpha=alpha, edgecolor="none")
                    ax.view_init(elev=elev, azim=azim)
                    ax.set_title(f"{label} (e={elev}, a={azim})", fontsize=8)

            elif "Function comparison" in mode:
                expr1, ok1 = QInputDialog.getText(self, "Function 1", "z = f(x,y):",
                                                    text=self.input_func.text())
                if not ok1:
                    return
                expr2, ok2 = QInputDialog.getText(self, "Function 2", "z = f(x,y):",
                                                    text="x**2 + y**2")
                if not ok2:
                    return
                for i, expr in enumerate([expr1, expr2]):
                    ax = self.figure.add_subplot(1, 2, i + 1, projection="3d")
                    ns = {"x": X, "y": Y, "np": np, "pi": np.pi}
                    Z = eval(expr, {"__builtins__": {}}, ns)
                    if np.isscalar(Z):
                        Z = np.full_like(X, Z)
                    ax.plot_surface(X, Y, Z, cmap=cmap_name, alpha=alpha, edgecolor="none")
                    ax.set_title(expr[:40], fontsize=8)
                    ax.view_init(elev=self.slider_elev.value(), azim=self.slider_azim.value())

            elif "Custom grid" in mode:
                preset_names = list(PRESETS.keys())[:6]
                nrows = 2
                ncols = 3
                for i, name in enumerate(preset_names):
                    ax = self.figure.add_subplot(nrows, ncols, i + 1, projection="3d")
                    ns = {"x": X, "y": Y, "np": np, "pi": np.pi}
                    Z = eval(PRESETS[name], {"__builtins__": {}}, ns)
                    ax.plot_surface(X, Y, Z, cmap=cmap_name, alpha=alpha, edgecolor="none")
                    ax.set_title(name, fontsize=8)
                    ax.view_init(elev=30, azim=45)

            self.figure.tight_layout()
            self.canvas.draw()
            self._log(f"3D multi-plot created: {mode}")
        except Exception as exc:
            self._log(f"Multi-plot error: {exc}")
            QMessageBox.warning(self, "Multi-Plot Error", str(exc))

    # ------------------------------------------------------------------
    # STL/OBJ Mesh visualization
    # ------------------------------------------------------------------

    def _load_mesh(self):
        """Load and render a 3D mesh from STL or OBJ file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load 3D Mesh", "",
            "3D Mesh Files (*.stl *.obj);;STL Files (*.stl);;OBJ Files (*.obj);;All Files (*)"
        )
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".stl":
                vertices, faces = self._parse_stl(path)
            elif ext == ".obj":
                vertices, faces = self._parse_obj(path)
            else:
                QMessageBox.warning(self, "Mesh Error", f"Unsupported format: {ext}")
                return

            if len(vertices) == 0 or len(faces) == 0:
                QMessageBox.warning(self, "Mesh Error", "No geometry found in file.")
                return

            self.figure.clear()
            ax = self.figure.add_subplot(111, projection="3d")

            # Build polygon collection
            mesh_polygons = vertices[faces]
            cmap_name = self._get_cmap()
            alpha = self._get_alpha()

            # Color by Z-height of face centers
            z_centers = mesh_polygons[:, :, 2].mean(axis=1)
            z_norm = (z_centers - z_centers.min()) / (z_centers.max() - z_centers.min() + 1e-10)
            facecolors = cm.get_cmap(cmap_name)(z_norm)

            collection = Poly3DCollection(mesh_polygons, alpha=alpha)
            collection.set_facecolor(facecolors)
            collection.set_edgecolor("gray")
            collection.set_linewidth(0.1)
            ax.add_collection3d(collection)

            # Set axis limits from data
            all_pts = vertices
            margin = 0.05
            for dim, setter in [(0, ax.set_xlim), (1, ax.set_ylim), (2, ax.set_zlim)]:
                lo, hi = all_pts[:, dim].min(), all_pts[:, dim].max()
                span = hi - lo
                setter(lo - margin * span, hi + margin * span)

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(os.path.basename(path))
            ax.view_init(elev=self.slider_elev.value(), azim=self.slider_azim.value())
            self._ax = ax

            self.figure.tight_layout()
            self.canvas.draw()
            self._log(f"Mesh loaded: {os.path.basename(path)} ({len(vertices)} vertices, {len(faces)} faces)")
        except Exception as exc:
            self._log(f"Mesh loading error: {exc}")
            QMessageBox.warning(self, "Mesh Error", str(exc))

    def _parse_stl(self, path: str):
        """Parse an STL file (ASCII or binary) and return vertices and face indices."""
        import struct

        vertices = []
        faces = []

        with open(path, "rb") as f:
            header = f.read(80)
            # Check if ASCII
            try:
                header_str = header.decode("ascii", errors="ignore").strip().lower()
            except Exception:
                header_str = ""

        if header_str.startswith("solid") and not header_str.startswith("solid \x00"):
            # Try ASCII STL
            try:
                vert_list = []
                with open(path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("vertex"):
                            parts = line.split()
                            vert_list.append([float(parts[1]), float(parts[2]), float(parts[3])])

                vertices = np.array(vert_list, dtype=np.float64)
                n_faces = len(vertices) // 3
                faces = np.arange(len(vertices)).reshape(n_faces, 3)
                return vertices, faces
            except Exception:
                pass

        # Binary STL
        with open(path, "rb") as f:
            f.read(80)  # header
            n_triangles = struct.unpack("<I", f.read(4))[0]
            vert_list = []
            for _ in range(n_triangles):
                f.read(12)  # normal vector
                for _ in range(3):
                    vx, vy, vz = struct.unpack("<fff", f.read(12))
                    vert_list.append([vx, vy, vz])
                f.read(2)  # attribute byte count

        vertices = np.array(vert_list, dtype=np.float64)
        n_faces = len(vertices) // 3
        faces = np.arange(len(vertices)).reshape(n_faces, 3)
        return vertices, faces

    def _parse_obj(self, path: str):
        """Parse a Wavefront OBJ file and return vertices and face indices."""
        vertices = []
        faces = []

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if parts[0] == "v" and len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == "f":
                    # OBJ face indices are 1-based; may contain v/vt/vn format
                    face_verts = []
                    for p in parts[1:]:
                        idx = int(p.split("/")[0]) - 1
                        face_verts.append(idx)
                    # Triangulate polygons with more than 3 vertices
                    for i in range(1, len(face_verts) - 1):
                        faces.append([face_verts[0], face_verts[i], face_verts[i + 1]])

        vertices = np.array(vertices, dtype=np.float64) if vertices else np.empty((0, 3))
        faces = np.array(faces, dtype=np.int64) if faces else np.empty((0, 3), dtype=np.int64)
        return vertices, faces

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
    # Convenience entry point
    # ------------------------------------------------------------------
    def run(self):
        """Show the widget and, if no QApplication exists, start the event loop."""
        app = QApplication.instance()
        standalone = app is None
        if standalone:
            app = QApplication(sys.argv)
        self.show()
        self._do_plot()  # render default plot on launch
        if standalone:
            sys.exit(app.exec_())


# ----------------------------------------------------------------------
# Stand-alone execution
# ----------------------------------------------------------------------
if __name__ == "__main__":
    widget = Plotter3DWidget()
    widget.run()
