"""
fractal_explorer.py - Comprehensive Fractal Explorer for Axiom Scientific Suite

Provides interactive exploration of 23+ fractal types including 2D escape-time fractals,
IFS/L-system fractals, 1D fractals, and 3D fractal projections. Features include
mouse-driven zoom/pan, progressive rendering, smooth coloring, custom formula editor,
animation export, and high-resolution image export.

Fractal types:
  2D Escape-Time: Mandelbrot, Julia, Burning Ship, Multibrot, Tricorn, Newton,
                  Phoenix, Magnet, Lyapunov, Buddhabrot
  2D IFS/L-System: Barnsley Fern, Sierpinski Triangle, Koch Snowflake,
                   Dragon Curve, Hilbert Curve, Apollonian Gasket
  1D: Cantor Set, Logistic Map / Bifurcation, Feigenbaum
  3D: Mandelbulb, Quaternion Julia, Menger Sponge, Sierpinski Tetrahedron
"""

import os
import io
import json
import time
import traceback
import colorsys
import math
import struct
from collections import deque

import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTreeWidget,
    QTreeWidgetItem, QLabel, QSlider, QSpinBox, QDoubleSpinBox,
    QComboBox, QPushButton, QGroupBox, QFormLayout, QLineEdit,
    QFileDialog, QMessageBox, QProgressBar, QCheckBox, QTextEdit,
    QScrollArea, QApplication, QToolBar, QAction, QSizePolicy,
    QGridLayout, QTabWidget, QFrame, QMenu
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QRectF, QPointF
from PyQt5.QtGui import QFont, QCursor, QPixmap, QImage, QColor, QPainter

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

try:
    from mpl_style import style_figure, style_axes
except ImportError:
    def style_figure(f): pass
    def style_axes(a): pass
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLORMAPS = [
    "hot", "inferno", "magma", "plasma", "viridis", "twilight", "jet",
    "cubehelix", "twilight_shifted", "hsv", "Spectral", "ocean",
    "turbo", "copper", "gnuplot2", "custom_cyclic"
]

JULIA_PRESETS = {
    "Douady Rabbit": (-0.123, 0.745),
    "Dendrite": (0.0, 1.0),
    "San Marco": (-0.75, 0.0),
    "Siegel Disk": (-0.391, -0.587),
    "Dragon": (0.36, 0.1),
    "Spiral": (-0.8, 0.156),
    "Star": (-0.7269, 0.1889),
    "Lightning": (-0.4, 0.6),
}

FRACTAL_CATEGORIES = {
    "2D Escape-Time": [
        "Mandelbrot", "Julia Set", "Burning Ship", "Multibrot",
        "Tricorn", "Newton Fractal", "Phoenix", "Magnet",
        "Lyapunov", "Buddhabrot"
    ],
    "2D IFS / L-System": [
        "Barnsley Fern", "Sierpinski Triangle", "Koch Snowflake",
        "Dragon Curve", "Hilbert Curve", "Apollonian Gasket"
    ],
    "1D Fractals": [
        "Cantor Set", "Logistic Map", "Feigenbaum"
    ],
    "3D Fractals": [
        "Mandelbulb", "Quaternion Julia", "Menger Sponge",
        "Sierpinski Tetrahedron"
    ],
    "Custom": [
        "Custom Formula", "Custom IFS", "Custom L-System",
        "Custom Newton"
    ]
}


def _make_custom_cyclic_cmap():
    """Create a smooth cyclic colormap for fractal rendering."""
    colors = []
    for i in range(256):
        t = i / 255.0
        r = 0.5 + 0.5 * math.cos(2 * math.pi * (t + 0.0))
        g = 0.5 + 0.5 * math.cos(2 * math.pi * (t + 0.33))
        b = 0.5 + 0.5 * math.cos(2 * math.pi * (t + 0.67))
        colors.append((r, g, b))
    return mcolors.ListedColormap(colors, name="custom_cyclic")


try:
    _custom_cmap = _make_custom_cyclic_cmap()
    plt.register_cmap(cmap=_custom_cmap)
except Exception:
    pass


# ---------------------------------------------------------------------------
# Rendering worker thread
# ---------------------------------------------------------------------------

class FractalRenderThread(QThread):
    """Background thread for rendering fractals without blocking the UI."""

    progress = pyqtSignal(int)
    finished = pyqtSignal(object, float)  # (image_array, elapsed_seconds)

    def __init__(self, render_func, params, parent=None):
        super().__init__(parent)
        self._render_func = render_func
        self._params = params
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        t0 = time.perf_counter()
        try:
            result = self._render_func(self._params, self.progress, lambda: self._cancelled)
        except Exception as exc:
            result = None
        elapsed = time.perf_counter() - t0
        if not self._cancelled:
            self.finished.emit(result, elapsed)


# ---------------------------------------------------------------------------
# Fractal computation functions (numpy-vectorized where applicable)
# ---------------------------------------------------------------------------

@np.errstate(over='ignore', invalid='ignore')
def compute_mandelbrot(params, progress_sig, is_cancelled):
    """Vectorized Mandelbrot with smooth coloring — optimized with batch iterations."""
    w = params["width"]
    h = params["height"]
    max_iter = params["max_iter"]
    xmin, xmax = params["xmin"], params["xmax"]
    ymin, ymax = params["ymin"], params["ymax"]
    ss = params.get("supersample", 1)

    sw, sh = w * ss, h * ss
    x = np.linspace(xmin, xmax, sw, dtype=np.float64)
    y = np.linspace(ymin, ymax, sh, dtype=np.float64)
    C = x[np.newaxis, :] + 1j * y[:, np.newaxis]

    Z = np.zeros_like(C, dtype=np.complex128)
    smooth = np.zeros((sh, sw), dtype=np.float64)
    not_escaped = np.ones((sh, sw), dtype=bool)

    # Batch iterations: check escapes every BATCH_SIZE iterations for speed
    BATCH_SIZE = max(1, min(50, max_iter // 10))
    LOG2 = math.log(2.0)
    i = 0
    while i < max_iter:
        if is_cancelled():
            return None
        batch_end = min(i + BATCH_SIZE, max_iter)
        for _ in range(i, batch_end):
            Z[not_escaped] = Z[not_escaped] ** 2 + C[not_escaped]
        i = batch_end
        # Check escapes after the batch
        abs_Z = np.abs(Z)
        escaped = not_escaped & (abs_Z > 4.0)
        if np.any(escaped):
            log_zn = np.log(abs_Z[escaped] + 1e-30) / 2.0
            nu = np.log(log_zn / LOG2 + 1e-30) / LOG2
            smooth[escaped] = i - nu
            not_escaped[escaped] = False
        # Early exit if everything escaped
        if not np.any(not_escaped):
            break
        progress_sig.emit(int(100 * i / max_iter))

    smooth[not_escaped] = 0  # interior
    if ss > 1:
        smooth = smooth.reshape(h, ss, w, ss).mean(axis=(1, 3))
    progress_sig.emit(100)
    return smooth


@np.errstate(over="ignore", invalid="ignore")
def compute_julia(params, progress_sig, is_cancelled):
    """Vectorized Julia set — optimized with batch iterations."""
    w = params["width"]
    h = params["height"]
    max_iter = params["max_iter"]
    xmin, xmax = params["xmin"], params["xmax"]
    ymin, ymax = params["ymin"], params["ymax"]
    cr, ci = params["julia_cr"], params["julia_ci"]
    ss = params.get("supersample", 1)

    sw, sh = w * ss, h * ss
    x = np.linspace(xmin, xmax, sw, dtype=np.float64)
    y = np.linspace(ymin, ymax, sh, dtype=np.float64)
    Z = x[np.newaxis, :] + 1j * y[:, np.newaxis]
    C = complex(cr, ci)

    smooth = np.zeros((sh, sw), dtype=np.float64)
    not_escaped = np.ones((sh, sw), dtype=bool)

    BATCH_SIZE = max(1, min(50, max_iter // 10))
    LOG2 = math.log(2.0)
    i = 0
    while i < max_iter:
        if is_cancelled():
            return None
        batch_end = min(i + BATCH_SIZE, max_iter)
        for _ in range(i, batch_end):
            Z[not_escaped] = Z[not_escaped] ** 2 + C
        i = batch_end
        abs_Z = np.abs(Z)
        escaped = not_escaped & (abs_Z > 4.0)
        if np.any(escaped):
            log_zn = np.log(abs_Z[escaped] + 1e-30) / 2.0
            nu = np.log(log_zn / LOG2 + 1e-30) / LOG2
            smooth[escaped] = i - nu
            not_escaped[escaped] = False
        if not np.any(not_escaped):
            break
        progress_sig.emit(int(100 * i / max_iter))

    smooth[not_escaped] = 0
    if ss > 1:
        smooth = smooth.reshape(h, ss, w, ss).mean(axis=(1, 3))
    progress_sig.emit(100)
    return smooth


@np.errstate(over="ignore", invalid="ignore")
def compute_burning_ship(params, progress_sig, is_cancelled):
    """Burning Ship fractal: z = (|Re(z)| + i|Im(z)|)^2 + c."""
    w, h = params["width"], params["height"]
    max_iter = params["max_iter"]
    xmin, xmax = params["xmin"], params["xmax"]
    ymin, ymax = params["ymin"], params["ymax"]

    x = np.linspace(xmin, xmax, w)
    y = np.linspace(ymin, ymax, h)
    C = x[np.newaxis, :] + 1j * y[:, np.newaxis]
    Z = np.zeros_like(C, dtype=np.complex128)
    smooth = np.zeros((h, w), dtype=np.float64)
    mask = np.ones((h, w), dtype=bool)

    chunk = max(1, max_iter // 20)
    for i in range(max_iter):
        if is_cancelled():
            return None
        Zr = np.abs(Z[mask].real)
        Zi = np.abs(Z[mask].imag)
        Z[mask] = (Zr + 1j * Zi) ** 2 + C[mask]
        escaped = mask & (np.abs(Z) > 4.0)
        log_zn = np.log(np.abs(Z[escaped]) + 1e-30) / 2.0
        nu = np.log(log_zn / math.log(2.0) + 1e-30) / math.log(2.0)
        smooth[escaped] = i + 1 - nu
        mask[escaped] = False
        if i % chunk == 0:
            progress_sig.emit(int(100 * i / max_iter))

    smooth[mask] = 0
    if params.get("supersample", 1) > 1:
        ss = params["supersample"]
        smooth = smooth.reshape(h // ss, ss, w // ss, ss).mean(axis=(1, 3))
    progress_sig.emit(100)
    return smooth


@np.errstate(over="ignore", invalid="ignore")
def compute_multibrot(params, progress_sig, is_cancelled):
    """Multibrot: z = z^d + c with configurable power d."""
    w, h = params["width"], params["height"]
    max_iter = params["max_iter"]
    xmin, xmax = params["xmin"], params["xmax"]
    ymin, ymax = params["ymin"], params["ymax"]
    power = params.get("power", 3)

    x = np.linspace(xmin, xmax, w)
    y = np.linspace(ymin, ymax, h)
    C = x[np.newaxis, :] + 1j * y[:, np.newaxis]
    Z = np.zeros_like(C, dtype=np.complex128)
    smooth = np.zeros((h, w), dtype=np.float64)
    mask = np.ones((h, w), dtype=bool)

    chunk = max(1, max_iter // 20)
    for i in range(max_iter):
        if is_cancelled():
            return None
        Z[mask] = Z[mask] ** power + C[mask]
        escaped = mask & (np.abs(Z) > 4.0)
        log_zn = np.log(np.abs(Z[escaped]) + 1e-30) / 2.0
        nu = np.log(log_zn / math.log(power) + 1e-30) / math.log(power)
        smooth[escaped] = i + 1 - nu
        mask[escaped] = False
        if i % chunk == 0:
            progress_sig.emit(int(100 * i / max_iter))

    smooth[mask] = 0
    progress_sig.emit(100)
    return smooth


@np.errstate(over="ignore", invalid="ignore")
def compute_tricorn(params, progress_sig, is_cancelled):
    """Tricorn (Mandelbar): z = conj(z)^2 + c."""
    w, h = params["width"], params["height"]
    max_iter = params["max_iter"]
    xmin, xmax = params["xmin"], params["xmax"]
    ymin, ymax = params["ymin"], params["ymax"]

    x = np.linspace(xmin, xmax, w)
    y = np.linspace(ymin, ymax, h)
    C = x[np.newaxis, :] + 1j * y[:, np.newaxis]
    Z = np.zeros_like(C, dtype=np.complex128)
    smooth = np.zeros((h, w), dtype=np.float64)
    mask = np.ones((h, w), dtype=bool)

    chunk = max(1, max_iter // 20)
    for i in range(max_iter):
        if is_cancelled():
            return None
        Z[mask] = np.conj(Z[mask]) ** 2 + C[mask]
        escaped = mask & (np.abs(Z) > 4.0)
        log_zn = np.log(np.abs(Z[escaped]) + 1e-30) / 2.0
        nu = np.log(log_zn / math.log(2.0) + 1e-30) / math.log(2.0)
        smooth[escaped] = i + 1 - nu
        mask[escaped] = False
        if i % chunk == 0:
            progress_sig.emit(int(100 * i / max_iter))

    smooth[mask] = 0
    progress_sig.emit(100)
    return smooth


@np.errstate(over="ignore", invalid="ignore")
def compute_newton(params, progress_sig, is_cancelled):
    """Newton fractal for z^n - 1."""
    w, h = params["width"], params["height"]
    max_iter = params["max_iter"]
    xmin, xmax = params["xmin"], params["xmax"]
    ymin, ymax = params["ymin"], params["ymax"]
    power = params.get("newton_power", 3)
    tol = 1e-6

    x = np.linspace(xmin, xmax, w)
    y = np.linspace(ymin, ymax, h)
    Z = x[np.newaxis, :] + 1j * y[:, np.newaxis]

    roots = np.exp(2j * np.pi * np.arange(power) / power)
    result = np.zeros((h, w), dtype=np.float64)
    shade = np.zeros((h, w), dtype=np.float64)
    mask = np.ones((h, w), dtype=bool)

    chunk = max(1, max_iter // 10)
    for i in range(max_iter):
        if is_cancelled():
            return None
        denom = power * Z[mask] ** (power - 1)
        denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
        Z[mask] = Z[mask] - (Z[mask] ** power - 1.0) / denom
        for r_idx, root in enumerate(roots):
            converged = mask & (np.abs(Z - root) < tol)
            result[converged] = r_idx + 1
            shade[converged] = i
            mask[converged] = False
        if i % chunk == 0:
            progress_sig.emit(int(100 * i / max_iter))

    combined = result * 50.0 + shade
    progress_sig.emit(100)
    return combined


@np.errstate(over="ignore", invalid="ignore")
def compute_phoenix(params, progress_sig, is_cancelled):
    """Phoenix fractal: z(n+1) = z(n)^2 + Re(c) + Im(c)*z(n-1)."""
    w, h = params["width"], params["height"]
    max_iter = params["max_iter"]
    xmin, xmax = params["xmin"], params["xmax"]
    ymin, ymax = params["ymin"], params["ymax"]
    cr = params.get("phoenix_cr", 0.5667)
    ci = params.get("phoenix_ci", -0.5)

    x = np.linspace(xmin, xmax, w)
    y = np.linspace(ymin, ymax, h)
    Z = x[np.newaxis, :] + 1j * y[:, np.newaxis]
    Z_prev = np.zeros_like(Z, dtype=np.complex128)
    smooth = np.zeros((h, w), dtype=np.float64)
    mask = np.ones((h, w), dtype=bool)

    chunk = max(1, max_iter // 20)
    for i in range(max_iter):
        if is_cancelled():
            return None
        Z_new = Z[mask] ** 2 + cr + ci * Z_prev[mask]
        Z_prev[mask] = Z[mask]
        Z[mask] = Z_new
        escaped = mask & (np.abs(Z) > 4.0)
        smooth[escaped] = i + 1
        mask[escaped] = False
        if i % chunk == 0:
            progress_sig.emit(int(100 * i / max_iter))

    smooth[mask] = 0
    progress_sig.emit(100)
    return smooth


@np.errstate(over="ignore", invalid="ignore")
def compute_magnet(params, progress_sig, is_cancelled):
    """Magnet fractal type I: z = [(z^2 + c - 1)/(2z + c - 2)]^2."""
    w, h = params["width"], params["height"]
    max_iter = params["max_iter"]
    xmin, xmax = params["xmin"], params["xmax"]
    ymin, ymax = params["ymin"], params["ymax"]

    x = np.linspace(xmin, xmax, w)
    y = np.linspace(ymin, ymax, h)
    C = x[np.newaxis, :] + 1j * y[:, np.newaxis]
    Z = np.zeros_like(C, dtype=np.complex128)
    smooth = np.zeros((h, w), dtype=np.float64)
    mask = np.ones((h, w), dtype=bool)

    chunk = max(1, max_iter // 20)
    for i in range(max_iter):
        if is_cancelled():
            return None
        numer = Z[mask] ** 2 + C[mask] - 1.0
        denom = 2.0 * Z[mask] + C[mask] - 2.0
        denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
        Z[mask] = (numer / denom) ** 2
        escaped = mask & (np.abs(Z) > 100.0)
        smooth[escaped] = i + 1
        mask[escaped] = False
        fixed = mask & (np.abs(Z - 1.0) < 1e-6)
        smooth[fixed] = i + 1
        mask[fixed] = False
        if i % chunk == 0:
            progress_sig.emit(int(100 * i / max_iter))

    smooth[mask] = 0
    progress_sig.emit(100)
    return smooth


@np.errstate(over="ignore", invalid="ignore")
def compute_lyapunov(params, progress_sig, is_cancelled):
    """Lyapunov fractal from a logistic map sequence string like 'AB'."""
    w, h = params["width"], params["height"]
    max_iter = min(params["max_iter"], 500)
    xmin, xmax = params.get("xmin", 2.0), params.get("xmax", 4.0)
    ymin, ymax = params.get("ymin", 2.0), params.get("ymax", 4.0)
    seq = params.get("lyapunov_seq", "AB")

    a_vals = np.linspace(xmin, xmax, w)
    b_vals = np.linspace(ymin, ymax, h)
    A, B = np.meshgrid(a_vals, b_vals)

    seq_map = []
    for ch in seq.upper():
        if ch == 'A':
            seq_map.append(0)
        else:
            seq_map.append(1)
    ab = np.stack([A, B], axis=-1)

    x_val = 0.5 * np.ones((h, w), dtype=np.float64)
    lyap = np.zeros((h, w), dtype=np.float64)
    n_seq = len(seq_map)

    warmup = 50
    chunk = max(1, max_iter // 10)
    for i in range(max_iter):
        if is_cancelled():
            return None
        r = ab[:, :, seq_map[i % n_seq]]
        x_val = r * x_val * (1.0 - x_val)
        x_val = np.clip(x_val, 1e-15, 1.0 - 1e-15)
        if i >= warmup:
            deriv = np.abs(r * (1.0 - 2.0 * x_val))
            lyap += np.log(deriv + 1e-30)
        if i % chunk == 0:
            progress_sig.emit(int(100 * i / max_iter))

    lyap /= max(1, max_iter - warmup)
    progress_sig.emit(100)
    return lyap


@np.errstate(over="ignore", invalid="ignore")
def compute_buddhabrot(params, progress_sig, is_cancelled):
    """Buddhabrot via random sampling of escaping orbits."""
    w, h = params["width"], params["height"]
    max_iter = min(params["max_iter"], 2000)
    xmin, xmax = params["xmin"], params["xmax"]
    ymin, ymax = params["ymin"], params["ymax"]
    n_samples = w * h * 2

    image = np.zeros((h, w), dtype=np.float64)
    batch_size = min(200000, n_samples)
    n_batches = max(1, n_samples // batch_size)

    for b in range(n_batches):
        if is_cancelled():
            return None
        cr = np.random.uniform(xmin, xmax, batch_size)
        ci = np.random.uniform(ymin, ymax, batch_size)
        C = cr + 1j * ci
        Z = np.zeros(batch_size, dtype=np.complex128)
        trajectories = np.zeros((max_iter, batch_size), dtype=np.complex128)
        escaped = np.zeros(batch_size, dtype=bool)
        escape_iter = np.full(batch_size, max_iter, dtype=int)

        for i in range(max_iter):
            Z = Z ** 2 + C
            just_escaped = (~escaped) & (np.abs(Z) > 4.0)
            escaped |= just_escaped
            escape_iter[just_escaped] = i
            trajectories[i] = Z

        for idx in np.where(escaped)[0]:
            n = escape_iter[idx]
            traj = trajectories[:n, idx]
            px = ((traj.real - xmin) / (xmax - xmin) * (w - 1)).astype(int)
            py = ((traj.imag - ymin) / (ymax - ymin) * (h - 1)).astype(int)
            valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
            for ppx, ppy in zip(px[valid], py[valid]):
                image[ppy, ppx] += 1

        progress_sig.emit(int(100 * (b + 1) / n_batches))

    image = np.log1p(image)
    progress_sig.emit(100)
    return image


@np.errstate(over="ignore", invalid="ignore")
def compute_custom_formula(params, progress_sig, is_cancelled):
    """Evaluate a user-provided formula string like 'z**3 + c'."""
    w, h = params["width"], params["height"]
    max_iter = params["max_iter"]
    xmin, xmax = params["xmin"], params["xmax"]
    ymin, ymax = params["ymin"], params["ymax"]
    formula = params.get("custom_formula", "z**2 + c")

    x = np.linspace(xmin, xmax, w)
    y = np.linspace(ymin, ymax, h)
    C = x[np.newaxis, :] + 1j * y[:, np.newaxis]
    Z = np.zeros_like(C, dtype=np.complex128)
    smooth = np.zeros((h, w), dtype=np.float64)
    mask = np.ones((h, w), dtype=bool)

    safe_ns = {
        "z": None, "c": None,
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "exp": np.exp, "log": np.log, "abs": np.abs,
        "sqrt": np.sqrt, "conj": np.conj, "pi": np.pi,
        "e": np.e, "np": np,
    }

    chunk = max(1, max_iter // 20)
    for i in range(max_iter):
        if is_cancelled():
            return None
        safe_ns["z"] = Z[mask]
        safe_ns["c"] = C[mask]
        try:
            Z[mask] = eval(formula, {"__builtins__": {}}, safe_ns)
        except Exception:
            return np.zeros((h, w))
        escaped = mask & (np.abs(Z) > 4.0)
        smooth[escaped] = i + 1
        mask[escaped] = False
        if i % chunk == 0:
            progress_sig.emit(int(100 * i / max_iter))

    smooth[mask] = 0
    progress_sig.emit(100)
    return smooth


# ---------------------------------------------------------------------------
# IFS / L-System / Geometric fractal renderers (return RGBA images directly)
# ---------------------------------------------------------------------------

def render_barnsley_fern(params, progress_sig, is_cancelled):
    """Barnsley fern via iterated function system."""
    w, h = params["width"], params["height"]
    n_points = max(100000, w * h)
    custom_ifs = params.get("custom_ifs", None)

    if custom_ifs:
        transforms = custom_ifs
    else:
        transforms = [
            (0.0, 0.0, 0.0, 0.16, 0.0, 0.0, 0.01),
            (0.85, 0.04, -0.04, 0.85, 0.0, 1.6, 0.85),
            (0.20, -0.26, 0.23, 0.22, 0.0, 1.6, 0.07),
            (-0.15, 0.28, 0.26, 0.24, 0.0, 0.44, 0.07),
        ]

    probs = np.array([t[6] for t in transforms])
    probs /= probs.sum()
    cum_probs = np.cumsum(probs)

    xs = np.zeros(n_points)
    ys = np.zeros(n_points)
    x_val, y_val = 0.0, 0.0

    for i in range(n_points):
        if is_cancelled():
            return None
        r = np.random.random()
        for k, t in enumerate(transforms):
            if r <= cum_probs[k]:
                a, b, c, d, e, f, _ = t
                x_new = a * x_val + b * y_val + e
                y_new = c * x_val + d * y_val + f
                x_val, y_val = x_new, y_new
                break
        xs[i] = x_val
        ys[i] = y_val
        if i % 50000 == 0:
            progress_sig.emit(int(100 * i / n_points))

    image = np.zeros((h, w), dtype=np.float64)
    x_min_v, x_max_v = xs.min(), xs.max()
    y_min_v, y_max_v = ys.min(), ys.max()
    px = ((xs - x_min_v) / (x_max_v - x_min_v + 1e-30) * (w - 1)).astype(int)
    py = ((ys - y_min_v) / (y_max_v - y_min_v + 1e-30) * (h - 1)).astype(int)
    py = h - 1 - py
    valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    np.add.at(image, (py[valid], px[valid]), 1)
    image = np.log1p(image)
    progress_sig.emit(100)
    return image


def render_sierpinski_triangle(params, progress_sig, is_cancelled):
    """Sierpinski triangle via chaos game."""
    w, h = params["width"], params["height"]
    n_points = max(100000, w * h)

    vertices = np.array([[w / 2, 0], [0, h - 1], [w - 1, h - 1]])
    image = np.zeros((h, w), dtype=np.float64)
    pt = np.array([w / 2.0, h / 2.0])

    for i in range(n_points):
        if is_cancelled():
            return None
        v = vertices[np.random.randint(3)]
        pt = (pt + v) / 2.0
        px, py = int(pt[0]), int(pt[1])
        if 0 <= px < w and 0 <= py < h:
            image[py, px] += 1
        if i % 50000 == 0:
            progress_sig.emit(int(100 * i / n_points))

    image = np.log1p(image)
    progress_sig.emit(100)
    return image


def _koch_points(p1, p2, depth):
    """Recursive Koch curve point generation."""
    if depth == 0:
        return [p1]
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    a = (p1[0] + dx / 3, p1[1] + dy / 3)
    b = (p1[0] + 2 * dx / 3, p1[1] + 2 * dy / 3)
    angle = math.pi / 3
    cx = a[0] + (dx / 3) * math.cos(angle) - (dy / 3) * math.sin(angle)
    cy = a[1] + (dx / 3) * math.sin(angle) + (dy / 3) * math.cos(angle)
    c = (cx, cy)
    pts = []
    pts.extend(_koch_points(p1, a, depth - 1))
    pts.extend(_koch_points(a, c, depth - 1))
    pts.extend(_koch_points(c, b, depth - 1))
    pts.extend(_koch_points(b, p2, depth - 1))
    return pts


def render_koch_snowflake(params, progress_sig, is_cancelled):
    """Koch snowflake drawn on a figure canvas."""
    depth = min(params.get("lsys_iterations", 5), 7)
    progress_sig.emit(10)

    s = 0.8
    p1 = (-s / 2, -s / (2 * math.sqrt(3)))
    p2 = (s / 2, -s / (2 * math.sqrt(3)))
    p3 = (0, s * math.sqrt(3) / 2 - s / (2 * math.sqrt(3)))

    pts = _koch_points(p1, p2, depth)
    pts.extend(_koch_points(p2, p3, depth))
    pts.extend(_koch_points(p3, p1, depth))
    pts.append(p1)

    progress_sig.emit(80)
    return ("line_data", pts)


def _apply_lsystem(axiom, rules, iterations):
    """Apply L-system rewriting rules."""
    current = axiom
    for _ in range(iterations):
        next_str = []
        for ch in current:
            next_str.append(rules.get(ch, ch))
        current = "".join(next_str)
    return current


def _lsystem_to_points(string, angle_deg, step=1.0):
    """Convert L-system string to line segments."""
    angle_rad = math.radians(angle_deg)
    x, y, theta = 0.0, 0.0, 0.0
    stack = []
    segments = []
    for ch in string:
        if ch == 'F' or ch == 'G':
            x2 = x + step * math.cos(theta)
            y2 = y + step * math.sin(theta)
            segments.append(((x, y), (x2, y2)))
            x, y = x2, y2
        elif ch == 'f':
            x += step * math.cos(theta)
            y += step * math.sin(theta)
        elif ch == '+':
            theta += angle_rad
        elif ch == '-':
            theta -= angle_rad
        elif ch == '[':
            stack.append((x, y, theta))
        elif ch == ']':
            x, y, theta = stack.pop()
    return segments


def render_dragon_curve(params, progress_sig, is_cancelled):
    """Dragon curve via L-system."""
    iters = min(params.get("lsys_iterations", 12), 18)
    progress_sig.emit(10)
    axiom = "FX"
    rules = {"X": "X+YF+", "Y": "-FX-Y"}
    string = _apply_lsystem(axiom, rules, iters)
    progress_sig.emit(50)
    segments = _lsystem_to_points(string, 90)
    progress_sig.emit(100)
    return ("segments", segments)


def render_hilbert_curve(params, progress_sig, is_cancelled):
    """Hilbert space-filling curve via L-system."""
    iters = min(params.get("lsys_iterations", 6), 9)
    progress_sig.emit(10)
    axiom = "A"
    rules = {"A": "-BF+AFA+FB-", "B": "+AF-BFB-FA+"}
    string = _apply_lsystem(axiom, rules, iters)
    progress_sig.emit(50)
    segments = _lsystem_to_points(string, 90)
    progress_sig.emit(100)
    return ("segments", segments)


def render_custom_lsystem(params, progress_sig, is_cancelled):
    """User-defined L-system."""
    axiom = params.get("lsys_axiom", "F")
    rules_str = params.get("lsys_rules", "F=F+F-F-F+F")
    angle = params.get("lsys_angle", 90)
    iters = min(params.get("lsys_iterations", 4), 10)

    rules = {}
    for part in rules_str.split(","):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            rules[k.strip()] = v.strip()

    progress_sig.emit(10)
    string = _apply_lsystem(axiom, rules, iters)
    progress_sig.emit(50)
    segments = _lsystem_to_points(string, angle)
    progress_sig.emit(100)
    return ("segments", segments)


def render_apollonian_gasket(params, progress_sig, is_cancelled):
    """Apollonian gasket via recursive circle packing."""
    depth = min(params.get("lsys_iterations", 5), 8)
    circles = []

    def _inscribed(c1, c2, c3, depth_left):
        if depth_left <= 0:
            return
        x1, y1, r1 = c1
        x2, y2, r2 = c2
        x3, y3, r3 = c3
        k1, k2, k3 = 1.0 / r1, 1.0 / r2, 1.0 / r3
        k4 = k1 + k2 + k3 + 2.0 * math.sqrt(abs(k1 * k2 + k2 * k3 + k1 * k3))
        if k4 < 1e-10:
            return
        r4 = 1.0 / k4
        if r4 < 0.001:
            return
        z1 = complex(x1, y1) * k1
        z2 = complex(x2, y2) * k2
        z3 = complex(x3, y3) * k3
        z4_sum = z1 + z2 + z3
        disc = z1 * z2 + z2 * z3 + z1 * z3
        try:
            sq = 2.0 * (disc ** 0.5)
        except Exception:
            return
        for sign in [1, -1]:
            z4 = (z4_sum + sign * sq) / k4 if k4 != 0 else 0
            x4, y4 = z4.real, z4.imag
            new_circle = (x4, y4, r4)
            circles.append(new_circle)
            _inscribed(c1, c2, new_circle, depth_left - 1)
            _inscribed(c1, new_circle, c3, depth_left - 1)
            _inscribed(new_circle, c2, c3, depth_left - 1)

    big_r = 1.0
    r_inner = big_r / (1.0 + 2.0 / math.sqrt(3))
    c_outer = (0, 0, big_r)
    c1 = (0, big_r - r_inner, r_inner)
    angle2 = 2 * math.pi / 3
    angle3 = 4 * math.pi / 3
    c2 = (r_inner * math.sin(angle2) * 0, big_r - r_inner, r_inner)

    small_r = big_r * (2.0 / 3.0)
    circles = [
        (0, 0, big_r),
        (0, big_r - small_r, small_r),
        (-small_r * math.cos(math.pi / 6), -small_r * math.sin(math.pi / 6) + big_r - 2 * small_r, small_r),
        (small_r * math.cos(math.pi / 6), -small_r * math.sin(math.pi / 6) + big_r - 2 * small_r, small_r),
    ]

    progress_sig.emit(100)
    return ("circles", circles)


def render_cantor_set(params, progress_sig, is_cancelled):
    """Cantor set -- recursive middle-third removal."""
    depth = min(params.get("lsys_iterations", 6), 12)
    segments = []

    def _cantor(x0, x1, level, y):
        if level > depth:
            return
        segments.append((x0, x1, y))
        third = (x1 - x0) / 3.0
        _cantor(x0, x0 + third, level + 1, y - 1)
        _cantor(x1 - third, x1, level + 1, y - 1)

    _cantor(0.0, 1.0, 0, 0)
    progress_sig.emit(100)
    return ("cantor", segments)


def compute_logistic_map(params, progress_sig, is_cancelled):
    """Logistic map bifurcation diagram."""
    w, h = params["width"], params["height"]
    r_min = max(0.0, min(params.get("xmin", 2.5), 3.99))
    r_max = min(4.0, max(params.get("xmax", 4.0), r_min + 0.01))
    n_r = w
    n_iter = 1000
    n_last = 300

    image = np.zeros((h, w), dtype=np.float64)
    r_vals = np.linspace(r_min, r_max, n_r)

    for idx, r in enumerate(r_vals):
        if is_cancelled():
            return None
        x = 0.5
        for _ in range(n_iter - n_last):
            x = r * x * (1.0 - x)
            if not np.isfinite(x) or x > 1e10 or x < -1e10:
                x = 0.5
                break
        for _ in range(n_last):
            x = r * x * (1.0 - x)
            if not np.isfinite(x) or x > 1e10 or x < -1e10:
                break
            if 0.0 <= x <= 1.0:
                py = int((1.0 - x) * (h - 1))
                if 0 <= py < h:
                    image[py, idx] += 1
        if idx % (n_r // 20 + 1) == 0:
            progress_sig.emit(int(100 * idx / n_r))

    image = np.log1p(image)
    progress_sig.emit(100)
    return image


def compute_feigenbaum(params, progress_sig, is_cancelled):
    """Feigenbaum / logistic map bifurcation (zoomed into period-doubling region)."""
    params_copy = dict(params)
    params_copy["xmin"] = params.get("xmin", 3.4)
    params_copy["xmax"] = params.get("xmax", 3.6)
    return compute_logistic_map(params_copy, progress_sig, is_cancelled)


# ---------------------------------------------------------------------------
# 3D Fractal renderers
# ---------------------------------------------------------------------------

@np.errstate(over="ignore", invalid="ignore")
def compute_mandelbulb(params, progress_sig, is_cancelled):
    """Mandelbulb rendered as a 2D distance-estimation cross-section."""
    w, h = params["width"], params["height"]
    max_iter = min(params["max_iter"], 200)
    power = params.get("power", 8)

    xmin, xmax = params.get("xmin", -1.5), params.get("xmax", 1.5)
    ymin, ymax = params.get("ymin", -1.5), params.get("ymax", 1.5)
    z_slice = params.get("z_slice", 0.0)

    x = np.linspace(xmin, xmax, w)
    y = np.linspace(ymin, ymax, h)
    X, Y = np.meshgrid(x, y)
    Z_val = np.full_like(X, z_slice)

    cx, cy, cz = X.copy(), Y.copy(), Z_val.copy()
    zx, zy, zz = np.zeros_like(X), np.zeros_like(X), np.zeros_like(X)
    escape = np.full((h, w), max_iter, dtype=np.float64)
    mask = np.ones((h, w), dtype=bool)

    chunk = max(1, max_iter // 10)
    for i in range(max_iter):
        if is_cancelled():
            return None
        r = np.sqrt(zx[mask] ** 2 + zy[mask] ** 2 + zz[mask] ** 2)
        theta = np.arctan2(np.sqrt(zx[mask] ** 2 + zy[mask] ** 2), zz[mask])
        phi = np.arctan2(zy[mask], zx[mask])
        rn = r ** power
        zx[mask] = rn * np.sin(theta * power) * np.cos(phi * power) + cx[mask]
        zy[mask] = rn * np.sin(theta * power) * np.sin(phi * power) + cy[mask]
        zz[mask] = rn * np.cos(theta * power) + cz[mask]
        mag = zx ** 2 + zy ** 2 + zz ** 2
        escaped = mask & (mag > 4.0)
        escape[escaped] = i
        mask[escaped] = False
        if i % chunk == 0:
            progress_sig.emit(int(100 * i / max_iter))

    progress_sig.emit(100)
    return escape


@np.errstate(over="ignore", invalid="ignore")
def compute_quaternion_julia(params, progress_sig, is_cancelled):
    """Quaternion Julia set -- 2D slice as height map."""
    w, h = params["width"], params["height"]
    max_iter = min(params["max_iter"], 200)
    xmin, xmax = params.get("xmin", -1.5), params.get("xmax", 1.5)
    ymin, ymax = params.get("ymin", -1.5), params.get("ymax", 1.5)
    cr, ci, cj, ck = -0.2, 0.8, 0.0, 0.0

    x = np.linspace(xmin, xmax, w)
    y = np.linspace(ymin, ymax, h)
    X, Y = np.meshgrid(x, y)

    qr, qi, qj, qk = X.copy(), Y.copy(), np.zeros_like(X), np.zeros_like(X)
    escape = np.full((h, w), max_iter, dtype=np.float64)
    mask = np.ones((h, w), dtype=bool)

    chunk = max(1, max_iter // 10)
    for it in range(max_iter):
        if is_cancelled():
            return None
        r, i_v, j, k = qr[mask], qi[mask], qj[mask], qk[mask]
        nr = r * r - i_v * i_v - j * j - k * k + cr
        ni = 2 * r * i_v + ci
        nj = 2 * r * j + cj
        nk = 2 * r * k + ck
        qr[mask], qi[mask], qj[mask], qk[mask] = nr, ni, nj, nk
        mag = qr ** 2 + qi ** 2 + qj ** 2 + qk ** 2
        escaped = mask & (mag > 4.0)
        escape[escaped] = it
        mask[escaped] = False
        if it % chunk == 0:
            progress_sig.emit(int(100 * it / max_iter))

    progress_sig.emit(100)
    return escape


def render_menger_sponge(params, progress_sig, is_cancelled):
    """Menger sponge as a set of cube coordinates for 3D display."""
    depth = min(params.get("lsys_iterations", 3), 4)
    cubes = [(0, 0, 0, 1)]

    for d in range(depth):
        if is_cancelled():
            return None
        new_cubes = []
        for (cx, cy, cz, size) in cubes:
            s = size / 3.0
            for ix in range(3):
                for iy in range(3):
                    for iz in range(3):
                        zeroes = (ix == 1) + (iy == 1) + (iz == 1)
                        if zeroes >= 2:
                            continue
                        new_cubes.append((cx + ix * s, cy + iy * s, cz + iz * s, s))
            progress_sig.emit(int(100 * (d + 1) / depth))
        cubes = new_cubes

    progress_sig.emit(100)
    return ("menger", cubes)


def render_sierpinski_tetrahedron(params, progress_sig, is_cancelled):
    """Sierpinski tetrahedron as a set of vertex lists for 3D display."""
    depth = min(params.get("lsys_iterations", 4), 6)

    v0 = np.array([1, 1, 1], dtype=float)
    v1 = np.array([1, -1, -1], dtype=float)
    v2 = np.array([-1, 1, -1], dtype=float)
    v3 = np.array([-1, -1, 1], dtype=float)

    tetrahedra = [(v0, v1, v2, v3)]

    for d in range(depth):
        if is_cancelled():
            return None
        new_t = []
        for (a, b, c, dd_) in tetrahedra:
            ab = (a + b) / 2
            ac = (a + c) / 2
            ad = (a + dd_) / 2
            bc = (b + c) / 2
            bd = (b + dd_) / 2
            cd = (c + dd_) / 2
            new_t.append((a, ab, ac, ad))
            new_t.append((b, ab, bc, bd))
            new_t.append((c, ac, bc, cd))
            new_t.append((dd_, ad, bd, cd))
        tetrahedra = new_t
        progress_sig.emit(int(100 * (d + 1) / depth))

    progress_sig.emit(100)
    return ("tetrahedron", tetrahedra)


@np.errstate(over="ignore", invalid="ignore")
def compute_custom_newton(params, progress_sig, is_cancelled):
    """Newton fractal for a user-defined polynomial."""
    w, h = params["width"], params["height"]
    max_iter = params["max_iter"]
    xmin, xmax = params["xmin"], params["xmax"]
    ymin, ymax = params["ymin"], params["ymax"]
    coeffs_str = params.get("newton_coeffs", "1,0,0,-1")
    tol = 1e-6

    coeffs = [float(c.strip()) for c in coeffs_str.split(",")]
    poly = np.array(coeffs)
    dpoly = np.polyder(poly)

    x = np.linspace(xmin, xmax, w)
    y = np.linspace(ymin, ymax, h)
    Z = x[np.newaxis, :] + 1j * y[:, np.newaxis]
    result = np.zeros((h, w), dtype=np.float64)
    shade = np.zeros((h, w), dtype=np.float64)
    mask = np.ones((h, w), dtype=bool)

    chunk = max(1, max_iter // 10)
    for i in range(max_iter):
        if is_cancelled():
            return None
        fz = np.polyval(poly, Z[mask])
        dfz = np.polyval(dpoly, Z[mask])
        dfz = np.where(np.abs(dfz) < 1e-30, 1e-30, dfz)
        Z[mask] = Z[mask] - fz / dfz
        converged = mask & (np.abs(np.polyval(poly, Z)) < tol)
        angle = np.angle(Z[converged])
        result[converged] = angle
        shade[converged] = i
        mask[converged] = False
        if i % chunk == 0:
            progress_sig.emit(int(100 * i / max_iter))

    combined = result * 30.0 + shade
    progress_sig.emit(100)
    return combined


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

FRACTAL_DISPATCH = {
    "Mandelbrot": compute_mandelbrot,
    "Julia Set": compute_julia,
    "Burning Ship": compute_burning_ship,
    "Multibrot": compute_multibrot,
    "Tricorn": compute_tricorn,
    "Newton Fractal": compute_newton,
    "Phoenix": compute_phoenix,
    "Magnet": compute_magnet,
    "Lyapunov": compute_lyapunov,
    "Buddhabrot": compute_buddhabrot,
    "Barnsley Fern": render_barnsley_fern,
    "Sierpinski Triangle": render_sierpinski_triangle,
    "Koch Snowflake": render_koch_snowflake,
    "Dragon Curve": render_dragon_curve,
    "Hilbert Curve": render_hilbert_curve,
    "Apollonian Gasket": render_apollonian_gasket,
    "Cantor Set": render_cantor_set,
    "Logistic Map": compute_logistic_map,
    "Feigenbaum": compute_feigenbaum,
    "Mandelbulb": compute_mandelbulb,
    "Quaternion Julia": compute_quaternion_julia,
    "Menger Sponge": render_menger_sponge,
    "Sierpinski Tetrahedron": render_sierpinski_tetrahedron,
    "Custom Formula": compute_custom_formula,
    "Custom IFS": render_barnsley_fern,
    "Custom L-System": render_custom_lsystem,
    "Custom Newton": compute_custom_newton,
}

DEFAULT_BOUNDS = {
    "Mandelbrot": (-2.5, 1.0, -1.25, 1.25),
    "Julia Set": (-2.0, 2.0, -1.5, 1.5),
    "Burning Ship": (-2.5, 1.5, -2.0, 1.0),
    "Multibrot": (-2.0, 2.0, -2.0, 2.0),
    "Tricorn": (-2.5, 1.0, -1.25, 1.25),
    "Newton Fractal": (-3.0, 3.0, -3.0, 3.0),
    "Phoenix": (-2.0, 2.0, -1.5, 1.5),
    "Magnet": (-2.0, 4.0, -3.0, 3.0),
    "Lyapunov": (2.0, 4.0, 2.0, 4.0),
    "Buddhabrot": (-2.5, 1.0, -1.25, 1.25),
    "Logistic Map": (2.5, 4.0, 0.0, 1.0),
    "Feigenbaum": (3.4, 3.6, 0.0, 1.0),
    "Mandelbulb": (-1.5, 1.5, -1.5, 1.5),
    "Quaternion Julia": (-1.5, 1.5, -1.5, 1.5),
}


# ---------------------------------------------------------------------------
# Main Widget
# ---------------------------------------------------------------------------

class FractalExplorerWidget(QWidget):
    """
    Comprehensive fractal exploration widget for the Axiom Scientific Suite.

    Supports 23+ fractal types with interactive zoom/pan, progressive rendering,
    smooth coloring, parameter controls, and high-resolution export.

    Public API:
        set_logger(fn)  -- set a logging callback
        run()            -- trigger a render
        export()         -- export the current fractal image
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._logger = None
        self._current_fractal = "Mandelbrot"
        self._render_thread = None
        self._render_data = None
        self._zoom_history = deque(maxlen=100)
        self._zoom_fwd_stack = deque(maxlen=100)
        self._rubber_band_origin = None
        self._rubber_band_rect = None
        self._dragging = False
        self._drag_start = None
        self._drag_moved = False
        self._last_render_time = 0.0

        # Current view bounds
        self._xmin, self._xmax = -2.5, 1.0
        self._ymin, self._ymax = -1.25, 1.25

        self._init_ui()
        QTimer.singleShot(100, self.run)

    # ----- Logging -----

    def set_logger(self, fn):
        """Set a callable for logging messages."""
        self._logger = fn

    def _log(self, msg):
        if self._logger:
            try:
                self._logger(msg)
            except Exception:
                pass

    # ----- UI Construction -----

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)

        # --- Left Panel ---
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(280)
        left_scroll.setMaximumWidth(400)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(4, 4, 4, 4)

        # Fractal type tree
        type_group = QGroupBox("Fractal Type")
        type_layout = QVBoxLayout()
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setMaximumHeight(320)
        for cat, items in FRACTAL_CATEGORIES.items():
            cat_item = QTreeWidgetItem([cat])
            cat_item.setFlags(cat_item.flags() & ~Qt.ItemIsSelectable)
            font = cat_item.font(0)
            font.setBold(True)
            cat_item.setFont(0, font)
            for name in items:
                child = QTreeWidgetItem([name])
                cat_item.addChild(child)
            self._tree.addTopLevelItem(cat_item)
        self._tree.expandAll()
        self._tree.currentItemChanged.connect(self._on_fractal_type_changed)
        type_layout.addWidget(self._tree)
        type_group.setLayout(type_layout)
        left_layout.addWidget(type_group)

        # Parameters group
        params_group = QGroupBox("Parameters")
        params_layout = QFormLayout()

        self._iter_slider = QSlider(Qt.Horizontal)
        self._iter_slider.setRange(50, 10000)
        self._iter_slider.setValue(500)
        self._iter_slider.setTickInterval(500)
        self._iter_label = QLabel("500")
        self._iter_slider.valueChanged.connect(lambda v: self._iter_label.setText(str(v)))
        iter_row = QHBoxLayout()
        iter_row.addWidget(self._iter_slider)
        iter_row.addWidget(self._iter_label)
        params_layout.addRow("Max Iterations:", iter_row)

        self._res_combo = QComboBox()
        for res in ["256x256", "512x512", "768x768", "1024x1024", "1536x1536",
                     "2048x2048", "3072x3072", "4096x4096"]:
            self._res_combo.addItem(res)
        self._res_combo.setCurrentIndex(1)  # Default 512x512 for fast interactive
        params_layout.addRow("Resolution:", self._res_combo)

        self._cmap_combo = QComboBox()
        for cname in COLORMAPS:
            self._cmap_combo.addItem(cname)
        self._cmap_combo.setCurrentIndex(0)
        params_layout.addRow("Colormap:", self._cmap_combo)

        self._supersample_combo = QComboBox()
        self._supersample_combo.addItems(["None", "2x2", "3x3"])
        params_layout.addRow("Supersampling:", self._supersample_combo)

        self._interior_combo = QComboBox()
        self._interior_combo.addItems(["Black", "Orbit Trap"])
        params_layout.addRow("Interior:", self._interior_combo)

        params_group.setLayout(params_layout)
        left_layout.addWidget(params_group)

        # Fractal-specific parameters
        specific_group = QGroupBox("Fractal Parameters")
        specific_layout = QFormLayout()

        self._julia_cr = QDoubleSpinBox()
        self._julia_cr.setRange(-3.0, 3.0)
        self._julia_cr.setDecimals(6)
        self._julia_cr.setSingleStep(0.01)
        self._julia_cr.setValue(-0.123)
        specific_layout.addRow("Julia c (real):", self._julia_cr)

        self._julia_ci = QDoubleSpinBox()
        self._julia_ci.setRange(-3.0, 3.0)
        self._julia_ci.setDecimals(6)
        self._julia_ci.setSingleStep(0.01)
        self._julia_ci.setValue(0.745)
        specific_layout.addRow("Julia c (imag):", self._julia_ci)

        self._julia_preset_combo = QComboBox()
        self._julia_preset_combo.addItem("-- Select Preset --")
        for name in JULIA_PRESETS:
            self._julia_preset_combo.addItem(name)
        self._julia_preset_combo.currentTextChanged.connect(self._on_julia_preset_changed)
        specific_layout.addRow("Julia Preset:", self._julia_preset_combo)

        self._power_spin = QSpinBox()
        self._power_spin.setRange(2, 8)
        self._power_spin.setValue(3)
        specific_layout.addRow("Power (d):", self._power_spin)

        self._newton_power_spin = QSpinBox()
        self._newton_power_spin.setRange(3, 8)
        self._newton_power_spin.setValue(3)
        specific_layout.addRow("Newton z^n:", self._newton_power_spin)

        self._phoenix_cr_spin = QDoubleSpinBox()
        self._phoenix_cr_spin.setRange(-3.0, 3.0)
        self._phoenix_cr_spin.setDecimals(4)
        self._phoenix_cr_spin.setValue(0.5667)
        specific_layout.addRow("Phoenix Re(c):", self._phoenix_cr_spin)

        self._phoenix_ci_spin = QDoubleSpinBox()
        self._phoenix_ci_spin.setRange(-3.0, 3.0)
        self._phoenix_ci_spin.setDecimals(4)
        self._phoenix_ci_spin.setValue(-0.5)
        specific_layout.addRow("Phoenix Im(c):", self._phoenix_ci_spin)

        self._lyapunov_seq = QLineEdit("AB")
        specific_layout.addRow("Lyapunov Seq:", self._lyapunov_seq)

        self._lsys_iters_spin = QSpinBox()
        self._lsys_iters_spin.setRange(1, 18)
        self._lsys_iters_spin.setValue(5)
        specific_layout.addRow("L-Sys Iters:", self._lsys_iters_spin)

        specific_group.setLayout(specific_layout)
        left_layout.addWidget(specific_group)

        # Custom editors in a tab widget
        custom_group = QGroupBox("Custom Editors")
        custom_layout = QVBoxLayout()
        self._custom_tabs = QTabWidget()
        self._custom_tabs.setMaximumHeight(200)

        # Custom formula tab
        formula_tab = QWidget()
        fl = QVBoxLayout(formula_tab)
        fl.addWidget(QLabel("Formula (use z and c):"))
        self._formula_edit = QLineEdit("z**2 + c")
        fl.addWidget(self._formula_edit)
        self._custom_tabs.addTab(formula_tab, "Formula")

        # Custom IFS tab
        ifs_tab = QWidget()
        il = QVBoxLayout(ifs_tab)
        il.addWidget(QLabel("IFS transforms (a,b,c,d,e,f,prob per line):"))
        self._ifs_edit = QTextEdit()
        self._ifs_edit.setPlainText(
            "0.0, 0.0, 0.0, 0.16, 0.0, 0.0, 0.01\n"
            "0.85, 0.04, -0.04, 0.85, 0.0, 1.6, 0.85\n"
            "0.20, -0.26, 0.23, 0.22, 0.0, 1.6, 0.07\n"
            "-0.15, 0.28, 0.26, 0.24, 0.0, 0.44, 0.07"
        )
        self._ifs_edit.setMaximumHeight(100)
        il.addWidget(self._ifs_edit)
        self._custom_tabs.addTab(ifs_tab, "IFS")

        # Custom L-System tab
        lsys_tab = QWidget()
        ll = QVBoxLayout(lsys_tab)
        ll.addWidget(QLabel("Axiom:"))
        self._lsys_axiom_edit = QLineEdit("F")
        ll.addWidget(self._lsys_axiom_edit)
        ll.addWidget(QLabel("Rules (comma-sep, e.g. F=F+F-F):"))
        self._lsys_rules_edit = QLineEdit("F=F+F-F-F+F")
        ll.addWidget(self._lsys_rules_edit)
        ll.addWidget(QLabel("Angle (degrees):"))
        self._lsys_angle_spin = QDoubleSpinBox()
        self._lsys_angle_spin.setRange(1, 360)
        self._lsys_angle_spin.setValue(90)
        ll.addWidget(self._lsys_angle_spin)
        self._custom_tabs.addTab(lsys_tab, "L-System")

        # Custom Newton tab
        newton_tab = QWidget()
        nl = QVBoxLayout(newton_tab)
        nl.addWidget(QLabel("Polynomial coefficients (high to low):"))
        self._newton_coeffs_edit = QLineEdit("1,0,0,-1")
        nl.addWidget(self._newton_coeffs_edit)
        nl.addWidget(QLabel("e.g. '1,0,0,-1' = z^3 - 1"))
        self._custom_tabs.addTab(newton_tab, "Newton")

        custom_layout.addWidget(self._custom_tabs)
        custom_group.setLayout(custom_layout)
        left_layout.addWidget(custom_group)

        # Action buttons
        btn_layout = QGridLayout()
        self._render_btn = QPushButton("Render")
        self._render_btn.clicked.connect(self.run)
        btn_layout.addWidget(self._render_btn, 0, 0)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_render)
        btn_layout.addWidget(self._stop_btn, 0, 1)

        self._reset_btn = QPushButton("Reset Zoom")
        self._reset_btn.clicked.connect(self._reset_zoom)
        btn_layout.addWidget(self._reset_btn, 1, 0)

        self._back_btn = QPushButton("< Back")
        self._back_btn.clicked.connect(self._zoom_back)
        btn_layout.addWidget(self._back_btn, 1, 1)

        self._fwd_btn = QPushButton("Forward >")
        self._fwd_btn.clicked.connect(self._nav_forward)
        btn_layout.addWidget(self._fwd_btn, 2, 0)

        self._export_btn = QPushButton("Export Image")
        self._export_btn.clicked.connect(self.export)
        btn_layout.addWidget(self._export_btn, 2, 1)

        self._export_params_btn = QPushButton("Save Params")
        self._export_params_btn.clicked.connect(self._save_params)
        btn_layout.addWidget(self._export_params_btn, 3, 0)

        self._load_params_btn = QPushButton("Load Params")
        self._load_params_btn.clicked.connect(self._load_params)
        btn_layout.addWidget(self._load_params_btn, 3, 1)

        self._export_anim_btn = QPushButton("Export Animation")
        self._export_anim_btn.clicked.connect(self._export_animation)
        btn_layout.addWidget(self._export_anim_btn, 4, 0, 1, 2)

        self._clipboard_btn = QPushButton("Copy Plot")
        self._clipboard_btn.setToolTip("Copy current plot to clipboard as image")
        self._clipboard_btn.clicked.connect(self._copy_plot_to_clipboard)
        btn_layout.addWidget(self._clipboard_btn, 5, 0, 1, 2)

        left_layout.addLayout(btn_layout)
        left_layout.addStretch()

        left_scroll.setWidget(left_widget)
        splitter.addWidget(left_scroll)

        # --- Center Panel (canvas) ---
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)

        self._figure = Figure(figsize=(8, 8), dpi=100)
        style_figure(self._figure)
        self._figure.set_facecolor("#1a1a2e")
        self._ax = self._figure.add_subplot(111)
        self._ax.set_facecolor("#1a1a2e")
        self._ax.set_xticks([])
        self._ax.set_yticks([])

        self._canvas = FigureCanvas(self._figure)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._canvas.mpl_connect("button_press_event", self._on_canvas_press)
        self._canvas.mpl_connect("button_release_event", self._on_canvas_release)
        self._canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)
        self._canvas.mpl_connect("scroll_event", self._on_canvas_scroll)

        self._toolbar = NavigationToolbar(self._canvas, self)
        center_layout.addWidget(self._toolbar)
        center_layout.addWidget(self._canvas)

        # Bottom status bar
        status_frame = QFrame()
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(4, 2, 4, 2)

        self._coord_label = QLabel("Cursor: (0, 0)")
        self._coord_label.setMinimumWidth(250)
        status_layout.addWidget(self._coord_label)

        self._zoom_label = QLabel("Zoom: 1x")
        self._zoom_label.setMinimumWidth(150)
        status_layout.addWidget(self._zoom_label)

        self._time_label = QLabel("Render: --")
        self._time_label.setMinimumWidth(120)
        status_layout.addWidget(self._time_label)

        self._iter_info_label = QLabel("Iters: --")
        status_layout.addWidget(self._iter_info_label)

        status_layout.addStretch()

        self._progress = QProgressBar()
        self._progress.setMaximumWidth(200)
        self._progress.setValue(0)
        status_layout.addWidget(self._progress)

        center_layout.addWidget(status_frame)
        splitter.addWidget(center_widget)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 800])
        main_layout.addWidget(splitter)

    # ----- Event handlers -----

    def _on_fractal_type_changed(self, current, previous):
        if current is None or current.childCount() > 0:
            return
        name = current.text(0)
        if name in FRACTAL_DISPATCH:
            self._current_fractal = name
            bounds = DEFAULT_BOUNDS.get(name, (-2.0, 2.0, -2.0, 2.0))
            self._xmin, self._xmax, self._ymin, self._ymax = bounds
            self._zoom_history.clear()
            self._zoom_fwd_stack.clear()
            self._log(f"Selected fractal: {name}")

    def _on_julia_preset_changed(self, text):
        if text in JULIA_PRESETS:
            cr, ci = JULIA_PRESETS[text]
            self._julia_cr.setValue(cr)
            self._julia_ci.setValue(ci)

    def _on_canvas_press(self, event):
        if event.inaxes != self._ax:
            return
        if event.button == 1 and event.dblclick:
            self._double_click_zoom(event)
            return
        if event.button in (1, 3):
            # Both left and right click start a drag/pan
            self._dragging = True
            self._drag_start = (event.xdata, event.ydata)
            self._drag_moved = False

    def _on_canvas_release(self, event):
        if not self._dragging:
            return

        if self._drag_moved and self._drag_start and event.xdata is not None:
            # Completed a drag — apply pan
            w, h = self._get_resolution()
            dx_px = event.xdata - self._drag_start[0]
            dy_px = event.ydata - self._drag_start[1]
            xrange = self._xmax - self._xmin
            yrange = self._ymax - self._ymin
            self._push_zoom()
            self._xmin -= dx_px / w * xrange
            self._xmax -= dx_px / w * xrange
            self._ymin -= dy_px / h * yrange
            self._ymax -= dy_px / h * yrange
            self.run()

        self._dragging = False
        self._drag_start = None
        self._drag_moved = False

    def _on_canvas_motion(self, event):
        if event.inaxes != self._ax or event.xdata is None:
            return
        # Update coordinate display
        w, h = self._get_resolution()
        frac_x = event.xdata / w if w else 0
        frac_y = event.ydata / h if h else 0
        real = self._xmin + frac_x * (self._xmax - self._xmin)
        imag = self._ymin + frac_y * (self._ymax - self._ymin)
        self._coord_label.setText(f"Cursor: ({real:.10f}, {imag:.10f}i)")

        # Track if we're dragging
        if self._dragging and self._drag_start:
            dx = abs(event.xdata - self._drag_start[0])
            dy = abs(event.ydata - self._drag_start[1])
            if dx > 3 or dy > 3:
                self._drag_moved = True

    def _on_canvas_scroll(self, event):
        if event.inaxes != self._ax or event.xdata is None:
            return
        w, h = self._get_resolution()
        frac_x = event.xdata / w if w else 0.5
        frac_y = event.ydata / h if h else 0.5
        center_r = self._xmin + frac_x * (self._xmax - self._xmin)
        center_i = self._ymin + frac_y * (self._ymax - self._ymin)

        factor = 0.7 if event.button == "up" else 1.0 / 0.7
        self._push_zoom()
        xrange = (self._xmax - self._xmin) * factor
        yrange = (self._ymax - self._ymin) * factor
        self._xmin = center_r - frac_x * xrange
        self._xmax = center_r + (1 - frac_x) * xrange
        self._ymin = center_i - frac_y * yrange
        self._ymax = center_i + (1 - frac_y) * yrange
        self._update_zoom_label()
        self.run()

    def _double_click_zoom(self, event):
        """Center and zoom 2x at double-click location."""
        if event.xdata is None:
            return
        w, h = self._get_resolution()
        frac_x = event.xdata / w if w else 0.5
        frac_y = event.ydata / h if h else 0.5
        center_r = self._xmin + frac_x * (self._xmax - self._xmin)
        center_i = self._ymin + frac_y * (self._ymax - self._ymin)

        self._push_zoom()
        xrange = (self._xmax - self._xmin) * 0.5
        yrange = (self._ymax - self._ymin) * 0.5
        self._xmin = center_r - xrange / 2
        self._xmax = center_r + xrange / 2
        self._ymin = center_i - yrange / 2
        self._ymax = center_i + yrange / 2
        self.run()

    # ----- Zoom history -----

    def _push_zoom(self):
        self._zoom_history.append((self._xmin, self._xmax, self._ymin, self._ymax))
        self._zoom_fwd_stack.clear()

    def _zoom_back(self):
        if not self._zoom_history:
            return
        self._zoom_fwd_stack.append((self._xmin, self._xmax, self._ymin, self._ymax))
        self._xmin, self._xmax, self._ymin, self._ymax = self._zoom_history.pop()
        self.run()

    def _nav_forward(self):
        if not self._zoom_fwd_stack:
            return
        self._zoom_history.append((self._xmin, self._xmax, self._ymin, self._ymax))
        self._xmin, self._xmax, self._ymin, self._ymax = self._zoom_fwd_stack.pop()
        self.run()

    def _reset_zoom(self):
        bounds = DEFAULT_BOUNDS.get(self._current_fractal, (-2.0, 2.0, -2.0, 2.0))
        self._push_zoom()
        self._xmin, self._xmax, self._ymin, self._ymax = bounds
        self.run()

    def _update_zoom_label(self):
        default = DEFAULT_BOUNDS.get(self._current_fractal, (-2.0, 2.0, -2.0, 2.0))
        default_range = default[1] - default[0]
        current_range = self._xmax - self._xmin
        if current_range > 0:
            mag = default_range / current_range
            if mag >= 1e6:
                exp = math.log10(mag)
                self._zoom_label.setText(f"Zoom: 10^{exp:.1f}x")
            else:
                self._zoom_label.setText(f"Zoom: {mag:.1f}x")
        else:
            self._zoom_label.setText("Zoom: --")

    # ----- Resolution helper -----

    def _get_resolution(self):
        text = self._res_combo.currentText()
        parts = text.split("x")
        return int(parts[0]), int(parts[1])

    # ----- Build params dict -----

    def _build_params(self, width=None, height=None):
        if width is None:
            width, height = self._get_resolution()
        ss_map = {"None": 1, "2x2": 2, "3x3": 3}
        ss = ss_map.get(self._supersample_combo.currentText(), 1)

        params = {
            "width": width,
            "height": height,
            "max_iter": self._iter_slider.value(),
            "xmin": self._xmin,
            "xmax": self._xmax,
            "ymin": self._ymin,
            "ymax": self._ymax,
            "supersample": ss,
            "julia_cr": self._julia_cr.value(),
            "julia_ci": self._julia_ci.value(),
            "power": self._power_spin.value(),
            "newton_power": self._newton_power_spin.value(),
            "phoenix_cr": self._phoenix_cr_spin.value(),
            "phoenix_ci": self._phoenix_ci_spin.value(),
            "lyapunov_seq": self._lyapunov_seq.text(),
            "lsys_iterations": self._lsys_iters_spin.value(),
            "custom_formula": self._formula_edit.text(),
            "lsys_axiom": self._lsys_axiom_edit.text(),
            "lsys_rules": self._lsys_rules_edit.text(),
            "lsys_angle": self._lsys_angle_spin.value(),
            "newton_coeffs": self._newton_coeffs_edit.text(),
            "colormap": self._cmap_combo.currentText(),
            "interior": self._interior_combo.currentText(),
        }

        # Parse custom IFS
        if self._current_fractal == "Custom IFS":
            try:
                lines = self._ifs_edit.toPlainText().strip().split("\n")
                transforms = []
                for line in lines:
                    vals = [float(x.strip()) for x in line.split(",")]
                    if len(vals) == 7:
                        transforms.append(tuple(vals))
                if transforms:
                    params["custom_ifs"] = transforms
            except Exception:
                pass

        return params

    # ----- Rendering -----

    def run(self):
        """Trigger fractal rendering (public API)."""
        if self._render_thread and self._render_thread.isRunning():
            self._render_thread.cancel()
            self._render_thread.wait(2000)

        name = self._current_fractal
        func = FRACTAL_DISPATCH.get(name)
        if func is None:
            self._log(f"Unknown fractal type: {name}")
            return

        params = self._build_params()
        self._progress.setValue(0)
        self._render_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._iter_info_label.setText(f"Iters: {params['max_iter']}")
        self._update_zoom_label()
        self._log(f"Rendering {name} at {params['width']}x{params['height']}...")

        # Progressive rendering: do a quick low-res preview first
        preview_w = min(256, params["width"])
        preview_h = min(256, params["height"])
        preview_params = dict(params)
        preview_params["width"] = preview_w
        preview_params["height"] = preview_h
        preview_params["max_iter"] = min(params["max_iter"], 100)
        preview_params["supersample"] = 1
        try:
            preview_data = func(preview_params, _DummySignal(), lambda: False)
            if preview_data is not None:
                self._display_result(preview_data, is_preview=True)
        except Exception:
            pass

        # Full render in background
        self._render_thread = FractalRenderThread(func, params, self)
        self._render_thread.progress.connect(self._on_render_progress)
        self._render_thread.finished.connect(self._on_render_finished)
        self._render_thread.start()

    def _stop_render(self):
        if self._render_thread and self._render_thread.isRunning():
            self._render_thread.cancel()
            self._log("Render cancelled.")
        self._render_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    def _on_render_progress(self, value):
        self._progress.setValue(value)

    def _on_render_finished(self, result, elapsed):
        self._render_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._last_render_time = elapsed
        self._time_label.setText(f"Render: {elapsed:.2f}s")
        self._progress.setValue(100)

        if result is None:
            self._log("Render returned no data.")
            return

        self._render_data = result
        self._display_result(result)
        self._log(f"Render complete in {elapsed:.2f}s")

    def _display_result(self, result, is_preview=False):
        """Display the rendered fractal on the matplotlib canvas."""
        self._ax.clear()
        self._ax.set_xticks([])
        self._ax.set_yticks([])

        cmap_name = self._cmap_combo.currentText()
        try:
            cmap = plt.get_cmap(cmap_name)
        except ValueError:
            cmap = plt.get_cmap("inferno")

        if isinstance(result, tuple):
            kind = result[0]
            data = result[1]
            if kind == "line_data":
                xs = [p[0] for p in data]
                ys = [p[1] for p in data]
                self._ax.plot(xs, ys, color="#00ff88", linewidth=0.5)
                self._ax.set_aspect("equal")
                self._ax.set_facecolor("#1a1a2e")
            elif kind == "segments":
                for (x1, y1), (x2, y2) in data:
                    self._ax.plot([x1, x2], [y1, y2], color="#00ff88", linewidth=0.3)
                self._ax.set_aspect("equal")
                self._ax.set_facecolor("#1a1a2e")
            elif kind == "circles":
                for (cx, cy, r) in data:
                    circle = mpatches.Circle((cx, cy), r, fill=False, edgecolor="#00ff88", linewidth=0.5)
                    self._ax.add_patch(circle)
                self._ax.set_aspect("equal")
                self._ax.autoscale_view()
                self._ax.set_facecolor("#1a1a2e")
            elif kind == "cantor":
                for (x0, x1, y) in data:
                    self._ax.plot([x0, x1], [y, y], color="#00ff88", linewidth=2)
                self._ax.set_facecolor("#1a1a2e")
            elif kind == "menger":
                from mpl_toolkits.mplot3d import Axes3D
                self._figure.clear()
                ax3 = self._figure.add_subplot(111, projection="3d")
                ax3.set_facecolor("#1a1a2e")
                # Draw a subset of cubes as wireframe
                max_draw = min(len(data), 5000)
                step = max(1, len(data) // max_draw)
                for idx in range(0, len(data), step):
                    cx, cy, cz, s = data[idx]
                    corners = np.array([
                        [cx, cy, cz], [cx + s, cy, cz], [cx + s, cy + s, cz], [cx, cy + s, cz],
                        [cx, cy, cz + s], [cx + s, cy, cz + s], [cx + s, cy + s, cz + s], [cx, cy + s, cz + s]
                    ])
                    edges = [
                        [0, 1], [1, 2], [2, 3], [3, 0],
                        [4, 5], [5, 6], [6, 7], [7, 4],
                        [0, 4], [1, 5], [2, 6], [3, 7]
                    ]
                    for e in edges:
                        ax3.plot3D(*zip(corners[e[0]], corners[e[1]]), color="#00ff88", linewidth=0.2)
                ax3.set_xlabel("X")
                ax3.set_ylabel("Y")
                ax3.set_zlabel("Z")
                self._ax = ax3
            elif kind == "tetrahedron":
                from mpl_toolkits.mplot3d import Axes3D
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                self._figure.clear()
                ax3 = self._figure.add_subplot(111, projection="3d")
                ax3.set_facecolor("#1a1a2e")
                max_draw = min(len(data), 3000)
                step = max(1, len(data) // max_draw)
                faces = []
                for idx in range(0, len(data), step):
                    a, b, c, d = data[idx]
                    faces.append([a, b, c])
                    faces.append([a, b, d])
                    faces.append([a, c, d])
                    faces.append([b, c, d])
                poly = Poly3DCollection(faces, alpha=0.3, facecolor="#00ff88", edgecolor="#008844", linewidth=0.2)
                ax3.add_collection3d(poly)
                ax3.auto_scale_xyz([-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5])
                self._ax = ax3
        elif isinstance(result, np.ndarray):
            # Apply colormap to 2D array — normalize excluding interior (0) for better contrast
            # Replace NaN/Inf with 0 first
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            interior_mask = (result == 0)
            exterior = result[~interior_mask]
            if exterior.size > 0:
                vmin, vmax = float(exterior.min()), float(exterior.max())
                denom = vmax - vmin
                if denom > 0:
                    norm = np.clip((result - vmin) / denom, 0, 1)
                else:
                    norm = np.ones_like(result) * 0.5
            else:
                norm = np.zeros_like(result)
            norm[interior_mask] = 0

            # Interior coloring
            if self._interior_combo.currentText() == "Black":
                colored = cmap(norm)
                colored[interior_mask] = [0, 0, 0, 1]
            else:
                colored = cmap(norm)

            interp = "bilinear" if not is_preview else "nearest"
            self._ax.imshow(colored, aspect="auto", origin="upper", interpolation=interp)

        self._canvas.draw_idle()

    # ----- Export -----

    def export(self):
        """Export the current fractal image (public API)."""
        path, filt = QFileDialog.getSaveFileName(
            self, "Export Fractal Image", "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tiff);;BMP (*.bmp);;All Files (*)"
        )
        if not path:
            return

        # Render at current or higher resolution
        w, h = self._get_resolution()
        msg = QMessageBox.question(
            self, "Export Resolution",
            f"Export at current resolution ({w}x{h})?\n\n"
            "Click 'No' to export at 4096x4096 or 'Yes' for current.",
            QMessageBox.Yes | QMessageBox.No
        )
        if msg == QMessageBox.No:
            w, h = 4096, 4096

        params = self._build_params(w, h)
        func = FRACTAL_DISPATCH.get(self._current_fractal)
        if func is None:
            return

        self._log(f"Exporting at {w}x{h}...")
        self._progress.setValue(0)

        class ExportThread(QThread):
            done = pyqtSignal(object)

            def __init__(self, f, p):
                super().__init__()
                self._f = f
                self._p = p

            def run(self):
                result = self._f(self._p, _DummySignal(), lambda: False)
                self.done.emit(result)

        def _on_export_done(result):
            if result is None or isinstance(result, tuple):
                # For line/geometry fractals, save the figure directly
                self._figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="#1a1a2e")
                self._log(f"Exported (figure) to {path}")
                return
            if isinstance(result, np.ndarray):
                cmap_name = self._cmap_combo.currentText()
                try:
                    cmap = plt.get_cmap(cmap_name)
                except ValueError:
                    cmap = plt.get_cmap("inferno")
                if result.max() > result.min():
                    norm = (result - result.min()) / (result.max() - result.min())
                else:
                    norm = np.zeros_like(result)
                interior_mask = (result == 0)
                colored = cmap(norm)
                colored[interior_mask] = [0, 0, 0, 1]
                img_data = (colored[:, :, :3] * 255).astype(np.uint8)

                from PIL import Image
                try:
                    img = Image.fromarray(img_data)
                    img.save(path)
                    self._log(f"Exported to {path}")
                except ImportError:
                    fig_tmp = Figure(figsize=(w / 100, h / 100), dpi=100)
                    style_figure(fig_tmp)
                    ax_tmp = fig_tmp.add_subplot(111)
                    ax_tmp.imshow(colored, aspect="auto")
                    ax_tmp.axis("off")
                    fig_tmp.savefig(path, dpi=100, bbox_inches="tight", pad_inches=0)
                    self._log(f"Exported (matplotlib) to {path}")

        thread = ExportThread(func, params)
        thread.done.connect(_on_export_done)
        thread.start()
        self._export_thread = thread  # prevent GC

    def _save_params(self):
        """Save current parameters to JSON."""
        path, _ = QFileDialog.getSaveFileName(self, "Save Parameters", "", "JSON (*.json)")
        if not path:
            return
        data = {
            "fractal": self._current_fractal,
            "xmin": self._xmin, "xmax": self._xmax,
            "ymin": self._ymin, "ymax": self._ymax,
            "max_iter": self._iter_slider.value(),
            "resolution": self._res_combo.currentText(),
            "colormap": self._cmap_combo.currentText(),
            "julia_cr": self._julia_cr.value(),
            "julia_ci": self._julia_ci.value(),
            "power": self._power_spin.value(),
            "newton_power": self._newton_power_spin.value(),
            "phoenix_cr": self._phoenix_cr_spin.value(),
            "phoenix_ci": self._phoenix_ci_spin.value(),
            "lyapunov_seq": self._lyapunov_seq.text(),
            "lsys_iterations": self._lsys_iters_spin.value(),
            "custom_formula": self._formula_edit.text(),
            "lsys_axiom": self._lsys_axiom_edit.text(),
            "lsys_rules": self._lsys_rules_edit.text(),
            "lsys_angle": self._lsys_angle_spin.value(),
            "newton_coeffs": self._newton_coeffs_edit.text(),
            "supersample": self._supersample_combo.currentText(),
            "interior": self._interior_combo.currentText(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self._log(f"Parameters saved to {path}")

    def _load_params(self):
        """Load parameters from JSON."""
        path, _ = QFileDialog.getOpenFileName(self, "Load Parameters", "", "JSON (*.json)")
        if not path or not os.path.isfile(path):
            return
        with open(path, "r") as f:
            data = json.load(f)

        if "fractal" in data:
            self._current_fractal = data["fractal"]
        if "xmin" in data:
            self._xmin = data["xmin"]
        if "xmax" in data:
            self._xmax = data["xmax"]
        if "ymin" in data:
            self._ymin = data["ymin"]
        if "ymax" in data:
            self._ymax = data["ymax"]
        if "max_iter" in data:
            self._iter_slider.setValue(data["max_iter"])
        if "resolution" in data:
            idx = self._res_combo.findText(data["resolution"])
            if idx >= 0:
                self._res_combo.setCurrentIndex(idx)
        if "colormap" in data:
            idx = self._cmap_combo.findText(data["colormap"])
            if idx >= 0:
                self._cmap_combo.setCurrentIndex(idx)
        if "julia_cr" in data:
            self._julia_cr.setValue(data["julia_cr"])
        if "julia_ci" in data:
            self._julia_ci.setValue(data["julia_ci"])
        if "power" in data:
            self._power_spin.setValue(data["power"])
        if "newton_power" in data:
            self._newton_power_spin.setValue(data["newton_power"])
        if "phoenix_cr" in data:
            self._phoenix_cr_spin.setValue(data["phoenix_cr"])
        if "phoenix_ci" in data:
            self._phoenix_ci_spin.setValue(data["phoenix_ci"])
        if "lyapunov_seq" in data:
            self._lyapunov_seq.setText(data["lyapunov_seq"])
        if "lsys_iterations" in data:
            self._lsys_iters_spin.setValue(data["lsys_iterations"])
        if "custom_formula" in data:
            self._formula_edit.setText(data["custom_formula"])
        if "lsys_axiom" in data:
            self._lsys_axiom_edit.setText(data["lsys_axiom"])
        if "lsys_rules" in data:
            self._lsys_rules_edit.setText(data["lsys_rules"])
        if "lsys_angle" in data:
            self._lsys_angle_spin.setValue(data["lsys_angle"])
        if "newton_coeffs" in data:
            self._newton_coeffs_edit.setText(data["newton_coeffs"])
        if "supersample" in data:
            idx = self._supersample_combo.findText(data["supersample"])
            if idx >= 0:
                self._supersample_combo.setCurrentIndex(idx)
        if "interior" in data:
            idx = self._interior_combo.findText(data["interior"])
            if idx >= 0:
                self._interior_combo.setCurrentIndex(idx)

        self._log(f"Parameters loaded from {path}")
        self.run()

    def _export_animation(self):
        """Export a zoom-in animation as individual frames or GIF."""
        path, filt = QFileDialog.getSaveFileName(
            self, "Export Animation", "",
            "GIF (*.gif);;PNG Frames (*.png);;All Files (*)"
        )
        if not path:
            return

        n_frames = 60
        zoom_factor = 0.92

        cx = (self._xmin + self._xmax) / 2
        cy = (self._ymin + self._ymax) / 2
        xrange = self._xmax - self._xmin
        yrange = self._ymax - self._ymin

        func = FRACTAL_DISPATCH.get(self._current_fractal)
        if func is None:
            return

        frame_w, frame_h = 512, 512
        frames = []

        self._log(f"Generating {n_frames} animation frames...")
        for i in range(n_frames):
            params = self._build_params(frame_w, frame_h)
            half_x = xrange / 2
            half_y = yrange / 2
            params["xmin"] = cx - half_x
            params["xmax"] = cx + half_x
            params["ymin"] = cy - half_y
            params["ymax"] = cy + half_y
            params["max_iter"] = min(params["max_iter"], 300)
            params["supersample"] = 1

            result = func(params, _DummySignal(), lambda: False)
            if result is not None and isinstance(result, np.ndarray):
                cmap_name = self._cmap_combo.currentText()
                try:
                    cmap = plt.get_cmap(cmap_name)
                except ValueError:
                    cmap = plt.get_cmap("inferno")
                if result.max() > result.min():
                    norm = (result - result.min()) / (result.max() - result.min())
                else:
                    norm = np.zeros_like(result)
                colored = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
                frames.append(colored)

            xrange *= zoom_factor
            yrange *= zoom_factor
            self._progress.setValue(int(100 * (i + 1) / n_frames))
            QApplication.processEvents()

        if not frames:
            self._log("No frames generated.")
            return

        if path.lower().endswith(".gif"):
            try:
                from PIL import Image
                pil_frames = [Image.fromarray(f) for f in frames]
                pil_frames[0].save(
                    path, save_all=True, append_images=pil_frames[1:],
                    duration=50, loop=0
                )
                self._log(f"Animation saved to {path}")
            except ImportError:
                self._log("PIL/Pillow required for GIF export. Saving frames as PNG instead.")
                base, _ = os.path.splitext(path)
                for idx, frame in enumerate(frames):
                    fig_tmp = Figure(figsize=(frame_w / 100, frame_h / 100), dpi=100)
                    style_figure(fig_tmp)
                    ax_tmp = fig_tmp.add_subplot(111)
                    ax_tmp.imshow(frame)
                    ax_tmp.axis("off")
                    fig_tmp.savefig(f"{base}_{idx:04d}.png", dpi=100, bbox_inches="tight", pad_inches=0)
                self._log(f"Frames saved as {base}_XXXX.png")
        else:
            base, _ = os.path.splitext(path)
            for idx, frame in enumerate(frames):
                fig_tmp = Figure(figsize=(frame_w / 100, frame_h / 100), dpi=100)
                style_figure(fig_tmp)
                ax_tmp = fig_tmp.add_subplot(111)
                ax_tmp.imshow(frame)
                ax_tmp.axis("off")
                fig_tmp.savefig(f"{base}_{idx:04d}.png", dpi=100, bbox_inches="tight", pad_inches=0)
            self._log(f"{len(frames)} frames saved as {base}_XXXX.png")

    def _copy_to_clipboard(self):
        """Copy current fractal image to the system clipboard (legacy)."""
        self._copy_plot_to_clipboard()

    def _copy_plot_to_clipboard(self):
        """Copy current plot to clipboard as image."""
        import io
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtGui import QImage
        buf = io.BytesIO()
        self._figure.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                             facecolor=self._figure.get_facecolor())
        buf.seek(0)
        img = QImage()
        img.loadFromData(buf.read())
        QApplication.clipboard().setImage(img)
        self._log("Plot copied to clipboard.")


# ---------------------------------------------------------------------------
# Dummy signal for synchronous preview rendering
# ---------------------------------------------------------------------------

class _DummySignal:
    """A no-op signal replacement for synchronous calls."""
    def emit(self, *args):
        pass


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

def main():
    """Launch the fractal explorer as a standalone application."""
    import sys
    from PyQt5.QtWidgets import QApplication, QMainWindow

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = app.palette()
    palette.setColor(palette.Window, QColor(26, 26, 46))
    palette.setColor(palette.WindowText, QColor(230, 230, 230))
    palette.setColor(palette.Base, QColor(30, 30, 50))
    palette.setColor(palette.AlternateBase, QColor(40, 40, 60))
    palette.setColor(palette.Text, QColor(230, 230, 230))
    palette.setColor(palette.Button, QColor(40, 40, 60))
    palette.setColor(palette.ButtonText, QColor(230, 230, 230))
    palette.setColor(palette.Highlight, QColor(0, 120, 215))
    app.setPalette(palette)

    win = QMainWindow()
    win.setWindowTitle("Axiom Scientific Suite - Fractal Explorer")
    win.resize(1400, 900)

    widget = FractalExplorerWidget()
    widget.set_logger(lambda msg: print(f"[FractalExplorer] {msg}"))
    win.setCentralWidget(widget)
    win.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
