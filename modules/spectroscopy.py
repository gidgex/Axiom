"""
Spectroscopy Analysis Widget for PyQt5 Scientific Suite.

Provides interactive spectrum loading, processing, peak finding/fitting,
and visualization for UV-Vis, IR, Raman, NMR, XRD, XPS, and Mass Spec data.
"""

import csv
import json
import os
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.polynomial import polynomial as P
from scipy import signal, optimize, sparse
from scipy.sparse.linalg import spsolve
from scipy.special import voigt_profile

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QComboBox,
    QPushButton, QSpinBox, QDoubleSpinBox, QFileDialog, QSplitter,
    QFormLayout, QTabWidget, QCheckBox, QMessageBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QSizePolicy, QLineEdit, QListWidget,
    QListWidgetItem, QColorDialog, QGridLayout,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

try:
    from mpl_style import style_figure, style_axes
except ImportError:
    def style_figure(f): pass
    def style_axes(a): pass


# ---------------------------------------------------------------------------
# Enums & constants
# ---------------------------------------------------------------------------

class SpectrumType(Enum):
    UV_VIS = "UV-Vis"
    IR = "IR"
    RAMAN = "Raman"
    NMR = "NMR"
    XRD = "XRD"
    XPS = "XPS"
    MASS_SPEC = "Mass Spec"


class BaselineMethod(Enum):
    POLYNOMIAL = "Polynomial"
    RUBBER_BAND = "Rubber Band"
    ALS = "Asymmetric Least Squares"


class PeakProfile(Enum):
    GAUSSIAN = "Gaussian"
    LORENTZIAN = "Lorentzian"
    VOIGT = "Voigt"


class SmoothingMethod(Enum):
    SAVITZKY_GOLAY = "Savitzky-Golay"
    MOVING_AVERAGE = "Moving Average"


class NormalizationMode(Enum):
    MAX = "Max"
    AREA = "Area"
    MIN_MAX = "Min-Max"


SPECTRUM_LABELS: Dict[SpectrumType, Tuple[str, str]] = {
    SpectrumType.UV_VIS: ("Wavelength (nm)", "Absorbance"),
    SpectrumType.IR: ("Wavenumber (cm\u207b\u00b9)", "Transmittance (%)"),
    SpectrumType.RAMAN: ("Raman Shift (cm\u207b\u00b9)", "Intensity"),
    SpectrumType.NMR: ("Chemical Shift (ppm)", "Intensity"),
    SpectrumType.XRD: ("2\u03b8 (\u00b0)", "Intensity"),
    SpectrumType.XPS: ("Binding Energy (eV)", "Counts/s"),
    SpectrumType.MASS_SPEC: ("m/z", "Relative Intensity"),
}


# ---------------------------------------------------------------------------
# Processing helpers (pure functions)
# ---------------------------------------------------------------------------

def _baseline_polynomial(x: np.ndarray, y: np.ndarray, order: int = 3) -> np.ndarray:
    """Fit and subtract a polynomial baseline."""
    coeffs = np.polyfit(x, y, order)
    baseline = np.polyval(coeffs, x)
    return y - baseline


def _baseline_rubber_band(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Rubber-band baseline correction using convex hull."""
    from scipy.spatial import ConvexHull
    points = np.column_stack([x, y])
    hull = ConvexHull(points)
    hull_verts = sorted(set(hull.vertices))
    lower = [v for v in hull_verts if y[v] <= np.median(y)]
    if len(lower) < 2:
        lower = [hull_verts[0], hull_verts[-1]]
    baseline = np.interp(x, x[lower], y[lower])
    return y - baseline


def _baseline_als(y: np.ndarray, lam: float = 1e6, p: float = 0.01,
                  n_iter: int = 10) -> np.ndarray:
    """Asymmetric least squares baseline estimation (Eilers & Boelens)."""
    length = len(y)
    diag = sparse.diags([1, -2, 1], [0, 1, 2], shape=(length - 2, length))
    D = diag.T @ diag
    w = np.ones(length)
    for _ in range(n_iter):
        W = sparse.spdiags(w, 0, length, length)
        Z = W + lam * D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return y - z


def _smooth_savgol(y: np.ndarray, window: int = 11, poly: int = 3) -> np.ndarray:
    window = window if window % 2 == 1 else window + 1
    window = max(window, poly + 2)
    return signal.savgol_filter(y, window, poly)


def _smooth_moving_avg(y: np.ndarray, window: int = 5) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def _normalize_max(y: np.ndarray) -> np.ndarray:
    ymax = np.max(np.abs(y))
    return y / ymax if ymax != 0 else y.copy()


def _normalize_area(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    area = np.trapz(np.abs(y), x)
    return y / area if area != 0 else y.copy()


def _normalize_minmax(y: np.ndarray) -> np.ndarray:
    ymin, ymax = y.min(), y.max()
    span = ymax - ymin
    return (y - ymin) / span if span != 0 else y.copy()


def _derivative(x: np.ndarray, y: np.ndarray, order: int = 1) -> np.ndarray:
    result = y.copy()
    for _ in range(order):
        result = np.gradient(result, x)
    return result


def _convert_wavelength_to_wavenumber(wl_nm: np.ndarray) -> np.ndarray:
    """nm -> cm^-1."""
    return 1e7 / wl_nm


def _convert_wavelength_to_ev(wl_nm: np.ndarray) -> np.ndarray:
    """nm -> eV  (E = hc / lambda)."""
    return 1239.841984 / wl_nm


def _convert_wavenumber_to_wavelength(wn: np.ndarray) -> np.ndarray:
    """cm^-1 -> nm."""
    return 1e7 / wn


def _convert_wavenumber_to_ev(wn: np.ndarray) -> np.ndarray:
    """cm^-1 -> eV."""
    return wn * 1.239841984e-4


# ---------------------------------------------------------------------------
# Peak profile models for curve fitting
# ---------------------------------------------------------------------------

def _gaussian(x, amplitude, center, sigma):
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _lorentzian(x, amplitude, center, gamma):
    return amplitude * gamma ** 2 / ((x - center) ** 2 + gamma ** 2)


def _voigt(x, amplitude, center, sigma, gamma):
    return amplitude * voigt_profile(x - center, sigma, gamma)


# ---------------------------------------------------------------------------
# Spectral peak database
# ---------------------------------------------------------------------------

# IR functional group frequencies: (name, wavenumber_cm-1, intensity_desc)
IR_PEAK_DATABASE: List[Tuple[str, float, str]] = [
    ("O-H stretch (alcohol)", 3400, "broad, strong"),
    ("O-H stretch (carboxylic acid)", 3000, "very broad, strong"),
    ("N-H stretch (amine)", 3350, "medium"),
    ("C-H stretch (alkane)", 2950, "strong"),
    ("C-H stretch (alkene)", 3080, "medium"),
    ("C-H stretch (aromatic)", 3030, "medium"),
    ("C-H stretch (aldehyde)", 2720, "medium"),
    ("C=C=N stretch (nitrile)", 2250, "medium-strong"),
    ("C=C stretch (alkyne)", 2150, "variable"),
    ("C=O stretch (ketone)", 1715, "strong"),
    ("C=O stretch (aldehyde)", 1725, "strong"),
    ("C=O stretch (carboxylic acid)", 1710, "strong"),
    ("C=O stretch (ester)", 1740, "strong"),
    ("C=O stretch (amide)", 1650, "strong"),
    ("C=C stretch (alkene)", 1650, "variable"),
    ("C=C stretch (aromatic)", 1600, "variable"),
    ("C=C stretch (aromatic)", 1500, "variable"),
    ("N-H bend (amine)", 1600, "medium"),
    ("C-H bend (alkane)", 1450, "medium"),
    ("C-H bend (methyl)", 1375, "medium"),
    ("C-O stretch (alcohol)", 1050, "strong"),
    ("C-O stretch (ether)", 1100, "strong"),
    ("C-O stretch (ester)", 1200, "strong"),
    ("C-F stretch", 1100, "strong"),
    ("C-Cl stretch", 750, "strong"),
    ("C-Br stretch", 650, "strong"),
    ("=C-H bend (alkene)", 900, "strong"),
    ("aromatic C-H bend", 750, "strong"),
]

# UV-Vis common chromophores: (name, lambda_max_nm, epsilon_approx)
UVVIS_PEAK_DATABASE: List[Tuple[str, float, float]] = [
    ("Benzene", 254, 200),
    ("Naphthalene", 286, 9300),
    ("Anthracene", 375, 7900),
    ("Acetone", 280, 15),
    ("Acetaldehyde", 293, 12),
    ("Nitromethane", 271, 19),
    ("Pyridine", 257, 2750),
    ("Phenol", 270, 1450),
    ("Aniline", 280, 1430),
    ("Styrene", 282, 450),
    ("1,3-Butadiene", 217, 21000),
    ("beta-Carotene", 450, 139000),
    ("Chlorophyll a", 430, 111700),
    ("Chlorophyll a (red)", 662, 87000),
    ("DNA (260 nm)", 260, 6600),
    ("Tryptophan", 280, 5500),
    ("NADH", 340, 6220),
    ("Cytochrome c (oxidized)", 530, 11000),
    ("Rhodamine B", 554, 106000),
    ("Fluorescein", 490, 76900),
]


# ---------------------------------------------------------------------------
# Spectrum simulator
# ---------------------------------------------------------------------------

def simulate_uvvis_spectrum(transitions: List[Tuple[float, float]],
                            x_min: float = 200, x_max: float = 800,
                            n_points: int = 1000,
                            fwhm: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate a UV-Vis absorption spectrum from transition energies.

    Parameters
    ----------
    transitions : list of (wavelength_nm, molar_absorptivity)
    fwhm : full width at half maximum in nm

    Returns (wavelength_array, absorbance_array).
    """
    x = np.linspace(x_min, x_max, n_points)
    y = np.zeros_like(x)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    for lam, eps in transitions:
        y += eps * np.exp(-0.5 * ((x - lam) / sigma) ** 2)
    return x, y


def simulate_ir_spectrum(peaks: List[Tuple[float, float]],
                         x_min: float = 400, x_max: float = 4000,
                         n_points: int = 2000,
                         fwhm: float = 30.0) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate an IR transmittance spectrum from force constant peaks.

    Parameters
    ----------
    peaks : list of (wavenumber_cm-1, relative_intensity 0-100)
    fwhm : full width at half maximum in cm-1

    Returns (wavenumber_array, transmittance_array).
    """
    x = np.linspace(x_min, x_max, n_points)
    absorbance = np.zeros_like(x)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    for wn, intensity in peaks:
        absorbance += (intensity / 100.0) * np.exp(-0.5 * ((x - wn) / sigma) ** 2)
    # Convert to transmittance (%)
    transmittance = 100.0 * np.exp(-absorbance * 2.303)
    return x, transmittance


# ---------------------------------------------------------------------------
# Deconvolution: automatic peak detection + multi-peak fitting
# ---------------------------------------------------------------------------

def auto_deconvolve(x: np.ndarray, y: np.ndarray,
                    profile: str = "Gaussian",
                    min_prominence: float = 0.01,
                    min_distance: int = 5,
                    max_peaks: int = 20,
                    max_iter: int = 5
                    ) -> Tuple[List[np.ndarray], List[Dict], np.ndarray]:
    """Automatic peak deconvolution via iterative fitting.

    1. Find peaks automatically
    2. Fit sum of profiles
    3. Check residual, find remaining peaks
    4. Refit with all peaks

    Returns (individual_curves, fit_parameters, composite_curve).
    """
    residual = y.copy()
    all_peaks_idx = []

    for iteration in range(max_iter):
        peaks, props = signal.find_peaks(residual,
                                          prominence=min_prominence,
                                          distance=min_distance)
        if len(peaks) == 0:
            break
        all_peaks_idx.extend(peaks.tolist())
        # Remove duplicates (close peaks)
        all_peaks_idx = list(set(all_peaks_idx))
        all_peaks_idx.sort()
        if len(all_peaks_idx) >= max_peaks:
            all_peaks_idx = all_peaks_idx[:max_peaks]
            break

        # Fit current set of peaks
        _, params, composite = _fit_multi_peak(x, y, all_peaks_idx, profile)
        residual = y - composite
        # Stop if residual is small
        if np.max(np.abs(residual)) < min_prominence * 0.5:
            break

    # Final fit with all detected peaks
    curves, params, composite = _fit_multi_peak(x, y, all_peaks_idx, profile)
    return curves, params, composite


def _fit_multi_peak(x: np.ndarray, y: np.ndarray,
                    peak_indices: List[int],
                    profile: str = "Gaussian"
                    ) -> Tuple[List[np.ndarray], List[Dict], np.ndarray]:
    """Fit multiple peaks simultaneously."""
    n_peaks = len(peak_indices)
    if n_peaks == 0:
        return [], [], np.zeros_like(x)

    # Build initial guesses
    p0 = []
    bounds_lo = []
    bounds_hi = []
    for idx in peak_indices:
        amp = y[idx]
        cen = x[idx]
        # Estimate width
        half_max = amp / 2.0
        left = idx
        while left > 0 and y[left] > half_max:
            left -= 1
        right = idx
        while right < len(y) - 1 and y[right] > half_max:
            right += 1
        dx = abs(x[right] - x[left]) / 2.0 if right > left else abs(x[1] - x[0]) * 5
        sig = max(dx / (2.0 * np.sqrt(2.0 * np.log(2.0))), abs(x[1] - x[0]))

        p0.extend([amp, cen, sig])
        bounds_lo.extend([0, x.min(), abs(x[1] - x[0]) * 0.1])
        bounds_hi.extend([amp * 5, x.max(), (x.max() - x.min()) / 2])

    if profile == "Gaussian":
        func = _gaussian
        params_per_peak = 3
    elif profile == "Lorentzian":
        func = _lorentzian
        params_per_peak = 3
    else:
        func = _gaussian
        params_per_peak = 3

    def multi_func(x_data, *params):
        result = np.zeros_like(x_data)
        for i in range(n_peaks):
            offset = i * params_per_peak
            result += func(x_data, *params[offset:offset + params_per_peak])
        return result

    try:
        popt, _ = optimize.curve_fit(multi_func, x, y, p0=p0,
                                      bounds=(bounds_lo, bounds_hi),
                                      maxfev=10000)
    except Exception:
        # Fall back to individual fitting
        popt = np.array(p0)

    # Extract individual curves and parameters
    curves = []
    params_list = []
    composite = np.zeros_like(x)
    for i in range(n_peaks):
        offset = i * params_per_peak
        p = popt[offset:offset + params_per_peak]
        curve = func(x, *p)
        curves.append(curve)
        composite += curve
        params_list.append({
            "amplitude": float(p[0]),
            "center": float(p[1]),
            "width": float(p[2]),
            "fwhm": float(2.0 * np.sqrt(2.0 * np.log(2.0)) * p[2]) if profile == "Gaussian"
                     else float(2.0 * p[2]),
            "area": float(np.trapz(curve, x)),
        })

    return curves, params_list, composite


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_html_report(x: np.ndarray, y: np.ndarray,
                         peaks_x: Optional[np.ndarray],
                         peaks_y: Optional[np.ndarray],
                         fit_params: Optional[List[Dict]],
                         spectrum_type: str = "UV-Vis",
                         image_path: Optional[str] = None,
                         title: str = "Spectroscopy Report"
                         ) -> str:
    """Generate an HTML report with spectrum data, peak table, and fit parameters."""
    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
h2 {{ color: #34495e; margin-top: 20px; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; background: white; }}
th {{ background: #3498db; color: white; padding: 8px 12px; text-align: left; }}
td {{ border: 1px solid #ddd; padding: 6px 12px; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
.info {{ background: white; padding: 15px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin: 10px 0; }}
img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 3px; }}
.footer {{ color: #999; font-size: 12px; margin-top: 30px; border-top: 1px solid #ddd; padding-top: 10px; }}
</style>
</head><body>
<h1>{title}</h1>
<div class="info">
<p><strong>Spectrum Type:</strong> {spectrum_type}</p>
<p><strong>Data Points:</strong> {len(x)}</p>
<p><strong>X Range:</strong> {x.min():.2f} - {x.max():.2f}</p>
<p><strong>Y Range:</strong> {y.min():.4f} - {y.max():.4f}</p>
</div>
"""
    if image_path:
        html += f'<h2>Spectrum</h2>\n<img src="{image_path}" alt="Spectrum">\n'

    if peaks_x is not None and len(peaks_x) > 0:
        html += "<h2>Detected Peaks</h2>\n<table>\n"
        html += "<tr><th>#</th><th>X Position</th><th>Y Intensity</th></tr>\n"
        for i in range(len(peaks_x)):
            html += f"<tr><td>{i+1}</td><td>{peaks_x[i]:.4f}</td><td>{peaks_y[i]:.4f}</td></tr>\n"
        html += "</table>\n"

    if fit_params:
        html += "<h2>Fit Parameters</h2>\n<table>\n"
        html += "<tr><th>Peak #</th><th>Center</th><th>Amplitude</th><th>Width</th><th>FWHM</th><th>Area</th></tr>\n"
        for i, p in enumerate(fit_params):
            html += (f"<tr><td>{i+1}</td><td>{p['center']:.4f}</td>"
                     f"<td>{p['amplitude']:.4f}</td><td>{p['width']:.4f}</td>"
                     f"<td>{p['fwhm']:.4f}</td><td>{p['area']:.4f}</td></tr>\n")
        html += "</table>\n"

    import datetime
    html += f'<div class="footer">Generated by QuantumRes Spectroscopy Module | {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>\n'
    html += "</body></html>"
    return html


# ---------------------------------------------------------------------------
# SpectroscopyWidget
# ---------------------------------------------------------------------------

class SpectroscopyWidget(QWidget):
    """Full-featured spectroscopy analysis widget for PyQt5."""

    peaks_found = pyqtSignal(list)
    data_loaded = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._logger: Optional[Callable] = None
        self._x_raw: Optional[np.ndarray] = None
        self._y_raw: Optional[np.ndarray] = None
        self._x: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._peaks_idx: Optional[np.ndarray] = None
        self._fit_curves: List[np.ndarray] = []
        self._file_path: Optional[str] = None
        self._spectrum_type = SpectrumType.UV_VIS
        # Multi-spectrum overlay storage
        self._overlay_spectra: List[Dict] = []  # [{name, x, y, color}]
        # Deconvolution results
        self._deconv_curves: List[np.ndarray] = []
        self._deconv_params: List[Dict] = []
        # Reference spectrum for difference
        self._ref_x: Optional[np.ndarray] = None
        self._ref_y: Optional[np.ndarray] = None
        self._init_ui()

    # -- public API ---------------------------------------------------------

    def set_logger(self, fn: Callable):
        """Register a callable *fn(message: str)* used for status logging."""
        self._logger = fn

    def load_file(self, path: str):
        """Programmatically load a CSV spectrum file at *path*."""
        if not os.path.isfile(path):
            self._log(f"File not found: {path}")
            return
        try:
            x, y = self._read_csv(path)
        except Exception as exc:
            self._log(f"Error reading {path}: {exc}")
            return
        self._ingest(x, y, path)

    def export(self) -> Optional[Dict]:
        """Return current processed data and peaks as a dictionary.

        Returns ``None`` when no data is loaded.
        """
        if self._x is None or self._y is None:
            return None
        result: Dict = {
            "file": self._file_path,
            "spectrum_type": self._spectrum_type.value,
            "x": self._x.tolist(),
            "y": self._y.tolist(),
        }
        if self._peaks_idx is not None and len(self._peaks_idx) > 0:
            result["peaks"] = {
                "x": self._x[self._peaks_idx].tolist(),
                "y": self._y[self._peaks_idx].tolist(),
                "indices": self._peaks_idx.tolist(),
            }
        if self._fit_curves:
            result["fits"] = [c.tolist() for c in self._fit_curves]
        return result

    # -- UI construction ----------------------------------------------------

    def _init_ui(self):
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # Left panel — controls
        ctrl = QWidget()
        ctrl_layout = QVBoxLayout(ctrl)
        ctrl_layout.setContentsMargins(4, 4, 4, 4)
        ctrl.setMinimumWidth(280)
        ctrl.setMaximumWidth(380)

        # File section
        file_grp = QGroupBox("File")
        fl = QVBoxLayout(file_grp)
        self._btn_load = QPushButton("Load Spectrum (CSV)")
        self._btn_load.clicked.connect(self._on_load)
        fl.addWidget(self._btn_load)
        self._lbl_file = QLabel("No file loaded")
        self._lbl_file.setWordWrap(True)
        fl.addWidget(self._lbl_file)
        ctrl_layout.addWidget(file_grp)

        # Spectrum type
        type_grp = QGroupBox("Spectrum Type")
        tl = QFormLayout(type_grp)
        self._combo_type = QComboBox()
        for st in SpectrumType:
            self._combo_type.addItem(st.value, st)
        self._combo_type.currentIndexChanged.connect(self._on_type_changed)
        tl.addRow("Type:", self._combo_type)
        ctrl_layout.addWidget(type_grp)

        # Processing tabs
        proc_tabs = QTabWidget()
        proc_tabs.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # -- Baseline tab
        bl_tab = QWidget()
        bl_lay = QFormLayout(bl_tab)
        self._combo_baseline = QComboBox()
        for bm in BaselineMethod:
            self._combo_baseline.addItem(bm.value, bm)
        bl_lay.addRow("Method:", self._combo_baseline)
        self._spin_poly_order = QSpinBox()
        self._spin_poly_order.setRange(1, 12)
        self._spin_poly_order.setValue(3)
        bl_lay.addRow("Poly Order:", self._spin_poly_order)
        self._spin_als_lam = QDoubleSpinBox()
        self._spin_als_lam.setRange(1e2, 1e10)
        self._spin_als_lam.setDecimals(0)
        self._spin_als_lam.setValue(1e6)
        bl_lay.addRow("ALS \u03bb:", self._spin_als_lam)
        self._spin_als_p = QDoubleSpinBox()
        self._spin_als_p.setRange(0.001, 0.5)
        self._spin_als_p.setDecimals(3)
        self._spin_als_p.setValue(0.01)
        self._spin_als_p.setSingleStep(0.005)
        bl_lay.addRow("ALS p:", self._spin_als_p)
        btn_bl = QPushButton("Apply Baseline")
        btn_bl.clicked.connect(self._on_baseline)
        bl_lay.addRow(btn_bl)
        proc_tabs.addTab(bl_tab, "Baseline")

        # -- Smoothing tab
        sm_tab = QWidget()
        sm_lay = QFormLayout(sm_tab)
        self._combo_smooth = QComboBox()
        for sm in SmoothingMethod:
            self._combo_smooth.addItem(sm.value, sm)
        sm_lay.addRow("Method:", self._combo_smooth)
        self._spin_smooth_win = QSpinBox()
        self._spin_smooth_win.setRange(3, 101)
        self._spin_smooth_win.setValue(11)
        self._spin_smooth_win.setSingleStep(2)
        sm_lay.addRow("Window:", self._spin_smooth_win)
        self._spin_smooth_poly = QSpinBox()
        self._spin_smooth_poly.setRange(1, 7)
        self._spin_smooth_poly.setValue(3)
        sm_lay.addRow("SG Poly:", self._spin_smooth_poly)
        btn_sm = QPushButton("Apply Smoothing")
        btn_sm.clicked.connect(self._on_smooth)
        sm_lay.addRow(btn_sm)
        proc_tabs.addTab(sm_tab, "Smooth")

        # -- Normalization tab
        norm_tab = QWidget()
        norm_lay = QFormLayout(norm_tab)
        self._combo_norm = QComboBox()
        for nm in NormalizationMode:
            self._combo_norm.addItem(nm.value, nm)
        norm_lay.addRow("Mode:", self._combo_norm)
        btn_norm = QPushButton("Normalize")
        btn_norm.clicked.connect(self._on_normalize)
        norm_lay.addRow(btn_norm)
        proc_tabs.addTab(norm_tab, "Normalize")

        # -- Derivative tab
        deriv_tab = QWidget()
        deriv_lay = QFormLayout(deriv_tab)
        self._spin_deriv_order = QSpinBox()
        self._spin_deriv_order.setRange(1, 2)
        self._spin_deriv_order.setValue(1)
        deriv_lay.addRow("Order:", self._spin_deriv_order)
        btn_deriv = QPushButton("Compute Derivative")
        btn_deriv.clicked.connect(self._on_derivative)
        deriv_lay.addRow(btn_deriv)
        proc_tabs.addTab(deriv_tab, "Derivative")

        # -- Unit conversion tab
        conv_tab = QWidget()
        conv_lay = QFormLayout(conv_tab)
        self._combo_conv = QComboBox()
        self._combo_conv.addItems([
            "nm -> cm\u207b\u00b9",
            "nm -> eV",
            "cm\u207b\u00b9 -> nm",
            "cm\u207b\u00b9 -> eV",
        ])
        conv_lay.addRow("Conversion:", self._combo_conv)
        btn_conv = QPushButton("Convert X-axis")
        btn_conv.clicked.connect(self._on_convert)
        conv_lay.addRow(btn_conv)
        proc_tabs.addTab(conv_tab, "Units")

        ctrl_layout.addWidget(proc_tabs)

        # -- Peak finding group
        peak_grp = QGroupBox("Peak Finding")
        pk_lay = QFormLayout(peak_grp)
        self._spin_prom = QDoubleSpinBox()
        self._spin_prom.setRange(0.0, 1e9)
        self._spin_prom.setDecimals(4)
        self._spin_prom.setValue(0.01)
        pk_lay.addRow("Prominence:", self._spin_prom)
        self._spin_width = QDoubleSpinBox()
        self._spin_width.setRange(0.0, 1000.0)
        self._spin_width.setDecimals(2)
        self._spin_width.setValue(1.0)
        pk_lay.addRow("Min Width:", self._spin_width)
        self._spin_dist = QSpinBox()
        self._spin_dist.setRange(1, 10000)
        self._spin_dist.setValue(5)
        pk_lay.addRow("Min Distance:", self._spin_dist)
        btn_find = QPushButton("Find Peaks")
        btn_find.clicked.connect(self._on_find_peaks)
        pk_lay.addRow(btn_find)
        ctrl_layout.addWidget(peak_grp)

        # -- Peak fitting group
        fit_grp = QGroupBox("Peak Fitting")
        fit_lay = QFormLayout(fit_grp)
        self._combo_profile = QComboBox()
        for pp in PeakProfile:
            self._combo_profile.addItem(pp.value, pp)
        fit_lay.addRow("Profile:", self._combo_profile)
        self._spin_fit_window = QSpinBox()
        self._spin_fit_window.setRange(3, 500)
        self._spin_fit_window.setValue(20)
        fit_lay.addRow("Fit Window:", self._spin_fit_window)
        btn_fit = QPushButton("Fit Peaks")
        btn_fit.clicked.connect(self._on_fit_peaks)
        fit_lay.addRow(btn_fit)
        ctrl_layout.addWidget(fit_grp)

        # Reset / Export
        btn_row = QHBoxLayout()
        btn_reset = QPushButton("Reset")
        btn_reset.clicked.connect(self._on_reset)
        btn_row.addWidget(btn_reset)
        btn_export = QPushButton("Export JSON")
        btn_export.clicked.connect(self._on_export)
        btn_row.addWidget(btn_export)
        ctrl_layout.addLayout(btn_row)

        # --- Spectrum Simulator ---
        sim_grp = QGroupBox("Spectrum Simulator")
        sim_lay = QFormLayout(sim_grp)
        self._combo_sim_type = QComboBox()
        self._combo_sim_type.addItems(["UV-Vis (from transitions)", "IR (from vibrations)"])
        sim_lay.addRow("Type:", self._combo_sim_type)
        self._combo_sim_compound = QComboBox()
        self._combo_sim_compound.addItem("-- Select compound --")
        for name, lam, eps in UVVIS_PEAK_DATABASE:
            self._combo_sim_compound.addItem(f"{name} ({lam:.0f} nm)")
        sim_lay.addRow("Compound:", self._combo_sim_compound)
        self._spin_sim_fwhm = QDoubleSpinBox()
        self._spin_sim_fwhm.setRange(1, 200)
        self._spin_sim_fwhm.setValue(20)
        sim_lay.addRow("FWHM:", self._spin_sim_fwhm)
        btn_simulate = QPushButton("Simulate Spectrum")
        btn_simulate.clicked.connect(self._on_simulate)
        sim_lay.addRow(btn_simulate)
        ctrl_layout.addWidget(sim_grp)

        # --- Multi-spectrum Overlay ---
        overlay_grp = QGroupBox("Multi-Spectrum Overlay")
        overlay_lay = QVBoxLayout(overlay_grp)
        btn_add_overlay = QPushButton("Add Current to Overlay")
        btn_add_overlay.clicked.connect(self._on_add_overlay)
        overlay_lay.addWidget(btn_add_overlay)
        btn_load_overlay = QPushButton("Load File to Overlay")
        btn_load_overlay.clicked.connect(self._on_load_overlay)
        overlay_lay.addWidget(btn_load_overlay)
        self._overlay_list = QListWidget()
        self._overlay_list.setMaximumHeight(80)
        overlay_lay.addWidget(self._overlay_list)
        overlay_btn_row = QHBoxLayout()
        btn_plot_overlay = QPushButton("Plot Overlay")
        btn_plot_overlay.clicked.connect(self._on_plot_overlay)
        overlay_btn_row.addWidget(btn_plot_overlay)
        btn_clear_overlay = QPushButton("Clear")
        btn_clear_overlay.clicked.connect(self._on_clear_overlay)
        overlay_btn_row.addWidget(btn_clear_overlay)
        overlay_lay.addLayout(overlay_btn_row)
        ctrl_layout.addWidget(overlay_grp)

        # --- Deconvolution ---
        deconv_grp = QGroupBox("Deconvolution")
        deconv_lay = QFormLayout(deconv_grp)
        self._combo_deconv_profile = QComboBox()
        self._combo_deconv_profile.addItems(["Gaussian", "Lorentzian"])
        deconv_lay.addRow("Profile:", self._combo_deconv_profile)
        self._spin_deconv_max_peaks = QSpinBox()
        self._spin_deconv_max_peaks.setRange(1, 30)
        self._spin_deconv_max_peaks.setValue(10)
        deconv_lay.addRow("Max Peaks:", self._spin_deconv_max_peaks)
        self._spin_deconv_iter = QSpinBox()
        self._spin_deconv_iter.setRange(1, 10)
        self._spin_deconv_iter.setValue(3)
        deconv_lay.addRow("Iterations:", self._spin_deconv_iter)
        btn_deconv = QPushButton("Deconvolve")
        btn_deconv.clicked.connect(self._on_deconvolve)
        deconv_lay.addRow(btn_deconv)
        ctrl_layout.addWidget(deconv_grp)

        # --- Difference Spectrum ---
        diff_grp = QGroupBox("Difference Spectrum")
        diff_lay = QVBoxLayout(diff_grp)
        btn_set_ref = QPushButton("Set Current as Reference")
        btn_set_ref.clicked.connect(self._on_set_reference)
        diff_lay.addWidget(btn_set_ref)
        btn_diff = QPushButton("Compute Difference")
        btn_diff.clicked.connect(self._on_difference)
        diff_lay.addWidget(btn_diff)
        self._lbl_ref = QLabel("No reference set")
        diff_lay.addWidget(self._lbl_ref)
        ctrl_layout.addWidget(diff_grp)

        # --- Report ---
        btn_report = QPushButton("Generate HTML Report")
        btn_report.clicked.connect(self._on_generate_report)
        ctrl_layout.addWidget(btn_report)

        # --- Database Lookup ---
        db_grp = QGroupBox("Peak Database")
        db_lay = QVBoxLayout(db_grp)
        self._combo_db_type = QComboBox()
        self._combo_db_type.addItems(["IR Functional Groups", "UV-Vis Chromophores"])
        db_lay.addWidget(self._combo_db_type)
        btn_show_db = QPushButton("Show Database")
        btn_show_db.clicked.connect(self._on_show_database)
        db_lay.addWidget(btn_show_db)
        ctrl_layout.addWidget(db_grp)

        ctrl_layout.addStretch()
        splitter.addWidget(ctrl)

        # Right panel — plot + peak table
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self._figure = Figure(figsize=(8, 5), tight_layout=True)
        style_figure(self._figure)
        self._canvas = FigureCanvas(self._figure)
        self._toolbar = NavigationToolbar(self._canvas, self)
        right_layout.addWidget(self._toolbar)
        right_layout.addWidget(self._canvas, stretch=3)

        # Peak table
        self._peak_table = QTableWidget(0, 4)
        self._peak_table.setHorizontalHeaderLabels(["#", "X", "Y", "FWHM est."])
        self._peak_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._peak_table.setMaximumHeight(180)
        right_layout.addWidget(self._peak_table, stretch=1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    # -- internal helpers ---------------------------------------------------

    def _log(self, msg: str):
        if self._logger:
            self._logger(msg)

    @staticmethod
    def _read_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Read a two-column CSV and return (x, y) arrays.

        Handles common delimiters (comma, tab, semicolon, space) and
        automatically skips header rows that cannot be parsed as floats.
        """
        delimiters = [",", "\t", ";", " "]
        with open(path, "r", newline="") as fh:
            sample = fh.read(4096)
        chosen_delim = ","
        for d in delimiters:
            if d in sample:
                chosen_delim = d
                break
        xs, ys = [], []
        with open(path, "r", newline="") as fh:
            reader = csv.reader(fh, delimiter=chosen_delim)
            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    xv = float(row[0].strip())
                    yv = float(row[1].strip())
                except ValueError:
                    continue
                xs.append(xv)
                ys.append(yv)
        if len(xs) == 0:
            raise ValueError("No numeric data found in file")
        return np.array(xs), np.array(ys)

    def _ingest(self, x: np.ndarray, y: np.ndarray, path: str):
        self._x_raw = x.copy()
        self._y_raw = y.copy()
        self._x = x.copy()
        self._y = y.copy()
        self._peaks_idx = None
        self._fit_curves = []
        self._file_path = path
        self._lbl_file.setText(os.path.basename(path))
        self._log(f"Loaded {len(x)} points from {os.path.basename(path)}")
        self.data_loaded.emit(path)
        self._update_plot()

    def _ensure_data(self) -> bool:
        if self._x is None or self._y is None:
            self._log("No spectrum data loaded.")
            return False
        return True

    # -- plot ---------------------------------------------------------------

    def _update_plot(self):
        self._figure.clear()
        if self._x is None:
            self._canvas.draw()
            return
        ax = self._figure.add_subplot(111)
        xlabel, ylabel = SPECTRUM_LABELS.get(
            self._spectrum_type, ("X", "Y"))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(self._spectrum_type.value + " Spectrum")

        ax.plot(self._x, self._y, color="#1f77b4", linewidth=1.0,
                label="Spectrum")

        # peaks
        if self._peaks_idx is not None and len(self._peaks_idx) > 0:
            ax.plot(self._x[self._peaks_idx], self._y[self._peaks_idx],
                    "rv", markersize=8, label="Peaks")
            for idx in self._peaks_idx:
                ax.annotate(
                    f"{self._x[idx]:.2f}",
                    xy=(self._x[idx], self._y[idx]),
                    xytext=(0, 10), textcoords="offset points",
                    fontsize=7, ha="center", color="red",
                )

        # fit curves
        for i, curve in enumerate(self._fit_curves):
            ax.plot(self._x, curve, "--", linewidth=1.0,
                    label=f"Fit {i + 1}")

        ax.legend(fontsize=8, loc="best")
        self._canvas.draw()

    def _populate_peak_table(self):
        self._peak_table.setRowCount(0)
        if self._peaks_idx is None or len(self._peaks_idx) == 0:
            return
        widths_result = signal.peak_widths(self._y, self._peaks_idx, rel_height=0.5)
        fwhms = widths_result[0]
        for i, idx in enumerate(self._peaks_idx):
            row = self._peak_table.rowCount()
            self._peak_table.insertRow(row)
            self._peak_table.setItem(row, 0, QTableWidgetItem(str(i + 1)))
            self._peak_table.setItem(row, 1, QTableWidgetItem(f"{self._x[idx]:.4f}"))
            self._peak_table.setItem(row, 2, QTableWidgetItem(f"{self._y[idx]:.4f}"))
            fwhm_x = 0.0
            if i < len(fwhms):
                dx = np.mean(np.diff(self._x)) if len(self._x) > 1 else 1.0
                fwhm_x = fwhms[i] * abs(dx)
            self._peak_table.setItem(row, 3, QTableWidgetItem(f"{fwhm_x:.4f}"))

    # -- slots --------------------------------------------------------------

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Spectrum File", "",
            "CSV Files (*.csv *.txt *.dat *.tsv);;All Files (*)")
        if path:
            self.load_file(path)

    def _on_type_changed(self, index: int):
        self._spectrum_type = self._combo_type.currentData()
        self._log(f"Spectrum type set to {self._spectrum_type.value}")
        self._update_plot()

    def _on_baseline(self):
        if not self._ensure_data():
            return
        method: BaselineMethod = self._combo_baseline.currentData()
        self._log(f"Applying baseline correction: {method.value}")
        try:
            if method == BaselineMethod.POLYNOMIAL:
                self._y = _baseline_polynomial(
                    self._x, self._y, self._spin_poly_order.value())
            elif method == BaselineMethod.RUBBER_BAND:
                self._y = _baseline_rubber_band(self._x, self._y)
            elif method == BaselineMethod.ALS:
                self._y = _baseline_als(
                    self._y,
                    lam=self._spin_als_lam.value(),
                    p=self._spin_als_p.value(),
                )
        except Exception as exc:
            self._log(f"Baseline correction failed: {exc}")
            return
        self._peaks_idx = None
        self._fit_curves = []
        self._update_plot()

    def _on_smooth(self):
        if not self._ensure_data():
            return
        method: SmoothingMethod = self._combo_smooth.currentData()
        win = self._spin_smooth_win.value()
        self._log(f"Smoothing: {method.value}, window={win}")
        if method == SmoothingMethod.SAVITZKY_GOLAY:
            poly = self._spin_smooth_poly.value()
            self._y = _smooth_savgol(self._y, window=win, poly=poly)
        else:
            self._y = _smooth_moving_avg(self._y, window=win)
        self._peaks_idx = None
        self._fit_curves = []
        self._update_plot()

    def _on_normalize(self):
        if not self._ensure_data():
            return
        mode: NormalizationMode = self._combo_norm.currentData()
        self._log(f"Normalizing: {mode.value}")
        if mode == NormalizationMode.MAX:
            self._y = _normalize_max(self._y)
        elif mode == NormalizationMode.AREA:
            self._y = _normalize_area(self._x, self._y)
        elif mode == NormalizationMode.MIN_MAX:
            self._y = _normalize_minmax(self._y)
        self._peaks_idx = None
        self._fit_curves = []
        self._update_plot()

    def _on_derivative(self):
        if not self._ensure_data():
            return
        order = self._spin_deriv_order.value()
        self._log(f"Computing derivative order={order}")
        self._y = _derivative(self._x, self._y, order)
        self._peaks_idx = None
        self._fit_curves = []
        self._update_plot()

    def _on_convert(self):
        if not self._ensure_data():
            return
        conv = self._combo_conv.currentText()
        self._log(f"Converting x-axis: {conv}")
        try:
            if conv.startswith("nm -> cm"):
                self._x = _convert_wavelength_to_wavenumber(self._x)
            elif conv == "nm -> eV":
                self._x = _convert_wavelength_to_ev(self._x)
            elif conv.startswith("cm") and "nm" in conv:
                self._x = _convert_wavenumber_to_wavelength(self._x)
            elif conv.startswith("cm") and "eV" in conv:
                self._x = _convert_wavenumber_to_ev(self._x)
        except Exception as exc:
            self._log(f"Conversion failed: {exc}")
            return
        self._peaks_idx = None
        self._fit_curves = []
        self._update_plot()

    def _on_find_peaks(self):
        if not self._ensure_data():
            return
        prominence = self._spin_prom.value()
        width = self._spin_width.value()
        distance = self._spin_dist.value()
        self._log(f"Finding peaks (prom={prominence}, width={width}, dist={distance})")
        try:
            peaks, properties = signal.find_peaks(
                self._y,
                prominence=prominence if prominence > 0 else None,
                width=width if width > 0 else None,
                distance=distance,
            )
        except Exception as exc:
            self._log(f"Peak finding error: {exc}")
            return
        self._peaks_idx = peaks
        self._fit_curves = []
        n = len(peaks)
        self._log(f"Found {n} peak(s)")
        self.peaks_found.emit(peaks.tolist())
        self._populate_peak_table()
        self._update_plot()

    def _on_fit_peaks(self):
        if not self._ensure_data():
            return
        if self._peaks_idx is None or len(self._peaks_idx) == 0:
            self._log("No peaks to fit. Run peak finding first.")
            return
        profile: PeakProfile = self._combo_profile.currentData()
        half_win = self._spin_fit_window.value()
        self._log(f"Fitting {len(self._peaks_idx)} peak(s) with {profile.value} profile")
        composite = np.zeros_like(self._y)
        self._fit_curves = []
        for pidx in self._peaks_idx:
            lo = max(0, pidx - half_win)
            hi = min(len(self._x), pidx + half_win + 1)
            xseg = self._x[lo:hi]
            yseg = self._y[lo:hi]
            amp0 = self._y[pidx]
            cen0 = self._x[pidx]
            sig0 = (self._x[hi - 1] - self._x[lo]) / 6.0 if hi > lo else 1.0
            try:
                if profile == PeakProfile.GAUSSIAN:
                    popt, _ = optimize.curve_fit(
                        _gaussian, xseg, yseg,
                        p0=[amp0, cen0, sig0], maxfev=5000)
                    fit_full = _gaussian(self._x, *popt)
                elif profile == PeakProfile.LORENTZIAN:
                    popt, _ = optimize.curve_fit(
                        _lorentzian, xseg, yseg,
                        p0=[amp0, cen0, sig0], maxfev=5000)
                    fit_full = _lorentzian(self._x, *popt)
                elif profile == PeakProfile.VOIGT:
                    popt, _ = optimize.curve_fit(
                        _voigt, xseg, yseg,
                        p0=[amp0, cen0, sig0, sig0], maxfev=5000)
                    fit_full = _voigt(self._x, *popt)
                else:
                    continue
                composite += fit_full
                self._fit_curves.append(fit_full)
            except Exception as exc:
                self._log(f"Fit failed for peak at x={cen0:.2f}: {exc}")
        if self._fit_curves:
            self._fit_curves.insert(0, composite)
            self._log(f"Fitting complete: {len(self._fit_curves) - 1} individual + composite")
        self._update_plot()

    # -- Spectrum simulator -------------------------------------------------

    def _on_simulate(self):
        sim_type = self._combo_sim_type.currentIndex()
        fwhm = self._spin_sim_fwhm.value()

        if sim_type == 0:  # UV-Vis
            compound_idx = self._combo_sim_compound.currentIndex()
            if compound_idx <= 0:
                # Default: simulate a multi-peak UV-Vis
                transitions = [(260, 500), (320, 200), (450, 100)]
            else:
                db_entry = UVVIS_PEAK_DATABASE[compound_idx - 1]
                transitions = [(db_entry[1], db_entry[2])]
            x, y = simulate_uvvis_spectrum(transitions, fwhm=fwhm)
            self._spectrum_type = SpectrumType.UV_VIS
            self._combo_type.setCurrentIndex(0)
        else:  # IR
            # Simulate common organic molecule IR
            peaks = [
                (3400, 60), (2950, 80), (1715, 90), (1450, 50),
                (1375, 40), (1050, 70), (750, 55),
            ]
            x, y = simulate_ir_spectrum(peaks, fwhm=fwhm)
            self._spectrum_type = SpectrumType.IR
            self._combo_type.setCurrentIndex(1)

        self._ingest(x, y, "Simulated Spectrum")
        self._log(f"Generated simulated {self._spectrum_type.value} spectrum")

    # -- Multi-spectrum overlay ---------------------------------------------

    def _on_add_overlay(self):
        if self._x is None or self._y is None:
            self._log("No spectrum to add to overlay.")
            return
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        idx = len(self._overlay_spectra)
        color = colors[idx % len(colors)]
        name = os.path.basename(self._file_path) if self._file_path else f"Spectrum {idx + 1}"
        self._overlay_spectra.append({
            "name": name,
            "x": self._x.copy(),
            "y": self._y.copy(),
            "color": color,
        })
        self._overlay_list.addItem(name)
        self._log(f"Added '{name}' to overlay ({len(self._overlay_spectra)} total)")

    def _on_load_overlay(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Spectrum for Overlay", "",
            "CSV Files (*.csv *.txt *.dat *.tsv);;All Files (*)")
        if not path:
            return
        try:
            x, y = self._read_csv(path)
        except Exception as exc:
            self._log(f"Error reading {path}: {exc}")
            return
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        idx = len(self._overlay_spectra)
        color = colors[idx % len(colors)]
        name = os.path.basename(path)
        self._overlay_spectra.append({
            "name": name, "x": x, "y": y, "color": color,
        })
        self._overlay_list.addItem(name)
        self._log(f"Loaded '{name}' for overlay")

    def _on_plot_overlay(self):
        if not self._overlay_spectra:
            self._log("No spectra in overlay.")
            return
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        xlabel, ylabel = SPECTRUM_LABELS.get(self._spectrum_type, ("X", "Y"))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title("Multi-Spectrum Overlay")

        n = len(self._overlay_spectra)
        for i, spec in enumerate(self._overlay_spectra):
            offset = i * 0.0  # No offset by default; can add vertical offset
            ax.plot(spec["x"], spec["y"] + offset,
                    color=spec["color"], linewidth=1.0,
                    label=spec["name"], alpha=0.85)

        ax.legend(fontsize=7, loc="best")
        self._canvas.draw()
        self._log(f"Plotted overlay with {n} spectra")

    def _on_clear_overlay(self):
        self._overlay_spectra = []
        self._overlay_list.clear()
        self._log("Overlay cleared")

    # -- Deconvolution ------------------------------------------------------

    def _on_deconvolve(self):
        if not self._ensure_data():
            return
        profile = self._combo_deconv_profile.currentText()
        max_peaks = self._spin_deconv_max_peaks.value()
        max_iter = self._spin_deconv_iter.value()
        prominence = self._spin_prom.value()

        self._log(f"Running deconvolution: {profile}, max_peaks={max_peaks}, "
                  f"iter={max_iter}")
        try:
            curves, params, composite = auto_deconvolve(
                self._x, self._y, profile=profile,
                min_prominence=prominence,
                max_peaks=max_peaks,
                max_iter=max_iter)
        except Exception as exc:
            self._log(f"Deconvolution failed: {exc}")
            return

        self._deconv_curves = curves
        self._deconv_params = params
        self._fit_curves = [composite] + curves

        # Update peak table with fit parameters
        self._peak_table.setRowCount(0)
        self._peak_table.setColumnCount(6)
        self._peak_table.setHorizontalHeaderLabels(
            ["#", "Center", "Amplitude", "Width", "FWHM", "Area"])
        for i, p in enumerate(params):
            row = self._peak_table.rowCount()
            self._peak_table.insertRow(row)
            self._peak_table.setItem(row, 0, QTableWidgetItem(str(i + 1)))
            self._peak_table.setItem(row, 1, QTableWidgetItem(f"{p['center']:.4f}"))
            self._peak_table.setItem(row, 2, QTableWidgetItem(f"{p['amplitude']:.4f}"))
            self._peak_table.setItem(row, 3, QTableWidgetItem(f"{p['width']:.4f}"))
            self._peak_table.setItem(row, 4, QTableWidgetItem(f"{p['fwhm']:.4f}"))
            self._peak_table.setItem(row, 5, QTableWidgetItem(f"{p['area']:.4f}"))

        self._update_plot()
        self._log(f"Deconvolution complete: {len(curves)} peaks found")

    # -- Difference spectrum ------------------------------------------------

    def _on_set_reference(self):
        if self._x is None or self._y is None:
            self._log("No spectrum loaded to use as reference.")
            return
        self._ref_x = self._x.copy()
        self._ref_y = self._y.copy()
        name = os.path.basename(self._file_path) if self._file_path else "Current"
        self._lbl_ref.setText(f"Ref: {name} ({len(self._ref_x)} pts)")
        self._log("Reference spectrum set")

    def _on_difference(self):
        if not self._ensure_data():
            return
        if self._ref_x is None or self._ref_y is None:
            self._log("Set a reference spectrum first.")
            return
        # Interpolate reference to match current x-axis
        try:
            ref_interp = np.interp(self._x, self._ref_x, self._ref_y)
            self._y = self._y - ref_interp
            self._peaks_idx = None
            self._fit_curves = []
            self._update_plot()
            self._log("Difference spectrum computed (current - reference)")
        except Exception as exc:
            self._log(f"Difference computation failed: {exc}")

    # -- Report generation --------------------------------------------------

    def _on_generate_report(self):
        if not self._ensure_data():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save HTML Report", "spectrum_report.html",
            "HTML Files (*.html);;All Files (*)")
        if not path:
            return

        # Save spectrum image alongside report
        img_path = path.replace(".html", "_spectrum.png")
        try:
            self._figure.savefig(img_path, dpi=150, bbox_inches="tight",
                                 facecolor="white")
        except Exception:
            img_path = None

        peaks_x = self._x[self._peaks_idx] if self._peaks_idx is not None and len(self._peaks_idx) > 0 else None
        peaks_y = self._y[self._peaks_idx] if self._peaks_idx is not None and len(self._peaks_idx) > 0 else None

        fit_params = self._deconv_params if self._deconv_params else None

        html = generate_html_report(
            self._x, self._y, peaks_x, peaks_y, fit_params,
            spectrum_type=self._spectrum_type.value,
            image_path=os.path.basename(img_path) if img_path else None,
            title=f"{self._spectrum_type.value} Spectroscopy Report")

        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(html)
            self._log(f"Report saved to {path}")
        except Exception as exc:
            self._log(f"Report generation failed: {exc}")

    # -- Database lookup ----------------------------------------------------

    def _on_show_database(self):
        db_type = self._combo_db_type.currentIndex()

        self._peak_table.setRowCount(0)
        if db_type == 0:  # IR
            self._peak_table.setColumnCount(3)
            self._peak_table.setHorizontalHeaderLabels(
                ["Functional Group", "Wavenumber (cm-1)", "Intensity"])
            for name, wn, intensity in IR_PEAK_DATABASE:
                row = self._peak_table.rowCount()
                self._peak_table.insertRow(row)
                self._peak_table.setItem(row, 0, QTableWidgetItem(name))
                self._peak_table.setItem(row, 1, QTableWidgetItem(f"{wn:.0f}"))
                self._peak_table.setItem(row, 2, QTableWidgetItem(intensity))
        else:  # UV-Vis
            self._peak_table.setColumnCount(3)
            self._peak_table.setHorizontalHeaderLabels(
                ["Chromophore", "lambda_max (nm)", "epsilon"])
            for name, lam, eps in UVVIS_PEAK_DATABASE:
                row = self._peak_table.rowCount()
                self._peak_table.insertRow(row)
                self._peak_table.setItem(row, 0, QTableWidgetItem(name))
                self._peak_table.setItem(row, 1, QTableWidgetItem(f"{lam:.0f}"))
                self._peak_table.setItem(row, 2, QTableWidgetItem(f"{eps:.0f}"))

        self._log(f"Showing {'IR' if db_type == 0 else 'UV-Vis'} peak database "
                  f"({self._peak_table.rowCount()} entries)")

    def _on_reset(self):
        if self._x_raw is None:
            return
        self._x = self._x_raw.copy()
        self._y = self._y_raw.copy()
        self._peaks_idx = None
        self._fit_curves = []
        self._peak_table.setRowCount(0)
        self._log("Data reset to original")
        self._update_plot()

    def _on_export(self):
        data = self.export()
        if data is None:
            self._log("Nothing to export")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Data", "", "JSON Files (*.json);;All Files (*)")
        if not path:
            return
        try:
            with open(path, "w") as fh:
                json.dump(data, fh, indent=2)
            self._log(f"Exported to {path}")
        except Exception as exc:
            self._log(f"Export failed: {exc}")
