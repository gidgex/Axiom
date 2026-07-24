"""Curve Fitting - Nonlinear fitting, peak fitting, model selection, and export."""
import io
import csv
import json
import traceback
import datetime
from collections import OrderedDict

import numpy as np
from scipy.stats import norm as _norm
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSplitter, QPushButton,
    QComboBox, QTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QGroupBox, QFormLayout, QCheckBox,
    QFileDialog, QMessageBox, QHeaderView, QFrame, QGridLayout, QScrollArea,
    QSizePolicy, QApplication
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

try:
    from mpl_style import style_figure, style_axes
except ImportError:
    def style_figure(f): pass
    def style_axes(a): pass

from scipy.optimize import curve_fit, least_squares
from scipy.signal import find_peaks
from scipy.special import voigt_profile


# ---------------------------------------------------------------------------
# Built-in model definitions
# ---------------------------------------------------------------------------

def _linear(x, a, b):
    return a * x + b


def _polynomial(x, *coeffs):
    return np.polyval(coeffs, x)


def _exponential(x, a, b, c):
    return a * np.exp(b * x) + c


def _power_law(x, a, b, c):
    return a * np.power(np.maximum(x, 1e-30), b) + c


def _gaussian(x, amp, cen, wid):
    return amp * np.exp(-0.5 * ((x - cen) / wid) ** 2)


def _lorentzian(x, amp, cen, wid):
    return amp * wid ** 2 / ((x - cen) ** 2 + wid ** 2)


def _voigt(x, amp, cen, sigma, gamma):
    return amp * voigt_profile(x - cen, sigma, gamma)


def _sigmoid(x, a, b, c, d):
    return a / (1.0 + np.exp(-b * (x - c))) + d


def _sine(x, amp, freq, phase, offset):
    return amp * np.sin(2 * np.pi * freq * x + phase) + offset


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = OrderedDict([
    ("Linear", {
        "func": _linear,
        "params": ["a (slope)", "b (intercept)"],
        "guess": lambda x, y: [
            (y[-1] - y[0]) / (x[-1] - x[0] + 1e-30),
            y[0],
        ],
    }),
    ("Polynomial", {
        "func": _polynomial,
        "params": None,  # dynamic, set by degree
        "guess": None,
    }),
    ("Exponential", {
        "func": _exponential,
        "params": ["a (amplitude)", "b (rate)", "c (offset)"],
        "guess": lambda x, y: [np.ptp(y), 0.01, np.min(y)],
    }),
    ("Power Law", {
        "func": _power_law,
        "params": ["a (scale)", "b (exponent)", "c (offset)"],
        "guess": lambda x, y: [1.0, 1.0, 0.0],
    }),
    ("Gaussian", {
        "func": _gaussian,
        "params": ["amplitude", "center", "width (sigma)"],
        "guess": lambda x, y: [np.max(y) - np.min(y), x[np.argmax(y)], np.std(x) / 3],
    }),
    ("Lorentzian", {
        "func": _lorentzian,
        "params": ["amplitude", "center", "width (gamma)"],
        "guess": lambda x, y: [np.max(y) - np.min(y), x[np.argmax(y)], np.std(x) / 3],
    }),
    ("Voigt", {
        "func": _voigt,
        "params": ["amplitude", "center", "sigma", "gamma"],
        "guess": lambda x, y: [np.max(y), x[np.argmax(y)], np.std(x) / 4, np.std(x) / 4],
    }),
    ("Sigmoid / Logistic", {
        "func": _sigmoid,
        "params": ["a (max)", "b (steepness)", "c (midpoint)", "d (offset)"],
        "guess": lambda x, y: [np.ptp(y), 1.0, np.median(x), np.min(y)],
    }),
    ("Sine Wave", {
        "func": _sine,
        "params": ["amplitude", "frequency", "phase", "offset"],
        "guess": lambda x, y: [np.ptp(y) / 2, 1.0 / (np.ptp(x) + 1e-30), 0.0, np.mean(y)],
    }),
    ("Custom Expression", {
        "func": None,
        "params": None,
        "guess": None,
    }),
])


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class CurveFittingWidget(QWidget):
    """Full-featured curve fitting tool for the QuantumRes scientific suite."""

    fit_completed = pyqtSignal(dict)

    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self._logger = None

        # Data arrays
        self.x_data = None
        self.y_data = None
        self.y_err = None

        # Fit state
        self.fit_result = None
        self.fit_popt = None
        self.fit_pcov = None
        self.fit_func = None
        self.fit_peaks = []

        self._init_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_logger(self, fn):
        """Set an external logging callback ``fn(message: str)``."""
        self._logger = fn

    def load_file(self, path: str):
        """Load X,Y[,Yerr] data from a file (CSV, TSV, or whitespace)."""
        self._log(f"Loading file: {path}")
        try:
            data = np.loadtxt(path, delimiter=None, comments="#", unpack=False)
            if data.ndim == 1:
                self.y_data = data
                self.x_data = np.arange(len(data), dtype=float)
                self.y_err = None
            elif data.shape[1] >= 3:
                self.x_data = data[:, 0]
                self.y_data = data[:, 1]
                self.y_err = data[:, 2]
            else:
                self.x_data = data[:, 0]
                self.y_data = data[:, 1]
                self.y_err = None
            self._populate_table_from_arrays()
            self._plot_raw_data()
            self._log(f"Loaded {len(self.x_data)} data points from {path}")
        except Exception as exc:
            self._log(f"Error loading file: {exc}")
            QMessageBox.warning(self, "Load Error", str(exc))

    def run(self):
        """Execute the current fit programmatically (same as pressing *Fit*)."""
        self._do_fit()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Horizontal)

        # ---- Left panel: controls ----
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(6)

        # Data input section
        data_group = QGroupBox("Data Input")
        data_lay = QVBoxLayout(data_group)

        btn_row = QHBoxLayout()
        self.btn_load = QPushButton("Load File...")
        self.btn_load.clicked.connect(self._on_load_file)
        btn_row.addWidget(self.btn_load)

        self.btn_paste = QPushButton("Paste Data")
        self.btn_paste.clicked.connect(self._on_paste_data)
        btn_row.addWidget(self.btn_paste)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self._on_clear_data)
        btn_row.addWidget(self.btn_clear)
        data_lay.addLayout(btn_row)

        self.data_table = QTableWidget(0, 3)
        self.data_table.setHorizontalHeaderLabels(["X", "Y", "Y Error"])
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.data_table.setMinimumHeight(120)
        data_lay.addWidget(self.data_table)

        left_layout.addWidget(data_group)

        # Model selection
        model_group = QGroupBox("Model")
        model_lay = QFormLayout(model_group)

        self.model_combo = QComboBox()
        self.model_combo.addItems(list(MODEL_REGISTRY.keys()))
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        model_lay.addRow("Model:", self.model_combo)

        self.poly_degree_spin = QSpinBox()
        self.poly_degree_spin.setRange(2, 20)
        self.poly_degree_spin.setValue(3)
        self.poly_degree_label = QLabel("Degree:")
        model_lay.addRow(self.poly_degree_label, self.poly_degree_spin)
        self.poly_degree_label.hide()
        self.poly_degree_spin.hide()

        self.custom_expr_edit = QLineEdit()
        self.custom_expr_edit.setPlaceholderText("e.g.  a*exp(-b*x) + c*sin(d*x)")
        self.custom_expr_label = QLabel("f(x) =")
        model_lay.addRow(self.custom_expr_label, self.custom_expr_edit)
        self.custom_expr_label.hide()
        self.custom_expr_edit.hide()

        self.custom_params_edit = QLineEdit()
        self.custom_params_edit.setPlaceholderText("a, b, c, d")
        self.custom_params_label = QLabel("Params:")
        model_lay.addRow(self.custom_params_label, self.custom_params_edit)
        self.custom_params_label.hide()
        self.custom_params_edit.hide()

        left_layout.addWidget(model_group)

        # Initial parameter guesses
        guess_group = QGroupBox("Initial Guesses")
        self.guess_layout = QFormLayout(guess_group)
        self.auto_guess_cb = QCheckBox("Auto-guess")
        self.auto_guess_cb.setChecked(True)
        self.guess_layout.addRow(self.auto_guess_cb)
        self.guess_edits = []
        left_layout.addWidget(guess_group)

        # Fit options
        opts_group = QGroupBox("Fit Options")
        opts_lay = QFormLayout(opts_group)

        self.method_combo = QComboBox()
        self.method_combo.addItems(["Levenberg-Marquardt (curve_fit)", "Trust Region (least_squares)"])
        opts_lay.addRow("Method:", self.method_combo)

        self.maxiter_spin = QSpinBox()
        self.maxiter_spin.setRange(100, 100000)
        self.maxiter_spin.setValue(5000)
        self.maxiter_spin.setSingleStep(500)
        opts_lay.addRow("Max Iterations:", self.maxiter_spin)

        self.weight_cb = QCheckBox("Use Y errors as weights")
        self.weight_cb.setChecked(True)
        opts_lay.addRow(self.weight_cb)

        left_layout.addWidget(opts_group)

        # Action buttons
        action_row = QHBoxLayout()
        self.btn_fit = QPushButton("Fit")
        self.btn_fit.setStyleSheet("background-color:#4a90d9; color:white; font-weight:bold; padding:6px;")
        self.btn_fit.clicked.connect(self._do_fit)
        action_row.addWidget(self.btn_fit)

        self.btn_peaks = QPushButton("Find Peaks")
        self.btn_peaks.clicked.connect(self._find_peaks)
        action_row.addWidget(self.btn_peaks)

        self.btn_multi_peak = QPushButton("Multi-Peak Fit")
        self.btn_multi_peak.clicked.connect(self._multi_peak_fit)
        action_row.addWidget(self.btn_multi_peak)
        left_layout.addLayout(action_row)

        export_row = QHBoxLayout()
        self.btn_export = QPushButton("Export Results...")
        self.btn_export.clicked.connect(self._export_results)
        export_row.addWidget(self.btn_export)

        self.btn_copy_plot = QPushButton("Copy Plot")
        self.btn_copy_plot.setToolTip("Copy current plot to clipboard as image")
        self.btn_copy_plot.clicked.connect(self._copy_plot_to_clipboard)
        export_row.addWidget(self.btn_copy_plot)
        left_layout.addLayout(export_row)

        # Advanced fitting actions
        adv_group = QGroupBox("Advanced Fitting")
        adv_lay = QVBoxLayout(adv_group)

        adv_row1 = QHBoxLayout()
        self.btn_conf_bands = QPushButton("Confidence Bands")
        self.btn_conf_bands.clicked.connect(self._plot_confidence_bands)
        adv_row1.addWidget(self.btn_conf_bands)

        self.btn_resid_diag = QPushButton("Residual Diagnostics")
        self.btn_resid_diag.clicked.connect(self._residual_diagnostics)
        adv_row1.addWidget(self.btn_resid_diag)
        adv_lay.addLayout(adv_row1)

        adv_row2 = QHBoxLayout()
        self.btn_model_compare = QPushButton("Compare Models")
        self.btn_model_compare.clicked.connect(self._compare_models)
        adv_row2.addWidget(self.btn_model_compare)

        self.btn_batch_fit = QPushButton("Batch Fit")
        self.btn_batch_fit.clicked.connect(self._batch_fit)
        adv_row2.addWidget(self.btn_batch_fit)
        adv_lay.addLayout(adv_row2)

        adv_row3 = QHBoxLayout()
        self.btn_global_fit = QPushButton("Global Fit")
        self.btn_global_fit.clicked.connect(self._global_fit)
        adv_row3.addWidget(self.btn_global_fit)

        self.btn_gen_report = QPushButton("Generate Report")
        self.btn_gen_report.clicked.connect(self._generate_fit_report)
        adv_row3.addWidget(self.btn_gen_report)
        adv_lay.addLayout(adv_row3)

        left_layout.addWidget(adv_group)

        # Results
        result_group = QGroupBox("Fit Results")
        result_lay = QVBoxLayout(result_group)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFont(QFont("Consolas", 9))
        self.result_text.setMinimumHeight(100)
        result_lay.addWidget(self.result_text)
        left_layout.addWidget(result_group)

        left_layout.addStretch()

        # ---- Right panel: plots ----
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(2, 2, 2, 2)

        self.figure = Figure(figsize=(7, 5), dpi=100, tight_layout=True)
        style_figure(self.figure)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)

        # Set up axes: main + residuals
        gs = self.figure.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
        self.ax_main = self.figure.add_subplot(gs[0])
        self.ax_resid = self.figure.add_subplot(gs[1], sharex=self.ax_main)
        self.ax_main.tick_params(labelbottom=False)
        self.ax_resid.set_xlabel("X")
        self.ax_resid.set_ylabel("Residuals")
        self.ax_main.set_ylabel("Y")

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter)

        # Trigger initial model selection state
        self._on_model_changed(self.model_combo.currentText())

    # ------------------------------------------------------------------
    # Data handling
    # ------------------------------------------------------------------

    def _on_load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "",
            "Data Files (*.csv *.tsv *.txt *.dat);;All Files (*)"
        )
        if path:
            self.load_file(path)

    def _on_paste_data(self):
        clipboard = QApplication.clipboard()
        text = clipboard.text()
        if not text.strip():
            QMessageBox.information(self, "Paste", "Clipboard is empty.")
            return
        self._parse_text_data(text)

    def _on_clear_data(self):
        self.x_data = None
        self.y_data = None
        self.y_err = None
        self.data_table.setRowCount(0)
        self.fit_result = None
        self.result_text.clear()
        self._clear_plots()

    def _parse_text_data(self, text):
        """Parse pasted or typed text into x, y, [y_err] arrays."""
        lines = text.strip().splitlines()
        rows = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Try comma, tab, whitespace
            for sep in [",", "\t", None]:
                parts = line.split(sep) if sep else line.split()
                if len(parts) >= 2:
                    try:
                        vals = [float(p.strip()) for p in parts[:3]]
                        rows.append(vals)
                        break
                    except ValueError:
                        continue
        if not rows:
            QMessageBox.warning(self, "Parse Error", "Could not parse any numeric X,Y pairs.")
            return
        arr = np.array(rows)
        self.x_data = arr[:, 0]
        self.y_data = arr[:, 1]
        self.y_err = arr[:, 2] if arr.shape[1] >= 3 else None
        self._populate_table_from_arrays()
        self._plot_raw_data()
        self._log(f"Parsed {len(self.x_data)} data points from clipboard.")

    def _populate_table_from_arrays(self):
        n = len(self.x_data)
        self.data_table.setRowCount(n)
        for i in range(n):
            self.data_table.setItem(i, 0, QTableWidgetItem(f"{self.x_data[i]:.6g}"))
            self.data_table.setItem(i, 1, QTableWidgetItem(f"{self.y_data[i]:.6g}"))
            if self.y_err is not None:
                self.data_table.setItem(i, 2, QTableWidgetItem(f"{self.y_err[i]:.6g}"))

    def _read_table_data(self):
        """Read data back from the table (user may have edited cells)."""
        rows = self.data_table.rowCount()
        if rows == 0:
            return False
        xs, ys, es = [], [], []
        for i in range(rows):
            try:
                x_item = self.data_table.item(i, 0)
                y_item = self.data_table.item(i, 1)
                if x_item is None or y_item is None:
                    continue
                xs.append(float(x_item.text()))
                ys.append(float(y_item.text()))
                e_item = self.data_table.item(i, 2)
                if e_item and e_item.text().strip():
                    es.append(float(e_item.text()))
            except ValueError:
                continue
        if len(xs) < 2:
            return False
        self.x_data = np.array(xs)
        self.y_data = np.array(ys)
        self.y_err = np.array(es) if len(es) == len(xs) else None
        return True

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------

    def _on_model_changed(self, name):
        is_poly = name == "Polynomial"
        is_custom = name == "Custom Expression"

        self.poly_degree_label.setVisible(is_poly)
        self.poly_degree_spin.setVisible(is_poly)
        self.custom_expr_label.setVisible(is_custom)
        self.custom_expr_edit.setVisible(is_custom)
        self.custom_params_label.setVisible(is_custom)
        self.custom_params_edit.setVisible(is_custom)

        self._rebuild_guess_fields()

    def _rebuild_guess_fields(self):
        """Rebuild the initial-guess input fields for the current model."""
        # Remove old fields (skip the auto-guess checkbox at row 0)
        while self.guess_layout.rowCount() > 1:
            self.guess_layout.removeRow(1)
        self.guess_edits.clear()

        model_name = self.model_combo.currentText()
        params = self._current_param_names()
        if params is None:
            return
        for pname in params:
            edit = QDoubleSpinBox()
            edit.setRange(-1e15, 1e15)
            edit.setDecimals(6)
            edit.setValue(1.0)
            self.guess_layout.addRow(pname + ":", edit)
            self.guess_edits.append(edit)

    def _current_param_names(self):
        name = self.model_combo.currentText()
        if name == "Polynomial":
            deg = self.poly_degree_spin.value()
            return [f"c{i}" for i in range(deg + 1)]
        if name == "Custom Expression":
            raw = self.custom_params_edit.text().strip()
            if raw:
                return [p.strip() for p in raw.split(",")]
            return None
        info = MODEL_REGISTRY.get(name, {})
        return info.get("params")

    def _build_model_func(self):
        """Return (callable, param_names, initial_guesses)."""
        name = self.model_combo.currentText()

        if name == "Polynomial":
            deg = self.poly_degree_spin.value()
            params = [f"c{i}" for i in range(deg + 1)]

            def poly_func(x, *coeffs):
                return np.polyval(coeffs, x)

            if self.auto_guess_cb.isChecked() and self.x_data is not None:
                p0 = list(np.polyfit(self.x_data, self.y_data, deg))
            else:
                p0 = [e.value() for e in self.guess_edits] if self.guess_edits else [1.0] * (deg + 1)
            return poly_func, params, p0

        if name == "Custom Expression":
            expr = self.custom_expr_edit.text().strip()
            raw_params = self.custom_params_edit.text().strip()
            if not expr or not raw_params:
                raise ValueError("Custom model requires both an expression and parameter names.")
            param_names = [p.strip() for p in raw_params.split(",")]
            safe_ns = {"np": np, "exp": np.exp, "sin": np.sin, "cos": np.cos,
                        "log": np.log, "sqrt": np.sqrt, "abs": np.abs,
                        "pi": np.pi, "e": np.e, "power": np.power, "tanh": np.tanh}
            code = compile(expr, "<custom>", "eval")

            def custom_func(x, *args):
                ns = dict(safe_ns)
                ns["x"] = x
                for k, v in zip(param_names, args):
                    ns[k] = v
                return eval(code, {"__builtins__": {}}, ns)

            p0 = [e.value() for e in self.guess_edits] if self.guess_edits else [1.0] * len(param_names)
            return custom_func, param_names, p0

        info = MODEL_REGISTRY[name]
        func = info["func"]
        param_names = info["params"]
        if self.auto_guess_cb.isChecked() and info["guess"] and self.x_data is not None:
            p0 = list(info["guess"](self.x_data, self.y_data))
        else:
            p0 = [e.value() for e in self.guess_edits] if self.guess_edits else [1.0] * len(param_names)
        return func, param_names, p0

    # ------------------------------------------------------------------
    # Fitting engine
    # ------------------------------------------------------------------

    def _do_fit(self):
        if not self._read_table_data():
            QMessageBox.warning(self, "No Data", "Please load or paste data before fitting.")
            return

        try:
            func, param_names, p0 = self._build_model_func()
        except Exception as exc:
            self._log(f"Model build error: {exc}")
            QMessageBox.warning(self, "Model Error", str(exc))
            return

        sigma = self.y_err if (self.y_err is not None and self.weight_cb.isChecked()) else None
        method = self.method_combo.currentText()

        self._log(f"Fitting model '{self.model_combo.currentText()}' with {len(p0)} parameters ...")

        try:
            if "curve_fit" in method:
                popt, pcov = curve_fit(
                    func, self.x_data, self.y_data,
                    p0=p0, sigma=sigma, absolute_sigma=True,
                    maxfev=self.maxiter_spin.value(),
                )
            else:
                # Use least_squares via a wrapper
                def residual_fn(params):
                    model_y = func(self.x_data, *params)
                    if sigma is not None:
                        return (self.y_data - model_y) / sigma
                    return self.y_data - model_y

                result = least_squares(
                    residual_fn, p0,
                    method="trf",
                    max_nfev=self.maxiter_spin.value(),
                )
                popt = result.x
                # Approximate covariance from Jacobian
                J = result.jac
                try:
                    pcov = np.linalg.inv(J.T @ J) * (result.fun @ result.fun) / max(len(self.x_data) - len(popt), 1)
                except np.linalg.LinAlgError:
                    pcov = np.full((len(popt), len(popt)), np.inf)

            self.fit_popt = popt
            self.fit_pcov = pcov
            self.fit_func = func

            self._compute_statistics(func, popt, pcov, param_names, sigma)
            self._plot_fit(func, popt)
            self._log("Fit completed successfully.")

        except Exception as exc:
            tb = traceback.format_exc()
            self._log(f"Fit failed: {exc}\n{tb}")
            QMessageBox.warning(self, "Fit Error", f"Fit did not converge:\n{exc}")

    def _compute_statistics(self, func, popt, pcov, param_names, sigma):
        """Compute fit statistics and display in the results panel."""
        y_fit = func(self.x_data, *popt)
        residuals = self.y_data - y_fit
        n = len(self.y_data)
        p = len(popt)

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((self.y_data - np.mean(self.y_data)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")

        dof = max(n - p, 1)
        if sigma is not None:
            chi2 = np.sum((residuals / sigma) ** 2)
        else:
            chi2 = ss_res / (np.var(self.y_data) + 1e-30)
        reduced_chi2 = chi2 / dof

        perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else np.full(p, float("nan"))

        aic = n * np.log(ss_res / n + 1e-30) + 2 * p
        bic = n * np.log(ss_res / n + 1e-30) + p * np.log(n)

        lines = []
        lines.append(f"Model: {self.model_combo.currentText()}")
        lines.append(f"{'='*45}")
        for i, name in enumerate(param_names):
            lines.append(f"  {name:>20s} = {popt[i]:>14.6g} +/- {perr[i]:.6g}")
        lines.append(f"{'='*45}")
        lines.append(f"  R-squared        = {r_squared:.8f}")
        lines.append(f"  Chi-square       = {chi2:.6g}")
        lines.append(f"  Reduced Chi-sq   = {reduced_chi2:.6g}")
        lines.append(f"  AIC              = {aic:.4f}")
        lines.append(f"  BIC              = {bic:.4f}")
        lines.append(f"  Residual SS      = {ss_res:.6g}")
        lines.append(f"  Points / Params  = {n} / {p}")

        self.result_text.setPlainText("\n".join(lines))

        self.fit_result = {
            "model": self.model_combo.currentText(),
            "params": {name: float(popt[i]) for i, name in enumerate(param_names)},
            "uncertainties": {name: float(perr[i]) for i, name in enumerate(param_names)},
            "r_squared": float(r_squared),
            "chi_square": float(chi2),
            "reduced_chi_square": float(reduced_chi2),
            "aic": float(aic),
            "bic": float(bic),
            "residual_ss": float(ss_res),
            "n_points": n,
            "n_params": p,
        }
        self.fit_completed.emit(self.fit_result)

    # ------------------------------------------------------------------
    # Peak finding and multi-peak fitting
    # ------------------------------------------------------------------

    def _find_peaks(self):
        if self.y_data is None:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return
        prominence = np.ptp(self.y_data) * 0.05
        peaks, properties = find_peaks(self.y_data, prominence=prominence, distance=3)
        self.fit_peaks = peaks
        if len(peaks) == 0:
            self._log("No peaks found.")
            QMessageBox.information(self, "Peaks", "No peaks detected above threshold.")
            return

        lines = [f"Found {len(peaks)} peak(s):"]
        for i, idx in enumerate(peaks):
            lines.append(f"  Peak {i+1}: X={self.x_data[idx]:.6g}, Y={self.y_data[idx]:.6g}")
        self.result_text.setPlainText("\n".join(lines))
        self._log(f"Found {len(peaks)} peaks.")

        # Plot peaks on main axes
        self.ax_main.plot(self.x_data[peaks], self.y_data[peaks], "rv", markersize=10, label="Peaks")
        self.ax_main.legend()
        self.canvas.draw_idle()

    def _multi_peak_fit(self):
        """Fit a sum of Gaussians, one per detected peak."""
        if self.y_data is None:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return
        if len(self.fit_peaks) == 0:
            self._find_peaks()
        if len(self.fit_peaks) == 0:
            return

        n_peaks = len(self.fit_peaks)

        def multi_gauss(x, *params):
            y = np.zeros_like(x)
            for i in range(n_peaks):
                amp = params[3 * i]
                cen = params[3 * i + 1]
                wid = params[3 * i + 2]
                y += amp * np.exp(-0.5 * ((x - cen) / wid) ** 2)
            return y + params[-1]  # baseline offset

        p0 = []
        for idx in self.fit_peaks:
            p0.extend([self.y_data[idx] - np.min(self.y_data), self.x_data[idx], np.std(self.x_data) / (2 * n_peaks)])
        p0.append(np.min(self.y_data))

        sigma = self.y_err if (self.y_err is not None and self.weight_cb.isChecked()) else None

        try:
            popt, pcov = curve_fit(multi_gauss, self.x_data, self.y_data, p0=p0,
                                   sigma=sigma, absolute_sigma=True, maxfev=self.maxiter_spin.value())
            perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else np.full(len(popt), float("nan"))

            lines = [f"Multi-Peak Gaussian Fit ({n_peaks} peaks + baseline)"]
            lines.append("=" * 50)
            for i in range(n_peaks):
                lines.append(f"  Peak {i+1}:")
                lines.append(f"    amplitude = {popt[3*i]:.6g} +/- {perr[3*i]:.6g}")
                lines.append(f"    center    = {popt[3*i+1]:.6g} +/- {perr[3*i+1]:.6g}")
                lines.append(f"    width     = {popt[3*i+2]:.6g} +/- {perr[3*i+2]:.6g}")
            lines.append(f"  Baseline offset = {popt[-1]:.6g} +/- {perr[-1]:.6g}")

            y_fit = multi_gauss(self.x_data, *popt)
            ss_res = np.sum((self.y_data - y_fit) ** 2)
            ss_tot = np.sum((self.y_data - np.mean(self.y_data)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")
            lines.append(f"  R-squared = {r2:.8f}")

            self.result_text.setPlainText("\n".join(lines))
            self._log(f"Multi-peak fit converged with R^2 = {r2:.6f}")

            # Plot multi-peak fit
            self.fit_popt = popt
            self.fit_pcov = pcov
            self.fit_func = multi_gauss
            self._plot_fit(multi_gauss, popt, label="Multi-Peak Fit")

            # Also draw individual peaks
            x_smooth = np.linspace(self.x_data.min(), self.x_data.max(), 500)
            for i in range(n_peaks):
                amp, cen, wid = popt[3*i], popt[3*i+1], popt[3*i+2]
                y_peak = amp * np.exp(-0.5 * ((x_smooth - cen) / wid) ** 2) + popt[-1]
                self.ax_main.plot(x_smooth, y_peak, "--", alpha=0.5, label=f"Peak {i+1}")
            self.ax_main.legend(fontsize=8)
            self.canvas.draw_idle()

        except Exception as exc:
            self._log(f"Multi-peak fit failed: {exc}")
            QMessageBox.warning(self, "Multi-Peak Fit Error", str(exc))

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _clear_plots(self):
        self.ax_main.cla()
        self.ax_resid.cla()
        self.ax_main.set_ylabel("Y")
        self.ax_resid.set_xlabel("X")
        self.ax_resid.set_ylabel("Residuals")
        self.ax_main.tick_params(labelbottom=False)
        self.canvas.draw_idle()

    def _plot_raw_data(self):
        self._clear_plots()
        if self.x_data is None:
            return
        if self.y_err is not None:
            self.ax_main.errorbar(self.x_data, self.y_data, yerr=self.y_err,
                                  fmt="o", ms=4, capsize=2, color="#4a90d9", label="Data")
        else:
            self.ax_main.plot(self.x_data, self.y_data, "o", ms=4, color="#4a90d9", label="Data")
        self.ax_main.legend(fontsize=8)
        self.canvas.draw_idle()

    def _plot_fit(self, func, popt, label="Fit"):
        self._plot_raw_data()
        x_smooth = np.linspace(self.x_data.min(), self.x_data.max(), 500)
        y_smooth = func(x_smooth, *popt)
        self.ax_main.plot(x_smooth, y_smooth, "-", color="#e74c3c", lw=2, label=label)
        self.ax_main.legend(fontsize=8)

        # Residuals
        y_fit = func(self.x_data, *popt)
        residuals = self.y_data - y_fit
        self.ax_resid.cla()
        self.ax_resid.axhline(0, color="gray", ls="--", lw=0.8)
        self.ax_resid.stem(self.x_data, residuals, linefmt="C2-", markerfmt="C2o", basefmt=" ", use_line_collection=True)
        self.ax_resid.set_xlabel("X")
        self.ax_resid.set_ylabel("Residuals")

        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_results(self):
        if self.fit_result is None:
            QMessageBox.information(self, "Export", "No fit results to export. Run a fit first.")
            return
        path, filt = QFileDialog.getSaveFileName(
            self, "Export Fit Results", "fit_results",
            "JSON (*.json);;CSV (*.csv);;Text (*.txt);;All Files (*)"
        )
        if not path:
            return
        try:
            if path.endswith(".json"):
                with open(path, "w") as f:
                    json.dump(self.fit_result, f, indent=2)
            elif path.endswith(".csv"):
                self._export_csv(path)
            else:
                with open(path, "w") as f:
                    f.write(self.result_text.toPlainText())
            self._log(f"Results exported to {path}")
        except Exception as exc:
            self._log(f"Export error: {exc}")
            QMessageBox.warning(self, "Export Error", str(exc))

    def _export_csv(self, path):
        """Export fit results and residuals as CSV."""
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Model", self.fit_result["model"]])
            writer.writerow(["R-squared", self.fit_result["r_squared"]])
            writer.writerow(["Chi-square", self.fit_result["chi_square"]])
            writer.writerow(["Reduced Chi-sq", self.fit_result["reduced_chi_square"]])
            writer.writerow(["AIC", self.fit_result["aic"]])
            writer.writerow(["BIC", self.fit_result["bic"]])
            writer.writerow([])
            writer.writerow(["Parameter", "Value", "Uncertainty"])
            for name in self.fit_result["params"]:
                writer.writerow([name, self.fit_result["params"][name],
                                 self.fit_result["uncertainties"][name]])
            writer.writerow([])
            writer.writerow(["X", "Y_data", "Y_fit", "Residual"])
            if self.fit_func is not None and self.fit_popt is not None:
                y_fit = self.fit_func(self.x_data, *self.fit_popt)
                for i in range(len(self.x_data)):
                    writer.writerow([self.x_data[i], self.y_data[i],
                                     y_fit[i], self.y_data[i] - y_fit[i]])

    # ------------------------------------------------------------------
    # Confidence / Prediction Bands
    # ------------------------------------------------------------------

    def _plot_confidence_bands(self):
        """Plot confidence and prediction intervals around the current fit."""
        if self.fit_func is None or self.fit_popt is None or self.fit_pcov is None:
            QMessageBox.information(self, "No Fit", "Run a fit first.")
            return
        if self.x_data is None:
            return

        try:
            x_smooth = np.linspace(self.x_data.min(), self.x_data.max(), 300)
            y_fit = self.fit_func(x_smooth, *self.fit_popt)
            n = len(self.x_data)
            p = len(self.fit_popt)
            dof = max(n - p, 1)
            y_data_fit = self.fit_func(self.x_data, *self.fit_popt)
            mse = np.sum((self.y_data - y_data_fit) ** 2) / dof

            # Compute Jacobian numerically for uncertainty propagation
            eps = 1e-8
            J = np.zeros((len(x_smooth), p))
            for i in range(p):
                params_up = self.fit_popt.copy()
                params_up[i] += eps
                J[:, i] = (self.fit_func(x_smooth, *params_up) - y_fit) / eps

            # Confidence band: uncertainty in the mean
            cov_y = J @ self.fit_pcov @ J.T
            se_conf = np.sqrt(np.maximum(np.diag(cov_y), 0))
            t_val = 1.96  # ~95% CI

            # Prediction band: adds residual variance
            se_pred = np.sqrt(se_conf ** 2 + mse)

            self._plot_raw_data()
            self.ax_main.plot(x_smooth, y_fit, "-", color="#e74c3c", lw=2, label="Fit")
            self.ax_main.fill_between(x_smooth, y_fit - t_val * se_conf, y_fit + t_val * se_conf,
                                      alpha=0.3, color="#e74c3c", label="95% Confidence")
            self.ax_main.fill_between(x_smooth, y_fit - t_val * se_pred, y_fit + t_val * se_pred,
                                      alpha=0.15, color="#ff7f0e", label="95% Prediction")
            self.ax_main.legend(fontsize=8)
            self.canvas.draw_idle()
            self._log("Plotted confidence and prediction bands")
        except Exception as exc:
            self._log(f"Confidence band error: {exc}")
            QMessageBox.warning(self, "Error", str(exc))

    # ------------------------------------------------------------------
    # Residual Diagnostics
    # ------------------------------------------------------------------

    def _residual_diagnostics(self):
        """Generate residual diagnostic plots: histogram, QQ plot, ACF."""
        if self.fit_func is None or self.fit_popt is None:
            QMessageBox.information(self, "No Fit", "Run a fit first.")
            return
        try:
            y_fit = self.fit_func(self.x_data, *self.fit_popt)
            residuals = self.y_data - y_fit

            dlg = QDialog(self)
            dlg.setWindowTitle("Residual Diagnostics")
            dlg.setMinimumSize(800, 500)
            lay = QVBoxLayout(dlg)

            fig = Figure(figsize=(10, 6), tight_layout=True)
            style_figure(fig)
            canvas = FigureCanvas(fig)

            # 1) Residuals vs fitted
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.scatter(y_fit, residuals, s=10, alpha=0.7)
            ax1.axhline(0, color="gray", ls="--", lw=0.8)
            ax1.set_xlabel("Fitted Values")
            ax1.set_ylabel("Residuals")
            ax1.set_title("Residuals vs Fitted")

            # 2) Histogram of residuals
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.hist(residuals, bins='auto', density=True, alpha=0.7, color="#4a90d9", edgecolor="white")
            # Overlay normal distribution
            mu, std = np.mean(residuals), np.std(residuals)
            x_norm = np.linspace(residuals.min(), residuals.max(), 100)
            ax2.plot(x_norm, _norm.pdf(x_norm, mu, std), 'r-', lw=1.5)
            ax2.set_xlabel("Residual")
            ax2.set_ylabel("Density")
            ax2.set_title("Residual Histogram")

            # 3) QQ plot
            ax3 = fig.add_subplot(2, 2, 3)
            sorted_res = np.sort(residuals)
            n = len(sorted_res)
            theoretical_q = _norm.ppf((np.arange(1, n + 1) - 0.5) / n)
            ax3.scatter(theoretical_q, sorted_res, s=10, alpha=0.7)
            lims = [min(theoretical_q.min(), sorted_res.min()),
                    max(theoretical_q.max(), sorted_res.max())]
            ax3.plot(lims, lims, "r--", lw=0.8)
            ax3.set_xlabel("Theoretical Quantiles")
            ax3.set_ylabel("Sample Quantiles")
            ax3.set_title("Q-Q Plot")

            # 4) Autocorrelation
            ax4 = fig.add_subplot(2, 2, 4)
            max_lag = min(40, len(residuals) - 1)
            acf = np.correlate(residuals - mu, residuals - mu, mode='full')
            acf = acf[len(acf) // 2:]
            acf = acf / acf[0]
            ax4.bar(range(max_lag + 1), acf[:max_lag + 1], width=0.6, color="#2ca02c", alpha=0.7)
            ci = 1.96 / np.sqrt(n)
            ax4.axhline(ci, color="red", ls="--", lw=0.8)
            ax4.axhline(-ci, color="red", ls="--", lw=0.8)
            ax4.set_xlabel("Lag")
            ax4.set_ylabel("ACF")
            ax4.set_title("Autocorrelation")

            lay.addWidget(canvas)
            btn = QPushButton("Close")
            btn.clicked.connect(dlg.accept)
            lay.addWidget(btn)
            dlg.exec_()
            self._log("Generated residual diagnostic plots")
        except Exception as exc:
            self._log(f"Residual diagnostics error: {exc}")
            QMessageBox.warning(self, "Error", str(exc))

    # ------------------------------------------------------------------
    # Model Comparison
    # ------------------------------------------------------------------

    def _compare_models(self):
        """Fit multiple models to the data and rank by AIC/BIC."""
        if not self._read_table_data():
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return

        sigma = self.y_err if (self.y_err is not None and self.weight_cb.isChecked()) else None
        results = []
        models_to_try = ["Linear", "Exponential", "Gaussian", "Lorentzian",
                         "Sigmoid / Logistic", "Power Law", "Sine Wave"]

        for model_name in models_to_try:
            if model_name not in MODEL_REGISTRY:
                continue
            info = MODEL_REGISTRY[model_name]
            func = info["func"]
            param_names = info["params"]
            if func is None or param_names is None:
                continue
            try:
                if info["guess"] and self.x_data is not None:
                    p0 = list(info["guess"](self.x_data, self.y_data))
                else:
                    p0 = [1.0] * len(param_names)
                popt, pcov = curve_fit(func, self.x_data, self.y_data, p0=p0,
                                       sigma=sigma, absolute_sigma=True, maxfev=5000)
                y_fit = func(self.x_data, *popt)
                residuals = self.y_data - y_fit
                n = len(self.y_data)
                p = len(popt)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((self.y_data - np.mean(self.y_data)) ** 2)
                r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")
                aic = n * np.log(ss_res / n + 1e-30) + 2 * p
                bic = n * np.log(ss_res / n + 1e-30) + p * np.log(n)
                results.append({
                    "model": model_name, "r2": r2, "aic": aic, "bic": bic,
                    "n_params": p, "ss_res": ss_res, "popt": popt, "func": func,
                })
            except Exception:
                continue

        if not results:
            QMessageBox.information(self, "Comparison", "No models converged.")
            return

        results.sort(key=lambda r: r["aic"])
        lines = [f"{'Model':<22s} {'R-sq':>10s} {'AIC':>12s} {'BIC':>12s} {'Params':>7s}"]
        lines.append("-" * 65)
        for r in results:
            lines.append(f"{r['model']:<22s} {r['r2']:>10.6f} {r['aic']:>12.2f} {r['bic']:>12.2f} {r['n_params']:>7d}")
        lines.append(f"\nBest model by AIC: {results[0]['model']}")
        self.result_text.setPlainText("\n".join(lines))

        # Plot best fit
        best = results[0]
        self._plot_fit(best["func"], best["popt"], label=f"Best: {best['model']}")
        self._log(f"Model comparison: best = {best['model']} (AIC={best['aic']:.2f})")

    # ------------------------------------------------------------------
    # Batch Fitting
    # ------------------------------------------------------------------

    def _batch_fit(self):
        """Load multiple data files and fit the same model to each, compare parameters."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Multiple Data Files", "",
            "Data Files (*.csv *.tsv *.txt *.dat);;All Files (*)")
        if not paths:
            return

        try:
            func, param_names, p0_template = self._build_model_func()
        except Exception as exc:
            QMessageBox.warning(self, "Model Error", str(exc))
            return

        results = []
        for path in paths:
            try:
                data = np.loadtxt(path, delimiter=None, comments="#")
                if data.ndim == 1:
                    y = data
                    x = np.arange(len(data), dtype=float)
                else:
                    x, y = data[:, 0], data[:, 1]
                popt, pcov = curve_fit(func, x, y, p0=p0_template, maxfev=5000)
                perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else np.full(len(popt), float("nan"))
                y_fit = func(x, *popt)
                ss_res = np.sum((y - y_fit) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")
                import os
                results.append({"file": os.path.basename(path), "popt": popt,
                                "perr": perr, "r2": r2})
            except Exception as exc:
                self._log(f"Batch fit skip {path}: {exc}")
                continue

        if not results:
            QMessageBox.information(self, "Batch Fit", "No files were successfully fit.")
            return

        lines = [f"Batch Fit Results ({len(results)} files)"]
        lines.append("=" * 70)
        header = f"{'File':<25s} " + " ".join(f"{p:>12s}" for p in param_names) + f" {'R-sq':>10s}"
        lines.append(header)
        lines.append("-" * len(header))
        for r in results:
            vals = " ".join(f"{v:>12.5g}" for v in r["popt"])
            lines.append(f"{r['file']:<25s} {vals} {r['r2']:>10.6f}")

        # Summary statistics
        all_popt = np.array([r["popt"] for r in results])
        lines.append("\n--- Parameter Statistics ---")
        for i, p in enumerate(param_names):
            col = all_popt[:, i]
            lines.append(f"  {p}: mean={np.mean(col):.5g}, std={np.std(col):.5g}, "
                         f"min={np.min(col):.5g}, max={np.max(col):.5g}")

        self.result_text.setPlainText("\n".join(lines))
        self._log(f"Batch fit completed for {len(results)} files")

    # ------------------------------------------------------------------
    # Global Fitting (shared parameters)
    # ------------------------------------------------------------------

    def _global_fit(self):
        """Fit multiple datasets simultaneously with shared parameters.

        Loads multiple files, then fits them all with a single parameter set.
        """
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Data Files for Global Fit", "",
            "Data Files (*.csv *.tsv *.txt *.dat);;All Files (*)")
        if not paths or len(paths) < 2:
            QMessageBox.information(self, "Global Fit",
                                    "Select at least 2 data files for global fitting.")
            return

        try:
            func, param_names, p0 = self._build_model_func()
        except Exception as exc:
            QMessageBox.warning(self, "Model Error", str(exc))
            return

        datasets = []
        for path in paths:
            try:
                data = np.loadtxt(path, delimiter=None, comments="#")
                if data.ndim == 1:
                    y = data
                    x = np.arange(len(data), dtype=float)
                else:
                    x, y = data[:, 0], data[:, 1]
                datasets.append((x, y))
            except Exception:
                continue

        if len(datasets) < 2:
            QMessageBox.warning(self, "Global Fit", "Could not load enough valid datasets.")
            return

        # Global residual: concatenate all datasets
        def global_residual(params):
            residuals = []
            for x, y in datasets:
                residuals.append(y - func(x, *params))
            return np.concatenate(residuals)

        try:
            from scipy.optimize import least_squares as _ls
            result = _ls(global_residual, p0, method="trf", max_nfev=self.maxiter_spin.value())
            popt = result.x
            J = result.jac
            try:
                pcov = np.linalg.inv(J.T @ J) * (result.fun @ result.fun) / max(
                    sum(len(d[0]) for d in datasets) - len(popt), 1)
            except np.linalg.LinAlgError:
                pcov = np.full((len(popt), len(popt)), np.inf)
            perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else np.full(len(popt), float("nan"))

            lines = [f"Global Fit ({len(datasets)} datasets, shared parameters)"]
            lines.append("=" * 50)
            for i, name in enumerate(param_names):
                lines.append(f"  {name:>20s} = {popt[i]:>14.6g} +/- {perr[i]:.6g}")
            total_n = sum(len(d[0]) for d in datasets)
            total_ss = np.sum(result.fun ** 2)
            lines.append(f"  Total Residual SS = {total_ss:.6g}")
            lines.append(f"  Total Points      = {total_n}")
            self.result_text.setPlainText("\n".join(lines))

            # Plot all datasets with the global fit
            self._clear_plots()
            colors = ["#4a90d9", "#e74c3c", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
            for i, (x, y) in enumerate(datasets):
                c = colors[i % len(colors)]
                self.ax_main.plot(x, y, "o", ms=3, color=c, alpha=0.6, label=f"Dataset {i+1}")
                x_s = np.linspace(x.min(), x.max(), 200)
                self.ax_main.plot(x_s, func(x_s, *popt), "-", color=c, lw=1.5)
            self.ax_main.legend(fontsize=7)
            self.canvas.draw_idle()
            self._log(f"Global fit completed across {len(datasets)} datasets")
        except Exception as exc:
            self._log(f"Global fit error: {exc}")
            QMessageBox.warning(self, "Global Fit Error", str(exc))

    # ------------------------------------------------------------------
    # Generate Fit Report (HTML)
    # ------------------------------------------------------------------

    def _generate_fit_report(self):
        """Generate an HTML fit report with data, fit, parameters, and plots."""
        if self.fit_result is None:
            QMessageBox.information(self, "No Fit", "Run a fit first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Fit Report", "fit_report.html",
            "HTML Files (*.html);;All Files (*)")
        if not path:
            return
        try:
            import base64
            # Render figure to PNG in-memory
            buf = io.BytesIO()
            self.figure.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode("ascii")
            buf.close()

            r = self.fit_result
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            params_rows = ""
            for name in r["params"]:
                params_rows += (f"<tr><td>{name}</td>"
                                f"<td>{r['params'][name]:.8g}</td>"
                                f"<td>{r['uncertainties'][name]:.6g}</td></tr>\n")

            html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Curve Fitting Report</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 900px; margin: auto; padding: 20px; }}
h1 {{ color: #333; border-bottom: 2px solid #4a90d9; padding-bottom: 10px; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #4a90d9; color: white; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
.stats {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
.stat-box {{ background: #f5f5f5; padding: 12px; border-radius: 6px; }}
.stat-label {{ font-weight: bold; color: #666; font-size: 0.9em; }}
.stat-value {{ font-size: 1.3em; color: #333; }}
img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin: 15px 0; }}
</style>
</head><body>
<h1>Curve Fitting Report</h1>
<p><strong>Generated:</strong> {now}</p>
<p><strong>Model:</strong> {r['model']}</p>

<h2>Fit Parameters</h2>
<table>
<tr><th>Parameter</th><th>Value</th><th>Uncertainty</th></tr>
{params_rows}
</table>

<h2>Goodness of Fit</h2>
<div class="stats">
<div class="stat-box"><div class="stat-label">R-squared</div><div class="stat-value">{r['r_squared']:.8f}</div></div>
<div class="stat-box"><div class="stat-label">Chi-square</div><div class="stat-value">{r['chi_square']:.6g}</div></div>
<div class="stat-box"><div class="stat-label">Reduced Chi-sq</div><div class="stat-value">{r['reduced_chi_square']:.6g}</div></div>
<div class="stat-box"><div class="stat-label">AIC</div><div class="stat-value">{r['aic']:.4f}</div></div>
<div class="stat-box"><div class="stat-label">BIC</div><div class="stat-value">{r['bic']:.4f}</div></div>
<div class="stat-box"><div class="stat-label">Residual SS</div><div class="stat-value">{r['residual_ss']:.6g}</div></div>
<div class="stat-box"><div class="stat-label">Data Points</div><div class="stat-value">{r['n_points']}</div></div>
<div class="stat-box"><div class="stat-label">Parameters</div><div class="stat-value">{r['n_params']}</div></div>
</div>

<h2>Fit Plot</h2>
<img src="data:image/png;base64,{img_b64}" alt="Fit Plot">

</body></html>"""

            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
            self._log(f"Fit report saved to {path}")
        except Exception as exc:
            self._log(f"Report generation error: {exc}")
            QMessageBox.warning(self, "Report Error", str(exc))

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
    # Logging
    # ------------------------------------------------------------------

    def _log(self, msg):
        if self._logger:
            self._logger(msg)
