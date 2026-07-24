"""
Control Systems Analysis Widget for PyQt5 Scientific Suite.

Provides transfer function entry, Bode/Nyquist/root-locus plots,
step/impulse response, pole-zero maps, PID tuning with live preview,
Ziegler-Nichols auto-tuning, state-space conversion, Routh-Hurwitz
stability analysis, and frequency-response data extraction.
"""

import numpy as np
from numpy import pi, polymul, polyadd, real, imag
from scipy import signal as sp_signal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QComboBox, QDoubleSpinBox, QSpinBox, QPushButton,
    QTabWidget, QLineEdit, QCheckBox, QSplitter, QFileDialog,
    QMessageBox, QFrame, QSizePolicy, QTextEdit, QScrollArea,
    QSlider, QFormLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QToolButton
)
from PyQt5.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
from matplotlib.figure import Figure

try:
    from mpl_style import style_figure, style_axes
except ImportError:
    def style_figure(f): pass
    def style_axes(a): pass


# ====================================================================== #
#  Presets                                                                #
# ====================================================================== #
PRESETS = {
    "First Order (1/(s+1))": {"num": "1", "den": "1, 1"},
    "Second Order Underdamped": {"num": "1", "den": "1, 0.4, 1"},
    "Second Order Overdamped": {"num": "1", "den": "1, 3, 2"},
    "Second Order Critically Damped": {"num": "1", "den": "1, 2, 1"},
    "Integrator (1/s)": {"num": "1", "den": "1, 0"},
    "Double Integrator (1/s^2)": {"num": "1", "den": "1, 0, 0"},
    "DC Motor": {"num": "10", "den": "1, 11, 10"},
    "Resonant System": {"num": "100", "den": "1, 2, 100"},
    "Type-1 System": {"num": "10", "den": "1, 1, 0"},
    "Lead Compensator": {"num": "1, 2", "den": "1, 10"},
    "Lag Compensator": {"num": "1, 10", "den": "1, 2"},
    "Unstable Plant": {"num": "1", "den": "1, -1"},
}


# ====================================================================== #
#  Helper utilities                                                       #
# ====================================================================== #
def _parse_coeffs(text):
    """Parse a comma/space separated string into a list of floats."""
    text = text.strip().replace("[", "").replace("]", "")
    parts = [t.strip() for t in text.replace(",", " ").split() if t.strip()]
    return [float(p) for p in parts]


def _poly_str(coeffs, var="s"):
    """Pretty-print a polynomial coefficient list."""
    n = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        power = n - i
        if abs(c) < 1e-14:
            continue
        if power == 0:
            terms.append(f"{c:g}")
        elif power == 1:
            terms.append(f"{c:g}{var}" if abs(c) != 1 else f"{'-' if c < 0 else ''}{var}")
        else:
            terms.append(f"{c:g}{var}^{power}" if abs(c) != 1 else f"{'-' if c < 0 else ''}{var}^{power}")
    return " + ".join(terms).replace("+ -", "- ") if terms else "0"


def _routh_table(den):
    """Build the Routh array and return (table, is_stable)."""
    n = len(den)
    if n < 2:
        return np.array([[den[0]]]), den[0] > 0
    rows = (n + 1) // 2
    table = np.zeros((n, rows))
    for i in range(n):
        idx = 0
        for j in range(i, n, 2):
            if idx < rows:
                table[i // 2 if i < 2 else -1][idx] = den[j]
                idx += 1
    # Build properly
    table = np.zeros((n, rows))
    table[0, :] = [den[i] if i < n else 0 for i in range(0, n, 2)][:rows]
    table[1, :] = [den[i] if i < n else 0 for i in range(1, n, 2)][:rows]
    for i in range(2, n):
        for j in range(rows - 1):
            a = table[i - 1, 0]
            if abs(a) < 1e-15:
                a = 1e-15
            table[i, j] = (table[i - 1, 0] * table[i - 2, j + 1]
                           - table[i - 2, 0] * table[i - 1, j + 1]) / a
        table[i, -1] = 0.0
    first_col = table[:, 0]
    sign_changes = 0
    for k in range(1, len(first_col)):
        if first_col[k - 1] * first_col[k] < 0:
            sign_changes += 1
    return table, sign_changes == 0


def _step_metrics(t, y):
    """Compute rise time, settling time, overshoot, steady-state error."""
    if len(y) < 2:
        return {}
    y_final = y[-1]
    y_init = y[0]
    if abs(y_final - y_init) < 1e-12:
        return {"steady_state": y_final, "steady_state_error": abs(1.0 - y_final)}
    # Rise time: 10% to 90% of final value
    y10 = y_init + 0.1 * (y_final - y_init)
    y90 = y_init + 0.9 * (y_final - y_init)
    t10 = t[np.argmax(y >= y10)] if np.any(y >= y10) else None
    t90 = t[np.argmax(y >= y90)] if np.any(y >= y90) else None
    rise_time = (t90 - t10) if (t10 is not None and t90 is not None) else None
    # Overshoot
    peak = np.max(y) if y_final > y_init else np.min(y)
    overshoot_pct = 100.0 * abs(peak - y_final) / abs(y_final - y_init) if abs(y_final - y_init) > 1e-12 else 0.0
    # Settling time (2% band)
    band = 0.02 * abs(y_final)
    settled = np.where(np.abs(y - y_final) > band)[0]
    settling_time = t[settled[-1]] if len(settled) > 0 and settled[-1] < len(t) - 1 else t[-1]
    # Steady-state error to unit step
    ss_error = abs(1.0 - y_final)
    return {
        "rise_time": rise_time,
        "settling_time": settling_time,
        "overshoot_pct": overshoot_pct,
        "peak": peak,
        "steady_state": y_final,
        "steady_state_error": ss_error,
    }


# ====================================================================== #
#  Main Widget                                                            #
# ====================================================================== #
class ControlSystemsWidget(QWidget):
    """Interactive control-systems analysis and design tool."""

    analysis_updated = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._logger = None
        self._sys = None            # current TransferFunction
        self._pid_sys = None        # PID closed-loop system
        self._build_ui()

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #
    def set_logger(self, fn):
        """Attach an external logging callback."""
        self._logger = fn

    def run(self):
        """Execute the analysis for the current transfer function."""
        self._apply_system()

    def export(self):
        """Export the active plot tab to an image file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Plot", "control_plot.png",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
        if not path:
            return
        idx = self._tabs.currentIndex()
        canvas = self._canvases.get(idx)
        if canvas:
            canvas.figure.savefig(path, dpi=150, bbox_inches="tight")
            self._log(f"Exported plot to {path}")

    # ------------------------------------------------------------------ #
    #  Logging                                                            #
    # ------------------------------------------------------------------ #
    def _log(self, msg):
        if self._logger:
            self._logger(msg)
        self._console.append(msg)

    # ------------------------------------------------------------------ #
    #  UI Construction                                                    #
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        root = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # ---- Left panel: inputs ---- #
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(4, 4, 4, 4)

        # Transfer function input group
        tf_group = QGroupBox("Transfer Function")
        tf_form = QFormLayout(tf_group)
        self._num_edit = QLineEdit("1")
        self._den_edit = QLineEdit("1, 2, 1")
        tf_form.addRow("Numerator:", self._num_edit)
        tf_form.addRow("Denominator:", self._den_edit)

        self._tf_label = QLabel("H(s) = ...")
        self._tf_label.setWordWrap(True)
        self._tf_label.setStyleSheet("font-family: monospace; font-size: 11pt; padding: 4px;")
        tf_form.addRow(self._tf_label)

        # Preset selector
        preset_row = QHBoxLayout()
        self._preset_combo = QComboBox()
        self._preset_combo.addItem("-- Load Preset --")
        for name in PRESETS:
            self._preset_combo.addItem(name)
        self._preset_combo.currentIndexChanged.connect(self._load_preset)
        preset_row.addWidget(QLabel("Preset:"))
        preset_row.addWidget(self._preset_combo, 1)
        tf_form.addRow(preset_row)

        # Buttons
        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Analyze")
        self._run_btn.clicked.connect(self._apply_system)
        self._export_btn = QPushButton("Export Plot")
        self._export_btn.clicked.connect(self.export)
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._export_btn)
        tf_form.addRow(btn_row)

        left_lay.addWidget(tf_group)

        # ---- PID Tuner group ---- #
        pid_group = QGroupBox("PID Tuner")
        pid_lay = QFormLayout(pid_group)

        self._kp_slider = self._make_pid_slider(0.0, 50.0, 1.0, "Kp")
        self._ki_slider = self._make_pid_slider(0.0, 50.0, 0.0, "Ki")
        self._kd_slider = self._make_pid_slider(0.0, 20.0, 0.0, "Kd")
        self._kp_label = QLabel("1.00")
        self._ki_label = QLabel("0.00")
        self._kd_label = QLabel("0.00")

        for label_w, slider, name in [
            (self._kp_label, self._kp_slider, "Kp"),
            (self._ki_label, self._ki_slider, "Ki"),
            (self._kd_label, self._kd_slider, "Kd"),
        ]:
            row = QHBoxLayout()
            row.addWidget(slider, 1)
            row.addWidget(label_w)
            label_w.setMinimumWidth(50)
            pid_lay.addRow(f"{name}:", row)

        self._kp_slider.valueChanged.connect(lambda v: self._pid_slider_changed())
        self._ki_slider.valueChanged.connect(lambda v: self._pid_slider_changed())
        self._kd_slider.valueChanged.connect(lambda v: self._pid_slider_changed())

        pid_btn_row = QHBoxLayout()
        self._pid_apply_btn = QPushButton("Apply PID")
        self._pid_apply_btn.clicked.connect(self._apply_pid)
        self._zn_btn = QPushButton("Ziegler-Nichols")
        self._zn_btn.clicked.connect(self._ziegler_nichols)
        pid_btn_row.addWidget(self._pid_apply_btn)
        pid_btn_row.addWidget(self._zn_btn)
        pid_lay.addRow(pid_btn_row)

        left_lay.addWidget(pid_group)

        # ---- State Space group ---- #
        ss_group = QGroupBox("State Space")
        ss_lay = QFormLayout(ss_group)
        self._mat_a = QLineEdit("0, 1; -1, -2")
        self._mat_b = QLineEdit("0; 1")
        self._mat_c = QLineEdit("1, 0")
        self._mat_d = QLineEdit("0")
        ss_lay.addRow("A:", self._mat_a)
        ss_lay.addRow("B:", self._mat_b)
        ss_lay.addRow("C:", self._mat_c)
        ss_lay.addRow("D:", self._mat_d)

        ss_btn_row = QHBoxLayout()
        self._ss_to_tf_btn = QPushButton("SS -> TF")
        self._ss_to_tf_btn.clicked.connect(self._ss_to_tf)
        self._tf_to_ss_btn = QPushButton("TF -> SS")
        self._tf_to_ss_btn.clicked.connect(self._tf_to_ss)
        ss_btn_row.addWidget(self._ss_to_tf_btn)
        ss_btn_row.addWidget(self._tf_to_ss_btn)
        ss_lay.addRow(ss_btn_row)

        left_lay.addWidget(ss_group)

        # ---- Console / info ---- #
        self._console = QTextEdit()
        self._console.setReadOnly(True)
        self._console.setMaximumHeight(140)
        self._console.setPlaceholderText("Analysis results...")
        left_lay.addWidget(self._console)
        left_lay.addStretch()

        left.setMaximumWidth(380)
        splitter.addWidget(left)

        # ---- Right panel: plot tabs ---- #
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(2, 2, 2, 2)

        self._tabs = QTabWidget()
        self._canvases = {}
        self._figures = {}
        self._toolbars = {}

        tab_defs = [
            ("Bode", self._plot_bode),
            ("Nyquist", self._plot_nyquist),
            ("Root Locus", self._plot_root_locus),
            ("Step Response", self._plot_step),
            ("Impulse Response", self._plot_impulse),
            ("Pole-Zero Map", self._plot_pzmap),
            ("PID Response", self._plot_pid_step),
            ("Stability", self._show_stability),
            ("Freq. Data", self._show_freq_data),
        ]
        self._plot_funcs = {}
        for i, (name, func) in enumerate(tab_defs):
            page = QWidget()
            page_lay = QVBoxLayout(page)
            page_lay.setContentsMargins(0, 0, 0, 0)
            fig = Figure(figsize=(7, 5), dpi=100)
            style_figure(fig)
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, page)
            page_lay.addWidget(toolbar)
            page_lay.addWidget(canvas)
            self._tabs.addTab(page, name)
            self._canvases[i] = canvas
            self._figures[i] = fig
            self._toolbars[i] = toolbar
            self._plot_funcs[i] = func

        right_lay.addWidget(self._tabs)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    # ------------------------------------------------------------------ #
    #  Slider helper                                                      #
    # ------------------------------------------------------------------ #
    def _make_pid_slider(self, lo, hi, default, name):
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(int(lo * 100))
        slider.setMaximum(int(hi * 100))
        slider.setValue(int(default * 100))
        slider.setTickInterval(int((hi - lo) * 10))
        slider.setTickPosition(QSlider.TicksBelow)
        return slider

    def _pid_slider_changed(self):
        kp = self._kp_slider.value() / 100.0
        ki = self._ki_slider.value() / 100.0
        kd = self._kd_slider.value() / 100.0
        self._kp_label.setText(f"{kp:.2f}")
        self._ki_label.setText(f"{ki:.2f}")
        self._kd_label.setText(f"{kd:.2f}")

    # ------------------------------------------------------------------ #
    #  Preset loader                                                      #
    # ------------------------------------------------------------------ #
    def _load_preset(self, idx):
        if idx <= 0:
            return
        name = self._preset_combo.currentText()
        p = PRESETS.get(name)
        if p:
            self._num_edit.setText(p["num"])
            self._den_edit.setText(p["den"])
            self._log(f"Loaded preset: {name}")
            self._apply_system()

    # ------------------------------------------------------------------ #
    #  Build / validate transfer function                                 #
    # ------------------------------------------------------------------ #
    def _apply_system(self):
        try:
            num = _parse_coeffs(self._num_edit.text())
            den = _parse_coeffs(self._den_edit.text())
            if not num or not den:
                raise ValueError("Numerator and denominator must not be empty.")
            self._sys = sp_signal.TransferFunction(num, den)
            self._tf_label.setText(
                f"H(s) = ({_poly_str(num)}) / ({_poly_str(den)})"
            )
            self._log(f"System: num={num}, den={den}")
            self._update_all_plots()
            self.analysis_updated.emit()
        except Exception as exc:
            self._log(f"Error: {exc}")
            QMessageBox.warning(self, "Input Error", str(exc))

    # ------------------------------------------------------------------ #
    #  Refresh all visible plots                                          #
    # ------------------------------------------------------------------ #
    def _update_all_plots(self):
        if self._sys is None:
            return
        for idx, func in self._plot_funcs.items():
            try:
                func(idx)
            except Exception as exc:
                self._log(f"Plot error (tab {idx}): {exc}")

    # ------------------------------------------------------------------ #
    #  Bode Plot                                                          #
    # ------------------------------------------------------------------ #
    def _plot_bode(self, idx):
        fig = self._figures[idx]
        fig.clear()
        ax_mag = fig.add_subplot(2, 1, 1)
        ax_phase = fig.add_subplot(2, 1, 2, sharex=ax_mag)
        style_axes(ax_mag)
        style_axes(ax_phase)

        w = np.logspace(-2, 4, 2000)
        w_out, mag, phase = sp_signal.bode(self._sys, w)

        ax_mag.semilogx(w_out, mag, "b-", linewidth=1.2)
        ax_mag.set_ylabel("Magnitude (dB)")
        ax_mag.set_title("Bode Diagram")
        ax_mag.grid(True, which="both", alpha=0.3)
        ax_mag.axhline(0, color="gray", linewidth=0.8, linestyle="--")

        ax_phase.semilogx(w_out, phase, "r-", linewidth=1.2)
        ax_phase.set_ylabel("Phase (deg)")
        ax_phase.set_xlabel("Frequency (rad/s)")
        ax_phase.grid(True, which="both", alpha=0.3)
        ax_phase.axhline(-180, color="gray", linewidth=0.8, linestyle="--")

        # Gain and phase margins
        try:
            gm, pm, wgc, wpc = self._compute_margins(w_out, mag, phase)
            if wgc is not None and np.isfinite(wgc):
                ax_mag.axvline(wgc, color="green", linestyle=":", alpha=0.7, label=f"GM = {gm:.1f} dB @ {wgc:.2f} rad/s")
                ax_mag.legend(fontsize=8)
            if wpc is not None and np.isfinite(wpc):
                ax_phase.axvline(wpc, color="purple", linestyle=":", alpha=0.7, label=f"PM = {pm:.1f} deg @ {wpc:.2f} rad/s")
                ax_phase.legend(fontsize=8)
            self._log(f"Gain margin: {gm:.2f} dB, Phase margin: {pm:.2f} deg")
        except Exception:
            pass

        fig.tight_layout()
        self._canvases[idx].draw()

    def _compute_margins(self, w, mag_db, phase_deg):
        """Return gain margin (dB), phase margin (deg), freq_gm, freq_pm."""
        # Gain crossover: where magnitude crosses 0 dB
        mag_sign = np.sign(mag_db)
        gc_crossings = np.where(np.diff(mag_sign))[0]
        if len(gc_crossings) > 0:
            i = gc_crossings[0]
            frac = -mag_db[i] / (mag_db[i + 1] - mag_db[i]) if abs(mag_db[i + 1] - mag_db[i]) > 1e-15 else 0
            w_gc = w[i] + frac * (w[i + 1] - w[i])
            phase_at_gc = np.interp(w_gc, w, phase_deg)
            pm = 180.0 + phase_at_gc
        else:
            w_gc = None
            pm = float("inf")

        # Phase crossover: where phase crosses -180 deg
        phase_shifted = phase_deg + 180
        ph_sign = np.sign(phase_shifted)
        pc_crossings = np.where(np.diff(ph_sign))[0]
        if len(pc_crossings) > 0:
            i = pc_crossings[0]
            frac = -phase_shifted[i] / (phase_shifted[i + 1] - phase_shifted[i]) if abs(phase_shifted[i + 1] - phase_shifted[i]) > 1e-15 else 0
            w_pc = w[i] + frac * (w[i + 1] - w[i])
            mag_at_pc = np.interp(w_pc, w, mag_db)
            gm = -mag_at_pc
        else:
            w_pc = None
            gm = float("inf")

        return gm, pm, w_pc, w_gc

    # ------------------------------------------------------------------ #
    #  Nyquist Diagram                                                    #
    # ------------------------------------------------------------------ #
    def _plot_nyquist(self, idx):
        fig = self._figures[idx]
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        style_axes(ax)

        w = np.logspace(-3, 4, 5000)
        w_out, H = sp_signal.freqresp(self._sys, w)
        re = np.real(H)
        im_part = np.imag(H)

        ax.plot(re, im_part, "b-", linewidth=1.2, label="G(j$\\omega$)")
        ax.plot(re, -im_part, "b--", linewidth=0.7, alpha=0.5, label="Mirror")

        # Unit circle
        theta = np.linspace(0, 2 * pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.6, alpha=0.4, label="Unit circle")

        # Critical point
        ax.plot(-1, 0, "rx", markersize=10, markeredgewidth=2, label="(-1, 0)")

        # Direction arrows
        n_arrows = 8
        step = max(1, len(re) // n_arrows)
        for k in range(0, len(re) - 1, step):
            dx = re[k + 1] - re[k]
            dy = im_part[k + 1] - im_part[k]
            norm = np.sqrt(dx**2 + dy**2)
            if norm > 1e-12:
                ax.annotate("", xy=(re[k + 1], im_part[k + 1]),
                            xytext=(re[k], im_part[k]),
                            arrowprops=dict(arrowstyle="->", color="blue", lw=0.8))

        ax.set_xlabel("Real")
        ax.set_ylabel("Imaginary")
        ax.set_title("Nyquist Diagram")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()
        self._canvases[idx].draw()

    # ------------------------------------------------------------------ #
    #  Root Locus                                                         #
    # ------------------------------------------------------------------ #
    def _plot_root_locus(self, idx):
        fig = self._figures[idx]
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        style_axes(ax)

        num = np.array(self._sys.num)
        den = np.array(self._sys.den)

        # Open-loop poles and zeros
        ol_poles = np.roots(den)
        ol_zeros = np.roots(num) if len(num) > 1 else np.array([])

        # Gain sweep
        gains = np.concatenate([
            np.linspace(0, 1, 200),
            np.linspace(1, 10, 200),
            np.linspace(10, 100, 200),
            np.linspace(100, 1000, 200),
            np.linspace(1000, 10000, 100),
        ])
        all_loci = []
        for K in gains:
            cl_char = polyadd(den, K * np.pad(num, (len(den) - len(num), 0)))
            roots = np.roots(cl_char)
            roots_sorted = sorted(roots, key=lambda z: (z.real, z.imag))
            all_loci.append(roots_sorted)

        all_loci = np.array(all_loci)
        n_branches = all_loci.shape[1]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                  "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
        for b in range(n_branches):
            c = colors[b % len(colors)]
            branch = all_loci[:, b]
            ax.plot(branch.real, branch.imag, "-", color=c, linewidth=0.9, alpha=0.8)

        # Mark open-loop poles and zeros
        ax.plot(ol_poles.real, ol_poles.imag, "kx", markersize=10,
                markeredgewidth=2, label="OL Poles")
        if len(ol_zeros) > 0:
            ax.plot(ol_zeros.real, ol_zeros.imag, "ko", markersize=8,
                    markerfacecolor="none", markeredgewidth=2, label="OL Zeros")

        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("Real Axis")
        ax.set_ylabel("Imaginary Axis")
        ax.set_title("Root Locus")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        self._canvases[idx].draw()

    # ------------------------------------------------------------------ #
    #  Step Response                                                      #
    # ------------------------------------------------------------------ #
    def _plot_step(self, idx):
        fig = self._figures[idx]
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        style_axes(ax)

        try:
            t, y = sp_signal.step(self._sys)
        except Exception:
            t = np.linspace(0, 20, 2000)
            t, y = sp_signal.step(self._sys, T=t)

        ax.plot(t, y, "b-", linewidth=1.2)
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", label="Unit step target")

        # Compute and display metrics
        metrics = _step_metrics(t, y)
        info_lines = []
        if metrics.get("rise_time") is not None:
            rt = metrics["rise_time"]
            ax.axvline(rt, color="green", linewidth=0.7, linestyle=":", alpha=0.6)
            info_lines.append(f"Rise time: {rt:.4f} s")
        if metrics.get("settling_time") is not None:
            st = metrics["settling_time"]
            ax.axvline(st, color="orange", linewidth=0.7, linestyle=":", alpha=0.6)
            info_lines.append(f"Settling time: {st:.4f} s")
        if metrics.get("overshoot_pct") is not None:
            info_lines.append(f"Overshoot: {metrics['overshoot_pct']:.2f}%")
        if metrics.get("peak") is not None:
            ax.axhline(metrics["peak"], color="red", linewidth=0.5, linestyle=":", alpha=0.5)
            info_lines.append(f"Peak: {metrics['peak']:.4f}")
        if metrics.get("steady_state") is not None:
            info_lines.append(f"Steady state: {metrics['steady_state']:.4f}")
        if metrics.get("steady_state_error") is not None:
            info_lines.append(f"SS error: {metrics['steady_state_error']:.4f}")

        # 2% settling band
        if metrics.get("steady_state") is not None:
            yss = metrics["steady_state"]
            band = 0.02 * abs(yss) if abs(yss) > 1e-12 else 0.02
            ax.axhspan(yss - band, yss + band, alpha=0.08, color="green", label="2% band")

        if info_lines:
            ax.text(0.98, 0.02, "\n".join(info_lines), transform=ax.transAxes,
                    fontsize=8, verticalalignment="bottom", horizontalalignment="right",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
            self._log("Step: " + " | ".join(info_lines))

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Step Response")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        self._canvases[idx].draw()

    # ------------------------------------------------------------------ #
    #  Impulse Response                                                   #
    # ------------------------------------------------------------------ #
    def _plot_impulse(self, idx):
        fig = self._figures[idx]
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        style_axes(ax)

        try:
            t, y = sp_signal.impulse(self._sys)
        except Exception:
            t = np.linspace(0, 20, 2000)
            t, y = sp_signal.impulse(self._sys, T=t)

        ax.plot(t, y, "m-", linewidth=1.2)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Impulse Response")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self._canvases[idx].draw()

    # ------------------------------------------------------------------ #
    #  Pole-Zero Map                                                      #
    # ------------------------------------------------------------------ #
    def _plot_pzmap(self, idx):
        fig = self._figures[idx]
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        style_axes(ax)

        poles = np.roots(self._sys.den)
        zeros = np.roots(self._sys.num) if len(self._sys.num) > 1 else np.array([])

        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)

        if len(poles) > 0:
            ax.plot(poles.real, poles.imag, "rx", markersize=12,
                    markeredgewidth=2.5, label="Poles", zorder=5)
        if len(zeros) > 0:
            ax.plot(zeros.real, zeros.imag, "bo", markersize=10,
                    markerfacecolor="none", markeredgewidth=2, label="Zeros", zorder=5)

        # Annotate pole locations
        for p in poles:
            ax.annotate(f"  {p:.3f}", (p.real, p.imag), fontsize=7, color="red")
        for z in zeros:
            ax.annotate(f"  {z:.3f}", (z.real, z.imag), fontsize=7, color="blue")

        # Draw damping ratio lines
        for zeta in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
            r = np.linspace(0, max(5, 1.5 * max(abs(poles.real)) if len(poles) > 0 else 5), 100)
            theta = np.arccos(zeta)
            x = -r * np.cos(theta)
            y_pos = r * np.sin(theta)
            ax.plot(x, y_pos, "k--", linewidth=0.3, alpha=0.3)
            ax.plot(x, -y_pos, "k--", linewidth=0.3, alpha=0.3)
            ax.text(x[-1], y_pos[-1], f"  {zeta}", fontsize=6, alpha=0.4)

        ax.set_xlabel("Real Axis")
        ax.set_ylabel("Imaginary Axis")
        ax.set_title("Pole-Zero Map")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()
        self._canvases[idx].draw()

    # ------------------------------------------------------------------ #
    #  PID Tuner response                                                 #
    # ------------------------------------------------------------------ #
    def _apply_pid(self):
        if self._sys is None:
            self._log("Set a plant transfer function first.")
            return
        kp = self._kp_slider.value() / 100.0
        ki = self._ki_slider.value() / 100.0
        kd = self._kd_slider.value() / 100.0
        self._log(f"PID gains: Kp={kp:.2f}, Ki={ki:.2f}, Kd={kd:.2f}")
        self._build_pid_closed_loop(kp, ki, kd)
        pid_tab_idx = 6  # PID Response tab
        self._plot_pid_step(pid_tab_idx)
        self._tabs.setCurrentIndex(pid_tab_idx)

    def _build_pid_closed_loop(self, kp, ki, kd):
        """Build closed-loop TF: C(s)*G(s) / (1 + C(s)*G(s))."""
        # PID controller: C(s) = Kd*s^2 + Kp*s + Ki / s
        # => C(s) = (Kd*s^2 + Kp*s + Ki) / s
        pid_num = np.array([kd, kp, ki])
        pid_den = np.array([1, 0])  # s

        # Open-loop: C(s)*G(s) = pid_num * sys.num / (pid_den * sys.den)
        ol_num = np.polymul(pid_num, self._sys.num)
        ol_den = np.polymul(pid_den, self._sys.den)

        # Closed-loop: OL / (1 + OL) = ol_num / (ol_den + ol_num)
        # Pad shorter array
        cl_den = np.polyadd(ol_den, ol_num)
        cl_num = ol_num.copy()

        self._pid_sys = sp_signal.TransferFunction(cl_num, cl_den)

    def _plot_pid_step(self, idx):
        fig = self._figures[idx]
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        style_axes(ax)

        if self._pid_sys is None:
            ax.text(0.5, 0.5, "Click 'Apply PID' to see response",
                    transform=ax.transAxes, ha="center", va="center", fontsize=12)
            self._canvases[idx].draw()
            return

        try:
            t = np.linspace(0, 30, 3000)
            t, y = sp_signal.step(self._pid_sys, T=t)
        except Exception as exc:
            ax.text(0.5, 0.5, f"Simulation error: {exc}",
                    transform=ax.transAxes, ha="center", va="center", fontsize=10)
            self._canvases[idx].draw()
            return

        ax.plot(t, y, "b-", linewidth=1.2, label="PID Closed-Loop")
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")

        # Also plot open-loop step for comparison
        try:
            t_ol, y_ol = sp_signal.step(self._sys, T=t)
            ax.plot(t_ol, y_ol, "r--", linewidth=0.8, alpha=0.6, label="Open-Loop Plant")
        except Exception:
            pass

        metrics = _step_metrics(t, y)
        info_lines = []
        if metrics.get("rise_time") is not None:
            info_lines.append(f"Rise: {metrics['rise_time']:.3f}s")
        if metrics.get("settling_time") is not None:
            info_lines.append(f"Settle: {metrics['settling_time']:.3f}s")
        if metrics.get("overshoot_pct") is not None:
            info_lines.append(f"OS: {metrics['overshoot_pct']:.1f}%")
        if metrics.get("steady_state_error") is not None:
            info_lines.append(f"SSE: {metrics['steady_state_error']:.4f}")

        kp = self._kp_slider.value() / 100.0
        ki = self._ki_slider.value() / 100.0
        kd = self._kd_slider.value() / 100.0
        info_lines.insert(0, f"Kp={kp:.2f}  Ki={ki:.2f}  Kd={kd:.2f}")

        if info_lines:
            ax.text(0.98, 0.02, "\n".join(info_lines), transform=ax.transAxes,
                    fontsize=8, va="bottom", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85))

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("PID Closed-Loop Step Response")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        self._canvases[idx].draw()

    # ------------------------------------------------------------------ #
    #  Ziegler-Nichols auto-tune                                          #
    # ------------------------------------------------------------------ #
    def _ziegler_nichols(self):
        """Estimate PID gains using Ziegler-Nichols ultimate-gain method."""
        if self._sys is None:
            self._log("Set a plant first.")
            return
        try:
            # Sweep gain K until closed-loop is marginally stable
            num = np.array(self._sys.num)
            den = np.array(self._sys.den)

            ku = None
            wu = None
            for K in np.concatenate([np.linspace(0.01, 10, 500),
                                     np.linspace(10, 100, 500),
                                     np.linspace(100, 10000, 500)]):
                cl_den = np.polyadd(den, K * np.pad(num, (len(den) - len(num), 0)))
                roots = np.roots(cl_den)
                # Check if any root is on the imaginary axis
                for r in roots:
                    if abs(r.real) < 0.05 * max(abs(r.imag), 0.01) and abs(r.imag) > 0.01:
                        if ku is None or K < ku:
                            ku = K
                            wu = abs(r.imag)

            if ku is None:
                self._log("Ziegler-Nichols: Could not find ultimate gain. System may be unconditionally stable.")
                QMessageBox.information(self, "Z-N Tuning",
                                        "Could not determine ultimate gain.\nThe system may be unconditionally stable or unstable.")
                return

            tu = 2 * pi / wu
            self._log(f"Ziegler-Nichols: Ku={ku:.4f}, Tu={tu:.4f} s, Wu={wu:.4f} rad/s")

            # Classic Z-N PID formulas
            kp = 0.6 * ku
            ki = 2.0 * kp / tu
            kd = kp * tu / 8.0

            self._log(f"Z-N PID: Kp={kp:.4f}, Ki={ki:.4f}, Kd={kd:.4f}")

            # Clamp to slider range and set
            self._kp_slider.setValue(min(int(kp * 100), self._kp_slider.maximum()))
            self._ki_slider.setValue(min(int(ki * 100), self._ki_slider.maximum()))
            self._kd_slider.setValue(min(int(kd * 100), self._kd_slider.maximum()))
            self._pid_slider_changed()
            self._apply_pid()
        except Exception as exc:
            self._log(f"Z-N error: {exc}")

    # ------------------------------------------------------------------ #
    #  State-Space conversion                                             #
    # ------------------------------------------------------------------ #
    def _parse_matrix(self, text):
        """Parse a semicolon-row, comma-column matrix string."""
        rows = text.strip().split(";")
        return np.array([[float(x) for x in row.split(",")] for row in rows])

    def _ss_to_tf(self):
        """Convert state-space (A,B,C,D) to transfer function."""
        try:
            A = self._parse_matrix(self._mat_a.text())
            B = self._parse_matrix(self._mat_b.text())
            C = self._parse_matrix(self._mat_c.text())
            D = self._parse_matrix(self._mat_d.text())
            ss = sp_signal.StateSpace(A, B, C, D)
            tf = ss.to_tf()
            num = tf.num.flatten().tolist()
            den = tf.den.flatten().tolist()
            self._num_edit.setText(", ".join(f"{c:g}" for c in num))
            self._den_edit.setText(", ".join(f"{c:g}" for c in den))
            self._log(f"SS->TF: num={num}, den={den}")
            self._apply_system()
        except Exception as exc:
            self._log(f"SS->TF error: {exc}")
            QMessageBox.warning(self, "State Space Error", str(exc))

    def _tf_to_ss(self):
        """Convert current transfer function to controllable canonical form."""
        if self._sys is None:
            self._log("Set a transfer function first.")
            return
        try:
            ss = self._sys.to_ss()
            A, B, C, D = ss.A, ss.B, ss.C, ss.D

            def mat_str(m):
                rows = []
                for row in np.atleast_2d(m):
                    rows.append(", ".join(f"{v:g}" for v in row))
                return "; ".join(rows)

            self._mat_a.setText(mat_str(A))
            self._mat_b.setText(mat_str(B))
            self._mat_c.setText(mat_str(C))
            self._mat_d.setText(mat_str(D))
            self._log(f"TF->SS conversion complete. A is {A.shape[0]}x{A.shape[1]}")
        except Exception as exc:
            self._log(f"TF->SS error: {exc}")
            QMessageBox.warning(self, "Conversion Error", str(exc))

    # ------------------------------------------------------------------ #
    #  Stability (Routh-Hurwitz)                                          #
    # ------------------------------------------------------------------ #
    def _show_stability(self, idx):
        fig = self._figures[idx]
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        ax.axis("off")
        style_axes(ax)

        den = list(self._sys.den)
        n = len(den)
        table, is_stable = _routh_table(den)

        lines = []
        lines.append("Routh-Hurwitz Stability Analysis")
        lines.append("=" * 50)
        lines.append(f"Characteristic polynomial: {_poly_str(den)}")
        lines.append(f"Order: {n - 1}")
        lines.append("")

        # Routh table display
        lines.append("Routh Array (first column):")
        for i in range(table.shape[0]):
            row_vals = "  ".join(f"{table[i, j]:10.4f}" for j in range(table.shape[1]))
            lines.append(f"  s^{n - 1 - i}:  {row_vals}")
        lines.append("")

        # Stability verdict
        poles = np.roots(den)
        rhp_poles = [p for p in poles if p.real > 1e-10]
        jw_poles = [p for p in poles if abs(p.real) < 1e-10]

        if is_stable and len(rhp_poles) == 0:
            verdict = "STABLE - All poles in LHP"
            color = "green"
        elif len(jw_poles) > 0 and len(rhp_poles) == 0:
            verdict = "MARGINALLY STABLE - Poles on imaginary axis"
            color = "orange"
        else:
            verdict = f"UNSTABLE - {len(rhp_poles)} RHP pole(s)"
            color = "red"

        lines.append(f"Verdict: {verdict}")
        lines.append("")
        lines.append("Poles:")
        for p in poles:
            loc = "LHP" if p.real < -1e-10 else ("RHP" if p.real > 1e-10 else "j-axis")
            lines.append(f"  {p:.4f}  [{loc}]")

        # Stability margins from Bode
        lines.append("")
        try:
            w = np.logspace(-2, 4, 2000)
            w_out, mag, phase = sp_signal.bode(self._sys, w)
            gm, pm, wgc, wpc = self._compute_margins(w_out, mag, phase)
            if np.isfinite(gm):
                lines.append(f"Gain Margin: {gm:.2f} dB")
            else:
                lines.append("Gain Margin: Inf (no phase crossover)")
            if np.isfinite(pm):
                lines.append(f"Phase Margin: {pm:.2f} deg")
            else:
                lines.append("Phase Margin: Inf (no gain crossover)")
        except Exception:
            pass

        text = "\n".join(lines)
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

        self._log(f"Stability: {verdict}")
        fig.tight_layout()
        self._canvases[idx].draw()

    # ------------------------------------------------------------------ #
    #  Frequency Response Data                                            #
    # ------------------------------------------------------------------ #
    def _show_freq_data(self, idx):
        fig = self._figures[idx]
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        ax.axis("off")
        style_axes(ax)

        w = np.logspace(-3, 4, 5000)
        w_out, mag_db, phase_deg = sp_signal.bode(self._sys, w)

        lines = []
        lines.append("Frequency Response Data")
        lines.append("=" * 50)

        # Gain crossover frequency (where |G(jw)| = 0 dB)
        gc_freqs = []
        mag_sign = np.sign(mag_db)
        crossings = np.where(np.diff(mag_sign))[0]
        for ci in crossings:
            frac = -mag_db[ci] / (mag_db[ci + 1] - mag_db[ci]) if abs(mag_db[ci + 1] - mag_db[ci]) > 1e-15 else 0
            wc = w_out[ci] + frac * (w_out[ci + 1] - w_out[ci])
            gc_freqs.append(wc)
        if gc_freqs:
            lines.append(f"Gain crossover freq(s): {', '.join(f'{f:.4f} rad/s' for f in gc_freqs)}")
        else:
            lines.append("Gain crossover freq: None (gain never crosses 0 dB)")

        # Phase crossover frequency (where phase = -180 deg)
        pc_freqs = []
        phase_shifted = phase_deg + 180
        ph_sign = np.sign(phase_shifted)
        crossings = np.where(np.diff(ph_sign))[0]
        for ci in crossings:
            frac = -phase_shifted[ci] / (phase_shifted[ci + 1] - phase_shifted[ci]) if abs(phase_shifted[ci + 1] - phase_shifted[ci]) > 1e-15 else 0
            wc = w_out[ci] + frac * (w_out[ci + 1] - w_out[ci])
            pc_freqs.append(wc)
        if pc_freqs:
            lines.append(f"Phase crossover freq(s): {', '.join(f'{f:.4f} rad/s' for f in pc_freqs)}")
        else:
            lines.append("Phase crossover freq: None (phase never crosses -180 deg)")

        # Bandwidth (-3dB)
        dc_gain = mag_db[0]
        bw_idx = np.where(mag_db < dc_gain - 3)[0]
        if len(bw_idx) > 0:
            bw = w_out[bw_idx[0]]
            lines.append(f"Bandwidth (-3dB): {bw:.4f} rad/s ({bw / (2*pi):.4f} Hz)")
        else:
            lines.append("Bandwidth (-3dB): > frequency range")

        lines.append(f"\nDC Gain: {mag_db[0]:.2f} dB ({10**(mag_db[0]/20):.4f} linear)")
        lines.append(f"HF Gain (at {w_out[-1]:.0f} rad/s): {mag_db[-1]:.2f} dB")

        # Resonance peak
        peak_idx = np.argmax(mag_db)
        peak_mag = mag_db[peak_idx]
        peak_w = w_out[peak_idx]
        if peak_mag > dc_gain + 0.5:
            lines.append(f"\nResonance peak: {peak_mag:.2f} dB at {peak_w:.4f} rad/s")
            lines.append(f"Peak magnitude: {10**(peak_mag/20):.4f} (linear)")
        else:
            lines.append("\nNo resonance peak detected.")

        # Gain/phase margins summary
        lines.append("")
        gm, pm, wgc, wpc = self._compute_margins(w_out, mag_db, phase_deg)
        lines.append(f"Gain Margin: {gm:.2f} dB" if np.isfinite(gm) else "Gain Margin: Inf")
        lines.append(f"Phase Margin: {pm:.2f} deg" if np.isfinite(pm) else "Phase Margin: Inf")

        # Poles and zeros
        poles = np.roots(self._sys.den)
        zeros = np.roots(self._sys.num) if len(self._sys.num) > 1 else np.array([])
        lines.append(f"\nSystem order: {len(self._sys.den) - 1}")
        lines.append(f"Number of poles: {len(poles)}")
        lines.append(f"Number of zeros: {len(zeros)}")
        for p in poles:
            wn = abs(p)
            zeta = -p.real / wn if wn > 1e-12 else 0
            lines.append(f"  Pole {p:.4f}  wn={wn:.4f}  zeta={zeta:.4f}")

        # Sample frequency table
        lines.append("\n--- Sample Frequency Table ---")
        lines.append(f"{'w (rad/s)':>12}  {'|G| (dB)':>10}  {'Phase (deg)':>12}")
        sample_w = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 1000.0]
        for ws in sample_w:
            if ws <= w_out[-1]:
                m = np.interp(ws, w_out, mag_db)
                ph = np.interp(ws, w_out, phase_deg)
                lines.append(f"{ws:12.3f}  {m:10.2f}  {ph:12.2f}")

        text = "\n".join(lines)
        ax.text(0.03, 0.97, text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))

        self._log("Frequency data updated.")
        fig.tight_layout()
        self._canvases[idx].draw()

    # ------------------------------------------------------------------ #
    #  Convenience: direct coefficient setters (for scripting)            #
    # ------------------------------------------------------------------ #
    def set_tf(self, num, den):
        """Programmatically set the transfer function and run analysis."""
        self._num_edit.setText(", ".join(str(c) for c in num))
        self._den_edit.setText(", ".join(str(c) for c in den))
        self._apply_system()

    def set_pid(self, kp, ki, kd):
        """Programmatically set PID gains and apply."""
        self._kp_slider.setValue(int(kp * 100))
        self._ki_slider.setValue(int(ki * 100))
        self._kd_slider.setValue(int(kd * 100))
        self._pid_slider_changed()
        self._apply_pid()

    def get_step_metrics(self):
        """Return step-response metrics dict for the current system."""
        if self._sys is None:
            return {}
        t, y = sp_signal.step(self._sys)
        return _step_metrics(t, y)

    def get_margins(self):
        """Return (gain_margin_dB, phase_margin_deg, w_gm, w_pm)."""
        if self._sys is None:
            return None
        w = np.logspace(-2, 4, 2000)
        w_out, mag, phase = sp_signal.bode(self._sys, w)
        return self._compute_margins(w_out, mag, phase)

    def is_stable(self):
        """Return True if all poles are in LHP."""
        if self._sys is None:
            return None
        poles = np.roots(self._sys.den)
        return all(p.real < 0 for p in poles)


# ====================================================================== #
#  Standalone test harness                                                #
# ====================================================================== #
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = ControlSystemsWidget()
    w.setWindowTitle("Control Systems Analysis")
    w.resize(1200, 750)
    w.show()
    w.set_tf([1], [1, 2, 1])
    sys.exit(app.exec_())
