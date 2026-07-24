"""
Acoustics Module Widget for QuantumRes.
SPL calculations, octave band analysis, room acoustics, resonance,
psychoacoustics, Doppler effect, noise reduction, and material database.
"""

import numpy as np
from scipy.interpolate import interp1d
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QLabel, QLineEdit, QComboBox, QGroupBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame,
    QDoubleSpinBox, QSpinBox, QTextEdit, QScrollArea, QCheckBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ── Constants ─────────────────────────────────────────────────────────────────
P_REF = 2e-5        # Pa, reference sound pressure
I_REF = 1e-12       # W/m^2, reference intensity
SPEED_SOUND = 343.0  # m/s at 20 C
RHO_AIR = 1.225      # kg/m^3

# ── A-weighting coefficients (IEC 61672) ──────────────────────────────────────
# Frequencies and corresponding A-weighting corrections in dB
A_WEIGHT_FREQ = np.array([10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80,
                           100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                           1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000,
                           6300, 8000, 10000, 12500, 16000, 20000])
A_WEIGHT_DB = np.array([-70.4, -63.4, -56.7, -50.5, -44.7, -39.4, -34.6,
                         -30.2, -26.2, -22.5, -19.1, -16.1, -13.4, -10.9,
                         -8.6, -6.6, -4.8, -3.2, -1.9, -0.8,
                         0.0, 0.6, 1.0, 1.2, 1.3, 1.2, 1.0, 0.5,
                         -0.1, -1.1, -2.5, -4.3, -6.6, -9.3])

# ── 1/1 Octave band center frequencies ───────────────────────────────────────
OCTAVE_CENTERS = np.array([31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
# 1/3 octave band center frequencies
THIRD_OCTAVE_CENTERS = np.array([
    20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500,
    630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
    10000, 12500, 16000, 20000
])

# ── Material Absorption Coefficients ──────────────────────────────────────────
# {material: [alpha at 125, 250, 500, 1000, 2000, 4000 Hz]}
MATERIAL_ABSORPTION = {
    "Concrete (unpainted)":     [0.01, 0.01, 0.02, 0.02, 0.02, 0.03],
    "Concrete (painted)":       [0.01, 0.01, 0.01, 0.02, 0.02, 0.02],
    "Brick (unglazed)":         [0.03, 0.03, 0.03, 0.04, 0.05, 0.07],
    "Brick (painted)":          [0.01, 0.01, 0.02, 0.02, 0.02, 0.03],
    "Plaster on brick":         [0.01, 0.02, 0.02, 0.03, 0.04, 0.05],
    "Gypsum board (1/2 in)":    [0.29, 0.10, 0.05, 0.04, 0.07, 0.09],
    "Plywood panel (3/8 in)":   [0.28, 0.22, 0.17, 0.09, 0.10, 0.11],
    "Glass (1/4 in)":           [0.18, 0.06, 0.04, 0.03, 0.02, 0.02],
    "Glass (heavy plate)":      [0.18, 0.06, 0.04, 0.03, 0.02, 0.02],
    "Carpet (heavy on concrete)":[0.02, 0.06, 0.14, 0.37, 0.60, 0.65],
    "Carpet (heavy on pad)":    [0.08, 0.24, 0.57, 0.69, 0.71, 0.73],
    "Carpet (thin)":            [0.02, 0.04, 0.08, 0.20, 0.35, 0.40],
    "Wood floor":               [0.15, 0.11, 0.10, 0.07, 0.06, 0.07],
    "Linoleum/vinyl on concrete":[0.02, 0.03, 0.03, 0.03, 0.03, 0.02],
    "Acoustic ceiling tile":    [0.70, 0.66, 0.72, 0.92, 0.88, 0.75],
    "Suspended ceiling (mineral)":[0.25, 0.28, 0.46, 0.71, 0.86, 0.93],
    "Fiberglass (1 in, rigid)": [0.06, 0.20, 0.65, 0.90, 0.95, 0.98],
    "Fiberglass (2 in, rigid)": [0.17, 0.55, 0.80, 0.95, 0.98, 0.99],
    "Fiberglass (4 in, rigid)": [0.45, 0.80, 0.95, 0.99, 0.99, 0.99],
    "Open-cell foam (2 in)":    [0.11, 0.30, 0.60, 0.85, 0.90, 0.88],
    "Heavy curtains":           [0.07, 0.31, 0.49, 0.75, 0.70, 0.60],
    "Light curtains":           [0.03, 0.04, 0.11, 0.17, 0.24, 0.35],
    "Upholstered seats (occupied)":[0.60, 0.74, 0.88, 0.96, 0.93, 0.85],
    "Upholstered seats (empty)":[0.49, 0.66, 0.80, 0.88, 0.82, 0.70],
    "Water surface":            [0.01, 0.01, 0.01, 0.02, 0.02, 0.03],
    "Grass/vegetation":         [0.11, 0.26, 0.60, 0.69, 0.92, 0.99],
    "Gravel (loose)":           [0.25, 0.60, 0.65, 0.70, 0.75, 0.80],
    "Snow (fresh, 4 in)":       [0.45, 0.75, 0.90, 0.95, 0.95, 0.95],
    "Audience (per person)":    [0.25, 0.35, 0.42, 0.46, 0.50, 0.50],
}

# ── ISO 226 Equal Loudness Contour Data (simplified) ─────────────────────────
# Frequencies for equal loudness contour data
ISO226_FREQ = np.array([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200,
                         250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
                         2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000,
                         12500])
# SPL values for selected phon levels (20, 40, 60, 80, 100 phon)
ISO226_DATA = {
    20: np.array([78.5, 68.7, 59.5, 51.1, 44.0, 37.5, 31.5, 26.5, 22.1,
                  17.9, 14.4, 11.4, 8.6, 6.2, 4.4, 3.0, 2.2, 2.4, 3.5,
                  1.7, -1.3, -4.2, -6.0, -5.4, -1.5, 6.0, 12.6, 13.9, 12.3]),
    40: np.array([92.4, 84.1, 76.3, 68.9, 62.4, 56.5, 51.0, 46.0, 41.5,
                  37.3, 33.6, 30.3, 27.4, 24.8, 22.7, 20.9, 19.8, 20.0,
                  20.7, 19.0, 16.8, 14.2, 12.5, 13.0, 16.6, 23.1, 28.7,
                  29.0, 26.9]),
    60: np.array([104.0, 97.0, 90.5, 84.2, 78.4, 73.2, 68.3, 63.9, 59.8,
                  56.0, 52.6, 49.5, 46.8, 44.3, 42.3, 40.5, 39.5, 40.0,
                  40.4, 38.8, 36.8, 34.4, 33.0, 33.5, 37.0, 43.0, 47.0,
                  46.0, 43.2]),
    80: np.array([115.0, 109.0, 103.5, 98.1, 93.1, 88.5, 84.0, 80.0, 76.2,
                  72.8, 69.6, 66.8, 64.3, 62.0, 60.1, 58.5, 57.6, 58.0,
                  58.2, 56.8, 55.0, 53.0, 52.0, 52.5, 55.5, 60.5, 63.5,
                  62.5, 59.5]),
    100: np.array([125.5, 120.5, 115.5, 111.0, 107.0, 103.0, 99.2, 96.0,
                   92.5, 89.5, 86.8, 84.3, 82.1, 80.1, 78.5, 77.0, 76.3,
                   77.0, 77.2, 75.8, 74.0, 72.3, 71.5, 72.0, 74.5, 79.0,
                   81.5, 80.0, 77.0]),
}

ABS_OCTAVE_FREQ = [125, 250, 500, 1000, 2000, 4000]  # Hz


# ── Helper Functions ──────────────────────────────────────────────────────────

def pa_to_db(pa):
    """Convert sound pressure [Pa] to dB SPL."""
    return 20 * np.log10(np.maximum(pa, 1e-20) / P_REF)


def db_to_pa(db):
    """Convert dB SPL to sound pressure [Pa]."""
    return P_REF * 10 ** (db / 20.0)


def a_weight(freq):
    """Return A-weighting correction [dB] at given frequency [Hz]."""
    interp = interp1d(A_WEIGHT_FREQ, A_WEIGHT_DB, kind='cubic',
                      bounds_error=False, fill_value='extrapolate')
    return float(interp(freq))


def add_db_levels(levels):
    """Incoherent addition of dB levels."""
    total = np.sum(10 ** (np.array(levels) / 10.0))
    return 10 * np.log10(total)


def sabine_rt60(V, S, alpha_avg):
    """Sabine RT60 reverberation time. V=volume [m^3], S=surface [m^2], alpha=avg absorption."""
    A = S * alpha_avg
    return 0.161 * V / A if A > 0 else float('inf')


def eyring_rt60(V, S, alpha_avg):
    """Eyring RT60 reverberation time."""
    if alpha_avg >= 1.0:
        return 0.0
    return -0.161 * V / (S * np.log(1 - alpha_avg)) if S > 0 else float('inf')


def inverse_square_spl(spl_ref, d_ref, d):
    """SPL at distance d given SPL at reference distance d_ref (point source)."""
    return spl_ref - 20 * np.log10(d / d_ref)


def helmholtz_freq(V_neck, L_neck, A_neck, V_cavity):
    """Helmholtz resonator frequency [Hz].
    V_neck not used; L_neck = neck length, A_neck = neck area, V_cavity = cavity volume."""
    L_eff = L_neck + 1.7 * np.sqrt(A_neck / np.pi)  # end correction
    return (SPEED_SOUND / (2 * np.pi)) * np.sqrt(A_neck / (V_cavity * L_eff))


def pipe_resonance(L, mode, open_both=True):
    """Pipe resonance frequency. mode=1,2,3... L=length [m]."""
    if open_both:
        return mode * SPEED_SOUND / (2 * L)
    else:  # one end closed
        return (2 * mode - 1) * SPEED_SOUND / (4 * L)


def string_resonance(L, T, mu, mode=1):
    """String resonance. L=length, T=tension [N], mu=linear density [kg/m]."""
    return mode / (2 * L) * np.sqrt(T / mu)


def doppler_freq(f_source, v_source, v_observer, v_medium=SPEED_SOUND):
    """Doppler effect: observed frequency.
    Positive v_source = approaching, positive v_observer = approaching."""
    return f_source * (v_medium + v_observer) / (v_medium - v_source)


def transmission_loss_mass_law(f, surface_density):
    """Transmission loss [dB] by mass law. surface_density in kg/m^2."""
    return 20 * np.log10(np.pi * f * surface_density / (RHO_AIR * SPEED_SOUND)) - 47


def stc_rating(tl_values, tl_freqs=None):
    """Estimate STC from TL values at 1/3 octave bands 125-4000 Hz (simplified)."""
    if tl_freqs is None:
        tl_freqs = THIRD_OCTAVE_CENTERS[8:27]  # 125 to 4000
    return int(np.mean(tl_values[:min(len(tl_values), 16)]))


# ══════════════════════════════════════════════════════════════════════════════
#  Widget
# ══════════════════════════════════════════════════════════════════════════════

class AcousticsWidget(QWidget):
    """Main acoustics calculator widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._log = lambda msg: None
        self._init_ui()

    def set_logger(self, fn):
        self._log = fn

    def run(self):
        """Entry point for module runner."""
        self._log("Acoustics module loaded")

    # ── UI Setup ──────────────────────────────────────────────────────────────

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_spl_tab(), "SPL Calculator")
        self.tabs.addTab(self._build_add_db_tab(), "Add dB Levels")
        self.tabs.addTab(self._build_octave_tab(), "Octave Bands")
        self.tabs.addTab(self._build_aweight_tab(), "A-Weighting")
        self.tabs.addTab(self._build_room_tab(), "Room Acoustics")
        self.tabs.addTab(self._build_barrier_tab(), "Noise Reduction")
        self.tabs.addTab(self._build_distance_tab(), "Distance / ISL")
        self.tabs.addTab(self._build_resonance_tab(), "Resonance")
        self.tabs.addTab(self._build_doppler_tab(), "Doppler Effect")
        self.tabs.addTab(self._build_loudness_tab(), "Equal Loudness")
        self.tabs.addTab(self._build_materials_tab(), "Material Database")
        layout.addWidget(self.tabs)

    # ── SPL Calculator Tab ────────────────────────────────────────────────────

    def _build_spl_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        grp = QGroupBox("Sound Pressure Level Converter")
        g = QGridLayout(grp)
        g.addWidget(QLabel("Pressure [Pa]:"), 0, 0)
        self.spl_pa = QDoubleSpinBox(); self.spl_pa.setRange(0, 1000); self.spl_pa.setValue(1.0)
        self.spl_pa.setDecimals(6)
        g.addWidget(self.spl_pa, 0, 1)
        g.addWidget(QLabel("dB SPL:"), 1, 0)
        self.spl_db = QDoubleSpinBox(); self.spl_db.setRange(-20, 200); self.spl_db.setValue(94)
        self.spl_db.setDecimals(2)
        g.addWidget(self.spl_db, 1, 1)
        g.addWidget(QLabel("Frequency [Hz] (for A-weighting):"), 2, 0)
        self.spl_freq = QDoubleSpinBox(); self.spl_freq.setRange(1, 20000); self.spl_freq.setValue(1000)
        g.addWidget(self.spl_freq, 2, 1)
        btn_pa = QPushButton("Pa -> dB"); btn_pa.clicked.connect(self._pa_to_db)
        btn_db = QPushButton("dB -> Pa"); btn_db.clicked.connect(self._db_to_pa)
        g.addWidget(btn_pa, 3, 0); g.addWidget(btn_db, 3, 1)
        self.spl_out = QLabel(""); self.spl_out.setFont(QFont("Consolas", 10))
        g.addWidget(self.spl_out, 4, 0, 1, 2)
        lay.addWidget(grp)
        lay.addStretch()
        return w

    def _pa_to_db(self):
        pa = self.spl_pa.value()
        db = pa_to_db(pa)
        freq = self.spl_freq.value()
        aw = a_weight(freq)
        dba = db + aw
        self.spl_db.setValue(db)
        self.spl_out.setText(f"{pa:.6f} Pa = {db:.2f} dB SPL = {dba:.2f} dB(A) @ {freq:.0f} Hz")
        self._log(f"SPL: {pa} Pa = {db:.2f} dB")

    def _db_to_pa(self):
        db = self.spl_db.value()
        pa = db_to_pa(db)
        freq = self.spl_freq.value()
        aw = a_weight(freq)
        dba = db + aw
        self.spl_pa.setValue(pa)
        self.spl_out.setText(f"{db:.2f} dB SPL = {pa:.6f} Pa = {dba:.2f} dB(A) @ {freq:.0f} Hz")
        self._log(f"SPL: {db} dB = {pa:.6f} Pa")

    # ── Add dB Levels Tab ─────────────────────────────────────────────────────

    def _build_add_db_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        grp = QGroupBox("Incoherent Addition of Sound Levels")
        g = QGridLayout(grp)
        g.addWidget(QLabel("dB levels (comma separated):"), 0, 0)
        self.add_db_input = QLineEdit("85, 88, 82, 90")
        g.addWidget(self.add_db_input, 0, 1)
        btn = QPushButton("Calculate Combined Level"); btn.clicked.connect(self._add_db_levels)
        g.addWidget(btn, 1, 0, 1, 2)
        self.add_db_out = QLabel(""); self.add_db_out.setFont(QFont("Consolas", 10))
        g.addWidget(self.add_db_out, 2, 0, 1, 2)
        lay.addWidget(grp)
        lay.addStretch()
        return w

    def _add_db_levels(self):
        try:
            levels = [float(x.strip()) for x in self.add_db_input.text().split(",") if x.strip()]
            total = add_db_levels(levels)
            self.add_db_out.setText(
                f"Sources: {', '.join(f'{l:.1f}' for l in levels)} dB\n"
                f"Combined: {total:.2f} dB  ({len(levels)} sources)")
            self._log(f"Added {len(levels)} dB levels -> {total:.2f} dB")
        except Exception as e:
            self.add_db_out.setText(f"Error: {e}")

    # ── Octave Band Tab ───────────────────────────────────────────────────────

    def _build_octave_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        top = QHBoxLayout()
        top.addWidget(QLabel("Band type:"))
        self.oct_type = QComboBox(); self.oct_type.addItems(["1/1 Octave", "1/3 Octave"])
        top.addWidget(self.oct_type)
        top.addWidget(QLabel("Levels (comma sep):"))
        self.oct_levels = QLineEdit("65,70,72,68,60,55,50,45,40,35")
        top.addWidget(self.oct_levels)
        btn = QPushButton("Plot"); btn.clicked.connect(self._plot_octave)
        top.addWidget(btn)
        lay.addLayout(top)
        self.oct_fig = Figure(figsize=(7, 3.5))
        self.oct_canvas = FigureCanvas(self.oct_fig)
        lay.addWidget(self.oct_canvas)
        return w

    def _plot_octave(self):
        try:
            levels = [float(x.strip()) for x in self.oct_levels.text().split(",") if x.strip()]
            band_type = self.oct_type.currentText()
            centers = OCTAVE_CENTERS if "1/1" in band_type else THIRD_OCTAVE_CENTERS
            n = min(len(levels), len(centers))
            levels = levels[:n]
            freqs = centers[:n]
            ax = self.oct_fig.clear()
            ax = self.oct_fig.add_subplot(111)
            ax.bar(range(n), levels, color='steelblue', edgecolor='navy', alpha=0.8)
            ax.set_xticks(range(n))
            ax.set_xticklabels([f"{f:.0f}" for f in freqs], rotation=45, fontsize=7)
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel("Level [dB]")
            ax.set_title(f"{band_type} Band Analysis")
            ax.grid(True, alpha=0.3, axis='y')
            self.oct_fig.tight_layout()
            self.oct_canvas.draw()
            self._log(f"Octave band plot: {band_type}, {n} bands")
        except Exception as e:
            self._log(f"Octave plot error: {e}")

    # ── A-Weighting Tab ───────────────────────────────────────────────────────

    def _build_aweight_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        top = QHBoxLayout()
        btn = QPushButton("Plot A-Weighting Curve"); btn.clicked.connect(self._plot_a_weight)
        top.addWidget(btn)
        self.aw_apply = QCheckBox("Apply to octave levels above")
        top.addWidget(self.aw_apply)
        lay.addLayout(top)
        self.aw_fig = Figure(figsize=(7, 4))
        self.aw_canvas = FigureCanvas(self.aw_fig)
        lay.addWidget(self.aw_canvas)
        return w

    def _plot_a_weight(self):
        ax = self.aw_fig.clear()
        ax = self.aw_fig.add_subplot(111)
        freqs = np.logspace(np.log10(10), np.log10(20000), 500)
        interp = interp1d(A_WEIGHT_FREQ, A_WEIGHT_DB, kind='cubic',
                          bounds_error=False, fill_value='extrapolate')
        aw = interp(freqs)
        ax.semilogx(freqs, aw, 'b-', linewidth=2, label='A-weighting')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Relative Response [dB]")
        ax.set_title("A-Weighting Frequency Response (IEC 61672)")
        ax.set_xlim(10, 20000)
        ax.set_ylim(-80, 5)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend()
        self.aw_fig.tight_layout()
        self.aw_canvas.draw()
        self._log("Plotted A-weighting curve")

    # ── Room Acoustics Tab ────────────────────────────────────────────────────

    def _build_room_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        grp = QGroupBox("Reverberation Time (RT60)")
        g = QGridLayout(grp)
        for i, (lbl, val) in enumerate([("Length [m]:", 10), ("Width [m]:", 8),
                                          ("Height [m]:", 3.5)]):
            g.addWidget(QLabel(lbl), i, 0)
            sb = QDoubleSpinBox(); sb.setRange(0.1, 200); sb.setValue(val); sb.setDecimals(2)
            setattr(self, f"room_dim_{i}", sb)
            g.addWidget(sb, i, 1)
        g.addWidget(QLabel("Avg absorption coeff:"), 3, 0)
        self.room_alpha = QDoubleSpinBox(); self.room_alpha.setRange(0.01, 0.99)
        self.room_alpha.setValue(0.15); self.room_alpha.setDecimals(3)
        g.addWidget(self.room_alpha, 3, 1)
        g.addWidget(QLabel("Method:"), 4, 0)
        self.room_method = QComboBox(); self.room_method.addItems(["Sabine", "Eyring", "Both"])
        g.addWidget(self.room_method, 4, 1)
        btn = QPushButton("Calculate RT60"); btn.clicked.connect(self._calc_rt60)
        g.addWidget(btn, 5, 0, 1, 2)
        self.room_out = QTextEdit(); self.room_out.setReadOnly(True); self.room_out.setMaximumHeight(120)
        g.addWidget(self.room_out, 6, 0, 1, 2)
        lay.addWidget(grp)
        lay.addStretch()
        return w

    def _calc_rt60(self):
        L = self.room_dim_0.value()
        W = self.room_dim_1.value()
        H = self.room_dim_2.value()
        alpha = self.room_alpha.value()
        V = L * W * H
        S = 2 * (L*W + L*H + W*H)
        method = self.room_method.currentText()
        lines = [f"Room: {L:.1f} x {W:.1f} x {H:.1f} m", f"Volume = {V:.1f} m^3, Surface = {S:.1f} m^2",
                 f"Avg alpha = {alpha:.3f}", ""]
        if method in ("Sabine", "Both"):
            rt = sabine_rt60(V, S, alpha)
            lines.append(f"Sabine RT60 = {rt:.3f} s")
        if method in ("Eyring", "Both"):
            rt = eyring_rt60(V, S, alpha)
            lines.append(f"Eyring RT60 = {rt:.3f} s")
        self.room_out.setPlainText("\n".join(lines))
        self._log(f"RT60 calc: V={V:.0f} m^3, alpha={alpha}")

    # ── Noise Reduction / Barrier Tab ─────────────────────────────────────────

    def _build_barrier_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        grp = QGroupBox("Transmission Loss (Mass Law)")
        g = QGridLayout(grp)
        g.addWidget(QLabel("Surface density [kg/m^2]:"), 0, 0)
        self.bar_density = QDoubleSpinBox(); self.bar_density.setRange(0.1, 500); self.bar_density.setValue(20)
        g.addWidget(self.bar_density, 0, 1)
        btn = QPushButton("Plot TL vs Frequency"); btn.clicked.connect(self._plot_tl)
        g.addWidget(btn, 1, 0, 1, 2)
        self.bar_out = QLabel(""); self.bar_out.setFont(QFont("Consolas", 10))
        g.addWidget(self.bar_out, 2, 0, 1, 2)
        lay.addWidget(grp)
        self.bar_fig = Figure(figsize=(6, 3.5))
        self.bar_canvas = FigureCanvas(self.bar_fig)
        lay.addWidget(self.bar_canvas)
        return w

    def _plot_tl(self):
        rho_s = self.bar_density.value()
        freqs = np.logspace(np.log10(50), np.log10(10000), 200)
        tl = transmission_loss_mass_law(freqs, rho_s)
        stc_est = stc_rating(tl[::10])
        ax = self.bar_fig.clear()
        ax = self.bar_fig.add_subplot(111)
        ax.semilogx(freqs, tl, 'b-', linewidth=2)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Transmission Loss [dB]")
        ax.set_title(f"Mass Law TL (surface density = {rho_s:.1f} kg/m^2)")
        ax.grid(True, which='both', alpha=0.3)
        self.bar_fig.tight_layout()
        self.bar_canvas.draw()
        self.bar_out.setText(f"Estimated STC ~ {stc_est}")
        self._log(f"TL plot: rho_s={rho_s} kg/m^2, STC~{stc_est}")

    # ── Inverse Square Law Tab ────────────────────────────────────────────────

    def _build_distance_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        grp = QGroupBox("SPL vs Distance (Inverse Square Law)")
        g = QGridLayout(grp)
        g.addWidget(QLabel("SPL at ref distance [dB]:"), 0, 0)
        self.isl_spl = QDoubleSpinBox(); self.isl_spl.setRange(0, 200); self.isl_spl.setValue(100)
        g.addWidget(self.isl_spl, 0, 1)
        g.addWidget(QLabel("Ref distance [m]:"), 1, 0)
        self.isl_dref = QDoubleSpinBox(); self.isl_dref.setRange(0.01, 1000); self.isl_dref.setValue(1.0)
        g.addWidget(self.isl_dref, 1, 1)
        g.addWidget(QLabel("Target distance [m]:"), 2, 0)
        self.isl_d = QDoubleSpinBox(); self.isl_d.setRange(0.01, 10000); self.isl_d.setValue(10.0)
        g.addWidget(self.isl_d, 2, 1)
        btn = QPushButton("Calculate & Plot"); btn.clicked.connect(self._calc_isl)
        g.addWidget(btn, 3, 0, 1, 2)
        self.isl_out = QLabel(""); self.isl_out.setFont(QFont("Consolas", 10))
        g.addWidget(self.isl_out, 4, 0, 1, 2)
        lay.addWidget(grp)
        self.isl_fig = Figure(figsize=(6, 3))
        self.isl_canvas = FigureCanvas(self.isl_fig)
        lay.addWidget(self.isl_canvas)
        return w

    def _calc_isl(self):
        spl_ref = self.isl_spl.value()
        d_ref = self.isl_dref.value()
        d = self.isl_d.value()
        spl_d = inverse_square_spl(spl_ref, d_ref, d)
        self.isl_out.setText(f"SPL at {d:.1f} m = {spl_d:.2f} dB (from {spl_ref:.1f} dB at {d_ref:.1f} m)")
        distances = np.logspace(np.log10(max(d_ref * 0.5, 0.1)), np.log10(d * 3), 200)
        spls = inverse_square_spl(spl_ref, d_ref, distances)
        ax = self.isl_fig.clear()
        ax = self.isl_fig.add_subplot(111)
        ax.semilogx(distances, spls, 'b-', linewidth=2)
        ax.axhline(spl_d, color='r', linestyle='--', alpha=0.5, label=f'{spl_d:.1f} dB at {d:.1f} m')
        ax.axvline(d, color='r', linestyle=':', alpha=0.5)
        ax.set_xlabel("Distance [m]"); ax.set_ylabel("SPL [dB]")
        ax.set_title("Inverse Square Law"); ax.legend(fontsize=8)
        ax.grid(True, which='both', alpha=0.3)
        self.isl_fig.tight_layout()
        self.isl_canvas.draw()
        self._log(f"ISL: {spl_ref} dB at {d_ref} m -> {spl_d:.2f} dB at {d} m")

    # ── Resonance Tab ─────────────────────────────────────────────────────────

    def _build_resonance_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        # Helmholtz
        grp1 = QGroupBox("Helmholtz Resonator")
        g1 = QGridLayout(grp1)
        g1.addWidget(QLabel("Neck length [m]:"), 0, 0)
        self.helm_L = QDoubleSpinBox(); self.helm_L.setRange(0.001, 2); self.helm_L.setValue(0.05); self.helm_L.setDecimals(4)
        g1.addWidget(self.helm_L, 0, 1)
        g1.addWidget(QLabel("Neck area [m^2]:"), 1, 0)
        self.helm_A = QDoubleSpinBox(); self.helm_A.setRange(1e-6, 1); self.helm_A.setValue(0.001); self.helm_A.setDecimals(6)
        g1.addWidget(self.helm_A, 1, 1)
        g1.addWidget(QLabel("Cavity volume [m^3]:"), 2, 0)
        self.helm_V = QDoubleSpinBox(); self.helm_V.setRange(1e-6, 10); self.helm_V.setValue(0.005); self.helm_V.setDecimals(6)
        g1.addWidget(self.helm_V, 2, 1)
        btn1 = QPushButton("Calculate"); btn1.clicked.connect(self._calc_helmholtz)
        g1.addWidget(btn1, 3, 0, 1, 2)
        self.helm_out = QLabel(""); self.helm_out.setFont(QFont("Consolas", 10))
        g1.addWidget(self.helm_out, 4, 0, 1, 2)
        lay.addWidget(grp1)
        # Pipe
        grp2 = QGroupBox("Pipe Resonance")
        g2 = QGridLayout(grp2)
        g2.addWidget(QLabel("Pipe length [m]:"), 0, 0)
        self.pipe_L = QDoubleSpinBox(); self.pipe_L.setRange(0.01, 50); self.pipe_L.setValue(1.0)
        g2.addWidget(self.pipe_L, 0, 1)
        g2.addWidget(QLabel("Modes to show:"), 1, 0)
        self.pipe_modes = QSpinBox(); self.pipe_modes.setRange(1, 10); self.pipe_modes.setValue(5)
        g2.addWidget(self.pipe_modes, 1, 1)
        self.pipe_type = QComboBox(); self.pipe_type.addItems(["Open-Open", "Open-Closed"])
        g2.addWidget(QLabel("Pipe type:"), 2, 0); g2.addWidget(self.pipe_type, 2, 1)
        btn2 = QPushButton("Calculate"); btn2.clicked.connect(self._calc_pipe)
        g2.addWidget(btn2, 3, 0, 1, 2)
        self.pipe_out = QTextEdit(); self.pipe_out.setReadOnly(True); self.pipe_out.setMaximumHeight(100)
        g2.addWidget(self.pipe_out, 4, 0, 1, 2)
        lay.addWidget(grp2)
        # String
        grp3 = QGroupBox("String Resonance")
        g3 = QGridLayout(grp3)
        g3.addWidget(QLabel("Length [m]:"), 0, 0)
        self.str_L = QDoubleSpinBox(); self.str_L.setRange(0.01, 10); self.str_L.setValue(0.65)
        g3.addWidget(self.str_L, 0, 1)
        g3.addWidget(QLabel("Tension [N]:"), 1, 0)
        self.str_T = QDoubleSpinBox(); self.str_T.setRange(0.1, 5000); self.str_T.setValue(73)
        g3.addWidget(self.str_T, 1, 1)
        g3.addWidget(QLabel("Linear density [kg/m]:"), 2, 0)
        self.str_mu = QDoubleSpinBox(); self.str_mu.setRange(1e-5, 1); self.str_mu.setValue(0.0039); self.str_mu.setDecimals(5)
        g3.addWidget(self.str_mu, 2, 1)
        btn3 = QPushButton("Calculate (first 5 modes)"); btn3.clicked.connect(self._calc_string)
        g3.addWidget(btn3, 3, 0, 1, 2)
        self.str_out = QLabel(""); self.str_out.setFont(QFont("Consolas", 10))
        g3.addWidget(self.str_out, 4, 0, 1, 2)
        lay.addWidget(grp3)
        return w

    def _calc_helmholtz(self):
        f = helmholtz_freq(0, self.helm_L.value(), self.helm_A.value(), self.helm_V.value())
        self.helm_out.setText(f"Helmholtz resonance = {f:.2f} Hz")
        self._log(f"Helmholtz: {f:.2f} Hz")

    def _calc_pipe(self):
        L = self.pipe_L.value()
        n = self.pipe_modes.value()
        open_both = "Open-Open" in self.pipe_type.currentText()
        lines = [f"Pipe: L={L:.2f} m, {'Open-Open' if open_both else 'Open-Closed'}"]
        for m in range(1, n + 1):
            f = pipe_resonance(L, m, open_both)
            lines.append(f"  Mode {m}: {f:.2f} Hz")
        self.pipe_out.setPlainText("\n".join(lines))
        self._log(f"Pipe resonance: {n} modes")

    def _calc_string(self):
        L = self.str_L.value()
        T = self.str_T.value()
        mu = self.str_mu.value()
        modes = [string_resonance(L, T, mu, m) for m in range(1, 6)]
        text = "  ".join(f"n{i+1}={f:.1f} Hz" for i, f in enumerate(modes))
        self.str_out.setText(text)
        self._log(f"String resonance: f1={modes[0]:.1f} Hz")

    # ── Doppler Effect Tab ────────────────────────────────────────────────────

    def _build_doppler_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        grp = QGroupBox("Doppler Effect Calculator")
        g = QGridLayout(grp)
        g.addWidget(QLabel("Source frequency [Hz]:"), 0, 0)
        self.dop_f = QDoubleSpinBox(); self.dop_f.setRange(1, 100000); self.dop_f.setValue(1000)
        g.addWidget(self.dop_f, 0, 1)
        g.addWidget(QLabel("Source velocity [m/s]:"), 1, 0)
        self.dop_vs = QDoubleSpinBox(); self.dop_vs.setRange(-340, 340); self.dop_vs.setValue(30)
        g.addWidget(self.dop_vs, 1, 1)
        g.addWidget(QLabel("Observer velocity [m/s]:"), 2, 0)
        self.dop_vo = QDoubleSpinBox(); self.dop_vo.setRange(-340, 340); self.dop_vo.setValue(0)
        g.addWidget(self.dop_vo, 2, 1)
        g.addWidget(QLabel("(positive = approaching)"), 3, 0, 1, 2)
        btn = QPushButton("Calculate"); btn.clicked.connect(self._calc_doppler)
        g.addWidget(btn, 4, 0, 1, 2)
        self.dop_out = QTextEdit(); self.dop_out.setReadOnly(True); self.dop_out.setMaximumHeight(100)
        g.addWidget(self.dop_out, 5, 0, 1, 2)
        lay.addWidget(grp)
        lay.addStretch()
        return w

    def _calc_doppler(self):
        f_s = self.dop_f.value()
        v_s = self.dop_vs.value()
        v_o = self.dop_vo.value()
        f_obs = doppler_freq(f_s, v_s, v_o)
        ratio = f_obs / f_s
        lines = [
            f"Source: {f_s:.1f} Hz, v_source = {v_s:.1f} m/s",
            f"Observer: v_observer = {v_o:.1f} m/s",
            f"Observed frequency = {f_obs:.2f} Hz",
            f"Frequency ratio = {ratio:.4f}",
            f"Apparent wavelength = {SPEED_SOUND / f_obs:.4f} m",
        ]
        self.dop_out.setPlainText("\n".join(lines))
        self._log(f"Doppler: {f_s} Hz -> {f_obs:.2f} Hz")

    # ── Equal Loudness Contours Tab ───────────────────────────────────────────

    def _build_loudness_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        btn = QPushButton("Plot Equal Loudness Contours (ISO 226)")
        btn.clicked.connect(self._plot_loudness)
        lay.addWidget(btn)
        self.loud_fig = Figure(figsize=(7, 5))
        self.loud_canvas = FigureCanvas(self.loud_fig)
        lay.addWidget(self.loud_canvas)
        return w

    def _plot_loudness(self):
        ax = self.loud_fig.clear()
        ax = self.loud_fig.add_subplot(111)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for idx, (phon, spl_vals) in enumerate(sorted(ISO226_DATA.items())):
            n = min(len(ISO226_FREQ), len(spl_vals))
            ax.semilogx(ISO226_FREQ[:n], spl_vals[:n], '-o', markersize=3,
                        color=colors[idx % len(colors)], label=f"{phon} phon")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("SPL [dB]")
        ax.set_title("Equal Loudness Contours (ISO 226)")
        ax.set_xlim(20, 16000)
        ax.set_ylim(-10, 130)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=8)
        self.loud_fig.tight_layout()
        self.loud_canvas.draw()
        self._log("Plotted equal loudness contours")

    # ── Material Database Tab ─────────────────────────────────────────────────

    def _build_materials_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        top = QHBoxLayout()
        top.addWidget(QLabel("Search:"))
        self.mat_search = QLineEdit(); self.mat_search.setPlaceholderText("Filter materials...")
        self.mat_search.textChanged.connect(self._filter_materials)
        top.addWidget(self.mat_search)
        lay.addLayout(top)
        self.mat_table = QTableWidget()
        self.mat_table.setColumnCount(7)
        headers = ["Material"] + [f"{f} Hz" for f in ABS_OCTAVE_FREQ]
        self.mat_table.setHorizontalHeaderLabels(headers)
        self.mat_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._populate_materials("")
        lay.addWidget(self.mat_table)
        return w

    def _populate_materials(self, filt):
        filt = filt.lower()
        rows = [(name, vals) for name, vals in MATERIAL_ABSORPTION.items() if filt in name.lower()]
        self.mat_table.setRowCount(len(rows))
        for i, (name, vals) in enumerate(rows):
            self.mat_table.setItem(i, 0, QTableWidgetItem(name))
            for j, v in enumerate(vals):
                self.mat_table.setItem(i, j + 1, QTableWidgetItem(f"{v:.3f}"))

    def _filter_materials(self, text):
        self._populate_materials(text)
