"""
Color Science Widget for QuantumRes.
Comprehensive color science tool: CIE chromaticity, color space conversions,
spectral-to-color, color temperature, color blindness simulation,
harmony, contrast checking, palette generation, and Delta-E computation.
"""

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QLabel, QLineEdit, QComboBox, QGroupBox, QSplitter, QFrame,
    QTabWidget, QSlider, QSpinBox, QDoubleSpinBox, QColorDialog,
    QScrollArea, QTextEdit, QApplication, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPainter, QPixmap, QBrush, QPen

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection


# ── CIE 1931 Standard Observer (2-degree) abridged data ─────────────────────
# Wavelengths 380-780nm in 5nm steps with corresponding x_bar, y_bar, z_bar
_CIE_WL = np.arange(380, 785, 5)
_CIE_X = np.array([
    0.0014,0.0022,0.0042,0.0076,0.0143,0.0232,0.0435,0.0776,0.1344,0.2148,
    0.2839,0.3285,0.3483,0.3481,0.3362,0.3187,0.2908,0.2511,0.1954,0.1421,
    0.0956,0.058,0.032,0.0147,0.0049,0.0024,0.0093,0.0291,0.0633,0.1096,
    0.1655,0.2257,0.2904,0.3597,0.4334,0.5121,0.5945,0.6784,0.7621,0.8425,
    0.9163,0.9786,1.0263,1.0567,1.0622,1.0456,1.0026,0.9384,0.8544,0.7514,
    0.6424,0.5419,0.4479,0.3608,0.2835,0.2187,0.1649,0.1212,0.0874,0.0636,
    0.0468,0.0329,0.0227,0.0158,0.0114,0.0081,0.0058,0.0041,0.0029,0.002,
    0.0014,0.001,0.0007,0.0005,0.0003,0.0002,0.0002,0.0001,0.0001,0.0001,
    0.0
])
_CIE_Y = np.array([
    0.0,0.0001,0.0001,0.0002,0.0004,0.0006,0.0012,0.0022,0.004,0.0073,
    0.0116,0.017,0.0241,0.0328,0.0468,0.061,0.079,0.109,0.139,0.208,
    0.323,0.503,0.71,0.862,0.954,0.995,0.995,0.952,0.87,0.757,
    0.631,0.503,0.381,0.265,0.175,0.107,0.061,0.032,0.017,0.0082,
    0.0041,0.0021,0.001,0.0005,0.00025,0.00012,0.00006,0.00003,0.00002,0.00001,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0
])
_CIE_Z = np.array([
    0.0065,0.0105,0.0201,0.0362,0.0679,0.1102,0.2074,0.3713,0.6456,1.0391,
    1.3856,1.623,1.7471,1.7826,1.7721,1.7441,1.6692,1.5281,1.2876,1.0419,
    0.8130,0.6162,0.4652,0.3533,0.272,0.2123,0.1582,0.1117,0.0782,0.0573,
    0.0422,0.0298,0.0203,0.0134,0.0087,0.0057,0.0039,0.0027,0.0021,0.0018,
    0.0017,0.0014,0.0011,0.001,0.0008,0.0006,0.0003,0.0002,0.0002,0.0001,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0
])


# ── Conversion Helpers ───────────────────────────────────────────────────────

def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def rgb_to_xyz(r, g, b):
    """sRGB (0-1) -> CIE XYZ (D65)."""
    def linearize(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    rl, gl, bl = linearize(r), linearize(g), linearize(b)
    X = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl
    Y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl
    Z = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl
    return X, Y, Z


def xyz_to_rgb(X, Y, Z):
    """CIE XYZ -> sRGB (0-1), clamped."""
    rl =  3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    gl = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    bl =  0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z
    def gamma(c):
        c = _clamp(c, 0, 1)
        return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1.0 / 2.4) - 0.055
    return gamma(rl), gamma(gl), gamma(bl)


def rgb_to_hsv(r, g, b):
    c = QColor.fromRgbF(r, g, b)
    return c.hsvHueF(), c.hsvSaturationF(), c.valueF()


def hsv_to_rgb(h, s, v):
    c = QColor.fromHsvF(h if h >= 0 else 0, s, v)
    return c.redF(), c.greenF(), c.blueF()


def rgb_to_hsl(r, g, b):
    c = QColor.fromRgbF(r, g, b)
    return c.hslHueF(), c.hslSaturationF(), c.lightnessF()


def hsl_to_rgb(h, s, l):
    c = QColor.fromHslF(h if h >= 0 else 0, s, l)
    return c.redF(), c.greenF(), c.blueF()


def xyz_to_lab(X, Y, Z):
    """XYZ -> CIELAB (D65 illuminant, Xn=0.9505, Yn=1.0, Zn=1.089)."""
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    def f(t):
        return t ** (1/3) if t > 0.008856 else 7.787 * t + 16/116
    L = 116 * f(Y / Yn) - 16
    a = 500 * (f(X / Xn) - f(Y / Yn))
    b = 200 * (f(Y / Yn) - f(Z / Zn))
    return L, a, b


def lab_to_xyz(L, a, b):
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    def inv_f(t):
        return t ** 3 if t ** 3 > 0.008856 else (t - 16/116) / 7.787
    return Xn * inv_f(fx), Yn * inv_f(fy), Zn * inv_f(fz)


def rgb_to_cmyk(r, g, b):
    k = 1 - max(r, g, b)
    if k >= 1.0:
        return 0, 0, 0, 1
    c = (1 - r - k) / (1 - k)
    m = (1 - g - k) / (1 - k)
    y = (1 - b - k) / (1 - k)
    return c, m, y, k


def cmyk_to_rgb(c, m, y, k):
    r = (1 - c) * (1 - k)
    g = (1 - m) * (1 - k)
    b = (1 - y) * (1 - k)
    return _clamp(r), _clamp(g), _clamp(b)


def wavelength_to_rgb(wl):
    """Convert wavelength (380-780 nm) to approximate sRGB (0-1)."""
    if wl < 380 or wl > 780:
        return 0, 0, 0
    # Use CIE data interpolation
    X = np.interp(wl, _CIE_WL, _CIE_X)
    Y = np.interp(wl, _CIE_WL, _CIE_Y)
    Z = np.interp(wl, _CIE_WL, _CIE_Z)
    r, g, b = xyz_to_rgb(X, Y, Z)
    # Intensity correction at edges
    if wl < 420:
        factor = 0.3 + 0.7 * (wl - 380) / 40
    elif wl > 700:
        factor = 0.3 + 0.7 * (780 - wl) / 80
    else:
        factor = 1.0
    return _clamp(r * factor), _clamp(g * factor), _clamp(b * factor)


def kelvin_to_rgb(T):
    """Approximate color of a blackbody at temperature T Kelvin."""
    t = T / 100.0
    if t <= 66:
        r = 255
        g = _clamp(99.4708025861 * np.log(t) - 161.1195681661, 0, 255)
        b = 0 if t <= 19 else _clamp(138.5177312231 * np.log(t - 10) - 305.0447927307, 0, 255)
    else:
        r = _clamp(329.698727446 * ((t - 60) ** -0.1332047592), 0, 255)
        g = _clamp(288.1221695283 * ((t - 60) ** -0.0755148492), 0, 255)
        b = 255
    return r / 255, g / 255, b / 255


def planckian_locus_xy(T):
    """CIE xy coordinates on the Planckian locus for temperature T."""
    if T < 1667:
        T = 1667
    if T <= 4000:
        x = (-0.2661239e9 / T**3 - 0.2343589e6 / T**2 + 0.8776956e3 / T + 0.179910)
    else:
        x = (-3.0258469e9 / T**3 + 2.1070379e6 / T**2 + 0.2226347e3 / T + 0.240390)
    if T <= 2222:
        y = (-1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683)
    elif T <= 4000:
        y = (-0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867)
    else:
        y = (3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483)
    return x, y


def relative_luminance(r, g, b):
    """WCAG relative luminance from sRGB 0-1."""
    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def contrast_ratio(c1, c2):
    L1 = relative_luminance(*c1) + 0.05
    L2 = relative_luminance(*c2) + 0.05
    return max(L1, L2) / min(L1, L2)


def delta_e_76(lab1, lab2):
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2)))


def delta_e_2000(lab1, lab2):
    """CIE DE2000 color difference."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    Lb = (L1 + L2) / 2
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    Cb = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cb**7 / (Cb**7 + 25**7)))
    a1p = a1 * (1 + G)
    a2p = a2 * (1 + G)
    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)
    Cbp = (C1p + C2p) / 2
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    if abs(h1p - h2p) <= 180:
        Hbp = (h1p + h2p) / 2
    elif h1p + h2p < 360:
        Hbp = (h1p + h2p + 360) / 2
    else:
        Hbp = (h1p + h2p - 360) / 2
    T = (1 - 0.17 * np.cos(np.radians(Hbp - 30)) + 0.24 * np.cos(np.radians(2 * Hbp))
         + 0.32 * np.cos(np.radians(3 * Hbp + 6)) - 0.20 * np.cos(np.radians(4 * Hbp - 63)))
    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    if abs(dhp) > 180:
        dhp += 360 if dhp < 0 else -360
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))
    SL = 1 + 0.015 * (Lb - 50)**2 / np.sqrt(20 + (Lb - 50)**2)
    SC = 1 + 0.045 * Cbp
    SH = 1 + 0.015 * Cbp * T
    RT = (-2 * np.sqrt(Cbp**7 / (Cbp**7 + 25**7))
          * np.sin(np.radians(60 * np.exp(-((Hbp - 275) / 25)**2))))
    return np.sqrt((dLp / SL)**2 + (dCp / SC)**2 + (dHp / SH)**2 + RT * (dCp / SC) * (dHp / SH))


# ── Color blindness simulation matrices (from Brettel/Vienot) ────────────────

_CB_MATRICES = {
    "Protanopia": np.array([
        [0.56667, 0.43333, 0.0],
        [0.55833, 0.44167, 0.0],
        [0.0, 0.24167, 0.75833]
    ]),
    "Deuteranopia": np.array([
        [0.625, 0.375, 0.0],
        [0.70, 0.30, 0.0],
        [0.0, 0.30, 0.70]
    ]),
    "Tritanopia": np.array([
        [0.95, 0.05, 0.0],
        [0.0, 0.43333, 0.56667],
        [0.0, 0.475, 0.525]
    ]),
}


def simulate_color_blindness(r, g, b, cb_type):
    mat = _CB_MATRICES.get(cb_type)
    if mat is None:
        return r, g, b
    rgb = np.array([r, g, b])
    result = mat @ rgb
    return tuple(_clamp(v) for v in result)


def _color_swatch_pixmap(r, g, b, w=48, h=48):
    pm = QPixmap(w, h)
    pm.fill(QColor(int(r * 255), int(g * 255), int(b * 255)))
    return pm


# ── Main Widget ──────────────────────────────────────────────────────────────

class ColorScienceWidget(QWidget):
    """Comprehensive color science tool with CIE diagram, conversions, and analysis."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._log = lambda msg: None
        self._current_rgb = (1.0, 0.333, 0.0)  # default orange
        self._second_rgb = (0.0, 0.0, 0.0)
        self._updating = False
        self._init_ui()
        self._refresh_all()

    def set_logger(self, fn):
        self._log = fn

    # ── UI Construction ──────────────────────────────────────────────────

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        tabs = QTabWidget()
        tabs.addTab(self._build_chromaticity_tab(), "CIE Diagram")
        tabs.addTab(self._build_conversion_tab(), "Conversions")
        tabs.addTab(self._build_spectral_tab(), "Spectral / Temp")
        tabs.addTab(self._build_blindness_tab(), "Color Blindness")
        tabs.addTab(self._build_harmony_tab(), "Harmony")
        tabs.addTab(self._build_contrast_tab(), "Contrast / Delta-E")
        tabs.addTab(self._build_palette_tab(), "Palette")
        layout.addWidget(tabs)

    # -- CIE Chromaticity Diagram tab --
    def _build_chromaticity_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self._cie_fig = Figure(figsize=(5, 5), dpi=100)
        self._cie_canvas = FigureCanvas(self._cie_fig)
        self._cie_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self._cie_canvas)
        btn_row = QHBoxLayout()
        btn = QPushButton("Refresh Diagram")
        btn.clicked.connect(self._draw_cie_diagram)
        btn_row.addWidget(btn)
        self._cie_temp_spin = QSpinBox()
        self._cie_temp_spin.setRange(1000, 25000)
        self._cie_temp_spin.setValue(6500)
        self._cie_temp_spin.setSuffix(" K")
        btn_row.addWidget(QLabel("Temp:"))
        btn_row.addWidget(self._cie_temp_spin)
        btn_show = QPushButton("Show Temp on Locus")
        btn_show.clicked.connect(self._draw_cie_diagram)
        btn_row.addWidget(btn_show)
        lay.addLayout(btn_row)
        return w

    def _draw_cie_diagram(self):
        fig = self._cie_fig
        fig.clear()
        ax = fig.add_subplot(111)
        # Spectral locus
        S = _CIE_X + _CIE_Y + _CIE_Z
        mask = S > 0
        sx = np.where(mask, _CIE_X / S, 0)
        sy = np.where(mask, _CIE_Y / S, 0)
        valid = mask & (S > 0.001)
        ax.plot(sx[valid], sy[valid], 'k-', linewidth=1.2, label='Spectral locus')
        ax.plot([sx[valid][-1], sx[valid][0]], [sy[valid][-1], sy[valid][0]], 'k--', linewidth=0.8)
        # Color fill approximation
        for i in range(len(_CIE_WL)):
            if not valid[i]:
                continue
            r, g, b = wavelength_to_rgb(_CIE_WL[i])
            ax.plot(sx[i], sy[i], 'o', color=(r, g, b), markersize=3)
        # sRGB gamut triangle
        srgb_verts = np.array([
            [0.64, 0.33], [0.30, 0.60], [0.15, 0.06], [0.64, 0.33]
        ])
        ax.plot(srgb_verts[:, 0], srgb_verts[:, 1], 'b-', linewidth=1.5, label='sRGB')
        # Planckian locus
        temps = np.arange(1667, 25001, 100)
        px, py = [], []
        for T in temps:
            x, y = planckian_locus_xy(T)
            px.append(x)
            py.append(y)
        ax.plot(px, py, 'r-', linewidth=1.2, label='Planckian locus')
        # Mark selected temperature
        T = self._cie_temp_spin.value()
        tx, ty = planckian_locus_xy(T)
        ax.plot(tx, ty, 'r*', markersize=12, label=f'{T} K')
        # Current color point
        X, Y, Z = rgb_to_xyz(*self._current_rgb)
        S_cur = X + Y + Z
        if S_cur > 0:
            ax.plot(X / S_cur, Y / S_cur, 'ko', markersize=8, markerfacecolor='white', label='Current')
        ax.set_xlim(-0.05, 0.85)
        ax.set_ylim(-0.05, 0.90)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('CIE 1931 Chromaticity Diagram')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
        self._cie_canvas.draw()

    # -- Conversion tab --
    def _build_conversion_tab(self):
        w = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        lay = QVBoxLayout(inner)

        # Color picker button and hex
        pick_row = QHBoxLayout()
        self._pick_btn = QPushButton("Pick Color")
        self._pick_btn.clicked.connect(self._open_color_dialog)
        pick_row.addWidget(self._pick_btn)
        pick_row.addWidget(QLabel("Hex:"))
        self._hex_edit = QLineEdit("#FF5500")
        self._hex_edit.setMaximumWidth(100)
        self._hex_edit.editingFinished.connect(self._hex_changed)
        pick_row.addWidget(self._hex_edit)
        self._swatch_label = QLabel()
        self._swatch_label.setFixedSize(48, 48)
        pick_row.addWidget(self._swatch_label)
        pick_row.addStretch()
        lay.addLayout(pick_row)

        # RGB sliders
        rgb_grp = QGroupBox("RGB (0-255)")
        rgb_lay = QGridLayout(rgb_grp)
        self._rgb_sliders = []
        for i, name in enumerate(["R", "G", "B"]):
            rgb_lay.addWidget(QLabel(name), i, 0)
            sl = QSlider(Qt.Horizontal)
            sl.setRange(0, 255)
            sl.valueChanged.connect(self._rgb_slider_changed)
            rgb_lay.addWidget(sl, i, 1)
            sp = QSpinBox()
            sp.setRange(0, 255)
            sp.valueChanged.connect(self._rgb_slider_changed)
            rgb_lay.addWidget(sp, i, 2)
            self._rgb_sliders.append((sl, sp))
        lay.addWidget(rgb_grp)

        # HSV sliders
        hsv_grp = QGroupBox("HSV")
        hsv_lay = QGridLayout(hsv_grp)
        self._hsv_sliders = []
        for i, (name, mx) in enumerate([("H", 359), ("S", 100), ("V", 100)]):
            hsv_lay.addWidget(QLabel(name), i, 0)
            sl = QSlider(Qt.Horizontal)
            sl.setRange(0, mx)
            sl.valueChanged.connect(self._hsv_slider_changed)
            hsv_lay.addWidget(sl, i, 1)
            sp = QSpinBox()
            sp.setRange(0, mx)
            sp.valueChanged.connect(self._hsv_slider_changed)
            hsv_lay.addWidget(sp, i, 2)
            self._hsv_sliders.append((sl, sp))
        lay.addWidget(hsv_grp)

        # Read-only displays for HSL, LAB, XYZ, CMYK
        info_grp = QGroupBox("Other Spaces")
        info_lay = QGridLayout(info_grp)
        self._hsl_label = QLabel()
        self._lab_label = QLabel()
        self._xyz_label = QLabel()
        self._cmyk_label = QLabel()
        for i, (name, lbl) in enumerate([("HSL", self._hsl_label), ("LAB", self._lab_label),
                                          ("XYZ", self._xyz_label), ("CMYK", self._cmyk_label)]):
            info_lay.addWidget(QLabel(name + ":"), i, 0)
            info_lay.addWidget(lbl, i, 1)
        lay.addWidget(info_grp)
        lay.addStretch()
        scroll.setWidget(inner)

        outer = QVBoxLayout(w)
        outer.addWidget(scroll)
        return w

    def _open_color_dialog(self):
        r, g, b = self._current_rgb
        c = QColorDialog.getColor(QColor(int(r * 255), int(g * 255), int(b * 255)), self)
        if c.isValid():
            self._current_rgb = (c.redF(), c.greenF(), c.blueF())
            self._refresh_all()

    def _hex_changed(self):
        if self._updating:
            return
        txt = self._hex_edit.text().strip()
        if not txt.startswith('#'):
            txt = '#' + txt
        c = QColor(txt)
        if c.isValid():
            self._current_rgb = (c.redF(), c.greenF(), c.blueF())
            self._refresh_all()

    def _rgb_slider_changed(self):
        if self._updating:
            return
        self._updating = True
        for sl, sp in self._rgb_sliders:
            if sl.value() != sp.value():
                if self.sender() == sl:
                    sp.setValue(sl.value())
                else:
                    sl.setValue(sp.value())
        r = self._rgb_sliders[0][0].value() / 255
        g = self._rgb_sliders[1][0].value() / 255
        b = self._rgb_sliders[2][0].value() / 255
        self._current_rgb = (r, g, b)
        self._updating = False
        self._refresh_all(skip_rgb=True)

    def _hsv_slider_changed(self):
        if self._updating:
            return
        self._updating = True
        for sl, sp in self._hsv_sliders:
            if sl.value() != sp.value():
                if self.sender() == sl:
                    sp.setValue(sl.value())
                else:
                    sl.setValue(sp.value())
        h = self._hsv_sliders[0][0].value() / 359
        s = self._hsv_sliders[1][0].value() / 100
        v = self._hsv_sliders[2][0].value() / 100
        self._current_rgb = hsv_to_rgb(h, s, v)
        self._updating = False
        self._refresh_all(skip_hsv=True)

    def _refresh_all(self, skip_rgb=False, skip_hsv=False):
        self._updating = True
        r, g, b = self._current_rgb
        # Swatch
        self._swatch_label.setPixmap(_color_swatch_pixmap(r, g, b))
        # Hex
        self._hex_edit.setText(f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}")
        # RGB sliders
        if not skip_rgb:
            for i, v in enumerate([r, g, b]):
                self._rgb_sliders[i][0].setValue(int(v * 255))
                self._rgb_sliders[i][1].setValue(int(v * 255))
        # HSV sliders
        if not skip_hsv:
            h, s, v = rgb_to_hsv(r, g, b)
            self._hsv_sliders[0][0].setValue(int(h * 359))
            self._hsv_sliders[0][1].setValue(int(h * 359))
            self._hsv_sliders[1][0].setValue(int(s * 100))
            self._hsv_sliders[1][1].setValue(int(s * 100))
            self._hsv_sliders[2][0].setValue(int(v * 100))
            self._hsv_sliders[2][1].setValue(int(v * 100))
        # HSL
        h2, s2, l2 = rgb_to_hsl(r, g, b)
        self._hsl_label.setText(f"H={h2*360:.1f}  S={s2*100:.1f}%  L={l2*100:.1f}%")
        # LAB
        X, Y, Z = rgb_to_xyz(r, g, b)
        L, a, bb = xyz_to_lab(X, Y, Z)
        self._lab_label.setText(f"L={L:.2f}  a={a:.2f}  b={bb:.2f}")
        # XYZ
        self._xyz_label.setText(f"X={X:.4f}  Y={Y:.4f}  Z={Z:.4f}")
        # CMYK
        c_, m_, y_, k_ = rgb_to_cmyk(r, g, b)
        self._cmyk_label.setText(f"C={c_*100:.1f}%  M={m_*100:.1f}%  Y={y_*100:.1f}%  K={k_*100:.1f}%")
        self._updating = False

    # -- Spectral / Temperature tab --
    def _build_spectral_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        # Wavelength
        wl_grp = QGroupBox("Spectral Wavelength to Color")
        wl_lay = QHBoxLayout(wl_grp)
        wl_lay.addWidget(QLabel("Wavelength (nm):"))
        self._wl_spin = QSpinBox()
        self._wl_spin.setRange(380, 780)
        self._wl_spin.setValue(550)
        wl_lay.addWidget(self._wl_spin)
        self._wl_btn = QPushButton("Show")
        self._wl_btn.clicked.connect(self._show_wavelength)
        wl_lay.addWidget(self._wl_btn)
        self._wl_swatch = QLabel()
        self._wl_swatch.setFixedSize(64, 32)
        wl_lay.addWidget(self._wl_swatch)
        self._wl_info = QLabel()
        wl_lay.addWidget(self._wl_info)
        lay.addWidget(wl_grp)

        # Color temperature
        kt_grp = QGroupBox("Color Temperature")
        kt_lay = QHBoxLayout(kt_grp)
        kt_lay.addWidget(QLabel("Kelvin:"))
        self._kt_spin = QSpinBox()
        self._kt_spin.setRange(1000, 40000)
        self._kt_spin.setValue(6500)
        kt_lay.addWidget(self._kt_spin)
        self._kt_btn = QPushButton("Show")
        self._kt_btn.clicked.connect(self._show_temperature)
        kt_lay.addWidget(self._kt_btn)
        self._kt_swatch = QLabel()
        self._kt_swatch.setFixedSize(64, 32)
        kt_lay.addWidget(self._kt_swatch)
        self._kt_info = QLabel()
        kt_lay.addWidget(self._kt_info)
        lay.addWidget(kt_grp)

        # Spectrum bar
        self._spectrum_fig = Figure(figsize=(6, 1.5), dpi=100)
        self._spectrum_canvas = FigureCanvas(self._spectrum_fig)
        lay.addWidget(self._spectrum_canvas)
        self._draw_spectrum_bar()
        lay.addStretch()
        return w

    def _draw_spectrum_bar(self):
        fig = self._spectrum_fig
        fig.clear()
        ax = fig.add_subplot(111)
        wls = np.arange(380, 781)
        colors = [wavelength_to_rgb(w) for w in wls]
        for i, wl in enumerate(wls):
            ax.axvline(wl, color=colors[i], linewidth=1)
        ax.set_xlim(380, 780)
        ax.set_yticks([])
        ax.set_xlabel("Wavelength (nm)")
        ax.set_title("Visible Spectrum")
        fig.tight_layout()
        self._spectrum_canvas.draw()

    def _show_wavelength(self):
        wl = self._wl_spin.value()
        r, g, b = wavelength_to_rgb(wl)
        self._wl_swatch.setPixmap(_color_swatch_pixmap(r, g, b, 64, 32))
        self._wl_info.setText(f"RGB=({int(r*255)},{int(g*255)},{int(b*255)})")
        self._current_rgb = (r, g, b)
        self._refresh_all()
        self._log(f"Wavelength {wl} nm -> RGB({int(r*255)},{int(g*255)},{int(b*255)})")

    def _show_temperature(self):
        T = self._kt_spin.value()
        r, g, b = kelvin_to_rgb(T)
        self._kt_swatch.setPixmap(_color_swatch_pixmap(r, g, b, 64, 32))
        x, y = planckian_locus_xy(min(T, 25000))
        self._kt_info.setText(f"xy=({x:.4f},{y:.4f})  RGB=({int(r*255)},{int(g*255)},{int(b*255)})")
        self._current_rgb = (r, g, b)
        self._refresh_all()
        self._log(f"Temperature {T} K -> xy({x:.4f},{y:.4f})")

    # -- Color Blindness tab --
    def _build_blindness_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Simulates how the current color appears under different types of color vision deficiency."))
        self._cb_labels = {}
        grid = QGridLayout()
        grid.addWidget(QLabel("Normal Vision"), 0, 0)
        self._cb_labels["Normal"] = QLabel()
        self._cb_labels["Normal"].setFixedSize(80, 80)
        grid.addWidget(self._cb_labels["Normal"], 1, 0)
        self._cb_info_labels = {}
        for i, name in enumerate(["Protanopia", "Deuteranopia", "Tritanopia"]):
            grid.addWidget(QLabel(name), 0, i + 1)
            lbl = QLabel()
            lbl.setFixedSize(80, 80)
            self._cb_labels[name] = lbl
            grid.addWidget(lbl, 1, i + 1)
            info = QLabel()
            self._cb_info_labels[name] = info
            grid.addWidget(info, 2, i + 1)
        lay.addLayout(grid)
        btn = QPushButton("Simulate Current Color")
        btn.clicked.connect(self._simulate_cb)
        lay.addWidget(btn)
        lay.addStretch()
        return w

    def _simulate_cb(self):
        r, g, b = self._current_rgb
        self._cb_labels["Normal"].setPixmap(_color_swatch_pixmap(r, g, b, 80, 80))
        for name in ["Protanopia", "Deuteranopia", "Tritanopia"]:
            sr, sg, sb = simulate_color_blindness(r, g, b, name)
            self._cb_labels[name].setPixmap(_color_swatch_pixmap(sr, sg, sb, 80, 80))
            self._cb_info_labels[name].setText(f"RGB({int(sr*255)},{int(sg*255)},{int(sb*255)})")
        self._log("Color blindness simulation updated")

    # -- Harmony tab --
    def _build_harmony_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Color harmonies derived from the current color (HSV-based)."))
        self._harmony_combo = QComboBox()
        self._harmony_combo.addItems(["Complementary", "Analogous", "Triadic", "Split-Complementary"])
        lay.addWidget(self._harmony_combo)
        btn = QPushButton("Generate Harmony")
        btn.clicked.connect(self._generate_harmony)
        lay.addWidget(btn)
        self._harmony_container = QHBoxLayout()
        self._harmony_swatches = []
        for _ in range(6):
            lbl = QLabel()
            lbl.setFixedSize(64, 64)
            self._harmony_container.addWidget(lbl)
            self._harmony_swatches.append(lbl)
        lay.addLayout(self._harmony_container)
        self._harmony_info = QTextEdit()
        self._harmony_info.setReadOnly(True)
        self._harmony_info.setMaximumHeight(100)
        lay.addWidget(self._harmony_info)
        lay.addStretch()
        return w

    def _generate_harmony(self):
        r, g, b = self._current_rgb
        h, s, v = rgb_to_hsv(r, g, b)
        h_deg = h * 360
        mode = self._harmony_combo.currentText()
        if mode == "Complementary":
            angles = [0, 180]
        elif mode == "Analogous":
            angles = [-30, -15, 0, 15, 30]
        elif mode == "Triadic":
            angles = [0, 120, 240]
        else:  # Split-Complementary
            angles = [0, 150, 210]
        info_lines = [f"Base: H={h_deg:.0f} ({mode})"]
        for i, ang in enumerate(angles):
            nh = ((h_deg + ang) % 360) / 360
            cr, cg, cb = hsv_to_rgb(nh, s, v)
            if i < len(self._harmony_swatches):
                self._harmony_swatches[i].setPixmap(_color_swatch_pixmap(cr, cg, cb, 64, 64))
            info_lines.append(f"  +{ang}: #{int(cr*255):02X}{int(cg*255):02X}{int(cb*255):02X}")
        # Hide unused
        for j in range(len(angles), len(self._harmony_swatches)):
            self._harmony_swatches[j].setPixmap(QPixmap())
        self._harmony_info.setPlainText("\n".join(info_lines))
        self._log(f"Generated {mode} harmony")

    # -- Contrast / Delta-E tab --
    def _build_contrast_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        # Second color picker
        row = QHBoxLayout()
        row.addWidget(QLabel("Color 1: current color | Color 2:"))
        self._c2_hex = QLineEdit("#000000")
        self._c2_hex.setMaximumWidth(100)
        row.addWidget(self._c2_hex)
        self._c2_swatch = QLabel()
        self._c2_swatch.setFixedSize(32, 32)
        self._c2_swatch.setPixmap(_color_swatch_pixmap(0, 0, 0, 32, 32))
        row.addWidget(self._c2_swatch)
        c2_btn = QPushButton("Pick Color 2")
        c2_btn.clicked.connect(self._pick_color2)
        row.addWidget(c2_btn)
        row.addStretch()
        lay.addLayout(row)

        btn = QPushButton("Compute Contrast & Delta-E")
        btn.clicked.connect(self._compute_contrast)
        lay.addWidget(btn)

        self._contrast_info = QTextEdit()
        self._contrast_info.setReadOnly(True)
        self._contrast_info.setMaximumHeight(200)
        lay.addWidget(self._contrast_info)

        # Side-by-side preview
        prev_row = QHBoxLayout()
        self._contrast_preview1 = QLabel("Sample Text")
        self._contrast_preview1.setAlignment(Qt.AlignCenter)
        self._contrast_preview1.setFixedSize(200, 60)
        self._contrast_preview1.setFont(QFont("Arial", 14))
        prev_row.addWidget(self._contrast_preview1)
        self._contrast_preview2 = QLabel("Sample Text")
        self._contrast_preview2.setAlignment(Qt.AlignCenter)
        self._contrast_preview2.setFixedSize(200, 60)
        self._contrast_preview2.setFont(QFont("Arial", 14))
        prev_row.addWidget(self._contrast_preview2)
        lay.addLayout(prev_row)
        lay.addStretch()
        return w

    def _pick_color2(self):
        c = QColorDialog.getColor(QColor(self._c2_hex.text()), self)
        if c.isValid():
            self._second_rgb = (c.redF(), c.greenF(), c.blueF())
            self._c2_hex.setText(c.name().upper())
            self._c2_swatch.setPixmap(_color_swatch_pixmap(*self._second_rgb, 32, 32))

    def _compute_contrast(self):
        # Parse color 2 from hex
        c2 = QColor(self._c2_hex.text().strip())
        if c2.isValid():
            self._second_rgb = (c2.redF(), c2.greenF(), c2.blueF())
            self._c2_swatch.setPixmap(_color_swatch_pixmap(*self._second_rgb, 32, 32))
        r1, g1, b1 = self._current_rgb
        r2, g2, b2 = self._second_rgb
        cr = contrast_ratio((r1, g1, b1), (r2, g2, b2))
        aa_normal = "PASS" if cr >= 4.5 else "FAIL"
        aa_large = "PASS" if cr >= 3.0 else "FAIL"
        aaa_normal = "PASS" if cr >= 7.0 else "FAIL"
        aaa_large = "PASS" if cr >= 4.5 else "FAIL"
        X1, Y1, Z1 = rgb_to_xyz(r1, g1, b1)
        X2, Y2, Z2 = rgb_to_xyz(r2, g2, b2)
        lab1 = xyz_to_lab(X1, Y1, Z1)
        lab2 = xyz_to_lab(X2, Y2, Z2)
        de76 = delta_e_76(lab1, lab2)
        de00 = delta_e_2000(lab1, lab2)
        lines = [
            f"Contrast Ratio: {cr:.2f}:1",
            f"",
            f"WCAG AA  Normal text: {aa_normal} (need 4.5:1)",
            f"WCAG AA  Large text:  {aa_large} (need 3.0:1)",
            f"WCAG AAA Normal text: {aaa_normal} (need 7.0:1)",
            f"WCAG AAA Large text:  {aaa_large} (need 4.5:1)",
            f"",
            f"Color 1 LAB: L={lab1[0]:.2f} a={lab1[1]:.2f} b={lab1[2]:.2f}",
            f"Color 2 LAB: L={lab2[0]:.2f} a={lab2[1]:.2f} b={lab2[2]:.2f}",
            f"Delta-E (CIE76):  {de76:.4f}",
            f"Delta-E (CIE2000): {de00:.4f}",
        ]
        self._contrast_info.setPlainText("\n".join(lines))
        # Preview
        c1_hex = f"#{int(r1*255):02X}{int(g1*255):02X}{int(b1*255):02X}"
        c2_hex = f"#{int(r2*255):02X}{int(g2*255):02X}{int(b2*255):02X}"
        self._contrast_preview1.setStyleSheet(f"background-color:{c2_hex}; color:{c1_hex}; border:1px solid #888;")
        self._contrast_preview2.setStyleSheet(f"background-color:{c1_hex}; color:{c2_hex}; border:1px solid #888;")
        self._log(f"Contrast ratio: {cr:.2f}:1  DE76={de76:.2f}  DE2000={de00:.2f}")

    # -- Palette tab --
    def _build_palette_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Generate a perceptually uniform palette in CIELAB space."))
        row = QHBoxLayout()
        row.addWidget(QLabel("Number of colors:"))
        self._palette_n = QSpinBox()
        self._palette_n.setRange(2, 24)
        self._palette_n.setValue(6)
        row.addWidget(self._palette_n)
        self._palette_mode = QComboBox()
        self._palette_mode.addItems(["Hue Sweep (fixed L,C)", "Lightness Ramp", "Rainbow"])
        row.addWidget(self._palette_mode)
        btn = QPushButton("Generate")
        btn.clicked.connect(self._generate_palette)
        row.addWidget(btn)
        lay.addLayout(row)

        self._palette_container = QHBoxLayout()
        self._palette_swatches = []
        for _ in range(24):
            lbl = QLabel()
            lbl.setFixedSize(40, 40)
            self._palette_container.addWidget(lbl)
            self._palette_swatches.append(lbl)
        lay.addLayout(self._palette_container)

        self._palette_info = QTextEdit()
        self._palette_info.setReadOnly(True)
        self._palette_info.setMaximumHeight(150)
        lay.addWidget(self._palette_info)
        lay.addStretch()
        return w

    def _generate_palette(self):
        n = self._palette_n.value()
        mode = self._palette_mode.currentText()
        colors = []
        if mode == "Hue Sweep (fixed L,C)":
            L, C = 65, 50
            for i in range(n):
                h_rad = 2 * np.pi * i / n
                a = C * np.cos(h_rad)
                b = C * np.sin(h_rad)
                X, Y, Z = lab_to_xyz(L, a, b)
                r, g, bb = xyz_to_rgb(X, Y, Z)
                colors.append((r, g, bb))
        elif mode == "Lightness Ramp":
            r0, g0, b0 = self._current_rgb
            X0, Y0, Z0 = rgb_to_xyz(r0, g0, b0)
            L0, a0, b0_ = xyz_to_lab(X0, Y0, Z0)
            for i in range(n):
                L = 15 + 70 * i / max(n - 1, 1)
                X, Y, Z = lab_to_xyz(L, a0, b0_)
                r, g, bb = xyz_to_rgb(X, Y, Z)
                colors.append((r, g, bb))
        else:  # Rainbow
            for i in range(n):
                wl = 380 + (780 - 380) * i / max(n - 1, 1)
                colors.append(wavelength_to_rgb(wl))

        info_lines = []
        for i, (r, g, b) in enumerate(colors):
            if i < len(self._palette_swatches):
                self._palette_swatches[i].setPixmap(_color_swatch_pixmap(r, g, b, 40, 40))
            info_lines.append(f"  [{i}] #{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}")
        for j in range(n, len(self._palette_swatches)):
            self._palette_swatches[j].setPixmap(QPixmap())
        self._palette_info.setPlainText(f"{mode} ({n} colors):\n" + "\n".join(info_lines))
        self._log(f"Generated {n}-color palette ({mode})")

    # ── Export ────────────────────────────────────────────────────────────

    def export(self):
        """Return a dict summarizing the current color state."""
        r, g, b = self._current_rgb
        X, Y, Z = rgb_to_xyz(r, g, b)
        L, a, bb = xyz_to_lab(X, Y, Z)
        h, s, v = rgb_to_hsv(r, g, b)
        c_, m_, y_, k_ = rgb_to_cmyk(r, g, b)
        return {
            "hex": f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}",
            "rgb": (int(r * 255), int(g * 255), int(b * 255)),
            "hsv": (round(h * 360, 1), round(s * 100, 1), round(v * 100, 1)),
            "hsl": tuple(round(x, 2) for x in rgb_to_hsl(r, g, b)),
            "lab": (round(L, 2), round(a, 2), round(bb, 2)),
            "xyz": (round(X, 5), round(Y, 5), round(Z, 5)),
            "cmyk": (round(c_ * 100, 1), round(m_ * 100, 1), round(y_ * 100, 1), round(k_ * 100, 1)),
        }
