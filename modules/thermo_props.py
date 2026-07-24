"""
Thermodynamic Properties Calculator Widget for QuantumRes.
Steam/water tables, ideal gas properties, refrigerant data, psychrometrics,
Mollier diagrams, equations of state, heat exchangers, and power cycles.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QLabel, QLineEdit, QComboBox, QGroupBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame,
    QDoubleSpinBox, QSpinBox, QScrollArea, QSplitter, QTextEdit
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ── Substance Database ────────────────────────────────────────────────────────
# name: (Tc [K], Pc [Pa], omega, M [kg/mol])
SUBSTANCE_DB = {
    "Water":        (647.096, 22064000, 0.3443, 0.01802),
    "Nitrogen":     (126.2,   3390000,  0.0377, 0.02801),
    "Oxygen":       (154.6,   5043000,  0.0222, 0.03200),
    "Argon":        (150.86,  4898000,  0.0000, 0.03995),
    "CO2":          (304.13,  7377300,  0.2236, 0.04401),
    "Helium":       (5.195,   227600,   -0.382, 0.00400),
    "Hydrogen":     (33.19,   1313000,  -0.217, 0.00202),
    "Methane":      (190.56,  4599000,  0.0115, 0.01604),
    "Ethane":       (305.32,  4872000,  0.0995, 0.03007),
    "Propane":      (369.83,  4248000,  0.1523, 0.04410),
    "n-Butane":     (425.12,  3796000,  0.2002, 0.05812),
    "n-Pentane":    (469.7,   3370000,  0.2515, 0.07215),
    "n-Hexane":     (507.6,   3025000,  0.3013, 0.08618),
    "n-Heptane":    (540.2,   2740000,  0.3495, 0.10021),
    "n-Octane":     (568.7,   2490000,  0.3996, 0.11423),
    "Ammonia":      (405.4,   11333000, 0.2526, 0.01703),
    "Benzene":      (562.05,  4895000,  0.2103, 0.07811),
    "Toluene":      (591.75,  4108000,  0.2640, 0.09214),
    "Acetylene":    (308.3,   6139000,  0.1912, 0.02604),
    "R-134a":       (374.21,  4059000,  0.3268, 0.10203),
    "R-410A":       (344.51,  4926000,  0.2960, 0.07258),
    "Carbon Monoxide": (132.86, 3494000, 0.0510, 0.02801),
    "Sulfur Dioxide":  (430.8, 7884000,  0.2450, 0.06407),
    "Hydrogen Sulfide":(373.4, 8963000,  0.0942, 0.03408),
    "Ethylene":     (282.34,  5041000,  0.0862, 0.02805),
    "Acetone":      (508.1,   4700000,  0.3065, 0.05808),
    "Methanol":     (512.64,  8097000,  0.5656, 0.03204),
    "Ethanol":      (513.92,  6148000,  0.6449, 0.04607),
    "Chlorine":     (416.9,   7977000,  0.0688, 0.07091),
    "Neon":         (44.49,   2679000,  -0.0387,0.02018),
    "Krypton":      (209.48,  5525000,  0.0000, 0.08380),
    "Xenon":        (289.73,  5841000,  0.0036, 0.13129),
}

# ── Ideal Gas Cp/Cv Data (Cp = a + bT + cT^2 + dT^3, J/(mol*K)) ─────────
# NASA 7-coeff simplified to polynomial for 300-1500 K range
IDEAL_GAS_CP = {
    "N2":     (28.90, 1.854e-3, 9.647e-6, -1.637e-8),
    "O2":     (25.48, 1.520e-2, -7.155e-6, 1.312e-9),
    "Ar":     (20.786, 0.0, 0.0, 0.0),
    "CO2":    (22.26, 5.981e-2, -3.501e-5, 7.469e-9),
    "He":     (20.786, 0.0, 0.0, 0.0),
    "H2":     (29.11, -1.916e-3, 4.003e-6, -8.704e-10),
    "Air":    (28.11, 1.967e-3, 4.802e-6, -1.966e-9),
    "CH4":    (19.89, 5.024e-2, 1.269e-5, -1.101e-8),
    "H2O_vap":(32.24, 1.924e-3, 1.055e-5, -3.596e-9),
}

IDEAL_GAS_M = {
    "N2": 0.02801, "O2": 0.03200, "Ar": 0.03995, "CO2": 0.04401,
    "He": 0.00400, "H2": 0.00202, "Air": 0.02897, "CH4": 0.01604,
    "H2O_vap": 0.01802,
}

R_UNIVERSAL = 8.31446  # J/(mol*K)

# ── R-134a Saturation Data (T [C], Psat [kPa], hf, hg [kJ/kg], sf, sg [kJ/(kg*K)]) ──
R134A_SAT = np.array([
    [-40, 51.8,  -7.4, 225.9, -0.032, 0.969],
    [-30, 84.4,   3.5, 232.8,  0.015, 0.955],
    [-20, 132.7, 14.6, 239.5,  0.062, 0.943],
    [-10, 200.7, 26.1, 246.0,  0.107, 0.933],
    [  0, 292.8, 37.9, 252.2,  0.152, 0.924],
    [ 10, 414.9, 50.1, 258.0,  0.197, 0.917],
    [ 20, 572.8, 62.7, 263.4,  0.241, 0.910],
    [ 30, 770.6, 75.8, 268.2,  0.285, 0.904],
    [ 40, 1017., 89.5, 272.3,  0.329, 0.898],
    [ 50, 1318., 104., 275.5,  0.374, 0.892],
    [ 60, 1682., 119., 277.5,  0.419, 0.885],
    [ 70, 2117., 136., 278.0,  0.465, 0.876],
    [ 80, 2633., 154., 276.1,  0.514, 0.864],
])

# ── R-410A Saturation Data (T [C], Psat [kPa], hf, hg, sf, sg) ──────────
R410A_SAT = np.array([
    [-40, 175.0,  -3.2, 274.5, -0.014, 1.092],
    [-30, 269.0,   9.8, 281.2,  0.042, 1.072],
    [-20, 399.6,  23.1, 287.5,  0.097, 1.055],
    [-10, 573.1,  36.9, 293.2,  0.151, 1.040],
    [  0, 798.0,  51.2, 298.2,  0.205, 1.026],
    [ 10, 1085.,  66.2, 302.4,  0.258, 1.013],
    [ 20, 1444.,  82.0, 305.5,  0.312, 1.001],
    [ 30, 1890.,  98.8, 307.2,  0.366, 0.988],
    [ 40, 2435., 117.,  306.8,  0.422, 0.974],
    [ 50, 3098., 137.,  303.5,  0.480, 0.957],
    [ 60, 3903., 160.,  296.0,  0.542, 0.935],
])

# ── Absorption Coefficients for Psychrometrics Constants ──────────────────
PATM = 101325.0  # Pa


def _antoine_water(T_C):
    """Antoine equation for water saturation pressure [Pa]. Valid 1-374 C."""
    T = T_C + 273.15
    if T < 373.15:
        A, B, C = 8.07131, 1730.63, 233.426
    else:
        A, B, C = 8.14019, 1810.94, 244.485
    log10_p_mmhg = A - B / (C + T_C)
    return 10.0 ** log10_p_mmhg * 133.322


def _steam_enthalpy(T_C, P_Pa, quality=None):
    """Simplified steam/water enthalpy [kJ/kg]."""
    Tsat = _saturation_T(P_Pa)
    hf = 4.18 * T_C if T_C <= Tsat else 4.18 * Tsat
    hfg = 2257.0 * (1 - T_C / 647.096) ** 0.38 if T_C < 374 else 0.0
    hg = hf + hfg
    if quality is not None:
        return hf + quality * hfg
    if T_C <= Tsat:
        return hf
    cp_steam = 1.996 + 0.0002 * (T_C - 100)
    return hg + cp_steam * (T_C - Tsat)


def _steam_entropy(T_C, P_Pa, quality=None):
    """Simplified steam/water entropy [kJ/(kg*K)]."""
    T_K = T_C + 273.15
    Tsat = _saturation_T(P_Pa)
    Tsat_K = Tsat + 273.15
    sf = 4.18 * np.log(T_K / 273.15) if T_C <= Tsat else 4.18 * np.log(Tsat_K / 273.15)
    sfg = 2257.0 / Tsat_K * (1 - T_C / 647.096) ** 0.38 if T_C < 374 else 0.0
    sg = sf + sfg
    if quality is not None:
        return sf + quality * sfg
    if T_C <= Tsat:
        return sf
    cp_steam = 1.996
    return sg + cp_steam * np.log(T_K / Tsat_K)


def _specific_volume(T_C, P_Pa, quality=None):
    """Simplified specific volume [m^3/kg]."""
    vf = 0.001 * (1 + 0.0004 * T_C)
    T_K = T_C + 273.15
    vg = R_UNIVERSAL * T_K / (0.01802 * P_Pa) if P_Pa > 0 else 1.0
    if quality is not None:
        return vf + quality * (vg - vf)
    Tsat = _saturation_T(P_Pa)
    return vf if T_C <= Tsat else vg


def _saturation_T(P_Pa):
    """Inverse Antoine: saturation temperature [C] from pressure [Pa]."""
    p_mmhg = P_Pa / 133.322
    if p_mmhg <= 0:
        return 100.0
    A, B, C = 8.07131, 1730.63, 233.426
    log_p = np.log10(p_mmhg)
    return B / (A - log_p) - C


# ── Equations of State ────────────────────────────────────────────────────────

def _van_der_waals_Z(T, P, Tc, Pc):
    """Van der Waals compressibility factor."""
    a = 27 * R_UNIVERSAL**2 * Tc**2 / (64 * Pc)
    b = R_UNIVERSAL * Tc / (8 * Pc)
    # Cubic in V: V^3 - (b + RT/P)V^2 + (a/P)V - ab/P = 0
    A = 1.0
    B = -(b + R_UNIVERSAL * T / P)
    C = a / P
    D = -a * b / P
    roots = np.roots([A, B, C, D])
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-10 and r.real > 0]
    if not real_roots:
        return P * R_UNIVERSAL * T / P  # fallback ideal
    V = max(real_roots)
    return P * V / (R_UNIVERSAL * T)


def _redlich_kwong_Z(T, P, Tc, Pc):
    """Redlich-Kwong compressibility factor."""
    a = 0.42748 * R_UNIVERSAL**2 * Tc**2.5 / Pc
    b = 0.08664 * R_UNIVERSAL * Tc / Pc
    A = a * P / (R_UNIVERSAL**2 * T**2.5)
    B = b * P / (R_UNIVERSAL * T)
    # Z^3 - Z^2 + (A - B - B^2)Z - AB = 0
    coeffs = [1, -1, A - B - B**2, -A * B]
    roots = np.roots(coeffs)
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-10 and r.real > 0]
    return max(real_roots) if real_roots else 1.0


def _peng_robinson_Z(T, P, Tc, Pc, omega):
    """Peng-Robinson compressibility factor."""
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    alpha = (1 + kappa * (1 - np.sqrt(T / Tc)))**2
    a = 0.45724 * R_UNIVERSAL**2 * Tc**2 * alpha / Pc
    b = 0.07780 * R_UNIVERSAL * Tc / Pc
    A = a * P / (R_UNIVERSAL * T)**2
    B = b * P / (R_UNIVERSAL * T)
    # Z^3 - (1-B)Z^2 + (A - 3B^2 - 2B)Z - (AB - B^2 - B^3) = 0
    coeffs = [1, -(1 - B), A - 3*B**2 - 2*B, -(A*B - B**2 - B**3)]
    roots = np.roots(coeffs)
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-10 and r.real > 0]
    return max(real_roots) if real_roots else 1.0


# ── Heat Exchanger Calculations ───────────────────────────────────────────────

def lmtd_calc(Th_in, Th_out, Tc_in, Tc_out, flow="counter"):
    """Log-mean temperature difference."""
    if flow == "counter":
        dT1 = Th_in - Tc_out
        dT2 = Th_out - Tc_in
    else:
        dT1 = Th_in - Tc_in
        dT2 = Th_out - Tc_out
    if abs(dT1 - dT2) < 0.01:
        return (dT1 + dT2) / 2
    if dT1 <= 0 or dT2 <= 0:
        return 0.0
    return (dT1 - dT2) / np.log(dT1 / dT2)


def effectiveness_ntu(NTU, Cr, flow="counter"):
    """Heat exchanger effectiveness from NTU and capacity ratio Cr=Cmin/Cmax."""
    if Cr < 1e-10:
        return 1 - np.exp(-NTU)
    if flow == "counter":
        if abs(Cr - 1.0) < 1e-10:
            return NTU / (1 + NTU)
        exp_term = np.exp(-NTU * (1 - Cr))
        return (1 - exp_term) / (1 - Cr * exp_term)
    elif flow == "parallel":
        return (1 - np.exp(-NTU * (1 + Cr))) / (1 + Cr)
    else:  # cross flow (unmixed)
        return 1 - np.exp((NTU**0.78 / Cr) * (np.exp(-Cr * NTU**0.22) - 1))


# ── Psychrometric Calculations ────────────────────────────────────────────────

def psychrometric_calc(Tdb, RH):
    """Compute psychrometric properties from dry-bulb T [C] and RH [0-100%]."""
    Pws = _antoine_water(Tdb)
    Pw = (RH / 100.0) * Pws
    W = 0.622 * Pw / (PATM - Pw)  # humidity ratio kg/kg
    Tdp_fn = lambda T: _antoine_water(T[0]) - Pw
    Tdp = fsolve(Tdp_fn, [Tdb - 10])[0]
    h = 1.006 * Tdb + W * (2501 + 1.86 * Tdb)  # enthalpy kJ/kg dry air
    v = (R_UNIVERSAL / 0.02897) * (Tdb + 273.15) / PATM * (1 + 1.6078 * W)
    # Wet bulb approximation (Stull formula)
    Twb = (Tdb * np.arctan(0.151977 * np.sqrt(RH + 8.313659))
           + np.arctan(Tdb + RH)
           - np.arctan(RH - 1.676331)
           + 0.00391838 * RH**1.5 * np.arctan(0.023101 * RH)
           - 4.686035)
    return {"Tdb": Tdb, "RH": RH, "Twb": round(Twb, 2), "Tdp": round(Tdp, 2),
            "W": round(W, 5), "h": round(h, 2), "v": round(v, 4)}


# ── Cycle Calculations ────────────────────────────────────────────────────────

def carnot_cycle(Th, Tc):
    """Carnot cycle efficiency and work ratio."""
    eta = 1 - Tc / Th
    return {"eta": eta, "Th": Th, "Tc": Tc, "Qh": 1.0, "Qc": Tc / Th, "W": eta}


def rankine_cycle(T_boiler, T_cond, eta_turbine=0.85, eta_pump=0.80):
    """Simplified Rankine cycle."""
    P_boiler = _antoine_water(T_boiler)
    P_cond = _antoine_water(T_cond)
    h1 = 4.18 * T_cond  # saturated liquid at condenser
    h2 = h1 + 0.001 * (P_boiler - P_cond) / eta_pump / 1000
    h3 = _steam_enthalpy(T_boiler, P_boiler, quality=1.0)
    s3 = _steam_entropy(T_boiler, P_boiler, quality=1.0)
    h4s = _steam_enthalpy(T_cond, P_cond, quality=0.9)
    h4 = h3 - eta_turbine * (h3 - h4s)
    w_turbine = h3 - h4
    w_pump = h2 - h1
    q_in = h3 - h2
    eta = (w_turbine - w_pump) / q_in if q_in > 0 else 0
    return {"eta": eta, "w_turbine": w_turbine, "w_pump": w_pump,
            "q_in": q_in, "h": [h1, h2, h3, h4],
            "T": [T_cond, T_cond, T_boiler, T_cond]}


def brayton_cycle(T1, rp, eta_comp=0.85, eta_turb=0.88, gamma=1.4):
    """Brayton (gas turbine) cycle with compressor/turbine efficiencies."""
    T2s = T1 * rp ** ((gamma - 1) / gamma)
    T2 = T1 + (T2s - T1) / eta_comp
    T3 = T2 * 3.5  # simplified: T3 ~ 3.5 * T2 for typical gas turbine
    T4s = T3 / rp ** ((gamma - 1) / gamma)
    T4 = T3 - eta_turb * (T3 - T4s)
    w_comp = 1.005 * (T2 - T1)
    w_turb = 1.005 * (T3 - T4)
    q_in = 1.005 * (T3 - T2)
    eta = (w_turb - w_comp) / q_in if q_in > 0 else 0
    return {"eta": eta, "T": [T1, T2, T3, T4],
            "w_comp": w_comp, "w_turb": w_turb, "q_in": q_in}


def otto_cycle(T1, P1, r, gamma=1.4, q_in=1800):
    """Otto cycle: T1, P1 [kPa], compression ratio r, heat input kJ/kg."""
    T2 = T1 * r ** (gamma - 1)
    P2 = P1 * r ** gamma
    T3 = T2 + q_in / 0.718
    P3 = P2 * T3 / T2
    T4 = T3 / r ** (gamma - 1)
    P4 = P3 / r ** gamma
    eta = 1 - 1 / r ** (gamma - 1)
    return {"eta": eta, "T": [T1, T2, T3, T4], "P": [P1, P2, P3, P4]}


def diesel_cycle(T1, P1, r, rc, gamma=1.4):
    """Diesel cycle: compression ratio r, cutoff ratio rc."""
    T2 = T1 * r ** (gamma - 1)
    P2 = P1 * r ** gamma
    T3 = T2 * rc
    P3 = P2
    T4 = T3 * (rc / r) ** (gamma - 1)
    P4 = P1
    eta = 1 - (1 / r ** (gamma - 1)) * (rc ** gamma - 1) / (gamma * (rc - 1))
    return {"eta": eta, "T": [T1, T2, T3, T4], "P": [P1, P2, P3, P4]}


# ══════════════════════════════════════════════════════════════════════════════
#  Widget
# ══════════════════════════════════════════════════════════════════════════════

class ThermoPropsWidget(QWidget):
    """Main thermodynamic properties calculator widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._log = lambda msg: None
        self._init_ui()

    def set_logger(self, fn):
        self._log = fn

    # ── UI Setup ──────────────────────────────────────────────────────────────

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_steam_tab(), "Steam Tables")
        self.tabs.addTab(self._build_ideal_gas_tab(), "Ideal Gas")
        self.tabs.addTab(self._build_refrigerant_tab(), "Refrigerants")
        self.tabs.addTab(self._build_psychro_tab(), "Psychrometrics")
        self.tabs.addTab(self._build_mollier_tab(), "Mollier Diagram")
        self.tabs.addTab(self._build_eos_tab(), "EOS / Z-factor")
        self.tabs.addTab(self._build_hx_tab(), "Heat Exchanger")
        self.tabs.addTab(self._build_cycles_tab(), "Power Cycles")
        self.tabs.addTab(self._build_props_table_tab(), "Property Tables")
        layout.addWidget(self.tabs)

    # ── Steam Tables Tab ──────────────────────────────────────────────────────

    def _build_steam_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        grp = QGroupBox("Steam/Water Properties")
        g = QGridLayout(grp)
        g.addWidget(QLabel("Temperature [C]:"), 0, 0)
        self.steam_T = QDoubleSpinBox(); self.steam_T.setRange(-50, 800); self.steam_T.setValue(150)
        g.addWidget(self.steam_T, 0, 1)
        g.addWidget(QLabel("Pressure [kPa]:"), 1, 0)
        self.steam_P = QDoubleSpinBox(); self.steam_P.setRange(0.1, 100000); self.steam_P.setValue(500); self.steam_P.setDecimals(1)
        g.addWidget(self.steam_P, 1, 1)
        g.addWidget(QLabel("Quality (0-1, blank=auto):"), 2, 0)
        self.steam_x = QLineEdit(""); self.steam_x.setPlaceholderText("auto")
        g.addWidget(self.steam_x, 2, 1)
        btn = QPushButton("Calculate"); btn.clicked.connect(self._calc_steam)
        g.addWidget(btn, 3, 0, 1, 2)
        self.steam_out = QTextEdit(); self.steam_out.setReadOnly(True); self.steam_out.setMaximumHeight(180)
        g.addWidget(self.steam_out, 4, 0, 1, 2)
        lay.addWidget(grp)
        lay.addStretch()
        return w

    def _calc_steam(self):
        T = self.steam_T.value()
        P = self.steam_P.value() * 1000  # kPa -> Pa
        x_text = self.steam_x.text().strip()
        quality = float(x_text) if x_text else None
        h = _steam_enthalpy(T, P, quality)
        s = _steam_entropy(T, P, quality)
        v = _specific_volume(T, P, quality)
        Tsat = _saturation_T(P)
        Psat = _antoine_water(T)
        lines = [
            f"T = {T:.2f} C,  P = {P/1000:.2f} kPa",
            f"T_sat(P) = {Tsat:.2f} C,  P_sat(T) = {Psat/1000:.2f} kPa",
            f"Enthalpy  h = {h:.2f} kJ/kg",
            f"Entropy   s = {s:.4f} kJ/(kg*K)",
            f"Sp. Vol   v = {v:.6f} m^3/kg",
        ]
        if quality is not None:
            lines.append(f"Quality   x = {quality:.4f}")
        elif T < Tsat:
            lines.append("Phase: Compressed liquid")
        elif abs(T - Tsat) < 0.5:
            lines.append("Phase: Saturated")
        else:
            lines.append("Phase: Superheated vapor")
        self.steam_out.setPlainText("\n".join(lines))
        self._log(f"Steam calc: T={T} C, P={P/1000} kPa")

    # ── Ideal Gas Tab ─────────────────────────────────────────────────────────

    def _build_ideal_gas_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        grp = QGroupBox("Ideal Gas Properties")
        g = QGridLayout(grp)
        g.addWidget(QLabel("Gas:"), 0, 0)
        self.ig_gas = QComboBox(); self.ig_gas.addItems(list(IDEAL_GAS_CP.keys()))
        g.addWidget(self.ig_gas, 0, 1)
        g.addWidget(QLabel("Temperature [K]:"), 1, 0)
        self.ig_T = QDoubleSpinBox(); self.ig_T.setRange(200, 3000); self.ig_T.setValue(300)
        g.addWidget(self.ig_T, 1, 1)
        btn = QPushButton("Calculate"); btn.clicked.connect(self._calc_ideal_gas)
        g.addWidget(btn, 2, 0, 1, 2)
        self.ig_out = QTextEdit(); self.ig_out.setReadOnly(True); self.ig_out.setMaximumHeight(160)
        g.addWidget(self.ig_out, 3, 0, 1, 2)
        lay.addWidget(grp)
        lay.addStretch()
        return w

    def _calc_ideal_gas(self):
        gas = self.ig_gas.currentText()
        T = self.ig_T.value()
        a, b, c, d = IDEAL_GAS_CP[gas]
        Cp_mol = a + b*T + c*T**2 + d*T**3  # J/(mol*K)
        M = IDEAL_GAS_M[gas]
        Cp = Cp_mol / M / 1000  # kJ/(kg*K)
        Cv = (Cp_mol - R_UNIVERSAL) / M / 1000
        gamma = Cp / Cv if Cv > 0 else 1.0
        c_sound = np.sqrt(gamma * R_UNIVERSAL * T / M)
        lines = [
            f"Gas: {gas} at T = {T:.1f} K",
            f"Cp = {Cp:.4f} kJ/(kg*K)  ({Cp_mol:.2f} J/(mol*K))",
            f"Cv = {Cv:.4f} kJ/(kg*K)",
            f"gamma = Cp/Cv = {gamma:.4f}",
            f"Speed of sound = {c_sound:.2f} m/s",
            f"Molar mass = {M*1000:.2f} g/mol",
        ]
        self.ig_out.setPlainText("\n".join(lines))
        self._log(f"Ideal gas: {gas} at {T} K")

    # ── Refrigerant Tab ───────────────────────────────────────────────────────

    def _build_refrigerant_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        grp = QGroupBox("Refrigerant Saturation Properties")
        g = QGridLayout(grp)
        g.addWidget(QLabel("Refrigerant:"), 0, 0)
        self.ref_sel = QComboBox(); self.ref_sel.addItems(["R-134a", "R-410A"])
        g.addWidget(self.ref_sel, 0, 1)
        self.ref_table = QTableWidget()
        self.ref_table.setColumnCount(6)
        self.ref_table.setHorizontalHeaderLabels(["T [C]", "P [kPa]", "hf [kJ/kg]", "hg [kJ/kg]", "sf", "sg"])
        self.ref_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        g.addWidget(self.ref_table, 1, 0, 1, 2)
        btn = QPushButton("Load Table"); btn.clicked.connect(self._load_refrigerant)
        g.addWidget(btn, 2, 0, 1, 2)
        lay.addWidget(grp)
        return w

    def _load_refrigerant(self):
        name = self.ref_sel.currentText()
        data = R134A_SAT if name == "R-134a" else R410A_SAT
        self.ref_table.setRowCount(len(data))
        for i, row in enumerate(data):
            for j, val in enumerate(row):
                self.ref_table.setItem(i, j, QTableWidgetItem(f"{val:.2f}"))
        self._log(f"Loaded {name} saturation table")

    # ── Psychrometric Tab ─────────────────────────────────────────────────────

    def _build_psychro_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        grp = QGroupBox("Psychrometric Calculator")
        g = QGridLayout(grp)
        g.addWidget(QLabel("Dry Bulb T [C]:"), 0, 0)
        self.psy_T = QDoubleSpinBox(); self.psy_T.setRange(-20, 60); self.psy_T.setValue(25)
        g.addWidget(self.psy_T, 0, 1)
        g.addWidget(QLabel("Relative Humidity [%]:"), 1, 0)
        self.psy_RH = QDoubleSpinBox(); self.psy_RH.setRange(0, 100); self.psy_RH.setValue(50)
        g.addWidget(self.psy_RH, 1, 1)
        btn = QPushButton("Calculate"); btn.clicked.connect(self._calc_psychro)
        g.addWidget(btn, 2, 0, 1, 2)
        self.psy_out = QTextEdit(); self.psy_out.setReadOnly(True); self.psy_out.setMaximumHeight(160)
        g.addWidget(self.psy_out, 3, 0, 1, 2)
        lay.addWidget(grp)
        lay.addStretch()
        return w

    def _calc_psychro(self):
        T = self.psy_T.value()
        RH = self.psy_RH.value()
        res = psychrometric_calc(T, RH)
        lines = [
            f"Dry Bulb T   = {res['Tdb']:.2f} C",
            f"Wet Bulb T   = {res['Twb']:.2f} C",
            f"Dew Point    = {res['Tdp']:.2f} C",
            f"Humidity Ratio = {res['W']:.5f} kg/kg",
            f"Enthalpy     = {res['h']:.2f} kJ/kg dry air",
            f"Specific Vol = {res['v']:.4f} m^3/kg",
            f"RH           = {res['RH']:.1f}%",
        ]
        self.psy_out.setPlainText("\n".join(lines))
        self._log(f"Psychrometric: Tdb={T}, RH={RH}%")

    # ── Mollier Diagram Tab ───────────────────────────────────────────────────

    def _build_mollier_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.mollier_fig = Figure(figsize=(6, 5))
        self.mollier_canvas = FigureCanvas(self.mollier_fig)
        lay.addWidget(self.mollier_canvas)
        btn = QPushButton("Generate Mollier (H-S) Diagram"); btn.clicked.connect(self._draw_mollier)
        lay.addWidget(btn)
        return w

    def _draw_mollier(self):
        ax = self.mollier_fig.clear()
        ax = self.mollier_fig.add_subplot(111)
        pressures = [10, 50, 100, 500, 1000, 5000, 10000, 22064]  # kPa
        for P_kpa in pressures:
            P = P_kpa * 1000
            temps = np.linspace(max(_saturation_T(P), 50), 600, 60)
            h_vals = [_steam_enthalpy(T, P) for T in temps]
            s_vals = [_steam_entropy(T, P) for T in temps]
            ax.plot(s_vals, h_vals, label=f"{P_kpa} kPa")
        # Saturation dome
        T_sat = np.linspace(10, 370, 80)
        hf_vals, sf_vals, hg_vals, sg_vals = [], [], [], []
        for T in T_sat:
            P = _antoine_water(T)
            hf_vals.append(_steam_enthalpy(T, P, 0.0))
            sf_vals.append(_steam_entropy(T, P, 0.0))
            hg_vals.append(_steam_enthalpy(T, P, 1.0))
            sg_vals.append(_steam_entropy(T, P, 1.0))
        ax.plot(sf_vals, hf_vals, 'k-', linewidth=2, label="Sat. liquid")
        ax.plot(sg_vals, hg_vals, 'k--', linewidth=2, label="Sat. vapor")
        ax.set_xlabel("Entropy s [kJ/(kg*K)]")
        ax.set_ylabel("Enthalpy h [kJ/kg]")
        ax.set_title("Mollier (H-S) Diagram for Water/Steam")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)
        self.mollier_canvas.draw()
        self._log("Generated Mollier diagram")

    # ── EOS / Compressibility Tab ─────────────────────────────────────────────

    def _build_eos_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        grp = QGroupBox("Compressibility Factor (Z)")
        g = QGridLayout(grp)
        g.addWidget(QLabel("Substance:"), 0, 0)
        self.eos_sub = QComboBox(); self.eos_sub.addItems(list(SUBSTANCE_DB.keys()))
        g.addWidget(self.eos_sub, 0, 1)
        g.addWidget(QLabel("Temperature [K]:"), 1, 0)
        self.eos_T = QDoubleSpinBox(); self.eos_T.setRange(10, 2000); self.eos_T.setValue(400)
        g.addWidget(self.eos_T, 1, 1)
        g.addWidget(QLabel("Pressure [kPa]:"), 2, 0)
        self.eos_P = QDoubleSpinBox(); self.eos_P.setRange(1, 200000); self.eos_P.setValue(1000); self.eos_P.setDecimals(1)
        g.addWidget(self.eos_P, 2, 1)
        g.addWidget(QLabel("EOS:"), 3, 0)
        self.eos_model = QComboBox()
        self.eos_model.addItems(["Peng-Robinson", "Redlich-Kwong", "van der Waals", "All"])
        g.addWidget(self.eos_model, 3, 1)
        btn = QPushButton("Calculate Z"); btn.clicked.connect(self._calc_eos)
        g.addWidget(btn, 4, 0, 1, 2)
        self.eos_out = QTextEdit(); self.eos_out.setReadOnly(True); self.eos_out.setMaximumHeight(160)
        g.addWidget(self.eos_out, 5, 0, 1, 2)
        lay.addWidget(grp)
        lay.addStretch()
        return w

    def _calc_eos(self):
        sub = self.eos_sub.currentText()
        T = self.eos_T.value()
        P = self.eos_P.value() * 1000  # Pa
        Tc, Pc, omega, M = SUBSTANCE_DB[sub]
        model = self.eos_model.currentText()
        Tr = T / Tc
        Pr = P / Pc
        lines = [f"{sub}: T={T:.1f} K, P={P/1000:.1f} kPa, Tr={Tr:.3f}, Pr={Pr:.3f}", ""]
        if model in ("Peng-Robinson", "All"):
            Z = _peng_robinson_Z(T, P, Tc, Pc, omega)
            lines.append(f"Peng-Robinson:   Z = {Z:.6f}")
        if model in ("Redlich-Kwong", "All"):
            Z = _redlich_kwong_Z(T, P, Tc, Pc)
            lines.append(f"Redlich-Kwong:   Z = {Z:.6f}")
        if model in ("van der Waals", "All"):
            Z = _van_der_waals_Z(T, P, Tc, Pc)
            lines.append(f"van der Waals:   Z = {Z:.6f}")
        lines.append(f"\nIdeal gas:       Z = 1.000000")
        self.eos_out.setPlainText("\n".join(lines))
        self._log(f"EOS calc: {sub}, T={T} K, P={P/1000} kPa")

    # ── Heat Exchanger Tab ────────────────────────────────────────────────────

    def _build_hx_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        # LMTD section
        grp1 = QGroupBox("LMTD Calculator")
        g1 = QGridLayout(grp1)
        for i, (lbl, default) in enumerate([("Th_in [C]:", 150), ("Th_out [C]:", 90),
                                             ("Tc_in [C]:", 30), ("Tc_out [C]:", 70)]):
            g1.addWidget(QLabel(lbl), i, 0)
            sb = QDoubleSpinBox(); sb.setRange(-200, 1000); sb.setValue(default)
            setattr(self, f"hx_lmtd_{i}", sb)
            g1.addWidget(sb, i, 1)
        self.hx_flow = QComboBox(); self.hx_flow.addItems(["counter", "parallel"])
        g1.addWidget(QLabel("Flow:"), 4, 0); g1.addWidget(self.hx_flow, 4, 1)
        btn1 = QPushButton("Calc LMTD"); btn1.clicked.connect(self._calc_lmtd)
        g1.addWidget(btn1, 5, 0, 1, 2)
        self.hx_lmtd_out = QLabel(""); self.hx_lmtd_out.setFont(QFont("Consolas", 10))
        g1.addWidget(self.hx_lmtd_out, 6, 0, 1, 2)
        lay.addWidget(grp1)
        # NTU section
        grp2 = QGroupBox("Effectiveness-NTU")
        g2 = QGridLayout(grp2)
        g2.addWidget(QLabel("NTU:"), 0, 0)
        self.hx_ntu = QDoubleSpinBox(); self.hx_ntu.setRange(0.01, 20); self.hx_ntu.setValue(2.0); self.hx_ntu.setDecimals(2)
        g2.addWidget(self.hx_ntu, 0, 1)
        g2.addWidget(QLabel("Cr (Cmin/Cmax):"), 1, 0)
        self.hx_Cr = QDoubleSpinBox(); self.hx_Cr.setRange(0, 1); self.hx_Cr.setValue(0.5); self.hx_Cr.setDecimals(3)
        g2.addWidget(self.hx_Cr, 1, 1)
        self.hx_ntu_flow = QComboBox(); self.hx_ntu_flow.addItems(["counter", "parallel", "crossflow"])
        g2.addWidget(QLabel("Flow:"), 2, 0); g2.addWidget(self.hx_ntu_flow, 2, 1)
        btn2 = QPushButton("Calc Effectiveness"); btn2.clicked.connect(self._calc_ntu)
        g2.addWidget(btn2, 3, 0, 1, 2)
        self.hx_ntu_out = QLabel(""); self.hx_ntu_out.setFont(QFont("Consolas", 10))
        g2.addWidget(self.hx_ntu_out, 4, 0, 1, 2)
        lay.addWidget(grp2)
        lay.addStretch()
        return w

    def _calc_lmtd(self):
        Th_in = self.hx_lmtd_0.value()
        Th_out = self.hx_lmtd_1.value()
        Tc_in = self.hx_lmtd_2.value()
        Tc_out = self.hx_lmtd_3.value()
        flow = self.hx_flow.currentText()
        val = lmtd_calc(Th_in, Th_out, Tc_in, Tc_out, flow)
        self.hx_lmtd_out.setText(f"LMTD ({flow}) = {val:.3f} C")
        self._log(f"LMTD = {val:.3f} C")

    def _calc_ntu(self):
        NTU = self.hx_ntu.value()
        Cr = self.hx_Cr.value()
        flow = self.hx_ntu_flow.currentText()
        eff = effectiveness_ntu(NTU, Cr, flow)
        self.hx_ntu_out.setText(f"Effectiveness ({flow}) = {eff:.4f}  ({eff*100:.2f}%)")
        self._log(f"NTU effectiveness = {eff:.4f}")

    # ── Power Cycles Tab ──────────────────────────────────────────────────────

    def _build_cycles_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        top = QHBoxLayout()
        top.addWidget(QLabel("Cycle:"))
        self.cycle_sel = QComboBox()
        self.cycle_sel.addItems(["Carnot", "Rankine", "Brayton", "Otto", "Diesel"])
        top.addWidget(self.cycle_sel)
        btn = QPushButton("Calculate & Plot"); btn.clicked.connect(self._calc_cycle)
        top.addWidget(btn)
        lay.addLayout(top)
        # Parameters
        self.cycle_params = QGroupBox("Parameters")
        self.cycle_param_layout = QGridLayout(self.cycle_params)
        self._cycle_inputs = {}
        params = [("Th / T1 [K]:", 600), ("Tc / T_cond [K] or r:", 300),
                  ("Param3 (rp/rc/q_in):", 8), ("Param4 (gamma):", 1.4)]
        for i, (lbl, default) in enumerate(params):
            self.cycle_param_layout.addWidget(QLabel(lbl), i, 0)
            sb = QDoubleSpinBox(); sb.setRange(0.1, 100000); sb.setValue(default); sb.setDecimals(2)
            self._cycle_inputs[i] = sb
            self.cycle_param_layout.addWidget(sb, i, 1)
        lay.addWidget(self.cycle_params)
        self.cycle_out = QTextEdit(); self.cycle_out.setReadOnly(True); self.cycle_out.setMaximumHeight(100)
        lay.addWidget(self.cycle_out)
        # Plots
        h_plot = QHBoxLayout()
        self.cycle_ts_fig = Figure(figsize=(3.5, 3))
        self.cycle_ts_canvas = FigureCanvas(self.cycle_ts_fig)
        h_plot.addWidget(self.cycle_ts_canvas)
        self.cycle_pv_fig = Figure(figsize=(3.5, 3))
        self.cycle_pv_canvas = FigureCanvas(self.cycle_pv_fig)
        h_plot.addWidget(self.cycle_pv_canvas)
        lay.addLayout(h_plot)
        return w

    def _calc_cycle(self):
        name = self.cycle_sel.currentText()
        v = [self._cycle_inputs[i].value() for i in range(4)]
        if name == "Carnot":
            res = carnot_cycle(v[0], v[1])
            T = [v[1], v[1], v[0], v[0], v[1]]
            S = [0, res['Qc'], res['Qc'], 0, 0]
            P_pts = None
        elif name == "Rankine":
            res = rankine_cycle(v[0] - 273.15, v[1] - 273.15)
            T = res['T'] + [res['T'][0]]
            S = [0, 0.1, 7, 5, 0]
            P_pts = None
        elif name == "Brayton":
            res = brayton_cycle(v[0], v[2], gamma=v[3])
            T = res['T'] + [res['T'][0]]
            S = [0, 0, 1, 1, 0]
            P_pts = None
        elif name == "Otto":
            res = otto_cycle(v[0], 101.325, v[2], v[3], v[2] * 100)
            T = res['T'] + [res['T'][0]]
            P_pts = res['P'] + [res['P'][0]]
            S = [0, 0, 1, 1, 0]
        elif name == "Diesel":
            res = diesel_cycle(v[0], 101.325, v[2], v[2] * 0.3, v[3])
            T = res['T'] + [res['T'][0]]
            P_pts = res['P'] + [res['P'][0]]
            S = [0, 0, 0.5, 1, 0]
        else:
            return

        self.cycle_out.setPlainText(f"{name} Cycle\nEfficiency = {res['eta']*100:.2f}%\n"
                                     + "\n".join(f"  {k} = {vv:.2f}" if isinstance(vv, float) else f"  {k} = {vv}"
                                                 for k, vv in res.items() if k != 'eta'))
        # T-S diagram
        ax1 = self.cycle_ts_fig.clear()
        ax1 = self.cycle_ts_fig.add_subplot(111)
        ax1.plot(S, T, 'b-o', markersize=4)
        ax1.fill(S, T, alpha=0.15)
        ax1.set_xlabel("s (relative)"); ax1.set_ylabel("T [K]")
        ax1.set_title(f"{name} T-S Diagram"); ax1.grid(True, alpha=0.3)
        self.cycle_ts_canvas.draw()
        # P-V diagram
        ax2 = self.cycle_pv_fig.clear()
        ax2 = self.cycle_pv_fig.add_subplot(111)
        if P_pts:
            V_pts = [R_UNIVERSAL * T[i] / (P_pts[i] * 1000) if P_pts[i] > 0 else 0 for i in range(len(P_pts))]
            ax2.plot(V_pts, P_pts, 'r-o', markersize=4)
            ax2.fill(V_pts, P_pts, alpha=0.15, color='red')
            ax2.set_ylabel("P [kPa]")
        else:
            ax2.text(0.5, 0.5, "P-V not applicable", ha='center', va='center', transform=ax2.transAxes)
        ax2.set_xlabel("v (relative)"); ax2.set_title(f"{name} P-V Diagram"); ax2.grid(True, alpha=0.3)
        self.cycle_pv_canvas.draw()
        self._log(f"{name} cycle: eta={res['eta']*100:.2f}%")

    # ── Property Tables Tab ───────────────────────────────────────────────────

    def _build_props_table_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        top = QHBoxLayout()
        top.addWidget(QLabel("Search:"))
        self.prop_search = QLineEdit(); self.prop_search.setPlaceholderText("Filter substances...")
        self.prop_search.textChanged.connect(self._filter_props)
        top.addWidget(self.prop_search)
        lay.addLayout(top)
        self.prop_table = QTableWidget()
        self.prop_table.setColumnCount(5)
        self.prop_table.setHorizontalHeaderLabels(["Substance", "Tc [K]", "Pc [kPa]", "omega", "M [g/mol]"])
        self.prop_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._populate_props_table("")
        lay.addWidget(self.prop_table)
        return w

    def _populate_props_table(self, filt):
        filt = filt.lower()
        rows = [(name, *vals) for name, vals in SUBSTANCE_DB.items() if filt in name.lower()]
        self.prop_table.setRowCount(len(rows))
        for i, (name, Tc, Pc, omega, M) in enumerate(rows):
            self.prop_table.setItem(i, 0, QTableWidgetItem(name))
            self.prop_table.setItem(i, 1, QTableWidgetItem(f"{Tc:.2f}"))
            self.prop_table.setItem(i, 2, QTableWidgetItem(f"{Pc/1000:.1f}"))
            self.prop_table.setItem(i, 3, QTableWidgetItem(f"{omega:.4f}"))
            self.prop_table.setItem(i, 4, QTableWidgetItem(f"{M*1000:.2f}"))

    def _filter_props(self, text):
        self._populate_props_table(text)
