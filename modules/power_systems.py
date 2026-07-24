"""
Power Systems Analysis Widget -- Electrical power systems modelling and analysis.

Provides single-line diagram visualisation, Gauss-Seidel / Newton-Raphson load
flow solvers, fault analysis (symmetrical 3-phase), transformer calculations,
per-unit conversion, power triangle, motor calculations, cable sizing, and
basic protection coordination plotting.
"""

import math
import numpy as np
from scipy import linalg
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
    QPushButton, QLabel, QGroupBox, QFormLayout, QLineEdit,
    QDoubleSpinBox, QSpinBox, QTextBrowser, QMessageBox, QFileDialog,
    QCheckBox, QFrame
)
from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPolygonF
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    from mpl_style import style_figure, style_axes
except ImportError:
    def style_figure(f): pass
    def style_axes(a): pass

# ── Preset Systems ──────────────────────────────────────────────────────────

PRESETS = {
    "3-bus": {
        "buses": [
            # voltage, angle, Pg, Qg, Pl, Ql, type
            [1.05, 0.0, 0.0, 0.0, 0.0, 0.0, "Slack"],
            [1.02, 0.0, 0.5, 0.0, 0.0, 0.0, "PV"],
            [1.00, 0.0, 0.0, 0.0, 0.8, 0.3, "PQ"],
        ],
        "lines": [
            # from, to, R, X, B
            [1, 2, 0.02, 0.06, 0.03],
            [1, 3, 0.01, 0.04, 0.02],
            [2, 3, 0.03, 0.08, 0.04],
        ],
    },
    "5-bus": {
        "buses": [
            [1.06, 0.0, 0.0, 0.0, 0.0, 0.0, "Slack"],
            [1.04, 0.0, 0.4, 0.0, 0.2, 0.1, "PV"],
            [1.00, 0.0, 0.0, 0.0, 0.45, 0.15, "PQ"],
            [1.00, 0.0, 0.0, 0.0, 0.40, 0.05, "PQ"],
            [1.00, 0.0, 0.0, 0.0, 0.60, 0.10, "PQ"],
        ],
        "lines": [
            [1, 2, 0.02, 0.06, 0.030],
            [1, 3, 0.08, 0.24, 0.025],
            [2, 3, 0.06, 0.18, 0.020],
            [2, 4, 0.04, 0.12, 0.015],
            [3, 4, 0.01, 0.03, 0.010],
            [4, 5, 0.08, 0.24, 0.025],
        ],
    },
    "IEEE 14-bus": {
        "buses": [
            [1.060, 0.0, 2.324, 0.0, 0.0, 0.0, "Slack"],
            [1.045, 0.0, 0.40, 0.0, 0.217, 0.127, "PV"],
            [1.010, 0.0, 0.0, 0.0, 0.942, 0.190, "PV"],
            [1.000, 0.0, 0.0, 0.0, 0.478, 0.04, "PQ"],
            [1.000, 0.0, 0.0, 0.0, 0.076, 0.016, "PQ"],
            [1.070, 0.0, 0.0, 0.0, 0.112, 0.075, "PV"],
            [1.000, 0.0, 0.0, 0.0, 0.0, 0.0, "PQ"],
            [1.080, 0.0, 0.0, 0.0, 0.0, 0.0, "PV"],
            [1.000, 0.0, 0.0, 0.0, 0.295, 0.166, "PQ"],
            [1.000, 0.0, 0.0, 0.0, 0.090, 0.058, "PQ"],
            [1.000, 0.0, 0.0, 0.0, 0.035, 0.018, "PQ"],
            [1.000, 0.0, 0.0, 0.0, 0.061, 0.016, "PQ"],
            [1.000, 0.0, 0.0, 0.0, 0.135, 0.058, "PQ"],
            [1.000, 0.0, 0.0, 0.0, 0.149, 0.050, "PQ"],
        ],
        "lines": [
            [1, 2, 0.01938, 0.05917, 0.0528],
            [1, 5, 0.05403, 0.22304, 0.0492],
            [2, 3, 0.04699, 0.19797, 0.0438],
            [2, 4, 0.05811, 0.17632, 0.0374],
            [2, 5, 0.05695, 0.17388, 0.0340],
            [3, 4, 0.06701, 0.17103, 0.0346],
            [4, 5, 0.01335, 0.04211, 0.0128],
            [4, 7, 0.0, 0.20912, 0.0],
            [4, 9, 0.0, 0.55618, 0.0],
            [5, 6, 0.0, 0.25202, 0.0],
            [6, 11, 0.09498, 0.19890, 0.0],
            [6, 12, 0.12291, 0.25581, 0.0],
            [6, 13, 0.06615, 0.13027, 0.0],
            [7, 8, 0.0, 0.17615, 0.0],
            [7, 9, 0.11001, 0.20640, 0.0],
            [9, 10, 0.03181, 0.08450, 0.0],
            [9, 14, 0.12711, 0.27038, 0.0],
            [10, 11, 0.08205, 0.19207, 0.0],
            [12, 13, 0.22092, 0.19988, 0.0],
            [13, 14, 0.17093, 0.34802, 0.0],
        ],
    },
}

# ── Single-Line Diagram Canvas ──────────────────────────────────────────────

class SingleLineDiagramCanvas(QFrame):
    """Custom widget that draws a single-line diagram for the power system."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.buses = []
        self.lines = []
        self.setMinimumSize(400, 300)
        self.setFrameShape(QFrame.StyledPanel)
        self._bus_positions = {}

    def set_data(self, buses, lines):
        self.buses = buses
        self.lines = lines
        self._compute_layout()
        self.update()

    def _compute_layout(self):
        """Arrange buses in a circular layout."""
        n = len(self.buses)
        if n == 0:
            self._bus_positions = {}
            return
        cx, cy = self.width() / 2, self.height() / 2
        radius = min(cx, cy) * 0.65
        self._bus_positions = {}
        for i in range(n):
            angle = 2 * math.pi * i / n - math.pi / 2
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            self._bus_positions[i] = (x, y)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._compute_layout()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if not self.buses:
            p.setPen(QPen(QColor(120, 120, 120)))
            p.drawText(self.rect(), Qt.AlignCenter, "No system data.\nLoad a preset or enter bus/line data.")
            p.end()
            return
        self._compute_layout()
        # Draw lines
        pen_line = QPen(QColor(80, 80, 80), 2)
        p.setPen(pen_line)
        for ln in self.lines:
            fb, tb = int(ln[0]) - 1, int(ln[1]) - 1
            if fb in self._bus_positions and tb in self._bus_positions:
                x1, y1 = self._bus_positions[fb]
                x2, y2 = self._bus_positions[tb]
                p.drawLine(int(x1), int(y1), int(x2), int(y2))
        # Draw buses
        bar_w, bar_h = 50, 6
        font = QFont("Consolas", 8)
        p.setFont(font)
        for i, bus in enumerate(self.buses):
            if i not in self._bus_positions:
                continue
            x, y = self._bus_positions[i]
            btype = bus[6] if len(bus) > 6 else "PQ"
            if btype == "Slack":
                colour = QColor(220, 60, 60)
            elif btype == "PV":
                colour = QColor(60, 140, 220)
            else:
                colour = QColor(60, 180, 80)
            p.setPen(QPen(colour, 2))
            p.setBrush(QBrush(colour))
            p.drawRect(int(x - bar_w / 2), int(y - bar_h / 2), bar_w, bar_h)
            # Generator symbol (circle)
            if btype in ("Slack", "PV"):
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(int(x - 8), int(y - 22), 16, 16)
                p.drawText(int(x - 3), int(y - 11), "G")
            # Load symbol (arrow down)
            pl = float(bus[4]) if len(bus) > 4 else 0
            if pl > 0:
                p.setPen(QPen(QColor(180, 120, 40), 2))
                p.drawLine(int(x), int(y + bar_h / 2), int(x), int(y + 25))
                tri = QPolygonF([
                    QPointF(x, y + 30),
                    QPointF(x - 5, y + 22),
                    QPointF(x + 5, y + 22),
                ])
                p.setBrush(QBrush(QColor(180, 120, 40)))
                p.drawPolygon(tri)
            # Label
            p.setPen(QPen(QColor(30, 30, 30)))
            p.drawText(int(x - 20), int(y + 40), f"Bus {i + 1}")
        p.end()


# ── Load Flow Solvers ────────────────────────────────────────────────────────

def build_ybus(n_bus, lines):
    """Build the admittance matrix Y_bus from line data."""
    Y = np.zeros((n_bus, n_bus), dtype=complex)
    for ln in lines:
        fb, tb = int(ln[0]) - 1, int(ln[1]) - 1
        r, x, b = float(ln[2]), float(ln[3]), float(ln[4])
        if abs(x) < 1e-14 and abs(r) < 1e-14:
            continue
        z = complex(r, x)
        y_line = 1.0 / z
        y_shunt = complex(0, b / 2.0)
        Y[fb, fb] += y_line + y_shunt
        Y[tb, tb] += y_line + y_shunt
        Y[fb, tb] -= y_line
        Y[tb, fb] -= y_line
    return Y


def gauss_seidel(buses, lines, max_iter=500, tol=1e-6):
    """Gauss-Seidel power flow solver.  Returns (V, iterations, converged)."""
    n = len(buses)
    Y = build_ybus(n, lines)
    V = np.array([complex(float(b[0]) * math.cos(math.radians(float(b[1]))),
                          float(b[0]) * math.sin(math.radians(float(b[1])))) for b in buses])
    P_spec = np.array([float(b[2]) - float(b[4]) for b in buses])
    Q_spec = np.array([float(b[3]) - float(b[5]) for b in buses])
    bus_types = [b[6] for b in buses]
    V_mag_spec = np.array([float(b[0]) for b in buses])

    for it in range(1, max_iter + 1):
        V_old = V.copy()
        for i in range(n):
            if bus_types[i] == "Slack":
                continue
            sigma = sum(Y[i, j] * V[j] for j in range(n) if j != i)
            if bus_types[i] == "PQ":
                S = complex(P_spec[i], Q_spec[i])
                V[i] = (1.0 / Y[i, i]) * (np.conj(S) / np.conj(V[i]) - sigma)
            elif bus_types[i] == "PV":
                Q_calc = -np.imag(np.conj(V[i]) * (sum(Y[i, j] * V[j] for j in range(n))))
                S = complex(P_spec[i], Q_calc)
                V[i] = (1.0 / Y[i, i]) * (np.conj(S) / np.conj(V[i]) - sigma)
                V[i] = V_mag_spec[i] * V[i] / abs(V[i])
        if np.max(np.abs(V - V_old)) < tol:
            return V, it, True
    return V, max_iter, False


def newton_raphson(buses, lines, max_iter=50, tol=1e-6):
    """Newton-Raphson power flow solver.  Returns (V, iterations, converged)."""
    n = len(buses)
    Y = build_ybus(n, lines)
    V_mag = np.array([float(b[0]) for b in buses], dtype=float)
    V_ang = np.array([math.radians(float(b[1])) for b in buses], dtype=float)
    P_spec = np.array([float(b[2]) - float(b[4]) for b in buses])
    Q_spec = np.array([float(b[3]) - float(b[5]) for b in buses])
    bus_types = [b[6] for b in buses]

    pq_indices = [i for i in range(n) if bus_types[i] == "PQ"]
    pv_indices = [i for i in range(n) if bus_types[i] == "PV"]
    non_slack = [i for i in range(n) if bus_types[i] != "Slack"]

    for it in range(1, max_iter + 1):
        P_calc = np.zeros(n)
        Q_calc = np.zeros(n)
        for i in range(n):
            for j in range(n):
                G = Y[i, j].real
                B = Y[i, j].imag
                P_calc[i] += V_mag[i] * V_mag[j] * (G * math.cos(V_ang[i] - V_ang[j]) +
                                                      B * math.sin(V_ang[i] - V_ang[j]))
                Q_calc[i] += V_mag[i] * V_mag[j] * (G * math.sin(V_ang[i] - V_ang[j]) -
                                                      B * math.cos(V_ang[i] - V_ang[j]))

        dP = P_spec - P_calc
        dQ = Q_spec - Q_calc
        mismatch = np.concatenate([dP[non_slack], dQ[pq_indices]])
        if np.max(np.abs(mismatch)) < tol:
            V = V_mag * np.exp(1j * V_ang)
            return V, it, True

        # Build Jacobian
        n_ns = len(non_slack)
        n_pq = len(pq_indices)
        dim = n_ns + n_pq
        J = np.zeros((dim, dim))
        # J1: dP/d_theta
        for ii, i in enumerate(non_slack):
            for jj, j in enumerate(non_slack):
                G = Y[i, j].real
                B = Y[i, j].imag
                if i == j:
                    J[ii, jj] = -Q_calc[i] - B * V_mag[i] ** 2
                else:
                    J[ii, jj] = V_mag[i] * V_mag[j] * (G * math.sin(V_ang[i] - V_ang[j]) -
                                                         B * math.cos(V_ang[i] - V_ang[j]))
        # J2: dP/dV
        for ii, i in enumerate(non_slack):
            for jj, j in enumerate(pq_indices):
                G = Y[i, j].real
                B = Y[i, j].imag
                if i == j:
                    J[ii, n_ns + jj] = P_calc[i] / V_mag[i] + G * V_mag[i]
                else:
                    J[ii, n_ns + jj] = V_mag[i] * (G * math.cos(V_ang[i] - V_ang[j]) +
                                                     B * math.sin(V_ang[i] - V_ang[j]))
        # J3: dQ/d_theta
        for ii, i in enumerate(pq_indices):
            for jj, j in enumerate(non_slack):
                G = Y[i, j].real
                B = Y[i, j].imag
                if i == j:
                    J[n_ns + ii, jj] = P_calc[i] - G * V_mag[i] ** 2
                else:
                    J[n_ns + ii, jj] = -V_mag[i] * V_mag[j] * (G * math.cos(V_ang[i] - V_ang[j]) +
                                                                  B * math.sin(V_ang[i] - V_ang[j]))
        # J4: dQ/dV
        for ii, i in enumerate(pq_indices):
            for jj, j in enumerate(pq_indices):
                G = Y[i, j].real
                B = Y[i, j].imag
                if i == j:
                    J[n_ns + ii, n_ns + jj] = Q_calc[i] / V_mag[i] - B * V_mag[i]
                else:
                    J[n_ns + ii, n_ns + jj] = V_mag[i] * (G * math.sin(V_ang[i] - V_ang[j]) -
                                                            B * math.cos(V_ang[i] - V_ang[j]))
        try:
            dx = np.linalg.solve(J, mismatch)
        except np.linalg.LinAlgError:
            V = V_mag * np.exp(1j * V_ang)
            return V, it, False
        for ii, i in enumerate(non_slack):
            V_ang[i] += dx[ii]
        for ii, i in enumerate(pq_indices):
            V_mag[i] += dx[n_ns + ii]

    V = V_mag * np.exp(1j * V_ang)
    return V, max_iter, False


# ── Fault Analysis ──────────────────────────────────────────────────────────

def symmetrical_fault(buses, lines, fault_bus):
    """3-phase symmetrical fault analysis using Z-bus method.
    Returns fault current magnitude (p.u.) and post-fault voltages."""
    n = len(buses)
    Y = build_ybus(n, lines)
    try:
        Z = np.linalg.inv(Y)
    except np.linalg.LinAlgError:
        return None, None
    fb = fault_bus - 1
    V_pre = np.array([float(b[0]) for b in buses], dtype=complex)
    I_fault = V_pre[fb] / Z[fb, fb]
    V_post = V_pre - (V_pre[fb] / Z[fb, fb]) * Z[:, fb]
    return abs(I_fault), np.abs(V_post)


# ── Transformer Calculations ────────────────────────────────────────────────

def transformer_calc(v1, v2, s_rated, r_pu, x_pu, pf_load=0.8):
    """Calculate transformer parameters.
    Returns dict with turns_ratio, Z_pu, voltage_reg, efficiency."""
    a = v1 / v2 if v2 != 0 else 0
    z_pu = complex(r_pu, x_pu)
    cos_phi = pf_load
    sin_phi = math.sqrt(1 - cos_phi ** 2)
    vr = (r_pu * cos_phi + x_pu * sin_phi) + 0.5 * (x_pu * cos_phi - r_pu * sin_phi) ** 2
    cu_loss = r_pu * s_rated  # at full load, MW
    p_out = s_rated * cos_phi
    eff = p_out / (p_out + cu_loss) * 100 if (p_out + cu_loss) > 0 else 0
    z_base = v2 ** 2 / s_rated if s_rated > 0 else 0
    z_actual = abs(z_pu) * z_base
    return {
        "turns_ratio": a,
        "Z_pu": z_pu,
        "Z_actual_ohm": z_actual,
        "voltage_regulation_pct": vr * 100,
        "efficiency_pct": eff,
    }


# ── Per-Unit Conversion ─────────────────────────────────────────────────────

def per_unit_convert(actual_value, base_mva, base_kv, quantity="impedance"):
    """Convert between actual and per-unit values."""
    base_v = base_kv * 1e3
    base_i = (base_mva * 1e6) / (math.sqrt(3) * base_v) if base_v > 0 else 0
    base_z = (base_kv ** 2) / base_mva if base_mva > 0 else 0
    if quantity == "impedance":
        return actual_value / base_z if base_z > 0 else 0
    elif quantity == "voltage":
        return actual_value / base_kv if base_kv > 0 else 0
    elif quantity == "current":
        return actual_value / base_i if base_i > 0 else 0
    elif quantity == "power":
        return actual_value / base_mva if base_mva > 0 else 0
    return 0


# ── Power Triangle ──────────────────────────────────────────────────────────

def power_triangle(known_qty, val1, val2):
    """Compute P, Q, S, pf from any two known quantities.
    known_qty is one of: 'P_Q', 'P_S', 'P_pf', 'S_pf', 'Q_S'."""
    if known_qty == "P_Q":
        P, Q = val1, val2
        S = math.sqrt(P ** 2 + Q ** 2)
        pf = P / S if S > 0 else 1.0
    elif known_qty == "P_S":
        P, S = val1, val2
        Q = math.sqrt(max(S ** 2 - P ** 2, 0))
        pf = P / S if S > 0 else 1.0
    elif known_qty == "P_pf":
        P, pf = val1, val2
        S = P / pf if pf > 0 else 0
        Q = math.sqrt(max(S ** 2 - P ** 2, 0))
    elif known_qty == "S_pf":
        S, pf = val1, val2
        P = S * pf
        Q = math.sqrt(max(S ** 2 - P ** 2, 0))
    elif known_qty == "Q_S":
        Q, S = val1, val2
        P = math.sqrt(max(S ** 2 - Q ** 2, 0))
        pf = P / S if S > 0 else 1.0
    else:
        P = Q = S = pf = 0
    return {"P": P, "Q": Q, "S": S, "pf": pf}


# ── Motor Calculations ──────────────────────────────────────────────────────

def induction_motor_calc(rated_hp, voltage, efficiency, pf, poles, freq=60.0, slip=None):
    """Induction motor calculations."""
    rated_w = rated_hp * 746
    p_input = rated_w / (efficiency / 100.0) if efficiency > 0 else 0
    i_line = p_input / (math.sqrt(3) * voltage * (pf / 100.0)) if voltage > 0 and pf > 0 else 0
    n_sync = 120 * freq / poles if poles > 0 else 0
    if slip is None:
        slip = 3.0  # default 3 %
    n_rotor = n_sync * (1 - slip / 100.0)
    torque = (rated_w / (2 * math.pi * n_rotor / 60)) if n_rotor > 0 else 0
    return {
        "rated_kW": rated_w / 1000,
        "input_kW": p_input / 1000,
        "line_current_A": i_line,
        "sync_speed_rpm": n_sync,
        "rotor_speed_rpm": n_rotor,
        "slip_pct": slip,
        "torque_Nm": torque,
    }


# ── Cable Sizing ─────────────────────────────────────────────────────────────

CABLE_DATA = {
    "1.5 mm2 Cu": {"ampacity": 17.5, "r_per_km": 12.10, "x_per_km": 0.115},
    "2.5 mm2 Cu": {"ampacity": 24, "r_per_km": 7.41, "x_per_km": 0.110},
    "4 mm2 Cu":   {"ampacity": 32, "r_per_km": 4.61, "x_per_km": 0.107},
    "6 mm2 Cu":   {"ampacity": 41, "r_per_km": 3.08, "x_per_km": 0.100},
    "10 mm2 Cu":  {"ampacity": 57, "r_per_km": 1.83, "x_per_km": 0.094},
    "16 mm2 Cu":  {"ampacity": 76, "r_per_km": 1.15, "x_per_km": 0.090},
    "25 mm2 Cu":  {"ampacity": 101, "r_per_km": 0.727, "x_per_km": 0.086},
    "35 mm2 Cu":  {"ampacity": 125, "r_per_km": 0.524, "x_per_km": 0.083},
    "50 mm2 Cu":  {"ampacity": 151, "r_per_km": 0.387, "x_per_km": 0.080},
    "70 mm2 Cu":  {"ampacity": 192, "r_per_km": 0.268, "x_per_km": 0.078},
    "95 mm2 Cu":  {"ampacity": 232, "r_per_km": 0.193, "x_per_km": 0.075},
    "120 mm2 Cu": {"ampacity": 269, "r_per_km": 0.153, "x_per_km": 0.073},
}


def cable_sizing(load_kw, voltage, pf, length_m, cable_type, phases=3):
    """Compute current and voltage drop for a given cable and load."""
    cos_phi = pf / 100.0 if pf > 1 else pf
    sin_phi = math.sqrt(1 - cos_phi ** 2)
    if phases == 3:
        current = (load_kw * 1000) / (math.sqrt(3) * voltage * cos_phi) if voltage > 0 and cos_phi > 0 else 0
    else:
        current = (load_kw * 1000) / (voltage * cos_phi) if voltage > 0 and cos_phi > 0 else 0
    cd = CABLE_DATA.get(cable_type, {})
    ampacity = cd.get("ampacity", 0)
    r_km = cd.get("r_per_km", 0)
    x_km = cd.get("x_per_km", 0)
    length_km = length_m / 1000.0
    if phases == 3:
        vdrop = math.sqrt(3) * current * length_km * (r_km * cos_phi + x_km * sin_phi)
    else:
        vdrop = 2 * current * length_km * (r_km * cos_phi + x_km * sin_phi)
    vdrop_pct = (vdrop / voltage * 100) if voltage > 0 else 0
    return {
        "current_A": current,
        "ampacity_A": ampacity,
        "adequate": current <= ampacity,
        "voltage_drop_V": vdrop,
        "voltage_drop_pct": vdrop_pct,
    }


# ── Main Widget ──────────────────────────────────────────────────────────────

class PowerSystemsWidget(QWidget):
    """Complete power-systems analysis widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._log = None
        self._init_ui()

    # -- public API ----------------------------------------------------------

    def set_logger(self, fn):
        self._log = fn

    def run(self):
        """Execute the analysis shown on the current tab."""
        idx = self.tabs.currentIndex()
        tab_name = self.tabs.tabText(idx)
        if "Load Flow" in tab_name or "Diagram" in tab_name:
            self._run_load_flow()
        elif "Fault" in tab_name:
            self._run_fault()
        elif "Transformer" in tab_name:
            self._run_transformer()
        elif "Per-Unit" in tab_name:
            self._run_per_unit()
        elif "Power Triangle" in tab_name:
            self._run_power_triangle()
        elif "Motor" in tab_name:
            self._run_motor()
        elif "Cable" in tab_name:
            self._run_cable()
        elif "Protection" in tab_name:
            self._run_protection()
        else:
            self._run_load_flow()

    def export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Results", "", "Text Files (*.txt);;CSV (*.csv)")
        if not path:
            return
        with open(path, "w") as f:
            f.write(self._results_browser.toPlainText())
        self._log_msg(f"Results exported to {path}")

    # -- UI setup ------------------------------------------------------------

    def _init_ui(self):
        root = QVBoxLayout(self)
        # Top bar
        top = QHBoxLayout()
        top.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(PRESETS.keys()))
        self.preset_combo.setMinimumWidth(120)
        top.addWidget(self.preset_combo)
        btn_load = QPushButton("Load Preset")
        btn_load.clicked.connect(self._load_preset)
        top.addWidget(btn_load)
        top.addStretch()
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Newton-Raphson", "Gauss-Seidel"])
        top.addWidget(QLabel("Solver:"))
        top.addWidget(self.method_combo)
        btn_run = QPushButton("Run Analysis")
        btn_run.clicked.connect(self.run)
        top.addWidget(btn_run)
        btn_exp = QPushButton("Export")
        btn_exp.clicked.connect(self.export)
        top.addWidget(btn_exp)
        root.addLayout(top)

        splitter = QSplitter(Qt.Vertical)

        # Tabs
        self.tabs = QTabWidget()

        # -- Tab 1: Single-line diagram + data tables --
        diag_widget = QWidget()
        dlay = QVBoxLayout(diag_widget)
        self.diagram_canvas = SingleLineDiagramCanvas()
        dlay.addWidget(self.diagram_canvas, 3)

        table_tabs = QTabWidget()
        # Bus table
        self.bus_table = QTableWidget(0, 7)
        self.bus_table.setHorizontalHeaderLabels(
            ["|V| (pu)", "Angle (deg)", "Pgen (pu)", "Qgen (pu)",
             "Pload (pu)", "Qload (pu)", "Type"])
        self.bus_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table_tabs.addTab(self.bus_table, "Bus Data")
        # Line table
        self.line_table = QTableWidget(0, 5)
        self.line_table.setHorizontalHeaderLabels(
            ["From Bus", "To Bus", "R (pu)", "X (pu)", "B (pu)"])
        self.line_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table_tabs.addTab(self.line_table, "Line Data")

        bus_btns = QHBoxLayout()
        btn_add_bus = QPushButton("+ Bus")
        btn_add_bus.clicked.connect(self._add_bus_row)
        btn_del_bus = QPushButton("- Bus")
        btn_del_bus.clicked.connect(lambda: self._del_row(self.bus_table))
        btn_add_line = QPushButton("+ Line")
        btn_add_line.clicked.connect(self._add_line_row)
        btn_del_line = QPushButton("- Line")
        btn_del_line.clicked.connect(lambda: self._del_row(self.line_table))
        bus_btns.addWidget(btn_add_bus)
        bus_btns.addWidget(btn_del_bus)
        bus_btns.addWidget(btn_add_line)
        bus_btns.addWidget(btn_del_line)
        bus_btns.addStretch()
        dlay.addLayout(bus_btns)
        dlay.addWidget(table_tabs, 2)
        self.tabs.addTab(diag_widget, "Diagram / Load Flow")

        # -- Tab 2: Fault Analysis --
        fault_w = QWidget()
        flay = QVBoxLayout(fault_w)
        ff = QHBoxLayout()
        ff.addWidget(QLabel("Fault at Bus:"))
        self.fault_bus_spin = QSpinBox()
        self.fault_bus_spin.setRange(1, 100)
        ff.addWidget(self.fault_bus_spin)
        btn_fault = QPushButton("Run Fault Analysis")
        btn_fault.clicked.connect(self._run_fault)
        ff.addWidget(btn_fault)
        ff.addStretch()
        flay.addLayout(ff)
        self.fault_result = QTextBrowser()
        flay.addWidget(self.fault_result)
        self.tabs.addTab(fault_w, "Fault Analysis")

        # -- Tab 3: Transformer --
        xfmr_w = QWidget()
        xlay = QFormLayout(xfmr_w)
        self.xf_v1 = QDoubleSpinBox(); self.xf_v1.setRange(0, 1e6); self.xf_v1.setValue(132)
        self.xf_v2 = QDoubleSpinBox(); self.xf_v2.setRange(0, 1e6); self.xf_v2.setValue(33)
        self.xf_srated = QDoubleSpinBox(); self.xf_srated.setRange(0, 1e6); self.xf_srated.setValue(100)
        self.xf_rpu = QDoubleSpinBox(); self.xf_rpu.setDecimals(4); self.xf_rpu.setRange(0, 1); self.xf_rpu.setValue(0.01)
        self.xf_xpu = QDoubleSpinBox(); self.xf_xpu.setDecimals(4); self.xf_xpu.setRange(0, 1); self.xf_xpu.setValue(0.05)
        self.xf_pf = QDoubleSpinBox(); self.xf_pf.setRange(0, 1); self.xf_pf.setDecimals(2); self.xf_pf.setValue(0.85)
        xlay.addRow("V1 (kV):", self.xf_v1)
        xlay.addRow("V2 (kV):", self.xf_v2)
        xlay.addRow("S rated (MVA):", self.xf_srated)
        xlay.addRow("R (pu):", self.xf_rpu)
        xlay.addRow("X (pu):", self.xf_xpu)
        xlay.addRow("Load PF:", self.xf_pf)
        btn_xf = QPushButton("Calculate")
        btn_xf.clicked.connect(self._run_transformer)
        xlay.addRow(btn_xf)
        self.xf_result = QTextBrowser()
        xlay.addRow(self.xf_result)
        self.tabs.addTab(xfmr_w, "Transformer")

        # -- Tab 4: Per-Unit --
        pu_w = QWidget()
        pul = QFormLayout(pu_w)
        self.pu_value = QDoubleSpinBox(); self.pu_value.setRange(0, 1e9); self.pu_value.setDecimals(4); self.pu_value.setValue(10)
        self.pu_base_mva = QDoubleSpinBox(); self.pu_base_mva.setRange(0.001, 1e6); self.pu_base_mva.setValue(100)
        self.pu_base_kv = QDoubleSpinBox(); self.pu_base_kv.setRange(0.001, 1e6); self.pu_base_kv.setValue(132)
        self.pu_qty = QComboBox(); self.pu_qty.addItems(["impedance", "voltage", "current", "power"])
        pul.addRow("Actual Value:", self.pu_value)
        pul.addRow("Base MVA:", self.pu_base_mva)
        pul.addRow("Base kV:", self.pu_base_kv)
        pul.addRow("Quantity:", self.pu_qty)
        btn_pu = QPushButton("Convert to Per-Unit")
        btn_pu.clicked.connect(self._run_per_unit)
        pul.addRow(btn_pu)
        self.pu_result = QTextBrowser()
        pul.addRow(self.pu_result)
        self.tabs.addTab(pu_w, "Per-Unit")

        # -- Tab 5: Power Triangle --
        pt_w = QWidget()
        ptl = QVBoxLayout(pt_w)
        ptf = QFormLayout()
        self.pt_known = QComboBox()
        self.pt_known.addItems(["P_Q", "P_S", "P_pf", "S_pf", "Q_S"])
        self.pt_val1 = QDoubleSpinBox(); self.pt_val1.setRange(-1e6, 1e6); self.pt_val1.setDecimals(3); self.pt_val1.setValue(100)
        self.pt_val2 = QDoubleSpinBox(); self.pt_val2.setRange(-1e6, 1e6); self.pt_val2.setDecimals(3); self.pt_val2.setValue(60)
        ptf.addRow("Known pair:", self.pt_known)
        ptf.addRow("Value 1:", self.pt_val1)
        ptf.addRow("Value 2:", self.pt_val2)
        btn_pt = QPushButton("Calculate")
        btn_pt.clicked.connect(self._run_power_triangle)
        ptf.addRow(btn_pt)
        ptl.addLayout(ptf)
        self.pt_canvas = FigureCanvas(Figure(figsize=(4, 3)))
        style_figure(self.pt_canvas.figure)
        ptl.addWidget(self.pt_canvas)
        self.pt_result = QTextBrowser()
        self.pt_result.setMaximumHeight(80)
        ptl.addWidget(self.pt_result)
        self.tabs.addTab(pt_w, "Power Triangle")

        # -- Tab 6: Motor --
        mot_w = QWidget()
        motl = QFormLayout(mot_w)
        self.mot_hp = QDoubleSpinBox(); self.mot_hp.setRange(0, 1e6); self.mot_hp.setValue(50)
        self.mot_v = QDoubleSpinBox(); self.mot_v.setRange(0, 1e6); self.mot_v.setValue(480)
        self.mot_eff = QDoubleSpinBox(); self.mot_eff.setRange(0, 100); self.mot_eff.setValue(92)
        self.mot_pf = QDoubleSpinBox(); self.mot_pf.setRange(0, 100); self.mot_pf.setValue(85)
        self.mot_poles = QSpinBox(); self.mot_poles.setRange(2, 24); self.mot_poles.setValue(4)
        self.mot_freq = QDoubleSpinBox(); self.mot_freq.setRange(1, 400); self.mot_freq.setValue(60)
        self.mot_slip = QDoubleSpinBox(); self.mot_slip.setRange(0, 100); self.mot_slip.setDecimals(2); self.mot_slip.setValue(3.0)
        motl.addRow("Rated HP:", self.mot_hp)
        motl.addRow("Voltage (V):", self.mot_v)
        motl.addRow("Efficiency (%):", self.mot_eff)
        motl.addRow("Power Factor (%):", self.mot_pf)
        motl.addRow("Poles:", self.mot_poles)
        motl.addRow("Frequency (Hz):", self.mot_freq)
        motl.addRow("Slip (%):", self.mot_slip)
        btn_mot = QPushButton("Calculate")
        btn_mot.clicked.connect(self._run_motor)
        motl.addRow(btn_mot)
        self.mot_result = QTextBrowser()
        motl.addRow(self.mot_result)
        self.tabs.addTab(mot_w, "Motor")

        # -- Tab 7: Cable Sizing --
        cab_w = QWidget()
        cabl = QFormLayout(cab_w)
        self.cab_load = QDoubleSpinBox(); self.cab_load.setRange(0, 1e6); self.cab_load.setValue(30)
        self.cab_v = QDoubleSpinBox(); self.cab_v.setRange(0, 1e6); self.cab_v.setValue(400)
        self.cab_pf = QDoubleSpinBox(); self.cab_pf.setRange(0, 1); self.cab_pf.setDecimals(2); self.cab_pf.setValue(0.85)
        self.cab_len = QDoubleSpinBox(); self.cab_len.setRange(0, 1e6); self.cab_len.setValue(100)
        self.cab_type = QComboBox(); self.cab_type.addItems(list(CABLE_DATA.keys()))
        self.cab_phases = QComboBox(); self.cab_phases.addItems(["3-phase", "1-phase"])
        cabl.addRow("Load (kW):", self.cab_load)
        cabl.addRow("Voltage (V):", self.cab_v)
        cabl.addRow("Power Factor:", self.cab_pf)
        cabl.addRow("Length (m):", self.cab_len)
        cabl.addRow("Cable:", self.cab_type)
        cabl.addRow("Phases:", self.cab_phases)
        btn_cab = QPushButton("Calculate")
        btn_cab.clicked.connect(self._run_cable)
        cabl.addRow(btn_cab)
        self.cab_result = QTextBrowser()
        cabl.addRow(self.cab_result)
        self.tabs.addTab(cab_w, "Cable Sizing")

        # -- Tab 8: Protection Coordination --
        prot_w = QWidget()
        protl = QVBoxLayout(prot_w)
        pf2 = QHBoxLayout()
        pf2.addWidget(QLabel("Fuse Rating (A):"))
        self.prot_fuse = QDoubleSpinBox(); self.prot_fuse.setRange(1, 10000); self.prot_fuse.setValue(100)
        pf2.addWidget(self.prot_fuse)
        pf2.addWidget(QLabel("Relay Pickup (A):"))
        self.prot_relay = QDoubleSpinBox(); self.prot_relay.setRange(1, 10000); self.prot_relay.setValue(80)
        pf2.addWidget(self.prot_relay)
        pf2.addWidget(QLabel("TDS:"))
        self.prot_tds = QDoubleSpinBox(); self.prot_tds.setRange(0.01, 15); self.prot_tds.setDecimals(2); self.prot_tds.setValue(2.0)
        pf2.addWidget(self.prot_tds)
        btn_prot = QPushButton("Plot TCC")
        btn_prot.clicked.connect(self._run_protection)
        pf2.addWidget(btn_prot)
        pf2.addStretch()
        protl.addLayout(pf2)
        self.prot_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        style_figure(self.prot_canvas.figure)
        protl.addWidget(self.prot_canvas)
        self.tabs.addTab(prot_w, "Protection")

        splitter.addWidget(self.tabs)

        # Results browser
        self._results_browser = QTextBrowser()
        self._results_browser.setMaximumHeight(180)
        splitter.addWidget(self._results_browser)
        root.addWidget(splitter)

    # -- helpers -------------------------------------------------------------

    def _log_msg(self, msg):
        if self._log:
            self._log(msg)
        self._results_browser.append(msg)

    def _add_bus_row(self):
        r = self.bus_table.rowCount()
        self.bus_table.insertRow(r)
        defaults = ["1.00", "0.0", "0.0", "0.0", "0.0", "0.0", "PQ"]
        for c, v in enumerate(defaults):
            self.bus_table.setItem(r, c, QTableWidgetItem(v))

    def _add_line_row(self):
        r = self.line_table.rowCount()
        self.line_table.insertRow(r)
        defaults = ["1", "2", "0.01", "0.05", "0.02"]
        for c, v in enumerate(defaults):
            self.line_table.setItem(r, c, QTableWidgetItem(v))

    @staticmethod
    def _del_row(table):
        row = table.currentRow()
        if row >= 0:
            table.removeRow(row)

    def _read_tables(self):
        buses, lines = [], []
        for r in range(self.bus_table.rowCount()):
            row = []
            for c in range(7):
                item = self.bus_table.item(r, c)
                val = item.text().strip() if item else "0"
                if c < 6:
                    try:
                        val = float(val)
                    except ValueError:
                        val = 0.0
                row.append(val)
            buses.append(row)
        for r in range(self.line_table.rowCount()):
            row = []
            for c in range(5):
                item = self.line_table.item(r, c)
                val = item.text().strip() if item else "0"
                try:
                    val = float(val)
                except ValueError:
                    val = 0.0
                row.append(val)
            lines.append(row)
        return buses, lines

    def _load_preset(self):
        name = self.preset_combo.currentText()
        data = PRESETS.get(name)
        if not data:
            return
        # Fill bus table
        self.bus_table.setRowCount(0)
        for b in data["buses"]:
            r = self.bus_table.rowCount()
            self.bus_table.insertRow(r)
            for c, v in enumerate(b):
                self.bus_table.setItem(r, c, QTableWidgetItem(str(v)))
        # Fill line table
        self.line_table.setRowCount(0)
        for ln in data["lines"]:
            r = self.line_table.rowCount()
            self.line_table.insertRow(r)
            for c, v in enumerate(ln):
                self.line_table.setItem(r, c, QTableWidgetItem(str(v)))
        # Update diagram
        buses, lines = self._read_tables()
        self.diagram_canvas.set_data(buses, lines)
        self.fault_bus_spin.setMaximum(len(buses))
        self._log_msg(f"Loaded preset: {name} ({len(buses)} buses, {len(lines)} lines)")

    # -- analysis routines ---------------------------------------------------

    def _run_load_flow(self):
        buses, lines = self._read_tables()
        if len(buses) < 2 or len(lines) < 1:
            self._log_msg("Need at least 2 buses and 1 line for load flow.")
            return
        self.diagram_canvas.set_data(buses, lines)
        method = self.method_combo.currentText()
        self._log_msg(f"\n{'='*60}\nLoad Flow Analysis ({method})\n{'='*60}")

        if method == "Gauss-Seidel":
            V, iters, conv = gauss_seidel(buses, lines)
        else:
            V, iters, conv = newton_raphson(buses, lines)

        status = "CONVERGED" if conv else "DID NOT CONVERGE"
        self._log_msg(f"Status: {status} in {iters} iterations\n")

        # Bus results
        self._log_msg(f"{'Bus':>4}  {'|V| (pu)':>10}  {'Angle (deg)':>12}  {'P (pu)':>10}  {'Q (pu)':>10}")
        self._log_msg("-" * 56)
        Y = build_ybus(len(buses), lines)
        for i in range(len(buses)):
            vmag = abs(V[i])
            vang = math.degrees(np.angle(V[i]))
            S_i = V[i] * np.conj(sum(Y[i, j] * V[j] for j in range(len(buses))))
            self._log_msg(f"{i+1:>4}  {vmag:>10.4f}  {vang:>12.4f}  {S_i.real:>10.4f}  {S_i.imag:>10.4f}")

        # Line flows
        self._log_msg(f"\n{'From':>5}  {'To':>5}  {'P_flow (pu)':>12}  {'Q_flow (pu)':>12}  {'P_loss (pu)':>12}")
        self._log_msg("-" * 56)
        total_p_loss = 0
        total_q_loss = 0
        for ln in lines:
            fb, tb = int(ln[0]) - 1, int(ln[1]) - 1
            r, x, b = float(ln[2]), float(ln[3]), float(ln[4])
            if abs(x) < 1e-14 and abs(r) < 1e-14:
                continue
            z = complex(r, x)
            y_line = 1.0 / z
            y_sh = complex(0, b / 2.0)
            I_ft = (V[fb] - V[tb]) * y_line + V[fb] * y_sh
            S_ft = V[fb] * np.conj(I_ft)
            I_tf = (V[tb] - V[fb]) * y_line + V[tb] * y_sh
            S_tf = V[tb] * np.conj(I_tf)
            p_loss = S_ft.real + S_tf.real
            q_loss = S_ft.imag + S_tf.imag
            total_p_loss += p_loss
            total_q_loss += q_loss
            self._log_msg(
                f"{fb+1:>5}  {tb+1:>5}  {S_ft.real:>12.4f}  {S_ft.imag:>12.4f}  {p_loss:>12.6f}")
        self._log_msg(f"\nTotal losses: P = {total_p_loss:.6f} pu, Q = {total_q_loss:.6f} pu")

    def _run_fault(self):
        buses, lines = self._read_tables()
        if len(buses) < 2:
            self.fault_result.setPlainText("Load a system first.")
            return
        fb = self.fault_bus_spin.value()
        if fb > len(buses):
            self.fault_result.setPlainText(f"Bus {fb} does not exist.")
            return
        I_fault, V_post = symmetrical_fault(buses, lines, fb)
        if I_fault is None:
            self.fault_result.setPlainText("Singular Y-bus -- cannot invert.")
            return
        txt = f"3-Phase Symmetrical Fault at Bus {fb}\n{'='*45}\n"
        txt += f"Fault current magnitude: {I_fault:.4f} pu\n\n"
        txt += "Post-fault bus voltages:\n"
        for i, v in enumerate(V_post):
            txt += f"  Bus {i+1}: {v:.4f} pu\n"
        self.fault_result.setPlainText(txt)
        self._log_msg(f"Fault analysis complete -- I_fault = {I_fault:.4f} pu at Bus {fb}")

    def _run_transformer(self):
        res = transformer_calc(
            self.xf_v1.value(), self.xf_v2.value(), self.xf_srated.value(),
            self.xf_rpu.value(), self.xf_xpu.value(), self.xf_pf.value())
        txt = "Transformer Calculations\n" + "=" * 40 + "\n"
        txt += f"Turns ratio (a):       {res['turns_ratio']:.4f}\n"
        txt += f"Z (pu):                {res['Z_pu'].real:.4f} + j{res['Z_pu'].imag:.4f}\n"
        txt += f"Z (actual):            {res['Z_actual_ohm']:.4f} ohm\n"
        txt += f"Voltage regulation:    {res['voltage_regulation_pct']:.2f} %\n"
        txt += f"Efficiency (FL):       {res['efficiency_pct']:.2f} %\n"
        self.xf_result.setPlainText(txt)
        self._log_msg("Transformer calculation complete.")

    def _run_per_unit(self):
        val = self.pu_value.value()
        base_mva = self.pu_base_mva.value()
        base_kv = self.pu_base_kv.value()
        qty = self.pu_qty.currentText()
        pu = per_unit_convert(val, base_mva, base_kv, qty)
        base_z = base_kv ** 2 / base_mva if base_mva > 0 else 0
        base_v = base_kv
        base_i = (base_mva * 1e3) / (math.sqrt(3) * base_kv) if base_kv > 0 else 0
        txt = f"Per-Unit Conversion\n{'='*40}\n"
        txt += f"Base MVA:    {base_mva}\n"
        txt += f"Base kV:     {base_kv}\n"
        txt += f"Base Z:      {base_z:.4f} ohm\n"
        txt += f"Base I:      {base_i:.4f} A\n\n"
        txt += f"Actual {qty}: {val}\n"
        txt += f"Per-unit:    {pu:.6f} pu\n"
        self.pu_result.setPlainText(txt)

    def _run_power_triangle(self):
        known = self.pt_known.currentText()
        v1 = self.pt_val1.value()
        v2 = self.pt_val2.value()
        res = power_triangle(known, v1, v2)
        txt = f"P = {res['P']:.3f},  Q = {res['Q']:.3f},  S = {res['S']:.3f},  pf = {res['pf']:.4f}"
        self.pt_result.setPlainText(txt)
        # Draw triangle
        fig = self.pt_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        style_axes(ax)
        P, Q, S = res["P"], res["Q"], res["S"]
        ax.annotate("", xy=(P, Q), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color="red", lw=2))
        ax.annotate("", xy=(P, 0), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color="blue", lw=2))
        ax.annotate("", xy=(P, Q), xytext=(P, 0),
                     arrowprops=dict(arrowstyle="->", color="green", lw=2))
        ax.text(P / 2, -max(abs(Q) * 0.15, 2), f"P={P:.1f}", ha="center", color="blue", fontsize=9)
        ax.text(P * 1.05, Q / 2, f"Q={Q:.1f}", ha="left", color="green", fontsize=9)
        ax.text(P / 2, Q / 2 + max(abs(Q) * 0.1, 1), f"S={S:.1f}", ha="center", color="red", fontsize=9)
        margin = max(S, P, abs(Q)) * 0.15
        ax.set_xlim(-margin, max(P, S) + margin)
        if Q >= 0:
            ax.set_ylim(-margin, Q + margin)
        else:
            ax.set_ylim(Q - margin, margin)
        ax.set_xlabel("Real Power (P)")
        ax.set_ylabel("Reactive Power (Q)")
        ax.set_title("Power Triangle")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()
        self.pt_canvas.draw()

    def _run_motor(self):
        res = induction_motor_calc(
            self.mot_hp.value(), self.mot_v.value(), self.mot_eff.value(),
            self.mot_pf.value(), self.mot_poles.value(), self.mot_freq.value(),
            self.mot_slip.value())
        txt = "Induction Motor Calculations\n" + "=" * 40 + "\n"
        txt += f"Rated power:     {res['rated_kW']:.2f} kW\n"
        txt += f"Input power:     {res['input_kW']:.2f} kW\n"
        txt += f"Line current:    {res['line_current_A']:.2f} A\n"
        txt += f"Sync speed:      {res['sync_speed_rpm']:.0f} rpm\n"
        txt += f"Rotor speed:     {res['rotor_speed_rpm']:.0f} rpm\n"
        txt += f"Slip:            {res['slip_pct']:.2f} %\n"
        txt += f"Torque:          {res['torque_Nm']:.2f} Nm\n"
        self.mot_result.setPlainText(txt)

    def _run_cable(self):
        phases = 3 if self.cab_phases.currentText() == "3-phase" else 1
        res = cable_sizing(
            self.cab_load.value(), self.cab_v.value(), self.cab_pf.value(),
            self.cab_len.value(), self.cab_type.currentText(), phases)
        status = "ADEQUATE" if res["adequate"] else "UNDERSIZED"
        txt = "Cable Sizing Results\n" + "=" * 40 + "\n"
        txt += f"Load current:    {res['current_A']:.2f} A\n"
        txt += f"Cable ampacity:  {res['ampacity_A']:.0f} A  [{status}]\n"
        txt += f"Voltage drop:    {res['voltage_drop_V']:.2f} V  ({res['voltage_drop_pct']:.2f} %)\n"
        if res["voltage_drop_pct"] > 5:
            txt += "WARNING: Voltage drop exceeds 5 % limit.\n"
        self.cab_result.setPlainText(txt)

    def _run_protection(self):
        """Plot time-current curves for fuse and inverse-time relay."""
        fuse_rating = self.prot_fuse.value()
        relay_pickup = self.prot_relay.value()
        tds = self.prot_tds.value()
        currents = np.linspace(1.1, 30, 500)  # multiples of pickup

        # Fuse: approximate log-log curve  t = k / (I/I_rated)^2
        fuse_k = 10.0
        t_fuse = fuse_k / (currents ** 2)

        # Very Inverse relay (IEC standard):  t = TDS * 13.5 / (I^1 - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_relay = tds * 13.5 / (currents - 1)
            t_relay = np.clip(t_relay, 0.01, 100)

        I_fuse = currents * fuse_rating
        I_relay = currents * relay_pickup

        fig = self.prot_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        style_axes(ax)
        ax.loglog(I_fuse, t_fuse, "r-", linewidth=2, label=f"Fuse {fuse_rating:.0f} A")
        ax.loglog(I_relay, t_relay, "b--", linewidth=2, label=f"Relay {relay_pickup:.0f} A (TDS={tds})")
        ax.set_xlabel("Current (A)")
        ax.set_ylabel("Time (s)")
        ax.set_title("Time-Current Coordination")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlim(min(fuse_rating, relay_pickup), max(I_fuse[-1], I_relay[-1]))
        ax.set_ylim(0.01, 100)
        fig.tight_layout()
        self.prot_canvas.draw()
        self._log_msg("Protection TCC plotted.")
