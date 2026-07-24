"""
Circuit Simulator Widget -- SPICE-like MNA-based circuit analysis engine.

Provides DC operating point, AC frequency sweep (Bode plots), and transient
analysis using Modified Nodal Analysis with backward-Euler integration for
reactive components.
"""

import numpy as np
from scipy import linalg, signal
from collections import defaultdict
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QPlainTextEdit,
    QComboBox, QPushButton, QLabel, QGroupBox, QFormLayout, QLineEdit,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QListWidget, QListWidgetItem, QDoubleSpinBox, QSpinBox, QFileDialog,
    QCheckBox, QTextBrowser
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    from mpl_style import style_figure, style_axes
except ImportError:
    def style_figure(f): pass
    def style_axes(a): pass


def _clean_num(x, tol=1e-10):
    """Clean floating-point noise for display."""
    if isinstance(x, (float,)):
        rounded = round(x)
        if abs(x - rounded) < tol:
            return int(rounded)
        return round(x, 10)
    return x


# ---------------------------------------------------------------------------
# Component data structures
# ---------------------------------------------------------------------------

COMPONENT_TYPES = {
    "R": "Resistor",
    "C": "Capacitor",
    "L": "Inductor",
    "V": "Voltage Source",
    "I": "Current Source",
    "D": "Diode",
    "O": "Op-Amp",
}

PRESET_CIRCUITS = {
    "Voltage Divider": (
        "V1 1 0 10\n"
        "R1 1 2 1000\n"
        "R2 2 0 2000\n"
    ),
    "RC Low-Pass Filter": (
        "V1 1 0 AC 1\n"
        "R1 1 2 1000\n"
        "C1 2 0 1e-6\n"
    ),
    "RLC Resonant": (
        "V1 1 0 AC 1\n"
        "R1 1 2 100\n"
        "L1 2 3 10e-3\n"
        "C1 3 0 1e-6\n"
    ),
    "Inverting Amplifier": (
        "V1 1 0 AC 1\n"
        "R1 1 2 1000\n"
        "R2 2 3 10000\n"
        "O1 0 2 3\n"
    ),
}

# ---------------------------------------------------------------------------
# Parsed component
# ---------------------------------------------------------------------------

class Component:
    """Represents a single circuit element parsed from a netlist line."""

    def __init__(self, ctype, name, n_plus, n_minus, value, extra=None):
        self.ctype = ctype      # R, C, L, V, I, D, O
        self.name = name        # e.g. R1
        self.n_plus = n_plus    # positive node (int)
        self.n_minus = n_minus  # negative node (int)
        self.value = value      # primary value (ohms, farads, henrys, volts, amps)
        self.extra = extra or {}  # ac amplitude, pulse params, etc.

    def __repr__(self):
        return f"<{self.name} {self.ctype} ({self.n_plus},{self.n_minus}) val={self.value}>"


# ---------------------------------------------------------------------------
# Netlist parser
# ---------------------------------------------------------------------------

class NetlistParser:
    """Parse a SPICE-like netlist into Component objects."""

    @staticmethod
    def parse(text: str):
        components = []
        nodes = {0}
        for raw_line in text.strip().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("*") or line.startswith("."):
                continue
            tokens = line.split()
            if len(tokens) < 4:
                continue
            name = tokens[0]
            ctype = name[0].upper()
            n_plus = int(tokens[1])
            n_minus = int(tokens[2])
            nodes.add(n_plus)
            nodes.add(n_minus)
            extra = {}

            if ctype in ("R", "C", "L", "I"):
                value = float(tokens[3])
            elif ctype == "V":
                # V1 1 0 10  or  V1 1 0 AC 1
                if tokens[3].upper() == "AC":
                    value = float(tokens[4]) if len(tokens) > 4 else 1.0
                    extra["ac"] = value
                    extra["dc"] = 0.0
                else:
                    value = float(tokens[3])
                    extra["dc"] = value
                    extra["ac"] = 0.0
            elif ctype == "D":
                value = float(tokens[3]) if len(tokens) > 3 else 1e-14  # Is
                extra["vt"] = float(tokens[4]) if len(tokens) > 4 else 0.026
                extra["n"] = float(tokens[5]) if len(tokens) > 5 else 1.0
            elif ctype == "O":
                # Op-Amp: O1 inp inn out  (value not used)
                value = 0.0
                # reinterpret: n_plus = non-inv input, n_minus = inv input
                # extra node for output
                if len(tokens) > 3:
                    extra["out"] = int(tokens[3])
                    nodes.add(extra["out"])
            else:
                value = float(tokens[3]) if len(tokens) > 3 else 0.0

            components.append(Component(ctype, name, n_plus, n_minus, value, extra))

        return components, sorted(nodes)


# ---------------------------------------------------------------------------
# MNA Solver
# ---------------------------------------------------------------------------

class MNASolver:
    """Modified Nodal Analysis solver for DC, AC and transient analyses."""

    def __init__(self, components, nodes, logger=None):
        self.components = components
        self.nodes = sorted(n for n in nodes if n != 0)  # exclude ground
        self.n = len(self.nodes)
        self.node_map = {nd: i for i, nd in enumerate(self.nodes)}
        self.logger = logger or (lambda m: None)

        # voltage sources and op-amps get extra MNA rows
        self.vsources = [c for c in components if c.ctype == "V"]
        self.opamps = [c for c in components if c.ctype == "O"]
        self.extra = len(self.vsources) + len(self.opamps)
        self.size = self.n + self.extra

    def _idx(self, node):
        if node == 0:
            return -1
        return self.node_map[node]

    def _stamp(self, row, col, matrix, value):
        if row < 0 or col < 0:
            return
        matrix[row, col] += value

    # -- DC operating point --------------------------------------------------

    def solve_dc(self):
        """Solve DC operating point. Returns node voltages dict and branch currents dict."""
        G = np.zeros((self.size, self.size), dtype=float)
        rhs = np.zeros(self.size, dtype=float)
        vs_idx = self.n  # starting row for voltage source currents

        for comp in self.components:
            ip = self._idx(comp.n_plus)
            im = self._idx(comp.n_minus)

            if comp.ctype == "R":
                g = 1.0 / comp.value
                self._stamp(ip, ip, G, g)
                self._stamp(im, im, G, g)
                self._stamp(ip, im, G, -g)
                self._stamp(im, ip, G, -g)

            elif comp.ctype == "V":
                k = vs_idx
                vs_idx += 1
                dc = comp.extra.get("dc", comp.value)
                # stamp voltage source into MNA
                if ip >= 0:
                    G[ip, k] += 1.0
                    G[k, ip] += 1.0
                if im >= 0:
                    G[im, k] -= 1.0
                    G[k, im] -= 1.0
                rhs[k] = dc

            elif comp.ctype == "I":
                if ip >= 0:
                    rhs[ip] -= comp.value
                if im >= 0:
                    rhs[im] += comp.value

            elif comp.ctype in ("C", "L"):
                # DC: capacitor = open, inductor = short
                if comp.ctype == "L":
                    # model inductor as short (very small resistance)
                    g = 1.0 / 1e-9
                    self._stamp(ip, ip, G, g)
                    self._stamp(im, im, G, g)
                    self._stamp(ip, im, G, -g)
                    self._stamp(im, ip, G, -g)
                # capacitor: open circuit, nothing stamped

            elif comp.ctype == "D":
                # linearise diode around 0.6 V forward drop
                Is = comp.value if comp.value > 0 else 1e-14
                vt = comp.extra.get("vt", 0.026)
                n_coeff = comp.extra.get("n", 1.0)
                vd = 0.6
                Id = Is * (np.exp(vd / (n_coeff * vt)) - 1.0)
                gd = Is / (n_coeff * vt) * np.exp(vd / (n_coeff * vt))
                Ieq = Id - gd * vd
                self._stamp(ip, ip, G, gd)
                self._stamp(im, im, G, gd)
                self._stamp(ip, im, G, -gd)
                self._stamp(im, ip, G, -gd)
                if ip >= 0:
                    rhs[ip] -= Ieq
                if im >= 0:
                    rhs[im] += Ieq

            elif comp.ctype == "O":
                # ideal op-amp: V(out) is whatever makes V+ = V-
                out_node = comp.extra.get("out")
                if out_node is None:
                    continue
                k = vs_idx
                vs_idx += 1
                io = self._idx(out_node)
                # current from op-amp output enters node 'out'
                if io >= 0:
                    G[io, k] += 1.0
                # constraint: V(n_plus) - V(n_minus) = 0
                if ip >= 0:
                    G[k, ip] += 1.0
                if im >= 0:
                    G[k, im] -= 1.0

        self.logger(f"[DC] Matrix size {self.size}x{self.size}, rank={np.linalg.matrix_rank(G)}")

        try:
            x = linalg.solve(G, rhs)
        except linalg.LinAlgError:
            self.logger("[DC] Singular matrix -- adding ground tie.")
            G[0, 0] += 1e-12
            x = linalg.solve(G, rhs)

        voltages = {0: 0.0}
        for nd, idx in self.node_map.items():
            voltages[nd] = x[idx]

        currents = {}
        vi = self.n
        for comp in self.vsources:
            currents[comp.name] = x[vi]
            vi += 1
        for comp in self.opamps:
            currents[comp.name] = x[vi]
            vi += 1

        # compute resistor currents
        for comp in self.components:
            if comp.ctype == "R":
                v_across = voltages.get(comp.n_plus, 0) - voltages.get(comp.n_minus, 0)
                currents[comp.name] = v_across / comp.value

        return voltages, currents

    # -- AC frequency sweep ---------------------------------------------------

    def solve_ac(self, f_start=1.0, f_stop=1e6, n_points=200):
        """AC sweep returning frequencies, node voltage magnitudes and phases."""
        freqs = np.logspace(np.log10(f_start), np.log10(f_stop), n_points)
        node_ids = sorted(self.node_map.keys())
        mag = {nd: np.zeros(n_points) for nd in node_ids}
        phase = {nd: np.zeros(n_points) for nd in node_ids}

        for fi, f in enumerate(freqs):
            omega = 2.0 * np.pi * f
            G = np.zeros((self.size, self.size), dtype=complex)
            rhs = np.zeros(self.size, dtype=complex)
            vs_idx = self.n

            for comp in self.components:
                ip = self._idx(comp.n_plus)
                im = self._idx(comp.n_minus)

                if comp.ctype == "R":
                    g = 1.0 / comp.value
                    self._stamp(ip, ip, G, g)
                    self._stamp(im, im, G, g)
                    self._stamp(ip, im, G, -g)
                    self._stamp(im, ip, G, -g)

                elif comp.ctype == "C":
                    yc = 1j * omega * comp.value
                    self._stamp(ip, ip, G, yc)
                    self._stamp(im, im, G, yc)
                    self._stamp(ip, im, G, -yc)
                    self._stamp(im, ip, G, -yc)

                elif comp.ctype == "L":
                    yl = 1.0 / (1j * omega * comp.value) if omega > 0 else 1e9
                    self._stamp(ip, ip, G, yl)
                    self._stamp(im, im, G, yl)
                    self._stamp(ip, im, G, -yl)
                    self._stamp(im, ip, G, -yl)

                elif comp.ctype == "V":
                    k = vs_idx
                    vs_idx += 1
                    ac_amp = comp.extra.get("ac", 0.0)
                    if ip >= 0:
                        G[ip, k] += 1.0
                        G[k, ip] += 1.0
                    if im >= 0:
                        G[im, k] -= 1.0
                        G[k, im] -= 1.0
                    rhs[k] = ac_amp

                elif comp.ctype == "I":
                    if ip >= 0:
                        rhs[ip] -= comp.value
                    if im >= 0:
                        rhs[im] += comp.value

                elif comp.ctype == "D":
                    # small-signal linearised model
                    Is = comp.value if comp.value > 0 else 1e-14
                    vt = comp.extra.get("vt", 0.026)
                    n_c = comp.extra.get("n", 1.0)
                    vd = 0.6
                    gd = Is / (n_c * vt) * np.exp(vd / (n_c * vt))
                    self._stamp(ip, ip, G, gd)
                    self._stamp(im, im, G, gd)
                    self._stamp(ip, im, G, -gd)
                    self._stamp(im, ip, G, -gd)

                elif comp.ctype == "O":
                    out_node = comp.extra.get("out")
                    if out_node is None:
                        continue
                    k = vs_idx
                    vs_idx += 1
                    io = self._idx(out_node)
                    if io >= 0:
                        G[io, k] += 1.0
                    if ip >= 0:
                        G[k, ip] += 1.0
                    if im >= 0:
                        G[k, im] -= 1.0

            try:
                x = linalg.solve(G, rhs)
            except linalg.LinAlgError:
                G[0, 0] += 1e-15
                x = linalg.solve(G, rhs)

            for nd, idx in self.node_map.items():
                mag[nd][fi] = np.abs(x[idx])
                phase[nd][fi] = np.degrees(np.angle(x[idx]))

        self.logger(f"[AC] Sweep {f_start:.1f} Hz - {f_stop:.0f} Hz, {n_points} points complete.")
        return freqs, mag, phase

    # -- Transient analysis (backward Euler) ----------------------------------

    def solve_transient(self, t_stop=1e-3, dt=1e-6, v_func=None):
        """Backward-Euler transient simulation.

        *v_func* maps ``(comp_name, t)`` -> voltage at time *t*.  If None a
        step from 0 to the DC value at t=0 is assumed for all voltage sources.
        """
        n_steps = int(t_stop / dt) + 1
        times = np.linspace(0, t_stop, n_steps)
        node_ids = sorted(self.node_map.keys())
        history = {nd: np.zeros(n_steps) for nd in node_ids}

        # state: voltages across caps, currents through inductors
        cap_v = {c.name: 0.0 for c in self.components if c.ctype == "C"}
        ind_i = {c.name: 0.0 for c in self.components if c.ctype == "L"}

        for ti in range(n_steps):
            t = times[ti]
            G = np.zeros((self.size, self.size), dtype=float)
            rhs = np.zeros(self.size, dtype=float)
            vs_idx = self.n

            for comp in self.components:
                ip = self._idx(comp.n_plus)
                im = self._idx(comp.n_minus)

                if comp.ctype == "R":
                    g = 1.0 / comp.value
                    self._stamp(ip, ip, G, g)
                    self._stamp(im, im, G, g)
                    self._stamp(ip, im, G, -g)
                    self._stamp(im, ip, G, -g)

                elif comp.ctype == "C":
                    # backward Euler: i_C = C/dt * (v_n - v_prev)
                    geq = comp.value / dt
                    ieq = geq * cap_v[comp.name]
                    self._stamp(ip, ip, G, geq)
                    self._stamp(im, im, G, geq)
                    self._stamp(ip, im, G, -geq)
                    self._stamp(im, ip, G, -geq)
                    if ip >= 0:
                        rhs[ip] += ieq
                    if im >= 0:
                        rhs[im] -= ieq

                elif comp.ctype == "L":
                    # backward Euler: v_L = L/dt * (i_n - i_prev)
                    geq = dt / comp.value
                    ieq = ind_i[comp.name]
                    self._stamp(ip, ip, G, geq)
                    self._stamp(im, im, G, geq)
                    self._stamp(ip, im, G, -geq)
                    self._stamp(im, ip, G, -geq)
                    if ip >= 0:
                        rhs[ip] += ieq
                    if im >= 0:
                        rhs[im] -= ieq

                elif comp.ctype == "V":
                    k = vs_idx
                    vs_idx += 1
                    if v_func is not None:
                        vval = v_func(comp.name, t)
                    else:
                        vval = comp.extra.get("dc", comp.value)
                    if ip >= 0:
                        G[ip, k] += 1.0
                        G[k, ip] += 1.0
                    if im >= 0:
                        G[im, k] -= 1.0
                        G[k, im] -= 1.0
                    rhs[k] = vval

                elif comp.ctype == "I":
                    if ip >= 0:
                        rhs[ip] -= comp.value
                    if im >= 0:
                        rhs[im] += comp.value

                elif comp.ctype == "D":
                    Is = comp.value if comp.value > 0 else 1e-14
                    vt = comp.extra.get("vt", 0.026)
                    n_c = comp.extra.get("n", 1.0)
                    vd = 0.6
                    gd = Is / (n_c * vt) * np.exp(vd / (n_c * vt))
                    Ieq = Is * (np.exp(vd / (n_c * vt)) - 1.0) - gd * vd
                    self._stamp(ip, ip, G, gd)
                    self._stamp(im, im, G, gd)
                    self._stamp(ip, im, G, -gd)
                    self._stamp(im, ip, G, -gd)
                    if ip >= 0:
                        rhs[ip] -= Ieq
                    if im >= 0:
                        rhs[im] += Ieq

                elif comp.ctype == "O":
                    out_node = comp.extra.get("out")
                    if out_node is None:
                        continue
                    k = vs_idx
                    vs_idx += 1
                    io = self._idx(out_node)
                    if io >= 0:
                        G[io, k] += 1.0
                    if ip >= 0:
                        G[k, ip] += 1.0
                    if im >= 0:
                        G[k, im] -= 1.0

            try:
                x = linalg.solve(G, rhs)
            except linalg.LinAlgError:
                G[0, 0] += 1e-12
                x = linalg.solve(G, rhs)

            # record node voltages
            for nd, idx in self.node_map.items():
                history[nd][ti] = np.real(x[idx])

            # update companion-model states
            for comp in self.components:
                ip = self._idx(comp.n_plus)
                im = self._idx(comp.n_minus)
                vp = x[ip] if ip >= 0 else 0.0
                vm = x[im] if im >= 0 else 0.0
                v_across = np.real(vp - vm)
                if comp.ctype == "C":
                    cap_v[comp.name] = v_across
                elif comp.ctype == "L":
                    ind_i[comp.name] += (dt / comp.value) * v_across

        self.logger(f"[TRAN] {n_steps} steps, dt={dt:.2e} s, t_stop={t_stop:.2e} s")
        return times, history

    # -- Impedance analysis ---------------------------------------------------

    def compute_impedance(self, node_a, node_b, f_start=1.0, f_stop=1e6, n_points=200):
        """Compute impedance between two nodes over frequency range.

        Injects a 1A test current between node_a and node_b and measures
        the resulting voltage difference.  Returns (freqs, Z_mag, Z_phase).
        """
        freqs = np.logspace(np.log10(f_start), np.log10(f_stop), n_points)
        Z_mag = np.zeros(n_points)
        Z_phase = np.zeros(n_points)

        for fi, f in enumerate(freqs):
            omega = 2.0 * np.pi * f
            G = np.zeros((self.size, self.size), dtype=complex)
            rhs = np.zeros(self.size, dtype=complex)
            vs_idx = self.n

            for comp in self.components:
                ip = self._idx(comp.n_plus)
                im = self._idx(comp.n_minus)
                if comp.ctype == "R":
                    g = 1.0 / comp.value
                    self._stamp(ip, ip, G, g); self._stamp(im, im, G, g)
                    self._stamp(ip, im, G, -g); self._stamp(im, ip, G, -g)
                elif comp.ctype == "C":
                    yc = 1j * omega * comp.value
                    self._stamp(ip, ip, G, yc); self._stamp(im, im, G, yc)
                    self._stamp(ip, im, G, -yc); self._stamp(im, ip, G, -yc)
                elif comp.ctype == "L":
                    yl = 1.0 / (1j * omega * comp.value) if omega > 0 else 1e9
                    self._stamp(ip, ip, G, yl); self._stamp(im, im, G, yl)
                    self._stamp(ip, im, G, -yl); self._stamp(im, ip, G, -yl)
                elif comp.ctype == "V":
                    k = vs_idx; vs_idx += 1
                    if ip >= 0: G[ip, k] += 1.0; G[k, ip] += 1.0
                    if im >= 0: G[im, k] -= 1.0; G[k, im] -= 1.0
                    rhs[k] = 0.0  # zero AC sources for impedance measurement
                elif comp.ctype == "O":
                    out_node = comp.extra.get("out")
                    if out_node is None: continue
                    k = vs_idx; vs_idx += 1
                    io = self._idx(out_node)
                    if io >= 0: G[io, k] += 1.0
                    if ip >= 0: G[k, ip] += 1.0
                    if im >= 0: G[k, im] -= 1.0

            # Inject 1A test current: +1A into node_a, -1A out of node_b
            ia = self._idx(node_a)
            ib = self._idx(node_b)
            if ia >= 0: rhs[ia] += 1.0
            if ib >= 0: rhs[ib] -= 1.0

            try:
                x = linalg.solve(G, rhs)
            except linalg.LinAlgError:
                G[0, 0] += 1e-15
                x = linalg.solve(G, rhs)

            va = x[ia] if ia >= 0 else 0.0
            vb = x[ib] if ib >= 0 else 0.0
            Z = va - vb  # V/I with I=1A
            Z_mag[fi] = np.abs(Z)
            Z_phase[fi] = np.degrees(np.angle(Z))

        self.logger(f"[Z] Impedance between nodes {node_a}-{node_b} computed.")
        return freqs, Z_mag, Z_phase

    # -- Power analysis -------------------------------------------------------

    def compute_power(self, voltages, currents):
        """Compute power dissipation per component from DC solution.

        Returns list of (component_name, power_watts) tuples.
        """
        power_list = []
        for comp in self.components:
            vp = voltages.get(comp.n_plus, 0.0)
            vm = voltages.get(comp.n_minus, 0.0)
            v_across = vp - vm

            if comp.ctype == "R":
                p = v_across ** 2 / comp.value
                power_list.append((comp.name, p))
            elif comp.ctype == "V":
                i = currents.get(comp.name, 0.0)
                p = abs(v_across * i)
                power_list.append((comp.name, p))
            elif comp.ctype == "I":
                p = abs(v_across * comp.value)
                power_list.append((comp.name, p))
            elif comp.ctype == "D":
                Is = comp.value if comp.value > 0 else 1e-14
                vt = comp.extra.get("vt", 0.026)
                n_c = comp.extra.get("n", 1.0)
                i_d = Is * (np.exp(v_across / (n_c * vt)) - 1.0) if v_across > 0 else 0
                p = abs(v_across * i_d)
                power_list.append((comp.name, p))
            elif comp.ctype in ("C", "L"):
                power_list.append((comp.name, 0.0))  # ideal reactive: zero DC power
            elif comp.ctype == "O":
                i = currents.get(comp.name, 0.0)
                out = comp.extra.get("out")
                v_out = voltages.get(out, 0.0) if out is not None else 0.0
                p = abs(v_out * i)
                power_list.append((comp.name, p))

        self.logger(f"[PWR] Total power: {sum(p for _, p in power_list):.6g} W")
        return power_list


# ---------------------------------------------------------------------------
# Filter Design Wizard
# ---------------------------------------------------------------------------

class FilterDesignWizard:
    """Generate component values for standard analog filter topologies."""

    @staticmethod
    def butterworth_coefficients(order):
        """Return normalized Butterworth polynomial denominator coefficients."""
        b, a = signal.butter(order, 1.0, analog=True, output='ba')
        return b, a

    @staticmethod
    def chebyshev_coefficients(order, ripple_db=1.0):
        """Return normalized Chebyshev Type I polynomial coefficients."""
        b, a = signal.cheby1(order, ripple_db, 1.0, analog=True, output='ba')
        return b, a

    @staticmethod
    def design_filter(filter_type="butterworth", topology="low-pass",
                      order=2, cutoff_hz=1000.0, impedance=1000.0,
                      ripple_db=1.0):
        """Design a passive ladder filter.

        Returns a dict with component names and values, plus a netlist string.
        Supported topologies: low-pass, high-pass, band-pass.
        """
        omega_c = 2.0 * np.pi * cutoff_hz
        R = impedance

        if filter_type == "butterworth":
            z, p, k = signal.buttap(order)
        else:
            z, p, k = signal.cheb1ap(order, ripple_db)

        # Prototype low-pass element values (Cauer synthesis, normalized)
        # Use scipy to get second-order sections for accuracy
        b_proto, a_proto = signal.zpk2tf(z, p, k)

        # For a simple ladder: use tabulated g-values
        g_values = FilterDesignWizard._ladder_g_values(order, filter_type, ripple_db)

        components = {}
        netlist_lines = [f"* {filter_type.title()} {topology} filter, order {order}",
                         f"* Cutoff: {cutoff_hz} Hz, Z0: {impedance} ohm",
                         f"V1 1 0 AC 1"]
        node = 1

        if topology == "low-pass":
            for i, g in enumerate(g_values):
                node_next = node + 1
                if i % 2 == 0:  # series inductor
                    L = g * R / omega_c
                    name = f"L{i + 1}"
                    components[name] = L
                    netlist_lines.append(f"{name} {node} {node_next} {L:.6e}")
                else:  # shunt capacitor
                    C = g / (R * omega_c)
                    name = f"C{i + 1}"
                    components[name] = C
                    netlist_lines.append(f"{name} {node_next} 0 {C:.6e}")
                node = node_next
            # Termination resistor
            components["R_load"] = R
            netlist_lines.append(f"R_load {node} 0 {R:.6e}")

        elif topology == "high-pass":
            for i, g in enumerate(g_values):
                node_next = node + 1
                if i % 2 == 0:  # series capacitor
                    C = 1.0 / (g * R * omega_c)
                    name = f"C{i + 1}"
                    components[name] = C
                    netlist_lines.append(f"{name} {node} {node_next} {C:.6e}")
                else:  # shunt inductor
                    L = R / (g * omega_c)
                    name = f"L{i + 1}"
                    components[name] = L
                    netlist_lines.append(f"{name} {node_next} 0 {L:.6e}")
                node = node_next
            components["R_load"] = R
            netlist_lines.append(f"R_load {node} 0 {R:.6e}")

        elif topology == "band-pass":
            # LP-to-BP transformation: L -> series LC, C -> parallel LC
            bw = cutoff_hz * 0.5  # default bandwidth = 50% of center
            omega_0 = omega_c
            for i, g in enumerate(g_values):
                node_next = node + 1
                if i % 2 == 0:
                    L_s = g * R / bw / (2 * np.pi)
                    C_s = 1.0 / (omega_0 ** 2 * L_s)
                    nm_l = f"L{i + 1}"
                    nm_c = f"C{i + 1}"
                    components[nm_l] = L_s
                    components[nm_c] = C_s
                    mid = node * 100 + i
                    netlist_lines.append(f"{nm_l} {node} {mid} {L_s:.6e}")
                    netlist_lines.append(f"{nm_c} {mid} {node_next} {C_s:.6e}")
                else:
                    C_p = g / (R * bw * 2 * np.pi)
                    L_p = 1.0 / (omega_0 ** 2 * C_p)
                    nm_l = f"L{i + 1}"
                    nm_c = f"C{i + 1}"
                    components[nm_l] = L_p
                    components[nm_c] = C_p
                    netlist_lines.append(f"{nm_c} {node_next} 0 {C_p:.6e}")
                    netlist_lines.append(f"{nm_l} {node_next} 0 {L_p:.6e}")
                node = node_next
            components["R_load"] = R
            netlist_lines.append(f"R_load {node} 0 {R:.6e}")

        netlist = "\n".join(netlist_lines)
        return components, netlist

    @staticmethod
    def _ladder_g_values(order, filter_type, ripple_db=1.0):
        """Compute normalized ladder element g-values."""
        if filter_type == "butterworth":
            return [2.0 * np.sin((2 * k - 1) * np.pi / (2 * order))
                    for k in range(1, order + 1)]
        else:  # Chebyshev
            eps = np.sqrt(10 ** (ripple_db / 10.0) - 1.0)
            gamma = np.arcsinh(1.0 / eps) / order
            g = []
            for k in range(1, order + 1):
                a_k = np.sin((2 * k - 1) * np.pi / (2 * order))
                b_k = np.sinh(gamma) ** 2 + np.sin(k * np.pi / order) ** 2
                if k == 1:
                    g.append(2.0 * a_k / np.sinh(gamma))
                else:
                    g.append(4.0 * a_k * a_prev / (b_prev * g[-1]))
                a_prev = a_k
                b_prev = b_k
            return g

    @staticmethod
    def transfer_function_symbolic(components, nodes):
        """Build symbolic transfer function H(s) string from RLC circuit.

        Returns a human-readable string of H(s) for simple RLC topologies.
        """
        # Identify topology from component types
        resistors = [c for c in components if c.ctype == "R"]
        capacitors = [c for c in components if c.ctype == "C"]
        inductors = [c for c in components if c.ctype == "L"]

        # Simple RC low-pass: V_out/V_in = 1/(1 + sRC)
        if len(resistors) == 1 and len(capacitors) == 1 and len(inductors) == 0:
            R = resistors[0].value
            C = capacitors[0].value
            tau = R * C
            fc = 1.0 / (2 * np.pi * tau)
            return (f"H(s) = 1 / (1 + s*R*C)\n"
                    f"     = 1 / (1 + s*{tau:.6e})\n"
                    f"Time constant tau = {tau:.6e} s\n"
                    f"Cutoff freq fc = {fc:.2f} Hz")

        # RLC series: H(s) = 1/(s^2*LC + s*RC + 1) for output across C
        if len(resistors) == 1 and len(capacitors) == 1 and len(inductors) == 1:
            R = resistors[0].value
            L_val = inductors[0].value
            C = capacitors[0].value
            omega0 = 1.0 / np.sqrt(L_val * C)
            Q = 1.0 / (R * np.sqrt(C / L_val))
            f0 = omega0 / (2 * np.pi)
            return (f"H(s) = 1 / (s^2*LC + s*RC + 1)\n"
                    f"L={L_val:.6e} H, C={C:.6e} F, R={R:.6e} ohm\n"
                    f"Resonant freq f0 = {f0:.2f} Hz\n"
                    f"Quality factor Q = {Q:.4f}\n"
                    f"omega_0 = {omega0:.2f} rad/s")

        # Voltage divider
        if len(resistors) == 2 and len(capacitors) == 0 and len(inductors) == 0:
            R1, R2 = resistors[0].value, resistors[1].value
            return (f"H(s) = R2 / (R1 + R2)  [frequency independent]\n"
                    f"R1={R1:.2f} ohm, R2={R2:.2f} ohm\n"
                    f"H = {R2 / (R1 + R2):.6f}")

        # General case: use scipy transfer function from AC data
        return "H(s): complex topology -- use AC sweep for numerical Bode plot"


# ---------------------------------------------------------------------------
# SPICE Netlist Exporter
# ---------------------------------------------------------------------------

class SPICEExporter:
    """Export circuit netlist to standard SPICE .cir format."""

    @staticmethod
    def export(netlist_text, title="QuantumRes Circuit", analyses=None):
        """Convert internal netlist to SPICE .cir format string.

        Parameters
        ----------
        netlist_text : str
            The raw netlist from the editor.
        title : str
            Title line for the SPICE file.
        analyses : list of str or None
            Analysis commands to append (e.g., ['.ac dec 100 1 1meg']).
        """
        lines = [f"* {title}", f"* Exported from QuantumRes Circuit Simulator", ""]
        for raw in netlist_text.strip().splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("*") or line.startswith("."):
                lines.append(line)
                continue
            tokens = line.split()
            name = tokens[0]
            ctype = name[0].upper()
            if ctype == "O":
                # Convert op-amp to SPICE subcircuit call
                lines.append(f"* Op-Amp {name}: non-inv={tokens[1]}, inv={tokens[2]}, out={tokens[3]}")
                lines.append(f"E{name[1:]} {tokens[3]} 0 {tokens[1]} {tokens[2]} 1e6")
            else:
                lines.append(line)

        lines.append("")
        if analyses:
            for a in analyses:
                lines.append(a)
        else:
            lines.append(".ac dec 100 1 1meg")
            lines.append(".tran 1u 1m")
        lines.append(".end")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# GUI Widget
# ---------------------------------------------------------------------------

class CircuitSimWidget(QWidget):
    """PyQt5 widget providing a full SPICE-like circuit simulator."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._logger = None
        self._init_ui()

    # -- public API -----------------------------------------------------------

    def set_logger(self, fn):
        """Set an external logging callback ``fn(message)``."""
        self._logger = fn

    def run(self):
        """Programmatic entry point -- execute the currently selected analysis."""
        self._run_analysis()

    # -- UI construction ------------------------------------------------------

    def _init_ui(self):
        layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Horizontal)

        # ---- Left panel: netlist + controls ----
        left = QWidget()
        left_layout = QVBoxLayout(left)

        # Component library
        lib_group = QGroupBox("Component Library")
        lib_layout = QVBoxLayout(lib_group)
        self.comp_list = QListWidget()
        for key, label in COMPONENT_TYPES.items():
            item = QListWidgetItem(f"{key} - {label}")
            item.setData(Qt.UserRole, key)
            self.comp_list.addItem(item)
        self.comp_list.setMaximumHeight(160)
        lib_layout.addWidget(self.comp_list)
        self.btn_insert = QPushButton("Insert Component Template")
        self.btn_insert.clicked.connect(self._insert_template)
        lib_layout.addWidget(self.btn_insert)
        left_layout.addWidget(lib_group)

        # Preset circuits
        preset_group = QGroupBox("Preset Circuits")
        preset_layout = QVBoxLayout(preset_group)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(PRESET_CIRCUITS.keys())
        preset_layout.addWidget(self.preset_combo)
        btn_load = QPushButton("Load Preset")
        btn_load.clicked.connect(self._load_preset)
        preset_layout.addWidget(btn_load)
        left_layout.addWidget(preset_group)

        # Netlist editor
        net_group = QGroupBox("Netlist Editor")
        net_layout = QVBoxLayout(net_group)
        self.netlist_edit = QPlainTextEdit()
        self.netlist_edit.setPlaceholderText(
            "Enter SPICE-like netlist, e.g.:\n"
            "V1 1 0 10\n"
            "R1 1 2 1000\n"
            "R2 2 0 2000"
        )
        net_layout.addWidget(self.netlist_edit)
        left_layout.addWidget(net_group)

        # Analysis controls
        ctrl_group = QGroupBox("Analysis")
        ctrl_form = QFormLayout(ctrl_group)
        self.analysis_combo = QComboBox()
        self.analysis_combo.addItems(["DC Operating Point", "AC Sweep", "Transient"])
        ctrl_form.addRow("Type:", self.analysis_combo)

        self.ac_fstart = QDoubleSpinBox()
        self.ac_fstart.setRange(0.01, 1e12)
        self.ac_fstart.setValue(1.0)
        self.ac_fstart.setDecimals(2)
        ctrl_form.addRow("AC f_start (Hz):", self.ac_fstart)

        self.ac_fstop = QDoubleSpinBox()
        self.ac_fstop.setRange(1, 1e12)
        self.ac_fstop.setValue(1e6)
        self.ac_fstop.setDecimals(0)
        ctrl_form.addRow("AC f_stop (Hz):", self.ac_fstop)

        self.ac_npts = QSpinBox()
        self.ac_npts.setRange(10, 5000)
        self.ac_npts.setValue(200)
        ctrl_form.addRow("AC points:", self.ac_npts)

        self.tran_tstop = QDoubleSpinBox()
        self.tran_tstop.setRange(1e-12, 100)
        self.tran_tstop.setValue(1e-3)
        self.tran_tstop.setDecimals(9)
        ctrl_form.addRow("Tran t_stop (s):", self.tran_tstop)

        self.tran_dt = QDoubleSpinBox()
        self.tran_dt.setRange(1e-15, 1)
        self.tran_dt.setValue(1e-6)
        self.tran_dt.setDecimals(12)
        ctrl_form.addRow("Tran dt (s):", self.tran_dt)

        left_layout.addWidget(ctrl_group)

        # --- Component Sweep controls ---
        sweep_group = QGroupBox("Component Value Sweep")
        sweep_form = QFormLayout(sweep_group)
        self.sweep_comp = QLineEdit()
        self.sweep_comp.setPlaceholderText("Component name, e.g. R1")
        sweep_form.addRow("Component:", self.sweep_comp)
        self.sweep_start = QDoubleSpinBox()
        self.sweep_start.setRange(1e-15, 1e15)
        self.sweep_start.setValue(100)
        self.sweep_start.setDecimals(4)
        sweep_form.addRow("Start value:", self.sweep_start)
        self.sweep_stop = QDoubleSpinBox()
        self.sweep_stop.setRange(1e-15, 1e15)
        self.sweep_stop.setValue(10000)
        self.sweep_stop.setDecimals(4)
        sweep_form.addRow("Stop value:", self.sweep_stop)
        self.sweep_steps = QSpinBox()
        self.sweep_steps.setRange(2, 20)
        self.sweep_steps.setValue(5)
        sweep_form.addRow("Steps:", self.sweep_steps)
        btn_sweep = QPushButton("Run Sweep")
        btn_sweep.clicked.connect(self._run_sweep)
        sweep_form.addRow(btn_sweep)
        left_layout.addWidget(sweep_group)

        # --- Filter Design Wizard ---
        filt_group = QGroupBox("Filter Design Wizard")
        filt_form = QFormLayout(filt_group)
        self.filt_type_combo = QComboBox()
        self.filt_type_combo.addItems(["butterworth", "chebyshev"])
        filt_form.addRow("Type:", self.filt_type_combo)
        self.filt_topo_combo = QComboBox()
        self.filt_topo_combo.addItems(["low-pass", "high-pass", "band-pass"])
        filt_form.addRow("Topology:", self.filt_topo_combo)
        self.filt_order = QSpinBox()
        self.filt_order.setRange(1, 10)
        self.filt_order.setValue(3)
        filt_form.addRow("Order:", self.filt_order)
        self.filt_cutoff = QDoubleSpinBox()
        self.filt_cutoff.setRange(0.1, 1e9)
        self.filt_cutoff.setValue(1000.0)
        self.filt_cutoff.setSuffix(" Hz")
        filt_form.addRow("Cutoff:", self.filt_cutoff)
        self.filt_impedance = QDoubleSpinBox()
        self.filt_impedance.setRange(1, 1e6)
        self.filt_impedance.setValue(1000.0)
        self.filt_impedance.setSuffix(" ohm")
        filt_form.addRow("Impedance:", self.filt_impedance)
        self.filt_ripple = QDoubleSpinBox()
        self.filt_ripple.setRange(0.01, 10.0)
        self.filt_ripple.setValue(1.0)
        self.filt_ripple.setSuffix(" dB")
        filt_form.addRow("Ripple (Cheby):", self.filt_ripple)
        btn_design = QPushButton("Design Filter")
        btn_design.clicked.connect(self._design_filter)
        filt_form.addRow(btn_design)
        left_layout.addWidget(filt_group)

        # --- Impedance Analyzer ---
        imp_group = QGroupBox("Impedance Analyzer")
        imp_form = QFormLayout(imp_group)
        self.imp_node_a = QSpinBox()
        self.imp_node_a.setRange(0, 999)
        self.imp_node_a.setValue(1)
        imp_form.addRow("Node A:", self.imp_node_a)
        self.imp_node_b = QSpinBox()
        self.imp_node_b.setRange(0, 999)
        self.imp_node_b.setValue(0)
        imp_form.addRow("Node B:", self.imp_node_b)
        btn_imp = QPushButton("Analyze Impedance")
        btn_imp.clicked.connect(self._run_impedance)
        imp_form.addRow(btn_imp)
        left_layout.addWidget(imp_group)

        # --- Export / Utility buttons ---
        util_group = QGroupBox("Tools")
        util_layout = QVBoxLayout(util_group)
        btn_spice = QPushButton("Export SPICE Netlist (.cir)")
        btn_spice.clicked.connect(self._export_spice)
        util_layout.addWidget(btn_spice)
        btn_tf = QPushButton("Show Transfer Function H(s)")
        btn_tf.clicked.connect(self._show_transfer_function)
        util_layout.addWidget(btn_tf)
        btn_power = QPushButton("Power Analysis (DC)")
        btn_power.clicked.connect(self._run_power_analysis)
        util_layout.addWidget(btn_power)
        left_layout.addWidget(util_group)

        btn_run = QPushButton("Run Analysis")
        btn_run.setStyleSheet("font-weight:bold; padding:8px;")
        btn_run.clicked.connect(self._run_analysis)
        left_layout.addWidget(btn_run)

        # ---- Right panel: results ----
        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.result_tabs = QTabWidget()

        # Plot tab
        self.figure = Figure(figsize=(7, 5))
        style_figure(self.figure)
        self.canvas = FigureCanvas(self.figure)
        self.result_tabs.addTab(self.canvas, "Plot")

        # Table tab
        self.result_table = QTableWidget()
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_tabs.addTab(self.result_table, "Results Table")

        # Transfer Function / Info tab
        self.tf_display = QTextBrowser()
        self.tf_display.setOpenExternalLinks(False)
        self.result_tabs.addTab(self.tf_display, "Transfer Function")

        # Log tab
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.result_tabs.addTab(self.log_edit, "Log")

        right_layout.addWidget(self.result_tabs)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter)

    # -- helpers --------------------------------------------------------------

    def _log(self, msg):
        self.log_edit.appendPlainText(msg)
        if self._logger:
            self._logger(msg)

    def _insert_template(self):
        item = self.comp_list.currentItem()
        if item is None:
            return
        ctype = item.data(Qt.UserRole)
        templates = {
            "R": "R1 1 2 1000",
            "C": "C1 1 0 1e-6",
            "L": "L1 1 2 10e-3",
            "V": "V1 1 0 10",
            "I": "I1 1 0 0.001",
            "D": "D1 1 2 1e-14",
            "O": "O1 inp inn out",
        }
        self.netlist_edit.appendPlainText(templates.get(ctype, ""))

    def _load_preset(self):
        name = self.preset_combo.currentText()
        netlist = PRESET_CIRCUITS.get(name, "")
        self.netlist_edit.setPlainText(netlist)
        self._log(f"Loaded preset: {name}")

    # -- analysis dispatch ----------------------------------------------------

    def _run_analysis(self):
        text = self.netlist_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Error", "Netlist is empty.")
            return

        try:
            components, nodes = NetlistParser.parse(text)
        except Exception as exc:
            QMessageBox.critical(self, "Parse Error", str(exc))
            return

        if not components:
            QMessageBox.warning(self, "Error", "No valid components parsed.")
            return

        self._log(f"Parsed {len(components)} components, nodes: {nodes}")
        solver = MNASolver(components, nodes, logger=self._log)

        analysis = self.analysis_combo.currentText()
        try:
            if analysis == "DC Operating Point":
                self._run_dc(solver)
            elif analysis == "AC Sweep":
                self._run_ac(solver)
            elif analysis == "Transient":
                self._run_transient(solver)
        except Exception as exc:
            QMessageBox.critical(self, "Solver Error", str(exc))
            self._log(f"ERROR: {exc}")

    # -- DC -------------------------------------------------------------------

    def _run_dc(self, solver: MNASolver):
        voltages, currents = solver.solve_dc()

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        nodes_sorted = sorted(voltages.keys())
        vals = [voltages[n] for n in nodes_sorted]
        colours = ["#2196F3" if n != 0 else "#9E9E9E" for n in nodes_sorted]
        ax.bar([str(n) for n in nodes_sorted], vals, color=colours)
        ax.set_xlabel("Node")
        ax.set_ylabel("Voltage (V)")
        ax.set_title("DC Operating Point")
        ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()

        # populate table
        rows = [(f"Node {n}", f"{_clean_num(float(voltages[n])):.6g} V") for n in nodes_sorted]
        rows += [(f"I({name})", f"{_clean_num(float(val)):.6g} A") for name, val in currents.items()]
        self._fill_table(["Quantity", "Value"], rows)
        self.result_tabs.setCurrentIndex(0)
        self._log("[DC] Analysis complete.")

    # -- AC -------------------------------------------------------------------

    def _run_ac(self, solver: MNASolver):
        f_start = self.ac_fstart.value()
        f_stop = self.ac_fstop.value()
        n_pts = self.ac_npts.value()
        freqs, mag, phase = solver.solve_ac(f_start, f_stop, n_pts)

        self.figure.clear()

        ax_mag = self.figure.add_subplot(211)
        ax_ph = self.figure.add_subplot(212, sharex=ax_mag)

        for nd in sorted(mag.keys()):
            mag_db = 20.0 * np.log10(np.maximum(mag[nd], 1e-30))
            ax_mag.semilogx(freqs, mag_db, label=f"Node {nd}")
            ax_ph.semilogx(freqs, phase[nd], label=f"Node {nd}")

        ax_mag.set_ylabel("Magnitude (dB)")
        ax_mag.set_title("Bode Plot")
        ax_mag.legend(fontsize=7)
        ax_mag.grid(True, which="both", alpha=0.3)

        ax_ph.set_xlabel("Frequency (Hz)")
        ax_ph.set_ylabel("Phase (deg)")
        ax_ph.legend(fontsize=7)
        ax_ph.grid(True, which="both", alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()

        # table: -3 dB frequency estimate for each node
        rows = []
        for nd in sorted(mag.keys()):
            mag_db = 20.0 * np.log10(np.maximum(mag[nd], 1e-30))
            max_db = np.max(mag_db)
            below = np.where(mag_db < max_db - 3.0)[0]
            f3db = freqs[below[0]] if len(below) > 0 else float("nan")
            rows.append((f"Node {nd}", f"{max_db:.2f} dB", f"{f3db:.2f} Hz"))
        self._fill_table(["Node", "Peak Mag", "-3 dB Freq"], rows)
        self.result_tabs.setCurrentIndex(0)
        self._log("[AC] Sweep complete.")

    # -- Transient ------------------------------------------------------------

    def _run_transient(self, solver: MNASolver):
        t_stop = self.tran_tstop.value()
        dt = self.tran_dt.value()
        times, history = solver.solve_transient(t_stop, dt)

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        for nd in sorted(history.keys()):
            ax.plot(times * 1e3, history[nd], label=f"Node {nd}")

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Voltage (V)")
        ax.set_title("Transient Analysis")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()

        # summary table
        rows = []
        for nd in sorted(history.keys()):
            v = history[nd]
            rows.append((
                f"Node {nd}",
                f"{_clean_num(float(np.min(v))):.6g} V",
                f"{_clean_num(float(np.max(v))):.6g} V",
                f"{_clean_num(float(np.mean(v))):.6g} V",
            ))
        self._fill_table(["Node", "Min", "Max", "Mean"], rows)
        self.result_tabs.setCurrentIndex(0)
        self._log("[TRAN] Analysis complete.")

    # -- Component Value Sweep ------------------------------------------------

    def _run_sweep(self):
        """Vary a single component value and plot family of AC Bode curves."""
        comp_name = self.sweep_comp.text().strip()
        if not comp_name:
            QMessageBox.warning(self, "Sweep", "Enter a component name to sweep.")
            return
        text = self.netlist_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Sweep", "Netlist is empty.")
            return

        start = self.sweep_start.value()
        stop = self.sweep_stop.value()
        n_steps = self.sweep_steps.value()
        values = np.logspace(np.log10(start), np.log10(stop), n_steps)

        f_start = self.ac_fstart.value()
        f_stop = self.ac_fstop.value()
        n_pts = self.ac_npts.value()

        self.figure.clear()
        ax_mag = self.figure.add_subplot(211)
        ax_ph = self.figure.add_subplot(212, sharex=ax_mag)

        for val in values:
            # Replace component value in netlist
            modified = []
            for line in text.splitlines():
                tokens = line.strip().split()
                if tokens and tokens[0].upper() == comp_name.upper():
                    # Replace the value token
                    ctype = tokens[0][0].upper()
                    if ctype in ("R", "C", "L", "I"):
                        tokens[3] = f"{val:.6e}"
                    modified.append(" ".join(tokens))
                else:
                    modified.append(line)
            try:
                components, nodes = NetlistParser.parse("\n".join(modified))
                solver = MNASolver(components, nodes, logger=self._log)
                freqs, mag, phase = solver.solve_ac(f_start, f_stop, n_pts)
                # Plot highest non-ground node
                nd = max(mag.keys())
                mag_db = 20.0 * np.log10(np.maximum(mag[nd], 1e-30))
                label = f"{comp_name}={val:.4g}"
                ax_mag.semilogx(freqs, mag_db, label=label)
                ax_ph.semilogx(freqs, phase[nd], label=label)
            except Exception as exc:
                self._log(f"[SWEEP] Error at {comp_name}={val}: {exc}")

        ax_mag.set_ylabel("Magnitude (dB)")
        ax_mag.set_title(f"Component Sweep: {comp_name}")
        ax_mag.legend(fontsize=6)
        ax_mag.grid(True, which="both", alpha=0.3)
        ax_ph.set_xlabel("Frequency (Hz)")
        ax_ph.set_ylabel("Phase (deg)")
        ax_ph.legend(fontsize=6)
        ax_ph.grid(True, which="both", alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()
        self.result_tabs.setCurrentIndex(0)
        self._log(f"[SWEEP] {comp_name} swept {start} to {stop} in {n_steps} steps.")

    # -- Filter Design Wizard -------------------------------------------------

    def _design_filter(self):
        """Use FilterDesignWizard to generate filter netlist and load it."""
        ftype = self.filt_type_combo.currentText()
        topo = self.filt_topo_combo.currentText()
        order = self.filt_order.value()
        cutoff = self.filt_cutoff.value()
        Z0 = self.filt_impedance.value()
        ripple = self.filt_ripple.value()

        try:
            components, netlist = FilterDesignWizard.design_filter(
                filter_type=ftype, topology=topo, order=order,
                cutoff_hz=cutoff, impedance=Z0, ripple_db=ripple
            )
        except Exception as exc:
            QMessageBox.critical(self, "Filter Design", str(exc))
            return

        self.netlist_edit.setPlainText(netlist)
        self._log(f"[FILTER] Designed {ftype} {topo} order {order}, fc={cutoff} Hz")

        # Show component values in table
        rows = [(name, f"{val:.6e}") for name, val in components.items()]
        self._fill_table(["Component", "Value"], rows)
        self.result_tabs.setCurrentIndex(1)

        # Also show in transfer function tab
        info_lines = [f"Filter Design: {ftype.title()} {topo}",
                      f"Order: {order}, Cutoff: {cutoff} Hz, Z0: {Z0} ohm"]
        if ftype == "chebyshev":
            info_lines.append(f"Passband ripple: {ripple} dB")
        info_lines.append("")
        for name, val in components.items():
            unit = "H" if name.startswith("L") else "F" if name.startswith("C") else "ohm"
            info_lines.append(f"  {name} = {val:.6e} {unit}")
        self.tf_display.setPlainText("\n".join(info_lines))

    # -- Impedance Analyzer ---------------------------------------------------

    def _run_impedance(self):
        """Plot impedance magnitude and phase between two nodes."""
        text = self.netlist_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Impedance", "Netlist is empty.")
            return
        try:
            components, nodes = NetlistParser.parse(text)
        except Exception as exc:
            QMessageBox.critical(self, "Parse Error", str(exc))
            return

        node_a = self.imp_node_a.value()
        node_b = self.imp_node_b.value()
        f_start = self.ac_fstart.value()
        f_stop = self.ac_fstop.value()
        n_pts = self.ac_npts.value()

        try:
            solver = MNASolver(components, nodes, logger=self._log)
            freqs, Z_mag, Z_phase = solver.compute_impedance(
                node_a, node_b, f_start, f_stop, n_pts)
        except Exception as exc:
            QMessageBox.critical(self, "Impedance Error", str(exc))
            return

        self.figure.clear()
        ax_mag = self.figure.add_subplot(211)
        ax_ph = self.figure.add_subplot(212, sharex=ax_mag)

        ax_mag.loglog(freqs, Z_mag, "b-", linewidth=1.5)
        ax_mag.set_ylabel("|Z| (ohm)")
        ax_mag.set_title(f"Impedance: Node {node_a} to Node {node_b}")
        ax_mag.grid(True, which="both", alpha=0.3)

        ax_ph.semilogx(freqs, Z_phase, "r-", linewidth=1.5)
        ax_ph.set_xlabel("Frequency (Hz)")
        ax_ph.set_ylabel("Phase (deg)")
        ax_ph.grid(True, which="both", alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()
        self.result_tabs.setCurrentIndex(0)

    # -- SPICE Netlist Export --------------------------------------------------

    def _export_spice(self):
        """Export current netlist as a .cir SPICE file."""
        text = self.netlist_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Export", "Netlist is empty.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save SPICE Netlist", "circuit.cir",
            "SPICE Files (*.cir);;All Files (*)")
        if not path:
            return

        spice_text = SPICEExporter.export(text, title="QuantumRes Circuit")
        try:
            with open(path, "w") as f:
                f.write(spice_text)
            self._log(f"[SPICE] Exported to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    # -- Transfer Function Display --------------------------------------------

    def _show_transfer_function(self):
        """Compute and display symbolic transfer function H(s)."""
        text = self.netlist_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "H(s)", "Netlist is empty.")
            return
        try:
            components, nodes = NetlistParser.parse(text)
        except Exception as exc:
            QMessageBox.critical(self, "Parse Error", str(exc))
            return

        tf_str = FilterDesignWizard.transfer_function_symbolic(components, nodes)
        self.tf_display.setPlainText(tf_str)
        self.result_tabs.setCurrentWidget(self.tf_display)
        self._log("[TF] Transfer function computed.")

    # -- Power Analysis -------------------------------------------------------

    def _run_power_analysis(self):
        """Run DC analysis and compute power dissipation per component."""
        text = self.netlist_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Power", "Netlist is empty.")
            return
        try:
            components, nodes = NetlistParser.parse(text)
        except Exception as exc:
            QMessageBox.critical(self, "Parse Error", str(exc))
            return

        solver = MNASolver(components, nodes, logger=self._log)
        try:
            voltages, currents = solver.solve_dc()
            power_list = solver.compute_power(voltages, currents)
        except Exception as exc:
            QMessageBox.critical(self, "Solver Error", str(exc))
            return

        total_power = sum(p for _, p in power_list)

        # Plot power bar chart
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        names = [n for n, _ in power_list]
        powers = [p for _, p in power_list]
        colors = ["#e74c3c" if p > 0 else "#3498db" for p in powers]
        ax.bar(names, powers, color=colors)
        ax.set_xlabel("Component")
        ax.set_ylabel("Power (W)")
        ax.set_title(f"Power Dissipation (Total: {total_power:.6g} W)")
        ax.grid(True, alpha=0.3, axis="y")
        self.figure.tight_layout()
        self.canvas.draw()

        # Table
        rows = [(name, f"{p:.6e} W", f"{p / total_power * 100:.1f}%" if total_power > 0 else "0%")
                for name, p in power_list]
        rows.append(("TOTAL", f"{total_power:.6e} W", "100%"))
        self._fill_table(["Component", "Power", "% of Total"], rows)
        self.result_tabs.setCurrentIndex(0)
        self._log(f"[PWR] Analysis complete. Total: {total_power:.6g} W")

    # -- table helper ---------------------------------------------------------

    def _fill_table(self, headers, rows):
        self.result_table.clear()
        self.result_table.setColumnCount(len(headers))
        self.result_table.setHorizontalHeaderLabels(headers)
        self.result_table.setRowCount(len(rows))
        for r, row_data in enumerate(rows):
            for c, val in enumerate(row_data):
                self.result_table.setItem(r, c, QTableWidgetItem(str(val)))
