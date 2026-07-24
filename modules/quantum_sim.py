"""
Quantum Mechanics Simulator Widget for PyQt5 Scientific Suite.
Solves 1D Schrodinger equation, computes tunneling coefficients,
hydrogen radial wavefunctions, and Kronig-Penney band structure.
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal, eigh
from scipy.special import genlaguerre, factorial
from scipy.fft import fft, ifft, fftfreq
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QDoubleSpinBox, QSpinBox, QPushButton, QSplitter,
    QFormLayout, QTabWidget, QTextEdit, QFileDialog, QMessageBox,
    QSlider, QCheckBox
)
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    from mpl_style import style_figure, style_axes
except ImportError:
    def style_figure(f): pass
    def style_axes(a): pass


# Physical constants (SI)
HBAR = 1.0545718e-34
ME = 9.10938e-31
EV = 1.602176634e-19
A0 = 5.29177e-11  # Bohr radius


def _clean_num(x, tol=1e-10):
    """Clean floating-point noise for display."""
    if isinstance(x, (float,)):
        rounded = round(x)
        if abs(x - rounded) < tol:
            return int(rounded)
        return round(x, 10)
    return x


POTENTIAL_TYPES = [
    "Infinite Well",
    "Finite Well",
    "Harmonic Oscillator",
    "Double Well",
    "Step Potential",
    "Barrier (Tunneling)",
]


class QuantumSimWidget(QWidget):
    """Interactive quantum mechanics simulation widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._logger = None
        self._build_ui()
        self._connect_signals()

    # -- public API ----------------------------------------------------------

    def set_logger(self, fn):
        self._logger = fn

    def run(self):
        self._on_run()

    # -- UI ------------------------------------------------------------------

    def _build_ui(self):
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        ctrl = QWidget()
        cl = QVBoxLayout(ctrl)

        # potential selector
        pot_group = QGroupBox("Potential")
        pl = QFormLayout()
        self.pot_combo = QComboBox()
        self.pot_combo.addItems(POTENTIAL_TYPES)
        pl.addRow("Type:", self.pot_combo)
        pot_group.setLayout(pl)
        cl.addWidget(pot_group)

        # parameters
        par_group = QGroupBox("Parameters")
        fl = QFormLayout()
        self.well_width_spin = QDoubleSpinBox(); self.well_width_spin.setRange(0.1, 50.0); self.well_width_spin.setValue(1.0); self.well_width_spin.setSuffix(" nm")
        fl.addRow("Well width:", self.well_width_spin)
        self.barrier_height_spin = QDoubleSpinBox(); self.barrier_height_spin.setRange(0.01, 100.0); self.barrier_height_spin.setValue(5.0); self.barrier_height_spin.setSuffix(" eV")
        fl.addRow("Barrier/well depth:", self.barrier_height_spin)
        self.barrier_width_spin = QDoubleSpinBox(); self.barrier_width_spin.setRange(0.01, 10.0); self.barrier_width_spin.setValue(0.5); self.barrier_width_spin.setSuffix(" nm")
        fl.addRow("Barrier width:", self.barrier_width_spin)
        self.n_states_spin = QSpinBox(); self.n_states_spin.setRange(1, 20); self.n_states_spin.setValue(5)
        fl.addRow("States to compute:", self.n_states_spin)
        self.mass_spin = QDoubleSpinBox(); self.mass_spin.setRange(0.01, 10.0); self.mass_spin.setValue(1.0); self.mass_spin.setSuffix(" m_e")
        fl.addRow("Eff. mass:", self.mass_spin)
        self.energy_spin = QDoubleSpinBox(); self.energy_spin.setRange(0.01, 50.0); self.energy_spin.setValue(2.0); self.energy_spin.setSuffix(" eV")
        fl.addRow("Particle energy:", self.energy_spin)
        par_group.setLayout(fl)
        cl.addWidget(par_group)

        # hydrogen
        h_group = QGroupBox("Hydrogen Atom")
        hl = QFormLayout()
        self.h_n_spin = QSpinBox(); self.h_n_spin.setRange(1, 10); self.h_n_spin.setValue(3)
        hl.addRow("Max n:", self.h_n_spin)
        self.h_btn = QPushButton("Plot Radial Wavefunctions")
        hl.addRow(self.h_btn)
        h_group.setLayout(hl)
        cl.addWidget(h_group)

        # band structure
        b_group = QGroupBox("Band Structure")
        bl = QFormLayout()
        self.kp_v_spin = QDoubleSpinBox(); self.kp_v_spin.setRange(0.1, 50.0); self.kp_v_spin.setValue(5.0); self.kp_v_spin.setSuffix(" eV")
        bl.addRow("Barrier V0:", self.kp_v_spin)
        self.kp_b_spin = QDoubleSpinBox(); self.kp_b_spin.setRange(0.01, 5.0); self.kp_b_spin.setValue(0.2); self.kp_b_spin.setSuffix(" nm")
        bl.addRow("Barrier width b:", self.kp_b_spin)
        self.kp_a_spin = QDoubleSpinBox(); self.kp_a_spin.setRange(0.1, 5.0); self.kp_a_spin.setValue(0.5); self.kp_a_spin.setSuffix(" nm")
        bl.addRow("Period a:", self.kp_a_spin)
        self.kp_btn = QPushButton("Compute Kronig-Penney")
        bl.addRow(self.kp_btn)
        b_group.setLayout(bl)
        cl.addWidget(b_group)

        # --- Time-dependent Schrodinger ---
        td_group = QGroupBox("Time-Dependent Evolution")
        td_layout = QFormLayout()
        self.td_x0_spin = QDoubleSpinBox()
        self.td_x0_spin.setRange(-50, 50)
        self.td_x0_spin.setValue(-0.3)
        self.td_x0_spin.setSuffix(" nm")
        td_layout.addRow("Packet center x0:", self.td_x0_spin)
        self.td_k0_spin = QDoubleSpinBox()
        self.td_k0_spin.setRange(0, 500)
        self.td_k0_spin.setValue(50.0)
        self.td_k0_spin.setSuffix(" nm^-1")
        td_layout.addRow("Packet momentum k0:", self.td_k0_spin)
        self.td_sigma_spin = QDoubleSpinBox()
        self.td_sigma_spin.setRange(0.01, 5.0)
        self.td_sigma_spin.setValue(0.1)
        self.td_sigma_spin.setSuffix(" nm")
        td_layout.addRow("Packet width sigma:", self.td_sigma_spin)
        self.td_nsteps_spin = QSpinBox()
        self.td_nsteps_spin.setRange(10, 500)
        self.td_nsteps_spin.setValue(100)
        td_layout.addRow("Animation frames:", self.td_nsteps_spin)
        self.td_btn = QPushButton("Animate Wave Packet")
        self.td_btn.clicked.connect(self._animate_wave_packet)
        td_layout.addRow(self.td_btn)
        self.td_stop_btn = QPushButton("Stop Animation")
        self.td_stop_btn.clicked.connect(self._stop_animation)
        td_layout.addRow(self.td_stop_btn)
        td_group.setLayout(td_layout)
        cl.addWidget(td_group)

        # --- Perturbation Theory ---
        pt_group = QGroupBox("Perturbation Theory")
        pt_layout = QFormLayout()
        self.pt_strength_spin = QDoubleSpinBox()
        self.pt_strength_spin.setRange(0.001, 10.0)
        self.pt_strength_spin.setValue(0.5)
        self.pt_strength_spin.setSuffix(" eV")
        pt_layout.addRow("Perturbation V':", self.pt_strength_spin)
        self.pt_type_combo = QComboBox()
        self.pt_type_combo.addItems(["Linear (electric field)", "Quadratic", "Gaussian bump"])
        pt_layout.addRow("Perturbation type:", self.pt_type_combo)
        self.pt_btn = QPushButton("Compute Corrections")
        self.pt_btn.clicked.connect(self._compute_perturbation)
        pt_layout.addRow(self.pt_btn)
        pt_group.setLayout(pt_layout)
        cl.addWidget(pt_group)

        # --- Multi-particle ---
        mp_group = QGroupBox("Multi-Particle (2-body 1D)")
        mp_layout = QFormLayout()
        self.mp_symmetry_combo = QComboBox()
        self.mp_symmetry_combo.addItems(["Distinguishable", "Bosonic", "Fermionic"])
        mp_layout.addRow("Symmetry:", self.mp_symmetry_combo)
        self.mp_interaction_spin = QDoubleSpinBox()
        self.mp_interaction_spin.setRange(0, 50)
        self.mp_interaction_spin.setValue(2.0)
        self.mp_interaction_spin.setSuffix(" eV")
        mp_layout.addRow("Interaction V_12:", self.mp_interaction_spin)
        self.mp_btn = QPushButton("Solve 2-Particle System")
        self.mp_btn.clicked.connect(self._solve_two_particle)
        mp_layout.addRow(self.mp_btn)
        mp_group.setLayout(mp_layout)
        cl.addWidget(mp_group)

        # --- Utility buttons ---
        util_group = QGroupBox("Tools")
        util_layout = QVBoxLayout()
        self.dos_btn = QPushButton("Density of States (from bands)")
        self.dos_btn.clicked.connect(self._compute_dos)
        util_layout.addWidget(self.dos_btn)
        self.tunnel_btn = QPushButton("Tunneling Visualizer")
        self.tunnel_btn.clicked.connect(self._visualize_tunneling)
        util_layout.addWidget(self.tunnel_btn)
        self.export_btn = QPushButton("Export Wavefunctions to CSV")
        self.export_btn.clicked.connect(self._export_csv)
        util_layout.addWidget(self.export_btn)
        util_group.setLayout(util_layout)
        cl.addWidget(util_group)

        # run
        self.run_btn = QPushButton("Solve Schrodinger Equation")
        cl.addWidget(self.run_btn)
        cl.addStretch()
        splitter.addWidget(ctrl)

        # right: plots + info
        right = QWidget()
        rl = QVBoxLayout(right)
        self.tabs = QTabWidget()

        self.fig_wf = Figure(figsize=(6, 4))
        style_figure(self.fig_wf)
        self.ax_wf = self.fig_wf.add_subplot(111)
        self.canvas_wf = FigureCanvas(self.fig_wf)
        self.tabs.addTab(self.canvas_wf, "Wavefunctions")

        self.fig_pd = Figure(figsize=(6, 4))
        style_figure(self.fig_pd)
        self.ax_pd = self.fig_pd.add_subplot(111)
        self.canvas_pd = FigureCanvas(self.fig_pd)
        self.tabs.addTab(self.canvas_pd, "Probability Density")

        self.fig_hy = Figure(figsize=(6, 4))
        style_figure(self.fig_hy)
        self.ax_hy = self.fig_hy.add_subplot(111)
        self.canvas_hy = FigureCanvas(self.fig_hy)
        self.tabs.addTab(self.canvas_hy, "Hydrogen Atom")

        self.fig_bs = Figure(figsize=(6, 4))
        style_figure(self.fig_bs)
        self.ax_bs = self.fig_bs.add_subplot(111)
        self.canvas_bs = FigureCanvas(self.fig_bs)
        self.tabs.addTab(self.canvas_bs, "Band Structure")

        self.fig_td = Figure(figsize=(6, 4))
        style_figure(self.fig_td)
        self.ax_td = self.fig_td.add_subplot(111)
        self.canvas_td = FigureCanvas(self.fig_td)
        self.tabs.addTab(self.canvas_td, "Time-Dependent")

        self.fig_mp = Figure(figsize=(6, 4))
        style_figure(self.fig_mp)
        self.ax_mp = self.fig_mp.add_subplot(111)
        self.canvas_mp = FigureCanvas(self.fig_mp)
        self.tabs.addTab(self.canvas_mp, "Multi-Particle")

        self.fig_dos = Figure(figsize=(6, 4))
        style_figure(self.fig_dos)
        self.ax_dos = self.fig_dos.add_subplot(111)
        self.canvas_dos = FigureCanvas(self.fig_dos)
        self.tabs.addTab(self.canvas_dos, "Density of States")

        rl.addWidget(self.tabs)

        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setMaximumHeight(120)
        rl.addWidget(self.info_box)

        # Internal state for caching last solution and animation
        self._last_solution = None  # (x, V_eV, energies, states, n_calc)
        self._anim_timer = None

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

    def _connect_signals(self):
        self.run_btn.clicked.connect(self._on_run)
        self.h_btn.clicked.connect(self._plot_hydrogen)
        self.kp_btn.clicked.connect(self._plot_kronig_penney)

    def _log(self, msg):
        if self._logger:
            self._logger(msg)

    # -- Schrodinger solver --------------------------------------------------

    def _on_run(self):
        pot_type = self.pot_combo.currentText()
        self._log(f"Solving: {pot_type}")
        L_nm = self.well_width_spin.value()
        V0_eV = self.barrier_height_spin.value()
        bw_nm = self.barrier_width_spin.value()
        n_states = self.n_states_spin.value()
        m_eff = self.mass_spin.value() * ME
        E_particle = self.energy_spin.value()

        L = L_nm * 1e-9
        V0 = V0_eV * EV
        bw = bw_nm * 1e-9

        N = 1000
        # extend grid beyond well for finite potentials
        pad = L * 1.5 if pot_type != "Infinite Well" else 0
        x = np.linspace(-pad, L + pad, N)
        dx = x[1] - x[0]

        V = self._build_potential(pot_type, x, L, V0, bw)

        # matrix diagonalization (finite-difference Hamiltonian)
        diag = HBAR ** 2 / (m_eff * dx ** 2) + V
        off_diag = -0.5 * HBAR ** 2 / (m_eff * dx ** 2) * np.ones(N - 1)

        n_calc = min(n_states, N - 2)
        energies, states = eigh_tridiagonal(diag / EV, off_diag / EV,
                                            select="i", select_range=(0, n_calc - 1))
        energies_eV = energies  # already in eV due to division

        # normalize wavefunctions
        for i in range(n_calc):
            norm = np.sqrt(np.trapz(states[:, i] ** 2, x))
            if norm > 0:
                states[:, i] /= norm

        # Cache solution for export and other tools
        self._last_solution = (x, V / EV, energies_eV, states, n_calc, pot_type,
                               L, V0, bw, m_eff)

        self._plot_wavefunctions(x, V / EV, energies_eV, states, n_calc, pot_type)
        self._plot_probability(x, V / EV, energies_eV, states, n_calc, pot_type)

        # info
        lines = [f"Potential: {pot_type}", f"Eigenvalues (eV):"]
        for i, e in enumerate(energies_eV):
            lines.append(f"  n={i + 1}: {_clean_num(float(e)):.12g} eV")

        # transmission coefficient for barrier
        if pot_type == "Barrier (Tunneling)":
            T = self._transmission_coefficient(E_particle * EV, V0, bw, m_eff)
            lines.append(f"\nTransmission (E={E_particle:.2f} eV): T = {T:.6e}")

        self.info_box.setPlainText("\n".join(lines))
        self._log("Solver complete")

    def _build_potential(self, pot_type, x, L, V0, bw):
        N = len(x)
        V = np.zeros(N)
        if pot_type == "Infinite Well":
            V[x < 0] = 1e6 * EV
            V[x > L] = 1e6 * EV
        elif pot_type == "Finite Well":
            V[x < 0] = V0
            V[x > L] = V0
        elif pot_type == "Harmonic Oscillator":
            omega = np.sqrt(2 * V0 / (ME * (L / 2) ** 2))
            V = 0.5 * ME * omega ** 2 * (x - L / 2) ** 2
        elif pot_type == "Double Well":
            cx1, cx2 = L * 0.3, L * 0.7
            hw = L * 0.15
            V[:] = V0
            V[np.abs(x - cx1) < hw] = 0
            V[np.abs(x - cx2) < hw] = 0
        elif pot_type == "Step Potential":
            V[x > L / 2] = V0
        elif pot_type == "Barrier (Tunneling)":
            center = L / 2
            V[(x > center - bw / 2) & (x < center + bw / 2)] = V0
        return V

    # -- plotting ------------------------------------------------------------

    def _plot_wavefunctions(self, x, V_eV, energies, states, n, title):
        ax = self.ax_wf
        ax.clear()
        x_nm = x * 1e9
        ax.fill_between(x_nm, 0, V_eV, alpha=0.15, color="gray", label="V(x)")
        ax.plot(x_nm, V_eV, "k-", linewidth=1)
        scale = (energies[-1] - energies[0]) / n if n > 1 else 1.0
        colors = ["#2c3e50", "#e74c3c", "#27ae60", "#8e44ad", "#e67e22",
                  "#1abc9c", "#d35400", "#2980b9", "#c0392b", "#16a085"]
        for i in range(n):
            psi = states[:, i]
            psi_scaled = psi * scale * 0.4
            ax.plot(x_nm, energies[i] + psi_scaled, color=colors[i % len(colors)],
                    linewidth=1.2, label=f"n={i + 1} ({energies[i]:.4f} eV)")
            ax.axhline(energies[i], color=colors[i % len(colors)], linewidth=0.5, linestyle="--", alpha=0.5)
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("Energy (eV) / \u03c8 (a.u.)")
        ax.set_title(f"Wavefunctions \u2014 {title}")
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.3)
        self.fig_wf.tight_layout()
        self.canvas_wf.draw()
        self.tabs.setCurrentIndex(0)

    def _plot_probability(self, x, V_eV, energies, states, n, title):
        ax = self.ax_pd
        ax.clear()
        x_nm = x * 1e9
        ax.fill_between(x_nm, 0, V_eV, alpha=0.1, color="gray")
        ax.plot(x_nm, V_eV, "k-", linewidth=0.8)
        scale = (energies[-1] - energies[0]) / n if n > 1 else 1.0
        colors = ["#2c3e50", "#e74c3c", "#27ae60", "#8e44ad", "#e67e22",
                  "#1abc9c", "#d35400", "#2980b9", "#c0392b", "#16a085"]
        for i in range(n):
            pd = states[:, i] ** 2
            pd_scaled = pd * scale * 0.4
            ax.fill_between(x_nm, energies[i], energies[i] + pd_scaled,
                            alpha=0.4, color=colors[i % len(colors)])
            ax.plot(x_nm, energies[i] + pd_scaled, color=colors[i % len(colors)], linewidth=1)
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("Energy (eV) / |\u03c8|\u00b2 (a.u.)")
        ax.set_title(f"Probability Density \u2014 {title}")
        ax.grid(True, alpha=0.3)
        self.fig_pd.tight_layout()
        self.canvas_pd.draw()

    # -- transmission coefficient --------------------------------------------

    def _transmission_coefficient(self, E, V0, bw, m):
        """Analytical transmission for rectangular barrier."""
        if E >= V0:
            k2 = np.sqrt(2 * m * (E - V0)) / HBAR
            if abs(k2 * bw) < 1e-12:
                return 1.0
            k1 = np.sqrt(2 * m * E) / HBAR
            denom = 1 + ((k1 ** 2 - k2 ** 2) * np.sin(k2 * bw)) ** 2 / (4 * k1 ** 2 * k2 ** 2)
            return 1.0 / denom
        else:
            kappa = np.sqrt(2 * m * (V0 - E)) / HBAR
            k1 = np.sqrt(2 * m * E) / HBAR
            denom = 1 + ((k1 ** 2 + kappa ** 2) * np.sinh(kappa * bw)) ** 2 / (4 * k1 ** 2 * kappa ** 2)
            return 1.0 / denom

    # -- hydrogen atom radial wavefunctions ----------------------------------

    def _plot_hydrogen(self):
        n_max = self.h_n_spin.value()
        self._log(f"Plotting hydrogen radial wavefunctions up to n={n_max}")
        ax = self.ax_hy
        ax.clear()
        r = np.linspace(1e-3, 40, 2000)  # in units of a0
        colors = ["#2c3e50", "#e74c3c", "#27ae60", "#8e44ad", "#e67e22",
                  "#1abc9c", "#d35400", "#2980b9", "#c0392b", "#16a085"]
        ci = 0
        for n in range(1, n_max + 1):
            for l in range(n):
                R = self._hydrogen_radial(n, l, r)
                ax.plot(r, R ** 2 * r ** 2, color=colors[ci % len(colors)],
                        linewidth=1.2, label=f"n={n}, l={l}")
                ci += 1
        ax.set_xlabel("r / a\u2080")
        ax.set_ylabel("r\u00b2 |R(r)|\u00b2")
        ax.set_title("Hydrogen Atom Radial Probability Density")
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.3)
        self.fig_hy.tight_layout()
        self.canvas_hy.draw()
        self.tabs.setCurrentIndex(2)

    @staticmethod
    def _hydrogen_radial(n, l, r):
        """Compute R_{nl}(r) for hydrogen, r in units of Bohr radius."""
        rho = 2.0 * r / n
        norm = np.sqrt((2.0 / n) ** 3 * factorial(n - l - 1, exact=True) /
                       (2.0 * n * factorial(n + l, exact=True)))
        L = genlaguerre(n - l - 1, 2 * l + 1)(rho)
        return norm * np.exp(-rho / 2) * rho ** l * L

    # -- Kronig-Penney band structure ----------------------------------------

    def _plot_kronig_penney(self):
        V0_eV = self.kp_v_spin.value()
        b_nm = self.kp_b_spin.value()
        a_nm = self.kp_a_spin.value()
        self._log(f"Kronig-Penney: V0={V0_eV} eV, b={b_nm} nm, a={a_nm} nm")

        V0 = V0_eV * EV
        b = b_nm * 1e-9
        a = a_nm * 1e-9
        period = a + b

        ax = self.ax_bs
        ax.clear()

        # scan energy to find allowed bands
        E_scan = np.linspace(0.001 * EV, 15 * EV, 5000)
        ka_vals = np.full_like(E_scan, np.nan)

        for i, E in enumerate(E_scan):
            alpha = np.sqrt(2 * ME * E) / HBAR
            if E < V0:
                beta = np.sqrt(2 * ME * (V0 - E)) / HBAR
                f_val = (np.cos(alpha * a) * np.cosh(beta * b)
                         - (alpha ** 2 - beta ** 2) / (2 * alpha * beta)
                         * np.sin(alpha * a) * np.sinh(beta * b))
            else:
                beta = np.sqrt(2 * ME * (E - V0)) / HBAR
                f_val = (np.cos(alpha * a) * np.cos(beta * b)
                         - (alpha ** 2 + beta ** 2) / (2 * alpha * beta)
                         * np.sin(alpha * a) * np.sin(beta * b))
            if abs(f_val) <= 1.0:
                ka_vals[i] = np.arccos(np.clip(f_val, -1, 1))

        E_eV = E_scan / EV
        # plot allowed bands (as filled regions)
        allowed = ~np.isnan(ka_vals)
        ka_norm = ka_vals / np.pi  # normalize to pi/period

        ax.plot(ka_norm[allowed], E_eV[allowed], "b.", markersize=0.5)
        ax.plot(-ka_norm[allowed], E_eV[allowed], "b.", markersize=0.5)

        # highlight band gaps
        prev = False
        gap_start = 0
        for i in range(len(allowed)):
            if prev and not allowed[i]:
                gap_start = E_eV[i]
            if not prev and allowed[i] and gap_start > 0:
                ax.axhspan(gap_start, E_eV[i], alpha=0.15, color="red")
                gap_start = 0
            prev = allowed[i]

        ax.set_xlabel("ka / \u03c0")
        ax.set_ylabel("Energy (eV)")
        ax.set_title("1D Kronig-Penney Band Structure")
        ax.set_xlim(-1, 1)
        ax.grid(True, alpha=0.3)
        self.fig_bs.tight_layout()
        self.canvas_bs.draw()
        self.tabs.setCurrentIndex(3)
        self._log("Band structure complete")

    # -- Time-dependent wave packet animation --------------------------------

    def _animate_wave_packet(self):
        """Create a Gaussian wave packet and propagate via split-operator FFT."""
        self._stop_animation()  # stop any running animation first

        pot_type = self.pot_combo.currentText()
        L_nm = self.well_width_spin.value()
        V0_eV = self.barrier_height_spin.value()
        bw_nm = self.barrier_width_spin.value()
        m_eff = self.mass_spin.value() * ME

        L = L_nm * 1e-9
        V0 = V0_eV * EV
        bw = bw_nm * 1e-9

        # Spatial grid -- wider than the well so the packet can travel
        pad = L * 3.0
        N = 1024  # power of 2 for FFT efficiency
        x = np.linspace(-pad, L + pad, N)
        dx = x[1] - x[0]

        # Build potential on this grid (in Joules)
        V = self._build_potential(pot_type, x, L, V0, bw)

        # Read user parameters for the wave packet
        x0 = self.td_x0_spin.value() * 1e-9        # centre position (m)
        k0 = self.td_k0_spin.value() * 1e9          # momentum wave-vector (1/m)
        sigma = self.td_sigma_spin.value() * 1e-9    # width (m)
        n_frames = self.td_nsteps_spin.value()

        # Time step chosen so the packet moves visibly but stays stable
        dt = 0.4 * m_eff * dx ** 2 / HBAR  # ~0.4 of the stability limit

        # --- Construct split-operator propagators ---
        # Half-step potential propagator  exp(-i V dt / 2 hbar)
        half_V_prop = np.exp(-0.5j * V * dt / HBAR)

        # Kinetic propagator in momentum space  exp(-i hbar k^2 dt / 2m)
        k = 2.0 * np.pi * fftfreq(N, d=dx)
        T_prop = np.exp(-0.5j * HBAR * k ** 2 * dt / m_eff)

        # --- Initial Gaussian wave packet ---
        psi = ((1.0 / (sigma * np.sqrt(np.pi))) ** 0.5
               * np.exp(-(x - x0) ** 2 / (2.0 * sigma ** 2))
               * np.exp(1j * k0 * x))

        # Pre-compute all frames so the animation is smooth
        x_nm = x * 1e9
        V_eV = V / EV
        prob_frames = []
        for _ in range(n_frames):
            prob_frames.append(np.abs(psi) ** 2)
            # Split-operator step: V/2 -> T -> V/2
            psi = half_V_prop * psi
            psi = ifft(T_prop * fft(psi))
            psi = half_V_prop * psi

        # Determine y-axis range from all frames
        max_prob = max(f.max() for f in prob_frames)
        max_y = max(max_prob * 1.15, V_eV.max() * 1.05) if V_eV.max() > 0 else max_prob * 1.15

        # --- Set up the plot ---
        ax = self.ax_td
        ax.clear()
        ax.fill_between(x_nm, 0, V_eV, alpha=0.20, color="gray", label="V(x)")
        ax.plot(x_nm, V_eV, "k-", linewidth=0.8)
        prob_line, = ax.plot(x_nm, prob_frames[0] * 1e-9, "b-", linewidth=1.4,
                             label="|ψ(x,t)|²")
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("|ψ|² (nm⁻¹)  /  V (eV)")
        ax.set_title(f"Wave Packet Propagation — {pot_type}")
        ax.set_ylim(0, max_y * 1e-9 if max_y * 1e-9 > V_eV.max() else V_eV.max() * 1.3)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
        self.fig_td.tight_layout()
        self.canvas_td.draw()
        self.tabs.setCurrentIndex(4)

        # --- QTimer-based animation ---
        self._anim_frame_idx = 0
        self._anim_frames = prob_frames
        self._anim_line = prob_line
        self._anim_x_nm = x_nm

        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(33)  # ~30 fps

        def _update_frame():
            idx = self._anim_frame_idx
            if idx >= len(self._anim_frames):
                self._anim_frame_idx = 0  # loop
                idx = 0
            self._anim_line.set_ydata(self._anim_frames[idx] * 1e-9)
            self.canvas_td.draw_idle()
            self._anim_frame_idx += 1

        self._anim_timer.timeout.connect(_update_frame)
        self._anim_timer.start()
        self._log(f"Wave packet animation started ({n_frames} frames, {pot_type})")

    def _stop_animation(self):
        """Stop any running wave packet animation."""
        if self._anim_timer is not None:
            self._anim_timer.stop()
            self._anim_timer = None
        self._log("Animation stopped")

    # -- Perturbation theory ---------------------------------------------------

    def _compute_perturbation(self):
        """Compute first- and second-order perturbation corrections.

        Computes <n|V'|n> for first order and the sum of
        |<m|V'|n>|^2 / (E_n - E_m) for second order.  Displays a
        comparison of unperturbed vs corrected energy levels.
        """
        if self._last_solution is None:
            QMessageBox.warning(self, "No solution",
                                "Run the Schrodinger solver first.")
            return
        x, V_eV, energies, states, n_calc = self._last_solution[:5]
        strength = self.pt_strength_spin.value()  # eV
        pt_type = self.pt_type_combo.currentText()
        L = x[-1] - x[0]

        # Build perturbation V'(x) in eV
        if pt_type.startswith("Linear"):
            Vp = strength * (x - x.mean()) / (L / 2)
        elif pt_type == "Quadratic":
            Vp = strength * ((x - x.mean()) / (L / 2)) ** 2
        else:  # Gaussian bump
            Vp = strength * np.exp(-((x - x.mean()) ** 2) / (0.05 * L) ** 2)

        # First-order corrections
        E1_arr = np.zeros(n_calc)
        for i in range(n_calc):
            E1_arr[i] = np.trapz(states[:, i] ** 2 * Vp, x)

        # Second-order corrections
        E2_arr = np.zeros(n_calc)
        for i in range(n_calc):
            for j in range(n_calc):
                if j == i:
                    continue
                V_ij = np.trapz(states[:, i] * Vp * states[:, j], x)
                dE = energies[i] - energies[j]
                if abs(dE) > 1e-12:
                    E2_arr[i] += V_ij ** 2 / dE

        # Info text
        lines = [f"Perturbation theory: {pt_type} (strength = {strength} eV)",
                 "",
                 f"{'n':>3} {'E0 (eV)':>12} {'E1 (eV)':>14} {'E2 (eV)':>14} {'E_corr (eV)':>14}",
                 "-" * 62]
        for i in range(n_calc):
            E_corr = energies[i] + E1_arr[i] + E2_arr[i]
            lines.append(f"  {i+1:>2}  {_clean_num(float(energies[i])):>12.6f}  {E1_arr[i]:>+14.6e}  "
                         f"{E2_arr[i]:>+14.6e}  {_clean_num(float(E_corr)):>14.6f}")
        self.info_box.setPlainText("\n".join(lines))

        # Plot comparison of energy levels
        ax = self.ax_wf
        ax.clear()
        x_nm = x * 1e9
        ax.fill_between(x_nm, 0, V_eV + Vp, alpha=0.15, color="orange", label="V+V'")
        ax.fill_between(x_nm, 0, V_eV, alpha=0.15, color="gray", label="V(x)")
        colors = ["#2c3e50", "#e74c3c", "#27ae60", "#8e44ad", "#e67e22",
                  "#1abc9c", "#d35400", "#2980b9", "#c0392b", "#16a085"]
        for i in range(n_calc):
            E0 = energies[i]
            E_corr = E0 + E1_arr[i] + E2_arr[i]
            ax.axhline(E0, color=colors[i % len(colors)], linewidth=0.8,
                       linestyle="--", alpha=0.5)
            ax.axhline(E_corr, color=colors[i % len(colors)], linewidth=1.5,
                       label=f"n={i+1}: {E0:.4f} -> {E_corr:.4f}")
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("Energy (eV)")
        ax.set_title("Perturbation Theory: Energy Level Shifts")
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.3)
        self.fig_wf.tight_layout()
        self.canvas_wf.draw()
        self.tabs.setCurrentIndex(0)
        self._log("[PT] First- and second-order corrections computed.")

    # -- Two-particle solver ---------------------------------------------------

    def _solve_two_particle(self):
        """Solve a two-particle 1D system with symmetry constraints.

        Uses a product basis of single-particle eigenstates and
        diagonalizes the full Hamiltonian including a contact interaction.
        """
        L_nm = self.well_width_spin.value()
        m_eff = self.mass_spin.value() * ME
        symmetry = self.mp_symmetry_combo.currentText()
        V_int_eV = self.mp_interaction_spin.value()

        L = L_nm * 1e-9

        # Small grid for the single-particle problem (keep 2D tractable)
        N = 60
        x = np.linspace(0, L, N)
        dx = x[1] - x[0]

        # Single-particle infinite-well Hamiltonian (tridiagonal)
        diag_sp = HBAR ** 2 / (m_eff * dx ** 2) * np.ones(N)
        off_sp = -0.5 * HBAR ** 2 / (m_eff * dx ** 2) * np.ones(N - 1)

        n_sp_max = min(6, N - 2)
        E_sp, psi_sp = eigh_tridiagonal(diag_sp / EV, off_sp / EV,
                                         select="i", select_range=(0, n_sp_max - 1))
        # Normalize
        for i in range(n_sp_max):
            norm = np.sqrt(np.trapz(psi_sp[:, i] ** 2, x))
            if norm > 0:
                psi_sp[:, i] /= norm

        # Build two-particle basis states
        basis_states = []
        if symmetry == "Distinguishable":
            for i in range(n_sp_max):
                for j in range(n_sp_max):
                    basis_states.append((i, j))
        elif symmetry == "Bosonic":
            for i in range(n_sp_max):
                for j in range(i, n_sp_max):
                    basis_states.append((i, j))
        else:  # Fermionic
            for i in range(n_sp_max):
                for j in range(i + 1, n_sp_max):
                    basis_states.append((i, j))

        n_basis = len(basis_states)
        if n_basis == 0:
            QMessageBox.warning(self, "Multi-Particle",
                                "No valid basis states (need more single-particle levels for Fermionic).")
            return

        H = np.zeros((n_basis, n_basis))

        for a, (i, j) in enumerate(basis_states):
            H[a, a] = E_sp[i] + E_sp[j]

            for b, (k, l) in enumerate(basis_states):
                # Contact interaction: V_int * delta(x1 - x2)
                if symmetry == "Distinguishable":
                    overlap = np.sum(psi_sp[:, i] * psi_sp[:, k] *
                                     psi_sp[:, j] * psi_sp[:, l]) * dx
                elif symmetry == "Bosonic":
                    fac_a = np.sqrt(2) if i != j else 1.0
                    fac_b = np.sqrt(2) if k != l else 1.0
                    psi_a = (psi_sp[:, i] * psi_sp[:, j] +
                             psi_sp[:, j] * psi_sp[:, i]) / fac_a
                    psi_b = (psi_sp[:, k] * psi_sp[:, l] +
                             psi_sp[:, l] * psi_sp[:, k]) / fac_b
                    overlap = np.sum(psi_a * psi_b) * dx
                else:  # Fermionic
                    psi_a = (psi_sp[:, i] * psi_sp[:, j] -
                             psi_sp[:, j] * psi_sp[:, i]) / np.sqrt(2)
                    psi_b = (psi_sp[:, k] * psi_sp[:, l] -
                             psi_sp[:, l] * psi_sp[:, k]) / np.sqrt(2)
                    overlap = np.sum(psi_a * psi_b) * dx

                H[a, b] += V_int_eV * overlap

        E_2p, C_2p = eigh(H)

        # --- Plot energy levels ---
        ax = self.ax_mp
        ax.clear()
        n_show = min(12, len(E_2p))
        colors = ["#2c3e50", "#e74c3c", "#27ae60", "#8e44ad", "#e67e22",
                  "#1abc9c", "#d35400", "#2980b9", "#c0392b", "#16a085",
                  "#34495e", "#f39c12"]

        for i in range(n_show):
            ax.hlines(E_2p[i], 0.2, 0.8, colors=colors[i % len(colors)], linewidth=2)
            ax.text(0.85, E_2p[i], f"{E_2p[i]:.4f}", fontsize=6, va="center")

        # Non-interacting comparison
        for i in range(n_sp_max):
            start = i if symmetry != "Fermionic" else i + 1
            for j in range(start, n_sp_max):
                E_ni = E_sp[i] + E_sp[j]
                ax.hlines(E_ni, 1.2, 1.8, colors="gray", linewidth=1,
                          linestyle="--", alpha=0.5)

        ax.set_xlim(0, 2.2)
        ax.set_ylabel("Energy (eV)")
        ax.set_title(f"Two-Particle System ({symmetry})")
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(["Interacting", "Non-interacting"])
        ax.grid(True, alpha=0.3, axis="y")
        self.fig_mp.tight_layout()
        self.canvas_mp.draw()
        self.tabs.setCurrentIndex(5)

        # Info
        lines = [f"Two-Particle 1D System ({symmetry})",
                 f"Well: {L_nm} nm, Interaction: {V_int_eV} eV",
                 f"Single-particle states: {n_sp_max}, Basis size: {n_basis}\n",
                 "Lowest two-particle energies:"]
        for i in range(min(10, len(E_2p))):
            dom = np.argmax(np.abs(C_2p[:, i]))
            st = basis_states[dom]
            lines.append(f"  E_{i+1} = {_clean_num(float(E_2p[i])):.12g} eV  (dominant: |{st[0]+1},{st[1]+1}>)")
        self.info_box.setPlainText("\n".join(lines))
        self._log(f"[MP] Solved {symmetry} 2-particle, {n_basis} basis states.")

    # -- Density of states from Kronig-Penney band structure ------------------

    def _compute_dos(self):
        """Compute density of states from Kronig-Penney band structure.

        Performs a dense energy scan, identifies allowed bands, and
        histograms the allowed energies to produce a DOS curve.
        """
        V0_eV = self.kp_v_spin.value()
        b_nm = self.kp_b_spin.value()
        a_nm = self.kp_a_spin.value()

        V0 = V0_eV * EV
        b = b_nm * 1e-9
        a = a_nm * 1e-9

        # Dense energy scan
        n_scan = 10000
        E_scan = np.linspace(0.001 * EV, 15 * EV, n_scan)
        allowed_energies = []

        for E in E_scan:
            alpha = np.sqrt(2 * ME * E) / HBAR
            if E < V0:
                beta_v = np.sqrt(2 * ME * (V0 - E)) / HBAR
                f_val = (np.cos(alpha * a) * np.cosh(beta_v * b)
                         - (alpha ** 2 - beta_v ** 2) / (2 * alpha * beta_v)
                         * np.sin(alpha * a) * np.sinh(beta_v * b))
            else:
                beta_v = np.sqrt(2 * ME * (E - V0)) / HBAR
                f_val = (np.cos(alpha * a) * np.cos(beta_v * b)
                         - (alpha ** 2 + beta_v ** 2) / (2 * alpha * beta_v)
                         * np.sin(alpha * a) * np.sin(beta_v * b))
            if abs(f_val) <= 1.0:
                allowed_energies.append(E / EV)

        if not allowed_energies:
            QMessageBox.warning(self, "DOS", "No allowed energies found. Adjust band structure parameters.")
            return

        E_arr = np.array(allowed_energies)
        n_bins = 200
        counts, bin_edges = np.histogram(E_arr, bins=n_bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        dE = bin_edges[1] - bin_edges[0]
        dos = counts / dE

        ax = self.ax_dos
        ax.clear()
        ax.fill_between(bin_centers, 0, dos, alpha=0.4, color="#3498db")
        ax.plot(bin_centers, dos, "b-", linewidth=1)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Density of States (a.u.)")
        ax.set_title(f"1D DOS (Kronig-Penney: V0={V0_eV} eV, a={a_nm} nm, b={b_nm} nm)")
        ax.grid(True, alpha=0.3)

        # Mark approximate band edges
        E_eV_full = E_scan / EV
        prev_allowed = False
        for i in range(n_scan):
            alpha = np.sqrt(2 * ME * E_scan[i]) / HBAR
            if E_scan[i] < V0:
                beta_v = np.sqrt(2 * ME * (V0 - E_scan[i])) / HBAR
                f_val = (np.cos(alpha * a) * np.cosh(beta_v * b)
                         - (alpha ** 2 - beta_v ** 2) / (2 * alpha * beta_v)
                         * np.sin(alpha * a) * np.sinh(beta_v * b))
            else:
                beta_v = np.sqrt(2 * ME * (E_scan[i] - V0)) / HBAR
                f_val = (np.cos(alpha * a) * np.cos(beta_v * b)
                         - (alpha ** 2 + beta_v ** 2) / (2 * alpha * beta_v)
                         * np.sin(alpha * a) * np.sin(beta_v * b))
            is_allowed = abs(f_val) <= 1.0
            if prev_allowed and not is_allowed:
                ax.axvline(E_eV_full[i], color="red", linestyle="--", alpha=0.5, linewidth=0.8)
            elif not prev_allowed and is_allowed:
                ax.axvline(E_eV_full[i], color="green", linestyle="--", alpha=0.5, linewidth=0.8)
            prev_allowed = is_allowed

        self.fig_dos.tight_layout()
        self.canvas_dos.draw()
        self.tabs.setCurrentIndex(6)
        self._log(f"[DOS] Computed from Kronig-Penney, {len(allowed_energies)} allowed states.")

    # -- Tunneling visualizer with wavefunction display ----------------------

    def _visualize_tunneling(self):
        """Visualize tunneling: incident, reflected, and transmitted waves,
        plus T(E) transmission curve.
        """
        V0_eV = self.barrier_height_spin.value()
        bw_nm = self.barrier_width_spin.value()
        E_eV = self.energy_spin.value()
        m_eff = self.mass_spin.value() * ME
        L_nm = self.well_width_spin.value()

        V0 = V0_eV * EV
        bw = bw_nm * 1e-9
        E = E_eV * EV
        L = L_nm * 1e-9

        T_coeff = self._transmission_coefficient(E, V0, bw, m_eff)
        R_coeff = 1 - T_coeff
        k1 = np.sqrt(2 * m_eff * E) / HBAR

        # Spatial grid
        x = np.linspace(-2 * L, 3 * L, 2000)
        x_nm = x * 1e9
        center = L / 2

        # Barrier region masks
        barrier = (x > center - bw / 2) & (x < center + bw / 2)
        left = x < center - bw / 2
        right = x > center + bw / 2

        # Build wavefunctions for visualization
        psi_inc = np.zeros_like(x, dtype=complex)
        psi_ref = np.zeros_like(x, dtype=complex)
        psi_trans = np.zeros_like(x, dtype=complex)
        psi_barrier = np.zeros_like(x, dtype=complex)

        psi_inc[left] = np.exp(1j * k1 * x[left])
        psi_ref[left] = np.sqrt(R_coeff) * np.exp(-1j * k1 * x[left])

        if E < V0:
            kappa = np.sqrt(2 * m_eff * (V0 - E)) / HBAR
            psi_barrier[barrier] = np.exp(-kappa * (x[barrier] - center + bw / 2))
        else:
            k2 = np.sqrt(2 * m_eff * (E - V0)) / HBAR
            psi_barrier[barrier] = np.exp(1j * k2 * x[barrier])

        psi_trans[right] = np.sqrt(T_coeff) * np.exp(1j * k1 * x[right])

        # --- Plot wavefunctions ---
        ax = self.ax_td
        ax.clear()

        V_plot = np.zeros_like(x)
        V_plot[barrier] = V0_eV
        ax.fill_between(x_nm, 0, V_plot, alpha=0.2, color="gray", label="Barrier")

        ax.plot(x_nm, np.abs(psi_inc) ** 2, "b-", linewidth=1.2, label="Incident")
        ax.plot(x_nm, np.abs(psi_ref) ** 2, "r-", linewidth=1.2,
                label=f"Reflected (R={R_coeff:.4f})")
        ax.plot(x_nm, np.abs(psi_trans) ** 2, "g-", linewidth=1.2,
                label=f"Transmitted (T={T_coeff:.4f})")
        ax.plot(x_nm, np.abs(psi_barrier) ** 2, "m-", linewidth=1.2,
                alpha=0.7, label="In barrier")

        ax.set_xlabel("x (nm)")
        ax.set_ylabel("|psi|^2")
        ax.set_title(f"Quantum Tunneling: E={E_eV:.2f} eV, V0={V0_eV:.2f} eV")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        self.fig_td.tight_layout()
        self.canvas_td.draw()
        self.tabs.setCurrentIndex(4)

        # --- T(E) curve on probability density tab ---
        E_sweep = np.linspace(0.01 * EV, V0_eV * 3 * EV, 500)
        T_sweep = np.array([self._transmission_coefficient(e, V0, bw, m_eff) for e in E_sweep])

        ax2 = self.ax_pd
        ax2.clear()
        ax2.semilogy(E_sweep / EV, T_sweep, "b-", linewidth=1.5)
        ax2.axvline(E_eV, color="r", linestyle="--", label=f"E = {E_eV} eV")
        ax2.axvline(V0_eV, color="gray", linestyle=":", label=f"V0 = {V0_eV} eV")
        ax2.set_xlabel("Energy (eV)")
        ax2.set_ylabel("Transmission T(E)")
        ax2.set_title("Transmission Coefficient vs Energy")
        ax2.legend(fontsize=7)
        ax2.grid(True, which="both", alpha=0.3)
        self.fig_pd.tight_layout()
        self.canvas_pd.draw()

        self.info_box.setPlainText(
            f"Tunneling Analysis:\n"
            f"  E = {E_eV:.4f} eV, V0 = {V0_eV:.4f} eV\n"
            f"  Barrier width = {bw_nm:.4f} nm\n"
            f"  Eff. mass = {self.mass_spin.value():.2f} m_e\n"
            f"  T = {T_coeff:.6e}, R = {R_coeff:.6e}\n"
            f"  k1 = {k1:.4e} m^-1"
        )
        self._log(f"[TUNNEL] T={T_coeff:.6e}, R={R_coeff:.6e}")

    # -- CSV export ----------------------------------------------------------

    def _export_csv(self):
        """Export the cached wavefunctions and potential to CSV."""
        if self._last_solution is None:
            QMessageBox.warning(self, "No data", "Run the solver first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "wavefunctions.csv",
                                               "CSV Files (*.csv)")
        if not path:
            return
        x, V_eV, energies, states, n_calc = self._last_solution[:5]
        header = "x(m),V(eV)," + ",".join(f"psi_{i + 1}" for i in range(n_calc))
        data = np.column_stack([x, V_eV] + [states[:, i] for i in range(n_calc)])
        np.savetxt(path, data, delimiter=",", header=header, comments="")
        self._log(f"Exported to {path}")


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = QuantumSimWidget()
    w.setWindowTitle("Quantum Mechanics Simulator")
    w.resize(1100, 700)
    w.show()
    sys.exit(app.exec_())
