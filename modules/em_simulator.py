"""
Electromagnetic Simulator Widget for PyQt5 Scientific Suite (FEMM-like).

Provides electrostatics (Poisson/Laplace), magnetostatics (Biot-Savart),
and current flow simulations on configurable 2D grids with interactive
visualization of potential contours, field lines, and magnitude maps.
"""

import os
import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QComboBox,
    QDoubleSpinBox, QSpinBox, QPushButton, QTabWidget, QFormLayout,
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox, QSplitter,
    QMessageBox, QLineEdit, QGridLayout, QFileDialog, QDialog,
    QDialogButtonBox, QPlainTextEdit,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

try:
    from mpl_style import style_figure, style_axes
except ImportError:
    def style_figure(f): pass
    def style_axes(a): pass
import matplotlib.pyplot as plt


def _clean_num(x, tol=1e-10):
    """Clean floating-point noise for display."""
    if isinstance(x, (float,)):
        rounded = round(x)
        if abs(x - rounded) < tol:
            return int(rounded)
        return round(x, 10)
    return x


# ---------------------------------------------------------------------------
# Solver worker thread
# ---------------------------------------------------------------------------
class _SolverThread(QThread):
    """Run the heavy numerical solve off the GUI thread."""
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int)

    def __init__(self, solver_func, params, parent=None):
        super().__init__(parent)
        self.solver_func = solver_func
        self.params = params

    def run(self):
        result = self.solver_func(self.params, progress_cb=self.progress.emit)
        self.finished.emit(result)


# ---------------------------------------------------------------------------
# Core solver routines
# ---------------------------------------------------------------------------

def _solve_electrostatics(params, progress_cb=None):
    """Solve Poisson / Laplace equation via finite-difference SOR on a 2D grid."""
    Nx = params["Nx"]
    Ny = params["Ny"]
    Lx = params["Lx"]
    Ly = params["Ly"]
    eps_r = params["eps_r"]
    omega = params["omega"]
    max_iter = params["max_iter"]
    tol = params["tol"]
    sources = params["sources"]          # list of dicts
    boundaries = params["boundaries"]    # list of dicts
    geometry = params["geometry"]

    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)

    phi = np.zeros((Ny, Nx), dtype=np.float64)
    rho = np.zeros_like(phi)
    fixed = np.zeros((Ny, Nx), dtype=bool)

    eps0 = 8.854187817e-12
    eps_base = eps_r * eps0

    # Multi-dielectric support: eps_map is a 2D array of relative permittivity
    dielectric_regions = params.get("dielectric_regions", [])
    eps_map = np.full((Ny, Nx), eps_base, dtype=np.float64)
    for region in dielectric_regions:
        rx0 = region.get("x0", 0.0)
        ry0 = region.get("y0", 0.0)
        rx1 = region.get("x1", Lx)
        ry1 = region.get("y1", Ly)
        r_eps_r = region.get("eps_r", 1.0)
        ix0 = int(round(rx0 / dx)); ix1 = int(round(rx1 / dx))
        iy0 = int(round(ry0 / dy)); iy1 = int(round(ry1 / dy))
        ix0, ix1 = max(0, ix0), min(Nx - 1, ix1)
        iy0, iy1 = max(0, iy0), min(Ny - 1, iy1)
        eps_map[iy0:iy1+1, ix0:ix1+1] = r_eps_r * eps0

    eps = eps_map  # Now a 2D array

    # --- Apply geometry presets -------------------------------------------------
    if geometry == "Parallel plates":
        plate_v = params.get("plate_voltage", 100.0)
        col_lo = max(int(0.2 * Nx), 1)
        col_hi = min(int(0.8 * Nx), Nx - 2)
        row_lo = max(int(Ny * 0.3), 1)
        row_hi = min(int(Ny * 0.7), Ny - 2)
        phi[row_lo, col_lo:col_hi] = plate_v
        phi[row_hi, col_lo:col_hi] = -plate_v
        fixed[row_lo, col_lo:col_hi] = True
        fixed[row_hi, col_lo:col_hi] = True

    elif geometry == "Coaxial":
        cx, cy = Nx // 2, Ny // 2
        r_inner = min(Nx, Ny) * 0.1
        r_outer = min(Nx, Ny) * 0.4
        v_inner = params.get("plate_voltage", 100.0)
        for j in range(Ny):
            for i in range(Nx):
                r = np.sqrt((i - cx) ** 2 + (j - cy) ** 2)
                if r <= r_inner:
                    phi[j, i] = v_inner
                    fixed[j, i] = True
                elif r >= r_outer:
                    phi[j, i] = 0.0
                    fixed[j, i] = True

    # --- Point / line charges ---------------------------------------------------
    for s in sources:
        kind = s.get("type", "point")
        mag = s.get("magnitude", 1.0)
        sx = s.get("x", Lx / 2)
        sy = s.get("y", Ly / 2)
        ix = int(round(sx / dx))
        iy = int(round(sy / dy))
        ix = np.clip(ix, 0, Nx - 1)
        iy = np.clip(iy, 0, Ny - 1)
        if kind == "point":
            rho[iy, ix] += mag / (dx * dy)
        elif kind == "line":
            length = s.get("length", Ly * 0.4)
            half = int(round(0.5 * length / dy))
            y_lo = max(iy - half, 0)
            y_hi = min(iy + half, Ny - 1)
            rho[y_lo:y_hi, ix] += mag / (dx * (y_hi - y_lo))

    # --- Fixed-potential boundaries ---------------------------------------------
    for b in boundaries:
        side = b.get("side", "all")
        val = b.get("value", 0.0)
        if side in ("all", "left"):
            phi[:, 0] = val; fixed[:, 0] = True
        if side in ("all", "right"):
            phi[:, -1] = val; fixed[:, -1] = True
        if side in ("all", "top"):
            phi[0, :] = val; fixed[0, :] = True
        if side in ("all", "bottom"):
            phi[-1, :] = val; fixed[-1, :] = True

    # Ensure at least grounded boundary when nothing else is specified
    if not np.any(fixed):
        phi[0, :] = 0; phi[-1, :] = 0; phi[:, 0] = 0; phi[:, -1] = 0
        fixed[0, :] = True; fixed[-1, :] = True
        fixed[:, 0] = True; fixed[:, -1] = True

    # --- SOR iteration ----------------------------------------------------------
    coeff = -rho / eps
    dx2 = dx * dx
    dy2 = dy * dy
    denom = 2.0 * (1.0 / dx2 + 1.0 / dy2)

    for it in range(max_iter):
        phi_old = phi.copy()
        for j in range(1, Ny - 1):
            for i in range(1, Nx - 1):
                if fixed[j, i]:
                    continue
                gs = ((phi[j, i - 1] + phi[j, i + 1]) / dx2 +
                      (phi[j - 1, i] + phi[j + 1, i]) / dy2 -
                      coeff[j, i]) / denom
                phi[j, i] += omega * (gs - phi[j, i])
        diff = np.max(np.abs(phi - phi_old))
        if progress_cb and it % 20 == 0:
            progress_cb(int(100 * it / max_iter))
        if diff < tol:
            break

    # --- Derive E field ---------------------------------------------------------
    Ey_field, Ex_field = np.gradient(-phi, dy, dx)
    E_mag = np.sqrt(Ex_field ** 2 + Ey_field ** 2)

    # Energy density  u = 0.5 * eps * |E|^2
    energy_density = 0.5 * eps_map * E_mag ** 2
    total_energy = np.sum(energy_density) * dx * dy

    return {
        "x": x, "y": y, "X": X, "Y": Y,
        "phi": phi, "Ex": Ex_field, "Ey": Ey_field, "E_mag": E_mag,
        "energy_density": energy_density, "total_energy": total_energy,
        "eps_map": eps_map, "fixed": fixed, "rho": rho,
        "dx": dx, "dy": dy,
        "type": "electrostatics",
    }


def _solve_magnetostatics(params, progress_cb=None):
    """Compute B field from current-carrying wires via Biot-Savart (2D cross-section)."""
    Nx = params["Nx"]
    Ny = params["Ny"]
    Lx = params["Lx"]
    Ly = params["Ly"]
    mu_r = params["mu_r"]
    sources = params["sources"]
    geometry = params["geometry"]

    mu0 = 4.0 * np.pi * 1e-7
    mu = mu_r * mu0

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)
    Bx = np.zeros_like(X)
    By = np.zeros_like(X)

    wire_list = []

    # Geometry presets
    if geometry == "Solenoid":
        n_turns = params.get("solenoid_turns", 10)
        current = params.get("solenoid_current", 1.0)
        cx = Lx / 2
        radius = Ly * 0.3
        for k in range(n_turns):
            angle = 2.0 * np.pi * k / n_turns
            wx = cx + radius * np.cos(angle)
            wy = Ly / 2 + radius * np.sin(angle)
            wire_list.append({"x": wx, "y": wy, "I": current})
        # Return path
        for k in range(n_turns):
            angle = 2.0 * np.pi * k / n_turns
            wx = cx + radius * 0.5 * np.cos(angle)
            wy = Ly / 2 + radius * 0.5 * np.sin(angle)
            wire_list.append({"x": wx, "y": wy, "I": -current})
    else:
        for s in sources:
            wire_list.append({
                "x": s.get("x", Lx / 2),
                "y": s.get("y", Ly / 2),
                "I": s.get("magnitude", 1.0),
            })

    # Biot-Savart for infinite straight wires (2D cross-section)
    for idx, w in enumerate(wire_list):
        rx = X - w["x"]
        ry = Y - w["y"]
        r2 = rx ** 2 + ry ** 2
        r2 = np.where(r2 < 1e-20, 1e-20, r2)
        coeff = mu * w["I"] / (2.0 * np.pi * r2)
        Bx += -coeff * ry
        By += coeff * rx
        if progress_cb and idx % max(1, len(wire_list) // 10) == 0:
            progress_cb(int(100 * idx / max(1, len(wire_list))))

    B_mag = np.sqrt(Bx ** 2 + By ** 2)
    # Magnetic energy density  u = B^2 / (2 mu)
    energy_density = B_mag ** 2 / (2.0 * mu)
    total_energy = np.sum(energy_density) * (Lx / (Nx - 1)) * (Ly / (Ny - 1))

    # Vector potential Az (for contour plot)
    Az = np.zeros_like(X)
    for w in wire_list:
        rx = X - w["x"]
        ry = Y - w["y"]
        r2 = rx ** 2 + ry ** 2
        r2 = np.where(r2 < 1e-20, 1e-20, r2)
        Az += -mu * w["I"] / (4.0 * np.pi) * np.log(r2)

    return {
        "x": x, "y": y, "X": X, "Y": Y,
        "Bx": Bx, "By": By, "B_mag": B_mag,
        "Az": Az,
        "energy_density": energy_density, "total_energy": total_energy,
        "wires": wire_list,
        "type": "magnetostatics",
    }


def _solve_current_flow(params, progress_cb=None):
    """Solve steady-state current flow (Laplace equation for voltage) with conductivity."""
    Nx = params["Nx"]
    Ny = params["Ny"]
    Lx = params["Lx"]
    Ly = params["Ly"]
    sigma = params.get("conductivity", 1.0)
    omega = params["omega"]
    max_iter = params["max_iter"]
    tol = params["tol"]
    boundaries = params["boundaries"]

    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)

    phi = np.zeros((Ny, Nx), dtype=np.float64)
    fixed = np.zeros((Ny, Nx), dtype=bool)

    # Default: left = V, right = 0
    if not boundaries:
        boundaries = [
            {"side": "left", "value": 1.0},
            {"side": "right", "value": 0.0},
        ]

    for b in boundaries:
        side = b.get("side", "left")
        val = b.get("value", 0.0)
        if side == "left":
            phi[:, 0] = val; fixed[:, 0] = True
        elif side == "right":
            phi[:, -1] = val; fixed[:, -1] = True
        elif side == "top":
            phi[0, :] = val; fixed[0, :] = True
        elif side == "bottom":
            phi[-1, :] = val; fixed[-1, :] = True

    dx2 = dx * dx
    dy2 = dy * dy
    denom = 2.0 * (1.0 / dx2 + 1.0 / dy2)

    for it in range(max_iter):
        phi_old = phi.copy()
        for j in range(1, Ny - 1):
            for i in range(1, Nx - 1):
                if fixed[j, i]:
                    continue
                gs = ((phi[j, i - 1] + phi[j, i + 1]) / dx2 +
                      (phi[j - 1, i] + phi[j + 1, i]) / dy2) / denom
                phi[j, i] += omega * (gs - phi[j, i])
        diff = np.max(np.abs(phi - phi_old))
        if progress_cb and it % 20 == 0:
            progress_cb(int(100 * it / max_iter))
        if diff < tol:
            break

    Ey_field, Ex_field = np.gradient(-phi, dy, dx)
    Jx = sigma * Ex_field
    Jy = sigma * Ey_field
    J_mag = np.sqrt(Jx ** 2 + Jy ** 2)
    power_density = J_mag ** 2 / sigma
    total_power = np.sum(power_density) * dx * dy

    return {
        "x": x, "y": y, "X": X, "Y": Y,
        "phi": phi, "Ex": Ex_field, "Ey": Ey_field,
        "Jx": Jx, "Jy": Jy, "J_mag": J_mag,
        "power_density": power_density, "total_power": total_power,
        "type": "current_flow",
    }


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class EMSimulatorWidget(QWidget):
    """
    Full-featured electromagnetic simulator widget providing electrostatics,
    magnetostatics, and current-flow analysis with interactive visualisation.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._logger = None
        self._result = None
        self._solver_thread = None
        self._dielectric_regions = []
        self._init_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_logger(self, fn):
        """Attach an external logging callback ``fn(str)``."""
        self._logger = fn

    def run(self):
        """Programmatic entry point -- triggers a solve."""
        self._on_solve()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _init_ui(self):
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # ---- Left panel: controls ------------------------------------------
        ctrl = QWidget()
        ctrl_lay = QVBoxLayout(ctrl)

        # Problem type
        grp_prob = QGroupBox("Problem Type")
        fl_prob = QFormLayout()
        self._combo_problem = QComboBox()
        self._combo_problem.addItems(["Electrostatics", "Magnetostatics", "Current Flow"])
        self._combo_problem.currentIndexChanged.connect(self._on_problem_changed)
        fl_prob.addRow("Type:", self._combo_problem)

        self._combo_geom = QComboBox()
        self._combo_geom.addItems(["Custom", "Parallel plates", "Coaxial", "Solenoid"])
        fl_prob.addRow("Geometry:", self._combo_geom)
        grp_prob.setLayout(fl_prob)
        ctrl_lay.addWidget(grp_prob)

        # Grid & material
        grp_grid = QGroupBox("Grid / Material")
        fl_grid = QFormLayout()

        self._spin_nx = QSpinBox(); self._spin_nx.setRange(20, 500); self._spin_nx.setValue(80)
        self._spin_ny = QSpinBox(); self._spin_ny.setRange(20, 500); self._spin_ny.setValue(80)
        self._spin_lx = QDoubleSpinBox(); self._spin_lx.setRange(0.01, 100); self._spin_lx.setValue(1.0)
        self._spin_ly = QDoubleSpinBox(); self._spin_ly.setRange(0.01, 100); self._spin_ly.setValue(1.0)
        self._spin_epsr = QDoubleSpinBox(); self._spin_epsr.setRange(0.1, 1e4); self._spin_epsr.setValue(1.0)
        self._spin_mur = QDoubleSpinBox(); self._spin_mur.setRange(0.1, 1e6); self._spin_mur.setValue(1.0)
        self._spin_sigma = QDoubleSpinBox(); self._spin_sigma.setRange(1e-10, 1e8); self._spin_sigma.setValue(1.0); self._spin_sigma.setDecimals(4)
        self._spin_omega = QDoubleSpinBox(); self._spin_omega.setRange(1.0, 1.99); self._spin_omega.setValue(1.85); self._spin_omega.setSingleStep(0.05)
        self._spin_maxiter = QSpinBox(); self._spin_maxiter.setRange(100, 100000); self._spin_maxiter.setValue(5000)
        self._spin_tol = QDoubleSpinBox(); self._spin_tol.setDecimals(8); self._spin_tol.setRange(1e-10, 1.0); self._spin_tol.setValue(1e-5)
        self._spin_plate_v = QDoubleSpinBox(); self._spin_plate_v.setRange(-1e6, 1e6); self._spin_plate_v.setValue(100.0)
        self._spin_sol_turns = QSpinBox(); self._spin_sol_turns.setRange(2, 200); self._spin_sol_turns.setValue(12)
        self._spin_sol_I = QDoubleSpinBox(); self._spin_sol_I.setRange(-1e6, 1e6); self._spin_sol_I.setValue(1.0)

        fl_grid.addRow("Nx:", self._spin_nx)
        fl_grid.addRow("Ny:", self._spin_ny)
        fl_grid.addRow("Lx (m):", self._spin_lx)
        fl_grid.addRow("Ly (m):", self._spin_ly)
        fl_grid.addRow("eps_r:", self._spin_epsr)
        fl_grid.addRow("mu_r:", self._spin_mur)
        fl_grid.addRow("sigma (S/m):", self._spin_sigma)
        fl_grid.addRow("SOR omega:", self._spin_omega)
        fl_grid.addRow("Max iterations:", self._spin_maxiter)
        fl_grid.addRow("Tolerance:", self._spin_tol)
        fl_grid.addRow("Plate / inner V:", self._spin_plate_v)
        fl_grid.addRow("Solenoid turns:", self._spin_sol_turns)
        fl_grid.addRow("Solenoid I (A):", self._spin_sol_I)
        grp_grid.setLayout(fl_grid)
        ctrl_lay.addWidget(grp_grid)

        # Source table
        grp_src = QGroupBox("Sources / Charges")
        src_lay = QVBoxLayout()
        self._tbl_sources = QTableWidget(0, 5)
        self._tbl_sources.setHorizontalHeaderLabels(["Type", "X", "Y", "Magnitude", "Length"])
        self._tbl_sources.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        src_lay.addWidget(self._tbl_sources)
        btn_row = QHBoxLayout()
        btn_add = QPushButton("Add"); btn_add.clicked.connect(self._add_source_row)
        btn_del = QPushButton("Remove"); btn_del.clicked.connect(self._del_source_row)
        btn_row.addWidget(btn_add); btn_row.addWidget(btn_del)
        src_lay.addLayout(btn_row)
        grp_src.setLayout(src_lay)
        ctrl_lay.addWidget(grp_src)

        # Boundary table
        grp_bc = QGroupBox("Boundary Conditions")
        bc_lay = QVBoxLayout()
        self._tbl_bc = QTableWidget(0, 2)
        self._tbl_bc.setHorizontalHeaderLabels(["Side", "Value (V)"])
        self._tbl_bc.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        bc_lay.addWidget(self._tbl_bc)
        btn_row2 = QHBoxLayout()
        btn_add2 = QPushButton("Add"); btn_add2.clicked.connect(self._add_bc_row)
        btn_del2 = QPushButton("Remove"); btn_del2.clicked.connect(self._del_bc_row)
        btn_row2.addWidget(btn_add2); btn_row2.addWidget(btn_del2)
        bc_lay.addLayout(btn_row2)
        grp_bc.setLayout(bc_lay)
        ctrl_lay.addWidget(grp_bc)

        # Solve button
        self._btn_solve = QPushButton("Solve")
        self._btn_solve.setStyleSheet("font-weight:bold; padding:6px;")
        self._btn_solve.clicked.connect(self._on_solve)
        ctrl_lay.addWidget(self._btn_solve)

        # --- Generation / Export tools ---
        grp_tools = QGroupBox("Generation / Export")
        tools_lay = QVBoxLayout(grp_tools)

        btn_export_csv = QPushButton("Export Field Data CSV")
        btn_export_csv.clicked.connect(self._on_export_field_csv)
        tools_lay.addWidget(btn_export_csv)

        btn_export_fieldlines = QPushButton("Export Field Lines PNG")
        btn_export_fieldlines.clicked.connect(self._on_export_field_lines_png)
        tools_lay.addWidget(btn_export_fieldlines)

        btn_capacitance = QPushButton("Compute Capacitance")
        btn_capacitance.clicked.connect(self._on_compute_capacitance)
        tools_lay.addWidget(btn_capacitance)

        btn_force = QPushButton("Compute Force Between Charges")
        btn_force.clicked.connect(self._on_compute_force)
        tools_lay.addWidget(btn_force)

        btn_dielectric = QPushButton("Define Dielectric Regions...")
        btn_dielectric.clicked.connect(self._on_define_dielectrics)
        tools_lay.addWidget(btn_dielectric)

        btn_place_charge = QPushButton("Place Charge on Grid...")
        btn_place_charge.clicked.connect(self._on_place_charge)
        tools_lay.addWidget(btn_place_charge)

        ctrl_lay.addWidget(grp_tools)

        ctrl_lay.addStretch()
        splitter.addWidget(ctrl)

        # ---- Right panel: plots + field calculator -------------------------
        right = QWidget()
        right_lay = QVBoxLayout(right)

        self._tabs_plot = QTabWidget()

        # Potential / Az tab
        self._fig_pot = Figure(figsize=(5, 4), dpi=100)
        style_figure(self._fig_pot)
        self._ax_pot = self._fig_pot.add_subplot(111)
        self._canvas_pot = FigureCanvas(self._fig_pot)
        self._toolbar_pot = NavigationToolbar(self._canvas_pot, self)
        w1 = QWidget(); l1 = QVBoxLayout(w1); l1.addWidget(self._toolbar_pot); l1.addWidget(self._canvas_pot)
        self._tabs_plot.addTab(w1, "Potential / Az")

        # Field lines tab
        self._fig_field = Figure(figsize=(5, 4), dpi=100)
        style_figure(self._fig_field)
        self._ax_field = self._fig_field.add_subplot(111)
        self._canvas_field = FigureCanvas(self._fig_field)
        self._toolbar_field = NavigationToolbar(self._canvas_field, self)
        w2 = QWidget(); l2 = QVBoxLayout(w2); l2.addWidget(self._toolbar_field); l2.addWidget(self._canvas_field)
        self._tabs_plot.addTab(w2, "Field Lines")

        # Magnitude tab
        self._fig_mag = Figure(figsize=(5, 4), dpi=100)
        style_figure(self._fig_mag)
        self._ax_mag = self._fig_mag.add_subplot(111)
        self._canvas_mag = FigureCanvas(self._fig_mag)
        self._toolbar_mag = NavigationToolbar(self._canvas_mag, self)
        w3 = QWidget(); l3 = QVBoxLayout(w3); l3.addWidget(self._toolbar_mag); l3.addWidget(self._canvas_mag)
        self._tabs_plot.addTab(w3, "Field Magnitude")

        # Energy tab
        self._fig_energy = Figure(figsize=(5, 4), dpi=100)
        style_figure(self._fig_energy)
        self._ax_energy = self._fig_energy.add_subplot(111)
        self._canvas_energy = FigureCanvas(self._fig_energy)
        self._toolbar_energy = NavigationToolbar(self._canvas_energy, self)
        w4 = QWidget(); l4 = QVBoxLayout(w4); l4.addWidget(self._toolbar_energy); l4.addWidget(self._canvas_energy)
        self._tabs_plot.addTab(w4, "Energy Density")

        right_lay.addWidget(self._tabs_plot, stretch=3)

        # Field calculator
        grp_calc = QGroupBox("Field Calculator")
        calc_lay = QGridLayout()
        calc_lay.addWidget(QLabel("X:"), 0, 0)
        self._calc_x = QDoubleSpinBox(); self._calc_x.setDecimals(4); self._calc_x.setRange(0, 100); self._calc_x.setValue(0.5)
        calc_lay.addWidget(self._calc_x, 0, 1)
        calc_lay.addWidget(QLabel("Y:"), 0, 2)
        self._calc_y = QDoubleSpinBox(); self._calc_y.setDecimals(4); self._calc_y.setRange(0, 100); self._calc_y.setValue(0.5)
        calc_lay.addWidget(self._calc_y, 0, 3)
        btn_calc = QPushButton("Evaluate"); btn_calc.clicked.connect(self._on_evaluate_field)
        calc_lay.addWidget(btn_calc, 0, 4)
        self._lbl_calc = QLabel("No results yet.")
        self._lbl_calc.setWordWrap(True)
        calc_lay.addWidget(self._lbl_calc, 1, 0, 1, 5)
        grp_calc.setLayout(calc_lay)
        right_lay.addWidget(grp_calc)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        # Defaults
        self._add_bc_row_data("all", 0.0)
        self._add_source_row_data("point", 0.5, 0.5, 1e-9, 0.0)

    # ------------------------------------------------------------------
    # Table helpers
    # ------------------------------------------------------------------

    def _add_source_row(self):
        self._add_source_row_data("point", 0.5, 0.5, 1e-9, 0.0)

    def _add_source_row_data(self, stype, x, y, mag, length):
        r = self._tbl_sources.rowCount()
        self._tbl_sources.insertRow(r)
        combo = QComboBox(); combo.addItems(["point", "line"])
        combo.setCurrentText(stype)
        self._tbl_sources.setCellWidget(r, 0, combo)
        self._tbl_sources.setItem(r, 1, QTableWidgetItem(str(x)))
        self._tbl_sources.setItem(r, 2, QTableWidgetItem(str(y)))
        self._tbl_sources.setItem(r, 3, QTableWidgetItem(str(mag)))
        self._tbl_sources.setItem(r, 4, QTableWidgetItem(str(length)))

    def _del_source_row(self):
        row = self._tbl_sources.currentRow()
        if row >= 0:
            self._tbl_sources.removeRow(row)

    def _add_bc_row(self):
        self._add_bc_row_data("all", 0.0)

    def _add_bc_row_data(self, side, val):
        r = self._tbl_bc.rowCount()
        self._tbl_bc.insertRow(r)
        combo = QComboBox(); combo.addItems(["all", "left", "right", "top", "bottom"])
        combo.setCurrentText(side)
        self._tbl_bc.setCellWidget(r, 0, combo)
        self._tbl_bc.setItem(r, 1, QTableWidgetItem(str(val)))

    def _del_bc_row(self):
        row = self._tbl_bc.currentRow()
        if row >= 0:
            self._tbl_bc.removeRow(row)

    # ------------------------------------------------------------------
    # Gather parameters
    # ------------------------------------------------------------------

    def _gather_params(self):
        sources = []
        for r in range(self._tbl_sources.rowCount()):
            combo = self._tbl_sources.cellWidget(r, 0)
            sources.append({
                "type": combo.currentText() if combo else "point",
                "x": float(self._tbl_sources.item(r, 1).text()) if self._tbl_sources.item(r, 1) else 0.5,
                "y": float(self._tbl_sources.item(r, 2).text()) if self._tbl_sources.item(r, 2) else 0.5,
                "magnitude": float(self._tbl_sources.item(r, 3).text()) if self._tbl_sources.item(r, 3) else 1e-9,
                "length": float(self._tbl_sources.item(r, 4).text()) if self._tbl_sources.item(r, 4) else 0.0,
            })

        boundaries = []
        for r in range(self._tbl_bc.rowCount()):
            combo = self._tbl_bc.cellWidget(r, 0)
            boundaries.append({
                "side": combo.currentText() if combo else "all",
                "value": float(self._tbl_bc.item(r, 1).text()) if self._tbl_bc.item(r, 1) else 0.0,
            })

        return {
            "Nx": self._spin_nx.value(),
            "Ny": self._spin_ny.value(),
            "Lx": self._spin_lx.value(),
            "Ly": self._spin_ly.value(),
            "eps_r": self._spin_epsr.value(),
            "mu_r": self._spin_mur.value(),
            "dielectric_regions": list(self._dielectric_regions),
            "conductivity": self._spin_sigma.value(),
            "omega": self._spin_omega.value(),
            "max_iter": self._spin_maxiter.value(),
            "tol": self._spin_tol.value(),
            "plate_voltage": self._spin_plate_v.value(),
            "solenoid_turns": self._spin_sol_turns.value(),
            "solenoid_current": self._spin_sol_I.value(),
            "sources": sources,
            "boundaries": boundaries,
            "geometry": self._combo_geom.currentText(),
        }

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def _log(self, msg):
        if self._logger:
            self._logger(msg)

    def _on_problem_changed(self, idx):
        """Toggle visibility hints when problem type changes."""
        pass  # Could hide/show irrelevant controls

    def _on_solve(self):
        if self._solver_thread and self._solver_thread.isRunning():
            self._log("Solver already running.")
            return

        params = self._gather_params()
        ptype = self._combo_problem.currentText()

        if ptype == "Electrostatics":
            solver = _solve_electrostatics
        elif ptype == "Magnetostatics":
            solver = _solve_magnetostatics
        else:
            solver = _solve_current_flow

        self._log(f"Starting {ptype} solve  (grid {params['Nx']}x{params['Ny']}) ...")
        self._btn_solve.setEnabled(False)
        self._btn_solve.setText("Solving ...")

        self._solver_thread = _SolverThread(solver, params)
        self._solver_thread.finished.connect(self._on_solve_done)
        self._solver_thread.start()

    def _on_solve_done(self, result):
        self._result = result
        self._btn_solve.setEnabled(True)
        self._btn_solve.setText("Solve")

        rtype = result["type"]
        self._log(f"Solve complete ({rtype}).")

        if rtype == "electrostatics":
            self._plot_electrostatics(result)
        elif rtype == "magnetostatics":
            self._plot_magnetostatics(result)
        else:
            self._plot_current_flow(result)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot_electrostatics(self, r):
        X, Y, phi = r["X"], r["Y"], r["phi"]
        Ex, Ey, E_mag = r["Ex"], r["Ey"], r["E_mag"]
        ed = r["energy_density"]

        # Potential contours
        ax = self._ax_pot; ax.clear()
        cs = ax.contourf(X, Y, phi, levels=40, cmap="RdBu_r")
        ax.contour(X, Y, phi, levels=20, colors="k", linewidths=0.3)
        ax.set_title("Electric Potential (V)")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
        self._fig_pot.colorbar(cs, ax=ax, label="V")
        self._canvas_pot.draw()

        # Field lines (streamlines)
        ax2 = self._ax_field; ax2.clear()
        speed = E_mag / (E_mag.max() + 1e-30)
        ax2.streamplot(r["x"], r["y"], Ex, Ey, color=speed, cmap="inferno",
                       linewidth=1, density=1.6, arrowsize=1.2)
        ax2.set_title("Electric Field Lines")
        ax2.set_xlabel("x (m)"); ax2.set_ylabel("y (m)")
        ax2.set_aspect("equal")
        self._canvas_field.draw()

        # Magnitude
        ax3 = self._ax_mag; ax3.clear()
        em = ax3.pcolormesh(X, Y, E_mag, cmap="hot", shading="auto")
        ax3.set_title("|E| (V/m)")
        ax3.set_xlabel("x (m)"); ax3.set_ylabel("y (m)")
        ax3.set_aspect("equal")
        self._fig_mag.colorbar(em, ax=ax3, label="V/m")
        self._canvas_mag.draw()

        # Energy density
        ax4 = self._ax_energy; ax4.clear()
        en = ax4.pcolormesh(X, Y, ed, cmap="magma", shading="auto")
        ax4.set_title(f"Energy Density  (total={r['total_energy']:.4e} J)")
        ax4.set_xlabel("x (m)"); ax4.set_ylabel("y (m)")
        ax4.set_aspect("equal")
        self._fig_energy.colorbar(en, ax=ax4, label="J/m^3")
        self._canvas_energy.draw()

    def _plot_magnetostatics(self, r):
        X, Y = r["X"], r["Y"]
        Bx, By, B_mag = r["Bx"], r["By"], r["B_mag"]
        Az, ed = r["Az"], r["energy_density"]

        ax = self._ax_pot; ax.clear()
        cs = ax.contourf(X, Y, Az, levels=40, cmap="coolwarm")
        ax.contour(X, Y, Az, levels=20, colors="k", linewidths=0.3)
        for w in r["wires"]:
            marker = "o" if w["I"] > 0 else "x"
            ax.plot(w["x"], w["y"], marker, color="lime", markersize=5)
        ax.set_title("Vector Potential Az")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
        self._fig_pot.colorbar(cs, ax=ax, label="Wb/m")
        self._canvas_pot.draw()

        ax2 = self._ax_field; ax2.clear()
        speed = B_mag / (B_mag.max() + 1e-30)
        ax2.streamplot(r["x"], r["y"], Bx, By, color=speed, cmap="plasma",
                       linewidth=1, density=1.6, arrowsize=1.2)
        for w in r["wires"]:
            marker = "o" if w["I"] > 0 else "x"
            ax2.plot(w["x"], w["y"], marker, color="lime", markersize=5)
        ax2.set_title("Magnetic Field Lines")
        ax2.set_xlabel("x (m)"); ax2.set_ylabel("y (m)")
        ax2.set_aspect("equal")
        self._canvas_field.draw()

        ax3 = self._ax_mag; ax3.clear()
        em = ax3.pcolormesh(X, Y, B_mag, cmap="hot", shading="auto")
        ax3.set_title("|B| (T)")
        ax3.set_xlabel("x (m)"); ax3.set_ylabel("y (m)")
        ax3.set_aspect("equal")
        self._fig_mag.colorbar(em, ax=ax3, label="T")
        self._canvas_mag.draw()

        ax4 = self._ax_energy; ax4.clear()
        en = ax4.pcolormesh(X, Y, ed, cmap="magma", shading="auto")
        ax4.set_title(f"Magnetic Energy Density  (total={r['total_energy']:.4e} J/m)")
        ax4.set_xlabel("x (m)"); ax4.set_ylabel("y (m)")
        ax4.set_aspect("equal")
        self._fig_energy.colorbar(en, ax=ax4, label="J/m^3")
        self._canvas_energy.draw()

    def _plot_current_flow(self, r):
        X, Y, phi = r["X"], r["Y"], r["phi"]
        Jx, Jy, J_mag = r["Jx"], r["Jy"], r["J_mag"]
        pd = r["power_density"]

        ax = self._ax_pot; ax.clear()
        cs = ax.contourf(X, Y, phi, levels=40, cmap="RdBu_r")
        ax.contour(X, Y, phi, levels=20, colors="k", linewidths=0.3)
        ax.set_title("Voltage (V)")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
        self._fig_pot.colorbar(cs, ax=ax, label="V")
        self._canvas_pot.draw()

        ax2 = self._ax_field; ax2.clear()
        speed = J_mag / (J_mag.max() + 1e-30)
        ax2.streamplot(r["x"], r["y"], Jx, Jy, color=speed, cmap="viridis",
                       linewidth=1, density=1.6, arrowsize=1.2)
        ax2.set_title("Current Density Streamlines")
        ax2.set_xlabel("x (m)"); ax2.set_ylabel("y (m)")
        ax2.set_aspect("equal")
        self._canvas_field.draw()

        ax3 = self._ax_mag; ax3.clear()
        em = ax3.pcolormesh(X, Y, J_mag, cmap="hot", shading="auto")
        ax3.set_title("|J| (A/m^2)")
        ax3.set_xlabel("x (m)"); ax3.set_ylabel("y (m)")
        ax3.set_aspect("equal")
        self._fig_mag.colorbar(em, ax=ax3, label="A/m^2")
        self._canvas_mag.draw()

        ax4 = self._ax_energy; ax4.clear()
        en = ax4.pcolormesh(X, Y, pd, cmap="magma", shading="auto")
        ax4.set_title(f"Power Dissipation  (total={r['total_power']:.4e} W/m)")
        ax4.set_xlabel("x (m)"); ax4.set_ylabel("y (m)")
        ax4.set_aspect("equal")
        self._fig_energy.colorbar(en, ax=ax4, label="W/m^3")
        self._canvas_energy.draw()

    # ------------------------------------------------------------------
    # Field calculator
    # ------------------------------------------------------------------

    def _on_evaluate_field(self):
        if self._result is None:
            self._lbl_calc.setText("Run a simulation first.")
            return

        r = self._result
        px = self._calc_x.value()
        py = self._calc_y.value()
        x, y = r["x"], r["y"]

        if px < x[0] or px > x[-1] or py < y[0] or py > y[-1]:
            self._lbl_calc.setText("Point outside domain.")
            return

        rtype = r["type"]
        lines = [f"Point ({px:.4f}, {py:.4f}):"]

        def _interp(field):
            interp = RegularGridInterpolator((y, x), field, method="linear", bounds_error=False, fill_value=0.0)
            return float(interp([[py, px]]))

        if rtype == "electrostatics":
            phi_val = _interp(r["phi"])
            ex_val = _interp(r["Ex"])
            ey_val = _interp(r["Ey"])
            e_val = np.sqrt(ex_val ** 2 + ey_val ** 2)
            u_val = _interp(r["energy_density"])
            lines.append(f"  Potential  V = {phi_val:.6e} V")
            lines.append(f"  Ex = {ex_val:.6e},  Ey = {ey_val:.6e} V/m")
            lines.append(f"  |E| = {e_val:.6e} V/m")
            lines.append(f"  Energy density = {u_val:.6e} J/m^3")

        elif rtype == "magnetostatics":
            bx_val = _interp(r["Bx"])
            by_val = _interp(r["By"])
            b_val = np.sqrt(bx_val ** 2 + by_val ** 2)
            az_val = _interp(r["Az"])
            u_val = _interp(r["energy_density"])
            lines.append(f"  Bx = {bx_val:.6e},  By = {by_val:.6e} T")
            lines.append(f"  |B| = {b_val:.6e} T")
            lines.append(f"  Az = {az_val:.6e} Wb/m")
            lines.append(f"  Energy density = {u_val:.6e} J/m^3")

        else:
            phi_val = _interp(r["phi"])
            jx_val = _interp(r["Jx"])
            jy_val = _interp(r["Jy"])
            j_val = np.sqrt(jx_val ** 2 + jy_val ** 2)
            p_val = _interp(r["power_density"])
            lines.append(f"  Voltage = {phi_val:.6e} V")
            lines.append(f"  Jx = {jx_val:.6e},  Jy = {jy_val:.6e} A/m^2")
            lines.append(f"  |J| = {j_val:.6e} A/m^2")
            lines.append(f"  Power density = {p_val:.6e} W/m^3")

        self._lbl_calc.setText("\n".join(lines))
        self._log("\n".join(lines))

    # ------------------------------------------------------------------
    # Export field data as CSV
    # ------------------------------------------------------------------

    def _on_export_field_csv(self):
        """Export field data as CSV with columns x, y, Ex, Ey, V (or Bx, By, etc.)."""
        if self._result is None:
            QMessageBox.warning(self, "No Data", "Run a simulation first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Field Data CSV", "em_field_data.csv",
            "CSV files (*.csv);;All Files (*)")
        if not path:
            return

        r = self._result
        X, Y = r["X"], r["Y"]
        Ny, Nx = X.shape
        rtype = r["type"]

        with open(path, 'w') as f:
            if rtype == "electrostatics":
                f.write("x,y,Ex,Ey,V,E_mag,energy_density\n")
                for j in range(Ny):
                    for i in range(Nx):
                        f.write(f"{X[j,i]},{Y[j,i]},{r['Ex'][j,i]},{r['Ey'][j,i]},"
                                f"{r['phi'][j,i]},{r['E_mag'][j,i]},{r['energy_density'][j,i]}\n")
            elif rtype == "magnetostatics":
                f.write("x,y,Bx,By,B_mag,Az,energy_density\n")
                for j in range(Ny):
                    for i in range(Nx):
                        f.write(f"{X[j,i]},{Y[j,i]},{r['Bx'][j,i]},{r['By'][j,i]},"
                                f"{r['B_mag'][j,i]},{r['Az'][j,i]},{r['energy_density'][j,i]}\n")
            else:  # current_flow
                f.write("x,y,Jx,Jy,J_mag,V,power_density\n")
                for j in range(Ny):
                    for i in range(Nx):
                        f.write(f"{X[j,i]},{Y[j,i]},{r['Jx'][j,i]},{r['Jy'][j,i]},"
                                f"{r['J_mag'][j,i]},{r['phi'][j,i]},{r['power_density'][j,i]}\n")

        self._log(f"Field data CSV exported: {path}")

    # ------------------------------------------------------------------
    # Publication-quality field line plot
    # ------------------------------------------------------------------

    def _on_export_field_lines_png(self):
        """Generate a publication-quality field line plot as PNG."""
        if self._result is None:
            QMessageBox.warning(self, "No Data", "Run a simulation first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Field Lines PNG", "field_lines.png",
            "PNG files (*.png);;All Files (*)")
        if not path:
            return

        r = self._result
        rtype = r["type"]
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

        if rtype == "electrostatics":
            E_mag = r["E_mag"]
            speed = E_mag / (E_mag.max() + 1e-30)
            strm = ax.streamplot(r["x"], r["y"], r["Ex"], r["Ey"],
                                  color=speed, cmap="inferno",
                                  linewidth=1.2, density=2.0, arrowsize=1.5)
            fig.colorbar(strm.lines, ax=ax, label="|E| (normalized)", shrink=0.85)
            ax.set_title("Electric Field Lines", fontsize=14)

            # Overlay equipotential contours
            ax.contour(r["X"], r["Y"], r["phi"], levels=20,
                       colors='gray', linewidths=0.4, alpha=0.5)

        elif rtype == "magnetostatics":
            B_mag = r["B_mag"]
            speed = B_mag / (B_mag.max() + 1e-30)
            strm = ax.streamplot(r["x"], r["y"], r["Bx"], r["By"],
                                  color=speed, cmap="plasma",
                                  linewidth=1.2, density=2.0, arrowsize=1.5)
            fig.colorbar(strm.lines, ax=ax, label="|B| (normalized)", shrink=0.85)
            for w in r.get("wires", []):
                marker = "o" if w["I"] > 0 else "x"
                ax.plot(w["x"], w["y"], marker, color="lime", markersize=6)
            ax.set_title("Magnetic Field Lines", fontsize=14)
        else:
            J_mag = r["J_mag"]
            speed = J_mag / (J_mag.max() + 1e-30)
            strm = ax.streamplot(r["x"], r["y"], r["Jx"], r["Jy"],
                                  color=speed, cmap="viridis",
                                  linewidth=1.2, density=2.0, arrowsize=1.5)
            fig.colorbar(strm.lines, ax=ax, label="|J| (normalized)", shrink=0.85)
            ax.set_title("Current Density Field Lines", fontsize=14)

        ax.set_xlabel("x (m)", fontsize=12)
        ax.set_ylabel("y (m)", fontsize=12)
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        self._log(f"Field lines PNG exported: {path}")

    # ------------------------------------------------------------------
    # Capacitance calculation
    # ------------------------------------------------------------------

    def _on_compute_capacitance(self):
        """Compute capacitance between conductors using stored energy: C = 2*W / V^2."""
        if self._result is None or self._result["type"] != "electrostatics":
            QMessageBox.warning(self, "No Data",
                                "Run an electrostatics simulation first.")
            return

        r = self._result
        total_energy = r["total_energy"]
        phi = r["phi"]
        fixed = r.get("fixed")

        # Find conductor voltages
        if fixed is not None:
            conductor_voltages = set()
            for j in range(phi.shape[0]):
                for i in range(phi.shape[1]):
                    if fixed[j, i]:
                        v = round(phi[j, i], 4)
                        conductor_voltages.add(v)
            conductor_voltages = sorted(conductor_voltages)
        else:
            conductor_voltages = [phi.min(), phi.max()]

        if len(conductor_voltages) < 2:
            QMessageBox.warning(self, "Insufficient Conductors",
                                "Need at least two different conductor voltages to compute capacitance.")
            return

        V_diff = abs(conductor_voltages[-1] - conductor_voltages[0])
        if V_diff < 1e-30:
            QMessageBox.warning(self, "Zero Voltage", "Voltage difference is zero.")
            return

        # C = 2 * W / V^2  (per unit depth for 2D)
        capacitance = 2.0 * total_energy / (V_diff ** 2)

        msg = (f"Conductor voltages: {[_clean_num(v) for v in conductor_voltages]}\n"
               f"Voltage difference: {V_diff:.4f} V\n"
               f"Total stored energy: {total_energy:.6e} J/m\n"
               f"Capacitance (per unit depth): {capacitance:.6e} F/m\n"
               f"  = {capacitance * 1e12:.4f} pF/m")

        self._log(msg)
        QMessageBox.information(self, "Capacitance Result", msg)

    # ------------------------------------------------------------------
    # Force calculation between charges
    # ------------------------------------------------------------------

    def _on_compute_force(self):
        """Calculate electrostatic force between point charges using Coulomb's law."""
        if self._result is None or self._result["type"] != "electrostatics":
            QMessageBox.warning(self, "No Data",
                                "Run an electrostatics simulation first.")
            return

        # Gather source charges from the table
        charges = []
        for row in range(self._tbl_sources.rowCount()):
            combo = self._tbl_sources.cellWidget(row, 0)
            stype = combo.currentText() if combo else "point"
            if stype != "point":
                continue
            x = float(self._tbl_sources.item(row, 1).text()) if self._tbl_sources.item(row, 1) else 0.5
            y = float(self._tbl_sources.item(row, 2).text()) if self._tbl_sources.item(row, 2) else 0.5
            q = float(self._tbl_sources.item(row, 3).text()) if self._tbl_sources.item(row, 3) else 1e-9
            charges.append({'x': x, 'y': y, 'q': q})

        if len(charges) < 2:
            QMessageBox.warning(self, "Insufficient Charges",
                                "Need at least 2 point charges to compute force.")
            return

        eps0 = 8.854187817e-12
        k_e = 1.0 / (4.0 * np.pi * eps0)

        lines = ["Coulomb Forces between point charges:\n"]
        for i in range(len(charges)):
            Fx_total, Fy_total = 0.0, 0.0
            for j in range(len(charges)):
                if i == j:
                    continue
                dx = charges[i]['x'] - charges[j]['x']
                dy = charges[i]['y'] - charges[j]['y']
                r = np.sqrt(dx**2 + dy**2)
                if r < 1e-15:
                    continue
                F_mag = k_e * charges[i]['q'] * charges[j]['q'] / (r ** 2)
                Fx_total += F_mag * dx / r
                Fy_total += F_mag * dy / r

            F_total = np.sqrt(Fx_total**2 + Fy_total**2)
            lines.append(f"  Charge {i+1} (q={charges[i]['q']:.3e} C at ({charges[i]['x']:.3f}, {charges[i]['y']:.3f})):")
            lines.append(f"    Fx = {Fx_total:.6e} N, Fy = {Fy_total:.6e} N, |F| = {F_total:.6e} N")

        msg = "\n".join(lines)
        self._log(msg)
        QMessageBox.information(self, "Force Calculation", msg)

    # ------------------------------------------------------------------
    # Define dielectric regions
    # ------------------------------------------------------------------

    def _on_define_dielectrics(self):
        """Open dialog to define rectangular regions with different permittivity."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Define Dielectric Regions")
        dlg.resize(450, 300)
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(
            "Define dielectric regions as lines: x0, y0, x1, y1, eps_r\n"
            "Each line defines a rectangular region with given relative permittivity.\n"
            "Example:\n  0.0, 0.0, 0.5, 1.0, 4.0\n  0.5, 0.0, 1.0, 1.0, 2.0"))

        txt = QPlainTextEdit()
        # Pre-fill with existing regions
        existing = []
        for r in self._dielectric_regions:
            existing.append(f"{r['x0']}, {r['y0']}, {r['x1']}, {r['y1']}, {r['eps_r']}")
        txt.setPlainText("\n".join(existing) if existing else "0.0, 0.0, 0.5, 1.0, 4.0")
        lay.addWidget(txt)

        bbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bbox.accepted.connect(dlg.accept)
        bbox.rejected.connect(dlg.reject)
        lay.addWidget(bbox)

        if dlg.exec_() != QDialog.Accepted:
            return

        regions = []
        for line in txt.toPlainText().strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.replace(';', ',').split(',')]
            if len(parts) >= 5:
                regions.append({
                    'x0': float(parts[0]), 'y0': float(parts[1]),
                    'x1': float(parts[2]), 'y1': float(parts[3]),
                    'eps_r': float(parts[4]),
                })

        self._dielectric_regions = regions
        self._log(f"Defined {len(regions)} dielectric region(s).")

    # ------------------------------------------------------------------
    # Place charge on grid (interactive-like via dialog)
    # ------------------------------------------------------------------

    def _on_place_charge(self):
        """Add a point charge at specified grid coordinates."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Place Charge on Grid")
        dlg.resize(300, 150)
        flay = QFormLayout(dlg)

        spin_x = QDoubleSpinBox(); spin_x.setDecimals(4)
        spin_x.setRange(0, self._spin_lx.value()); spin_x.setValue(self._spin_lx.value() / 2)
        spin_y = QDoubleSpinBox(); spin_y.setDecimals(4)
        spin_y.setRange(0, self._spin_ly.value()); spin_y.setValue(self._spin_ly.value() / 2)
        spin_q = QDoubleSpinBox(); spin_q.setDecimals(12)
        spin_q.setRange(-1e6, 1e6); spin_q.setValue(1e-9)

        flay.addRow("X:", spin_x)
        flay.addRow("Y:", spin_y)
        flay.addRow("Charge (C):", spin_q)

        bbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bbox.accepted.connect(dlg.accept)
        bbox.rejected.connect(dlg.reject)
        flay.addRow(bbox)

        if dlg.exec_() != QDialog.Accepted:
            return

        x = spin_x.value()
        y = spin_y.value()
        q = spin_q.value()

        # Add to the sources table
        self._add_source_row_data("point", x, y, q, 0.0)
        self._log(f"Placed charge: q={q:.3e} C at ({x:.4f}, {y:.4f})")
