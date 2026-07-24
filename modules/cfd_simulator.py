"""
CFD Simulator Widget
====================
A fully functional Computational Fluid Dynamics simulator for 2D incompressible
Navier-Stokes equations using the finite difference method with the projection
(fractional step) method for pressure-velocity coupling.

Supported problem types:
  - Lid-driven cavity
  - Channel flow
  - Flow around obstacle
  - Natural convection

Visualization includes velocity magnitude contours, streamlines, pressure fields,
velocity vector plots, convergence history, and time-stepping animation.
"""

import os
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import time
import traceback

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton,
    QProgressBar, QTabWidget, QSplitter, QCheckBox, QTextEdit,
    QSizePolicy, QFrame, QFileDialog, QMessageBox, QDialog,
    QFormLayout, QDialogButtonBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation

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
# Navier-Stokes Solver Core
# ---------------------------------------------------------------------------

class NavierStokesSolver:
    """2D incompressible Navier-Stokes solver using the projection method.

    The algorithm (Chorin's projection / fractional step):
        1. Compute an intermediate velocity field (ignoring pressure).
        2. Solve a pressure Poisson equation to enforce incompressibility.
        3. Correct the velocity field with the pressure gradient.

    Spatial discretisation: second-order central differences on a staggered grid.
    Time integration: explicit Euler (first-order) for the advection-diffusion step.
    """

    PROBLEM_LID_CAVITY = "Lid-driven cavity"
    PROBLEM_CHANNEL = "Channel flow"
    PROBLEM_OBSTACLE = "Flow around obstacle"
    PROBLEM_CONVECTION = "Natural convection"

    INLET_UNIFORM = "Uniform"
    INLET_PARABOLIC = "Parabolic"
    INLET_PULSATILE = "Pulsatile (sinusoidal)"

    def __init__(self, problem_type=PROBLEM_LID_CAVITY, nx=41, ny=41,
                 re=100.0, dt=0.001, max_iter=500, tol=1e-6,
                 inlet_profile=None):
        self.problem_type = problem_type
        self.nx = nx
        self.ny = ny
        self.re = re
        self.dt = dt
        self.max_iter = max_iter
        self.tol = tol
        self.inlet_profile = inlet_profile or self.INLET_UNIFORM

        # Domain length
        self.lx = 1.0
        self.ly = 1.0
        self.dx = self.lx / (self.nx - 1)
        self.dy = self.ly / (self.ny - 1)

        # Fields
        self.u = np.zeros((ny, nx))   # x-velocity
        self.v = np.zeros((ny, nx))   # y-velocity
        self.p = np.zeros((ny, nx))   # pressure

        # For natural convection
        self.T = np.zeros((ny, nx))   # temperature

        # Obstacle mask (True where obstacle exists)
        self.obstacle = np.zeros((ny, nx), dtype=bool)

        # Snapshot history for animation export
        self._snapshots = []

        # Convergence history
        self.residuals = []
        self.iteration = 0
        self._stop = False

        self._setup_problem()

    def _inlet_velocity(self):
        """Compute the inlet velocity profile based on the current setting."""
        y = np.linspace(0, self.ly, self.ny)
        if self.inlet_profile == self.INLET_PARABOLIC:
            return 4.0 * y * (self.ly - y) / (self.ly ** 2)
        elif self.inlet_profile == self.INLET_PULSATILE:
            base = 4.0 * y * (self.ly - y) / (self.ly ** 2)
            phase = np.sin(2.0 * np.pi * self.iteration * self.dt / 1.0)  # period=1s
            return base * (1.0 + 0.5 * phase)
        else:  # Uniform
            return np.ones(self.ny)

    def _setup_problem(self):
        """Initialise fields and boundary conditions for the selected problem."""
        self.u[:] = 0.0
        self.v[:] = 0.0
        self.p[:] = 0.0
        self.T[:] = 0.0
        self.obstacle[:] = False

        if self.problem_type == self.PROBLEM_LID_CAVITY:
            # Top wall moves to the right with unit velocity
            self.u[-1, :] = 1.0

        elif self.problem_type == self.PROBLEM_CHANNEL:
            # Inlet velocity based on selected profile
            self.u[:, 0] = self._inlet_velocity()

        elif self.problem_type == self.PROBLEM_OBSTACLE:
            # Inlet velocity based on selected profile + circular obstacle
            self.u[:, 0] = self._inlet_velocity()
            cx, cy = self.nx // 4, self.ny // 2
            radius = min(self.nx, self.ny) // 10
            for j in range(self.ny):
                for i in range(self.nx):
                    if (i - cx) ** 2 + (j - cy) ** 2 <= radius ** 2:
                        self.obstacle[j, i] = True
            self.u[self.obstacle] = 0.0
            self.v[self.obstacle] = 0.0

        elif self.problem_type == self.PROBLEM_CONVECTION:
            # Hot left wall, cold right wall
            self.T[:, 0] = 1.0
            self.T[:, -1] = 0.0
            y = np.linspace(0, 1, self.ny)
            for i in range(self.nx):
                self.T[:, i] = 1.0 - i / (self.nx - 1)

    def _apply_bc(self):
        """Apply boundary conditions after each time step."""
        if self.problem_type == self.PROBLEM_LID_CAVITY:
            # No-slip walls
            self.u[0, :] = 0.0
            self.u[-1, :] = 1.0   # lid velocity
            self.u[:, 0] = 0.0
            self.u[:, -1] = 0.0
            self.v[0, :] = 0.0
            self.v[-1, :] = 0.0
            self.v[:, 0] = 0.0
            self.v[:, -1] = 0.0

        elif self.problem_type == self.PROBLEM_CHANNEL:
            self.u[:, 0] = self._inlet_velocity()
            self.v[:, 0] = 0.0
            # Outlet: zero-gradient (Neumann)
            self.u[:, -1] = self.u[:, -2]
            self.v[:, -1] = self.v[:, -2]
            # Walls (top / bottom): no-slip
            self.u[0, :] = 0.0
            self.u[-1, :] = 0.0
            self.v[0, :] = 0.0
            self.v[-1, :] = 0.0
            # Outlet pressure reference
            self.p[:, -1] = 0.0

        elif self.problem_type == self.PROBLEM_OBSTACLE:
            self.u[:, 0] = self._inlet_velocity()
            self.v[:, 0] = 0.0
            self.u[:, -1] = self.u[:, -2]
            self.v[:, -1] = self.v[:, -2]
            self.u[0, :] = 0.0
            self.u[-1, :] = 0.0
            self.v[0, :] = 0.0
            self.v[-1, :] = 0.0
            self.u[self.obstacle] = 0.0
            self.v[self.obstacle] = 0.0
            self.p[:, -1] = 0.0

        elif self.problem_type == self.PROBLEM_CONVECTION:
            self.u[0, :] = 0.0
            self.u[-1, :] = 0.0
            self.u[:, 0] = 0.0
            self.u[:, -1] = 0.0
            self.v[0, :] = 0.0
            self.v[-1, :] = 0.0
            self.v[:, 0] = 0.0
            self.v[:, -1] = 0.0
            self.T[:, 0] = 1.0
            self.T[:, -1] = 0.0
            self.T[0, :] = self.T[1, :]    # insulated
            self.T[-1, :] = self.T[-2, :]  # insulated

    def _build_pressure_poisson_rhs(self, u_star, v_star):
        """Build the RHS of the pressure Poisson equation: div(u*)/dt."""
        rhs = np.zeros((self.ny, self.nx))
        dx, dy, dt = self.dx, self.dy, self.dt
        rhs[1:-1, 1:-1] = (1.0 / dt) * (
            (u_star[1:-1, 2:] - u_star[1:-1, :-2]) / (2.0 * dx) +
            (v_star[2:, 1:-1] - v_star[:-2, 1:-1]) / (2.0 * dy)
        )
        return rhs

    def _solve_pressure_poisson(self, rhs, n_sub=50):
        """Solve the pressure Poisson equation using iterative Jacobi relaxation."""
        p = self.p.copy()
        dx2 = self.dx ** 2
        dy2 = self.dy ** 2
        denom = 2.0 * (1.0 / dx2 + 1.0 / dy2)

        for _ in range(n_sub):
            pn = p.copy()
            p[1:-1, 1:-1] = (
                (pn[1:-1, 2:] + pn[1:-1, :-2]) / dx2 +
                (pn[2:, 1:-1] + pn[:-2, 1:-1]) / dy2 -
                rhs[1:-1, 1:-1]
            ) / denom

            # Pressure boundary conditions
            p[:, 0] = p[:, 1]     # left: dp/dx = 0
            p[:, -1] = p[:, -2]   # right: dp/dx = 0 (or p=0 for outlet)
            p[0, :] = p[1, :]     # bottom: dp/dy = 0
            p[-1, :] = p[-2, :]   # top: dp/dy = 0

            if self.problem_type in (self.PROBLEM_CHANNEL, self.PROBLEM_OBSTACLE):
                p[:, -1] = 0.0  # outlet pressure reference

        return p

    def step(self):
        """Advance the solution by one time step using the projection method."""
        u, v, p = self.u, self.v, self.p
        nx, ny = self.nx, self.ny
        dx, dy, dt = self.dx, self.dy, self.dt
        nu = 1.0 / self.re  # kinematic viscosity

        # --- Step 1: Compute intermediate velocity (u*, v*) ---
        u_star = u.copy()
        v_star = v.copy()

        # Advection + diffusion for u
        u_star[1:-1, 1:-1] = u[1:-1, 1:-1] + dt * (
            # Diffusion
            nu * (
                (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dx**2 +
                (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dy**2
            )
            # Advection (central difference)
            - u[1:-1, 1:-1] * (u[1:-1, 2:] - u[1:-1, :-2]) / (2*dx)
            - v[1:-1, 1:-1] * (u[2:, 1:-1] - u[:-2, 1:-1]) / (2*dy)
        )

        # Advection + diffusion for v
        v_star[1:-1, 1:-1] = v[1:-1, 1:-1] + dt * (
            nu * (
                (v[1:-1, 2:] - 2*v[1:-1, 1:-1] + v[1:-1, :-2]) / dx**2 +
                (v[2:, 1:-1] - 2*v[1:-1, 1:-1] + v[:-2, 1:-1]) / dy**2
            )
            - u[1:-1, 1:-1] * (v[1:-1, 2:] - v[1:-1, :-2]) / (2*dx)
            - v[1:-1, 1:-1] * (v[2:, 1:-1] - v[:-2, 1:-1]) / (2*dy)
        )

        # Buoyancy for natural convection (Boussinesq approximation)
        if self.problem_type == self.PROBLEM_CONVECTION:
            ra = self.re  # reinterpret Re as Rayleigh number for convection
            pr = 0.71     # Prandtl number for air
            gr = ra * pr  # Grashof-like scaling
            v_star[1:-1, 1:-1] += dt * gr * self.T[1:-1, 1:-1]

        # --- Step 2: Solve pressure Poisson equation ---
        rhs = self._build_pressure_poisson_rhs(u_star, v_star)
        self.p = self._solve_pressure_poisson(rhs, n_sub=80)

        # --- Step 3: Correct velocity with pressure gradient ---
        self.u[1:-1, 1:-1] = u_star[1:-1, 1:-1] - dt * (
            (self.p[1:-1, 2:] - self.p[1:-1, :-2]) / (2*dx)
        )
        self.v[1:-1, 1:-1] = v_star[1:-1, 1:-1] - dt * (
            (self.p[2:, 1:-1] - self.p[:-2, 1:-1]) / (2*dy)
        )

        # --- Update temperature for natural convection ---
        if self.problem_type == self.PROBLEM_CONVECTION:
            pr = 0.71
            alpha = nu / pr
            T = self.T
            self.T[1:-1, 1:-1] = T[1:-1, 1:-1] + dt * (
                alpha * (
                    (T[1:-1, 2:] - 2*T[1:-1, 1:-1] + T[1:-1, :-2]) / dx**2 +
                    (T[2:, 1:-1] - 2*T[1:-1, 1:-1] + T[:-2, 1:-1]) / dy**2
                )
                - self.u[1:-1, 1:-1] * (T[1:-1, 2:] - T[1:-1, :-2]) / (2*dx)
                - self.v[1:-1, 1:-1] * (T[2:, 1:-1] - T[:-2, 1:-1]) / (2*dy)
            )

        # Apply boundary conditions
        self._apply_bc()

        # Compute residual (divergence of velocity ~ should be zero)
        div = np.abs(
            (self.u[1:-1, 2:] - self.u[1:-1, :-2]) / (2*dx) +
            (self.v[2:, 1:-1] - self.v[:-2, 1:-1]) / (2*dy)
        ).max()

        self.iteration += 1
        self.residuals.append(div)
        return div

    def run(self, callback=None):
        """Run the solver for max_iter steps or until convergence."""
        self._stop = False
        for n in range(self.max_iter):
            if self._stop:
                break
            residual = self.step()
            if callback:
                callback(n, residual)
            if residual < self.tol and n > 10:
                break
        return self.u, self.v, self.p

    def stop(self):
        self._stop = True

    def reset(self):
        self.u[:] = 0.0
        self.v[:] = 0.0
        self.p[:] = 0.0
        self.T[:] = 0.0
        self.obstacle[:] = False
        self.residuals = []
        self.iteration = 0
        self._stop = False
        self._setup_problem()

    def velocity_magnitude(self):
        return np.sqrt(self.u ** 2 + self.v ** 2)

    def vorticity(self):
        dvdx = np.gradient(self.v, self.dx, axis=1)
        dudy = np.gradient(self.u, self.dy, axis=0)
        return dvdx - dudy

    def stream_function(self):
        """Compute stream function by integrating u along y."""
        psi = np.zeros_like(self.u)
        for j in range(1, self.ny):
            psi[j, :] = psi[j-1, :] + self.u[j, :] * self.dy
        return psi

    def save_snapshot(self):
        """Save the current velocity and pressure fields for later animation export."""
        self._snapshots.append({
            'u': self.u.copy(),
            'v': self.v.copy(),
            'p': self.p.copy(),
            'iteration': self.iteration,
        })

    def drag_lift_coefficients(self):
        """Compute drag and lift coefficients for obstacle flows.

        Uses a simple pressure-integration approach around the obstacle boundary.
        Returns (Cd, Cl) or (0, 0) if no obstacle is present.
        """
        if not self.obstacle.any():
            return 0.0, 0.0

        dx, dy = self.dx, self.dy
        nu = 1.0 / self.re
        Fx, Fy = 0.0, 0.0

        # Find obstacle boundary cells (obstacle cells adjacent to fluid)
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                if not self.obstacle[j, i]:
                    continue
                # Check neighbours
                for dj, di, nx_dir, ny_dir in [(-1, 0, 0, -1), (1, 0, 0, 1),
                                                 (0, -1, -1, 0), (0, 1, 1, 0)]:
                    nj, ni = j + dj, i + di
                    if 0 <= nj < self.ny and 0 <= ni < self.nx and not self.obstacle[nj, ni]:
                        # Pressure contribution
                        Fx -= self.p[nj, ni] * nx_dir * dy
                        Fy -= self.p[nj, ni] * ny_dir * dx
                        # Viscous contribution (approximate)
                        Fx += nu * (self.u[nj, ni]) / dx * abs(ny_dir) * dy
                        Fy += nu * (self.v[nj, ni]) / dy * abs(nx_dir) * dx

        U_inf = 1.0  # reference velocity
        L_ref = min(self.nx, self.ny) // 5 * dy  # approximate obstacle diameter
        denom = 0.5 * U_inf ** 2 * L_ref
        if denom < 1e-30:
            return 0.0, 0.0
        Cd = Fx / denom
        Cl = Fy / denom
        return Cd, Cl


# ---------------------------------------------------------------------------
# Solver Worker Thread
# ---------------------------------------------------------------------------

class SolverThread(QThread):
    """Runs the CFD solver in a background thread to keep the GUI responsive."""
    step_done = pyqtSignal(int, float)       # iteration, residual
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, solver: NavierStokesSolver, parent=None):
        super().__init__(parent)
        self.solver = solver

    def run(self):
        try:
            def _cb(it, res):
                self.step_done.emit(it, res)
            self.solver.run(callback=_cb)
        except Exception as exc:
            self.error_occurred.emit(traceback.format_exc())
        finally:
            self.finished.emit()


# ---------------------------------------------------------------------------
# CFD Simulator Widget
# ---------------------------------------------------------------------------

class CFDSimulatorWidget(QWidget):
    """Main CFD Simulator widget for embedding in a PyQt5 scientific suite."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._logger = None
        self._solver = None
        self._thread = None
        self._animation_timer = QTimer(self)
        self._animation_timer.timeout.connect(self._animation_step)
        self._animation_running = False
        self._init_ui()

    # ---- public API -------------------------------------------------------

    def set_logger(self, fn):
        """Register a callable *fn(message: str)* used for status logging."""
        self._logger = fn

    def run(self):
        """Programmatic trigger equivalent to pressing the Run button."""
        self._on_run()

    # ---- UI construction --------------------------------------------------

    def _init_ui(self):
        root = QHBoxLayout(self)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # --- Left panel: controls ------------------------------------------
        ctrl_frame = QFrame()
        ctrl_frame.setMaximumWidth(340)
        ctrl_frame.setMinimumWidth(280)
        ctrl_layout = QVBoxLayout(ctrl_frame)

        # Problem type
        prob_group = QGroupBox("Problem Type")
        prob_lay = QVBoxLayout(prob_group)
        self._combo_problem = QComboBox()
        self._combo_problem.addItems([
            NavierStokesSolver.PROBLEM_LID_CAVITY,
            NavierStokesSolver.PROBLEM_CHANNEL,
            NavierStokesSolver.PROBLEM_OBSTACLE,
            NavierStokesSolver.PROBLEM_CONVECTION,
        ])
        prob_lay.addWidget(self._combo_problem)

        prob_lay.addWidget(QLabel("Inlet Velocity Profile:"))
        self._combo_inlet = QComboBox()
        self._combo_inlet.addItems([
            NavierStokesSolver.INLET_UNIFORM,
            NavierStokesSolver.INLET_PARABOLIC,
            NavierStokesSolver.INLET_PULSATILE,
        ])
        prob_lay.addWidget(self._combo_inlet)
        ctrl_layout.addWidget(prob_group)

        # Solver parameters
        param_group = QGroupBox("Solver Parameters")
        param_grid = QGridLayout(param_group)

        row = 0
        param_grid.addWidget(QLabel("Reynolds Number:"), row, 0)
        self._spin_re = QDoubleSpinBox()
        self._spin_re.setRange(1, 100000)
        self._spin_re.setValue(100)
        self._spin_re.setDecimals(1)
        param_grid.addWidget(self._spin_re, row, 1)

        row += 1
        param_grid.addWidget(QLabel("Grid Nx:"), row, 0)
        self._spin_nx = QSpinBox()
        self._spin_nx.setRange(11, 301)
        self._spin_nx.setValue(41)
        self._spin_nx.setSingleStep(10)
        param_grid.addWidget(self._spin_nx, row, 1)

        row += 1
        param_grid.addWidget(QLabel("Grid Ny:"), row, 0)
        self._spin_ny = QSpinBox()
        self._spin_ny.setRange(11, 301)
        self._spin_ny.setValue(41)
        self._spin_ny.setSingleStep(10)
        param_grid.addWidget(self._spin_ny, row, 1)

        row += 1
        param_grid.addWidget(QLabel("Time Step (dt):"), row, 0)
        self._spin_dt = QDoubleSpinBox()
        self._spin_dt.setRange(1e-5, 0.1)
        self._spin_dt.setValue(0.001)
        self._spin_dt.setDecimals(5)
        self._spin_dt.setSingleStep(0.0005)
        param_grid.addWidget(self._spin_dt, row, 1)

        row += 1
        param_grid.addWidget(QLabel("Max Iterations:"), row, 0)
        self._spin_maxiter = QSpinBox()
        self._spin_maxiter.setRange(10, 100000)
        self._spin_maxiter.setValue(500)
        self._spin_maxiter.setSingleStep(100)
        param_grid.addWidget(self._spin_maxiter, row, 1)

        row += 1
        param_grid.addWidget(QLabel("Convergence Tol:"), row, 0)
        self._spin_tol = QDoubleSpinBox()
        self._spin_tol.setRange(1e-10, 1e-1)
        self._spin_tol.setValue(1e-6)
        self._spin_tol.setDecimals(10)
        self._spin_tol.setSingleStep(1e-7)
        param_grid.addWidget(self._spin_tol, row, 1)

        ctrl_layout.addWidget(param_group)

        # Visualization options
        vis_group = QGroupBox("Visualization")
        vis_lay = QVBoxLayout(vis_group)
        self._combo_viz = QComboBox()
        self._combo_viz.addItems([
            "Velocity Magnitude",
            "Streamlines",
            "Pressure Field",
            "Velocity Vectors",
            "Vorticity",
        ])
        self._combo_viz.currentIndexChanged.connect(self._update_plot)
        vis_lay.addWidget(self._combo_viz)

        self._chk_overlay_vectors = QCheckBox("Overlay velocity vectors")
        self._chk_overlay_vectors.stateChanged.connect(self._update_plot)
        vis_lay.addWidget(self._chk_overlay_vectors)

        self._chk_show_obstacle = QCheckBox("Highlight obstacle")
        self._chk_show_obstacle.setChecked(True)
        self._chk_show_obstacle.stateChanged.connect(self._update_plot)
        vis_lay.addWidget(self._chk_show_obstacle)

        ctrl_layout.addWidget(vis_group)

        # Animation controls
        anim_group = QGroupBox("Animation")
        anim_lay = QHBoxLayout(anim_group)
        self._btn_anim_start = QPushButton("Play")
        self._btn_anim_start.clicked.connect(self._toggle_animation)
        anim_lay.addWidget(self._btn_anim_start)

        self._spin_anim_interval = QSpinBox()
        self._spin_anim_interval.setRange(10, 2000)
        self._spin_anim_interval.setValue(100)
        self._spin_anim_interval.setSuffix(" ms")
        anim_lay.addWidget(QLabel("Interval:"))
        anim_lay.addWidget(self._spin_anim_interval)
        ctrl_layout.addWidget(anim_group)

        # Run / Stop / Reset
        btn_layout = QHBoxLayout()
        self._btn_run = QPushButton("Run")
        self._btn_run.clicked.connect(self._on_run)
        self._btn_stop = QPushButton("Stop")
        self._btn_stop.clicked.connect(self._on_stop)
        self._btn_stop.setEnabled(False)
        self._btn_reset = QPushButton("Reset")
        self._btn_reset.clicked.connect(self._on_reset)
        btn_layout.addWidget(self._btn_run)
        btn_layout.addWidget(self._btn_stop)
        btn_layout.addWidget(self._btn_reset)
        ctrl_layout.addLayout(btn_layout)

        # Progress
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        ctrl_layout.addWidget(self._progress)

        self._lbl_status = QLabel("Ready")
        ctrl_layout.addWidget(self._lbl_status)

        # --- Generation / Export tools ---
        gen_group = QGroupBox("Generation / Export")
        gen_lay = QVBoxLayout(gen_group)

        btn_export_csv = QPushButton("Export Fields CSV")
        btn_export_csv.clicked.connect(self._on_export_csv)
        gen_lay.addWidget(btn_export_csv)

        btn_export_gif = QPushButton("Export Animation GIF")
        btn_export_gif.clicked.connect(self._on_export_gif)
        gen_lay.addWidget(btn_export_gif)

        btn_drag_lift = QPushButton("Compute Drag/Lift")
        btn_drag_lift.clicked.connect(self._on_drag_lift)
        gen_lay.addWidget(btn_drag_lift)

        btn_vorticity_plot = QPushButton("Export Vorticity PNG")
        btn_vorticity_plot.clicked.connect(self._on_export_vorticity)
        gen_lay.addWidget(btn_vorticity_plot)

        btn_re_sweep = QPushButton("Reynolds Number Sweep...")
        btn_re_sweep.clicked.connect(self._on_re_sweep)
        gen_lay.addWidget(btn_re_sweep)

        ctrl_layout.addWidget(gen_group)

        # Log console
        log_group = QGroupBox("Log")
        log_lay = QVBoxLayout(log_group)
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumHeight(140)
        log_lay.addWidget(self._log_text)
        ctrl_layout.addWidget(log_group)

        ctrl_layout.addStretch()
        splitter.addWidget(ctrl_frame)

        # --- Right panel: plots -------------------------------------------
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)

        self._tabs = QTabWidget()

        # Main field plot
        self._fig_main = Figure(figsize=(6, 5), dpi=100)
        style_figure(self._fig_main)
        self._ax_main = self._fig_main.add_subplot(111)
        self._canvas_main = FigureCanvas(self._fig_main)
        self._toolbar_main = NavigationToolbar(self._canvas_main, self)
        main_tab = QWidget()
        ml = QVBoxLayout(main_tab)
        ml.addWidget(self._toolbar_main)
        ml.addWidget(self._canvas_main)
        self._tabs.addTab(main_tab, "Flow Field")

        # Convergence plot
        self._fig_conv = Figure(figsize=(6, 3), dpi=100)
        style_figure(self._fig_conv)
        self._ax_conv = self._fig_conv.add_subplot(111)
        self._canvas_conv = FigureCanvas(self._fig_conv)
        self._toolbar_conv = NavigationToolbar(self._canvas_conv, self)
        conv_tab = QWidget()
        cl = QVBoxLayout(conv_tab)
        cl.addWidget(self._toolbar_conv)
        cl.addWidget(self._canvas_conv)
        self._tabs.addTab(conv_tab, "Convergence")

        plot_layout.addWidget(self._tabs)
        splitter.addWidget(plot_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    # ---- Logging ----------------------------------------------------------

    def _log(self, msg):
        self._log_text.append(msg)
        if self._logger:
            self._logger(msg)

    # ---- Solver lifecycle -------------------------------------------------

    def _create_solver(self):
        problem = self._combo_problem.currentText()
        nx = self._spin_nx.value()
        ny = self._spin_ny.value()
        re = self._spin_re.value()
        dt = self._spin_dt.value()
        max_iter = self._spin_maxiter.value()
        tol = self._spin_tol.value()

        inlet = self._combo_inlet.currentText()
        self._solver = NavierStokesSolver(
            problem_type=problem, nx=nx, ny=ny,
            re=re, dt=dt, max_iter=max_iter, tol=tol,
            inlet_profile=inlet
        )
        self._log(f"Solver created: {problem}, grid={nx}x{ny}, Re={re}, dt={dt}, inlet={inlet}")

    def _on_run(self):
        if self._thread and self._thread.isRunning():
            return
        self._create_solver()
        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._progress.setValue(0)
        self._lbl_status.setText("Running...")
        self._log("Simulation started.")

        self._thread = SolverThread(self._solver, self)
        self._thread.step_done.connect(self._on_step_done)
        self._thread.finished.connect(self._on_finished)
        self._thread.error_occurred.connect(self._on_error)
        self._thread.start()

    def _on_stop(self):
        if self._solver:
            self._solver.stop()
        self._lbl_status.setText("Stopping...")
        self._log("Stop requested.")

    def _on_reset(self):
        self._on_stop()
        if self._thread and self._thread.isRunning():
            self._thread.wait(2000)
        self._solver = None
        self._ax_main.clear()
        self._canvas_main.draw()
        self._ax_conv.clear()
        self._canvas_conv.draw()
        self._progress.setValue(0)
        self._lbl_status.setText("Ready")
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._log("Simulation reset.")

    def _on_step_done(self, iteration, residual):
        max_iter = self._spin_maxiter.value()
        pct = int(100 * (iteration + 1) / max_iter)
        self._progress.setValue(min(pct, 100))
        self._lbl_status.setText(f"Iter {iteration+1}/{max_iter}  |  Residual: {residual:.3e}")

        # Save snapshot every 20 steps for animation export
        if self._solver and iteration % 20 == 0:
            self._solver.save_snapshot()

        # Live convergence update every 20 steps
        if iteration % 20 == 0 or iteration == max_iter - 1:
            self._plot_convergence()

    def _on_finished(self):
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._progress.setValue(100)
        if self._solver:
            n = self._solver.iteration
            res = self._solver.residuals[-1] if self._solver.residuals else float('nan')
            self._lbl_status.setText(f"Done: {n} iterations, final residual {res:.3e}")
            self._log(f"Simulation finished: {n} iterations, residual={res:.3e}")
        self._update_plot()
        self._plot_convergence()

    def _on_error(self, tb_str):
        self._log(f"ERROR:\n{tb_str}")
        self._lbl_status.setText("Error occurred (see log)")
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)

    # ---- Plotting ---------------------------------------------------------

    def _update_plot(self):
        if self._solver is None:
            return
        s = self._solver
        self._ax_main.clear()

        x = np.linspace(0, s.lx, s.nx)
        y = np.linspace(0, s.ly, s.ny)
        X, Y = np.meshgrid(x, y)

        viz = self._combo_viz.currentText()

        if viz == "Velocity Magnitude":
            mag = s.velocity_magnitude()
            cf = self._ax_main.contourf(X, Y, mag, levels=30, cmap='jet')
            self._fig_main.colorbar(cf, ax=self._ax_main, label='|V|')
            self._ax_main.set_title("Velocity Magnitude")

        elif viz == "Streamlines":
            mag = s.velocity_magnitude()
            self._ax_main.contourf(X, Y, mag, levels=30, cmap='jet', alpha=0.4)
            # Ensure no zero-velocity everywhere (streamplot needs non-trivial field)
            speed = mag.max()
            if speed > 1e-12:
                self._ax_main.streamplot(X, Y, s.u, s.v, color='k',
                                         linewidth=0.7, density=1.5, arrowsize=1.0)
            self._ax_main.set_title("Streamlines")

        elif viz == "Pressure Field":
            cf = self._ax_main.contourf(X, Y, s.p, levels=30, cmap='coolwarm')
            self._fig_main.colorbar(cf, ax=self._ax_main, label='Pressure')
            self._ax_main.set_title("Pressure Field")

        elif viz == "Velocity Vectors":
            mag = s.velocity_magnitude()
            step = max(1, s.nx // 20)
            self._ax_main.quiver(
                X[::step, ::step], Y[::step, ::step],
                s.u[::step, ::step], s.v[::step, ::step],
                mag[::step, ::step], cmap='jet', scale=None
            )
            self._ax_main.set_title("Velocity Vectors")

        elif viz == "Vorticity":
            vort = s.vorticity()
            lim = max(abs(vort.min()), abs(vort.max()), 1e-12)
            cf = self._ax_main.contourf(X, Y, vort, levels=30,
                                        cmap='RdBu_r', vmin=-lim, vmax=lim)
            self._fig_main.colorbar(cf, ax=self._ax_main, label='Vorticity')
            self._ax_main.set_title("Vorticity")

        # Overlay velocity vectors
        if self._chk_overlay_vectors.isChecked() and viz != "Velocity Vectors":
            step = max(1, s.nx // 16)
            self._ax_main.quiver(
                X[::step, ::step], Y[::step, ::step],
                s.u[::step, ::step], s.v[::step, ::step],
                color='white', alpha=0.6, scale=None
            )

        # Highlight obstacle
        if self._chk_show_obstacle.isChecked() and s.obstacle.any():
            self._ax_main.contour(X, Y, s.obstacle.astype(float),
                                  levels=[0.5], colors='black', linewidths=2)
            self._ax_main.contourf(X, Y, s.obstacle.astype(float),
                                   levels=[0.5, 1.5], colors=['gray'], alpha=0.5)

        self._ax_main.set_xlabel("x")
        self._ax_main.set_ylabel("y")
        self._ax_main.set_aspect('equal')
        self._fig_main.tight_layout()
        self._canvas_main.draw()

    def _plot_convergence(self):
        if self._solver is None or not self._solver.residuals:
            return
        self._ax_conv.clear()
        res = self._solver.residuals
        self._ax_conv.semilogy(range(len(res)), res, 'b-', linewidth=0.8)
        self._ax_conv.set_xlabel("Iteration")
        self._ax_conv.set_ylabel("Max Divergence Residual")
        self._ax_conv.set_title("Convergence History")
        self._ax_conv.grid(True, which='both', linestyle='--', alpha=0.5)
        if len(res) > 1:
            self._ax_conv.axhline(y=self._spin_tol.value(), color='r',
                                  linestyle='--', linewidth=0.8, label='Tolerance')
            self._ax_conv.legend(fontsize=8)
        self._fig_conv.tight_layout()
        self._canvas_conv.draw()

    # ---- Animation --------------------------------------------------------

    def _toggle_animation(self):
        if self._animation_running:
            self._animation_timer.stop()
            self._animation_running = False
            self._btn_anim_start.setText("Play")
            self._log("Animation paused.")
        else:
            if self._solver is None:
                self._create_solver()
            self._animation_timer.setInterval(self._spin_anim_interval.value())
            self._animation_timer.start()
            self._animation_running = True
            self._btn_anim_start.setText("Pause")
            self._log("Animation started.")

    def _animation_step(self):
        """Perform a single solver step and redraw (called by QTimer)."""
        if self._solver is None:
            return
        try:
            residual = self._solver.step()
            self._lbl_status.setText(
                f"Anim iter {self._solver.iteration}  |  Residual: {residual:.3e}"
            )
            self._update_plot()
            if self._solver.iteration % 10 == 0:
                self._plot_convergence()
            if residual < self._solver.tol and self._solver.iteration > 10:
                self._animation_timer.stop()
                self._animation_running = False
                self._btn_anim_start.setText("Play")
                self._log("Animation: converged.")
        except Exception as exc:
            self._animation_timer.stop()
            self._animation_running = False
            self._btn_anim_start.setText("Play")
            self._log(f"Animation error: {exc}")

    # ---- Export: velocity/pressure fields as CSV --------------------------

    def _on_export_csv(self):
        """Export velocity and pressure fields as CSV."""
        if self._solver is None:
            QMessageBox.warning(self, "No Data", "Run the simulation first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Fields CSV", "cfd_fields.csv",
            "CSV files (*.csv);;All Files (*)")
        if not path:
            return

        s = self._solver
        x = np.linspace(0, s.lx, s.nx)
        y = np.linspace(0, s.ly, s.ny)
        with open(path, 'w') as f:
            f.write("x,y,u,v,velocity_mag,pressure,vorticity\n")
            vort = s.vorticity()
            mag = s.velocity_magnitude()
            for j in range(s.ny):
                for i in range(s.nx):
                    f.write(f"{x[i]},{y[j]},{s.u[j,i]},{s.v[j,i]},"
                            f"{mag[j,i]},{s.p[j,i]},{vort[j,i]}\n")
        self._log(f"Fields CSV exported: {path}")

    # ---- Export: flow animation GIF ----------------------------------------

    def _on_export_gif(self):
        """Generate a GIF of flow development over time from saved snapshots."""
        if self._solver is None or not self._solver._snapshots:
            QMessageBox.warning(self, "No Data",
                                "Run the simulation first. Snapshots are saved automatically every 20 steps.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Animation GIF", "cfd_animation.gif",
            "GIF files (*.gif);;All Files (*)")
        if not path:
            return

        s = self._solver
        snapshots = s._snapshots
        x = np.linspace(0, s.lx, s.nx)
        y = np.linspace(0, s.ly, s.ny)
        X, Y = np.meshgrid(x, y)

        fig, ax = plt.subplots(figsize=(6, 5), dpi=100)

        def _frame(idx):
            ax.clear()
            snap = snapshots[idx]
            mag = np.sqrt(snap['u']**2 + snap['v']**2)
            cf = ax.contourf(X, Y, mag, levels=20, cmap='jet')
            ax.set_title(f"Velocity Magnitude - Iteration {snap['iteration']}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect('equal')
            return []

        anim = animation.FuncAnimation(fig, _frame, frames=len(snapshots),
                                        interval=200, blit=False)
        try:
            anim.save(path, writer='pillow', fps=5)
            self._log(f"Animation GIF exported: {path} ({len(snapshots)} frames)")
        except Exception as exc:
            QMessageBox.warning(self, "Export Error", f"Failed to save GIF:\n{exc}")
            self._log(f"GIF export error: {exc}")
        finally:
            plt.close(fig)

    # ---- Drag/Lift coefficient calculation --------------------------------

    def _on_drag_lift(self):
        """Compute and display drag/lift coefficients."""
        if self._solver is None:
            QMessageBox.warning(self, "No Data", "Run the simulation first.")
            return
        Cd, Cl = self._solver.drag_lift_coefficients()
        msg = f"Drag coefficient Cd = {Cd:.6f}\nLift coefficient Cl = {Cl:.6f}"
        if not self._solver.obstacle.any():
            msg += "\n\n(No obstacle detected -- use 'Flow around obstacle' problem type)"
        self._log(msg)
        QMessageBox.information(self, "Drag / Lift Coefficients", msg)

    # ---- Vorticity field export -------------------------------------------

    def _on_export_vorticity(self):
        """Export a publication-quality vorticity contour plot as PNG."""
        if self._solver is None:
            QMessageBox.warning(self, "No Data", "Run the simulation first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Vorticity PNG", "vorticity.png",
            "PNG files (*.png);;All Files (*)")
        if not path:
            return

        s = self._solver
        x = np.linspace(0, s.lx, s.nx)
        y = np.linspace(0, s.ly, s.ny)
        X, Y = np.meshgrid(x, y)
        vort = s.vorticity()

        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        lim = max(abs(vort.min()), abs(vort.max()), 1e-12)
        cf = ax.contourf(X, Y, vort, levels=40, cmap='RdBu_r', vmin=-lim, vmax=lim)
        cb = fig.colorbar(cf, ax=ax, shrink=0.85)
        cb.set_label('Vorticity (1/s)', fontsize=11)
        if s.obstacle.any():
            ax.contourf(X, Y, s.obstacle.astype(float), levels=[0.5, 1.5],
                        colors=['gray'], alpha=0.6)
        ax.set_title(f'Vorticity Field (Re={s.re:.0f})', fontsize=13)
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        ax.set_aspect('equal')
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        self._log(f"Vorticity PNG exported: {path}")

    # ---- Reynolds number sweep -------------------------------------------

    def _on_re_sweep(self):
        """Run simulations at multiple Re values and plot comparison."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Reynolds Number Sweep")
        dlg.resize(320, 180)
        flay = QFormLayout(dlg)

        spin_start = QDoubleSpinBox(); spin_start.setRange(1, 100000); spin_start.setValue(50)
        spin_end = QDoubleSpinBox(); spin_end.setRange(1, 100000); spin_end.setValue(500)
        spin_steps = QSpinBox(); spin_steps.setRange(3, 20); spin_steps.setValue(5)
        spin_iter = QSpinBox(); spin_iter.setRange(50, 50000); spin_iter.setValue(300)

        flay.addRow("Re start:", spin_start)
        flay.addRow("Re end:", spin_end)
        flay.addRow("Number of Re:", spin_steps)
        flay.addRow("Iterations each:", spin_iter)

        bbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bbox.accepted.connect(dlg.accept)
        bbox.rejected.connect(dlg.reject)
        flay.addRow(bbox)

        if dlg.exec_() != QDialog.Accepted:
            return

        re_start = spin_start.value()
        re_end = spin_end.value()
        n_re = spin_steps.value()
        n_iter = spin_iter.value()

        re_values = np.linspace(re_start, re_end, n_re)
        problem = self._combo_problem.currentText()
        nx = self._spin_nx.value()
        ny = self._spin_ny.value()
        dt = self._spin_dt.value()
        inlet = self._combo_inlet.currentText()

        self._log(f"Reynolds sweep: {[_clean_num(v) for v in re_values.tolist()]}, {n_iter} iterations each")

        # Store results
        results = []
        for re_val in re_values:
            solver = NavierStokesSolver(
                problem_type=problem, nx=nx, ny=ny,
                re=re_val, dt=dt, max_iter=n_iter, tol=1e-8,
                inlet_profile=inlet
            )
            solver.run()
            results.append({
                'Re': re_val,
                'u_max': solver.velocity_magnitude().max(),
                'vort_max': np.abs(solver.vorticity()).max(),
                'final_residual': solver.residuals[-1] if solver.residuals else 0.0,
            })
            self._log(f"  Re={re_val:.0f}: u_max={results[-1]['u_max']:.4f}, "
                       f"vort_max={results[-1]['vort_max']:.4f}")

        # Plot comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)

        re_list = [r['Re'] for r in results]
        ax1, ax2, ax3 = axes

        ax1.plot(re_list, [r['u_max'] for r in results], 'b-o', markersize=5)
        ax1.set_xlabel('Reynolds Number')
        ax1.set_ylabel('Max Velocity')
        ax1.set_title('Max Velocity vs Re')
        ax1.grid(True, alpha=0.3)

        ax2.plot(re_list, [r['vort_max'] for r in results], 'r-s', markersize=5)
        ax2.set_xlabel('Reynolds Number')
        ax2.set_ylabel('Max |Vorticity|')
        ax2.set_title('Max Vorticity vs Re')
        ax2.grid(True, alpha=0.3)

        ax3.semilogy(re_list, [r['final_residual'] for r in results], 'g-^', markersize=5)
        ax3.set_xlabel('Reynolds Number')
        ax3.set_ylabel('Final Residual')
        ax3.set_title('Final Residual vs Re')
        ax3.grid(True, which='both', alpha=0.3)

        fig.suptitle(f'Reynolds Number Sweep: {problem}', fontsize=13)
        fig.tight_layout()
        fig.show()
        self._log("Reynolds number sweep complete.")


# ---------------------------------------------------------------------------
# Standalone entry point (for testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication, QMainWindow

    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setWindowTitle("CFD Simulator - Navier-Stokes 2D")
    win.resize(1200, 750)
    widget = CFDSimulatorWidget()
    widget.set_logger(print)
    win.setCentralWidget(widget)
    win.show()
    sys.exit(app.exec_())
