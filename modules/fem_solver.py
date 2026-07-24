"""
FEM Solver Widget for PyQt5 Scientific Suite
=============================================
Finite Element Method solver supporting 2D heat conduction, structural mechanics,
and electrostatics problems on rectangular, L-shaped, and circular domains.

Uses linear triangular elements with scipy sparse solvers.
"""

import os
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QComboBox,
    QDoubleSpinBox, QSpinBox, QPushButton, QSplitter, QTextEdit,
    QFormLayout, QTabWidget, QCheckBox, QProgressBar, QFrame,
    QGridLayout, QSizePolicy, QFileDialog, QMessageBox, QLineEdit,
    QDialog, QDialogButtonBox, QPlainTextEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.tri import Triangulation

try:
    from mpl_style import style_figure, style_axes
except ImportError:
    def style_figure(f): pass
    def style_axes(a): pass
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def _clean_num(x, tol=1e-10):
    """Clean floating-point noise for display."""
    if isinstance(x, (float,)):
        rounded = round(x)
        if abs(x - rounded) < tol:
            return int(rounded)
        return round(x, 10)
    return x


# ---------------------------------------------------------------------------
# Mesh generation helpers
# ---------------------------------------------------------------------------

def generate_rectangular_mesh(width, height, nx, ny):
    """Generate a structured triangular mesh over a rectangular domain."""
    x = np.linspace(0, width, nx + 1)
    y = np.linspace(0, height, ny + 1)
    xx, yy = np.meshgrid(x, y)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])

    triangles = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n0 + (nx + 1)
            n3 = n2 + 1
            triangles.append([n0, n1, n3])
            triangles.append([n0, n3, n2])
    triangles = np.array(triangles, dtype=int)

    boundary = _extract_rect_boundary(nodes, width, height)
    return nodes, triangles, boundary


def generate_l_shaped_mesh(size, resolution):
    """Generate a mesh for an L-shaped domain (unit square minus upper-right quarter)."""
    half = size / 2.0
    nx = resolution
    ny = resolution

    nodes_list = []
    # Bottom half: full width
    x_full = np.linspace(0, size, 2 * nx + 1)
    y_bottom = np.linspace(0, half, ny + 1)
    for yv in y_bottom:
        for xv in x_full:
            nodes_list.append([xv, yv])

    # Top half: left half only
    x_left = np.linspace(0, half, nx + 1)
    y_top = np.linspace(half, size, ny + 1)[1:]  # skip duplicate row
    for yv in y_top:
        for xv in x_left:
            nodes_list.append([xv, yv])

    nodes = np.array(nodes_list)
    tri = Delaunay(nodes)
    triangles = tri.simplices

    tol = 1e-10 * size
    boundary = {
        'bottom': np.where(np.abs(nodes[:, 1]) < tol)[0],
        'right_lower': np.where(np.abs(nodes[:, 0] - size) < tol)[0],
        'top_step': np.where(
            (np.abs(nodes[:, 1] - half) < tol) & (nodes[:, 0] > half - tol)
        )[0],
        'right_step': np.where(
            (np.abs(nodes[:, 0] - half) < tol) & (nodes[:, 1] > half - tol)
        )[0],
        'top': np.where(np.abs(nodes[:, 1] - size) < tol)[0],
        'left': np.where(np.abs(nodes[:, 0]) < tol)[0],
    }
    return nodes, triangles, boundary


def generate_circular_mesh(radius, resolution):
    """Generate a mesh for a circular domain using polar-like node placement."""
    nodes_list = [[0.0, 0.0]]
    nr = resolution
    for i in range(1, nr + 1):
        r = radius * i / nr
        n_theta = max(6, int(6 * i))
        for j in range(n_theta):
            theta = 2.0 * np.pi * j / n_theta
            nodes_list.append([r * np.cos(theta), r * np.sin(theta)])

    nodes = np.array(nodes_list)
    tri = Delaunay(nodes)
    triangles = tri.simplices

    dist = np.sqrt(nodes[:, 0] ** 2 + nodes[:, 1] ** 2)
    tol = radius * 0.05
    boundary = {
        'outer': np.where(np.abs(dist - radius) < tol)[0],
        'center': np.array([0]),
    }
    return nodes, triangles, boundary


def generate_polygon_mesh(boundary_points, resolution):
    """Generate a triangular mesh inside an arbitrary polygon defined by boundary_points.

    Parameters
    ----------
    boundary_points : array-like, shape (N, 2)
        Ordered vertices of the polygon boundary.
    resolution : int
        Approximate number of interior points along each axis.

    Returns
    -------
    nodes, triangles, boundary_dict
    """
    pts = np.asarray(boundary_points, dtype=float)
    if len(pts) < 3:
        raise ValueError("Need at least 3 boundary points to define a polygon.")

    # Bounding box
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    margin = 0.0

    # Create interior candidate grid
    xs = np.linspace(xmin - margin, xmax + margin, resolution)
    ys = np.linspace(ymin - margin, ymax + margin, resolution)
    xx, yy = np.meshgrid(xs, ys)
    candidates = np.column_stack([xx.ravel(), yy.ravel()])

    # Ray-casting point-in-polygon test
    from matplotlib.path import Path
    poly_path = Path(pts)
    inside = poly_path.contains_points(candidates)
    interior = candidates[inside]

    # Combine boundary + interior nodes
    nodes = np.vstack([pts, interior])
    tri = Delaunay(nodes)
    triangles = tri.simplices

    # Remove triangles whose centroid is outside the polygon
    centroids = nodes[triangles].mean(axis=1)
    valid = poly_path.contains_points(centroids)
    triangles = triangles[valid]

    # Identify boundary node indices
    n_bnd = len(pts)
    boundary = {
        'polygon_boundary': np.arange(n_bnd),
    }
    return nodes, triangles, boundary


def _extract_rect_boundary(nodes, width, height):
    """Identify boundary node indices for a rectangular domain."""
    tol = 1e-10 * max(width, height, 1.0)
    return {
        'left': np.where(np.abs(nodes[:, 0]) < tol)[0],
        'right': np.where(np.abs(nodes[:, 0] - width) < tol)[0],
        'bottom': np.where(np.abs(nodes[:, 1]) < tol)[0],
        'top': np.where(np.abs(nodes[:, 1] - height) < tol)[0],
    }


# ---------------------------------------------------------------------------
# Element stiffness and force routines
# ---------------------------------------------------------------------------

def _tri_area(p0, p1, p2):
    """Signed area of a triangle."""
    return 0.5 * ((p1[0] - p0[0]) * (p2[1] - p0[1]) -
                   (p2[0] - p0[0]) * (p1[1] - p0[1]))


def _shape_grad(p0, p1, p2):
    """Gradient of linear shape functions for a triangle (2x3 matrix)."""
    A2 = 2.0 * _tri_area(p0, p1, p2)
    if abs(A2) < 1e-30:
        return np.zeros((2, 3)), 0.0
    dN = np.array([
        [p1[1] - p2[1], p2[1] - p0[1], p0[1] - p1[1]],
        [p2[0] - p1[0], p0[0] - p2[0], p1[0] - p0[0]],
    ]) / A2
    return dN, abs(A2) / 2.0


def element_stiffness_scalar(nodes_e, k=1.0):
    """Element stiffness for scalar problems (heat / electrostatics).

    K_e = k * A * (dN^T dN)  where dN is 2x3 gradient matrix.
    """
    dN, A = _shape_grad(nodes_e[0], nodes_e[1], nodes_e[2])
    if A < 1e-30:
        return np.zeros((3, 3))
    Ke = k * A * (dN.T @ dN)
    return Ke


def element_stiffness_structural(nodes_e, E, nu, plane_stress=True):
    """Element stiffness for 2D structural (plane stress or plane strain).

    Uses 6x6 element stiffness (2 DOF per node).
    """
    if plane_stress:
        factor = E / (1.0 - nu ** 2)
        D = factor * np.array([
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ])
    else:
        factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        D = factor * np.array([
            [1.0 - nu, nu, 0.0],
            [nu, 1.0 - nu, 0.0],
            [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0],
        ])

    dN, A = _shape_grad(nodes_e[0], nodes_e[1], nodes_e[2])
    if A < 1e-30:
        return np.zeros((6, 6))

    B = np.zeros((3, 6))
    for i in range(3):
        B[0, 2 * i] = dN[0, i]
        B[1, 2 * i + 1] = dN[1, i]
        B[2, 2 * i] = dN[1, i]
        B[2, 2 * i + 1] = dN[0, i]

    Ke = A * (B.T @ D @ B)
    return Ke


def element_force_scalar(nodes_e, source=0.0):
    """Consistent element load vector for a uniform source term."""
    _, A = _shape_grad(nodes_e[0], nodes_e[1], nodes_e[2])
    return source * A / 3.0 * np.ones(3)


def element_force_structural(nodes_e, body_force):
    """Consistent element body force vector (2 components per node)."""
    _, A = _shape_grad(nodes_e[0], nodes_e[1], nodes_e[2])
    fe = np.zeros(6)
    for i in range(3):
        fe[2 * i] = body_force[0] * A / 3.0
        fe[2 * i + 1] = body_force[1] * A / 3.0
    return fe


# ---------------------------------------------------------------------------
# Global assembly
# ---------------------------------------------------------------------------

def assemble_scalar(nodes, elements, k=1.0, source=0.0):
    """Assemble global stiffness matrix and force vector for scalar problem."""
    n_nodes = len(nodes)
    rows, cols, vals = [], [], []
    F = np.zeros(n_nodes)

    for elem in elements:
        nodes_e = nodes[elem]
        Ke = element_stiffness_scalar(nodes_e, k)
        fe = element_force_scalar(nodes_e, source)
        for a in range(3):
            F[elem[a]] += fe[a]
            for b in range(3):
                rows.append(elem[a])
                cols.append(elem[b])
                vals.append(Ke[a, b])

    K = sparse.coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    return K, F


def assemble_structural(nodes, elements, E, nu, plane_stress=True, body_force=(0, 0)):
    """Assemble global stiffness and force for 2D structural problem."""
    n_dof = 2 * len(nodes)
    rows, cols, vals = [], [], []
    F = np.zeros(n_dof)
    bf = np.array(body_force, dtype=float)

    for elem in elements:
        nodes_e = nodes[elem]
        Ke = element_stiffness_structural(nodes_e, E, nu, plane_stress)
        fe = element_force_structural(nodes_e, bf)
        dofs = []
        for n_id in elem:
            dofs.extend([2 * n_id, 2 * n_id + 1])
        for a in range(6):
            F[dofs[a]] += fe[a]
            for b in range(6):
                rows.append(dofs[a])
                cols.append(dofs[b])
                vals.append(Ke[a, b])

    K = sparse.coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()
    return K, F


# ---------------------------------------------------------------------------
# Boundary condition application
# ---------------------------------------------------------------------------

def apply_dirichlet_scalar(K, F, bc_nodes, bc_value):
    """Apply Dirichlet BCs by modifying K and F (penalty method)."""
    penalty = 1e20
    K = K.tolil()
    for n in bc_nodes:
        K[n, n] += penalty
        F[n] = penalty * bc_value
    return K.tocsr(), F


def apply_neumann_scalar(nodes, F, edge_nodes, flux):
    """Apply Neumann (flux) BC on an edge.  Distributes flux to edge nodes."""
    if len(edge_nodes) < 2:
        return F
    coords = nodes[edge_nodes]
    order = np.argsort(np.arctan2(coords[:, 1] - coords[:, 1].mean(),
                                   coords[:, 0] - coords[:, 0].mean()))
    sorted_nodes = edge_nodes[order]
    for i in range(len(sorted_nodes) - 1):
        n0, n1 = sorted_nodes[i], sorted_nodes[i + 1]
        length = np.linalg.norm(nodes[n1] - nodes[n0])
        F[n0] += flux * length / 2.0
        F[n1] += flux * length / 2.0
    return F


def apply_dirichlet_structural(K, F, bc_nodes, bc_component, bc_value):
    """Apply Dirichlet BC for structural (component 0=x, 1=y)."""
    penalty = 1e20
    K = K.tolil()
    for n in bc_nodes:
        dof = 2 * n + bc_component
        K[dof, dof] += penalty
        F[dof] = penalty * bc_value
    return K.tocsr(), F


# ---------------------------------------------------------------------------
# Solver thread
# ---------------------------------------------------------------------------

class SolverThread(QThread):
    """Run the FEM solve in a background thread to keep the GUI responsive."""
    progress = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            self._solve()
        except Exception as exc:
            self.error_signal.emit(str(exc))

    def _solve(self):
        p = self.params
        t0 = time.perf_counter()

        # --- mesh ---
        self.progress.emit("Generating mesh...")
        geom = p['geometry']
        res = p['resolution']
        if geom == 'Rectangular':
            nodes, elems, bnd = generate_rectangular_mesh(
                p['width'], p['height'], res, res)
        elif geom == 'L-shaped':
            nodes, elems, bnd = generate_l_shaped_mesh(p['width'], res)
        else:
            nodes, elems, bnd = generate_circular_mesh(p['width'] / 2.0, res)

        t_mesh = time.perf_counter() - t0
        self.progress.emit(
            f"Mesh: {len(nodes)} nodes, {len(elems)} elements ({t_mesh:.3f}s)")

        # --- assemble ---
        self.progress.emit("Assembling system...")
        problem = p['problem']
        t1 = time.perf_counter()

        if problem in ('Heat Conduction', 'Electrostatics'):
            prop = p['conductivity'] if problem == 'Heat Conduction' else p['permittivity']
            K, F = assemble_scalar(nodes, elems, k=prop, source=p.get('source', 0.0))
        else:
            K, F = assemble_structural(
                nodes, elems, p['youngs_modulus'], p['poisson_ratio'],
                plane_stress=(p['struct_type'] == 'Plane Stress'),
                body_force=(p.get('body_fx', 0.0), p.get('body_fy', 0.0)))

        t_assemble = time.perf_counter() - t1
        self.progress.emit(f"Assembly done ({t_assemble:.3f}s)")

        # --- boundary conditions ---
        self.progress.emit("Applying boundary conditions...")
        bc_specs = p.get('bc_specs', [])

        if problem in ('Heat Conduction', 'Electrostatics'):
            for bc in bc_specs:
                edge_key = bc['edge']
                matching = [v for k, v in bnd.items() if edge_key.lower() in k.lower()]
                if not matching:
                    continue
                edge_nodes = matching[0]
                if bc['type'] == 'Dirichlet':
                    K, F = apply_dirichlet_scalar(K, F, edge_nodes, bc['value'])
                else:
                    F = apply_neumann_scalar(nodes, F, edge_nodes, bc['value'])
        else:
            for bc in bc_specs:
                edge_key = bc['edge']
                matching = [v for k, v in bnd.items() if edge_key.lower() in k.lower()]
                if not matching:
                    continue
                edge_nodes = matching[0]
                comp = bc.get('component', 0)
                if bc['type'] == 'Dirichlet':
                    K, F = apply_dirichlet_structural(
                        K, F, edge_nodes, comp, bc['value'])

        # --- solve ---
        self.progress.emit("Solving linear system...")
        t2 = time.perf_counter()
        u = spsolve(K.tocsc(), F)
        t_solve = time.perf_counter() - t2
        self.progress.emit(f"Solve done ({t_solve:.3f}s)")

        total_time = time.perf_counter() - t0

        result = {
            'nodes': nodes,
            'elements': elems,
            'boundary': bnd,
            'solution': u,
            'problem': problem,
            'time_mesh': t_mesh,
            'time_assembly': t_assemble,
            'time_solve': t_solve,
            'time_total': total_time,
            'n_nodes': len(nodes),
            'n_elements': len(elems),
            'n_dof': len(u),
        }

        if problem == 'Structural':
            ux = u[0::2]
            uy = u[1::2]
            result['ux'] = ux
            result['uy'] = uy
            result['u_mag'] = np.sqrt(ux ** 2 + uy ** 2)

            # Compute von Mises stress per element
            E = p['youngs_modulus']
            nu = p['poisson_ratio']
            ps = p['struct_type'] == 'Plane Stress'
            stresses = []
            for elem in elems:
                nodes_e = nodes[elem]
                dN, A = _shape_grad(nodes_e[0], nodes_e[1], nodes_e[2])
                if A < 1e-30:
                    stresses.append(0.0)
                    continue
                B = np.zeros((3, 6))
                for i in range(3):
                    B[0, 2 * i] = dN[0, i]
                    B[1, 2 * i + 1] = dN[1, i]
                    B[2, 2 * i] = dN[1, i]
                    B[2, 2 * i + 1] = dN[0, i]
                if ps:
                    fac = E / (1.0 - nu ** 2)
                    D = fac * np.array([
                        [1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
                else:
                    fac = E / ((1 + nu) * (1 - 2 * nu))
                    D = fac * np.array([
                        [1 - nu, nu, 0], [nu, 1 - nu, 0],
                        [0, 0, (1 - 2 * nu) / 2]])
                dofs = []
                for nid in elem:
                    dofs.extend([2 * nid, 2 * nid + 1])
                ue = u[dofs]
                strain = B @ ue
                stress = D @ strain
                sx, sy, sxy = stress
                vm = np.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3 * sxy ** 2)
                stresses.append(vm)
            result['von_mises'] = np.array(stresses)

        self.finished_signal.emit(result)


# ---------------------------------------------------------------------------
# Boundary Condition Editor row
# ---------------------------------------------------------------------------

class BCRow(QFrame):
    """A single boundary condition specification row."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)

        self.edge_combo = QComboBox()
        self.edge_combo.addItems(['left', 'right', 'bottom', 'top'])
        lay.addWidget(QLabel("Edge:"))
        lay.addWidget(self.edge_combo)

        self.type_combo = QComboBox()
        self.type_combo.addItems(['Dirichlet', 'Neumann'])
        lay.addWidget(QLabel("Type:"))
        lay.addWidget(self.type_combo)

        self.value_spin = QDoubleSpinBox()
        self.value_spin.setRange(-1e12, 1e12)
        self.value_spin.setDecimals(4)
        self.value_spin.setValue(0.0)
        lay.addWidget(QLabel("Value:"))
        lay.addWidget(self.value_spin)

        self.comp_combo = QComboBox()
        self.comp_combo.addItems(['x (0)', 'y (1)'])
        self.comp_label = QLabel("Comp:")
        lay.addWidget(self.comp_label)
        lay.addWidget(self.comp_combo)
        self.comp_label.hide()
        self.comp_combo.hide()

        self.remove_btn = QPushButton("X")
        self.remove_btn.setFixedWidth(28)
        lay.addWidget(self.remove_btn)

    def set_structural(self, is_struct):
        self.comp_label.setVisible(is_struct)
        self.comp_combo.setVisible(is_struct)

    def get_spec(self):
        spec = {
            'edge': self.edge_combo.currentText(),
            'type': self.type_combo.currentText(),
            'value': self.value_spin.value(),
        }
        if self.comp_combo.isVisible():
            spec['component'] = self.comp_combo.currentIndex()
        return spec


# ---------------------------------------------------------------------------
# Main Widget
# ---------------------------------------------------------------------------

class FEMSolverWidget(QWidget):
    """Full-featured FEM Solver widget for a PyQt5 scientific application."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._logger = None
        self._result = None
        self._solver_thread = None
        self._bc_rows = []
        self._init_ui()

    # -- public API ----------------------------------------------------------

    def set_logger(self, fn):
        """Set an external logging callback ``fn(message: str)``."""
        self._logger = fn

    def run(self):
        """Programmatic trigger to start the solver (same as clicking Solve)."""
        self._on_solve()

    # -- UI construction -----------------------------------------------------

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # ---- Left panel: controls ----
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(4, 4, 4, 4)

        # Problem type
        grp_problem = QGroupBox("Problem Type")
        fl = QFormLayout(grp_problem)
        self.problem_combo = QComboBox()
        self.problem_combo.addItems(['Heat Conduction', 'Structural', 'Electrostatics'])
        self.problem_combo.currentTextChanged.connect(self._on_problem_changed)
        fl.addRow("Problem:", self.problem_combo)

        self.struct_type_combo = QComboBox()
        self.struct_type_combo.addItems(['Plane Stress', 'Plane Strain'])
        self.struct_type_label = QLabel("Type:")
        fl.addRow(self.struct_type_label, self.struct_type_combo)
        self.struct_type_label.hide()
        self.struct_type_combo.hide()
        left_lay.addWidget(grp_problem)

        # Geometry
        grp_geom = QGroupBox("Geometry")
        gl = QFormLayout(grp_geom)
        self.geom_combo = QComboBox()
        self.geom_combo.addItems(['Rectangular', 'L-shaped', 'Circular'])
        gl.addRow("Shape:", self.geom_combo)

        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.01, 1000)
        self.width_spin.setValue(1.0)
        self.width_spin.setDecimals(3)
        gl.addRow("Width / Size:", self.width_spin)

        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(0.01, 1000)
        self.height_spin.setValue(1.0)
        self.height_spin.setDecimals(3)
        gl.addRow("Height:", self.height_spin)

        self.res_spin = QSpinBox()
        self.res_spin.setRange(2, 200)
        self.res_spin.setValue(20)
        gl.addRow("Resolution:", self.res_spin)
        left_lay.addWidget(grp_geom)

        # Material
        grp_mat = QGroupBox("Material Properties")
        ml = QFormLayout(grp_mat)

        self.cond_spin = QDoubleSpinBox()
        self.cond_spin.setRange(1e-6, 1e6)
        self.cond_spin.setValue(1.0)
        self.cond_spin.setDecimals(4)
        self.cond_label = QLabel("Conductivity:")
        ml.addRow(self.cond_label, self.cond_spin)

        self.perm_spin = QDoubleSpinBox()
        self.perm_spin.setRange(1e-15, 1e6)
        self.perm_spin.setValue(8.854e-12)
        self.perm_spin.setDecimals(15)
        self.perm_label = QLabel("Permittivity:")
        ml.addRow(self.perm_label, self.perm_spin)
        self.perm_label.hide()
        self.perm_spin.hide()

        self.E_spin = QDoubleSpinBox()
        self.E_spin.setRange(1e-3, 1e12)
        self.E_spin.setValue(2.1e11)
        self.E_spin.setDecimals(1)
        self.E_label = QLabel("Young's Modulus:")
        ml.addRow(self.E_label, self.E_spin)
        self.E_label.hide()
        self.E_spin.hide()

        self.nu_spin = QDoubleSpinBox()
        self.nu_spin.setRange(0.0, 0.499)
        self.nu_spin.setValue(0.3)
        self.nu_spin.setDecimals(3)
        self.nu_label = QLabel("Poisson Ratio:")
        ml.addRow(self.nu_label, self.nu_spin)
        self.nu_label.hide()
        self.nu_spin.hide()

        self.source_spin = QDoubleSpinBox()
        self.source_spin.setRange(-1e12, 1e12)
        self.source_spin.setValue(0.0)
        self.source_spin.setDecimals(4)
        ml.addRow("Source term:", self.source_spin)

        self.body_fx_spin = QDoubleSpinBox()
        self.body_fx_spin.setRange(-1e12, 1e12)
        self.body_fx_spin.setValue(0.0)
        self.body_fy_spin = QDoubleSpinBox()
        self.body_fy_spin.setRange(-1e12, 1e12)
        self.body_fy_spin.setValue(-9810.0)
        self.body_fx_label = QLabel("Body Fx:")
        self.body_fy_label = QLabel("Body Fy:")
        ml.addRow(self.body_fx_label, self.body_fx_spin)
        ml.addRow(self.body_fy_label, self.body_fy_spin)
        self.body_fx_label.hide()
        self.body_fx_spin.hide()
        self.body_fy_label.hide()
        self.body_fy_spin.hide()

        left_lay.addWidget(grp_mat)

        # Boundary conditions
        grp_bc = QGroupBox("Boundary Conditions")
        self._bc_layout = QVBoxLayout(grp_bc)
        add_bc_btn = QPushButton("+ Add BC")
        add_bc_btn.clicked.connect(self._add_bc_row)
        self._bc_layout.addWidget(add_bc_btn)
        left_lay.addWidget(grp_bc)

        # Add default BCs
        self._add_default_bcs()

        # Solve button / progress
        self.solve_btn = QPushButton("Solve")
        self.solve_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "font-weight: bold; padding: 8px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1976D2; }"
            "QPushButton:disabled { background-color: #90CAF9; }"
        )
        self.solve_btn.clicked.connect(self._on_solve)
        left_lay.addWidget(self.solve_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate
        self.progress_bar.hide()
        left_lay.addWidget(self.progress_bar)

        # --- Generation / Export tools ---
        grp_tools = QGroupBox("Generation / Export")
        tools_lay = QVBoxLayout(grp_tools)

        btn_export_mesh = QPushButton("Export Mesh (VTK/CSV)")
        btn_export_mesh.clicked.connect(self._on_export_mesh)
        tools_lay.addWidget(btn_export_mesh)

        btn_export_result = QPushButton("Export Result CSV")
        btn_export_result.clicked.connect(self._on_export_result_csv)
        tools_lay.addWidget(btn_export_result)

        btn_export_png = QPushButton("Export Contour PNG")
        btn_export_png.clicked.connect(self._on_export_contour_png)
        tools_lay.addWidget(btn_export_png)

        btn_custom_geom = QPushButton("Custom Polygon Geometry...")
        btn_custom_geom.clicked.connect(self._on_custom_geometry)
        tools_lay.addWidget(btn_custom_geom)

        btn_param_study = QPushButton("Parametric Study...")
        btn_param_study.clicked.connect(self._on_parametric_study)
        tools_lay.addWidget(btn_param_study)

        btn_convergence = QPushButton("Convergence Study...")
        btn_convergence.clicked.connect(self._on_convergence_study)
        tools_lay.addWidget(btn_convergence)

        left_lay.addWidget(grp_tools)

        left_lay.addStretch()
        left.setMaximumWidth(380)
        splitter.addWidget(left)

        # ---- Right panel: results ----
        right = QWidget()
        right_lay = QVBoxLayout(right)

        self.tabs = QTabWidget()

        # Tab 1: Solution contour
        self.fig_solution = Figure(figsize=(6, 5), dpi=100)
        style_figure(self.fig_solution)
        self.canvas_solution = FigureCanvas(self.fig_solution)
        self.toolbar_solution = NavigationToolbar(self.canvas_solution, self)
        tab1 = QWidget()
        t1l = QVBoxLayout(tab1)
        t1l.addWidget(self.toolbar_solution)
        t1l.addWidget(self.canvas_solution)
        self.tabs.addTab(tab1, "Solution")

        # Tab 2: Mesh
        self.fig_mesh = Figure(figsize=(6, 5), dpi=100)
        style_figure(self.fig_mesh)
        self.canvas_mesh = FigureCanvas(self.fig_mesh)
        self.toolbar_mesh = NavigationToolbar(self.canvas_mesh, self)
        tab2 = QWidget()
        t2l = QVBoxLayout(tab2)
        t2l.addWidget(self.toolbar_mesh)
        t2l.addWidget(self.canvas_mesh)
        self.tabs.addTab(tab2, "Mesh")

        # Tab 3: Deformed shape (structural only)
        self.fig_deformed = Figure(figsize=(6, 5), dpi=100)
        style_figure(self.fig_deformed)
        self.canvas_deformed = FigureCanvas(self.fig_deformed)
        self.toolbar_deformed = NavigationToolbar(self.canvas_deformed, self)
        tab3 = QWidget()
        t3l = QVBoxLayout(tab3)
        t3l.addWidget(self.toolbar_deformed)
        t3l.addWidget(self.canvas_deformed)
        self.tabs.addTab(tab3, "Deformed Shape")

        # Tab 4: Stress (structural only)
        self.fig_stress = Figure(figsize=(6, 5), dpi=100)
        style_figure(self.fig_stress)
        self.canvas_stress = FigureCanvas(self.fig_stress)
        self.toolbar_stress = NavigationToolbar(self.canvas_stress, self)
        tab4 = QWidget()
        t4l = QVBoxLayout(tab4)
        t4l.addWidget(self.toolbar_stress)
        t4l.addWidget(self.canvas_stress)
        self.tabs.addTab(tab4, "Stress")

        right_lay.addWidget(self.tabs)

        # Results info panel
        grp_results = QGroupBox("Results Summary")
        rl = QVBoxLayout(grp_results)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(180)
        self.results_text.setStyleSheet(
            "QTextEdit { font-family: 'Consolas', 'Courier New', monospace; font-size: 11px; }")
        rl.addWidget(self.results_text)
        right_lay.addWidget(grp_results)

        # Log panel
        grp_log = QGroupBox("Solver Log")
        ll = QVBoxLayout(grp_log)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        self.log_text.setStyleSheet(
            "QTextEdit { font-family: 'Consolas', 'Courier New', monospace; "
            "font-size: 10px; color: #333; }")
        ll.addWidget(self.log_text)
        right_lay.addWidget(grp_log)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self._on_problem_changed(self.problem_combo.currentText())

    # -- BC management -------------------------------------------------------

    def _add_bc_row(self):
        row = BCRow()
        is_struct = self.problem_combo.currentText() == 'Structural'
        row.set_structural(is_struct)
        row.remove_btn.clicked.connect(lambda: self._remove_bc_row(row))
        self._bc_rows.append(row)
        self._bc_layout.insertWidget(self._bc_layout.count() - 1, row)

    def _remove_bc_row(self, row):
        if row in self._bc_rows:
            self._bc_rows.remove(row)
            self._bc_layout.removeWidget(row)
            row.deleteLater()

    def _add_default_bcs(self):
        # Left edge: Dirichlet = 100
        self._add_bc_row()
        self._bc_rows[-1].edge_combo.setCurrentText('left')
        self._bc_rows[-1].type_combo.setCurrentText('Dirichlet')
        self._bc_rows[-1].value_spin.setValue(100.0)

        # Right edge: Dirichlet = 0
        self._add_bc_row()
        self._bc_rows[-1].edge_combo.setCurrentText('right')
        self._bc_rows[-1].type_combo.setCurrentText('Dirichlet')
        self._bc_rows[-1].value_spin.setValue(0.0)

    # -- UI event handlers ---------------------------------------------------

    def _on_problem_changed(self, text):
        is_struct = (text == 'Structural')
        is_electro = (text == 'Electrostatics')
        is_heat = (text == 'Heat Conduction')

        self.struct_type_label.setVisible(is_struct)
        self.struct_type_combo.setVisible(is_struct)
        self.cond_label.setVisible(is_heat)
        self.cond_spin.setVisible(is_heat)
        self.perm_label.setVisible(is_electro)
        self.perm_spin.setVisible(is_electro)
        self.E_label.setVisible(is_struct)
        self.E_spin.setVisible(is_struct)
        self.nu_label.setVisible(is_struct)
        self.nu_spin.setVisible(is_struct)
        self.body_fx_label.setVisible(is_struct)
        self.body_fx_spin.setVisible(is_struct)
        self.body_fy_label.setVisible(is_struct)
        self.body_fy_spin.setVisible(is_struct)

        for row in self._bc_rows:
            row.set_structural(is_struct)

        # Show / hide structural-specific tabs
        if is_struct:
            self.tabs.setTabEnabled(2, True)
            self.tabs.setTabEnabled(3, True)
        else:
            self.tabs.setTabEnabled(2, False)
            self.tabs.setTabEnabled(3, False)

    def _on_solve(self):
        if self._solver_thread and self._solver_thread.isRunning():
            self._log("Solver is already running.")
            return

        self.log_text.clear()
        self.results_text.clear()
        self._log("Preparing solver parameters...")

        params = {
            'problem': self.problem_combo.currentText(),
            'geometry': self.geom_combo.currentText(),
            'width': self.width_spin.value(),
            'height': self.height_spin.value(),
            'resolution': self.res_spin.value(),
            'conductivity': self.cond_spin.value(),
            'permittivity': self.perm_spin.value(),
            'youngs_modulus': self.E_spin.value(),
            'poisson_ratio': self.nu_spin.value(),
            'struct_type': self.struct_type_combo.currentText(),
            'source': self.source_spin.value(),
            'body_fx': self.body_fx_spin.value(),
            'body_fy': self.body_fy_spin.value(),
            'bc_specs': [row.get_spec() for row in self._bc_rows],
        }

        self.solve_btn.setEnabled(False)
        self.progress_bar.show()

        self._solver_thread = SolverThread(params)
        self._solver_thread.progress.connect(self._log)
        self._solver_thread.finished_signal.connect(self._on_solve_finished)
        self._solver_thread.error_signal.connect(self._on_solve_error)
        self._solver_thread.start()

    def _on_solve_finished(self, result):
        self._result = result
        self.solve_btn.setEnabled(True)
        self.progress_bar.hide()
        self._log(f"Solution complete. Total time: {result['time_total']:.4f}s")
        self._update_results_panel(result)
        self._plot_mesh(result)
        self._plot_solution(result)
        if result['problem'] == 'Structural':
            self._plot_deformed(result)
            self._plot_stress(result)

    def _on_solve_error(self, msg):
        self.solve_btn.setEnabled(True)
        self.progress_bar.hide()
        self._log(f"ERROR: {msg}")

    # -- Logging -------------------------------------------------------------

    def _log(self, msg):
        self.log_text.append(msg)
        if self._logger:
            self._logger(msg)

    # -- Results panel -------------------------------------------------------

    def _update_results_panel(self, result):
        lines = []
        lines.append(f"Problem type : {result['problem']}")
        lines.append(f"Nodes        : {result['n_nodes']}")
        lines.append(f"Elements     : {result['n_elements']}")
        lines.append(f"DOFs         : {result['n_dof']}")
        lines.append(f"")
        lines.append(f"Mesh time    : {result['time_mesh']:.4f} s")
        lines.append(f"Assembly time: {result['time_assembly']:.4f} s")
        lines.append(f"Solve time   : {result['time_solve']:.4f} s")
        lines.append(f"Total time   : {result['time_total']:.4f} s")
        lines.append(f"")

        u = result['solution']
        if result['problem'] == 'Structural':
            lines.append(f"Max |u|      : {_clean_num(float(result['u_mag'].max())):.6e}")
            lines.append(f"Max ux       : {_clean_num(float(result['ux'].max())):.6e}")
            lines.append(f"Min ux       : {_clean_num(float(result['ux'].min())):.6e}")
            lines.append(f"Max uy       : {_clean_num(float(result['uy'].max())):.6e}")
            lines.append(f"Min uy       : {_clean_num(float(result['uy'].min())):.6e}")
            if 'von_mises' in result:
                vm = result['von_mises']
                lines.append(f"Max von Mises: {_clean_num(float(vm.max())):.6e}")
                lines.append(f"Min von Mises: {_clean_num(float(vm.min())):.6e}")
        else:
            lines.append(f"Max value    : {_clean_num(float(u.max())):.6e}")
            lines.append(f"Min value    : {_clean_num(float(u.min())):.6e}")
            lines.append(f"Mean value   : {_clean_num(float(u.mean())):.6e}")

        self.results_text.setPlainText("\n".join(lines))

    # -- Plotting ------------------------------------------------------------

    def _plot_mesh(self, result):
        self.fig_mesh.clear()
        ax = self.fig_mesh.add_subplot(111)
        nodes = result['nodes']
        elems = result['elements']
        tri = Triangulation(nodes[:, 0], nodes[:, 1], elems)
        ax.triplot(tri, 'b-', linewidth=0.3, alpha=0.7)

        # Highlight boundary nodes
        bnd = result['boundary']
        for key, indices in bnd.items():
            ax.plot(nodes[indices, 0], nodes[indices, 1], 'o',
                    markersize=1.5, label=key)

        ax.set_aspect('equal')
        ax.set_title(f"Mesh: {result['n_nodes']} nodes, {result['n_elements']} elements")
        ax.legend(fontsize=7, loc='upper right')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        self.fig_mesh.tight_layout()
        self.canvas_mesh.draw()

    def _plot_solution(self, result):
        self.fig_solution.clear()
        ax = self.fig_solution.add_subplot(111)
        nodes = result['nodes']
        elems = result['elements']
        u = result['solution']
        tri = Triangulation(nodes[:, 0], nodes[:, 1], elems)

        if result['problem'] == 'Structural':
            field = result['u_mag']
            label = 'Displacement magnitude'
        elif result['problem'] == 'Heat Conduction':
            field = u
            label = 'Temperature'
        else:
            field = u
            label = 'Electric potential'

        tpc = ax.tripcolor(tri, field, shading='gouraud', cmap='jet')
        cb = self.fig_solution.colorbar(tpc, ax=ax, shrink=0.8)
        cb.set_label(label)
        ax.set_aspect('equal')
        ax.set_title(f"{result['problem']} - {label}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        # Overlay contour lines
        try:
            ax.tricontour(tri, field, levels=12, colors='k',
                          linewidths=0.3, alpha=0.5)
        except Exception:
            pass

        self.fig_solution.tight_layout()
        self.canvas_solution.draw()

    def _plot_deformed(self, result):
        self.fig_deformed.clear()
        ax = self.fig_deformed.add_subplot(111)
        nodes = result['nodes']
        elems = result['elements']
        ux = result['ux']
        uy = result['uy']
        u_mag = result['u_mag']

        max_disp = u_mag.max()
        char_len = max(nodes[:, 0].max() - nodes[:, 0].min(),
                       nodes[:, 1].max() - nodes[:, 1].min())
        if max_disp > 1e-30:
            scale = 0.1 * char_len / max_disp
        else:
            scale = 1.0

        deformed = nodes.copy()
        deformed[:, 0] += scale * ux
        deformed[:, 1] += scale * uy

        tri_orig = Triangulation(nodes[:, 0], nodes[:, 1], elems)
        tri_def = Triangulation(deformed[:, 0], deformed[:, 1], elems)

        ax.triplot(tri_orig, 'b--', linewidth=0.3, alpha=0.3, label='Original')
        tpc = ax.tripcolor(tri_def, u_mag, shading='gouraud', cmap='hot')
        ax.triplot(tri_def, 'k-', linewidth=0.2, alpha=0.4)
        cb = self.fig_deformed.colorbar(tpc, ax=ax, shrink=0.8)
        cb.set_label('|u|')
        ax.set_aspect('equal')
        ax.set_title(f"Deformed shape (scale: {scale:.2e})")
        ax.legend(fontsize=7)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        self.fig_deformed.tight_layout()
        self.canvas_deformed.draw()

    def _plot_stress(self, result):
        self.fig_stress.clear()
        ax = self.fig_stress.add_subplot(111)
        nodes = result['nodes']
        elems = result['elements']
        vm = result.get('von_mises')
        if vm is None:
            ax.text(0.5, 0.5, 'No stress data', transform=ax.transAxes,
                    ha='center', va='center')
            self.canvas_stress.draw()
            return

        tri = Triangulation(nodes[:, 0], nodes[:, 1], elems)
        tpc = ax.tripcolor(tri, facecolors=vm, cmap='jet')
        cb = self.fig_stress.colorbar(tpc, ax=ax, shrink=0.8)
        cb.set_label('von Mises stress')
        ax.set_aspect('equal')
        ax.set_title(f"von Mises stress (max: {vm.max():.4e})")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        self.fig_stress.tight_layout()
        self.canvas_stress.draw()


    # -- Mesh export ---------------------------------------------------------

    def _on_export_mesh(self):
        """Export the current mesh to VTK or CSV file."""
        if self._result is None:
            QMessageBox.warning(self, "No Data", "Run the solver first to generate a mesh.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Mesh", "mesh.vtk",
            "VTK files (*.vtk);;CSV files (*.csv);;All Files (*)")
        if not path:
            return

        nodes = self._result['nodes']
        elems = self._result['elements']

        if path.lower().endswith('.vtk'):
            with open(path, 'w') as f:
                f.write("# vtk DataFile Version 3.0\n")
                f.write("FEM Mesh Export\n")
                f.write("ASCII\n")
                f.write("DATASET UNSTRUCTURED_GRID\n")
                f.write(f"POINTS {len(nodes)} float\n")
                for n in nodes:
                    f.write(f"{n[0]} {n[1]} 0.0\n")
                f.write(f"CELLS {len(elems)} {len(elems) * 4}\n")
                for e in elems:
                    f.write(f"3 {e[0]} {e[1]} {e[2]}\n")
                f.write(f"CELL_TYPES {len(elems)}\n")
                for _ in elems:
                    f.write("5\n")  # VTK_TRIANGLE
            self._log(f"Mesh exported to VTK: {path}")
        else:
            with open(path, 'w') as f:
                f.write("# Nodes: node_id, x, y\n")
                for i, n in enumerate(nodes):
                    f.write(f"{i},{n[0]},{n[1]}\n")
                f.write("# Elements: elem_id, n0, n1, n2\n")
                for i, e in enumerate(elems):
                    f.write(f"{i},{e[0]},{e[1]},{e[2]}\n")
            self._log(f"Mesh exported to CSV: {path}")

    # -- Result CSV export ---------------------------------------------------

    def _on_export_result_csv(self):
        """Export the solution field as CSV with columns x, y, value."""
        if self._result is None:
            QMessageBox.warning(self, "No Data", "Run the solver first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Result", "fem_result.csv",
            "CSV files (*.csv);;All Files (*)")
        if not path:
            return

        nodes = self._result['nodes']
        u = self._result['solution']
        problem = self._result['problem']

        with open(path, 'w') as f:
            if problem == 'Structural':
                f.write("x,y,ux,uy,u_mag\n")
                ux = self._result['ux']
                uy = self._result['uy']
                umag = self._result['u_mag']
                for i in range(len(nodes)):
                    f.write(f"{nodes[i,0]},{nodes[i,1]},{ux[i]},{uy[i]},{umag[i]}\n")
            else:
                f.write("x,y,value\n")
                for i in range(len(nodes)):
                    f.write(f"{nodes[i,0]},{nodes[i,1]},{u[i]}\n")

        self._log(f"Result CSV exported: {path}")

    # -- Publication-quality PNG export --------------------------------------

    def _on_export_contour_png(self):
        """Export the current solution contour as a high-resolution PNG."""
        if self._result is None:
            QMessageBox.warning(self, "No Data", "Run the solver first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Contour PNG", "fem_contour.png",
            "PNG files (*.png);;All Files (*)")
        if not path:
            return

        result = self._result
        nodes = result['nodes']
        elems = result['elements']
        tri = Triangulation(nodes[:, 0], nodes[:, 1], elems)

        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        if result['problem'] == 'Structural':
            field = result['u_mag']
            label = 'Displacement magnitude'
        elif result['problem'] == 'Heat Conduction':
            field = result['solution']
            label = 'Temperature'
        else:
            field = result['solution']
            label = 'Electric potential'

        tpc = ax.tripcolor(tri, field, shading='gouraud', cmap='jet')
        cb = fig.colorbar(tpc, ax=ax, shrink=0.85)
        cb.set_label(label, fontsize=11)
        try:
            ax.tricontour(tri, field, levels=15, colors='k', linewidths=0.3, alpha=0.5)
        except Exception:
            pass
        ax.set_aspect('equal')
        ax.set_title(f"{result['problem']} Solution", fontsize=13)
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        self._log(f"Contour PNG exported: {path}")

    # -- Custom polygon geometry --------------------------------------------

    def _on_custom_geometry(self):
        """Open a dialog to define polygon boundary points for a custom 2D domain."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Custom Polygon Geometry")
        dlg.resize(420, 300)
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(
            "Enter polygon vertices as x,y pairs (one per line).\n"
            "Example for a triangle:\n  0, 0\n  1, 0\n  0.5, 1"))
        txt = QPlainTextEdit()
        txt.setPlainText("0, 0\n1, 0\n1, 0.5\n0.5, 0.5\n0.5, 1\n0, 1")
        lay.addWidget(txt)
        bbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bbox.accepted.connect(dlg.accept)
        bbox.rejected.connect(dlg.reject)
        lay.addWidget(bbox)

        if dlg.exec_() != QDialog.Accepted:
            return

        try:
            lines = txt.toPlainText().strip().split('\n')
            pts = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.replace(';', ',').split(',')
                pts.append([float(parts[0].strip()), float(parts[1].strip())])
            pts = np.array(pts)
            if len(pts) < 3:
                raise ValueError("Need at least 3 points.")

            res = self.res_spin.value()
            nodes, elems, bnd = generate_polygon_mesh(pts, res)
            self._log(f"Custom polygon mesh: {len(nodes)} nodes, {len(elems)} elements")

            # Store as a result for preview
            self._custom_mesh = {'nodes': nodes, 'elements': elems, 'boundary': bnd}
            # Plot the mesh
            self.fig_mesh.clear()
            ax = self.fig_mesh.add_subplot(111)
            tri = Triangulation(nodes[:, 0], nodes[:, 1], elems)
            ax.triplot(tri, 'b-', linewidth=0.3, alpha=0.7)
            ax.plot(pts[:, 0], pts[:, 1], 'ro-', markersize=4, linewidth=1.5, label='Boundary')
            ax.set_aspect('equal')
            ax.set_title(f"Custom Polygon Mesh: {len(nodes)} nodes, {len(elems)} elements")
            ax.legend(fontsize=7)
            self.fig_mesh.tight_layout()
            self.canvas_mesh.draw()
            self.tabs.setCurrentIndex(1)

        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to create polygon mesh:\n{exc}")

    # -- Parametric study ---------------------------------------------------

    def _on_parametric_study(self):
        """Sweep a parameter and plot the solution's max/min/mean vs that parameter."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Parametric Study")
        dlg.resize(350, 200)
        flay = QFormLayout(dlg)

        combo_param = QComboBox()
        problem = self.problem_combo.currentText()
        if problem == 'Heat Conduction':
            combo_param.addItems(['conductivity', 'source', 'resolution'])
        elif problem == 'Structural':
            combo_param.addItems(['youngs_modulus', 'poisson_ratio', 'resolution'])
        else:
            combo_param.addItems(['permittivity', 'source', 'resolution'])
        flay.addRow("Parameter:", combo_param)

        spin_start = QDoubleSpinBox(); spin_start.setRange(-1e15, 1e15); spin_start.setDecimals(6)
        spin_end = QDoubleSpinBox(); spin_end.setRange(-1e15, 1e15); spin_end.setDecimals(6)
        spin_steps = QSpinBox(); spin_steps.setRange(3, 50); spin_steps.setValue(8)

        # Default ranges
        if problem == 'Heat Conduction':
            spin_start.setValue(0.1); spin_end.setValue(10.0)
        elif problem == 'Structural':
            spin_start.setValue(1e9); spin_end.setValue(1e12)
        else:
            spin_start.setValue(1e-12); spin_end.setValue(1e-10)

        flay.addRow("Start:", spin_start)
        flay.addRow("End:", spin_end)
        flay.addRow("Steps:", spin_steps)

        bbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bbox.accepted.connect(dlg.accept)
        bbox.rejected.connect(dlg.reject)
        flay.addRow(bbox)

        if dlg.exec_() != QDialog.Accepted:
            return

        param_name = combo_param.currentText()
        p_start = spin_start.value()
        p_end = spin_end.value()
        n_steps = spin_steps.value()

        values = np.linspace(p_start, p_end, n_steps)
        max_vals, min_vals, mean_vals = [], [], []

        self._log(f"Parametric study: sweeping '{param_name}' from {p_start} to {p_end} ({n_steps} steps)")

        for val in values:
            params = {
                'problem': self.problem_combo.currentText(),
                'geometry': self.geom_combo.currentText(),
                'width': self.width_spin.value(),
                'height': self.height_spin.value(),
                'resolution': self.res_spin.value(),
                'conductivity': self.cond_spin.value(),
                'permittivity': self.perm_spin.value(),
                'youngs_modulus': self.E_spin.value(),
                'poisson_ratio': self.nu_spin.value(),
                'struct_type': self.struct_type_combo.currentText(),
                'source': self.source_spin.value(),
                'body_fx': self.body_fx_spin.value(),
                'body_fy': self.body_fy_spin.value(),
                'bc_specs': [row.get_spec() for row in self._bc_rows],
            }
            if param_name == 'resolution':
                params['resolution'] = int(val)
            else:
                params[param_name] = val

            try:
                thread = SolverThread(params)
                thread.start()
                thread.wait()
                if hasattr(thread, '_result_data'):
                    r = thread._result_data
                else:
                    continue
            except Exception:
                # Run synchronously as fallback
                geom = params['geometry']
                res = params['resolution']
                if geom == 'Rectangular':
                    nodes, elems, bnd = generate_rectangular_mesh(params['width'], params['height'], res, res)
                elif geom == 'L-shaped':
                    nodes, elems, bnd = generate_l_shaped_mesh(params['width'], res)
                else:
                    nodes, elems, bnd = generate_circular_mesh(params['width'] / 2.0, res)

                prob = params['problem']
                if prob in ('Heat Conduction', 'Electrostatics'):
                    prop = params['conductivity'] if prob == 'Heat Conduction' else params['permittivity']
                    K, F = assemble_scalar(nodes, elems, k=prop, source=params.get('source', 0.0))
                else:
                    K, F = assemble_structural(nodes, elems, params['youngs_modulus'], params['poisson_ratio'],
                                               plane_stress=(params['struct_type'] == 'Plane Stress'),
                                               body_force=(params.get('body_fx', 0.0), params.get('body_fy', 0.0)))

                bc_specs = params.get('bc_specs', [])
                if prob in ('Heat Conduction', 'Electrostatics'):
                    for bc in bc_specs:
                        edge_key = bc['edge']
                        matching = [v for k, v in bnd.items() if edge_key.lower() in k.lower()]
                        if not matching:
                            continue
                        edge_nodes = matching[0]
                        if bc['type'] == 'Dirichlet':
                            K, F = apply_dirichlet_scalar(K, F, edge_nodes, bc['value'])
                        else:
                            F = apply_neumann_scalar(nodes, F, edge_nodes, bc['value'])
                else:
                    for bc in bc_specs:
                        edge_key = bc['edge']
                        matching = [v for k, v in bnd.items() if edge_key.lower() in k.lower()]
                        if not matching:
                            continue
                        edge_nodes = matching[0]
                        comp = bc.get('component', 0)
                        if bc['type'] == 'Dirichlet':
                            K, F = apply_dirichlet_structural(K, F, edge_nodes, comp, bc['value'])

                u = spsolve(K.tocsc(), F)
                if prob == 'Structural':
                    u_field = np.sqrt(u[0::2]**2 + u[1::2]**2)
                else:
                    u_field = u
                max_vals.append(u_field.max())
                min_vals.append(u_field.min())
                mean_vals.append(u_field.mean())
                continue
            max_vals.append(0.0); min_vals.append(0.0); mean_vals.append(0.0)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        ax.plot(values, max_vals, 'r-o', label='Max', markersize=4)
        ax.plot(values, min_vals, 'b-s', label='Min', markersize=4)
        ax.plot(values, mean_vals, 'g-^', label='Mean', markersize=4)
        ax.set_xlabel(param_name, fontsize=11)
        ax.set_ylabel('Solution Value', fontsize=11)
        ax.set_title(f'Parametric Study: {param_name}', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.show()
        self._log(f"Parametric study complete: {n_steps} runs.")

    # -- Convergence study --------------------------------------------------

    def _on_convergence_study(self):
        """Run the solver with increasing mesh density and plot error vs mesh size."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Convergence Study")
        dlg.resize(300, 150)
        flay = QFormLayout(dlg)

        spin_min = QSpinBox(); spin_min.setRange(3, 50); spin_min.setValue(5)
        spin_max = QSpinBox(); spin_max.setRange(10, 200); spin_max.setValue(40)
        spin_n = QSpinBox(); spin_n.setRange(3, 20); spin_n.setValue(6)
        flay.addRow("Min resolution:", spin_min)
        flay.addRow("Max resolution:", spin_max)
        flay.addRow("Number of runs:", spin_n)

        bbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bbox.accepted.connect(dlg.accept)
        bbox.rejected.connect(dlg.reject)
        flay.addRow(bbox)

        if dlg.exec_() != QDialog.Accepted:
            return

        res_min = spin_min.value()
        res_max = spin_max.value()
        n_runs = spin_n.value()
        resolutions = np.linspace(res_min, res_max, n_runs, dtype=int)

        h_values = []       # characteristic mesh size
        max_vals = []       # max solution value (used as convergence metric)
        n_dofs_list = []

        self._log(f"Convergence study: resolutions {list(resolutions)}")

        prev_max = None
        errors = []

        for res in resolutions:
            params = {
                'problem': self.problem_combo.currentText(),
                'geometry': self.geom_combo.currentText(),
                'width': self.width_spin.value(),
                'height': self.height_spin.value(),
                'resolution': int(res),
                'conductivity': self.cond_spin.value(),
                'permittivity': self.perm_spin.value(),
                'youngs_modulus': self.E_spin.value(),
                'poisson_ratio': self.nu_spin.value(),
                'struct_type': self.struct_type_combo.currentText(),
                'source': self.source_spin.value(),
                'body_fx': self.body_fx_spin.value(),
                'body_fy': self.body_fy_spin.value(),
                'bc_specs': [row.get_spec() for row in self._bc_rows],
            }

            geom = params['geometry']
            if geom == 'Rectangular':
                nodes, elems, bnd = generate_rectangular_mesh(params['width'], params['height'], int(res), int(res))
            elif geom == 'L-shaped':
                nodes, elems, bnd = generate_l_shaped_mesh(params['width'], int(res))
            else:
                nodes, elems, bnd = generate_circular_mesh(params['width'] / 2.0, int(res))

            prob = params['problem']
            if prob in ('Heat Conduction', 'Electrostatics'):
                prop = params['conductivity'] if prob == 'Heat Conduction' else params['permittivity']
                K, F = assemble_scalar(nodes, elems, k=prop, source=params.get('source', 0.0))
            else:
                K, F = assemble_structural(nodes, elems, params['youngs_modulus'], params['poisson_ratio'],
                                           plane_stress=(params['struct_type'] == 'Plane Stress'),
                                           body_force=(params.get('body_fx', 0.0), params.get('body_fy', 0.0)))

            bc_specs = params.get('bc_specs', [])
            if prob in ('Heat Conduction', 'Electrostatics'):
                for bc in bc_specs:
                    edge_key = bc['edge']
                    matching = [v for k, v in bnd.items() if edge_key.lower() in k.lower()]
                    if not matching:
                        continue
                    edge_nodes = matching[0]
                    if bc['type'] == 'Dirichlet':
                        K, F = apply_dirichlet_scalar(K, F, edge_nodes, bc['value'])
                    else:
                        F = apply_neumann_scalar(nodes, F, edge_nodes, bc['value'])
            else:
                for bc in bc_specs:
                    edge_key = bc['edge']
                    matching = [v for k, v in bnd.items() if edge_key.lower() in k.lower()]
                    if not matching:
                        continue
                    edge_nodes = matching[0]
                    comp = bc.get('component', 0)
                    if bc['type'] == 'Dirichlet':
                        K, F = apply_dirichlet_structural(K, F, edge_nodes, comp, bc['value'])

            u = spsolve(K.tocsc(), F)
            n_dof = len(u)

            if prob == 'Structural':
                u_field = np.sqrt(u[0::2]**2 + u[1::2]**2)
            else:
                u_field = u

            cur_max = u_field.max()
            h = params['width'] / float(res)
            h_values.append(h)
            max_vals.append(cur_max)
            n_dofs_list.append(n_dof)

            if prev_max is not None:
                errors.append(abs(cur_max - prev_max))
            prev_max = cur_max
            self._log(f"  res={res}, h={h:.4f}, n_dof={n_dof}, max={cur_max:.6e}")

        # Plot convergence
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

        ax1.plot(h_values, max_vals, 'b-o', markersize=5)
        ax1.set_xlabel('Mesh size h', fontsize=11)
        ax1.set_ylabel('Max solution value', fontsize=11)
        ax1.set_title('Solution vs Mesh Size', fontsize=13)
        ax1.grid(True, alpha=0.3)

        if errors:
            ax2.loglog(h_values[1:], errors, 'r-s', markersize=5)
            ax2.set_xlabel('Mesh size h', fontsize=11)
            ax2.set_ylabel('Change in max value', fontsize=11)
            ax2.set_title('Convergence (successive difference)', fontsize=13)
            ax2.grid(True, which='both', alpha=0.3)

            # Estimate convergence rate
            if len(errors) >= 2:
                log_h = np.log(np.array(h_values[1:]))
                log_e = np.log(np.array(errors) + 1e-30)
                valid = np.isfinite(log_e)
                if valid.sum() >= 2:
                    p = np.polyfit(log_h[valid], log_e[valid], 1)
                    ax2.set_title(f'Convergence rate ~ O(h^{p[0]:.2f})', fontsize=13)

        fig.tight_layout()
        fig.show()
        self._log("Convergence study complete.")

    # -- End of FEMSolverWidget new methods ----------------------------------


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    w = FEMSolverWidget()
    w.setWindowTitle("FEM Solver")
    w.resize(1200, 800)
    w.set_logger(lambda msg: print(f"[LOG] {msg}"))
    w.show()
    sys.exit(app.exec_())
