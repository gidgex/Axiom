"""
Tensor Calculator Widget for PyQt5 Scientific Suite.
Provides Einstein summation, index operations, metric tensors,
Christoffel symbols, Levi-Civita, Voigt notation, stress/strain
analysis, and common tensor algebra operations.
"""

import numpy as np
import json
from itertools import product as iter_product
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QPushButton, QSplitter, QFormLayout, QTabWidget,
    QTextEdit, QLineEdit, QSpinBox, QCheckBox, QTableWidget,
    QTableWidgetItem, QMessageBox, QHeaderView, QDoubleSpinBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

try:
    from sympy import symbols, diff, Rational, Array, simplify
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False


# ---------------------------------------------------------------------------
# Preset tensors
# ---------------------------------------------------------------------------

PAULI_MATRICES = {
    "sigma_1": np.array([[0, 1], [1, 0]], dtype=complex),
    "sigma_2": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "sigma_3": np.array([[1, 0], [0, -1]], dtype=complex),
}

GELL_MANN_MATRICES = {
    "lambda_1": np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex),
    "lambda_2": np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex),
    "lambda_3": np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex),
    "lambda_4": np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex),
    "lambda_5": np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=complex),
    "lambda_6": np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex),
    "lambda_7": np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=complex),
    "lambda_8": np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=complex) / np.sqrt(3),
}


def _identity(n):
    return np.eye(n)


def _levi_civita_3d():
    """Return the 3D Levi-Civita symbol as a rank-3 tensor."""
    eps = np.zeros((3, 3, 3), dtype=float)
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
    eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0
    return eps


def _levi_civita_4d():
    """Return the 4D Levi-Civita symbol as a rank-4 tensor."""
    eps = np.zeros((4, 4, 4, 4), dtype=float)
    for i, j, k, l in iter_product(range(4), repeat=4):
        idx = [i, j, k, l]
        if len(set(idx)) < 4:
            continue
        # count inversions
        inversions = 0
        for a in range(4):
            for b in range(a + 1, 4):
                if idx[a] > idx[b]:
                    inversions += 1
        eps[i, j, k, l] = (-1) ** inversions
    return eps


def _euclidean_metric(n=3):
    return np.eye(n)


def _minkowski_metric(signature="mostly_plus"):
    if signature == "mostly_plus":
        return np.diag([-1.0, 1.0, 1.0, 1.0])
    else:
        return np.diag([1.0, -1.0, -1.0, -1.0])


def _schwarzschild_metric(r, M=1.0, G=1.0, c=1.0):
    """Schwarzschild metric at radius r (diagonal, in Schwarzschild coords).
    Returns 4x4 diagonal metric g_{mu nu} for (t, r, theta, phi) at theta=pi/2."""
    rs = 2.0 * G * M / (c * c)
    if abs(r - rs) < 1e-12:
        r = rs + 1e-10
    f = 1.0 - rs / r
    return np.diag([-f * c * c, 1.0 / f, r * r, r * r])


def _flrw_metric(a, k=0.0, r=1.0):
    """FLRW metric for scale factor a, curvature k, at coordinate r, theta=pi/2.
    ds^2 = -dt^2 + a^2 [ dr^2/(1-kr^2) + r^2 dOmega^2 ]."""
    denom = 1.0 - k * r * r
    if abs(denom) < 1e-12:
        denom = 1e-10
    return np.diag([-1.0, a * a / denom, a * a * r * r, a * a * r * r])


def _cubic_elasticity_tensor(c11, c12, c44):
    """Build the rank-4 elasticity tensor C_{ijkl} for a cubic crystal.
    Uses Voigt mapping from 3 independent constants."""
    C = np.zeros((3, 3, 3, 3), dtype=float)
    delta = np.eye(3)
    for i, j, k, l in iter_product(range(3), repeat=4):
        C[i, j, k, l] = (
            c12 * delta[i, j] * delta[k, l]
            + c44 * (delta[i, k] * delta[j, l] + delta[i, l] * delta[j, k])
        )
        if i == j == k == l:
            C[i, j, k, l] = c11
        elif i == j and k == l and i != k:
            C[i, j, k, l] = c12
        elif ((i == k and j == l) or (i == l and j == k)) and i != j:
            C[i, j, k, l] = c44
    return C


# ---------------------------------------------------------------------------
# Voigt notation helpers
# ---------------------------------------------------------------------------

_VOIGT_MAP = {(0, 0): 0, (1, 1): 1, (2, 2): 2,
              (1, 2): 3, (2, 1): 3, (0, 2): 4, (2, 0): 4, (0, 1): 5, (1, 0): 5}

_VOIGT_INV = {0: (0, 0), 1: (1, 1), 2: (2, 2),
              3: (1, 2), 4: (0, 2), 5: (0, 1)}


def tensor_to_voigt_2(T):
    """Convert a 3x3 symmetric tensor to 6-component Voigt vector."""
    v = np.zeros(6)
    for (i, j), vi in _VOIGT_MAP.items():
        if i == j:
            v[vi] = T[i, j]
        else:
            v[vi] = T[i, j]
    return v


def voigt_to_tensor_2(v):
    """Convert 6-component Voigt vector to 3x3 symmetric tensor."""
    T = np.zeros((3, 3))
    for vi, (i, j) in _VOIGT_INV.items():
        T[i, j] = v[vi]
        T[j, i] = v[vi]
    return T


def tensor4_to_voigt_matrix(C):
    """Convert rank-4 tensor C_{ijkl} (3x3x3x3) to 6x6 Voigt matrix."""
    V = np.zeros((6, 6))
    for I in range(6):
        i, j = _VOIGT_INV[I]
        for J in range(6):
            k, l = _VOIGT_INV[J]
            V[I, J] = C[i, j, k, l]
    return V


def voigt_matrix_to_tensor4(V):
    """Convert 6x6 Voigt matrix to rank-4 tensor C_{ijkl}."""
    C = np.zeros((3, 3, 3, 3))
    for I in range(6):
        i, j = _VOIGT_INV[I]
        for J in range(6):
            k, l = _VOIGT_INV[J]
            C[i, j, k, l] = V[I, J]
            C[j, i, k, l] = V[I, J]
            C[i, j, l, k] = V[I, J]
            C[j, i, l, k] = V[I, J]
    return C


# ---------------------------------------------------------------------------
# Stress / strain helpers
# ---------------------------------------------------------------------------

def principal_stresses(sigma):
    """Compute principal stresses from 3x3 stress tensor."""
    eigvals = np.linalg.eigvalsh(sigma)
    return np.sort(eigvals)[::-1]


def von_mises_stress(sigma):
    """Compute von Mises equivalent stress from 3x3 stress tensor."""
    s = sigma - np.trace(sigma) / 3.0 * np.eye(3)
    return np.sqrt(1.5 * np.sum(s * s))


def hydrostatic_stress(sigma):
    """Compute hydrostatic stress."""
    return np.trace(sigma) / 3.0


def stress_invariants(sigma):
    """Compute the three invariants I1, I2, I3 of a 3x3 stress tensor."""
    I1 = np.trace(sigma)
    I2 = 0.5 * (np.trace(sigma) ** 2 - np.trace(sigma @ sigma))
    I3 = np.linalg.det(sigma)
    return I1, I2, I3


# ---------------------------------------------------------------------------
# Christoffel symbols (numerical)
# ---------------------------------------------------------------------------

def christoffel_symbols_numerical(metric_func, coords, h=1e-6):
    """Compute Christoffel symbols Gamma^sigma_{mu nu} numerically.

    Parameters
    ----------
    metric_func : callable
        Function(coords) -> NxN metric tensor at given coordinates.
    coords : array-like
        Coordinate values at which to evaluate.
    h : float
        Finite difference step size.

    Returns
    -------
    Gamma : ndarray of shape (N, N, N)
        Christoffel symbols Gamma^sigma_{mu nu}.
    """
    coords = np.asarray(coords, dtype=float)
    g = metric_func(coords)
    n = len(coords)
    g_inv = np.linalg.inv(g)

    # partial derivatives of metric: dg[alpha][mu][nu] = dg_{mu nu}/dx^alpha
    dg = np.zeros((n, n, n))
    for alpha in range(n):
        coords_plus = coords.copy()
        coords_minus = coords.copy()
        coords_plus[alpha] += h
        coords_minus[alpha] -= h
        g_plus = metric_func(coords_plus)
        g_minus = metric_func(coords_minus)
        dg[alpha] = (g_plus - g_minus) / (2.0 * h)

    Gamma = np.zeros((n, n, n))
    for sigma in range(n):
        for mu in range(n):
            for nu in range(n):
                s = 0.0
                for lam in range(n):
                    s += 0.5 * g_inv[sigma, lam] * (
                        dg[mu][lam, nu] + dg[nu][lam, mu] - dg[lam][mu, nu]
                    )
                Gamma[sigma, mu, nu] = s
    return Gamma


# ---------------------------------------------------------------------------
# Tensor operations
# ---------------------------------------------------------------------------

def tensor_trace(T, axis1=0, axis2=1):
    return np.trace(T, axis1=axis1, axis2=axis2)


def tensor_transpose(T, axes=None):
    if axes is None:
        axes = list(reversed(range(T.ndim)))
    return np.transpose(T, axes)


def symmetrize(T):
    """Symmetrize a rank-2 tensor."""
    return 0.5 * (T + T.T)


def antisymmetrize(T):
    """Antisymmetrize a rank-2 tensor."""
    return 0.5 * (T - T.T)


def raise_index(T, metric_inv, index=0):
    """Raise an index of tensor T using the inverse metric.
    T with lower index -> T with upper index via g^{ab} T_b..."""
    return np.tensordot(metric_inv, T, axes=([1], [index]))


def lower_index(T, metric, index=0):
    """Lower an index of tensor T using the metric.
    T with upper index -> T with lower index via g_{ab} T^b..."""
    return np.tensordot(metric, T, axes=([1], [index]))


def contract_indices(T, axis1, axis2):
    """Contract (trace over) two indices of a tensor."""
    return np.trace(T, axis1=axis1, axis2=axis2)


def cross_product_levi_civita(a, b):
    """Compute cross product of 3-vectors using Levi-Civita contraction."""
    eps = _levi_civita_3d()
    return np.einsum('ijk,j,k', eps, a, b)


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class TensorCalcWidget(QWidget):
    """Tensor calculator widget for PyQt5 scientific suite."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._logger = None
        self._tensors = {}  # name -> ndarray storage
        self._init_ui()

    # -- public API ----------------------------------------------------------

    def set_logger(self, fn):
        """Set the logging callback function."""
        self._logger = fn

    def run(self):
        """Activate / refresh the widget."""
        self._log("Tensor Calculator ready.")

    # -- logging -------------------------------------------------------------

    def _log(self, msg):
        if self._logger:
            self._logger(str(msg))
        self.output.append(str(msg))

    # -- UI setup ------------------------------------------------------------

    def _init_ui(self):
        root = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # Left panel: controls
        left = QWidget()
        left_layout = QVBoxLayout(left)
        tabs = QTabWidget()
        left_layout.addWidget(tabs)

        tabs.addTab(self._build_input_tab(), "Input")
        tabs.addTab(self._build_einsum_tab(), "Einstein Sum")
        tabs.addTab(self._build_index_tab(), "Index Ops")
        tabs.addTab(self._build_metric_tab(), "Metrics")
        tabs.addTab(self._build_christoffel_tab(), "Christoffel")
        tabs.addTab(self._build_levi_civita_tab(), "Levi-Civita")
        tabs.addTab(self._build_products_tab(), "Products")
        tabs.addTab(self._build_common_tab(), "Common Ops")
        tabs.addTab(self._build_voigt_tab(), "Voigt")
        tabs.addTab(self._build_stress_tab(), "Stress/Strain")
        tabs.addTab(self._build_presets_tab(), "Presets")

        splitter.addWidget(left)

        # Right panel: output
        right = QWidget()
        right_layout = QVBoxLayout(right)
        lbl = QLabel("Output")
        lbl.setStyleSheet("font-weight: bold; font-size: 13px;")
        right_layout.addWidget(lbl)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QFont("Consolas", 10))
        right_layout.addWidget(self.output)

        btn_clear = QPushButton("Clear Output")
        btn_clear.clicked.connect(self.output.clear)
        right_layout.addWidget(btn_clear)

        self.tensor_table = QTableWidget()
        self.tensor_table.setColumnCount(3)
        self.tensor_table.setHorizontalHeaderLabels(["Name", "Shape", "Rank"])
        self.tensor_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        right_layout.addWidget(QLabel("Stored Tensors:"))
        right_layout.addWidget(self.tensor_table)

        splitter.addWidget(right)
        splitter.setSizes([520, 480])

    # -- Tab builders --------------------------------------------------------

    def _build_input_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        grp = QGroupBox("Tensor Input")
        form = QFormLayout()

        self.input_name = QLineEdit("T")
        form.addRow("Name:", self.input_name)

        self.input_data = QTextEdit()
        self.input_data.setPlaceholderText(
            "Enter as nested list, e.g.:\n"
            "  Scalar: 5\n"
            "  Vector: [1, 2, 3]\n"
            "  Matrix: [[1,0],[0,1]]\n"
            "  Rank-3: [[[1,0],[0,1]],[[0,1],[1,0]]]"
        )
        self.input_data.setMaximumHeight(160)
        form.addRow("Data:", self.input_data)

        btn_store = QPushButton("Store Tensor")
        btn_store.clicked.connect(self._store_tensor)
        form.addRow(btn_store)

        btn_show = QPushButton("Display Tensor")
        btn_show.clicked.connect(self._display_current_tensor)
        form.addRow(btn_show)

        grp.setLayout(form)
        layout.addWidget(grp)
        layout.addStretch()
        return w

    def _build_einsum_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        grp = QGroupBox("Einstein Summation")
        form = QFormLayout()

        self.einsum_expr = QLineEdit("ij,jk->ik")
        form.addRow("Expression:", self.einsum_expr)

        self.einsum_operands = QLineEdit("A,B")
        self.einsum_operands.setToolTip("Comma-separated tensor names from storage")
        form.addRow("Operands:", self.einsum_operands)

        self.einsum_result_name = QLineEdit("result")
        form.addRow("Result name:", self.einsum_result_name)

        btn = QPushButton("Compute einsum")
        btn.clicked.connect(self._compute_einsum)
        form.addRow(btn)

        grp.setLayout(form)
        layout.addWidget(grp)

        # Quick examples
        eg = QGroupBox("Quick Examples")
        eg_layout = QVBoxLayout()
        examples = [
            ("Matrix multiply", "ij,jk->ik", "A,B"),
            ("Trace", "ii->", "A"),
            ("Outer product", "i,j->ij", "u,v"),
            ("Dot product", "i,i->", "u,v"),
            ("Batch matmul", "bij,bjk->bik", "A,B"),
            ("Tensor contraction", "ijkl,jl->ik", "C,g"),
        ]
        for label, expr, ops in examples:
            b = QPushButton(f"{label}:  {expr}")
            b.clicked.connect(lambda _, e=expr, o=ops: self._set_einsum(e, o))
            eg_layout.addWidget(b)
        eg.setLayout(eg_layout)
        layout.addWidget(eg)
        layout.addStretch()
        return w

    def _build_index_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        grp = QGroupBox("Raise / Lower Index")
        form = QFormLayout()

        self.idx_tensor = QLineEdit("T")
        form.addRow("Tensor:", self.idx_tensor)

        self.idx_metric = QLineEdit("g")
        form.addRow("Metric:", self.idx_metric)

        self.idx_which = QSpinBox()
        self.idx_which.setRange(0, 10)
        form.addRow("Index position:", self.idx_which)

        self.idx_result = QLineEdit("T_raised")
        form.addRow("Result name:", self.idx_result)

        btn_raise = QPushButton("Raise Index")
        btn_raise.clicked.connect(self._raise_index)
        form.addRow(btn_raise)

        btn_lower = QPushButton("Lower Index")
        btn_lower.clicked.connect(self._lower_index)
        form.addRow(btn_lower)

        grp.setLayout(form)
        layout.addWidget(grp)

        grp2 = QGroupBox("Contract Indices")
        form2 = QFormLayout()
        self.contract_tensor = QLineEdit("T")
        form2.addRow("Tensor:", self.contract_tensor)
        self.contract_ax1 = QSpinBox()
        self.contract_ax1.setRange(0, 10)
        form2.addRow("Axis 1:", self.contract_ax1)
        self.contract_ax2 = QSpinBox()
        self.contract_ax2.setRange(0, 10)
        self.contract_ax2.setValue(1)
        form2.addRow("Axis 2:", self.contract_ax2)
        self.contract_result = QLineEdit("T_contracted")
        form2.addRow("Result name:", self.contract_result)
        btn_contract = QPushButton("Contract")
        btn_contract.clicked.connect(self._contract_indices)
        form2.addRow(btn_contract)
        grp2.setLayout(form2)
        layout.addWidget(grp2)

        layout.addStretch()
        return w

    def _build_metric_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        grp = QGroupBox("Built-in Metric Tensors")
        form = QFormLayout()

        self.metric_type = QComboBox()
        self.metric_type.addItems([
            "Euclidean (3D)", "Euclidean (4D)",
            "Minkowski (mostly +)", "Minkowski (mostly -)",
            "Schwarzschild", "FLRW"
        ])
        form.addRow("Metric:", self.metric_type)

        self.metric_name = QLineEdit("g")
        form.addRow("Store as:", self.metric_name)

        # Schwarzschild params
        self.schw_r = QDoubleSpinBox()
        self.schw_r.setRange(0.01, 1e10)
        self.schw_r.setValue(10.0)
        self.schw_r.setDecimals(4)
        form.addRow("r (Schwarzschild):", self.schw_r)

        self.schw_M = QDoubleSpinBox()
        self.schw_M.setRange(0.001, 1e10)
        self.schw_M.setValue(1.0)
        self.schw_M.setDecimals(4)
        form.addRow("M (Schwarzschild):", self.schw_M)

        # FLRW params
        self.flrw_a = QDoubleSpinBox()
        self.flrw_a.setRange(0.001, 1e6)
        self.flrw_a.setValue(1.0)
        self.flrw_a.setDecimals(4)
        form.addRow("a (FLRW scale):", self.flrw_a)

        self.flrw_k = QDoubleSpinBox()
        self.flrw_k.setRange(-1.0, 1.0)
        self.flrw_k.setValue(0.0)
        self.flrw_k.setDecimals(4)
        form.addRow("k (FLRW curvature):", self.flrw_k)

        btn = QPushButton("Generate & Store")
        btn.clicked.connect(self._generate_metric)
        form.addRow(btn)

        grp.setLayout(form)
        layout.addWidget(grp)
        layout.addStretch()
        return w

    def _build_christoffel_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        grp = QGroupBox("Christoffel Symbols (Numerical)")
        form = QFormLayout()

        info = QLabel(
            "Computes Gamma^sigma_{mu nu} numerically\n"
            "from a stored diagonal metric at given coordinates."
        )
        info.setWordWrap(True)
        form.addRow(info)

        self.chris_metric_type = QComboBox()
        self.chris_metric_type.addItems([
            "Schwarzschild", "FLRW"
        ])
        form.addRow("Metric type:", self.chris_metric_type)

        self.chris_coords = QLineEdit("10.0, 1.0, 1.5708, 0.0")
        self.chris_coords.setToolTip("Coordinate values (comma-separated)")
        form.addRow("Coordinates:", self.chris_coords)

        self.chris_M = QDoubleSpinBox()
        self.chris_M.setRange(0.001, 1e10)
        self.chris_M.setValue(1.0)
        self.chris_M.setDecimals(4)
        form.addRow("M / a parameter:", self.chris_M)

        self.chris_result = QLineEdit("Gamma")
        form.addRow("Store as:", self.chris_result)

        btn = QPushButton("Compute Christoffel")
        btn.clicked.connect(self._compute_christoffel)
        form.addRow(btn)

        grp.setLayout(form)
        layout.addWidget(grp)
        layout.addStretch()
        return w

    def _build_levi_civita_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        grp = QGroupBox("Levi-Civita Symbol")
        form = QFormLayout()

        self.lc_dim = QComboBox()
        self.lc_dim.addItems(["3D", "4D"])
        form.addRow("Dimension:", self.lc_dim)

        self.lc_name = QLineEdit("eps")
        form.addRow("Store as:", self.lc_name)

        btn = QPushButton("Generate Levi-Civita")
        btn.clicked.connect(self._generate_levi_civita)
        form.addRow(btn)

        grp.setLayout(form)
        layout.addWidget(grp)

        grp2 = QGroupBox("Cross Product via Levi-Civita")
        form2 = QFormLayout()
        self.cross_a = QLineEdit("u")
        form2.addRow("Vector a:", self.cross_a)
        self.cross_b = QLineEdit("v")
        form2.addRow("Vector b:", self.cross_b)
        self.cross_result = QLineEdit("cross")
        form2.addRow("Result name:", self.cross_result)
        btn2 = QPushButton("Compute Cross Product")
        btn2.clicked.connect(self._compute_cross_product)
        form2.addRow(btn2)
        grp2.setLayout(form2)
        layout.addWidget(grp2)

        layout.addStretch()
        return w

    def _build_products_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        grp = QGroupBox("Tensor Products")
        form = QFormLayout()

        self.prod_a = QLineEdit("A")
        form.addRow("Tensor A:", self.prod_a)
        self.prod_b = QLineEdit("B")
        form.addRow("Tensor B:", self.prod_b)
        self.prod_result = QLineEdit("result")
        form.addRow("Result name:", self.prod_result)

        btn_outer = QPushButton("Outer Product (A x B)")
        btn_outer.clicked.connect(self._outer_product)
        form.addRow(btn_outer)

        btn_kron = QPushButton("Kronecker Product (A kron B)")
        btn_kron.clicked.connect(self._kronecker_product)
        form.addRow(btn_kron)

        grp.setLayout(form)
        layout.addWidget(grp)

        grp2 = QGroupBox("Tensor Contraction")
        form2 = QFormLayout()
        self.tc_tensor = QLineEdit("T")
        form2.addRow("Tensor:", self.tc_tensor)
        self.tc_ax1 = QSpinBox()
        self.tc_ax1.setRange(0, 10)
        form2.addRow("Axis 1:", self.tc_ax1)
        self.tc_ax2 = QSpinBox()
        self.tc_ax2.setRange(0, 10)
        self.tc_ax2.setValue(1)
        form2.addRow("Axis 2:", self.tc_ax2)
        self.tc_result = QLineEdit("contracted")
        form2.addRow("Result name:", self.tc_result)
        btn_tc = QPushButton("Contract")
        btn_tc.clicked.connect(self._tensor_contraction)
        form2.addRow(btn_tc)
        grp2.setLayout(form2)
        layout.addWidget(grp2)

        layout.addStretch()
        return w

    def _build_common_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        grp = QGroupBox("Common Tensor Operations")
        form = QFormLayout()

        self.common_tensor = QLineEdit("A")
        form.addRow("Tensor:", self.common_tensor)
        self.common_result = QLineEdit("result")
        form.addRow("Result name:", self.common_result)

        btn_trace = QPushButton("Trace")
        btn_trace.clicked.connect(lambda: self._common_op("trace"))
        form.addRow(btn_trace)

        btn_trans = QPushButton("Transpose")
        btn_trans.clicked.connect(lambda: self._common_op("transpose"))
        form.addRow(btn_trans)

        btn_sym = QPushButton("Symmetrize")
        btn_sym.clicked.connect(lambda: self._common_op("symmetrize"))
        form.addRow(btn_sym)

        btn_antisym = QPushButton("Antisymmetrize")
        btn_antisym.clicked.connect(lambda: self._common_op("antisymmetrize"))
        form.addRow(btn_antisym)

        btn_det = QPushButton("Determinant")
        btn_det.clicked.connect(lambda: self._common_op("determinant"))
        form.addRow(btn_det)

        btn_inv = QPushButton("Inverse")
        btn_inv.clicked.connect(lambda: self._common_op("inverse"))
        form.addRow(btn_inv)

        btn_eig = QPushButton("Eigenvalues")
        btn_eig.clicked.connect(lambda: self._common_op("eigenvalues"))
        form.addRow(btn_eig)

        btn_norm = QPushButton("Frobenius Norm")
        btn_norm.clicked.connect(lambda: self._common_op("norm"))
        form.addRow(btn_norm)

        grp.setLayout(form)
        layout.addWidget(grp)
        layout.addStretch()
        return w

    def _build_voigt_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        grp = QGroupBox("Voigt Notation Conversion")
        form = QFormLayout()

        info = QLabel(
            "Convert between full tensor notation and\n"
            "Voigt notation (for elasticity / engineering).\n"
            "Rank-2: 3x3 <-> 6-vector\n"
            "Rank-4: 3x3x3x3 <-> 6x6 matrix"
        )
        info.setWordWrap(True)
        form.addRow(info)

        self.voigt_tensor = QLineEdit("T")
        form.addRow("Tensor name:", self.voigt_tensor)
        self.voigt_result = QLineEdit("V")
        form.addRow("Result name:", self.voigt_result)

        btn_to = QPushButton("Tensor -> Voigt")
        btn_to.clicked.connect(self._to_voigt)
        form.addRow(btn_to)

        btn_from = QPushButton("Voigt -> Tensor")
        btn_from.clicked.connect(self._from_voigt)
        form.addRow(btn_from)

        grp.setLayout(form)
        layout.addWidget(grp)
        layout.addStretch()
        return w

    def _build_stress_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        grp = QGroupBox("Stress / Strain Analysis")
        form = QFormLayout()

        info = QLabel(
            "Enter or select a stored 3x3 stress tensor.\n"
            "Computes principal stresses, von Mises, hydrostatic, invariants."
        )
        info.setWordWrap(True)
        form.addRow(info)

        self.stress_tensor = QLineEdit("sigma")
        form.addRow("Stress tensor:", self.stress_tensor)

        btn = QPushButton("Analyze Stress Tensor")
        btn.clicked.connect(self._analyze_stress)
        form.addRow(btn)

        # Quick entry: symmetric 3x3
        grp2 = QGroupBox("Quick Stress Entry (symmetric)")
        g2 = QFormLayout()
        self.stress_entries = {}
        labels = ["sigma_11", "sigma_22", "sigma_33",
                  "sigma_12", "sigma_13", "sigma_23"]
        for lbl in labels:
            sb = QDoubleSpinBox()
            sb.setRange(-1e12, 1e12)
            sb.setDecimals(4)
            sb.setValue(0.0)
            self.stress_entries[lbl] = sb
            g2.addRow(lbl + ":", sb)
        btn_quick = QPushButton("Store & Analyze")
        btn_quick.clicked.connect(self._quick_stress)
        g2.addRow(btn_quick)
        grp2.setLayout(g2)

        grp.setLayout(form)
        layout.addWidget(grp)
        layout.addWidget(grp2)
        layout.addStretch()
        return w

    def _build_presets_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        grp = QGroupBox("Preset Tensors")
        g_layout = QVBoxLayout()

        presets = [
            ("Identity 2x2", lambda: _identity(2)),
            ("Identity 3x3", lambda: _identity(3)),
            ("Identity 4x4", lambda: _identity(4)),
            ("Pauli sigma_1", lambda: PAULI_MATRICES["sigma_1"]),
            ("Pauli sigma_2", lambda: PAULI_MATRICES["sigma_2"]),
            ("Pauli sigma_3", lambda: PAULI_MATRICES["sigma_3"]),
            ("Gell-Mann lambda_1", lambda: GELL_MANN_MATRICES["lambda_1"]),
            ("Gell-Mann lambda_2", lambda: GELL_MANN_MATRICES["lambda_2"]),
            ("Gell-Mann lambda_3", lambda: GELL_MANN_MATRICES["lambda_3"]),
            ("Gell-Mann lambda_4", lambda: GELL_MANN_MATRICES["lambda_4"]),
            ("Gell-Mann lambda_5", lambda: GELL_MANN_MATRICES["lambda_5"]),
            ("Gell-Mann lambda_6", lambda: GELL_MANN_MATRICES["lambda_6"]),
            ("Gell-Mann lambda_7", lambda: GELL_MANN_MATRICES["lambda_7"]),
            ("Gell-Mann lambda_8", lambda: GELL_MANN_MATRICES["lambda_8"]),
            ("Levi-Civita 3D", _levi_civita_3d),
            ("Levi-Civita 4D", _levi_civita_4d),
            ("Cubic Elasticity (Cu)", lambda: _cubic_elasticity_tensor(168.4, 121.4, 75.4)),
            ("Cubic Elasticity (Fe)", lambda: _cubic_elasticity_tensor(231.4, 134.7, 116.4)),
        ]

        self.preset_name = QLineEdit("preset")
        g_layout.addWidget(QLabel("Store as:"))
        g_layout.addWidget(self.preset_name)

        for label, factory in presets:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, f=factory, l=label: self._load_preset(f, l))
            g_layout.addWidget(btn)

        grp.setLayout(g_layout)
        layout.addWidget(grp)
        layout.addStretch()
        return w

    # -- Handlers ------------------------------------------------------------

    def _store_tensor(self):
        name = self.input_name.text().strip()
        if not name:
            self._log("Error: tensor name cannot be empty.")
            return
        text = self.input_data.toPlainText().strip()
        if not text:
            self._log("Error: no data entered.")
            return
        try:
            data = json.loads(text)
            arr = np.array(data, dtype=complex if 'j' in text else float)
        except Exception:
            try:
                arr = np.array(eval(text, {"__builtins__": {}},
                                    {"np": np, "j": 1j, "pi": np.pi, "e": np.e}))
            except Exception as exc:
                self._log(f"Error parsing tensor data: {exc}")
                return
        self._tensors[name] = arr
        self._log(f"Stored tensor '{name}': shape={arr.shape}, rank={arr.ndim}")
        self._display_tensor(name, arr)
        self._update_table()

    def _display_current_tensor(self):
        name = self.input_name.text().strip()
        if name not in self._tensors:
            self._log(f"Tensor '{name}' not found in storage.")
            return
        self._display_tensor(name, self._tensors[name])

    def _display_tensor(self, name, arr):
        self._log(f"\n--- {name} ---")
        self._log(f"Shape: {arr.shape}  Rank: {arr.ndim}")
        if arr.ndim == 0:
            self._log(f"  {arr.item()}")
        elif arr.ndim == 1:
            self._log(f"  {arr}")
        elif arr.ndim == 2:
            self._log(self._format_matrix(arr))
        elif arr.ndim == 3:
            for i in range(arr.shape[0]):
                self._log(f"  [{i}]:")
                self._log(self._format_matrix(arr[i], indent=4))
        elif arr.ndim == 4:
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    self._log(f"  [{i},{j}]:")
                    self._log(self._format_matrix(arr[i, j], indent=4))
        else:
            self._log(f"  (rank-{arr.ndim} tensor, showing flat):")
            self._log(f"  {arr}")
        self._log("")

    def _format_matrix(self, mat, indent=2):
        lines = []
        prefix = " " * indent
        for row in mat:
            entries = []
            for v in row:
                if np.iscomplex(v) and abs(v.imag) > 1e-14:
                    entries.append(f"{v:.6g}")
                else:
                    val = v.real if np.iscomplex(v) else v
                    entries.append(f"{val: .6g}")
            lines.append(prefix + "[ " + "  ".join(entries) + " ]")
        return "\n".join(lines)

    def _update_table(self):
        self.tensor_table.setRowCount(len(self._tensors))
        for i, (name, arr) in enumerate(self._tensors.items()):
            self.tensor_table.setItem(i, 0, QTableWidgetItem(name))
            self.tensor_table.setItem(i, 1, QTableWidgetItem(str(arr.shape)))
            self.tensor_table.setItem(i, 2, QTableWidgetItem(str(arr.ndim)))

    def _get_tensor(self, name):
        name = name.strip()
        if name not in self._tensors:
            self._log(f"Error: tensor '{name}' not found.")
            return None
        return self._tensors[name]

    # -- Einstein summation --------------------------------------------------

    def _set_einsum(self, expr, ops):
        self.einsum_expr.setText(expr)
        self.einsum_operands.setText(ops)

    def _compute_einsum(self):
        expr = self.einsum_expr.text().strip()
        op_names = [s.strip() for s in self.einsum_operands.text().split(",")]
        result_name = self.einsum_result_name.text().strip() or "result"

        operands = []
        for n in op_names:
            t = self._get_tensor(n)
            if t is None:
                return
            operands.append(t)

        try:
            result = np.einsum(expr, *operands)
        except Exception as exc:
            self._log(f"einsum error: {exc}")
            return

        result = np.array(result)
        self._tensors[result_name] = result
        self._log(f"einsum('{expr}', {', '.join(op_names)}) -> '{result_name}'")
        self._display_tensor(result_name, result)
        self._update_table()

    # -- Index operations ----------------------------------------------------

    def _raise_index(self):
        T = self._get_tensor(self.idx_tensor.text())
        g = self._get_tensor(self.idx_metric.text())
        if T is None or g is None:
            return
        idx = self.idx_which.value()
        name = self.idx_result.text().strip() or "raised"
        try:
            g_inv = np.linalg.inv(g)
            result = raise_index(T, g_inv, index=idx)
            self._tensors[name] = result
            self._log(f"Raised index {idx} of '{self.idx_tensor.text()}' -> '{name}'")
            self._display_tensor(name, result)
            self._update_table()
        except Exception as exc:
            self._log(f"Error raising index: {exc}")

    def _lower_index(self):
        T = self._get_tensor(self.idx_tensor.text())
        g = self._get_tensor(self.idx_metric.text())
        if T is None or g is None:
            return
        idx = self.idx_which.value()
        name = self.idx_result.text().strip() or "lowered"
        try:
            result = lower_index(T, g, index=idx)
            self._tensors[name] = result
            self._log(f"Lowered index {idx} of '{self.idx_tensor.text()}' -> '{name}'")
            self._display_tensor(name, result)
            self._update_table()
        except Exception as exc:
            self._log(f"Error lowering index: {exc}")

    def _contract_indices(self):
        T = self._get_tensor(self.contract_tensor.text())
        if T is None:
            return
        ax1 = self.contract_ax1.value()
        ax2 = self.contract_ax2.value()
        name = self.contract_result.text().strip() or "contracted"
        try:
            result = contract_indices(T, ax1, ax2)
            self._tensors[name] = result
            self._log(f"Contracted indices ({ax1},{ax2}) of '{self.contract_tensor.text()}' -> '{name}'")
            self._display_tensor(name, result)
            self._update_table()
        except Exception as exc:
            self._log(f"Error contracting: {exc}")

    # -- Metrics -------------------------------------------------------------

    def _generate_metric(self):
        choice = self.metric_type.currentText()
        name = self.metric_name.text().strip() or "g"
        try:
            if "Euclidean (3D)" in choice:
                g = _euclidean_metric(3)
            elif "Euclidean (4D)" in choice:
                g = _euclidean_metric(4)
            elif "mostly +" in choice:
                g = _minkowski_metric("mostly_plus")
            elif "mostly -" in choice:
                g = _minkowski_metric("mostly_minus")
            elif "Schwarzschild" in choice:
                g = _schwarzschild_metric(self.schw_r.value(), self.schw_M.value())
            elif "FLRW" in choice:
                g = _flrw_metric(self.flrw_a.value(), self.flrw_k.value())
            else:
                self._log("Unknown metric type.")
                return

            self._tensors[name] = g
            self._log(f"Generated {choice} metric -> '{name}'")
            self._display_tensor(name, g)
            self._update_table()
        except Exception as exc:
            self._log(f"Error generating metric: {exc}")

    # -- Christoffel ---------------------------------------------------------

    def _compute_christoffel(self):
        mtype = self.chris_metric_type.currentText()
        coords_text = self.chris_coords.text().strip()
        name = self.chris_result.text().strip() or "Gamma"
        M_val = self.chris_M.value()

        try:
            coords = np.array([float(x) for x in coords_text.split(",")])
        except Exception:
            self._log("Error parsing coordinates.")
            return

        if "Schwarzschild" in mtype:
            def metric_func(c):
                # c = [t, r, theta, phi] but metric only depends on r
                return _schwarzschild_metric(c[1], M=M_val)
        elif "FLRW" in mtype:
            def metric_func(c):
                return _flrw_metric(M_val, k=0.0, r=c[1])
        else:
            self._log("Unknown metric for Christoffel computation.")
            return

        try:
            Gamma = christoffel_symbols_numerical(metric_func, coords)
            self._tensors[name] = Gamma
            self._log(f"Christoffel symbols for {mtype} at coords={coords} -> '{name}'")
            self._log(f"Shape: {Gamma.shape}")

            # Display non-zero components
            n = Gamma.shape[0]
            coord_labels = ["t", "r", "theta", "phi"] if n == 4 else [str(i) for i in range(n)]
            count = 0
            for s in range(n):
                for m in range(n):
                    for nu in range(m, n):
                        val = Gamma[s, m, nu]
                        if abs(val) > 1e-12:
                            self._log(
                                f"  Gamma^{coord_labels[s]}_{{{coord_labels[m]}{coord_labels[nu]}}} "
                                f"= {val:.8g}"
                            )
                            count += 1
            if count == 0:
                self._log("  All components zero (flat space or coordinate artifact).")
            self._update_table()
        except Exception as exc:
            self._log(f"Error computing Christoffel symbols: {exc}")

    # -- Levi-Civita ---------------------------------------------------------

    def _generate_levi_civita(self):
        dim = self.lc_dim.currentText()
        name = self.lc_name.text().strip() or "eps"
        if "3D" in dim:
            eps = _levi_civita_3d()
        else:
            eps = _levi_civita_4d()
        self._tensors[name] = eps
        self._log(f"Generated {dim} Levi-Civita symbol -> '{name}', shape={eps.shape}")
        self._update_table()

    def _compute_cross_product(self):
        a = self._get_tensor(self.cross_a.text())
        b = self._get_tensor(self.cross_b.text())
        if a is None or b is None:
            return
        if a.shape != (3,) or b.shape != (3,):
            self._log("Error: both vectors must be 3D for cross product.")
            return
        name = self.cross_result.text().strip() or "cross"
        result = cross_product_levi_civita(a.real, b.real)
        self._tensors[name] = result
        self._log(f"Cross product {self.cross_a.text()} x {self.cross_b.text()} -> '{name}'")
        self._display_tensor(name, result)
        self._update_table()

    # -- Products ------------------------------------------------------------

    def _outer_product(self):
        A = self._get_tensor(self.prod_a.text())
        B = self._get_tensor(self.prod_b.text())
        if A is None or B is None:
            return
        name = self.prod_result.text().strip() or "result"
        result = np.tensordot(A, B, axes=0)
        self._tensors[name] = result
        self._log(f"Outer product {self.prod_a.text()} x {self.prod_b.text()} -> '{name}'")
        self._display_tensor(name, result)
        self._update_table()

    def _kronecker_product(self):
        A = self._get_tensor(self.prod_a.text())
        B = self._get_tensor(self.prod_b.text())
        if A is None or B is None:
            return
        if A.ndim != 2 or B.ndim != 2:
            self._log("Kronecker product requires rank-2 tensors (matrices).")
            return
        name = self.prod_result.text().strip() or "result"
        result = np.kron(A, B)
        self._tensors[name] = result
        self._log(f"Kronecker product {self.prod_a.text()} kron {self.prod_b.text()} -> '{name}'")
        self._display_tensor(name, result)
        self._update_table()

    def _tensor_contraction(self):
        T = self._get_tensor(self.tc_tensor.text())
        if T is None:
            return
        ax1 = self.tc_ax1.value()
        ax2 = self.tc_ax2.value()
        name = self.tc_result.text().strip() or "contracted"
        try:
            result = contract_indices(T, ax1, ax2)
            self._tensors[name] = result
            self._log(f"Contraction of '{self.tc_tensor.text()}' axes ({ax1},{ax2}) -> '{name}'")
            self._display_tensor(name, result)
            self._update_table()
        except Exception as exc:
            self._log(f"Contraction error: {exc}")

    # -- Common ops ----------------------------------------------------------

    def _common_op(self, op):
        name_in = self.common_tensor.text().strip()
        T = self._get_tensor(name_in)
        if T is None:
            return
        name_out = self.common_result.text().strip() or "result"

        try:
            if op == "trace":
                if T.ndim < 2:
                    self._log("Trace requires rank >= 2.")
                    return
                result = tensor_trace(T)
                self._tensors[name_out] = np.array(result)
                self._log(f"Trace({name_in}) = ")
                self._display_tensor(name_out, np.array(result))

            elif op == "transpose":
                result = tensor_transpose(T)
                self._tensors[name_out] = result
                self._log(f"Transpose({name_in}) -> '{name_out}'")
                self._display_tensor(name_out, result)

            elif op == "symmetrize":
                if T.ndim != 2:
                    self._log("Symmetrize requires rank-2 tensor.")
                    return
                result = symmetrize(T)
                self._tensors[name_out] = result
                self._log(f"Symmetrize({name_in}) -> '{name_out}'")
                self._display_tensor(name_out, result)

            elif op == "antisymmetrize":
                if T.ndim != 2:
                    self._log("Antisymmetrize requires rank-2 tensor.")
                    return
                result = antisymmetrize(T)
                self._tensors[name_out] = result
                self._log(f"Antisymmetrize({name_in}) -> '{name_out}'")
                self._display_tensor(name_out, result)

            elif op == "determinant":
                if T.ndim != 2 or T.shape[0] != T.shape[1]:
                    self._log("Determinant requires a square matrix.")
                    return
                det = np.linalg.det(T)
                self._log(f"det({name_in}) = {det:.10g}")

            elif op == "inverse":
                if T.ndim != 2 or T.shape[0] != T.shape[1]:
                    self._log("Inverse requires a square matrix.")
                    return
                result = np.linalg.inv(T)
                self._tensors[name_out] = result
                self._log(f"Inverse({name_in}) -> '{name_out}'")
                self._display_tensor(name_out, result)

            elif op == "eigenvalues":
                if T.ndim != 2 or T.shape[0] != T.shape[1]:
                    self._log("Eigenvalues require a square matrix.")
                    return
                eigvals = np.linalg.eigvals(T)
                self._log(f"Eigenvalues of {name_in}:")
                for i, ev in enumerate(eigvals):
                    if abs(ev.imag) < 1e-14:
                        self._log(f"  lambda_{i} = {ev.real:.10g}")
                    else:
                        self._log(f"  lambda_{i} = {ev:.10g}")

            elif op == "norm":
                n = np.linalg.norm(T)
                self._log(f"Frobenius norm of {name_in} = {n:.10g}")

            self._update_table()
        except Exception as exc:
            self._log(f"Error in {op}: {exc}")

    # -- Voigt ---------------------------------------------------------------

    def _to_voigt(self):
        T = self._get_tensor(self.voigt_tensor.text())
        if T is None:
            return
        name = self.voigt_result.text().strip() or "V"
        try:
            if T.ndim == 2 and T.shape == (3, 3):
                result = tensor_to_voigt_2(T.real)
                self._tensors[name] = result
                self._log(f"Tensor '{self.voigt_tensor.text()}' (3x3) -> Voigt '{name}' (6-vector)")
                self._display_tensor(name, result)
            elif T.ndim == 4 and T.shape == (3, 3, 3, 3):
                result = tensor4_to_voigt_matrix(T.real)
                self._tensors[name] = result
                self._log(f"Tensor '{self.voigt_tensor.text()}' (3x3x3x3) -> Voigt '{name}' (6x6)")
                self._display_tensor(name, result)
            else:
                self._log("Voigt conversion requires 3x3 or 3x3x3x3 tensor.")
            self._update_table()
        except Exception as exc:
            self._log(f"Voigt conversion error: {exc}")

    def _from_voigt(self):
        V = self._get_tensor(self.voigt_tensor.text())
        if V is None:
            return
        name = self.voigt_result.text().strip() or "T"
        try:
            if V.ndim == 1 and V.shape == (6,):
                result = voigt_to_tensor_2(V.real)
                self._tensors[name] = result
                self._log(f"Voigt '{self.voigt_tensor.text()}' (6-vector) -> Tensor '{name}' (3x3)")
                self._display_tensor(name, result)
            elif V.ndim == 2 and V.shape == (6, 6):
                result = voigt_matrix_to_tensor4(V.real)
                self._tensors[name] = result
                self._log(f"Voigt '{self.voigt_tensor.text()}' (6x6) -> Tensor '{name}' (3x3x3x3)")
                self._display_tensor(name, result)
            else:
                self._log("Voigt->Tensor requires 6-vector or 6x6 matrix.")
            self._update_table()
        except Exception as exc:
            self._log(f"Voigt conversion error: {exc}")

    # -- Stress/Strain -------------------------------------------------------

    def _analyze_stress(self):
        sigma = self._get_tensor(self.stress_tensor.text())
        if sigma is None:
            return
        if sigma.shape != (3, 3):
            self._log("Stress analysis requires a 3x3 tensor.")
            return
        sigma_real = sigma.real
        self._log(f"\n=== Stress Analysis: '{self.stress_tensor.text()}' ===")
        self._display_tensor(self.stress_tensor.text(), sigma_real)

        ps = principal_stresses(sigma_real)
        self._log(f"Principal stresses:")
        self._log(f"  sigma_1 = {ps[0]:.8g}")
        self._log(f"  sigma_2 = {ps[1]:.8g}")
        self._log(f"  sigma_3 = {ps[2]:.8g}")

        vm = von_mises_stress(sigma_real)
        self._log(f"Von Mises stress: {vm:.8g}")

        hs = hydrostatic_stress(sigma_real)
        self._log(f"Hydrostatic stress: {hs:.8g}")

        I1, I2, I3 = stress_invariants(sigma_real)
        self._log(f"Stress invariants:")
        self._log(f"  I1 = {I1:.8g}")
        self._log(f"  I2 = {I2:.8g}")
        self._log(f"  I3 = {I3:.8g}")

        # Deviatoric
        dev = sigma_real - hs * np.eye(3)
        self._tensors["dev_" + self.stress_tensor.text()] = dev
        self._log(f"Deviatoric stress stored as 'dev_{self.stress_tensor.text()}'")
        self._display_tensor("dev_" + self.stress_tensor.text(), dev)

        # Max shear
        tau_max = 0.5 * (ps[0] - ps[2])
        self._log(f"Maximum shear stress: {tau_max:.8g}")

        # Lode angle
        J2 = 0.5 * np.sum(dev * dev)
        J3 = np.linalg.det(dev)
        if J2 > 1e-30:
            cos3theta = (3.0 * np.sqrt(3.0) / 2.0) * J3 / (J2 ** 1.5)
            cos3theta = np.clip(cos3theta, -1.0, 1.0)
            theta_lode = np.arccos(cos3theta) / 3.0
            self._log(f"Lode angle: {np.degrees(theta_lode):.4f} deg")
        else:
            self._log("Lode angle: N/A (J2 ~ 0)")

        self._log("")
        self._update_table()

    def _quick_stress(self):
        s = self.stress_entries
        sigma = np.array([
            [s["sigma_11"].value(), s["sigma_12"].value(), s["sigma_13"].value()],
            [s["sigma_12"].value(), s["sigma_22"].value(), s["sigma_23"].value()],
            [s["sigma_13"].value(), s["sigma_23"].value(), s["sigma_33"].value()],
        ])
        name = self.stress_tensor.text().strip() or "sigma"
        self._tensors[name] = sigma
        self._log(f"Stored symmetric stress tensor as '{name}'")
        self._update_table()
        self._analyze_stress()

    # -- Presets -------------------------------------------------------------

    def _load_preset(self, factory, label):
        name = self.preset_name.text().strip() or "preset"
        try:
            arr = factory()
            self._tensors[name] = arr
            self._log(f"Loaded preset '{label}' -> '{name}', shape={arr.shape}")
            self._display_tensor(name, arr)
            self._update_table()
        except Exception as exc:
            self._log(f"Error loading preset: {exc}")
