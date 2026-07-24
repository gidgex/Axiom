"""
Coordinate Transforms Widget for QuantumRes.
Interactive coordinate system conversions with 3D visualization,
Jacobian display, differential operators, rotation matrices,
Lorentz boosts, and matrix transform visualization.
"""

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QLabel, QLineEdit, QComboBox, QGroupBox, QTabWidget, QTextEdit,
    QDoubleSpinBox, QSpinBox, QSplitter, QScrollArea, QSizePolicy,
    QFrame, QSlider
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.pyplot as plt


# ── Conversion Functions ─────────────────────────────────────────────────────

def cartesian_to_cylindrical(x, y, z):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta, z


def cylindrical_to_cartesian(r, theta, z):
    return r * np.cos(theta), r * np.sin(theta), z


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r) if r > 1e-12 else 0.0
    phi = np.arctan2(y, x)
    return r, theta, phi


def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def cylindrical_to_spherical(r_cyl, theta_cyl, z):
    x, y, zz = cylindrical_to_cartesian(r_cyl, theta_cyl, z)
    return cartesian_to_spherical(x, y, zz)


def spherical_to_cylindrical(r_sph, theta_sph, phi_sph):
    x, y, z = spherical_to_cartesian(r_sph, theta_sph, phi_sph)
    return cartesian_to_cylindrical(x, y, z)


# ── Jacobian Matrices ────────────────────────────────────────────────────────

def jacobian_cart_to_cyl(x, y, z):
    """Jacobian d(r,theta,z)/d(x,y,z)."""
    r = np.sqrt(x**2 + y**2)
    if r < 1e-12:
        return np.eye(3)
    return np.array([
        [x / r, y / r, 0],
        [-y / r**2, x / r**2, 0],
        [0, 0, 1]
    ])


def jacobian_cart_to_sph(x, y, z):
    """Jacobian d(r,theta,phi)/d(x,y,z)."""
    r = np.sqrt(x**2 + y**2 + z**2)
    rho = np.sqrt(x**2 + y**2)
    if r < 1e-12:
        return np.eye(3)
    J = np.zeros((3, 3))
    J[0] = [x / r, y / r, z / r]
    if rho > 1e-12:
        J[1] = [x * z / (r**2 * rho), y * z / (r**2 * rho), -rho / r**2]
        J[2] = [-y / rho**2, x / rho**2, 0]
    return J


def jacobian_cyl_to_cart(r, theta, z):
    """Jacobian d(x,y,z)/d(r,theta,z)."""
    return np.array([
        [np.cos(theta), -r * np.sin(theta), 0],
        [np.sin(theta), r * np.cos(theta), 0],
        [0, 0, 1]
    ])


def jacobian_sph_to_cart(r, theta, phi):
    """Jacobian d(x,y,z)/d(r,theta,phi)."""
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    return np.array([
        [st * cp, r * ct * cp, -r * st * sp],
        [st * sp, r * ct * sp, r * st * cp],
        [ct, -r * st, 0]
    ])


# ── Scale Factors ────────────────────────────────────────────────────────────

def scale_factors_cylindrical(r, theta, z):
    return 1.0, r, 1.0


def scale_factors_spherical(r, theta, phi):
    return 1.0, r, r * np.sin(theta)


# ── Rotation Matrices ────────────────────────────────────────────────────────

def rotation_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_rotation(alpha, beta, gamma):
    """ZYZ Euler angles."""
    return rotation_z(alpha) @ rotation_y(beta) @ rotation_z(gamma)


def lorentz_boost(beta_x, beta_y=0, beta_z=0):
    """4x4 Lorentz boost matrix for velocity beta = v/c."""
    bx, by, bz = beta_x, beta_y, beta_z
    b2 = bx**2 + by**2 + bz**2
    if b2 >= 1.0:
        b2 = 0.9999
        s = np.sqrt(b2 / (bx**2 + by**2 + bz**2 + 1e-30))
        bx, by, bz = bx * s, by * s, bz * s
    gamma = 1.0 / np.sqrt(1.0 - b2)
    L = np.eye(4)
    L[0, 0] = gamma
    beta = np.array([bx, by, bz])
    for i in range(3):
        L[0, i + 1] = -gamma * beta[i]
        L[i + 1, 0] = -gamma * beta[i]
        for j in range(3):
            L[i + 1, j + 1] = (gamma - 1) * beta[i] * beta[j] / (b2 + 1e-30) + (1 if i == j else 0)
    return L


# ── Differential Operators (symbolic formulas) ──────────────────────────────

DIFF_OPERATORS = {
    "Cartesian": {
        "Gradient": "grad(f) = (df/dx) x_hat + (df/dy) y_hat + (df/dz) z_hat",
        "Divergence": "div(F) = dFx/dx + dFy/dy + dFz/dz",
        "Curl": "curl(F) = (dFz/dy - dFy/dz) x_hat + (dFx/dz - dFz/dx) y_hat + (dFy/dx - dFx/dy) z_hat",
        "Laplacian": "nabla^2(f) = d2f/dx2 + d2f/dy2 + d2f/dz2",
    },
    "Cylindrical": {
        "Gradient": "grad(f) = (df/dr) r_hat + (1/r)(df/dtheta) theta_hat + (df/dz) z_hat",
        "Divergence": "div(F) = (1/r) d(r Fr)/dr + (1/r) dFtheta/dtheta + dFz/dz",
        "Curl": "curl(F) = [(1/r) dFz/dtheta - dFtheta/dz] r_hat + [dFr/dz - dFz/dr] theta_hat + (1/r)[d(r Ftheta)/dr - dFr/dtheta] z_hat",
        "Laplacian": "nabla^2(f) = (1/r) d/dr(r df/dr) + (1/r^2) d2f/dtheta2 + d2f/dz2",
    },
    "Spherical": {
        "Gradient": "grad(f) = (df/dr) r_hat + (1/r)(df/dtheta) theta_hat + (1/(r sin(theta)))(df/dphi) phi_hat",
        "Divergence": "div(F) = (1/r^2) d(r^2 Fr)/dr + (1/(r sin(theta))) d(sin(theta) Ftheta)/dtheta + (1/(r sin(theta))) dFphi/dphi",
        "Curl": "[complex expression in spherical coords]",
        "Laplacian": "nabla^2(f) = (1/r^2) d/dr(r^2 df/dr) + (1/(r^2 sin(theta))) d/dtheta(sin(theta) df/dtheta) + (1/(r^2 sin^2(theta))) d2f/dphi2",
    },
}


# ── Main Widget ──────────────────────────────────────────────────────────────

class CoordTransformsWidget(QWidget):
    """Coordinate system transforms with 3D visualization."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._log = lambda msg: None
        self._anim_timer = QTimer()
        self._anim_timer.setInterval(50)
        self._anim_timer.timeout.connect(self._anim_step)
        self._anim_t = 0.0
        self._init_ui()

    def set_logger(self, fn):
        self._log = fn

    def run(self):
        """Trigger a coordinate conversion from current inputs."""
        self._convert()

    # ── UI Construction ──────────────────────────────────────────────────

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        tabs = QTabWidget()
        tabs.addTab(self._build_converter_tab(), "Converter")
        tabs.addTab(self._build_visual_tab(), "3D Visual")
        tabs.addTab(self._build_jacobian_tab(), "Jacobian / Scale")
        tabs.addTab(self._build_diffops_tab(), "Diff Operators")
        tabs.addTab(self._build_transforms_tab(), "Transforms")
        tabs.addTab(self._build_matrix_vis_tab(), "Matrix Visualizer")
        tabs.addTab(self._build_vector_field_tab(), "Vector Field")
        layout.addWidget(tabs)

    # -- Converter tab --
    def _build_converter_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        # Cartesian input
        cart_grp = QGroupBox("Cartesian (x, y, z)")
        cart_lay = QHBoxLayout(cart_grp)
        self._cart_spins = []
        for label in ["x", "y", "z"]:
            cart_lay.addWidget(QLabel(label + ":"))
            sp = QDoubleSpinBox()
            sp.setRange(-1000, 1000)
            sp.setDecimals(4)
            sp.setValue(1.0)
            sp.valueChanged.connect(self._cart_changed)
            cart_lay.addWidget(sp)
            self._cart_spins.append(sp)
        lay.addWidget(cart_grp)

        # Cylindrical display
        cyl_grp = QGroupBox("Cylindrical (r, theta, z)")
        cyl_lay = QHBoxLayout(cyl_grp)
        self._cyl_spins = []
        for label in ["r", "theta", "z"]:
            cyl_lay.addWidget(QLabel(label + ":"))
            sp = QDoubleSpinBox()
            sp.setRange(-1000, 1000)
            sp.setDecimals(4)
            sp.valueChanged.connect(self._cyl_changed)
            cyl_lay.addWidget(sp)
            self._cyl_spins.append(sp)
        lay.addWidget(cyl_grp)

        # Spherical display
        sph_grp = QGroupBox("Spherical (r, theta, phi)")
        sph_lay = QHBoxLayout(sph_grp)
        self._sph_spins = []
        for label in ["r", "theta", "phi"]:
            sph_lay.addWidget(QLabel(label + ":"))
            sp = QDoubleSpinBox()
            sp.setRange(-1000, 1000)
            sp.setDecimals(4)
            sp.valueChanged.connect(self._sph_changed)
            sph_lay.addWidget(sp)
            self._sph_spins.append(sp)
        lay.addWidget(sph_grp)

        btn = QPushButton("Convert from Cartesian")
        btn.clicked.connect(self._convert)
        lay.addWidget(btn)
        lay.addStretch()
        return w

    def _block_signals(self, block):
        for sp in self._cart_spins + self._cyl_spins + self._sph_spins:
            sp.blockSignals(block)

    def _cart_changed(self):
        x = self._cart_spins[0].value()
        y = self._cart_spins[1].value()
        z = self._cart_spins[2].value()
        self._block_signals(True)
        r_c, th_c, z_c = cartesian_to_cylindrical(x, y, z)
        self._cyl_spins[0].setValue(r_c)
        self._cyl_spins[1].setValue(np.degrees(th_c))
        self._cyl_spins[2].setValue(z_c)
        r_s, th_s, ph_s = cartesian_to_spherical(x, y, z)
        self._sph_spins[0].setValue(r_s)
        self._sph_spins[1].setValue(np.degrees(th_s))
        self._sph_spins[2].setValue(np.degrees(ph_s))
        self._block_signals(False)

    def _cyl_changed(self):
        r = self._cyl_spins[0].value()
        theta = np.radians(self._cyl_spins[1].value())
        z = self._cyl_spins[2].value()
        x, y, zz = cylindrical_to_cartesian(r, theta, z)
        self._block_signals(True)
        self._cart_spins[0].setValue(x)
        self._cart_spins[1].setValue(y)
        self._cart_spins[2].setValue(zz)
        r_s, th_s, ph_s = cartesian_to_spherical(x, y, zz)
        self._sph_spins[0].setValue(r_s)
        self._sph_spins[1].setValue(np.degrees(th_s))
        self._sph_spins[2].setValue(np.degrees(ph_s))
        self._block_signals(False)

    def _sph_changed(self):
        r = self._sph_spins[0].value()
        theta = np.radians(self._sph_spins[1].value())
        phi = np.radians(self._sph_spins[2].value())
        x, y, z = spherical_to_cartesian(r, theta, phi)
        self._block_signals(True)
        self._cart_spins[0].setValue(x)
        self._cart_spins[1].setValue(y)
        self._cart_spins[2].setValue(z)
        r_c, th_c, z_c = cartesian_to_cylindrical(x, y, z)
        self._cyl_spins[0].setValue(r_c)
        self._cyl_spins[1].setValue(np.degrees(th_c))
        self._cyl_spins[2].setValue(z_c)
        self._block_signals(False)

    def _convert(self):
        """Convert from Cartesian and update all fields plus plots."""
        x = self._cart_spins[0].value()
        y = self._cart_spins[1].value()
        z = self._cart_spins[2].value()
        self._cart_changed()
        self._update_3d_plot(x, y, z)
        self._update_jacobian(x, y, z)
        self._log(f"Converted ({x:.4f}, {y:.4f}, {z:.4f})")

    # -- 3D Visualization tab --
    def _build_visual_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self._3d_fig = Figure(figsize=(5, 5), dpi=100)
        self._3d_canvas = FigureCanvas(self._3d_fig)
        self._3d_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self._3d_canvas)

        btn_row = QHBoxLayout()
        btn = QPushButton("Update 3D")
        btn.clicked.connect(lambda: self._convert())
        btn_row.addWidget(btn)
        anim_btn = QPushButton("Animate Basis Vectors")
        anim_btn.clicked.connect(self._toggle_animation)
        btn_row.addWidget(anim_btn)
        lay.addLayout(btn_row)
        return w

    def _update_3d_plot(self, x, y, z):
        fig = self._3d_fig
        fig.clear()
        ax = fig.add_subplot(111, projection='3d')

        # Plot point
        ax.scatter([x], [y], [z], color='red', s=80, zorder=5)

        # Cartesian basis
        origin = np.array([x, y, z])
        scale = max(abs(x), abs(y), abs(z), 1) * 0.3
        colors_cart = ['#e74c3c', '#2ecc71', '#3498db']
        labels_cart = ['x', 'y', 'z']
        for i, c in enumerate(colors_cart):
            d = np.zeros(3)
            d[i] = scale
            ax.quiver(*origin, *d, color=c, arrow_length_ratio=0.15, linewidth=2, label=f'{labels_cart[i]}_hat')

        # Cylindrical basis vectors at point
        r_cyl, theta_cyl, _ = cartesian_to_cylindrical(x, y, z)
        if r_cyl > 1e-6:
            r_hat = np.array([np.cos(theta_cyl), np.sin(theta_cyl), 0]) * scale
            th_hat = np.array([-np.sin(theta_cyl), np.cos(theta_cyl), 0]) * scale
            z_hat = np.array([0, 0, scale])
            ax.quiver(*origin, *r_hat, color='orange', arrow_length_ratio=0.15, linewidth=1.5, linestyle='--', label='r_hat (cyl)')
            ax.quiver(*origin, *th_hat, color='gold', arrow_length_ratio=0.15, linewidth=1.5, linestyle='--', label='theta_hat (cyl)')

        # Spherical basis vectors
        r_sph, theta_sph, phi_sph = cartesian_to_spherical(x, y, z)
        if r_sph > 1e-6:
            st, ct = np.sin(theta_sph), np.cos(theta_sph)
            sp, cp = np.sin(phi_sph), np.cos(phi_sph)
            er = np.array([st * cp, st * sp, ct]) * scale
            eth = np.array([ct * cp, ct * sp, -st]) * scale
            eph = np.array([-sp, cp, 0]) * scale
            ax.quiver(*origin, *er, color='purple', arrow_length_ratio=0.15, linewidth=1.5, linestyle=':', label='r_hat (sph)')
            ax.quiver(*origin, *eth, color='magenta', arrow_length_ratio=0.15, linewidth=1.5, linestyle=':', label='theta_hat (sph)')

        # Draw coordinate surfaces (faint)
        # Cylinder surface
        if r_cyl > 0.1:
            th_range = np.linspace(0, 2 * np.pi, 40)
            z_range = np.linspace(z - scale, z + scale, 10)
            TH, ZZ = np.meshgrid(th_range, z_range)
            XX = r_cyl * np.cos(TH)
            YY = r_cyl * np.sin(TH)
            ax.plot_surface(XX, YY, ZZ, alpha=0.05, color='orange')

        lim = max(abs(x), abs(y), abs(z), 1) * 1.5
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Coordinate Visualization')
        ax.legend(fontsize=6, loc='upper left')
        self._3d_canvas.draw()

    def _toggle_animation(self):
        if self._anim_timer.isActive():
            self._anim_timer.stop()
        else:
            self._anim_t = 0.0
            self._anim_timer.start()

    def _anim_step(self):
        self._anim_t += 0.05
        if self._anim_t > 2 * np.pi:
            self._anim_timer.stop()
            return
        t = self._anim_t
        x = 2 * np.cos(t)
        y = 2 * np.sin(t)
        z = 1.0
        self._block_signals(True)
        self._cart_spins[0].setValue(x)
        self._cart_spins[1].setValue(y)
        self._cart_spins[2].setValue(z)
        self._block_signals(False)
        self._cart_changed()
        self._update_3d_plot(x, y, z)

    # -- Jacobian / Scale factors tab --
    def _build_jacobian_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self._jacobian_text = QTextEdit()
        self._jacobian_text.setReadOnly(True)
        self._jacobian_text.setFont(QFont("Courier New", 10))
        lay.addWidget(self._jacobian_text)
        btn = QPushButton("Compute Jacobians")
        btn.clicked.connect(self._convert)
        lay.addWidget(btn)
        return w

    def _update_jacobian(self, x, y, z):
        lines = []
        # Cartesian -> Cylindrical
        J1 = jacobian_cart_to_cyl(x, y, z)
        r_c, th_c, z_c = cartesian_to_cylindrical(x, y, z)
        h1, h2, h3 = scale_factors_cylindrical(r_c, th_c, z_c)
        lines.append("=== Cartesian -> Cylindrical ===")
        lines.append(f"Point: r={r_c:.4f}, theta={np.degrees(th_c):.2f} deg, z={z_c:.4f}")
        lines.append("Jacobian d(r,theta,z)/d(x,y,z):")
        for row in J1:
            lines.append("  [" + "  ".join(f"{v:10.6f}" for v in row) + "]")
        lines.append(f"Scale factors: h_r={h1:.4f}, h_theta={h2:.4f}, h_z={h3:.4f}")
        lines.append(f"det(J) = {np.linalg.det(J1):.6f}")
        lines.append("")

        # Cartesian -> Spherical
        J2 = jacobian_cart_to_sph(x, y, z)
        r_s, th_s, ph_s = cartesian_to_spherical(x, y, z)
        h1s, h2s, h3s = scale_factors_spherical(r_s, th_s, ph_s)
        lines.append("=== Cartesian -> Spherical ===")
        lines.append(f"Point: r={r_s:.4f}, theta={np.degrees(th_s):.2f} deg, phi={np.degrees(ph_s):.2f} deg")
        lines.append("Jacobian d(r,theta,phi)/d(x,y,z):")
        for row in J2:
            lines.append("  [" + "  ".join(f"{v:10.6f}" for v in row) + "]")
        lines.append(f"Scale factors: h_r={h1s:.4f}, h_theta={h2s:.4f}, h_phi={h3s:.4f}")
        lines.append(f"det(J) = {np.linalg.det(J2):.6f}")
        lines.append("")

        # Cylindrical -> Cartesian
        J3 = jacobian_cyl_to_cart(r_c, th_c, z_c)
        lines.append("=== Cylindrical -> Cartesian ===")
        lines.append("Jacobian d(x,y,z)/d(r,theta,z):")
        for row in J3:
            lines.append("  [" + "  ".join(f"{v:10.6f}" for v in row) + "]")
        lines.append(f"det(J) = {np.linalg.det(J3):.6f}  (= r = {r_c:.6f})")
        lines.append("")

        # Spherical -> Cartesian
        J4 = jacobian_sph_to_cart(r_s, th_s, ph_s)
        lines.append("=== Spherical -> Cartesian ===")
        lines.append("Jacobian d(x,y,z)/d(r,theta,phi):")
        for row in J4:
            lines.append("  [" + "  ".join(f"{v:10.6f}" for v in row) + "]")
        lines.append(f"det(J) = {np.linalg.det(J4):.6f}  (= r^2 sin(theta) = {r_s**2 * np.sin(th_s):.6f})")

        self._jacobian_text.setPlainText("\n".join(lines))

    # -- Differential Operators tab --
    def _build_diffops_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self._diffops_combo = QComboBox()
        self._diffops_combo.addItems(["Cartesian", "Cylindrical", "Spherical"])
        self._diffops_combo.currentTextChanged.connect(self._show_diffops)
        lay.addWidget(self._diffops_combo)
        self._diffops_text = QTextEdit()
        self._diffops_text.setReadOnly(True)
        self._diffops_text.setFont(QFont("Courier New", 10))
        lay.addWidget(self._diffops_text)
        self._show_diffops("Cartesian")
        return w

    def _show_diffops(self, system):
        ops = DIFF_OPERATORS.get(system, {})
        lines = [f"Differential Operators in {system} Coordinates", "=" * 50, ""]
        for name, formula in ops.items():
            lines.append(f"{name}:")
            lines.append(f"  {formula}")
            lines.append("")
        self._diffops_text.setPlainText("\n".join(lines))

    # -- Transforms tab (Euler, Lorentz, Affine) --
    def _build_transforms_tab(self):
        w = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        lay = QVBoxLayout(inner)

        # Euler angles
        euler_grp = QGroupBox("Rotation Matrix (Euler ZYZ)")
        euler_lay = QGridLayout(euler_grp)
        self._euler_spins = []
        for i, name in enumerate(["alpha", "beta", "gamma"]):
            euler_lay.addWidget(QLabel(name + " (deg):"), i, 0)
            sp = QDoubleSpinBox()
            sp.setRange(-360, 360)
            sp.setDecimals(2)
            euler_lay.addWidget(sp, i, 1)
            self._euler_spins.append(sp)
        euler_btn = QPushButton("Compute Rotation")
        euler_btn.clicked.connect(self._compute_euler)
        euler_lay.addWidget(euler_btn, 3, 0, 1, 2)
        self._euler_result = QTextEdit()
        self._euler_result.setReadOnly(True)
        self._euler_result.setFont(QFont("Courier New", 10))
        self._euler_result.setMaximumHeight(120)
        euler_lay.addWidget(self._euler_result, 4, 0, 1, 2)
        lay.addWidget(euler_grp)

        # Lorentz boost
        lor_grp = QGroupBox("Lorentz Boost")
        lor_lay = QGridLayout(lor_grp)
        self._lor_spins = []
        for i, name in enumerate(["beta_x", "beta_y", "beta_z"]):
            lor_lay.addWidget(QLabel(name + " (v/c):"), i, 0)
            sp = QDoubleSpinBox()
            sp.setRange(-0.9999, 0.9999)
            sp.setDecimals(4)
            sp.setSingleStep(0.01)
            lor_lay.addWidget(sp, i, 1)
            self._lor_spins.append(sp)
        lor_btn = QPushButton("Compute Boost")
        lor_btn.clicked.connect(self._compute_lorentz)
        lor_lay.addWidget(lor_btn, 3, 0, 1, 2)
        self._lor_result = QTextEdit()
        self._lor_result.setReadOnly(True)
        self._lor_result.setFont(QFont("Courier New", 10))
        self._lor_result.setMaximumHeight(140)
        lor_lay.addWidget(self._lor_result, 4, 0, 1, 2)
        lay.addWidget(lor_grp)

        lay.addStretch()
        scroll.setWidget(inner)
        outer = QVBoxLayout(w)
        outer.addWidget(scroll)
        return w

    def _compute_euler(self):
        a = np.radians(self._euler_spins[0].value())
        b = np.radians(self._euler_spins[1].value())
        g = np.radians(self._euler_spins[2].value())
        R = euler_rotation(a, b, g)
        lines = ["Rotation Matrix (ZYZ Euler):"]
        for row in R:
            lines.append("  [" + "  ".join(f"{v:10.6f}" for v in row) + "]")
        lines.append(f"det(R) = {np.linalg.det(R):.6f}")
        self._euler_result.setPlainText("\n".join(lines))
        self._log("Computed Euler rotation matrix")

    def _compute_lorentz(self):
        bx = self._lor_spins[0].value()
        by = self._lor_spins[1].value()
        bz = self._lor_spins[2].value()
        L = lorentz_boost(bx, by, bz)
        b2 = bx**2 + by**2 + bz**2
        gamma = 1.0 / np.sqrt(max(1.0 - b2, 1e-10))
        lines = [f"Lorentz Boost (gamma={gamma:.4f}):", "Lambda ="]
        for row in L:
            lines.append("  [" + "  ".join(f"{v:10.6f}" for v in row) + "]")
        lines.append(f"det(Lambda) = {np.linalg.det(L):.6f}")
        self._lor_result.setPlainText("\n".join(lines))
        self._log(f"Computed Lorentz boost, gamma={gamma:.4f}")

    # -- Matrix Visualizer tab --
    def _build_matrix_vis_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Enter a 2x2 or 3x3 matrix to see how it transforms the unit square/cube."))

        row = QHBoxLayout()
        self._mat_size = QComboBox()
        self._mat_size.addItems(["2x2", "3x3"])
        row.addWidget(QLabel("Size:"))
        row.addWidget(self._mat_size)
        lay.addLayout(row)

        grid = QGridLayout()
        self._mat_entries = []
        for i in range(3):
            row_entries = []
            for j in range(3):
                sp = QDoubleSpinBox()
                sp.setRange(-100, 100)
                sp.setDecimals(3)
                sp.setValue(1.0 if i == j else 0.0)
                grid.addWidget(sp, i, j)
                row_entries.append(sp)
            self._mat_entries.append(row_entries)
        lay.addLayout(grid)

        btn = QPushButton("Visualize Transform")
        btn.clicked.connect(self._visualize_matrix)
        lay.addWidget(btn)

        self._mat_fig = Figure(figsize=(5, 4), dpi=100)
        self._mat_canvas = FigureCanvas(self._mat_fig)
        self._mat_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self._mat_canvas)
        return w

    def _visualize_matrix(self):
        fig = self._mat_fig
        fig.clear()
        is_2d = self._mat_size.currentText() == "2x2"

        if is_2d:
            M = np.array([[self._mat_entries[i][j].value() for j in range(2)] for i in range(2)])
            ax = fig.add_subplot(111)
            # Unit square vertices
            sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T
            transformed = M @ sq
            ax.plot(sq[0], sq[1], 'b-', linewidth=2, label='Original')
            ax.fill(sq[0, :4], sq[1, :4], alpha=0.15, color='blue')
            ax.plot(transformed[0], transformed[1], 'r-', linewidth=2, label='Transformed')
            ax.fill(transformed[0, :4], transformed[1, :4], alpha=0.15, color='red')
            # Basis vectors
            e1 = M @ np.array([1, 0])
            e2 = M @ np.array([0, 1])
            ax.quiver(0, 0, e1[0], e1[1], angles='xy', scale_units='xy', scale=1, color='red', alpha=0.7, width=0.02)
            ax.quiver(0, 0, e2[0], e2[1], angles='xy', scale_units='xy', scale=1, color='green', alpha=0.7, width=0.02)
            lim = max(abs(transformed).max(), 1.5) * 1.2
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_title(f"2x2 Transform  det={np.linalg.det(M):.4f}")
        else:
            M = np.array([[self._mat_entries[i][j].value() for j in range(3)] for i in range(3)])
            ax = fig.add_subplot(111, projection='3d')
            # Unit cube edges
            edges = []
            for a in [0, 1]:
                for b in [0, 1]:
                    edges.append(([a, a], [b, b], [0, 1]))
                    edges.append(([a, a], [0, 1], [b, b]))
                    edges.append(([0, 1], [a, a], [b, b]))
            for ex, ey, ez in edges:
                pts = np.array([ex, ey, ez])
                ax.plot(*pts, 'b-', alpha=0.4)
                tp = M @ pts
                ax.plot(*tp, 'r-', alpha=0.6)
            # Basis vectors
            colors = ['red', 'green', 'blue']
            for i in range(3):
                e = np.zeros(3)
                e[i] = 1
                te = M @ e
                ax.quiver(0, 0, 0, te[0], te[1], te[2], color=colors[i], arrow_length_ratio=0.1, linewidth=2)
            lim = max(abs(M).max(), 1.5) * 1.5
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(-lim, lim)
            ax.set_title(f"3x3 Transform  det={np.linalg.det(M):.4f}")

        self._mat_canvas.draw()
        self._log("Matrix transform visualized")

    # -- Vector Field tab --
    def _build_vector_field_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Enter a vector field in Cartesian (using x, y, z). Example: -y, x, 0"))

        row = QHBoxLayout()
        row.addWidget(QLabel("Fx, Fy, Fz:"))
        self._vf_input = QLineEdit("-y, x, 0")
        row.addWidget(self._vf_input)
        btn = QPushButton("Plot Field")
        btn.clicked.connect(self._plot_vector_field)
        row.addWidget(btn)
        lay.addLayout(row)

        self._vf_fig = Figure(figsize=(5, 4), dpi=100)
        self._vf_canvas = FigureCanvas(self._vf_fig)
        self._vf_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self._vf_canvas)

        self._vf_info = QTextEdit()
        self._vf_info.setReadOnly(True)
        self._vf_info.setMaximumHeight(100)
        self._vf_info.setFont(QFont("Courier New", 9))
        lay.addWidget(self._vf_info)
        return w

    def _plot_vector_field(self):
        expr = self._vf_input.text().strip()
        parts = [p.strip() for p in expr.split(",")]
        if len(parts) != 3:
            self._vf_info.setPlainText("Error: enter three comma-separated expressions (Fx, Fy, Fz).")
            return

        fig = self._vf_fig
        fig.clear()

        try:
            # 2D slice at z=0
            ax = fig.add_subplot(121)
            grid_1d = np.linspace(-2, 2, 10)
            X, Y = np.meshgrid(grid_1d, grid_1d)
            Z = np.zeros_like(X)
            ns = {"x": X, "y": Y, "z": Z, "np": np, "sin": np.sin,
                  "cos": np.cos, "sqrt": np.sqrt, "pi": np.pi, "exp": np.exp}
            Fx = eval(parts[0], {"__builtins__": {}}, ns)
            Fy = eval(parts[1], {"__builtins__": {}}, ns)
            Fz = eval(parts[2], {"__builtins__": {}}, ns)
            if np.isscalar(Fx):
                Fx = np.full_like(X, Fx)
            if np.isscalar(Fy):
                Fy = np.full_like(X, Fy)
            mag = np.sqrt(Fx**2 + Fy**2 + 1e-10)
            ax.quiver(X, Y, Fx / mag, Fy / mag, mag, cmap='viridis', alpha=0.8)
            ax.set_title("XY plane (z=0)")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

            # 3D quiver
            ax3 = fig.add_subplot(122, projection='3d')
            g3 = np.linspace(-2, 2, 5)
            X3, Y3, Z3 = np.meshgrid(g3, g3, g3)
            ns3 = {"x": X3, "y": Y3, "z": Z3, "np": np, "sin": np.sin,
                   "cos": np.cos, "sqrt": np.sqrt, "pi": np.pi, "exp": np.exp}
            Fx3 = eval(parts[0], {"__builtins__": {}}, ns3)
            Fy3 = eval(parts[1], {"__builtins__": {}}, ns3)
            Fz3 = eval(parts[2], {"__builtins__": {}}, ns3)
            if np.isscalar(Fx3):
                Fx3 = np.full_like(X3, Fx3)
            if np.isscalar(Fy3):
                Fy3 = np.full_like(X3, Fy3)
            if np.isscalar(Fz3):
                Fz3 = np.full_like(X3, Fz3)
            mag3 = np.sqrt(Fx3**2 + Fy3**2 + Fz3**2 + 1e-10)
            ax3.quiver(X3.ravel(), Y3.ravel(), Z3.ravel(),
                       (Fx3 / mag3).ravel(), (Fy3 / mag3).ravel(), (Fz3 / mag3).ravel(),
                       length=0.3, normalize=True, alpha=0.6)
            ax3.set_title("3D Field")
            ax3.set_xlabel("x")
            ax3.set_ylabel("y")
            ax3.set_zlabel("z")

            fig.tight_layout()
            self._vf_canvas.draw()

            # Curvilinear representation info
            info = [f"Field: F = ({parts[0]}, {parts[1]}, {parts[2]})",
                    "",
                    "In cylindrical coords (r, theta, z):",
                    "  Fr     = Fx cos(theta) + Fy sin(theta)",
                    "  Ftheta = -Fx sin(theta) + Fy cos(theta)",
                    "  Fz     = Fz",
                    "",
                    "In spherical coords (r, theta, phi):",
                    "  Fr     = Fx sin(theta)cos(phi) + Fy sin(theta)sin(phi) + Fz cos(theta)",
                    "  Ftheta = Fx cos(theta)cos(phi) + Fy cos(theta)sin(phi) - Fz sin(theta)",
                    "  Fphi   = -Fx sin(phi) + Fy cos(phi)"]
            self._vf_info.setPlainText("\n".join(info))
            self._log(f"Plotted vector field: ({expr})")

        except Exception as e:
            self._vf_info.setPlainText(f"Error evaluating field: {e}")
