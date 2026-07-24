"""
Optics Simulator Widget for PyQt5 Scientific Suite.
Provides single/double slit diffraction, diffraction grating,
thin lens ray tracing, and Fresnel/Fraunhofer diffraction simulations.
"""

import numpy as np
from scipy.fft import fft2, fftshift
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QDoubleSpinBox, QSpinBox, QPushButton, QSplitter,
    QFormLayout, QTabWidget, QFileDialog, QMessageBox, QTextEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QListWidget,
    QListWidgetItem, QCheckBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    from mpl_style import style_figure, style_axes
except ImportError:
    def style_figure(f): pass
    def style_axes(a): pass


# ---------------------------------------------------------------------------
# Optical element classes for multi-element system
# ---------------------------------------------------------------------------

class OpticalElement:
    """Represents a single optical element with its ABCD (ray transfer) matrix."""

    def __init__(self, name, matrix, position=0.0):
        self.name = name
        self.matrix = np.array(matrix, dtype=float)
        self.position = position

    @staticmethod
    def thin_lens(f):
        return OpticalElement(f"Lens f={f*1e3:.1f}mm", [[1, 0], [-1/f, 1]])

    @staticmethod
    def free_space(d):
        return OpticalElement(f"Space {d*1e3:.1f}mm", [[1, d], [0, 1]])

    @staticmethod
    def flat_mirror():
        return OpticalElement("Flat Mirror", [[1, 0], [0, 1]])

    @staticmethod
    def curved_mirror(R):
        return OpticalElement(f"Mirror R={R*1e3:.1f}mm", [[1, 0], [-2/R, 1]])

    @staticmethod
    def thick_lens(n, R1, R2, d):
        """Thick lens with refractive index n, radii R1, R2, thickness d."""
        M1 = np.array([[1, 0], [(n-1)/R1, 1]])   # first surface (corrected sign)
        M_prop = np.array([[1, d/n], [0, 1]])
        M2 = np.array([[1, 0], [(1-n)/R2, 1]])    # second surface
        M = M2 @ M_prop @ M1
        return OpticalElement(f"ThickLens n={n:.2f}", M)


# ---------------------------------------------------------------------------
# Jones Matrix library for polarization
# ---------------------------------------------------------------------------

class JonesMatrices:
    """Library of Jones matrices for common polarization optics."""

    @staticmethod
    def linear_polarizer(theta=0.0):
        """Linear polarizer at angle theta (radians)."""
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c**2, c*s], [c*s, s**2]], dtype=complex)

    @staticmethod
    def half_wave_plate(theta=0.0):
        """Half-wave plate with fast axis at angle theta."""
        c, s = np.cos(2*theta), np.sin(2*theta)
        return np.array([[c, s], [s, -c]], dtype=complex)

    @staticmethod
    def quarter_wave_plate(theta=0.0):
        """Quarter-wave plate with fast axis at angle theta."""
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c**2 + 1j*s**2, (1-1j)*c*s],
                         [(1-1j)*c*s, s**2 + 1j*c**2]], dtype=complex) / np.sqrt(1)

    @staticmethod
    def rotator(phi):
        """Optical rotator (e.g., Faraday) by angle phi."""
        c, s = np.cos(phi), np.sin(phi)
        return np.array([[c, -s], [s, c]], dtype=complex)

    @staticmethod
    def general_waveplate(theta, delta):
        """General waveplate: fast axis at theta, phase retardation delta."""
        c, s = np.cos(theta), np.sin(theta)
        M = np.array([
            [c**2 * np.exp(1j*delta/2) + s**2 * np.exp(-1j*delta/2),
             c*s*(np.exp(1j*delta/2) - np.exp(-1j*delta/2))],
            [c*s*(np.exp(1j*delta/2) - np.exp(-1j*delta/2)),
             s**2 * np.exp(1j*delta/2) + c**2 * np.exp(-1j*delta/2)]
        ], dtype=complex)
        return M


# ---------------------------------------------------------------------------
# Gaussian beam propagation
# ---------------------------------------------------------------------------

class GaussianBeam:
    """Gaussian beam propagation using complex beam parameter q."""

    def __init__(self, wavelength, waist, z=0.0):
        self.wavelength = wavelength
        self.w0 = waist
        self.z_R = np.pi * waist**2 / wavelength  # Rayleigh range
        self.q = z + 1j * self.z_R  # complex beam parameter

    def propagate_abcd(self, M):
        """Propagate through ABCD matrix. Returns new GaussianBeam."""
        A, B, C, D = M[0,0], M[0,1], M[1,0], M[1,1]
        q_new = (A * self.q + B) / (C * self.q + D)
        beam = GaussianBeam.__new__(GaussianBeam)
        beam.wavelength = self.wavelength
        beam.q = q_new
        beam.z_R = np.imag(q_new)
        beam.w0 = np.sqrt(beam.z_R * self.wavelength / np.pi) if beam.z_R > 0 else self.w0
        return beam

    def spot_size(self, z=None):
        """Beam radius w(z)."""
        if z is None:
            q = self.q
        else:
            q = z + 1j * self.z_R
        inv_q = 1.0 / q
        w_sq = -self.wavelength / (np.pi * np.imag(inv_q))
        return np.sqrt(np.abs(w_sq))

    def divergence(self):
        """Far-field half-angle divergence (radians)."""
        return self.wavelength / (np.pi * self.w0)

    def radius_of_curvature(self, z):
        """Radius of curvature R(z)."""
        if abs(z) < 1e-15:
            return np.inf
        return z * (1 + (self.z_R / z)**2)

    def spot_size_array(self, z_array):
        """Compute spot size over an array of z positions."""
        return self.w0 * np.sqrt(1 + (z_array / self.z_R)**2)


PRESETS = {
    "Red Laser Single Slit": dict(sim="Single Slit Diffraction", wavelength=632.8, slit_width=50.0, slit_sep=0.0, n_slits=1, focal_length=100.0, obj_dist=200.0),
    "Green Laser Double Slit": dict(sim="Double Slit Interference", wavelength=532.0, slit_width=20.0, slit_sep=100.0, n_slits=2, focal_length=100.0, obj_dist=200.0),
    "Diffraction Grating 5 slits": dict(sim="Diffraction Grating", wavelength=532.0, slit_width=10.0, slit_sep=50.0, n_slits=5, focal_length=100.0, obj_dist=200.0),
    "Converging Lens f=100mm": dict(sim="Thin Lens Ray Tracing", wavelength=550.0, slit_width=25.0, slit_sep=0.0, n_slits=1, focal_length=100.0, obj_dist=200.0),
    "Diverging Lens f=-80mm": dict(sim="Thin Lens Ray Tracing", wavelength=550.0, slit_width=25.0, slit_sep=0.0, n_slits=1, focal_length=-80.0, obj_dist=150.0),
    "Fraunhofer Circular Aperture": dict(sim="Fresnel/Fraunhofer Diffraction", wavelength=500.0, slit_width=40.0, slit_sep=0.0, n_slits=1, focal_length=100.0, obj_dist=200.0),
}


class OpticsSimWidget(QWidget):
    """Interactive optics simulation widget."""

    SIM_TYPES = [
        "Single Slit Diffraction",
        "Double Slit Interference",
        "Diffraction Grating",
        "Thin Lens Ray Tracing",
        "Fresnel/Fraunhofer Diffraction",
        "Multi-Element Optical System",
        "Aberration Analysis",
        "Interference Patterns",
        "Polarization (Jones Calculus)",
        "Gaussian Beam Propagation",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._logger = None
        self._build_ui()
        self._connect_signals()

    # -- public API ----------------------------------------------------------

    def set_logger(self, fn):
        """Set external logging callback fn(str)."""
        self._logger = fn

    def run(self):
        """Run the currently selected simulation."""
        self._on_run()

    # -- UI construction -----------------------------------------------------

    def _build_ui(self):
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # left: controls
        ctrl = QWidget()
        cl = QVBoxLayout(ctrl)

        # simulation selector
        sim_group = QGroupBox("Simulation")
        sl = QFormLayout()
        self.sim_combo = QComboBox()
        self.sim_combo.addItems(self.SIM_TYPES)
        sl.addRow("Type:", self.sim_combo)
        sim_group.setLayout(sl)
        cl.addWidget(sim_group)

        # parameters
        par_group = QGroupBox("Parameters")
        pl = QFormLayout()
        self.wavelength_spin = QDoubleSpinBox(); self.wavelength_spin.setRange(200, 1100); self.wavelength_spin.setValue(532.0); self.wavelength_spin.setSuffix(" nm")
        pl.addRow("Wavelength:", self.wavelength_spin)
        self.slit_width_spin = QDoubleSpinBox(); self.slit_width_spin.setRange(0.1, 1000); self.slit_width_spin.setValue(25.0); self.slit_width_spin.setSuffix(" \u00b5m")
        pl.addRow("Slit width:", self.slit_width_spin)
        self.slit_sep_spin = QDoubleSpinBox(); self.slit_sep_spin.setRange(0, 5000); self.slit_sep_spin.setValue(100.0); self.slit_sep_spin.setSuffix(" \u00b5m")
        pl.addRow("Slit separation:", self.slit_sep_spin)
        self.n_slits_spin = QSpinBox(); self.n_slits_spin.setRange(1, 100); self.n_slits_spin.setValue(2)
        pl.addRow("Number of slits:", self.n_slits_spin)
        self.focal_spin = QDoubleSpinBox(); self.focal_spin.setRange(-500, 500); self.focal_spin.setValue(100.0); self.focal_spin.setSuffix(" mm")
        pl.addRow("Focal length:", self.focal_spin)
        self.obj_dist_spin = QDoubleSpinBox(); self.obj_dist_spin.setRange(1, 2000); self.obj_dist_spin.setValue(200.0); self.obj_dist_spin.setSuffix(" mm")
        pl.addRow("Object distance:", self.obj_dist_spin)
        par_group.setLayout(pl)
        cl.addWidget(par_group)

        # presets
        pre_group = QGroupBox("Presets")
        prl = QVBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["(select preset)"] + list(PRESETS.keys()))
        prl.addWidget(self.preset_combo)
        pre_group.setLayout(prl)
        cl.addWidget(pre_group)

        # --- Multi-element system ---
        me_group = QGroupBox("Multi-Element System")
        me_layout = QFormLayout()
        self.me_elements = QListWidget()
        self.me_elements.setMaximumHeight(100)
        me_layout.addRow("Elements:", self.me_elements)
        self.me_elem_type = QComboBox()
        self.me_elem_type.addItems(["Thin Lens", "Free Space", "Curved Mirror", "Flat Mirror"])
        me_layout.addRow("Add type:", self.me_elem_type)
        self.me_param = QDoubleSpinBox()
        self.me_param.setRange(-1000, 1000)
        self.me_param.setValue(100.0)
        self.me_param.setSuffix(" mm")
        me_layout.addRow("f/d/R:", self.me_param)
        btn_add_elem = QPushButton("Add Element")
        btn_add_elem.clicked.connect(self._add_optical_element)
        me_layout.addRow(btn_add_elem)
        btn_clear_elem = QPushButton("Clear Elements")
        btn_clear_elem.clicked.connect(lambda: self.me_elements.clear())
        me_layout.addRow(btn_clear_elem)
        me_group.setLayout(me_layout)
        cl.addWidget(me_group)

        # --- Interference controls ---
        intf_group = QGroupBox("Interference")
        intf_layout = QFormLayout()
        self.intf_type_combo = QComboBox()
        self.intf_type_combo.addItems(["Michelson", "Fabry-Perot", "Thin Film"])
        intf_layout.addRow("Type:", self.intf_type_combo)
        self.intf_d_spin = QDoubleSpinBox()
        self.intf_d_spin.setRange(0.01, 1000.0)
        self.intf_d_spin.setValue(10.0)
        self.intf_d_spin.setSuffix(" um")
        intf_layout.addRow("Path diff / thickness:", self.intf_d_spin)
        self.intf_n_spin = QDoubleSpinBox()
        self.intf_n_spin.setRange(1.0, 4.0)
        self.intf_n_spin.setValue(1.5)
        intf_layout.addRow("Refractive index:", self.intf_n_spin)
        self.intf_finesse_spin = QDoubleSpinBox()
        self.intf_finesse_spin.setRange(1.0, 1000.0)
        self.intf_finesse_spin.setValue(30.0)
        intf_layout.addRow("Finesse (FP):", self.intf_finesse_spin)
        intf_group.setLayout(intf_layout)
        cl.addWidget(intf_group)

        # --- Gaussian beam controls ---
        gb_group = QGroupBox("Gaussian Beam")
        gb_layout = QFormLayout()
        self.gb_waist_spin = QDoubleSpinBox()
        self.gb_waist_spin.setRange(0.1, 10000.0)
        self.gb_waist_spin.setValue(100.0)
        self.gb_waist_spin.setSuffix(" um")
        gb_layout.addRow("Beam waist w0:", self.gb_waist_spin)
        self.gb_lens_f_spin = QDoubleSpinBox()
        self.gb_lens_f_spin.setRange(-1000, 1000)
        self.gb_lens_f_spin.setValue(50.0)
        self.gb_lens_f_spin.setSuffix(" mm")
        gb_layout.addRow("Lens f (optional):", self.gb_lens_f_spin)
        self.gb_lens_pos_spin = QDoubleSpinBox()
        self.gb_lens_pos_spin.setRange(0, 10000)
        self.gb_lens_pos_spin.setValue(100.0)
        self.gb_lens_pos_spin.setSuffix(" mm")
        gb_layout.addRow("Lens position:", self.gb_lens_pos_spin)
        gb_group.setLayout(gb_layout)
        cl.addWidget(gb_group)

        # --- Export button ---
        btn_export = QPushButton("Export Pattern as Image")
        btn_export.clicked.connect(self._export_image)
        cl.addWidget(btn_export)

        # run
        self.run_btn = QPushButton("Run Simulation")
        cl.addWidget(self.run_btn)
        cl.addStretch()
        splitter.addWidget(ctrl)

        # right: plots
        plot_widget = QWidget()
        pvl = QVBoxLayout(plot_widget)
        self.tabs = QTabWidget()

        self.fig1d = Figure(figsize=(6, 4))
        style_figure(self.fig1d)
        self.ax1d = self.fig1d.add_subplot(111)
        self.canvas1d = FigureCanvas(self.fig1d)
        self.tabs.addTab(self.canvas1d, "1D Pattern")

        self.fig2d = Figure(figsize=(6, 4))
        style_figure(self.fig2d)
        self.ax2d = self.fig2d.add_subplot(111)
        self.canvas2d = FigureCanvas(self.fig2d)
        self.tabs.addTab(self.canvas2d, "2D Pattern")

        self.fig_ray = Figure(figsize=(6, 4))
        style_figure(self.fig_ray)
        self.ax_ray = self.fig_ray.add_subplot(111)
        self.canvas_ray = FigureCanvas(self.fig_ray)
        self.tabs.addTab(self.canvas_ray, "Ray Trace")

        self.fig_polar = Figure(figsize=(6, 4))
        style_figure(self.fig_polar)
        self.ax_polar = self.fig_polar.add_subplot(111)
        self.canvas_polar = FigureCanvas(self.fig_polar)
        self.tabs.addTab(self.canvas_polar, "Polarization / Beam")

        self.fig_intf = Figure(figsize=(6, 4))
        style_figure(self.fig_intf)
        self.ax_intf = self.fig_intf.add_subplot(111)
        self.canvas_intf = FigureCanvas(self.fig_intf)
        self.tabs.addTab(self.canvas_intf, "Interference")

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(120)
        pvl.addWidget(self.tabs)
        pvl.addWidget(self.info_text)
        splitter.addWidget(plot_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

    def _connect_signals(self):
        self.run_btn.clicked.connect(self._on_run)
        self.preset_combo.currentIndexChanged.connect(self._on_preset)

    # -- helpers -------------------------------------------------------------

    def _log(self, msg):
        if self._logger:
            self._logger(msg)

    def _on_preset(self, idx):
        name = self.preset_combo.currentText()
        if name not in PRESETS:
            return
        p = PRESETS[name]
        self.sim_combo.setCurrentText(p["sim"])
        self.wavelength_spin.setValue(p["wavelength"])
        self.slit_width_spin.setValue(p["slit_width"])
        self.slit_sep_spin.setValue(p["slit_sep"])
        self.n_slits_spin.setValue(p["n_slits"])
        self.focal_spin.setValue(p["focal_length"])
        self.obj_dist_spin.setValue(p["obj_dist"])
        self._log(f"Preset loaded: {name}")

    def _params(self):
        return dict(
            wavelength=self.wavelength_spin.value() * 1e-9,
            slit_width=self.slit_width_spin.value() * 1e-6,
            slit_sep=self.slit_sep_spin.value() * 1e-6,
            n_slits=self.n_slits_spin.value(),
            focal_length=self.focal_spin.value() * 1e-3,
            obj_dist=self.obj_dist_spin.value() * 1e-3,
        )

    # -- simulation dispatch -------------------------------------------------

    def _on_run(self):
        sim = self.sim_combo.currentText()
        self._log(f"Running: {sim}")
        p = self._params()
        dispatch = {
            "Single Slit Diffraction": self._sim_single_slit,
            "Double Slit Interference": self._sim_double_slit,
            "Diffraction Grating": self._sim_grating,
            "Thin Lens Ray Tracing": self._sim_ray_trace,
            "Fresnel/Fraunhofer Diffraction": self._sim_fresnel_fraunhofer,
            "Multi-Element Optical System": self._sim_multi_element,
            "Aberration Analysis": self._sim_aberrations,
            "Interference Patterns": self._sim_interference,
            "Polarization (Jones Calculus)": self._sim_polarization,
            "Gaussian Beam Propagation": self._sim_gaussian_beam,
        }
        func = dispatch.get(sim)
        if func:
            func(p)
        self._log(f"Completed: {sim}")

    # -- single slit ---------------------------------------------------------

    def _sim_single_slit(self, p):
        lam, a = p["wavelength"], p["slit_width"]
        theta = np.linspace(-0.05, 0.05, 4000)
        beta = np.pi * a * np.sin(theta) / lam
        with np.errstate(divide="ignore", invalid="ignore"):
            intensity = np.where(np.abs(beta) < 1e-12, 1.0, (np.sin(beta) / beta) ** 2)
        self.ax1d.clear()
        self.ax1d.plot(np.degrees(theta), intensity, "b-")
        self.ax1d.set_xlabel("Angle (degrees)")
        self.ax1d.set_ylabel("Normalized Intensity")
        self.ax1d.set_title("Single Slit Diffraction")
        self.ax1d.grid(True, alpha=0.3)
        self.canvas1d.draw()
        self._draw_2d_pattern(a, 0, 1, lam)
        self.tabs.setCurrentIndex(0)

    # -- double slit ---------------------------------------------------------

    def _sim_double_slit(self, p):
        lam, a, d = p["wavelength"], p["slit_width"], p["slit_sep"]
        theta = np.linspace(-0.03, 0.03, 4000)
        beta = np.pi * a * np.sin(theta) / lam
        delta = np.pi * d * np.sin(theta) / lam
        with np.errstate(divide="ignore", invalid="ignore"):
            env = np.where(np.abs(beta) < 1e-12, 1.0, (np.sin(beta) / beta) ** 2)
        intensity = env * np.cos(delta) ** 2
        self.ax1d.clear()
        self.ax1d.plot(np.degrees(theta), intensity, "r-", label="Combined")
        self.ax1d.plot(np.degrees(theta), env, "b--", alpha=0.5, label="Envelope")
        self.ax1d.set_xlabel("Angle (degrees)")
        self.ax1d.set_ylabel("Normalized Intensity")
        self.ax1d.set_title("Double Slit Interference")
        self.ax1d.legend()
        self.ax1d.grid(True, alpha=0.3)
        self.canvas1d.draw()
        self._draw_2d_pattern(a, d, 2, lam)
        self.tabs.setCurrentIndex(0)

    # -- grating -------------------------------------------------------------

    def _sim_grating(self, p):
        lam, a, d, N = p["wavelength"], p["slit_width"], p["slit_sep"], p["n_slits"]
        theta = np.linspace(-0.03, 0.03, 8000)
        beta = np.pi * a * np.sin(theta) / lam
        delta = np.pi * d * np.sin(theta) / lam
        with np.errstate(divide="ignore", invalid="ignore"):
            env = np.where(np.abs(beta) < 1e-12, 1.0, (np.sin(beta) / beta) ** 2)
            multi = np.where(np.abs(np.sin(delta)) < 1e-12, float(N ** 2),
                             (np.sin(N * delta) / np.sin(delta)) ** 2)
        intensity = env * multi / N ** 2
        self.ax1d.clear()
        self.ax1d.plot(np.degrees(theta), intensity, "g-")
        self.ax1d.set_xlabel("Angle (degrees)")
        self.ax1d.set_ylabel("Normalized Intensity")
        self.ax1d.set_title(f"Diffraction Grating (N={N})")
        self.ax1d.grid(True, alpha=0.3)
        self.canvas1d.draw()
        self._draw_2d_pattern(a, d, N, lam)
        self.tabs.setCurrentIndex(0)

    # -- 2D diffraction via FFT ----------------------------------------------

    def _draw_2d_pattern(self, a, d, N, lam):
        sz = 512
        L = max(a, d if d > 0 else a) * 20
        x = np.linspace(-L / 2, L / 2, sz)
        y = np.linspace(-L / 2, L / 2, sz)
        X, Y = np.meshgrid(x, y)
        aperture = np.zeros((sz, sz))
        for n in range(N):
            cx = (n - (N - 1) / 2) * d if d > 0 else 0
            mask = (np.abs(X - cx) < a / 2) & (np.abs(Y) < a * 4)
            aperture[mask] = 1.0
        E = fftshift(fft2(aperture))
        I2d = np.abs(E) ** 2
        I2d /= I2d.max() if I2d.max() > 0 else 1
        self.ax2d.clear()
        self.ax2d.imshow(I2d ** 0.3, cmap="inferno", extent=[-1, 1, -1, 1])
        self.ax2d.set_title("2D Diffraction Pattern")
        self.ax2d.set_xlabel("kx (a.u.)")
        self.ax2d.set_ylabel("ky (a.u.)")
        self.canvas2d.draw()

    # -- thin lens ray tracing -----------------------------------------------

    def _sim_ray_trace(self, p):
        f = p["focal_length"]
        do = p["obj_dist"]
        ax = self.ax_ray
        ax.clear()

        # image distance from thin lens equation 1/f = 1/do + 1/di
        if abs(1 / f - 1 / do) > 1e-12:
            di = 1.0 / (1.0 / f - 1.0 / do)
        else:
            di = 1e6  # effectively at infinity

        obj_h = 0.03  # object height (m)
        img_h = -obj_h * di / do if abs(do) > 1e-12 else 0

        xlim = max(abs(do), abs(di), abs(f)) * 1.5
        ax.axhline(0, color="k", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=1, linestyle="--", label="Lens")

        # draw focal points
        ax.plot(f, 0, "ro", ms=6, label=f"F ({f*1e3:.0f} mm)")
        ax.plot(-f, 0, "ro", ms=6)

        # draw object
        ax.annotate("", xy=(-do, obj_h), xytext=(-do, 0),
                     arrowprops=dict(arrowstyle="->", color="blue", lw=2))
        ax.text(-do, obj_h * 1.15, "Object", ha="center", color="blue", fontsize=8)

        # draw image
        if abs(di) < xlim * 5:
            ax.annotate("", xy=(di, img_h), xytext=(di, 0),
                         arrowprops=dict(arrowstyle="->", color="green", lw=2))
            label = "Image (real)" if di > 0 else "Image (virtual)"
            ax.text(di, img_h - 0.005 * np.sign(img_h), label, ha="center", color="green", fontsize=8)

        # trace rays
        colors = ["#e67e22", "#e74c3c", "#9b59b6"]
        ray_heights = [obj_h, obj_h * 0.5, obj_h * 0.25]
        for rh, c in zip(ray_heights, colors):
            # ray parallel to axis, then through focal point
            ax.plot([-do, 0], [rh, rh], c, linewidth=1)
            if abs(di) < xlim * 5:
                ax.plot([0, di], [rh, img_h * rh / obj_h], c, linewidth=1)
            else:
                ax.plot([0, xlim], [rh, rh - xlim * rh / f], c, linewidth=1)
            # ray through center
            ax.plot([-do, 0], [rh, 0], c, linewidth=1, linestyle="--")
            if abs(di) < xlim * 5:
                slope = -rh / do
                ax.plot([0, di], [0, slope * di], c, linewidth=1, linestyle="--")

        # lens representation
        lens_h = obj_h * 1.8
        if f > 0:
            ax.annotate("", xy=(0, lens_h), xytext=(0, -lens_h),
                         arrowprops=dict(arrowstyle="<->", color="gray", lw=2))
        else:
            ax.annotate("", xy=(0, lens_h), xytext=(0, -lens_h),
                         arrowprops=dict(arrowstyle=">-<", color="gray", lw=2))

        kind = "Converging" if f > 0 else "Diverging"
        mag = abs(img_h / obj_h) if abs(obj_h) > 1e-12 else 0
        ax.set_title(f"{kind} Lens  |  f={f*1e3:.1f} mm  do={do*1e3:.1f} mm  di={di*1e3:.1f} mm  M={mag:.2f}")
        ax.set_xlabel("Position along axis (m)")
        ax.set_ylabel("Height (m)")
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-lens_h * 1.5, lens_h * 1.5)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
        self.canvas_ray.draw()
        self.tabs.setCurrentIndex(2)

    # -- Fresnel / Fraunhofer diffraction ------------------------------------

    def _sim_fresnel_fraunhofer(self, p):
        lam, a = p["wavelength"], p["slit_width"]
        # 1D Fraunhofer
        theta = np.linspace(-0.06, 0.06, 4000)
        # circular aperture Airy pattern: J1(x)/x
        from scipy.special import j1
        k = 2 * np.pi / lam
        R = a / 2
        u = k * R * np.sin(theta)
        with np.errstate(divide="ignore", invalid="ignore"):
            airy = np.where(np.abs(u) < 1e-12, 1.0, (2 * j1(u) / u) ** 2)
        self.ax1d.clear()
        self.ax1d.plot(np.degrees(theta), airy, "m-", label="Airy (circular)")
        # also rectangular
        beta = np.pi * a * np.sin(theta) / lam
        with np.errstate(divide="ignore", invalid="ignore"):
            rect = np.where(np.abs(beta) < 1e-12, 1.0, (np.sin(beta) / beta) ** 2)
        self.ax1d.plot(np.degrees(theta), rect, "c--", alpha=0.6, label="Rect slit")
        self.ax1d.set_xlabel("Angle (degrees)")
        self.ax1d.set_ylabel("Normalized Intensity")
        self.ax1d.set_title("Fraunhofer Diffraction")
        self.ax1d.legend()
        self.ax1d.grid(True, alpha=0.3)
        self.canvas1d.draw()

        # 2D circular aperture FFT
        sz = 512
        L = a * 20
        x = np.linspace(-L / 2, L / 2, sz)
        X, Y = np.meshgrid(x, x)
        circ = ((X ** 2 + Y ** 2) <= R ** 2).astype(float)
        E = fftshift(fft2(circ))
        I2d = np.abs(E) ** 2
        I2d /= I2d.max() if I2d.max() > 0 else 1
        self.ax2d.clear()
        self.ax2d.imshow(I2d ** 0.3, cmap="inferno", extent=[-1, 1, -1, 1])
        self.ax2d.set_title("2D Fraunhofer (Circular Aperture)")
        self.ax2d.set_xlabel("kx (a.u.)")
        self.ax2d.set_ylabel("ky (a.u.)")
        self.canvas2d.draw()
        self.tabs.setCurrentIndex(0)

    # -- helper: add optical element to list ----------------------------------

    def _add_optical_element(self):
        etype = self.me_elem_type.currentText()
        val = self.me_param.value() * 1e-3  # mm -> m
        if etype == "Thin Lens":
            elem = OpticalElement.thin_lens(val)
        elif etype == "Free Space":
            elem = OpticalElement.free_space(val)
        elif etype == "Curved Mirror":
            elem = OpticalElement.curved_mirror(val)
        else:
            elem = OpticalElement.flat_mirror()
        item = QListWidgetItem(elem.name)
        item.setData(Qt.UserRole, elem)
        self.me_elements.addItem(item)

    # -- Multi-element optical system -----------------------------------------

    def _sim_multi_element(self, p):
        """Chain multiple optical elements and trace rays through the system."""
        elements = []
        for i in range(self.me_elements.count()):
            elem = self.me_elements.item(i).data(Qt.UserRole)
            if elem is not None:
                elements.append(elem)

        if not elements:
            # Default demo: two lenses with space
            f1, f2, d = 0.1, 0.05, 0.15
            elements = [
                OpticalElement.thin_lens(f1),
                OpticalElement.free_space(d),
                OpticalElement.thin_lens(f2),
            ]
            self._log("[Multi] Using default two-lens system")

        # Compute total ABCD matrix
        M_total = np.eye(2)
        for elem in elements:
            M_total = elem.matrix @ M_total

        ax = self.ax_ray
        ax.clear()

        # Trace rays through the system
        n_rays = 7
        ray_heights = np.linspace(-0.02, 0.02, n_rays)
        ray_angles = [0.0, 0.01, -0.01]
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#e67e22", "#9b59b6",
                  "#1abc9c", "#f39c12"]

        # Build cumulative matrices at each element boundary
        positions = [0.0]
        cum_matrices = [np.eye(2)]
        pos = 0.0
        for elem in elements:
            # Estimate position from free-space elements
            if "Space" in elem.name:
                pos += elem.matrix[0, 1]
            else:
                pos += 0.01  # small offset for lens/mirror
            positions.append(pos)
            cum_matrices.append(elem.matrix @ cum_matrices[-1])

        for ri, h0 in enumerate(ray_heights):
            for angle in ray_angles:
                ray = np.array([h0, angle])
                trace_x = [positions[0]]
                trace_y = [ray[0]]
                for j, elem in enumerate(elements):
                    ray = elem.matrix @ ray
                    trace_x.append(positions[j + 1])
                    trace_y.append(ray[0])
                ax.plot(np.array(trace_x) * 1e3, np.array(trace_y) * 1e3,
                        color=colors[ri % len(colors)], linewidth=0.8, alpha=0.7)

        # Draw element positions
        for j, elem in enumerate(elements):
            x_pos = positions[j + 1] * 1e3
            if "Lens" in elem.name:
                ax.axvline(x_pos, color="gray", linestyle="--", linewidth=1.5)
                ax.text(x_pos, ax.get_ylim()[1] * 0.9, elem.name,
                        ha="center", fontsize=6, rotation=45)
            elif "Mirror" in elem.name:
                ax.axvline(x_pos, color="blue", linestyle="-", linewidth=2)

        ax.set_xlabel("Position (mm)")
        ax.set_ylabel("Height (mm)")
        ax.set_title("Multi-Element Ray Trace")
        ax.grid(True, alpha=0.3)
        self.canvas_ray.draw()
        self.tabs.setCurrentIndex(2)

        # Info
        det = np.linalg.det(M_total)
        info = (f"System ABCD Matrix:\n"
                f"  A={M_total[0,0]:.4f}  B={M_total[0,1]:.4e}\n"
                f"  C={M_total[1,0]:.4e}  D={M_total[1,1]:.4f}\n"
                f"Determinant: {det:.6f}\n"
                f"Effective focal length: {-1/M_total[1,0]:.4f} m" if abs(M_total[1,0]) > 1e-12 else
                f"System ABCD Matrix:\n"
                f"  A={M_total[0,0]:.4f}  B={M_total[0,1]:.4e}\n"
                f"  C={M_total[1,0]:.4e}  D={M_total[1,1]:.4f}\n"
                f"Afocal system (C ~ 0)")
        self.info_text.setPlainText(info)

    # -- Aberration Analysis --------------------------------------------------

    def _sim_aberrations(self, p):
        """Simulate spherical and chromatic aberration for a single lens."""
        f = p["focal_length"]
        lam = p["wavelength"]

        ax = self.ax_ray
        ax.clear()

        # --- Spherical aberration ---
        # Rays at different heights focus at different points
        n_rays = 20
        heights = np.linspace(0.001, 0.03, n_rays)

        # Spherical aberration model: f_eff(h) = f * (1 - C_s * h^2)
        # C_s is the spherical aberration coefficient
        R = 2 * f  # approximate lens curvature radius
        n_lens = 1.5
        C_s = 1.0 / (8 * f**3)  # third-order Seidel coefficient (simplified)

        focus_shifts = []
        colors_r = []
        for h in heights:
            # paraxial focus vs marginal focus
            f_marginal = f / (1 + C_s * h**2 * f**2)
            focus_shifts.append(f_marginal)

            # Draw ray
            ax.plot([-0.1, 0], [h, h], 'b-', linewidth=0.5, alpha=0.5)
            ax.plot([0, f_marginal], [h, 0], 'b-', linewidth=0.5, alpha=0.5)

        # Paraxial focus
        ax.axvline(f, color='green', linestyle='--', linewidth=1.5, label='Paraxial focus')
        # Marginal focus
        f_marg = min(focus_shifts)
        ax.axvline(f_marg, color='red', linestyle='--', linewidth=1.5, label='Marginal focus')

        # --- Chromatic aberration ---
        # Different wavelengths focus at different points
        wavelengths = [450e-9, 550e-9, 650e-9]  # blue, green, red
        colors_wl = ['blue', 'green', 'red']
        n_cauchy = lambda lam_nm: 1.5 + 0.004 / (lam_nm * 1e6)**2  # Cauchy dispersion

        for wl, c in zip(wavelengths, colors_wl):
            n_wl = n_cauchy(wl)
            f_wl = R / (2 * (n_wl - 1))
            ax.axvline(f_wl, color=c, linestyle=':', linewidth=1.5, alpha=0.7,
                       label=f'{wl*1e9:.0f}nm f={f_wl*1e3:.1f}mm')

        # Lens
        ax.axvline(0, color='gray', linewidth=2)

        ax.set_xlabel("Position (m)")
        ax.set_ylabel("Height (m)")
        ax.set_title("Aberration Analysis (Spherical + Chromatic)")
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.3)
        self.canvas_ray.draw()
        self.tabs.setCurrentIndex(2)

        # Longitudinal spherical aberration
        LSA = f - f_marg
        # Chromatic aberration
        f_blue = R / (2 * (n_cauchy(wavelengths[0]) - 1))
        f_red = R / (2 * (n_cauchy(wavelengths[2]) - 1))
        LCA = f_red - f_blue

        self.info_text.setPlainText(
            f"Spherical Aberration:\n"
            f"  Longitudinal SA = {LSA*1e3:.4f} mm\n"
            f"  Paraxial f = {f*1e3:.2f} mm\n"
            f"  Marginal f = {f_marg*1e3:.2f} mm\n\n"
            f"Chromatic Aberration:\n"
            f"  f(450nm) = {f_blue*1e3:.2f} mm\n"
            f"  f(650nm) = {f_red*1e3:.2f} mm\n"
            f"  Longitudinal CA = {LCA*1e3:.4f} mm"
        )

    # -- Interference Patterns ------------------------------------------------

    def _sim_interference(self, p):
        """Generate Michelson, Fabry-Perot, or thin-film interference patterns."""
        intf_type = self.intf_type_combo.currentText()
        lam = p["wavelength"]
        d = self.intf_d_spin.value() * 1e-6   # um -> m
        n_film = self.intf_n_spin.value()
        finesse = self.intf_finesse_spin.value()

        ax1 = self.ax_intf
        ax1.clear()
        ax2d = self.ax2d
        ax2d.clear()

        if intf_type == "Michelson":
            # Michelson interferometer: circular fringes
            r = np.linspace(0, 0.05, 1000)  # radial coordinate (angle-like)
            delta = 4 * np.pi * d / lam * np.cos(r)
            I = 0.5 * (1 + np.cos(delta))

            ax1.plot(r * 1e3, I, 'b-')
            ax1.set_xlabel("Observation angle (mrad)")
            ax1.set_ylabel("Normalized intensity")
            ax1.set_title(f"Michelson Interferometer (d={d*1e6:.1f} um)")
            ax1.grid(True, alpha=0.3)

            # 2D circular fringe pattern
            sz = 400
            x = np.linspace(-0.05, 0.05, sz)
            X, Y = np.meshgrid(x, x)
            R = np.sqrt(X**2 + Y**2)
            delta_2d = 4 * np.pi * d / lam * np.cos(R)
            I2d = 0.5 * (1 + np.cos(delta_2d))
            ax2d.imshow(I2d, cmap="gray", extent=[-50, 50, -50, 50])
            ax2d.set_title("Michelson Fringe Pattern")
            ax2d.set_xlabel("x (mrad)")
            ax2d.set_ylabel("y (mrad)")

        elif intf_type == "Fabry-Perot":
            # Airy function
            R_coeff = 1 - np.pi / finesse  # mirror reflectivity from finesse
            R_coeff = max(0.01, min(R_coeff, 0.9999))
            F = 4 * R_coeff / (1 - R_coeff)**2

            delta = np.linspace(0, 6 * np.pi, 2000)
            I = 1.0 / (1 + F * np.sin(delta / 2)**2)

            ax1.plot(delta / np.pi, I, 'r-')
            ax1.set_xlabel("Phase difference / pi")
            ax1.set_ylabel("Transmission")
            ax1.set_title(f"Fabry-Perot (Finesse={finesse:.0f})")
            ax1.grid(True, alpha=0.3)

            # 2D ring pattern
            sz = 400
            x = np.linspace(-0.03, 0.03, sz)
            X, Y = np.meshgrid(x, x)
            R_rad = np.sqrt(X**2 + Y**2)
            delta_2d = 4 * np.pi * n_film * d / lam * np.cos(R_rad)
            I2d = 1.0 / (1 + F * np.sin(delta_2d / 2)**2)
            ax2d.imshow(I2d, cmap="hot", extent=[-30, 30, -30, 30])
            ax2d.set_title("Fabry-Perot Ring Pattern")
            ax2d.set_xlabel("x (mrad)")
            ax2d.set_ylabel("y (mrad)")

        elif intf_type == "Thin Film":
            # Thin film interference: reflectance vs wavelength
            wavelengths = np.linspace(380e-9, 780e-9, 1000)
            # Reflectance for thin film: R = 2*r^2*(1-cos(delta)) / (1 + r^4 - 2*r^2*cos(delta))
            n_air = 1.0
            r12 = (n_air - n_film) / (n_air + n_film)
            r23 = (n_film - 1.5) / (n_film + 1.5)  # film on glass substrate

            delta_arr = 4 * np.pi * n_film * d / wavelengths
            R_film = (r12**2 + r23**2 + 2*r12*r23*np.cos(delta_arr)) / \
                     (1 + r12**2 * r23**2 + 2*r12*r23*np.cos(delta_arr))

            ax1.plot(wavelengths * 1e9, R_film, 'g-')
            ax1.set_xlabel("Wavelength (nm)")
            ax1.set_ylabel("Reflectance")
            ax1.set_title(f"Thin Film (n={n_film:.2f}, d={d*1e9:.0f} nm)")
            ax1.grid(True, alpha=0.3)

            # Color map: thin film color vs thickness
            thicknesses = np.linspace(50e-9, 1000e-9, 200)
            R_map = np.zeros((len(wavelengths), len(thicknesses)))
            for j, thick in enumerate(thicknesses):
                delta_t = 4 * np.pi * n_film * thick / wavelengths
                R_map[:, j] = (r12**2 + r23**2 + 2*r12*r23*np.cos(delta_t)) / \
                               (1 + r12**2 * r23**2 + 2*r12*r23*np.cos(delta_t))
            ax2d.imshow(R_map, aspect='auto', cmap="Spectral",
                        extent=[50, 1000, 380, 780], origin='lower')
            ax2d.set_xlabel("Film thickness (nm)")
            ax2d.set_ylabel("Wavelength (nm)")
            ax2d.set_title("Thin Film Reflectance Map")

        self.canvas_intf.draw()
        self.canvas2d.draw()
        self.tabs.setCurrentIndex(4)  # Interference tab

    # -- Polarization (Jones Calculus) ----------------------------------------

    def _sim_polarization(self, p):
        """Jones matrix calculator for optical element chains."""
        ax = self.ax_polar
        ax.clear()

        # Demonstrate: light through polarizer -> QWP -> analyzer
        # Sweep analyzer angle and plot transmitted intensity
        angles = np.linspace(0, 2 * np.pi, 360)

        # Input: horizontal polarization
        E_in = np.array([1.0, 0.0], dtype=complex)

        # Configuration 1: Polarizer(0) -> QWP(45) -> Analyzer(theta)
        QWP = JonesMatrices.quarter_wave_plate(np.pi / 4)
        E_after_qwp = QWP @ E_in

        I_config1 = []
        for theta in angles:
            analyzer = JonesMatrices.linear_polarizer(theta)
            E_out = analyzer @ E_after_qwp
            I_config1.append(np.abs(E_out[0])**2 + np.abs(E_out[1])**2)

        # Configuration 2: Polarizer(0) -> HWP(22.5) -> Analyzer(theta)
        HWP = JonesMatrices.half_wave_plate(np.pi / 8)
        E_after_hwp = HWP @ E_in

        I_config2 = []
        for theta in angles:
            analyzer = JonesMatrices.linear_polarizer(theta)
            E_out = analyzer @ E_after_hwp
            I_config2.append(np.abs(E_out[0])**2 + np.abs(E_out[1])**2)

        # Configuration 3: Polarizer(0) -> Rotator(45) -> Analyzer(theta)
        ROT = JonesMatrices.rotator(np.pi / 4)
        E_after_rot = ROT @ E_in

        I_config3 = []
        for theta in angles:
            analyzer = JonesMatrices.linear_polarizer(theta)
            E_out = analyzer @ E_after_rot
            I_config3.append(np.abs(E_out[0])**2 + np.abs(E_out[1])**2)

        ax.plot(np.degrees(angles), I_config1, 'b-', label='QWP(45)')
        ax.plot(np.degrees(angles), I_config2, 'r-', label='HWP(22.5)')
        ax.plot(np.degrees(angles), I_config3, 'g-', label='Rotator(45)')
        ax.set_xlabel("Analyzer angle (degrees)")
        ax.set_ylabel("Transmitted intensity")
        ax.set_title("Polarization: Malus Law with Waveplates")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self.canvas_polar.draw()
        self.tabs.setCurrentIndex(3)

        # Polarization ellipse info
        def stokes_from_jones(E):
            S0 = np.abs(E[0])**2 + np.abs(E[1])**2
            S1 = np.abs(E[0])**2 - np.abs(E[1])**2
            S2 = 2 * np.real(E[0] * np.conj(E[1]))
            S3 = 2 * np.imag(E[0] * np.conj(E[1]))
            return S0, S1, S2, S3

        info_lines = ["Jones Matrix Analysis:\n"]
        for label, E in [("After QWP(45)", E_after_qwp),
                         ("After HWP(22.5)", E_after_hwp),
                         ("After Rotator(45)", E_after_rot)]:
            S0, S1, S2, S3 = stokes_from_jones(E)
            dop = np.sqrt(S1**2 + S2**2 + S3**2) / S0 if S0 > 0 else 0
            ellipticity = S3 / S0 if S0 > 0 else 0
            info_lines.append(f"{label}:")
            info_lines.append(f"  E = [{E[0]:.3f}, {E[1]:.3f}]")
            info_lines.append(f"  Stokes: S0={S0:.3f} S1={S1:.3f} S2={S2:.3f} S3={S3:.3f}")
            info_lines.append(f"  DOP={dop:.3f}, Ellipticity={ellipticity:.3f}\n")
        self.info_text.setPlainText("\n".join(info_lines))

    # -- Gaussian Beam Propagation --------------------------------------------

    def _sim_gaussian_beam(self, p):
        """Propagate Gaussian beam and optionally through a lens."""
        lam = p["wavelength"]
        w0 = self.gb_waist_spin.value() * 1e-6  # um -> m
        f_lens = self.gb_lens_f_spin.value() * 1e-3  # mm -> m
        z_lens = self.gb_lens_pos_spin.value() * 1e-3  # mm -> m

        beam = GaussianBeam(lam, w0)

        ax = self.ax_polar
        ax.clear()

        # Propagate without lens
        z_max = max(z_lens * 3, 5 * beam.z_R)
        z = np.linspace(-z_max, z_max, 1000)
        w_free = beam.spot_size_array(z)

        ax.plot(z * 1e3, w_free * 1e6, 'b-', linewidth=1.5, label='Free propagation')
        ax.plot(z * 1e3, -w_free * 1e6, 'b-', linewidth=1.5)

        # Propagate through lens at z_lens
        if abs(f_lens) > 1e-6:
            # Propagate to lens
            M_space1 = OpticalElement.free_space(z_lens).matrix
            beam_at_lens = beam.propagate_abcd(M_space1)

            # Through lens
            M_lens = OpticalElement.thin_lens(f_lens).matrix
            beam_after = beam_at_lens.propagate_abcd(M_lens)

            # Propagate after lens
            z_after = np.linspace(0, z_max, 500)
            w_after = beam_after.spot_size_array(z_after)
            z_plot = z_after + z_lens

            ax.plot(z_plot * 1e3, w_after * 1e6, 'r-', linewidth=1.5, label='After lens')
            ax.plot(z_plot * 1e3, -w_after * 1e6, 'r-', linewidth=1.5)
            ax.axvline(z_lens * 1e3, color='gray', linestyle='--', linewidth=1.5,
                       label=f'Lens f={f_lens*1e3:.1f}mm')

        # Rayleigh range markers
        ax.axvline(beam.z_R * 1e3, color='green', linestyle=':', alpha=0.5,
                   label=f'Rayleigh range = {beam.z_R*1e3:.2f} mm')
        ax.axvline(-beam.z_R * 1e3, color='green', linestyle=':', alpha=0.5)

        ax.set_xlabel("z (mm)")
        ax.set_ylabel("Beam radius (um)")
        ax.set_title("Gaussian Beam Propagation")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.5)
        self.canvas_polar.draw()
        self.tabs.setCurrentIndex(3)

        # Info
        divergence = beam.divergence()
        info = (f"Gaussian Beam Parameters:\n"
                f"  Wavelength: {lam*1e9:.1f} nm\n"
                f"  Beam waist w0: {w0*1e6:.2f} um\n"
                f"  Rayleigh range z_R: {beam.z_R*1e3:.4f} mm\n"
                f"  Divergence half-angle: {divergence*1e3:.4f} mrad\n"
                f"  Confocal parameter: {2*beam.z_R*1e3:.4f} mm")
        if abs(f_lens) > 1e-6:
            info += (f"\n\nAfter lens (f={f_lens*1e3:.1f} mm at z={z_lens*1e3:.1f} mm):\n"
                     f"  New waist: {beam_after.w0*1e6:.2f} um\n"
                     f"  New z_R: {beam_after.z_R*1e3:.4f} mm\n"
                     f"  New divergence: {beam_after.divergence()*1e3:.4f} mrad")
        self.info_text.setPlainText(info)

    # -- Export patterns as publication-quality images -------------------------

    def _export_image(self):
        """Export the current plot as a high-resolution publication-quality image."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Image", "optics_pattern.png",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)")
        if not path:
            return

        # Determine which figure is currently visible
        idx = self.tabs.currentIndex()
        fig_map = {
            0: self.fig1d,
            1: self.fig2d,
            2: self.fig_ray,
            3: self.fig_polar,
            4: self.fig_intf,
        }
        fig = fig_map.get(idx, self.fig1d)

        try:
            fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.1)
            self._log(f"Exported image to {path}")
        except Exception as exc:
            self._log(f"Export error: {exc}")


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = OpticsSimWidget()
    w.setWindowTitle("Optics Simulator")
    w.resize(1000, 600)
    w.show()
    sys.exit(app.exec_())
