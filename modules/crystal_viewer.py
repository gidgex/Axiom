"""
Crystal Structure Viewer Widget
================================
A comprehensive PyQt5-based crystal structure visualization tool inspired by
VESTA and XCrysDen. Provides interactive 3D rendering of crystal structures
with support for built-in structure types, CIF file loading, supercell
generation, bond display, Miller plane visualization, and simulated powder
XRD diffraction patterns.
"""

import re
import math
import json
import itertools
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Dict, Any

import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QComboBox, QDoubleSpinBox, QSpinBox, QPushButton,
    QCheckBox, QTabWidget, QTextEdit, QFileDialog, QSplitter,
    QScrollArea, QFrame, QSizePolicy, QMessageBox, QLineEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QFormLayout,
    QDialog, QDialogButtonBox, QPlainTextEdit,
)
from PyQt5.QtCore import Qt, pyqtSignal

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    from mpl_style import style_figure, style_axes
except ImportError:
    def style_figure(f): pass
    def style_axes(a): pass
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection


# ---------------------------------------------------------------------------
# CPK element data: symbol -> (color_hex, covalent_radius_angstrom)
# ---------------------------------------------------------------------------
CPK_DATA: Dict[str, Tuple[str, float]] = {
    "H":  ("#FFFFFF", 0.31), "He": ("#D9FFFF", 0.28),
    "Li": ("#CC80FF", 1.28), "Be": ("#C2FF00", 0.96),
    "B":  ("#FFB5B5", 0.84), "C":  ("#909090", 0.76),
    "N":  ("#3050F8", 0.71), "O":  ("#FF0D0D", 0.66),
    "F":  ("#90E050", 0.57), "Ne": ("#B3E3F5", 0.58),
    "Na": ("#AB5CF2", 1.66), "Mg": ("#8AFF00", 1.41),
    "Al": ("#BFA6A6", 1.21), "Si": ("#F0C8A0", 1.11),
    "P":  ("#FF8000", 1.07), "S":  ("#FFFF30", 1.05),
    "Cl": ("#1FF01F", 1.02), "Ar": ("#80D1E3", 1.06),
    "K":  ("#8F40D4", 2.03), "Ca": ("#3DFF00", 1.76),
    "Ti": ("#BFC2C7", 1.60), "V":  ("#A6A6AB", 1.53),
    "Cr": ("#8A99C7", 1.39), "Mn": ("#9C7AC7", 1.39),
    "Fe": ("#E06633", 1.32), "Co": ("#F090A0", 1.26),
    "Ni": ("#50D050", 1.24), "Cu": ("#C88033", 1.32),
    "Zn": ("#7D80B0", 1.22), "Ga": ("#C28F8F", 1.22),
    "Ge": ("#668F8F", 1.20), "As": ("#BD80E3", 1.19),
    "Se": ("#FFA100", 1.20), "Br": ("#A62929", 1.20),
    "Sr": ("#00FF00", 1.95), "Zr": ("#94E0E0", 1.75),
    "Nb": ("#73C2C9", 1.64), "Mo": ("#54B5B5", 1.54),
    "Ag": ("#C0C0C0", 1.45), "Cd": ("#FFD98F", 1.44),
    "In": ("#A67573", 1.42), "Sn": ("#668080", 1.39),
    "Sb": ("#9E63B5", 1.39), "Te": ("#D47A00", 1.38),
    "I":  ("#940094", 1.39), "Cs": ("#57178F", 2.44),
    "Ba": ("#00C900", 2.15), "La": ("#70D4FF", 2.07),
    "Au": ("#FFD123", 1.36), "Pb": ("#575961", 1.46),
    "Bi": ("#9E4FB5", 1.48), "U":  ("#008FFF", 1.96),
}

# Atomic masses (amu) for density estimation
ATOMIC_MASSES: Dict[str, float] = {
    "H": 1.008, "He": 4.003, "Li": 6.941, "Be": 9.012, "B": 10.81,
    "C": 12.01, "N": 14.01, "O": 16.00, "F": 19.00, "Ne": 20.18,
    "Na": 22.99, "Mg": 24.31, "Al": 26.98, "Si": 28.09, "P": 30.97,
    "S": 32.07, "Cl": 35.45, "Ar": 39.95, "K": 39.10, "Ca": 40.08,
    "Ti": 47.87, "V": 50.94, "Cr": 52.00, "Mn": 54.94, "Fe": 55.85,
    "Co": 58.93, "Ni": 58.69, "Cu": 63.55, "Zn": 65.38, "Ga": 69.72,
    "Ge": 72.63, "As": 74.92, "Se": 78.97, "Br": 79.90, "Sr": 87.62,
    "Zr": 91.22, "Nb": 92.91, "Mo": 95.95, "Ag": 107.87, "Cd": 112.41,
    "In": 114.82, "Sn": 118.71, "Sb": 121.76, "Te": 127.60, "I": 126.90,
    "Cs": 132.91, "Ba": 137.33, "La": 138.91, "Au": 196.97, "Pb": 207.2,
    "Bi": 208.98, "U": 238.03,
}

DEFAULT_COLOR = "#FF69B4"
DEFAULT_RADIUS = 1.20


def _element_color(symbol: str) -> str:
    return CPK_DATA.get(symbol, (DEFAULT_COLOR, DEFAULT_RADIUS))[0]


def _element_radius(symbol: str) -> float:
    return CPK_DATA.get(symbol, (DEFAULT_COLOR, DEFAULT_RADIUS))[1]


# ---------------------------------------------------------------------------
# Lattice math helpers
# ---------------------------------------------------------------------------

def lattice_vectors(a: float, b: float, c: float,
                    alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Return 3x3 matrix whose rows are the lattice vectors (Angstrom)."""
    alpha_r = math.radians(alpha)
    beta_r = math.radians(beta)
    gamma_r = math.radians(gamma)

    cos_a, cos_b, cos_g = math.cos(alpha_r), math.cos(beta_r), math.cos(gamma_r)
    sin_g = math.sin(gamma_r)

    v1 = np.array([a, 0.0, 0.0])
    v2 = np.array([b * cos_g, b * sin_g, 0.0])

    cx = c * cos_b
    cy = c * (cos_a - cos_b * cos_g) / sin_g
    cz = math.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    v3 = np.array([cx, cy, cz])

    return np.array([v1, v2, v3])


def cell_volume(mat: np.ndarray) -> float:
    return abs(np.dot(mat[0], np.cross(mat[1], mat[2])))


def frac_to_cart(frac: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Convert fractional coordinates to Cartesian."""
    return frac @ mat


# ---------------------------------------------------------------------------
# Built-in structure definitions (fractional coords)
# ---------------------------------------------------------------------------

def _builtin_structures() -> Dict[str, Dict[str, Any]]:
    structures: Dict[str, Dict[str, Any]] = {}

    # Simple cubic
    structures["Simple cubic"] = {
        "a": 3.0, "b": 3.0, "c": 3.0,
        "alpha": 90, "beta": 90, "gamma": 90,
        "atoms": [("Fe", 0.0, 0.0, 0.0)],
        "spacegroup": "Pm-3m (221)",
    }

    # FCC
    structures["FCC"] = {
        "a": 3.615, "b": 3.615, "c": 3.615,
        "alpha": 90, "beta": 90, "gamma": 90,
        "atoms": [
            ("Cu", 0.0, 0.0, 0.0),
            ("Cu", 0.5, 0.5, 0.0),
            ("Cu", 0.5, 0.0, 0.5),
            ("Cu", 0.0, 0.5, 0.5),
        ],
        "spacegroup": "Fm-3m (225)",
    }

    # BCC
    structures["BCC"] = {
        "a": 2.87, "b": 2.87, "c": 2.87,
        "alpha": 90, "beta": 90, "gamma": 90,
        "atoms": [
            ("Fe", 0.0, 0.0, 0.0),
            ("Fe", 0.5, 0.5, 0.5),
        ],
        "spacegroup": "Im-3m (229)",
    }

    # HCP
    structures["HCP"] = {
        "a": 3.21, "b": 3.21, "c": 5.21,
        "alpha": 90, "beta": 90, "gamma": 120,
        "atoms": [
            ("Mg", 0.0, 0.0, 0.0),
            ("Mg", 1.0 / 3.0, 2.0 / 3.0, 0.5),
        ],
        "spacegroup": "P6_3/mmc (194)",
    }

    # Diamond cubic
    structures["Diamond cubic"] = {
        "a": 3.567, "b": 3.567, "c": 3.567,
        "alpha": 90, "beta": 90, "gamma": 90,
        "atoms": [
            ("C", 0.0, 0.0, 0.0),
            ("C", 0.5, 0.5, 0.0),
            ("C", 0.5, 0.0, 0.5),
            ("C", 0.0, 0.5, 0.5),
            ("C", 0.25, 0.25, 0.25),
            ("C", 0.75, 0.75, 0.25),
            ("C", 0.75, 0.25, 0.75),
            ("C", 0.25, 0.75, 0.75),
        ],
        "spacegroup": "Fd-3m (227)",
    }

    # NaCl (rock salt)
    structures["NaCl"] = {
        "a": 5.64, "b": 5.64, "c": 5.64,
        "alpha": 90, "beta": 90, "gamma": 90,
        "atoms": [
            ("Na", 0.0, 0.0, 0.0),
            ("Na", 0.5, 0.5, 0.0),
            ("Na", 0.5, 0.0, 0.5),
            ("Na", 0.0, 0.5, 0.5),
            ("Cl", 0.5, 0.0, 0.0),
            ("Cl", 0.0, 0.5, 0.0),
            ("Cl", 0.0, 0.0, 0.5),
            ("Cl", 0.5, 0.5, 0.5),
        ],
        "spacegroup": "Fm-3m (225)",
    }

    # CsCl
    structures["CsCl"] = {
        "a": 4.123, "b": 4.123, "c": 4.123,
        "alpha": 90, "beta": 90, "gamma": 90,
        "atoms": [
            ("Cs", 0.0, 0.0, 0.0),
            ("Cl", 0.5, 0.5, 0.5),
        ],
        "spacegroup": "Pm-3m (221)",
    }

    # Perovskite (SrTiO3)
    structures["Perovskite"] = {
        "a": 3.905, "b": 3.905, "c": 3.905,
        "alpha": 90, "beta": 90, "gamma": 90,
        "atoms": [
            ("Sr", 0.0, 0.0, 0.0),
            ("Ti", 0.5, 0.5, 0.5),
            ("O",  0.5, 0.5, 0.0),
            ("O",  0.5, 0.0, 0.5),
            ("O",  0.0, 0.5, 0.5),
        ],
        "spacegroup": "Pm-3m (221)",
    }

    # Zincblende (ZnS)
    structures["Zincblende"] = {
        "a": 5.41, "b": 5.41, "c": 5.41,
        "alpha": 90, "beta": 90, "gamma": 90,
        "atoms": [
            ("Zn", 0.0, 0.0, 0.0),
            ("Zn", 0.5, 0.5, 0.0),
            ("Zn", 0.5, 0.0, 0.5),
            ("Zn", 0.0, 0.5, 0.5),
            ("S",  0.25, 0.25, 0.25),
            ("S",  0.75, 0.75, 0.25),
            ("S",  0.75, 0.25, 0.75),
            ("S",  0.25, 0.75, 0.75),
        ],
        "spacegroup": "F-43m (216)",
    }

    # Wurtzite (ZnS)
    structures["Wurtzite"] = {
        "a": 3.82, "b": 3.82, "c": 6.26,
        "alpha": 90, "beta": 90, "gamma": 120,
        "atoms": [
            ("Zn", 1.0 / 3.0, 2.0 / 3.0, 0.0),
            ("Zn", 2.0 / 3.0, 1.0 / 3.0, 0.5),
            ("S",  1.0 / 3.0, 2.0 / 3.0, 0.375),
            ("S",  2.0 / 3.0, 1.0 / 3.0, 0.875),
        ],
        "spacegroup": "P6_3mc (186)",
    }

    # Graphene (single layer, pseudo-3D cell)
    structures["Graphene"] = {
        "a": 2.46, "b": 2.46, "c": 6.70,
        "alpha": 90, "beta": 90, "gamma": 120,
        "atoms": [
            ("C", 0.0, 0.0, 0.0),
            ("C", 1.0 / 3.0, 2.0 / 3.0, 0.0),
        ],
        "spacegroup": "P6/mmm (191)",
    }

    return structures


BUILTIN_STRUCTURES = _builtin_structures()


# ---------------------------------------------------------------------------
# Simple CIF parser
# ---------------------------------------------------------------------------

def parse_cif(text: str) -> Dict[str, Any]:
    """
    Parse a basic CIF file and return a dict with lattice parameters and atoms.
    Handles the most common CIF tags; does not implement full CIF spec.
    """
    lines = text.splitlines()

    params: Dict[str, float] = {}
    tag_map = {
        "_cell_length_a": "a",
        "_cell_length_b": "b",
        "_cell_length_c": "c",
        "_cell_angle_alpha": "alpha",
        "_cell_angle_beta": "beta",
        "_cell_angle_gamma": "gamma",
    }

    atoms: List[Tuple[str, float, float, float]] = []
    spacegroup = "P1"

    # --- scalar tags ---
    for line in lines:
        stripped = line.strip()
        for tag, key in tag_map.items():
            if stripped.startswith(tag):
                val_str = stripped.split()[-1]
                val_str = re.sub(r"\(.*?\)", "", val_str)
                try:
                    params[key] = float(val_str)
                except ValueError:
                    pass
        if stripped.startswith("_symmetry_space_group_name_H-M") or \
           stripped.startswith("_space_group_name_H-M_alt"):
            parts = stripped.split(None, 1)
            if len(parts) > 1:
                spacegroup = parts[1].strip().strip("'\"")

    # --- loop_ block for atom sites ---
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            headers: List[str] = []
            i += 1
            while i < len(lines) and lines[i].strip().startswith("_"):
                headers.append(lines[i].strip().split()[0])
                i += 1
            # Check if this loop contains atom site data
            label_idx = None
            symbol_idx = None
            x_idx = y_idx = z_idx = None
            for hi, h in enumerate(headers):
                h_low = h.lower()
                if "atom_site_type_symbol" in h_low:
                    symbol_idx = hi
                elif "atom_site_label" in h_low:
                    label_idx = hi
                elif "atom_site_fract_x" in h_low:
                    x_idx = hi
                elif "atom_site_fract_y" in h_low:
                    y_idx = hi
                elif "atom_site_fract_z" in h_low:
                    z_idx = hi

            has_coords = x_idx is not None and y_idx is not None and z_idx is not None
            has_symbol = symbol_idx is not None or label_idx is not None

            if has_coords and has_symbol:
                while i < len(lines):
                    row = lines[i].strip()
                    if not row or row.startswith("loop_") or row.startswith("_") or row.startswith("#"):
                        break
                    parts = row.split()
                    if len(parts) < len(headers):
                        break
                    sym = parts[symbol_idx] if symbol_idx is not None else parts[label_idx]
                    sym = re.sub(r"[^A-Za-z]", "", sym)
                    if len(sym) > 2:
                        sym = sym[:2]
                    sym = sym.capitalize()
                    try:
                        fx = float(re.sub(r"\(.*?\)", "", parts[x_idx]))
                        fy = float(re.sub(r"\(.*?\)", "", parts[y_idx]))
                        fz = float(re.sub(r"\(.*?\)", "", parts[z_idx]))
                        atoms.append((sym, fx, fy, fz))
                    except ValueError:
                        pass
                    i += 1
            else:
                # Skip non-atom loop rows
                while i < len(lines):
                    row = lines[i].strip()
                    if not row or row.startswith("loop_") or row.startswith("_") or row.startswith("#"):
                        break
                    i += 1
        else:
            i += 1

    # Defaults for missing parameters
    for key, default in [("a", 5.0), ("b", 5.0), ("c", 5.0),
                         ("alpha", 90.0), ("beta", 90.0), ("gamma", 90.0)]:
        params.setdefault(key, default)

    return {
        "a": params["a"], "b": params["b"], "c": params["c"],
        "alpha": params["alpha"], "beta": params["beta"], "gamma": params["gamma"],
        "atoms": atoms if atoms else [("X", 0.0, 0.0, 0.0)],
        "spacegroup": spacegroup,
    }


# ---------------------------------------------------------------------------
# Powder XRD simulation helpers
# ---------------------------------------------------------------------------

def _generate_hkl(max_index: int = 5) -> List[Tuple[int, int, int]]:
    """Generate unique (h, k, l) Miller indices up to max_index."""
    indices = []
    for h in range(-max_index, max_index + 1):
        for k in range(-max_index, max_index + 1):
            for l in range(-max_index, max_index + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                # Keep only one of (h,k,l) and (-h,-k,-l)
                if (h, k, l) > (0, 0, 0) or (h == 0 and k == 0 and l > 0) or (h == 0 and k > 0):
                    indices.append((h, k, l))
    return indices


def compute_d_spacing(h: int, k: int, l: int, mat: np.ndarray) -> float:
    """Compute d-spacing for Miller indices (h,k,l) given lattice matrix."""
    recip = np.linalg.inv(mat).T
    g = h * recip[0] + k * recip[1] + l * recip[2]
    g_len = np.linalg.norm(g)
    if g_len < 1e-12:
        return 1e12
    return 1.0 / g_len


def simulate_xrd(mat: np.ndarray, atoms: List[Tuple[str, float, float, float]],
                 wavelength: float = 1.5406, two_theta_max: float = 90.0,
                 max_index: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a simple powder XRD pattern using Bragg's law and a basic
    structure factor calculation.

    Returns (two_theta_array, intensity_array) for plotting.
    """
    hkl_list = _generate_hkl(max_index)
    peaks_2theta = []
    peaks_intensity = []

    # Approximate atomic scattering factors (very simplified, Z-based)
    z_approx = {}
    for sym, _, _, _ in atoms:
        if sym not in z_approx:
            mass = ATOMIC_MASSES.get(sym, 28.0)
            z_approx[sym] = mass / 2.0  # rough proxy

    for h, k, l in hkl_list:
        d = compute_d_spacing(h, k, l, mat)
        sin_theta = wavelength / (2.0 * d)
        if abs(sin_theta) > 1.0:
            continue
        theta = math.asin(sin_theta)
        two_theta = math.degrees(2.0 * theta)
        if two_theta > two_theta_max or two_theta < 1.0:
            continue

        # Structure factor (simplified)
        f_real = 0.0
        f_imag = 0.0
        for sym, fx, fy, fz in atoms:
            z = z_approx.get(sym, 14.0)
            phase = 2.0 * math.pi * (h * fx + k * fy + l * fz)
            f_real += z * math.cos(phase)
            f_imag += z * math.sin(phase)

        intensity = f_real * f_real + f_imag * f_imag

        # Lorentz-polarization factor (simplified)
        cos2t = math.cos(2.0 * theta)
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)
        if sin_t > 1e-10 and cos_t > 1e-10:
            lp = (1.0 + cos2t * cos2t) / (sin_t * sin_t * cos_t)
        else:
            lp = 1.0

        intensity *= lp

        # Multiplicity approximation: simple count
        multiplicity = 1
        vals = sorted([abs(h), abs(k), abs(l)])
        if vals[0] == vals[1] == vals[2]:
            multiplicity = 8
        elif vals[0] == vals[1] or vals[1] == vals[2]:
            multiplicity = 24
        else:
            multiplicity = 48

        intensity *= multiplicity

        peaks_2theta.append(two_theta)
        peaks_intensity.append(intensity)

    if not peaks_2theta:
        return np.array([0.0]), np.array([0.0])

    # Broaden peaks into a profile (pseudo-Voigt with Gaussian approx)
    two_theta_arr = np.linspace(1.0, two_theta_max, 2000)
    intensity_arr = np.zeros_like(two_theta_arr)
    fwhm = 0.15  # degrees

    for t2, inten in zip(peaks_2theta, peaks_intensity):
        sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        intensity_arr += inten * np.exp(-0.5 * ((two_theta_arr - t2) / sigma) ** 2)

    # Normalize
    max_i = np.max(intensity_arr)
    if max_i > 0:
        intensity_arr = intensity_arr / max_i * 100.0

    return two_theta_arr, intensity_arr


# ---------------------------------------------------------------------------
# Space group symmetry operations (common groups)
# ---------------------------------------------------------------------------

# Mapping of space group name -> list of (rotation_matrix, translation_vector)
# Only the most common space groups are included for the crystal builder.
_SPACEGROUP_OPS: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}


def _rot(matrix_flat, trans):
    """Helper to define a symmetry operation."""
    return (np.array(matrix_flat).reshape(3, 3), np.array(trans))


def _init_spacegroup_ops():
    """Populate common space group symmetry operations."""
    I = [1, 0, 0, 0, 1, 0, 0, 0, 1]  # noqa: E741
    # P1: only identity
    _SPACEGROUP_OPS["P1"] = [_rot(I, [0, 0, 0])]
    # P-1: identity + inversion
    _SPACEGROUP_OPS["P-1"] = [
        _rot(I, [0, 0, 0]),
        _rot([-1, 0, 0, 0, -1, 0, 0, 0, -1], [0, 0, 0]),
    ]
    # Pm-3m (221): full cubic ops (subset generating the group)
    cubic_ops = [_rot(I, [0, 0, 0])]
    for perm in itertools.permutations([0, 1, 2]):
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    m = np.zeros((3, 3))
                    signs = [sx, sy, sz]
                    for row, col in enumerate(perm):
                        m[row, col] = signs[row]
                    if np.linalg.det(m) > 0.5:
                        op = (m, np.array([0, 0, 0]))
                        # Check for duplicates
                        dup = False
                        for existing_m, _ in cubic_ops:
                            if np.allclose(existing_m, m):
                                dup = True
                                break
                        if not dup:
                            cubic_ops.append(op)
    # Add inversion
    inv_ops = []
    for m, t in cubic_ops:
        inv_ops.append((-m, t))
    _SPACEGROUP_OPS["Pm-3m"] = cubic_ops + inv_ops

    # Fm-3m (225): cubic + face-centering translations
    fc_trans = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    fm3m_ops = []
    for m, t in _SPACEGROUP_OPS["Pm-3m"]:
        for ft in fc_trans:
            fm3m_ops.append((m.copy(), np.array(ft)))
    _SPACEGROUP_OPS["Fm-3m"] = fm3m_ops

    # Im-3m (229): cubic + body-centering
    bcc_trans = [[0, 0, 0], [0.5, 0.5, 0.5]]
    im3m_ops = []
    for m, t in _SPACEGROUP_OPS["Pm-3m"]:
        for bt in bcc_trans:
            im3m_ops.append((m.copy(), np.array(bt)))
    _SPACEGROUP_OPS["Im-3m"] = im3m_ops

    # Fd-3m (227): cubic + face-centering + d-glide (simplified)
    _SPACEGROUP_OPS["Fd-3m"] = list(_SPACEGROUP_OPS["Fm-3m"])

    # F-43m (216)
    _SPACEGROUP_OPS["F-43m"] = list(_SPACEGROUP_OPS["Fm-3m"])

    # P6_3/mmc (194): hexagonal - simplified with basic 6-fold + mirror
    hex_ops = [_rot(I, [0, 0, 0])]
    c60 = math.cos(math.radians(60))
    s60 = math.sin(math.radians(60))
    for n in range(1, 6):
        angle = math.radians(60 * n)
        ca, sa = math.cos(angle), math.sin(angle)
        m = [ca, -sa, 0, sa, ca, 0, 0, 0, 1]
        hex_ops.append(_rot(m, [0, 0, 0.5 * (n % 2)]))
    # Mirror z
    hex_ops.append(_rot([1, 0, 0, 0, 1, 0, 0, 0, -1], [0, 0, 0]))
    _SPACEGROUP_OPS["P6_3/mmc"] = hex_ops

    # P6/mmm (191)
    _SPACEGROUP_OPS["P6/mmm"] = list(hex_ops)

    # P6_3mc (186)
    _SPACEGROUP_OPS["P6_3mc"] = list(hex_ops)


_init_spacegroup_ops()

AVAILABLE_SPACEGROUPS = sorted(_SPACEGROUP_OPS.keys())


def apply_symmetry(wyckoff_atoms: List[Tuple[str, float, float, float]],
                   spacegroup: str) -> List[Tuple[str, float, float, float]]:
    """Apply space group symmetry operations to Wyckoff positions.

    Returns a list of unique atoms (fractional coords wrapped to [0, 1)).
    """
    ops = _SPACEGROUP_OPS.get(spacegroup, [_rot([1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0])])
    result = []
    seen = set()
    for sym, fx, fy, fz in wyckoff_atoms:
        frac = np.array([fx, fy, fz])
        for rot_m, trans in ops:
            new_frac = rot_m @ frac + trans
            # Wrap to [0, 1)
            new_frac = new_frac % 1.0
            key = (sym, round(new_frac[0], 4), round(new_frac[1], 4), round(new_frac[2], 4))
            if key not in seen:
                seen.add(key)
                result.append((sym, float(new_frac[0]), float(new_frac[1]), float(new_frac[2])))
    return result


# ---------------------------------------------------------------------------
# Surface slab generator
# ---------------------------------------------------------------------------

def generate_surface_slab(structure: Dict[str, Any],
                          h: int, k: int, l: int,
                          n_layers: int = 4,
                          vacuum: float = 15.0) -> Dict[str, Any]:
    """Generate a surface slab from a bulk crystal structure.

    Parameters
    ----------
    structure : dict with lattice params and atoms
    h, k, l : Miller indices defining the surface orientation
    n_layers : number of atomic layers in the slab
    vacuum : vacuum thickness in Angstroms added above slab

    Returns a new structure dict for the slab.
    """
    mat = lattice_vectors(structure["a"], structure["b"], structure["c"],
                          structure["alpha"], structure["beta"], structure["gamma"])

    # Build a supercell large enough to contain the slab
    # Use a simple approach: replicate and project
    rep = max(n_layers + 2, 4)
    all_atoms = []
    for sym, fx, fy, fz in structure["atoms"]:
        for ix in range(-rep, rep + 1):
            for iy in range(-rep, rep + 1):
                for iz in range(-rep, rep + 1):
                    frac = np.array([fx + ix, fy + iy, fz + iz])
                    cart = frac_to_cart(frac, mat)
                    all_atoms.append((sym, cart))

    # Surface normal direction
    hkl = np.array([h, k, l], dtype=float)
    recip = np.linalg.inv(mat).T
    normal = recip @ hkl
    normal = normal / np.linalg.norm(normal)

    # Project all atoms onto the surface normal to determine layer heights
    heights = np.array([np.dot(pos, normal) for _, pos in all_atoms])

    # Find unique layer heights
    sorted_h = np.sort(np.unique(np.round(heights, 3)))
    if len(sorted_h) == 0:
        return structure

    # Select n_layers from the middle
    mid = len(sorted_h) // 2
    start = max(0, mid - n_layers // 2)
    end = min(len(sorted_h), start + n_layers)
    layer_heights = sorted_h[start:end]

    if len(layer_heights) == 0:
        return structure

    h_min, h_max = layer_heights[0], layer_heights[-1]
    tolerance = 0.2

    # Filter atoms within the slab layers
    slab_atoms_cart = []
    for sym, pos in all_atoms:
        proj = np.dot(pos, normal)
        if h_min - tolerance <= proj <= h_max + tolerance:
            slab_atoms_cart.append((sym, pos))

    if not slab_atoms_cart:
        return structure

    # Build orthogonal slab cell
    # a_slab and b_slab are in-plane, c_slab is along normal
    slab_thickness = h_max - h_min + 0.1
    c_length = slab_thickness + vacuum

    # Find two in-plane lattice vectors
    if abs(h) >= abs(k) and abs(h) >= abs(l):
        a_slab = mat[1].copy()
        b_slab = mat[2].copy()
    elif abs(k) >= abs(l):
        a_slab = mat[0].copy()
        b_slab = mat[2].copy()
    else:
        a_slab = mat[0].copy()
        b_slab = mat[1].copy()

    c_slab = normal * c_length
    slab_mat = np.array([a_slab, b_slab, c_slab])

    # Convert slab atoms to fractional coordinates of new cell
    try:
        inv_slab = np.linalg.inv(slab_mat)
    except np.linalg.LinAlgError:
        return structure

    # Shift atoms so slab starts at z_frac ~ 0
    min_cart_z = min(np.dot(pos, normal) for _, pos in slab_atoms_cart)
    slab_frac_atoms = []
    seen = set()
    for sym, pos in slab_atoms_cart:
        shifted = pos - min_cart_z * normal
        frac = shifted @ inv_slab.T
        # Keep only atoms within [0, 1) in a, b
        fa, fb = frac[0] % 1.0, frac[1] % 1.0
        fc = frac[2]
        if fc < -0.01 or fc > 1.01:
            continue
        fc = max(0.0, min(fc, 0.99))
        key = (sym, round(fa, 3), round(fb, 3), round(fc, 3))
        if key not in seen:
            seen.add(key)
            slab_frac_atoms.append((sym, float(fa), float(fb), float(fc)))

    a_len = np.linalg.norm(a_slab)
    b_len = np.linalg.norm(b_slab)
    c_len = np.linalg.norm(c_slab)

    # Compute angles
    cos_alpha = np.dot(b_slab, c_slab) / (b_len * c_len) if b_len * c_len > 0 else 0
    cos_beta = np.dot(a_slab, c_slab) / (a_len * c_len) if a_len * c_len > 0 else 0
    cos_gamma = np.dot(a_slab, b_slab) / (a_len * b_len) if a_len * b_len > 0 else 0

    return {
        "a": a_len, "b": b_len, "c": c_len,
        "alpha": math.degrees(math.acos(np.clip(cos_alpha, -1, 1))),
        "beta": math.degrees(math.acos(np.clip(cos_beta, -1, 1))),
        "gamma": math.degrees(math.acos(np.clip(cos_gamma, -1, 1))),
        "atoms": slab_frac_atoms if slab_frac_atoms else [("X", 0.0, 0.0, 0.0)],
        "spacegroup": "P1 (slab)",
    }


# ---------------------------------------------------------------------------
# Supercell defect generators
# ---------------------------------------------------------------------------

def create_supercell_atoms(structure: Dict[str, Any],
                           nx: int, ny: int, nz: int
                           ) -> Tuple[List[Tuple[str, float, float, float]], Dict[str, Any]]:
    """Create supercell atoms with scaled fractional coords.

    Returns (atoms_list, new_structure_dict).
    """
    atoms = []
    for sym, fx, fy, fz in structure["atoms"]:
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    new_fx = (fx + ix) / nx
                    new_fy = (fy + iy) / ny
                    new_fz = (fz + iz) / nz
                    atoms.append((sym, new_fx, new_fy, new_fz))
    new_s = {
        "a": structure["a"] * nx,
        "b": structure["b"] * ny,
        "c": structure["c"] * nz,
        "alpha": structure["alpha"],
        "beta": structure["beta"],
        "gamma": structure["gamma"],
        "atoms": atoms,
        "spacegroup": "P1 (supercell)",
    }
    return atoms, new_s


def add_vacancy(structure: Dict[str, Any], atom_index: int) -> Dict[str, Any]:
    """Remove atom at given index to create a vacancy defect."""
    new_atoms = list(structure["atoms"])
    if 0 <= atom_index < len(new_atoms):
        removed = new_atoms.pop(atom_index)
        new_s = dict(structure)
        new_s["atoms"] = new_atoms
        new_s["spacegroup"] = structure.get("spacegroup", "P1") + " (vacancy)"
        return new_s
    return structure


def add_substitutional(structure: Dict[str, Any], atom_index: int,
                       new_element: str) -> Dict[str, Any]:
    """Replace atom at given index with new_element."""
    new_atoms = list(structure["atoms"])
    if 0 <= atom_index < len(new_atoms):
        old = new_atoms[atom_index]
        new_atoms[atom_index] = (new_element, old[1], old[2], old[3])
        new_s = dict(structure)
        new_s["atoms"] = new_atoms
        new_s["spacegroup"] = structure.get("spacegroup", "P1") + " (substitutional)"
        return new_s
    return structure


def add_interstitial(structure: Dict[str, Any], element: str,
                     fx: float, fy: float, fz: float) -> Dict[str, Any]:
    """Add an interstitial atom at the given fractional position."""
    new_atoms = list(structure["atoms"])
    new_atoms.append((element, fx, fy, fz))
    new_s = dict(structure)
    new_s["atoms"] = new_atoms
    new_s["spacegroup"] = structure.get("spacegroup", "P1") + " (interstitial)"
    return new_s


# ---------------------------------------------------------------------------
# Export: POSCAR (VASP) and XYZ formats
# ---------------------------------------------------------------------------

def export_poscar(structure: Dict[str, Any], comment: str = "Generated by QuantumRes") -> str:
    """Export structure to VASP POSCAR format string."""
    mat = lattice_vectors(structure["a"], structure["b"], structure["c"],
                          structure["alpha"], structure["beta"], structure["gamma"])
    atoms = structure["atoms"]

    # Group atoms by element, preserving order of first appearance
    elem_order = []
    elem_groups: Dict[str, List[Tuple[float, float, float]]] = {}
    for sym, fx, fy, fz in atoms:
        if sym not in elem_groups:
            elem_order.append(sym)
            elem_groups[sym] = []
        elem_groups[sym].append((fx, fy, fz))

    lines = [comment]
    lines.append("1.0")
    for row in mat:
        lines.append(f"  {row[0]:16.10f} {row[1]:16.10f} {row[2]:16.10f}")
    lines.append("  " + "  ".join(elem_order))
    lines.append("  " + "  ".join(str(len(elem_groups[e])) for e in elem_order))
    lines.append("Direct")
    for elem in elem_order:
        for fx, fy, fz in elem_groups[elem]:
            lines.append(f"  {fx:12.8f}  {fy:12.8f}  {fz:12.8f}")
    return "\n".join(lines) + "\n"


def export_xyz_crystal(structure: Dict[str, Any]) -> str:
    """Export structure to XYZ format (Cartesian coordinates)."""
    mat = lattice_vectors(structure["a"], structure["b"], structure["c"],
                          structure["alpha"], structure["beta"], structure["gamma"])
    atoms = structure["atoms"]
    lines = [str(len(atoms))]
    sg = structure.get("spacegroup", "P1")
    lines.append(f"Generated by QuantumRes | {sg} | a={structure['a']:.4f} b={structure['b']:.4f} c={structure['c']:.4f}")
    for sym, fx, fy, fz in atoms:
        cart = frac_to_cart(np.array([fx, fy, fz]), mat)
        lines.append(f"{sym:4s} {cart[0]:12.6f} {cart[1]:12.6f} {cart[2]:12.6f}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Bond length/angle measurement
# ---------------------------------------------------------------------------

def compute_all_bonds(structure: Dict[str, Any],
                      cutoff: float = 3.0
                      ) -> List[Tuple[int, int, str, str, float]]:
    """Compute all bonds within cutoff distance.

    Returns list of (i, j, elem_i, elem_j, distance).
    """
    mat = lattice_vectors(structure["a"], structure["b"], structure["c"],
                          structure["alpha"], structure["beta"], structure["gamma"])
    atoms = structure["atoms"]
    n = len(atoms)
    cart_coords = []
    for sym, fx, fy, fz in atoms:
        cart_coords.append(frac_to_cart(np.array([fx, fy, fz]), mat))

    bonds = []
    for i in range(n):
        ri = _element_radius(atoms[i][0])
        for j in range(i + 1, n):
            rj = _element_radius(atoms[j][0])
            dist = np.linalg.norm(cart_coords[i] - cart_coords[j])
            max_bond = ri + rj + 0.45
            if dist < min(cutoff, max_bond) and dist > 0.3:
                bonds.append((i, j, atoms[i][0], atoms[j][0], float(dist)))
    return bonds


def compute_bond_angles(structure: Dict[str, Any],
                        bonds: List[Tuple[int, int, str, str, float]]
                        ) -> List[Tuple[int, int, int, str, str, str, float]]:
    """Compute angles formed by shared-vertex bond pairs.

    Returns list of (i, center, k, elem_i, elem_center, elem_k, angle_degrees).
    """
    mat = lattice_vectors(structure["a"], structure["b"], structure["c"],
                          structure["alpha"], structure["beta"], structure["gamma"])
    atoms = structure["atoms"]
    cart_coords = []
    for sym, fx, fy, fz in atoms:
        cart_coords.append(frac_to_cart(np.array([fx, fy, fz]), mat))

    # Build adjacency
    from collections import defaultdict
    adj = defaultdict(list)
    for i, j, ei, ej, d in bonds:
        adj[i].append(j)
        adj[j].append(i)

    angles = []
    seen = set()
    for center, neighbors in adj.items():
        for ni in range(len(neighbors)):
            for nj in range(ni + 1, len(neighbors)):
                a_idx, b_idx = neighbors[ni], neighbors[nj]
                key = tuple(sorted([a_idx, b_idx])) + (center,)
                if key in seen:
                    continue
                seen.add(key)
                v1 = cart_coords[a_idx] - cart_coords[center]
                v2 = cart_coords[b_idx] - cart_coords[center]
                cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
                cos_a = np.clip(cos_a, -1.0, 1.0)
                angle = math.degrees(math.acos(cos_a))
                angles.append((a_idx, center, b_idx,
                                atoms[a_idx][0], atoms[center][0], atoms[b_idx][0],
                                angle))
    return angles


# ---------------------------------------------------------------------------
# Coordination number analysis
# ---------------------------------------------------------------------------

def compute_coordination(structure: Dict[str, Any],
                         cutoff: float = 3.0
                         ) -> List[Tuple[int, str, int, List[str]]]:
    """Compute coordination number for each atom.

    Returns list of (atom_index, element, coordination_number, neighbor_elements).
    """
    bonds = compute_all_bonds(structure, cutoff)
    from collections import defaultdict
    adj: Dict[int, List[int]] = defaultdict(list)
    for i, j, ei, ej, d in bonds:
        adj[i].append(j)
        adj[j].append(i)

    atoms = structure["atoms"]
    result = []
    for idx, (sym, fx, fy, fz) in enumerate(atoms):
        neighbors = adj.get(idx, [])
        neighbor_elems = [atoms[n][0] for n in neighbors]
        result.append((idx, sym, len(neighbors), neighbor_elems))
    return result


# ---------------------------------------------------------------------------
# The main widget
# ---------------------------------------------------------------------------

class CrystalViewerWidget(QWidget):
    """
    Crystal Structure Viewer widget for PyQt5.

    Provides interactive 3D visualization of crystal structures with
    controls for lattice parameters, supercell generation, bond display,
    Miller plane overlay, and simulated powder XRD.
    """

    structure_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._logger: Optional[Callable] = None
        self._current_structure: Optional[Dict[str, Any]] = None
        self._loaded_file_path: Optional[str] = None

        self._init_ui()
        self._load_builtin("FCC")

    # ------------------------------------------------------------------ log
    def set_logger(self, fn: Callable):
        """Attach an external logging callback ``fn(message: str)``."""
        self._logger = fn

    def _log(self, msg: str):
        if self._logger:
            self._logger(msg)

    # --------------------------------------------------------------- UI
    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left: controls
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setMinimumWidth(280)
        ctrl_scroll.setMaximumWidth(380)
        ctrl_widget = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_widget)
        ctrl_scroll.setWidget(ctrl_widget)

        # --- Structure selector ---
        grp_struct = QGroupBox("Structure")
        gl = QVBoxLayout(grp_struct)
        self._combo_struct = QComboBox()
        self._combo_struct.addItems(sorted(BUILTIN_STRUCTURES.keys()))
        self._combo_struct.currentTextChanged.connect(self._on_structure_selected)
        gl.addWidget(self._combo_struct)

        btn_load = QPushButton("Load CIF File...")
        btn_load.clicked.connect(self._on_load_cif)
        gl.addWidget(btn_load)
        ctrl_layout.addWidget(grp_struct)

        # --- Lattice parameters ---
        grp_lat = QGroupBox("Lattice Parameters")
        lat_grid = QGridLayout(grp_lat)
        self._spins_lat: Dict[str, QDoubleSpinBox] = {}
        labels = [("a", 0, 0), ("b", 0, 2), ("c", 0, 4),
                  ("alpha", 1, 0), ("beta", 1, 2), ("gamma", 1, 4)]
        for name, row, col in labels:
            lat_grid.addWidget(QLabel(name), row, col)
            sp = QDoubleSpinBox()
            sp.setDecimals(3)
            if name in ("alpha", "beta", "gamma"):
                sp.setRange(10.0, 170.0)
                sp.setValue(90.0)
                sp.setSuffix(" deg")
            else:
                sp.setRange(0.1, 100.0)
                sp.setValue(3.0)
                sp.setSuffix(" A")
            sp.valueChanged.connect(self._on_params_changed)
            self._spins_lat[name] = sp
            lat_grid.addWidget(sp, row, col + 1)
        ctrl_layout.addWidget(grp_lat)

        # --- Supercell ---
        grp_super = QGroupBox("Supercell")
        sg = QGridLayout(grp_super)
        self._spins_super: Dict[str, QSpinBox] = {}
        for i, axis in enumerate(("Nx", "Ny", "Nz")):
            sg.addWidget(QLabel(axis), 0, 2 * i)
            sp = QSpinBox()
            sp.setRange(1, 10)
            sp.setValue(1)
            sp.valueChanged.connect(self._on_params_changed)
            self._spins_super[axis] = sp
            sg.addWidget(sp, 0, 2 * i + 1)
        ctrl_layout.addWidget(grp_super)

        # --- Display options ---
        grp_disp = QGroupBox("Display")
        dl = QVBoxLayout(grp_disp)
        self._chk_bonds = QCheckBox("Show bonds")
        self._chk_bonds.setChecked(True)
        self._chk_bonds.stateChanged.connect(self._on_params_changed)
        dl.addWidget(self._chk_bonds)

        self._chk_cell = QCheckBox("Show unit cell box")
        self._chk_cell.setChecked(True)
        self._chk_cell.stateChanged.connect(self._on_params_changed)
        dl.addWidget(self._chk_cell)

        self._chk_axes = QCheckBox("Show axes labels")
        self._chk_axes.setChecked(True)
        self._chk_axes.stateChanged.connect(self._on_params_changed)
        dl.addWidget(self._chk_axes)

        bond_row = QHBoxLayout()
        bond_row.addWidget(QLabel("Bond cutoff"))
        self._spin_bond_cut = QDoubleSpinBox()
        self._spin_bond_cut.setRange(0.5, 10.0)
        self._spin_bond_cut.setValue(3.0)
        self._spin_bond_cut.setDecimals(2)
        self._spin_bond_cut.setSuffix(" A")
        self._spin_bond_cut.valueChanged.connect(self._on_params_changed)
        bond_row.addWidget(self._spin_bond_cut)
        dl.addLayout(bond_row)

        ctrl_layout.addWidget(grp_disp)

        # --- Miller plane ---
        grp_miller = QGroupBox("Miller Plane (hkl)")
        ml = QGridLayout(grp_miller)
        self._spins_miller: Dict[str, QSpinBox] = {}
        for i, lbl in enumerate(("h", "k", "l")):
            ml.addWidget(QLabel(lbl), 0, 2 * i)
            sp = QSpinBox()
            sp.setRange(-9, 9)
            sp.setValue(1 if lbl == "h" else 0)
            self._spins_miller[lbl] = sp
            ml.addWidget(sp, 0, 2 * i + 1)
        self._chk_miller = QCheckBox("Show plane")
        self._chk_miller.setChecked(False)
        self._chk_miller.stateChanged.connect(self._on_params_changed)
        ml.addWidget(self._chk_miller, 1, 0, 1, 6)
        btn_miller = QPushButton("Update Plane")
        btn_miller.clicked.connect(self._on_params_changed)
        ml.addWidget(btn_miller, 2, 0, 1, 6)
        ctrl_layout.addWidget(grp_miller)

        # --- Crystal Structure Builder ---
        grp_builder = QGroupBox("Crystal Builder")
        bld_lay = QFormLayout(grp_builder)
        self._combo_spacegroup = QComboBox()
        self._combo_spacegroup.addItems(AVAILABLE_SPACEGROUPS)
        bld_lay.addRow("Space Group:", self._combo_spacegroup)
        self._txt_wyckoff = QPlainTextEdit()
        self._txt_wyckoff.setPlaceholderText("Element fx fy fz (one per line)\ne.g.:\nSi 0.0 0.0 0.0\nO 0.25 0.25 0.25")
        self._txt_wyckoff.setMaximumHeight(80)
        bld_lay.addRow("Wyckoff Pos:", self._txt_wyckoff)
        btn_build = QPushButton("Build Crystal")
        btn_build.clicked.connect(self._on_build_crystal)
        bld_lay.addRow(btn_build)
        ctrl_layout.addWidget(grp_builder)

        # --- Surface Slab Generator ---
        grp_slab = QGroupBox("Surface Slab")
        slab_lay = QFormLayout(grp_slab)
        self._slab_h = QSpinBox()
        self._slab_h.setRange(-9, 9); self._slab_h.setValue(1)
        self._slab_k = QSpinBox()
        self._slab_k.setRange(-9, 9); self._slab_k.setValue(0)
        self._slab_l = QSpinBox()
        self._slab_l.setRange(-9, 9); self._slab_l.setValue(0)
        hkl_row = QHBoxLayout()
        hkl_row.addWidget(QLabel("h")); hkl_row.addWidget(self._slab_h)
        hkl_row.addWidget(QLabel("k")); hkl_row.addWidget(self._slab_k)
        hkl_row.addWidget(QLabel("l")); hkl_row.addWidget(self._slab_l)
        slab_lay.addRow(hkl_row)
        self._spin_slab_layers = QSpinBox()
        self._spin_slab_layers.setRange(1, 20); self._spin_slab_layers.setValue(4)
        slab_lay.addRow("Layers:", self._spin_slab_layers)
        self._spin_vacuum = QDoubleSpinBox()
        self._spin_vacuum.setRange(1.0, 50.0); self._spin_vacuum.setValue(15.0)
        self._spin_vacuum.setSuffix(" A")
        slab_lay.addRow("Vacuum:", self._spin_vacuum)
        btn_slab = QPushButton("Generate Slab")
        btn_slab.clicked.connect(self._on_generate_slab)
        slab_lay.addRow(btn_slab)
        ctrl_layout.addWidget(grp_slab)

        # --- Defect Generator ---
        grp_defect = QGroupBox("Defects")
        def_lay = QFormLayout(grp_defect)
        self._spin_defect_idx = QSpinBox()
        self._spin_defect_idx.setRange(0, 999)
        self._spin_defect_idx.setValue(0)
        def_lay.addRow("Atom Index:", self._spin_defect_idx)
        self._txt_defect_elem = QLineEdit("X")
        self._txt_defect_elem.setMaximumWidth(60)
        def_lay.addRow("Element:", self._txt_defect_elem)
        def_btn_row = QHBoxLayout()
        btn_vacancy = QPushButton("Vacancy")
        btn_vacancy.clicked.connect(self._on_add_vacancy)
        def_btn_row.addWidget(btn_vacancy)
        btn_subst = QPushButton("Substitution")
        btn_subst.clicked.connect(self._on_add_substitutional)
        def_btn_row.addWidget(btn_subst)
        def_lay.addRow(def_btn_row)
        int_row = QHBoxLayout()
        self._spin_int_x = QDoubleSpinBox()
        self._spin_int_x.setRange(0, 1); self._spin_int_x.setValue(0.25); self._spin_int_x.setDecimals(4)
        self._spin_int_y = QDoubleSpinBox()
        self._spin_int_y.setRange(0, 1); self._spin_int_y.setValue(0.25); self._spin_int_y.setDecimals(4)
        self._spin_int_z = QDoubleSpinBox()
        self._spin_int_z.setRange(0, 1); self._spin_int_z.setValue(0.25); self._spin_int_z.setDecimals(4)
        int_row.addWidget(QLabel("x")); int_row.addWidget(self._spin_int_x)
        int_row.addWidget(QLabel("y")); int_row.addWidget(self._spin_int_y)
        int_row.addWidget(QLabel("z")); int_row.addWidget(self._spin_int_z)
        def_lay.addRow("Interstitial:", int_row)
        btn_interstitial = QPushButton("Add Interstitial")
        btn_interstitial.clicked.connect(self._on_add_interstitial)
        def_lay.addRow(btn_interstitial)
        ctrl_layout.addWidget(grp_defect)

        # --- Buttons ---
        btn_refresh = QPushButton("Refresh View")
        btn_refresh.clicked.connect(self._on_params_changed)
        ctrl_layout.addWidget(btn_refresh)

        # Export buttons
        export_row = QHBoxLayout()
        btn_export = QPushButton("Export JSON")
        btn_export.clicked.connect(self._on_export_clicked)
        export_row.addWidget(btn_export)
        btn_poscar = QPushButton("Export POSCAR")
        btn_poscar.clicked.connect(self._on_export_poscar)
        export_row.addWidget(btn_poscar)
        btn_xyz = QPushButton("Export XYZ")
        btn_xyz.clicked.connect(self._on_export_xyz)
        export_row.addWidget(btn_xyz)
        ctrl_layout.addLayout(export_row)

        ctrl_layout.addStretch()
        splitter.addWidget(ctrl_scroll)

        # Right: canvas + tabs
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Tabs: 3D view + XRD + Info
        self._tabs = QTabWidget()

        # --- 3D canvas tab ---
        canvas_widget = QWidget()
        canvas_layout = QVBoxLayout(canvas_widget)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        self._figure = Figure(figsize=(6, 6), dpi=100)
        style_figure(self._figure)
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas_layout.addWidget(self._canvas)
        self._tabs.addTab(canvas_widget, "3D View")

        # --- XRD tab ---
        xrd_widget = QWidget()
        xrd_layout = QVBoxLayout(xrd_widget)
        xrd_layout.setContentsMargins(0, 0, 0, 0)
        self._xrd_figure = Figure(figsize=(6, 3), dpi=100)
        style_figure(self._xrd_figure)
        self._xrd_canvas = FigureCanvas(self._xrd_figure)
        self._xrd_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        xrd_layout.addWidget(self._xrd_canvas)

        xrd_ctrl = QHBoxLayout()
        xrd_ctrl.addWidget(QLabel("Wavelength (A):"))
        self._spin_wavelength = QDoubleSpinBox()
        self._spin_wavelength.setRange(0.1, 5.0)
        self._spin_wavelength.setValue(1.5406)
        self._spin_wavelength.setDecimals(4)
        xrd_ctrl.addWidget(self._spin_wavelength)
        btn_xrd = QPushButton("Simulate XRD")
        btn_xrd.clicked.connect(self._on_simulate_xrd)
        xrd_ctrl.addWidget(btn_xrd)
        xrd_ctrl.addStretch()
        xrd_layout.addLayout(xrd_ctrl)

        self._tabs.addTab(xrd_widget, "XRD Pattern")

        # --- Info tab ---
        self._info_text = QTextEdit()
        self._info_text.setReadOnly(True)
        self._tabs.addTab(self._info_text, "Crystal Info")

        # --- Bond Analysis tab ---
        bond_widget = QWidget()
        bond_layout = QVBoxLayout(bond_widget)
        bond_layout.setContentsMargins(2, 2, 2, 2)
        bond_ctrl = QHBoxLayout()
        bond_ctrl.addWidget(QLabel("Cutoff:"))
        self._spin_bond_analysis_cut = QDoubleSpinBox()
        self._spin_bond_analysis_cut.setRange(0.5, 10.0)
        self._spin_bond_analysis_cut.setValue(3.0)
        self._spin_bond_analysis_cut.setSuffix(" A")
        bond_ctrl.addWidget(self._spin_bond_analysis_cut)
        btn_analyze_bonds = QPushButton("Analyze Bonds")
        btn_analyze_bonds.clicked.connect(self._on_analyze_bonds)
        bond_ctrl.addWidget(btn_analyze_bonds)
        bond_ctrl.addStretch()
        bond_layout.addLayout(bond_ctrl)
        self._bond_table = QTableWidget(0, 5)
        self._bond_table.setHorizontalHeaderLabels(["#", "Atom 1", "Atom 2", "Distance (A)", "Type"])
        self._bond_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        bond_layout.addWidget(self._bond_table)
        self._angle_table = QTableWidget(0, 5)
        self._angle_table.setHorizontalHeaderLabels(["#", "Atom 1", "Center", "Atom 3", "Angle (deg)"])
        self._angle_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        bond_layout.addWidget(self._angle_table)
        self._tabs.addTab(bond_widget, "Bond Analysis")

        # --- Coordination tab ---
        coord_widget = QWidget()
        coord_layout = QVBoxLayout(coord_widget)
        coord_layout.setContentsMargins(2, 2, 2, 2)
        coord_ctrl = QHBoxLayout()
        coord_ctrl.addWidget(QLabel("Cutoff:"))
        self._spin_coord_cut = QDoubleSpinBox()
        self._spin_coord_cut.setRange(0.5, 10.0)
        self._spin_coord_cut.setValue(3.0)
        self._spin_coord_cut.setSuffix(" A")
        coord_ctrl.addWidget(self._spin_coord_cut)
        btn_coord = QPushButton("Analyze Coordination")
        btn_coord.clicked.connect(self._on_analyze_coordination)
        coord_ctrl.addWidget(btn_coord)
        coord_ctrl.addStretch()
        coord_layout.addLayout(coord_ctrl)
        self._coord_table = QTableWidget(0, 4)
        self._coord_table.setHorizontalHeaderLabels(["Atom #", "Element", "CN", "Neighbors"])
        self._coord_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        coord_layout.addWidget(self._coord_table)
        self._coord_figure = Figure(figsize=(6, 3), dpi=100)
        style_figure(self._coord_figure)
        self._coord_canvas = FigureCanvas(self._coord_figure)
        coord_layout.addWidget(self._coord_canvas)
        self._tabs.addTab(coord_widget, "Coordination")

        right_layout.addWidget(self._tabs)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    # -------------------------------------------------------- Structure loading

    def _load_builtin(self, name: str):
        if name not in BUILTIN_STRUCTURES:
            return
        s = BUILTIN_STRUCTURES[name]
        self._current_structure = dict(s)
        self._apply_structure_to_ui(s)
        self._update_view()
        self._log(f"Loaded built-in structure: {name}")

    def _apply_structure_to_ui(self, s: Dict[str, Any]):
        """Push structure dict values into the UI spin boxes."""
        for key in ("a", "b", "c", "alpha", "beta", "gamma"):
            self._spins_lat[key].blockSignals(True)
            self._spins_lat[key].setValue(s[key])
            self._spins_lat[key].blockSignals(False)

    def _on_structure_selected(self, name: str):
        self._load_builtin(name)

    def _on_load_cif(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open CIF File", "", "CIF Files (*.cif);;All Files (*)"
        )
        if path:
            self.load_file(path)

    def load_file(self, path: str):
        """Load a CIF file from *path* and display the structure."""
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
            s = parse_cif(text)
            self._current_structure = s
            self._loaded_file_path = path
            self._apply_structure_to_ui(s)
            self._update_view()
            self._log(f"Loaded CIF: {path}")
        except Exception as exc:
            self._log(f"Error loading CIF: {exc}")
            QMessageBox.warning(self, "CIF Error", f"Failed to load CIF file:\n{exc}")

    # -------------------------------------------------------- Parameter changes

    def _on_params_changed(self, *_args):
        self._update_current_structure_from_ui()
        self._update_view()

    def _update_current_structure_from_ui(self):
        if self._current_structure is None:
            return
        for key in ("a", "b", "c", "alpha", "beta", "gamma"):
            self._current_structure[key] = self._spins_lat[key].value()

    # -------------------------------------------------------- 3D rendering

    def _update_view(self):
        if self._current_structure is None:
            return

        s = self._current_structure
        mat = lattice_vectors(s["a"], s["b"], s["c"],
                              s["alpha"], s["beta"], s["gamma"])

        nx = self._spins_super["Nx"].value()
        ny = self._spins_super["Ny"].value()
        nz = self._spins_super["Nz"].value()

        # Build supercell atoms in Cartesian
        cart_atoms: List[Tuple[str, np.ndarray]] = []
        for sym, fx, fy, fz in s["atoms"]:
            for ix in range(nx):
                for iy in range(ny):
                    for iz in range(nz):
                        frac = np.array([fx + ix, fy + iy, fz + iz])
                        pos = frac_to_cart(frac, mat)
                        cart_atoms.append((sym, pos))

        # Supercell lattice matrix
        sup_mat = mat.copy()
        sup_mat[0] *= nx
        sup_mat[1] *= ny
        sup_mat[2] *= nz

        self._figure.clear()
        ax = self._figure.add_subplot(111, projection="3d")
        ax.set_facecolor("#1a1a2e")
        self._figure.patch.set_facecolor("#0f0f23")

        # Plot atoms
        for sym, pos in cart_atoms:
            color = _element_color(sym)
            rad = _element_radius(sym) * 0.35
            ax.scatter(*pos, s=rad * 200, c=color, edgecolors="black",
                       linewidths=0.3, alpha=0.95, depthshade=True)

        # Bonds
        if self._chk_bonds.isChecked() and len(cart_atoms) > 1:
            cutoff = self._spin_bond_cut.value()
            bond_lines = []
            n = len(cart_atoms)
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.linalg.norm(cart_atoms[i][1] - cart_atoms[j][1])
                    if d < cutoff:
                        bond_lines.append([cart_atoms[i][1], cart_atoms[j][1]])
            if bond_lines:
                lc = Line3DCollection(bond_lines, colors="#aaaaaa",
                                      linewidths=0.8, alpha=0.5)
                ax.add_collection3d(lc)

        # Unit cell box
        if self._chk_cell.isChecked():
            self._draw_cell_box(ax, sup_mat)

        # Miller plane
        if self._chk_miller.isChecked():
            h_val = self._spins_miller["h"].value()
            k_val = self._spins_miller["k"].value()
            l_val = self._spins_miller["l"].value()
            if not (h_val == 0 and k_val == 0 and l_val == 0):
                self._draw_miller_plane(ax, mat, h_val, k_val, l_val, nx, ny, nz)

        # Axes labels
        if self._chk_axes.isChecked():
            origin = np.zeros(3)
            for i, (label, color) in enumerate(zip(["a", "b", "c"],
                                                     ["#ff4444", "#44ff44", "#4444ff"])):
                vec = sup_mat[i]
                ax.quiver(*origin, *vec, color=color, arrow_length_ratio=0.08,
                          linewidth=1.5, alpha=0.8)
                ax.text(*(vec * 1.1), label, color=color, fontsize=10, fontweight="bold")

        # Axis settings
        all_pos = np.array([p for _, p in cart_atoms]) if cart_atoms else np.zeros((1, 3))
        margin = 1.5
        for setter, idx in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
            lo = all_pos[:, idx].min() - margin
            hi = all_pos[:, idx].max() + margin
            setter(lo, hi)

        ax.set_xlabel("X (A)", color="white", fontsize=8)
        ax.set_ylabel("Y (A)", color="white", fontsize=8)
        ax.set_zlabel("Z (A)", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=6)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        self._canvas.draw()
        self._update_info_panel()
        self.structure_changed.emit()

    def _draw_cell_box(self, ax, mat: np.ndarray):
        """Draw the parallelepiped unit cell."""
        o = np.zeros(3)
        a, b, c = mat[0], mat[1], mat[2]
        corners = [o, a, b, c, a + b, a + c, b + c, a + b + c]
        edges = [
            (0, 1), (0, 2), (0, 3),
            (1, 4), (1, 5),
            (2, 4), (2, 6),
            (3, 5), (3, 6),
            (4, 7), (5, 7), (6, 7),
        ]
        lines = [[corners[i], corners[j]] for i, j in edges]
        lc = Line3DCollection(lines, colors="#ffffff", linewidths=1.0, alpha=0.4,
                              linestyles="dashed")
        ax.add_collection3d(lc)

    def _draw_miller_plane(self, ax, mat: np.ndarray,
                           h: int, k: int, l: int,
                           nx: int, ny: int, nz: int):
        """Draw a Miller plane (hkl) as a semi-transparent polygon."""
        intercepts = []
        axes = [mat[0] * nx, mat[1] * ny, mat[2] * nz]
        hkl = [h, k, l]

        for i, idx_val in enumerate(hkl):
            if idx_val != 0:
                intercepts.append(axes[i] / idx_val)

        if len(intercepts) < 2:
            return

        if len(intercepts) == 2:
            # Plane parallel to one axis; extend along that axis
            missing_axis = [i for i, v in enumerate(hkl) if v == 0][0]
            ext = axes[missing_axis]
            p1, p2 = intercepts[0], intercepts[1]
            verts = [p1, p2, p2 + ext, p1 + ext]
        else:
            verts = intercepts

        poly = Poly3DCollection([verts], alpha=0.25, facecolor="#ffcc00",
                                edgecolor="#ffcc00", linewidth=1.5)
        ax.add_collection3d(poly)

    # -------------------------------------------------------- XRD

    def _on_simulate_xrd(self):
        if self._current_structure is None:
            return
        s = self._current_structure
        mat = lattice_vectors(s["a"], s["b"], s["c"],
                              s["alpha"], s["beta"], s["gamma"])
        wl = self._spin_wavelength.value()

        two_theta, intensity = simulate_xrd(mat, s["atoms"], wavelength=wl)

        self._xrd_figure.clear()
        ax = self._xrd_figure.add_subplot(111)
        ax.set_facecolor("#0f0f23")
        self._xrd_figure.patch.set_facecolor("#0f0f23")
        ax.plot(two_theta, intensity, color="#00ccff", linewidth=0.8)
        ax.fill_between(two_theta, intensity, alpha=0.15, color="#00ccff")
        ax.set_xlabel("2-theta (degrees)", color="white", fontsize=9)
        ax.set_ylabel("Intensity (a.u.)", color="white", fontsize=9)
        ax.set_title(f"Simulated Powder XRD (lambda={wl:.4f} A)", color="white", fontsize=10)
        ax.tick_params(colors="white", labelsize=7)
        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(5, 90)
        self._xrd_canvas.draw()
        self._log("XRD simulation complete")

    # -------------------------------------------------------- Info panel

    def _update_info_panel(self):
        if self._current_structure is None:
            return
        s = self._current_structure
        mat = lattice_vectors(s["a"], s["b"], s["c"],
                              s["alpha"], s["beta"], s["gamma"])
        vol = cell_volume(mat)
        sg = s.get("spacegroup", "Unknown")

        # Density estimate
        total_mass = 0.0
        elem_count: Dict[str, int] = {}
        for sym, *_ in s["atoms"]:
            total_mass += ATOMIC_MASSES.get(sym, 28.0)
            elem_count[sym] = elem_count.get(sym, 0) + 1

        avogadro = 6.02214076e23
        vol_cm3 = vol * 1e-24  # Angstrom^3 -> cm^3
        if vol_cm3 > 0:
            density = (total_mass / avogadro) / vol_cm3
        else:
            density = 0.0

        lines = [
            "=== Crystal Structure Information ===",
            "",
            f"Space Group:  {sg}",
            "",
            "--- Lattice Parameters ---",
            f"  a = {s['a']:.4f} A",
            f"  b = {s['b']:.4f} A",
            f"  c = {s['c']:.4f} A",
            f"  alpha = {s['alpha']:.2f} deg",
            f"  beta  = {s['beta']:.2f} deg",
            f"  gamma = {s['gamma']:.2f} deg",
            "",
            f"Cell Volume:  {vol:.4f} A^3",
            f"Density (est):  {density:.4f} g/cm^3",
            "",
            "--- Lattice Vectors (Cartesian) ---",
            f"  a = [{mat[0][0]:.4f}, {mat[0][1]:.4f}, {mat[0][2]:.4f}]",
            f"  b = [{mat[1][0]:.4f}, {mat[1][1]:.4f}, {mat[1][2]:.4f}]",
            f"  c = [{mat[2][0]:.4f}, {mat[2][1]:.4f}, {mat[2][2]:.4f}]",
            "",
            "--- Reciprocal Lattice Vectors ---",
        ]
        try:
            recip = np.linalg.inv(mat).T * 2.0 * np.pi
            lines.append(f"  a* = [{recip[0][0]:.4f}, {recip[0][1]:.4f}, {recip[0][2]:.4f}]")
            lines.append(f"  b* = [{recip[1][0]:.4f}, {recip[1][1]:.4f}, {recip[1][2]:.4f}]")
            lines.append(f"  c* = [{recip[2][0]:.4f}, {recip[2][1]:.4f}, {recip[2][2]:.4f}]")
        except np.linalg.LinAlgError:
            lines.append("  (singular matrix)")

        lines += [
            "",
            f"--- Atoms in Unit Cell ({len(s['atoms'])}) ---",
        ]
        for sym, fx, fy, fz in s["atoms"]:
            lines.append(f"  {sym:4s}  ({fx:.4f}, {fy:.4f}, {fz:.4f})")

        lines += [
            "",
            "--- Element Composition ---",
        ]
        for sym in sorted(elem_count.keys()):
            cnt = elem_count[sym]
            mass = ATOMIC_MASSES.get(sym, 0.0)
            lines.append(f"  {sym}: {cnt} atom(s), mass = {mass:.3f} amu")

        lines += [
            "",
            f"--- Atom Count ---",
            f"  Unit cell atoms: {len(s['atoms'])}",
            f"  Unique elements: {len(elem_count)}",
        ]

        # Nearest-neighbor distance calculation
        if len(s["atoms"]) >= 2:
            cart_coords = []
            for sym, fx, fy, fz in s["atoms"]:
                cart = frac_to_cart(np.array([fx, fy, fz]), mat)
                cart_coords.append(cart)
            cart_coords = np.array(cart_coords)
            min_dist = float("inf")
            nn_pair = ("", "")
            for i in range(len(cart_coords)):
                for j in range(i + 1, len(cart_coords)):
                    d = np.linalg.norm(cart_coords[i] - cart_coords[j])
                    if d < min_dist and d > 1e-6:
                        min_dist = d
                        nn_pair = (s["atoms"][i][0], s["atoms"][j][0])
            if min_dist < float("inf"):
                lines += [
                    "",
                    f"--- Nearest-Neighbor Distance ---",
                    f"  {nn_pair[0]} - {nn_pair[1]}: {min_dist:.4f} A",
                ]

        nx = self._spins_super["Nx"].value()
        ny = self._spins_super["Ny"].value()
        nz = self._spins_super["Nz"].value()
        total_atoms = len(s["atoms"]) * nx * ny * nz
        lines += [
            "",
            f"--- Supercell ---",
            f"  Repetitions: {nx} x {ny} x {nz}",
            f"  Total atoms: {total_atoms}",
            f"  Supercell volume: {vol * nx * ny * nz:.4f} A^3",
        ]

        if self._loaded_file_path:
            lines += ["", f"Source file: {self._loaded_file_path}"]

        self._info_text.setPlainText("\n".join(lines))

    # -------------------------------------------------------- Crystal Builder

    def _on_build_crystal(self):
        """Build a crystal from space group and Wyckoff positions."""
        if self._current_structure is None:
            self._current_structure = {
                "a": 5.0, "b": 5.0, "c": 5.0,
                "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
                "atoms": [], "spacegroup": "P1",
            }
        sg = self._combo_spacegroup.currentText()
        text = self._txt_wyckoff.toPlainText().strip()
        if not text:
            self._log("No Wyckoff positions specified.")
            return

        wyckoff_atoms = []
        for line in text.splitlines():
            parts = line.split()
            if len(parts) >= 4:
                try:
                    sym = parts[0].capitalize()
                    fx, fy, fz = float(parts[1]), float(parts[2]), float(parts[3])
                    wyckoff_atoms.append((sym, fx, fy, fz))
                except ValueError:
                    continue

        if not wyckoff_atoms:
            self._log("Could not parse Wyckoff positions.")
            return

        # Apply symmetry operations
        all_atoms = apply_symmetry(wyckoff_atoms, sg)

        # Update structure using current lattice params
        self._update_current_structure_from_ui()
        self._current_structure["atoms"] = all_atoms
        self._current_structure["spacegroup"] = sg
        self._update_view()
        self._log(f"Built crystal: {sg} with {len(all_atoms)} atoms from "
                  f"{len(wyckoff_atoms)} Wyckoff position(s)")

    # -------------------------------------------------------- Surface Slab

    def _on_generate_slab(self):
        """Generate a surface slab from current structure."""
        if self._current_structure is None:
            self._log("No structure loaded.")
            return
        self._update_current_structure_from_ui()
        h = self._slab_h.value()
        k = self._slab_k.value()
        l_val = self._slab_l.value()
        if h == 0 and k == 0 and l_val == 0:
            self._log("Miller indices (0,0,0) are not valid.")
            return
        n_layers = self._spin_slab_layers.value()
        vacuum = self._spin_vacuum.value()

        slab = generate_surface_slab(self._current_structure, h, k, l_val,
                                     n_layers=n_layers, vacuum=vacuum)
        self._current_structure = slab
        self._apply_structure_to_ui(slab)
        self._update_view()
        self._log(f"Generated ({h}{k}{l_val}) surface slab: "
                  f"{len(slab['atoms'])} atoms, {n_layers} layers, {vacuum} A vacuum")

    # -------------------------------------------------------- Defects

    def _on_add_vacancy(self):
        if self._current_structure is None:
            return
        self._update_current_structure_from_ui()
        idx = self._spin_defect_idx.value()
        n = len(self._current_structure["atoms"])
        if idx >= n:
            self._log(f"Atom index {idx} out of range (0-{n-1}).")
            return
        removed = self._current_structure["atoms"][idx]
        self._current_structure = add_vacancy(self._current_structure, idx)
        self._update_view()
        self._log(f"Created vacancy: removed {removed[0]} at index {idx}")

    def _on_add_substitutional(self):
        if self._current_structure is None:
            return
        self._update_current_structure_from_ui()
        idx = self._spin_defect_idx.value()
        elem = self._txt_defect_elem.text().strip().capitalize()
        if not elem:
            self._log("Specify an element symbol for substitution.")
            return
        n = len(self._current_structure["atoms"])
        if idx >= n:
            self._log(f"Atom index {idx} out of range (0-{n-1}).")
            return
        old_elem = self._current_structure["atoms"][idx][0]
        self._current_structure = add_substitutional(self._current_structure, idx, elem)
        self._update_view()
        self._log(f"Substitution: replaced {old_elem} with {elem} at index {idx}")

    def _on_add_interstitial(self):
        if self._current_structure is None:
            return
        self._update_current_structure_from_ui()
        elem = self._txt_defect_elem.text().strip().capitalize()
        if not elem:
            self._log("Specify an element symbol for interstitial.")
            return
        fx = self._spin_int_x.value()
        fy = self._spin_int_y.value()
        fz = self._spin_int_z.value()
        self._current_structure = add_interstitial(self._current_structure, elem, fx, fy, fz)
        self._update_view()
        self._log(f"Added interstitial {elem} at ({fx:.4f}, {fy:.4f}, {fz:.4f})")

    # -------------------------------------------------------- Bond Analysis

    def _on_analyze_bonds(self):
        if self._current_structure is None:
            return
        self._update_current_structure_from_ui()
        cutoff = self._spin_bond_analysis_cut.value()

        bonds = compute_all_bonds(self._current_structure, cutoff)
        angles = compute_bond_angles(self._current_structure, bonds)

        # Fill bond table
        self._bond_table.setRowCount(0)
        for row_idx, (i, j, ei, ej, dist) in enumerate(bonds):
            self._bond_table.insertRow(row_idx)
            self._bond_table.setItem(row_idx, 0, QTableWidgetItem(str(row_idx + 1)))
            self._bond_table.setItem(row_idx, 1, QTableWidgetItem(f"{ei}#{i}"))
            self._bond_table.setItem(row_idx, 2, QTableWidgetItem(f"{ej}#{j}"))
            self._bond_table.setItem(row_idx, 3, QTableWidgetItem(f"{dist:.4f}"))
            self._bond_table.setItem(row_idx, 4, QTableWidgetItem(f"{ei}-{ej}"))

        # Fill angle table
        self._angle_table.setRowCount(0)
        for row_idx, (a, c, b, ea, ec, eb, angle) in enumerate(angles):
            self._angle_table.insertRow(row_idx)
            self._angle_table.setItem(row_idx, 0, QTableWidgetItem(str(row_idx + 1)))
            self._angle_table.setItem(row_idx, 1, QTableWidgetItem(f"{ea}#{a}"))
            self._angle_table.setItem(row_idx, 2, QTableWidgetItem(f"{ec}#{c}"))
            self._angle_table.setItem(row_idx, 3, QTableWidgetItem(f"{eb}#{b}"))
            self._angle_table.setItem(row_idx, 4, QTableWidgetItem(f"{angle:.2f}"))

        self._log(f"Bond analysis: {len(bonds)} bonds, {len(angles)} angles (cutoff={cutoff} A)")

    # -------------------------------------------------------- Coordination

    def _on_analyze_coordination(self):
        if self._current_structure is None:
            return
        self._update_current_structure_from_ui()
        cutoff = self._spin_coord_cut.value()

        coord_data = compute_coordination(self._current_structure, cutoff)

        # Fill coordination table
        self._coord_table.setRowCount(0)
        for row_idx, (idx, elem, cn, neighbors) in enumerate(coord_data):
            self._coord_table.insertRow(row_idx)
            self._coord_table.setItem(row_idx, 0, QTableWidgetItem(str(idx)))
            self._coord_table.setItem(row_idx, 1, QTableWidgetItem(elem))
            self._coord_table.setItem(row_idx, 2, QTableWidgetItem(str(cn)))
            self._coord_table.setItem(row_idx, 3, QTableWidgetItem(", ".join(neighbors)))

        # Plot coordination number distribution
        self._coord_figure.clear()
        ax = self._coord_figure.add_subplot(111)
        ax.set_facecolor("#0f0f23")
        self._coord_figure.patch.set_facecolor("#0f0f23")

        # Group by element
        from collections import defaultdict
        elem_cn: Dict[str, List[int]] = defaultdict(list)
        for idx, elem, cn, _ in coord_data:
            elem_cn[elem].append(cn)

        if elem_cn:
            elements = sorted(elem_cn.keys())
            avg_cns = [np.mean(elem_cn[e]) for e in elements]
            colors = [_element_color(e) for e in elements]
            bars = ax.bar(range(len(elements)), avg_cns, color=colors,
                          edgecolor="white", linewidth=0.5)
            ax.set_xticks(range(len(elements)))
            ax.set_xticklabels(elements, color="white", fontsize=9)
            ax.set_ylabel("Avg. Coordination Number", color="white", fontsize=9)
            ax.set_title("Coordination Number by Element", color="white", fontsize=10)
            ax.tick_params(colors="white", labelsize=7)
            # Add value labels on bars
            for bar, val in zip(bars, avg_cns):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f"{val:.1f}", ha="center", va="bottom", color="white", fontsize=8)

        self._coord_canvas.draw()
        self._log(f"Coordination analysis complete: {len(coord_data)} atoms analyzed")

    # -------------------------------------------------------- Export (POSCAR/XYZ)

    def _on_export_poscar(self):
        if self._current_structure is None:
            return
        self._update_current_structure_from_ui()
        path, _ = QFileDialog.getSaveFileName(
            self, "Export POSCAR", "POSCAR",
            "POSCAR Files (POSCAR*);;All Files (*)")
        if not path:
            return
        try:
            content = export_poscar(self._current_structure)
            Path(path).write_text(content, encoding="utf-8")
            self._log(f"Exported POSCAR to {path}")
        except Exception as exc:
            self._log(f"POSCAR export error: {exc}")
            QMessageBox.warning(self, "Export Error", str(exc))

    def _on_export_xyz(self):
        if self._current_structure is None:
            return
        self._update_current_structure_from_ui()
        path, _ = QFileDialog.getSaveFileName(
            self, "Export XYZ", "structure.xyz",
            "XYZ Files (*.xyz);;All Files (*)")
        if not path:
            return
        try:
            content = export_xyz_crystal(self._current_structure)
            Path(path).write_text(content, encoding="utf-8")
            self._log(f"Exported XYZ to {path}")
        except Exception as exc:
            self._log(f"XYZ export error: {exc}")
            QMessageBox.warning(self, "Export Error", str(exc))

    # -------------------------------------------------------- Export (JSON)

    def export(self) -> dict:
        """
        Export the current crystal structure as a dictionary suitable for
        JSON serialization.
        """
        if self._current_structure is None:
            return {}

        s = self._current_structure
        mat = lattice_vectors(s["a"], s["b"], s["c"],
                              s["alpha"], s["beta"], s["gamma"])

        nx = self._spins_super["Nx"].value()
        ny = self._spins_super["Ny"].value()
        nz = self._spins_super["Nz"].value()

        data = {
            "lattice_parameters": {
                "a": s["a"], "b": s["b"], "c": s["c"],
                "alpha": s["alpha"], "beta": s["beta"], "gamma": s["gamma"],
            },
            "spacegroup": s.get("spacegroup", "Unknown"),
            "lattice_vectors": mat.tolist(),
            "cell_volume_angstrom3": cell_volume(mat),
            "atoms_fractional": [
                {"element": sym, "x": fx, "y": fy, "z": fz}
                for sym, fx, fy, fz in s["atoms"]
            ],
            "supercell": {"Nx": nx, "Ny": ny, "Nz": nz},
            "source_file": self._loaded_file_path,
        }

        self._log("Structure exported")
        return data

    def _on_export_clicked(self):
        data = self.export()
        if not data:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Structure", "structure.json",
            "JSON Files (*.json);;All Files (*)"
        )
        if path:
            try:
                Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
                self._log(f"Exported to {path}")
            except Exception as exc:
                self._log(f"Export error: {exc}")
                QMessageBox.warning(self, "Export Error", str(exc))


# ---------------------------------------------------------------------------
# Standalone launcher for testing
# ---------------------------------------------------------------------------

def main():
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = QWidget()
    window.setWindowTitle("Crystal Structure Viewer")
    window.resize(1200, 800)
    layout = QVBoxLayout(window)

    viewer = CrystalViewerWidget()
    viewer.set_logger(lambda msg: print(f"[CrystalViewer] {msg}"))
    layout.addWidget(viewer)

    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
