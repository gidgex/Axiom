"""
Molecule Viewer Widget
======================
A PyQt5 widget providing 3D molecule visualization with ball-and-stick
representation, file loading (XYZ/PDB), built-in molecules, measurement
tools, and an interactive coordinate table.
"""

import math
import os
import re
import random
from collections import Counter
from typing import List, Tuple, Dict, Optional

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox,
    QComboBox, QPushButton, QLabel, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QTextEdit, QCheckBox, QSlider,
    QFormLayout, QMessageBox, QAbstractItemView, QTabWidget,
    QLineEdit, QSpinBox, QDoubleSpinBox, QDialog, QDialogButtonBox,
    QGridLayout, QPlainTextEdit,
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    from mpl_style import style_figure, style_axes
except ImportError:
    def style_figure(f): pass
    def style_axes(a): pass
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------------------------------------------------------------------
# Element data: symbol -> (name, CPK color hex, van der Waals radius A, mass)
# ---------------------------------------------------------------------------
ELEMENT_DATA = {
    "H":  ("Hydrogen",   "#FFFFFF", 1.20, 1.008),
    "He": ("Helium",     "#D9FFFF", 1.40, 4.003),
    "Li": ("Lithium",    "#CC80FF", 1.82, 6.941),
    "Be": ("Beryllium",  "#C2FF00", 1.53, 9.012),
    "B":  ("Boron",      "#FFB5B5", 1.92, 10.81),
    "C":  ("Carbon",     "#404040", 1.70, 12.011),
    "N":  ("Nitrogen",   "#3050F8", 1.55, 14.007),
    "O":  ("Oxygen",     "#FF0D0D", 1.52, 15.999),
    "F":  ("Fluorine",   "#90E050", 1.47, 18.998),
    "Ne": ("Neon",       "#B3E3F5", 1.54, 20.180),
    "Na": ("Sodium",     "#AB5CF2", 2.27, 22.990),
    "Mg": ("Magnesium",  "#8AFF00", 1.73, 24.305),
    "Al": ("Aluminium",  "#BFA6A6", 1.84, 26.982),
    "Si": ("Silicon",    "#F0C8A0", 2.10, 28.086),
    "P":  ("Phosphorus", "#FF8000", 1.80, 30.974),
    "S":  ("Sulfur",     "#FFFF30", 1.80, 32.065),
    "Cl": ("Chlorine",   "#1FF01F", 1.75, 35.453),
    "Ar": ("Argon",      "#80D1E3", 1.88, 39.948),
    "K":  ("Potassium",  "#8F40D4", 2.75, 39.098),
    "Ca": ("Calcium",    "#3DFF00", 2.31, 40.078),
    "Fe": ("Iron",       "#E06633", 2.04, 55.845),
    "Co": ("Cobalt",     "#F090A0", 2.00, 58.933),
    "Ni": ("Nickel",     "#50D050", 1.97, 58.693),
    "Cu": ("Copper",     "#C88033", 1.96, 63.546),
    "Zn": ("Zinc",       "#7D80B0", 2.01, 65.380),
    "Br": ("Bromine",    "#A62929", 1.85, 79.904),
    "I":  ("Iodine",     "#940094", 1.98, 126.90),
}

# Covalent radii for bond detection (Angstroms)
COVALENT_RADII = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84,
    "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07,
    "S": 1.05, "Cl": 1.02, "Ar": 1.06, "K": 2.03, "Ca": 1.76,
    "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
    "Br": 1.20, "I": 1.39,
}

BOND_TOLERANCE = 0.45  # extra margin for bond detection


# ---------------------------------------------------------------------------
# Built-in molecule definitions: list of (element, x, y, z)
# ---------------------------------------------------------------------------

def _builtin_water():
    return [
        ("O", 0.000, 0.000, 0.000),
        ("H", 0.757, 0.586, 0.000),
        ("H", -0.757, 0.586, 0.000),
    ]


def _builtin_methane():
    d = 1.09
    t = math.radians(109.47)
    return [
        ("C", 0.0, 0.0, 0.0),
        ("H", 0.0, 0.0, d),
        ("H", d * math.sin(t) * math.cos(0), d * math.sin(t) * math.sin(0), d * math.cos(t)),
        ("H", d * math.sin(t) * math.cos(2 * math.pi / 3), d * math.sin(t) * math.sin(2 * math.pi / 3), d * math.cos(t)),
        ("H", d * math.sin(t) * math.cos(4 * math.pi / 3), d * math.sin(t) * math.sin(4 * math.pi / 3), d * math.cos(t)),
    ]


def _builtin_ethanol():
    return [
        ("C", 0.000, 0.000, 0.000),
        ("C", 1.520, 0.000, 0.000),
        ("O", 2.040, 1.260, 0.000),
        ("H", -0.360, -0.510, 0.890),
        ("H", -0.360, -0.510, -0.890),
        ("H", -0.360, 1.020, 0.000),
        ("H", 1.880, -0.510, 0.890),
        ("H", 1.880, -0.510, -0.890),
        ("H", 2.960, 1.260, 0.000),
    ]


def _builtin_benzene():
    atoms = []
    r_cc = 1.40
    r_ch = 2.48
    for i in range(6):
        angle = math.radians(60 * i)
        atoms.append(("C", r_cc * math.cos(angle), r_cc * math.sin(angle), 0.0))
    for i in range(6):
        angle = math.radians(60 * i)
        atoms.append(("H", r_ch * math.cos(angle), r_ch * math.sin(angle), 0.0))
    return atoms


def _builtin_caffeine():
    return [
        ("C", 0.000, 0.000, 0.000), ("N", 1.350, 0.000, 0.000),
        ("C", 2.100, 1.200, 0.000), ("C", 1.350, 2.400, 0.000),
        ("N", 0.000, 2.400, 0.000), ("C", -0.600, 1.200, 0.000),
        ("O", -1.800, 1.200, 0.000), ("N", 3.450, 1.200, 0.000),
        ("C", 3.900, 2.400, 0.000), ("N", 2.850, 3.200, 0.000),
        ("O", 1.800, 3.600, 0.000), ("C", -0.600, 3.600, 0.000),
        ("C", 1.800, -1.200, 0.000), ("C", 4.200, 0.000, 0.000),
        ("H", 4.950, 2.700, 0.000),
        ("H", -0.200, 4.550, 0.000), ("H", -1.650, 3.450, 0.000),
        ("H", 0.200, 3.850, 0.800),
        ("H", 1.200, -2.050, 0.000), ("H", 2.400, -1.300, 0.890),
        ("H", 2.400, -1.300, -0.890),
        ("H", 5.250, 0.200, 0.000), ("H", 3.800, -0.500, 0.890),
        ("H", 3.800, -0.500, -0.890),
    ]


def _builtin_aspirin():
    return [
        ("C", 0.000, 0.000, 0.000), ("C", 1.400, 0.000, 0.000),
        ("C", 2.100, 1.200, 0.000), ("C", 1.400, 2.400, 0.000),
        ("C", 0.000, 2.400, 0.000), ("C", -0.700, 1.200, 0.000),
        ("O", -0.700, -1.000, 0.000), ("C", -0.200, -2.200, 0.000),
        ("O", 0.800, -2.600, 0.000), ("O", 2.100, -1.200, 0.000),
        ("C", 3.300, -1.200, 0.000), ("O", 3.800, -2.200, 0.000),
        ("C", 3.900, 0.000, 0.000),
        ("H", 3.150, 1.200, 0.000), ("H", 1.850, 3.350, 0.000),
        ("H", -0.450, 3.350, 0.000), ("H", -1.750, 1.200, 0.000),
        ("H", -0.800, -3.050, 0.000), ("H", -0.800, -2.050, 0.890),
        ("H", 4.450, 0.200, 0.890), ("H", 4.450, 0.200, -0.890),
        ("H", 3.300, 0.850, 0.000),
    ]


def _builtin_glucose():
    return [
        ("C", 0.000, 0.000, 0.000), ("C", 1.520, 0.000, 0.000),
        ("C", 2.040, 1.430, 0.000), ("C", 1.520, 2.150, 1.260),
        ("C", 0.000, 2.150, 1.260), ("C", -0.520, 0.720, 1.260),
        ("O", -0.520, 0.720, -0.400), ("O", 2.040, -0.720, -1.100),
        ("O", 3.450, 1.430, 0.000), ("O", 2.040, 3.450, 1.260),
        ("O", -0.520, 2.870, 2.360), ("O", -1.930, 0.720, 1.260),
        ("H", 0.000, -1.020, 0.000), ("H", 1.880, -0.510, 0.890),
        ("H", 1.680, 1.940, -0.890), ("H", 1.880, 1.640, 2.150),
        ("H", -0.360, 2.660, 0.370), ("H", -0.160, 0.210, 2.150),
        ("H", 2.040, -1.660, -0.900), ("H", 3.800, 0.720, 0.600),
        ("H", 1.680, 3.900, 0.500), ("H", -0.160, 3.750, 2.200),
        ("H", -2.300, 1.400, 0.700),
    ]


def _builtin_alanine():
    return [
        ("N", 0.000, 0.000, 0.000),
        ("C", 1.470, 0.000, 0.000),
        ("C", 2.000, 1.420, 0.000),
        ("O", 1.350, 2.300, 0.600),
        ("O", 3.100, 1.650, -0.500),
        ("C", 2.000, -0.750, 1.260),
        ("H", -0.350, -0.480, 0.830),
        ("H", -0.350, -0.480, -0.830),
        ("H", 1.830, -0.510, -0.890),
        ("H", 1.600, -0.240, 2.150),
        ("H", 1.600, -1.760, 1.260),
        ("H", 3.080, -0.750, 1.260),
        ("H", 3.500, 2.500, -0.400),
    ]


def _builtin_buckyball():
    """Generate C60 Buckyball coordinates using a golden-ratio icosahedral approach."""
    phi = (1 + math.sqrt(5)) / 2
    raw = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            raw.append((0, s1, s2 * 3 * phi))
            raw.append((s1, s2 * 3 * phi, 0))
            raw.append((s2 * 3 * phi, 0, s1))
            raw.append((s1 * 2, s2 * (1 + 2 * phi), s2 * phi))
            raw.append((s2 * (1 + 2 * phi), s2 * phi, s1 * 2))
            raw.append((s2 * phi, s1 * 2, s2 * (1 + 2 * phi)))
            for s3 in (-1, 1):
                raw.append((s1 * 1, s2 * (2 + phi), s3 * 2 * phi))
                raw.append((s2 * (2 + phi), s3 * 2 * phi, s1 * 1))
                raw.append((s3 * 2 * phi, s1 * 1, s2 * (2 + phi)))
    # Deduplicate and normalise to radius ~3.55 A
    seen = set()
    unique = []
    for p in raw:
        key = (round(p[0], 4), round(p[1], 4), round(p[2], 4))
        if key not in seen:
            seen.add(key)
            unique.append(np.array(p, dtype=float))
    # Sort by distance from origin and take 60 closest (truncated icosahedron vertices)
    unique.sort(key=lambda v: np.linalg.norm(v))
    verts = unique[:60]
    scale = 3.55 / np.linalg.norm(verts[0]) if len(verts) > 0 else 1.0
    atoms = []
    for v in verts:
        sv = v * scale
        atoms.append(("C", float(sv[0]), float(sv[1]), float(sv[2])))
    return atoms


def _builtin_dna_base_pair():
    """Adenine-Thymine base pair (simplified planar coords)."""
    return [
        # Adenine
        ("N", 0.000, 0.000, 0.000), ("C", 1.300, 0.400, 0.000),
        ("N", 2.200, -0.500, 0.000), ("C", 1.700, -1.700, 0.000),
        ("C", 0.300, -1.400, 0.000), ("N", -0.500, -2.400, 0.000),
        ("C", 0.200, -3.500, 0.000), ("N", 1.500, -3.600, 0.000),
        ("C", 2.100, -2.500, 0.000), ("N", 1.600, 1.700, 0.000),
        ("H", 2.500, 2.100, 0.000), ("H", 0.900, 2.300, 0.000),
        ("H", -0.300, -4.400, 0.000), ("H", 3.150, -2.500, 0.000),
        # Thymine (offset along x)
        ("N", 5.500, 0.000, 0.000), ("C", 6.800, 0.400, 0.000),
        ("O", 7.100, 1.600, 0.000), ("N", 7.700, -0.600, 0.000),
        ("C", 7.300, -1.900, 0.000), ("O", 8.100, -2.800, 0.000),
        ("C", 5.900, -2.200, 0.000), ("C", 5.100, -1.200, 0.000),
        ("C", 5.400, -3.600, 0.000),
        ("H", 8.650, -0.400, 0.000), ("H", 4.050, -1.300, 0.000),
        ("H", 4.700, -3.700, 0.900), ("H", 4.700, -3.700, -0.900),
        ("H", 6.200, -4.300, 0.000),
        # H-bonds (represented as H atoms bridging)
        ("H", 3.400, 0.200, 0.000), ("H", 4.400, -2.000, 0.000),
    ]


BUILTIN_MOLECULES = {
    "Water (H2O)": _builtin_water,
    "Methane (CH4)": _builtin_methane,
    "Ethanol (C2H5OH)": _builtin_ethanol,
    "Benzene (C6H6)": _builtin_benzene,
    "Caffeine (C8H10N4O2)": _builtin_caffeine,
    "Aspirin (C9H8O4)": _builtin_aspirin,
    "Glucose (C6H12O6)": _builtin_glucose,
    "Alanine (C3H7NO2)": _builtin_alanine,
    "Buckyball (C60)": _builtin_buckyball,
    "DNA Base Pair (A-T)": _builtin_dna_base_pair,
}


# ---------------------------------------------------------------------------
# File parsers
# ---------------------------------------------------------------------------

def parse_xyz(path):
    """Parse an XYZ format file and return list of (element, x, y, z)."""
    atoms = []
    with open(path, "r") as fh:
        lines = fh.readlines()
    if len(lines) < 3:
        return atoms
    # First line: atom count; second line: comment; then atom lines
    try:
        n_atoms = int(lines[0].strip())
    except ValueError:
        n_atoms = len(lines) - 2
    for line in lines[2: 2 + n_atoms]:
        parts = line.split()
        if len(parts) >= 4:
            elem = parts[0].capitalize()
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            atoms.append((elem, x, y, z))
    return atoms


def parse_pdb(path):
    """Basic PDB parser extracting ATOM/HETATM records."""
    atoms = []
    with open(path, "r") as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec in ("ATOM", "HETATM"):
                try:
                    elem = line[76:78].strip()
                    if not elem:
                        elem = line[12:16].strip()
                        elem = re.sub(r"[0-9]", "", elem).strip()
                        if len(elem) > 2:
                            elem = elem[:1]
                    elem = elem.capitalize()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    atoms.append((elem, x, y, z))
                except (ValueError, IndexError):
                    continue
    return atoms


# ---------------------------------------------------------------------------
# Bond detection
# ---------------------------------------------------------------------------

def detect_bonds(atoms):
    """Return list of (i, j) index pairs where a bond is likely."""
    bonds = []
    n = len(atoms)
    coords = np.array([(a[1], a[2], a[3]) for a in atoms])
    elems = [a[0] for a in atoms]
    for i in range(n):
        ri = COVALENT_RADII.get(elems[i], 0.77)
        for j in range(i + 1, n):
            rj = COVALENT_RADII.get(elems[j], 0.77)
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < (ri + rj + BOND_TOLERANCE) and dist > 0.4:
                bonds.append((i, j))
    return bonds


# ---------------------------------------------------------------------------
# Molecule info helpers
# ---------------------------------------------------------------------------

def molecular_formula(atoms):
    counts = Counter(a[0] for a in atoms)
    order = ["C", "H"]
    rest = sorted(k for k in counts if k not in order)
    parts = []
    for e in order + rest:
        if e in counts:
            parts.append(f"{e}{counts[e] if counts[e] > 1 else ''}")
    return "".join(parts)


def molecular_weight(atoms):
    return sum(ELEMENT_DATA.get(a[0], ("?", "#888888", 1.5, 0.0))[3] for a in atoms)


# ---------------------------------------------------------------------------
# Geometry optimization (simple spring model)
# ---------------------------------------------------------------------------

# Equilibrium bond lengths (Angstroms) for common pairs
EQUILIBRIUM_BONDS = {
    ("C", "C"): 1.54, ("C", "H"): 1.09, ("C", "N"): 1.47, ("C", "O"): 1.43,
    ("C", "F"): 1.35, ("C", "Cl"): 1.77, ("C", "S"): 1.82, ("N", "H"): 1.01,
    ("O", "H"): 0.96, ("N", "N"): 1.45, ("O", "O"): 1.48, ("S", "H"): 1.34,
    ("C", "Br"): 1.94, ("C", "I"): 2.14, ("N", "O"): 1.40, ("S", "O"): 1.43,
    ("P", "O"): 1.63, ("Si", "O"): 1.63, ("Si", "C"): 1.89,
}
# Equilibrium bond angles (degrees) for center atoms
EQUILIBRIUM_ANGLES = {
    "C": 109.5, "N": 107.0, "O": 104.5, "S": 92.0, "Si": 109.5, "P": 93.0,
}


def _get_eq_bond(e1: str, e2: str) -> float:
    """Get equilibrium bond length for a pair."""
    pair = tuple(sorted([e1, e2]))
    return EQUILIBRIUM_BONDS.get(pair, 1.50)


def geometry_optimize(atoms: List[Tuple[str, float, float, float]],
                      bonds: List[Tuple[int, int]],
                      n_steps: int = 200,
                      dt: float = 0.01,
                      k_bond: float = 50.0,
                      k_angle: float = 5.0,
                      k_repul: float = 2.0
                      ) -> List[Tuple[str, float, float, float]]:
    """Simple force-field optimization using spring model for bonds,
    angle bending terms, and non-bonded repulsion.

    Uses velocity Verlet integration.
    """
    n = len(atoms)
    if n < 2:
        return list(atoms)

    elems = [a[0] for a in atoms]
    pos = np.array([[a[1], a[2], a[3]] for a in atoms], dtype=float)
    vel = np.zeros_like(pos)
    masses = np.array([ELEMENT_DATA.get(e, ("?", "#888", 1.5, 12.0))[3] for e in elems])
    masses = np.maximum(masses, 1.0)

    # Build adjacency
    from collections import defaultdict
    adj = defaultdict(list)
    for i, j in bonds:
        adj[i].append(j)
        adj[j].append(i)

    # Find angles (triplets connected by bonds)
    angle_triplets = []
    for center in range(n):
        nbrs = adj[center]
        for ni in range(len(nbrs)):
            for nj in range(ni + 1, len(nbrs)):
                angle_triplets.append((nbrs[ni], center, nbrs[nj]))

    for step in range(n_steps):
        forces = np.zeros_like(pos)

        # Bond stretching forces
        for i, j in bonds:
            r_vec = pos[j] - pos[i]
            r = np.linalg.norm(r_vec)
            if r < 1e-8:
                continue
            r_eq = _get_eq_bond(elems[i], elems[j])
            f_mag = k_bond * (r - r_eq)
            f_dir = r_vec / r
            forces[i] += f_mag * f_dir
            forces[j] -= f_mag * f_dir

        # Angle bending forces
        for a, center, b in angle_triplets:
            v1 = pos[a] - pos[center]
            v2 = pos[b] - pos[center]
            r1 = np.linalg.norm(v1)
            r2 = np.linalg.norm(v2)
            if r1 < 1e-8 or r2 < 1e-8:
                continue
            cos_theta = np.clip(np.dot(v1, v2) / (r1 * r2), -1.0, 1.0)
            theta = math.acos(cos_theta)
            eq_angle = math.radians(EQUILIBRIUM_ANGLES.get(elems[center], 109.5))
            torque = k_angle * (theta - eq_angle)

            # Apply perpendicular forces to a and b
            n1 = v1 / r1
            n2 = v2 / r2
            perp1 = n2 - cos_theta * n1
            perp2 = n1 - cos_theta * n2
            norm_p1 = np.linalg.norm(perp1)
            norm_p2 = np.linalg.norm(perp2)
            if norm_p1 > 1e-8:
                forces[a] -= torque * perp1 / (norm_p1 * r1)
            if norm_p2 > 1e-8:
                forces[b] -= torque * perp2 / (norm_p2 * r2)

        # Non-bonded repulsion (1/r^2 repulsion for close non-bonded pairs)
        bonded_set = set()
        for i, j in bonds:
            bonded_set.add((min(i, j), max(i, j)))
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in bonded_set:
                    continue
                r_vec = pos[j] - pos[i]
                r = np.linalg.norm(r_vec)
                vdw_sum = (ELEMENT_DATA.get(elems[i], ("?", "#888", 1.5, 12.0))[2] +
                           ELEMENT_DATA.get(elems[j], ("?", "#888", 1.5, 12.0))[2]) * 0.5
                if r < vdw_sum and r > 0.1:
                    f_mag = k_repul * (vdw_sum - r) / r
                    f_dir = r_vec / r
                    forces[i] -= f_mag * f_dir
                    forces[j] += f_mag * f_dir

        # Damped velocity Verlet
        damping = 0.9
        for i in range(n):
            acc = forces[i] / masses[i]
            vel[i] = vel[i] * damping + acc * dt
            pos[i] += vel[i] * dt

    return [(elems[i], float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2]))
            for i in range(n)]


# ---------------------------------------------------------------------------
# Conformer generation
# ---------------------------------------------------------------------------

def generate_conformers(atoms: List[Tuple[str, float, float, float]],
                        bonds: List[Tuple[int, int]],
                        n_conformers: int = 10,
                        angle_step: float = 30.0
                        ) -> List[List[Tuple[str, float, float, float]]]:
    """Generate conformers by rotating around rotatable dihedral bonds.

    Only single bonds between non-terminal heavy atoms are rotated.
    """
    from collections import defaultdict
    adj = defaultdict(list)
    for i, j in bonds:
        adj[i].append(j)
        adj[j].append(i)

    # Find rotatable bonds: bonds between atoms each with >= 2 connections, non-H
    rotatable = []
    for i, j in bonds:
        if atoms[i][0] == "H" or atoms[j][0] == "H":
            continue
        if len(adj[i]) >= 2 and len(adj[j]) >= 2:
            rotatable.append((i, j))

    if not rotatable:
        return [list(atoms)]

    conformers = []
    for conf_idx in range(n_conformers):
        pos = np.array([[a[1], a[2], a[3]] for a in atoms], dtype=float)
        elems = [a[0] for a in atoms]

        for bond_i, bond_j in rotatable:
            angle = random.uniform(-180, 180) if conf_idx > 0 else 0.0
            angle_rad = math.radians(angle)
            if abs(angle_rad) < 1e-6:
                continue

            # Rotation axis
            axis = pos[bond_j] - pos[bond_i]
            axis_len = np.linalg.norm(axis)
            if axis_len < 1e-8:
                continue
            axis = axis / axis_len

            # Find atoms on the j-side via BFS
            visited = {bond_i}
            to_rotate = set()
            queue = [bond_j]
            while queue:
                curr = queue.pop(0)
                if curr in visited:
                    continue
                visited.add(curr)
                to_rotate.add(curr)
                for nbr in adj[curr]:
                    if nbr not in visited:
                        queue.append(nbr)

            # Rodrigues' rotation formula
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            pivot = pos[bond_i]
            for idx in to_rotate:
                v = pos[idx] - pivot
                v_rot = (v * cos_a +
                         np.cross(axis, v) * sin_a +
                         axis * np.dot(axis, v) * (1 - cos_a))
                pos[idx] = pivot + v_rot

        conf_atoms = [(elems[i], float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2]))
                       for i in range(len(atoms))]
        conformers.append(conf_atoms)

    return conformers


# ---------------------------------------------------------------------------
# SMILES parser (basic implementation for simple molecules)
# ---------------------------------------------------------------------------

# Simple SMILES element mapping
_SMILES_ATOMS = {
    'C': 'C', 'N': 'N', 'O': 'O', 'S': 'S', 'P': 'P', 'F': 'F',
    'I': 'I', 'c': 'C', 'n': 'N', 'o': 'O', 's': 'S',
}

_SMILES_BRACKET = re.compile(r'\[([A-Z][a-z]?)')


def smiles_to_atoms(smiles: str) -> List[Tuple[str, float, float, float]]:
    """Parse a simple SMILES string and generate approximate 3D coordinates.

    This is a basic implementation handling:
    - Linear chains (C, N, O, S, F, Cl, Br, I)
    - Branches via parentheses
    - Ring closures via single digits
    - Implicit hydrogens for C, N, O
    - Bracket notation [Fe], [Cu], etc.

    For production use, consider RDKit instead.
    """
    parsed_atoms = []  # list of element symbols
    parsed_bonds = []  # list of (i, j) pairs
    stack = []         # for branch handling
    ring_opens = {}    # digit -> atom index
    current_idx = -1
    i = 0

    while i < len(smiles):
        ch = smiles[i]

        if ch == '(':
            stack.append(current_idx)
            i += 1
            continue
        elif ch == ')':
            if stack:
                current_idx = stack.pop()
            i += 1
            continue
        elif ch == '[':
            # Bracket atom
            m = _SMILES_BRACKET.match(smiles[i:])
            if m:
                elem = m.group(1)
                parsed_atoms.append(elem)
                new_idx = len(parsed_atoms) - 1
                if current_idx >= 0:
                    parsed_bonds.append((current_idx, new_idx))
                current_idx = new_idx
                # Skip to closing bracket
                close = smiles.index(']', i)
                i = close + 1
                continue
            i += 1
            continue
        elif ch in ('=', '#', '/', '\\', '+', '-', '@', '.'):
            # Bond order/stereo markers - skip for 3D generation
            i += 1
            continue
        elif ch.isdigit():
            digit = int(ch)
            if digit in ring_opens:
                parsed_bonds.append((ring_opens[digit], current_idx))
                del ring_opens[digit]
            else:
                ring_opens[digit] = current_idx
            i += 1
            continue
        elif ch in _SMILES_ATOMS:
            elem = _SMILES_ATOMS[ch]
            parsed_atoms.append(elem)
            new_idx = len(parsed_atoms) - 1
            if current_idx >= 0:
                parsed_bonds.append((current_idx, new_idx))
            current_idx = new_idx
            i += 1
            continue
        elif ch == 'l' and i > 0 and smiles[i - 1] == 'C':
            # Cl - fix the last atom
            parsed_atoms[-1] = 'Cl'
            i += 1
            continue
        elif ch == 'r' and i > 0 and smiles[i - 1] == 'B':
            parsed_atoms[-1] = 'Br'
            i += 1
            continue
        elif ch == 'B':
            parsed_atoms.append('B')
            new_idx = len(parsed_atoms) - 1
            if current_idx >= 0:
                parsed_bonds.append((current_idx, new_idx))
            current_idx = new_idx
            i += 1
            continue
        else:
            i += 1
            continue

    # Add implicit hydrogens
    from collections import defaultdict
    adj_count = defaultdict(int)
    for a, b in parsed_bonds:
        adj_count[a] += 1
        adj_count[b] += 1

    valence_map = {"C": 4, "N": 3, "O": 2, "S": 2, "P": 3, "B": 3, "Si": 4}
    h_atoms = []
    h_bonds = []
    for idx, elem in enumerate(parsed_atoms):
        max_val = valence_map.get(elem, 0)
        n_h = max(0, max_val - adj_count[idx])
        for _ in range(n_h):
            h_idx = len(parsed_atoms) + len(h_atoms)
            h_atoms.append("H")
            h_bonds.append((idx, h_idx))

    all_atoms = parsed_atoms + h_atoms
    all_bonds = parsed_bonds + h_bonds

    # Generate 3D coordinates using a simple distance geometry approach
    n = len(all_atoms)
    if n == 0:
        return []

    coords = np.zeros((n, 3))
    placed = np.zeros(n, dtype=bool)

    # Place first atom at origin
    placed[0] = True

    # BFS placement
    queue = [0]
    visited = {0}
    adj = defaultdict(list)
    for a, b in all_bonds:
        adj[a].append(b)
        adj[b].append(a)

    while queue:
        curr = queue.pop(0)
        neighbors = adj[curr]
        n_nbrs = len(neighbors)
        placed_nbrs = [n for n in neighbors if placed[n]]
        unplaced_nbrs = [n for n in neighbors if not placed[n]]

        for k, nbr in enumerate(unplaced_nbrs):
            elem_curr = all_atoms[curr]
            elem_nbr = all_atoms[nbr]
            bond_len = _get_eq_bond(elem_curr, elem_nbr)

            # Find a direction that avoids existing neighbors
            if not placed_nbrs and k == 0:
                direction = np.array([1.0, 0.0, 0.0])
            else:
                # Distribute around the central atom
                angle = (2 * math.pi * (k + len(placed_nbrs))) / max(n_nbrs, 1)
                # Find a local coordinate system
                ref = np.array([0.0, 0.0, 1.0])
                if placed_nbrs:
                    ref = coords[placed_nbrs[0]] - coords[curr]
                    ref_len = np.linalg.norm(ref)
                    if ref_len < 1e-8:
                        ref = np.array([0.0, 0.0, 1.0])
                    else:
                        ref = ref / ref_len

                # Create perpendicular vectors
                up = np.array([0.0, 1.0, 0.0])
                if abs(np.dot(ref, up)) > 0.9:
                    up = np.array([1.0, 0.0, 0.0])
                perp1 = np.cross(ref, up)
                perp1 = perp1 / (np.linalg.norm(perp1) + 1e-12)
                perp2 = np.cross(ref, perp1)

                # Tetrahedral-like placement
                theta = math.radians(109.5)
                direction = (ref * math.cos(theta) +
                             perp1 * math.sin(theta) * math.cos(angle) +
                             perp2 * math.sin(theta) * math.sin(angle))
                d_len = np.linalg.norm(direction)
                if d_len > 1e-8:
                    direction = direction / d_len

            coords[nbr] = coords[curr] + direction * bond_len
            placed[nbr] = True

            if nbr not in visited:
                visited.add(nbr)
                queue.append(nbr)

    return [(all_atoms[i], float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2]))
            for i in range(n)]


# ---------------------------------------------------------------------------
# Molecular surface (van der Waals)
# ---------------------------------------------------------------------------

def compute_vdw_surface(atoms: List[Tuple[str, float, float, float]],
                        n_points: int = 50
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute van der Waals surface points for visualization.

    Returns (xs, ys, zs) arrays of surface points.
    """
    all_x, all_y, all_z = [], [], []
    for elem, cx, cy, cz in atoms:
        radius = ELEMENT_DATA.get(elem, ("?", "#888", 1.50, 0.0))[2]
        # Generate points on a sphere
        for u_idx in range(n_points):
            u = math.pi * u_idx / n_points
            for v_idx in range(n_points):
                v = 2 * math.pi * v_idx / n_points
                x = cx + radius * math.sin(u) * math.cos(v)
                y = cy + radius * math.sin(u) * math.sin(v)
                z = cz + radius * math.cos(u)
                # Check if point is inside any other atom's VdW sphere
                inside_other = False
                for elem2, cx2, cy2, cz2 in atoms:
                    if (cx2, cy2, cz2) == (cx, cy, cz):
                        continue
                    r2 = ELEMENT_DATA.get(elem2, ("?", "#888", 1.50, 0.0))[2]
                    dist = math.sqrt((x - cx2) ** 2 + (y - cy2) ** 2 + (z - cz2) ** 2)
                    if dist < r2 * 0.95:
                        inside_other = True
                        break
                if not inside_other:
                    all_x.append(x)
                    all_y.append(y)
                    all_z.append(z)
    return np.array(all_x), np.array(all_y), np.array(all_z)


# ---------------------------------------------------------------------------
# Export functions: XYZ, PDB, MOL2
# ---------------------------------------------------------------------------

def export_molecule_xyz(atoms: List[Tuple[str, float, float, float]],
                        comment: str = "Generated by QuantumRes") -> str:
    """Export molecule to XYZ format."""
    lines = [str(len(atoms)), comment]
    for elem, x, y, z in atoms:
        lines.append(f"{elem:4s} {x:12.6f} {y:12.6f} {z:12.6f}")
    return "\n".join(lines) + "\n"


def export_molecule_pdb(atoms: List[Tuple[str, float, float, float]],
                        bonds: List[Tuple[int, int]],
                        mol_name: str = "MOL") -> str:
    """Export molecule to PDB format with CONECT records."""
    lines = [f"HEADER    {mol_name}",
             f"REMARK   Generated by QuantumRes"]
    for i, (elem, x, y, z) in enumerate(atoms):
        atom_name = f"{elem}{i+1}"[:4]
        lines.append(
            f"HETATM{i+1:5d} {atom_name:4s} {mol_name:3s} A   1    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}"
        )
    # CONECT records
    from collections import defaultdict
    adj = defaultdict(list)
    for a, b in bonds:
        adj[a].append(b)
        adj[b].append(a)
    for i in sorted(adj.keys()):
        nbrs = " ".join(f"{n+1:5d}" for n in sorted(adj[i]))
        lines.append(f"CONECT{i+1:5d} {nbrs}")
    lines.append("END")
    return "\n".join(lines) + "\n"


def export_molecule_mol2(atoms: List[Tuple[str, float, float, float]],
                         bonds: List[Tuple[int, int]],
                         mol_name: str = "MOL") -> str:
    """Export molecule to Tripos MOL2 format."""
    lines = ["@<TRIPOS>MOLECULE",
             mol_name,
             f" {len(atoms)} {len(bonds)} 0 0 0",
             "SMALL", "GASTEIGER", ""]
    lines.append("@<TRIPOS>ATOM")
    for i, (elem, x, y, z) in enumerate(atoms):
        atom_name = f"{elem}{i+1}"
        atom_type = elem
        lines.append(
            f"  {i+1:5d} {atom_name:8s} {x:10.4f} {y:10.4f} {z:10.4f} "
            f"{atom_type:8s}  1 MOL1       0.0000"
        )
    lines.append("@<TRIPOS>BOND")
    for i, (a, b) in enumerate(bonds):
        lines.append(f"  {i+1:5d} {a+1:5d} {b+1:5d} 1")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# MoleculeViewerWidget
# ---------------------------------------------------------------------------

class MoleculeViewerWidget(QWidget):
    """Interactive 3D molecule viewer widget for PyQt5 applications."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._logger = None
        self._atoms = []      # list of (element, x, y, z)
        self._bonds = []      # list of (i, j)
        self._selected = []   # indices of selected atoms (max 3)
        self._show_labels = True
        self._show_bonds = True
        self._init_ui()

    # -- public API ---------------------------------------------------------

    def set_logger(self, fn):
        """Set an external logging callback ``fn(message: str)``."""
        self._logger = fn

    def load_file(self, path):
        """Load a molecule from an XYZ or PDB file."""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".xyz":
            atoms = parse_xyz(path)
        elif ext == ".pdb":
            atoms = parse_pdb(path)
        else:
            self._log(f"Unsupported file format: {ext}")
            return
        if not atoms:
            self._log("No atoms found in file.")
            return
        self._set_molecule(atoms, source=os.path.basename(path))

    def export(self):
        """Export the current 3D view as a PNG image via a save dialog."""
        if not self._atoms:
            self._log("Nothing to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Molecule Image", "molecule.png",
            "PNG Images (*.png);;All Files (*)"
        )
        if path:
            self._fig.savefig(path, dpi=150, bbox_inches="tight")
            self._log(f"Exported image to {path}")

    # -- UI setup -----------------------------------------------------------

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        # Top toolbar
        toolbar = QHBoxLayout()
        self._combo_builtin = QComboBox()
        self._combo_builtin.addItem("-- Select built-in molecule --")
        for name in BUILTIN_MOLECULES:
            self._combo_builtin.addItem(name)
        self._combo_builtin.currentIndexChanged.connect(self._on_builtin_selected)
        toolbar.addWidget(QLabel("Built-in:"))
        toolbar.addWidget(self._combo_builtin, 1)

        btn_load = QPushButton("Load File...")
        btn_load.clicked.connect(self._on_load_file)
        toolbar.addWidget(btn_load)

        btn_export_img = QPushButton("Export PNG")
        btn_export_img.clicked.connect(self.export)
        toolbar.addWidget(btn_export_img)

        # Export formats dropdown
        self._combo_export = QComboBox()
        self._combo_export.addItems(["Export...", "Export XYZ", "Export PDB", "Export MOL2"])
        self._combo_export.currentIndexChanged.connect(self._on_export_format)
        toolbar.addWidget(self._combo_export)

        self._chk_labels = QCheckBox("Labels")
        self._chk_labels.setChecked(True)
        self._chk_labels.toggled.connect(self._on_toggle_labels)
        toolbar.addWidget(self._chk_labels)

        self._chk_bonds = QCheckBox("Bonds")
        self._chk_bonds.setChecked(True)
        self._chk_bonds.toggled.connect(self._on_toggle_bonds)
        toolbar.addWidget(self._chk_bonds)

        main_layout.addLayout(toolbar)

        # Splitter: 3D canvas | side panel
        splitter = QSplitter(Qt.Horizontal)

        # Matplotlib 3D canvas
        self._fig = Figure(figsize=(5, 5), facecolor="#1e1e2e")
        style_figure(self._fig)
        self._canvas = FigureCanvas(self._fig)
        self._ax = self._fig.add_subplot(111, projection="3d")
        self._ax.set_facecolor("#1e1e2e")
        self._canvas.mpl_connect("button_press_event", self._on_canvas_click)
        splitter.addWidget(self._canvas)

        # Side panel
        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(4, 4, 4, 4)

        # Info group
        info_group = QGroupBox("Molecule Info")
        info_layout = QFormLayout()
        self._lbl_name = QLabel("-")
        self._lbl_formula = QLabel("-")
        self._lbl_weight = QLabel("-")
        self._lbl_atoms = QLabel("-")
        self._lbl_bonds = QLabel("-")
        info_layout.addRow("Name:", self._lbl_name)
        info_layout.addRow("Formula:", self._lbl_formula)
        info_layout.addRow("Mol. Weight:", self._lbl_weight)
        info_layout.addRow("Atoms:", self._lbl_atoms)
        info_layout.addRow("Bonds:", self._lbl_bonds)
        self._lbl_com = QLabel("-")
        self._lbl_com.setWordWrap(True)
        info_layout.addRow("Center of Mass:", self._lbl_com)
        self._lbl_inertia = QLabel("-")
        self._lbl_inertia.setWordWrap(True)
        info_layout.addRow("Inertia:", self._lbl_inertia)
        info_group.setLayout(info_layout)
        side_layout.addWidget(info_group)

        # Measurement group
        meas_group = QGroupBox("Measurements")
        meas_layout = QVBoxLayout()
        self._lbl_meas = QLabel("Click atoms in 3D view to measure.\n"
                                "2 atoms -> distance\n3 atoms -> angle\n4 atoms -> dihedral")
        self._lbl_meas.setWordWrap(True)
        meas_layout.addWidget(self._lbl_meas)
        btn_clear_sel = QPushButton("Clear Selection")
        btn_clear_sel.clicked.connect(self._clear_selection)
        meas_layout.addWidget(btn_clear_sel)
        meas_group.setLayout(meas_layout)
        side_layout.addWidget(meas_group)

        # Zoom slider
        zoom_group = QGroupBox("Zoom")
        zoom_layout = QHBoxLayout()
        self._zoom_slider = QSlider(Qt.Horizontal)
        self._zoom_slider.setRange(10, 200)
        self._zoom_slider.setValue(100)
        self._zoom_slider.valueChanged.connect(self._on_zoom)
        zoom_layout.addWidget(QLabel("Near"))
        zoom_layout.addWidget(self._zoom_slider, 1)
        zoom_layout.addWidget(QLabel("Far"))
        zoom_group.setLayout(zoom_layout)
        side_layout.addWidget(zoom_group)

        # Tabbed side panel for tools
        side_tabs = QTabWidget()

        # --- Coordinates tab ---
        table_tab = QWidget()
        table_layout = QVBoxLayout(table_tab)
        table_layout.setContentsMargins(2, 2, 2, 2)
        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(["#", "Elem", "X", "Y", "Z"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table_layout.addWidget(self._table)
        side_tabs.addTab(table_tab, "Coords")

        # --- Molecule Builder tab ---
        builder_tab = QWidget()
        builder_layout = QVBoxLayout(builder_tab)
        builder_layout.setContentsMargins(4, 4, 4, 4)

        # SMILES input
        smiles_grp = QGroupBox("SMILES Input")
        smiles_lay = QVBoxLayout(smiles_grp)
        self._txt_smiles = QLineEdit()
        self._txt_smiles.setPlaceholderText("e.g. CCO, c1ccccc1, CC(=O)O")
        smiles_lay.addWidget(self._txt_smiles)
        btn_smiles = QPushButton("Parse SMILES")
        btn_smiles.clicked.connect(self._on_parse_smiles)
        smiles_lay.addWidget(btn_smiles)
        builder_layout.addWidget(smiles_grp)

        # Manual atom builder
        manual_grp = QGroupBox("Add Atom")
        manual_lay = QFormLayout(manual_grp)
        self._txt_add_elem = QLineEdit("C")
        self._txt_add_elem.setMaximumWidth(50)
        manual_lay.addRow("Element:", self._txt_add_elem)
        coord_row = QHBoxLayout()
        self._spin_add_x = QDoubleSpinBox()
        self._spin_add_x.setRange(-100, 100); self._spin_add_x.setDecimals(3)
        self._spin_add_y = QDoubleSpinBox()
        self._spin_add_y.setRange(-100, 100); self._spin_add_y.setDecimals(3)
        self._spin_add_z = QDoubleSpinBox()
        self._spin_add_z.setRange(-100, 100); self._spin_add_z.setDecimals(3)
        coord_row.addWidget(QLabel("X")); coord_row.addWidget(self._spin_add_x)
        coord_row.addWidget(QLabel("Y")); coord_row.addWidget(self._spin_add_y)
        coord_row.addWidget(QLabel("Z")); coord_row.addWidget(self._spin_add_z)
        manual_lay.addRow(coord_row)
        btn_add_atom = QPushButton("Add Atom")
        btn_add_atom.clicked.connect(self._on_add_atom)
        manual_lay.addRow(btn_add_atom)
        self._spin_bond_a = QSpinBox(); self._spin_bond_a.setRange(0, 999)
        self._spin_bond_b = QSpinBox(); self._spin_bond_b.setRange(0, 999)
        bond_row = QHBoxLayout()
        bond_row.addWidget(QLabel("Atom A:")); bond_row.addWidget(self._spin_bond_a)
        bond_row.addWidget(QLabel("Atom B:")); bond_row.addWidget(self._spin_bond_b)
        manual_lay.addRow(bond_row)
        btn_add_bond = QPushButton("Add Bond")
        btn_add_bond.clicked.connect(self._on_add_bond)
        manual_lay.addRow(btn_add_bond)
        btn_remove_atom = QPushButton("Remove Last Atom")
        btn_remove_atom.clicked.connect(self._on_remove_last_atom)
        manual_lay.addRow(btn_remove_atom)
        btn_clear_all = QPushButton("Clear Molecule")
        btn_clear_all.clicked.connect(self._on_clear_molecule)
        manual_lay.addRow(btn_clear_all)
        builder_layout.addWidget(manual_grp)
        builder_layout.addStretch()
        side_tabs.addTab(builder_tab, "Builder")

        # --- Optimization & Conformers tab ---
        opt_tab = QWidget()
        opt_layout = QVBoxLayout(opt_tab)
        opt_layout.setContentsMargins(4, 4, 4, 4)

        opt_grp = QGroupBox("Geometry Optimization")
        opt_lay = QFormLayout(opt_grp)
        self._spin_opt_steps = QSpinBox()
        self._spin_opt_steps.setRange(10, 2000); self._spin_opt_steps.setValue(200)
        opt_lay.addRow("Max Steps:", self._spin_opt_steps)
        self._spin_k_bond = QDoubleSpinBox()
        self._spin_k_bond.setRange(1, 200); self._spin_k_bond.setValue(50.0)
        opt_lay.addRow("Bond k:", self._spin_k_bond)
        self._spin_k_angle = QDoubleSpinBox()
        self._spin_k_angle.setRange(0.1, 50); self._spin_k_angle.setValue(5.0)
        opt_lay.addRow("Angle k:", self._spin_k_angle)
        btn_optimize = QPushButton("Optimize Geometry")
        btn_optimize.clicked.connect(self._on_optimize)
        opt_lay.addRow(btn_optimize)
        opt_layout.addWidget(opt_grp)

        conf_grp = QGroupBox("Conformer Generation")
        conf_lay = QFormLayout(conf_grp)
        self._spin_n_conf = QSpinBox()
        self._spin_n_conf.setRange(2, 50); self._spin_n_conf.setValue(5)
        conf_lay.addRow("# Conformers:", self._spin_n_conf)
        btn_gen_conf = QPushButton("Generate Conformers")
        btn_gen_conf.clicked.connect(self._on_generate_conformers)
        conf_lay.addRow(btn_gen_conf)
        self._combo_conformer = QComboBox()
        self._combo_conformer.currentIndexChanged.connect(self._on_select_conformer)
        conf_lay.addRow("Select:", self._combo_conformer)
        opt_layout.addWidget(conf_grp)

        # Surface display
        surf_grp = QGroupBox("Molecular Surface")
        surf_lay = QVBoxLayout(surf_grp)
        self._chk_vdw_surface = QCheckBox("Show VdW Surface")
        self._chk_vdw_surface.setChecked(False)
        self._chk_vdw_surface.toggled.connect(self._draw)
        surf_lay.addWidget(self._chk_vdw_surface)
        opt_layout.addWidget(surf_grp)

        opt_layout.addStretch()
        side_tabs.addTab(opt_tab, "Tools")

        side_layout.addWidget(side_tabs, 1)

        splitter.addWidget(side)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter, 1)

        # Internal state for conformers
        self._conformers: List[List[Tuple[str, float, float, float]]] = []

        # Draw empty scene
        self._draw()

    # -- internal helpers ---------------------------------------------------

    def _log(self, msg):
        if self._logger:
            self._logger(msg)

    def _set_molecule(self, atoms, source=""):
        self._atoms = atoms
        self._bonds = detect_bonds(atoms)
        self._selected = []
        self._update_info(source)
        self._update_table()
        self._draw()
        self._log(f"Loaded molecule: {source or 'built-in'} "
                   f"({len(atoms)} atoms, {len(self._bonds)} bonds)")

    def _update_info(self, name=""):
        if not self._atoms:
            for lbl in (self._lbl_name, self._lbl_formula,
                        self._lbl_weight, self._lbl_atoms, self._lbl_bonds):
                lbl.setText("-")
            self._lbl_com.setText("-")
            self._lbl_inertia.setText("-")
            return
        self._lbl_name.setText(name if name else "Custom")
        self._lbl_formula.setText(molecular_formula(self._atoms))
        self._lbl_weight.setText(f"{molecular_weight(self._atoms):.3f} g/mol")
        self._lbl_atoms.setText(str(len(self._atoms)))
        self._lbl_bonds.setText(str(len(self._bonds)))

        # Center of mass
        com = self._calc_center_of_mass()
        if com is not None:
            self._lbl_com.setText(f"({com[0]:.4f}, {com[1]:.4f}, {com[2]:.4f})")
        else:
            self._lbl_com.setText("-")

        # Moment of inertia (principal moments)
        inertia = self._calc_moment_of_inertia()
        if inertia is not None:
            self._lbl_inertia.setText(
                f"Ia={inertia[0]:.3f}  Ib={inertia[1]:.3f}  Ic={inertia[2]:.3f} amu*A^2"
            )
        else:
            self._lbl_inertia.setText("-")

    def _update_table(self):
        self._table.setRowCount(len(self._atoms))
        for i, (elem, x, y, z) in enumerate(self._atoms):
            self._table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self._table.setItem(i, 1, QTableWidgetItem(elem))
            self._table.setItem(i, 2, QTableWidgetItem(f"{x:.4f}"))
            self._table.setItem(i, 3, QTableWidgetItem(f"{y:.4f}"))
            self._table.setItem(i, 4, QTableWidgetItem(f"{z:.4f}"))

    # -- drawing ------------------------------------------------------------

    def _draw(self):
        self._ax.cla()
        self._ax.set_facecolor("#1e1e2e")
        self._ax.grid(False)
        self._ax.set_xlabel("X (\u00c5)", color="#aaaaaa", fontsize=8)
        self._ax.set_ylabel("Y (\u00c5)", color="#aaaaaa", fontsize=8)
        self._ax.set_zlabel("Z (\u00c5)", color="#aaaaaa", fontsize=8)
        self._ax.tick_params(colors="#888888", labelsize=7)

        if not self._atoms:
            self._ax.set_title("No molecule loaded", color="#cccccc", fontsize=10)
            self._canvas.draw_idle()
            return

        coords = np.array([(a[1], a[2], a[3]) for a in self._atoms])
        elems = [a[0] for a in self._atoms]

        # Draw bonds
        if self._show_bonds:
            for i, j in self._bonds:
                xs = [coords[i][0], coords[j][0]]
                ys = [coords[i][1], coords[j][1]]
                zs = [coords[i][2], coords[j][2]]
                self._ax.plot(xs, ys, zs, color="#999999", linewidth=1.5, alpha=0.7)

        # Draw atoms as scatter points
        for idx in range(len(self._atoms)):
            elem = elems[idx]
            data = ELEMENT_DATA.get(elem, ("Unknown", "#888888", 1.50, 0.0))
            color = data[1]
            radius = data[2]
            size = max(20, radius * 60)
            edge = "yellow" if idx in self._selected else "#333333"
            lw = 2.0 if idx in self._selected else 0.5
            self._ax.scatter(
                coords[idx][0], coords[idx][1], coords[idx][2],
                s=size, c=color, edgecolors=edge, linewidths=lw,
                depthshade=True, alpha=0.95
            )

        # Labels
        if self._show_labels:
            for idx, (elem, x, y, z) in enumerate(self._atoms):
                self._ax.text(x, y, z + 0.25, elem, fontsize=6,
                              color="#dddddd", ha="center", va="bottom")

        # Van der Waals surface
        if hasattr(self, '_chk_vdw_surface') and self._chk_vdw_surface.isChecked():
            try:
                sx, sy, sz = compute_vdw_surface(self._atoms, n_points=20)
                if len(sx) > 0:
                    self._ax.scatter(sx, sy, sz, s=1, c="#4488ff", alpha=0.08,
                                     depthshade=True)
            except Exception:
                pass

        # Equal aspect
        center = coords.mean(axis=0)
        max_range = max((coords.max(axis=0) - coords.min(axis=0)).max() / 2, 1.0)
        zoom = self._zoom_slider.value() / 100.0
        r = max_range * (1.0 / zoom)
        self._ax.set_xlim(center[0] - r, center[0] + r)
        self._ax.set_ylim(center[1] - r, center[1] + r)
        self._ax.set_zlim(center[2] - r, center[2] + r)

        self._canvas.draw_idle()

    # -- callbacks ----------------------------------------------------------

    def _on_builtin_selected(self, index):
        if index <= 0:
            return
        name = self._combo_builtin.currentText()
        builder = BUILTIN_MOLECULES.get(name)
        if builder:
            atoms = builder()
            self._set_molecule(atoms, source=name)

    def _on_load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Molecule File", "",
            "XYZ Files (*.xyz);;PDB Files (*.pdb);;All Files (*)"
        )
        if path:
            self.load_file(path)

    def _on_toggle_labels(self, checked):
        self._show_labels = checked
        self._draw()

    def _on_toggle_bonds(self, checked):
        self._show_bonds = checked
        self._draw()

    def _on_zoom(self, _value):
        self._draw()

    def _on_canvas_click(self, event):
        """Select nearest atom to the click for distance/angle measurement."""
        if event.inaxes is not self._ax or not self._atoms:
            return
        # Use projected 2D coordinates to find nearest atom
        coords = np.array([(a[1], a[2], a[3]) for a in self._atoms])
        # Get 2D projection of all atoms
        from mpl_toolkits.mplot3d import proj3d
        proj_coords = []
        for c in coords:
            x2, y2, _ = proj3d.proj_transform(c[0], c[1], c[2], self._ax.get_proj())
            proj_coords.append((x2, y2))
        proj_coords = np.array(proj_coords)
        click = np.array([event.xdata, event.ydata])
        dists = np.linalg.norm(proj_coords - click, axis=1)
        nearest = int(np.argmin(dists))

        if nearest in self._selected:
            self._selected.remove(nearest)
        else:
            self._selected.append(nearest)
            if len(self._selected) > 4:
                self._selected.pop(0)

        self._update_measurement()
        self._draw()

    def _clear_selection(self):
        self._selected = []
        self._lbl_meas.setText("Click atoms in 3D view to measure.\n"
                               "2 atoms -> distance\n3 atoms -> angle\n4 atoms -> dihedral")
        self._draw()

    def _update_measurement(self):
        if len(self._selected) == 0:
            self._lbl_meas.setText("No atoms selected.")
            return
        sel_text = ", ".join(
            f"{self._atoms[i][0]}#{i+1}" for i in self._selected
        )
        info = f"Selected: {sel_text}\n"
        coords = np.array([(self._atoms[i][1], self._atoms[i][2], self._atoms[i][3])
                           for i in self._selected])

        if len(self._selected) == 2:
            d = np.linalg.norm(coords[0] - coords[1])
            info += f"Distance: {d:.4f} \u00c5"
        elif len(self._selected) == 3:
            v1 = coords[0] - coords[1]
            v2 = coords[2] - coords[1]
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
            cos_a = np.clip(cos_a, -1.0, 1.0)
            angle_deg = math.degrees(math.acos(cos_a))
            info += f"Angle at {self._atoms[self._selected[1]][0]}#{self._selected[1]+1}: "
            info += f"{angle_deg:.2f}\u00b0"
            d01 = np.linalg.norm(coords[0] - coords[1])
            d12 = np.linalg.norm(coords[1] - coords[2])
            info += f"\nDist 1-2: {d01:.4f} \u00c5  |  Dist 2-3: {d12:.4f} \u00c5"
        elif len(self._selected) == 4:
            # Dihedral angle: angle between planes defined by atoms 1-2-3 and 2-3-4
            b1 = coords[1] - coords[0]
            b2 = coords[2] - coords[1]
            b3 = coords[3] - coords[2]
            n1 = np.cross(b1, b2)
            n2 = np.cross(b2, b3)
            n1_norm = np.linalg.norm(n1)
            n2_norm = np.linalg.norm(n2)
            if n1_norm > 1e-12 and n2_norm > 1e-12:
                n1 = n1 / n1_norm
                n2 = n2 / n2_norm
                m1 = np.cross(n1, b2 / np.linalg.norm(b2))
                x_val = np.dot(n1, n2)
                y_val = np.dot(m1, n2)
                dihedral = -math.degrees(math.atan2(y_val, x_val))
                info += f"Dihedral: {dihedral:.2f}\u00b0"
            else:
                info += "Dihedral: undefined (collinear atoms)"
            # Also show individual distances
            for k in range(3):
                dk = np.linalg.norm(coords[k] - coords[k + 1])
                info += f"\nDist {k+1}-{k+2}: {dk:.4f} \u00c5"
        else:
            info += "Select 2 for distance, 3 for angle, 4 for dihedral."

        self._lbl_meas.setText(info)

    # -- Center of Mass & Moment of Inertia ----------------------------------

    def _calc_center_of_mass(self):
        """Calculate the center of mass of the current molecule."""
        if not self._atoms:
            return None
        total_mass = 0.0
        com = np.zeros(3)
        for elem, x, y, z in self._atoms:
            m = ELEMENT_DATA.get(elem, ("", "", 0.0, 1.0))[3]
            com += m * np.array([x, y, z])
            total_mass += m
        if total_mass > 0:
            return com / total_mass
        return None

    def _calc_moment_of_inertia(self):
        """Calculate the principal moments of inertia (eigenvalues of the inertia tensor)."""
        if len(self._atoms) < 2:
            return None
        com = self._calc_center_of_mass()
        if com is None:
            return None

        # Build inertia tensor
        I = np.zeros((3, 3))
        for elem, x, y, z in self._atoms:
            m = ELEMENT_DATA.get(elem, ("", "", 0.0, 1.0))[3]
            r = np.array([x, y, z]) - com
            I[0, 0] += m * (r[1]**2 + r[2]**2)
            I[1, 1] += m * (r[0]**2 + r[2]**2)
            I[2, 2] += m * (r[0]**2 + r[1]**2)
            I[0, 1] -= m * r[0] * r[1]
            I[0, 2] -= m * r[0] * r[2]
            I[1, 2] -= m * r[1] * r[2]
        I[1, 0] = I[0, 1]
        I[2, 0] = I[0, 2]
        I[2, 1] = I[1, 2]

        eigenvalues = np.linalg.eigvalsh(I)
        return np.sort(eigenvalues)

    # -- SMILES parser ------------------------------------------------------

    def _on_parse_smiles(self):
        smiles = self._txt_smiles.text().strip()
        if not smiles:
            self._log("Enter a SMILES string.")
            return
        try:
            atoms = smiles_to_atoms(smiles)
            if not atoms:
                self._log("Could not parse SMILES string.")
                return
            self._set_molecule(atoms, source=f"SMILES: {smiles}")
            self._log(f"Parsed SMILES '{smiles}': {len(atoms)} atoms")
        except Exception as exc:
            self._log(f"SMILES parse error: {exc}")

    # -- Molecule builder ---------------------------------------------------

    def _on_add_atom(self):
        elem = self._txt_add_elem.text().strip().capitalize()
        if not elem:
            return
        x = self._spin_add_x.value()
        y = self._spin_add_y.value()
        z = self._spin_add_z.value()
        self._atoms.append((elem, x, y, z))
        self._bonds = detect_bonds(self._atoms)
        self._update_info("Custom")
        self._update_table()
        self._draw()
        self._log(f"Added {elem} at ({x:.3f}, {y:.3f}, {z:.3f}), total: {len(self._atoms)}")

    def _on_add_bond(self):
        a = self._spin_bond_a.value()
        b = self._spin_bond_b.value()
        if a == b or a >= len(self._atoms) or b >= len(self._atoms):
            self._log("Invalid atom indices for bond.")
            return
        pair = (min(a, b), max(a, b))
        if pair not in self._bonds:
            self._bonds.append(pair)
            self._draw()
            self._log(f"Added bond {a}-{b}")

    def _on_remove_last_atom(self):
        if not self._atoms:
            return
        removed = self._atoms.pop()
        self._bonds = detect_bonds(self._atoms)
        self._selected = [s for s in self._selected if s < len(self._atoms)]
        self._update_info("Custom")
        self._update_table()
        self._draw()
        self._log(f"Removed {removed[0]}, remaining: {len(self._atoms)}")

    def _on_clear_molecule(self):
        self._atoms = []
        self._bonds = []
        self._selected = []
        self._conformers = []
        self._update_info()
        self._update_table()
        self._draw()
        self._log("Molecule cleared")

    # -- Geometry optimization ----------------------------------------------

    def _on_optimize(self):
        if not self._atoms or len(self._atoms) < 2:
            self._log("Need at least 2 atoms to optimize.")
            return
        n_steps = self._spin_opt_steps.value()
        k_bond = self._spin_k_bond.value()
        k_angle = self._spin_k_angle.value()
        self._log(f"Optimizing geometry ({n_steps} steps, k_bond={k_bond}, k_angle={k_angle})...")
        try:
            optimized = geometry_optimize(self._atoms, self._bonds,
                                          n_steps=n_steps, k_bond=k_bond, k_angle=k_angle)
            self._atoms = optimized
            self._bonds = detect_bonds(self._atoms)
            self._update_info("Optimized")
            self._update_table()
            self._draw()
            self._log("Geometry optimization complete")
        except Exception as exc:
            self._log(f"Optimization error: {exc}")

    # -- Conformer generation -----------------------------------------------

    def _on_generate_conformers(self):
        if not self._atoms or len(self._atoms) < 2:
            self._log("Need at least 2 atoms.")
            return
        n = self._spin_n_conf.value()
        self._log(f"Generating {n} conformers...")
        try:
            self._conformers = generate_conformers(self._atoms, self._bonds,
                                                    n_conformers=n)
            self._combo_conformer.blockSignals(True)
            self._combo_conformer.clear()
            for i in range(len(self._conformers)):
                self._combo_conformer.addItem(f"Conformer {i + 1}")
            self._combo_conformer.blockSignals(False)
            self._log(f"Generated {len(self._conformers)} conformers")
        except Exception as exc:
            self._log(f"Conformer generation error: {exc}")

    def _on_select_conformer(self, index):
        if 0 <= index < len(self._conformers):
            self._atoms = list(self._conformers[index])
            self._bonds = detect_bonds(self._atoms)
            self._update_info(f"Conformer {index + 1}")
            self._update_table()
            self._draw()

    # -- Export formats -----------------------------------------------------

    def _on_export_format(self, index):
        if index <= 0 or not self._atoms:
            return
        self._combo_export.blockSignals(True)
        self._combo_export.setCurrentIndex(0)
        self._combo_export.blockSignals(False)

        fmt = ["", "xyz", "pdb", "mol2"][index]
        filters = {
            "xyz": "XYZ Files (*.xyz);;All Files (*)",
            "pdb": "PDB Files (*.pdb);;All Files (*)",
            "mol2": "MOL2 Files (*.mol2);;All Files (*)",
        }
        path, _ = QFileDialog.getSaveFileName(
            self, f"Export {fmt.upper()}", f"molecule.{fmt}", filters[fmt])
        if not path:
            return

        try:
            if fmt == "xyz":
                content = export_molecule_xyz(self._atoms)
            elif fmt == "pdb":
                content = export_molecule_pdb(self._atoms, self._bonds)
            elif fmt == "mol2":
                content = export_molecule_mol2(self._atoms, self._bonds)
            else:
                return
            with open(path, "w") as fh:
                fh.write(content)
            self._log(f"Exported to {path}")
        except Exception as exc:
            self._log(f"Export error: {exc}")
