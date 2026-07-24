"""
Dimensional Analysis Engine for QuantumRes.
Provides dimensional consistency checking, Buckingham Pi theorem,
unit conversion, dimensionless numbers reference, and expression parsing.
"""

import re
import numpy as np
from functools import reduce
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QLabel, QLineEdit, QComboBox, QListWidget, QListWidgetItem,
    QGroupBox, QSplitter, QFrame, QMessageBox, QApplication,
    QTextEdit, QScrollArea, QTableWidget, QTableWidgetItem,
    QHeaderView, QTabWidget, QPlainTextEdit, QFormLayout
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor

# ── Fundamental Dimensions ────────────────────────────────────────────────────
# Order: M, L, T, A, K, mol, cd  (SI base quantities)

DIM_LABELS = ["M", "L", "T", "A", "K", "mol", "cd"]
DIM_COUNT = len(DIM_LABELS)

def _d(*args):
    """Create a dimension vector from exponents in order M, L, T, A, K, mol, cd."""
    v = [0] * DIM_COUNT
    for i, a in enumerate(args):
        v[i] = a
    return tuple(v)

DIMENSIONLESS = _d()

# ── Dimension Database (50+ quantities) ──────────────────────────────────────

DIMENSION_DB = {
    # Mechanics
    "Mass":                  {"dim": _d(1),            "si": "kg"},
    "Length":                {"dim": _d(0, 1),         "si": "m"},
    "Time":                  {"dim": _d(0, 0, 1),      "si": "s"},
    "Velocity":              {"dim": _d(0, 1, -1),     "si": "m/s"},
    "Acceleration":          {"dim": _d(0, 1, -2),     "si": "m/s^2"},
    "Force":                 {"dim": _d(1, 1, -2),     "si": "N = kg*m/s^2"},
    "Energy":                {"dim": _d(1, 2, -2),     "si": "J = kg*m^2/s^2"},
    "Power":                 {"dim": _d(1, 2, -3),     "si": "W = kg*m^2/s^3"},
    "Pressure":              {"dim": _d(1, -1, -2),    "si": "Pa = kg/(m*s^2)"},
    "Momentum":              {"dim": _d(1, 1, -1),     "si": "kg*m/s"},
    "Angular momentum":      {"dim": _d(1, 2, -1),     "si": "kg*m^2/s"},
    "Torque":                {"dim": _d(1, 2, -2),     "si": "N*m"},
    "Frequency":             {"dim": _d(0, 0, -1),     "si": "Hz = 1/s"},
    "Angular velocity":      {"dim": _d(0, 0, -1),     "si": "rad/s"},
    "Angular acceleration":  {"dim": _d(0, 0, -2),     "si": "rad/s^2"},
    "Moment of inertia":     {"dim": _d(1, 2),         "si": "kg*m^2"},
    "Density":               {"dim": _d(1, -3),        "si": "kg/m^3"},
    "Specific volume":       {"dim": _d(-1, 3),        "si": "m^3/kg"},
    "Area":                  {"dim": _d(0, 2),         "si": "m^2"},
    "Volume":                {"dim": _d(0, 3),         "si": "m^3"},
    "Surface tension":       {"dim": _d(1, 0, -2),     "si": "N/m"},
    "Strain":                {"dim": DIMENSIONLESS,     "si": "dimensionless"},
    "Stress":                {"dim": _d(1, -1, -2),    "si": "Pa"},
    "Elastic modulus":       {"dim": _d(1, -1, -2),    "si": "Pa"},
    # Thermodynamics
    "Temperature":           {"dim": _d(0, 0, 0, 0, 1), "si": "K"},
    "Entropy":               {"dim": _d(1, 2, -2, 0, -1), "si": "J/K"},
    "Specific heat":         {"dim": _d(0, 2, -2, 0, -1), "si": "J/(kg*K)"},
    "Thermal conductivity":  {"dim": _d(1, 1, -3, 0, -1), "si": "W/(m*K)"},
    "Heat flux":             {"dim": _d(1, 0, -3),     "si": "W/m^2"},
    "Thermal diffusivity":   {"dim": _d(0, 2, -1),     "si": "m^2/s"},
    # Fluid dynamics
    "Dynamic viscosity":     {"dim": _d(1, -1, -1),    "si": "Pa*s"},
    "Kinematic viscosity":   {"dim": _d(0, 2, -1),     "si": "m^2/s"},
    "Diffusivity":           {"dim": _d(0, 2, -1),     "si": "m^2/s"},
    "Volume flow rate":      {"dim": _d(0, 3, -1),     "si": "m^3/s"},
    "Mass flow rate":        {"dim": _d(1, 0, -1),     "si": "kg/s"},
    # Electromagnetism
    "Electric current":      {"dim": _d(0, 0, 0, 1),   "si": "A"},
    "Electric charge":       {"dim": _d(0, 0, 1, 1),   "si": "C = A*s"},
    "Voltage":               {"dim": _d(1, 2, -3, -1), "si": "V = kg*m^2/(A*s^3)"},
    "Resistance":            {"dim": _d(1, 2, -3, -2), "si": "ohm"},
    "Conductance":           {"dim": _d(-1, -2, 3, 2), "si": "S = A^2*s^3/(kg*m^2)"},
    "Capacitance":           {"dim": _d(-1, -2, 4, 2), "si": "F = A^2*s^4/(kg*m^2)"},
    "Inductance":            {"dim": _d(1, 2, -2, -2), "si": "H = kg*m^2/(A^2*s^2)"},
    "Magnetic field":        {"dim": _d(1, 0, -2, -1), "si": "T = kg/(A*s^2)"},
    "Magnetic flux":         {"dim": _d(1, 2, -2, -1), "si": "Wb = kg*m^2/(A*s^2)"},
    "Electric field":        {"dim": _d(1, 1, -3, -1), "si": "V/m"},
    "Permittivity":          {"dim": _d(-1, -3, 4, 2), "si": "F/m"},
    "Permeability":          {"dim": _d(1, 1, -2, -2), "si": "H/m"},
    "Electric displacement": {"dim": _d(0, -2, 1, 1),  "si": "C/m^2"},
    "Magnetization":         {"dim": _d(0, -1, 0, 1),  "si": "A/m"},
    # Optics / Radiation
    "Luminous intensity":    {"dim": _d(0, 0, 0, 0, 0, 0, 1), "si": "cd"},
    "Luminous flux":         {"dim": _d(0, 0, 0, 0, 0, 0, 1), "si": "lm (cd*sr)"},
    "Illuminance":           {"dim": _d(0, -2, 0, 0, 0, 0, 1), "si": "lx = lm/m^2"},
    "Radiant intensity":     {"dim": _d(1, 2, -3),     "si": "W/sr"},
    # Chemical
    "Amount of substance":   {"dim": _d(0, 0, 0, 0, 0, 1), "si": "mol"},
    "Molar mass":            {"dim": _d(1, 0, 0, 0, 0, -1), "si": "kg/mol"},
    "Concentration":         {"dim": _d(0, -3, 0, 0, 0, 1), "si": "mol/m^3"},
    "Catalytic activity":    {"dim": _d(0, 0, -1, 0, 0, 1), "si": "kat = mol/s"},
    # Misc
    "Wavenumber":            {"dim": _d(0, -1),        "si": "1/m"},
    "Specific energy":       {"dim": _d(0, 2, -2),     "si": "J/kg"},
    "Energy density":        {"dim": _d(1, -1, -2),    "si": "J/m^3"},
    "Action":                {"dim": _d(1, 2, -1),     "si": "J*s"},
    "Gravitational constant": {"dim": _d(-1, 3, -2),   "si": "m^3/(kg*s^2)"},
}

# ── Dimensionless Numbers ────────────────────────────────────────────────────

DIMENSIONLESS_NUMBERS = {
    "Reynolds (Re)": {
        "formula": "Re = rho * v * L / mu",
        "meaning": "Ratio of inertial to viscous forces. Determines flow regime (laminar vs turbulent).",
        "variables": "rho=density, v=velocity, L=characteristic length, mu=dynamic viscosity",
    },
    "Mach (Ma)": {
        "formula": "Ma = v / c",
        "meaning": "Ratio of flow velocity to speed of sound. Classifies compressibility regime.",
        "variables": "v=flow velocity, c=speed of sound",
    },
    "Froude (Fr)": {
        "formula": "Fr = v / sqrt(g * L)",
        "meaning": "Ratio of flow inertia to gravity. Important for free-surface flows.",
        "variables": "v=velocity, g=gravitational acceleration, L=characteristic length",
    },
    "Prandtl (Pr)": {
        "formula": "Pr = mu * cp / k",
        "meaning": "Ratio of momentum diffusivity to thermal diffusivity.",
        "variables": "mu=dynamic viscosity, cp=specific heat, k=thermal conductivity",
    },
    "Nusselt (Nu)": {
        "formula": "Nu = h * L / k",
        "meaning": "Ratio of convective to conductive heat transfer at a boundary.",
        "variables": "h=heat transfer coefficient, L=characteristic length, k=thermal conductivity",
    },
    "Grashof (Gr)": {
        "formula": "Gr = g * beta * dT * L^3 / nu^2",
        "meaning": "Ratio of buoyancy to viscous forces in natural convection.",
        "variables": "g=gravity, beta=thermal expansion coeff, dT=temp difference, L=length, nu=kinematic viscosity",
    },
    "Rayleigh (Ra)": {
        "formula": "Ra = Gr * Pr = g * beta * dT * L^3 / (nu * alpha)",
        "meaning": "Product of Grashof and Prandtl. Determines onset of natural convection.",
        "variables": "alpha=thermal diffusivity, nu=kinematic viscosity, etc.",
    },
    "Weber (We)": {
        "formula": "We = rho * v^2 * L / sigma",
        "meaning": "Ratio of inertia to surface tension. Important for droplets and bubbles.",
        "variables": "rho=density, v=velocity, L=length, sigma=surface tension",
    },
    "Peclet (Pe)": {
        "formula": "Pe = v * L / alpha  (thermal) or v * L / D (mass)",
        "meaning": "Ratio of advective to diffusive transport.",
        "variables": "v=velocity, L=length, alpha=thermal diffusivity, D=mass diffusivity",
    },
    "Schmidt (Sc)": {
        "formula": "Sc = nu / D",
        "meaning": "Ratio of momentum diffusivity to mass diffusivity.",
        "variables": "nu=kinematic viscosity, D=mass diffusivity",
    },
    "Sherwood (Sh)": {
        "formula": "Sh = k_m * L / D",
        "meaning": "Ratio of convective to diffusive mass transfer. Mass transfer analogue of Nusselt.",
        "variables": "k_m=mass transfer coefficient, L=length, D=diffusivity",
    },
    "Lewis (Le)": {
        "formula": "Le = alpha / D = Sc / Pr",
        "meaning": "Ratio of thermal to mass diffusivity.",
        "variables": "alpha=thermal diffusivity, D=mass diffusivity",
    },
    "Biot (Bi)": {
        "formula": "Bi = h * L / k_s",
        "meaning": "Ratio of surface convection to body conduction. Determines if lumped capacitance is valid.",
        "variables": "h=heat transfer coefficient, L=characteristic length, k_s=solid thermal conductivity",
    },
    "Fourier (Fo)": {
        "formula": "Fo = alpha * t / L^2",
        "meaning": "Dimensionless time for transient heat conduction.",
        "variables": "alpha=thermal diffusivity, t=time, L=characteristic length",
    },
    "Euler (Eu)": {
        "formula": "Eu = dP / (rho * v^2)",
        "meaning": "Ratio of pressure forces to inertial forces.",
        "variables": "dP=pressure difference, rho=density, v=velocity",
    },
    "Strouhal (St)": {
        "formula": "St = f * L / v",
        "meaning": "Ratio of oscillatory inertia to convective inertia. Describes vortex shedding.",
        "variables": "f=frequency, L=characteristic length, v=velocity",
    },
}

# ── Preset Equations ─────────────────────────────────────────────────────────

PRESET_EQUATIONS = {
    "Newton's 2nd law":        "F = m * a",
    "Ideal gas law":           "P * V = n * R * T",
    "Kinetic energy":          "E = 0.5 * m * v**2",
    "Gravitational force":     "F = G * m1 * m2 / r**2",
    "Bernoulli equation":      "P + 0.5 * rho * v**2 + rho * g * h = const",
    "Navier-Stokes (energy)":  "rho * cp * dT_dt = k * nabla2_T + mu * Phi",
    "Wave equation":           "v = f * wavelength",
    "Coulomb's law":           "F = k_e * q1 * q2 / r**2",
    "Ohm's law":               "V = I * R_ohm",
    "Power (electrical)":      "P_elec = V * I",
    "Schrodinger (energy)":    "E = hbar * omega",
    "Maxwell (Faraday)":       "EMF = -dPhi_B_dt",
}

PRESET_VARIABLES = {
    "F": _d(1, 1, -2),      "m": _d(1),             "a": _d(0, 1, -2),
    "P": _d(1, -1, -2),     "V": _d(0, 3),          "n": _d(0, 0, 0, 0, 0, 1),
    "R": _d(1, 2, -2, 0, -1, -1),  "T": _d(0, 0, 0, 0, 1),
    "E": _d(1, 2, -2),      "v": _d(0, 1, -1),      "G": _d(-1, 3, -2),
    "m1": _d(1),             "m2": _d(1),             "r": _d(0, 1),
    "rho": _d(1, -3),        "g": _d(0, 1, -2),      "h": _d(0, 1),
    "cp": _d(0, 2, -2, 0, -1), "dT_dt": _d(0, 0, -1, 0, 1),
    "k": _d(1, 1, -3, 0, -1), "nabla2_T": _d(0, -2, 0, 0, 1),
    "mu": _d(1, -1, -1),     "Phi": _d(0, 0, -2),
    "f": _d(0, 0, -1),       "wavelength": _d(0, 1),
    "k_e": _d(1, 3, -4, -2), "q1": _d(0, 0, 1, 1),  "q2": _d(0, 0, 1, 1),
    "I": _d(0, 0, 0, 1),     "R_ohm": _d(1, 2, -3, -2),
    "P_elec": _d(1, 2, -3),  "hbar": _d(1, 2, -1),   "omega": _d(0, 0, -1),
    "EMF": _d(1, 2, -3, -1), "dPhi_B_dt": _d(1, 2, -3, -1),
    "const": _d(1, -1, -2),  # Bernoulli constant has pressure dimensions
}

# ── Unit Conversion Data (within dimensions) ─────────────────────────────────

UNIT_CONVERSIONS = {
    "Force": {
        "base": "N",
        "units": {
            "N": 1.0, "dyn": 1e-5, "kgf": 9.80665, "lbf": 4.44822,
            "poundal": 0.138255, "kN": 1e3, "MN": 1e6,
        },
    },
    "Energy": {
        "base": "J",
        "units": {
            "J": 1.0, "erg": 1e-7, "cal": 4.184, "kcal": 4184.0,
            "kWh": 3.6e6, "eV": 1.602176634e-19, "BTU": 1055.06,
            "ft*lbf": 1.35582, "kJ": 1e3, "MJ": 1e6,
        },
    },
    "Pressure": {
        "base": "Pa",
        "units": {
            "Pa": 1.0, "kPa": 1e3, "MPa": 1e6, "bar": 1e5, "atm": 101325.0,
            "torr": 133.322, "mmHg": 133.322, "psi": 6894.76,
            "inHg": 3386.39, "mbar": 100.0,
        },
    },
    "Power": {
        "base": "W",
        "units": {
            "W": 1.0, "kW": 1e3, "MW": 1e6, "hp": 745.7,
            "BTU/h": 0.293071, "erg/s": 1e-7, "ft*lbf/s": 1.35582,
        },
    },
    "Velocity": {
        "base": "m/s",
        "units": {
            "m/s": 1.0, "km/h": 1.0 / 3.6, "mph": 0.44704,
            "ft/s": 0.3048, "knot": 0.514444, "cm/s": 0.01,
            "c": 299792458.0,
        },
    },
    "Length": {
        "base": "m",
        "units": {
            "m": 1.0, "km": 1e3, "cm": 1e-2, "mm": 1e-3,
            "um": 1e-6, "nm": 1e-9, "in": 0.0254, "ft": 0.3048,
            "yd": 0.9144, "mi": 1609.344, "AU": 1.496e11,
        },
    },
    "Mass": {
        "base": "kg",
        "units": {
            "kg": 1.0, "g": 1e-3, "mg": 1e-6, "lb": 0.453592,
            "oz": 0.0283495, "tonne": 1e3, "slug": 14.5939,
            "grain": 6.47989e-5, "amu": 1.66054e-27,
        },
    },
    "Temperature": {
        "base": "K",
        "units": {"K": 1.0, "degC": 1.0, "degF": 5.0 / 9.0, "degR": 5.0 / 9.0},
    },
}


# ── Dimension Utilities ──────────────────────────────────────────────────────

def dim_to_str(dim, style="bracket"):
    """Convert dimension tuple to string representation."""
    if all(d == 0 for d in dim):
        return "dimensionless" if style == "bracket" else "1"
    parts = []
    for i, label in enumerate(DIM_LABELS):
        if i < len(dim) and dim[i] != 0:
            exp = dim[i]
            if exp == 1:
                parts.append(label)
            elif int(exp) == exp:
                parts.append(f"{label}^{int(exp)}")
            else:
                parts.append(f"{label}^{exp}")
    if style == "bracket":
        return "[" + " ".join(parts) + "]"
    return " ".join(parts)


def parse_dim_str(s):
    """Parse a dimension string like '[M L T^-2]' or 'MLT^-2' into a tuple."""
    s = s.strip().strip("[]")
    vec = [0] * DIM_COUNT
    pattern = re.compile(r'([A-Za-z]+)(?:\^([+-]?\d+(?:\.\d+)?))?')
    for match in pattern.finditer(s):
        label = match.group(1)
        exp = float(match.group(2)) if match.group(2) else 1.0
        if int(exp) == exp:
            exp = int(exp)
        if label in DIM_LABELS:
            idx = DIM_LABELS.index(label)
            vec[idx] = exp
        elif label == "mol":
            vec[5] = exp
        elif label == "cd":
            vec[6] = exp
    return tuple(vec)


def dim_multiply(a, b):
    """Multiply dimensions (add exponents)."""
    return tuple(a[i] + b[i] for i in range(DIM_COUNT))


def dim_divide(a, b):
    """Divide dimensions (subtract exponents)."""
    return tuple(a[i] - b[i] for i in range(DIM_COUNT))


def dim_power(a, n):
    """Raise dimension to a power."""
    return tuple(a[i] * n for i in range(DIM_COUNT))


def dim_equal(a, b):
    """Check if two dimensions are equal."""
    for i in range(DIM_COUNT):
        ai = a[i] if i < len(a) else 0
        bi = b[i] if i < len(b) else 0
        if abs(ai - bi) > 1e-12:
            return False
    return True


# ── Expression Parser ────────────────────────────────────────────────────────

class DimExprParser:
    """Parse mathematical expressions and compute their dimensions."""

    def __init__(self, variables=None):
        self.variables = dict(variables) if variables else {}

    def set_variable(self, name, dim):
        self.variables[name] = dim

    def parse(self, expr):
        """Parse an expression and return its dimension tuple.
        Raises ValueError on dimensional inconsistency.
        """
        expr = expr.strip()
        if not expr:
            raise ValueError("Empty expression")
        return self._parse_add_sub(expr)

    def _tokenize(self, expr):
        """Tokenize an expression into numbers, identifiers, and operators."""
        tokens = []
        i = 0
        while i < len(expr):
            c = expr[i]
            if c in ' \t':
                i += 1
                continue
            if c in '+-':
                # Distinguish unary minus from binary minus
                if tokens and (tokens[-1][0] in ('NUM', 'ID', ')')):
                    tokens.append(('OP', c))
                else:
                    # Unary plus/minus: absorb into number or treat as sign
                    j = i + 1
                    while j < len(expr) and expr[j] in ' \t':
                        j += 1
                    tokens.append(('UNARY', c))
                i += 1
            elif c in '*/':
                tokens.append(('OP', c))
                i += 1
            elif c == '*' and i + 1 < len(expr) and expr[i + 1] == '*':
                tokens.append(('OP', '**'))
                i += 2
            elif c == '(':
                tokens.append(('(', '('))
                i += 1
            elif c == ')':
                tokens.append((')', ')'))
                i += 1
            elif c.isdigit() or c == '.':
                j = i
                while j < len(expr) and (expr[j].isdigit() or expr[j] == '.' or expr[j] in 'eE'
                                          or (expr[j] in '+-' and j > i and expr[j-1] in 'eE')):
                    j += 1
                tokens.append(('NUM', expr[i:j]))
                i = j
            elif c.isalpha() or c == '_':
                j = i
                while j < len(expr) and (expr[j].isalnum() or expr[j] == '_'):
                    j += 1
                tokens.append(('ID', expr[i:j]))
                i = j
            else:
                i += 1
        return tokens

    def _parse_add_sub(self, expr):
        """Handle addition and subtraction: operands must have same dimensions."""
        # Split at top-level + and - (not inside parentheses)
        parts = self._split_top_level(expr, ['+', '-'])
        if len(parts) == 1:
            return self._parse_mul_div(parts[0].strip())

        dims = []
        for part in parts:
            part = part.strip()
            if part:
                dims.append(self._parse_mul_div(part))

        if not dims:
            return DIMENSIONLESS

        base = dims[0]
        for i, d in enumerate(dims[1:], 1):
            if not dim_equal(base, d):
                raise ValueError(
                    f"Dimensional mismatch in addition/subtraction: "
                    f"{dim_to_str(base)} vs {dim_to_str(d)}"
                )
        return base

    def _parse_mul_div(self, expr):
        """Handle multiplication and division."""
        parts_ops = self._split_top_level_with_ops(expr, ['*', '/'])
        if len(parts_ops) == 1:
            return self._parse_power(parts_ops[0][1].strip())

        result = self._parse_power(parts_ops[0][1].strip())
        for op, part in parts_ops[1:]:
            part = part.strip()
            if not part:
                continue
            # Handle ** which got split on first *
            if part.startswith('*'):
                # This was **, rejoin
                part = part[1:]
                result = dim_power(result, self._parse_number(part))
                continue
            d = self._parse_power(part)
            if op == '*':
                result = dim_multiply(result, d)
            elif op == '/':
                result = dim_divide(result, d)
        return result

    def _parse_power(self, expr):
        """Handle exponentiation."""
        if '**' in expr:
            # Find the last ** at top level
            idx = self._find_top_level(expr, '**')
            if idx >= 0:
                base_expr = expr[:idx].strip()
                exp_expr = expr[idx + 2:].strip()
                base_dim = self._parse_atom(base_expr)
                exp_val = self._parse_number(exp_expr)
                return dim_power(base_dim, exp_val)
        return self._parse_atom(expr)

    def _parse_atom(self, expr):
        """Parse an atomic expression: number, variable, or parenthesized."""
        expr = expr.strip()
        if not expr:
            return DIMENSIONLESS

        # Remove unary minus/plus
        while expr and expr[0] in '+-':
            expr = expr[1:].strip()

        # Parenthesized expression
        if expr.startswith('(') and self._find_matching_paren(expr, 0) == len(expr) - 1:
            return self.parse(expr[1:-1])

        # Number
        try:
            float(expr)
            return DIMENSIONLESS
        except ValueError:
            pass

        # Variable
        if expr in self.variables:
            return self.variables[expr]

        # Check dimension database by name
        for name, info in DIMENSION_DB.items():
            if expr.lower() == name.lower().replace(" ", "_"):
                return info["dim"]

        raise ValueError(f"Unknown variable: '{expr}'")

    def _parse_number(self, expr):
        """Parse a numeric value from expression."""
        expr = expr.strip()
        try:
            return float(expr)
        except ValueError:
            raise ValueError(f"Expected number, got '{expr}'")

    def _split_top_level(self, expr, ops):
        """Split expression at top-level operators (outside parentheses)."""
        parts = []
        depth = 0
        current = []
        i = 0
        while i < len(expr):
            c = expr[i]
            if c == '(':
                depth += 1
                current.append(c)
                i += 1
            elif c == ')':
                depth -= 1
                current.append(c)
                i += 1
            elif depth == 0 and c in ops:
                # Check for ** vs *
                if c == '*' and i + 1 < len(expr) and expr[i + 1] == '*':
                    current.append('*')
                    current.append('*')
                    i += 2
                    continue
                parts.append(''.join(current))
                current = []
                i += 1
            else:
                current.append(c)
                i += 1
        parts.append(''.join(current))
        return parts

    def _split_top_level_with_ops(self, expr, ops):
        """Split expression at top-level operators, returning (op, part) pairs."""
        parts = []
        depth = 0
        current = []
        current_op = None
        i = 0
        while i < len(expr):
            c = expr[i]
            if c == '(':
                depth += 1
                current.append(c)
                i += 1
            elif c == ')':
                depth -= 1
                current.append(c)
                i += 1
            elif depth == 0 and c in ops:
                if c == '*' and i + 1 < len(expr) and expr[i + 1] == '*':
                    current.append('*')
                    current.append('*')
                    i += 2
                    continue
                parts.append((current_op, ''.join(current)))
                current = []
                current_op = c
                i += 1
            else:
                current.append(c)
                i += 1
        parts.append((current_op, ''.join(current)))
        return parts

    def _find_top_level(self, expr, op):
        """Find index of a top-level operator string."""
        depth = 0
        for i in range(len(expr) - len(op) + 1):
            if expr[i] == '(':
                depth += 1
            elif expr[i] == ')':
                depth -= 1
            elif depth == 0 and expr[i:i + len(op)] == op:
                # Make sure ** is not confused with *
                if op == '**':
                    return i
                if op == '*' and i + 1 < len(expr) and expr[i + 1] == '*':
                    continue
                return i
        return -1

    def _find_matching_paren(self, expr, start):
        """Find the matching closing parenthesis."""
        depth = 0
        for i in range(start, len(expr)):
            if expr[i] == '(':
                depth += 1
            elif expr[i] == ')':
                depth -= 1
                if depth == 0:
                    return i
        return -1


# ── Buckingham Pi Theorem ────────────────────────────────────────────────────

def buckingham_pi(variables):
    """
    Compute dimensionless groups from a list of (name, dim_tuple) pairs.
    Returns a list of dicts with 'name' and 'exponents' keys.
    variables: list of (name, dim_tuple)
    """
    n = len(variables)
    if n == 0:
        return []

    # Build dimension matrix: each column is a variable, each row is a dimension
    # Only use dimensions that are actually present
    active_dims = []
    for i in range(DIM_COUNT):
        if any(v[1][i] != 0 for v in variables):
            active_dims.append(i)

    k = len(active_dims)  # number of independent dimensions
    if k == 0:
        return [{"name": v[0], "exponents": {v[0]: 1}} for v in variables]

    # Build matrix A (k x n)
    A = np.zeros((k, n))
    for j, (name, dim) in enumerate(variables):
        for row, dim_idx in enumerate(active_dims):
            A[row, j] = dim[dim_idx]

    # Find the null space of A using SVD
    rank = np.linalg.matrix_rank(A)
    num_pi = n - rank  # number of Pi groups

    if num_pi <= 0:
        return []

    # Use SVD to find null space
    U, S, Vt = np.linalg.svd(A)
    null_space = Vt[rank:].T  # columns are null space vectors

    pi_groups = []
    for i in range(null_space.shape[1]):
        vec = null_space[:, i]
        # Try to find integer exponents by scaling
        vec = _rationalize_vector(vec)
        exponents = {}
        for j, (name, _) in enumerate(variables):
            if abs(vec[j]) > 1e-10:
                exp = vec[j]
                if abs(exp - round(exp)) < 1e-6:
                    exp = int(round(exp))
                exponents[name] = exp
        if exponents:
            pi_groups.append({
                "name": f"Pi_{i + 1}",
                "exponents": exponents,
            })

    return pi_groups


def _rationalize_vector(vec):
    """Try to scale a vector to have small integer entries."""
    vec = vec.copy()
    # Find the element with smallest nonzero absolute value
    nonzero = [abs(v) for v in vec if abs(v) > 1e-10]
    if not nonzero:
        return vec
    min_val = min(nonzero)
    vec = vec / min_val

    # Check if all entries are close to integers
    for scale in [1, 2, 3, 4, 5, 6]:
        scaled = vec * scale
        if all(abs(s - round(s)) < 0.05 for s in scaled):
            return np.array([round(s) for s in scaled])
    return vec


def pi_group_to_str(pi):
    """Convert a Pi group dict to a readable string."""
    num_parts = []
    den_parts = []
    for name, exp in sorted(pi["exponents"].items()):
        if exp > 0:
            if exp == 1:
                num_parts.append(name)
            else:
                num_parts.append(f"{name}^{exp}")
        elif exp < 0:
            if exp == -1:
                den_parts.append(name)
            else:
                den_parts.append(f"{name}^{abs(exp)}")

    num_str = "*".join(num_parts) if num_parts else "1"
    if den_parts:
        den_str = "*".join(den_parts)
        return f"{pi['name']} = {num_str} / ({den_str})"
    return f"{pi['name']} = {num_str}"


# ── Main Widget ──────────────────────────────────────────────────────────────

class DimensionalAnalysisWidget(QWidget):
    """Dimensional Analysis Engine widget for QuantumRes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._logger = None
        self._user_vars = dict(PRESET_VARIABLES)
        self._parser = DimExprParser(self._user_vars)
        self._init_ui()

    # ── Public API ────────────────────────────────────────────────────────

    def set_logger(self, fn):
        """Set a logging callback function."""
        self._logger = fn

    def run(self):
        """Initialize and activate the widget."""
        self._log("Dimensional Analysis Engine initialized.")
        self._populate_reference_table()
        self._populate_dimensionless_table()
        self._populate_conversion_combos()
        self._populate_preset_combos()
        self._populate_pi_example()

    # ── Logging ───────────────────────────────────────────────────────────

    def _log(self, msg):
        if self._logger:
            self._logger(msg)

    # ── UI Construction ───────────────────────────────────────────────────

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        title = QLabel("Dimensional Analysis Engine")
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        tabs = QTabWidget()
        tabs.setFont(QFont("Segoe UI", 10))
        layout.addWidget(tabs)

        tabs.addTab(self._build_checker_tab(), "Unit Checker")
        tabs.addTab(self._build_variable_tab(), "Variables")
        tabs.addTab(self._build_pi_tab(), "Buckingham Pi")
        tabs.addTab(self._build_conversion_tab(), "Conversions")
        tabs.addTab(self._build_dimensionless_tab(), "Dimensionless Numbers")
        tabs.addTab(self._build_reference_tab(), "Reference Table")
        tabs.addTab(self._build_preset_tab(), "Presets")

    # ── Tab: Unit Checker ─────────────────────────────────────────────────

    def _build_checker_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        info = QLabel(
            "Enter an equation (e.g., F = m * a) to verify dimensional consistency.\n"
            "Variables must be defined in the Variables tab or presets."
        )
        info.setWordWrap(True)
        lay.addWidget(info)

        row = QHBoxLayout()
        row.addWidget(QLabel("Equation:"))
        self._eq_input = QLineEdit()
        self._eq_input.setPlaceholderText("F = m * a")
        self._eq_input.returnPressed.connect(self._check_equation)
        row.addWidget(self._eq_input)
        btn = QPushButton("Check")
        btn.clicked.connect(self._check_equation)
        row.addWidget(btn)
        lay.addLayout(row)

        self._eq_result = QTextEdit()
        self._eq_result.setReadOnly(True)
        self._eq_result.setFont(QFont("Consolas", 11))
        self._eq_result.setMaximumHeight(220)
        lay.addWidget(self._eq_result)

        # Expression dimension calculator
        grp = QGroupBox("Expression Dimension Calculator")
        grp_lay = QVBoxLayout(grp)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Expression:"))
        self._expr_input = QLineEdit()
        self._expr_input.setPlaceholderText("m * v**2 / r")
        self._expr_input.returnPressed.connect(self._calc_expression_dim)
        row2.addWidget(self._expr_input)
        btn2 = QPushButton("Compute")
        btn2.clicked.connect(self._calc_expression_dim)
        row2.addWidget(btn2)
        grp_lay.addLayout(row2)
        self._expr_result = QLabel("")
        self._expr_result.setFont(QFont("Consolas", 11))
        self._expr_result.setWordWrap(True)
        grp_lay.addWidget(self._expr_result)
        lay.addWidget(grp)

        lay.addStretch()
        return w

    # ── Tab: Variables ────────────────────────────────────────────────────

    def _build_variable_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        info = QLabel(
            "Assign dimensions to variables. Format: name = [M L T^-2]\n"
            "Example: v = [L T^-1]   or   rho = [M L^-3]"
        )
        info.setWordWrap(True)
        lay.addWidget(info)

        row = QHBoxLayout()
        self._var_input = QLineEdit()
        self._var_input.setPlaceholderText("variable = [dimensions]")
        self._var_input.returnPressed.connect(self._assign_variable)
        row.addWidget(self._var_input)
        btn = QPushButton("Assign")
        btn.clicked.connect(self._assign_variable)
        row.addWidget(btn)
        btn_clear = QPushButton("Reset to Defaults")
        btn_clear.clicked.connect(self._reset_variables)
        row.addWidget(btn_clear)
        lay.addLayout(row)

        self._var_table = QTableWidget()
        self._var_table.setColumnCount(3)
        self._var_table.setHorizontalHeaderLabels(["Variable", "Dimensions", "SI Units"])
        self._var_table.horizontalHeader().setStretchLastSection(True)
        self._var_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._var_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._var_table.setAlternatingRowColors(True)
        lay.addWidget(self._var_table)

        self._refresh_var_table()
        return w

    # ── Tab: Buckingham Pi ────────────────────────────────────────────────

    def _build_pi_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        info = QLabel(
            "Buckingham Pi Theorem: find dimensionless groups.\n"
            "Enter variables one per line as: name [dim]  e.g. F [M L T^-2]\n"
            "Or use the example below."
        )
        info.setWordWrap(True)
        lay.addWidget(info)

        self._pi_input = QPlainTextEdit()
        self._pi_input.setFont(QFont("Consolas", 10))
        self._pi_input.setMaximumHeight(180)
        lay.addWidget(self._pi_input)

        row = QHBoxLayout()
        btn = QPushButton("Compute Pi Groups")
        btn.clicked.connect(self._compute_pi)
        row.addWidget(btn)
        btn_ex = QPushButton("Load Example (Drag on sphere)")
        btn_ex.clicked.connect(self._populate_pi_example)
        row.addWidget(btn_ex)
        lay.addLayout(row)

        self._pi_result = QTextEdit()
        self._pi_result.setReadOnly(True)
        self._pi_result.setFont(QFont("Consolas", 11))
        lay.addWidget(self._pi_result)

        return w

    # ── Tab: Conversions ──────────────────────────────────────────────────

    def _build_conversion_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        info = QLabel("Convert between units within the same dimension.")
        lay.addWidget(info)

        form = QGridLayout()
        form.addWidget(QLabel("Category:"), 0, 0)
        self._conv_category = QComboBox()
        self._conv_category.currentTextChanged.connect(self._on_conv_category_changed)
        form.addWidget(self._conv_category, 0, 1, 1, 3)

        form.addWidget(QLabel("Value:"), 1, 0)
        self._conv_value = QLineEdit("1.0")
        form.addWidget(self._conv_value, 1, 1)

        form.addWidget(QLabel("From:"), 2, 0)
        self._conv_from = QComboBox()
        form.addWidget(self._conv_from, 2, 1)

        form.addWidget(QLabel("To:"), 2, 2)
        self._conv_to = QComboBox()
        form.addWidget(self._conv_to, 2, 3)

        btn = QPushButton("Convert")
        btn.clicked.connect(self._do_conversion)
        form.addWidget(btn, 3, 0, 1, 4)

        self._conv_result = QLabel("")
        self._conv_result.setFont(QFont("Consolas", 12))
        self._conv_result.setWordWrap(True)
        form.addWidget(self._conv_result, 4, 0, 1, 4)

        lay.addLayout(form)

        # Full conversion table
        self._conv_table = QTableWidget()
        self._conv_table.setAlternatingRowColors(True)
        self._conv_table.setEditTriggers(QTableWidget.NoEditTriggers)
        lay.addWidget(self._conv_table)

        lay.addStretch()
        return w

    # ── Tab: Dimensionless Numbers ────────────────────────────────────────

    def _build_dimensionless_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        info = QLabel("Common dimensionless numbers with formulas and physical meaning.")
        info.setWordWrap(True)
        lay.addWidget(info)

        row = QHBoxLayout()
        row.addWidget(QLabel("Search:"))
        self._dimless_search = QLineEdit()
        self._dimless_search.setPlaceholderText("Filter by name...")
        self._dimless_search.textChanged.connect(self._filter_dimensionless)
        row.addWidget(self._dimless_search)
        lay.addLayout(row)

        self._dimless_table = QTableWidget()
        self._dimless_table.setColumnCount(4)
        self._dimless_table.setHorizontalHeaderLabels(["Name", "Formula", "Variables", "Physical Meaning"])
        self._dimless_table.horizontalHeader().setStretchLastSection(True)
        self._dimless_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self._dimless_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._dimless_table.setAlternatingRowColors(True)
        self._dimless_table.setWordWrap(True)
        lay.addWidget(self._dimless_table)

        return w

    # ── Tab: Reference Table ──────────────────────────────────────────────

    def _build_reference_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        info = QLabel("Searchable table of physical quantities with dimensions and SI units.")
        lay.addWidget(info)

        row = QHBoxLayout()
        row.addWidget(QLabel("Search:"))
        self._ref_search = QLineEdit()
        self._ref_search.setPlaceholderText("Filter quantities...")
        self._ref_search.textChanged.connect(self._filter_reference)
        row.addWidget(self._ref_search)

        self._ref_style = QComboBox()
        self._ref_style.addItems(["Bracket notation", "SI unit notation"])
        self._ref_style.currentIndexChanged.connect(self._populate_reference_table)
        row.addWidget(self._ref_style)
        lay.addLayout(row)

        self._ref_table = QTableWidget()
        self._ref_table.setColumnCount(3)
        self._ref_table.setHorizontalHeaderLabels(["Quantity", "Dimensions", "SI Unit"])
        self._ref_table.horizontalHeader().setStretchLastSection(True)
        self._ref_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self._ref_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._ref_table.setAlternatingRowColors(True)
        self._ref_table.setSortingEnabled(True)
        lay.addWidget(self._ref_table)

        return w

    # ── Tab: Presets ──────────────────────────────────────────────────────

    def _build_preset_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        info = QLabel(
            "Common equations to check for dimensional consistency.\n"
            "Select a preset and click Check to verify."
        )
        info.setWordWrap(True)
        lay.addWidget(info)

        row = QHBoxLayout()
        row.addWidget(QLabel("Preset:"))
        self._preset_combo = QComboBox()
        row.addWidget(self._preset_combo)
        btn = QPushButton("Load && Check")
        btn.clicked.connect(self._load_and_check_preset)
        row.addWidget(btn)
        lay.addLayout(row)

        self._preset_result = QTextEdit()
        self._preset_result.setReadOnly(True)
        self._preset_result.setFont(QFont("Consolas", 11))
        lay.addWidget(self._preset_result)

        return w

    # ── Checker Logic ─────────────────────────────────────────────────────

    def _check_equation(self):
        eq = self._eq_input.text().strip()
        if not eq:
            return
        result_text, is_consistent = self._analyze_equation(eq)
        color = "#e6ffe6" if is_consistent else "#ffe6e6"
        border = "#22aa22" if is_consistent else "#cc2222"
        self._eq_result.setStyleSheet(
            f"QTextEdit {{ background-color: {color}; border: 2px solid {border}; "
            f"border-radius: 6px; padding: 8px; }}"
        )
        self._eq_result.setPlainText(result_text)
        self._log(f"Equation check: {eq} -> {'CONSISTENT' if is_consistent else 'INCONSISTENT'}")

    def _analyze_equation(self, eq):
        """Analyze an equation for dimensional consistency."""
        # Handle = sign
        if '=' not in eq:
            return "Error: equation must contain '=' sign.", False

        sides = eq.split('=')
        if len(sides) != 2:
            return "Error: equation must have exactly one '=' sign.", False

        lhs_expr = sides[0].strip()
        rhs_expr = sides[1].strip()

        try:
            lhs_dim = self._parser.parse(lhs_expr)
        except ValueError as e:
            return f"Error in LHS ({lhs_expr}):\n  {e}", False

        try:
            rhs_dim = self._parser.parse(rhs_expr)
        except ValueError as e:
            return f"Error in RHS ({rhs_expr}):\n  {e}", False

        lhs_str = dim_to_str(lhs_dim)
        rhs_str = dim_to_str(rhs_dim)
        consistent = dim_equal(lhs_dim, rhs_dim)

        lines = [
            f"Equation: {eq}",
            f"",
            f"LHS: {lhs_expr}",
            f"  Dimensions: {lhs_str}",
            f"",
            f"RHS: {rhs_expr}",
            f"  Dimensions: {rhs_str}",
            f"",
        ]
        if consistent:
            lines.append("RESULT: Dimensionally CONSISTENT")
            lines.append(f"  {lhs_str} = {rhs_str}")
        else:
            lines.append("RESULT: Dimensionally INCONSISTENT")
            lines.append(f"  {lhs_str} != {rhs_str}")
            # Show difference
            diff = tuple(lhs_dim[i] - rhs_dim[i] for i in range(DIM_COUNT))
            lines.append(f"  Difference: {dim_to_str(diff)}")

        return "\n".join(lines), consistent

    def _calc_expression_dim(self):
        """Calculate the dimensions of a single expression."""
        expr = self._expr_input.text().strip()
        if not expr:
            return
        try:
            dim = self._parser.parse(expr)
            dim_str = dim_to_str(dim, "bracket")
            si_str = self._dim_to_si_units(dim)
            self._expr_result.setText(
                f"Dimensions of '{expr}':  {dim_str}    (SI: {si_str})"
            )
            self._expr_result.setStyleSheet("color: #006600; font-weight: bold;")
        except ValueError as e:
            self._expr_result.setText(f"Error: {e}")
            self._expr_result.setStyleSheet("color: #cc0000; font-weight: bold;")

    # ── Variable Management ───────────────────────────────────────────────

    def _assign_variable(self):
        text = self._var_input.text().strip()
        if not text or '=' not in text:
            QMessageBox.warning(self, "Input Error", "Format: name = [dimensions]\nExample: v = [L T^-1]")
            return

        name, dim_str = text.split('=', 1)
        name = name.strip()
        dim_str = dim_str.strip()

        try:
            dim = parse_dim_str(dim_str)
        except Exception as e:
            QMessageBox.warning(self, "Parse Error", f"Could not parse dimensions: {e}")
            return

        self._user_vars[name] = dim
        self._parser.set_variable(name, dim)
        self._var_input.clear()
        self._refresh_var_table()
        self._log(f"Assigned variable: {name} = {dim_to_str(dim)}")

    def _reset_variables(self):
        self._user_vars = dict(PRESET_VARIABLES)
        self._parser = DimExprParser(self._user_vars)
        self._refresh_var_table()
        self._log("Variables reset to defaults.")

    def _refresh_var_table(self):
        self._var_table.setRowCount(len(self._user_vars))
        for row, (name, dim) in enumerate(sorted(self._user_vars.items())):
            self._var_table.setItem(row, 0, QTableWidgetItem(name))
            self._var_table.setItem(row, 1, QTableWidgetItem(dim_to_str(dim)))
            self._var_table.setItem(row, 2, QTableWidgetItem(self._dim_to_si_units(dim)))

    # ── Buckingham Pi ─────────────────────────────────────────────────────

    def _populate_pi_example(self):
        example = (
            "F [M L T^-2]\n"
            "v [L T^-1]\n"
            "rho [M L^-3]\n"
            "L_char [L]\n"
            "mu [M L^-1 T^-1]"
        )
        self._pi_input.setPlainText(example)

    def _compute_pi(self):
        text = self._pi_input.toPlainText().strip()
        if not text:
            return

        variables = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Parse "name [dim]" or "name dim"
            match = re.match(r'(\w+)\s+\[?([^\]]+)\]?', line)
            if match:
                name = match.group(1)
                dim = parse_dim_str(match.group(2))
                variables.append((name, dim))
            else:
                self._pi_result.setPlainText(f"Error: cannot parse line: {line}")
                return

        n = len(variables)
        if n < 2:
            self._pi_result.setPlainText("Need at least 2 variables.")
            return

        # Count independent dimensions
        active_dims = []
        for i in range(DIM_COUNT):
            if any(v[1][i] != 0 for v in variables):
                active_dims.append(i)
        k = len(active_dims)

        # Build dimension matrix for rank computation
        A = np.zeros((k, n))
        for j, (name, dim) in enumerate(variables):
            for row, dim_idx in enumerate(active_dims):
                A[row, j] = dim[dim_idx]
        rank = np.linalg.matrix_rank(A)
        num_pi = n - rank

        lines = [
            f"Variables: {n}",
            f"Independent dimensions: {rank}",
            f"Number of Pi groups: {num_pi}",
            f"",
            "Input variables:",
        ]
        for name, dim in variables:
            lines.append(f"  {name}: {dim_to_str(dim)}")
        lines.append("")

        pi_groups = buckingham_pi(variables)
        if pi_groups:
            lines.append("Dimensionless groups:")
            for pi in pi_groups:
                lines.append(f"  {pi_group_to_str(pi)}")
                # Verify dimensionless
                result_dim = DIMENSIONLESS
                for vname, exp in pi["exponents"].items():
                    for orig_name, orig_dim in variables:
                        if orig_name == vname:
                            result_dim = dim_multiply(result_dim, dim_power(orig_dim, exp))
                            break
                lines.append(f"    Verification: {dim_to_str(result_dim)}")
        else:
            lines.append("No dimensionless groups found (rank equals number of variables).")

        self._pi_result.setPlainText("\n".join(lines))
        self._log(f"Buckingham Pi: {n} variables -> {num_pi} Pi groups")

    # ── Conversions ───────────────────────────────────────────────────────

    def _populate_conversion_combos(self):
        self._conv_category.clear()
        for cat in sorted(UNIT_CONVERSIONS.keys()):
            self._conv_category.addItem(cat)
        if self._conv_category.count() > 0:
            self._on_conv_category_changed(self._conv_category.currentText())

    def _on_conv_category_changed(self, cat):
        self._conv_from.clear()
        self._conv_to.clear()
        if cat in UNIT_CONVERSIONS:
            units = list(UNIT_CONVERSIONS[cat]["units"].keys())
            self._conv_from.addItems(units)
            self._conv_to.addItems(units)
            if len(units) > 1:
                self._conv_to.setCurrentIndex(1)
            self._update_conversion_table(cat)

    def _update_conversion_table(self, cat):
        """Fill the conversion table showing all conversions from 1 base unit."""
        if cat not in UNIT_CONVERSIONS:
            return
        data = UNIT_CONVERSIONS[cat]
        units = list(data["units"].keys())
        n = len(units)
        self._conv_table.setColumnCount(n + 1)
        self._conv_table.setRowCount(n)
        self._conv_table.setHorizontalHeaderLabels(["From \\ To"] + units)
        self._conv_table.setVerticalHeaderLabels(units)

        for i, u_from in enumerate(units):
            self._conv_table.setItem(i, 0, QTableWidgetItem(u_from))
            f_from = data["units"][u_from]
            for j, u_to in enumerate(units):
                f_to = data["units"][u_to]
                factor = f_from / f_to
                item = QTableWidgetItem(f"{factor:.6g}")
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if i == j:
                    item.setBackground(QColor("#e0e0ff"))
                self._conv_table.setItem(i, j + 1, item)

        self._conv_table.resizeColumnsToContents()

    def _do_conversion(self):
        cat = self._conv_category.currentText()
        if cat not in UNIT_CONVERSIONS:
            return
        data = UNIT_CONVERSIONS[cat]
        try:
            val = float(self._conv_value.text())
        except ValueError:
            self._conv_result.setText("Error: invalid number")
            return

        u_from = self._conv_from.currentText()
        u_to = self._conv_to.currentText()
        if u_from not in data["units"] or u_to not in data["units"]:
            return

        f_from = data["units"][u_from]
        f_to = data["units"][u_to]
        result = val * f_from / f_to
        factor = f_from / f_to

        self._conv_result.setText(
            f"{val} {u_from} = {result:.10g} {u_to}\n"
            f"Conversion factor: 1 {u_from} = {factor:.10g} {u_to}"
        )
        self._log(f"Conversion: {val} {u_from} -> {result:.10g} {u_to}")

    # ── Dimensionless Numbers Table ───────────────────────────────────────

    def _populate_dimensionless_table(self):
        nums = DIMENSIONLESS_NUMBERS
        self._dimless_table.setRowCount(len(nums))
        for row, (name, info) in enumerate(sorted(nums.items())):
            self._dimless_table.setItem(row, 0, QTableWidgetItem(name))
            self._dimless_table.setItem(row, 1, QTableWidgetItem(info["formula"]))
            self._dimless_table.setItem(row, 2, QTableWidgetItem(info["variables"]))
            self._dimless_table.setItem(row, 3, QTableWidgetItem(info["meaning"]))
        self._dimless_table.resizeColumnsToContents()
        self._dimless_table.resizeRowsToContents()

    def _filter_dimensionless(self, text):
        text = text.lower()
        for row in range(self._dimless_table.rowCount()):
            name_item = self._dimless_table.item(row, 0)
            if name_item:
                visible = text in name_item.text().lower()
                self._dimless_table.setRowHidden(row, not visible)

    # ── Reference Table ───────────────────────────────────────────────────

    def _populate_reference_table(self):
        style = "bracket" if self._ref_style.currentIndex() == 0 else "si"
        items = sorted(DIMENSION_DB.items())
        self._ref_table.setRowCount(len(items))
        for row, (name, info) in enumerate(items):
            self._ref_table.setItem(row, 0, QTableWidgetItem(name))
            if style == "bracket":
                self._ref_table.setItem(row, 1, QTableWidgetItem(dim_to_str(info["dim"])))
            else:
                self._ref_table.setItem(row, 1, QTableWidgetItem(self._dim_to_si_units(info["dim"])))
            self._ref_table.setItem(row, 2, QTableWidgetItem(info["si"]))
        self._ref_table.resizeColumnsToContents()

    def _filter_reference(self, text):
        text = text.lower()
        for row in range(self._ref_table.rowCount()):
            name_item = self._ref_table.item(row, 0)
            if name_item:
                visible = text in name_item.text().lower()
                self._ref_table.setRowHidden(row, not visible)

    # ── Presets ───────────────────────────────────────────────────────────

    def _populate_preset_combos(self):
        self._preset_combo.clear()
        for name in PRESET_EQUATIONS:
            self._preset_combo.addItem(name)

    def _load_and_check_preset(self):
        name = self._preset_combo.currentText()
        if name not in PRESET_EQUATIONS:
            return
        eq = PRESET_EQUATIONS[name]
        result_text, is_consistent = self._analyze_equation(eq)

        header = f"Preset: {name}\nEquation: {eq}\n{'=' * 50}\n\n"
        self._preset_result.setPlainText(header + result_text)

        color = "#e6ffe6" if is_consistent else "#ffe6e6"
        border = "#22aa22" if is_consistent else "#cc2222"
        self._preset_result.setStyleSheet(
            f"QTextEdit {{ background-color: {color}; border: 2px solid {border}; "
            f"border-radius: 6px; padding: 8px; }}"
        )
        self._log(f"Preset check: {name} -> {'CONSISTENT' if is_consistent else 'INCONSISTENT'}")

    # ── Utilities ─────────────────────────────────────────────────────────

    def _dim_to_si_units(self, dim):
        """Convert a dimension tuple to SI unit string."""
        si_base = ["kg", "m", "s", "A", "K", "mol", "cd"]
        if all(d == 0 for d in dim):
            return "1 (dimensionless)"
        num_parts = []
        den_parts = []
        for i, label in enumerate(si_base):
            if i < len(dim) and dim[i] != 0:
                exp = dim[i]
                if exp > 0:
                    if exp == 1:
                        num_parts.append(label)
                    elif int(exp) == exp:
                        num_parts.append(f"{label}^{int(exp)}")
                    else:
                        num_parts.append(f"{label}^{exp}")
                else:
                    aexp = abs(exp)
                    if aexp == 1:
                        den_parts.append(label)
                    elif int(aexp) == aexp:
                        den_parts.append(f"{label}^{int(aexp)}")
                    else:
                        den_parts.append(f"{label}^{aexp}")
        num_str = "*".join(num_parts) if num_parts else "1"
        if den_parts:
            den_str = "*".join(den_parts)
            if len(den_parts) > 1:
                return f"{num_str}/({den_str})"
            return f"{num_str}/{den_str}"
        return num_str
