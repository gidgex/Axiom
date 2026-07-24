"""
Math Engine Widget - Scientific computing suite for PyQt5.

Provides matrix operations, equation solving, symbolic mathematics,
and expression evaluation in a tabbed interface.
"""

import traceback
import math
import csv
import io
import numpy as np
from scipy import linalg as sp_linalg
from scipy.optimize import fsolve, minimize as sp_minimize
from scipy.integrate import solve_ivp

import sympy
from sympy import (
    symbols, sympify, diff, integrate, limit, series, simplify,
    factor, expand, solve, Poly, oo, sin, cos, tan, exp, log,
    sqrt, pi, E, I, Matrix as SympyMatrix, det, Rational,
    Symbol, latex, pretty, Function, factorint, gcd, lcm,
    isprime, nextprime, mod_inverse, totient, primitive_root
)
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application,
    convert_xor
)

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QTextEdit,
    QLineEdit, QPushButton, QLabel, QGridLayout, QComboBox,
    QGroupBox, QSplitter, QSpinBox, QPlainTextEdit, QFormLayout,
    QMessageBox, QFrame, QSizePolicy, QFileDialog, QCheckBox,
    QDoubleSpinBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QTextCursor


def _clean_float(x, tol=1e-10):
    """Clean up floating-point noise. Returns a clean number."""
    if isinstance(x, (complex, np.complexfloating)):
        re = _clean_float(x.real, tol)
        im = _clean_float(x.imag, tol)
        if im == 0:
            return re
        return complex(re, im)
    if isinstance(x, (float, np.floating)):
        rounded = round(x)
        if abs(x - rounded) < tol:
            return int(rounded)
        # Try rounding to 10 significant digits
        if x != 0:
            magnitude = 10 ** (10 - int(np.floor(np.log10(abs(x)))) - 1)
            cleaned = round(x * magnitude) / magnitude
            rounded2 = round(cleaned)
            if abs(cleaned - rounded2) < tol:
                return int(rounded2)
            return cleaned
    return x


def _fmt_num(x, tol=1e-10):
    """Format a number cleanly as string."""
    cleaned = _clean_float(x, tol)
    if isinstance(cleaned, complex):
        re, im = cleaned.real, cleaned.imag
        if re == 0 and im == 0:
            return "0"
        if re == 0:
            return f"{im}j"
        if im == 0:
            return str(re)
        sign = "+" if im >= 0 else ""
        return f"{re}{sign}{im}j"
    return str(cleaned)


def _fmt_array(arr, tol=1e-10):
    """Format a numpy array with cleaned floats."""
    if arr.ndim == 1:
        items = [_fmt_num(v, tol) for v in arr]
        return "[" + ", ".join(items) + "]"
    elif arr.ndim == 2:
        rows = []
        for i in range(arr.shape[0]):
            items = [_fmt_num(arr[i, j], tol) for j in range(arr.shape[1])]
            rows.append("[" + ", ".join(items) + "]")
        return "[\n  " + "\n  ".join(rows) + "\n]"
    return str(arr)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TRANSFORM = standard_transformations + (implicit_multiplication_application, convert_xor)


def _parse(expr_str: str):
    """Parse a string into a SymPy expression with convenience transforms."""
    local_dict = {
        "pi": pi, "e": E, "i": I, "E": E, "I": I,
        "sin": sin, "cos": cos, "tan": tan, "exp": exp,
        "log": log, "sqrt": sqrt, "oo": oo, "inf": oo,
    }
    return parse_expr(expr_str, local_dict=local_dict, transformations=TRANSFORM)


def _parse_matrix_text(text: str) -> np.ndarray:
    """Parse a text block into a NumPy matrix.

    Accepted formats:
        1 2 3
        4 5 6
    or
        1, 2, 3; 4, 5, 6
    or
        [[1,2],[3,4]]
    """
    text = text.strip()
    if not text:
        raise ValueError("Empty matrix input.")

    # Try numpy-style literal
    if text.startswith("["):
        text_clean = text.replace(";", ",")
        return np.array(eval(text_clean, {"__builtins__": {}}, {"np": np}), dtype=float)

    # Semicolon-delimited rows
    if ";" in text:
        rows = [r.strip() for r in text.split(";") if r.strip()]
    else:
        rows = [r.strip() for r in text.splitlines() if r.strip()]

    parsed_rows = []
    for r in rows:
        r = r.replace(",", " ")
        parsed_rows.append([float(x) for x in r.split()])
    return np.array(parsed_rows, dtype=float)


# ---------------------------------------------------------------------------
# Styled result text widget
# ---------------------------------------------------------------------------

class ResultDisplay(QTextEdit):
    """Read-only text area for showing computation results."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 10))
        self.setStyleSheet(
            "QTextEdit { background-color: #1e1e2e; color: #cdd6f4; "
            "border: 1px solid #45475a; border-radius: 4px; padding: 6px; }"
        )
        self._last_title = ""
        self._last_body = ""
        self._last_sympy_expr = None  # store last SymPy expression for LaTeX

    def append_result(self, title: str, body: str):
        self._last_title = title
        self._last_body = body
        self.moveCursor(QTextCursor.End)
        self.append(f"--- {title} ---\n{body}\n")
        self.moveCursor(QTextCursor.End)

    def set_result(self, title: str, body: str):
        self._last_title = title
        self._last_body = body
        self.clear()
        self.append_result(title, body)

    def store_sympy_expr(self, expr):
        """Store the last SymPy expression for LaTeX generation."""
        self._last_sympy_expr = expr

    def get_last_sympy_expr(self):
        return self._last_sympy_expr

    def get_last_result(self):
        return self._last_title, self._last_body


# ---------------------------------------------------------------------------
# Calculator Tab
# ---------------------------------------------------------------------------

class CalculatorTab(QWidget):
    """General-purpose expression evaluator."""

    def __init__(self, result_display: ResultDisplay, parent=None):
        super().__init__(parent)
        self.result_display = result_display
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        info = QLabel("Enter a mathematical expression (e.g. sin(pi/4) + sqrt(2), 3**2 + log(10)):")
        info.setWordWrap(True)
        layout.addWidget(info)

        self.expr_input = QLineEdit()
        self.expr_input.setPlaceholderText("e.g.  2**10 + sin(pi/6)")
        self.expr_input.returnPressed.connect(self._evaluate)
        layout.addWidget(self.expr_input)

        btn_row = QHBoxLayout()
        eval_btn = QPushButton("Evaluate")
        eval_btn.clicked.connect(self._evaluate)
        eval_num_btn = QPushButton("Evaluate (numeric)")
        eval_num_btn.clicked.connect(self._evaluate_numeric)
        simplify_btn = QPushButton("Simplify")
        simplify_btn.clicked.connect(self._simplify)
        expand_btn = QPushButton("Expand")
        expand_btn.clicked.connect(self._expand)
        factor_btn = QPushButton("Factor")
        factor_btn.clicked.connect(self._factor)
        btn_row.addWidget(eval_btn)
        btn_row.addWidget(eval_num_btn)
        btn_row.addWidget(simplify_btn)
        btn_row.addWidget(expand_btn)
        btn_row.addWidget(factor_btn)
        layout.addLayout(btn_row)

        # Quick-insert buttons
        quick_row = QHBoxLayout()
        for label in ["pi", "e", "sqrt(", "sin(", "cos(", "tan(", "log(", "exp(", "**", "("]:
            b = QPushButton(label)
            b.setMaximumWidth(60)
            b.clicked.connect(lambda checked, t=label: self._insert(t))
            quick_row.addWidget(b)
        layout.addLayout(quick_row)

        layout.addStretch()

    def _get_expr(self):
        text = self.expr_input.text().strip()
        if not text:
            raise ValueError("Please enter an expression.")
        return _parse(text)

    def _evaluate(self):
        try:
            expr = self._get_expr()
            result = sympy.simplify(expr)
            self.result_display.set_result("Evaluate", f"{self.expr_input.text().strip()}\n= {pretty(result, use_unicode=True)}")
        except Exception as exc:
            self.result_display.set_result("Error", str(exc))

    def _evaluate_numeric(self):
        try:
            expr = self._get_expr()
            result = complex(expr.evalf())
            if result.imag == 0:
                self.result_display.set_result("Numeric Result", f"{self.expr_input.text().strip()}\n= {result.real}")
            else:
                self.result_display.set_result("Numeric Result", f"{self.expr_input.text().strip()}\n= {result}")
        except Exception as exc:
            self.result_display.set_result("Error", str(exc))

    def _simplify(self):
        try:
            expr = self._get_expr()
            self.result_display.set_result("Simplify", pretty(simplify(expr), use_unicode=True))
        except Exception as exc:
            self.result_display.set_result("Error", str(exc))

    def _expand(self):
        try:
            expr = self._get_expr()
            self.result_display.set_result("Expand", pretty(expand(expr), use_unicode=True))
        except Exception as exc:
            self.result_display.set_result("Error", str(exc))

    def _factor(self):
        try:
            expr = self._get_expr()
            self.result_display.set_result("Factor", pretty(factor(expr), use_unicode=True))
        except Exception as exc:
            self.result_display.set_result("Error", str(exc))

    def _insert(self, text):
        self.expr_input.insert(text)
        self.expr_input.setFocus()


# ---------------------------------------------------------------------------
# Matrix Operations Tab
# ---------------------------------------------------------------------------

class MatrixTab(QWidget):
    """Matrix calculator with NumPy / SciPy operations."""

    def __init__(self, result_display: ResultDisplay, parent=None):
        super().__init__(parent)
        self.result_display = result_display
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # --- Matrix A ---
        ga = QGroupBox("Matrix A")
        ga_layout = QVBoxLayout(ga)
        ga_layout.addWidget(QLabel("Enter matrix (rows on separate lines, values separated by spaces or commas):"))
        self.mat_a = QPlainTextEdit()
        self.mat_a.setMaximumHeight(120)
        self.mat_a.setPlaceholderText("1 2 3\n4 5 6\n7 8 9")
        ga_layout.addWidget(self.mat_a)
        layout.addWidget(ga)

        # --- Matrix B (for multiplication) ---
        gb = QGroupBox("Matrix B (for multiplication / solving Ax=b)")
        gb_layout = QVBoxLayout(gb)
        self.mat_b = QPlainTextEdit()
        self.mat_b.setMaximumHeight(100)
        self.mat_b.setPlaceholderText("1 0\n0 1\n1 1")
        gb_layout.addWidget(self.mat_b)
        layout.addWidget(gb)

        # --- Operation buttons ---
        ops_grid = QGridLayout()
        operations = [
            ("Determinant", self._determinant),
            ("Inverse", self._inverse),
            ("Eigenvalues", self._eigenvalues),
            ("SVD", self._svd),
            ("Rank", self._rank),
            ("Trace", self._trace),
            ("Transpose", self._transpose),
            ("A x B", self._multiply),
            ("Solve Ax=b", self._solve_linear),
            ("LU Decomposition", self._lu),
            ("QR Decomposition", self._qr),
            ("Cholesky", self._cholesky),
            ("Norm", self._norm),
            ("Condition Number", self._cond),
        ]
        for idx, (label, slot) in enumerate(operations):
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            ops_grid.addWidget(btn, idx // 4, idx % 4)
        layout.addLayout(ops_grid)
        layout.addStretch()

    def _get_a(self) -> np.ndarray:
        return _parse_matrix_text(self.mat_a.toPlainText())

    def _get_b(self) -> np.ndarray:
        return _parse_matrix_text(self.mat_b.toPlainText())

    def _show(self, title, body):
        self.result_display.set_result(title, body)

    def _determinant(self):
        try:
            A = self._get_a()
            if A.shape[0] != A.shape[1]:
                raise ValueError("Determinant requires a square matrix.")
            d = np.linalg.det(A)
            self._show("Determinant", f"det(A) = {_fmt_num(d)}")
        except Exception as e:
            self._show("Error", str(e))

    def _inverse(self):
        try:
            A = self._get_a()
            inv = np.linalg.inv(A)
            self._show("Inverse of A", _fmt_array(inv))
        except Exception as e:
            self._show("Error", str(e))

    def _eigenvalues(self):
        try:
            A = self._get_a()
            vals, vecs = np.linalg.eig(A)
            lines = ["Eigenvalues:"]
            for i, v in enumerate(vals):
                lines.append(f"  lambda_{i+1} = {_fmt_num(v)}")
            lines.append("\nEigenvectors (columns):")
            lines.append(_fmt_array(vecs))
            self._show("Eigenvalues & Eigenvectors", "\n".join(lines))
        except Exception as e:
            self._show("Error", str(e))

    def _svd(self):
        try:
            A = self._get_a()
            U, s, Vt = np.linalg.svd(A)
            self._show("Singular Value Decomposition",
                        f"U:\n{_fmt_array(U)}\n\nSigma (singular values):\n{_fmt_array(s)}\n\nV^T:\n{_fmt_array(Vt)}")
        except Exception as e:
            self._show("Error", str(e))

    def _rank(self):
        try:
            A = self._get_a()
            r = np.linalg.matrix_rank(A)
            self._show("Matrix Rank", f"rank(A) = {r}")
        except Exception as e:
            self._show("Error", str(e))

    def _trace(self):
        try:
            A = self._get_a()
            self._show("Trace", f"trace(A) = {_fmt_num(np.trace(A))}")
        except Exception as e:
            self._show("Error", str(e))

    def _transpose(self):
        try:
            A = self._get_a()
            self._show("Transpose of A", _fmt_array(A.T))
        except Exception as e:
            self._show("Error", str(e))

    def _multiply(self):
        try:
            A = self._get_a()
            B = self._get_b()
            self._show("A x B", _fmt_array(A @ B))
        except Exception as e:
            self._show("Error", str(e))

    def _solve_linear(self):
        try:
            A = self._get_a()
            b = self._get_b()
            if b.ndim == 2 and b.shape[1] == 1:
                b = b.flatten()
            x = np.linalg.solve(A, b)
            self._show("Solution of Ax = b", f"x =\n{_fmt_array(x)}")
        except Exception as e:
            self._show("Error", str(e))

    def _lu(self):
        try:
            A = self._get_a()
            P, L, U = sp_linalg.lu(A)
            self._show("LU Decomposition", f"P:\n{_fmt_array(P)}\n\nL:\n{_fmt_array(L)}\n\nU:\n{_fmt_array(U)}")
        except Exception as e:
            self._show("Error", str(e))

    def _qr(self):
        try:
            A = self._get_a()
            Q, R = np.linalg.qr(A)
            self._show("QR Decomposition", f"Q:\n{_fmt_array(Q)}\n\nR:\n{_fmt_array(R)}")
        except Exception as e:
            self._show("Error", str(e))

    def _cholesky(self):
        try:
            A = self._get_a()
            L = np.linalg.cholesky(A)
            self._show("Cholesky Decomposition", f"L (lower triangular):\n{_fmt_array(L)}")
        except Exception as e:
            self._show("Error", str(e))

    def _norm(self):
        try:
            A = self._get_a()
            norms = {
                "Frobenius": np.linalg.norm(A, "fro"),
                "2-norm": np.linalg.norm(A, 2),
                "1-norm": np.linalg.norm(A, 1),
                "inf-norm": np.linalg.norm(A, np.inf),
            }
            lines = [f"  {k}: {_fmt_num(v)}" for k, v in norms.items()]
            self._show("Matrix Norms", "\n".join(lines))
        except Exception as e:
            self._show("Error", str(e))

    def _cond(self):
        try:
            A = self._get_a()
            c = np.linalg.cond(A)
            self._show("Condition Number", f"cond(A) = {_fmt_num(c)}")
        except Exception as e:
            self._show("Error", str(e))


# ---------------------------------------------------------------------------
# Equation Solver Tab
# ---------------------------------------------------------------------------

class EquationSolverTab(QWidget):
    """Solve polynomials, systems, and nonlinear equations."""

    def __init__(self, result_display: ResultDisplay, parent=None):
        super().__init__(parent)
        self.result_display = result_display
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # --- Polynomial roots ---
        pg = QGroupBox("Polynomial Roots")
        pl = QVBoxLayout(pg)
        pl.addWidget(QLabel("Enter coefficients highest-degree first, separated by spaces (e.g. 1 0 -1 for x^2-1):"))
        self.poly_input = QLineEdit()
        self.poly_input.setPlaceholderText("1 0 -1")
        pl.addWidget(self.poly_input)
        poly_btn = QPushButton("Find Roots")
        poly_btn.clicked.connect(self._poly_roots)
        pl.addWidget(poly_btn)
        layout.addWidget(pg)

        # --- Symbolic equation solving ---
        sg = QGroupBox("Symbolic Equation Solver")
        sl = QVBoxLayout(sg)
        sl.addWidget(QLabel("Enter equation(s), one per line (e.g. x**2 - 4 = 0, or x + y = 5):"))
        self.sym_eq_input = QPlainTextEdit()
        self.sym_eq_input.setMaximumHeight(80)
        self.sym_eq_input.setPlaceholderText("x**2 + 2*x - 3 = 0")
        sl.addWidget(self.sym_eq_input)

        var_row = QHBoxLayout()
        var_row.addWidget(QLabel("Variables:"))
        self.var_input = QLineEdit("x")
        self.var_input.setPlaceholderText("x, y")
        var_row.addWidget(self.var_input)
        sl.addLayout(var_row)

        sym_btn = QPushButton("Solve Symbolically")
        sym_btn.clicked.connect(self._solve_symbolic)
        sl.addWidget(sym_btn)
        layout.addWidget(sg)

        # --- Nonlinear numeric solver ---
        ng = QGroupBox("Nonlinear Numeric Solver (scipy.fsolve)")
        nl = QVBoxLayout(ng)
        nl.addWidget(QLabel("Enter equation as f(x)=0 expression (e.g. cos(x) - x):"))
        self.nonlin_input = QLineEdit()
        self.nonlin_input.setPlaceholderText("cos(x) - x")
        nl.addWidget(self.nonlin_input)

        x0_row = QHBoxLayout()
        x0_row.addWidget(QLabel("Initial guess x0:"))
        self.x0_input = QLineEdit("1.0")
        x0_row.addWidget(self.x0_input)
        nl.addLayout(x0_row)

        nl_btn = QPushButton("Solve Numerically")
        nl_btn.clicked.connect(self._solve_nonlinear)
        nl.addWidget(nl_btn)
        layout.addWidget(ng)

        layout.addStretch()

    def _poly_roots(self):
        try:
            text = self.poly_input.text().strip()
            coeffs = [float(c) for c in text.replace(",", " ").split()]
            roots = np.roots(coeffs)
            lines = [f"Polynomial degree: {len(coeffs) - 1}", "Roots:"]
            for i, r in enumerate(roots):
                if np.isreal(r):
                    lines.append(f"  r{i+1} = {r.real}")
                else:
                    lines.append(f"  r{i+1} = {r}")
            self.result_display.set_result("Polynomial Roots", "\n".join(lines))
        except Exception as e:
            self.result_display.set_result("Error", str(e))

    def _solve_symbolic(self):
        try:
            var_names = [v.strip() for v in self.var_input.text().split(",") if v.strip()]
            sym_vars = symbols(var_names)
            if not isinstance(sym_vars, tuple):
                sym_vars = (sym_vars,)

            raw_lines = [l.strip() for l in self.sym_eq_input.toPlainText().splitlines() if l.strip()]
            equations = []
            for line in raw_lines:
                if "=" in line:
                    lhs, rhs = line.split("=", 1)
                    equations.append(_parse(lhs) - _parse(rhs))
                else:
                    equations.append(_parse(line))

            sol = solve(equations, sym_vars, dict=True)
            if not sol:
                sol = solve(equations, sym_vars)

            lines = []
            if isinstance(sol, list):
                for i, s in enumerate(sol):
                    lines.append(f"Solution {i+1}: {s}")
            else:
                lines.append(str(sol))

            self.result_display.set_result("Symbolic Solution", "\n".join(lines))
        except Exception as e:
            self.result_display.set_result("Error", str(e))

    def _solve_nonlinear(self):
        try:
            x = symbols("x")
            expr = _parse(self.nonlin_input.text().strip())
            f_lambda = sympy.lambdify(x, expr, modules=["numpy"])
            x0 = float(self.x0_input.text().strip())
            root, info, ier, msg = fsolve(f_lambda, x0, full_output=True)
            lines = [f"Root: x = {root[0]}", f"f(x) = {f_lambda(root[0])}", f"Converged: {'Yes' if ier == 1 else 'No'}", f"Message: {msg}"]
            self.result_display.set_result("Nonlinear Solver", "\n".join(lines))
        except Exception as e:
            self.result_display.set_result("Error", str(e))


# ---------------------------------------------------------------------------
# Symbolic Math Tab
# ---------------------------------------------------------------------------

class SymbolicMathTab(QWidget):
    """Differentiation, integration, limits, Taylor series."""

    def __init__(self, result_display: ResultDisplay, parent=None):
        super().__init__(parent)
        self.result_display = result_display
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Expression input
        expr_group = QGroupBox("Expression")
        eg_layout = QFormLayout(expr_group)
        self.expr_input = QLineEdit()
        self.expr_input.setPlaceholderText("e.g.  x**3 * sin(x)")
        eg_layout.addRow("f(x) =", self.expr_input)

        self.var_input = QLineEdit("x")
        eg_layout.addRow("Variable:", self.var_input)
        layout.addWidget(expr_group)

        # --- Differentiation ---
        diff_group = QGroupBox("Differentiation")
        dl = QHBoxLayout(diff_group)
        dl.addWidget(QLabel("Order:"))
        self.diff_order = QSpinBox()
        self.diff_order.setRange(1, 10)
        self.diff_order.setValue(1)
        dl.addWidget(self.diff_order)
        diff_btn = QPushButton("Differentiate")
        diff_btn.clicked.connect(self._differentiate)
        dl.addWidget(diff_btn)
        layout.addWidget(diff_group)

        # --- Integration ---
        int_group = QGroupBox("Integration")
        il = QGridLayout(int_group)
        il.addWidget(QLabel("Lower:"), 0, 0)
        self.int_lower = QLineEdit()
        self.int_lower.setPlaceholderText("(leave empty for indefinite)")
        il.addWidget(self.int_lower, 0, 1)
        il.addWidget(QLabel("Upper:"), 0, 2)
        self.int_upper = QLineEdit()
        il.addWidget(self.int_upper, 0, 3)
        int_btn = QPushButton("Integrate")
        int_btn.clicked.connect(self._integrate)
        il.addWidget(int_btn, 1, 0, 1, 4)
        layout.addWidget(int_group)

        # --- Limits ---
        lim_group = QGroupBox("Limits")
        ll = QHBoxLayout(lim_group)
        ll.addWidget(QLabel("As var ->"))
        self.lim_point = QLineEdit("0")
        self.lim_point.setMaximumWidth(80)
        ll.addWidget(self.lim_point)
        self.lim_dir = QComboBox()
        self.lim_dir.addItems(["both", "+", "-"])
        ll.addWidget(self.lim_dir)
        lim_btn = QPushButton("Compute Limit")
        lim_btn.clicked.connect(self._limit)
        ll.addWidget(lim_btn)
        layout.addWidget(lim_group)

        # --- Taylor / Laurent series ---
        tay_group = QGroupBox("Taylor Series")
        tl = QHBoxLayout(tay_group)
        tl.addWidget(QLabel("About:"))
        self.taylor_point = QLineEdit("0")
        self.taylor_point.setMaximumWidth(60)
        tl.addWidget(self.taylor_point)
        tl.addWidget(QLabel("Order:"))
        self.taylor_order = QSpinBox()
        self.taylor_order.setRange(1, 20)
        self.taylor_order.setValue(6)
        tl.addWidget(self.taylor_order)
        tay_btn = QPushButton("Expand")
        tay_btn.clicked.connect(self._taylor)
        tl.addWidget(tay_btn)
        layout.addWidget(tay_group)

        # --- Simplify ---
        simp_row = QHBoxLayout()
        simp_btn = QPushButton("Simplify Expression")
        simp_btn.clicked.connect(self._simplify_expr)
        simp_row.addWidget(simp_btn)
        trig_btn = QPushButton("Trig Simplify")
        trig_btn.clicked.connect(self._trig_simplify)
        simp_row.addWidget(trig_btn)
        layout.addLayout(simp_row)

        layout.addStretch()

    def _get_expr_and_var(self):
        var = Symbol(self.var_input.text().strip() or "x")
        expr = _parse(self.expr_input.text().strip())
        return expr, var

    def _differentiate(self):
        try:
            expr, var = self._get_expr_and_var()
            order = self.diff_order.value()
            result = diff(expr, var, order)
            label = f"d{''.join(['']*order)}{'(' + str(order) + ')' if order > 1 else ''}"
            self.result_display.set_result(
                f"Derivative (order {order})",
                f"f({var}) = {pretty(expr, use_unicode=True)}\n\n"
                f"f{'(' + chr(8304+order) + ')' if order < 4 else '('+str(order)+')'}({var}) = {pretty(result, use_unicode=True)}"
            )
        except Exception as e:
            self.result_display.set_result("Error", str(e))

    def _integrate(self):
        try:
            expr, var = self._get_expr_and_var()
            lo = self.int_lower.text().strip()
            hi = self.int_upper.text().strip()
            if lo and hi:
                lo_val = _parse(lo)
                hi_val = _parse(hi)
                result = integrate(expr, (var, lo_val, hi_val))
                self.result_display.set_result(
                    "Definite Integral",
                    f"Integral of {pretty(expr, use_unicode=True)} from {lo} to {hi}\n\n= {pretty(result, use_unicode=True)}"
                )
            else:
                result = integrate(expr, var)
                self.result_display.set_result(
                    "Indefinite Integral",
                    f"Integral of {pretty(expr, use_unicode=True)} d{var}\n\n= {pretty(result, use_unicode=True)} + C"
                )
        except Exception as e:
            self.result_display.set_result("Error", str(e))

    def _limit(self):
        try:
            expr, var = self._get_expr_and_var()
            point = _parse(self.lim_point.text().strip())
            direction = self.lim_dir.currentText()
            if direction == "both":
                result = limit(expr, var, point)
            else:
                result = limit(expr, var, point, direction)
            self.result_display.set_result(
                "Limit",
                f"lim ({var} -> {point}{'+' if direction == '+' else '-' if direction == '-' else ''}) "
                f"of {pretty(expr, use_unicode=True)}\n\n= {pretty(result, use_unicode=True)}"
            )
        except Exception as e:
            self.result_display.set_result("Error", str(e))

    def _taylor(self):
        try:
            expr, var = self._get_expr_and_var()
            point = _parse(self.taylor_point.text().strip())
            order = self.taylor_order.value()
            result = series(expr, var, point, n=order)
            self.result_display.set_result(
                f"Taylor Series (order {order})",
                f"f({var}) = {pretty(expr, use_unicode=True)}\n\n"
                f"Series about {var}={point}:\n{pretty(result, use_unicode=True)}"
            )
        except Exception as e:
            self.result_display.set_result("Error", str(e))

    def _simplify_expr(self):
        try:
            expr, var = self._get_expr_and_var()
            result = simplify(expr)
            self.result_display.set_result("Simplified", pretty(result, use_unicode=True))
        except Exception as e:
            self.result_display.set_result("Error", str(e))

    def _trig_simplify(self):
        try:
            expr, var = self._get_expr_and_var()
            result = sympy.trigsimp(expr)
            self.result_display.set_result("Trig Simplified", pretty(result, use_unicode=True))
        except Exception as e:
            self.result_display.set_result("Error", str(e))


# ---------------------------------------------------------------------------
# ODE Solver Tab
# ---------------------------------------------------------------------------

class ODESolverTab(QWidget):
    """Solve ordinary differential equations dy/dx = f(x, y)."""

    def __init__(self, result_display: ResultDisplay, parent=None):
        super().__init__(parent)
        self.result_display = result_display
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        info = QLabel("Solve dy/dx = f(x, y) with initial condition y(x0) = y0")
        info.setWordWrap(True)
        layout.addWidget(info)

        form = QFormLayout()

        self.ode_input = QLineEdit()
        self.ode_input.setPlaceholderText("e.g.  -2*y + sin(x)  or  x**2 - y")
        form.addRow("f(x, y) =", self.ode_input)

        self.x0_input = QLineEdit("0")
        form.addRow("x0:", self.x0_input)

        self.y0_input = QLineEdit("1")
        form.addRow("y0:", self.y0_input)

        self.x_end_input = QLineEdit("10")
        form.addRow("x_end:", self.x_end_input)

        self.n_points_spin = QSpinBox()
        self.n_points_spin.setRange(50, 10000)
        self.n_points_spin.setValue(500)
        form.addRow("Points:", self.n_points_spin)

        self.method_combo = QComboBox()
        self.method_combo.addItems(["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"])
        form.addRow("Method:", self.method_combo)

        layout.addLayout(form)

        btn_row = QHBoxLayout()
        solve_btn = QPushButton("Solve ODE")
        solve_btn.clicked.connect(self._solve)
        btn_row.addWidget(solve_btn)

        plot_btn = QPushButton("Solve and Plot")
        plot_btn.clicked.connect(self._solve_and_plot)
        btn_row.addWidget(plot_btn)
        layout.addLayout(btn_row)

        # Preset examples
        preset_group = QGroupBox("Presets")
        preset_layout = QHBoxLayout(preset_group)
        presets = [
            ("Exponential decay", "-y", "0", "1", "5"),
            ("Logistic growth", "y*(1-y/10)", "0", "0.5", "10"),
            ("Damped oscillation", "-0.5*y + sin(x)", "0", "0", "20"),
        ]
        for name, expr, x0, y0, xend in presets:
            btn = QPushButton(name)
            btn.clicked.connect(
                lambda checked, e=expr, a=x0, b=y0, c=xend:
                self._load_preset(e, a, b, c))
            preset_layout.addWidget(btn)
        layout.addWidget(preset_group)

        layout.addStretch()

    def _load_preset(self, expr, x0, y0, x_end):
        self.ode_input.setText(expr)
        self.x0_input.setText(x0)
        self.y0_input.setText(y0)
        self.x_end_input.setText(x_end)

    def _get_ode_func(self):
        """Parse the ODE expression and return a callable f(x, y)."""
        x_sym, y_sym = symbols('x y')
        expr_str = self.ode_input.text().strip()
        if not expr_str:
            raise ValueError("Please enter an ODE expression f(x, y).")
        expr = _parse(expr_str)
        f_lambda = sympy.lambdify((x_sym, y_sym), expr, modules=["numpy"])
        return expr, f_lambda

    def _solve(self):
        try:
            expr, f_lambda = self._get_ode_func()
            x0 = float(self.x0_input.text())
            y0 = float(self.y0_input.text())
            x_end = float(self.x_end_input.text())
            n_pts = self.n_points_spin.value()
            method = self.method_combo.currentText()

            def ode_rhs(t, y_vec):
                return [f_lambda(t, y_vec[0])]

            t_eval = np.linspace(x0, x_end, n_pts)
            sol = solve_ivp(ode_rhs, (x0, x_end), [y0], t_eval=t_eval, method=method)

            if sol.success:
                lines = [
                    f"ODE: dy/dx = {expr}",
                    f"Method: {method}",
                    f"Initial condition: y({x0}) = {y0}",
                    f"Interval: [{x0}, {x_end}]",
                    f"Points evaluated: {len(sol.t)}",
                    f"y(x_end) = {sol.y[0, -1]:.10g}",
                    f"Min y = {sol.y[0].min():.6g}, Max y = {sol.y[0].max():.6g}",
                ]
                self.result_display.set_result("ODE Solution", "\n".join(lines))
            else:
                self.result_display.set_result("ODE Solver Failed", sol.message)
        except Exception as e:
            self.result_display.set_result("Error", str(e))

    def _solve_and_plot(self):
        try:
            expr, f_lambda = self._get_ode_func()
            x0 = float(self.x0_input.text())
            y0 = float(self.y0_input.text())
            x_end = float(self.x_end_input.text())
            n_pts = self.n_points_spin.value()
            method = self.method_combo.currentText()

            def ode_rhs(t, y_vec):
                return [f_lambda(t, y_vec[0])]

            t_eval = np.linspace(x0, x_end, n_pts)
            sol = solve_ivp(ode_rhs, (x0, x_end), [y0], t_eval=t_eval, method=method)

            if sol.success:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.plot(sol.t, sol.y[0], 'b-', lw=1.5, label=f"y(x), method={method}")
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_title(f"ODE Solution: dy/dx = {expr}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()

                self.result_display.set_result("ODE Solution (plotted)",
                    f"dy/dx = {expr}\ny({x0}) = {y0}\nMethod: {method}\n"
                    f"y({x_end}) = {sol.y[0, -1]:.10g}")
            else:
                self.result_display.set_result("ODE Solver Failed", sol.message)
        except Exception as e:
            self.result_display.set_result("Error", str(e))


# ---------------------------------------------------------------------------
# Optimization Tab
# ---------------------------------------------------------------------------

class OptimizationTab(QWidget):
    """Minimize/maximize functions using scipy.optimize.minimize."""

    def __init__(self, result_display: ResultDisplay, parent=None):
        super().__init__(parent)
        self.result_display = result_display
        self._last_result = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        info = QLabel("Minimize f(x, y) or f(x) using scipy.optimize.minimize")
        info.setWordWrap(True)
        layout.addWidget(info)

        form = QFormLayout()

        self.func_input = QLineEdit()
        self.func_input.setPlaceholderText("e.g.  (x-1)**2 + (y-2.5)**2  or  x**4 - 3*x**2 + x")
        form.addRow("f =", self.func_input)

        self.vars_input = QLineEdit("x, y")
        self.vars_input.setPlaceholderText("x  or  x, y")
        form.addRow("Variables:", self.vars_input)

        self.x0_input = QLineEdit("0, 0")
        self.x0_input.setPlaceholderText("Initial guess (comma-separated)")
        form.addRow("x0:", self.x0_input)

        self.method_combo = QComboBox()
        self.method_combo.addItems(["Nelder-Mead", "BFGS", "L-BFGS-B", "Powell",
                                     "CG", "TNC", "COBYLA"])
        form.addRow("Method:", self.method_combo)

        self.bounds_input = QLineEdit()
        self.bounds_input.setPlaceholderText("Optional: (-5,5), (-5,5)")
        form.addRow("Bounds:", self.bounds_input)

        self.maximize_cb = QCheckBox("Maximize (negate function)")
        form.addRow("", self.maximize_cb)

        layout.addLayout(form)

        btn_row = QHBoxLayout()
        opt_btn = QPushButton("Optimize")
        opt_btn.clicked.connect(self._optimize)
        btn_row.addWidget(opt_btn)

        plot_btn = QPushButton("Optimize and Contour Plot")
        plot_btn.clicked.connect(self._optimize_and_plot)
        btn_row.addWidget(plot_btn)
        layout.addLayout(btn_row)

        # Presets
        preset_group = QGroupBox("Presets")
        preset_layout = QHBoxLayout(preset_group)
        presets = [
            ("Rosenbrock", "(1-x)**2 + 100*(y-x**2)**2", "x, y", "-1, -1"),
            ("Himmelblau", "(x**2+y-11)**2 + (x+y**2-7)**2", "x, y", "0, 0"),
            ("Booth", "(x+2*y-7)**2 + (2*x+y-5)**2", "x, y", "0, 0"),
        ]
        for name, expr, vars_str, x0_str in presets:
            btn = QPushButton(name)
            btn.clicked.connect(
                lambda checked, e=expr, v=vars_str, g=x0_str:
                self._load_preset(e, v, g))
            preset_layout.addWidget(btn)
        layout.addWidget(preset_group)

        layout.addStretch()

    def _load_preset(self, expr, vars_str, x0_str):
        self.func_input.setText(expr)
        self.vars_input.setText(vars_str)
        self.x0_input.setText(x0_str)

    def _parse_bounds(self):
        text = self.bounds_input.text().strip()
        if not text:
            return None
        # Parse bounds like (-5,5), (-5,5)
        bounds = []
        parts = text.split("),")
        for p in parts:
            p = p.strip().strip("()")
            lo, hi = p.split(",")
            lo = float(lo.strip()) if lo.strip() else None
            hi = float(hi.strip()) if hi.strip() else None
            bounds.append((lo, hi))
        return bounds

    def _optimize(self):
        try:
            var_names = [v.strip() for v in self.vars_input.text().split(",") if v.strip()]
            sym_vars = symbols(var_names)
            if not isinstance(sym_vars, tuple):
                sym_vars = (sym_vars,)

            expr = _parse(self.func_input.text().strip())
            sign = -1 if self.maximize_cb.isChecked() else 1
            f_lambda = sympy.lambdify(sym_vars, sign * expr, modules=["numpy"])

            x0_vals = [float(v.strip()) for v in self.x0_input.text().split(",")]
            method = self.method_combo.currentText()
            bounds = self._parse_bounds()

            def objective(x_vec):
                return float(f_lambda(*x_vec))

            result = sp_minimize(objective, x0_vals, method=method, bounds=bounds)
            self._last_result = result

            opt_type = "Maximum" if self.maximize_cb.isChecked() else "Minimum"
            lines = [
                f"Objective: f = {expr}",
                f"Method: {method}",
                f"{'Maximize' if self.maximize_cb.isChecked() else 'Minimize'}",
                f"",
                f"{opt_type} found at:",
            ]
            for name, val in zip(var_names, result.x):
                lines.append(f"  {name} = {val:.10g}")
            lines.append(f"")
            lines.append(f"f* = {sign * result.fun:.10g}")
            lines.append(f"Converged: {'Yes' if result.success else 'No'}")
            lines.append(f"Iterations: {result.get('nit', 'N/A')}")
            lines.append(f"Function evaluations: {result.get('nfev', 'N/A')}")
            lines.append(f"Message: {result.message}")

            self.result_display.set_result(f"Optimization ({opt_type})", "\n".join(lines))
        except Exception as e:
            self.result_display.set_result("Error", str(e))

    def _optimize_and_plot(self):
        try:
            var_names = [v.strip() for v in self.vars_input.text().split(",") if v.strip()]
            if len(var_names) != 2:
                self.result_display.set_result("Error",
                    "Contour plot requires exactly 2 variables.")
                return

            sym_vars = symbols(var_names)
            expr = _parse(self.func_input.text().strip())
            sign = -1 if self.maximize_cb.isChecked() else 1
            f_lambda = sympy.lambdify(sym_vars, sign * expr, modules=["numpy"])

            x0_vals = [float(v.strip()) for v in self.x0_input.text().split(",")]
            method = self.method_combo.currentText()
            bounds = self._parse_bounds()

            # Track path
            path = [np.array(x0_vals)]

            def objective(x_vec):
                return float(f_lambda(*x_vec))

            def callback(xk):
                path.append(np.array(xk))

            result = sp_minimize(objective, x0_vals, method=method,
                                 bounds=bounds, callback=callback)
            path.append(result.x)

            # Create contour plot
            import matplotlib.pyplot as plt

            # Determine plot range around the optimum
            all_pts = np.array(path)
            x_center = result.x[0]
            y_center = result.x[1]
            span = max(abs(all_pts[:, 0].max() - all_pts[:, 0].min()),
                       abs(all_pts[:, 1].max() - all_pts[:, 1].min()), 2.0) * 1.5

            xr = np.linspace(x_center - span, x_center + span, 200)
            yr = np.linspace(y_center - span, y_center + span, 200)
            X, Y = np.meshgrid(xr, yr)

            f_plot = sympy.lambdify(sym_vars, expr, modules=["numpy"])
            Z = f_plot(X, Y)

            fig, ax = plt.subplots(figsize=(8, 6))
            contour = ax.contourf(X, Y, Z, levels=40, cmap='viridis', alpha=0.8)
            ax.contour(X, Y, Z, levels=40, colors='white', linewidths=0.3, alpha=0.3)
            plt.colorbar(contour, ax=ax, label='f(x, y)')

            # Plot optimization path
            path_arr = np.array(path)
            ax.plot(path_arr[:, 0], path_arr[:, 1], 'r.-', markersize=4,
                    linewidth=1.5, label='optimization path', zorder=5)
            ax.plot(path_arr[0, 0], path_arr[0, 1], 'ws', markersize=10,
                    label='start', zorder=6)
            ax.plot(result.x[0], result.x[1], 'r*', markersize=15,
                    label='optimum', zorder=6)

            ax.set_xlabel(var_names[0])
            ax.set_ylabel(var_names[1])
            opt_type = "max" if self.maximize_cb.isChecked() else "min"
            ax.set_title(f"Optimization: {opt_type} of f = {expr}\n"
                         f"Result: ({result.x[0]:.4f}, {result.x[1]:.4f}), "
                         f"f* = {sign * result.fun:.6g}")
            ax.legend()
            plt.tight_layout()
            plt.show()

            self.result_display.set_result("Optimization (plotted)",
                f"f = {expr}\n{opt_type} at ({result.x[0]:.6g}, {result.x[1]:.6g})\n"
                f"f* = {sign * result.fun:.10g}")
        except Exception as e:
            self.result_display.set_result("Error", str(e))


# ---------------------------------------------------------------------------
# Number Theory Tab
# ---------------------------------------------------------------------------

class NumberTheoryTab(QWidget):
    """Number theory tools: prime factorization, GCD/LCM, modular arithmetic."""

    def __init__(self, result_display: ResultDisplay, parent=None):
        super().__init__(parent)
        self.result_display = result_display
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Prime factorization
        pf_group = QGroupBox("Prime Factorization")
        pf_layout = QHBoxLayout(pf_group)
        pf_layout.addWidget(QLabel("n ="))
        self.pf_input = QLineEdit()
        self.pf_input.setPlaceholderText("e.g. 360")
        pf_layout.addWidget(self.pf_input)
        pf_btn = QPushButton("Factorize")
        pf_btn.clicked.connect(self._factorize)
        pf_layout.addWidget(pf_btn)
        layout.addWidget(pf_group)

        # GCD / LCM
        gl_group = QGroupBox("GCD / LCM")
        gl_layout = QHBoxLayout(gl_group)
        gl_layout.addWidget(QLabel("a ="))
        self.gcd_a = QLineEdit()
        self.gcd_a.setPlaceholderText("e.g. 48")
        gl_layout.addWidget(self.gcd_a)
        gl_layout.addWidget(QLabel("b ="))
        self.gcd_b = QLineEdit()
        self.gcd_b.setPlaceholderText("e.g. 36")
        gl_layout.addWidget(self.gcd_b)
        gcd_btn = QPushButton("GCD")
        gcd_btn.clicked.connect(self._gcd)
        gl_layout.addWidget(gcd_btn)
        lcm_btn = QPushButton("LCM")
        lcm_btn.clicked.connect(self._lcm)
        gl_layout.addWidget(lcm_btn)
        layout.addWidget(gl_group)

        # Primality test
        pt_group = QGroupBox("Primality Test")
        pt_layout = QHBoxLayout(pt_group)
        pt_layout.addWidget(QLabel("n ="))
        self.prime_input = QLineEdit()
        self.prime_input.setPlaceholderText("e.g. 104729")
        pt_layout.addWidget(self.prime_input)
        prime_btn = QPushButton("Test Primality")
        prime_btn.clicked.connect(self._test_prime)
        pt_layout.addWidget(prime_btn)
        next_prime_btn = QPushButton("Next Prime")
        next_prime_btn.clicked.connect(self._next_prime)
        pt_layout.addWidget(next_prime_btn)
        layout.addWidget(pt_group)

        # Modular arithmetic
        mod_group = QGroupBox("Modular Arithmetic")
        mod_form = QFormLayout(mod_group)

        mod_row1 = QHBoxLayout()
        mod_row1.addWidget(QLabel("a ="))
        self.mod_a = QLineEdit()
        self.mod_a.setPlaceholderText("e.g. 7")
        mod_row1.addWidget(self.mod_a)
        mod_row1.addWidget(QLabel("b ="))
        self.mod_b = QLineEdit()
        self.mod_b.setPlaceholderText("e.g. 3")
        mod_row1.addWidget(self.mod_b)
        mod_row1.addWidget(QLabel("mod m ="))
        self.mod_m = QLineEdit()
        self.mod_m.setPlaceholderText("e.g. 11")
        mod_row1.addWidget(self.mod_m)
        mod_form.addRow(mod_row1)

        mod_btn_row = QHBoxLayout()
        for label, slot in [
            ("a^b mod m", self._mod_pow),
            ("a^(-1) mod m", self._mod_inv),
            ("(a+b) mod m", self._mod_add),
            ("(a*b) mod m", self._mod_mul),
            ("Euler totient(m)", self._euler_totient),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            mod_btn_row.addWidget(btn)
        mod_form.addRow(mod_btn_row)

        layout.addWidget(mod_group)
        layout.addStretch()

    def _factorize(self):
        try:
            n = int(self.pf_input.text().strip())
            if n < 2:
                self.result_display.set_result("Error", "Enter an integer >= 2.")
                return
            factors = factorint(n)
            factor_str = " * ".join(
                f"{p}^{e}" if e > 1 else str(p)
                for p, e in sorted(factors.items())
            )
            lines = [
                f"n = {n}",
                f"Prime factorization: {n} = {factor_str}",
                f"Number of distinct prime factors: {len(factors)}",
                f"Total number of divisors: {sympy.divisor_count(n)}",
                f"Sum of divisors: {sympy.divisor_sigma(n)}",
                f"Euler totient: {totient(n)}",
            ]
            self.result_display.set_result("Prime Factorization", "\n".join(lines))
        except Exception as e:
            self.result_display.set_result("Error", str(e))

    def _gcd(self):
        try:
            a = int(self.gcd_a.text().strip())
            b = int(self.gcd_b.text().strip())
            g = int(gcd(a, b))
            self.result_display.set_result("GCD", f"gcd({a}, {b}) = {g}")
        except Exception as e:
            self.result_display.set_result("Error", str(e))

    def _lcm(self):
        try:
            a = int(self.gcd_a.text().strip())
            b = int(self.gcd_b.text().strip())
            l = int(lcm(a, b))
            self.result_display.set_result("LCM", f"lcm({a}, {b}) = {l}")
        except Exception as e:
            self.result_display.set_result("Error", str(e))

    def _test_prime(self):
        try:
            n = int(self.prime_input.text().strip())
            is_p = isprime(n)
            lines = [f"n = {n}", f"Is prime: {'Yes' if is_p else 'No'}"]
            if not is_p and n > 1:
                factors = factorint(n)
                factor_str = " * ".join(
                    f"{p}^{e}" if e > 1 else str(p)
                    for p, e in sorted(factors.items())
                )
                lines.append(f"Factorization: {factor_str}")
            self.result_display.set_result("Primality Test", "\n".join(lines))
        except Exception as e:
            self.result_display.set_result("Error", str(e))

    def _next_prime(self):
        try:
            n = int(self.prime_input.text().strip())
            np_ = nextprime(n)
            self.result_display.set_result("Next Prime",
                f"The next prime after {n} is {np_}")
        except Exception as e:
            self.result_display.set_result("Error", str(e))

    def _mod_pow(self):
        try:
            a = int(self.mod_a.text().strip())
            b = int(self.mod_b.text().strip())
            m = int(self.mod_m.text().strip())
            result = pow(a, b, m)
            self.result_display.set_result("Modular Exponentiation",
                f"{a}^{b} mod {m} = {result}")
        except Exception as e:
            self.result_display.set_result("Error", str(e))

    def _mod_inv(self):
        try:
            a = int(self.mod_a.text().strip())
            m = int(self.mod_m.text().strip())
            result = int(mod_inverse(a, m))
            self.result_display.set_result("Modular Inverse",
                f"{a}^(-1) mod {m} = {result}\n"
                f"Verification: {a} * {result} mod {m} = {(a * result) % m}")
        except Exception as e:
            self.result_display.set_result("Error", str(e))

    def _mod_add(self):
        try:
            a = int(self.mod_a.text().strip())
            b = int(self.mod_b.text().strip())
            m = int(self.mod_m.text().strip())
            result = (a + b) % m
            self.result_display.set_result("Modular Addition",
                f"({a} + {b}) mod {m} = {result}")
        except Exception as e:
            self.result_display.set_result("Error", str(e))

    def _mod_mul(self):
        try:
            a = int(self.mod_a.text().strip())
            b = int(self.mod_b.text().strip())
            m = int(self.mod_m.text().strip())
            result = (a * b) % m
            self.result_display.set_result("Modular Multiplication",
                f"({a} * {b}) mod {m} = {result}")
        except Exception as e:
            self.result_display.set_result("Error", str(e))

    def _euler_totient(self):
        try:
            m = int(self.mod_m.text().strip())
            phi = int(totient(m))
            self.result_display.set_result("Euler Totient",
                f"phi({m}) = {phi}")
        except Exception as e:
            self.result_display.set_result("Error", str(e))


# ---------------------------------------------------------------------------
# Main Widget
# ---------------------------------------------------------------------------

class MathEngineWidget(QWidget):
    """Top-level Math Engine widget with tabbed interface.

    Provides:
      - Expression calculator
      - Matrix operations (det, inv, eigen, SVD, ...)
      - Equation solver (polynomial, symbolic, nonlinear)
      - Symbolic math (derivatives, integrals, limits, series)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._logger = None
        self._build_ui()

    # -- Public API ---------------------------------------------------------

    def set_logger(self, fn):
        """Set a logging callback ``fn(message: str)``."""
        self._logger = fn

    def run(self):
        """Activate the widget (refresh / reinitialise if needed)."""
        self._log("MathEngine: run() called")
        self.result_display.set_result("Math Engine", "Ready.  Select a tab and enter data.")

    # -- Internal -----------------------------------------------------------

    def _log(self, msg: str):
        if self._logger:
            try:
                self._logger(msg)
            except Exception:
                pass

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # Header
        header = QLabel("Math Engine")
        header.setFont(QFont("Segoe UI", 14, QFont.Bold))
        header.setStyleSheet("color: #cba6f7; padding: 4px;")
        main_layout.addWidget(header)

        splitter = QSplitter(Qt.Vertical)

        # Tab widget (top part)
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            "QTabWidget::pane { border: 1px solid #45475a; border-radius: 4px; }"
            "QTabBar::tab { padding: 6px 16px; }"
            "QTabBar::tab:selected { background: #45475a; color: #cdd6f4; }"
        )

        # Result display (bottom part)
        self.result_display = ResultDisplay()

        # Build tabs
        self.calc_tab = CalculatorTab(self.result_display)
        self.matrix_tab = MatrixTab(self.result_display)
        self.eq_tab = EquationSolverTab(self.result_display)
        self.sym_tab = SymbolicMathTab(self.result_display)
        self.ode_tab = ODESolverTab(self.result_display)
        self.opt_tab = OptimizationTab(self.result_display)
        self.nt_tab = NumberTheoryTab(self.result_display)

        self.tabs.addTab(self.calc_tab, "Calculator")
        self.tabs.addTab(self.matrix_tab, "Matrix Operations")
        self.tabs.addTab(self.eq_tab, "Equation Solver")
        self.tabs.addTab(self.sym_tab, "Symbolic Math")
        self.tabs.addTab(self.ode_tab, "ODE Solver")
        self.tabs.addTab(self.opt_tab, "Optimization")
        self.tabs.addTab(self.nt_tab, "Number Theory")

        splitter.addWidget(self.tabs)
        splitter.addWidget(self.result_display)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        main_layout.addWidget(splitter)

        # Button row: Generate LaTeX + Export
        btn_row = QHBoxLayout()

        latex_btn = QPushButton("Generate LaTeX")
        latex_btn.setToolTip("Convert the current result to LaTeX markup")
        latex_btn.clicked.connect(self._generate_latex)
        btn_row.addWidget(latex_btn)

        export_csv_btn = QPushButton("Export CSV")
        export_csv_btn.setToolTip("Export current result as CSV")
        export_csv_btn.clicked.connect(self._export_csv)
        btn_row.addWidget(export_csv_btn)

        export_latex_btn = QPushButton("Export LaTeX File")
        export_latex_btn.setToolTip("Export current result as a .tex file")
        export_latex_btn.clicked.connect(self._export_latex_file)
        btn_row.addWidget(export_latex_btn)

        export_html_btn = QPushButton("Export HTML")
        export_html_btn.setToolTip("Export current result as HTML")
        export_html_btn.clicked.connect(self._export_html)
        btn_row.addWidget(export_html_btn)

        main_layout.addLayout(btn_row)

        # Status bar
        self.status = QLabel("Ready")
        self.status.setStyleSheet("color: #a6adc8; font-size: 11px; padding: 2px;")
        main_layout.addWidget(self.status)

    # -- LaTeX Generation ---------------------------------------------------

    def _generate_latex(self):
        """Convert the current computation to LaTeX markup."""
        title, body = self.result_display.get_last_result()
        if not title or title == "Math Engine":
            self.result_display.set_result("LaTeX", "No computation to convert. "
                                           "Run a calculation first.")
            return

        # Try to parse mathematical expressions from the result body for LaTeX
        latex_lines = [r"\section*{" + title + "}"]

        # Attempt to extract and convert SymPy expressions
        current_tab = self.tabs.currentWidget()
        latex_content = None

        if isinstance(current_tab, CalculatorTab):
            try:
                expr = _parse(current_tab.expr_input.text().strip())
                result = sympy.simplify(expr)
                latex_content = (
                    f"Expression:\n$$\n{latex(expr)}\n$$\n\n"
                    f"Result:\n$$\n{latex(result)}\n$$"
                )
            except Exception:
                pass
        elif isinstance(current_tab, SymbolicMathTab):
            try:
                expr = _parse(current_tab.expr_input.text().strip())
                latex_content = f"$$\n{latex(expr)}\n$$"
            except Exception:
                pass
        elif isinstance(current_tab, EquationSolverTab):
            try:
                text = current_tab.sym_eq_input.toPlainText().strip()
                if text:
                    lines_raw = [l.strip() for l in text.splitlines() if l.strip()]
                    latex_eqs = []
                    for line in lines_raw:
                        if "=" in line:
                            lhs, rhs = line.split("=", 1)
                            latex_eqs.append(f"{latex(_parse(lhs))} = {latex(_parse(rhs))}")
                        else:
                            latex_eqs.append(latex(_parse(line)) + " = 0")
                    latex_content = "Equations:\n\\begin{align}\n"
                    latex_content += " \\\\\n".join(latex_eqs)
                    latex_content += "\n\\end{align}"
            except Exception:
                pass
        elif isinstance(current_tab, MatrixTab):
            try:
                A = current_tab._get_a()
                sym_A = SympyMatrix(A.tolist())
                latex_content = f"Matrix A:\n$$\n{latex(sym_A)}\n$$"
            except Exception:
                pass
        elif isinstance(current_tab, ODESolverTab):
            try:
                expr = _parse(current_tab.ode_input.text().strip())
                x0 = current_tab.x0_input.text().strip()
                y0 = current_tab.y0_input.text().strip()
                latex_content = (
                    f"ODE:\n$$\n\\frac{{dy}}{{dx}} = {latex(expr)}\n$$\n\n"
                    f"Initial condition: $y({x0}) = {y0}$"
                )
            except Exception:
                pass
        elif isinstance(current_tab, OptimizationTab):
            try:
                expr = _parse(current_tab.func_input.text().strip())
                var_names = [v.strip() for v in current_tab.vars_input.text().split(",")]
                op = r"\max" if current_tab.maximize_cb.isChecked() else r"\min"
                vars_str = ", ".join(var_names)
                latex_content = (
                    f"Optimization problem:\n"
                    f"$$\n{op}_{{{vars_str}}} \\; {latex(expr)}\n$$"
                )
            except Exception:
                pass

        if latex_content:
            latex_lines.append(latex_content)
        else:
            # Fall back to verbatim
            latex_lines.append("\\begin{verbatim}")
            latex_lines.append(body)
            latex_lines.append("\\end{verbatim}")

        full_latex = "\n".join(latex_lines)
        self.result_display.set_result("LaTeX Output", full_latex)
        self.status.setText("LaTeX generated - copy from result display")

    # -- Export Methods ------------------------------------------------------

    def _export_csv(self):
        """Export the current result as CSV."""
        title, body = self.result_display.get_last_result()
        if not body:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export as CSV", "result.csv",
            "CSV Files (*.csv);;All Files (*)")
        if not path:
            return

        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Title", title])
                writer.writerow([])
                for line in body.split("\n"):
                    # Try to parse numeric data
                    parts = line.strip().split()
                    writer.writerow(parts if parts else [line])
            self.status.setText(f"Exported CSV: {path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))

    def _export_latex_file(self):
        """Export the current result as a standalone .tex file."""
        title, body = self.result_display.get_last_result()
        if not body:
            return

        # First generate the LaTeX content
        self._generate_latex()
        _, latex_body = self.result_display.get_last_result()

        path, _ = QFileDialog.getSaveFileName(
            self, "Export as LaTeX", "result.tex",
            "LaTeX Files (*.tex);;All Files (*)")
        if not path:
            return

        try:
            tex_doc = (
                "\\documentclass{article}\n"
                "\\usepackage{amsmath, amssymb, amsfonts}\n"
                "\\usepackage[margin=1in]{geometry}\n"
                "\\title{QuantumRes Math Engine Result}\n"
                "\\date{\\today}\n"
                "\\begin{document}\n"
                "\\maketitle\n\n"
                f"{latex_body}\n\n"
                "\\end{document}\n"
            )
            with open(path, 'w', encoding='utf-8') as f:
                f.write(tex_doc)
            self.status.setText(f"Exported LaTeX: {path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))

    def _export_html(self):
        """Export the current result as HTML."""
        title, body = self.result_display.get_last_result()
        if not body:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export as HTML", "result.html",
            "HTML Files (*.html);;All Files (*)")
        if not path:
            return

        try:
            html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>{title} - QuantumRes</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; margin: 2em; background: #fafafa; }}
h1 {{ color: #2c3e50; border-bottom: 2px solid #9b59b6; padding-bottom: 8px; }}
pre {{ background: #1e1e2e; color: #cdd6f4; padding: 16px; border-radius: 6px;
       overflow-x: auto; font-size: 14px; line-height: 1.5; }}
.footer {{ color: #888; font-size: 0.85em; margin-top: 2em; }}
</style>
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head><body>
<h1>{title}</h1>
<pre>{body}</pre>
<div class="footer">Generated by QuantumRes Math Engine</div>
</body></html>"""
            with open(path, 'w', encoding='utf-8') as f:
                f.write(html)
            self.status.setText(f"Exported HTML: {path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))


# ---------------------------------------------------------------------------
# Standalone entry point (for testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MathEngineWidget()
    win.setWindowTitle("Math Engine - Standalone")
    win.resize(900, 700)
    win.show()
    win.run()
    sys.exit(app.exec_())
