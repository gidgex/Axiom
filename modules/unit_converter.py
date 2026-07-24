"""
Universal Unit Converter Widget for QuantumRes.
Supports 16 categories with bidirectional real-time conversion,
favorites, scientific notation, and comprehensive unit coverage.
"""

import json
import os
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QLabel, QLineEdit, QComboBox, QListWidget, QListWidgetItem,
    QGroupBox, QSplitter, QFrame, QMessageBox, QApplication,
    QDialog, QFormLayout, QDialogButtonBox, QTextEdit, QScrollArea
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QDoubleValidator


# ── Conversion Data ─────────────────────────────────────────────────────────
# Each category maps unit names to a factor relative to a base unit (multiply to convert TO base).
# Temperature is handled specially.

UNIT_DATA = {
    "Length": {
        "_base": "meter",
        "meter (m)": 1.0,
        "kilometer (km)": 1e3,
        "centimeter (cm)": 1e-2,
        "millimeter (mm)": 1e-3,
        "micrometer (\u00b5m)": 1e-6,
        "nanometer (nm)": 1e-9,
        "picometer (pm)": 1e-12,
        "angstrom (\u00c5)": 1e-10,
        "inch (in)": 0.0254,
        "foot (ft)": 0.3048,
        "yard (yd)": 0.9144,
        "mile (mi)": 1609.344,
        "nautical mile (nmi)": 1852.0,
        "astronomical unit (AU)": 1.495978707e11,
        "light-year (ly)": 9.4607304725808e15,
        "parsec (pc)": 3.0856775814913673e16,
        "Bohr radius (a\u2080)": 5.29177210903e-11,
    },
    "Mass": {
        "_base": "kilogram",
        "kilogram (kg)": 1.0,
        "gram (g)": 1e-3,
        "milligram (mg)": 1e-6,
        "microgram (\u00b5g)": 1e-9,
        "metric ton (t)": 1e3,
        "pound (lb)": 0.45359237,
        "ounce (oz)": 0.028349523125,
        "stone (st)": 6.35029318,
        "slug": 14.593903,
        "atomic mass unit (u)": 1.66053906660e-27,
        "electron mass (m_e)": 9.1093837015e-31,
        "proton mass (m_p)": 1.67262192369e-27,
        "solar mass (M\u2609)": 1.989e30,
    },
    "Time": {
        "_base": "second",
        "second (s)": 1.0,
        "millisecond (ms)": 1e-3,
        "microsecond (\u00b5s)": 1e-6,
        "nanosecond (ns)": 1e-9,
        "picosecond (ps)": 1e-12,
        "femtosecond (fs)": 1e-15,
        "minute (min)": 60.0,
        "hour (h)": 3600.0,
        "day (d)": 86400.0,
        "week": 604800.0,
        "year (Julian)": 31557600.0,
        "Planck time (t_P)": 5.391247e-44,
    },
    "Temperature": {
        "_base": "special",
        "Kelvin (K)": "K",
        "Celsius (\u00b0C)": "C",
        "Fahrenheit (\u00b0F)": "F",
        "Rankine (\u00b0R)": "R",
    },
    "Force": {
        "_base": "newton",
        "newton (N)": 1.0,
        "kilonewton (kN)": 1e3,
        "dyne (dyn)": 1e-5,
        "pound-force (lbf)": 4.4482216152605,
        "kilogram-force (kgf)": 9.80665,
        "poundal": 0.138254954376,
    },
    "Energy": {
        "_base": "joule",
        "joule (J)": 1.0,
        "kilojoule (kJ)": 1e3,
        "megajoule (MJ)": 1e6,
        "calorie (cal)": 4.184,
        "kilocalorie (kcal)": 4184.0,
        "electronvolt (eV)": 1.602176634e-19,
        "keV": 1.602176634e-16,
        "MeV": 1.602176634e-13,
        "GeV": 1.602176634e-10,
        "erg": 1e-7,
        "watt-hour (Wh)": 3600.0,
        "kilowatt-hour (kWh)": 3.6e6,
        "BTU": 1055.06,
        "hartree (E_h)": 4.3597447222071e-18,
        "Rydberg (Ry)": 2.1798723611e-18,
    },
    "Power": {
        "_base": "watt",
        "watt (W)": 1.0,
        "kilowatt (kW)": 1e3,
        "megawatt (MW)": 1e6,
        "gigawatt (GW)": 1e9,
        "milliwatt (mW)": 1e-3,
        "horsepower (hp)": 745.69987158,
        "BTU/hour": 0.29307107,
        "calorie/second": 4.184,
        "erg/second": 1e-7,
    },
    "Pressure": {
        "_base": "pascal",
        "pascal (Pa)": 1.0,
        "kilopascal (kPa)": 1e3,
        "megapascal (MPa)": 1e6,
        "gigapascal (GPa)": 1e9,
        "bar": 1e5,
        "millibar (mbar)": 100.0,
        "atmosphere (atm)": 101325.0,
        "torr (mmHg)": 133.322,
        "psi": 6894.757,
        "dyne/cm\u00b2": 0.1,
    },
    "Frequency": {
        "_base": "hertz",
        "hertz (Hz)": 1.0,
        "kilohertz (kHz)": 1e3,
        "megahertz (MHz)": 1e6,
        "gigahertz (GHz)": 1e9,
        "terahertz (THz)": 1e12,
        "rpm": 1.0 / 60.0,
        "radian/second": 1.0 / (2.0 * np.pi),
    },
    "Electric Current": {
        "_base": "ampere",
        "ampere (A)": 1.0,
        "milliampere (mA)": 1e-3,
        "microampere (\u00b5A)": 1e-6,
        "nanoampere (nA)": 1e-9,
        "kiloampere (kA)": 1e3,
    },
    "Voltage": {
        "_base": "volt",
        "volt (V)": 1.0,
        "millivolt (mV)": 1e-3,
        "microvolt (\u00b5V)": 1e-6,
        "kilovolt (kV)": 1e3,
        "megavolt (MV)": 1e6,
    },
    "Speed": {
        "_base": "m/s",
        "meter/second (m/s)": 1.0,
        "kilometer/hour (km/h)": 1.0 / 3.6,
        "mile/hour (mph)": 0.44704,
        "knot (kn)": 0.514444,
        "foot/second (ft/s)": 0.3048,
        "speed of light (c)": 2.99792458e8,
        "mach (sea level)": 343.0,
    },
    "Area": {
        "_base": "m\u00b2",
        "square meter (m\u00b2)": 1.0,
        "square kilometer (km\u00b2)": 1e6,
        "square centimeter (cm\u00b2)": 1e-4,
        "square millimeter (mm\u00b2)": 1e-6,
        "hectare (ha)": 1e4,
        "acre": 4046.8564224,
        "square foot (ft\u00b2)": 0.09290304,
        "square inch (in\u00b2)": 6.4516e-4,
        "square mile (mi\u00b2)": 2.589988110336e6,
        "barn (b)": 1e-28,
    },
    "Volume": {
        "_base": "m\u00b3",
        "cubic meter (m\u00b3)": 1.0,
        "liter (L)": 1e-3,
        "milliliter (mL)": 1e-6,
        "cubic centimeter (cm\u00b3)": 1e-6,
        "gallon (US)": 3.785411784e-3,
        "quart (US)": 9.46352946e-4,
        "pint (US)": 4.73176473e-4,
        "cup (US)": 2.365882365e-4,
        "fluid ounce (US)": 2.95735295625e-5,
        "cubic foot (ft\u00b3)": 0.028316846592,
        "cubic inch (in\u00b3)": 1.6387064e-5,
        "barrel (oil)": 0.158987294928,
    },
    "Angle": {
        "_base": "radian",
        "radian (rad)": 1.0,
        "degree (\u00b0)": np.pi / 180.0,
        "arcminute (')": np.pi / 10800.0,
        "arcsecond (\")": np.pi / 648000.0,
        "gradian (grad)": np.pi / 200.0,
        "revolution (rev)": 2.0 * np.pi,
        "milliradian (mrad)": 1e-3,
    },
    "Data Storage": {
        "_base": "byte",
        "bit (b)": 0.125,
        "byte (B)": 1.0,
        "kilobyte (KB)": 1e3,
        "megabyte (MB)": 1e6,
        "gigabyte (GB)": 1e9,
        "terabyte (TB)": 1e12,
        "petabyte (PB)": 1e15,
        "kibibyte (KiB)": 1024.0,
        "mebibyte (MiB)": 1048576.0,
        "gibibyte (GiB)": 1073741824.0,
        "tebibyte (TiB)": 1099511627776.0,
    },
}


def _convert_temperature(value, from_unit, to_unit):
    """Convert temperature between K, C, F, R."""
    # Convert to Kelvin first
    if from_unit == "C":
        k = value + 273.15
    elif from_unit == "F":
        k = (value - 32) * 5.0 / 9.0 + 273.15
    elif from_unit == "R":
        k = value * 5.0 / 9.0
    else:
        k = value
    # Convert from Kelvin
    if to_unit == "C":
        return k - 273.15
    elif to_unit == "F":
        return (k - 273.15) * 9.0 / 5.0 + 32.0
    elif to_unit == "R":
        return k * 9.0 / 5.0
    return k


def convert(value, category, from_unit, to_unit):
    """Convert a value between units in a given category."""
    if category == "Temperature":
        fu = UNIT_DATA[category][from_unit]
        tu = UNIT_DATA[category][to_unit]
        return _convert_temperature(value, fu, tu)
    data = UNIT_DATA[category]
    from_factor = data[from_unit]
    to_factor = data[to_unit]
    base_value = value * from_factor
    return base_value / to_factor


def _format_result(value):
    """Format a numeric result, using scientific notation for extreme values."""
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) > 1e6:
        return f"{value:.10e}"
    if abs(value - round(value)) < 1e-12:
        return str(int(round(value)))
    return f"{value:.10g}"


class UnitConverterWidget(QWidget):
    """Universal unit converter with 16 categories."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._log = None
        self._favorites = []
        self._recent = []
        self._fav_path = os.path.join(os.path.dirname(__file__), ".unit_conv_favorites.json")
        self._load_favorites()
        self._init_ui()

    def set_logger(self, fn):
        """Set external logging function."""
        self._log = fn

    def _emit_log(self, msg):
        if self._log:
            self._log(msg)

    # ── UI ──────────────────────────────────────────────────────────────
    def _init_ui(self):
        main = QVBoxLayout(self)

        # Category selector
        cat_row = QHBoxLayout()
        cat_row.addWidget(QLabel("Category:"))
        self._cat_combo = QComboBox()
        self._cat_combo.addItems(sorted(UNIT_DATA.keys()))
        self._cat_combo.currentTextChanged.connect(self._on_category_changed)
        cat_row.addWidget(self._cat_combo)
        main.addLayout(cat_row)

        # Conversion area
        conv_group = QGroupBox("Conversion")
        conv_layout = QGridLayout(conv_group)

        # From
        conv_layout.addWidget(QLabel("From:"), 0, 0)
        self._from_combo = QComboBox()
        self._from_combo.currentTextChanged.connect(self._do_convert)
        conv_layout.addWidget(self._from_combo, 0, 1)

        self._from_input = QLineEdit("1")
        self._from_input.setFont(QFont("Consolas", 12))
        self._from_input.textChanged.connect(self._do_convert)
        conv_layout.addWidget(self._from_input, 0, 2)

        # Swap button
        swap_btn = QPushButton("\u21c4 Swap")
        swap_btn.clicked.connect(self._swap_units)
        conv_layout.addWidget(swap_btn, 0, 3)

        # To
        conv_layout.addWidget(QLabel("To:"), 1, 0)
        self._to_combo = QComboBox()
        self._to_combo.currentTextChanged.connect(self._do_convert)
        conv_layout.addWidget(self._to_combo, 1, 1)

        self._to_output = QLineEdit()
        self._to_output.setFont(QFont("Consolas", 12))
        self._to_output.setReadOnly(True)
        conv_layout.addWidget(self._to_output, 1, 2)

        copy_btn = QPushButton("Copy Result")
        copy_btn.clicked.connect(self._copy_result)
        conv_layout.addWidget(copy_btn, 1, 3)

        # Reverse input (bidirectional)
        self._to_input_btn = QPushButton("Enter result value \u2192")
        self._to_input_btn.clicked.connect(self._enable_reverse)
        conv_layout.addWidget(self._to_input_btn, 2, 0, 1, 2)

        # Formula display
        self._formula_label = QLabel("")
        self._formula_label.setStyleSheet("color: #666; font-style: italic;")
        conv_layout.addWidget(self._formula_label, 2, 2, 1, 2)

        main.addWidget(conv_group)

        # Favorites / Recent
        splitter = QSplitter(Qt.Horizontal)

        fav_group = QGroupBox("Favorites")
        fav_layout = QVBoxLayout(fav_group)
        self._fav_list = QListWidget()
        self._fav_list.itemDoubleClicked.connect(self._load_favorite)
        fav_layout.addWidget(self._fav_list)
        fav_btns = QHBoxLayout()
        add_fav = QPushButton("Add Current")
        add_fav.clicked.connect(self._add_favorite)
        fav_btns.addWidget(add_fav)
        rm_fav = QPushButton("Remove")
        rm_fav.clicked.connect(self._remove_favorite)
        fav_btns.addWidget(rm_fav)
        fav_layout.addLayout(fav_btns)
        splitter.addWidget(fav_group)

        recent_group = QGroupBox("Recent Conversions")
        recent_layout = QVBoxLayout(recent_group)
        self._recent_list = QListWidget()
        self._recent_list.itemDoubleClicked.connect(self._load_recent)
        recent_layout.addWidget(self._recent_list)
        splitter.addWidget(recent_group)

        main.addWidget(splitter)

        # Advanced features group
        adv_group = QGroupBox("Advanced Features")
        adv_lay = QVBoxLayout(adv_group)

        adv_row1 = QHBoxLayout()
        chain_btn = QPushButton("Conversion Chain")
        chain_btn.clicked.connect(self._show_conversion_chain)
        adv_row1.addWidget(chain_btn)

        dim_btn = QPushButton("Dimensional Analysis")
        dim_btn.clicked.connect(self._dimensional_analysis_dialog)
        adv_row1.addWidget(dim_btn)

        custom_btn = QPushButton("Custom Unit")
        custom_btn.clicked.connect(self._add_custom_unit_dialog)
        adv_row1.addWidget(custom_btn)
        adv_lay.addLayout(adv_row1)

        self._chain_display = QLabel("")
        self._chain_display.setWordWrap(True)
        self._chain_display.setStyleSheet("color: #444; font-style: italic; padding: 4px;")
        adv_lay.addWidget(self._chain_display)

        main.addWidget(adv_group)

        # Initialize combos
        self._on_category_changed(self._cat_combo.currentText())
        self._refresh_favorites()

    # ── Category Change ─────────────────────────────────────────────────
    def _on_category_changed(self, category):
        self._from_combo.blockSignals(True)
        self._to_combo.blockSignals(True)
        self._from_combo.clear()
        self._to_combo.clear()

        if category in UNIT_DATA:
            units = [u for u in UNIT_DATA[category] if u != "_base"]
            self._from_combo.addItems(units)
            self._to_combo.addItems(units)
            if len(units) > 1:
                self._to_combo.setCurrentIndex(1)

        self._from_combo.blockSignals(False)
        self._to_combo.blockSignals(False)
        self._do_convert()

    # ── Conversion ──────────────────────────────────────────────────────
    def _do_convert(self):
        category = self._cat_combo.currentText()
        from_unit = self._from_combo.currentText()
        to_unit = self._to_combo.currentText()
        text = self._from_input.text().strip()

        if not from_unit or not to_unit or not text:
            self._to_output.clear()
            return

        try:
            value = float(text)
        except ValueError:
            self._to_output.setText("Invalid input")
            return

        try:
            result = convert(value, category, from_unit, to_unit)
            self._to_output.setText(_format_result(result))
            self._formula_label.setText(f"{_format_result(value)} {from_unit} = {_format_result(result)} {to_unit}")

            # Add to recent
            desc = f"{_format_result(value)} {from_unit} \u2192 {_format_result(result)} {to_unit}"
            self._recent.insert(0, {
                "category": category, "from": from_unit, "to": to_unit,
                "value": value, "desc": desc
            })
            if len(self._recent) > 20:
                self._recent = self._recent[:20]
            self._refresh_recent()

        except Exception as e:
            self._to_output.setText(f"Error: {e}")

    def _swap_units(self):
        fi = self._from_combo.currentIndex()
        ti = self._to_combo.currentIndex()
        self._from_combo.setCurrentIndex(ti)
        self._to_combo.setCurrentIndex(fi)
        self._emit_log("Swapped units")

    def _enable_reverse(self):
        """Allow typing in the result field to do reverse conversion."""
        current_result = self._to_output.text().strip()
        if not current_result or current_result.startswith("Error") or current_result == "Invalid input":
            return
        # Swap and put the result value in input
        fi = self._from_combo.currentIndex()
        ti = self._to_combo.currentIndex()
        self._from_combo.setCurrentIndex(ti)
        self._to_combo.setCurrentIndex(fi)
        self._from_input.setText(current_result)
        self._emit_log("Reversed conversion direction")

    def _copy_result(self):
        text = self._to_output.text()
        if text and text != "Invalid input" and not text.startswith("Error"):
            QApplication.clipboard().setText(text)
            self._emit_log(f"Copied result: {text}")

    # ── Favorites ───────────────────────────────────────────────────────
    def _load_favorites(self):
        try:
            if os.path.exists(self._fav_path):
                with open(self._fav_path, "r") as f:
                    self._favorites = json.load(f)
        except Exception:
            self._favorites = []

    def _save_favorites(self):
        try:
            with open(self._fav_path, "w") as f:
                json.dump(self._favorites, f)
        except Exception:
            pass

    def _refresh_favorites(self):
        self._fav_list.clear()
        for fav in self._favorites:
            self._fav_list.addItem(fav.get("desc", "???"))

    def _refresh_recent(self):
        self._recent_list.clear()
        for r in self._recent:
            self._recent_list.addItem(r.get("desc", "???"))

    def _add_favorite(self):
        category = self._cat_combo.currentText()
        from_unit = self._from_combo.currentText()
        to_unit = self._to_combo.currentText()
        if not from_unit or not to_unit:
            return
        desc = f"[{category}] {from_unit} \u2194 {to_unit}"
        fav = {"category": category, "from": from_unit, "to": to_unit, "desc": desc}
        # Check duplicate
        for f in self._favorites:
            if f["category"] == category and f["from"] == from_unit and f["to"] == to_unit:
                return
        self._favorites.append(fav)
        self._save_favorites()
        self._refresh_favorites()
        self._emit_log(f"Added favorite: {desc}")

    def _remove_favorite(self):
        row = self._fav_list.currentRow()
        if 0 <= row < len(self._favorites):
            removed = self._favorites.pop(row)
            self._save_favorites()
            self._refresh_favorites()
            self._emit_log(f"Removed favorite: {removed.get('desc', '')}")

    def _load_favorite(self, item):
        row = self._fav_list.currentRow()
        if 0 <= row < len(self._favorites):
            fav = self._favorites[row]
            idx = self._cat_combo.findText(fav["category"])
            if idx >= 0:
                self._cat_combo.setCurrentIndex(idx)
                QTimer.singleShot(50, lambda: self._set_units(fav["from"], fav["to"]))

    def _load_recent(self, item):
        row = self._recent_list.currentRow()
        if 0 <= row < len(self._recent):
            r = self._recent[row]
            idx = self._cat_combo.findText(r["category"])
            if idx >= 0:
                self._cat_combo.setCurrentIndex(idx)
                QTimer.singleShot(50, lambda: self._set_units_and_value(
                    r["from"], r["to"], r["value"]))

    def _set_units(self, from_u, to_u):
        fi = self._from_combo.findText(from_u)
        ti = self._to_combo.findText(to_u)
        if fi >= 0:
            self._from_combo.setCurrentIndex(fi)
        if ti >= 0:
            self._to_combo.setCurrentIndex(ti)

    def _set_units_and_value(self, from_u, to_u, value):
        self._set_units(from_u, to_u)
        self._from_input.setText(str(value))

    # ── Conversion Chain ───────────────────────────────────────────────
    def _show_conversion_chain(self):
        """Show the conversion path from source to destination unit via base unit."""
        category = self._cat_combo.currentText()
        from_unit = self._from_combo.currentText()
        to_unit = self._to_combo.currentText()
        text = self._from_input.text().strip()
        if not from_unit or not to_unit or not text:
            return
        try:
            value = float(text)
        except ValueError:
            return

        data = UNIT_DATA.get(category, {})
        base_name = data.get("_base", "base")

        if category == "Temperature":
            chain_text = (f"{_format_result(value)} {from_unit}\n"
                          f"  -> convert to Kelvin\n"
                          f"  -> convert to {to_unit}\n"
                          f"= {self._to_output.text()}")
        else:
            from_factor = data.get(from_unit, 1)
            to_factor = data.get(to_unit, 1)
            base_val = value * from_factor
            result = base_val / to_factor
            chain_text = (f"{_format_result(value)} {from_unit}\n"
                          f"  x {from_factor} = {_format_result(base_val)} {base_name}\n"
                          f"  / {to_factor} = {_format_result(result)} {to_unit}")

        self._chain_display.setText(chain_text)
        self._emit_log(f"Conversion chain: {from_unit} -> {base_name} -> {to_unit}")

    # ── Dimensional Analysis ───────────────────────────────────────────
    def _dimensional_analysis_dialog(self):
        """Check if an expression has consistent dimensions."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Dimensional Analysis")
        dlg.setMinimumWidth(500)
        lay = QVBoxLayout(dlg)

        lay.addWidget(QLabel("Enter an expression using units (e.g., 'kg * m / s^2'):"))
        expr_edit = QLineEdit("kg * m / s^2")
        lay.addWidget(expr_edit)

        result_label = QTextEdit()
        result_label.setReadOnly(True)
        result_label.setMaximumHeight(200)
        lay.addWidget(result_label)

        # Basic dimension map: unit -> {dimension: power}
        dim_map = {
            "m": {"L": 1}, "km": {"L": 1}, "cm": {"L": 1}, "mm": {"L": 1},
            "ft": {"L": 1}, "in": {"L": 1}, "mi": {"L": 1},
            "kg": {"M": 1}, "g": {"M": 1}, "mg": {"M": 1}, "lb": {"M": 1},
            "s": {"T": 1}, "ms": {"T": 1}, "min": {"T": 1}, "h": {"T": 1},
            "A": {"I": 1}, "mA": {"I": 1},
            "K": {"Theta": 1},
            "mol": {"N": 1},
            "cd": {"J": 1},
            # Derived
            "N": {"M": 1, "L": 1, "T": -2},
            "J": {"M": 1, "L": 2, "T": -2},
            "W": {"M": 1, "L": 2, "T": -3},
            "Pa": {"M": 1, "L": -1, "T": -2},
            "Hz": {"T": -1},
            "V": {"M": 1, "L": 2, "T": -3, "I": -1},
            "C": {"T": 1, "I": 1},
        }

        def analyze():
            expr = expr_edit.text().strip()
            if not expr:
                return
            try:
                result_dims = {}
                # Simple parser: split by * and /, handle ^
                parts = expr.replace("/", " / ").replace("*", " * ").split()
                sign = 1  # 1 for multiply, -1 for divide
                for part in parts:
                    if part == "*":
                        sign = 1
                        continue
                    elif part == "/":
                        sign = -1
                        continue

                    # Handle ^power
                    power = 1
                    if "^" in part:
                        base, exp_str = part.split("^", 1)
                        power = int(exp_str)
                        part = base

                    if part in dim_map:
                        for dim, p in dim_map[part].items():
                            result_dims[dim] = result_dims.get(dim, 0) + sign * p * power
                    else:
                        result_label.setPlainText(f"Unknown unit: '{part}'\n\n"
                                                  f"Known units: {', '.join(sorted(dim_map.keys()))}")
                        return

                # Format result
                # Remove zero dimensions
                result_dims = {k: v for k, v in result_dims.items() if v != 0}

                if not result_dims:
                    dim_str = "dimensionless"
                else:
                    pos = [f"{k}^{v}" if v != 1 else k for k, v in result_dims.items() if v > 0]
                    neg = [f"{k}^{abs(v)}" if abs(v) != 1 else k for k, v in result_dims.items() if v < 0]
                    dim_str = " * ".join(pos)
                    if neg:
                        dim_str += " / (" + " * ".join(neg) + ")"

                # Identify known derived unit
                known = ""
                known_derived = {
                    "Force (N)": {"M": 1, "L": 1, "T": -2},
                    "Energy (J)": {"M": 1, "L": 2, "T": -2},
                    "Power (W)": {"M": 1, "L": 2, "T": -3},
                    "Pressure (Pa)": {"M": 1, "L": -1, "T": -2},
                    "Velocity": {"L": 1, "T": -1},
                    "Acceleration": {"L": 1, "T": -2},
                    "Frequency (Hz)": {"T": -1},
                    "Voltage (V)": {"M": 1, "L": 2, "T": -3, "I": -1},
                }
                for name, dims in known_derived.items():
                    if dims == result_dims:
                        known = f"\nRecognized as: {name}"
                        break

                result_label.setPlainText(f"Expression: {expr}\n"
                                          f"Dimensions: [{dim_str}]{known}")
            except Exception as e:
                result_label.setPlainText(f"Parse error: {e}")

        check_btn = QPushButton("Analyze Dimensions")
        check_btn.clicked.connect(analyze)
        lay.addWidget(check_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        lay.addWidget(close_btn)
        dlg.exec_()

    # ── Custom Unit Definitions ────────────────────────────────────────
    def _add_custom_unit_dialog(self):
        """Allow users to define custom units in any category."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Define Custom Unit")
        form = QFormLayout(dlg)

        cat_combo = QComboBox()
        cat_combo.addItems(sorted(k for k in UNIT_DATA.keys() if k != "Temperature"))
        form.addRow("Category:", cat_combo)

        name_edit = QLineEdit()
        name_edit.setPlaceholderText("e.g., furlong")
        form.addRow("Unit Name:", name_edit)

        factor_edit = QLineEdit()
        factor_edit.setPlaceholderText("Factor relative to base unit (e.g., 201.168 for furlong->meter)")
        form.addRow("Conversion Factor:", factor_edit)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)

        if dlg.exec_() != QDialog.Accepted:
            return

        category = cat_combo.currentText()
        unit_name = name_edit.text().strip()
        factor_text = factor_edit.text().strip()
        if not unit_name or not factor_text:
            QMessageBox.warning(self, "Error", "Name and factor are required.")
            return
        try:
            factor = float(factor_text)
        except ValueError:
            QMessageBox.warning(self, "Error", "Factor must be a number.")
            return

        if category not in UNIT_DATA:
            return

        # Check for duplicate
        if unit_name in UNIT_DATA[category]:
            QMessageBox.warning(self, "Duplicate", f"Unit '{unit_name}' already exists.")
            return

        UNIT_DATA[category][unit_name] = factor
        self._emit_log(f"Added custom unit: {unit_name} = {factor} (in {category})")

        # Refresh combos if current category matches
        if self._cat_combo.currentText() == category:
            self._on_category_changed(category)

        QMessageBox.information(self, "Custom Unit Added",
                                f"'{unit_name}' added to {category} with factor {factor}.")
