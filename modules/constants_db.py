"""
Physical Constants and Material Properties Database Widget for QuantumRes.
Searchable tables with unit toggling, category filtering, and clipboard support.
"""

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QLabel, QLineEdit, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QTabWidget, QGroupBox, QApplication, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


# ── Physical Constants Data ─────────────────────────────────────────────────
# (name, symbol, SI_value, SI_units, CGS_value, CGS_units, uncertainty, category)
CONSTANTS = [
    ("Speed of light in vacuum", "c", 2.99792458e8, "m/s", 2.99792458e10, "cm/s", "exact", "Electromagnetic"),
    ("Planck constant", "h", 6.62607015e-34, "J s", 6.62607015e-27, "erg s", "exact", "Quantum"),
    ("Reduced Planck constant", "\u0127", 1.054571817e-34, "J s", 1.054571817e-27, "erg s", "exact", "Quantum"),
    ("Boltzmann constant", "k_B", 1.380649e-23, "J/K", 1.380649e-16, "erg/K", "exact", "Thermodynamic"),
    ("Avogadro constant", "N_A", 6.02214076e23, "1/mol", 6.02214076e23, "1/mol", "exact", "Atomic"),
    ("Elementary charge", "e", 1.602176634e-19, "C", 4.80320427e-10, "esu", "exact", "Electromagnetic"),
    ("Electron mass", "m_e", 9.1093837015e-31, "kg", 9.1093837015e-28, "g", "3.0e-40 kg", "Atomic"),
    ("Proton mass", "m_p", 1.67262192369e-27, "kg", 1.67262192369e-24, "g", "5.1e-37 kg", "Atomic"),
    ("Neutron mass", "m_n", 1.67492749804e-27, "kg", 1.67492749804e-24, "g", "9.5e-37 kg", "Atomic"),
    ("Fine-structure constant", "\u03b1", 7.2973525693e-3, "", 7.2973525693e-3, "", "1.1e-12", "Electromagnetic"),
    ("Gravitational constant", "G", 6.67430e-11, "m\u00b3/(kg s\u00b2)", 6.67430e-8, "cm\u00b3/(g s\u00b2)", "1.5e-15", "Gravitational"),
    ("Magnetic constant (vacuum permeability)", "\u03bc_0", 1.25663706212e-6, "N/A\u00b2", 1.0, "dimensionless", "exact (CGS)", "Electromagnetic"),
    ("Electric constant (vacuum permittivity)", "\u03b5_0", 8.8541878128e-12, "F/m", 1.0, "dimensionless", "exact (CGS)", "Electromagnetic"),
    ("Coulomb constant", "k_e", 8.9875517923e9, "N m\u00b2/C\u00b2", 1.0, "dyne cm\u00b2/esu\u00b2", "exact (CGS)", "Electromagnetic"),
    ("Bohr radius", "a_0", 5.29177210903e-11, "m", 5.29177210903e-9, "cm", "8.0e-21 m", "Atomic"),
    ("Bohr magneton", "\u03bc_B", 9.2740100783e-24, "J/T", 9.2740100783e-21, "erg/G", "2.8e-33 J/T", "Electromagnetic"),
    ("Nuclear magneton", "\u03bc_N", 5.0507837461e-27, "J/T", 5.0507837461e-24, "erg/G", "1.5e-36 J/T", "Electromagnetic"),
    ("Rydberg constant", "R_inf", 1.0973731568160e7, "1/m", 1.0973731568160e5, "1/cm", "2.1e-5 1/m", "Atomic"),
    ("Stefan-Boltzmann constant", "\u03c3", 5.670374419e-8, "W/(m\u00b2 K\u2074)", 5.670374419e-5, "erg/(cm\u00b2 s K\u2074)", "exact", "Thermodynamic"),
    ("Wien displacement constant", "b", 2.897771955e-3, "m K", 2.897771955e-1, "cm K", "exact", "Thermodynamic"),
    ("Gas constant", "R", 8.314462618, "J/(mol K)", 8.314462618e7, "erg/(mol K)", "exact", "Thermodynamic"),
    ("Faraday constant", "F", 96485.33212, "C/mol", 96485.33212, "C/mol", "exact", "Electromagnetic"),
    ("Atomic mass unit", "u", 1.66053906660e-27, "kg", 1.66053906660e-24, "g", "5.0e-37 kg", "Atomic"),
    ("Electron volt", "eV", 1.602176634e-19, "J", 1.602176634e-12, "erg", "exact", "Atomic"),
    ("Standard atmosphere", "atm", 101325.0, "Pa", 1013250.0, "dyn/cm\u00b2", "exact", "Thermodynamic"),
    ("Standard gravity", "g_n", 9.80665, "m/s\u00b2", 980.665, "cm/s\u00b2", "exact", "Gravitational"),
    ("Compton wavelength", "\u03bb_C", 2.42631023867e-12, "m", 2.42631023867e-10, "cm", "7.3e-22 m", "Quantum"),
    ("Classical electron radius", "r_e", 2.8179403262e-15, "m", 2.8179403262e-13, "cm", "1.3e-24 m", "Electromagnetic"),
    ("Thomson cross section", "\u03c3_T", 6.6524587321e-29, "m\u00b2", 6.6524587321e-25, "cm\u00b2", "6.0e-38 m\u00b2", "Electromagnetic"),
    ("Impedance of vacuum", "Z_0", 376.730313668, "\u03a9", 376.730313668, "\u03a9", "exact", "Electromagnetic"),
    ("Molar mass of carbon-12", "M(12C)", 11.9999999958e-3, "kg/mol", 11.9999999958, "g/mol", "3.6e-12", "Atomic"),
    ("Hartree energy", "E_h", 4.3597447222071e-18, "J", 4.3597447222071e-11, "erg", "8.5e-30 J", "Atomic"),
    ("Vacuum magnetic flux quantum", "\u03a6_0", 2.067833848e-15, "Wb", 2.067833848e-7, "Mx", "exact", "Quantum"),
    ("Conductance quantum", "G_0", 7.748091729e-5, "S", 7.748091729e-5, "S", "exact", "Quantum"),
    ("Josephson constant", "K_J", 483597.8484e9, "Hz/V", 483597.8484e9, "Hz/V", "exact", "Quantum"),
    ("von Klitzing constant", "R_K", 25812.80745, "\u03a9", 25812.80745, "\u03a9", "exact", "Quantum"),
    ("First radiation constant", "c_1", 3.741771852e-16, "W m\u00b2", 3.741771852e-5, "erg cm\u00b2/s", "exact", "Thermodynamic"),
    ("Second radiation constant", "c_2", 1.438776877e-2, "m K", 1.438776877, "cm K", "exact", "Thermodynamic"),
    ("Loschmidt constant (273.15K)", "n_0", 2.6867774e25, "1/m\u00b3", 2.6867774e19, "1/cm\u00b3", "exact", "Thermodynamic"),
    ("Molar Planck constant", "N_A h", 3.990312712e-10, "J s/mol", 3.990312712e-3, "erg s/mol", "exact", "Quantum"),
    ("Electron g-factor", "g_e", -2.00231930436256, "", -2.00231930436256, "", "3.5e-13", "Atomic"),
    ("Proton g-factor", "g_p", 5.5856946893, "", 5.5856946893, "", "1.6e-9", "Atomic"),
    ("Muon mass", "m_\u03bc", 1.883531627e-28, "kg", 1.883531627e-25, "g", "4.2e-36 kg", "Atomic"),
    ("Tau mass", "m_\u03c4", 3.16754e-27, "kg", 3.16754e-24, "g", "2.1e-31 kg", "Atomic"),
    ("Deuteron mass", "m_d", 3.3435837724e-27, "kg", 3.3435837724e-24, "g", "1.0e-36 kg", "Atomic"),
    ("Alpha particle mass", "m_\u03b1", 6.6446573357e-27, "kg", 6.6446573357e-24, "g", "2.0e-36 kg", "Atomic"),
    ("Planck length", "l_P", 1.616255e-35, "m", 1.616255e-33, "cm", "1.8e-40 m", "Quantum"),
    ("Planck mass", "m_P", 2.176434e-8, "kg", 2.176434e-5, "g", "2.4e-13 kg", "Quantum"),
    ("Planck time", "t_P", 5.391247e-44, "s", 5.391247e-44, "s", "6.0e-49 s", "Quantum"),
    ("Planck temperature", "T_P", 1.416784e32, "K", 1.416784e32, "K", "1.6e27 K", "Quantum"),
    ("Cosmological constant (approx)", "\u039b", 1.1056e-52, "1/m\u00b2", 1.1056e-56, "1/cm\u00b2", "~order of magnitude", "Gravitational"),
    ("Hubble constant (approx)", "H_0", 2.2e-18, "1/s", 2.2e-18, "1/s", "~67-74 km/s/Mpc", "Gravitational"),
    ("Solar mass", "M_sun", 1.989e30, "kg", 1.989e33, "g", "~1e26 kg", "Gravitational"),
    ("Earth mass", "M_earth", 5.972e24, "kg", 5.972e27, "g", "~1e20 kg", "Gravitational"),
]

# ── Material Properties Data ────────────────────────────────────────────────
# (name, density_kg_m3, thermal_cond_W_mK, youngs_modulus_GPa)
MATERIALS = [
    ("Steel (mild)", 7850, 50.2, 200),
    ("Steel (stainless 304)", 8000, 16.2, 193),
    ("Aluminium (6061)", 2700, 167, 68.9),
    ("Copper", 8960, 401, 117),
    ("Gold", 19300, 318, 79),
    ("Silver", 10490, 429, 83),
    ("Titanium", 4507, 21.9, 116),
    ("Iron (pure)", 7874, 80.4, 211),
    ("Nickel", 8908, 90.9, 200),
    ("Tungsten", 19250, 173, 411),
    ("Platinum", 21450, 71.6, 168),
    ("Lead", 11340, 35.3, 16),
    ("Zinc", 7134, 116, 108),
    ("Silicon", 2329, 149, 130),
    ("Germanium", 5323, 60.2, 103),
    ("Glass (soda-lime)", 2500, 1.0, 72),
    ("Glass (borosilicate)", 2230, 1.14, 63),
    ("Diamond", 3510, 2200, 1220),
    ("Graphite", 2260, 25, 8),
    ("Concrete", 2400, 1.7, 30),
    ("Brick", 1920, 0.72, 15),
    ("Wood (oak)", 750, 0.17, 12),
    ("Wood (pine)", 510, 0.12, 9),
    ("Water (20C)", 998, 0.598, None),
    ("Ice (0C)", 917, 2.22, 9.33),
    ("Air (STP)", 1.225, 0.0262, None),
    ("Rubber (natural)", 920, 0.13, 0.05),
    ("Nylon", 1150, 0.25, 2.7),
    ("Polyethylene (HDPE)", 960, 0.46, 0.8),
    ("PTFE (Teflon)", 2200, 0.25, 0.5),
    ("Epoxy", 1200, 0.2, 3.5),
    ("Carbon fiber composite", 1600, 7.0, 181),
    ("Kevlar", 1440, 0.04, 70.5),
    ("Quartz (fused)", 2200, 1.38, 73),
    ("Sapphire", 3980, 46.06, 345),
    ("Bone (cortical)", 1900, 0.32, 14),
]


class ConstantsDBWidget(QWidget):
    """Physical constants and material properties database."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._log = None
        self._si_mode = True  # True = SI, False = CGS
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
        tabs = QTabWidget()
        main.addWidget(tabs)

        # Tab 1: Physical Constants
        const_page = QWidget()
        const_layout = QVBoxLayout(const_page)

        # Controls row
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Search:"))
        self._const_search = QLineEdit()
        self._const_search.setPlaceholderText("Filter by name or symbol...")
        self._const_search.textChanged.connect(self._filter_constants)
        ctrl.addWidget(self._const_search)

        ctrl.addWidget(QLabel("Category:"))
        self._cat_filter = QComboBox()
        categories = sorted(set(c[7] for c in CONSTANTS))
        self._cat_filter.addItem("All")
        self._cat_filter.addItems(categories)
        self._cat_filter.currentTextChanged.connect(self._filter_constants)
        ctrl.addWidget(self._cat_filter)

        self._unit_toggle = QPushButton("Units: SI")
        self._unit_toggle.setCheckable(True)
        self._unit_toggle.clicked.connect(self._toggle_units)
        ctrl.addWidget(self._unit_toggle)

        copy_btn = QPushButton("Copy Selected Value")
        copy_btn.clicked.connect(self._copy_constant)
        ctrl.addWidget(copy_btn)
        const_layout.addLayout(ctrl)

        # Constants table
        self._const_table = QTableWidget()
        self._const_table.setColumnCount(5)
        self._const_table.setHorizontalHeaderLabels(["Name", "Symbol", "Value", "Units", "Uncertainty"])
        self._const_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._const_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._const_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._const_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._const_table.setAlternatingRowColors(True)
        const_layout.addWidget(self._const_table)
        self._populate_constants()

        tabs.addTab(const_page, "Physical Constants")

        # Tab 2: Material Properties
        mat_page = QWidget()
        mat_layout = QVBoxLayout(mat_page)

        mat_ctrl = QHBoxLayout()
        mat_ctrl.addWidget(QLabel("Search:"))
        self._mat_search = QLineEdit()
        self._mat_search.setPlaceholderText("Filter materials...")
        self._mat_search.textChanged.connect(self._filter_materials)
        mat_ctrl.addWidget(self._mat_search)

        copy_mat_btn = QPushButton("Copy Selected Value")
        copy_mat_btn.clicked.connect(self._copy_material)
        mat_ctrl.addWidget(copy_mat_btn)
        mat_layout.addLayout(mat_ctrl)

        self._mat_table = QTableWidget()
        self._mat_table.setColumnCount(4)
        self._mat_table.setHorizontalHeaderLabels([
            "Material", "Density (kg/m\u00b3)", "Thermal Cond. (W/mK)", "Young's Modulus (GPa)"
        ])
        self._mat_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._mat_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._mat_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._mat_table.setAlternatingRowColors(True)
        mat_layout.addWidget(self._mat_table)
        self._populate_materials()

        tabs.addTab(mat_page, "Material Properties")

    # ── Constants Table ─────────────────────────────────────────────────
    def _populate_constants(self):
        search = self._const_search.text().strip().lower() if hasattr(self, '_const_search') else ""
        cat = self._cat_filter.currentText() if hasattr(self, '_cat_filter') else "All"

        filtered = []
        for c in CONSTANTS:
            name, sym, si_val, si_unit, cgs_val, cgs_unit, unc, category = c
            if search and search not in name.lower() and search not in sym.lower():
                continue
            if cat != "All" and category != cat:
                continue
            if self._si_mode:
                filtered.append((name, sym, si_val, si_unit, unc))
            else:
                filtered.append((name, sym, cgs_val, cgs_unit, unc))

        self._const_table.setRowCount(len(filtered))
        for r, (name, sym, val, unit, unc) in enumerate(filtered):
            self._const_table.setItem(r, 0, QTableWidgetItem(name))
            self._const_table.setItem(r, 1, QTableWidgetItem(sym))
            val_str = f"{val:.10e}" if isinstance(val, float) and (abs(val) < 1e-3 or abs(val) > 1e6) else str(val)
            self._const_table.setItem(r, 2, QTableWidgetItem(val_str))
            self._const_table.setItem(r, 3, QTableWidgetItem(unit))
            self._const_table.setItem(r, 4, QTableWidgetItem(str(unc)))

    def _filter_constants(self):
        self._populate_constants()

    def _toggle_units(self):
        self._si_mode = not self._si_mode
        label = "Units: SI" if self._si_mode else "Units: CGS"
        self._unit_toggle.setText(label)
        self._emit_log(f"Switched to {'SI' if self._si_mode else 'CGS'} units")
        self._populate_constants()

    def _copy_constant(self):
        row = self._const_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Copy", "Select a row first.")
            return
        val_item = self._const_table.item(row, 2)
        if val_item:
            QApplication.clipboard().setText(val_item.text())
            name_item = self._const_table.item(row, 0)
            self._emit_log(f"Copied value of {name_item.text()}: {val_item.text()}")

    # ── Materials Table ─────────────────────────────────────────────────
    def _populate_materials(self):
        search = self._mat_search.text().strip().lower() if hasattr(self, '_mat_search') else ""
        filtered = [m for m in MATERIALS if not search or search in m[0].lower()]

        self._mat_table.setRowCount(len(filtered))
        for r, (name, density, tcond, young) in enumerate(filtered):
            self._mat_table.setItem(r, 0, QTableWidgetItem(name))
            self._mat_table.setItem(r, 1, QTableWidgetItem(str(density)))
            self._mat_table.setItem(r, 2, QTableWidgetItem(str(tcond)))
            self._mat_table.setItem(r, 3, QTableWidgetItem(str(young) if young is not None else "N/A"))

    def _filter_materials(self):
        self._populate_materials()

    def _copy_material(self):
        row = self._mat_table.currentRow()
        col = self._mat_table.currentColumn()
        if row < 0:
            QMessageBox.information(self, "Copy", "Select a cell first.")
            return
        item = self._mat_table.item(row, max(col, 0))
        if item:
            QApplication.clipboard().setText(item.text())
            self._emit_log(f"Copied: {item.text()}")
