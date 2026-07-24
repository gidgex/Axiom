"""
Phase Diagram Tool for Materials Science
Binary and ternary phase diagrams with lever rule, tie lines,
cooling curves, and Gibbs phase rule calculations.
"""

import numpy as np
from scipy.interpolate import interp1d
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QComboBox, QDoubleSpinBox, QPushButton, QCheckBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QSplitter,
    QScrollArea, QFileDialog, QSpinBox, QTextEdit
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
import matplotlib.tri as mtri
import csv


# ---------------------------------------------------------------- Data
# Each system: { "name", "comp_a", "comp_b", "liquidus_pts", "solidus_pts",
#                "eutectic" (optional), "phase_labels" }
# Points are (composition_B_pct, temperature_C)

BUILT_IN_SYSTEMS = {
    "Cu-Ni": {
        "comp_a": "Cu", "comp_b": "Ni",
        "liquidus_pts": [(0, 1085), (10, 1120), (20, 1160), (30, 1200),
                         (40, 1240), (50, 1280), (60, 1320), (70, 1360),
                         (80, 1400), (90, 1430), (100, 1455)],
        "solidus_pts": [(0, 1085), (10, 1090), (20, 1100), (30, 1120),
                        (40, 1150), (50, 1190), (60, 1240), (70, 1290),
                        (80, 1340), (90, 1390), (100, 1455)],
        "eutectic": None,
        "type": "isomorphous",
    },
    "Pb-Sn": {
        "comp_a": "Pb", "comp_b": "Sn",
        "liquidus_pts": [(0, 327), (10, 300), (20, 270), (30, 245),
                         (38.1, 183), (50, 210), (60, 225), (70, 215),
                         (80, 210), (90, 210), (100, 232)],
        "solidus_pts": [(0, 327), (10, 250), (18.3, 183), (38.1, 183),
                        (61.9, 183), (80, 183), (97.8, 183), (100, 232)],
        "eutectic": {"comp": 61.9, "temp": 183},
        "type": "eutectic",
    },
    "Fe-C": {
        "comp_a": "Fe", "comp_b": "C",
        "liquidus_pts": [(0, 1538), (0.5, 1495), (1.0, 1470), (1.5, 1440),
                         (2.0, 1400), (2.5, 1350), (3.0, 1300), (3.5, 1250),
                         (4.0, 1200), (4.3, 1147), (5.0, 1200), (6.67, 1227)],
        "solidus_pts": [(0, 1538), (0.1, 1510), (0.17, 1495), (0.5, 1400),
                        (0.8, 1300), (1.0, 1200), (2.14, 1147),
                        (4.3, 1147), (6.67, 1227)],
        "eutectic": {"comp": 4.3, "temp": 1147},
        "type": "eutectic",
    },
    "Cu-Zn": {
        "comp_a": "Cu", "comp_b": "Zn",
        "liquidus_pts": [(0, 1085), (5, 1060), (10, 1030), (15, 1000),
                         (20, 975), (25, 960), (30, 940), (35, 910),
                         (37, 903), (40, 880), (50, 840), (60, 800),
                         (70, 750), (80, 700), (90, 600), (100, 420)],
        "solidus_pts": [(0, 1085), (5, 1040), (10, 1000), (15, 970),
                        (20, 940), (25, 920), (30, 905), (35, 903),
                        (37, 903), (40, 860), (50, 820), (60, 780),
                        (70, 720), (80, 650), (90, 550), (100, 420)],
        "eutectic": None,
        "type": "isomorphous",
    },
    "Al-Si": {
        "comp_a": "Al", "comp_b": "Si",
        "liquidus_pts": [(0, 660), (2, 640), (5, 615), (8, 595),
                         (10, 580), (12.6, 577), (15, 590), (20, 650),
                         (30, 780), (40, 900), (50, 1000), (60, 1100),
                         (80, 1300), (100, 1414)],
        "solidus_pts": [(0, 660), (1.65, 577), (12.6, 577),
                        (99.83, 577), (100, 1414)],
        "eutectic": {"comp": 12.6, "temp": 577},
        "type": "eutectic",
    },
    "Au-Si": {
        "comp_a": "Au", "comp_b": "Si",
        "liquidus_pts": [(0, 1064), (5, 950), (10, 800), (15, 650),
                         (18.6, 363), (25, 500), (40, 750), (60, 1000),
                         (80, 1200), (100, 1414)],
        "solidus_pts": [(0, 1064), (2, 363), (18.6, 363),
                        (98, 363), (100, 1414)],
        "eutectic": {"comp": 18.6, "temp": 363},
        "type": "eutectic",
    },
}

TERNARY_EXAMPLES = {
    "Fe-Cr-Ni": {
        "components": ["Fe", "Cr", "Ni"],
        "regions": [
            {"label": "Austenite", "centroid": (0.70, 0.10, 0.20), "color": "#4488FF"},
            {"label": "Ferrite", "centroid": (0.60, 0.30, 0.10), "color": "#FF8844"},
            {"label": "Martensite", "centroid": (0.85, 0.05, 0.10), "color": "#44FF88"},
        ],
    },
}


class PhaseDiagramWidget(QWidget):
    """Interactive phase diagram tool for materials science."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._log = None
        self._current_system = None
        self._liquidus_interp = None
        self._solidus_interp = None
        self._custom_liquidus = []
        self._custom_solidus = []
        self._build_ui()
        self._load_system("Cu-Ni")

    def set_logger(self, fn):
        self._log = fn

    def _emit_log(self, msg):
        if self._log:
            self._log(msg)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        main_layout = QHBoxLayout(self)

        # Left panel: controls
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setMaximumWidth(320)
        ctrl_widget = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_widget)

        # System selector
        sys_grp = QGroupBox("Phase System")
        sys_lay = QVBoxLayout(sys_grp)
        self.system_combo = QComboBox()
        self.system_combo.addItems(list(BUILT_IN_SYSTEMS.keys()) + ["Custom..."])
        self.system_combo.currentTextChanged.connect(self._on_system_changed)
        sys_lay.addWidget(self.system_combo)
        ctrl_layout.addWidget(sys_grp)

        # Interactive query
        query_grp = QGroupBox("Query Point")
        query_lay = QGridLayout(query_grp)
        query_lay.addWidget(QLabel("Composition (%B):"), 0, 0)
        self.query_comp = QDoubleSpinBox()
        self.query_comp.setRange(0, 100)
        self.query_comp.setValue(30)
        self.query_comp.setDecimals(1)
        query_lay.addWidget(self.query_comp, 0, 1)

        query_lay.addWidget(QLabel("Temperature (\u00b0C):"), 1, 0)
        self.query_temp = QDoubleSpinBox()
        self.query_temp.setRange(-273, 3500)
        self.query_temp.setValue(800)
        self.query_temp.setDecimals(1)
        query_lay.addWidget(self.query_temp, 1, 1)

        self.btn_query = QPushButton("Query Phase")
        self.btn_query.clicked.connect(self._query_phase)
        query_lay.addWidget(self.btn_query, 2, 0, 1, 2)
        ctrl_layout.addWidget(query_grp)

        # Lever rule
        lever_grp = QGroupBox("Lever Rule")
        lever_lay = QVBoxLayout(lever_grp)
        self.btn_lever = QPushButton("Calculate Lever Rule")
        self.btn_lever.clicked.connect(self._calculate_lever_rule)
        lever_lay.addWidget(self.btn_lever)
        self.lever_result = QLabel("Click a point in two-phase region.")
        self.lever_result.setWordWrap(True)
        lever_lay.addWidget(self.lever_result)
        ctrl_layout.addWidget(lever_grp)

        # Tie line
        tie_grp = QGroupBox("Tie Line")
        tie_lay = QGridLayout(tie_grp)
        tie_lay.addWidget(QLabel("Temperature (\u00b0C):"), 0, 0)
        self.tie_temp = QDoubleSpinBox()
        self.tie_temp.setRange(-273, 3500)
        self.tie_temp.setValue(600)
        tie_lay.addWidget(self.tie_temp, 0, 1)
        self.btn_tie = QPushButton("Draw Tie Line")
        self.btn_tie.clicked.connect(self._draw_tie_line)
        tie_lay.addWidget(self.btn_tie, 1, 0, 1, 2)
        ctrl_layout.addWidget(tie_grp)

        # Cooling curve
        cool_grp = QGroupBox("Cooling Curve")
        cool_lay = QGridLayout(cool_grp)
        cool_lay.addWidget(QLabel("Composition (%B):"), 0, 0)
        self.cool_comp = QDoubleSpinBox()
        self.cool_comp.setRange(0, 100)
        self.cool_comp.setValue(40)
        cool_lay.addWidget(self.cool_comp, 0, 1)
        self.btn_cool = QPushButton("Simulate Cooling")
        self.btn_cool.clicked.connect(self._simulate_cooling)
        cool_lay.addWidget(self.btn_cool, 1, 0, 1, 2)
        ctrl_layout.addWidget(cool_grp)

        # Gibbs phase rule
        gibbs_grp = QGroupBox("Gibbs Phase Rule")
        gibbs_lay = QGridLayout(gibbs_grp)
        gibbs_lay.addWidget(QLabel("Components (C):"), 0, 0)
        self.gibbs_c = QSpinBox()
        self.gibbs_c.setRange(1, 10)
        self.gibbs_c.setValue(2)
        gibbs_lay.addWidget(self.gibbs_c, 0, 1)
        gibbs_lay.addWidget(QLabel("Phases (P):"), 1, 0)
        self.gibbs_p = QSpinBox()
        self.gibbs_p.setRange(1, 10)
        self.gibbs_p.setValue(1)
        gibbs_lay.addWidget(self.gibbs_p, 1, 1)
        self.btn_gibbs = QPushButton("Calculate F")
        self.btn_gibbs.clicked.connect(self._calc_gibbs)
        gibbs_lay.addWidget(self.btn_gibbs, 2, 0, 1, 2)
        self.gibbs_result = QLabel("F = C - P + 2")
        gibbs_lay.addWidget(self.gibbs_result, 3, 0, 1, 2)
        ctrl_layout.addWidget(gibbs_grp)

        # Custom diagram editor
        custom_grp = QGroupBox("Custom Diagram Editor")
        custom_lay = QVBoxLayout(custom_grp)
        custom_lay.addWidget(QLabel("Liquidus (comp,temp per line):"))
        self.custom_liquidus_edit = QTextEdit()
        self.custom_liquidus_edit.setMaximumHeight(80)
        self.custom_liquidus_edit.setPlaceholderText("0,1000\n50,800\n100,1200")
        custom_lay.addWidget(self.custom_liquidus_edit)
        custom_lay.addWidget(QLabel("Solidus (comp,temp per line):"))
        self.custom_solidus_edit = QTextEdit()
        self.custom_solidus_edit.setMaximumHeight(80)
        self.custom_solidus_edit.setPlaceholderText("0,1000\n50,700\n100,1200")
        custom_lay.addWidget(self.custom_solidus_edit)
        self.btn_custom_load = QPushButton("Load Custom Diagram")
        self.btn_custom_load.clicked.connect(self._load_custom)
        custom_lay.addWidget(self.btn_custom_load)
        ctrl_layout.addWidget(custom_grp)

        # Action buttons
        btn_lay = QHBoxLayout()
        self.btn_run = QPushButton("Run / Refresh")
        self.btn_run.clicked.connect(self.run)
        btn_lay.addWidget(self.btn_run)
        self.btn_export = QPushButton("Export")
        self.btn_export.clicked.connect(self.export)
        btn_lay.addWidget(self.btn_export)
        ctrl_layout.addLayout(btn_lay)

        ctrl_layout.addStretch()
        ctrl_scroll.setWidget(ctrl_widget)
        main_layout.addWidget(ctrl_scroll)

        # Right: plots
        self.tabs = QTabWidget()

        # Binary phase diagram
        self.fig_binary = Figure(figsize=(7, 6), facecolor="#fafafa")
        self.canvas_binary = FigureCanvas(self.fig_binary)
        self.ax_binary = self.fig_binary.add_subplot(111)
        self.canvas_binary.mpl_connect("button_press_event", self._on_binary_click)
        self.canvas_binary.mpl_connect("motion_notify_event", self._on_binary_hover)
        self.tabs.addTab(self.canvas_binary, "Binary Diagram")

        # Cooling curve plot
        self.fig_cool = Figure(figsize=(5, 5), facecolor="#fafafa")
        self.canvas_cool = FigureCanvas(self.fig_cool)
        self.ax_cool = self.fig_cool.add_subplot(111)
        self.tabs.addTab(self.canvas_cool, "Cooling Curve")

        # Ternary diagram
        self.fig_ternary = Figure(figsize=(6, 6), facecolor="#fafafa")
        self.canvas_ternary = FigureCanvas(self.fig_ternary)
        self.ax_ternary = self.fig_ternary.add_subplot(111)
        self.tabs.addTab(self.canvas_ternary, "Ternary Diagram")

        # Info panel
        self.info_label = QLabel("Hover over diagram for phase info.")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("padding: 6px; background: #eef; border: 1px solid #bbc;")
        self.info_label.setMaximumHeight(60)

        right_panel = QSplitter(Qt.Vertical)
        right_panel.addWidget(self.tabs)
        right_panel.addWidget(self.info_label)
        main_layout.addWidget(right_panel, 1)

    # -------------------------------------------------------------- System
    def _on_system_changed(self, name):
        if name == "Custom...":
            return
        self._load_system(name)

    def _load_system(self, name):
        if name not in BUILT_IN_SYSTEMS:
            return
        sys = BUILT_IN_SYSTEMS[name]
        self._current_system = sys
        self._current_system["name"] = name
        liq = sorted(sys["liquidus_pts"], key=lambda p: p[0])
        sol = sorted(sys["solidus_pts"], key=lambda p: p[0])
        comps_l = [p[0] for p in liq]
        temps_l = [p[1] for p in liq]
        comps_s = [p[0] for p in sol]
        temps_s = [p[1] for p in sol]
        self._liquidus_interp = interp1d(comps_l, temps_l, kind="cubic",
                                         fill_value="extrapolate")
        self._solidus_interp = interp1d(comps_s, temps_s, kind="cubic",
                                        fill_value="extrapolate")
        self._draw_binary()
        self._draw_ternary()
        self._emit_log(f"Loaded phase system: {name}")

    def _load_custom(self):
        """Parse user-entered liquidus/solidus points and build diagram."""
        try:
            liq_text = self.custom_liquidus_edit.toPlainText().strip()
            sol_text = self.custom_solidus_edit.toPlainText().strip()
            liq_pts = []
            for line in liq_text.splitlines():
                parts = line.split(",")
                liq_pts.append((float(parts[0]), float(parts[1])))
            sol_pts = []
            for line in sol_text.splitlines():
                parts = line.split(",")
                sol_pts.append((float(parts[0]), float(parts[1])))
            if len(liq_pts) < 2 or len(sol_pts) < 2:
                self._emit_log("Need at least 2 points for each curve.")
                return
            self._current_system = {
                "name": "Custom",
                "comp_a": "A", "comp_b": "B",
                "liquidus_pts": liq_pts, "solidus_pts": sol_pts,
                "eutectic": None, "type": "custom",
            }
            liq = sorted(liq_pts, key=lambda p: p[0])
            sol = sorted(sol_pts, key=lambda p: p[0])
            self._liquidus_interp = interp1d(
                [p[0] for p in liq], [p[1] for p in liq],
                kind="cubic", fill_value="extrapolate"
            )
            self._solidus_interp = interp1d(
                [p[0] for p in sol], [p[1] for p in sol],
                kind="cubic", fill_value="extrapolate"
            )
            self._draw_binary()
            self._emit_log("Custom phase diagram loaded.")
        except Exception as e:
            self._emit_log(f"Error parsing custom diagram: {e}")

    # ----------------------------------------------------------- Drawing
    def _draw_binary(self):
        ax = self.ax_binary
        ax.clear()
        if self._current_system is None:
            return
        sys = self._current_system
        x_fine = np.linspace(0, 100, 500)
        t_liq = self._liquidus_interp(x_fine)
        t_sol = self._solidus_interp(x_fine)

        # Phase region fills
        t_max = max(np.max(t_liq), np.max(t_sol)) + 100
        t_min = min(np.min(t_liq), np.min(t_sol)) - 100

        # Liquid region (above liquidus)
        ax.fill_between(x_fine, t_liq, t_max, color="#FFD0D0", alpha=0.5, label="Liquid")
        # Two-phase region (between liquidus and solidus)
        ax.fill_between(x_fine, t_sol, t_liq, color="#D0FFD0", alpha=0.5, label="L + S")
        # Solid region (below solidus)
        ax.fill_between(x_fine, t_min, t_sol, color="#D0D0FF", alpha=0.5, label="Solid")

        # Lines
        ax.plot(x_fine, t_liq, "r-", linewidth=2, label="Liquidus")
        ax.plot(x_fine, t_sol, "b-", linewidth=2, label="Solidus")

        # Eutectic marker
        if sys.get("eutectic"):
            ec = sys["eutectic"]
            ax.plot(ec["comp"], ec["temp"], "ko", markersize=8)
            ax.annotate(f"Eutectic\n({ec['comp']}%, {ec['temp']}\u00b0C)",
                        xy=(ec["comp"], ec["temp"]),
                        xytext=(ec["comp"] + 5, ec["temp"] + 40),
                        fontsize=8, arrowprops=dict(arrowstyle="->"))

        # Phase labels
        mid_liq = (t_max + np.mean(t_liq)) / 2
        ax.text(50, mid_liq, "LIQUID", ha="center", fontsize=12, fontweight="bold",
                color="#CC0000", alpha=0.6)
        mid_sol = (t_min + np.mean(t_sol)) / 2
        ax.text(50, mid_sol, "SOLID", ha="center", fontsize=12, fontweight="bold",
                color="#0000CC", alpha=0.6)
        mid_two = (np.mean(t_liq) + np.mean(t_sol)) / 2
        ax.text(50, mid_two, "L + S", ha="center", fontsize=10, fontweight="bold",
                color="#008800", alpha=0.6)

        comp_a = sys.get("comp_a", "A")
        comp_b = sys.get("comp_b", "B")
        ax.set_xlabel(f"Composition (wt% {comp_b})", fontsize=10)
        ax.set_ylabel("Temperature (\u00b0C)", fontsize=10)
        ax.set_title(f"{comp_a}-{comp_b} Phase Diagram", fontsize=12, fontweight="bold")
        ax.set_xlim(0, 100)
        ax.set_ylim(t_min, t_max)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        self.canvas_binary.draw_idle()

    def _draw_ternary(self):
        """Draw a basic ternary diagram for Fe-Cr-Ni."""
        ax = self.ax_ternary
        ax.clear()
        ax.set_aspect("equal")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.0)
        ax.axis("off")
        ax.set_title("Ternary Phase Diagram (Fe-Cr-Ni)", fontsize=11, fontweight="bold")

        # Triangle vertices: A at (0,0), B at (1,0), C at (0.5, sqrt(3)/2)
        h = np.sqrt(3) / 2
        verts = np.array([[0, 0], [1, 0], [0.5, h]])
        triangle = Polygon(verts, closed=True, fill=False, edgecolor="black", linewidth=2)
        ax.add_patch(triangle)

        # Grid lines
        for frac in np.arange(0.1, 1.0, 0.1):
            # Lines parallel to each side
            p1 = verts[0] * (1 - frac) + verts[1] * frac
            p2 = verts[2] * (1 - frac) + verts[1] * frac
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k-", alpha=0.15, linewidth=0.5)
            p1 = verts[0] * (1 - frac) + verts[2] * frac
            p2 = verts[1] * (1 - frac) + verts[2] * frac
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k-", alpha=0.15, linewidth=0.5)
            p1 = verts[0] * (1 - frac) + verts[1] * frac
            p2 = verts[0] * (1 - frac) + verts[2] * frac
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k-", alpha=0.15, linewidth=0.5)

        # Vertex labels
        ax.text(0, -0.05, "Fe", ha="center", fontsize=10, fontweight="bold")
        ax.text(1, -0.05, "Cr", ha="center", fontsize=10, fontweight="bold")
        ax.text(0.5, h + 0.04, "Ni", ha="center", fontsize=10, fontweight="bold")

        # Phase regions (approximate)
        tern = TERNARY_EXAMPLES.get("Fe-Cr-Ni", {})
        for region in tern.get("regions", []):
            a, b, c = region["centroid"]
            # Barycentric to Cartesian
            x = b * 1.0 + c * 0.5
            y = c * h
            ax.plot(x, y, "o", color=region["color"], markersize=14, alpha=0.6)
            ax.text(x, y - 0.04, region["label"], ha="center", fontsize=7,
                    fontweight="bold", color=region["color"])

        self.canvas_ternary.draw_idle()

    # -------------------------------------------------------- Interaction
    def _on_binary_click(self, event):
        if event.inaxes != self.ax_binary or event.xdata is None:
            return
        comp = event.xdata
        temp = event.ydata
        self.query_comp.setValue(comp)
        self.query_temp.setValue(temp)
        self._query_phase()

    def _on_binary_hover(self, event):
        if event.inaxes != self.ax_binary or event.xdata is None:
            return
        comp = event.xdata
        temp = event.ydata
        phase = self._identify_phase(comp, temp)
        t_liq = float(self._liquidus_interp(comp))
        t_sol = float(self._solidus_interp(comp))
        dof = self._degrees_of_freedom(phase)
        self.info_label.setText(
            f"Comp: {comp:.1f}%  Temp: {temp:.0f}\u00b0C  |  Phase: {phase}  |  "
            f"Liquidus: {t_liq:.0f}\u00b0C  Solidus: {t_sol:.0f}\u00b0C  |  DOF(F): {dof}"
        )

    def _identify_phase(self, comp, temp):
        if self._liquidus_interp is None:
            return "Unknown"
        t_liq = float(self._liquidus_interp(np.clip(comp, 0, 100)))
        t_sol = float(self._solidus_interp(np.clip(comp, 0, 100)))
        if temp > t_liq:
            return "Liquid"
        elif temp < t_sol:
            return "Solid"
        else:
            return "Liquid + Solid"

    def _degrees_of_freedom(self, phase_str):
        c = 2  # binary system
        if phase_str == "Liquid":
            p = 1
        elif phase_str == "Solid":
            p = 1
        elif "+" in phase_str:
            p = 2
        else:
            p = 1
        return c - p + 2

    # -------------------------------------------------------- Query / Lever
    def _query_phase(self):
        comp = self.query_comp.value()
        temp = self.query_temp.value()
        phase = self._identify_phase(comp, temp)
        dof = self._degrees_of_freedom(phase)
        self._emit_log(
            f"At {comp:.1f}% B, {temp:.0f}\u00b0C: Phase = {phase}, "
            f"Degrees of freedom (F) = {dof}"
        )
        # Mark on plot
        self._draw_binary()
        self.ax_binary.plot(comp, temp, "k+", markersize=15, markeredgewidth=2)
        self.ax_binary.annotate(f"{phase}\nF={dof}", xy=(comp, temp),
                                xytext=(comp + 3, temp + 20), fontsize=8,
                                arrowprops=dict(arrowstyle="->"))
        self.canvas_binary.draw_idle()

    def _calculate_lever_rule(self):
        comp = self.query_comp.value()
        temp = self.query_temp.value()
        phase = self._identify_phase(comp, temp)
        if "+" not in phase:
            self.lever_result.setText("Point is not in a two-phase region.")
            self._emit_log("Lever rule: point must be in a two-phase region.")
            return

        # Find compositions at liquidus and solidus at this temperature
        # Solve liquidus_interp(x) = temp and solidus_interp(x) = temp
        x_range = np.linspace(0, 100, 5000)
        t_liq = self._liquidus_interp(x_range)
        t_sol = self._solidus_interp(x_range)

        # Find liquidus composition closest to temp
        liq_idx = np.argmin(np.abs(t_liq - temp))
        comp_liq = x_range[liq_idx]
        # Find solidus composition closest to temp
        sol_idx = np.argmin(np.abs(t_sol - temp))
        comp_sol = x_range[sol_idx]

        if abs(comp_liq - comp_sol) < 0.01:
            self.lever_result.setText("Compositions too close to apply lever rule.")
            return

        # Ensure ordering
        c_alpha = min(comp_sol, comp_liq)
        c_beta = max(comp_sol, comp_liq)

        if comp < c_alpha or comp > c_beta:
            self.lever_result.setText("Point outside tie-line range.")
            return

        f_liquid = (comp - c_alpha) / (c_beta - c_alpha)
        f_solid = 1.0 - f_liquid

        result_text = (
            f"At {comp:.1f}% B, {temp:.0f}\u00b0C:\n"
            f"Liquid comp: {comp_liq:.1f}%  Solid comp: {comp_sol:.1f}%\n"
            f"Fraction liquid: {f_liquid:.3f}  Fraction solid: {f_solid:.3f}"
        )
        self.lever_result.setText(result_text)
        self._emit_log(f"Lever rule: {result_text}")

        # Draw tie line at this temperature
        self._draw_binary()
        self.ax_binary.plot([c_alpha, c_beta], [temp, temp], "k--", linewidth=1.5)
        self.ax_binary.plot(comp, temp, "k+", markersize=15, markeredgewidth=2)
        self.ax_binary.plot(comp_liq, temp, "ro", markersize=8)
        self.ax_binary.plot(comp_sol, temp, "bs", markersize=8)
        self.canvas_binary.draw_idle()

    # ----------------------------------------------------------- Tie Line
    def _draw_tie_line(self):
        temp = self.tie_temp.value()
        x_range = np.linspace(0, 100, 5000)
        t_liq = self._liquidus_interp(x_range)
        t_sol = self._solidus_interp(x_range)

        liq_idx = np.argmin(np.abs(t_liq - temp))
        sol_idx = np.argmin(np.abs(t_sol - temp))
        comp_liq = x_range[liq_idx]
        comp_sol = x_range[sol_idx]

        self._draw_binary()
        self.ax_binary.plot([comp_sol, comp_liq], [temp, temp],
                            "k-", linewidth=2, label=f"Tie @ {temp:.0f}\u00b0C")
        self.ax_binary.plot(comp_liq, temp, "ro", markersize=8)
        self.ax_binary.plot(comp_sol, temp, "bs", markersize=8)
        self.ax_binary.legend(loc="upper right", fontsize=8)
        self.canvas_binary.draw_idle()
        self._emit_log(f"Tie line drawn at {temp:.0f}\u00b0C: "
                       f"liquid at {comp_liq:.1f}%, solid at {comp_sol:.1f}%")

    # -------------------------------------------------------- Cooling Curve
    def _simulate_cooling(self):
        comp = self.cool_comp.value()
        t_liq = float(self._liquidus_interp(comp))
        t_sol = float(self._solidus_interp(comp))
        t_start = t_liq + 150
        t_end = t_sol - 150

        temps = np.linspace(t_start, t_end, 1000)
        # Simple cooling curve: time vs temperature with plateaus at phase boundaries
        time_vals = np.zeros_like(temps)
        dt_base = 1.0  # seconds per degree
        t_accum = 0.0
        for i in range(1, len(temps)):
            dT = abs(temps[i] - temps[i - 1])
            # Slow down near phase boundaries (latent heat)
            if t_sol - 5 < temps[i] < t_liq + 5:
                factor = 3.0  # two-phase: slower cooling
            else:
                factor = 1.0
            # Extra slowdown at eutectic if applicable
            eut = self._current_system.get("eutectic")
            if eut and abs(temps[i] - eut["temp"]) < 10:
                factor = 5.0
            t_accum += dT * dt_base * factor
            time_vals[i] = t_accum

        ax = self.ax_cool
        ax.clear()
        ax.plot(time_vals, temps, "b-", linewidth=1.5)
        ax.axhline(t_liq, color="r", linestyle="--", linewidth=1, label=f"Liquidus {t_liq:.0f}\u00b0C")
        ax.axhline(t_sol, color="b", linestyle="--", linewidth=1, label=f"Solidus {t_sol:.0f}\u00b0C")
        if self._current_system.get("eutectic"):
            eut = self._current_system["eutectic"]
            ax.axhline(eut["temp"], color="g", linestyle=":", linewidth=1,
                       label=f"Eutectic {eut['temp']}\u00b0C")
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Temperature (\u00b0C)", fontsize=10)
        ax.set_title(f"Cooling Curve at {comp:.1f}% B", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
        self.canvas_cool.draw_idle()

        # Also draw vertical line on binary diagram
        self._draw_binary()
        self.ax_binary.axvline(comp, color="orange", linewidth=1.5, linestyle="--",
                               label=f"Cooling @ {comp:.1f}%")
        self.ax_binary.legend(loc="upper right", fontsize=8)
        self.canvas_binary.draw_idle()

        self._emit_log(f"Cooling curve simulated at {comp:.1f}% B. "
                       f"Liquidus: {t_liq:.0f}\u00b0C, Solidus: {t_sol:.0f}\u00b0C")

    # ----------------------------------------------------------- Gibbs
    def _calc_gibbs(self):
        c = self.gibbs_c.value()
        p = self.gibbs_p.value()
        f = c - p + 2
        self.gibbs_result.setText(f"F = {c} - {p} + 2 = {f}")
        self._emit_log(f"Gibbs phase rule: F = {c} - {p} + 2 = {f}")

    # --------------------------------------------------------- Run / Export
    def run(self):
        self._draw_binary()
        self._draw_ternary()
        self._emit_log("Phase diagram refreshed.")

    def export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Phase Diagram", "",
            "PNG Image (*.png);;CSV Data (*.csv)"
        )
        if not path:
            return

        if path.lower().endswith(".png"):
            idx = self.tabs.currentIndex()
            figs = [self.fig_binary, self.fig_cool, self.fig_ternary]
            if 0 <= idx < len(figs):
                figs[idx].savefig(path, dpi=150, bbox_inches="tight")
            self._emit_log(f"Diagram exported to {path}")

        elif path.lower().endswith(".csv"):
            x = np.linspace(0, 100, 500)
            t_liq = self._liquidus_interp(x)
            t_sol = self._solidus_interp(x)
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Composition (%B)", "Liquidus (C)", "Solidus (C)"])
                for i in range(len(x)):
                    writer.writerow([f"{x[i]:.2f}", f"{t_liq[i]:.1f}", f"{t_sol[i]:.1f}"])
            self._emit_log(f"Phase data exported to {path}")
