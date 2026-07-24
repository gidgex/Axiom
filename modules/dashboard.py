"""Dashboard - Home screen with module overview and quick access."""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout, QPushButton,
    QScrollArea, QFrame, QGroupBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class DashboardWidget(QWidget):
    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self._init_ui()

    def _init_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(20)
        layout.setContentsMargins(40, 30, 40, 30)

        # Title
        title = QLabel("Axiom Scientific Suite")
        title.setFont(QFont("Segoe UI", 28, QFont.Bold))
        title.setStyleSheet("color: #4a90d9;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Universal Scientific Computing Platform")
        subtitle.setFont(QFont("Segoe UI", 14))
        subtitle.setStyleSheet("color: #888;")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        layout.addSpacing(10)

        # Quick Start tips
        qs = QGroupBox("Quick Start")
        qs.setStyleSheet("""
            QGroupBox { font-size: 14px; font-weight: bold; color: #50c878;
                        border: 1px solid #355; border-radius: 6px; margin-top: 12px; padding-top: 18px; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
        """)
        qs_layout = QVBoxLayout(qs)
        tips = [
            ("Ctrl+Shift+P", "Command Palette — search any module, tool, or action"),
            ("Ctrl+O", "Open any scientific file (auto-routes to the right module)"),
            ("Ctrl+D", "Detach current tab into its own window"),
            ("F11", "Toggle full screen"),
            ("F12", "Toggle output log panel"),
            ("View > Color Theme", "Switch between 6 color themes (Quantum, Hyper Light, etc.)"),
            ("Click a tab", "Modules load on demand — just click to activate"),
        ]
        for shortcut, desc in tips:
            row = QHBoxLayout()
            key = QLabel(shortcut)
            key.setFixedWidth(160)
            key.setStyleSheet("color: #4a90d9; font-family: Consolas; font-weight: bold; font-size: 12px;")
            row.addWidget(key)
            d = QLabel(desc)
            d.setStyleSheet("color: #aaa; font-size: 12px;")
            row.addWidget(d)
            row.addStretch()
            qs_layout.addLayout(row)
        layout.addWidget(qs)

        layout.addSpacing(10)

        # Module categories
        categories = [
            ("Computing & Programming", [
                ("Python Console", "Interactive Python with NumPy, SciPy, SymPy"),
                ("Math Engine", "Matrix algebra, equation solving, symbolic math"),
                ("Notebook", "Jupyter-style computational notebook"),
            ]),
            ("Data Analysis & Statistics", [
                ("Data Analysis", "Import, explore, transform, analyze datasets"),
                ("Statistics", "Hypothesis testing, ANOVA, regression, distributions"),
                ("Curve Fitting", "Nonlinear fitting, peak fitting, model selection"),
                ("Signal Processing", "FFT, filters, wavelets, spectrograms"),
                ("Spectroscopy", "UV-Vis, IR, Raman, NMR, XRD analysis"),
            ]),
            ("Visualization & Plotting", [
                ("2D Plotter", "Publication-quality plots: line, scatter, bar, contour"),
                ("3D Plotter", "Surface, wireframe, volume rendering, isosurfaces"),
                ("Image Processor", "Filters, segmentation, measurements, FFT"),
                ("Fractal Explorer", "Mandelbrot, Julia, Burning Ship, Newton, 3D fractals, custom formulas"),
            ]),
            ("Simulation & Modeling", [
                ("FEM Solver", "Finite element analysis: structural, thermal, modal"),
                ("CFD Simulator", "Navier-Stokes fluid flow simulation"),
                ("EM Simulator", "Electrostatic/magnetostatic field computation"),
                ("Circuit Sim", "SPICE-like electronic circuit simulation"),
                ("Optics Sim", "Ray tracing, diffraction, interference patterns"),
                ("Quantum Sim", "Quantum mechanics: wavefunctions, potentials, bands"),
            ]),
            ("Chemistry & Materials", [
                ("Crystal Viewer", "Unit cells, symmetry, diffraction patterns (VESTA-like)"),
                ("Molecule Viewer", "3D molecular visualization (Chimera-like)"),
                ("Phase Diagrams", "Binary/ternary phase diagrams, lever rule, cooling curves"),
                ("Periodic Table", "Interactive periodic table with element data"),
            ]),
            ("CAD & Design", [
                ("2D CAD", "Technical drawing, dimensions, DXF export"),
                ("3D CAD", "Solid modeling, boolean operations, STL export"),
                ("IC Layout", "Integrated circuit layout editor (KLayout-like)"),
            ]),
            ("Documents & Publishing", [
                ("LaTeX Editor", "Scientific document editor with preview"),
                ("PDF Tools", "View, merge, annotate PDF documents"),
            ]),
            ("Life Sciences & Geoscience", [
                ("Genomics", "Sequence alignment, DNA/RNA analysis, phylogenetics"),
                ("GIS / Mapping", "Geographic data, projections, spatial analysis"),
            ]),
            ("AI / Machine Learning", [
                ("AI / ML", "Classification, regression, clustering, neural networks"),
            ]),
            ("Calculators & References", [
                ("Graphing Calc", "Desmos-like graphing: explicit, implicit, parametric, polar"),
                ("Waveform Gen", "Oscilloscope, Fourier synthesis, Lissajous figures"),
                ("Formula Ref", "Searchable database of 100+ physics/math/chemistry formulas"),
                ("Color Science", "CIE chromaticity, color spaces, color blindness simulation"),
                ("Coord Transforms", "Cartesian/cylindrical/spherical with 3D visualization"),
            ]),
            ("Engineering", [
                ("Control Systems", "Bode plots, root locus, PID tuning, Nyquist diagrams"),
                ("Power Systems", "Load flow, fault analysis, single-line diagrams"),
                ("Thermo Props", "Steam tables, psychrometrics, EOS, thermodynamic cycles"),
                ("Acoustics", "SPL calculator, RT60, octave bands, Helmholtz resonator"),
                ("Tensor Calc", "Einstein summation, Christoffel symbols, stress tensors"),
            ]),
            ("Utilities", [
                ("Unit Converter", "Convert between all scientific units"),
                ("Constants DB", "Physical constants, material properties"),
                ("Dim. Analysis", "Check dimensional consistency, Buckingham Pi, dimensionless numbers"),
            ]),
        ]

        for cat_name, modules in categories:
            group = QGroupBox(cat_name)
            group.setStyleSheet("""
                QGroupBox { font-size: 15px; font-weight: bold; color: #4a90d9;
                            border: 1px solid #444; border-radius: 6px; margin-top: 12px; padding-top: 18px; }
                QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
            """)
            grid = QGridLayout(group)
            grid.setSpacing(8)
            for i, (mod_name, mod_desc) in enumerate(modules):
                btn = QPushButton(f"  {mod_name}")
                btn.setToolTip(mod_desc)
                btn.setMinimumHeight(36)
                btn.setStyleSheet("""
                    QPushButton { text-align: left; padding-left: 12px; font-size: 12px;
                                  background: #2a2a2a; border: 1px solid #444; border-radius: 4px; }
                    QPushButton:hover { background: #3a3a3a; border-color: #4a90d9; }
                """)
                btn.clicked.connect(lambda checked, n=mod_name: self._go_to(n))
                desc = QLabel(mod_desc)
                desc.setStyleSheet("color: #888; font-size: 11px;")
                row = i // 2
                col = (i % 2) * 2
                grid.addWidget(btn, row, col)
                grid.addWidget(desc, row, col + 1)
            layout.addWidget(group)

        layout.addStretch()
        scroll.setWidget(container)

        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.addWidget(scroll)

    def _go_to(self, name):
        if self.main_window:
            idx = self.main_window._module_index(name)
            self.main_window.tabs.setCurrentIndex(idx)
