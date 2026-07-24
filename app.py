"""Axiom Scientific Suite - Main Application Window"""
import sys
import os
import json
import traceback
import logging
from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QAction, QMenuBar, QStatusBar, QToolBar,
    QFileDialog, QMessageBox, QLabel, QDockWidget, QTextEdit, QWidget,
    QVBoxLayout, QHBoxLayout, QSplitter, QApplication, QStyleFactory,
    QDialog, QLineEdit, QListWidget, QListWidgetItem, QShortcut
)
from PyQt5.QtCore import Qt, QTimer, QSize, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QKeySequence

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

RECENT_FILES_PATH = os.path.join(os.path.expanduser("~"), ".axiom_recent.json")


class DetachedWindow(QMainWindow):
    """A window that holds a tab detached from the main tab widget."""
    closed = pyqtSignal(QWidget, str)  # widget, tab_name

    def __init__(self, widget, tab_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Axiom - {tab_name}")
        self._widget = widget
        self._tab_name = tab_name
        self.setCentralWidget(widget)
        self.resize(1000, 700)

    def closeEvent(self, event):
        self.closed.emit(self._widget, self._tab_name)
        super().closeEvent(event)


class CommandPaletteDialog(QDialog):
    """Floating command palette (Ctrl+Shift+P) with fuzzy search."""

    def __init__(self, items, parent=None):
        """items: list of (display_text, callable)"""
        super().__init__(parent, Qt.FramelessWindowHint | Qt.Popup)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._items = items

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        container = QWidget()
        container.setObjectName("palette_container")
        container.setStyleSheet("""
            #palette_container {
                background: rgba(30, 30, 30, 240);
                border: 1px solid #555;
                border-radius: 10px;
            }
            QLineEdit {
                background: #2a2a2a; color: #eee; border: 1px solid #444;
                border-radius: 6px; padding: 8px 12px; font-size: 14px;
            }
            QListWidget {
                background: transparent; color: #ddd; border: none;
                font-size: 13px;
            }
            QListWidget::item { padding: 6px 12px; border-radius: 4px; }
            QListWidget::item:selected { background: #3a6fbf; }
            QListWidget::item:hover { background: #333; }
        """)

        inner = QVBoxLayout(container)
        inner.setContentsMargins(12, 12, 12, 12)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Type a command...")
        self._search.textChanged.connect(self._filter)
        inner.addWidget(self._search)

        self._list = QListWidget()
        self._list.itemActivated.connect(self._execute)
        inner.addWidget(self._list)

        outer.addWidget(container)
        self.resize(520, 420)

        self._populate("")

    def _populate(self, query):
        self._list.clear()
        words = query.lower().split()
        for display, callback in self._items:
            text_lower = display.lower()
            if all(w in text_lower for w in words):
                item = QListWidgetItem(display)
                item.setData(Qt.UserRole, callback)
                self._list.addItem(item)
        if self._list.count():
            self._list.setCurrentRow(0)

    def _filter(self, text):
        self._populate(text)

    def _execute(self, item):
        cb = item.data(Qt.UserRole)
        self.accept()
        if callable(cb):
            cb()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.reject()
        elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
            cur = self._list.currentItem()
            if cur:
                self._execute(cur)
        elif event.key() == Qt.Key_Down:
            row = self._list.currentRow()
            if row < self._list.count() - 1:
                self._list.setCurrentRow(row + 1)
        elif event.key() == Qt.Key_Up:
            row = self._list.currentRow()
            if row > 0:
                self._list.setCurrentRow(row - 1)
        else:
            super().keyPressEvent(event)


class QuantumResMainWindow(QMainWindow):
    def __init__(self, splash=None):
        super().__init__()
        self.setWindowTitle("Axiom Scientific Suite")
        self.setMinimumSize(1400, 900)
        self.showMaximized()

        self._modules = {}
        self._shared_data = {}
        self._recent_files = self._load_recent_files()
        self._detached_windows = []
        self._init_menubar()
        self._init_toolbar()
        self._init_tabs()
        self._init_statusbar()
        self._init_log_dock()
        self._load_modules(splash=splash)

        # Command palette shortcut
        palette_shortcut = QShortcut(QKeySequence("Ctrl+Shift+P"), self)
        palette_shortcut.activated.connect(self._show_command_palette)

        # Detach tab shortcut
        detach_shortcut = QShortcut(QKeySequence("Ctrl+D"), self)
        detach_shortcut.activated.connect(self._detach_current_tab)

        # Memory usage timer
        self._mem_timer = QTimer(self)
        self._mem_timer.timeout.connect(self._update_memory_label)
        self._mem_timer.start(5000)
        self._update_memory_label()

    def _init_menubar(self):
        mb = self.menuBar()

        file_menu = mb.addMenu("&File")
        file_menu.addAction(self._action("New Project", "Ctrl+N", self._new_project))
        file_menu.addAction(self._action("Open File", "Ctrl+O", self._open_file))
        file_menu.addAction(self._action("Save", "Ctrl+S", self._save))
        file_menu.addAction(self._action("Save As...", "Ctrl+Shift+S", self._save_as))
        file_menu.addSeparator()
        file_menu.addAction(self._action("Import Data...", "Ctrl+I", self._import_data))
        file_menu.addAction(self._action("Export...", "Ctrl+E", self._export))
        file_menu.addSeparator()
        file_menu.addAction(self._action("Save Project...", "", self._save_project))
        file_menu.addAction(self._action("Load Project...", "", self._load_project))
        file_menu.addSeparator()
        self._recent_menu = file_menu.addMenu("Recent Files")
        self._rebuild_recent_menu()
        file_menu.addSeparator()
        file_menu.addAction(self._action("Exit", "Ctrl+Q", self.close))

        edit_menu = mb.addMenu("&Edit")
        edit_menu.addAction(self._action("Undo", "Ctrl+Z", self._undo))
        edit_menu.addAction(self._action("Redo", "Ctrl+Y", self._redo))
        edit_menu.addSeparator()
        edit_menu.addAction(self._action("Cut", "Ctrl+X", self._cut))
        edit_menu.addAction(self._action("Copy", "Ctrl+C", self._copy))
        edit_menu.addAction(self._action("Paste", "Ctrl+V", self._paste))
        edit_menu.addSeparator()
        edit_menu.addAction(self._action("Preferences...", "", self._preferences))

        view_menu = mb.addMenu("&View")
        view_menu.addAction(self._action("Toggle Log Panel", "F12", self._toggle_log))
        view_menu.addAction(self._action("Full Screen", "F11", self._toggle_fullscreen))
        view_menu.addAction(self._action("Detach Tab", "Ctrl+D", self._detach_current_tab))
        view_menu.addSeparator()

        # Theme submenu
        from themes import THEMES, get_stylesheet, get_palette
        theme_menu = view_menu.addMenu("Color Theme")
        for theme_name, theme_data in THEMES.items():
            desc = theme_data.get("description", "")
            act = QAction(f"{theme_name}", self)
            act.setToolTip(desc)
            act.triggered.connect(lambda checked, tn=theme_name: self._apply_theme(tn))
            theme_menu.addAction(act)

        tools_menu = mb.addMenu("&Tools")
        tools_menu.addAction(self._action("Unit Converter", "", lambda: self.tabs.setCurrentIndex(self._module_index("Unit Converter"))))
        tools_menu.addAction(self._action("Periodic Table", "", lambda: self.tabs.setCurrentIndex(self._module_index("Periodic Table"))))
        tools_menu.addAction(self._action("Constants Database", "", lambda: self.tabs.setCurrentIndex(self._module_index("Constants DB"))))
        tools_menu.addSeparator()
        tools_menu.addAction(self._action("Data Bus", "", self._show_data_bus))

        sim_menu = mb.addMenu("&Simulation")
        sim_menu.addAction(self._action("FEM Solver", "", lambda: self.tabs.setCurrentIndex(self._module_index("FEM Solver"))))
        sim_menu.addAction(self._action("CFD Simulator", "", lambda: self.tabs.setCurrentIndex(self._module_index("CFD Simulator"))))
        sim_menu.addAction(self._action("EM Simulator", "", lambda: self.tabs.setCurrentIndex(self._module_index("EM Simulator"))))
        sim_menu.addAction(self._action("Circuit Simulator", "", lambda: self.tabs.setCurrentIndex(self._module_index("Circuit Sim"))))
        sim_menu.addAction(self._action("Optics Simulator", "", lambda: self.tabs.setCurrentIndex(self._module_index("Optics Sim"))))

        analysis_menu = mb.addMenu("&Analysis")
        analysis_menu.addAction(self._action("Data Analysis", "", lambda: self.tabs.setCurrentIndex(self._module_index("Data Analysis"))))
        analysis_menu.addAction(self._action("Statistics", "", lambda: self.tabs.setCurrentIndex(self._module_index("Statistics"))))
        analysis_menu.addAction(self._action("Curve Fitting", "", lambda: self.tabs.setCurrentIndex(self._module_index("Curve Fitting"))))
        analysis_menu.addAction(self._action("Signal Processing", "", lambda: self.tabs.setCurrentIndex(self._module_index("Signal Processing"))))

        help_menu = mb.addMenu("&Help")
        help_menu.addAction(self._action("Documentation", "F1", self._show_help))
        help_menu.addAction(self._action("About", "", self._show_about))

    def _action(self, name, shortcut, handler):
        a = QAction(name, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.triggered.connect(handler)
        return a

    def _init_toolbar(self):
        tb = QToolBar("Main Toolbar")
        tb.setIconSize(QSize(20, 20))
        tb.setMovable(False)
        self.addToolBar(tb)

        for label, slot in [
            ("New", self._new_project), ("Open", self._open_file),
            ("Save", self._save), ("|", None),
            ("Import", self._import_data), ("Export", self._export), ("|", None),
            ("Run", self._run_current),
        ]:
            if label == "|":
                tb.addSeparator()
            else:
                btn = QAction(label, self)
                btn.triggered.connect(slot)
                tb.addAction(btn)

    def _init_tabs(self):
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.West)
        self.tabs.setMovable(True)
        self.tabs.setTabsClosable(False)
        self.tabs.setDocumentMode(True)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tabs)

    def _init_statusbar(self):
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status_label = QLabel("Ready")
        self.memory_label = QLabel("")
        self.module_label = QLabel("")
        self.module_count_label = QLabel("")
        self.status.addWidget(self.status_label, 1)
        self.status.addPermanentWidget(self.memory_label)
        self.status.addPermanentWidget(self.module_label)
        self.status.addPermanentWidget(self.module_count_label)

    def _init_log_dock(self):
        self.log_dock = QDockWidget("Output Log", self)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setMaximumHeight(200)
        self.log_dock.setWidget(self.log_text)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
        self.log_dock.hide()

    def log(self, msg, level="info"):
        colors = {"info": "#aaa", "success": "#6f6", "warning": "#ff6", "error": "#f66"}
        color = colors.get(level, "#aaa")
        self.log_text.append(f'<span style="color:{color}">[{level.upper()}] {msg}</span>')
        self.status_label.setText(msg)

    # Module definitions: (tab_name, module_path, class_name) or (tab_name, factory_func)
    MODULE_DEFS = [
        ("Dashboard", None, None),
        ("Python Console", "modules.python_console", "PythonConsoleWidget"),
        ("Math Engine", "modules.math_engine", "MathEngineWidget"),
        ("2D Plotter", "modules.plotter2d", "Plotter2DWidget"),
        ("3D Plotter", "modules.plotter3d", "Plotter3DWidget"),
        ("Data Analysis", "modules.data_analysis", "DataAnalysisWidget"),
        ("Statistics", "modules.statistics_mod", "StatisticsWidget"),
        ("Curve Fitting", "modules.curve_fitting", "CurveFittingWidget"),
        ("Signal Processing", "modules.signal_proc", "SignalProcessingWidget"),
        ("Image Processor", "modules.image_processor", "ImageProcessorWidget"),
        ("Spectroscopy", "modules.spectroscopy", "SpectroscopyWidget"),
        ("Crystal Viewer", "modules.crystal_viewer", "CrystalViewerWidget"),
        ("Molecule Viewer", "modules.molecule_viewer", "MoleculeViewerWidget"),
        ("FEM Solver", "modules.fem_solver", "FEMSolverWidget"),
        ("CFD Simulator", "modules.cfd_simulator", "CFDSimulatorWidget"),
        ("EM Simulator", "modules.em_simulator", "EMSimulatorWidget"),
        ("Circuit Sim", "modules.circuit_sim", "CircuitSimWidget"),
        ("Optics Sim", "modules.optics_sim", "OpticsSimWidget"),
        ("Quantum Sim", "modules.quantum_sim", "QuantumSimWidget"),
        ("2D CAD", "modules.cad2d", "CAD2DWidget"),
        ("3D CAD", "modules.cad3d", "CAD3DWidget"),
        ("IC Layout", "modules.ic_layout", "ICLayoutWidget"),
        ("LaTeX Editor", "modules.latex_editor", "LaTeXEditorWidget"),
        ("PDF Tools", "modules.pdf_tools", "PDFToolsWidget"),
        ("Notebook", "modules.notebook", "NotebookWidget"),
        ("AI / ML", "modules.ml_tools", "MLToolsWidget"),
        ("Genomics", "modules.genomics", "GenomicsWidget"),
        ("GIS / Mapping", "modules.gis_module", "GISWidget"),
        ("Fractal Explorer", "modules.fractal_explorer", "FractalExplorerWidget"),
        ("Graphing Calc", "modules.graphing_calc", "GraphingCalcWidget"),
        ("Waveform Gen", "modules.waveform_gen", "WaveformGenWidget"),
        ("Formula Ref", "modules.formula_ref", "FormulaRefWidget"),
        ("Phase Diagrams", "modules.phase_diagram", "PhaseDiagramWidget"),
        ("Color Science", "modules.color_science", "ColorScienceWidget"),
        ("Coord Transforms", "modules.coord_transforms", "CoordTransformsWidget"),
        ("Tensor Calc", "modules.tensor_calc", "TensorCalcWidget"),
        ("Control Systems", "modules.control_systems", "ControlSystemsWidget"),
        ("Thermo Props", "modules.thermo_props", "ThermoPropsWidget"),
        ("Power Systems", "modules.power_systems", "PowerSystemsWidget"),
        ("Acoustics", "modules.acoustics", "AcousticsWidget"),
        ("Dim. Analysis", "modules.dimensional_analysis", "DimensionalAnalysisWidget"),
        ("Periodic Table", "modules.periodic_table", "PeriodicTableWidget"),
        ("Constants DB", "modules.constants_db", "ConstantsDBWidget"),
        ("Unit Converter", "modules.unit_converter", "UnitConverterWidget"),
    ]

    def _load_modules(self, splash=None):
        """Create lazy-loading placeholder tabs for all modules.
        Dashboard loads immediately; everything else loads on first tab click."""
        total = len(self.MODULE_DEFS)
        for i, (name, mod_path, cls_name) in enumerate(self.MODULE_DEFS):
            if splash:
                splash.showMessage(
                    f"  Preparing {name}...  ({i+1}/{total})",
                    Qt.AlignBottom | Qt.AlignHCenter,
                    QApplication.instance().palette().color(
                        QApplication.instance().palette().WindowText))
                QApplication.processEvents()

            if mod_path is None:
                # Dashboard loads immediately (it's lightweight)
                try:
                    widget = self._make_dashboard()
                    self.tabs.addTab(widget, name)
                    self._modules[name] = widget
                except Exception as e:
                    self.tabs.addTab(self._make_error_tab(name, str(e)), f"{name} (!)")
            else:
                # Create a lazy placeholder
                placeholder = self._make_lazy_tab(name, mod_path, cls_name)
                self.tabs.addTab(placeholder, name)

    def _make_lazy_tab(self, name, mod_path, cls_name):
        """Create a placeholder widget that loads the real module on first display."""
        w = QWidget()
        w._axiom_lazy = True
        w._axiom_mod_path = mod_path
        w._axiom_cls_name = cls_name
        w._axiom_tab_name = name
        layout = QVBoxLayout(w)
        lbl = QLabel(f"Loading {name}...")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("color: #888; font-size: 16px; padding: 60px;")
        layout.addWidget(lbl)
        return w

    def _on_tab_changed(self, index):
        """When a tab is selected, load it if it's still a lazy placeholder."""
        widget = self.tabs.widget(index)
        name = self.tabs.tabText(index)

        if hasattr(widget, '_axiom_lazy') and widget._axiom_lazy:
            # Replace placeholder with real module
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                real_widget = self._load_module(widget._axiom_mod_path, widget._axiom_cls_name)
                tab_name = widget._axiom_tab_name
                self.tabs.removeTab(index)
                self.tabs.insertTab(index, real_widget, tab_name)
                self.tabs.setCurrentIndex(index)
                self._modules[tab_name] = real_widget
                logging.info(f"Lazy-loaded: {tab_name}")
            except Exception as e:
                err = self._make_error_tab(name, str(e))
                self.tabs.removeTab(index)
                self.tabs.insertTab(index, err, f"{name} (!)")
                self.tabs.setCurrentIndex(index)
                logging.error(f"Error lazy-loading {name}: {e}")
                logging.error(traceback.format_exc())
            finally:
                QApplication.restoreOverrideCursor()
        else:
            self.module_label.setText(name.replace(" (!)", ""))
            self.status_label.setText(f"Active: {name}")

    def _load_module(self, module_path, class_name):
        import importlib
        logging.info(f"Loading module: {module_path}.{class_name}")
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        widget = cls()
        if hasattr(widget, 'set_logger'):
            widget.set_logger(self.log)
        return widget

    def _make_error_tab(self, name, error):
        w = QWidget()
        layout = QVBoxLayout(w)
        lbl = QLabel(f"Module '{name}' failed to load:\n\n{error}\n\n"
                     "This module may require additional dependencies.")
        lbl.setWordWrap(True)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("color: #f66; font-size: 14px; padding: 40px;")
        layout.addWidget(lbl)
        return w

    def _make_dashboard(self):
        from modules.dashboard import DashboardWidget
        return DashboardWidget(self)

    def _module_index(self, name):
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i).replace(" (!)", "") == name:
                return i
        return 0

    # _on_tab_changed defined above with lazy-loading support

    # File operations
    def _new_project(self):
        self.log("New project created")

    def _open_file(self):
        try:
            from file_formats import get_import_filter
            filt = get_import_filter()
        except ImportError:
            filt = "All Files (*)"
        path, _ = QFileDialog.getOpenFileName(self, "Open File", "", filt)
        if path:
            self.log(f"Opened: {path}")
            self._add_recent_file(path)
            self._route_file(path)

    def _route_file(self, path):
        try:
            from file_formats import route_file
            target = route_file(path)
        except ImportError:
            ext = path.lower().rsplit('.', 1)[-1] if '.' in path else ''
            routing = {
                'py': 'Python Console', 'csv': 'Data Analysis', 'dat': 'Data Analysis',
                'txt': 'Data Analysis', 'xlsx': 'Data Analysis', 'xls': 'Data Analysis',
                'png': 'Image Processor', 'jpg': 'Image Processor', 'jpeg': 'Image Processor',
                'tif': 'Image Processor', 'tiff': 'Image Processor', 'bmp': 'Image Processor',
                'cif': 'Crystal Viewer', 'xyz': 'Molecule Viewer', 'pdb': 'Molecule Viewer',
                'mol': 'Molecule Viewer', 'mol2': 'Molecule Viewer',
                'tex': 'LaTeX Editor', 'pdf': 'PDF Tools',
                'stl': '3D CAD', 'obj': '3D CAD', 'dxf': '2D CAD',
                'gds': 'IC Layout', 'fasta': 'Genomics', 'fa': 'Genomics',
            }
            target = routing.get(ext, 'Python Console')
        idx = self._module_index(target)
        self.tabs.setCurrentIndex(idx)
        widget = self.tabs.widget(idx)
        if hasattr(widget, 'load_file'):
            widget.load_file(path)

    def _save(self):
        current = self.tabs.currentWidget()
        if hasattr(current, 'save'):
            current.save()
        self.log("Saved")

    def _save_as(self):
        """Universal Save As with format-appropriate options."""
        tab_name = self.tabs.tabText(self.tabs.currentIndex()).replace(' (!)', '')
        # Map tab categories to export format groups
        format_map = {
            'Data Analysis': 'data', '2D Plotter': 'plot', '3D Plotter': 'plot',
            'Curve Fitting': 'plot', 'Signal Processing': 'plot', 'Statistics': 'plot',
            'Image Processor': 'image', 'Crystal Viewer': 'structure',
            'Molecule Viewer': 'structure', '2D CAD': 'cad2d', '3D CAD': 'cad3d',
            'IC Layout': 'layout', 'LaTeX Editor': 'document', 'PDF Tools': 'document',
            'Notebook': 'document', 'Genomics': 'sequence', 'Circuit Sim': 'circuit',
            'FEM Solver': 'data', 'CFD Simulator': 'data', 'EM Simulator': 'data',
            'Fractal Explorer': 'plot', 'GIS / Mapping': 'plot', 'ML / AI': 'data',
        }
        cat = format_map.get(tab_name, 'plot')
        try:
            from file_formats import get_export_filter
            filt = get_export_filter(cat)
        except ImportError:
            filt = "All Files (*)"
        path, selected_filter = QFileDialog.getSaveFileName(self, f"Save As — {tab_name}", "", filt)
        if path:
            current = self.tabs.currentWidget()
            if hasattr(current, 'save_as'):
                current.save_as(path)
            elif hasattr(current, 'export'):
                current.export(path)
            else:
                # Fallback: try to save any figure as image
                if hasattr(current, '_figure'):
                    current._figure.savefig(path, dpi=300, bbox_inches='tight',
                                           facecolor=current._figure.get_facecolor())
                else:
                    self.log("Current module doesn't support Save As", "warning")
                    return
            self.log(f"Saved as: {path}", "success")

    def _import_data(self):
        try:
            from file_formats import get_import_filter
            filt = get_import_filter()
        except ImportError:
            filt = "Data Files (*.csv *.tsv *.dat *.txt *.xlsx *.xls *.json)"
        path, _ = QFileDialog.getOpenFileName(self, "Import Data", "", filt)
        if path:
            self._add_recent_file(path)
            self._route_file(path)

    def _export(self):
        """Quick export using the module's built-in export."""
        current = self.tabs.currentWidget()
        if hasattr(current, 'export'):
            current.export()
        else:
            # Fallback to Save As
            self._save_as()

    def _undo(self):
        w = QApplication.focusWidget()
        if hasattr(w, 'undo'):
            w.undo()

    def _redo(self):
        w = QApplication.focusWidget()
        if hasattr(w, 'redo'):
            w.redo()

    def _cut(self):
        w = QApplication.focusWidget()
        if hasattr(w, 'cut'):
            w.cut()

    def _copy(self):
        w = QApplication.focusWidget()
        if hasattr(w, 'copy'):
            w.copy()

    def _paste(self):
        w = QApplication.focusWidget()
        if hasattr(w, 'paste'):
            w.paste()

    def _run_current(self):
        current = self.tabs.currentWidget()
        if hasattr(current, 'run'):
            current.run()
        else:
            self.log("Current module doesn't support run", "warning")

    def _preferences(self):
        QMessageBox.information(self, "Preferences", "Preferences dialog - configure modules and settings.")

    def _toggle_log(self):
        self.log_dock.setVisible(not self.log_dock.isVisible())

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showMaximized()
        else:
            self.showFullScreen()

    def _show_help(self):
        QMessageBox.information(self, "Help",
            "Axiom Scientific Suite\n\n"
            "A comprehensive scientific computing environment combining:\n"
            "- Mathematical computing (MATLAB/Octave/Mathematica)\n"
            "- Data analysis & statistics (Origin/R/SPSS)\n"
            "- 2D/3D plotting & visualization (ParaView/gnuplot)\n"
            "- Image processing (Fiji/ImageJ/Gwyddion)\n"
            "- Crystal & molecular visualization (VESTA/XCrysDen/Chimera)\n"
            "- FEM/FEA simulation (COMSOL/ANSYS/FEMM)\n"
            "- CFD simulation (OpenFOAM)\n"
            "- Electromagnetic simulation\n"
            "- Circuit simulation (SPICE)\n"
            "- Optics simulation (Zemax)\n"
            "- Quantum mechanics simulation (Quantum ESPRESSO)\n"
            "- 2D/3D CAD (AutoCAD/SolidWorks)\n"
            "- IC Layout (KLayout)\n"
            "- LaTeX document preparation\n"
            "- PDF tools (GhostScript)\n"
            "- AI/Machine Learning\n"
            "- Genomics & bioinformatics\n"
            "- GIS & mapping\n"
            "- Spectroscopy analysis\n"
            "- Signal processing\n"
            "- And much more!\n\n"
            "Use Ctrl+O to open files or use the sidebar tabs.")

    def _show_about(self):
        QMessageBox.about(self, "About Axiom",
            "Axiom Scientific Suite v1.0\n\n"
            "The universal scientific computing platform.\n"
            "All-in-one replacement for commercial and open-source\n"
            "scientific software packages.\n\n"
            "Built with Python, NumPy, SciPy, Matplotlib, VTK,\n"
            "scikit-learn, scikit-image, SymPy, and more.")

    def _apply_theme(self, theme_name):
        """Switch the entire application to a different color theme."""
        from themes import get_stylesheet, get_palette
        app = QApplication.instance()
        app.setPalette(get_palette(theme_name))
        app.setStyleSheet(get_stylesheet(theme_name))
        app._current_theme = theme_name
        self.log(f"Theme changed to: {theme_name}", "info")

    # ── Project Save/Load ─────────────────────────────────────────────

    def _save_project(self):
        """Save current workspace state to a .axiom project file."""
        path, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "Axiom Project (*.axiom)")
        if not path:
            return
        project = {
            "version": "1.0",
            "active_tab": self.tabs.currentIndex(),
            "theme": getattr(QApplication.instance(), '_current_theme', 'Axiom Dark'),
            "window_geometry": [self.x(), self.y(), self.width(), self.height()],
            "shared_data_keys": list(self._shared_data.keys()),
        }
        try:
            with open(path, 'w') as f:
                json.dump(project, f, indent=2)
            self.log(f"Project saved: {path}", "success")
        except Exception as e:
            self.log(f"Failed to save project: {e}", "error")

    def _load_project(self):
        """Load workspace state from a .axiom project file."""
        path, _ = QFileDialog.getOpenFileName(self, "Load Project", "", "Axiom Project (*.axiom)")
        if not path:
            return
        try:
            with open(path, 'r') as f:
                project = json.load(f)
            # Restore theme
            theme = project.get("theme", "Axiom Dark")
            self._apply_theme(theme)
            # Restore active tab
            tab_idx = project.get("active_tab", 0)
            if 0 <= tab_idx < self.tabs.count():
                self.tabs.setCurrentIndex(tab_idx)
            # Restore window geometry
            geom = project.get("window_geometry")
            if geom and len(geom) == 4:
                self.setGeometry(*geom)
            self.log(f"Project loaded: {path}", "success")
        except Exception as e:
            self.log(f"Failed to load project: {e}", "error")

    # ── Recent Files ──────────────────────────────────────────────────

    def _load_recent_files(self):
        try:
            with open(RECENT_FILES_PATH, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data[:10]
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        return []

    def _save_recent_files(self):
        try:
            with open(RECENT_FILES_PATH, "w") as f:
                json.dump(self._recent_files, f)
        except OSError:
            pass

    def _add_recent_file(self, path):
        path = os.path.normpath(path)
        if path in self._recent_files:
            self._recent_files.remove(path)
        self._recent_files.insert(0, path)
        self._recent_files = self._recent_files[:10]
        self._save_recent_files()
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self):
        self._recent_menu.clear()
        for p in self._recent_files:
            act = QAction(p, self)
            act.triggered.connect(lambda checked, fp=p: self._route_file(fp))
            self._recent_menu.addAction(act)
        if self._recent_files:
            self._recent_menu.addSeparator()
        clear_act = QAction("Clear Recent", self)
        clear_act.triggered.connect(self._clear_recent_files)
        self._recent_menu.addAction(clear_act)

    def _clear_recent_files(self):
        self._recent_files.clear()
        self._save_recent_files()
        self._rebuild_recent_menu()

    # ── Status Bar Memory ─────────────────────────────────────────────

    def _update_memory_label(self):
        try:
            if _HAS_PSUTIL:
                proc = psutil.Process(os.getpid())
                mem_mb = proc.memory_info().rss / (1024 * 1024)
            else:
                # Fallback: rough estimate via sys.getsizeof is not useful;
                # report only what we can.
                import resource
                mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except Exception:
            mem_mb = 0
        self.memory_label.setText(f"  Mem: {mem_mb:.0f} MB  ")

    # ── Command Palette ───────────────────────────────────────────────

    def _show_command_palette(self):
        from themes import THEMES
        items = []

        # All tab names
        for i in range(self.tabs.count()):
            name = self.tabs.tabText(i).replace(" (!)", "")
            idx = i
            items.append((f"Go to: {name}", lambda _i=idx: self.tabs.setCurrentIndex(_i)))

        # Common operations
        ops = [
            ("Open File", self._open_file),
            ("Save", self._save),
            ("Save As...", self._save_as),
            ("Export", self._export),
            ("Import Data...", self._import_data),
            ("Run", self._run_current),
            ("Preferences", self._preferences),
            ("Full Screen", self._toggle_fullscreen),
            ("Toggle Log", self._toggle_log),
            ("New Project", self._new_project),
            ("Detach Tab", self._detach_current_tab),
            ("Documentation", self._show_help),
            ("About", self._show_about),
            ("Undo", self._undo),
            ("Redo", self._redo),
        ]
        items.extend(ops)

        # Tools
        for tool_name in ["Unit Converter", "Periodic Table", "Constants DB"]:
            idx = self._module_index(tool_name)
            items.append((tool_name, lambda _i=idx: self.tabs.setCurrentIndex(_i)))

        # Simulations
        for sim_name in ["FEM Solver", "CFD Simulator", "EM Simulator",
                         "Circuit Sim", "Optics Sim", "Quantum Sim"]:
            idx = self._module_index(sim_name)
            items.append((sim_name, lambda _i=idx: self.tabs.setCurrentIndex(_i)))

        # Themes
        for theme_name in THEMES:
            items.append((f"Theme: {theme_name}",
                          lambda tn=theme_name: self._apply_theme(tn)))

        dlg = CommandPaletteDialog(items, self)
        # Center on the main window
        geo = self.geometry()
        dlg.move(geo.x() + (geo.width() - dlg.width()) // 2,
                 geo.y() + int(geo.height() * 0.2))
        dlg.exec_()

    # ── Detachable Tabs ───────────────────────────────────────────────

    def _detach_current_tab(self):
        idx = self.tabs.currentIndex()
        if idx < 0:
            return
        tab_name = self.tabs.tabText(idx)
        widget = self.tabs.widget(idx)
        self.tabs.removeTab(idx)

        win = DetachedWindow(widget, tab_name, self)
        win.closed.connect(self._reattach_tab)
        self._detached_windows.append(win)
        win.show()
        self.log(f"Detached tab: {tab_name}")

    def _reattach_tab(self, widget, tab_name):
        self.tabs.addTab(widget, tab_name)
        self.tabs.setCurrentWidget(widget)
        # Clean up reference
        self._detached_windows = [
            w for w in self._detached_windows if w._tab_name != tab_name
        ]
        self.log(f"Reattached tab: {tab_name}")

    # ------------------------------------------------------------------
    # Shared Data Bus
    # ------------------------------------------------------------------

    def share_data(self, key, data):
        """Store data in the shared data bus with the given key."""
        self._shared_data[key] = data

    def get_shared_data(self, key):
        """Retrieve data from the shared data bus by key."""
        return self._shared_data.get(key)

    def list_shared_data(self):
        """Return a list of all available keys in the shared data bus."""
        return list(self._shared_data.keys())

    def _show_data_bus(self):
        """Show a dialog listing all shared data items."""
        from PyQt5.QtWidgets import (
            QDialog, QVBoxLayout, QListWidget, QDialogButtonBox, QLabel
        )
        dlg = QDialog(self)
        dlg.setWindowTitle("Data Bus")
        dlg.setMinimumSize(400, 300)
        layout = QVBoxLayout(dlg)

        layout.addWidget(QLabel("Shared data items available across modules:"))

        lst = QListWidget()
        keys = self.list_shared_data()
        if keys:
            for k in keys:
                val = self._shared_data[k]
                type_str = type(val).__name__
                try:
                    size_info = f" ({len(val)} items)" if hasattr(val, '__len__') else ""
                except Exception:
                    size_info = ""
                lst.addItem(f"{k}  [{type_str}{size_info}]")
        else:
            lst.addItem("(no shared data)")
        layout.addWidget(lst)

        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        dlg.exec_()
