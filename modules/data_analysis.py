"""
Data Analysis Widget for QuantumRes Scientific Suite.

Provides an Origin/Excel-like data analysis environment with import/export,
column statistics, data operations, plotting, and formula-based computed columns.
"""

import os
import json
import csv
import io
import traceback
import datetime
import tempfile
import webbrowser
from functools import partial

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy import signal

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QFileDialog, QLabel, QComboBox, QLineEdit, QGroupBox,
    QSplitter, QTextEdit, QToolBar, QAction, QMenu, QMessageBox,
    QInputDialog, QHeaderView, QDialog, QFormLayout, QDialogButtonBox,
    QCheckBox, QSpinBox, QTabWidget, QApplication, QProgressBar,
    QAbstractItemView, QSizePolicy
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QIcon, QColor, QFont, QKeySequence

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    from mpl_style import style_figure, style_axes
except ImportError:
    def style_figure(f): pass
    def style_axes(a): pass


# ---------------------------------------------------------------------------
# Helper dialogs
# ---------------------------------------------------------------------------

class FilterDialog(QDialog):
    """Dialog for setting up column filters."""

    def __init__(self, columns, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filter Data")
        self.setMinimumWidth(380)
        layout = QFormLayout(self)

        self.col_combo = QComboBox()
        self.col_combo.addItems(columns)
        layout.addRow("Column:", self.col_combo)

        self.op_combo = QComboBox()
        self.op_combo.addItems(["==", "!=", ">", ">=", "<", "<=", "contains", "not contains", "is null", "not null"])
        layout.addRow("Operator:", self.op_combo)

        self.value_edit = QLineEdit()
        layout.addRow("Value:", self.value_edit)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def get_filter(self):
        return self.col_combo.currentText(), self.op_combo.currentText(), self.value_edit.text()


class ComputedColumnDialog(QDialog):
    """Dialog for creating a computed column via a pandas expression."""

    def __init__(self, columns, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Computed Column")
        self.setMinimumWidth(440)
        layout = QFormLayout(self)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("new_column_name")
        layout.addRow("Column name:", self.name_edit)

        self.expr_edit = QLineEdit()
        self.expr_edit.setPlaceholderText("e.g. col_a + col_b * 2")
        layout.addRow("Expression:", self.expr_edit)

        hint = QLabel("Use column names as variables. Numpy available as np.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666; font-size: 11px;")
        layout.addRow(hint)

        avail = QLabel("Columns: " + ", ".join(columns))
        avail.setWordWrap(True)
        avail.setStyleSheet("color: #888; font-size: 11px;")
        layout.addRow(avail)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def get_values(self):
        return self.name_edit.text().strip(), self.expr_edit.text().strip()


class RenameColumnDialog(QDialog):
    """Dialog for renaming a column."""

    def __init__(self, columns, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Rename Column")
        self.setMinimumWidth(340)
        layout = QFormLayout(self)

        self.col_combo = QComboBox()
        self.col_combo.addItems(columns)
        layout.addRow("Column:", self.col_combo)

        self.new_name_edit = QLineEdit()
        layout.addRow("New name:", self.new_name_edit)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def get_values(self):
        return self.col_combo.currentText(), self.new_name_edit.text().strip()


class PlotDialog(QDialog):
    """Quick plot dialog with embedded matplotlib canvas."""

    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Quick Plot")
        self.resize(720, 540)
        self.df = df

        main_layout = QVBoxLayout(self)

        # Controls
        ctrl = QHBoxLayout()
        self.x_combo = QComboBox()
        self.x_combo.addItem("(index)")
        self.x_combo.addItems(df.columns.tolist())
        ctrl.addWidget(QLabel("X:"))
        ctrl.addWidget(self.x_combo)

        self.y_combo = QComboBox()
        self.y_combo.addItems(df.select_dtypes(include=[np.number]).columns.tolist())
        ctrl.addWidget(QLabel("Y:"))
        ctrl.addWidget(self.y_combo)

        self.plot_type = QComboBox()
        self.plot_type.addItems(["Line", "Scatter", "Bar", "Histogram", "Box"])
        ctrl.addWidget(QLabel("Type:"))
        ctrl.addWidget(self.plot_type)

        plot_btn = QPushButton("Plot")
        plot_btn.clicked.connect(self._do_plot)
        ctrl.addWidget(plot_btn)
        main_layout.addLayout(ctrl)

        # Canvas
        self.figure = Figure(figsize=(7, 4), dpi=100)
        style_figure(self.figure)
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)

        if self.y_combo.count() > 0:
            self._do_plot()

    def _do_plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        y_col = self.y_combo.currentText()
        if not y_col:
            return
        x_col = self.x_combo.currentText()
        x = self.df.index if x_col == "(index)" else self.df[x_col]
        y = self.df[y_col]
        kind = self.plot_type.currentText()

        try:
            if kind == "Line":
                ax.plot(x, y, marker="o", markersize=3, linewidth=1)
            elif kind == "Scatter":
                ax.scatter(x, y, s=12, alpha=0.7)
            elif kind == "Bar":
                ax.bar(range(len(y)), y, tick_label=[str(v) for v in x])
                if len(y) > 20:
                    ax.tick_params(axis="x", rotation=45)
            elif kind == "Histogram":
                ax.hist(y.dropna(), bins="auto", edgecolor="black", alpha=0.75)
            elif kind == "Box":
                ax.boxplot(y.dropna(), vert=True)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{kind}: {y_col}")
            self.figure.tight_layout()
        except Exception as exc:
            ax.text(0.5, 0.5, f"Plot error:\n{exc}", transform=ax.transAxes,
                    ha="center", va="center", color="red")
        self.canvas.draw()


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class DataAnalysisWidget(QWidget):
    """
    Full-featured data analysis widget for a PyQt5 scientific suite.

    Features
    --------
    * Tabular data view with editable cells
    * Import CSV / TSV / Excel / JSON with automatic delimiter detection
    * Column statistics (mean, median, std, min, max, skewness, kurtosis)
    * Data operations: sort, filter, normalize, interpolate, remove duplicates,
      fill missing values
    * Column operations: add, remove, rename, computed (formula) columns
    * Quick-plot dialog (line, scatter, bar, histogram, box)
    * Summary panel with live statistics
    * Export to CSV / Excel
    * Programmatic API: set_logger, load_file, export
    """

    data_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._logger = None
        self._df = pd.DataFrame()
        self._undo_stack: list[pd.DataFrame] = []
        self._max_undo = 30
        self._file_path: str | None = None
        self._init_ui()

    # ------------------------------------------------------------------ UI
    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        tb = QToolBar()
        tb.setIconSize(QSize(18, 18))
        tb.setMovable(False)

        self._act_open = tb.addAction("Open")
        self._act_open.setToolTip("Import CSV / TSV / XLSX / JSON")
        self._act_open.triggered.connect(self._on_open)

        self._act_save = tb.addAction("Export")
        self._act_save.setToolTip("Export data to CSV or Excel")
        self._act_save.triggered.connect(self._on_export)

        tb.addSeparator()

        self._act_undo = tb.addAction("Undo")
        self._act_undo.setShortcut(QKeySequence.Undo)
        self._act_undo.triggered.connect(self._undo)

        tb.addSeparator()

        self._act_add_col = tb.addAction("+ Col")
        self._act_add_col.triggered.connect(self._add_column)

        self._act_rm_col = tb.addAction("- Col")
        self._act_rm_col.triggered.connect(self._remove_column)

        self._act_rename_col = tb.addAction("Rename Col")
        self._act_rename_col.triggered.connect(self._rename_column)

        self._act_computed = tb.addAction("Formula Col")
        self._act_computed.triggered.connect(self._add_computed_column)

        tb.addSeparator()

        self._act_sort = tb.addAction("Sort")
        self._act_sort.triggered.connect(self._sort_data)

        self._act_filter = tb.addAction("Filter")
        self._act_filter.triggered.connect(self._filter_data)

        self._act_dedup = tb.addAction("Dedup")
        self._act_dedup.setToolTip("Remove duplicate rows")
        self._act_dedup.triggered.connect(self._remove_duplicates)

        self._act_fill = tb.addAction("Fill NA")
        self._act_fill.triggered.connect(self._fill_missing)

        self._act_interp = tb.addAction("Interpolate")
        self._act_interp.triggered.connect(self._interpolate)

        self._act_norm = tb.addAction("Normalize")
        self._act_norm.triggered.connect(self._normalize)

        tb.addSeparator()

        self._act_plot = tb.addAction("Plot")
        self._act_plot.triggered.connect(self._quick_plot)

        self._act_stats = tb.addAction("Stats")
        self._act_stats.triggered.connect(self._show_column_stats)

        self._act_corr_heatmap = tb.addAction("Correlation")
        self._act_corr_heatmap.setToolTip("Generate correlation matrix heatmap")
        self._act_corr_heatmap.triggered.connect(self._show_correlation_heatmap)

        self._act_outliers = tb.addAction("Outliers")
        self._act_outliers.setToolTip("Detect outliers using IQR method")
        self._act_outliers.triggered.connect(self._detect_outliers)

        tb.addSeparator()

        self._act_synth = tb.addAction("Synthetic Data")
        self._act_synth.setToolTip("Generate synthetic datasets")
        self._act_synth.triggered.connect(self._generate_synthetic_data)

        self._act_transform = tb.addAction("Transform")
        self._act_transform.setToolTip("Apply data transformation pipeline")
        self._act_transform.triggered.connect(self._transform_pipeline)

        self._act_pivot = tb.addAction("Pivot Table")
        self._act_pivot.setToolTip("Create pivot table")
        self._act_pivot.triggered.connect(self._create_pivot_table)

        self._act_report = tb.addAction("Report")
        self._act_report.setToolTip("Generate automated HTML analysis report")
        self._act_report.triggered.connect(self._generate_report)

        self._act_profile = tb.addAction("Profile")
        self._act_profile.setToolTip("Auto-detect column types, distributions, outliers")
        self._act_profile.triggered.connect(self._profile_data)

        tb.addSeparator()

        self._act_sql = tb.addAction("SQL Query")
        self._act_sql.setToolTip("Run SQL-like queries: SELECT col WHERE col > 5 ORDER BY col")
        self._act_sql.triggered.connect(self._sql_query)

        self._act_join = tb.addAction("Join")
        self._act_join.setToolTip("Join/merge with a second dataset on a common column")
        self._act_join.triggered.connect(self._join_dataset)

        self._act_decompose = tb.addAction("Decompose")
        self._act_decompose.setToolTip("Time series decomposition: trend + seasonal + residual")
        self._act_decompose.triggered.connect(self._time_series_decompose)

        self._act_infer = tb.addAction("Infer Types")
        self._act_infer.setToolTip("Auto-detect and convert column data types")
        self._act_infer.triggered.connect(self._infer_types)

        tb.addSeparator()

        self._act_copy_table = tb.addAction("Copy Table")
        self._act_copy_table.setToolTip("Copy table data to clipboard as tab-separated text (for Excel)")
        self._act_copy_table.triggered.connect(self._copy_table_to_clipboard)

        root.addWidget(tb)

        # Splitter: table | summary
        splitter = QSplitter(Qt.Horizontal)

        # Table
        self._table = QTableWidget()
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self._table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self._table.setContextMenuPolicy(Qt.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._table_context_menu)
        self._table.cellChanged.connect(self._on_cell_changed)
        splitter.addWidget(self._table)

        # Summary panel
        summary_widget = QWidget()
        summary_layout = QVBoxLayout(summary_widget)
        summary_layout.setContentsMargins(4, 4, 4, 4)

        lbl = QLabel("Data Summary")
        lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
        summary_layout.addWidget(lbl)

        self._summary_text = QTextEdit()
        self._summary_text.setReadOnly(True)
        self._summary_text.setFont(QFont("Consolas", 9))
        summary_layout.addWidget(self._summary_text)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setMaximumHeight(16)
        summary_layout.addWidget(self._progress)

        splitter.addWidget(summary_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        root.addWidget(splitter)

        # Status bar
        self._status = QLabel("Ready")
        self._status.setStyleSheet("color: #555; font-size: 11px; padding: 2px;")
        root.addWidget(self._status)

    # --------------------------------------------------------------- logger
    def set_logger(self, fn):
        """Set an external logging callback: fn(message: str)."""
        self._logger = fn

    def _log(self, msg: str):
        if self._logger:
            self._logger(msg)

    # ------------------------------------------------------------ undo helpers
    def _push_undo(self):
        self._undo_stack.append(self._df.copy())
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)

    def _undo(self):
        if not self._undo_stack:
            self._set_status("Nothing to undo.")
            return
        self._df = self._undo_stack.pop()
        self._refresh_table()
        self._set_status("Undo applied.")

    # ---------------------------------------------------------- status / summary
    def _set_status(self, msg: str):
        self._status.setText(msg)
        self._log(msg)

    def _update_summary(self):
        if self._df.empty:
            self._summary_text.setPlainText("No data loaded.")
            return
        lines = []
        lines.append(f"Rows: {len(self._df)}    Columns: {len(self._df.columns)}")
        lines.append(f"Memory: {self._df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        lines.append("")
        lines.append("Column Types:")
        for c in self._df.columns:
            null_count = int(self._df[c].isna().sum())
            dtype = self._df[c].dtype
            lines.append(f"  {c}: {dtype}  (nulls: {null_count})")
        lines.append("")
        num_cols = self._df.select_dtypes(include=[np.number]).columns
        if len(num_cols):
            lines.append("Numeric Summary:")
            desc = self._df[num_cols].describe().round(4)
            lines.append(desc.to_string())
        self._summary_text.setPlainText("\n".join(lines))

    # ------------------------------------------------------------ table sync
    def _refresh_table(self):
        """Rebuild QTableWidget from the internal DataFrame."""
        self._table.blockSignals(True)
        df = self._df
        self._table.setRowCount(len(df))
        self._table.setColumnCount(len(df.columns))
        self._table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        self._table.setVerticalHeaderLabels([str(i) for i in df.index])

        for r in range(len(df)):
            for c in range(len(df.columns)):
                val = df.iat[r, c]
                text = "" if pd.isna(val) else str(val)
                item = QTableWidgetItem(text)
                if pd.isna(val):
                    item.setBackground(QColor(255, 255, 200))
                self._table.setItem(r, c, item)

        self._table.blockSignals(False)
        self._update_summary()
        self.data_changed.emit()

    def _on_cell_changed(self, row, col):
        """Propagate manual edits back to the DataFrame."""
        item = self._table.item(row, col)
        if item is None:
            return
        text = item.text()
        self._push_undo()
        try:
            self._df.iat[row, col] = pd.to_numeric(text)
        except (ValueError, TypeError):
            self._df.iat[row, col] = text if text != "" else np.nan
        self._update_summary()

    # --------------------------------------------------------------- Import
    def _detect_delimiter(self, path: str) -> str:
        """Sniff the delimiter of a text file."""
        with open(path, "r", newline="", encoding="utf-8", errors="replace") as f:
            sample = f.read(8192)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
            return dialect.delimiter
        except csv.Error:
            if "\t" in sample:
                return "\t"
            return ","

    def load_file(self, path: str):
        """Load a data file (CSV, TSV, XLSX, JSON) into the widget."""
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            self._set_status(f"File not found: {path}")
            return
        ext = os.path.splitext(path)[1].lower()
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)
        QApplication.processEvents()
        try:
            if ext in (".csv", ".tsv", ".txt", ".dat"):
                sep = self._detect_delimiter(path)
                df = pd.read_csv(path, sep=sep, engine="python")
            elif ext in (".xls", ".xlsx", ".xlsm"):
                df = pd.read_excel(path, engine="openpyxl")
            elif ext == ".json":
                df = pd.read_json(path)
            else:
                self._set_status(f"Unsupported format: {ext}")
                return
            self._push_undo()
            self._df = df
            self._file_path = path
            self._refresh_table()
            self._set_status(f"Loaded {os.path.basename(path)} ({len(df)} rows x {len(df.columns)} cols)")
        except Exception as exc:
            self._set_status(f"Import error: {exc}")
            self._log(traceback.format_exc())
        finally:
            self._progress.setVisible(False)

    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "",
            "All Supported (*.csv *.tsv *.txt *.dat *.xlsx *.xls *.json);;"
            "CSV (*.csv *.tsv *.txt *.dat);;Excel (*.xlsx *.xls);;JSON (*.json)"
        )
        if path:
            self.load_file(path)

    # --------------------------------------------------------------- Export
    def export(self, path: str | None = None):
        """Export current data to file. If *path* is None, opens a save dialog."""
        if self._df.empty:
            self._set_status("No data to export.")
            return
        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Data", "",
                "CSV (*.csv);;Excel (*.xlsx)"
            )
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".xlsx":
                self._df.to_excel(path, index=False, engine="openpyxl")
            else:
                self._df.to_csv(path, index=False)
            self._set_status(f"Exported to {os.path.basename(path)}")
        except Exception as exc:
            self._set_status(f"Export error: {exc}")

    def _on_export(self):
        self.export()

    # --------------------------------------------------------- Column ops
    def _columns(self):
        return list(self._df.columns)

    def _add_column(self):
        if self._df.empty:
            self._set_status("Load data first.")
            return
        name, ok = QInputDialog.getText(self, "Add Column", "Column name:")
        if ok and name.strip():
            self._push_undo()
            self._df[name.strip()] = np.nan
            self._refresh_table()
            self._set_status(f"Added column '{name.strip()}'")

    def _remove_column(self):
        cols = self._columns()
        if not cols:
            return
        col, ok = QInputDialog.getItem(self, "Remove Column", "Select column:", cols, editable=False)
        if ok:
            self._push_undo()
            self._df.drop(columns=[col], inplace=True)
            self._refresh_table()
            self._set_status(f"Removed column '{col}'")

    def _rename_column(self):
        cols = self._columns()
        if not cols:
            return
        dlg = RenameColumnDialog(cols, self)
        if dlg.exec_() == QDialog.Accepted:
            old, new = dlg.get_values()
            if new:
                self._push_undo()
                self._df.rename(columns={old: new}, inplace=True)
                self._refresh_table()
                self._set_status(f"Renamed '{old}' -> '{new}'")

    def _add_computed_column(self):
        cols = self._columns()
        if not cols:
            return
        dlg = ComputedColumnDialog(cols, self)
        if dlg.exec_() == QDialog.Accepted:
            name, expr = dlg.get_values()
            if not name or not expr:
                return
            self._push_undo()
            try:
                self._df[name] = self._df.eval(expr)
                self._refresh_table()
                self._set_status(f"Computed column '{name}' = {expr}")
            except Exception as exc:
                self._undo_stack.pop()
                self._set_status(f"Expression error: {exc}")

    # --------------------------------------------------------- Data operations
    def _sort_data(self):
        cols = self._columns()
        if not cols:
            return
        col, ok = QInputDialog.getItem(self, "Sort", "Sort by column:", cols, editable=False)
        if not ok:
            return
        order, ok2 = QInputDialog.getItem(self, "Sort Order", "Order:", ["Ascending", "Descending"], editable=False)
        if ok2:
            self._push_undo()
            asc = order == "Ascending"
            self._df.sort_values(by=col, ascending=asc, inplace=True, ignore_index=True)
            self._refresh_table()
            self._set_status(f"Sorted by '{col}' ({'asc' if asc else 'desc'})")

    def _filter_data(self):
        cols = self._columns()
        if not cols:
            return
        dlg = FilterDialog(cols, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        col, op, val = dlg.get_filter()
        self._push_undo()
        try:
            series = self._df[col]
            if op == "is null":
                mask = series.isna()
            elif op == "not null":
                mask = series.notna()
            elif op == "contains":
                mask = series.astype(str).str.contains(val, case=False, na=False)
            elif op == "not contains":
                mask = ~series.astype(str).str.contains(val, case=False, na=False)
            else:
                try:
                    val_num = pd.to_numeric(val)
                except (ValueError, TypeError):
                    val_num = val
                if op == "==":
                    mask = series == val_num
                elif op == "!=":
                    mask = series != val_num
                elif op == ">":
                    mask = series > val_num
                elif op == ">=":
                    mask = series >= val_num
                elif op == "<":
                    mask = series < val_num
                elif op == "<=":
                    mask = series <= val_num
                else:
                    mask = pd.Series([True] * len(self._df))
            self._df = self._df[mask].reset_index(drop=True)
            self._refresh_table()
            self._set_status(f"Filtered: {col} {op} {val} -> {len(self._df)} rows")
        except Exception as exc:
            self._undo_stack.pop()
            self._set_status(f"Filter error: {exc}")

    def _remove_duplicates(self):
        if self._df.empty:
            return
        self._push_undo()
        before = len(self._df)
        self._df.drop_duplicates(inplace=True, ignore_index=True)
        after = len(self._df)
        self._refresh_table()
        self._set_status(f"Removed {before - after} duplicate rows.")

    def _fill_missing(self):
        methods = ["Forward fill", "Backward fill", "Mean", "Median", "Zero", "Interpolate (linear)"]
        method, ok = QInputDialog.getItem(self, "Fill Missing Values", "Method:", methods, editable=False)
        if not ok:
            return
        self._push_undo()
        try:
            if method == "Forward fill":
                self._df.ffill(inplace=True)
            elif method == "Backward fill":
                self._df.bfill(inplace=True)
            elif method == "Mean":
                num = self._df.select_dtypes(include=[np.number]).columns
                self._df[num] = self._df[num].fillna(self._df[num].mean())
            elif method == "Median":
                num = self._df.select_dtypes(include=[np.number]).columns
                self._df[num] = self._df[num].fillna(self._df[num].median())
            elif method == "Zero":
                self._df.fillna(0, inplace=True)
            elif method == "Interpolate (linear)":
                self._df.interpolate(method="linear", inplace=True)
            self._refresh_table()
            self._set_status(f"Filled missing values ({method}).")
        except Exception as exc:
            self._undo_stack.pop()
            self._set_status(f"Fill error: {exc}")

    def _interpolate(self):
        cols = self._df.select_dtypes(include=[np.number]).columns.tolist()
        if not cols:
            self._set_status("No numeric columns for interpolation.")
            return
        col, ok = QInputDialog.getItem(self, "Interpolate", "Column:", cols, editable=False)
        if not ok:
            return
        methods = ["linear", "quadratic", "cubic", "spline", "polynomial", "nearest"]
        method, ok2 = QInputDialog.getItem(self, "Interpolation Method", "Method:", methods, editable=False)
        if not ok2:
            return
        self._push_undo()
        try:
            kwargs = {}
            if method in ("spline", "polynomial"):
                order, ok3 = QInputDialog.getInt(self, "Order", "Polynomial order:", 3, 1, 10)
                if not ok3:
                    self._undo_stack.pop()
                    return
                kwargs["order"] = order
            self._df[col] = self._df[col].interpolate(method=method, **kwargs)
            self._refresh_table()
            self._set_status(f"Interpolated '{col}' ({method}).")
        except Exception as exc:
            self._undo_stack.pop()
            self._set_status(f"Interpolation error: {exc}")

    def _normalize(self):
        cols = self._df.select_dtypes(include=[np.number]).columns.tolist()
        if not cols:
            self._set_status("No numeric columns to normalize.")
            return
        col, ok = QInputDialog.getItem(self, "Normalize", "Column:", cols, editable=False)
        if not ok:
            return
        methods = ["Min-Max [0,1]", "Z-score", "Max-Abs"]
        method, ok2 = QInputDialog.getItem(self, "Normalization", "Method:", methods, editable=False)
        if not ok2:
            return
        self._push_undo()
        try:
            s = self._df[col]
            if method == "Min-Max [0,1]":
                mn, mx = s.min(), s.max()
                self._df[col] = (s - mn) / (mx - mn) if mx != mn else 0.0
            elif method == "Z-score":
                self._df[col] = (s - s.mean()) / s.std()
            elif method == "Max-Abs":
                self._df[col] = s / s.abs().max() if s.abs().max() != 0 else 0.0
            self._refresh_table()
            self._set_status(f"Normalized '{col}' ({method}).")
        except Exception as exc:
            self._undo_stack.pop()
            self._set_status(f"Normalization error: {exc}")

    # --------------------------------------------------------------- Stats
    def _show_column_stats(self):
        cols = self._df.select_dtypes(include=[np.number]).columns.tolist()
        if not cols:
            self._set_status("No numeric columns for statistics.")
            return
        col, ok = QInputDialog.getItem(self, "Column Statistics", "Column:", cols, editable=False)
        if not ok:
            return
        s = self._df[col].dropna()
        if s.empty:
            QMessageBox.information(self, "Stats", f"Column '{col}' has no non-null numeric values.")
            return
        # Mode (top value)
        mode_val = s.mode()
        top_str = f"{mode_val.iloc[0]:.6g}" if len(mode_val) > 0 else "N/A"
        info = (
            f"Column: {col}\n"
            f"Count:     {len(s)}\n"
            f"Unique:    {s.nunique()}\n"
            f"Top (mode):{top_str}\n"
            f"Mean:      {s.mean():.6g}\n"
            f"Median:    {s.median():.6g}\n"
            f"Std Dev:   {s.std():.6g}\n"
            f"Variance:  {s.var():.6g}\n"
            f"Min:       {s.min():.6g}\n"
            f"Max:       {s.max():.6g}\n"
            f"Range:     {s.max() - s.min():.6g}\n"
            f"Skewness:  {sp_stats.skew(s):.6g}\n"
            f"Kurtosis:  {sp_stats.kurtosis(s):.6g}\n"
            f"Sum:       {s.sum():.6g}\n"
            f"Null count:{int(self._df[col].isna().sum())}"
        )
        QMessageBox.information(self, f"Statistics - {col}", info)
        self._log(f"Stats for '{col}':\n{info}")

    # --------------------------------------------------------------- Plot
    def _quick_plot(self):
        if self._df.empty:
            self._set_status("Load data before plotting.")
            return
        dlg = PlotDialog(self._df, self)
        dlg.exec_()

    # ------------------------------------------------------ Context menu
    def _table_context_menu(self, pos):
        menu = QMenu(self)
        copy_act = menu.addAction("Copy Selection")
        paste_act = menu.addAction("Paste")
        menu.addSeparator()
        del_rows_act = menu.addAction("Delete Selected Rows")
        menu.addSeparator()
        sel_stats_act = menu.addAction("Selection Statistics")

        action = menu.exec_(self._table.viewport().mapToGlobal(pos))
        if action == copy_act:
            self._copy_selection()
        elif action == paste_act:
            self._paste_selection()
        elif action == del_rows_act:
            self._delete_selected_rows()
        elif action == sel_stats_act:
            self._selection_statistics()

    def _copy_selection(self):
        selection = self._table.selectedRanges()
        if not selection:
            return
        sr = selection[0]
        rows = range(sr.topRow(), sr.bottomRow() + 1)
        cols = range(sr.leftColumn(), sr.rightColumn() + 1)
        lines = []
        for r in rows:
            cells = []
            for c in cols:
                item = self._table.item(r, c)
                cells.append(item.text() if item else "")
            lines.append("\t".join(cells))
        QApplication.clipboard().setText("\n".join(lines))
        self._set_status("Copied to clipboard.")

    def _paste_selection(self):
        text = QApplication.clipboard().text()
        if not text:
            return
        self._push_undo()
        rows = text.strip().split("\n")
        sel = self._table.selectedRanges()
        start_row = sel[0].topRow() if sel else 0
        start_col = sel[0].leftColumn() if sel else 0
        for r_offset, row_text in enumerate(rows):
            cells = row_text.split("\t")
            for c_offset, val in enumerate(cells):
                r = start_row + r_offset
                c = start_col + c_offset
                if r < len(self._df) and c < len(self._df.columns):
                    try:
                        self._df.iat[r, c] = pd.to_numeric(val)
                    except (ValueError, TypeError):
                        self._df.iat[r, c] = val if val != "" else np.nan
        self._refresh_table()
        self._set_status("Pasted from clipboard.")

    def _delete_selected_rows(self):
        rows = sorted(set(idx.row() for idx in self._table.selectedIndexes()), reverse=True)
        if not rows:
            return
        self._push_undo()
        self._df.drop(index=[self._df.index[r] for r in rows], inplace=True)
        self._df.reset_index(drop=True, inplace=True)
        self._refresh_table()
        self._set_status(f"Deleted {len(rows)} row(s).")

    def _selection_statistics(self):
        """Show statistics on the currently selected numeric cells."""
        vals = []
        for idx in self._table.selectedIndexes():
            item = self._table.item(idx.row(), idx.column())
            if item:
                try:
                    vals.append(float(item.text()))
                except ValueError:
                    pass
        if not vals:
            QMessageBox.information(self, "Selection Stats", "No numeric values selected.")
            return
        arr = np.array(vals)
        info = (
            f"Count:    {len(arr)}\n"
            f"Sum:      {arr.sum():.6g}\n"
            f"Mean:     {arr.mean():.6g}\n"
            f"Median:   {np.median(arr):.6g}\n"
            f"Std Dev:  {arr.std():.6g}\n"
            f"Min:      {arr.min():.6g}\n"
            f"Max:      {arr.max():.6g}"
        )
        QMessageBox.information(self, "Selection Statistics", info)

    # ------------------------------------------------------ Synthetic Data Generation
    def _generate_synthetic_data(self):
        """Generate synthetic datasets from various distributions and patterns."""
        dist_types = [
            "Normal (Gaussian)",
            "Uniform",
            "Exponential",
            "Poisson",
            "Log-Normal",
            "Random Walk",
            "Time Series (trend + seasonality + noise)",
            "Multi-variate Correlated",
        ]
        dist, ok = QInputDialog.getItem(
            self, "Generate Synthetic Data", "Distribution / Pattern:", dist_types, editable=False
        )
        if not ok:
            return

        n_rows, ok = QInputDialog.getInt(self, "Rows", "Number of data points:", 500, 10, 1000000)
        if not ok:
            return

        n_cols, ok = QInputDialog.getInt(self, "Columns", "Number of columns:", 3, 1, 100)
        if not ok:
            return

        self._push_undo()
        try:
            if "Normal" in dist:
                mean, ok = QInputDialog.getDouble(self, "Mean", "Mean:", 0.0)
                if not ok:
                    return
                std, ok = QInputDialog.getDouble(self, "Std Dev", "Standard deviation:", 1.0, 0.001)
                if not ok:
                    return
                data = np.random.normal(mean, std, (n_rows, n_cols))
                cols = [f"normal_{i+1}" for i in range(n_cols)]

            elif "Uniform" in dist:
                lo, ok = QInputDialog.getDouble(self, "Low", "Low bound:", 0.0)
                if not ok:
                    return
                hi, ok = QInputDialog.getDouble(self, "High", "High bound:", 1.0)
                if not ok:
                    return
                data = np.random.uniform(lo, hi, (n_rows, n_cols))
                cols = [f"uniform_{i+1}" for i in range(n_cols)]

            elif "Exponential" in dist:
                scale, ok = QInputDialog.getDouble(self, "Scale", "Scale (1/lambda):", 1.0, 0.001)
                if not ok:
                    return
                data = np.random.exponential(scale, (n_rows, n_cols))
                cols = [f"exponential_{i+1}" for i in range(n_cols)]

            elif "Poisson" in dist:
                lam, ok = QInputDialog.getDouble(self, "Lambda", "Lambda (rate):", 5.0, 0.01)
                if not ok:
                    return
                data = np.random.poisson(lam, (n_rows, n_cols))
                cols = [f"poisson_{i+1}" for i in range(n_cols)]

            elif "Log-Normal" in dist:
                mean, ok = QInputDialog.getDouble(self, "Mean", "Mean (of log):", 0.0)
                if not ok:
                    return
                sigma, ok = QInputDialog.getDouble(self, "Sigma", "Sigma (of log):", 0.5, 0.001)
                if not ok:
                    return
                data = np.random.lognormal(mean, sigma, (n_rows, n_cols))
                cols = [f"lognormal_{i+1}" for i in range(n_cols)]

            elif "Random Walk" in dist:
                steps = np.random.randn(n_rows, n_cols)
                data = np.cumsum(steps, axis=0)
                cols = [f"walk_{i+1}" for i in range(n_cols)]

            elif "Time Series" in dist:
                t = np.arange(n_rows)
                data_list = []
                for i in range(n_cols):
                    trend = 0.01 * (i + 1) * t
                    seasonal = np.sin(2 * np.pi * t / (50 + 10 * i)) * (2 + i)
                    noise = np.random.randn(n_rows) * 0.5
                    data_list.append(trend + seasonal + noise)
                data = np.column_stack(data_list)
                cols = [f"ts_{i+1}" for i in range(n_cols)]
                # Prepend a time index column
                time_idx = pd.date_range("2020-01-01", periods=n_rows, freq="D")
                df = pd.DataFrame(data, columns=cols)
                df.insert(0, "date", time_idx)
                self._df = df
                self._refresh_table()
                self._set_status(f"Synthetic time series: {n_rows} rows x {n_cols + 1} cols")
                return

            elif "Multi-variate Correlated" in dist:
                corr_strength, ok = QInputDialog.getDouble(
                    self, "Correlation", "Correlation strength (0-0.99):", 0.7, 0.0, 0.99, 2
                )
                if not ok:
                    return
                # Build a correlation matrix with specified off-diagonal correlation
                cov = np.full((n_cols, n_cols), corr_strength)
                np.fill_diagonal(cov, 1.0)
                mean = np.zeros(n_cols)
                data = np.random.multivariate_normal(mean, cov, n_rows)
                cols = [f"corr_{i+1}" for i in range(n_cols)]

            else:
                return

            self._df = pd.DataFrame(data, columns=cols)
            self._refresh_table()
            self._set_status(f"Synthetic data ({dist}): {n_rows} rows x {n_cols} cols")
        except Exception as exc:
            self._undo_stack.pop() if self._undo_stack else None
            self._set_status(f"Synthetic data error: {exc}")
            QMessageBox.warning(self, "Synthetic Data Error", str(exc))

    # ------------------------------------------------------ Data Transformation Pipeline
    def _transform_pipeline(self):
        """Chain multiple data transformations on selected columns."""
        cols = self._df.select_dtypes(include=[np.number]).columns.tolist()
        if not cols:
            self._set_status("No numeric columns for transformation.")
            return

        col, ok = QInputDialog.getItem(
            self, "Transform Pipeline", "Column to transform:", cols, editable=False
        )
        if not ok:
            return

        transforms = [
            "Normalize (0-1)",
            "Standardize (z-score)",
            "Log transform (ln)",
            "Log10 transform",
            "First difference",
            "Second difference",
            "Rolling mean",
            "Rolling std",
            "Exponential moving average",
            "Lag (shift)",
            "Cumulative sum",
            "Rank",
            "Percent change",
            "Abs (absolute value)",
            "Power transform (Box-Cox)",
            "Detrend (linear)",
        ]

        # Allow selecting multiple transforms in sequence
        applied = []
        while True:
            remaining = [t for t in transforms if t not in applied]
            if not remaining:
                break
            choices = remaining + ["-- Done (apply all) --"]
            transform, ok = QInputDialog.getItem(
                self, "Transform Pipeline",
                f"Select transformation (applied so far: {len(applied)}):",
                choices, editable=False
            )
            if not ok or "Done" in transform:
                break
            applied.append(transform)

        if not applied:
            return

        self._push_undo()
        try:
            s = self._df[col].copy()
            suffix_parts = []

            for transform in applied:
                if "Normalize" in transform:
                    mn, mx = s.min(), s.max()
                    s = (s - mn) / (mx - mn) if mx != mn else s * 0.0
                    suffix_parts.append("norm")
                elif "Standardize" in transform:
                    s = (s - s.mean()) / s.std()
                    suffix_parts.append("zscore")
                elif "Log transform" in transform and "Log10" not in transform:
                    s = np.log(s.clip(lower=1e-10))
                    suffix_parts.append("ln")
                elif "Log10" in transform:
                    s = np.log10(s.clip(lower=1e-10))
                    suffix_parts.append("log10")
                elif "First difference" in transform:
                    s = s.diff()
                    suffix_parts.append("diff1")
                elif "Second difference" in transform:
                    s = s.diff().diff()
                    suffix_parts.append("diff2")
                elif "Rolling mean" in transform:
                    window, ok_w = QInputDialog.getInt(self, "Window", "Rolling window size:", 5, 2, len(s))
                    if ok_w:
                        s = s.rolling(window).mean()
                    suffix_parts.append(f"rmean{window if ok_w else 5}")
                elif "Rolling std" in transform:
                    window, ok_w = QInputDialog.getInt(self, "Window", "Rolling window size:", 5, 2, len(s))
                    if ok_w:
                        s = s.rolling(window).std()
                    suffix_parts.append(f"rstd{window if ok_w else 5}")
                elif "Exponential moving average" in transform:
                    span, ok_s = QInputDialog.getInt(self, "Span", "EMA span:", 10, 2, len(s))
                    if ok_s:
                        s = s.ewm(span=span).mean()
                    suffix_parts.append("ema")
                elif "Lag" in transform:
                    lag, ok_l = QInputDialog.getInt(self, "Lag", "Lag periods:", 1, 1, len(s) - 1)
                    if ok_l:
                        s = s.shift(lag)
                    suffix_parts.append(f"lag{lag if ok_l else 1}")
                elif "Cumulative sum" in transform:
                    s = s.cumsum()
                    suffix_parts.append("cumsum")
                elif "Rank" in transform:
                    s = s.rank()
                    suffix_parts.append("rank")
                elif "Percent change" in transform:
                    s = s.pct_change()
                    suffix_parts.append("pctchg")
                elif "Abs" in transform:
                    s = s.abs()
                    suffix_parts.append("abs")
                elif "Power transform" in transform:
                    valid = s.dropna()
                    if (valid > 0).all():
                        from scipy.stats import boxcox
                        transformed, _ = boxcox(valid.values)
                        s = pd.Series(np.nan, index=s.index)
                        s.loc[valid.index] = transformed
                    else:
                        self._set_status("Box-Cox requires all positive values.")
                    suffix_parts.append("boxcox")
                elif "Detrend" in transform:
                    valid_mask = s.notna()
                    if valid_mask.sum() > 1:
                        x_vals = np.arange(len(s))[valid_mask]
                        y_vals = s[valid_mask].values
                        coeffs = np.polyfit(x_vals, y_vals, 1)
                        trend = np.polyval(coeffs, np.arange(len(s)))
                        s = s - trend
                    suffix_parts.append("detrend")

            new_col_name = f"{col}_{'_'.join(suffix_parts)}"
            self._df[new_col_name] = s
            self._refresh_table()
            self._set_status(f"Transform pipeline applied: {' -> '.join(applied)} => '{new_col_name}'")
        except Exception as exc:
            self._undo_stack.pop() if self._undo_stack else None
            self._set_status(f"Transform error: {exc}")

    # ------------------------------------------------------ Pivot Table
    def _create_pivot_table(self):
        """Create a pivot table from the current data."""
        if self._df.empty:
            self._set_status("Load data first.")
            return

        cols = self._columns()
        if len(cols) < 2:
            self._set_status("Need at least 2 columns for pivot table.")
            return

        index_col, ok = QInputDialog.getItem(
            self, "Pivot Table", "Row index column:", cols, editable=False
        )
        if not ok:
            return

        remaining = [c for c in cols if c != index_col]
        columns_col, ok = QInputDialog.getItem(
            self, "Pivot Table", "Column header column:", remaining, editable=False
        )
        if not ok:
            return

        num_cols = self._df.select_dtypes(include=[np.number]).columns.tolist()
        val_choices = [c for c in num_cols if c != index_col and c != columns_col]
        if not val_choices:
            self._set_status("No numeric value columns available.")
            return

        values_col, ok = QInputDialog.getItem(
            self, "Pivot Table", "Values column:", val_choices, editable=False
        )
        if not ok:
            return

        agg_funcs = ["mean", "sum", "count", "min", "max", "median", "std"]
        agg, ok = QInputDialog.getItem(
            self, "Pivot Table", "Aggregation function:", agg_funcs, editable=False
        )
        if not ok:
            return

        self._push_undo()
        try:
            pivot = pd.pivot_table(
                self._df, values=values_col, index=index_col,
                columns=columns_col, aggfunc=agg, fill_value=0
            )
            # Flatten multi-level columns
            pivot.columns = [str(c) for c in pivot.columns]
            pivot = pivot.reset_index()
            self._df = pivot
            self._refresh_table()
            self._set_status(f"Pivot table: {index_col} x {columns_col}, values={values_col}, agg={agg}")
        except Exception as exc:
            self._undo_stack.pop() if self._undo_stack else None
            self._set_status(f"Pivot table error: {exc}")
            QMessageBox.warning(self, "Pivot Table Error", str(exc))

    # ------------------------------------------------------ Automated Report Generation
    def _generate_report(self):
        """Generate an HTML summary report with statistics, plots, and correlations."""
        if self._df.empty:
            self._set_status("Load data before generating report.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Report", "data_report.html", "HTML (*.html);;All Files (*)"
        )
        if not path:
            return

        try:
            df = self._df
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            html_parts = []
            html_parts.append("<!DOCTYPE html><html><head>")
            html_parts.append("<meta charset='utf-8'>")
            html_parts.append("<title>Data Analysis Report</title>")
            html_parts.append("<style>")
            html_parts.append("""
                body { font-family: 'Segoe UI', Arial, sans-serif; margin: 30px; color: #333; max-width: 1200px; margin: 0 auto; padding: 30px; }
                h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
                h2 { color: #2980b9; border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-top: 30px; }
                h3 { color: #555; }
                table { border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 13px; }
                th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: right; }
                th { background: #3498db; color: white; font-weight: 600; }
                tr:nth-child(even) { background: #f8f9fa; }
                tr:hover { background: #e8f4fd; }
                .stat-card { display: inline-block; background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px 20px; margin: 5px; min-width: 150px; }
                .stat-card h4 { margin: 0 0 5px 0; color: #666; font-size: 12px; }
                .stat-card .value { font-size: 24px; font-weight: bold; color: #2c3e50; }
                .warning { background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px; padding: 10px; margin: 10px 0; }
                img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin: 10px 0; }
                .footer { margin-top: 40px; padding-top: 15px; border-top: 1px solid #ddd; color: #999; font-size: 12px; }
            """)
            html_parts.append("</style></head><body>")

            # Header
            html_parts.append(f"<h1>Data Analysis Report</h1>")
            html_parts.append(f"<p>Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
            if self._file_path:
                html_parts.append(f"<p>Source: {os.path.basename(self._file_path)}</p>")

            # Overview cards
            html_parts.append("<h2>Overview</h2>")
            html_parts.append("<div>")
            cards = [
                ("Rows", str(len(df))),
                ("Columns", str(len(df.columns))),
                ("Numeric Cols", str(len(num_cols))),
                ("Missing Values", str(int(df.isna().sum().sum()))),
                ("Duplicates", str(int(df.duplicated().sum()))),
                ("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB"),
            ]
            for label, value in cards:
                html_parts.append(f"<div class='stat-card'><h4>{label}</h4><div class='value'>{value}</div></div>")
            html_parts.append("</div>")

            # Column info
            html_parts.append("<h2>Column Information</h2>")
            html_parts.append("<table><tr><th>Column</th><th>Type</th><th>Non-Null</th><th>Null</th><th>Unique</th></tr>")
            for c in df.columns:
                n_null = int(df[c].isna().sum())
                html_parts.append(
                    f"<tr><td style='text-align:left'>{c}</td><td>{df[c].dtype}</td>"
                    f"<td>{len(df) - n_null}</td><td>{n_null}</td><td>{df[c].nunique()}</td></tr>"
                )
            html_parts.append("</table>")

            # Descriptive statistics
            if num_cols:
                html_parts.append("<h2>Descriptive Statistics</h2>")
                desc = df[num_cols].describe().round(4)
                # Add skewness and kurtosis
                desc.loc["skewness"] = df[num_cols].skew().round(4)
                desc.loc["kurtosis"] = df[num_cols].kurtosis().round(4)
                html_parts.append(desc.to_html())

            # Missing values
            missing = df.isna().sum()
            if missing.sum() > 0:
                html_parts.append("<h2>Missing Values</h2>")
                missing_df = pd.DataFrame({
                    "Column": missing.index,
                    "Missing": missing.values,
                    "Percent": (missing.values / len(df) * 100).round(2)
                })
                missing_df = missing_df[missing_df["Missing"] > 0].sort_values("Missing", ascending=False)
                html_parts.append(missing_df.to_html(index=False))

            # Correlation matrix
            if len(num_cols) >= 2:
                html_parts.append("<h2>Correlation Matrix</h2>")
                corr = df[num_cols].corr().round(3)
                html_parts.append(corr.to_html())

                # Generate correlation heatmap
                try:
                    fig = Figure(figsize=(8, 6), dpi=100)
                    style_figure(fig)
                    ax = fig.add_subplot(111)
                    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
                    fig.colorbar(im, ax=ax, shrink=0.8)
                    ax.set_xticks(range(len(corr.columns)))
                    ax.set_yticks(range(len(corr.columns)))
                    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
                    ax.set_yticklabels(corr.columns, fontsize=8)
                    ax.set_title("Correlation Heatmap")
                    fig.tight_layout()

                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=100)
                    import base64
                    img_b64 = base64.b64encode(buf.getvalue()).decode()
                    html_parts.append(f"<img src='data:image/png;base64,{img_b64}' alt='Correlation Heatmap'>")
                    buf.close()
                except Exception:
                    pass

            # Distribution plots
            if num_cols:
                html_parts.append("<h2>Distribution Plots</h2>")
                for col_name in num_cols[:10]:  # Limit to 10 columns
                    try:
                        fig = Figure(figsize=(6, 3), dpi=100)
                        style_figure(fig)
                        ax = fig.add_subplot(111)
                        data = df[col_name].dropna()
                        ax.hist(data, bins="auto", color="#3498db", alpha=0.7, edgecolor="white")
                        ax.set_title(col_name, fontsize=10)
                        ax.set_xlabel(col_name, fontsize=9)
                        ax.set_ylabel("Count", fontsize=9)
                        fig.tight_layout()

                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", dpi=100)
                        import base64
                        img_b64 = base64.b64encode(buf.getvalue()).decode()
                        html_parts.append(f"<img src='data:image/png;base64,{img_b64}' alt='{col_name} distribution' style='max-width:48%; display:inline-block;'>")
                        buf.close()
                    except Exception:
                        pass

            # Footer
            html_parts.append("<div class='footer'>")
            html_parts.append("Generated by QuantumRes Data Analysis Suite")
            html_parts.append("</div>")
            html_parts.append("</body></html>")

            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(html_parts))

            self._set_status(f"Report saved: {path}")
            QMessageBox.information(self, "Report",
                                    f"HTML report saved to:\n{path}\n\nOpen in browser?")
            try:
                webbrowser.open(f"file://{os.path.abspath(path)}")
            except Exception:
                pass

        except Exception as exc:
            self._set_status(f"Report error: {exc}")
            QMessageBox.warning(self, "Report Error", str(exc))

    # ------------------------------------------------------ Data Profiling
    def _profile_data(self):
        """Auto-detect column types, distributions, outliers, and missing patterns."""
        if self._df.empty:
            self._set_status("Load data before profiling.")
            return

        df = self._df
        lines = []
        lines.append("=" * 60)
        lines.append("DATA PROFILE REPORT")
        lines.append("=" * 60)
        lines.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        lines.append(f"Memory: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        lines.append(f"Duplicates: {df.duplicated().sum()} rows")
        lines.append("")

        for col in df.columns:
            s = df[col]
            lines.append("-" * 50)
            lines.append(f"Column: {col}")
            lines.append(f"  Dtype: {s.dtype}")
            lines.append(f"  Non-null: {s.notna().sum()}/{len(s)} ({s.notna().sum()/len(s)*100:.1f}%)")
            lines.append(f"  Unique: {s.nunique()}")

            # Auto-detect type
            if pd.api.types.is_numeric_dtype(s):
                clean = s.dropna()
                if len(clean) > 0:
                    lines.append(f"  Detected: Numeric")
                    lines.append(f"  Mean: {clean.mean():.4g}  Std: {clean.std():.4g}")
                    lines.append(f"  Min: {clean.min():.4g}  Max: {clean.max():.4g}")
                    lines.append(f"  Skewness: {clean.skew():.4g}  Kurtosis: {clean.kurtosis():.4g}")

                    # Distribution detection
                    if len(clean) >= 20:
                        _, p_normal = sp_stats.normaltest(clean)
                        if p_normal > 0.05:
                            lines.append(f"  Distribution: Likely Normal (p={p_normal:.4f})")
                        elif (clean > 0).all():
                            try:
                                log_data = np.log(clean)
                                _, p_log = sp_stats.normaltest(log_data)
                                if p_log > 0.05:
                                    lines.append(f"  Distribution: Likely Log-Normal")
                            except Exception:
                                pass
                        if clean.skew() > 1:
                            lines.append(f"  Distribution: Right-skewed")
                        elif clean.skew() < -1:
                            lines.append(f"  Distribution: Left-skewed")

                    # Outlier detection (IQR method)
                    Q1 = clean.quantile(0.25)
                    Q3 = clean.quantile(0.75)
                    IQR = Q3 - Q1
                    outlier_mask = (clean < Q1 - 1.5 * IQR) | (clean > Q3 + 1.5 * IQR)
                    n_outliers = outlier_mask.sum()
                    if n_outliers > 0:
                        lines.append(f"  Outliers (IQR): {n_outliers} ({n_outliers/len(clean)*100:.1f}%)")

                    # Zero and negative counts
                    n_zeros = (clean == 0).sum()
                    n_neg = (clean < 0).sum()
                    if n_zeros > 0:
                        lines.append(f"  Zeros: {n_zeros}")
                    if n_neg > 0:
                        lines.append(f"  Negatives: {n_neg}")

            elif pd.api.types.is_datetime64_any_dtype(s):
                lines.append(f"  Detected: Datetime")
                clean = s.dropna()
                if len(clean) > 0:
                    lines.append(f"  Range: {clean.min()} to {clean.max()}")

            else:
                lines.append(f"  Detected: Categorical/Text")
                clean = s.dropna()
                if len(clean) > 0:
                    # Check if could be datetime
                    try:
                        pd.to_datetime(clean.head(10))
                        lines.append(f"  Note: May be parseable as datetime")
                    except Exception:
                        pass

                    # Check if could be numeric
                    try:
                        pd.to_numeric(clean.head(10))
                        lines.append(f"  Note: May be parseable as numeric")
                    except Exception:
                        pass

                    n_unique = s.nunique()
                    if n_unique <= 20:
                        top5 = s.value_counts().head(5)
                        lines.append(f"  Top values:")
                        for val, cnt in top5.items():
                            lines.append(f"    {val}: {cnt} ({cnt/len(s)*100:.1f}%)")

            # Missing value pattern
            null_count = s.isna().sum()
            if null_count > 0:
                lines.append(f"  Missing: {null_count} ({null_count/len(s)*100:.1f}%)")
                # Check if missing at beginning, end, or random
                first_null = s.isna().idxmax() if s.isna().any() else None
                if first_null == 0:
                    lines.append(f"  Missing pattern: Starts with missing values")
                elif s.isna().iloc[-5:].all():
                    lines.append(f"  Missing pattern: Trailing missing values")

        lines.append("")
        lines.append("=" * 60)

        # Show in summary panel
        self._summary_text.setPlainText("\n".join(lines))
        self._set_status(f"Data profiling complete: {len(df.columns)} columns analyzed")

    # ------------------------------------------------------ Correlation Heatmap
    def _show_correlation_heatmap(self):
        """Generate and display a correlation matrix heatmap."""
        if self._df.empty:
            self._set_status("Load data before generating correlation heatmap.")
            return
        num_cols = self._df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            self._set_status("Need at least 2 numeric columns for correlation heatmap.")
            return

        corr = self._df[num_cols].corr()

        dlg = QDialog(self)
        dlg.setWindowTitle("Correlation Matrix Heatmap")
        dlg.setMinimumSize(600, 500)
        layout = QVBoxLayout(dlg)

        fig = Figure(figsize=(8, 6), dpi=100)
        style_figure(fig)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        im = ax.imshow(corr.values, cmap="RdBu_r", aspect="auto",
                       vmin=-1, vmax=1, interpolation="nearest")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")

        ax.set_xticks(range(len(num_cols)))
        ax.set_yticks(range(len(num_cols)))
        ax.set_xticklabels(num_cols, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(num_cols, fontsize=8)

        # Annotate cells with correlation values
        for i in range(len(num_cols)):
            for j in range(len(num_cols)):
                val = corr.values[i, j]
                text_color = "white" if abs(val) > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=text_color)

        ax.set_title("Correlation Matrix", fontsize=11)
        fig.tight_layout()

        layout.addWidget(canvas)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)
        dlg.exec_()
        self._log("Displayed correlation heatmap.")

    # ------------------------------------------------------ Outlier Detection
    def _detect_outliers(self):
        """Detect outliers using the IQR method and highlight them in the table."""
        if self._df.empty:
            self._set_status("Load data before detecting outliers.")
            return
        num_cols = self._df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            self._set_status("No numeric columns for outlier detection.")
            return
        col, ok = QInputDialog.getItem(
            self, "Outlier Detection (IQR)", "Column:", num_cols, editable=False)
        if not ok:
            return

        s = self._df[col].dropna()
        if s.empty:
            self._set_status(f"Column '{col}' has no numeric data.")
            return

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        col_idx = self._df.columns.get_loc(col)
        outlier_count = 0

        # Clear previous highlighting in this column
        for row in range(self._table.rowCount()):
            item = self._table.item(row, col_idx)
            if item:
                item.setBackground(QColor("#1e1e2e") if row % 2 == 0 else QColor("#2a2a3e"))

        # Highlight outliers in red
        for row in range(len(self._df)):
            val = self._df.iloc[row][col]
            if pd.notna(val):
                try:
                    fval = float(val)
                    if fval < lower or fval > upper:
                        item = self._table.item(row, col_idx)
                        if item:
                            item.setBackground(QColor("#8B0000"))
                            outlier_count += 1
                except (ValueError, TypeError):
                    pass

        info = (f"Outlier detection (IQR) for '{col}':\n"
                f"  Q1 = {q1:.6g}, Q3 = {q3:.6g}, IQR = {iqr:.6g}\n"
                f"  Lower fence = {lower:.6g}, Upper fence = {upper:.6g}\n"
                f"  Outliers found: {outlier_count} / {len(s)}")
        QMessageBox.information(self, "Outlier Detection", info)
        self._log(info)
        self._set_status(f"Highlighted {outlier_count} outlier(s) in '{col}'.")

    # ------------------------------------------------------ SQL-like Query Engine
    def _sql_query(self):
        """Execute a SQL-like query on the loaded data.

        Supports: SELECT col1, col2 WHERE col3 > 5 ORDER BY col4 [ASC|DESC] LIMIT n
        """
        if self._df.empty:
            self._set_status("Load data first.")
            return
        query, ok = QInputDialog.getText(
            self, "SQL-like Query",
            "Query (e.g. SELECT col1 WHERE col2 > 5 ORDER BY col3 DESC LIMIT 100):",
        )
        if not ok or not query.strip():
            return
        self._push_undo()
        try:
            result = self._execute_sql_query(query.strip())
            self._df = result
            self._refresh_table()
            self._set_status(f"Query executed: {len(result)} rows returned.")
        except Exception as exc:
            self._undo_stack.pop() if self._undo_stack else None
            self._set_status(f"Query error: {exc}")
            QMessageBox.warning(self, "Query Error", str(exc))

    def _execute_sql_query(self, query: str) -> pd.DataFrame:
        """Parse and execute a simplified SQL-like query string."""
        import re
        q = query.strip()
        # Parse SELECT
        sel_match = re.match(r'(?i)SELECT\s+(.+?)(?:\s+WHERE\s+|\s+ORDER\s+|\s+LIMIT\s+|$)', q)
        if not sel_match:
            raise ValueError("Query must start with SELECT column_list")

        columns_part = sel_match.group(1).strip()
        if columns_part == '*':
            selected_cols = list(self._df.columns)
        else:
            selected_cols = [c.strip() for c in columns_part.split(',')]
            for c in selected_cols:
                if c not in self._df.columns:
                    raise ValueError(f"Column '{c}' not found. Available: {list(self._df.columns)}")

        result = self._df.copy()

        # Parse WHERE
        where_match = re.search(r'(?i)WHERE\s+(.+?)(?:\s+ORDER\s+|\s+LIMIT\s+|$)', q)
        if where_match:
            condition = where_match.group(1).strip()
            # Support AND/OR
            try:
                result = result.query(condition)
            except Exception:
                # Fallback: manual parsing of simple conditions
                ops = {'>=': 'ge', '<=': 'le', '!=': 'ne', '>': 'gt', '<': 'lt', '==': 'eq', '=': 'eq'}
                for op_str, _ in sorted(ops.items(), key=lambda x: -len(x[0])):
                    if op_str in condition:
                        parts = condition.split(op_str, 1)
                        col_name = parts[0].strip()
                        val_str = parts[1].strip().strip("'\"")
                        try:
                            val = pd.to_numeric(val_str)
                        except (ValueError, TypeError):
                            val = val_str
                        if op_str in ('>=',):
                            result = result[result[col_name] >= val]
                        elif op_str in ('<=',):
                            result = result[result[col_name] <= val]
                        elif op_str in ('!=',):
                            result = result[result[col_name] != val]
                        elif op_str in ('>',):
                            result = result[result[col_name] > val]
                        elif op_str in ('<',):
                            result = result[result[col_name] < val]
                        elif op_str in ('==', '='):
                            result = result[result[col_name] == val]
                        break

        # Parse ORDER BY
        order_match = re.search(r'(?i)ORDER\s+BY\s+(\w+)(?:\s+(ASC|DESC))?', q)
        if order_match:
            order_col = order_match.group(1).strip()
            direction = order_match.group(2)
            ascending = True if direction is None or direction.upper() == 'ASC' else False
            result = result.sort_values(by=order_col, ascending=ascending)

        # Parse LIMIT
        limit_match = re.search(r'(?i)LIMIT\s+(\d+)', q)
        if limit_match:
            limit = int(limit_match.group(1))
            result = result.head(limit)

        return result[selected_cols].reset_index(drop=True)

    # ------------------------------------------------------ Join/Merge Datasets
    def _join_dataset(self):
        """Load a second CSV and join/merge it with the current dataset."""
        if self._df.empty:
            self._set_status("Load a primary dataset first.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Second Dataset for Join", "",
            "CSV (*.csv *.tsv *.txt);;Excel (*.xlsx *.xls);;All Files (*)"
        )
        if not path:
            return
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext in (".xls", ".xlsx", ".xlsm"):
                df2 = pd.read_excel(path, engine="openpyxl")
            else:
                sep = self._detect_delimiter(path)
                df2 = pd.read_csv(path, sep=sep, engine="python")
        except Exception as exc:
            self._set_status(f"Error loading second file: {exc}")
            return

        # Find common columns for join key
        common = sorted(set(self._df.columns) & set(df2.columns))
        if not common:
            QMessageBox.warning(self, "Join Error", "No common columns found between the two datasets.")
            return

        key, ok = QInputDialog.getItem(self, "Join Key", "Join on column:", common, editable=False)
        if not ok:
            return

        join_types = ["inner", "left", "right", "outer"]
        how, ok = QInputDialog.getItem(self, "Join Type", "Join type:", join_types, editable=False)
        if not ok:
            return

        self._push_undo()
        try:
            merged = pd.merge(self._df, df2, on=key, how=how, suffixes=('_left', '_right'))
            self._df = merged
            self._refresh_table()
            self._set_status(
                f"Joined on '{key}' ({how}): {len(merged)} rows x {len(merged.columns)} cols"
            )
        except Exception as exc:
            self._undo_stack.pop() if self._undo_stack else None
            self._set_status(f"Join error: {exc}")
            QMessageBox.warning(self, "Join Error", str(exc))

    # ------------------------------------------------------ Time Series Decomposition
    def _time_series_decompose(self):
        """Decompose a numeric column into trend, seasonal, and residual components."""
        if self._df.empty:
            self._set_status("Load data first.")
            return
        num_cols = self._df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            self._set_status("No numeric columns for decomposition.")
            return

        col, ok = QInputDialog.getItem(
            self, "Time Series Decomposition", "Column to decompose:", num_cols, editable=False
        )
        if not ok:
            return

        period, ok = QInputDialog.getInt(
            self, "Period", "Seasonal period (number of observations per cycle):",
            12, 2, max(len(self._df) // 2, 3)
        )
        if not ok:
            return

        models = ["additive", "multiplicative"]
        model_type, ok = QInputDialog.getItem(
            self, "Model", "Decomposition model:", models, editable=False
        )
        if not ok:
            return

        try:
            series = self._df[col].dropna()
            if len(series) < 2 * period:
                QMessageBox.warning(self, "Decomposition Error",
                                    f"Need at least {2 * period} observations for period={period}.")
                return

            # Manual decomposition (no statsmodels dependency)
            n = len(series)
            vals = series.values.astype(float)

            # 1. Trend: centered moving average
            trend = np.full(n, np.nan)
            half = period // 2
            for i in range(half, n - half):
                window = vals[i - half:i + half + 1]
                if period % 2 == 0:
                    window = vals[i - half:i + half]
                    trend[i] = np.mean(window)
                else:
                    trend[i] = np.mean(window)

            # 2. Detrended
            if model_type == "additive":
                detrended = vals - trend
            else:
                detrended = vals / np.where(trend != 0, trend, np.nan)

            # 3. Seasonal: average over each position in cycle
            seasonal = np.full(n, np.nan)
            for pos in range(period):
                indices = list(range(pos, n, period))
                cycle_vals = [detrended[i] for i in indices if not np.isnan(detrended[i])]
                if cycle_vals:
                    avg = np.mean(cycle_vals)
                    for i in indices:
                        seasonal[i] = avg

            # 4. Residual
            if model_type == "additive":
                residual = vals - trend - seasonal
            else:
                residual = vals / (np.where(trend != 0, trend, np.nan) *
                                   np.where(seasonal != 0, seasonal, np.nan))

            # Add columns to dataframe
            self._push_undo()
            self._df[f'{col}_trend'] = np.nan
            self._df.loc[series.index, f'{col}_trend'] = trend
            self._df[f'{col}_seasonal'] = np.nan
            self._df.loc[series.index, f'{col}_seasonal'] = seasonal
            self._df[f'{col}_residual'] = np.nan
            self._df.loc[series.index, f'{col}_residual'] = residual
            self._refresh_table()

            # Plot decomposition
            dlg = QDialog(self)
            dlg.setWindowTitle(f"Time Series Decomposition - {col}")
            dlg.resize(800, 600)
            layout = QVBoxLayout(dlg)
            fig = Figure(figsize=(8, 8), dpi=100)
            style_figure(fig)
            canvas = FigureCanvas(fig)

            x_axis = np.arange(n)
            axes = fig.subplots(4, 1, sharex=True)
            axes[0].plot(x_axis, vals, 'b-', linewidth=0.8)
            axes[0].set_ylabel("Observed")
            axes[0].set_title(f"Time Series Decomposition ({model_type})")

            axes[1].plot(x_axis, trend, 'r-', linewidth=1.2)
            axes[1].set_ylabel("Trend")

            axes[2].plot(x_axis, seasonal, 'g-', linewidth=0.8)
            axes[2].set_ylabel("Seasonal")

            axes[3].plot(x_axis, residual, 'm-', linewidth=0.5)
            axes[3].set_ylabel("Residual")
            axes[3].set_xlabel("Observation Index")

            for a in axes:
                a.grid(True, alpha=0.3)
            fig.tight_layout()

            layout.addWidget(canvas)
            btn = QPushButton("Close")
            btn.clicked.connect(dlg.accept)
            layout.addWidget(btn)
            dlg.exec_()

            self._set_status(f"Decomposed '{col}' ({model_type}, period={period})")
        except Exception as exc:
            self._set_status(f"Decomposition error: {exc}")
            QMessageBox.warning(self, "Decomposition Error", str(exc))

    # ------------------------------------------------------ Smart Type Inference
    def _infer_types(self):
        """Auto-detect and convert column types with smart parsing."""
        if self._df.empty:
            self._set_status("Load data first.")
            return
        self._push_undo()
        report = []
        for col in self._df.columns:
            original_dtype = str(self._df[col].dtype)
            series = self._df[col]
            non_null = series.dropna()
            if len(non_null) == 0:
                report.append(f"  {col}: all null, skipped")
                continue

            converted = False

            # Try numeric
            if not pd.api.types.is_numeric_dtype(series):
                try:
                    numeric_vals = pd.to_numeric(non_null, errors='coerce')
                    valid_ratio = numeric_vals.notna().sum() / len(non_null)
                    if valid_ratio >= 0.8:
                        self._df[col] = pd.to_numeric(series, errors='coerce')
                        # Distinguish int vs float
                        clean = self._df[col].dropna()
                        if len(clean) > 0 and (clean == clean.astype(int)).all():
                            self._df[col] = self._df[col].astype('Int64')
                            report.append(f"  {col}: {original_dtype} -> Int64")
                        else:
                            report.append(f"  {col}: {original_dtype} -> float64")
                        converted = True
                except Exception:
                    pass

            # Try datetime
            if not converted and not pd.api.types.is_datetime64_any_dtype(series):
                try:
                    dt_vals = pd.to_datetime(non_null.head(50), errors='coerce', infer_datetime_format=True)
                    valid_ratio = dt_vals.notna().sum() / min(len(non_null), 50)
                    if valid_ratio >= 0.7:
                        self._df[col] = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
                        report.append(f"  {col}: {original_dtype} -> datetime64")
                        converted = True
                except Exception:
                    pass

            # Try boolean
            if not converted:
                try:
                    lower_vals = non_null.astype(str).str.strip().str.lower()
                    bool_set = {'true', 'false', 'yes', 'no', '1', '0', 't', 'f', 'y', 'n'}
                    if set(lower_vals.unique()).issubset(bool_set):
                        true_set = {'true', 'yes', '1', 't', 'y'}
                        self._df[col] = series.astype(str).str.strip().str.lower().map(
                            lambda x: True if x in true_set else (False if x in bool_set else None)
                        )
                        report.append(f"  {col}: {original_dtype} -> boolean")
                        converted = True
                except Exception:
                    pass

            # Try categorical (low cardinality strings)
            if not converted and series.dtype == object:
                n_unique = series.nunique()
                if n_unique <= max(20, len(series) * 0.05):
                    self._df[col] = series.astype('category')
                    report.append(f"  {col}: {original_dtype} -> category ({n_unique} levels)")
                    converted = True

            if not converted:
                report.append(f"  {col}: {original_dtype} (no change)")

        self._refresh_table()
        report_text = "Data Type Inference Report:\n" + "\n".join(report)
        self._summary_text.setPlainText(report_text)
        self._set_status(f"Type inference complete for {len(self._df.columns)} columns.")
        self._log(report_text)

    # ------------------------------------------------------ Public helpers
    def get_dataframe(self) -> pd.DataFrame:
        """Return a copy of the current DataFrame."""
        return self._df.copy()

    def set_dataframe(self, df: pd.DataFrame):
        """Replace the current data with *df*."""
        self._push_undo()
        self._df = df.copy()
        self._refresh_table()
        self._set_status(f"DataFrame set ({len(df)} rows x {len(df.columns)} cols).")

    def current_file_path(self) -> str | None:
        """Return the path of the currently loaded file, or None."""
        return self._file_path

    # ------------------------------------------------------------------
    # Clipboard
    # ------------------------------------------------------------------

    def _copy_table_to_clipboard(self):
        """Copy current table data to clipboard as tab-separated text."""
        if self._df.empty:
            self._set_status("Nothing to copy - table is empty.")
            return
        text = self._df.to_csv(sep='\t', index=False)
        QApplication.clipboard().setText(text)
        self._set_status(f"Copied {len(self._df)} rows to clipboard.")


# ---------------------------------------------------------------------------
# Standalone test harness
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = DataAnalysisWidget()
    w.set_logger(print)
    w.setWindowTitle("QuantumRes - Data Analysis")
    w.resize(1100, 680)
    w.show()
    # Load a file from command line if provided
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        w.load_file(sys.argv[1])
    sys.exit(app.exec_())
