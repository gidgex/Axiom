"""
notebook.py - Computational Notebook Widget for PyQt5 Scientific Suite

Provides a Jupyter-like cell-based notebook interface with support for
Python code execution, Markdown rendering, inline matplotlib figures,
rich display of numpy arrays and pandas DataFrames, shared namespace,
and save/load in JSON format.
"""

import sys
import io
import os
import json
import base64
import traceback
import keyword
import builtins
import time
import textwrap
from datetime import datetime

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QPlainTextEdit,
    QPushButton, QLabel, QToolBar, QAction, QComboBox, QFrame,
    QFileDialog, QApplication, QSizePolicy, QTextEdit, QMenu,
    QMessageBox, QSplitter, QShortcut, QInputDialog, QTableWidget,
    QTableWidgetItem, QHeaderView, QDialog, QDialogButtonBox,
    QTreeWidget, QTreeWidgetItem, QTabWidget,
)
from PyQt5.QtCore import Qt, QRegExp, QTimer, pyqtSignal, QSize, QMimeData
from PyQt5.QtGui import (
    QFont, QColor, QTextCharFormat, QSyntaxHighlighter,
    QTextCursor, QPalette, QKeySequence, QPixmap, QImage,
    QIcon, QPainter,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CELL_TYPE_CODE = "code"
CELL_TYPE_MARKDOWN = "markdown"

FONT_FAMILY = "Consolas"
FONT_SIZE = 10

COLOR_CELL_BG = "#1e1e2e"
COLOR_CELL_BORDER = "#45475a"
COLOR_CELL_ACTIVE = "#585b70"
COLOR_OUTPUT_BG = "#181825"
COLOR_TOOLBAR_BG = "#11111b"
COLOR_TEXT = "#cdd6f4"
COLOR_KEYWORD = "#cba6f7"
COLOR_STRING = "#a6e3a1"
COLOR_COMMENT = "#6c7086"
COLOR_NUMBER = "#fab387"
COLOR_BUILTIN = "#89b4fa"
COLOR_PROMPT = "#f9e2af"
COLOR_ERROR = "#f38ba8"
COLOR_MARKDOWN_BG = "#1e1e2e"


# ---------------------------------------------------------------------------
# Syntax Highlighter
# ---------------------------------------------------------------------------

class NotebookSyntaxHighlighter(QSyntaxHighlighter):
    """Minimal Python syntax highlighter for notebook code cells."""

    def __init__(self, document):
        super().__init__(document)
        self._rules = []

        # Keywords
        kw_fmt = QTextCharFormat()
        kw_fmt.setForeground(QColor(COLOR_KEYWORD))
        kw_fmt.setFontWeight(QFont.Bold)
        kw_pattern = r"\b(?:" + "|".join(keyword.kwlist) + r")\b"
        self._rules.append((QRegExp(kw_pattern), kw_fmt))

        # Builtins
        bi_fmt = QTextCharFormat()
        bi_fmt.setForeground(QColor(COLOR_BUILTIN))
        builtin_names = [b for b in dir(builtins) if not b.startswith("_")]
        bi_pattern = r"\b(?:" + "|".join(builtin_names) + r")\b"
        self._rules.append((QRegExp(bi_pattern), bi_fmt))

        # Numbers
        num_fmt = QTextCharFormat()
        num_fmt.setForeground(QColor(COLOR_NUMBER))
        self._rules.append((QRegExp(r"\b[0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?\b"), num_fmt))

        # Strings (double and single quoted)
        str_fmt = QTextCharFormat()
        str_fmt.setForeground(QColor(COLOR_STRING))
        self._rules.append((QRegExp(r'"[^"\\]*(\\.[^"\\]*)*"'), str_fmt))
        self._rules.append((QRegExp(r"'[^'\\]*(\\.[^'\\]*)*'"), str_fmt))

        # Comments
        cmt_fmt = QTextCharFormat()
        cmt_fmt.setForeground(QColor(COLOR_COMMENT))
        cmt_fmt.setFontItalic(True)
        self._rules.append((QRegExp(r"#[^\n]*"), cmt_fmt))

        # Decorators
        dec_fmt = QTextCharFormat()
        dec_fmt.setForeground(QColor(COLOR_PROMPT))
        self._rules.append((QRegExp(r"@\w+"), dec_fmt))

    def highlightBlock(self, text):
        for pattern, fmt in self._rules:
            index = pattern.indexIn(text)
            while index >= 0:
                length = pattern.matchedLength()
                self.setFormat(index, length, fmt)
                index = pattern.indexIn(text, index + length)


# ---------------------------------------------------------------------------
# Code Editor (for code cells)
# ---------------------------------------------------------------------------

class CellCodeEditor(QPlainTextEdit):
    """Code editor with line-count-based auto-resize and tab handling."""

    execute_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFont(QFont(FONT_FAMILY, FONT_SIZE))
        self.setTabStopDistance(self.fontMetrics().horizontalAdvance(" ") * 4)
        self.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.setStyleSheet(
            f"QPlainTextEdit {{ background-color: {COLOR_CELL_BG}; color: {COLOR_TEXT};"
            f" border: none; padding: 6px; selection-background-color: #585b70; }}"
        )
        self.setMinimumHeight(60)
        self.setMaximumHeight(600)
        self._highlighter = NotebookSyntaxHighlighter(self.document())
        self.document().contentsChanged.connect(self._adjust_height)
        QTimer.singleShot(0, self._adjust_height)

    def _adjust_height(self):
        doc = self.document()
        lines = max(doc.blockCount(), 3)
        line_h = self.fontMetrics().lineSpacing()
        new_h = min(max(lines * line_h + 20, 60), 600)
        self.setFixedHeight(new_h)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Tab:
            self.insertPlainText("    ")
            return
        # Shift+Enter or Ctrl+Enter to run cell
        if event.key() == Qt.Key_Return and (event.modifiers() & Qt.ShiftModifier
                                              or event.modifiers() & Qt.ControlModifier):
            self.execute_requested.emit()
            return
        if event.key() == Qt.Key_Return:
            # Auto-indent
            cursor = self.textCursor()
            block_text = cursor.block().text()
            indent = len(block_text) - len(block_text.lstrip())
            extra = 4 if block_text.rstrip().endswith(":") else 0
            super().keyPressEvent(event)
            self.insertPlainText(" " * (indent + extra))
            return
        super().keyPressEvent(event)


# ---------------------------------------------------------------------------
# Output Area
# ---------------------------------------------------------------------------

class CellOutputArea(QTextEdit):
    """Read-only rich-text output display for a cell."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont(FONT_FAMILY, FONT_SIZE))
        self.setStyleSheet(
            f"QTextEdit {{ background-color: {COLOR_OUTPUT_BG}; color: {COLOR_TEXT};"
            f" border: none; border-top: 1px solid {COLOR_CELL_BORDER}; padding: 6px; }}"
        )
        self.setMinimumHeight(0)
        self.setMaximumHeight(800)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.hide()

    def set_output(self, text, is_error=False):
        """Set plain-text output."""
        self.clear()
        if not text.strip():
            self.hide()
            return
        color = COLOR_ERROR if is_error else COLOR_TEXT
        self.setHtml(f"<pre style='color:{color}; margin:0;'>{_escape_html(text)}</pre>")
        self._fit_height()
        self.show()

    def set_html_output(self, html):
        """Set rich HTML output (DataFrames, images, etc.)."""
        self.clear()
        if not html.strip():
            self.hide()
            return
        self.setHtml(html)
        self._fit_height()
        self.show()

    def append_image(self, pixmap, alt_text="Figure"):
        """Append a QPixmap image into the output."""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertHtml("<br>")
        cursor.insertImage(pixmap.toImage(), alt_text)
        cursor.insertHtml("<br>")
        self.setTextCursor(cursor)
        self._fit_height()
        self.show()

    def clear_output(self):
        self.clear()
        self.hide()

    def _fit_height(self):
        doc_h = int(self.document().size().height()) + 16
        self.setFixedHeight(min(max(doc_h, 30), 800))


# ---------------------------------------------------------------------------
# Notebook Cell Widget
# ---------------------------------------------------------------------------

class NotebookCell(QFrame):
    """A single cell in the notebook (code or markdown)."""

    run_requested = pyqtSignal(object)
    delete_requested = pyqtSignal(object)
    move_up_requested = pyqtSignal(object)
    move_down_requested = pyqtSignal(object)
    insert_above_requested = pyqtSignal(object)
    insert_below_requested = pyqtSignal(object)
    focused = pyqtSignal(object)

    def __init__(self, cell_type=CELL_TYPE_CODE, source="", parent=None):
        super().__init__(parent)
        self._cell_type = cell_type
        self._execution_count = None
        self._is_active = False

        self.setFrameShape(QFrame.Box)
        self.setStyleSheet(
            f"NotebookCell {{ background-color: {COLOR_CELL_BG};"
            f" border: 1px solid {COLOR_CELL_BORDER}; border-radius: 4px; margin: 2px 4px; }}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header bar
        header = QHBoxLayout()
        header.setContentsMargins(8, 4, 8, 4)

        self._prompt_label = QLabel("")
        self._prompt_label.setFont(QFont(FONT_FAMILY, FONT_SIZE - 1))
        self._prompt_label.setStyleSheet(f"color: {COLOR_PROMPT}; font-weight: bold; border:none;")
        self._prompt_label.setFixedWidth(80)
        header.addWidget(self._prompt_label)

        header.addStretch()

        self._type_label = QLabel(cell_type.capitalize())
        self._type_label.setStyleSheet(f"color: {COLOR_COMMENT}; font-size: 9pt; border:none;")
        header.addWidget(self._type_label)

        # Run button
        self._run_btn = QPushButton("Run")
        self._run_btn.setFixedSize(50, 22)
        self._run_btn.setStyleSheet(
            "QPushButton { background-color: #45475a; color: #cdd6f4; border-radius: 3px; font-size: 9pt; border:none; }"
            "QPushButton:hover { background-color: #585b70; }"
        )
        self._run_btn.clicked.connect(lambda: self.run_requested.emit(self))
        header.addWidget(self._run_btn)

        layout.addLayout(header)

        # Editor
        self._editor = CellCodeEditor()
        self._editor.setPlainText(source)
        self._editor.execute_requested.connect(lambda: self.run_requested.emit(self))
        layout.addWidget(self._editor)

        # Markdown rendered view (hidden by default)
        self._markdown_view = QTextEdit()
        self._markdown_view.setReadOnly(True)
        self._markdown_view.setStyleSheet(
            f"QTextEdit {{ background-color: {COLOR_MARKDOWN_BG}; color: {COLOR_TEXT};"
            f" border: none; padding: 8px; }}"
        )
        self._markdown_view.setFont(QFont("Segoe UI", FONT_SIZE))
        self._markdown_view.hide()
        layout.addWidget(self._markdown_view)

        # Output area
        self._output = CellOutputArea()
        layout.addWidget(self._output)

        self._update_for_type()

        # Context menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    # -- Properties ---------------------------------------------------------

    @property
    def cell_type(self):
        return self._cell_type

    @cell_type.setter
    def cell_type(self, value):
        self._cell_type = value
        self._type_label.setText(value.capitalize())
        self._update_for_type()

    @property
    def source(self):
        return self._editor.toPlainText()

    @source.setter
    def source(self, text):
        self._editor.setPlainText(text)

    @property
    def execution_count(self):
        return self._execution_count

    @execution_count.setter
    def execution_count(self, val):
        self._execution_count = val
        if val is not None and self._cell_type == CELL_TYPE_CODE:
            self._prompt_label.setText(f"In [{val}]:")
        elif self._cell_type == CELL_TYPE_CODE:
            self._prompt_label.setText("In [ ]:")
        else:
            self._prompt_label.setText("")

    @property
    def output_area(self):
        return self._output

    # -- Internal -----------------------------------------------------------

    def _update_for_type(self):
        if self._cell_type == CELL_TYPE_CODE:
            self._editor.show()
            self._markdown_view.hide()
            self._run_btn.show()
            if self._execution_count is not None:
                self._prompt_label.setText(f"In [{self._execution_count}]:")
            else:
                self._prompt_label.setText("In [ ]:")
        else:
            self._editor.show()
            self._markdown_view.hide()
            self._run_btn.setText("Render")
            self._prompt_label.setText("")

    def render_markdown(self):
        """Render the editor content as markdown (basic HTML conversion)."""
        raw = self._editor.toPlainText()
        html = _markdown_to_html(raw)
        self._markdown_view.setHtml(
            f"<div style='color:{COLOR_TEXT}; font-family: Segoe UI, sans-serif;'>{html}</div>"
        )
        self._editor.hide()
        self._markdown_view.show()
        # Auto-size
        doc_h = int(self._markdown_view.document().size().height()) + 20
        self._markdown_view.setFixedHeight(min(max(doc_h, 40), 600))

    def enter_edit_mode(self):
        """Switch back to editor from rendered markdown."""
        self._markdown_view.hide()
        self._editor.show()

    def set_active(self, active):
        self._is_active = active
        border_color = COLOR_CELL_ACTIVE if active else COLOR_CELL_BORDER
        self.setStyleSheet(
            f"NotebookCell {{ background-color: {COLOR_CELL_BG};"
            f" border: 2px solid {border_color}; border-radius: 4px; margin: 2px 4px; }}"
        )

    def mousePressEvent(self, event):
        self.focused.emit(self)
        super().mousePressEvent(event)

    def _show_context_menu(self, pos):
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; }"
            "QMenu::item:selected { background-color: #585b70; }"
        )
        menu.addAction("Run Cell", lambda: self.run_requested.emit(self))
        menu.addSeparator()
        menu.addAction("Insert Cell Above", lambda: self.insert_above_requested.emit(self))
        menu.addAction("Insert Cell Below", lambda: self.insert_below_requested.emit(self))
        menu.addSeparator()
        menu.addAction("Move Up", lambda: self.move_up_requested.emit(self))
        menu.addAction("Move Down", lambda: self.move_down_requested.emit(self))
        menu.addSeparator()
        menu.addAction("Delete Cell", lambda: self.delete_requested.emit(self))
        menu.exec_(self.mapToGlobal(pos))

    def to_dict(self):
        """Serialize cell to dict."""
        return {
            "cell_type": self._cell_type,
            "source": self._editor.toPlainText(),
            "execution_count": self._execution_count,
            "outputs": self._output.toHtml() if self._output.isVisible() else "",
        }

    @classmethod
    def from_dict(cls, data, parent=None):
        """Deserialize cell from dict."""
        cell = cls(
            cell_type=data.get("cell_type", CELL_TYPE_CODE),
            source=data.get("source", ""),
            parent=parent,
        )
        cell.execution_count = data.get("execution_count")
        output_html = data.get("outputs", "")
        if output_html:
            cell._output.set_html_output(output_html)
        return cell


# ---------------------------------------------------------------------------
# Main Notebook Widget
# ---------------------------------------------------------------------------

class NotebookWidget(QWidget):
    """Jupyter-like computational notebook widget.

    Features:
    - Cell-based interface with code and markdown cells
    - Shared execution namespace across cells
    - Inline matplotlib figure capture
    - Rich display of numpy arrays and pandas DataFrames
    - Save/Load as JSON
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cells = []
        self._active_cell = None
        self._exec_counter = 0
        self._total_executions = 0
        self._namespace = {"__builtins__": builtins}
        self._log = None
        self._current_path = None

        self._init_namespace()
        self._build_ui()
        # Start with one empty code cell
        if not self._cells:
            self._add_cell(CELL_TYPE_CODE, index=0)

    # -- Public API ---------------------------------------------------------

    def set_logger(self, fn):
        """Set a logging callback: fn(message: str)."""
        self._log = fn

    def load_file(self, path):
        """Load a notebook from a JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._load_from_data(data)
            self._current_path = path
            self._emit_log(f"Loaded notebook from {path}")
        except Exception as exc:
            self._emit_log(f"Error loading notebook: {exc}")

    def save_as(self, path):
        """Save the notebook to a JSON file."""
        try:
            data = self._serialize()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self._current_path = path
            self._emit_log(f"Saved notebook to {path}")
        except Exception as exc:
            self._emit_log(f"Error saving notebook: {exc}")

    def run(self):
        """Run all cells sequentially (alias for run_all)."""
        self._run_all()

    def reset_namespace(self):
        """Clear the shared execution namespace."""
        self._namespace = {"__builtins__": builtins}
        self._exec_counter = 0
        self._init_namespace()
        self._emit_log("Namespace reset")

    def get_cell_count(self):
        return len(self._cells)

    # -- Namespace ----------------------------------------------------------

    def _init_namespace(self):
        """Pre-populate the namespace with common scientific imports."""
        preamble = textwrap.dedent("""\
            import numpy as np
            import math
        """)
        try:
            exec(preamble, self._namespace)
        except ImportError:
            pass
        # Try optional imports
        for mod in ("pandas as pd", "scipy", "matplotlib", "matplotlib.pyplot as plt"):
            try:
                exec(f"import {mod}", self._namespace)
            except ImportError:
                pass

    # -- UI Construction ----------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Toolbar
        self._toolbar = QToolBar()
        self._toolbar.setMovable(False)
        self._toolbar.setStyleSheet(
            f"QToolBar {{ background-color: {COLOR_TOOLBAR_BG}; border-bottom: 1px solid {COLOR_CELL_BORDER};"
            f" spacing: 4px; padding: 2px 4px; }}"
        )
        self._toolbar.setIconSize(QSize(18, 18))

        btn_style = (
            "QPushButton { background-color: #45475a; color: #cdd6f4; border-radius: 3px;"
            " padding: 4px 10px; font-size: 9pt; }"
            "QPushButton:hover { background-color: #585b70; }"
        )

        # Run button
        run_btn = QPushButton("Run")
        run_btn.setStyleSheet(btn_style)
        run_btn.setToolTip("Run selected cell (Shift+Enter)")
        run_btn.clicked.connect(self._run_active_cell)
        self._toolbar.addWidget(run_btn)

        # Run All
        run_all_btn = QPushButton("Run All")
        run_all_btn.setStyleSheet(btn_style)
        run_all_btn.setToolTip("Run all cells")
        run_all_btn.clicked.connect(self._run_all)
        self._toolbar.addWidget(run_all_btn)

        self._toolbar.addSeparator()

        # Add Cell
        add_btn = QPushButton("+ Cell")
        add_btn.setStyleSheet(btn_style)
        add_btn.setToolTip("Add new cell below")
        add_btn.clicked.connect(self._add_cell_below_active)
        self._toolbar.addWidget(add_btn)

        # Delete Cell
        del_btn = QPushButton("Delete")
        del_btn.setStyleSheet(btn_style)
        del_btn.setToolTip("Delete selected cell")
        del_btn.clicked.connect(self._delete_active_cell)
        self._toolbar.addWidget(del_btn)

        # Move Up
        up_btn = QPushButton("Up")
        up_btn.setStyleSheet(btn_style)
        up_btn.setToolTip("Move cell up")
        up_btn.clicked.connect(lambda: self._move_cell_up(self._active_cell))
        self._toolbar.addWidget(up_btn)

        # Move Down
        down_btn = QPushButton("Down")
        down_btn.setStyleSheet(btn_style)
        down_btn.setToolTip("Move cell down")
        down_btn.clicked.connect(lambda: self._move_cell_down(self._active_cell))
        self._toolbar.addWidget(down_btn)

        self._toolbar.addSeparator()

        # Cell type selector
        type_label = QLabel(" Type: ")
        type_label.setStyleSheet(f"color: {COLOR_TEXT}; font-size: 9pt;")
        self._toolbar.addWidget(type_label)

        self._type_combo = QComboBox()
        self._type_combo.addItems(["Code", "Markdown"])
        self._type_combo.setStyleSheet(
            "QComboBox { background-color: #45475a; color: #cdd6f4; border-radius: 3px;"
            " padding: 3px 8px; font-size: 9pt; }"
            "QComboBox::drop-down { border: none; }"
            "QComboBox QAbstractItemView { background-color: #313244; color: #cdd6f4;"
            " selection-background-color: #585b70; }"
        )
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        self._toolbar.addWidget(self._type_combo)

        self._toolbar.addSeparator()

        # Reset namespace
        reset_btn = QPushButton("Reset")
        reset_btn.setStyleSheet(btn_style)
        reset_btn.setToolTip("Reset execution namespace")
        reset_btn.clicked.connect(self.reset_namespace)
        self._toolbar.addWidget(reset_btn)

        # Save / Load
        save_btn = QPushButton("Save")
        save_btn.setStyleSheet(btn_style)
        save_btn.clicked.connect(self._save_dialog)
        self._toolbar.addWidget(save_btn)

        load_btn = QPushButton("Load")
        load_btn.setStyleSheet(btn_style)
        load_btn.clicked.connect(self._load_dialog)
        self._toolbar.addWidget(load_btn)

        self._toolbar.addSeparator()

        # Export HTML
        export_html_btn = QPushButton("Export HTML")
        export_html_btn.setStyleSheet(btn_style)
        export_html_btn.setToolTip("Export notebook to HTML")
        export_html_btn.clicked.connect(self._export_html)
        self._toolbar.addWidget(export_html_btn)

        # Export .py
        export_py_btn = QPushButton("Export .py")
        export_py_btn.setStyleSheet(btn_style)
        export_py_btn.setToolTip("Export code cells as Python script")
        export_py_btn.clicked.connect(self._export_python)
        self._toolbar.addWidget(export_py_btn)

        # Variable Explorer
        var_btn = QPushButton("Vars")
        var_btn.setStyleSheet(btn_style)
        var_btn.setToolTip("Variable explorer")
        var_btn.clicked.connect(self._show_variable_explorer)
        self._toolbar.addWidget(var_btn)

        # Cell Templates
        tmpl_btn = QPushButton("Templates")
        tmpl_btn.setStyleSheet(btn_style)
        tmpl_btn.setToolTip("Insert pre-built cell templates")
        tmpl_btn.clicked.connect(self._show_cell_templates_menu)
        self._toolbar.addWidget(tmpl_btn)

        # Table of Contents
        toc_btn = QPushButton("TOC")
        toc_btn.setStyleSheet(btn_style)
        toc_btn.setToolTip("Table of contents from markdown headers")
        toc_btn.clicked.connect(self._show_table_of_contents)
        self._toolbar.addWidget(toc_btn)

        root.addWidget(self._toolbar)

        # Scroll area for cells
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet(
            f"QScrollArea {{ background-color: #11111b; border: none; }}"
            f"QScrollBar:vertical {{ background: #181825; width: 10px; }}"
            f"QScrollBar::handle:vertical {{ background: #45475a; border-radius: 4px; min-height: 30px; }}"
            f"QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}"
        )

        self._cell_container = QWidget()
        self._cell_layout = QVBoxLayout(self._cell_container)
        self._cell_layout.setContentsMargins(8, 8, 8, 8)
        self._cell_layout.setSpacing(4)
        self._cell_layout.addStretch()

        self._scroll.setWidget(self._cell_container)
        root.addWidget(self._scroll)

        # Status bar
        self._status = QLabel("Ready")
        self._status.setStyleSheet(
            f"QLabel {{ background-color: {COLOR_TOOLBAR_BG}; color: {COLOR_COMMENT};"
            f" padding: 2px 8px; font-size: 8pt; border-top: 1px solid {COLOR_CELL_BORDER}; }}"
        )
        root.addWidget(self._status)

        # Keyboard shortcuts
        QShortcut(QKeySequence("Shift+Return"), self, self._run_active_cell)
        QShortcut(QKeySequence("Ctrl+Return"), self, self._run_active_cell)
        QShortcut(QKeySequence("Ctrl+Shift+Return"), self, self._run_all)
        QShortcut(QKeySequence("Ctrl+S"), self, self._save_dialog)
        QShortcut(QKeySequence("Ctrl+B"), self, self._add_cell_below_active)

    # -- Cell Management ----------------------------------------------------

    def _add_cell(self, cell_type=CELL_TYPE_CODE, index=None, source=""):
        """Create and insert a new cell."""
        cell = NotebookCell(cell_type=cell_type, source=source)
        cell.run_requested.connect(self._run_cell)
        cell.delete_requested.connect(self._delete_cell)
        cell.move_up_requested.connect(self._move_cell_up)
        cell.move_down_requested.connect(self._move_cell_down)
        cell.insert_above_requested.connect(self._insert_cell_above)
        cell.insert_below_requested.connect(self._insert_cell_below)
        cell.focused.connect(self._set_active_cell)

        if index is None:
            index = len(self._cells)

        self._cells.insert(index, cell)
        # Insert before the stretch
        self._cell_layout.insertWidget(index, cell)
        self._set_active_cell(cell)
        self._emit_log(f"Added {cell_type} cell at position {index}")
        return cell

    def _delete_cell(self, cell):
        if cell not in self._cells:
            return
        if len(self._cells) <= 1:
            self._emit_log("Cannot delete the last cell")
            return
        idx = self._cells.index(cell)
        self._cells.remove(cell)
        self._cell_layout.removeWidget(cell)
        cell.deleteLater()

        # Activate neighbor
        if self._cells:
            new_idx = min(idx, len(self._cells) - 1)
            self._set_active_cell(self._cells[new_idx])
        self._emit_log(f"Deleted cell at position {idx}")

    def _delete_active_cell(self):
        if self._active_cell:
            self._delete_cell(self._active_cell)

    def _move_cell_up(self, cell):
        if cell is None or cell not in self._cells:
            return
        idx = self._cells.index(cell)
        if idx == 0:
            return
        self._swap_cells(idx, idx - 1)

    def _move_cell_down(self, cell):
        if cell is None or cell not in self._cells:
            return
        idx = self._cells.index(cell)
        if idx >= len(self._cells) - 1:
            return
        self._swap_cells(idx, idx + 1)

    def _swap_cells(self, i, j):
        """Swap two cells in the list and layout."""
        self._cells[i], self._cells[j] = self._cells[j], self._cells[i]
        # Rebuild layout order
        for idx, c in enumerate(self._cells):
            self._cell_layout.removeWidget(c)
        for idx, c in enumerate(self._cells):
            self._cell_layout.insertWidget(idx, c)

    def _insert_cell_above(self, cell):
        idx = self._cells.index(cell) if cell in self._cells else 0
        self._add_cell(CELL_TYPE_CODE, index=idx)

    def _insert_cell_below(self, cell):
        idx = self._cells.index(cell) + 1 if cell in self._cells else len(self._cells)
        self._add_cell(CELL_TYPE_CODE, index=idx)

    def _add_cell_below_active(self):
        if self._active_cell:
            self._insert_cell_below(self._active_cell)
        else:
            self._add_cell(CELL_TYPE_CODE)

    def _set_active_cell(self, cell):
        if self._active_cell:
            self._active_cell.set_active(False)
        self._active_cell = cell
        if cell:
            cell.set_active(True)
            # Sync type combo
            if cell.cell_type == CELL_TYPE_CODE:
                self._type_combo.setCurrentIndex(0)
            else:
                self._type_combo.setCurrentIndex(1)

    def _on_type_changed(self, index):
        if not self._active_cell:
            return
        new_type = CELL_TYPE_CODE if index == 0 else CELL_TYPE_MARKDOWN
        if self._active_cell.cell_type != new_type:
            self._active_cell.cell_type = new_type
            if new_type == CELL_TYPE_MARKDOWN:
                self._active_cell.enter_edit_mode()

    # -- Execution ----------------------------------------------------------

    def _run_cell(self, cell):
        """Execute a single cell."""
        self._set_active_cell(cell)

        if cell.cell_type == CELL_TYPE_MARKDOWN:
            cell.render_markdown()
            self._status.setText("Rendered markdown cell")
            return

        # Code cell execution
        self._exec_counter += 1
        cell.execution_count = self._exec_counter
        source = cell.source.strip()
        if not source:
            cell.output_area.clear_output()
            self._status.setText(f"In [{self._exec_counter}]: (empty)")
            return

        self._status.setText(f"Running In [{self._exec_counter}]...")
        QApplication.processEvents()

        # Handle magic commands
        magic_result = self._handle_magic_commands(source, cell)
        if magic_result is not None:
            cell.output_area.set_html_output(
                f"<div style='font-family:{FONT_FAMILY}; font-size:{FONT_SIZE}pt;'>"
                f"{magic_result}</div>"
            )
            self._status.setText(f"In [{self._exec_counter}]: ok (magic)")
            return

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        figures = []

        # Set up matplotlib figure capture
        mpl_backend_patched = False
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            plt.close("all")
            original_backend = matplotlib.get_backend()
            matplotlib.use("Agg")
            mpl_backend_patched = True
        except ImportError:
            plt = None

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        result_repr = None
        is_error = False

        try:
            # Split into statements: exec all but last, eval last for expression result
            code_obj = compile(source, f"<cell {self._exec_counter}>", "exec")
            exec(code_obj, self._namespace)

            # Try to get the result of the last expression
            lines = source.strip().split("\n")
            last_line = lines[-1].strip()
            if last_line and not last_line.startswith(("import ", "from ", "def ", "class ",
                                                        "if ", "for ", "while ", "with ",
                                                        "try:", "except", "finally:",
                                                        "return", "yield", "raise",
                                                        "#", "pass", "break", "continue",
                                                        "elif ", "else:")) \
               and "=" not in last_line.split("#")[0] or \
               (last_line.count("=") >= 2 and "==" in last_line):
                try:
                    val = eval(last_line, self._namespace)
                    if val is not None:
                        result_repr = self._format_result(val)
                except Exception:
                    pass

        except Exception:
            is_error = True
            traceback.print_exc(file=stderr_capture)

        sys.stdout = old_stdout
        sys.stderr = old_stderr

        # Capture matplotlib figures
        if plt is not None:
            fig_nums = plt.get_fignums()
            for num in fig_nums:
                fig = plt.figure(num)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                            facecolor="#1e1e2e", edgecolor="none")
                buf.seek(0)
                img_data = base64.b64encode(buf.read()).decode("utf-8")
                figures.append(img_data)
                buf.close()
            plt.close("all")

        if mpl_backend_patched:
            try:
                matplotlib.use(original_backend)
            except Exception:
                pass

        # Build output
        stdout_text = stdout_capture.getvalue()
        stderr_text = stderr_capture.getvalue()

        output_parts = []
        if stdout_text.strip():
            output_parts.append(f"<pre style='color:{COLOR_TEXT}; margin:0;'>{_escape_html(stdout_text)}</pre>")

        if result_repr:
            output_parts.append(result_repr)

        if stderr_text.strip():
            output_parts.append(f"<pre style='color:{COLOR_ERROR}; margin:0;'>{_escape_html(stderr_text)}</pre>")

        for img_b64 in figures:
            output_parts.append(
                f"<div style='text-align:center; padding:4px;'>"
                f"<img src='data:image/png;base64,{img_b64}' style='max-width:100%;'/>"
                f"</div>"
            )

        if output_parts:
            cell.output_area.set_html_output(
                f"<div style='font-family:{FONT_FAMILY}; font-size:{FONT_SIZE}pt;'>"
                + "\n".join(output_parts)
                + "</div>"
            )
        else:
            cell.output_area.clear_output()

        self._total_executions += 1
        status = "error" if is_error else "ok"
        mem_str = self._get_memory_usage()
        self._status.setText(
            f"In [{self._exec_counter}]: {status}  |  "
            f"Total executions: {self._total_executions}  |  {mem_str}"
        )
        self._emit_log(f"Executed cell {self._exec_counter}: {status}")

    def _get_memory_usage(self) -> str:
        """Return current process memory usage as a formatted string."""
        try:
            import psutil
            proc = psutil.Process(os.getpid())
            mem = proc.memory_info().rss
            if mem >= 1024 ** 3:
                return f"Mem: {mem / (1024 ** 3):.2f} GB"
            else:
                return f"Mem: {mem / (1024 ** 2):.1f} MB"
        except ImportError:
            # Fallback: use sys.getsizeof on the namespace
            import sys as _sys
            total = sum(_sys.getsizeof(v) for v in self._namespace.values()
                        if not isinstance(v, type))
            return f"NS size: {total / 1024:.1f} KB"

    def _run_active_cell(self):
        if self._active_cell:
            self._run_cell(self._active_cell)

    def _run_all(self):
        """Execute all cells in order."""
        self._emit_log("Running all cells...")
        for cell in self._cells:
            self._run_cell(cell)
            QApplication.processEvents()
        self._emit_log("All cells executed")

    def _format_result(self, val):
        """Format a Python value for rich display."""
        # pandas DataFrame
        try:
            import pandas as pd
            if isinstance(val, pd.DataFrame):
                html = val.to_html(max_rows=50, max_cols=20, classes="df-table")
                styled = (
                    "<style>"
                    ".df-table { border-collapse: collapse; font-size: 9pt; }"
                    f".df-table th {{ background-color: #313244; color: {COLOR_TEXT};"
                    " padding: 4px 8px; border: 1px solid #45475a; }"
                    f".df-table td {{ background-color: {COLOR_CELL_BG}; color: {COLOR_TEXT};"
                    " padding: 3px 8px; border: 1px solid #45475a; }}"
                    "</style>"
                ) + html
                return styled
            if isinstance(val, pd.Series):
                return self._format_result(val.to_frame())
        except ImportError:
            pass

        # numpy array
        try:
            import numpy as np
            if isinstance(val, np.ndarray):
                with np.printoptions(threshold=200, linewidth=100, precision=4):
                    arr_str = repr(val)
                return (
                    f"<pre style='color:{COLOR_BUILTIN}; margin:0;'>{_escape_html(arr_str)}</pre>"
                    f"<pre style='color:{COLOR_COMMENT}; margin:0; font-size:8pt;'>"
                    f"  shape={val.shape}, dtype={val.dtype}</pre>"
                )
        except ImportError:
            pass

        # Default repr
        repr_str = repr(val)
        if len(repr_str) > 2000:
            repr_str = repr_str[:2000] + "..."
        return f"<pre style='color:{COLOR_TEXT}; margin:0;'>{_escape_html(repr_str)}</pre>"

    # -- Magic Commands -----------------------------------------------------

    def _handle_magic_commands(self, source, cell):
        """Handle magic commands. Returns HTML string if handled, else None."""
        stripped = source.strip()
        if not stripped.startswith("%"):
            return None

        parts = stripped.split(None, 1)
        magic = parts[0]
        args = parts[1] if len(parts) > 1 else ""

        if magic == "%timeit":
            if not args:
                return f"<pre style='color:{COLOR_ERROR};'>Usage: %timeit expression</pre>"
            try:
                times = []
                ns = self._namespace.copy()
                for _ in range(7):
                    t0 = time.perf_counter()
                    exec(compile(args, "<timeit>", "exec"), ns)
                    times.append(time.perf_counter() - t0)
                mean_t = sum(times) / len(times)
                std_t = (sum((t - mean_t) ** 2 for t in times) / len(times)) ** 0.5
                if mean_t < 1e-3:
                    unit, scale = "us", 1e6
                elif mean_t < 1:
                    unit, scale = "ms", 1e3
                else:
                    unit, scale = "s", 1
                return (
                    f"<pre style='color:{COLOR_TEXT};'>"
                    f"{mean_t * scale:.2f} {unit} +/- {std_t * scale:.2f} {unit} per loop "
                    f"(mean +/- std. dev. of 7 runs)</pre>"
                )
            except Exception as exc:
                return f"<pre style='color:{COLOR_ERROR};'>timeit error: {_escape_html(str(exc))}</pre>"

        elif magic in ("%matplotlib",):
            return f"<pre style='color:{COLOR_COMMENT};'>matplotlib inline mode is default in this notebook.</pre>"

        elif magic == "%who":
            user_vars = [k for k in self._namespace if not k.startswith("_")
                         and k not in ("np", "pd", "plt", "math", "scipy", "matplotlib")]
            if not user_vars:
                return f"<pre style='color:{COLOR_COMMENT};'>No user-defined variables.</pre>"
            return f"<pre style='color:{COLOR_TEXT};'>{' '.join(sorted(user_vars))}</pre>"

        elif magic == "%whos":
            user_vars = {k: v for k, v in self._namespace.items()
                         if not k.startswith("_")
                         and k not in ("np", "pd", "plt", "math", "scipy", "matplotlib")}
            if not user_vars:
                return f"<pre style='color:{COLOR_COMMENT};'>No user-defined variables.</pre>"
            lines = [f"{'Variable':<20s} {'Type':<20s} {'Info':<30s}"]
            lines.append("-" * 70)
            for name, val in sorted(user_vars.items()):
                type_str = type(val).__name__
                info = ""
                if hasattr(val, "shape"):
                    info = f"shape={val.shape}"
                elif hasattr(val, "__len__"):
                    info = f"len={len(val)}"
                else:
                    r = repr(val)
                    info = r[:30] if len(r) > 30 else r
                lines.append(f"{name:<20s} {type_str:<20s} {info:<30s}")
            return f"<pre style='color:{COLOR_TEXT};'>{_escape_html(chr(10).join(lines))}</pre>"

        elif magic == "%clear":
            cell.output_area.clear_output()
            return f"<pre style='color:{COLOR_COMMENT};'>Output cleared.</pre>"

        return None

    # -- Export HTML --------------------------------------------------------

    def _export_html(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export as HTML", "", "HTML Files (*.html);;All Files (*)"
        )
        if not path:
            return

        try:
            html_parts = [
                "<!DOCTYPE html>",
                "<html><head><meta charset='utf-8'>",
                "<title>Notebook Export</title>",
                "<style>",
                "body { font-family: 'Segoe UI', sans-serif; max-width: 900px; "
                "margin: 40px auto; padding: 0 20px; background: #1e1e2e; color: #cdd6f4; }",
                ".cell { border: 1px solid #45475a; border-radius: 6px; margin: 10px 0; "
                "overflow: hidden; }",
                ".cell-header { background: #313244; padding: 4px 12px; font-size: 0.8em; "
                "color: #a6adc8; }",
                ".code-input { background: #1e1e2e; padding: 12px; font-family: 'Consolas', "
                "monospace; font-size: 10pt; white-space: pre-wrap; overflow-x: auto; }",
                ".code-output { background: #181825; padding: 12px; border-top: 1px solid "
                "#45475a; font-family: 'Consolas', monospace; font-size: 10pt; "
                "white-space: pre-wrap; }",
                ".markdown-cell { background: #1e1e2e; padding: 12px 16px; }",
                ".keyword { color: #cba6f7; font-weight: bold; }",
                ".string { color: #a6e3a1; }",
                ".comment { color: #6c7086; font-style: italic; }",
                ".number { color: #fab387; }",
                ".builtin { color: #89b4fa; }",
                "img { max-width: 100%; }",
                "</style></head><body>",
                f"<h1>Notebook Export</h1>",
                f"<p style='color:#6c7086;'>Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
                "<hr style='border-color:#45475a;'>",
            ]

            for i, cell in enumerate(self._cells):
                if cell.cell_type == CELL_TYPE_CODE:
                    source = cell.source
                    highlighted = self._syntax_highlight_html(source)
                    prompt = f"In [{cell.execution_count or ' '}]" if cell.execution_count else "In [ ]"
                    html_parts.append("<div class='cell'>")
                    html_parts.append(f"<div class='cell-header'>{prompt}</div>")
                    html_parts.append(f"<div class='code-input'>{highlighted}</div>")
                    if cell.output_area.isVisible():
                        output_html = cell.output_area.toHtml()
                        html_parts.append(f"<div class='code-output'>{output_html}</div>")
                    html_parts.append("</div>")
                else:
                    md_html = _markdown_to_html(cell.source)
                    html_parts.append("<div class='cell'>")
                    html_parts.append(f"<div class='cell-header'>Markdown</div>")
                    html_parts.append(f"<div class='markdown-cell'>{md_html}</div>")
                    html_parts.append("</div>")

            html_parts.append("</body></html>")

            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(html_parts))

            self._emit_log(f"Exported HTML to {path}")
        except Exception as exc:
            self._emit_log(f"HTML export error: {exc}")

    def _syntax_highlight_html(self, code):
        """Apply basic syntax highlighting to Python code for HTML export."""
        import re as _re
        escaped = _escape_html(code)
        # Keywords
        kw_pattern = r'\b(' + '|'.join(keyword.kwlist) + r')\b'
        escaped = _re.sub(kw_pattern, r"<span class='keyword'>\1</span>", escaped)
        # Strings (simplified)
        escaped = _re.sub(r'(&quot;.*?&quot;)', r"<span class='string'>\1</span>", escaped)
        escaped = _re.sub(r"(&#39;.*?&#39;)", r"<span class='string'>\1</span>", escaped)
        # Comments
        escaped = _re.sub(r'(#[^\n]*)', r"<span class='comment'>\1</span>", escaped)
        # Numbers
        escaped = _re.sub(r'\b(\d+\.?\d*)\b', r"<span class='number'>\1</span>", escaped)
        return escaped

    # -- Export Python Script ----------------------------------------------

    def _export_python(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export as Python Script", "", "Python Files (*.py);;All Files (*)"
        )
        if not path:
            return

        try:
            lines = [
                f"# -*- coding: utf-8 -*-",
                f"# Exported from QuantumRes Notebook",
                f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
            ]
            for i, cell in enumerate(self._cells):
                if cell.cell_type == CELL_TYPE_CODE:
                    source = cell.source.strip()
                    if source:
                        lines.append(f"# --- Cell {i + 1} ---")
                        lines.append(source)
                        lines.append("")
                elif cell.cell_type == CELL_TYPE_MARKDOWN:
                    source = cell.source.strip()
                    if source:
                        lines.append(f"# --- Markdown Cell {i + 1} ---")
                        for md_line in source.split("\n"):
                            lines.append(f"# {md_line}")
                        lines.append("")

            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

            self._emit_log(f"Exported Python script to {path}")
        except Exception as exc:
            self._emit_log(f"Python export error: {exc}")

    # -- Variable Explorer -------------------------------------------------

    def _show_variable_explorer(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Variable Explorer")
        dlg.setMinimumSize(700, 450)
        dlg.setStyleSheet(
            f"QDialog {{ background-color: #1e1e2e; color: {COLOR_TEXT}; }}"
        )

        lay = QVBoxLayout(dlg)

        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["Name", "Type", "Shape/Len", "Size (bytes)", "Value Preview"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setStyleSheet(
            f"QTableWidget {{ background-color: #181825; color: {COLOR_TEXT}; gridline-color: #45475a; }}"
            f"QHeaderView::section {{ background-color: #313244; color: {COLOR_TEXT}; padding: 4px; }}"
        )

        user_vars = {k: v for k, v in self._namespace.items()
                     if not k.startswith("_")
                     and k not in ("np", "pd", "plt", "math", "scipy", "matplotlib")}

        table.setRowCount(len(user_vars))
        for row, (name, val) in enumerate(sorted(user_vars.items())):
            table.setItem(row, 0, QTableWidgetItem(name))
            table.setItem(row, 1, QTableWidgetItem(type(val).__name__))

            # Shape/Len
            shape_str = ""
            if hasattr(val, "shape"):
                shape_str = str(val.shape)
            elif hasattr(val, "__len__"):
                try:
                    shape_str = f"len={len(val)}"
                except Exception:
                    shape_str = "?"
            table.setItem(row, 2, QTableWidgetItem(shape_str))

            # Memory size
            try:
                size = sys.getsizeof(val)
                if hasattr(val, "nbytes"):
                    size = val.nbytes
                if size > 1024 * 1024:
                    size_str = f"{size / (1024*1024):.1f} MB"
                elif size > 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size} B"
            except Exception:
                size_str = "?"
            table.setItem(row, 3, QTableWidgetItem(size_str))

            # Value preview
            try:
                r = repr(val)
                preview = r[:80] + "..." if len(r) > 80 else r
            except Exception:
                preview = "<error>"
            table.setItem(row, 4, QTableWidgetItem(preview))

        lay.addWidget(table)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        lay.addWidget(btn_close)

        dlg.exec_()

    # -- Cell Templates ----------------------------------------------------

    def _show_cell_templates_menu(self):
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; }"
            "QMenu::item:selected { background-color: #585b70; }"
        )

        templates = {
            "Load CSV Data": (
                "import pandas as pd\n\n"
                "# Load CSV data\n"
                "df = pd.read_csv('data.csv')\n"
                "print(f'Shape: {df.shape}')\n"
                "df.head()"
            ),
            "Basic Plot": (
                "import matplotlib.pyplot as plt\n"
                "import numpy as np\n\n"
                "x = np.linspace(0, 10, 100)\n"
                "y = np.sin(x)\n\n"
                "plt.figure(figsize=(8, 4))\n"
                "plt.plot(x, y)\n"
                "plt.xlabel('x')\n"
                "plt.ylabel('y')\n"
                "plt.title('Plot')\n"
                "plt.grid(True, alpha=0.3)\n"
                "plt.show()"
            ),
            "Scatter Plot": (
                "import matplotlib.pyplot as plt\n"
                "import numpy as np\n\n"
                "x = np.random.randn(100)\n"
                "y = x + np.random.randn(100) * 0.5\n\n"
                "plt.figure(figsize=(6, 6))\n"
                "plt.scatter(x, y, alpha=0.6, edgecolors='k', linewidths=0.5)\n"
                "plt.xlabel('X')\n"
                "plt.ylabel('Y')\n"
                "plt.title('Scatter Plot')\n"
                "plt.grid(True, alpha=0.3)\n"
                "plt.show()"
            ),
            "Curve Fitting": (
                "import numpy as np\n"
                "from scipy.optimize import curve_fit\n"
                "import matplotlib.pyplot as plt\n\n"
                "# Define model function\n"
                "def model(x, a, b, c):\n"
                "    return a * np.exp(-b * x) + c\n\n"
                "# Generate sample data\n"
                "xdata = np.linspace(0, 4, 50)\n"
                "ydata = model(xdata, 2.5, 1.3, 0.5) + 0.2 * np.random.randn(50)\n\n"
                "# Fit\n"
                "popt, pcov = curve_fit(model, xdata, ydata)\n"
                "print(f'Parameters: a={popt[0]:.3f}, b={popt[1]:.3f}, c={popt[2]:.3f}')\n\n"
                "plt.figure(figsize=(8, 4))\n"
                "plt.scatter(xdata, ydata, label='Data', s=20)\n"
                "plt.plot(xdata, model(xdata, *popt), 'r-', label='Fit')\n"
                "plt.legend()\n"
                "plt.show()"
            ),
            "Descriptive Statistics": (
                "import pandas as pd\n"
                "import numpy as np\n\n"
                "# Generate sample data\n"
                "data = pd.DataFrame({\n"
                "    'A': np.random.randn(100),\n"
                "    'B': np.random.exponential(2, 100),\n"
                "    'C': np.random.uniform(0, 10, 100),\n"
                "})\n\n"
                "print('=== Descriptive Statistics ===')\n"
                "data.describe()"
            ),
            "Histogram": (
                "import matplotlib.pyplot as plt\n"
                "import numpy as np\n\n"
                "data = np.random.randn(1000)\n\n"
                "plt.figure(figsize=(8, 4))\n"
                "plt.hist(data, bins=40, edgecolor='black', alpha=0.7)\n"
                "plt.xlabel('Value')\n"
                "plt.ylabel('Frequency')\n"
                "plt.title('Histogram')\n"
                "plt.axvline(data.mean(), color='r', linestyle='--', label=f'Mean={data.mean():.2f}')\n"
                "plt.legend()\n"
                "plt.show()"
            ),
            "Linear Algebra": (
                "import numpy as np\n\n"
                "A = np.array([[1, 2], [3, 4]])\n"
                "b = np.array([5, 6])\n\n"
                "# Solve Ax = b\n"
                "x = np.linalg.solve(A, b)\n"
                "print(f'Solution: x = {x}')\n"
                "print(f'Eigenvalues: {np.linalg.eigvals(A)}')\n"
                "print(f'Determinant: {np.linalg.det(A):.4f}')\n"
                "print(f'Inverse:\\n{np.linalg.inv(A)}')"
            ),
        }

        for name, code in templates.items():
            menu.addAction(name, lambda c=code, n=name: self._insert_template_cell(c, n))

        # Position menu at the toolbar button
        menu.exec_(self.mapToGlobal(self._toolbar.pos()))

    def _insert_template_cell(self, code, name):
        idx = 0
        if self._active_cell and self._active_cell in self._cells:
            idx = self._cells.index(self._active_cell) + 1
        cell = self._add_cell(CELL_TYPE_CODE, index=idx, source=code)
        self._emit_log(f"Inserted template: {name}")

    # -- Table of Contents -------------------------------------------------

    def _show_table_of_contents(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Table of Contents")
        dlg.setMinimumSize(400, 350)
        dlg.setStyleSheet(
            f"QDialog {{ background-color: #1e1e2e; color: {COLOR_TEXT}; }}"
        )

        lay = QVBoxLayout(dlg)

        tree = QTreeWidget()
        tree.setHeaderLabels(["Header", "Cell"])
        tree.setStyleSheet(
            f"QTreeWidget {{ background-color: #181825; color: {COLOR_TEXT}; border: none; }}"
            f"QTreeWidget::item:selected {{ background-color: #585b70; }}"
        )

        import re as _re
        toc_items = []
        for i, cell in enumerate(self._cells):
            if cell.cell_type == CELL_TYPE_MARKDOWN:
                for line in cell.source.split("\n"):
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        level = 0
                        for ch in stripped:
                            if ch == "#":
                                level += 1
                            else:
                                break
                        title = stripped[level:].strip()
                        toc_items.append((level, title, i, cell))

        if not toc_items:
            tree.addTopLevelItem(QTreeWidgetItem(["No markdown headers found", ""]))
        else:
            for level, title, cell_idx, cell in toc_items:
                indent = "  " * (level - 1)
                item = QTreeWidgetItem([f"{indent}{'#' * level} {title}", f"Cell {cell_idx + 1}"])
                item.setData(0, Qt.UserRole, cell)
                tree.addTopLevelItem(item)

        def _on_toc_click(item, column):
            cell = item.data(0, Qt.UserRole)
            if cell and cell in self._cells:
                self._set_active_cell(cell)
                self._scroll.ensureWidgetVisible(cell)
            dlg.accept()

        tree.itemDoubleClicked.connect(_on_toc_click)
        lay.addWidget(tree)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        lay.addWidget(btn_close)

        dlg.exec_()

    # -- Save / Load --------------------------------------------------------

    def _serialize(self):
        """Serialize the notebook state to a dict."""
        return {
            "format": "quantumres-notebook",
            "version": 1,
            "created": datetime.now().isoformat(),
            "exec_counter": self._exec_counter,
            "cells": [c.to_dict() for c in self._cells],
        }

    def _load_from_data(self, data):
        """Load notebook state from a dict."""
        # Clear existing cells
        for cell in self._cells[:]:
            self._cell_layout.removeWidget(cell)
            cell.deleteLater()
        self._cells.clear()
        self._active_cell = None

        self._exec_counter = data.get("exec_counter", 0)
        cells_data = data.get("cells", [])
        if not cells_data:
            cells_data = [{"cell_type": CELL_TYPE_CODE, "source": ""}]

        for cd in cells_data:
            cell = NotebookCell.from_dict(cd)
            cell.run_requested.connect(self._run_cell)
            cell.delete_requested.connect(self._delete_cell)
            cell.move_up_requested.connect(self._move_cell_up)
            cell.move_down_requested.connect(self._move_cell_down)
            cell.insert_above_requested.connect(self._insert_cell_above)
            cell.insert_below_requested.connect(self._insert_cell_below)
            cell.focused.connect(self._set_active_cell)
            self._cells.append(cell)
            self._cell_layout.insertWidget(len(self._cells) - 1, cell)

        if self._cells:
            self._set_active_cell(self._cells[0])

        # Reset namespace on load
        self._namespace = {"__builtins__": builtins}
        self._init_namespace()

    def _save_dialog(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Notebook", self._current_path or "",
            "Notebook Files (*.qnb *.json);;All Files (*)"
        )
        if path:
            self.save_as(path)

    def _load_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Notebook", "",
            "Notebook Files (*.qnb *.json);;All Files (*)"
        )
        if path:
            self.load_file(path)

    # -- Helpers ------------------------------------------------------------

    def _emit_log(self, msg):
        self._status.setText(msg)
        if self._log:
            try:
                self._log(msg)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def _escape_html(text):
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
    )


def _markdown_to_html(text):
    """Basic markdown-to-HTML converter (no external dependencies).

    Supports: headers, bold, italic, code blocks, inline code, lists,
    horizontal rules, links, and paragraphs.
    """
    lines = text.split("\n")
    html_parts = []
    in_code_block = False
    in_list = False
    list_type = None  # 'ul' or 'ol'

    for line in lines:
        stripped = line.strip()

        # Fenced code blocks
        if stripped.startswith("```"):
            if in_code_block:
                html_parts.append("</code></pre>")
                in_code_block = False
            else:
                lang = stripped[3:].strip()
                html_parts.append(
                    f"<pre style='background-color:#181825; padding:8px; border-radius:4px;'>"
                    f"<code>"
                )
                in_code_block = True
            continue

        if in_code_block:
            html_parts.append(_escape_html(line))
            html_parts.append("\n")
            continue

        # Close open list if needed
        if in_list and not stripped.startswith(("- ", "* ", "1.", "2.", "3.", "4.", "5.",
                                                 "6.", "7.", "8.", "9.")):
            html_parts.append(f"</{list_type}>")
            in_list = False

        # Horizontal rule
        if stripped in ("---", "***", "___"):
            html_parts.append("<hr style='border-color:#45475a;'>")
            continue

        # Headers
        if stripped.startswith("#"):
            level = 0
            for ch in stripped:
                if ch == "#":
                    level += 1
                else:
                    break
            level = min(level, 6)
            content = _inline_markdown(stripped[level:].strip())
            sizes = {1: "1.6em", 2: "1.4em", 3: "1.2em", 4: "1.1em", 5: "1em", 6: "0.9em"}
            font_size = sizes.get(level, "1em")
            html_parts.append(
                f"<h{level} style='font-size:{font_size}; margin:8px 0;'>"
                f"{content}</h{level}>"
            )
            continue

        # Unordered list
        if stripped.startswith(("- ", "* ")):
            if not in_list or list_type != "ul":
                if in_list:
                    html_parts.append(f"</{list_type}>")
                html_parts.append("<ul style='margin:4px 0 4px 20px;'>")
                in_list = True
                list_type = "ul"
            content = _inline_markdown(stripped[2:])
            html_parts.append(f"<li>{content}</li>")
            continue

        # Ordered list (simple detection)
        if len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in ".)" :
            if not in_list or list_type != "ol":
                if in_list:
                    html_parts.append(f"</{list_type}>")
                html_parts.append("<ol style='margin:4px 0 4px 20px;'>")
                in_list = True
                list_type = "ol"
            content = _inline_markdown(stripped[2:].lstrip())
            html_parts.append(f"<li>{content}</li>")
            continue

        # Empty line
        if not stripped:
            html_parts.append("<br>")
            continue

        # Paragraph
        content = _inline_markdown(stripped)
        html_parts.append(f"<p style='margin:4px 0;'>{content}</p>")

    # Close any open blocks
    if in_code_block:
        html_parts.append("</code></pre>")
    if in_list:
        html_parts.append(f"</{list_type}>")

    return "\n".join(html_parts)


def _inline_markdown(text):
    """Process inline markdown: bold, italic, code, links."""
    import re

    # Inline code
    text = re.sub(r'`([^`]+)`',
                  r"<code style='background-color:#313244; padding:1px 4px; border-radius:2px;'>\1</code>",
                  text)
    # Bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)
    # Italic
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    text = re.sub(r'_(.+?)_', r'<i>\1</i>', text)
    # Links
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)',
                  r"<a style='color:#89b4fa;' href='\2'>\1</a>", text)

    return text
