"""
latex_editor.py - LaTeX Editor Widget for PyQt5 Scientific Suite

Provides a code editor with LaTeX syntax highlighting, live math preview
via matplotlib mathtext, template library, equation builder, symbol palette,
auto-completion for environments, find/replace, and export to .tex / PNG.
"""

import os
import re
import traceback

import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QPlainTextEdit,
    QPushButton, QLabel, QTabWidget, QToolBar, QAction, QFileDialog,
    QComboBox, QGroupBox, QGridLayout, QLineEdit, QCheckBox,
    QMessageBox, QScrollArea, QSizePolicy, QCompleter, QTextEdit,
    QFormLayout, QDialog, QDialogButtonBox, QSpinBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QListWidget, QListWidgetItem,
)
from PyQt5.QtCore import Qt, QRegExp, QTimer, pyqtSignal, QStringListModel
from PyQt5.QtGui import (
    QFont, QColor, QTextCharFormat, QSyntaxHighlighter,
    QTextCursor, QPalette, QKeySequence, QPixmap, QImage,
)

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# LaTeX Syntax Highlighter
# ---------------------------------------------------------------------------

class LaTeXSyntaxHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for LaTeX source code."""

    def __init__(self, document):
        super().__init__(document)
        self._rules = []

        # Commands  \commandname
        cmd_fmt = QTextCharFormat()
        cmd_fmt.setForeground(QColor("#CC7832"))
        cmd_fmt.setFontWeight(QFont.Bold)
        self._rules.append((QRegExp(r"\\[A-Za-z]+"), cmd_fmt))

        # Environments \begin{...} \end{...}
        env_fmt = QTextCharFormat()
        env_fmt.setForeground(QColor("#6897BB"))
        env_fmt.setFontWeight(QFont.Bold)
        self._rules.append((QRegExp(r"\\(?:begin|end)\{[^}]*\}"), env_fmt))

        # Math delimiters $ ... $ and $$ ... $$
        math_fmt = QTextCharFormat()
        math_fmt.setForeground(QColor("#6A8759"))
        self._rules.append((QRegExp(r"\$\$?[^$]*\$\$?"), math_fmt))

        # Braces
        brace_fmt = QTextCharFormat()
        brace_fmt.setForeground(QColor("#A9B7C6"))
        brace_fmt.setFontWeight(QFont.Bold)
        self._rules.append((QRegExp(r"[{}]"), brace_fmt))

        # Brackets
        bracket_fmt = QTextCharFormat()
        bracket_fmt.setForeground(QColor("#A9B7C6"))
        self._rules.append((QRegExp(r"[\[\]]"), bracket_fmt))

        # Comments  % ...
        self._comment_fmt = QTextCharFormat()
        self._comment_fmt.setForeground(QColor("#808080"))
        self._comment_fmt.setFontItalic(True)

        # Numbers
        num_fmt = QTextCharFormat()
        num_fmt.setForeground(QColor("#6897BB"))
        self._rules.append((QRegExp(r"\b[0-9]+(?:\.[0-9]+)?\b"), num_fmt))

    def highlightBlock(self, text):
        for pattern, fmt in self._rules:
            index = pattern.indexIn(text, 0)
            while index >= 0:
                length = pattern.matchedLength()
                self.setFormat(index, length, fmt)
                index = pattern.indexIn(text, index + length)
        # Comments (override everything after %)
        ci = text.find("%")
        while ci >= 0:
            # Check it is not escaped
            if ci == 0 or text[ci - 1] != "\\":
                self.setFormat(ci, len(text) - ci, self._comment_fmt)
                break
            ci = text.find("%", ci + 1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEMPLATES = {
    "Article": (
        "\\documentclass{article}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{amsmath, amssymb}\n\n"
        "\\title{Title}\n\\author{Author}\n\\date{\\today}\n\n"
        "\\begin{document}\n\\maketitle\n\n"
        "\\section{Introduction}\n\n\\end{document}\n"
    ),
    "Report": (
        "\\documentclass{report}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{amsmath, amssymb, graphicx}\n\n"
        "\\title{Report Title}\n\\author{Author}\n\\date{\\today}\n\n"
        "\\begin{document}\n\\maketitle\n\\tableofcontents\n\n"
        "\\chapter{Introduction}\n\n\\end{document}\n"
    ),
    "Letter": (
        "\\documentclass{letter}\n"
        "\\usepackage[utf8]{inputenc}\n\n"
        "\\signature{Your Name}\n"
        "\\address{Your Address}\n\n"
        "\\begin{document}\n"
        "\\begin{letter}{Recipient \\\\ Address}\n"
        "\\opening{Dear Sir/Madam,}\n\n"
        "Body of the letter.\n\n"
        "\\closing{Yours sincerely,}\n"
        "\\end{letter}\n\\end{document}\n"
    ),
    "Beamer": (
        "\\documentclass{beamer}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usetheme{Madrid}\n\n"
        "\\title{Presentation Title}\n\\author{Author}\n"
        "\\date{\\today}\n\n"
        "\\begin{document}\n\n"
        "\\begin{frame}\n\\titlepage\n\\end{frame}\n\n"
        "\\begin{frame}{Slide Title}\n"
        "Content here.\n"
        "\\end{frame}\n\n\\end{document}\n"
    ),
    "CV": (
        "\\documentclass[11pt,a4paper]{article}\n"
        "\\usepackage[margin=1in]{geometry}\n"
        "\\usepackage{enumitem, titlesec}\n\n"
        "\\titleformat{\\section}{\\large\\bfseries}{}{0em}{}\n"
        "[\\titlerule]\n\n"
        "\\begin{document}\n\n"
        "\\begin{center}\n"
        "{\\LARGE\\bfseries Your Name}\\\\[4pt]\n"
        "email@example.com | (555) 123-4567\n"
        "\\end{center}\n\n"
        "\\section{Education}\n\n"
        "\\section{Experience}\n\n"
        "\\section{Skills}\n\n\\end{document}\n"
    ),
}

EQUATION_TEMPLATES = {
    "Fraction": "\\frac{a}{b}",
    "Square Root": "\\sqrt{x}",
    "Nth Root": "\\sqrt[n]{x}",
    "Integral": "\\int_{a}^{b} f(x)\\, dx",
    "Double Integral": "\\iint_{D} f(x,y)\\, dA",
    "Sum": "\\sum_{i=1}^{n} a_i",
    "Product": "\\prod_{i=1}^{n} a_i",
    "Limit": "\\lim_{x \\to \\infty} f(x)",
    "Matrix 2x2": "\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}",
    "Matrix 3x3": "\\begin{pmatrix} a & b & c \\\\ d & e & f \\\\ g & h & i \\end{pmatrix}",
    "Derivative": "\\frac{d}{dx} f(x)",
    "Partial": "\\frac{\\partial f}{\\partial x}",
    "Binomial": "\\binom{n}{k}",
    "Cases": "\\begin{cases} x & \\text{if } x \\geq 0 \\\\ -x & \\text{if } x < 0 \\end{cases}",
    "Aligned": "\\begin{aligned} a &= b + c \\\\ d &= e + f \\end{aligned}",
}

GREEK_LETTERS = [
    "\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon", "\\zeta",
    "\\eta", "\\theta", "\\iota", "\\kappa", "\\lambda", "\\mu",
    "\\nu", "\\xi", "\\pi", "\\rho", "\\sigma", "\\tau",
    "\\upsilon", "\\phi", "\\chi", "\\psi", "\\omega",
    "\\Gamma", "\\Delta", "\\Theta", "\\Lambda", "\\Xi",
    "\\Pi", "\\Sigma", "\\Phi", "\\Psi", "\\Omega",
]

OPERATORS = [
    "\\pm", "\\mp", "\\times", "\\div", "\\cdot", "\\ast",
    "\\star", "\\circ", "\\bullet", "\\oplus", "\\otimes",
    "\\nabla", "\\partial", "\\infty", "\\forall", "\\exists",
    "\\neg", "\\wedge", "\\vee", "\\cap", "\\cup",
]

ARROWS = [
    "\\leftarrow", "\\rightarrow", "\\leftrightarrow",
    "\\Leftarrow", "\\Rightarrow", "\\Leftrightarrow",
    "\\uparrow", "\\downarrow", "\\mapsto", "\\longmapsto",
    "\\nearrow", "\\searrow", "\\swarrow", "\\nwarrow",
]

RELATIONS = [
    "\\leq", "\\geq", "\\neq", "\\approx", "\\equiv",
    "\\sim", "\\simeq", "\\cong", "\\propto", "\\subset",
    "\\supset", "\\subseteq", "\\supseteq", "\\in", "\\notin",
    "\\ni", "\\parallel", "\\perp", "\\mid",
]

ENVIRONMENTS = [
    "document", "equation", "equation*", "align", "align*",
    "gather", "gather*", "figure", "table", "tabular",
    "itemize", "enumerate", "description", "verbatim",
    "abstract", "quote", "center", "flushleft", "flushright",
    "minipage", "array", "matrix", "pmatrix", "bmatrix",
    "cases", "frame", "block", "theorem", "proof",
]


# ---------------------------------------------------------------------------
# LaTeX Editor Widget
# ---------------------------------------------------------------------------

class LaTeXEditorWidget(QWidget):
    """Full-featured LaTeX editor with live math preview."""

    contentChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._log = lambda msg: None
        self._current_path = None
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(600)
        self._preview_timer.timeout.connect(self._update_preview)
        self._init_ui()

    # -- logging -----------------------------------------------------------

    def set_logger(self, fn):
        """Attach an external logging callback ``fn(str)``."""
        self._log = fn

    def _emit_log(self, msg):
        try:
            self._log(msg)
        except Exception:
            pass

    # -- UI ----------------------------------------------------------------

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self._add_toolbar_actions(toolbar)
        root.addWidget(toolbar)

        # Main splitter: editor | right panel
        splitter = QSplitter(Qt.Horizontal)

        # Left: code editor
        editor_container = QWidget()
        editor_lay = QVBoxLayout(editor_container)
        editor_lay.setContentsMargins(0, 0, 0, 0)

        self._editor = QPlainTextEdit()
        font = QFont("Consolas", 11)
        font.setStyleHint(QFont.Monospace)
        self._editor.setFont(font)
        self._editor.setTabStopDistance(
            self._editor.fontMetrics().horizontalAdvance(" ") * 4
        )
        self._editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        self._highlighter = LaTeXSyntaxHighlighter(self._editor.document())
        self._editor.textChanged.connect(self._on_text_changed)
        self._setup_completer()

        # Find/replace bar
        find_bar = self._build_find_replace_bar()

        editor_lay.addWidget(self._editor, 1)
        editor_lay.addWidget(find_bar)
        splitter.addWidget(editor_container)

        # Right: tabs (Preview / Templates / Equations / Symbols)
        right_tabs = QTabWidget()

        # Preview tab
        preview_widget = QWidget()
        pv_lay = QVBoxLayout(preview_widget)
        pv_lay.setContentsMargins(2, 2, 2, 2)
        self._figure = Figure(figsize=(5, 4), dpi=100)
        self._figure.patch.set_facecolor("white")
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        pv_lay.addWidget(self._canvas)
        btn_refresh = QPushButton("Refresh Preview")
        btn_refresh.clicked.connect(self._update_preview)
        pv_lay.addWidget(btn_refresh)
        right_tabs.addTab(preview_widget, "Preview")

        # Templates tab
        right_tabs.addTab(self._build_templates_tab(), "Templates")

        # Equations tab
        right_tabs.addTab(self._build_equations_tab(), "Equations")

        # Symbols tab
        right_tabs.addTab(self._build_symbols_tab(), "Symbols")

        # Document Generator tab
        right_tabs.addTab(self._build_document_generator_tab(), "Doc Gen")

        # Table Generator tab
        right_tabs.addTab(self._build_table_generator_tab(), "Tables")

        # Figure Generator tab
        right_tabs.addTab(self._build_figure_generator_tab(), "Figures")

        # Bibliography Manager tab
        right_tabs.addTab(self._build_bibliography_tab(), "Bib")

        splitter.addWidget(right_tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)

        # Status label
        self._status = QLabel("Ready")
        root.addWidget(self._status)

    # -- toolbar -----------------------------------------------------------

    def _add_toolbar_actions(self, toolbar):
        actions = [
            ("New", self._action_new),
            ("Open", self._action_open),
            ("Save", self._action_save),
            ("Save As", self._action_save_as),
            (None, None),
            ("Bold", lambda: self._insert_command("\\textbf{", "}")),
            ("Italic", lambda: self._insert_command("\\textit{", "}")),
            ("Underline", lambda: self._insert_command("\\underline{", "}")),
            (None, None),
            ("$ Math $", lambda: self._insert_command("$", "$")),
            ("Equation", lambda: self._insert_env("equation")),
            ("Align", lambda: self._insert_env("align")),
            (None, None),
            ("Eq Num", lambda: self._insert_numbered_equation()),
            ("Ref", lambda: self._insert_cross_reference()),
            (None, None),
            ("Export PNG", self._export_png),
        ]
        for name, callback in actions:
            if name is None:
                toolbar.addSeparator()
            else:
                act = QAction(name, self)
                act.triggered.connect(callback)
                toolbar.addAction(act)

    # -- completer ---------------------------------------------------------

    def _setup_completer(self):
        env_strings = [f"\\begin{{{e}}}" for e in ENVIRONMENTS]
        cmd_strings = GREEK_LETTERS + OPERATORS + ARROWS + RELATIONS
        all_words = env_strings + cmd_strings + [
            "\\section{}", "\\subsection{}", "\\subsubsection{}",
            "\\textbf{}", "\\textit{}", "\\underline{}", "\\emph{}",
            "\\cite{}", "\\ref{}", "\\label{}", "\\footnote{}",
            "\\includegraphics{}", "\\caption{}", "\\usepackage{}",
        ]
        self._completer_model = QStringListModel(sorted(set(all_words)))
        self._completer = QCompleter()
        self._completer.setModel(self._completer_model)
        self._completer.setCompletionMode(QCompleter.PopupCompletion)
        self._completer.setCaseSensitivity(Qt.CaseSensitive)
        self._completer.setWidget(self._editor)
        self._completer.activated.connect(self._insert_completion)
        self._editor.textChanged.connect(self._trigger_completer)

    def _trigger_completer(self):
        cursor = self._editor.textCursor()
        cursor.movePosition(QTextCursor.StartOfBlock, QTextCursor.KeepAnchor)
        line = cursor.selectedText()
        # Find current word starting with backslash
        match = re.search(r"(\\[A-Za-z*{]*)$", line)
        if match and len(match.group(1)) >= 2:
            prefix = match.group(1)
            self._completer.setCompletionPrefix(prefix)
            if self._completer.completionCount() > 0:
                popup = self._completer.popup()
                popup.setCurrentIndex(self._completer.completionModel().index(0, 0))
                cr = self._editor.cursorRect()
                cr.setWidth(260)
                self._completer.complete(cr)
            else:
                self._completer.popup().hide()
        else:
            self._completer.popup().hide()

    def _insert_completion(self, completion):
        cursor = self._editor.textCursor()
        # Remove the already-typed prefix
        prefix = self._completer.completionPrefix()
        for _ in range(len(prefix)):
            cursor.deletePreviousChar()
        cursor.insertText(completion)
        # Auto-close \begin{env} with \end{env}
        m = re.match(r"\\begin\{([^}]+)\}", completion)
        if m:
            env = m.group(1)
            cursor.insertText(f"\n\n\\end{{{env}}}")
            cursor.movePosition(QTextCursor.Up)
        self._editor.setTextCursor(cursor)

    # -- find / replace ----------------------------------------------------

    def _build_find_replace_bar(self):
        bar = QGroupBox()
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(4, 2, 4, 2)

        lay.addWidget(QLabel("Find:"))
        self._find_input = QLineEdit()
        self._find_input.setMaximumWidth(180)
        lay.addWidget(self._find_input)

        btn_find = QPushButton("Find")
        btn_find.clicked.connect(self._find_next)
        lay.addWidget(btn_find)

        lay.addWidget(QLabel("Replace:"))
        self._replace_input = QLineEdit()
        self._replace_input.setMaximumWidth(180)
        lay.addWidget(self._replace_input)

        btn_replace = QPushButton("Replace")
        btn_replace.clicked.connect(self._replace_current)
        lay.addWidget(btn_replace)

        btn_replace_all = QPushButton("Replace All")
        btn_replace_all.clicked.connect(self._replace_all)
        lay.addWidget(btn_replace_all)

        self._find_case = QCheckBox("Aa")
        self._find_case.setToolTip("Case sensitive")
        lay.addWidget(self._find_case)

        lay.addStretch()
        return bar

    def _find_next(self):
        text = self._find_input.text()
        if not text:
            return
        flags = QTextCursor.MoveAnchor
        doc = self._editor.document()
        cursor = self._editor.textCursor()
        if self._find_case.isChecked():
            found = doc.find(text, cursor)
        else:
            found = doc.find(text, cursor)
        if not found.isNull():
            self._editor.setTextCursor(found)
        else:
            # Wrap around
            cursor2 = QTextCursor(doc)
            found2 = doc.find(text, cursor2)
            if not found2.isNull():
                self._editor.setTextCursor(found2)
            else:
                self._status.setText("Not found")

    def _replace_current(self):
        cursor = self._editor.textCursor()
        if cursor.hasSelection() and cursor.selectedText() == self._find_input.text():
            cursor.insertText(self._replace_input.text())
        self._find_next()

    def _replace_all(self):
        text = self._find_input.text()
        repl = self._replace_input.text()
        if not text:
            return
        content = self._editor.toPlainText()
        if self._find_case.isChecked():
            new_content = content.replace(text, repl)
        else:
            new_content = re.sub(re.escape(text), repl, content, flags=re.IGNORECASE)
        count = content.count(text) if self._find_case.isChecked() else len(
            re.findall(re.escape(text), content, re.IGNORECASE)
        )
        self._editor.setPlainText(new_content)
        self._status.setText(f"Replaced {count} occurrences")

    # -- templates ---------------------------------------------------------

    def _build_templates_tab(self):
        widget = QWidget()
        lay = QVBoxLayout(widget)
        lay.addWidget(QLabel("Select a document template:"))
        for name, content in TEMPLATES.items():
            btn = QPushButton(name)
            btn.clicked.connect(lambda checked, c=content, n=name: self._apply_template(n, c))
            lay.addWidget(btn)
        lay.addStretch()
        return widget

    def _apply_template(self, name, content):
        if self._editor.toPlainText().strip():
            reply = QMessageBox.question(
                self, "Replace content?",
                "Current content will be replaced. Continue?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        self._editor.setPlainText(content)
        self._emit_log(f"Applied template: {name}")

    # -- equations ---------------------------------------------------------

    def _build_equations_tab(self):
        widget = QWidget()
        lay = QVBoxLayout(widget)
        lay.addWidget(QLabel("Insert equation template:"))
        for name, latex in EQUATION_TEMPLATES.items():
            btn = QPushButton(name)
            btn.setToolTip(latex)
            btn.clicked.connect(lambda checked, l=latex: self._insert_at_cursor(l))
            lay.addWidget(btn)
        lay.addStretch()
        return widget

    # -- symbols -----------------------------------------------------------

    def _build_symbols_tab(self):
        tabs = QTabWidget()
        tabs.addTab(self._symbol_grid(GREEK_LETTERS), "Greek")
        tabs.addTab(self._symbol_grid(OPERATORS), "Operators")
        tabs.addTab(self._symbol_grid(ARROWS), "Arrows")
        tabs.addTab(self._symbol_grid(RELATIONS), "Relations")
        return tabs

    def _symbol_grid(self, symbols):
        widget = QWidget()
        grid = QGridLayout(widget)
        grid.setSpacing(2)
        cols = 4
        for i, sym in enumerate(symbols):
            btn = QPushButton(sym)
            btn.setFixedHeight(28)
            btn.setToolTip(sym)
            btn.clicked.connect(lambda checked, s=sym: self._insert_at_cursor(s + " "))
            grid.addWidget(btn, i // cols, i % cols)
        return widget

    # -- insertion helpers --------------------------------------------------

    def _insert_at_cursor(self, text):
        cursor = self._editor.textCursor()
        cursor.insertText(text)
        self._editor.setTextCursor(cursor)
        self._editor.setFocus()

    def _insert_command(self, prefix, suffix):
        cursor = self._editor.textCursor()
        sel = cursor.selectedText()
        cursor.insertText(prefix + sel + suffix)
        if not sel:
            cursor.movePosition(QTextCursor.Left, n=len(suffix))
        self._editor.setTextCursor(cursor)
        self._editor.setFocus()

    def _insert_env(self, env):
        cursor = self._editor.textCursor()
        cursor.insertText(f"\\begin{{{env}}}\n\n\\end{{{env}}}")
        cursor.movePosition(QTextCursor.Up)
        self._editor.setTextCursor(cursor)
        self._editor.setFocus()

    # -- document generator ------------------------------------------------

    def _build_document_generator_tab(self):
        widget = QWidget()
        lay = QVBoxLayout(widget)
        lay.addWidget(QLabel("Generate a complete LaTeX document:"))

        form = QFormLayout()
        self._docgen_title = QLineEdit("My Document")
        form.addRow("Title:", self._docgen_title)
        self._docgen_author = QLineEdit("Author Name")
        form.addRow("Author:", self._docgen_author)
        self._docgen_class = QComboBox()
        self._docgen_class.addItems(["article", "report", "book", "memoir"])
        form.addRow("Class:", self._docgen_class)

        self._docgen_abstract = QCheckBox("Include abstract")
        self._docgen_abstract.setChecked(True)
        form.addRow(self._docgen_abstract)

        self._docgen_toc = QCheckBox("Table of contents")
        form.addRow(self._docgen_toc)

        self._docgen_bib = QCheckBox("Bibliography section")
        form.addRow(self._docgen_bib)

        self._docgen_figures = QCheckBox("Sample figure environment")
        form.addRow(self._docgen_figures)

        self._docgen_tables = QCheckBox("Sample table")
        form.addRow(self._docgen_tables)

        self._docgen_sections = QSpinBox()
        self._docgen_sections.setRange(1, 10)
        self._docgen_sections.setValue(3)
        form.addRow("Sections:", self._docgen_sections)

        lay.addLayout(form)

        btn = QPushButton("Generate Document")
        btn.clicked.connect(self._generate_full_document)
        lay.addWidget(btn)
        lay.addStretch()
        return widget

    def _generate_full_document(self):
        doc_class = self._docgen_class.currentText()
        title = self._docgen_title.text() or "Title"
        author = self._docgen_author.text() or "Author"
        n_sections = self._docgen_sections.value()

        packages = [
            "\\usepackage[utf8]{inputenc}",
            "\\usepackage[T1]{fontenc}",
            "\\usepackage{amsmath, amssymb, amsthm}",
            "\\usepackage{graphicx}",
            "\\usepackage{booktabs}",
            "\\usepackage{hyperref}",
            "\\usepackage[margin=1in]{geometry}",
        ]
        if self._docgen_bib.isChecked():
            packages.append("\\usepackage{natbib}")

        lines = [f"\\documentclass{{{doc_class}}}"]
        lines.extend(packages)
        lines.append("")
        lines.append(f"\\title{{{title}}}")
        lines.append(f"\\author{{{author}}}")
        lines.append("\\date{\\today}")
        lines.append("")
        lines.append("\\begin{document}")
        lines.append("\\maketitle")

        if self._docgen_abstract.isChecked():
            lines.append("")
            lines.append("\\begin{abstract}")
            lines.append("Enter your abstract text here.")
            lines.append("\\end{abstract}")

        if self._docgen_toc.isChecked():
            lines.append("")
            lines.append("\\tableofcontents")
            lines.append("\\newpage")

        for i in range(1, n_sections + 1):
            lines.append("")
            lines.append(f"\\section{{Section {i}}}")
            lines.append(f"\\label{{sec:section{i}}}")
            lines.append(f"Content for section {i}.")

        if self._docgen_figures.isChecked():
            lines.append("")
            lines.append("\\begin{figure}[htbp]")
            lines.append("    \\centering")
            lines.append("    \\includegraphics[width=0.8\\textwidth]{figure.png}")
            lines.append("    \\caption{A sample figure.}")
            lines.append("    \\label{fig:sample}")
            lines.append("\\end{figure}")

        if self._docgen_tables.isChecked():
            lines.append("")
            lines.append("\\begin{table}[htbp]")
            lines.append("    \\centering")
            lines.append("    \\caption{A sample table.}")
            lines.append("    \\label{tab:sample}")
            lines.append("    \\begin{tabular}{lcc}")
            lines.append("        \\toprule")
            lines.append("        Item & Value & Unit \\\\")
            lines.append("        \\midrule")
            lines.append("        Alpha & 1.0 & m/s \\\\")
            lines.append("        Beta  & 2.5 & kg  \\\\")
            lines.append("        \\bottomrule")
            lines.append("    \\end{tabular}")
            lines.append("\\end{table}")

        if self._docgen_bib.isChecked():
            lines.append("")
            lines.append("\\bibliographystyle{plain}")
            lines.append("\\bibliography{references}")

        lines.append("")
        lines.append("\\end{document}")

        content = "\n".join(lines)
        if self._editor.toPlainText().strip():
            reply = QMessageBox.question(
                self, "Replace content?",
                "Current content will be replaced. Continue?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        self._editor.setPlainText(content)
        self._emit_log("Generated full document")

    # -- table generator ---------------------------------------------------

    def _build_table_generator_tab(self):
        widget = QWidget()
        lay = QVBoxLayout(widget)
        lay.addWidget(QLabel("Generate a LaTeX table with booktabs:"))

        form = QFormLayout()
        self._tbl_rows = QSpinBox()
        self._tbl_rows.setRange(1, 50)
        self._tbl_rows.setValue(3)
        form.addRow("Rows:", self._tbl_rows)

        self._tbl_cols = QSpinBox()
        self._tbl_cols.setRange(1, 20)
        self._tbl_cols.setValue(3)
        form.addRow("Columns:", self._tbl_cols)

        self._tbl_caption = QLineEdit("Table caption")
        form.addRow("Caption:", self._tbl_caption)

        self._tbl_label = QLineEdit("tab:mytable")
        form.addRow("Label:", self._tbl_label)

        self._tbl_position = QComboBox()
        self._tbl_position.addItems(["htbp", "h", "t", "b", "p", "H"])
        form.addRow("Position:", self._tbl_position)

        lay.addLayout(form)

        lay.addWidget(QLabel("Enter data (editable grid):"))
        self._tbl_grid = QTableWidget(3, 3)
        self._tbl_grid.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for r in range(3):
            for c in range(3):
                if r == 0:
                    self._tbl_grid.setItem(r, c, QTableWidgetItem(f"Header {c+1}"))
                else:
                    self._tbl_grid.setItem(r, c, QTableWidgetItem(f"r{r}c{c+1}"))
        lay.addWidget(self._tbl_grid)

        self._tbl_rows.valueChanged.connect(self._resize_table_grid)
        self._tbl_cols.valueChanged.connect(self._resize_table_grid)

        btn = QPushButton("Insert Table Code")
        btn.clicked.connect(self._insert_table_code)
        lay.addWidget(btn)
        return widget

    def _resize_table_grid(self):
        rows = self._tbl_rows.value()
        cols = self._tbl_cols.value()
        self._tbl_grid.setRowCount(rows)
        self._tbl_grid.setColumnCount(cols)

    def _insert_table_code(self):
        rows = self._tbl_grid.rowCount()
        cols = self._tbl_grid.columnCount()
        pos = self._tbl_position.currentText()
        caption = self._tbl_caption.text()
        label = self._tbl_label.text()

        col_spec = "l" + "c" * (cols - 1)
        lines = [
            f"\\begin{{table}}[{pos}]",
            "    \\centering",
            f"    \\caption{{{caption}}}",
            f"    \\label{{{label}}}",
            f"    \\begin{{tabular}}{{{col_spec}}}",
            "        \\toprule",
        ]

        for r in range(rows):
            cells = []
            for c in range(cols):
                item = self._tbl_grid.item(r, c)
                cells.append(item.text() if item else "")
            row_str = " & ".join(cells) + " \\\\"
            lines.append(f"        {row_str}")
            if r == 0:
                lines.append("        \\midrule")

        lines.append("        \\bottomrule")
        lines.append("    \\end{tabular}")
        lines.append("\\end{table}")

        self._insert_at_cursor("\n".join(lines) + "\n")
        self._emit_log("Inserted table code")

    # -- figure generator --------------------------------------------------

    def _build_figure_generator_tab(self):
        widget = QWidget()
        lay = QVBoxLayout(widget)
        lay.addWidget(QLabel("Generate a figure environment:"))

        form = QFormLayout()
        self._fig_path = QLineEdit("figures/image.png")
        form.addRow("Image path:", self._fig_path)

        self._fig_width = QComboBox()
        self._fig_width.addItems([
            "0.5\\textwidth", "0.6\\textwidth", "0.7\\textwidth",
            "0.8\\textwidth", "0.9\\textwidth", "\\textwidth",
            "0.5\\linewidth", "\\linewidth",
        ])
        self._fig_width.setCurrentIndex(3)
        self._fig_width.setEditable(True)
        form.addRow("Width:", self._fig_width)

        self._fig_caption = QLineEdit("Figure caption.")
        form.addRow("Caption:", self._fig_caption)

        self._fig_label = QLineEdit("fig:myfigure")
        form.addRow("Label:", self._fig_label)

        self._fig_position = QComboBox()
        self._fig_position.addItems(["htbp", "h", "t", "b", "p", "H"])
        form.addRow("Position:", self._fig_position)

        self._fig_centering = QCheckBox("Centering")
        self._fig_centering.setChecked(True)
        form.addRow(self._fig_centering)

        lay.addLayout(form)

        btn_single = QPushButton("Insert Figure")
        btn_single.clicked.connect(self._insert_figure_code)
        lay.addWidget(btn_single)

        lay.addWidget(QLabel("--- Subfigures ---"))
        self._fig_sub_count = QSpinBox()
        self._fig_sub_count.setRange(2, 6)
        self._fig_sub_count.setValue(2)
        sub_form = QFormLayout()
        sub_form.addRow("Subfigure count:", self._fig_sub_count)
        lay.addLayout(sub_form)

        btn_sub = QPushButton("Insert Subfigures")
        btn_sub.clicked.connect(self._insert_subfigure_code)
        lay.addWidget(btn_sub)

        lay.addStretch()
        return widget

    def _insert_figure_code(self):
        pos = self._fig_position.currentText()
        path = self._fig_path.text()
        width = self._fig_width.currentText()
        caption = self._fig_caption.text()
        label = self._fig_label.text()
        centering = "    \\centering\n" if self._fig_centering.isChecked() else ""

        code = (
            f"\\begin{{figure}}[{pos}]\n"
            f"{centering}"
            f"    \\includegraphics[width={width}]{{{path}}}\n"
            f"    \\caption{{{caption}}}\n"
            f"    \\label{{{label}}}\n"
            f"\\end{{figure}}\n"
        )
        self._insert_at_cursor(code)
        self._emit_log("Inserted figure environment")

    def _insert_subfigure_code(self):
        n = self._fig_sub_count.value()
        pos = self._fig_position.currentText()
        sub_width = f"{0.9 / n:.2f}\\textwidth"
        lines = [f"\\begin{{figure}}[{pos}]", "    \\centering"]
        for i in range(1, n + 1):
            lines.append(f"    \\begin{{subfigure}}{{{sub_width}}}")
            lines.append("        \\centering")
            lines.append(f"        \\includegraphics[width=\\textwidth]{{subfig{i}.png}}")
            lines.append(f"        \\caption{{Subfigure {i}}}")
            lines.append(f"        \\label{{fig:sub{i}}}")
            lines.append("    \\end{subfigure}")
            if i < n:
                lines.append("    \\hfill")
        lines.append("    \\caption{Combined figure}")
        lines.append("    \\label{fig:combined}")
        lines.append("\\end{figure}")
        self._insert_at_cursor("\n".join(lines) + "\n")
        self._emit_log("Inserted subfigure environment")

    # -- bibliography manager ----------------------------------------------

    def _build_bibliography_tab(self):
        widget = QWidget()
        lay = QVBoxLayout(widget)
        lay.addWidget(QLabel("Bibliography Manager"))

        # BibTeX entry generator
        form = QFormLayout()
        self._bib_type = QComboBox()
        self._bib_type.addItems(["article", "book", "inproceedings", "phdthesis",
                                  "misc", "techreport", "unpublished"])
        form.addRow("Entry type:", self._bib_type)

        self._bib_key = QLineEdit("author2024")
        form.addRow("Cite key:", self._bib_key)

        self._bib_author = QLineEdit("Last, First and Last2, First2")
        form.addRow("Author:", self._bib_author)

        self._bib_title = QLineEdit("Paper Title")
        form.addRow("Title:", self._bib_title)

        self._bib_year = QLineEdit("2024")
        form.addRow("Year:", self._bib_year)

        self._bib_journal = QLineEdit("Journal Name")
        form.addRow("Journal/Book:", self._bib_journal)

        self._bib_volume = QLineEdit("")
        form.addRow("Volume:", self._bib_volume)

        self._bib_pages = QLineEdit("")
        form.addRow("Pages:", self._bib_pages)

        self._bib_doi = QLineEdit("")
        form.addRow("DOI:", self._bib_doi)

        lay.addLayout(form)

        btn_row = QHBoxLayout()
        btn_bib = QPushButton("Copy BibTeX Entry")
        btn_bib.clicked.connect(self._generate_bibtex_entry)
        btn_row.addWidget(btn_bib)

        btn_cite = QPushButton("Insert \\cite{}")
        btn_cite.clicked.connect(self._insert_cite_command)
        btn_row.addWidget(btn_cite)
        lay.addLayout(btn_row)

        # Entries list
        lay.addWidget(QLabel("Stored entries:"))
        self._bib_list = QListWidget()
        lay.addWidget(self._bib_list)
        self._bib_entries = []

        btn_row2 = QHBoxLayout()
        btn_add = QPushButton("Add to List")
        btn_add.clicked.connect(self._add_bib_entry)
        btn_row2.addWidget(btn_add)

        btn_export = QPushButton("Export .bib File")
        btn_export.clicked.connect(self._export_bib_file)
        btn_row2.addWidget(btn_export)
        lay.addLayout(btn_row2)

        return widget

    def _make_bibtex_string(self):
        btype = self._bib_type.currentText()
        key = self._bib_key.text() or "key"
        fields = []
        if self._bib_author.text():
            fields.append(f"  author    = {{{self._bib_author.text()}}}")
        if self._bib_title.text():
            fields.append(f"  title     = {{{self._bib_title.text()}}}")
        if self._bib_year.text():
            fields.append(f"  year      = {{{self._bib_year.text()}}}")
        if self._bib_journal.text():
            jfield = "journal" if btype == "article" else "booktitle"
            fields.append(f"  {jfield:<9s} = {{{self._bib_journal.text()}}}")
        if self._bib_volume.text():
            fields.append(f"  volume    = {{{self._bib_volume.text()}}}")
        if self._bib_pages.text():
            fields.append(f"  pages     = {{{self._bib_pages.text()}}}")
        if self._bib_doi.text():
            fields.append(f"  doi       = {{{self._bib_doi.text()}}}")
        entry = f"@{btype}{{{key},\n" + ",\n".join(fields) + "\n}"
        return entry

    def _generate_bibtex_entry(self):
        from PyQt5.QtWidgets import QApplication as _QApp
        entry = self._make_bibtex_string()
        _QApp.clipboard().setText(entry)
        self._status.setText("BibTeX entry copied to clipboard")
        self._emit_log("BibTeX entry copied")

    def _insert_cite_command(self):
        key = self._bib_key.text() or "key"
        self._insert_at_cursor(f"\\cite{{{key}}}")

    def _add_bib_entry(self):
        entry = self._make_bibtex_string()
        key = self._bib_key.text() or "key"
        self._bib_entries.append(entry)
        self._bib_list.addItem(f"{self._bib_type.currentText()}: {key}")
        self._emit_log(f"Added bib entry: {key}")

    def _export_bib_file(self):
        if not self._bib_entries:
            QMessageBox.warning(self, "Export", "No bibliography entries to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save BibTeX File", "", "BibTeX Files (*.bib);;All Files (*)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(self._bib_entries) + "\n")
            self._status.setText(f"Exported {len(self._bib_entries)} entries to {os.path.basename(path)}")
            self._emit_log(f"Exported bib file: {path}")

    # -- equation numbering and cross-referencing --------------------------

    def _insert_numbered_equation(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Numbered Equation")
        dlg_lay = QVBoxLayout(dlg)

        form = QFormLayout()
        eq_input = QLineEdit("E = mc^2")
        form.addRow("Equation:", eq_input)
        label_input = QLineEdit("eq:myeq")
        form.addRow("Label:", label_input)
        dlg_lay.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        dlg_lay.addWidget(buttons)

        if dlg.exec_() == QDialog.Accepted:
            eq = eq_input.text()
            label = label_input.text()
            code = (
                f"\\begin{{equation}}\n"
                f"    {eq}\n"
                f"    \\label{{{label}}}\n"
                f"\\end{{equation}}\n"
            )
            self._insert_at_cursor(code)
            self._emit_log(f"Inserted numbered equation: {label}")

    def _insert_cross_reference(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Cross Reference")
        dlg_lay = QVBoxLayout(dlg)

        form = QFormLayout()
        label_input = QLineEdit("eq:myeq")
        form.addRow("Label:", label_input)

        ref_type = QComboBox()
        ref_type.addItems(["\\ref", "\\eqref", "\\pageref", "\\autoref"])
        form.addRow("Ref type:", ref_type)
        dlg_lay.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        dlg_lay.addWidget(buttons)

        if dlg.exec_() == QDialog.Accepted:
            label = label_input.text()
            cmd = ref_type.currentText()
            self._insert_at_cursor(f"{cmd}{{{label}}}")

    # -- preview -----------------------------------------------------------

    def _on_text_changed(self):
        self._preview_timer.start()
        self.contentChanged.emit()

    def _update_preview(self):
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.set_axis_off()

        text = self._editor.toPlainText()

        # Build a structured preview of the full document
        preview_items = []

        # Extract document class
        cls_match = re.search(r"\\documentclass(?:\[.*?\])?\{(\w+)\}", text)
        if cls_match:
            preview_items.append(("heading", f"Document: {cls_match.group(1)}", 14))

        # Extract title
        title_match = re.search(r"\\title\{([^}]+)\}", text)
        if title_match:
            preview_items.append(("text", title_match.group(1), 16))

        # Extract author
        author_match = re.search(r"\\author\{([^}]+)\}", text)
        if author_match:
            preview_items.append(("text", author_match.group(1), 10))

        # Extract sections/chapters
        for m in re.finditer(r"\\(chapter|section|subsection|subsubsection)\{([^}]+)\}", text):
            level = m.group(1)
            sizes = {"chapter": 14, "section": 12, "subsection": 11, "subsubsection": 10}
            preview_items.append(("heading", f"[{level}] {m.group(2)}", sizes.get(level, 11)))

        # Extract math expressions from $ ... $, $$ ... $$, and equation environments
        math_patterns = re.findall(r"\$\$?(.*?)\$\$?", text, re.DOTALL)
        env_patterns = re.findall(
            r"\\begin\{(?:equation|align|gather)\*?\}(.*?)\\end\{(?:equation|align|gather)\*?\}",
            text, re.DOTALL)
        all_math = math_patterns + env_patterns

        for expr in all_math[:6]:
            expr = expr.strip().split("\\\\")[0].strip()
            expr = re.sub(r"\\label\{[^}]*\}", "", expr).strip()
            if expr:
                preview_items.append(("math", expr, 14))

        # Extract figure/table references
        for m in re.finditer(r"\\caption\{([^}]+)\}", text):
            preview_items.append(("text", f"[Caption: {m.group(1)[:40]}]", 9))

        if not preview_items:
            ax.text(0.5, 0.5, "No previewable content found.\n"
                    "Use $...$ for math, \\section{} for structure.",
                    ha="center", va="center", fontsize=12, color="gray",
                    transform=ax.transAxes)
            self._canvas.draw_idle()
            return

        y_pos = 0.97
        step = max(0.06, 0.92 / max(len(preview_items), 1))
        for kind, content, fsize in preview_items[:15]:
            if y_pos < 0.03:
                break
            try:
                if kind == "math":
                    ax.text(0.5, y_pos, f"${content}$", ha="center", va="top",
                            fontsize=fsize, transform=ax.transAxes)
                elif kind == "heading":
                    ax.text(0.05, y_pos, content, ha="left", va="top",
                            fontsize=fsize, fontweight="bold", transform=ax.transAxes)
                else:
                    ax.text(0.5, y_pos, content, ha="center", va="top",
                            fontsize=fsize, color="#555555", transform=ax.transAxes)
            except Exception:
                ax.text(0.5, y_pos, f"[render error]",
                        ha="center", va="top", fontsize=9, color="red",
                        transform=ax.transAxes)
            y_pos -= step
        self._canvas.draw_idle()

    # -- file actions ------------------------------------------------------

    def _action_new(self):
        if self._editor.toPlainText().strip():
            reply = QMessageBox.question(
                self, "New Document",
                "Discard current content?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        self._editor.clear()
        self._current_path = None
        self._status.setText("New document")
        self._emit_log("New document created")

    def _action_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open LaTeX File", "", "LaTeX Files (*.tex *.latex);;All Files (*)"
        )
        if path:
            self.load_file(path)

    def _action_save(self):
        if self._current_path:
            self.save_as(self._current_path)
        else:
            self._action_save_as()

    def _action_save_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save LaTeX File", "", "LaTeX Files (*.tex);;All Files (*)"
        )
        if path:
            self.save_as(path)

    def _export_png(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Equations as PNG", "", "PNG Images (*.png);;All Files (*)"
        )
        if path:
            self.export(path)

    # -- public API --------------------------------------------------------

    def load_file(self, path: str):
        """Load a .tex file into the editor."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._editor.setPlainText(f.read())
            self._current_path = path
            self._status.setText(f"Loaded: {os.path.basename(path)}")
            self._emit_log(f"Loaded file: {path}")
        except Exception as exc:
            self._status.setText(f"Error loading file: {exc}")
            self._emit_log(f"Error loading file: {exc}")

    def save_as(self, path: str):
        """Save editor content to *path* as .tex."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._editor.toPlainText())
            self._current_path = path
            self._status.setText(f"Saved: {os.path.basename(path)}")
            self._emit_log(f"Saved file: {path}")
        except Exception as exc:
            self._status.setText(f"Error saving: {exc}")
            self._emit_log(f"Error saving file: {exc}")

    def export(self, path: str = None):
        """Render math expressions to a PNG file via matplotlib."""
        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export PNG", "", "PNG Images (*.png)"
            )
            if not path:
                return
        try:
            fig = Figure(figsize=(8, 6), dpi=150)
            fig.patch.set_facecolor("white")
            ax = fig.add_subplot(111)
            ax.set_axis_off()

            text = self._editor.toPlainText()
            patterns = re.findall(r"\$\$?(.*?)\$\$?", text, re.DOTALL)

            if not patterns:
                ax.text(0.5, 0.5, "No math expressions to export.",
                        ha="center", va="center", fontsize=14)
            else:
                y = 0.95
                step = max(0.1, 0.85 / max(len(patterns), 1))
                for expr in patterns[:12]:
                    expr = expr.strip()
                    if expr:
                        ax.text(0.5, y, f"${expr}$",
                                ha="center", va="top", fontsize=18)
                        y -= step

            fig.savefig(path, bbox_inches="tight", facecolor="white")
            self._status.setText(f"Exported: {os.path.basename(path)}")
            self._emit_log(f"Exported equations to: {path}")
        except Exception as exc:
            self._status.setText(f"Export error: {exc}")
            self._emit_log(f"Export error: {exc}")
