"""
python_console.py - Interactive Python Console Widget for PyQt5 Scientific Suite

Provides an embedded Python REPL, multi-line code editor with syntax highlighting,
output capture, variable inspector, and pre-loaded scientific libraries.
"""

import sys
import io
import os
import json
import pickle
import datetime
import traceback
import keyword
import builtins
from code import InteractiveConsole
from types import ModuleType

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QPlainTextEdit,
    QPushButton, QLabel, QTreeWidget, QTreeWidgetItem, QTabWidget,
    QToolBar, QAction, QFileDialog, QApplication, QGroupBox,
    QHeaderView, QLineEdit, QShortcut, QSizePolicy, QStatusBar,
    QMenu, QMenuBar, QCompleter, QListWidget, QListWidgetItem,
    QInputDialog, QMessageBox,
)
from PyQt5.QtCore import Qt, QRegExp, QTimer, pyqtSignal, QStringListModel
from PyQt5.QtGui import (
    QFont, QColor, QTextCharFormat, QSyntaxHighlighter,
    QTextCursor, QPalette, QKeySequence,
)


# ---------------------------------------------------------------------------
# Syntax Highlighter
# ---------------------------------------------------------------------------

class PythonSyntaxHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for the Python language."""

    def __init__(self, document):
        super().__init__(document)
        self._rules = []

        # --- formats ---
        kw_fmt = QTextCharFormat()
        kw_fmt.setForeground(QColor("#CC7832"))
        kw_fmt.setFontWeight(QFont.Bold)

        builtin_fmt = QTextCharFormat()
        builtin_fmt.setForeground(QColor("#8888C6"))

        string_fmt = QTextCharFormat()
        string_fmt.setForeground(QColor("#6A8759"))

        number_fmt = QTextCharFormat()
        number_fmt.setForeground(QColor("#6897BB"))

        comment_fmt = QTextCharFormat()
        comment_fmt.setForeground(QColor("#808080"))
        comment_fmt.setFontItalic(True)

        decorator_fmt = QTextCharFormat()
        decorator_fmt.setForeground(QColor("#BBB529"))

        self_fmt = QTextCharFormat()
        self_fmt.setForeground(QColor("#94558D"))
        self_fmt.setFontItalic(True)

        func_fmt = QTextCharFormat()
        func_fmt.setForeground(QColor("#FFC66D"))

        # --- rules (order matters) ---
        # keywords
        kw_patterns = [r'\b{}\b'.format(w) for w in keyword.kwlist]
        for pat in kw_patterns:
            self._rules.append((QRegExp(pat), kw_fmt))

        # builtins
        builtin_names = [name for name in dir(builtins) if not name.startswith('_')]
        for name in builtin_names:
            self._rules.append((QRegExp(r'\b{}\b'.format(name)), builtin_fmt))

        # self
        self._rules.append((QRegExp(r'\bself\b'), self_fmt))

        # decorators
        self._rules.append((QRegExp(r'@\w+'), decorator_fmt))

        # function/method definitions
        self._rules.append((QRegExp(r'\bdef\b\s+(\w+)'), func_fmt))
        self._rules.append((QRegExp(r'\bclass\b\s+(\w+)'), func_fmt))

        # numbers
        self._rules.append((QRegExp(r'\b[0-9]+\.?[0-9]*([eE][+-]?[0-9]+)?\b'), number_fmt))
        self._rules.append((QRegExp(r'\b0[xX][0-9A-Fa-f]+\b'), number_fmt))

        # strings (single-line)
        self._rules.append((QRegExp(r'"[^"\\]*(\\.[^"\\]*)*"'), string_fmt))
        self._rules.append((QRegExp(r"'[^'\\]*(\\.[^'\\]*)*'"), string_fmt))

        # comments
        self._rules.append((QRegExp(r'#[^\n]*'), comment_fmt))

        # triple-quoted strings stored separately for multi-line handling
        self._tri_single = (QRegExp(r"'''"), QRegExp(r"'''"), string_fmt)
        self._tri_double = (QRegExp(r'"""'), QRegExp(r'"""'), string_fmt)

    # ---- QSyntaxHighlighter interface ----

    def highlightBlock(self, text):
        # single-line rules
        for pattern, fmt in self._rules:
            index = pattern.indexIn(text)
            while index >= 0:
                length = pattern.matchedLength()
                self.setFormat(index, length, fmt)
                index = pattern.indexIn(text, index + length)

        # multi-line strings
        self._match_multiline(text, *self._tri_single, state=1)
        self._match_multiline(text, *self._tri_double, state=2)

    def _match_multiline(self, text, start_re, end_re, fmt, state):
        if self.previousBlockState() == state:
            start = 0
            add = 0
        else:
            start = start_re.indexIn(text)
            if start == -1:
                return
            add = start_re.matchedLength()

        while start >= 0:
            end = end_re.indexIn(text, start + add)
            if end == -1:
                self.setCurrentBlockState(state)
                length = len(text) - start
            else:
                length = end - start + end_re.matchedLength()
            self.setFormat(start, length, fmt)
            start = start_re.indexIn(text, start + length)
            add = start_re.matchedLength() if start >= 0 else 0


# ---------------------------------------------------------------------------
# Console Input Widget (REPL line)
# ---------------------------------------------------------------------------

class ConsoleInput(QLineEdit):
    """Single-line input with command history navigation and auto-complete."""

    command_entered = pyqtSignal(str)

    # Common completions for popular scientific prefixes
    _COMPLETIONS = {
        "np.": [
            "np.array(", "np.zeros(", "np.ones(", "np.linspace(",
            "np.arange(", "np.random.rand(", "np.random.randn(",
            "np.reshape(", "np.dot(", "np.matmul(", "np.linalg.inv(",
            "np.linalg.eig(", "np.linalg.svd(", "np.linalg.norm(",
            "np.linalg.solve(", "np.linalg.det(", "np.mean(",
            "np.std(", "np.sum(", "np.max(", "np.min(",
            "np.concatenate(", "np.stack(", "np.meshgrid(",
            "np.fft.fft(", "np.fft.ifft(", "np.polyfit(",
            "np.histogram(", "np.corrcoef(", "np.cov(",
            "np.loadtxt(", "np.savetxt(", "np.save(", "np.load(",
        ],
        "plt.": [
            "plt.plot(", "plt.scatter(", "plt.bar(", "plt.hist(",
            "plt.imshow(", "plt.contour(", "plt.contourf(",
            "plt.subplot(", "plt.subplots(", "plt.figure(",
            "plt.xlabel(", "plt.ylabel(", "plt.title(",
            "plt.legend(", "plt.colorbar(", "plt.show(",
            "plt.savefig(", "plt.tight_layout(", "plt.grid(",
            "plt.xlim(", "plt.ylim(", "plt.loglog(",
            "plt.semilogx(", "plt.semilogy(", "plt.errorbar(",
            "plt.fill_between(", "plt.axhline(", "plt.axvline(",
        ],
        "pd.": [
            "pd.DataFrame(", "pd.Series(", "pd.read_csv(",
            "pd.read_excel(", "pd.concat(", "pd.merge(",
            "pd.to_datetime(", "pd.cut(", "pd.qcut(",
            "pd.pivot_table(", "pd.crosstab(", "pd.get_dummies(",
        ],
        "sp.": [
            "sp.symbols(", "sp.solve(", "sp.simplify(",
            "sp.expand(", "sp.factor(", "sp.diff(",
            "sp.integrate(", "sp.limit(", "sp.series(",
            "sp.Matrix(", "sp.Rational(", "sp.latex(",
        ],
        "scipy.": [
            "scipy.optimize.minimize(", "scipy.optimize.curve_fit(",
            "scipy.integrate.solve_ivp(", "scipy.integrate.quad(",
            "scipy.interpolate.interp1d(", "scipy.signal.find_peaks(",
            "scipy.stats.norm(", "scipy.stats.ttest_ind(",
            "scipy.linalg.lu(", "scipy.linalg.qr(",
        ],
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._history: list[str] = []
        self._history_index: int = 0
        self._current_input: str = ""
        self.setPlaceholderText(">>> Type Python expressions here ...")
        self.returnPressed.connect(self._on_return)

        # Auto-completer
        self._completer = QCompleter([], self)
        self._completer.setWidget(self)
        self._completer.setCompletionMode(QCompleter.PopupCompletion)
        self._completer.setCaseSensitivity(Qt.CaseSensitive)
        self._completer.activated.connect(self._insert_completion)
        self.textChanged.connect(self._update_completions)

    # -- public API --

    def history(self) -> list[str]:
        return list(self._history)

    # -- slots / events --

    def _on_return(self):
        text = self.text().strip()
        if text:
            self._history.append(text)
        self._history_index = len(self._history)
        self._current_input = ""
        self.command_entered.emit(text)
        self.clear()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Up:
            self._navigate_history(-1)
        elif key == Qt.Key_Down:
            self._navigate_history(1)
        else:
            super().keyPressEvent(event)

    # -- internal --

    def _navigate_history(self, direction: int):
        if not self._history:
            return
        if self._history_index == len(self._history):
            self._current_input = self.text()

        new_index = self._history_index + direction
        if new_index < 0:
            new_index = 0
        elif new_index > len(self._history):
            new_index = len(self._history)

        self._history_index = new_index
        if new_index == len(self._history):
            self.setText(self._current_input)
        else:
            self.setText(self._history[new_index])

    def _update_completions(self, text):
        """Show auto-complete popup when a known prefix is typed."""
        for prefix, completions in self._COMPLETIONS.items():
            if text.endswith(prefix) or (prefix[:-1] + "." in text):
                # Find the relevant part
                idx = text.rfind(prefix[:2])
                if idx >= 0:
                    partial = text[idx:]
                    matches = [c for c in completions if c.startswith(partial)]
                    if matches:
                        model = QStringListModel(matches)
                        self._completer.setModel(model)
                        self._completer.setCompletionPrefix(partial)
                        cr = self.cursorRect()
                        cr.setWidth(300)
                        self._completer.complete(cr)
                        return
        self._completer.popup().hide()

    def _insert_completion(self, completion):
        """Insert the selected completion, replacing the typed prefix."""
        text = self.text()
        # Find how much of the completion is already typed
        for prefix in self._COMPLETIONS:
            idx = text.rfind(prefix[:2])
            if idx >= 0:
                self.setText(text[:idx] + completion)
                return
        self.setText(text + completion)


# ---------------------------------------------------------------------------
# Output Display
# ---------------------------------------------------------------------------

class OutputDisplay(QPlainTextEdit):
    """Read-only pane that shows stdout / stderr output."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        font = QFont("Consolas", 10)
        font.setStyleHint(QFont.Monospace)
        self.setFont(font)
        pal = self.palette()
        pal.setColor(QPalette.Base, QColor("#1E1E1E"))
        pal.setColor(QPalette.Text, QColor("#DCDCDC"))
        self.setPalette(pal)
        self.setMaximumBlockCount(10000)

    def append_stdout(self, text: str):
        self._append_colored(text, "#DCDCDC")

    def append_stderr(self, text: str):
        self._append_colored(text, "#FF6B68")

    def append_info(self, text: str):
        self._append_colored(text, "#569CD6")

    def _append_colored(self, text: str, color: str):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cursor.setCharFormat(fmt)
        cursor.insertText(text)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()


# ---------------------------------------------------------------------------
# Variable Inspector
# ---------------------------------------------------------------------------

class VariableInspector(QTreeWidget):
    """Tree widget that displays variables in the execution namespace."""

    _IGNORE_TYPES = (ModuleType,)
    _IGNORE_NAMES = {
        '__builtins__', '__name__', '__doc__', '__package__',
        '__loader__', '__spec__', '__annotations__',
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(["Name", "Type", "Shape / Size", "Value Preview"])
        self.setAlternatingRowColors(True)
        self.setRootIsDecorated(False)
        header = self.header()
        header.setSectionResizeMode(0, QHeaderView.Interactive)
        header.setSectionResizeMode(1, QHeaderView.Interactive)
        header.setSectionResizeMode(2, QHeaderView.Interactive)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        self.setColumnWidth(0, 140)
        self.setColumnWidth(1, 110)
        self.setColumnWidth(2, 100)

    def refresh(self, namespace: dict):
        self.clear()
        for name, obj in sorted(namespace.items()):
            if name.startswith('_') or name in self._IGNORE_NAMES:
                continue
            if isinstance(obj, self._IGNORE_TYPES):
                continue
            if callable(obj) and not isinstance(obj, type):
                continue
            type_str = type(obj).__name__
            shape_str = self._get_shape(obj)
            preview = self._get_preview(obj)
            item = QTreeWidgetItem([name, type_str, shape_str, preview])
            self.addTopLevelItem(item)

    @staticmethod
    def _get_shape(obj) -> str:
        # numpy / pandas shapes
        if hasattr(obj, 'shape'):
            return str(obj.shape)
        if hasattr(obj, '__len__'):
            try:
                return "len={}".format(len(obj))
            except Exception:
                pass
        return ""

    @staticmethod
    def _get_preview(obj) -> str:
        try:
            text = repr(obj)
            if len(text) > 120:
                text = text[:117] + "..."
            return text
        except Exception:
            return "<unable to repr>"


# ---------------------------------------------------------------------------
# Code Editor
# ---------------------------------------------------------------------------

class _LineNumberArea(QWidget):
    """Line number gutter for CodeEditor."""

    def __init__(self, editor):
        super().__init__(editor)
        self._editor = editor

    def sizeHint(self):
        return QSize(self._editor._line_number_area_width(), 0)

    def paintEvent(self, event):
        self._editor._paint_line_numbers(event)


class CodeEditor(QPlainTextEdit):
    """Multi-line code editor with syntax highlighting and basic amenities."""

    def __init__(self, parent=None):
        super().__init__(parent)
        font = QFont("Consolas", 10)
        font.setStyleHint(QFont.Monospace)
        self.setFont(font)
        self.setTabStopWidth(4 * self.fontMetrics().width(' '))
        self.setLineWrapMode(QPlainTextEdit.NoWrap)
        pal = self.palette()
        pal.setColor(QPalette.Base, QColor("#2B2B2B"))
        pal.setColor(QPalette.Text, QColor("#A9B7C6"))
        self.setPalette(pal)
        self._highlighter = PythonSyntaxHighlighter(self.document())

        # Line numbers
        self._line_number_area = _LineNumberArea(self)
        self.blockCountChanged.connect(self._update_line_number_width)
        self.updateRequest.connect(self._update_line_number_area)
        self._update_line_number_width(0)

        # Error line tracking
        self._error_line = -1

    def _line_number_area_width(self):
        digits = max(1, len(str(self.blockCount())))
        return 10 + self.fontMetrics().width('9') * digits

    def _update_line_number_width(self, _):
        self.setViewportMargins(self._line_number_area_width(), 0, 0, 0)

    def _update_line_number_area(self, rect, dy):
        if dy:
            self._line_number_area.scroll(0, dy)
        else:
            self._line_number_area.update(0, rect.y(), self._line_number_area.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self._update_line_number_width(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cr = self.contentsRect()
        self._line_number_area.setGeometry(cr.left(), cr.top(),
                                           self._line_number_area_width(), cr.height())

    def _paint_line_numbers(self, event):
        from PyQt5.QtGui import QPainter
        painter = QPainter(self._line_number_area)
        painter.fillRect(event.rect(), QColor("#313335"))
        block = self.firstVisibleBlock()
        block_num = block.blockNumber()
        top = int(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(block_num + 1)
                if block_num == self._error_line:
                    painter.fillRect(0, top, self._line_number_area.width(),
                                     int(self.blockBoundingRect(block).height()),
                                     QColor("#5C1010"))
                    painter.setPen(QColor("#FF6B68"))
                else:
                    painter.setPen(QColor("#606366"))
                painter.drawText(0, top, self._line_number_area.width() - 4,
                                 int(self.blockBoundingRect(block).height()),
                                 Qt.AlignRight | Qt.AlignVCenter, number)
            block = block.next()
            top = bottom
            bottom = top + int(self.blockBoundingRect(block).height())
            block_num += 1
        painter.end()

    def highlight_error_line(self, line_number: int):
        """Highlight a specific line (0-based) as an error line."""
        self._error_line = line_number
        # Also highlight the line in the editor via extra selections
        selections = []
        if line_number >= 0:
            block = self.document().findBlockByNumber(line_number)
            if block.isValid():
                sel = QPlainTextEdit.ExtraSelection()  # type: ignore[attr-defined]
                sel.format.setBackground(QColor("#5C1010"))
                sel.format.setProperty(QTextCharFormat.FullWidthSelection, True)
                sel.cursor = QTextCursor(block)
                sel.cursor.clearSelection()
                selections.append(sel)
        self.setExtraSelections(selections)
        self._line_number_area.update()

    def clear_error_highlight(self):
        """Clear any error line highlighting."""
        self._error_line = -1
        self.setExtraSelections([])
        self._line_number_area.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Tab:
            cursor = self.textCursor()
            if cursor.hasSelection():
                self._indent_selection(cursor, indent=True)
                return
            else:
                cursor.insertText("    ")
                return
        if event.key() == Qt.Key_Backtab:
            cursor = self.textCursor()
            if cursor.hasSelection():
                self._indent_selection(cursor, indent=False)
                return
        # auto-indent on Enter
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            cursor = self.textCursor()
            line = cursor.block().text()
            indent = len(line) - len(line.lstrip())
            if line.rstrip().endswith(':'):
                indent += 4
            super().keyPressEvent(event)
            self.textCursor().insertText(' ' * indent)
            return
        super().keyPressEvent(event)

    def _indent_selection(self, cursor, indent=True):
        cursor.beginEditBlock()
        start = cursor.selectionStart()
        end = cursor.selectionEnd()
        cursor.setPosition(start)
        cursor.movePosition(QTextCursor.StartOfBlock)
        while cursor.position() <= end and not cursor.atEnd():
            cursor.movePosition(QTextCursor.StartOfBlock)
            if indent:
                cursor.insertText("    ")
                end += 4
            else:
                line = cursor.block().text()
                remove = min(4, len(line) - len(line.lstrip()))
                if remove > 0:
                    for _ in range(remove):
                        cursor.deleteChar()
                    end -= remove
            if not cursor.movePosition(QTextCursor.NextBlock):
                break
        cursor.endEditBlock()


# ---------------------------------------------------------------------------
# Stream Redirector
# ---------------------------------------------------------------------------

class _StreamRedirector(io.TextIOBase):
    """Redirect writes to a callable."""

    def __init__(self, callback):
        super().__init__()
        self._callback = callback

    def write(self, text):
        if text:
            self._callback(text)
        return len(text) if text else 0

    def flush(self):
        pass


# ---------------------------------------------------------------------------
# Script Templates
# ---------------------------------------------------------------------------

SCRIPT_TEMPLATES = {
    "Data Generation": '''\
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
n = 500
x = np.linspace(0, 10, n)
noise = np.random.normal(0, 0.3, n)
y = np.sin(2 * np.pi * x / 5) + 0.5 * x + noise

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x, y, s=4, alpha=0.6, label='data')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('Synthetic Dataset')
ax.legend()
plt.tight_layout()
plt.show()
''',
    "Monte Carlo Simulation": '''\
import numpy as np
import matplotlib.pyplot as plt

# Monte Carlo estimation of pi
n_samples = 100000
x = np.random.uniform(-1, 1, n_samples)
y = np.random.uniform(-1, 1, n_samples)
inside = x**2 + y**2 <= 1.0
pi_estimate = 4.0 * np.sum(inside) / n_samples
print(f"Pi estimate ({n_samples} samples): {pi_estimate:.6f}")
print(f"Error: {abs(pi_estimate - np.pi):.6f}")

# Visualization
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(x[inside], y[inside], s=0.5, c='blue', alpha=0.3)
ax.scatter(x[~inside], y[~inside], s=0.5, c='red', alpha=0.3)
circle = plt.Circle((0, 0), 1, fill=False, color='black', lw=2)
ax.add_patch(circle)
ax.set_aspect('equal')
ax.set_title(f'Monte Carlo Pi = {pi_estimate:.4f}')
plt.show()
''',
    "Numerical ODE Solver": '''\
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the ODE: dy/dt = f(t, y)
# Example: damped harmonic oscillator y'' + 2*zeta*omega*y' + omega^2*y = 0
omega = 2 * np.pi  # natural frequency
zeta = 0.1         # damping ratio

def osc(t, state):
    y, v = state
    dydt = v
    dvdt = -2 * zeta * omega * v - omega**2 * y
    return [dydt, dvdt]

# Solve
t_span = (0, 5)
y0 = [1.0, 0.0]  # initial displacement=1, velocity=0
sol = solve_ivp(osc, t_span, y0, t_eval=np.linspace(*t_span, 500), method='RK45')

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax1.plot(sol.t, sol.y[0], label='displacement y(t)')
ax1.set_ylabel('y(t)'); ax1.legend(); ax1.grid(True)
ax2.plot(sol.t, sol.y[1], 'r', label='velocity y\\'(t)')
ax2.set_xlabel('t'); ax2.set_ylabel("y'(t)"); ax2.legend(); ax2.grid(True)
fig.suptitle(f'Damped Oscillator (zeta={zeta})')
plt.tight_layout()
plt.show()
''',
    "Curve Fitting Template": '''\
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Model function
def model(x, a, b, c):
    return a * np.exp(-b * x) + c

# Generate noisy data
x_data = np.linspace(0, 5, 80)
y_true = model(x_data, 2.5, 1.3, 0.5)
y_data = y_true + 0.2 * np.random.randn(len(x_data))

# Fit
popt, pcov = curve_fit(model, x_data, y_data, p0=[1, 1, 0])
perr = np.sqrt(np.diag(pcov))
print("Fitted parameters:")
for name, val, err in zip(['a', 'b', 'c'], popt, perr):
    print(f"  {name} = {val:.4f} +/- {err:.4f}")

# Plot
x_fit = np.linspace(0, 5, 300)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x_data, y_data, s=15, label='data', zorder=5)
ax.plot(x_fit, model(x_fit, *popt), 'r-', lw=2, label='fit')
ax.plot(x_fit, y_true, 'g--', alpha=0.5, label='true')
ax.legend(); ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('Curve Fitting: a*exp(-b*x) + c')
plt.tight_layout(); plt.show()
''',
    "Image Analysis Template": '''\
import numpy as np
import matplotlib.pyplot as plt

# Create a synthetic image (Gaussian blob + noise)
size = 256
x = np.linspace(-3, 3, size)
X, Y = np.meshgrid(x, x)
image = np.exp(-(X**2 + Y**2) / 2) + 0.1 * np.random.randn(size, size)

# Analysis
print(f"Image shape: {image.shape}")
print(f"Min: {image.min():.4f}, Max: {image.max():.4f}")
print(f"Mean: {image.mean():.4f}, Std: {image.std():.4f}")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].imshow(image, cmap='viridis'); axes[0].set_title('Image')
axes[1].hist(image.ravel(), bins=100); axes[1].set_title('Histogram')
axes[2].imshow(image > 0.5, cmap='gray'); axes[2].set_title('Threshold > 0.5')
for ax in axes: ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.show()
''',
    "Statistical Test Template": '''\
import numpy as np
from scipy import stats

# Generate two sample groups
np.random.seed(42)
group_a = np.random.normal(loc=100, scale=15, size=50)
group_b = np.random.normal(loc=108, scale=15, size=50)

print("=== Descriptive Statistics ===")
for name, data in [("Group A", group_a), ("Group B", group_b)]:
    print(f"{name}: n={len(data)}, mean={data.mean():.2f}, std={data.std():.2f}, "
          f"median={np.median(data):.2f}")

print("\\n=== Normality Tests (Shapiro-Wilk) ===")
for name, data in [("Group A", group_a), ("Group B", group_b)]:
    stat, p = stats.shapiro(data)
    print(f"{name}: W={stat:.4f}, p={p:.4f} {'(normal)' if p > 0.05 else '(non-normal)'}")

print("\\n=== Independent t-test ===")
t_stat, p_val = stats.ttest_ind(group_a, group_b)
print(f"t = {t_stat:.4f}, p = {p_val:.4f}")
print(f"Significant at alpha=0.05: {'Yes' if p_val < 0.05 else 'No'}")

print("\\n=== Mann-Whitney U test ===")
u_stat, p_val_mw = stats.mannwhitneyu(group_a, group_b)
print(f"U = {u_stat:.4f}, p = {p_val_mw:.4f}")

# Effect size (Cohen\'s d)
pooled_std = np.sqrt((group_a.std()**2 + group_b.std()**2) / 2)
cohens_d = (group_b.mean() - group_a.mean()) / pooled_std
print(f"\\nCohen\'s d = {cohens_d:.4f}")
''',
}

# ---------------------------------------------------------------------------
# Snippet Library Data
# ---------------------------------------------------------------------------

SNIPPET_LIBRARY = {
    "NumPy Operations": {
        "Create array": "arr = np.array([1, 2, 3, 4, 5])",
        "Zeros/Ones": "z = np.zeros((3, 3))\no = np.ones((3, 3))",
        "Linspace": "x = np.linspace(0, 10, 100)",
        "Random normal": "data = np.random.randn(1000)",
        "Reshape": "mat = np.arange(12).reshape(3, 4)",
        "Matrix multiply": "C = A @ B  # or np.matmul(A, B)",
        "Eigenvalues": "vals, vecs = np.linalg.eig(A)",
        "FFT": "freq = np.fft.fft(signal)\npower = np.abs(freq)**2",
        "Polyfit": "coeffs = np.polyfit(x, y, deg=3)",
        "Broadcast add": "result = arr[:, np.newaxis] + arr[np.newaxis, :]",
    },
    "SciPy Solvers": {
        "Minimize": "from scipy.optimize import minimize\nres = minimize(func, x0, method='Nelder-Mead')",
        "Curve fit": "from scipy.optimize import curve_fit\npopt, pcov = curve_fit(model, xdata, ydata)",
        "Solve IVP": "from scipy.integrate import solve_ivp\nsol = solve_ivp(func, t_span, y0, method='RK45')",
        "Quad integral": "from scipy.integrate import quad\nresult, error = quad(func, a, b)",
        "Interpolate": "from scipy.interpolate import interp1d\nf = interp1d(x, y, kind='cubic')",
        "Find peaks": "from scipy.signal import find_peaks\npeaks, props = find_peaks(signal, height=0.5)",
        "KDE": "from scipy.stats import gaussian_kde\nkde = gaussian_kde(data)",
        "Sparse solve": "from scipy.sparse.linalg import spsolve\nx = spsolve(A_sparse, b)",
    },
    "Matplotlib Plots": {
        "Line plot": "plt.figure(figsize=(8, 5))\nplt.plot(x, y, 'b-', label='data')\nplt.xlabel('x'); plt.ylabel('y')\nplt.legend(); plt.show()",
        "Scatter": "plt.scatter(x, y, c=colors, s=sizes, alpha=0.6)\nplt.colorbar(); plt.show()",
        "Histogram": "plt.hist(data, bins=50, density=True, alpha=0.7)\nplt.xlabel('Value'); plt.ylabel('Density'); plt.show()",
        "Subplots": "fig, axes = plt.subplots(2, 2, figsize=(10, 8))\naxes[0, 0].plot(x, y); plt.tight_layout(); plt.show()",
        "Contour": "X, Y = np.meshgrid(x, y)\nplt.contourf(X, Y, Z, levels=20, cmap='viridis')\nplt.colorbar(); plt.show()",
        "Error bars": "plt.errorbar(x, y, yerr=err, fmt='o-', capsize=3)\nplt.show()",
        "3D surface": "from mpl_toolkits.mplot3d import Axes3D\nfig = plt.figure()\nax = fig.add_subplot(111, projection='3d')\nax.plot_surface(X, Y, Z, cmap='viridis')\nplt.show()",
        "Heatmap": "plt.imshow(data, cmap='hot', aspect='auto')\nplt.colorbar(); plt.show()",
    },
    "SymPy Symbolic": {
        "Define symbol": "x, y, z = sp.symbols('x y z')",
        "Differentiate": "expr = sp.sin(x) * sp.exp(x)\nresult = sp.diff(expr, x)",
        "Integrate": "result = sp.integrate(sp.exp(-x**2), (x, -sp.oo, sp.oo))",
        "Solve equation": "sol = sp.solve(x**2 - 5*x + 6, x)",
        "Limit": "lim = sp.limit(sp.sin(x)/x, x, 0)",
        "Taylor series": "ser = sp.series(sp.cos(x), x, 0, n=6)",
        "Simplify": "simple = sp.simplify(sp.sin(x)**2 + sp.cos(x)**2)",
        "LaTeX output": "print(sp.latex(expr))",
        "Matrix": "M = sp.Matrix([[1, 2], [3, 4]])\nprint(M.eigenvals())",
    },
    "Pandas Operations": {
        "Create DataFrame": "df = pd.DataFrame({'x': x, 'y': y, 'group': groups})",
        "Read CSV": "df = pd.read_csv('data.csv')",
        "Describe": "print(df.describe())",
        "Group by": "grouped = df.groupby('category').agg({'value': ['mean', 'std']})",
        "Filter rows": "filtered = df[df['column'] > threshold]",
        "Pivot table": "pivot = pd.pivot_table(df, values='val', index='row', columns='col', aggfunc='mean')",
        "Merge": "merged = pd.merge(df1, df2, on='key', how='inner')",
        "Apply function": "df['new_col'] = df['col'].apply(lambda x: x**2)",
        "To CSV": "df.to_csv('output.csv', index=False)",
    },
}


# ---------------------------------------------------------------------------
# Snippet Library Panel Widget
# ---------------------------------------------------------------------------

class SnippetLibraryPanel(QWidget):
    """Collapsible panel displaying categorized code snippets."""

    snippet_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        title = QLabel("Code Snippets")
        title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        layout.addWidget(title)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Snippet"])
        self._tree.setRootIsDecorated(True)
        self._tree.setAlternatingRowColors(True)
        self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self._tree)

        insert_btn = QPushButton("Insert Selected")
        insert_btn.clicked.connect(self._insert_selected)
        layout.addWidget(insert_btn)

        self._populate()

    def _populate(self):
        for category, snippets in SNIPPET_LIBRARY.items():
            cat_item = QTreeWidgetItem([category])
            cat_item.setFont(0, QFont("Segoe UI", 9, QFont.Bold))
            for name in snippets:
                child = QTreeWidgetItem([name])
                child.setToolTip(0, snippets[name])
                child.setData(0, Qt.UserRole, snippets[name])
                cat_item.addChild(child)
            self._tree.addTopLevelItem(cat_item)

    def _on_item_double_clicked(self, item, column):
        code = item.data(0, Qt.UserRole)
        if code:
            self.snippet_selected.emit(code)

    def _insert_selected(self):
        item = self._tree.currentItem()
        if item:
            code = item.data(0, Qt.UserRole)
            if code:
                self.snippet_selected.emit(code)


# ---------------------------------------------------------------------------
# Main Widget
# ---------------------------------------------------------------------------

class PythonConsoleWidget(QWidget):
    """
    Full-featured embedded Python console for a PyQt5 scientific application.

    Features
    --------
    * Interactive single-line REPL with history (up/down arrows)
    * Multi-line code editor with Python syntax highlighting
    * Run button to execute editor contents
    * Captured stdout / stderr display
    * Pre-imported scientific stack (numpy, scipy, matplotlib, sympy, pandas)
    * Variable inspector showing name, type, shape, value preview
    * ``run()`` -- execute the current editor content programmatically
    * ``set_logger(fn)`` -- attach an external logging callback
    * ``load_file(path)`` -- load a ``.py`` file into the editor
    """

    execution_finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._logger = None
        self._namespace: dict = {}
        self._init_namespace()
        self._build_ui()
        self._connect_signals()
        self._output.append_info("Python {} on {}\n".format(
            sys.version.split()[0], sys.platform))
        self._output.append_info("Scientific libraries loaded: "
                                 "numpy, scipy, sympy, matplotlib, pandas\n")
        self._output.append_info("Type expressions in the REPL or write scripts "
                                 "in the editor.\n\n")

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def run(self):
        """Execute the current editor content."""
        code = self._editor.toPlainText()
        if code.strip():
            self._execute(code, source="<editor>")

    def set_logger(self, fn):
        """
        Attach a logging callback.

        Parameters
        ----------
        fn : callable(str)
            Will be called with log messages from the console.
        """
        self._logger = fn

    def load_file(self, path: str):
        """
        Load a ``.py`` file into the code editor.

        Parameters
        ----------
        path : str
            Filesystem path to a Python source file.
        """
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            self._output.append_stderr("File not found: {}\n".format(path))
            return
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                source = fh.read()
            self._editor.setPlainText(source)
            self._output.append_info("Loaded: {}\n".format(path))
            self._log("Loaded file: {}".format(path))
        except Exception as exc:
            self._output.append_stderr("Error loading file: {}\n".format(exc))

    def get_namespace(self) -> dict:
        """Return the current execution namespace (for embedding)."""
        return self._namespace

    def inject(self, name: str, obj):
        """Inject a variable into the execution namespace."""
        self._namespace[name] = obj
        self._inspector.refresh(self._namespace)

    def clear_output(self):
        """Clear the output pane."""
        self._output.clear()

    # ------------------------------------------------------------------ #
    #  UI Construction                                                    #
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        # -- menu bar with Script Templates --
        self._menubar = QMenuBar()
        self._templates_menu = self._menubar.addMenu("Script Templates")
        for name, code in SCRIPT_TEMPLATES.items():
            action = self._templates_menu.addAction(name)
            action.triggered.connect(lambda checked, c=code: self._load_template(c))

        session_menu = self._menubar.addMenu("Session")
        save_session_action = session_menu.addAction("Save Session...")
        save_session_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_session_action.triggered.connect(self._save_session)
        restore_session_action = session_menu.addAction("Restore Session...")
        restore_session_action.setShortcut(QKeySequence("Ctrl+Shift+R"))
        restore_session_action.triggered.connect(self._restore_session)

        root_layout.addWidget(self._menubar)

        # -- toolbar --
        toolbar = QToolBar()
        toolbar.setMovable(False)

        self._run_action = QAction("Run", self)
        self._run_action.setShortcut(QKeySequence("Ctrl+Return"))
        self._run_action.setToolTip("Execute editor contents  (Ctrl+Enter)")
        toolbar.addAction(self._run_action)

        self._clear_action = QAction("Clear Output", self)
        self._clear_action.setToolTip("Clear the output pane")
        toolbar.addAction(self._clear_action)

        self._open_action = QAction("Open File", self)
        self._open_action.setShortcut(QKeySequence("Ctrl+O"))
        self._open_action.setToolTip("Load a .py file into the editor")
        toolbar.addAction(self._open_action)

        self._reset_action = QAction("Reset Namespace", self)
        self._reset_action.setToolTip("Clear all user variables")
        toolbar.addAction(self._reset_action)

        # -- Generate toolbar section --
        toolbar.addSeparator()
        gen_label = QLabel("  Generate: ")
        gen_label.setFont(QFont("Segoe UI", 9, QFont.Bold))
        toolbar.addWidget(gen_label)

        self._gen_data_action = QAction("Generate Data", self)
        self._gen_data_action.setToolTip("Create synthetic datasets")
        toolbar.addAction(self._gen_data_action)

        self._gen_plot_action = QAction("Generate Plot", self)
        self._gen_plot_action.setToolTip("Quick plot from namespace variables")
        toolbar.addAction(self._gen_plot_action)

        self._gen_report_action = QAction("Generate Report", self)
        self._gen_report_action.setToolTip("Export session output as HTML")
        toolbar.addAction(self._gen_report_action)

        root_layout.addWidget(toolbar)

        # -- main splitter (horizontal): left = editor+output, right = inspector+snippets --
        main_splitter = QSplitter(Qt.Horizontal)

        # left side: vertical splitter for editor (top) and output+REPL (bottom)
        left_splitter = QSplitter(Qt.Vertical)

        # editor group
        editor_group = QGroupBox("Code Editor")
        editor_layout = QVBoxLayout(editor_group)
        editor_layout.setContentsMargins(2, 2, 2, 2)
        self._editor = CodeEditor()
        self._editor.setPlaceholderText("# Write your Python script here ...")
        editor_layout.addWidget(self._editor)
        left_splitter.addWidget(editor_group)

        # output + repl group
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout(output_group)
        output_layout.setContentsMargins(2, 2, 2, 2)
        self._output = OutputDisplay()
        output_layout.addWidget(self._output)

        repl_layout = QHBoxLayout()
        prompt_label = QLabel(">>>")
        prompt_label.setFont(QFont("Consolas", 10, QFont.Bold))
        repl_layout.addWidget(prompt_label)
        self._repl_input = ConsoleInput()
        self._repl_input.setFont(QFont("Consolas", 10))
        repl_layout.addWidget(self._repl_input)
        output_layout.addLayout(repl_layout)
        left_splitter.addWidget(output_group)

        left_splitter.setStretchFactor(0, 3)
        left_splitter.setStretchFactor(1, 2)

        main_splitter.addWidget(left_splitter)

        # right side: vertical splitter with inspector (top) and snippet library (bottom)
        right_splitter = QSplitter(Qt.Vertical)

        inspector_group = QGroupBox("Variable Inspector")
        inspector_layout = QVBoxLayout(inspector_group)
        inspector_layout.setContentsMargins(2, 2, 2, 2)
        self._inspector = VariableInspector()
        inspector_layout.addWidget(self._inspector)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setToolTip("Refresh the variable list")
        inspector_layout.addWidget(self._refresh_btn)

        right_splitter.addWidget(inspector_group)

        # Snippet library panel (collapsible via splitter)
        snippet_group = QGroupBox("Code Snippet Library")
        snippet_layout = QVBoxLayout(snippet_group)
        snippet_layout.setContentsMargins(2, 2, 2, 2)
        self._snippet_panel = SnippetLibraryPanel()
        snippet_layout.addWidget(self._snippet_panel)
        right_splitter.addWidget(snippet_group)

        right_splitter.setStretchFactor(0, 2)
        right_splitter.setStretchFactor(1, 1)

        main_splitter.addWidget(right_splitter)

        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 1)

        root_layout.addWidget(main_splitter)

        # -- status bar --
        self._status = QStatusBar()
        self._status.showMessage("Ready")
        root_layout.addWidget(self._status)

    # ------------------------------------------------------------------ #
    #  Signals                                                            #
    # ------------------------------------------------------------------ #

    def _connect_signals(self):
        self._run_action.triggered.connect(self.run)
        self._clear_action.triggered.connect(self.clear_output)
        self._open_action.triggered.connect(self._open_file_dialog)
        self._reset_action.triggered.connect(self._reset_namespace)
        self._repl_input.command_entered.connect(self._on_repl_command)
        self._refresh_btn.clicked.connect(
            lambda: self._inspector.refresh(self._namespace))
        # Generate toolbar
        self._gen_data_action.triggered.connect(self._generate_data)
        self._gen_plot_action.triggered.connect(self._generate_plot)
        self._gen_report_action.triggered.connect(self._generate_report)
        # Snippet panel
        self._snippet_panel.snippet_selected.connect(self._insert_snippet)

    # ------------------------------------------------------------------ #
    #  Execution Engine                                                   #
    # ------------------------------------------------------------------ #

    def _execute(self, code: str, source: str = "<input>"):
        """Compile and execute *code* inside the shared namespace."""
        self._status.showMessage("Running ...")
        QApplication.processEvents()

        # Clear previous error highlights
        self._editor.clear_error_highlight()

        # Output timestamp
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self._output.append_info(f"[{timestamp}] Executing...\n")

        stdout_capture = _StreamRedirector(self._output.append_stdout)
        stderr_capture = _StreamRedirector(self._output.append_stderr)

        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout_capture, stderr_capture

        try:
            # Try to compile as an expression first (so we can print the result)
            try:
                compiled = compile(code, source, "eval")
            except SyntaxError:
                compiled = compile(code, source, "exec")
                exec(compiled, self._namespace)
            else:
                result = eval(compiled, self._namespace)
                if result is not None:
                    self._namespace['_'] = result
                    sys.stdout.write(repr(result) + "\n")
        except SystemExit:
            self._output.append_stderr("SystemExit caught (ignored).\n")
        except SyntaxError as se:
            tb_text = traceback.format_exc()
            self._output.append_stderr(tb_text)
            # Highlight the error line in the editor
            if se.lineno is not None:
                self._editor.highlight_error_line(se.lineno - 1)
            self._log("Syntax error:\n" + tb_text)
        except Exception:
            tb_text = traceback.format_exc()
            self._output.append_stderr(tb_text)
            # Try to extract error line number from traceback
            import re as _re
            match = _re.search(r'File "[^"]*", line (\d+)', tb_text)
            if match:
                err_line = int(match.group(1)) - 1
                self._editor.highlight_error_line(err_line)
            self._log("Execution error:\n" + tb_text)
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

        end_timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self._output.append_info(f"[{end_timestamp}] Done.\n")

        self._inspector.refresh(self._namespace)
        self._status.showMessage("Ready")
        self.execution_finished.emit()
        self._log("Executed code block from {}".format(source))

    # ------------------------------------------------------------------ #
    #  REPL                                                               #
    # ------------------------------------------------------------------ #

    def _on_repl_command(self, text: str):
        if not text:
            return
        self._output.append_info(">>> {}\n".format(text))
        self._execute(text, source="<repl>")

    # ------------------------------------------------------------------ #
    #  Namespace Management                                               #
    # ------------------------------------------------------------------ #

    def _init_namespace(self):
        """Set up the execution namespace with scientific libraries."""
        self._namespace = {'__name__': '__console__', '__doc__': None}

        _imports = {
            'numpy': 'np',
            'scipy': None,
            'scipy.linalg': None,
            'scipy.optimize': None,
            'scipy.integrate': None,
            'scipy.interpolate': None,
            'scipy.signal': None,
            'scipy.stats': None,
            'sympy': 'sp',
            'pandas': 'pd',
            'matplotlib': 'mpl',
            'matplotlib.pyplot': 'plt',
        }

        for module_name, alias in _imports.items():
            try:
                mod = __import__(module_name, fromlist=[''])
                key = alias if alias else module_name.split('.')[-1]
                self._namespace[key] = mod
            except ImportError:
                pass  # silently skip unavailable libraries

        # Convenience: make math available
        import math
        self._namespace['math'] = math

    def _reset_namespace(self):
        """Clear user variables and re-initialise the scientific stack."""
        self._init_namespace()
        self._inspector.refresh(self._namespace)
        self._output.append_info("Namespace reset.\n")
        self._log("Namespace was reset")

    # ------------------------------------------------------------------ #
    #  File I/O                                                           #
    # ------------------------------------------------------------------ #

    def _open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Python File", "",
            "Python Files (*.py);;All Files (*)")
        if path:
            self.load_file(path)

    # ------------------------------------------------------------------ #
    #  Script Templates                                                   #
    # ------------------------------------------------------------------ #

    def _load_template(self, code: str):
        """Load a script template into the code editor."""
        self._editor.setPlainText(code)
        self._output.append_info("Template loaded into editor.\n")
        self._log("Script template loaded")

    # ------------------------------------------------------------------ #
    #  Snippet Insertion                                                  #
    # ------------------------------------------------------------------ #

    def _insert_snippet(self, code: str):
        """Insert a code snippet at the current editor cursor position."""
        cursor = self._editor.textCursor()
        cursor.insertText(code + "\n")
        self._editor.setTextCursor(cursor)
        self._editor.setFocus()
        self._output.append_info("Snippet inserted into editor.\n")

    # ------------------------------------------------------------------ #
    #  Generate Data                                                      #
    # ------------------------------------------------------------------ #

    def _generate_data(self):
        """Generate synthetic datasets and inject them into the namespace."""
        items = [
            "Sine + Noise",
            "Exponential Decay",
            "Gaussian Peaks",
            "Random Walk",
        ]
        choice, ok = QInputDialog.getItem(
            self, "Generate Data", "Select dataset type:", items, 0, False)
        if not ok:
            return

        code_map = {
            "Sine + Noise": (
                "import numpy as np\n"
                "n = 500\n"
                "x_gen = np.linspace(0, 4 * np.pi, n)\n"
                "y_gen = np.sin(x_gen) + 0.3 * np.random.randn(n)\n"
                "print(f'Generated sine+noise: x_gen ({n}), y_gen ({n})')"
            ),
            "Exponential Decay": (
                "import numpy as np\n"
                "n = 300\n"
                "x_gen = np.linspace(0, 10, n)\n"
                "y_gen = 5.0 * np.exp(-0.5 * x_gen) + 0.2 * np.random.randn(n)\n"
                "print(f'Generated exponential decay: x_gen ({n}), y_gen ({n})')"
            ),
            "Gaussian Peaks": (
                "import numpy as np\n"
                "n = 500\n"
                "x_gen = np.linspace(-5, 15, n)\n"
                "y_gen = (2.0 * np.exp(-0.5 * ((x_gen - 2)/0.8)**2) + \n"
                "         3.0 * np.exp(-0.5 * ((x_gen - 7)/1.2)**2) + \n"
                "         1.5 * np.exp(-0.5 * ((x_gen - 11)/0.5)**2) + \n"
                "         0.1 * np.random.randn(n))\n"
                "print(f'Generated Gaussian peaks: x_gen ({n}), y_gen ({n})')"
            ),
            "Random Walk": (
                "import numpy as np\n"
                "n = 1000\n"
                "steps = np.random.choice([-1, 1], size=n)\n"
                "x_gen = np.arange(n)\n"
                "y_gen = np.cumsum(steps).astype(float)\n"
                "print(f'Generated random walk: x_gen ({n}), y_gen ({n})')"
            ),
        }

        code = code_map.get(choice, "")
        if code:
            self._output.append_info(f"Generating dataset: {choice}\n")
            self._execute(code, source="<generate_data>")

    # ------------------------------------------------------------------ #
    #  Generate Plot                                                      #
    # ------------------------------------------------------------------ #

    def _generate_plot(self):
        """Quick-plot arrays found in the current namespace."""
        # Find plottable arrays
        arrays = {}
        for name, obj in self._namespace.items():
            if name.startswith('_'):
                continue
            try:
                import numpy as _np
                if isinstance(obj, _np.ndarray) and obj.ndim == 1 and len(obj) > 1:
                    arrays[name] = obj
            except ImportError:
                break

        if not arrays:
            self._output.append_stderr("No 1-D arrays found in namespace to plot.\n")
            return

        names = sorted(arrays.keys())
        y_choice, ok = QInputDialog.getItem(
            self, "Generate Plot", "Select Y variable:", names, 0, False)
        if not ok:
            return

        x_options = ["Auto (index)"] + names
        x_choice, ok = QInputDialog.getItem(
            self, "Generate Plot", "Select X variable:", x_options, 0, False)
        if not ok:
            return

        if x_choice == "Auto (index)":
            code = (
                f"import matplotlib.pyplot as plt\n"
                f"plt.figure(figsize=(9, 5))\n"
                f"plt.plot({y_choice}, 'o-', markersize=2)\n"
                f"plt.xlabel('Index'); plt.ylabel('{y_choice}')\n"
                f"plt.title('Quick Plot: {y_choice}')\n"
                f"plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()"
            )
        else:
            code = (
                f"import matplotlib.pyplot as plt\n"
                f"plt.figure(figsize=(9, 5))\n"
                f"plt.plot({x_choice}, {y_choice}, 'o-', markersize=2)\n"
                f"plt.xlabel('{x_choice}'); plt.ylabel('{y_choice}')\n"
                f"plt.title('Quick Plot: {y_choice} vs {x_choice}')\n"
                f"plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()"
            )

        self._execute(code, source="<generate_plot>")

    # ------------------------------------------------------------------ #
    #  Generate Report                                                    #
    # ------------------------------------------------------------------ #

    def _generate_report(self):
        """Export the current session output as an HTML report."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Report As", "session_report.html",
            "HTML Files (*.html);;All Files (*)")
        if not path:
            return

        output_text = self._output.toPlainText()
        history = self._repl_input.history()
        editor_code = self._editor.toPlainText()

        # Build variable summary
        var_lines = []
        for name, obj in sorted(self._namespace.items()):
            if name.startswith('_') or name in VariableInspector._IGNORE_NAMES:
                continue
            if isinstance(obj, ModuleType) or (callable(obj) and not isinstance(obj, type)):
                continue
            type_str = type(obj).__name__
            preview = repr(obj)
            if len(preview) > 200:
                preview = preview[:197] + "..."
            var_lines.append(f"<tr><td><code>{name}</code></td>"
                             f"<td>{type_str}</td><td><pre>{preview}</pre></td></tr>")

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>QuantumRes Session Report</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; margin: 2em; background: #fafafa; }}
h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 8px; }}
h2 {{ color: #2980b9; }}
pre {{ background: #1e1e1e; color: #dcdcdc; padding: 12px; border-radius: 6px;
       overflow-x: auto; font-size: 13px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #3498db; color: white; }}
tr:nth-child(even) {{ background: #f2f2f2; }}
.footer {{ color: #888; font-size: 0.85em; margin-top: 2em; }}
</style></head><body>
<h1>QuantumRes Session Report</h1>
<p>Generated: {timestamp}</p>

<h2>Editor Code</h2>
<pre>{editor_code if editor_code.strip() else '(empty)'}</pre>

<h2>REPL History</h2>
<pre>{chr(10).join(history) if history else '(no commands)'}</pre>

<h2>Output</h2>
<pre>{output_text if output_text.strip() else '(no output)'}</pre>

<h2>Variables</h2>
<table><tr><th>Name</th><th>Type</th><th>Value</th></tr>
{''.join(var_lines) if var_lines else '<tr><td colspan="3">(none)</td></tr>'}
</table>

<div class="footer">Generated by QuantumRes Python Console</div>
</body></html>"""

        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(html)
            self._output.append_info(f"Report saved to: {path}\n")
            self._log(f"Report exported: {path}")
        except Exception as exc:
            self._output.append_stderr(f"Error saving report: {exc}\n")

    # ------------------------------------------------------------------ #
    #  Session Save / Restore                                             #
    # ------------------------------------------------------------------ #

    def _save_session(self):
        """Save the entire session (namespace, history, editor code) to a .qrs file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", "session.qrs",
            "QuantumRes Session (*.qrs);;All Files (*)")
        if not path:
            return

        # Collect serialisable variables from namespace
        safe_ns = {}
        for name, obj in self._namespace.items():
            if name.startswith('_') or name in VariableInspector._IGNORE_NAMES:
                continue
            if isinstance(obj, ModuleType):
                continue
            if callable(obj) and not isinstance(obj, type):
                continue
            try:
                pickle.dumps(obj)  # test if picklable
                safe_ns[name] = obj
            except Exception:
                pass

        session_data = {
            'version': 1,
            'timestamp': datetime.datetime.now().isoformat(),
            'editor_code': self._editor.toPlainText(),
            'history': self._repl_input.history(),
            'output': self._output.toPlainText(),
            'namespace': safe_ns,
        }

        try:
            with open(path, 'wb') as f:
                pickle.dump(session_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            self._output.append_info(f"Session saved: {path}\n")
            self._output.append_info(f"  Saved {len(safe_ns)} variables, "
                                     f"{len(session_data['history'])} history entries.\n")
            self._log(f"Session saved: {path}")
        except Exception as exc:
            self._output.append_stderr(f"Error saving session: {exc}\n")

    def _restore_session(self):
        """Restore a session from a .qrs file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Restore Session", "",
            "QuantumRes Session (*.qrs);;All Files (*)")
        if not path:
            return

        try:
            with open(path, 'rb') as f:
                session_data = pickle.load(f)

            # Restore editor code
            if 'editor_code' in session_data:
                self._editor.setPlainText(session_data['editor_code'])

            # Restore output
            if 'output' in session_data:
                self._output.clear()
                self._output.append_info(session_data['output'])
                self._output.append_info("\n--- Session restored ---\n")

            # Restore history
            if 'history' in session_data:
                self._repl_input._history = session_data['history']
                self._repl_input._history_index = len(session_data['history'])

            # Restore namespace (merge into current namespace)
            if 'namespace' in session_data:
                for name, obj in session_data['namespace'].items():
                    self._namespace[name] = obj

            self._inspector.refresh(self._namespace)
            n_vars = len(session_data.get('namespace', {}))
            n_hist = len(session_data.get('history', []))
            ts = session_data.get('timestamp', 'unknown')
            self._output.append_info(
                f"Restored {n_vars} variables, {n_hist} history entries "
                f"from session dated {ts}.\n")
            self._log(f"Session restored: {path}")
        except Exception as exc:
            self._output.append_stderr(f"Error restoring session: {exc}\n")

    # ------------------------------------------------------------------ #
    #  Logging                                                            #
    # ------------------------------------------------------------------ #

    def _log(self, message: str):
        if self._logger is not None:
            try:
                self._logger(message)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Standalone entry point (for testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette for standalone testing
    dark = QPalette()
    dark.setColor(QPalette.Window, QColor(53, 53, 53))
    dark.setColor(QPalette.WindowText, QColor(220, 220, 220))
    dark.setColor(QPalette.Base, QColor(35, 35, 35))
    dark.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
    dark.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
    dark.setColor(QPalette.Text, QColor(220, 220, 220))
    dark.setColor(QPalette.Button, QColor(53, 53, 53))
    dark.setColor(QPalette.ButtonText, QColor(220, 220, 220))
    dark.setColor(QPalette.BrightText, QColor(255, 0, 0))
    dark.setColor(QPalette.Link, QColor(42, 130, 218))
    dark.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark.setColor(QPalette.HighlightedText, QColor(35, 35, 35))
    app.setPalette(dark)

    console = PythonConsoleWidget()
    console.setWindowTitle("Python Console - QuantumRes")
    console.resize(1100, 700)
    console.show()

    sys.exit(app.exec_())
