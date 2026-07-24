"""
pdf_tools.py - PDF Tools Widget for PyQt5 Scientific Suite

Provides PDF information display, merge, split, text extraction, PDF creation
from text, PostScript generation, and page thumbnail rendering.  Uses PyPDF2
or pikepdf when available; falls back gracefully with user instructions.
"""

import os
import re
import struct
import traceback

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTabWidget,
    QPushButton, QLabel, QFileDialog, QPlainTextEdit, QLineEdit,
    QListWidget, QListWidgetItem, QGroupBox, QFormLayout, QSpinBox,
    QMessageBox, QToolBar, QAction, QTextEdit, QSizePolicy,
    QProgressBar, QGridLayout, QScrollArea, QComboBox, QCheckBox,
    QDoubleSpinBox, QColorDialog, QSlider, QTableWidget, QTableWidgetItem,
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QPixmap, QImage, QIcon, QColor

# Optional dependencies -------------------------------------------------
try:
    from PyPDF2 import PdfReader, PdfWriter
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import pikepdf
    HAS_PIKEPDF = True
except ImportError:
    HAS_PIKEPDF = False

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.pdfgen import canvas as rl_canvas
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


def _available_backend():
    """Return the best available PDF backend name."""
    if HAS_PYPDF2:
        return "PyPDF2"
    if HAS_PIKEPDF:
        return "pikepdf"
    return None


# ---------------------------------------------------------------------------
# PDF Tools Widget
# ---------------------------------------------------------------------------

class PDFToolsWidget(QWidget):
    """Multi-tab PDF utility widget with info, merge, split, extract,
    create, PostScript, and thumbnail capabilities."""

    contentChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._log = lambda msg: None
        self._current_path = None
        self._merge_list = []
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
        act_open = QAction("Open PDF", self)
        act_open.triggered.connect(self._action_open)
        toolbar.addAction(act_open)
        toolbar.addSeparator()

        backend = _available_backend()
        lbl = QLabel(f"  Backend: {backend or 'none (install PyPDF2)'}")
        lbl.setStyleSheet("color: " + ("#4CAF50" if backend else "#F44336"))
        toolbar.addWidget(lbl)
        root.addWidget(toolbar)

        # Tabs
        self._tabs = QTabWidget()

        self._tabs.addTab(self._build_info_tab(), "Info")
        self._tabs.addTab(self._build_merge_tab(), "Merge")
        self._tabs.addTab(self._build_split_tab(), "Split")
        self._tabs.addTab(self._build_extract_tab(), "Extract Text")
        self._tabs.addTab(self._build_create_tab(), "Create PDF")
        self._tabs.addTab(self._build_ps_tab(), "PostScript")
        self._tabs.addTab(self._build_thumbnails_tab(), "Thumbnails")
        self._tabs.addTab(self._build_report_generator_tab(), "Report Gen")
        self._tabs.addTab(self._build_form_creator_tab(), "Forms")
        self._tabs.addTab(self._build_annotation_tab(), "Annotate")
        self._tabs.addTab(self._build_watermark_tab(), "Watermark")
        self._tabs.addTab(self._build_comparison_tab(), "Compare")
        self._tabs.addTab(self._build_batch_convert_tab(), "Batch Convert")

        root.addWidget(self._tabs, 1)

        # Status
        self._status = QLabel("Ready")
        root.addWidget(self._status)

    # -- Info tab ----------------------------------------------------------

    def _build_info_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self._info_text = QTextEdit()
        self._info_text.setReadOnly(True)
        self._info_text.setFont(QFont("Consolas", 10))
        lay.addWidget(self._info_text)
        btn = QPushButton("Refresh Info")
        btn.clicked.connect(self._refresh_info)
        lay.addWidget(btn)
        return w

    def _refresh_info(self):
        if not self._current_path:
            self._info_text.setPlainText("No PDF loaded. Use 'Open PDF' to load a file.")
            return
        info_lines = [f"File: {self._current_path}"]
        try:
            size = os.path.getsize(self._current_path)
            info_lines.append(f"Size: {size:,} bytes ({size / 1024:.1f} KB)")
        except OSError:
            pass

        if HAS_PYPDF2:
            try:
                reader = PdfReader(self._current_path)
                info_lines.append(f"Pages: {len(reader.pages)}")
                meta = reader.metadata
                if meta:
                    for key in ("/Title", "/Author", "/Subject", "/Creator",
                                "/Producer", "/CreationDate", "/ModDate"):
                        val = meta.get(key)
                        if val:
                            info_lines.append(f"{key[1:]}: {val}")
                # Page size of first page
                if reader.pages:
                    page = reader.pages[0]
                    box = page.mediabox
                    w_pt = float(box.width)
                    h_pt = float(box.height)
                    info_lines.append(
                        f"Page size: {w_pt:.0f} x {h_pt:.0f} pt "
                        f"({w_pt / 72:.2f} x {h_pt / 72:.2f} in)"
                    )
            except Exception as exc:
                info_lines.append(f"Error reading PDF: {exc}")
        elif HAS_PIKEPDF:
            try:
                pdf = pikepdf.open(self._current_path)
                info_lines.append(f"Pages: {len(pdf.pages)}")
                with pdf.open_metadata() as meta:
                    for k, v in meta.items():
                        info_lines.append(f"{k}: {v}")
                pdf.close()
            except Exception as exc:
                info_lines.append(f"Error reading PDF: {exc}")
        else:
            info_lines.append("")
            info_lines.append("Install PyPDF2 or pikepdf for full metadata:")
            info_lines.append("  pip install PyPDF2")
            # Basic parsing: count pages via xref
            try:
                with open(self._current_path, "rb") as f:
                    data = f.read()
                page_count = data.count(b"/Type /Page") - data.count(b"/Type /Pages")
                info_lines.append(f"Pages (estimated): {max(page_count, 0)}")
            except Exception:
                pass

        self._info_text.setPlainText("\n".join(info_lines))
        self._emit_log("PDF info refreshed")

    # -- Merge tab ---------------------------------------------------------

    def _build_merge_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Add PDFs to merge (order matters):"))

        self._merge_listwidget = QListWidget()
        lay.addWidget(self._merge_listwidget)

        btn_row = QHBoxLayout()
        btn_add = QPushButton("Add File")
        btn_add.clicked.connect(self._merge_add)
        btn_row.addWidget(btn_add)

        btn_remove = QPushButton("Remove Selected")
        btn_remove.clicked.connect(self._merge_remove)
        btn_row.addWidget(btn_remove)

        btn_up = QPushButton("Move Up")
        btn_up.clicked.connect(lambda: self._merge_move(-1))
        btn_row.addWidget(btn_up)

        btn_down = QPushButton("Move Down")
        btn_down.clicked.connect(lambda: self._merge_move(1))
        btn_row.addWidget(btn_down)
        lay.addLayout(btn_row)

        btn_merge = QPushButton("Merge and Save")
        btn_merge.clicked.connect(self._merge_execute)
        lay.addWidget(btn_merge)
        return w

    def _merge_add(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select PDFs", "", "PDF Files (*.pdf)"
        )
        for p in paths:
            self._merge_list.append(p)
            self._merge_listwidget.addItem(os.path.basename(p))

    def _merge_remove(self):
        row = self._merge_listwidget.currentRow()
        if row >= 0:
            self._merge_list.pop(row)
            self._merge_listwidget.takeItem(row)

    def _merge_move(self, direction):
        row = self._merge_listwidget.currentRow()
        new_row = row + direction
        if 0 <= row < len(self._merge_list) and 0 <= new_row < len(self._merge_list):
            self._merge_list[row], self._merge_list[new_row] = (
                self._merge_list[new_row], self._merge_list[row]
            )
            item = self._merge_listwidget.takeItem(row)
            self._merge_listwidget.insertItem(new_row, item)
            self._merge_listwidget.setCurrentRow(new_row)

    def _merge_execute(self):
        if len(self._merge_list) < 2:
            QMessageBox.warning(self, "Merge", "Add at least two PDF files.")
            return
        if not HAS_PYPDF2 and not HAS_PIKEPDF:
            QMessageBox.information(
                self, "Missing Dependency",
                "PDF merge requires PyPDF2 or pikepdf.\n\n"
                "Install with: pip install PyPDF2"
            )
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Merged PDF", "", "PDF Files (*.pdf)"
        )
        if not out_path:
            return

        try:
            if HAS_PYPDF2:
                writer = PdfWriter()
                for p in self._merge_list:
                    reader = PdfReader(p)
                    for page in reader.pages:
                        writer.add_page(page)
                with open(out_path, "wb") as f:
                    writer.write(f)
            elif HAS_PIKEPDF:
                pdf_out = pikepdf.Pdf.new()
                for p in self._merge_list:
                    src = pikepdf.open(p)
                    pdf_out.pages.extend(src.pages)
                pdf_out.save(out_path)
                pdf_out.close()
            self._status.setText(f"Merged {len(self._merge_list)} files")
            self._emit_log(f"Merged {len(self._merge_list)} PDFs -> {out_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Merge Error", str(exc))
            self._emit_log(f"Merge error: {exc}")

    # -- Split tab ---------------------------------------------------------

    def _build_split_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Extract page range from the loaded PDF:"))

        form = QFormLayout()
        self._split_from = QSpinBox()
        self._split_from.setMinimum(1)
        self._split_from.setValue(1)
        form.addRow("From page:", self._split_from)

        self._split_to = QSpinBox()
        self._split_to.setMinimum(1)
        self._split_to.setValue(1)
        form.addRow("To page:", self._split_to)
        lay.addLayout(form)

        btn = QPushButton("Split and Save")
        btn.clicked.connect(self._split_execute)
        lay.addWidget(btn)
        lay.addStretch()
        return w

    def _split_execute(self):
        if not self._current_path:
            QMessageBox.warning(self, "Split", "Open a PDF first.")
            return
        if not HAS_PYPDF2 and not HAS_PIKEPDF:
            QMessageBox.information(
                self, "Missing Dependency",
                "PDF split requires PyPDF2 or pikepdf.\n\n"
                "Install with: pip install PyPDF2"
            )
            return

        start = self._split_from.value() - 1
        end = self._split_to.value()
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Split PDF", "", "PDF Files (*.pdf)"
        )
        if not out_path:
            return

        try:
            if HAS_PYPDF2:
                reader = PdfReader(self._current_path)
                writer = PdfWriter()
                for i in range(start, min(end, len(reader.pages))):
                    writer.add_page(reader.pages[i])
                with open(out_path, "wb") as f:
                    writer.write(f)
            elif HAS_PIKEPDF:
                src = pikepdf.open(self._current_path)
                dst = pikepdf.Pdf.new()
                for i in range(start, min(end, len(src.pages))):
                    dst.pages.append(src.pages[i])
                dst.save(out_path)
                dst.close()
                src.close()

            self._status.setText(f"Split pages {start+1}-{end}")
            self._emit_log(f"Split pages {start+1}-{end} -> {out_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Split Error", str(exc))
            self._emit_log(f"Split error: {exc}")

    # -- Extract Text tab --------------------------------------------------

    def _build_extract_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Extracted text from loaded PDF:"))
        self._extract_text = QPlainTextEdit()
        self._extract_text.setReadOnly(True)
        self._extract_text.setFont(QFont("Consolas", 10))
        lay.addWidget(self._extract_text)
        btn = QPushButton("Extract Text")
        btn.clicked.connect(self._extract_execute)
        lay.addWidget(btn)
        btn_save = QPushButton("Save Text As...")
        btn_save.clicked.connect(self._extract_save)
        lay.addWidget(btn_save)
        return w

    def _extract_execute(self):
        if not self._current_path:
            self._extract_text.setPlainText("No PDF loaded.")
            return
        text_parts = []
        try:
            if HAS_PYPDF2:
                reader = PdfReader(self._current_path)
                for i, page in enumerate(reader.pages):
                    t = page.extract_text() or ""
                    text_parts.append(f"--- Page {i+1} ---\n{t}\n")
            elif HAS_PIKEPDF:
                pdf = pikepdf.open(self._current_path)
                text_parts.append(
                    "pikepdf does not support text extraction.\n"
                    "Install PyPDF2: pip install PyPDF2"
                )
                pdf.close()
            else:
                text_parts.append(
                    "No PDF library available for text extraction.\n"
                    "Install PyPDF2: pip install PyPDF2"
                )
        except Exception as exc:
            text_parts.append(f"Error: {exc}")

        self._extract_text.setPlainText("\n".join(text_parts))
        self._emit_log("Text extracted from PDF")

    def _extract_save(self):
        text = self._extract_text.toPlainText()
        if not text.strip():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Extracted Text", "", "Text Files (*.txt);;All Files (*)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            self._emit_log(f"Extracted text saved to {path}")

    # -- Create PDF tab ----------------------------------------------------

    def _build_create_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Enter text to generate a PDF:"))
        self._create_text = QPlainTextEdit()
        self._create_text.setFont(QFont("Consolas", 10))
        self._create_text.setPlaceholderText("Type or paste text here...")
        lay.addWidget(self._create_text)

        opt_row = QHBoxLayout()
        opt_row.addWidget(QLabel("Page size:"))
        self._page_size_combo = QComboBox()
        self._page_size_combo.addItems(["Letter", "A4"])
        opt_row.addWidget(self._page_size_combo)
        opt_row.addStretch()
        lay.addLayout(opt_row)

        btn = QPushButton("Generate PDF")
        btn.clicked.connect(self._create_execute)
        lay.addWidget(btn)
        return w

    def _create_execute(self):
        text = self._create_text.toPlainText()
        if not text.strip():
            QMessageBox.warning(self, "Create PDF", "Enter some text first.")
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF", "", "PDF Files (*.pdf)"
        )
        if not out_path:
            return

        if HAS_REPORTLAB:
            try:
                page_size = letter if self._page_size_combo.currentText() == "Letter" else A4
                c = rl_canvas.Canvas(out_path, pagesize=page_size)
                width, height = page_size
                margin = 72
                y = height - margin
                c.setFont("Helvetica", 11)
                for line in text.split("\n"):
                    if y < margin:
                        c.showPage()
                        c.setFont("Helvetica", 11)
                        y = height - margin
                    c.drawString(margin, y, line)
                    y -= 14
                c.save()
                self._status.setText(f"Created: {os.path.basename(out_path)}")
                self._emit_log(f"Created PDF: {out_path}")
            except Exception as exc:
                QMessageBox.critical(self, "Create Error", str(exc))
                self._emit_log(f"PDF create error: {exc}")
        else:
            # Fallback: write minimal valid PDF manually
            try:
                self._write_minimal_pdf(out_path, text)
                self._status.setText(f"Created (basic): {os.path.basename(out_path)}")
                self._emit_log(f"Created basic PDF: {out_path}")
            except Exception as exc:
                QMessageBox.critical(self, "Create Error", str(exc))
                self._emit_log(f"PDF create error: {exc}")

    def _write_minimal_pdf(self, path, text):
        """Write a minimal valid PDF with text content (no external deps)."""
        lines_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        # Build content stream
        stream_lines = ["BT", "/F1 11 Tf", "72 720 Td", "14 TL"]
        for line in lines_text.split("\n"):
            stream_lines.append(f"({line}) '")
        stream_lines.append("ET")
        stream = "\n".join(stream_lines)

        objects = []

        # 1 - Catalog
        objects.append("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj")
        # 2 - Pages
        objects.append("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj")
        # 3 - Page
        objects.append(
            "3 0 obj\n<< /Type /Page /Parent 2 0 R "
            "/MediaBox [0 0 612 792] "
            "/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj"
        )
        # 4 - Content stream
        objects.append(
            f"4 0 obj\n<< /Length {len(stream)} >>\nstream\n{stream}\nendstream\nendobj"
        )
        # 5 - Font
        objects.append(
            "5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj"
        )

        with open(path, "w", encoding="latin-1") as f:
            f.write("%PDF-1.4\n")
            offsets = []
            for obj in objects:
                offsets.append(f.tell())
                f.write(obj + "\n")
            xref_pos = f.tell()
            f.write("xref\n")
            f.write(f"0 {len(objects) + 1}\n")
            f.write("0000000000 65535 f \n")
            for off in offsets:
                f.write(f"{off:010d} 00000 n \n")
            f.write("trailer\n")
            f.write(f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n")
            f.write("startxref\n")
            f.write(f"{xref_pos}\n")
            f.write("%%EOF\n")

    # -- PostScript tab ----------------------------------------------------

    def _build_ps_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Generate a PostScript (.ps) file from text:"))
        self._ps_text = QPlainTextEdit()
        self._ps_text.setFont(QFont("Consolas", 10))
        self._ps_text.setPlaceholderText("Type or paste text for PS output...")
        lay.addWidget(self._ps_text)
        btn = QPushButton("Generate PostScript")
        btn.clicked.connect(self._ps_execute)
        lay.addWidget(btn)
        return w

    def _ps_execute(self):
        text = self._ps_text.toPlainText()
        if not text.strip():
            QMessageBox.warning(self, "PostScript", "Enter some text first.")
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save PostScript", "", "PostScript Files (*.ps);;All Files (*)"
        )
        if not out_path:
            return
        try:
            with open(out_path, "w", encoding="latin-1") as f:
                f.write("%!PS-Adobe-3.0\n")
                f.write("%%Title: Generated PostScript\n")
                f.write("%%Pages: 1\n")
                f.write("%%EndComments\n\n")
                f.write("%%Page: 1 1\n")
                f.write("/Helvetica findfont 11 scalefont setfont\n")
                y = 720
                for line in text.split("\n"):
                    escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
                    f.write(f"72 {y} moveto ({escaped}) show\n")
                    y -= 14
                    if y < 72:
                        f.write("showpage\n")
                        f.write("%%Page: next next\n")
                        f.write("/Helvetica findfont 11 scalefont setfont\n")
                        y = 720
                f.write("showpage\n")
                f.write("%%EOF\n")
            self._status.setText(f"PostScript saved: {os.path.basename(out_path)}")
            self._emit_log(f"PostScript generated: {out_path}")
        except Exception as exc:
            QMessageBox.critical(self, "PS Error", str(exc))
            self._emit_log(f"PostScript error: {exc}")

    # -- Thumbnails tab ----------------------------------------------------

    def _build_thumbnails_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Page thumbnails (requires PyMuPDF):"))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._thumb_container = QWidget()
        self._thumb_layout = QGridLayout(self._thumb_container)
        self._thumb_layout.setSpacing(8)
        scroll.setWidget(self._thumb_container)
        lay.addWidget(scroll)

        btn = QPushButton("Generate Thumbnails")
        btn.clicked.connect(self._generate_thumbnails)
        lay.addWidget(btn)
        return w

    def _generate_thumbnails(self):
        # Clear existing
        while self._thumb_layout.count():
            item = self._thumb_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        if not self._current_path:
            lbl = QLabel("No PDF loaded.")
            self._thumb_layout.addWidget(lbl, 0, 0)
            return

        if HAS_FITZ:
            try:
                doc = fitz.open(self._current_path)
                cols = 3
                for i in range(len(doc)):
                    page = doc[i]
                    pix = page.get_pixmap(matrix=fitz.Matrix(0.3, 0.3))
                    img = QImage(pix.samples, pix.width, pix.height,
                                 pix.stride, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(img)
                    lbl = QLabel()
                    lbl.setPixmap(pixmap)
                    lbl.setToolTip(f"Page {i + 1}")
                    lbl.setStyleSheet("border: 1px solid #999;")
                    self._thumb_layout.addWidget(lbl, i // cols, i % cols)
                doc.close()
                self._emit_log(f"Generated {len(doc)} thumbnails")
            except Exception as exc:
                lbl = QLabel(f"Error: {exc}")
                self._thumb_layout.addWidget(lbl, 0, 0)
        else:
            # Fallback: use matplotlib to show placeholder
            try:
                fig = Figure(figsize=(2, 2.5), dpi=72)
                fig.patch.set_facecolor("#f0f0f0")
                ax = fig.add_subplot(111)
                ax.set_axis_off()
                ax.text(0.5, 0.5, "Install PyMuPDF\nfor thumbnails:\npip install PyMuPDF",
                        ha="center", va="center", fontsize=8, transform=ax.transAxes)
                canvas = FigureCanvas(fig)
                canvas.setFixedSize(QSize(160, 200))
                self._thumb_layout.addWidget(canvas, 0, 0)
            except Exception:
                lbl = QLabel("Install PyMuPDF for page thumbnails:\n  pip install PyMuPDF")
                self._thumb_layout.addWidget(lbl, 0, 0)

    # -- Report Generator tab ----------------------------------------------

    def _build_report_generator_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Generate a multi-page PDF report:"))

        form = QFormLayout()
        self._rpt_title = QLineEdit("Report Title")
        form.addRow("Title:", self._rpt_title)
        self._rpt_author = QLineEdit("Author Name")
        form.addRow("Author:", self._rpt_author)
        self._rpt_page_size = QComboBox()
        self._rpt_page_size.addItems(["Letter", "A4"])
        form.addRow("Page size:", self._rpt_page_size)
        self._rpt_header = QCheckBox("Include headers")
        self._rpt_header.setChecked(True)
        form.addRow(self._rpt_header)
        self._rpt_page_nums = QCheckBox("Include page numbers")
        self._rpt_page_nums.setChecked(True)
        form.addRow(self._rpt_page_nums)
        self._rpt_title_page = QCheckBox("Title page")
        self._rpt_title_page.setChecked(True)
        form.addRow(self._rpt_title_page)
        self._rpt_plot = QCheckBox("Embed sample plot")
        form.addRow(self._rpt_plot)
        lay.addLayout(form)

        lay.addWidget(QLabel("Report body text:"))
        self._rpt_body = QPlainTextEdit()
        self._rpt_body.setFont(QFont("Consolas", 10))
        self._rpt_body.setPlaceholderText("Enter report content. Use blank lines to separate paragraphs.")
        lay.addWidget(self._rpt_body)

        btn = QPushButton("Generate PDF Report")
        btn.clicked.connect(self._generate_report)
        lay.addWidget(btn)
        return w

    def _generate_report(self):
        if not HAS_REPORTLAB:
            QMessageBox.information(
                self, "Missing Dependency",
                "PDF report generation requires reportlab.\n\n"
                "Install with: pip install reportlab"
            )
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF Report", "", "PDF Files (*.pdf)"
        )
        if not out_path:
            return

        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.pdfgen import canvas as rl_canvas
            from reportlab.lib.units import inch

            page_size = letter if self._rpt_page_size.currentText() == "Letter" else A4
            width, height = page_size
            margin = 72
            c = rl_canvas.Canvas(out_path, pagesize=page_size)

            title = self._rpt_title.text() or "Report"
            author = self._rpt_author.text() or ""
            page_num = [0]

            def finish_page():
                page_num[0] += 1
                if self._rpt_header.isChecked():
                    c.saveState()
                    c.setFont("Helvetica", 8)
                    c.drawString(margin, height - 40, title)
                    c.drawRightString(width - margin, height - 40, author)
                    c.line(margin, height - 45, width - margin, height - 45)
                    c.restoreState()
                if self._rpt_page_nums.isChecked():
                    c.saveState()
                    c.setFont("Helvetica", 9)
                    c.drawCentredString(width / 2, 30, f"- {page_num[0]} -")
                    c.restoreState()
                c.showPage()

            # Title page
            if self._rpt_title_page.isChecked():
                c.setFont("Helvetica-Bold", 28)
                c.drawCentredString(width / 2, height / 2 + 40, title)
                c.setFont("Helvetica", 16)
                c.drawCentredString(width / 2, height / 2 - 10, author)
                c.setFont("Helvetica", 12)
                from datetime import datetime
                c.drawCentredString(width / 2, height / 2 - 50, datetime.now().strftime("%B %d, %Y"))
                finish_page()

            # Body
            body = self._rpt_body.toPlainText()
            if body.strip():
                y = height - margin - 20
                c.setFont("Helvetica", 11)
                for line in body.split("\n"):
                    if y < margin + 30:
                        finish_page()
                        y = height - margin - 20
                        c.setFont("Helvetica", 11)
                    if not line.strip():
                        y -= 10
                        continue
                    # Word wrap at ~80 chars
                    while len(line) > 80:
                        split_at = line.rfind(" ", 0, 80)
                        if split_at == -1:
                            split_at = 80
                        c.drawString(margin, y, line[:split_at])
                        line = line[split_at:].lstrip()
                        y -= 14
                        if y < margin + 30:
                            finish_page()
                            y = height - margin - 20
                            c.setFont("Helvetica", 11)
                    c.drawString(margin, y, line)
                    y -= 14

            # Sample plot
            if self._rpt_plot.isChecked():
                try:
                    fig = Figure(figsize=(5, 3), dpi=100)
                    ax = fig.add_subplot(111)
                    x = np.linspace(0, 10, 100)
                    ax.plot(x, np.sin(x), label="sin(x)")
                    ax.plot(x, np.cos(x), label="cos(x)")
                    ax.set_title("Sample Plot")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    fig.tight_layout()

                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        fig.savefig(tmp.name, dpi=150)
                        tmp_path = tmp.name

                    if y < margin + 250:
                        finish_page()
                        y = height - margin - 20
                    c.drawImage(tmp_path, margin, y - 230, width=400, height=220)
                    y -= 250
                    os.unlink(tmp_path)
                except Exception as exc:
                    self._emit_log(f"Plot embed error: {exc}")

            finish_page()
            c.save()
            self._status.setText(f"Report generated: {os.path.basename(out_path)}")
            self._emit_log(f"Generated report: {out_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Report Error", str(exc))
            self._emit_log(f"Report error: {exc}")

    # -- Form Creator tab --------------------------------------------------

    def _build_form_creator_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Create a fillable PDF form (requires reportlab):"))

        self._form_fields_list = QListWidget()
        lay.addWidget(QLabel("Form fields:"))
        lay.addWidget(self._form_fields_list)
        self._form_fields_data = []

        form = QFormLayout()
        self._form_field_name = QLineEdit("field_name")
        form.addRow("Field name:", self._form_field_name)

        self._form_field_type = QComboBox()
        self._form_field_type.addItems(["Text Field", "Checkbox", "Dropdown"])
        form.addRow("Field type:", self._form_field_type)

        self._form_field_label = QLineEdit("Label:")
        form.addRow("Label:", self._form_field_label)

        self._form_dropdown_opts = QLineEdit("Option1, Option2, Option3")
        form.addRow("Dropdown options:", self._form_dropdown_opts)
        lay.addLayout(form)

        btn_add = QPushButton("Add Field")
        btn_add.clicked.connect(self._add_form_field)
        lay.addWidget(btn_add)

        btn_gen = QPushButton("Generate PDF Form")
        btn_gen.clicked.connect(self._generate_pdf_form)
        lay.addWidget(btn_gen)
        return w

    def _add_form_field(self):
        name = self._form_field_name.text()
        ftype = self._form_field_type.currentText()
        label = self._form_field_label.text()
        opts = self._form_dropdown_opts.text()
        self._form_fields_data.append({
            "name": name, "type": ftype, "label": label, "options": opts
        })
        self._form_fields_list.addItem(f"{ftype}: {label} ({name})")

    def _generate_pdf_form(self):
        if not HAS_REPORTLAB:
            QMessageBox.information(
                self, "Missing Dependency",
                "PDF form creation requires reportlab.\n\nInstall with: pip install reportlab"
            )
            return
        if not self._form_fields_data:
            QMessageBox.warning(self, "Form Creator", "Add at least one field.")
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF Form", "", "PDF Files (*.pdf)"
        )
        if not out_path:
            return

        try:
            from reportlab.pdfgen import canvas as rl_canvas
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors

            width, height = letter
            c = rl_canvas.Canvas(out_path, pagesize=letter)
            c.setFont("Helvetica-Bold", 16)
            c.drawString(72, height - 72, "Fillable Form")
            c.setFont("Helvetica", 11)

            y = height - 120
            form = c.acroForm

            for field in self._form_fields_data:
                if y < 100:
                    c.showPage()
                    y = height - 72

                c.drawString(72, y, field["label"])

                if field["type"] == "Text Field":
                    form.textfield(
                        name=field["name"],
                        x=200, y=y - 5,
                        width=250, height=20,
                        borderWidth=1,
                        borderColor=colors.black,
                        fillColor=colors.white,
                        fontSize=10,
                    )
                elif field["type"] == "Checkbox":
                    form.checkbox(
                        name=field["name"],
                        x=200, y=y - 5,
                        size=15,
                        borderWidth=1,
                        borderColor=colors.black,
                    )
                elif field["type"] == "Dropdown":
                    options = [o.strip() for o in field["options"].split(",") if o.strip()]
                    form.choice(
                        name=field["name"],
                        options=options,
                        x=200, y=y - 5,
                        width=250, height=20,
                        borderWidth=1,
                        borderColor=colors.black,
                        fillColor=colors.white,
                        fontSize=10,
                    )
                y -= 40

            c.save()
            self._status.setText(f"Form created: {os.path.basename(out_path)}")
            self._emit_log(f"Created PDF form: {out_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Form Error", str(exc))
            self._emit_log(f"Form error: {exc}")

    # -- Annotation tab ----------------------------------------------------

    def _build_annotation_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Annotate the loaded PDF (requires PyMuPDF):"))

        form = QFormLayout()
        self._annot_page = QSpinBox()
        self._annot_page.setMinimum(1)
        self._annot_page.setValue(1)
        form.addRow("Page:", self._annot_page)

        self._annot_type = QComboBox()
        self._annot_type.addItems(["Highlight", "Text Note", "Stamp"])
        form.addRow("Annotation:", self._annot_type)

        self._annot_text = QLineEdit("Annotation text here")
        form.addRow("Text/content:", self._annot_text)

        self._annot_x = QSpinBox()
        self._annot_x.setRange(0, 1000)
        self._annot_x.setValue(72)
        form.addRow("X position:", self._annot_x)

        self._annot_y = QSpinBox()
        self._annot_y.setRange(0, 1000)
        self._annot_y.setValue(700)
        form.addRow("Y position:", self._annot_y)

        self._annot_stamp = QComboBox()
        self._annot_stamp.addItems(["Draft", "Confidential", "Approved", "Final",
                                     "Expired", "NotApproved"])
        form.addRow("Stamp type:", self._annot_stamp)

        lay.addLayout(form)

        btn = QPushButton("Add Annotation & Save")
        btn.clicked.connect(self._add_annotation)
        lay.addWidget(btn)
        lay.addStretch()
        return w

    def _add_annotation(self):
        if not self._current_path:
            QMessageBox.warning(self, "Annotate", "Open a PDF first.")
            return
        if not HAS_FITZ:
            QMessageBox.information(
                self, "Missing Dependency",
                "PDF annotation requires PyMuPDF.\n\nInstall with: pip install PyMuPDF"
            )
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Annotated PDF", "", "PDF Files (*.pdf)"
        )
        if not out_path:
            return

        try:
            doc = fitz.open(self._current_path)
            page_idx = self._annot_page.value() - 1
            if page_idx >= len(doc):
                QMessageBox.warning(self, "Annotate", f"Page {page_idx+1} does not exist.")
                doc.close()
                return

            page = doc[page_idx]
            atype = self._annot_type.currentText()
            x = self._annot_x.value()
            y = self._annot_y.value()
            text = self._annot_text.text()

            if atype == "Highlight":
                rect = fitz.Rect(x, y, x + 300, y + 20)
                annot = page.add_highlight_annot(rect)
                annot.set_info(content=text)
                annot.update()
            elif atype == "Text Note":
                point = fitz.Point(x, y)
                annot = page.add_text_annot(point, text)
                annot.update()
            elif atype == "Stamp":
                rect = fitz.Rect(x, y, x + 200, y + 60)
                stamp_name = self._annot_stamp.currentText()
                stamp_map = {
                    "Draft": 0, "Confidential": 2, "Approved": 4,
                    "Final": 6, "Expired": 8, "NotApproved": 10,
                }
                annot = page.add_stamp_annot(rect, stamp=stamp_map.get(stamp_name, 0))
                annot.update()

            doc.save(out_path)
            doc.close()
            self._status.setText(f"Annotated PDF saved: {os.path.basename(out_path)}")
            self._emit_log(f"Annotated PDF saved: {out_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Annotation Error", str(exc))
            self._emit_log(f"Annotation error: {exc}")

    # -- Watermark tab -----------------------------------------------------

    def _build_watermark_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Add watermark to loaded PDF:"))

        form = QFormLayout()
        self._wm_text = QLineEdit("CONFIDENTIAL")
        form.addRow("Watermark text:", self._wm_text)

        self._wm_fontsize = QSpinBox()
        self._wm_fontsize.setRange(10, 200)
        self._wm_fontsize.setValue(60)
        form.addRow("Font size:", self._wm_fontsize)

        self._wm_opacity = QDoubleSpinBox()
        self._wm_opacity.setRange(0.05, 1.0)
        self._wm_opacity.setSingleStep(0.05)
        self._wm_opacity.setValue(0.15)
        form.addRow("Opacity:", self._wm_opacity)

        self._wm_angle = QSpinBox()
        self._wm_angle.setRange(-90, 90)
        self._wm_angle.setValue(45)
        form.addRow("Rotation angle:", self._wm_angle)

        self._wm_color = QComboBox()
        self._wm_color.addItems(["Red", "Gray", "Blue", "Green", "Black"])
        form.addRow("Color:", self._wm_color)

        lay.addLayout(form)

        btn = QPushButton("Apply Watermark & Save")
        btn.clicked.connect(self._apply_watermark)
        lay.addWidget(btn)
        lay.addStretch()
        return w

    def _apply_watermark(self):
        if not self._current_path:
            QMessageBox.warning(self, "Watermark", "Open a PDF first.")
            return

        if not HAS_FITZ:
            QMessageBox.information(
                self, "Missing Dependency",
                "Watermarking requires PyMuPDF.\n\nInstall with: pip install PyMuPDF"
            )
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Watermarked PDF", "", "PDF Files (*.pdf)"
        )
        if not out_path:
            return

        try:
            doc = fitz.open(self._current_path)
            wm_text = self._wm_text.text() or "WATERMARK"
            fontsize = self._wm_fontsize.value()
            opacity = self._wm_opacity.value()
            angle = self._wm_angle.value()
            color_map = {
                "Red": (1, 0, 0), "Gray": (0.5, 0.5, 0.5),
                "Blue": (0, 0, 1), "Green": (0, 0.5, 0),
                "Black": (0, 0, 0),
            }
            color = color_map.get(self._wm_color.currentText(), (0.5, 0.5, 0.5))

            for page in doc:
                rect = page.rect
                center = fitz.Point(rect.width / 2, rect.height / 2)
                text_length = fitz.get_text_length(wm_text, fontsize=fontsize)
                insert_point = fitz.Point(center.x - text_length / 2, center.y)
                page.insert_text(
                    insert_point,
                    wm_text,
                    fontsize=fontsize,
                    color=color,
                    rotate=angle,
                    overlay=True,
                    opacity=opacity,
                )

            doc.save(out_path)
            doc.close()
            self._status.setText(f"Watermarked: {os.path.basename(out_path)}")
            self._emit_log(f"Watermarked PDF: {out_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Watermark Error", str(exc))
            self._emit_log(f"Watermark error: {exc}")

    # -- Comparison tab ----------------------------------------------------

    def _build_comparison_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Compare two PDFs side-by-side (requires PyMuPDF):"))

        file_row = QHBoxLayout()
        self._cmp_path1 = QLineEdit()
        self._cmp_path1.setPlaceholderText("First PDF...")
        file_row.addWidget(self._cmp_path1)
        btn1 = QPushButton("Browse")
        btn1.clicked.connect(lambda: self._browse_cmp_file(self._cmp_path1))
        file_row.addWidget(btn1)
        lay.addLayout(file_row)

        file_row2 = QHBoxLayout()
        self._cmp_path2 = QLineEdit()
        self._cmp_path2.setPlaceholderText("Second PDF...")
        file_row2.addWidget(self._cmp_path2)
        btn2 = QPushButton("Browse")
        btn2.clicked.connect(lambda: self._browse_cmp_file(self._cmp_path2))
        file_row2.addWidget(btn2)
        lay.addLayout(file_row2)

        self._cmp_page = QSpinBox()
        self._cmp_page.setMinimum(1)
        self._cmp_page.setValue(1)
        page_row = QHBoxLayout()
        page_row.addWidget(QLabel("Page:"))
        page_row.addWidget(self._cmp_page)
        page_row.addStretch()
        lay.addLayout(page_row)

        btn_compare = QPushButton("Compare")
        btn_compare.clicked.connect(self._compare_pdfs)
        lay.addWidget(btn_compare)

        # Side-by-side display
        self._cmp_splitter = QSplitter(Qt.Horizontal)
        self._cmp_label1 = QLabel("PDF 1")
        self._cmp_label1.setAlignment(Qt.AlignCenter)
        self._cmp_label1.setStyleSheet("border: 1px solid #999; background: white;")
        self._cmp_label1.setMinimumHeight(300)
        self._cmp_label2 = QLabel("PDF 2")
        self._cmp_label2.setAlignment(Qt.AlignCenter)
        self._cmp_label2.setStyleSheet("border: 1px solid #999; background: white;")
        self._cmp_label2.setMinimumHeight(300)
        self._cmp_splitter.addWidget(self._cmp_label1)
        self._cmp_splitter.addWidget(self._cmp_label2)
        lay.addWidget(self._cmp_splitter, 1)
        return w

    def _browse_cmp_file(self, line_edit):
        path, _ = QFileDialog.getOpenFileName(self, "Select PDF", "", "PDF Files (*.pdf)")
        if path:
            line_edit.setText(path)

    def _compare_pdfs(self):
        if not HAS_FITZ:
            QMessageBox.information(
                self, "Missing Dependency",
                "PDF comparison requires PyMuPDF.\n\nInstall with: pip install PyMuPDF"
            )
            return

        p1 = self._cmp_path1.text()
        p2 = self._cmp_path2.text()
        if not p1 or not p2:
            QMessageBox.warning(self, "Compare", "Select two PDF files.")
            return

        page_idx = self._cmp_page.value() - 1

        try:
            for path, label in [(p1, self._cmp_label1), (p2, self._cmp_label2)]:
                doc = fitz.open(path)
                if page_idx >= len(doc):
                    label.setText(f"Page {page_idx+1} not found")
                    doc.close()
                    continue
                page = doc[page_idx]
                pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0))
                img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(img).scaledToWidth(
                    400, Qt.SmoothTransformation
                )
                label.setPixmap(pixmap)
                doc.close()

            self._status.setText(f"Comparing page {page_idx+1}")
            self._emit_log(f"PDF comparison: page {page_idx+1}")
        except Exception as exc:
            QMessageBox.critical(self, "Compare Error", str(exc))
            self._emit_log(f"Compare error: {exc}")

    # -- Batch Convert tab -------------------------------------------------

    def _build_batch_convert_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Convert multiple images into a single PDF:"))

        self._batch_list = QListWidget()
        lay.addWidget(self._batch_list)
        self._batch_files = []

        btn_row = QHBoxLayout()
        btn_add = QPushButton("Add Images")
        btn_add.clicked.connect(self._batch_add_images)
        btn_row.addWidget(btn_add)
        btn_remove = QPushButton("Remove Selected")
        btn_remove.clicked.connect(self._batch_remove)
        btn_row.addWidget(btn_remove)
        btn_clear = QPushButton("Clear All")
        btn_clear.clicked.connect(self._batch_clear)
        btn_row.addWidget(btn_clear)
        lay.addLayout(btn_row)

        opt_row = QHBoxLayout()
        opt_row.addWidget(QLabel("Page size:"))
        self._batch_page_size = QComboBox()
        self._batch_page_size.addItems(["Fit to Image", "Letter", "A4"])
        opt_row.addWidget(self._batch_page_size)
        opt_row.addStretch()
        lay.addLayout(opt_row)

        btn_convert = QPushButton("Convert to PDF")
        btn_convert.clicked.connect(self._batch_convert)
        lay.addWidget(btn_convert)
        lay.addStretch()
        return w

    def _batch_add_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.gif);;All Files (*)"
        )
        for p in paths:
            self._batch_files.append(p)
            self._batch_list.addItem(os.path.basename(p))

    def _batch_remove(self):
        row = self._batch_list.currentRow()
        if row >= 0:
            self._batch_files.pop(row)
            self._batch_list.takeItem(row)

    def _batch_clear(self):
        self._batch_files.clear()
        self._batch_list.clear()

    def _batch_convert(self):
        if not self._batch_files:
            QMessageBox.warning(self, "Batch Convert", "Add at least one image.")
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF", "", "PDF Files (*.pdf)"
        )
        if not out_path:
            return

        page_mode = self._batch_page_size.currentText()

        if HAS_FITZ:
            try:
                doc = fitz.open()
                for img_path in self._batch_files:
                    img_doc = fitz.open(img_path)
                    if page_mode == "Fit to Image":
                        page = img_doc[0]
                        rect = page.rect
                    elif page_mode == "A4":
                        rect = fitz.paper_rect("a4")
                    else:
                        rect = fitz.paper_rect("letter")

                    pdf_bytes = img_doc.convert_to_pdf()
                    img_doc.close()
                    img_pdf = fitz.open("pdf", pdf_bytes)
                    page = doc.new_page(width=rect.width, height=rect.height)
                    page.show_pdf_page(page.rect, img_pdf, 0)
                    img_pdf.close()

                doc.save(out_path)
                doc.close()
                self._status.setText(f"Converted {len(self._batch_files)} images to PDF")
                self._emit_log(f"Batch convert: {len(self._batch_files)} images -> {out_path}")
            except Exception as exc:
                QMessageBox.critical(self, "Convert Error", str(exc))
                self._emit_log(f"Batch convert error: {exc}")
        elif HAS_REPORTLAB:
            try:
                from reportlab.lib.pagesizes import letter, A4
                from reportlab.pdfgen import canvas as rl_canvas
                from reportlab.lib.utils import ImageReader

                psize = letter if page_mode == "Letter" else A4
                c = rl_canvas.Canvas(out_path, pagesize=psize)
                pw, ph = psize

                for img_path in self._batch_files:
                    try:
                        img = ImageReader(img_path)
                        iw, ih = img.getSize()
                        if page_mode == "Fit to Image":
                            c.setPageSize((iw, ih))
                            c.drawImage(img_path, 0, 0, iw, ih)
                        else:
                            scale = min(pw / iw, ph / ih) * 0.9
                            nw, nh = iw * scale, ih * scale
                            x = (pw - nw) / 2
                            y = (ph - nh) / 2
                            c.drawImage(img_path, x, y, nw, nh)
                        c.showPage()
                    except Exception as img_exc:
                        self._emit_log(f"Skipping {img_path}: {img_exc}")

                c.save()
                self._status.setText(f"Converted {len(self._batch_files)} images to PDF")
                self._emit_log(f"Batch convert: {len(self._batch_files)} images -> {out_path}")
            except Exception as exc:
                QMessageBox.critical(self, "Convert Error", str(exc))
                self._emit_log(f"Batch convert error: {exc}")
        else:
            QMessageBox.information(
                self, "Missing Dependency",
                "Batch image-to-PDF requires PyMuPDF or reportlab.\n\n"
                "Install with: pip install PyMuPDF  or  pip install reportlab"
            )

    # -- file actions ------------------------------------------------------

    def _action_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open PDF", "", "PDF Files (*.pdf);;All Files (*)"
        )
        if path:
            self.load_file(path)

    # -- public API --------------------------------------------------------

    def load_file(self, path: str):
        """Load a PDF file for inspection and operations."""
        if not os.path.isfile(path):
            self._status.setText(f"File not found: {path}")
            self._emit_log(f"File not found: {path}")
            return
        self._current_path = path
        self._status.setText(f"Loaded: {os.path.basename(path)}")
        self._emit_log(f"Loaded PDF: {path}")
        self._refresh_info()
        # Update split spinner max
        if HAS_PYPDF2:
            try:
                reader = PdfReader(path)
                n = len(reader.pages)
                self._split_from.setMaximum(n)
                self._split_to.setMaximum(n)
                self._split_to.setValue(n)
            except Exception:
                pass
        self.contentChanged.emit()

    def export(self, path: str = None):
        """Export extracted text to a file, or trigger PDF creation."""
        text = self._extract_text.toPlainText()
        if not text.strip():
            self._extract_execute()
            text = self._extract_text.toPlainText()

        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Text", "", "Text Files (*.txt);;All Files (*)"
            )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            self._status.setText(f"Exported: {os.path.basename(path)}")
            self._emit_log(f"Exported text to {path}")
        except Exception as exc:
            self._status.setText(f"Export error: {exc}")
            self._emit_log(f"Export error: {exc}")
