"""
Image Processor Widget for PyQt5 Scientific Suite.
Provides Fiji/ImageJ/Gwyddion-like image processing capabilities including
filters, morphological operations, thresholding, measurements, FFT, and more.
"""

import os
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QToolBar, QAction, QFileDialog,
    QLabel, QSlider, QSpinBox, QDoubleSpinBox, QComboBox, QGroupBox,
    QFormLayout, QSplitter, QTextEdit, QTabWidget, QCheckBox,
    QPushButton, QMessageBox, QInputDialog, QDockWidget, QStatusBar,
    QGridLayout, QScrollArea, QFrame, QMenu, QActionGroup, QSizePolicy,
    QProgressDialog
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon, QCursor

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
from matplotlib.figure import Figure

try:
    from mpl_style import style_figure, style_axes
except ImportError:
    def style_figure(f): pass
    def style_axes(a): pass
import matplotlib.pyplot as plt

from PIL import Image
from scipy import ndimage as ndi

try:
    from skimage import filters as ski_filters
    from skimage import morphology as ski_morphology
    from skimage import measure as ski_measure
    from skimage import exposure as ski_exposure
    from skimage import feature as ski_feature
    from skimage import restoration as ski_restoration
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _to_gray(img):
    """Convert an image array to grayscale."""
    if img is None:
        return None
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(img.dtype)
    return img


def _ensure_uint8(img):
    """Normalise an image to uint8 range."""
    if img.dtype == np.uint8:
        return img
    if img.dtype in (np.float32, np.float64):
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if img.dtype == np.uint16:
        return (img / 256).astype(np.uint8)
    return img.astype(np.uint8)


def _ensure_float(img):
    """Normalise an image to float64 [0, 1]."""
    if img.dtype in (np.float32, np.float64):
        return img.astype(np.float64)
    return img.astype(np.float64) / np.iinfo(img.dtype).max


# ---------------------------------------------------------------------------
# Test pattern generators
# ---------------------------------------------------------------------------

def generate_checkerboard(width=512, height=512, square_size=32):
    """Generate a checkerboard test pattern."""
    img = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            if ((x // square_size) + (y // square_size)) % 2 == 0:
                img[y, x] = 255
    return img


def generate_gradient(width=512, height=512, direction="horizontal"):
    """Generate a smooth gradient test pattern."""
    if direction == "horizontal":
        row = np.linspace(0, 255, width, dtype=np.uint8)
        img = np.tile(row, (height, 1))
    elif direction == "vertical":
        col = np.linspace(0, 255, height, dtype=np.uint8).reshape(-1, 1)
        img = np.tile(col, (1, width))
    elif direction == "radial":
        y, x = np.mgrid[-height // 2:height // 2, -width // 2:width // 2]
        r = np.sqrt(x.astype(float) ** 2 + y.astype(float) ** 2)
        r = r / r.max() * 255
        img = r.astype(np.uint8)
    else:
        img = np.zeros((height, width), dtype=np.uint8)
    return img


def generate_concentric_circles(width=512, height=512, n_rings=20):
    """Generate concentric circles test pattern."""
    y, x = np.mgrid[-height // 2:height // 2, -width // 2:width // 2]
    r = np.sqrt(x.astype(float) ** 2 + y.astype(float) ** 2)
    max_r = min(width, height) / 2
    # Create alternating rings
    ring_width = max_r / n_rings
    img = ((r / ring_width).astype(int) % 2 * 255).astype(np.uint8)
    return img


def generate_siemens_star(width=512, height=512, n_spokes=36):
    """Generate a Siemens star resolution test pattern."""
    y, x = np.mgrid[-height // 2:height // 2, -width // 2:width // 2]
    theta = np.arctan2(y.astype(float), x.astype(float))
    sector = (theta / (2 * np.pi) * n_spokes).astype(int) % 2
    img = (sector * 255).astype(np.uint8)
    return img


def generate_resolution_target(width=512, height=512):
    """Generate a USAF-style resolution target with bar groups at different spacings."""
    img = np.ones((height, width), dtype=np.uint8) * 255
    spacings = [32, 24, 16, 12, 8, 6, 4, 3, 2]
    y_offset = 20
    for sp_idx, spacing in enumerate(spacings):
        x_start = 20
        y_start = y_offset
        bar_height = max(spacing * 5, 10)
        # Horizontal bars
        for i in range(5):
            y_pos = y_start + i * spacing
            if y_pos + spacing // 2 < height:
                img[y_pos:y_pos + spacing // 2, x_start:x_start + bar_height] = 0
        # Vertical bars next to horizontal
        x_vert = x_start + bar_height + 10
        for i in range(5):
            x_pos = x_vert + i * spacing
            if x_pos + spacing // 2 < width:
                img[y_start:y_start + bar_height, x_pos:x_pos + spacing // 2] = 0
        y_offset += bar_height + 20
        if y_offset > height - 50:
            break
    return img


# ---------------------------------------------------------------------------
# Particle / grain analysis
# ---------------------------------------------------------------------------

def detect_particles(binary_img, min_area=10):
    """Detect particles in a binary image. Returns list of dicts with properties."""
    from scipy.ndimage import label, find_objects
    labeled, n_features = label(binary_img > 0)
    particles = []
    for i in range(1, n_features + 1):
        mask = labeled == i
        area = mask.sum()
        if area < min_area:
            continue
        coords = np.argwhere(mask)
        cy, cx = coords.mean(axis=0)
        # Perimeter: count boundary pixels
        eroded = ndi.binary_erosion(mask)
        perimeter = (mask.astype(int) - eroded.astype(int)).sum()
        # Equivalent diameter
        eq_diameter = np.sqrt(4 * area / np.pi)
        # Circularity
        circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0
        particles.append({
            "id": i, "area": int(area), "centroid": (float(cx), float(cy)),
            "perimeter": int(perimeter), "eq_diameter": float(eq_diameter),
            "circularity": float(circularity),
        })
    return particles, labeled


# ---------------------------------------------------------------------------
# Simple translation-based image stitching
# ---------------------------------------------------------------------------

def stitch_images_translation(images):
    """Stitch a list of images using simple translation (horizontal strip).

    Finds overlap between consecutive images using cross-correlation.
    """
    if not images:
        return None
    if len(images) == 1:
        return images[0].copy()
    result = _ensure_uint8(images[0])
    for i in range(1, len(images)):
        next_img = _ensure_uint8(images[i])
        result = _stitch_pair(result, next_img)
    return result


def _stitch_pair(img1, img2, overlap_fraction=0.3):
    """Stitch two images with an estimated horizontal overlap."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    gray1 = _to_gray(img1) if img1.ndim == 3 else img1
    gray2 = _to_gray(img2) if img2.ndim == 3 else img2
    # Estimate overlap region
    overlap_w = int(w1 * overlap_fraction)
    if overlap_w < 10:
        overlap_w = min(w1, w2) // 3
    strip1 = gray1[:, -overlap_w:].astype(float)
    strip2 = gray2[:, :overlap_w].astype(float)
    # Cross-correlate to find best vertical and horizontal shift
    min_h = min(strip1.shape[0], strip2.shape[0])
    strip1 = strip1[:min_h, :]
    strip2 = strip2[:min_h, :]
    # Normalized cross-correlation for horizontal offset
    best_shift = 0
    best_corr = -1
    for shift in range(-overlap_w // 2, overlap_w // 2):
        if shift >= 0:
            s1 = strip1[:, shift:]
            s2 = strip2[:, :s1.shape[1]]
        else:
            s2 = strip2[:, -shift:]
            s1 = strip1[:, :s2.shape[1]]
        if s1.size == 0 or s2.size == 0:
            continue
        corr = np.sum(s1 * s2) / (np.sqrt(np.sum(s1 ** 2) * np.sum(s2 ** 2)) + 1e-12)
        if corr > best_corr:
            best_corr = corr
            best_shift = shift
    # Compute final offset
    x_offset = w1 - overlap_w + best_shift
    out_h = max(h1, h2)
    out_w = x_offset + w2
    if img1.ndim == 3:
        out = np.zeros((out_h, out_w, img1.shape[2]), dtype=np.uint8)
        out[:h1, :w1, :] = img1
        out[:h2, x_offset:x_offset + w2, :] = img2
    else:
        out = np.zeros((out_h, out_w), dtype=np.uint8)
        out[:h1, :w1] = img1
        out[:h2, x_offset:x_offset + w2] = img2
    return out


# ---------------------------------------------------------------------------
# Main Widget
# ---------------------------------------------------------------------------

class ImageProcessorWidget(QWidget):
    """Full-featured image processing widget with matplotlib canvas."""

    image_loaded = pyqtSignal(str)
    image_processed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._logger = None
        self._original = None       # original loaded image (numpy array)
        self._current = None        # current working image
        self._undo_stack = []
        self._redo_stack = []
        self._line_profile_active = False
        self._pixel_inspect_active = False
        self._line_pts = []
        self._pixels_per_unit = 1.0
        self._calibration_unit = "px"
        self._last_particles = []
        self._annotations = []
        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_logger(self, fn):
        """Set an external logging callback ``fn(message: str)``."""
        self._logger = fn

    def load_file(self, path):
        """Load an image from *path* and display it."""
        if not os.path.isfile(path):
            self._log(f"File not found: {path}")
            return
        try:
            pil_img = Image.open(path)
            arr = np.array(pil_img)
            self._set_image(arr, record_undo=False)
            self._original = arr.copy()
            self._undo_stack.clear()
            self._redo_stack.clear()
            self._update_info_panel()
            self._log(f"Loaded: {path}  ({arr.shape}, {arr.dtype})")
            self.image_loaded.emit(path)
        except Exception as exc:
            self._log(f"Error loading image: {exc}")

    def export(self):
        """Export the current image to a file chosen by the user."""
        if self._current is None:
            self._log("No image to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Image", "",
            "PNG (*.png);;TIFF (*.tiff *.tif);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)"
        )
        if path:
            try:
                out = _ensure_uint8(self._current)
                Image.fromarray(out).save(path)
                self._log(f"Exported to {path}")
            except Exception as exc:
                self._log(f"Export error: {exc}")

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)

        # Toolbar
        self._toolbar = self._make_toolbar()
        root.addWidget(self._toolbar)

        # Main splitter: canvas | side panel
        splitter = QSplitter(Qt.Horizontal)

        # --- Canvas area ---
        canvas_widget = QWidget()
        canvas_layout = QVBoxLayout(canvas_widget)
        canvas_layout.setContentsMargins(0, 0, 0, 0)

        self._figure = Figure(figsize=(6, 5), dpi=100)
        style_figure(self._figure)
        self._canvas = FigureCanvas(self._figure)
        self._ax = self._figure.add_subplot(111)
        self._ax.set_axis_off()
        self._nav_toolbar = NavigationToolbar(self._canvas, self)
        canvas_layout.addWidget(self._nav_toolbar)
        canvas_layout.addWidget(self._canvas)

        self._canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self._canvas.mpl_connect("motion_notify_event", self._on_canvas_move)

        splitter.addWidget(canvas_widget)

        # --- Side panel ---
        side = QTabWidget()
        side.setMaximumWidth(320)
        side.setMinimumWidth(220)

        # Filters tab
        side.addTab(self._make_filters_tab(), "Filters")
        # Morphology tab
        side.addTab(self._make_morphology_tab(), "Morph")
        # Threshold tab
        side.addTab(self._make_threshold_tab(), "Threshold")
        # Color tab
        side.addTab(self._make_color_tab(), "Color")
        # Measurements tab
        side.addTab(self._make_measurements_tab(), "Measure")
        # Generation tab
        side.addTab(self._make_generation_tab(), "Generate")
        # Analysis tab (particle/grain + calibration)
        side.addTab(self._make_analysis_tab(), "Analysis")
        # Annotation tab
        side.addTab(self._make_annotation_tab(), "Annotate")
        # Info tab
        side.addTab(self._make_info_tab(), "Info")

        splitter.addWidget(side)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        root.addWidget(splitter, stretch=1)

        # Status bar
        self._status = QLabel("Ready")
        self._status.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        root.addWidget(self._status)

    # ------------------------------------------------------------------
    # Toolbar
    # ------------------------------------------------------------------

    def _make_toolbar(self):
        tb = QToolBar("Image Tools")
        tb.setMovable(False)

        act_open = QAction("Open", self)
        act_open.setToolTip("Open image file")
        act_open.triggered.connect(self._action_open)
        tb.addAction(act_open)

        act_export = QAction("Export", self)
        act_export.triggered.connect(self.export)
        tb.addAction(act_export)

        tb.addSeparator()

        act_undo = QAction("Undo", self)
        act_undo.setShortcut("Ctrl+Z")
        act_undo.triggered.connect(self._action_undo)
        tb.addAction(act_undo)

        act_redo = QAction("Redo", self)
        act_redo.setShortcut("Ctrl+Y")
        act_redo.triggered.connect(self._action_redo)
        tb.addAction(act_redo)

        tb.addSeparator()

        act_reset = QAction("Reset", self)
        act_reset.setToolTip("Reset to original image")
        act_reset.triggered.connect(self._action_reset)
        tb.addAction(act_reset)

        tb.addSeparator()

        act_hist = QAction("Histogram", self)
        act_hist.triggered.connect(self._show_histogram)
        tb.addAction(act_hist)

        act_fft = QAction("FFT", self)
        act_fft.triggered.connect(self._show_fft)
        tb.addAction(act_fft)

        act_histeq = QAction("Hist Eq", self)
        act_histeq.setToolTip("Histogram equalization")
        act_histeq.triggered.connect(self._apply_histogram_eq)
        tb.addAction(act_histeq)

        tb.addSeparator()

        act_batch = QAction("Batch", self)
        act_batch.setToolTip("Apply filter chain to multiple images")
        act_batch.triggered.connect(self._batch_process)
        tb.addAction(act_batch)

        act_stitch = QAction("Stitch", self)
        act_stitch.setToolTip("Stitch overlapping images")
        act_stitch.triggered.connect(self._stitch_images)
        tb.addAction(act_stitch)

        return tb

    # ------------------------------------------------------------------
    # Side-panel tab builders
    # ------------------------------------------------------------------

    def _make_filters_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        # Gaussian
        grp_gauss = QGroupBox("Gaussian Blur")
        fl = QFormLayout(grp_gauss)
        self._gauss_sigma = QDoubleSpinBox()
        self._gauss_sigma.setRange(0.1, 50.0)
        self._gauss_sigma.setValue(1.0)
        self._gauss_sigma.setSingleStep(0.5)
        fl.addRow("Sigma:", self._gauss_sigma)
        btn = QPushButton("Apply")
        btn.clicked.connect(self._filter_gaussian)
        fl.addRow(btn)
        lay.addWidget(grp_gauss)

        # Median
        grp_med = QGroupBox("Median Filter")
        fl2 = QFormLayout(grp_med)
        self._median_size = QSpinBox()
        self._median_size.setRange(1, 31)
        self._median_size.setValue(3)
        self._median_size.setSingleStep(2)
        fl2.addRow("Size:", self._median_size)
        btn2 = QPushButton("Apply")
        btn2.clicked.connect(self._filter_median)
        fl2.addRow(btn2)
        lay.addWidget(grp_med)

        # Bilateral
        grp_bil = QGroupBox("Bilateral Filter")
        fl3 = QFormLayout(grp_bil)
        self._bilateral_sigma_s = QDoubleSpinBox()
        self._bilateral_sigma_s.setRange(1.0, 100.0)
        self._bilateral_sigma_s.setValue(10.0)
        self._bilateral_sigma_r = QDoubleSpinBox()
        self._bilateral_sigma_r.setRange(0.01, 1.0)
        self._bilateral_sigma_r.setValue(0.1)
        self._bilateral_sigma_r.setSingleStep(0.05)
        fl3.addRow("Sigma spatial:", self._bilateral_sigma_s)
        fl3.addRow("Sigma range:", self._bilateral_sigma_r)
        btn3 = QPushButton("Apply")
        btn3.clicked.connect(self._filter_bilateral)
        fl3.addRow(btn3)
        lay.addWidget(grp_bil)

        # Edge detection combo
        grp_edge = QGroupBox("Edge Detection")
        fl4 = QFormLayout(grp_edge)
        self._edge_combo = QComboBox()
        self._edge_combo.addItems(["Sobel", "Canny", "Laplacian"])
        fl4.addRow("Method:", self._edge_combo)
        self._canny_lo = QDoubleSpinBox()
        self._canny_lo.setRange(0.0, 1.0)
        self._canny_lo.setValue(0.1)
        self._canny_lo.setSingleStep(0.05)
        self._canny_hi = QDoubleSpinBox()
        self._canny_hi.setRange(0.0, 1.0)
        self._canny_hi.setValue(0.3)
        self._canny_hi.setSingleStep(0.05)
        fl4.addRow("Canny low:", self._canny_lo)
        fl4.addRow("Canny high:", self._canny_hi)
        btn4 = QPushButton("Apply")
        btn4.clicked.connect(self._filter_edge)
        fl4.addRow(btn4)
        lay.addWidget(grp_edge)

        # Sharpen
        btn_sharp = QPushButton("Sharpen (Unsharp Mask)")
        btn_sharp.clicked.connect(self._filter_sharpen)
        lay.addWidget(btn_sharp)

        # Denoise
        btn_denoise = QPushButton("Denoise (NL-Means / Gaussian)")
        btn_denoise.clicked.connect(self._filter_denoise)
        lay.addWidget(btn_denoise)

        lay.addStretch()
        return w

    def _make_morphology_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        grp = QGroupBox("Morphological Operations")
        fl = QFormLayout(grp)

        self._morph_op = QComboBox()
        self._morph_op.addItems(["Erosion", "Dilation", "Opening", "Closing"])
        fl.addRow("Operation:", self._morph_op)

        self._morph_elem = QComboBox()
        self._morph_elem.addItems(["Disk", "Square", "Cross"])
        fl.addRow("Element:", self._morph_elem)

        self._morph_radius = QSpinBox()
        self._morph_radius.setRange(1, 25)
        self._morph_radius.setValue(2)
        fl.addRow("Radius/Size:", self._morph_radius)

        self._morph_iters = QSpinBox()
        self._morph_iters.setRange(1, 20)
        self._morph_iters.setValue(1)
        fl.addRow("Iterations:", self._morph_iters)

        btn = QPushButton("Apply")
        btn.clicked.connect(self._apply_morphology)
        fl.addRow(btn)

        lay.addWidget(grp)
        lay.addStretch()
        return w

    def _make_threshold_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        grp = QGroupBox("Thresholding")
        fl = QFormLayout(grp)

        self._thresh_method = QComboBox()
        self._thresh_method.addItems(["Manual", "Otsu", "Adaptive Mean", "Adaptive Gaussian"])
        fl.addRow("Method:", self._thresh_method)

        self._thresh_val = QSlider(Qt.Horizontal)
        self._thresh_val.setRange(0, 255)
        self._thresh_val.setValue(128)
        self._thresh_label = QLabel("128")
        self._thresh_val.valueChanged.connect(
            lambda v: self._thresh_label.setText(str(v))
        )
        fl.addRow("Value:", self._thresh_val)
        fl.addRow("", self._thresh_label)

        self._thresh_block = QSpinBox()
        self._thresh_block.setRange(3, 201)
        self._thresh_block.setValue(11)
        self._thresh_block.setSingleStep(2)
        fl.addRow("Block size (adaptive):", self._thresh_block)

        btn = QPushButton("Apply")
        btn.clicked.connect(self._apply_threshold)
        fl.addRow(btn)

        lay.addWidget(grp)
        lay.addStretch()
        return w

    def _make_color_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        btn_gray = QPushButton("Convert to Grayscale")
        btn_gray.clicked.connect(self._color_grayscale)
        lay.addWidget(btn_gray)

        btn_invert = QPushButton("Invert")
        btn_invert.clicked.connect(self._color_invert)
        lay.addWidget(btn_invert)

        # RGB split
        grp_rgb = QGroupBox("RGB Channel Split")
        rl = QHBoxLayout(grp_rgb)
        for ch, name in enumerate(["Red", "Green", "Blue"]):
            b = QPushButton(name)
            b.clicked.connect(lambda checked, c=ch: self._color_channel(c))
            rl.addWidget(b)
        lay.addWidget(grp_rgb)

        # Brightness / Contrast
        grp_bc = QGroupBox("Brightness / Contrast")
        fl = QFormLayout(grp_bc)
        self._brightness = QSlider(Qt.Horizontal)
        self._brightness.setRange(-100, 100)
        self._brightness.setValue(0)
        self._bright_label = QLabel("0")
        self._brightness.valueChanged.connect(
            lambda v: self._bright_label.setText(str(v))
        )
        fl.addRow("Brightness:", self._brightness)
        fl.addRow("", self._bright_label)

        self._contrast = QDoubleSpinBox()
        self._contrast.setRange(0.1, 5.0)
        self._contrast.setValue(1.0)
        self._contrast.setSingleStep(0.1)
        fl.addRow("Contrast:", self._contrast)

        btn_bc = QPushButton("Apply")
        btn_bc.clicked.connect(self._apply_brightness_contrast)
        fl.addRow(btn_bc)
        lay.addWidget(grp_bc)

        lay.addStretch()
        return w

    def _make_measurements_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        btn_lp = QPushButton("Line Profile (click 2 points)")
        btn_lp.setCheckable(True)
        btn_lp.toggled.connect(self._toggle_line_profile)
        lay.addWidget(btn_lp)
        self._btn_line_profile = btn_lp

        btn_pixel = QPushButton("Pixel Inspector (hover)")
        btn_pixel.setCheckable(True)
        btn_pixel.toggled.connect(self._toggle_pixel_inspector)
        lay.addWidget(btn_pixel)
        self._btn_pixel_inspect = btn_pixel

        btn_region = QPushButton("Region Statistics (whole image)")
        btn_region.clicked.connect(self._region_stats)
        lay.addWidget(btn_region)

        self._measure_output = QTextEdit()
        self._measure_output.setReadOnly(True)
        lay.addWidget(self._measure_output)

        lay.addStretch()
        return w

    def _make_generation_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        grp = QGroupBox("Generate Test Pattern")
        fl = QFormLayout(grp)
        self._gen_type = QComboBox()
        self._gen_type.addItems(["Checkerboard", "Gradient (H)", "Gradient (V)", "Gradient (Radial)",
                                 "Concentric Circles", "Siemens Star", "Resolution Target"])
        fl.addRow("Pattern:", self._gen_type)
        self._gen_width = QSpinBox()
        self._gen_width.setRange(64, 4096)
        self._gen_width.setValue(512)
        fl.addRow("Width:", self._gen_width)
        self._gen_height = QSpinBox()
        self._gen_height.setRange(64, 4096)
        self._gen_height.setValue(512)
        fl.addRow("Height:", self._gen_height)
        self._gen_param = QSpinBox()
        self._gen_param.setRange(2, 200)
        self._gen_param.setValue(32)
        fl.addRow("Param (size/rings):", self._gen_param)
        btn = QPushButton("Generate")
        btn.clicked.connect(self._generate_pattern)
        fl.addRow(btn)
        lay.addWidget(grp)
        lay.addStretch()
        return w

    def _make_analysis_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        # Scale calibration
        grp_cal = QGroupBox("Scale Calibration")
        fl = QFormLayout(grp_cal)
        self._cal_pixels = QDoubleSpinBox()
        self._cal_pixels.setRange(0.001, 100000)
        self._cal_pixels.setValue(100.0)
        self._cal_pixels.setDecimals(2)
        fl.addRow("Known pixels:", self._cal_pixels)
        self._cal_unit_val = QDoubleSpinBox()
        self._cal_unit_val.setRange(0.001, 100000)
        self._cal_unit_val.setValue(1.0)
        self._cal_unit_val.setDecimals(4)
        fl.addRow("Known length:", self._cal_unit_val)
        self._cal_unit = QComboBox()
        self._cal_unit.addItems(["mm", "um", "nm", "cm", "inch"])
        fl.addRow("Unit:", self._cal_unit)
        btn_cal = QPushButton("Set Scale")
        btn_cal.clicked.connect(self._set_scale_calibration)
        fl.addRow(btn_cal)
        lay.addWidget(grp_cal)
        # Particle analysis
        grp_part = QGroupBox("Particle/Grain Analysis")
        pl = QFormLayout(grp_part)
        self._part_min_area = QSpinBox()
        self._part_min_area.setRange(1, 10000)
        self._part_min_area.setValue(10)
        pl.addRow("Min area (px):", self._part_min_area)
        btn_part = QPushButton("Detect Particles")
        btn_part.clicked.connect(self._detect_particles)
        pl.addRow(btn_part)
        btn_hist = QPushButton("Size Distribution")
        btn_hist.clicked.connect(self._particle_histogram)
        pl.addRow(btn_hist)
        lay.addWidget(grp_part)
        # Region measurement
        grp_meas = QGroupBox("Region Measurement")
        ml = QFormLayout(grp_meas)
        btn_regions = QPushButton("Measure Regions (threshold first)")
        btn_regions.clicked.connect(self._measure_regions)
        ml.addRow(btn_regions)
        lay.addWidget(grp_meas)
        self._analysis_output = QTextEdit()
        self._analysis_output.setReadOnly(True)
        lay.addWidget(self._analysis_output)
        lay.addStretch()
        return w

    def _make_annotation_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        grp = QGroupBox("Annotations")
        fl = QFormLayout(grp)
        self._annot_type = QComboBox()
        self._annot_type.addItems(["Text", "Arrow", "Line", "Scale Bar", "Circle", "Rectangle"])
        fl.addRow("Type:", self._annot_type)
        self._annot_text = QComboBox()
        self._annot_text.setEditable(True)
        self._annot_text.addItems(["Sample A", "100 um", "Region of Interest"])
        fl.addRow("Label:", self._annot_text)
        self._annot_color = QComboBox()
        self._annot_color.addItems(["red", "green", "blue", "yellow", "white", "cyan", "magenta"])
        fl.addRow("Color:", self._annot_color)
        self._annot_size = QSpinBox()
        self._annot_size.setRange(1, 100)
        self._annot_size.setValue(12)
        fl.addRow("Font/Line size:", self._annot_size)
        self._annot_x = QSpinBox()
        self._annot_x.setRange(0, 10000)
        self._annot_x.setValue(50)
        fl.addRow("X:", self._annot_x)
        self._annot_y = QSpinBox()
        self._annot_y.setRange(0, 10000)
        self._annot_y.setValue(50)
        fl.addRow("Y:", self._annot_y)
        self._annot_x2 = QSpinBox()
        self._annot_x2.setRange(0, 10000)
        self._annot_x2.setValue(200)
        fl.addRow("X2 (end):", self._annot_x2)
        self._annot_y2 = QSpinBox()
        self._annot_y2.setRange(0, 10000)
        self._annot_y2.setValue(50)
        fl.addRow("Y2 (end):", self._annot_y2)
        btn_add = QPushButton("Add Annotation")
        btn_add.clicked.connect(self._add_annotation)
        fl.addRow(btn_add)
        btn_export = QPushButton("Export Annotated Image")
        btn_export.clicked.connect(self._export_annotated)
        fl.addRow(btn_export)
        lay.addWidget(grp)
        lay.addStretch()
        return w

    def _make_info_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self._info_text = QTextEdit()
        self._info_text.setReadOnly(True)
        lay.addWidget(self._info_text)
        return w

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg):
        if self._logger:
            self._logger(msg)
        self._status.setText(msg)

    def _set_image(self, img, record_undo=True):
        """Set the current working image and refresh the display."""
        if record_undo and self._current is not None:
            self._undo_stack.append(self._current.copy())
            if len(self._undo_stack) > 50:
                self._undo_stack.pop(0)
            self._redo_stack.clear()
        self._current = img
        self._refresh_display()
        self._update_info_panel()
        self.image_processed.emit()

    def _refresh_display(self):
        self._ax.clear()
        self._ax.set_axis_off()
        if self._current is not None:
            if self._current.ndim == 2:
                self._ax.imshow(self._current, cmap="gray", aspect="equal")
            else:
                disp = _ensure_uint8(self._current)
                self._ax.imshow(disp, aspect="equal")
        self._canvas.draw_idle()

    def _update_info_panel(self):
        if self._current is None:
            self._info_text.setPlainText("No image loaded.")
            return
        img = self._current
        lines = [
            f"Dimensions : {img.shape[1]} x {img.shape[0]} px",
            f"Channels   : {img.shape[2] if img.ndim == 3 else 1}",
            f"Dtype      : {img.dtype}",
            f"Min value  : {img.min()}",
            f"Max value  : {img.max()}",
            f"Mean       : {img.mean():.2f}",
            f"Std dev    : {img.std():.2f}",
            f"Size (bytes): {img.nbytes}",
        ]
        self._info_text.setPlainText("\n".join(lines))

    def _require_image(self):
        if self._current is None:
            self._log("No image loaded.")
            return False
        return True

    def _gray_copy(self):
        """Return a grayscale float64 copy of the current image."""
        return _ensure_float(_to_gray(self._current))

    # ------------------------------------------------------------------
    # Toolbar actions
    # ------------------------------------------------------------------

    def _action_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.tiff *.tif *.bmp);;All Files (*)"
        )
        if path:
            self.load_file(path)

    def _action_undo(self):
        if not self._undo_stack:
            self._log("Nothing to undo.")
            return
        self._redo_stack.append(self._current.copy())
        self._current = self._undo_stack.pop()
        self._refresh_display()
        self._update_info_panel()
        self._log("Undo")

    def _action_redo(self):
        if not self._redo_stack:
            self._log("Nothing to redo.")
            return
        self._undo_stack.append(self._current.copy())
        self._current = self._redo_stack.pop()
        self._refresh_display()
        self._update_info_panel()
        self._log("Redo")

    def _action_reset(self):
        if self._original is None:
            return
        self._set_image(self._original.copy())
        self._log("Reset to original")

    # ------------------------------------------------------------------
    # Histogram & FFT
    # ------------------------------------------------------------------

    def _show_histogram(self):
        if not self._require_image():
            return
        fig, axes = plt.subplots(1, 1, figsize=(6, 4))
        fig.canvas.manager.set_window_title("Histogram")
        img = self._current
        if img.ndim == 2:
            axes.hist(img.ravel(), bins=256, range=(0, 255), color="gray", alpha=0.8)
        else:
            for idx, color in enumerate(["red", "green", "blue"]):
                if idx < img.shape[2]:
                    axes.hist(img[..., idx].ravel(), bins=256, range=(0, 255),
                              color=color, alpha=0.5, label=color)
            axes.legend()
        axes.set_title("Intensity Histogram")
        axes.set_xlabel("Pixel value")
        axes.set_ylabel("Count")
        fig.tight_layout()
        plt.show()

    def _apply_histogram_eq(self):
        if not self._require_image():
            return
        gray = _to_gray(self._current)
        img8 = _ensure_uint8(gray)
        if HAS_SKIMAGE:
            eq = ski_exposure.equalize_hist(img8)
            eq = (eq * 255).astype(np.uint8)
        else:
            hist, bins = np.histogram(img8.ravel(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_m = np.ma.masked_equal(cdf, 0)
            cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
            cdf_final = np.ma.filled(cdf_m, 0).astype(np.uint8)
            eq = cdf_final[img8]
        self._set_image(eq)
        self._log("Histogram equalization applied")

    def _show_fft(self):
        if not self._require_image():
            return
        gray = _to_gray(self._current).astype(np.float64)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log1p(np.abs(f_shift))

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.canvas.manager.set_window_title("2D FFT")
        axes[0].imshow(gray, cmap="gray")
        axes[0].set_title("Original (gray)")
        axes[0].axis("off")
        axes[1].imshow(magnitude, cmap="inferno")
        axes[1].set_title("FFT Magnitude (log)")
        axes[1].axis("off")
        fig.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _filter_gaussian(self):
        if not self._require_image():
            return
        sigma = self._gauss_sigma.value()
        img = self._current.astype(np.float64)
        if img.ndim == 3:
            out = np.stack([ndi.gaussian_filter(img[..., c], sigma)
                            for c in range(img.shape[2])], axis=-1)
        else:
            out = ndi.gaussian_filter(img, sigma)
        self._set_image(_ensure_uint8(out / out.max() if out.max() > 0 else out))
        self._log(f"Gaussian blur (sigma={sigma})")

    def _filter_median(self):
        if not self._require_image():
            return
        size = self._median_size.value()
        if size % 2 == 0:
            size += 1
        img = _ensure_uint8(self._current)
        if img.ndim == 3:
            out = np.stack([ndi.median_filter(img[..., c], size=size)
                            for c in range(img.shape[2])], axis=-1)
        else:
            out = ndi.median_filter(img, size=size)
        self._set_image(out)
        self._log(f"Median filter (size={size})")

    def _filter_bilateral(self):
        if not self._require_image():
            return
        sigma_s = self._bilateral_sigma_s.value()
        sigma_r = self._bilateral_sigma_r.value()
        gray = self._gray_copy()
        if HAS_SKIMAGE:
            from skimage.restoration import denoise_bilateral
            out = denoise_bilateral(gray, sigma_color=sigma_r, sigma_spatial=sigma_s)
            self._set_image(_ensure_uint8(out))
        elif HAS_CV2:
            img8 = _ensure_uint8(_to_gray(self._current))
            out = cv2.bilateralFilter(img8, d=int(sigma_s), sigmaColor=sigma_r * 255,
                                      sigmaSpace=sigma_s)
            self._set_image(out)
        else:
            # Approximate with Gaussian as fallback
            out = ndi.gaussian_filter(gray, sigma=sigma_s)
            self._set_image(_ensure_uint8(out))
            self._log("Bilateral unavailable; used Gaussian fallback")
            return
        self._log(f"Bilateral filter (spatial={sigma_s}, range={sigma_r})")

    def _filter_edge(self):
        if not self._require_image():
            return
        method = self._edge_combo.currentText()
        gray = self._gray_copy()
        if method == "Sobel":
            if HAS_SKIMAGE:
                out = ski_filters.sobel(gray)
            else:
                sx = ndi.sobel(gray, axis=0)
                sy = ndi.sobel(gray, axis=1)
                out = np.hypot(sx, sy)
            out = (out / out.max() * 255).astype(np.uint8) if out.max() > 0 else out.astype(np.uint8)
        elif method == "Canny":
            lo = self._canny_lo.value()
            hi = self._canny_hi.value()
            if HAS_SKIMAGE:
                out = ski_feature.canny(gray, sigma=1.0, low_threshold=lo,
                                        high_threshold=hi).astype(np.uint8) * 255
            elif HAS_CV2:
                img8 = _ensure_uint8(_to_gray(self._current))
                out = cv2.Canny(img8, int(lo * 255), int(hi * 255))
            else:
                sx = ndi.sobel(gray, axis=0)
                sy = ndi.sobel(gray, axis=1)
                mag = np.hypot(sx, sy)
                out = (mag > hi).astype(np.uint8) * 255
        elif method == "Laplacian":
            out = ndi.laplace(gray)
            out = np.abs(out)
            out = (out / out.max() * 255).astype(np.uint8) if out.max() > 0 else out.astype(np.uint8)
        else:
            return
        self._set_image(out)
        self._log(f"Edge detection: {method}")

    def _filter_sharpen(self):
        if not self._require_image():
            return
        img = self._current.astype(np.float64)
        if img.ndim == 3:
            blurred = np.stack([ndi.gaussian_filter(img[..., c], sigma=1.0)
                                for c in range(img.shape[2])], axis=-1)
        else:
            blurred = ndi.gaussian_filter(img, sigma=1.0)
        sharpened = img + (img - blurred) * 1.5
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        self._set_image(sharpened)
        self._log("Unsharp mask sharpening applied")

    def _filter_denoise(self):
        if not self._require_image():
            return
        gray = self._gray_copy()
        if HAS_SKIMAGE:
            from skimage.restoration import denoise_nl_means, estimate_sigma
            sigma_est = estimate_sigma(gray)
            out = denoise_nl_means(gray, h=1.15 * sigma_est, fast_mode=True,
                                   patch_size=5, patch_distance=6)
            self._set_image(_ensure_uint8(out))
            self._log("NL-Means denoising applied")
        else:
            out = ndi.gaussian_filter(gray, sigma=1.5)
            self._set_image(_ensure_uint8(out))
            self._log("Denoise (Gaussian fallback, sigma=1.5)")

    # ------------------------------------------------------------------
    # Morphological operations
    # ------------------------------------------------------------------

    def _get_structuring_element(self):
        elem = self._morph_elem.currentText()
        r = self._morph_radius.value()
        if HAS_SKIMAGE:
            if elem == "Disk":
                return ski_morphology.disk(r)
            elif elem == "Square":
                return ski_morphology.square(2 * r + 1)
            else:
                return ski_morphology.diamond(r)
        else:
            size = 2 * r + 1
            if elem == "Disk":
                y, x = np.ogrid[-r:r + 1, -r:r + 1]
                return (x * x + y * y <= r * r).astype(np.uint8)
            elif elem == "Square":
                return np.ones((size, size), dtype=np.uint8)
            else:
                se = np.zeros((size, size), dtype=np.uint8)
                se[r, :] = 1
                se[:, r] = 1
                return se

    def _apply_morphology(self):
        if not self._require_image():
            return
        op = self._morph_op.currentText()
        iters = self._morph_iters.value()
        selem = self._get_structuring_element()
        gray = _ensure_uint8(_to_gray(self._current))

        ops = {
            "Erosion": ndi.grey_erosion,
            "Dilation": ndi.grey_dilation,
        }

        if op in ("Erosion", "Dilation"):
            result = gray.copy()
            func = ops[op]
            for _ in range(iters):
                result = func(result, footprint=selem)
        elif op == "Opening":
            result = gray.copy()
            for _ in range(iters):
                result = ndi.grey_erosion(result, footprint=selem)
                result = ndi.grey_dilation(result, footprint=selem)
        elif op == "Closing":
            result = gray.copy()
            for _ in range(iters):
                result = ndi.grey_dilation(result, footprint=selem)
                result = ndi.grey_erosion(result, footprint=selem)
        else:
            return

        self._set_image(result)
        self._log(f"Morphology: {op} (r={self._morph_radius.value()}, iters={iters})")

    # ------------------------------------------------------------------
    # Thresholding
    # ------------------------------------------------------------------

    def _apply_threshold(self):
        if not self._require_image():
            return
        method = self._thresh_method.currentText()
        gray = _ensure_uint8(_to_gray(self._current))

        if method == "Manual":
            val = self._thresh_val.value()
            result = (gray > val).astype(np.uint8) * 255
        elif method == "Otsu":
            if HAS_SKIMAGE:
                val = ski_filters.threshold_otsu(gray)
            else:
                hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
                total = gray.size
                sum_total = np.dot(np.arange(256), hist)
                sum_bg = 0.0
                w_bg = 0
                max_var = 0.0
                val = 0
                for t in range(256):
                    w_bg += hist[t]
                    if w_bg == 0:
                        continue
                    w_fg = total - w_bg
                    if w_fg == 0:
                        break
                    sum_bg += t * hist[t]
                    mean_bg = sum_bg / w_bg
                    mean_fg = (sum_total - sum_bg) / w_fg
                    var = w_bg * w_fg * (mean_bg - mean_fg) ** 2
                    if var > max_var:
                        max_var = var
                        val = t
            result = (gray > val).astype(np.uint8) * 255
            self._log(f"Otsu threshold = {val}")
        elif method in ("Adaptive Mean", "Adaptive Gaussian"):
            block = self._thresh_block.value()
            if block % 2 == 0:
                block += 1
            if HAS_CV2:
                adapt_method = (cv2.ADAPTIVE_THRESH_MEAN_C if "Mean" in method
                                else cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
                result = cv2.adaptiveThreshold(gray, 255, adapt_method,
                                               cv2.THRESH_BINARY, block, 2)
            else:
                # Manual adaptive: local mean threshold
                from scipy.ndimage import uniform_filter, gaussian_filter
                if "Mean" in method:
                    local_mean = uniform_filter(gray.astype(np.float64), size=block)
                else:
                    local_mean = gaussian_filter(gray.astype(np.float64), sigma=block / 6.0)
                result = (gray > local_mean - 2).astype(np.uint8) * 255
        else:
            return

        self._set_image(result)
        self._log(f"Threshold: {method}")

    # ------------------------------------------------------------------
    # Color operations
    # ------------------------------------------------------------------

    def _color_grayscale(self):
        if not self._require_image():
            return
        self._set_image(_ensure_uint8(_to_gray(self._current)))
        self._log("Converted to grayscale")

    def _color_invert(self):
        if not self._require_image():
            return
        img = _ensure_uint8(self._current)
        self._set_image(255 - img)
        self._log("Inverted")

    def _color_channel(self, ch):
        if not self._require_image():
            return
        img = _ensure_uint8(self._current)
        if img.ndim != 3 or img.shape[2] < 3:
            self._log("Image does not have RGB channels.")
            return
        channel_img = img[..., ch]
        names = {0: "Red", 1: "Green", 2: "Blue"}
        self._set_image(channel_img)
        self._log(f"Extracted {names.get(ch, '?')} channel")

    def _apply_brightness_contrast(self):
        if not self._require_image():
            return
        if self._original is None:
            return
        brightness = self._brightness.value()
        contrast = self._contrast.value()
        img = self._original.astype(np.float64)
        result = contrast * img + brightness
        result = np.clip(result, 0, 255).astype(np.uint8)
        self._set_image(result)
        self._log(f"Brightness={brightness}, Contrast={contrast:.2f}")

    # ------------------------------------------------------------------
    # Measurements
    # ------------------------------------------------------------------

    def _toggle_line_profile(self, checked):
        self._line_profile_active = checked
        self._line_pts.clear()
        if checked:
            self._pixel_inspect_active = False
            self._btn_pixel_inspect.setChecked(False)
            self._log("Line profile: click two points on the image")
        else:
            self._log("Line profile deactivated")

    def _toggle_pixel_inspector(self, checked):
        self._pixel_inspect_active = checked
        if checked:
            self._line_profile_active = False
            self._btn_line_profile.setChecked(False)
            self._log("Pixel inspector active - hover over image")
        else:
            self._log("Pixel inspector deactivated")

    def _on_canvas_click(self, event):
        if event.inaxes != self._ax or self._current is None:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))

        if self._line_profile_active:
            self._line_pts.append((x, y))
            self._ax.plot(x, y, "ro", markersize=5)
            self._canvas.draw_idle()
            if len(self._line_pts) == 2:
                self._compute_line_profile()
                self._line_pts.clear()

    def _on_canvas_move(self, event):
        if not self._pixel_inspect_active or self._current is None:
            return
        if event.inaxes != self._ax:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        img = self._current
        if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
            val = img[y, x]
            self._measure_output.setPlainText(
                f"Pixel ({x}, {y})\nValue: {val}\n"
                f"Type: {img.dtype}"
            )

    def _compute_line_profile(self):
        (x0, y0), (x1, y1) = self._line_pts
        gray = _to_gray(self._current).astype(np.float64)
        length = int(np.hypot(x1 - x0, y1 - y0))
        if length == 0:
            return
        xs = np.linspace(x0, x1, length)
        ys = np.linspace(y0, y1, length)

        # Bilinear interpolation via map_coordinates
        profile = ndi.map_coordinates(gray, [ys, xs], order=1)

        fig, ax = plt.subplots(figsize=(6, 3))
        fig.canvas.manager.set_window_title("Line Profile")
        ax.plot(profile, "b-")
        ax.set_xlabel("Distance (px)")
        ax.set_ylabel("Intensity")
        ax.set_title(f"Line profile ({x0},{y0}) -> ({x1},{y1})")
        fig.tight_layout()
        plt.show()

        self._measure_output.setPlainText(
            f"Line: ({x0},{y0}) -> ({x1},{y1})\n"
            f"Length: {length} px\n"
            f"Min: {profile.min():.2f}\n"
            f"Max: {profile.max():.2f}\n"
            f"Mean: {profile.mean():.2f}\n"
            f"Std: {profile.std():.2f}"
        )
        self._log("Line profile computed")

    def _region_stats(self):
        if not self._require_image():
            return
        gray = _to_gray(self._current).astype(np.float64)
        text_lines = [
            "--- Region Statistics (full image) ---",
            f"Dimensions : {gray.shape[1]} x {gray.shape[0]}",
            f"Min        : {gray.min():.4f}",
            f"Max        : {gray.max():.4f}",
            f"Mean       : {gray.mean():.4f}",
            f"Median     : {np.median(gray):.4f}",
            f"Std Dev    : {gray.std():.4f}",
            f"Variance   : {gray.var():.4f}",
            f"Sum        : {gray.sum():.1f}",
            f"Total px   : {gray.size}",
            f"RMS        : {np.sqrt(np.mean(gray ** 2)):.4f}",
        ]
        if HAS_SKIMAGE:
            from skimage.measure import shannon_entropy
            text_lines.append(f"Entropy    : {shannon_entropy(gray):.4f}")
        self._measure_output.setPlainText("\n".join(text_lines))
        self._log("Region statistics computed")

    # ------------------------------------------------------------------
    # Image generation
    # ------------------------------------------------------------------

    def _generate_pattern(self):
        """Generate a test pattern image."""
        pattern = self._gen_type.currentText()
        w = self._gen_width.value()
        h = self._gen_height.value()
        param = self._gen_param.value()
        if pattern == "Checkerboard":
            img = generate_checkerboard(w, h, square_size=param)
        elif pattern == "Gradient (H)":
            img = generate_gradient(w, h, direction="horizontal")
        elif pattern == "Gradient (V)":
            img = generate_gradient(w, h, direction="vertical")
        elif pattern == "Gradient (Radial)":
            img = generate_gradient(w, h, direction="radial")
        elif pattern == "Concentric Circles":
            img = generate_concentric_circles(w, h, n_rings=param)
        elif pattern == "Siemens Star":
            img = generate_siemens_star(w, h, n_spokes=param)
        elif pattern == "Resolution Target":
            img = generate_resolution_target(w, h)
        else:
            return
        self._set_image(img, record_undo=False)
        self._original = img.copy()
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._update_info_panel()
        self._log(f"Generated {pattern} ({w}x{h})")

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def _batch_process(self):
        """Apply current filter chain to multiple images and save results."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images for Batch Processing", "",
            "Images (*.png *.jpg *.jpeg *.tiff *.tif *.bmp);;All Files (*)")
        if not paths:
            return
        # Ask which filter to apply
        filters = ["Gaussian Blur", "Median Filter", "Sharpen", "Grayscale",
                    "Invert", "Histogram Eq", "Edge (Sobel)"]
        chosen, ok = QInputDialog.getItem(self, "Batch Filter", "Filter to apply:", filters, 0, False)
        if not ok:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Output Directory")
        if not out_dir:
            return
        count = 0
        for p in paths:
            try:
                pil_img = Image.open(p)
                arr = np.array(pil_img)
                processed = self._apply_batch_filter(arr, chosen)
                if processed is not None:
                    out_path = os.path.join(out_dir, "batch_" + os.path.basename(p))
                    Image.fromarray(_ensure_uint8(processed)).save(out_path)
                    count += 1
            except Exception as exc:
                self._log(f"Batch error on {p}: {exc}")
        self._log(f"Batch processed {count}/{len(paths)} images to {out_dir}")

    def _apply_batch_filter(self, img, filter_name):
        """Apply a named filter to an image array and return result."""
        if filter_name == "Gaussian Blur":
            sigma = self._gauss_sigma.value()
            fimg = img.astype(np.float64)
            if fimg.ndim == 3:
                out = np.stack([ndi.gaussian_filter(fimg[..., c], sigma)
                                for c in range(fimg.shape[2])], axis=-1)
            else:
                out = ndi.gaussian_filter(fimg, sigma)
            return _ensure_uint8(out / (out.max() + 1e-12))
        elif filter_name == "Median Filter":
            size = self._median_size.value()
            if size % 2 == 0:
                size += 1
            img8 = _ensure_uint8(img)
            if img8.ndim == 3:
                return np.stack([ndi.median_filter(img8[..., c], size=size)
                                 for c in range(img8.shape[2])], axis=-1)
            return ndi.median_filter(img8, size=size)
        elif filter_name == "Sharpen":
            fimg = img.astype(np.float64)
            if fimg.ndim == 3:
                blurred = np.stack([ndi.gaussian_filter(fimg[..., c], 1.0)
                                    for c in range(fimg.shape[2])], axis=-1)
            else:
                blurred = ndi.gaussian_filter(fimg, 1.0)
            return np.clip(fimg + (fimg - blurred) * 1.5, 0, 255).astype(np.uint8)
        elif filter_name == "Grayscale":
            return _ensure_uint8(_to_gray(img))
        elif filter_name == "Invert":
            return 255 - _ensure_uint8(img)
        elif filter_name == "Histogram Eq":
            gray = _ensure_uint8(_to_gray(img))
            if HAS_SKIMAGE:
                eq = ski_exposure.equalize_hist(gray)
                return (eq * 255).astype(np.uint8)
            hist, _ = np.histogram(gray.ravel(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_m = np.ma.masked_equal(cdf, 0)
            cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
            return np.ma.filled(cdf_m, 0).astype(np.uint8)[gray]
        elif filter_name == "Edge (Sobel)":
            gray = _ensure_float(_to_gray(img))
            sx = ndi.sobel(gray, axis=0)
            sy = ndi.sobel(gray, axis=1)
            out = np.hypot(sx, sy)
            return (out / (out.max() + 1e-12) * 255).astype(np.uint8)
        return img

    # ------------------------------------------------------------------
    # Scale calibration and region measurement
    # ------------------------------------------------------------------

    def _set_scale_calibration(self):
        """Set pixels-per-unit scale calibration."""
        pixels = self._cal_pixels.value()
        length = self._cal_unit_val.value()
        unit = self._cal_unit.currentText()
        self._pixels_per_unit = pixels / length
        self._calibration_unit = unit
        self._log(f"Scale set: {self._pixels_per_unit:.4f} px/{unit}")
        if hasattr(self, '_analysis_output'):
            self._analysis_output.setPlainText(
                f"Scale calibration set:\n"
                f"  {pixels:.2f} pixels = {length:.4f} {unit}\n"
                f"  {self._pixels_per_unit:.4f} px/{unit}")

    def _measure_regions(self):
        """Measure area and perimeter of segmented regions in current image."""
        if not self._require_image():
            return
        gray = _ensure_uint8(_to_gray(self._current))
        # Threshold if not already binary
        if gray.max() > 1:
            thresh = 128
            binary = gray > thresh
        else:
            binary = gray > 0
        from scipy.ndimage import label
        labeled, n = label(binary)
        ppu = self._pixels_per_unit
        unit = self._calibration_unit
        lines = [f"Found {n} regions (scale: {ppu:.2f} px/{unit})", ""]
        for i in range(1, min(n + 1, 101)):  # limit to 100 regions
            mask = labeled == i
            area_px = mask.sum()
            area_real = area_px / (ppu ** 2)
            eroded = ndi.binary_erosion(mask)
            perim_px = (mask.astype(int) - eroded.astype(int)).sum()
            perim_real = perim_px / ppu
            lines.append(f"Region {i}: area={area_real:.4f} {unit}^2, perim={perim_real:.4f} {unit}")
        self._analysis_output.setPlainText("\n".join(lines))
        self._log(f"Measured {n} regions")

    # ------------------------------------------------------------------
    # Particle / grain analysis
    # ------------------------------------------------------------------

    def _detect_particles(self):
        """Detect and count particles in the current image."""
        if not self._require_image():
            return
        gray = _ensure_uint8(_to_gray(self._current))
        # Auto-threshold
        if HAS_SKIMAGE:
            thresh = ski_filters.threshold_otsu(gray)
        else:
            thresh = 128
        binary = gray > thresh
        min_area = self._part_min_area.value()
        particles, labeled = detect_particles(binary, min_area=min_area)
        self._last_particles = particles
        # Show detected particles on image
        overlay = np.stack([gray, gray, gray], axis=-1) if gray.ndim == 2 else _ensure_uint8(self._current).copy()
        # Mark centroids
        for p in particles:
            cx, cy = int(p["centroid"][0]), int(p["centroid"][1])
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < overlay.shape[0] and 0 <= nx < overlay.shape[1]:
                        overlay[ny, nx] = [255, 0, 0]
        self._set_image(overlay)
        ppu = self._pixels_per_unit
        unit = self._calibration_unit
        lines = [f"Detected {len(particles)} particles (min_area={min_area} px)", ""]
        for p in particles[:50]:
            area_real = p["area"] / (ppu ** 2)
            lines.append(f"ID {p['id']}: area={area_real:.3f} {unit}^2, "
                         f"d_eq={p['eq_diameter'] / ppu:.3f} {unit}, "
                         f"circ={p['circularity']:.3f}")
        if len(particles) > 50:
            lines.append(f"... and {len(particles) - 50} more")
        self._analysis_output.setPlainText("\n".join(lines))
        self._log(f"Detected {len(particles)} particles")

    def _particle_histogram(self):
        """Show size distribution histogram of last detected particles."""
        if not self._last_particles:
            self._log("Run particle detection first.")
            return
        ppu = self._pixels_per_unit
        unit = self._calibration_unit
        diameters = [p["eq_diameter"] / ppu for p in self._last_particles]
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.canvas.manager.set_window_title("Particle Size Distribution")
        ax.hist(diameters, bins=min(30, max(5, len(diameters) // 3)), color="#4488cc", alpha=0.8, edgecolor="black")
        ax.set_xlabel(f"Equivalent Diameter ({unit})")
        ax.set_ylabel("Count")
        ax.set_title(f"Size Distribution (N={len(diameters)})")
        fig.tight_layout()
        plt.show()
        self._log("Particle size histogram displayed")

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def _batch_process(self):
        """Apply the currently selected filter to multiple images and save."""
        from PyQt5.QtWidgets import QApplication

        # Select input images
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images for Batch Processing", "",
            "Images (*.png *.jpg *.jpeg *.tiff *.tif *.bmp);;All Files (*)")
        if not paths:
            return

        # Select output directory
        out_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory")
        if not out_dir:
            return

        # Ask which filter to apply
        filters = ["Gaussian Blur", "Median Filter", "Sharpen",
                    "Edge Detection (Sobel)", "Denoise", "Histogram Equalization",
                    "Grayscale", "Invert"]
        chosen, ok = QInputDialog.getItem(
            self, "Batch Filter", "Select filter to apply:", filters, 0, False)
        if not ok:
            return

        # Set up progress dialog
        progress = QProgressDialog(
            "Processing images...", "Cancel", 0, len(paths), self)
        progress.setWindowTitle("Batch Processing")
        progress.setMinimumDuration(0)
        progress.setValue(0)

        processed = 0
        errors = 0

        for i, path in enumerate(paths):
            if progress.wasCanceled():
                self._log(f"Batch processing cancelled after {processed} images.")
                break

            progress.setValue(i)
            progress.setLabelText(
                f"Processing {os.path.basename(path)} ({i + 1}/{len(paths)})")
            QApplication.processEvents()

            try:
                img = np.array(Image.open(path))
                out = self._apply_batch_filter(img, chosen)
                if out is not None:
                    base = os.path.splitext(os.path.basename(path))[0]
                    out_path = os.path.join(out_dir, f"{base}_processed.png")
                    Image.fromarray(_ensure_uint8(out)).save(out_path)
                    processed += 1
                else:
                    errors += 1
                    self._log(
                        f"Filter returned None for {os.path.basename(path)}")
            except Exception as exc:
                errors += 1
                self._log(f"Error processing {os.path.basename(path)}: {exc}")

        progress.setValue(len(paths))
        self._log(
            f"Batch complete: {processed} processed, {errors} errors, "
            f"output in {out_dir}")
        QMessageBox.information(
            self, "Batch Processing Complete",
            f"Processed {processed} of {len(paths)} images.\n"
            f"Errors: {errors}\nOutput directory: {out_dir}")

    def _apply_batch_filter(self, img, filter_name):
        """Apply a named filter to a single image array and return result."""
        if filter_name == "Gaussian Blur":
            sigma = self._gauss_sigma.value()
            img_f = img.astype(np.float64)
            if img_f.ndim == 3:
                out = np.stack(
                    [ndi.gaussian_filter(img_f[..., c], sigma)
                     for c in range(img_f.shape[2])], axis=-1)
            else:
                out = ndi.gaussian_filter(img_f, sigma)
            mx = out.max()
            return _ensure_uint8(out / mx if mx > 0 else out)

        elif filter_name == "Median Filter":
            size = self._median_size.value()
            if size % 2 == 0:
                size += 1
            img8 = _ensure_uint8(img)
            if img8.ndim == 3:
                return np.stack(
                    [ndi.median_filter(img8[..., c], size=size)
                     for c in range(img8.shape[2])], axis=-1)
            else:
                return ndi.median_filter(img8, size=size)

        elif filter_name == "Sharpen":
            img_f = img.astype(np.float64)
            if img_f.ndim == 3:
                blurred = np.stack(
                    [ndi.gaussian_filter(img_f[..., c], sigma=1.0)
                     for c in range(img_f.shape[2])], axis=-1)
            else:
                blurred = ndi.gaussian_filter(img_f, sigma=1.0)
            sharpened = img_f + (img_f - blurred) * 1.5
            return np.clip(sharpened, 0, 255).astype(np.uint8)

        elif filter_name == "Edge Detection (Sobel)":
            gray = _ensure_float(_to_gray(img))
            sx = ndi.sobel(gray, axis=0)
            sy = ndi.sobel(gray, axis=1)
            out = np.hypot(sx, sy)
            mx = out.max()
            return ((out / mx * 255).astype(np.uint8)
                    if mx > 0 else out.astype(np.uint8))

        elif filter_name == "Denoise":
            gray = _ensure_float(_to_gray(img))
            out = ndi.gaussian_filter(gray, sigma=1.5)
            return _ensure_uint8(out)

        elif filter_name == "Histogram Equalization":
            gray = _to_gray(img)
            img8 = _ensure_uint8(gray)
            hist, bins = np.histogram(img8.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_m = np.ma.masked_equal(cdf, 0)
            cdf_m = ((cdf_m - cdf_m.min()) * 255
                     / (cdf_m.max() - cdf_m.min()))
            cdf_final = np.ma.filled(cdf_m, 0).astype(np.uint8)
            return cdf_final[img8]

        elif filter_name == "Grayscale":
            return _ensure_uint8(_to_gray(img))

        elif filter_name == "Invert":
            img8 = _ensure_uint8(img)
            return 255 - img8

        return None

    # ------------------------------------------------------------------
    # Image stitching
    # ------------------------------------------------------------------

    def _stitch_images(self):
        """Load multiple overlapping images and stitch them."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Overlapping Images (in order)", "",
            "Images (*.png *.jpg *.jpeg *.tiff *.tif *.bmp);;All Files (*)")
        if len(paths) < 2:
            self._log("Need at least 2 images for stitching.")
            return
        images = []
        for p in paths:
            try:
                arr = np.array(Image.open(p))
                images.append(arr)
            except Exception as exc:
                self._log(f"Error loading {p}: {exc}")
                return
        result = stitch_images_translation(images)
        if result is not None:
            self._set_image(result, record_undo=False)
            self._original = result.copy()
            self._undo_stack.clear()
            self._redo_stack.clear()
            self._update_info_panel()
            self._log(f"Stitched {len(images)} images ({result.shape[1]}x{result.shape[0]})")
        else:
            self._log("Stitching failed.")

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------

    def _add_annotation(self):
        """Add an annotation to the current image display."""
        if not self._require_image():
            return
        annot = {
            "type": self._annot_type.currentText(),
            "text": self._annot_text.currentText(),
            "color": self._annot_color.currentText(),
            "size": self._annot_size.value(),
            "x": self._annot_x.value(),
            "y": self._annot_y.value(),
            "x2": self._annot_x2.value(),
            "y2": self._annot_y2.value(),
        }
        self._annotations.append(annot)
        self._draw_annotations()
        self._log(f"Added {annot['type']} annotation")

    def _draw_annotations(self):
        """Redraw the image with all annotations overlaid."""
        self._ax.clear()
        self._ax.set_axis_off()
        if self._current is not None:
            if self._current.ndim == 2:
                self._ax.imshow(self._current, cmap="gray", aspect="equal")
            else:
                self._ax.imshow(_ensure_uint8(self._current), aspect="equal")
        for a in self._annotations:
            color = a["color"]
            sz = a["size"]
            x, y = a["x"], a["y"]
            x2, y2 = a["x2"], a["y2"]
            if a["type"] == "Text":
                self._ax.text(x, y, a["text"], color=color, fontsize=sz,
                              fontweight="bold", va="top")
            elif a["type"] == "Arrow":
                self._ax.annotate("", xy=(x2, y2), xytext=(x, y),
                                  arrowprops=dict(arrowstyle="->", color=color, lw=sz / 4))
                if a["text"]:
                    self._ax.text(x, y - 10, a["text"], color=color, fontsize=max(8, sz))
            elif a["type"] == "Line":
                self._ax.plot([x, x2], [y, y2], color=color, linewidth=sz / 4)
            elif a["type"] == "Scale Bar":
                self._ax.plot([x, x2], [y, y], color=color, linewidth=max(2, sz / 3))
                self._ax.plot([x, x], [y - 3, y + 3], color=color, linewidth=max(1, sz / 4))
                self._ax.plot([x2, x2], [y - 3, y + 3], color=color, linewidth=max(1, sz / 4))
                bar_len_px = abs(x2 - x)
                ppu = self._pixels_per_unit
                unit = self._calibration_unit
                real_len = bar_len_px / ppu
                label = a["text"] if a["text"] else f"{real_len:.1f} {unit}"
                self._ax.text((x + x2) / 2, y - 8, label, color=color,
                              fontsize=max(8, sz), ha="center")
            elif a["type"] == "Circle":
                r = np.sqrt((x2 - x) ** 2 + (y2 - y) ** 2)
                circle = plt.Circle((x, y), r, fill=False, edgecolor=color, linewidth=sz / 4)
                self._ax.add_patch(circle)
            elif a["type"] == "Rectangle":
                from matplotlib.patches import Rectangle as MplRect
                w_rect = abs(x2 - x)
                h_rect = abs(y2 - y)
                rect = MplRect((min(x, x2), min(y, y2)), w_rect, h_rect,
                                fill=False, edgecolor=color, linewidth=sz / 4)
                self._ax.add_patch(rect)
        self._canvas.draw_idle()

    def _export_annotated(self):
        """Export the current view with annotations to a file."""
        if self._current is None:
            self._log("No image to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Annotated Image", "",
            "PNG (*.png);;TIFF (*.tiff *.tif);;JPEG (*.jpg *.jpeg)")
        if not path:
            return
        # Use matplotlib savefig to capture annotations
        self._figure.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.05,
                              facecolor=self._figure.get_facecolor())
        self._log(f"Exported annotated image: {path}")
