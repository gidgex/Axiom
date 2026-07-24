"""
gis_module.py - GIS/Mapping Widget for PyQt5 Scientific Suite

Provides a fully functional geographic information system widget with:
- 2D map canvas via matplotlib with continent/country outlines
- Multiple map projections (Equirectangular, Mercator, Mollweide)
- Coordinate systems (WGS84 Lat/Lon, UTM)
- Point plotting with labels, color-coded by value
- Great circle distance calculator
- Coordinate converter (Decimal Degrees <-> DMS)
- CSV data loading and plotting
- Heatmap overlay
- Scale bar and grid lines
- Polygon drawing from coordinate lists
- Export capabilities

No external GIS libraries required - all projections implemented mathematically.
"""

import os
import csv
import math
import traceback
import io

import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QPushButton,
    QComboBox, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QCheckBox, QFileDialog, QGroupBox, QFormLayout,
    QMessageBox, QToolBar, QAction, QSizePolicy, QGridLayout,
    QTextEdit, QColorDialog, QListWidget, QListWidgetItem,
    QAbstractItemView, QDialog, QDialogButtonBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

try:
    from mpl_style import style_figure, style_axes
except ImportError:
    def style_figure(f): pass
    def style_axes(a): pass
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection, PolyCollection


# ---------------------------------------------------------------------------
# Simplified world coastline/continent data (lon, lat pairs)
# Each entry is a list of (lon, lat) coordinate pairs forming outlines
# ---------------------------------------------------------------------------

WORLD_CONTINENTS = {
    "North America": [
        (-130, 55), (-125, 60), (-120, 65), (-110, 70), (-100, 72),
        (-85, 75), (-70, 72), (-65, 65), (-60, 55), (-65, 45),
        (-70, 40), (-75, 35), (-80, 30), (-82, 25), (-88, 20),
        (-92, 18), (-97, 20), (-100, 22), (-105, 25), (-110, 30),
        (-115, 32), (-120, 35), (-122, 40), (-125, 45), (-128, 50),
        (-130, 55),
    ],
    "South America": [
        (-80, 10), (-75, 12), (-70, 12), (-62, 10), (-55, 5),
        (-50, 2), (-45, -2), (-40, -5), (-38, -10), (-37, -15),
        (-40, -22), (-45, -25), (-48, -28), (-52, -33), (-55, -35),
        (-58, -38), (-62, -40), (-65, -45), (-68, -50), (-72, -52),
        (-75, -50), (-75, -45), (-72, -40), (-72, -35), (-70, -30),
        (-70, -25), (-70, -20), (-72, -15), (-75, -10), (-78, -5),
        (-80, 0), (-80, 5), (-80, 10),
    ],
    "Europe": [
        (-10, 36), (-8, 38), (-9, 42), (-5, 44), (0, 43), (3, 43),
        (5, 44), (8, 44), (10, 46), (12, 46), (14, 48), (15, 50),
        (18, 54), (20, 55), (22, 56), (25, 58), (28, 60), (30, 62),
        (32, 65), (28, 68), (22, 70), (18, 70), (12, 68), (8, 65),
        (5, 62), (3, 58), (0, 56), (-3, 55), (-5, 52), (-6, 50),
        (-8, 48), (-10, 44), (-10, 36),
    ],
    "Africa": [
        (-17, 15), (-15, 20), (-12, 25), (-10, 30), (-5, 34),
        (0, 36), (5, 37), (10, 37), (15, 35), (20, 33), (25, 32),
        (30, 31), (33, 30), (35, 28), (38, 22), (40, 15), (42, 12),
        (45, 10), (48, 8), (50, 5), (48, 2), (45, 0), (42, -2),
        (40, -8), (38, -12), (36, -18), (34, -22), (32, -26),
        (30, -30), (28, -33), (25, -34), (20, -34), (18, -32),
        (16, -28), (14, -22), (12, -18), (10, -8), (8, 0), (5, 5),
        (2, 5), (-2, 5), (-5, 5), (-8, 5), (-10, 7), (-13, 10),
        (-15, 12), (-17, 15),
    ],
    "Asia": [
        (30, 35), (35, 37), (40, 40), (45, 42), (50, 44), (55, 45),
        (60, 48), (65, 50), (70, 55), (75, 55), (80, 52), (85, 50),
        (90, 48), (95, 45), (100, 42), (105, 40), (108, 35),
        (110, 30), (112, 25), (115, 22), (118, 20), (120, 22),
        (122, 25), (125, 30), (128, 35), (130, 40), (135, 42),
        (140, 45), (145, 48), (150, 52), (155, 55), (160, 58),
        (165, 60), (170, 62), (175, 65), (180, 68), (175, 70),
        (170, 72), (160, 72), (150, 70), (140, 68), (130, 65),
        (120, 62), (110, 60), (100, 58), (90, 56), (80, 58),
        (70, 60), (60, 62), (50, 60), (45, 55), (40, 50), (35, 45),
        (30, 40), (30, 35),
    ],
    "Australia": [
        (115, -20), (118, -18), (122, -15), (128, -14), (132, -12),
        (136, -12), (140, -15), (143, -12), (148, -18), (150, -22),
        (152, -25), (153, -28), (152, -32), (150, -35), (148, -38),
        (145, -38), (140, -36), (136, -34), (132, -32), (128, -32),
        (124, -34), (118, -34), (115, -32), (114, -28), (114, -24),
        (115, -20),
    ],
    "Antarctica": [
        (-180, -70), (-150, -72), (-120, -75), (-90, -78), (-60, -75),
        (-30, -70), (0, -68), (30, -70), (60, -72), (90, -75),
        (120, -72), (150, -70), (180, -70),
    ],
}

# Major country borders (simplified) for additional detail
COUNTRY_BORDERS = {
    "USA-Canada": [
        (-125, 49), (-120, 49), (-115, 49), (-110, 49), (-105, 49),
        (-100, 49), (-95, 49), (-90, 48), (-85, 46), (-80, 44),
        (-75, 45), (-70, 47), (-67, 47),
    ],
    "USA-Mexico": [
        (-117, 32), (-112, 31), (-108, 31), (-105, 30), (-100, 28),
        (-97, 26),
    ],
    "India outline": [
        (68, 24), (70, 22), (72, 20), (73, 16), (76, 10), (78, 8),
        (80, 10), (82, 14), (84, 18), (86, 22), (88, 22), (90, 24),
        (92, 26), (90, 28), (85, 28), (80, 30), (76, 32), (72, 30),
        (68, 24),
    ],
}

# ---------------------------------------------------------------------------
# Projection math utilities
# ---------------------------------------------------------------------------

def _equirectangular(lon, lat, center_lon=0):
    """Equirectangular (Plate Carree) projection."""
    x = np.radians(lon - center_lon)
    y = np.radians(lat)
    return np.degrees(x), np.degrees(y)


def _mercator(lon, lat, center_lon=0, clip_lat=85.0):
    """Mercator projection. Clips latitude to avoid infinity at poles."""
    lat = np.clip(lat, -clip_lat, clip_lat)
    x = np.radians(lon - center_lon)
    y = np.log(np.tan(np.radians(45 + lat / 2.0)))
    return np.degrees(x), np.degrees(y)


def _mollweide(lon, lat, center_lon=0, iterations=20):
    """Mollweide equal-area projection using Newton-Raphson iteration."""
    lon_r = np.radians(lon - center_lon)
    lat_r = np.radians(lat)

    # Newton-Raphson to solve 2*theta + sin(2*theta) = pi*sin(lat)
    theta = np.copy(lat_r)
    target = np.pi * np.sin(lat_r)
    for _ in range(iterations):
        dtheta = -(2 * theta + np.sin(2 * theta) - target) / (2 + 2 * np.cos(2 * theta) + 1e-12)
        theta += dtheta

    R = 1.0
    x = (2 * np.sqrt(2) / np.pi) * R * lon_r * np.cos(theta)
    y = np.sqrt(2) * R * np.sin(theta)
    return np.degrees(x), np.degrees(y)


PROJECTIONS = {
    "Equirectangular": _equirectangular,
    "Mercator": _mercator,
    "Mollweide": _mollweide,
}


def project_coords(lons, lats, projection="Equirectangular", center_lon=0):
    """Apply a map projection to arrays of lon/lat values."""
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    fn = PROJECTIONS.get(projection, _equirectangular)
    return fn(lons, lats, center_lon=center_lon)


# ---------------------------------------------------------------------------
# Coordinate conversion utilities
# ---------------------------------------------------------------------------

def decimal_to_dms(deg):
    """Convert decimal degrees to (degrees, minutes, seconds)."""
    d = int(deg)
    md = abs(deg - d) * 60
    m = int(md)
    s = (md - m) * 60
    return d, m, round(s, 4)


def dms_to_decimal(d, m, s):
    """Convert degrees/minutes/seconds to decimal degrees."""
    sign = -1 if d < 0 else 1
    return sign * (abs(d) + m / 60.0 + s / 3600.0)


def lat_to_str(lat):
    """Format latitude as DMS string."""
    d, m, s = decimal_to_dms(abs(lat))
    hemi = "N" if lat >= 0 else "S"
    return f"{d}\u00b0{m}'{s:.2f}\"{hemi}"


def lon_to_str(lon):
    """Format longitude as DMS string."""
    d, m, s = decimal_to_dms(abs(lon))
    hemi = "E" if lon >= 0 else "W"
    return f"{d}\u00b0{m}'{s:.2f}\"{hemi}"


# ---------------------------------------------------------------------------
# UTM conversion (simplified WGS84)
# ---------------------------------------------------------------------------

_WGS84_A = 6378137.0
_WGS84_F = 1 / 298.257223563
_WGS84_E2 = 2 * _WGS84_F - _WGS84_F ** 2
_K0 = 0.9996


def latlon_to_utm(lat, lon):
    """Convert WGS84 lat/lon to UTM easting/northing and zone."""
    zone = int((lon + 180) / 6) + 1
    lon0 = (zone - 1) * 6 - 180 + 3

    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    lon0_r = math.radians(lon0)

    e2 = _WGS84_E2
    ep2 = e2 / (1 - e2)
    N = _WGS84_A / math.sqrt(1 - e2 * math.sin(lat_r) ** 2)
    T = math.tan(lat_r) ** 2
    C = ep2 * math.cos(lat_r) ** 2
    A_val = (lon_r - lon0_r) * math.cos(lat_r)

    # Meridional arc
    M = _WGS84_A * (
        (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * lat_r
        - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * math.sin(2 * lat_r)
        + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * math.sin(4 * lat_r)
        - (35 * e2 ** 3 / 3072) * math.sin(6 * lat_r)
    )

    easting = _K0 * N * (
        A_val + (1 - T + C) * A_val ** 3 / 6
        + (5 - 18 * T + T ** 2 + 72 * C - 58 * ep2) * A_val ** 5 / 120
    ) + 500000.0

    northing = _K0 * (
        M + N * math.tan(lat_r) * (
            A_val ** 2 / 2
            + (5 - T + 9 * C + 4 * C ** 2) * A_val ** 4 / 24
            + (61 - 58 * T + T ** 2 + 600 * C - 330 * ep2) * A_val ** 6 / 720
        )
    )
    if lat < 0:
        northing += 10000000.0

    letter = "N" if lat >= 0 else "S"
    return easting, northing, zone, letter


# ---------------------------------------------------------------------------
# Great circle distance (Haversine)
# ---------------------------------------------------------------------------

def great_circle_distance(lat1, lon1, lat2, lon2, radius=6371.0):
    """Compute great-circle distance in km between two points on Earth."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(min(1.0, math.sqrt(a)))
    return radius * c


def great_circle_path(lat1, lon1, lat2, lon2, n_points=100):
    """Return arrays of lat/lon along the great circle arc."""
    lat1r, lon1r = math.radians(lat1), math.radians(lon1)
    lat2r, lon2r = math.radians(lat2), math.radians(lon2)
    d = great_circle_distance(lat1, lon1, lat2, lon2) / 6371.0  # angular distance

    if d < 1e-12:
        return np.array([lat1, lat2]), np.array([lon1, lon2])

    fracs = np.linspace(0, 1, n_points)
    lats, lons = [], []
    for f in fracs:
        A = math.sin((1 - f) * d) / math.sin(d)
        B = math.sin(f * d) / math.sin(d)
        x = A * math.cos(lat1r) * math.cos(lon1r) + B * math.cos(lat2r) * math.cos(lon2r)
        y = A * math.cos(lat1r) * math.sin(lon1r) + B * math.cos(lat2r) * math.sin(lon2r)
        z = A * math.sin(lat1r) + B * math.sin(lat2r)
        lats.append(math.degrees(math.atan2(z, math.sqrt(x ** 2 + y ** 2))))
        lons.append(math.degrees(math.atan2(y, x)))
    return np.array(lats), np.array(lons)


# ---------------------------------------------------------------------------
# GIS Widget
# ---------------------------------------------------------------------------

class GISWidget(QWidget):
    """Full-featured GIS / Mapping widget for the PyQt5 Scientific Suite."""

    dataChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._logger = None
        self._projection = "Equirectangular"
        self._center_lon = 0.0
        self._show_grid = True
        self._show_coastlines = True
        self._show_borders = True
        self._show_scale = True
        self._points = []        # list of dict(lat, lon, label, value)
        self._polygons = []      # list of dict(coords=[(lon,lat),...], label, color)
        self._heatmap_data = None  # (lats, lons, values)
        self._csv_data = None
        self._colormap = "viridis"
        self._init_ui()

    # ---- Logging -----------------------------------------------------------

    def set_logger(self, fn):
        """Set external logging callback: fn(message_string)."""
        self._logger = fn

    def _log(self, msg):
        if self._logger:
            try:
                self._logger(str(msg))
            except Exception:
                pass

    # ---- UI Setup ----------------------------------------------------------

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # Left: controls panel
        ctrl_widget = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_widget)
        ctrl_layout.setContentsMargins(4, 4, 4, 4)

        tabs = QTabWidget()
        tabs.addTab(self._build_projection_tab(), "Projection")
        tabs.addTab(self._build_points_tab(), "Points")
        tabs.addTab(self._build_distance_tab(), "Distance")
        tabs.addTab(self._build_converter_tab(), "Converter")
        tabs.addTab(self._build_polygon_tab(), "Polygons")
        tabs.addTab(self._build_data_tab(), "Data/CSV")
        ctrl_layout.addWidget(tabs)

        # Right: map canvas
        canvas_widget = QWidget()
        canvas_layout = QVBoxLayout(canvas_widget)
        canvas_layout.setContentsMargins(0, 0, 0, 0)

        self._figure = Figure(figsize=(10, 6), dpi=100)
        style_figure(self._figure)
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._toolbar = NavigationToolbar(self._canvas, self)

        canvas_layout.addWidget(self._toolbar)
        canvas_layout.addWidget(self._canvas)

        splitter.addWidget(ctrl_widget)
        splitter.addWidget(canvas_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        main_layout.addWidget(splitter)
        self._redraw()

    # ---- Tab builders ------------------------------------------------------

    def _build_projection_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        grp = QGroupBox("Map Projection")
        form = QFormLayout()

        self._proj_combo = QComboBox()
        self._proj_combo.addItems(list(PROJECTIONS.keys()))
        self._proj_combo.currentTextChanged.connect(self._on_projection_changed)
        form.addRow("Projection:", self._proj_combo)

        self._center_lon_spin = QDoubleSpinBox()
        self._center_lon_spin.setRange(-180, 180)
        self._center_lon_spin.setValue(0)
        self._center_lon_spin.setSuffix("\u00b0")
        self._center_lon_spin.valueChanged.connect(self._on_center_lon_changed)
        form.addRow("Center Lon:", self._center_lon_spin)

        self._grid_chk = QCheckBox("Show Grid Lines")
        self._grid_chk.setChecked(True)
        self._grid_chk.toggled.connect(self._on_toggle_grid)
        form.addRow(self._grid_chk)

        self._coast_chk = QCheckBox("Show Coastlines")
        self._coast_chk.setChecked(True)
        self._coast_chk.toggled.connect(self._on_toggle_coastlines)
        form.addRow(self._coast_chk)

        self._border_chk = QCheckBox("Show Borders")
        self._border_chk.setChecked(True)
        self._border_chk.toggled.connect(self._on_toggle_borders)
        form.addRow(self._border_chk)

        self._scale_chk = QCheckBox("Show Scale Bar")
        self._scale_chk.setChecked(True)
        self._scale_chk.toggled.connect(self._on_toggle_scale)
        form.addRow(self._scale_chk)

        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(["viridis", "plasma", "inferno", "magma",
                                   "cividis", "hot", "cool", "jet", "terrain",
                                   "RdYlGn", "Spectral", "Blues", "Reds"])
        self._cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        form.addRow("Colormap:", self._cmap_combo)

        grp.setLayout(form)
        layout.addWidget(grp)

        btn_redraw = QPushButton("Redraw Map")
        btn_redraw.clicked.connect(self._redraw)
        layout.addWidget(btn_redraw)

        btn_export = QPushButton("Export Map Image...")
        btn_export.clicked.connect(self._export_dialog)
        layout.addWidget(btn_export)

        layout.addStretch()
        return w

    def _build_points_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        grp = QGroupBox("Add Point")
        form = QFormLayout()

        self._pt_lat = QDoubleSpinBox()
        self._pt_lat.setRange(-90, 90)
        self._pt_lat.setDecimals(6)
        self._pt_lat.setSuffix("\u00b0")
        form.addRow("Latitude:", self._pt_lat)

        self._pt_lon = QDoubleSpinBox()
        self._pt_lon.setRange(-180, 180)
        self._pt_lon.setDecimals(6)
        self._pt_lon.setSuffix("\u00b0")
        form.addRow("Longitude:", self._pt_lon)

        self._pt_label = QLineEdit()
        self._pt_label.setPlaceholderText("Point label")
        form.addRow("Label:", self._pt_label)

        self._pt_value = QDoubleSpinBox()
        self._pt_value.setRange(-1e9, 1e9)
        self._pt_value.setDecimals(4)
        self._pt_value.setValue(0)
        form.addRow("Value:", self._pt_value)

        grp.setLayout(form)
        layout.addWidget(grp)

        btn_add = QPushButton("Add Point")
        btn_add.clicked.connect(self._add_point)
        layout.addWidget(btn_add)

        self._points_list = QListWidget()
        layout.addWidget(QLabel("Points:"))
        layout.addWidget(self._points_list)

        btn_rm = QPushButton("Remove Selected Point")
        btn_rm.clicked.connect(self._remove_point)
        layout.addWidget(btn_rm)

        btn_clear = QPushButton("Clear All Points")
        btn_clear.clicked.connect(self._clear_points)
        layout.addWidget(btn_clear)

        layout.addStretch()
        return w

    def _build_distance_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        grp = QGroupBox("Great Circle Distance")
        form = QFormLayout()

        self._gc_lat1 = QDoubleSpinBox(); self._gc_lat1.setRange(-90, 90); self._gc_lat1.setDecimals(6)
        self._gc_lon1 = QDoubleSpinBox(); self._gc_lon1.setRange(-180, 180); self._gc_lon1.setDecimals(6)
        self._gc_lat2 = QDoubleSpinBox(); self._gc_lat2.setRange(-90, 90); self._gc_lat2.setDecimals(6)
        self._gc_lon2 = QDoubleSpinBox(); self._gc_lon2.setRange(-180, 180); self._gc_lon2.setDecimals(6)

        form.addRow("Point 1 Lat:", self._gc_lat1)
        form.addRow("Point 1 Lon:", self._gc_lon1)
        form.addRow("Point 2 Lat:", self._gc_lat2)
        form.addRow("Point 2 Lon:", self._gc_lon2)

        grp.setLayout(form)
        layout.addWidget(grp)

        btn_calc = QPushButton("Calculate Distance")
        btn_calc.clicked.connect(self._calc_distance)
        layout.addWidget(btn_calc)

        self._dist_result = QTextEdit()
        self._dist_result.setReadOnly(True)
        self._dist_result.setMaximumHeight(120)
        layout.addWidget(self._dist_result)

        btn_draw = QPushButton("Draw Great Circle on Map")
        btn_draw.clicked.connect(self._draw_great_circle)
        layout.addWidget(btn_draw)

        layout.addStretch()
        return w

    def _build_converter_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        # Decimal -> DMS
        grp1 = QGroupBox("Decimal Degrees -> DMS")
        form1 = QFormLayout()
        self._dec_lat = QDoubleSpinBox(); self._dec_lat.setRange(-90, 90); self._dec_lat.setDecimals(8)
        self._dec_lon = QDoubleSpinBox(); self._dec_lon.setRange(-180, 180); self._dec_lon.setDecimals(8)
        form1.addRow("Latitude:", self._dec_lat)
        form1.addRow("Longitude:", self._dec_lon)
        grp1.setLayout(form1)
        layout.addWidget(grp1)

        btn_to_dms = QPushButton("Convert to DMS")
        btn_to_dms.clicked.connect(self._convert_to_dms)
        layout.addWidget(btn_to_dms)

        # DMS -> Decimal
        grp2 = QGroupBox("DMS -> Decimal Degrees")
        form2 = QFormLayout()
        self._dms_d = QSpinBox(); self._dms_d.setRange(-180, 180)
        self._dms_m = QSpinBox(); self._dms_m.setRange(0, 59)
        self._dms_s = QDoubleSpinBox(); self._dms_s.setRange(0, 59.9999); self._dms_s.setDecimals(4)
        form2.addRow("Degrees:", self._dms_d)
        form2.addRow("Minutes:", self._dms_m)
        form2.addRow("Seconds:", self._dms_s)
        grp2.setLayout(form2)
        layout.addWidget(grp2)

        btn_to_dec = QPushButton("Convert to Decimal")
        btn_to_dec.clicked.connect(self._convert_to_decimal)
        layout.addWidget(btn_to_dec)

        # UTM
        grp3 = QGroupBox("Lat/Lon -> UTM")
        form3 = QFormLayout()
        self._utm_lat = QDoubleSpinBox(); self._utm_lat.setRange(-84, 84); self._utm_lat.setDecimals(8)
        self._utm_lon = QDoubleSpinBox(); self._utm_lon.setRange(-180, 180); self._utm_lon.setDecimals(8)
        form3.addRow("Latitude:", self._utm_lat)
        form3.addRow("Longitude:", self._utm_lon)
        grp3.setLayout(form3)
        layout.addWidget(grp3)

        btn_utm = QPushButton("Convert to UTM")
        btn_utm.clicked.connect(self._convert_to_utm)
        layout.addWidget(btn_utm)

        self._conv_result = QTextEdit()
        self._conv_result.setReadOnly(True)
        self._conv_result.setMaximumHeight(140)
        layout.addWidget(self._conv_result)

        layout.addStretch()
        return w

    def _build_polygon_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        grp = QGroupBox("Draw Polygon")
        form = QFormLayout()

        self._poly_coords = QTextEdit()
        self._poly_coords.setPlaceholderText("Enter coordinates as lon,lat per line:\n-10,50\n10,50\n10,40\n-10,40")
        self._poly_coords.setMaximumHeight(160)
        form.addRow("Coordinates:", self._poly_coords)

        self._poly_label = QLineEdit()
        self._poly_label.setPlaceholderText("Polygon label")
        form.addRow("Label:", self._poly_label)

        self._poly_color = QLineEdit("blue")
        form.addRow("Color:", self._poly_color)

        grp.setLayout(form)
        layout.addWidget(grp)

        btn_add = QPushButton("Add Polygon")
        btn_add.clicked.connect(self._add_polygon)
        layout.addWidget(btn_add)

        self._poly_list = QListWidget()
        layout.addWidget(QLabel("Polygons:"))
        layout.addWidget(self._poly_list)

        btn_rm = QPushButton("Remove Selected Polygon")
        btn_rm.clicked.connect(self._remove_polygon)
        layout.addWidget(btn_rm)

        btn_clear = QPushButton("Clear All Polygons")
        btn_clear.clicked.connect(self._clear_polygons)
        layout.addWidget(btn_clear)

        layout.addStretch()
        return w

    def _build_data_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        btn_csv = QPushButton("Load CSV with Lat/Lon...")
        btn_csv.clicked.connect(self._load_csv_dialog)
        layout.addWidget(btn_csv)

        self._csv_info = QTextEdit()
        self._csv_info.setReadOnly(True)
        self._csv_info.setMaximumHeight(100)
        layout.addWidget(self._csv_info)

        grp = QGroupBox("CSV Column Mapping")
        form = QFormLayout()
        self._csv_lat_col = QLineEdit("lat")
        self._csv_lon_col = QLineEdit("lon")
        self._csv_val_col = QLineEdit("value")
        self._csv_label_col = QLineEdit("name")
        form.addRow("Lat column:", self._csv_lat_col)
        form.addRow("Lon column:", self._csv_lon_col)
        form.addRow("Value column:", self._csv_val_col)
        form.addRow("Label column:", self._csv_label_col)
        grp.setLayout(form)
        layout.addWidget(grp)

        btn_plot_csv = QPushButton("Plot CSV Data on Map")
        btn_plot_csv.clicked.connect(self._plot_csv_on_map)
        layout.addWidget(btn_plot_csv)

        # Heatmap controls
        grp2 = QGroupBox("Heatmap Overlay")
        form2 = QFormLayout()

        self._heat_res = QSpinBox()
        self._heat_res.setRange(10, 200)
        self._heat_res.setValue(50)
        form2.addRow("Grid Resolution:", self._heat_res)

        self._heat_radius = QDoubleSpinBox()
        self._heat_radius.setRange(0.5, 50.0)
        self._heat_radius.setValue(5.0)
        self._heat_radius.setSuffix("\u00b0")
        form2.addRow("Influence Radius:", self._heat_radius)

        grp2.setLayout(form2)
        layout.addWidget(grp2)

        btn_heat = QPushButton("Generate Heatmap from Points")
        btn_heat.clicked.connect(self._generate_heatmap)
        layout.addWidget(btn_heat)

        btn_clear_heat = QPushButton("Clear Heatmap")
        btn_clear_heat.clicked.connect(self._clear_heatmap)
        layout.addWidget(btn_clear_heat)

        layout.addStretch()
        return w

    # ---- Projection / display callbacks ------------------------------------

    def _on_projection_changed(self, name):
        self._projection = name
        self._redraw()

    def _on_center_lon_changed(self, val):
        self._center_lon = val
        self._redraw()

    def _on_toggle_grid(self, state):
        self._show_grid = state
        self._redraw()

    def _on_toggle_coastlines(self, state):
        self._show_coastlines = state
        self._redraw()

    def _on_toggle_borders(self, state):
        self._show_borders = state
        self._redraw()

    def _on_toggle_scale(self, state):
        self._show_scale = state
        self._redraw()

    def _on_cmap_changed(self, name):
        self._colormap = name
        self._redraw()

    # ---- Point operations --------------------------------------------------

    def _add_point(self):
        lat = self._pt_lat.value()
        lon = self._pt_lon.value()
        label = self._pt_label.text().strip() or f"({lat:.2f}, {lon:.2f})"
        value = self._pt_value.value()
        pt = {"lat": lat, "lon": lon, "label": label, "value": value}
        self._points.append(pt)
        self._points_list.addItem(f"{label}  [{lat:.4f}, {lon:.4f}]  val={value}")
        self._log(f"Added point: {label} at ({lat}, {lon})")
        self._redraw()

    def _remove_point(self):
        row = self._points_list.currentRow()
        if row >= 0:
            self._points.pop(row)
            self._points_list.takeItem(row)
            self._redraw()

    def _clear_points(self):
        self._points.clear()
        self._points_list.clear()
        self._redraw()

    # ---- Distance ----------------------------------------------------------

    def _calc_distance(self):
        lat1 = self._gc_lat1.value()
        lon1 = self._gc_lon1.value()
        lat2 = self._gc_lat2.value()
        lon2 = self._gc_lon2.value()

        dist_km = great_circle_distance(lat1, lon1, lat2, lon2)
        dist_mi = dist_km * 0.621371
        dist_nm = dist_km * 0.539957

        bearing = self._initial_bearing(lat1, lon1, lat2, lon2)

        self._dist_result.setPlainText(
            f"From: {lat_to_str(lat1)}, {lon_to_str(lon1)}\n"
            f"To:   {lat_to_str(lat2)}, {lon_to_str(lon2)}\n\n"
            f"Distance: {dist_km:.3f} km\n"
            f"          {dist_mi:.3f} miles\n"
            f"          {dist_nm:.3f} nautical miles\n"
            f"Initial Bearing: {bearing:.2f}\u00b0"
        )
        self._log(f"Great circle distance: {dist_km:.3f} km")

    @staticmethod
    def _initial_bearing(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        return (math.degrees(math.atan2(x, y)) + 360) % 360

    def _draw_great_circle(self):
        lat1 = self._gc_lat1.value()
        lon1 = self._gc_lon1.value()
        lat2 = self._gc_lat2.value()
        lon2 = self._gc_lon2.value()

        lats, lons = great_circle_path(lat1, lon1, lat2, lon2, n_points=200)
        xs, ys = project_coords(lons, lats, self._projection, self._center_lon)

        ax = self._figure.axes[0] if self._figure.axes else self._figure.add_subplot(111)
        ax.plot(xs, ys, 'r-', linewidth=2, label="Great Circle", zorder=8)
        ax.plot(xs[0], ys[0], 'go', markersize=8, zorder=9)
        ax.plot(xs[-1], ys[-1], 'ro', markersize=8, zorder=9)
        ax.legend(loc="lower left", fontsize=8)
        self._canvas.draw()
        self._log("Drew great circle path on map")

    # ---- Converter ---------------------------------------------------------

    def _convert_to_dms(self):
        lat = self._dec_lat.value()
        lon = self._dec_lon.value()
        self._conv_result.setPlainText(
            f"Latitude:  {lat_to_str(lat)}\n"
            f"Longitude: {lon_to_str(lon)}"
        )

    def _convert_to_decimal(self):
        d = self._dms_d.value()
        m = self._dms_m.value()
        s = self._dms_s.value()
        dec = dms_to_decimal(d, m, s)
        self._conv_result.setPlainText(f"Decimal Degrees: {dec:.8f}\u00b0")

    def _convert_to_utm(self):
        lat = self._utm_lat.value()
        lon = self._utm_lon.value()
        try:
            e, n, zone, letter = latlon_to_utm(lat, lon)
            self._conv_result.setPlainText(
                f"UTM Zone: {zone}{letter}\n"
                f"Easting:  {e:.2f} m\n"
                f"Northing: {n:.2f} m"
            )
        except Exception as exc:
            self._conv_result.setPlainText(f"Error: {exc}")

    # ---- Polygon operations ------------------------------------------------

    def _add_polygon(self):
        text = self._poly_coords.toPlainText().strip()
        if not text:
            return
        coords = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    lon, lat = float(parts[0]), float(parts[1])
                    coords.append((lon, lat))
                except ValueError:
                    continue
        if len(coords) < 3:
            QMessageBox.warning(self, "Polygon", "Need at least 3 coordinate pairs.")
            return

        label = self._poly_label.text().strip() or f"Polygon {len(self._polygons)+1}"
        color = self._poly_color.text().strip() or "blue"
        self._polygons.append({"coords": coords, "label": label, "color": color})
        self._poly_list.addItem(f"{label} ({len(coords)} vertices)")
        self._log(f"Added polygon: {label}")
        self._redraw()

    def _remove_polygon(self):
        row = self._poly_list.currentRow()
        if row >= 0:
            self._polygons.pop(row)
            self._poly_list.takeItem(row)
            self._redraw()

    def _clear_polygons(self):
        self._polygons.clear()
        self._poly_list.clear()
        self._redraw()

    # ---- CSV / Data --------------------------------------------------------

    def _load_csv_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load CSV", "", "CSV Files (*.csv);;All Files (*)")
        if path:
            self.load_file(path)

    def load_file(self, path):
        """Load a CSV file for plotting. Expects columns for lat/lon."""
        try:
            rows = []
            with open(path, "r", newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                self._csv_columns = reader.fieldnames or []
                for row in reader:
                    rows.append(row)
            self._csv_data = rows
            self._csv_info.setPlainText(
                f"Loaded: {os.path.basename(path)}\n"
                f"Rows: {len(rows)}\n"
                f"Columns: {', '.join(self._csv_columns)}"
            )
            self._log(f"Loaded CSV: {path} ({len(rows)} rows)")
        except Exception as exc:
            self._log(f"CSV load error: {exc}")
            QMessageBox.critical(self, "CSV Error", str(exc))

    def _plot_csv_on_map(self):
        if not self._csv_data:
            QMessageBox.information(self, "CSV", "No CSV data loaded.")
            return

        lat_col = self._csv_lat_col.text().strip()
        lon_col = self._csv_lon_col.text().strip()
        val_col = self._csv_val_col.text().strip()
        label_col = self._csv_label_col.text().strip()

        # Parse CSV rows into points
        new_points = []
        for row in self._csv_data:
            try:
                lat = float(row.get(lat_col, ""))
                lon = float(row.get(lon_col, ""))
            except (ValueError, TypeError):
                continue
            label = row.get(label_col, "")
            try:
                value = float(row.get(val_col, 0))
            except (ValueError, TypeError):
                value = 0.0
            new_points.append({"lat": lat, "lon": lon, "label": label, "value": value})

        self._points.extend(new_points)
        for pt in new_points:
            self._points_list.addItem(
                f"{pt['label']}  [{pt['lat']:.4f}, {pt['lon']:.4f}]  val={pt['value']}"
            )
        self._log(f"Plotted {len(new_points)} points from CSV")
        self._redraw()

    # ---- Heatmap -----------------------------------------------------------

    def _generate_heatmap(self):
        if not self._points:
            QMessageBox.information(self, "Heatmap", "Add points first to generate a heatmap.")
            return

        res = self._heat_res.value()
        radius = self._heat_radius.value()

        lats = np.array([p["lat"] for p in self._points])
        lons = np.array([p["lon"] for p in self._points])
        vals = np.array([p["value"] for p in self._points])

        # Build a grid and compute weighted influence
        grid_lat = np.linspace(-90, 90, res)
        grid_lon = np.linspace(-180, 180, res * 2)
        grid_lon_2d, grid_lat_2d = np.meshgrid(grid_lon, grid_lat)
        heat = np.zeros_like(grid_lat_2d)

        for i in range(len(lats)):
            dist = np.sqrt((grid_lat_2d - lats[i]) ** 2 + (grid_lon_2d - lons[i]) ** 2)
            weight = np.exp(-0.5 * (dist / radius) ** 2)
            heat += weight * (vals[i] if vals[i] != 0 else 1.0)

        self._heatmap_data = (grid_lat_2d, grid_lon_2d, heat)
        self._log("Generated heatmap overlay")
        self._redraw()

    def _clear_heatmap(self):
        self._heatmap_data = None
        self._redraw()

    # ---- Export ------------------------------------------------------------

    def export(self, path=None):
        """Export the current map to an image file. Returns the path used."""
        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Map", "map.png",
                "PNG (*.png);;JPEG (*.jpg);;SVG (*.svg);;PDF (*.pdf);;All Files (*)"
            )
        if path:
            self._figure.savefig(path, dpi=150, bbox_inches="tight")
            self._log(f"Exported map to: {path}")
        return path

    def _export_dialog(self):
        self.export()

    # ---- Main redraw -------------------------------------------------------

    def _redraw(self):
        """Full redraw of the map canvas."""
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.set_facecolor("#e6f2ff")

        proj = self._projection
        clon = self._center_lon

        # Draw coastlines / continent outlines
        if self._show_coastlines:
            for name, coords in WORLD_CONTINENTS.items():
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                xs, ys = project_coords(lons, lats, proj, clon)
                ax.fill(xs, ys, facecolor="#d4e8c2", edgecolor="#3a6e28",
                        linewidth=0.8, alpha=0.7, zorder=2)

        # Draw borders
        if self._show_borders:
            for name, coords in COUNTRY_BORDERS.items():
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                xs, ys = project_coords(lons, lats, proj, clon)
                ax.plot(xs, ys, color="#888888", linewidth=0.6,
                        linestyle="--", zorder=3)

        # Grid lines
        if self._show_grid:
            self._draw_grid(ax, proj, clon)

        # Heatmap overlay
        if self._heatmap_data is not None:
            glat, glon, heat = self._heatmap_data
            gx, gy = project_coords(glon.ravel(), glat.ravel(), proj, clon)
            gx = gx.reshape(glat.shape)
            gy = gy.reshape(glat.shape)
            ax.contourf(gx, gy, heat, levels=30, cmap=self._colormap,
                        alpha=0.5, zorder=4)

        # Polygons
        for poly in self._polygons:
            lons = [c[0] for c in poly["coords"]]
            lats = [c[1] for c in poly["coords"]]
            xs, ys = project_coords(lons, lats, proj, clon)
            polygon = MplPolygon(list(zip(xs, ys)), closed=True,
                                 facecolor=poly["color"], alpha=0.3,
                                 edgecolor=poly["color"], linewidth=1.5, zorder=5)
            ax.add_patch(polygon)
            cx, cy = np.mean(xs), np.mean(ys)
            ax.text(cx, cy, poly["label"], fontsize=7, ha="center",
                    color=poly["color"], weight="bold", zorder=6)

        # Points (color-coded by value)
        if self._points:
            lats = np.array([p["lat"] for p in self._points])
            lons = np.array([p["lon"] for p in self._points])
            vals = np.array([p["value"] for p in self._points])
            labels = [p["label"] for p in self._points]

            xs, ys = project_coords(lons, lats, proj, clon)

            if np.ptp(vals) > 0:
                norm = mcolors.Normalize(vmin=vals.min(), vmax=vals.max())
                cmap_obj = cm.get_cmap(self._colormap)
                colors = cmap_obj(norm(vals))
                sc = ax.scatter(xs, ys, c=vals, cmap=self._colormap, norm=norm,
                                s=50, edgecolors="black", linewidths=0.5, zorder=7)
                self._figure.colorbar(sc, ax=ax, shrink=0.6, label="Value")
            else:
                ax.scatter(xs, ys, c="red", s=50, edgecolors="black",
                           linewidths=0.5, zorder=7)

            for i, label in enumerate(labels):
                if label:
                    ax.annotate(label, (xs[i], ys[i]), fontsize=6,
                                xytext=(4, 4), textcoords="offset points",
                                zorder=8, color="#333333",
                                bbox=dict(boxstyle="round,pad=0.2",
                                          facecolor="white", alpha=0.7, linewidth=0.3))

        # Scale bar
        if self._show_scale:
            self._draw_scale_bar(ax, proj, clon)

        ax.set_aspect("equal" if proj != "Mercator" else "auto")
        ax.set_title(f"Map Projection: {proj}", fontsize=10, pad=10)
        ax.tick_params(labelsize=7)

        self._canvas.draw()

    # ---- Grid drawing ------------------------------------------------------

    def _draw_grid(self, ax, proj, clon):
        """Draw latitude/longitude grid lines."""
        # Longitude lines
        for lon in range(-180, 181, 30):
            lats = np.linspace(-85, 85, 200)
            lons = np.full_like(lats, lon)
            xs, ys = project_coords(lons, lats, proj, clon)
            ax.plot(xs, ys, color="#cccccc", linewidth=0.4, zorder=1)
            # Label
            x_lbl, y_lbl = project_coords([lon], [0], proj, clon)
            ax.text(x_lbl[0], y_lbl[0], f"{lon}\u00b0", fontsize=5,
                    ha="center", va="top", color="#999999", zorder=1)

        # Latitude lines
        for lat in range(-60, 91, 30):
            lons = np.linspace(-180, 180, 400)
            lats = np.full_like(lons, lat)
            xs, ys = project_coords(lons, lats, proj, clon)
            ax.plot(xs, ys, color="#cccccc", linewidth=0.4, zorder=1)
            x_lbl, y_lbl = project_coords([clon - 175], [lat], proj, clon)
            ax.text(x_lbl[0], y_lbl[0], f"{lat}\u00b0", fontsize=5,
                    ha="right", va="center", color="#999999", zorder=1)

    # ---- Scale bar ---------------------------------------------------------

    def _draw_scale_bar(self, ax, proj, clon):
        """Draw an approximate scale bar in the lower-left corner."""
        # Approximate scale at the equator
        # 1 degree of longitude at equator ~ 111.32 km
        scale_deg = 30  # degrees shown
        scale_km = scale_deg * 111.32

        x0, y0 = project_coords([-160], [-75], proj, clon)
        x1, y1 = project_coords([-160 + scale_deg], [-75], proj, clon)

        ax.plot([x0[0], x1[0]], [y0[0], y1[0]], color="black", linewidth=2.5, zorder=10)
        ax.plot([x0[0], x0[0]], [y0[0] - 1, y0[0] + 1], color="black", linewidth=1.5, zorder=10)
        ax.plot([x1[0], x1[0]], [y1[0] - 1, y1[0] + 1], color="black", linewidth=1.5, zorder=10)
        mid_x = (x0[0] + x1[0]) / 2
        ax.text(mid_x, y0[0] + 3, f"~{scale_km:.0f} km", fontsize=6,
                ha="center", va="bottom", color="black", weight="bold", zorder=10)

    # ---- Public API for programmatic use -----------------------------------

    def add_point(self, lat, lon, label="", value=0.0):
        """Programmatically add a point to the map."""
        pt = {"lat": float(lat), "lon": float(lon),
              "label": str(label), "value": float(value)}
        self._points.append(pt)
        self._points_list.addItem(f"{pt['label']}  [{lat:.4f}, {lon:.4f}]  val={value}")
        self._redraw()

    def add_polygon_coords(self, coords, label="", color="blue"):
        """Programmatically add a polygon. coords = list of (lon, lat) tuples."""
        self._polygons.append({"coords": list(coords), "label": label, "color": color})
        self._poly_list.addItem(f"{label} ({len(coords)} vertices)")
        self._redraw()

    def set_projection(self, name):
        """Set the map projection programmatically."""
        if name in PROJECTIONS:
            self._projection = name
            self._proj_combo.setCurrentText(name)
            self._redraw()

    def set_center_longitude(self, lon):
        """Set the center longitude for projections."""
        self._center_lon = float(lon)
        self._center_lon_spin.setValue(lon)
        self._redraw()

    def get_points(self):
        """Return the current list of points as list of dicts."""
        return list(self._points)

    def get_distance(self, lat1, lon1, lat2, lon2):
        """Return great circle distance in km."""
        return great_circle_distance(lat1, lon1, lat2, lon2)

    def convert_dd_to_dms(self, degrees):
        """Convert decimal degrees to (d, m, s) tuple."""
        return decimal_to_dms(degrees)

    def convert_dms_to_dd(self, d, m, s):
        """Convert DMS to decimal degrees."""
        return dms_to_decimal(d, m, s)

    def convert_to_utm(self, lat, lon):
        """Convert lat/lon to UTM. Returns (easting, northing, zone, letter)."""
        return latlon_to_utm(lat, lon)

    def set_heatmap(self, lats, lons, values, resolution=50, radius=5.0):
        """Programmatically generate and display a heatmap overlay."""
        lats_arr = np.asarray(lats)
        lons_arr = np.asarray(lons)
        vals_arr = np.asarray(values)

        grid_lat = np.linspace(-90, 90, resolution)
        grid_lon = np.linspace(-180, 180, resolution * 2)
        grid_lon_2d, grid_lat_2d = np.meshgrid(grid_lon, grid_lat)
        heat = np.zeros_like(grid_lat_2d)

        for i in range(len(lats_arr)):
            dist = np.sqrt((grid_lat_2d - lats_arr[i]) ** 2 +
                           (grid_lon_2d - lons_arr[i]) ** 2)
            weight = np.exp(-0.5 * (dist / radius) ** 2)
            heat += weight * (vals_arr[i] if vals_arr[i] != 0 else 1.0)

        self._heatmap_data = (grid_lat_2d, grid_lon_2d, heat)
        self._redraw()

    def clear_all(self):
        """Clear all data layers (points, polygons, heatmap)."""
        self._points.clear()
        self._points_list.clear()
        self._polygons.clear()
        self._poly_list.clear()
        self._heatmap_data = None
        self._csv_data = None
        self._redraw()
