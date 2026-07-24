"""
IC Layout Editor Widget for PyQt5 Scientific Suite.
Provides a KLayout-like integrated circuit layout editor with drawing tools,
layer management, design rule checking, cell hierarchy, and GDS-II export.
"""

import struct
import math
import json
import os
from enum import Enum, auto
from typing import List, Optional, Dict, Tuple, Callable, Any

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTreeWidget, QTreeWidgetItem,
    QToolBar, QAction, QActionGroup, QLabel, QSpinBox, QDoubleSpinBox,
    QComboBox, QGroupBox, QFormLayout, QLineEdit, QPushButton, QCheckBox,
    QColorDialog, QMessageBox, QFileDialog, QMenu, QInputDialog, QScrollArea,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView
)
from PyQt5.QtCore import Qt, QPointF, QRectF, QPoint, pyqtSignal, QTimer
from PyQt5.QtGui import (
    QPainter, QPen, QBrush, QColor, QPolygonF, QFont, QTransform,
    QWheelEvent, QMouseEvent, QPainterPath, QKeyEvent, QPixmap, QIcon
)


# ---------------------------------------------------------------------------
# Layer definitions
# ---------------------------------------------------------------------------

class LayerDef:
    """Definition for a single IC fabrication layer."""

    def __init__(self, name: str, gds_layer: int, gds_datatype: int,
                 color: QColor, pattern: Qt.BrushStyle = Qt.SolidPattern,
                 visible: bool = True, selectable: bool = True):
        self.name = name
        self.gds_layer = gds_layer
        self.gds_datatype = gds_datatype
        self.color = QColor(color)
        self.fill_color = QColor(color)
        self.fill_color.setAlpha(60)
        self.pattern = pattern
        self.visible = visible
        self.selectable = selectable


DEFAULT_LAYERS: List[LayerDef] = [
    LayerDef("Diffusion", 1, 0, QColor(255, 255, 0), Qt.Dense4Pattern),
    LayerDef("Poly", 2, 0, QColor(255, 0, 0), Qt.FDiagPattern),
    LayerDef("Contact", 3, 0, QColor(100, 100, 100), Qt.CrossPattern),
    LayerDef("Metal1", 4, 0, QColor(0, 100, 255), Qt.SolidPattern),
    LayerDef("Via", 5, 0, QColor(180, 0, 180), Qt.Dense6Pattern),
    LayerDef("Metal2", 6, 0, QColor(0, 200, 100), Qt.BDiagPattern),
]


# ---------------------------------------------------------------------------
# Drawing tool enumeration
# ---------------------------------------------------------------------------

class DrawTool(Enum):
    SELECT = auto()
    RECTANGLE = auto()
    POLYGON = auto()
    PATH = auto()
    VIA = auto()
    TEXT = auto()


# ---------------------------------------------------------------------------
# Shape classes
# ---------------------------------------------------------------------------

class Shape:
    """Base class for all layout shapes."""

    _next_id = 1

    def __init__(self, layer_index: int):
        self.id = Shape._next_id
        Shape._next_id += 1
        self.layer_index = layer_index
        self.selected = False
        self.properties: Dict[str, Any] = {}

    def bounding_rect(self) -> QRectF:
        return QRectF()

    def contains(self, pt: QPointF) -> bool:
        return self.bounding_rect().contains(pt)

    def translate(self, dx: float, dy: float):
        pass

    def clone(self) -> "Shape":
        raise NotImplementedError

    def to_dict(self) -> dict:
        return {"type": self.__class__.__name__, "layer": self.layer_index,
                "props": self.properties}


class RectShape(Shape):
    def __init__(self, layer_index: int, rect: QRectF):
        super().__init__(layer_index)
        self.rect = QRectF(rect)

    def bounding_rect(self) -> QRectF:
        return QRectF(self.rect)

    def contains(self, pt: QPointF) -> bool:
        return self.rect.contains(pt)

    def translate(self, dx: float, dy: float):
        self.rect.translate(dx, dy)

    def clone(self) -> "RectShape":
        c = RectShape(self.layer_index, QRectF(self.rect))
        c.properties = dict(self.properties)
        return c

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({"x": self.rect.x(), "y": self.rect.y(),
                  "w": self.rect.width(), "h": self.rect.height()})
        return d


class PolygonShape(Shape):
    def __init__(self, layer_index: int, points: List[QPointF]):
        super().__init__(layer_index)
        self.points = list(points)

    def bounding_rect(self) -> QRectF:
        if not self.points:
            return QRectF()
        poly = QPolygonF(self.points)
        return poly.boundingRect()

    def contains(self, pt: QPointF) -> bool:
        poly = QPolygonF(self.points)
        return poly.containsPoint(pt, Qt.OddEvenFill)

    def translate(self, dx: float, dy: float):
        self.points = [QPointF(p.x() + dx, p.y() + dy) for p in self.points]

    def clone(self) -> "PolygonShape":
        c = PolygonShape(self.layer_index, [QPointF(p) for p in self.points])
        c.properties = dict(self.properties)
        return c

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["points"] = [[p.x(), p.y()] for p in self.points]
        return d


class PathShape(Shape):
    """Wire / path shape with a width."""

    def __init__(self, layer_index: int, points: List[QPointF], width: float = 100.0):
        super().__init__(layer_index)
        self.points = list(points)
        self.width = width

    def bounding_rect(self) -> QRectF:
        if not self.points:
            return QRectF()
        xs = [p.x() for p in self.points]
        ys = [p.y() for p in self.points]
        hw = self.width / 2.0
        return QRectF(min(xs) - hw, min(ys) - hw,
                      max(xs) - min(xs) + self.width,
                      max(ys) - min(ys) + self.width)

    def contains(self, pt: QPointF) -> bool:
        for i in range(len(self.points) - 1):
            a = np.array([self.points[i].x(), self.points[i].y()])
            b = np.array([self.points[i + 1].x(), self.points[i + 1].y()])
            p = np.array([pt.x(), pt.y()])
            ab = b - a
            ap = p - a
            t = np.clip(np.dot(ap, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
            closest = a + t * ab
            if np.linalg.norm(p - closest) <= self.width / 2.0:
                return True
        return False

    def translate(self, dx: float, dy: float):
        self.points = [QPointF(p.x() + dx, p.y() + dy) for p in self.points]

    def clone(self) -> "PathShape":
        c = PathShape(self.layer_index, [QPointF(p) for p in self.points], self.width)
        c.properties = dict(self.properties)
        return c

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["points"] = [[p.x(), p.y()] for p in self.points]
        d["width"] = self.width
        return d


class ViaShape(Shape):
    def __init__(self, layer_index: int, center: QPointF, size: float = 100.0):
        super().__init__(layer_index)
        self.center = QPointF(center)
        self.size = size

    def bounding_rect(self) -> QRectF:
        hs = self.size / 2.0
        return QRectF(self.center.x() - hs, self.center.y() - hs, self.size, self.size)

    def contains(self, pt: QPointF) -> bool:
        return self.bounding_rect().contains(pt)

    def translate(self, dx: float, dy: float):
        self.center = QPointF(self.center.x() + dx, self.center.y() + dy)

    def clone(self) -> "ViaShape":
        c = ViaShape(self.layer_index, QPointF(self.center), self.size)
        c.properties = dict(self.properties)
        return c

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({"cx": self.center.x(), "cy": self.center.y(), "size": self.size})
        return d


class TextShape(Shape):
    def __init__(self, layer_index: int, position: QPointF, text: str,
                 font_size: float = 200.0):
        super().__init__(layer_index)
        self.position = QPointF(position)
        self.text = text
        self.font_size = font_size

    def bounding_rect(self) -> QRectF:
        w = len(self.text) * self.font_size * 0.6
        return QRectF(self.position.x(), self.position.y() - self.font_size,
                      w, self.font_size * 1.2)

    def contains(self, pt: QPointF) -> bool:
        return self.bounding_rect().contains(pt)

    def translate(self, dx: float, dy: float):
        self.position = QPointF(self.position.x() + dx, self.position.y() + dy)

    def clone(self) -> "TextShape":
        c = TextShape(self.layer_index, QPointF(self.position), self.text, self.font_size)
        c.properties = dict(self.properties)
        return c

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({"x": self.position.x(), "y": self.position.y(),
                  "text": self.text, "font_size": self.font_size})
        return d


# ---------------------------------------------------------------------------
# Cell (hierarchical unit)
# ---------------------------------------------------------------------------

class Cell:
    """A cell is a named collection of shapes and cell instances."""

    def __init__(self, name: str):
        self.name = name
        self.shapes: List[Shape] = []
        self.instances: List[Tuple[str, QPointF]] = []  # (cell_name, origin)

    def add_shape(self, shape: Shape):
        self.shapes.append(shape)

    def remove_shape(self, shape: Shape):
        if shape in self.shapes:
            self.shapes.remove(shape)

    def add_instance(self, cell_name: str, origin: QPointF):
        self.instances.append((cell_name, origin))

    def bounding_rect(self) -> QRectF:
        if not self.shapes:
            return QRectF(0, 0, 1000, 1000)
        rects = [s.bounding_rect() for s in self.shapes]
        x0 = min(r.left() for r in rects)
        y0 = min(r.top() for r in rects)
        x1 = max(r.right() for r in rects)
        y1 = max(r.bottom() for r in rects)
        return QRectF(x0, y0, x1 - x0, y1 - y0)


# ---------------------------------------------------------------------------
# Design Rule Check
# ---------------------------------------------------------------------------

class DRCViolation:
    def __init__(self, message: str, shapes: List[Shape], region: QRectF):
        self.message = message
        self.shapes = shapes
        self.region = region


class DesignRuleChecker:
    """Simple DRC engine checking minimum width and minimum spacing."""

    def __init__(self):
        self.min_width: Dict[int, float] = {
            0: 200, 1: 180, 2: 100, 3: 200, 4: 100, 5: 200
        }
        self.min_spacing: Dict[int, float] = {
            0: 200, 1: 200, 2: 150, 3: 200, 4: 150, 5: 200
        }

    def check(self, shapes: List[Shape]) -> List[DRCViolation]:
        violations: List[DRCViolation] = []
        for s in shapes:
            if isinstance(s, RectShape):
                li = s.layer_index
                mw = self.min_width.get(li, 100)
                if s.rect.width() < mw:
                    violations.append(DRCViolation(
                        f"Width {s.rect.width():.0f} < min {mw:.0f} on {li}",
                        [s], s.bounding_rect()))
                if s.rect.height() < mw:
                    violations.append(DRCViolation(
                        f"Height {s.rect.height():.0f} < min {mw:.0f} on {li}",
                        [s], s.bounding_rect()))
            elif isinstance(s, PathShape):
                li = s.layer_index
                mw = self.min_width.get(li, 100)
                if s.width < mw:
                    violations.append(DRCViolation(
                        f"Path width {s.width:.0f} < min {mw:.0f} on {li}",
                        [s], s.bounding_rect()))

        # Spacing check between same-layer rectangles
        rects_by_layer: Dict[int, List[RectShape]] = {}
        for s in shapes:
            if isinstance(s, RectShape):
                rects_by_layer.setdefault(s.layer_index, []).append(s)

        for li, rlist in rects_by_layer.items():
            ms = self.min_spacing.get(li, 100)
            for i in range(len(rlist)):
                for j in range(i + 1, len(rlist)):
                    r1, r2 = rlist[i].rect, rlist[j].rect
                    dx = max(0, max(r1.left(), r2.left()) - min(r1.right(), r2.right()))
                    dy = max(0, max(r1.top(), r2.top()) - min(r1.bottom(), r2.bottom()))
                    dist = math.sqrt(dx * dx + dy * dy)
                    if 0 < dist < ms:
                        union = r1.united(r2)
                        violations.append(DRCViolation(
                            f"Spacing {dist:.0f} < min {ms:.0f} on layer {li}",
                            [rlist[i], rlist[j]], union))
        return violations


# ---------------------------------------------------------------------------
# GDS-II binary export helpers
# ---------------------------------------------------------------------------

class GDSExporter:
    """Minimal GDS-II stream format writer."""

    def __init__(self, db_unit: float = 1e-9, user_unit: float = 1e-3):
        self.db_unit = db_unit
        self.user_unit = user_unit

    @staticmethod
    def _pack_record(rec_type: int, data: bytes = b"") -> bytes:
        length = len(data) + 4
        return struct.pack(">HBB", length, rec_type >> 8, rec_type & 0xFF) + data

    @staticmethod
    def _pack_int16(values: List[int]) -> bytes:
        return b"".join(struct.pack(">h", v) for v in values)

    @staticmethod
    def _pack_int32(values: List[int]) -> bytes:
        return b"".join(struct.pack(">i", v) for v in values)

    @staticmethod
    def _pack_ascii(text: str) -> bytes:
        b = text.encode("ascii")
        if len(b) % 2:
            b += b"\x00"
        return b

    @staticmethod
    def _pack_real8(value: float) -> bytes:
        if value == 0:
            return b"\x00" * 8
        sign = 0
        if value < 0:
            sign = 0x80
            value = -value
        exp = 0
        mantissa = value
        while mantissa >= 1.0:
            mantissa /= 16.0
            exp += 1
        while mantissa < 1.0 / 16.0:
            mantissa *= 16.0
            exp -= 1
        mant_int = int(mantissa * (2 ** 56))
        exp_byte = (exp + 64) & 0x7F
        result = struct.pack(">Q", mant_int)
        result = bytes([sign | exp_byte]) + result[1:]
        return result

    def export_cells(self, cells: Dict[str, Cell], path: str):
        """Write cells to a GDS-II file."""
        buf = bytearray()
        # Header
        buf += self._pack_record(0x0002, self._pack_int16([600]))  # HEADER
        buf += self._pack_record(0x0102, self._pack_int16([1, 1, 1, 1, 1, 1,
                                                           1, 1, 1, 1, 1, 1]))  # BGNLIB
        buf += self._pack_record(0x0206, self._pack_ascii("LAYOUT"))  # LIBNAME
        buf += self._pack_record(0x0305,
                                 self._pack_real8(self.db_unit) + self._pack_real8(
                                     self.db_unit / self.user_unit))  # UNITS

        for cell in cells.values():
            buf += self._pack_record(0x0502, self._pack_int16([1, 1, 1, 1, 1, 1,
                                                                1, 1, 1, 1, 1, 1]))  # BGNSTR
            buf += self._pack_record(0x0606, self._pack_ascii(cell.name))  # STRNAME

            for shape in cell.shapes:
                if isinstance(shape, RectShape):
                    buf += self._write_boundary(shape)
                elif isinstance(shape, PolygonShape):
                    buf += self._write_polygon(shape)
                elif isinstance(shape, PathShape):
                    buf += self._write_path(shape)
                elif isinstance(shape, TextShape):
                    buf += self._write_text(shape)

            for inst_name, origin in cell.instances:
                buf += self._write_sref(inst_name, origin)

            buf += self._pack_record(0x0700)  # ENDSTR

        buf += self._pack_record(0x0400)  # ENDLIB

        with open(path, "wb") as f:
            f.write(bytes(buf))

    def _write_boundary(self, shape: RectShape) -> bytes:
        buf = bytearray()
        buf += self._pack_record(0x0800)  # BOUNDARY
        buf += self._pack_record(0x0D02,
                                 self._pack_int16([shape.layer_index]))  # LAYER
        buf += self._pack_record(0x0E02, self._pack_int16([0]))  # DATATYPE
        r = shape.rect
        coords = [
            int(r.left()), int(r.top()),
            int(r.right()), int(r.top()),
            int(r.right()), int(r.bottom()),
            int(r.left()), int(r.bottom()),
            int(r.left()), int(r.top()),
        ]
        buf += self._pack_record(0x1003, self._pack_int32(coords))  # XY
        buf += self._pack_record(0x1100)  # ENDEL
        return bytes(buf)

    def _write_polygon(self, shape: PolygonShape) -> bytes:
        buf = bytearray()
        buf += self._pack_record(0x0800)  # BOUNDARY
        buf += self._pack_record(0x0D02,
                                 self._pack_int16([shape.layer_index]))
        buf += self._pack_record(0x0E02, self._pack_int16([0]))
        coords = []
        for p in shape.points:
            coords.extend([int(p.x()), int(p.y())])
        if shape.points:
            coords.extend([int(shape.points[0].x()), int(shape.points[0].y())])
        buf += self._pack_record(0x1003, self._pack_int32(coords))
        buf += self._pack_record(0x1100)
        return bytes(buf)

    def _write_path(self, shape: PathShape) -> bytes:
        buf = bytearray()
        buf += self._pack_record(0x0900)  # PATH
        buf += self._pack_record(0x0D02,
                                 self._pack_int16([shape.layer_index]))
        buf += self._pack_record(0x0E02, self._pack_int16([0]))
        buf += self._pack_record(0x0F03,
                                 self._pack_int32([int(shape.width)]))  # WIDTH
        coords = []
        for p in shape.points:
            coords.extend([int(p.x()), int(p.y())])
        buf += self._pack_record(0x1003, self._pack_int32(coords))
        buf += self._pack_record(0x1100)
        return bytes(buf)

    def _write_text(self, shape: TextShape) -> bytes:
        buf = bytearray()
        buf += self._pack_record(0x0C00)  # TEXT
        buf += self._pack_record(0x0D02,
                                 self._pack_int16([shape.layer_index]))
        buf += self._pack_record(0x1602, self._pack_int16([0]))  # TEXTTYPE
        buf += self._pack_record(0x1003,
                                 self._pack_int32([int(shape.position.x()),
                                                   int(shape.position.y())]))
        buf += self._pack_record(0x1906,
                                 self._pack_ascii(shape.text))  # STRING
        buf += self._pack_record(0x1100)
        return bytes(buf)

    def _write_sref(self, name: str, origin: QPointF) -> bytes:
        buf = bytearray()
        buf += self._pack_record(0x0A00)  # SREF
        buf += self._pack_record(0x1206, self._pack_ascii(name))  # SNAME
        buf += self._pack_record(0x1003,
                                 self._pack_int32([int(origin.x()),
                                                   int(origin.y())]))
        buf += self._pack_record(0x1100)
        return bytes(buf)


# ---------------------------------------------------------------------------
# Layout Canvas Widget (the drawing surface)
# ---------------------------------------------------------------------------

class LayoutCanvas(QWidget):
    """Canvas for rendering and interacting with IC layout shapes."""

    coordinate_changed = pyqtSignal(float, float)
    selection_changed = pyqtSignal()

    def __init__(self, layers: List[LayerDef], parent=None):
        super().__init__(parent)
        self.layers = layers
        self.cells: Dict[str, Cell] = {"TOP": Cell("TOP")}
        self.current_cell_name = "TOP"
        self.active_layer = 3  # Metal1 by default

        # View state
        self._zoom = 1.0
        self._pan_offset = QPointF(0, 0)
        self._panning = False
        self._pan_start = QPointF()

        # Grid
        self.grid_spacing = 100.0  # in database units (nm)
        self.grid_visible = True
        self.snap_enabled = True

        # Tool state
        self.current_tool = DrawTool.SELECT
        self._draw_start: Optional[QPointF] = None
        self._draw_points: List[QPointF] = []
        self._mouse_pos = QPointF()
        self._drag_offset = QPointF()
        self._moving = False

        # DRC
        self.drc = DesignRuleChecker()
        self.drc_violations: List[DRCViolation] = []
        self.show_drc = True

        # Path/wire width
        self.path_width = 200.0
        self.via_size = 200.0
        self.text_label = "Label"
        self.font_size = 300.0

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(400, 300)

    @property
    def current_cell(self) -> Cell:
        return self.cells[self.current_cell_name]

    def snap(self, pt: QPointF) -> QPointF:
        if not self.snap_enabled or self.grid_spacing <= 0:
            return pt
        g = self.grid_spacing
        return QPointF(round(pt.x() / g) * g, round(pt.y() / g) * g)

    def screen_to_world(self, pos: QPointF) -> QPointF:
        return QPointF((pos.x() - self._pan_offset.x()) / self._zoom,
                       (pos.y() - self._pan_offset.y()) / self._zoom)

    def world_to_screen(self, pt: QPointF) -> QPointF:
        return QPointF(pt.x() * self._zoom + self._pan_offset.x(),
                       pt.y() * self._zoom + self._pan_offset.y())

    # ---- Painting ----

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(20, 20, 30))

        painter.save()
        painter.translate(self._pan_offset)
        painter.scale(self._zoom, self._zoom)

        if self.grid_visible and self._zoom > 0.05:
            self._draw_grid(painter)

        self._draw_cell(painter, self.current_cell, QPointF(0, 0))

        # Draw in-progress shape preview
        self._draw_preview(painter)

        # DRC violations
        if self.show_drc:
            self._draw_drc(painter)

        painter.restore()

    def _draw_grid(self, painter: QPainter):
        g = self.grid_spacing
        if g <= 0:
            return
        pen = QPen(QColor(50, 50, 60), 0)
        painter.setPen(pen)
        view_rect = QRectF(
            -self._pan_offset.x() / self._zoom,
            -self._pan_offset.y() / self._zoom,
            self.width() / self._zoom,
            self.height() / self._zoom
        )
        x0 = math.floor(view_rect.left() / g) * g
        y0 = math.floor(view_rect.top() / g) * g
        x = x0
        max_lines = 500
        count = 0
        while x <= view_rect.right() and count < max_lines:
            painter.drawLine(QPointF(x, view_rect.top()), QPointF(x, view_rect.bottom()))
            x += g
            count += 1
        y = y0
        count = 0
        while y <= view_rect.bottom() and count < max_lines:
            painter.drawLine(QPointF(view_rect.left(), y), QPointF(view_rect.right(), y))
            y += g
            count += 1

    def _draw_cell(self, painter: QPainter, cell: Cell, origin: QPointF):
        painter.save()
        painter.translate(origin)
        for shape in cell.shapes:
            layer = self.layers[shape.layer_index] if shape.layer_index < len(self.layers) else None
            if layer and not layer.visible:
                continue
            self._draw_shape(painter, shape, layer)
        # Draw instances
        for inst_name, inst_origin in cell.instances:
            if inst_name in self.cells:
                self._draw_cell(painter, self.cells[inst_name], inst_origin)
        painter.restore()

    def _draw_shape(self, painter: QPainter, shape: Shape, layer: Optional[LayerDef]):
        if layer is None:
            pen_color = QColor(200, 200, 200)
            fill = QBrush(QColor(200, 200, 200, 40))
        else:
            pen_color = layer.color
            fill = QBrush(layer.fill_color, layer.pattern)

        pen = QPen(pen_color, max(1, 2.0 / self._zoom))
        if shape.selected:
            pen.setStyle(Qt.DashLine)
            pen.setColor(QColor(255, 255, 255))
            pen.setWidthF(max(1, 3.0 / self._zoom))
        painter.setPen(pen)
        painter.setBrush(fill)

        if isinstance(shape, RectShape):
            painter.drawRect(shape.rect)
        elif isinstance(shape, PolygonShape):
            if len(shape.points) >= 3:
                painter.drawPolygon(QPolygonF(shape.points))
        elif isinstance(shape, PathShape):
            if len(shape.points) >= 2:
                path = QPainterPath()
                path.moveTo(shape.points[0])
                for p in shape.points[1:]:
                    path.lineTo(p)
                old_pen = QPen(pen)
                old_pen.setWidthF(shape.width)
                old_pen.setCapStyle(Qt.FlatCap)
                old_pen.setJoinStyle(Qt.MiterJoin)
                painter.setPen(old_pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawPath(path)
        elif isinstance(shape, ViaShape):
            painter.drawRect(shape.bounding_rect())
            # Draw cross
            br = shape.bounding_rect()
            painter.drawLine(QPointF(br.left(), br.top()),
                             QPointF(br.right(), br.bottom()))
            painter.drawLine(QPointF(br.right(), br.top()),
                             QPointF(br.left(), br.bottom()))
        elif isinstance(shape, TextShape):
            font = QFont("Monospace", max(1, int(shape.font_size)))
            font.setPixelSize(max(1, int(shape.font_size)))
            painter.setFont(font)
            painter.drawText(shape.position, shape.text)

    def _draw_preview(self, painter: QPainter):
        if self._draw_start is None and not self._draw_points:
            return
        layer = self.layers[self.active_layer] if self.active_layer < len(self.layers) else None
        pen = QPen(QColor(255, 255, 255, 180), max(1, 1.5 / self._zoom), Qt.DashDotLine)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(255, 255, 255, 30)))

        world_mouse = self.snap(self.screen_to_world(self._mouse_pos))

        if self.current_tool == DrawTool.RECTANGLE and self._draw_start:
            r = QRectF(self._draw_start, world_mouse).normalized()
            painter.drawRect(r)
        elif self.current_tool in (DrawTool.POLYGON, DrawTool.PATH) and self._draw_points:
            pts = self._draw_points + [world_mouse]
            if self.current_tool == DrawTool.POLYGON and len(pts) >= 3:
                painter.drawPolygon(QPolygonF(pts))
            else:
                path = QPainterPath()
                path.moveTo(pts[0])
                for p in pts[1:]:
                    path.lineTo(p)
                painter.drawPath(path)
        elif self.current_tool == DrawTool.VIA and self._draw_start:
            hs = self.via_size / 2
            painter.drawRect(QRectF(world_mouse.x() - hs, world_mouse.y() - hs,
                                    self.via_size, self.via_size))

    def _draw_drc(self, painter: QPainter):
        pen = QPen(QColor(255, 0, 0), max(1, 2.0 / self._zoom), Qt.DotLine)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(255, 0, 0, 30)))
        for v in self.drc_violations:
            painter.drawRect(v.region.adjusted(-50, -50, 50, 50))

    # ---- Mouse handling ----

    def wheelEvent(self, event: QWheelEvent):
        old_world = self.screen_to_world(QPointF(event.pos()))
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self._zoom = max(0.001, min(self._zoom * factor, 500.0))
        new_screen = self.world_to_screen(old_world)
        self._pan_offset += QPointF(event.pos()) - new_screen
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        world = self.snap(self.screen_to_world(QPointF(event.pos())))

        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = QPointF(event.pos())
            return

        if event.button() == Qt.RightButton:
            if self._draw_points and self.current_tool in (DrawTool.POLYGON, DrawTool.PATH):
                self._finish_multi_point(world)
            return

        if event.button() != Qt.LeftButton:
            return

        if self.current_tool == DrawTool.SELECT:
            self._handle_select(world, event)
        elif self.current_tool == DrawTool.RECTANGLE:
            self._draw_start = world
        elif self.current_tool in (DrawTool.POLYGON, DrawTool.PATH):
            self._draw_points.append(world)
        elif self.current_tool == DrawTool.VIA:
            via = ViaShape(self.active_layer, world, self.via_size)
            self.current_cell.add_shape(via)
            self.update()
        elif self.current_tool == DrawTool.TEXT:
            txt = TextShape(self.active_layer, world, self.text_label, self.font_size)
            self.current_cell.add_shape(txt)
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        self._mouse_pos = QPointF(event.pos())
        world = self.screen_to_world(self._mouse_pos)
        self.coordinate_changed.emit(world.x(), world.y())

        if self._panning:
            delta = QPointF(event.pos()) - self._pan_start
            self._pan_offset += delta
            self._pan_start = QPointF(event.pos())
            self.update()
            return

        if self._moving:
            snapped = self.snap(world)
            dx = snapped.x() - self._drag_offset.x()
            dy = snapped.y() - self._drag_offset.y()
            for s in self.current_cell.shapes:
                if s.selected:
                    s.translate(dx, dy)
            self._drag_offset = snapped
            self.update()
            return

        if self._draw_start or self._draw_points:
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        world = self.snap(self.screen_to_world(QPointF(event.pos())))

        if event.button() == Qt.MiddleButton:
            self._panning = False
            return

        if event.button() == Qt.LeftButton:
            if self._moving:
                self._moving = False
                return

            if self.current_tool == DrawTool.RECTANGLE and self._draw_start:
                r = QRectF(self._draw_start, world).normalized()
                if r.width() > 0 and r.height() > 0:
                    rect = RectShape(self.active_layer, r)
                    self.current_cell.add_shape(rect)
                self._draw_start = None
                self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if self.current_tool in (DrawTool.POLYGON, DrawTool.PATH):
            world = self.snap(self.screen_to_world(QPointF(event.pos())))
            self._finish_multi_point(world)

    def _finish_multi_point(self, world: QPointF):
        if self.current_tool == DrawTool.POLYGON and len(self._draw_points) >= 3:
            poly = PolygonShape(self.active_layer, self._draw_points)
            self.current_cell.add_shape(poly)
        elif self.current_tool == DrawTool.PATH and len(self._draw_points) >= 2:
            path = PathShape(self.active_layer, self._draw_points, self.path_width)
            self.current_cell.add_shape(path)
        self._draw_points = []
        self.update()

    def _handle_select(self, world: QPointF, event: QMouseEvent):
        shift = event.modifiers() & Qt.ShiftModifier
        hit = None
        for s in reversed(self.current_cell.shapes):
            layer = self.layers[s.layer_index] if s.layer_index < len(self.layers) else None
            if layer and (not layer.visible or not layer.selectable):
                continue
            if s.contains(world):
                hit = s
                break

        if not shift:
            for s in self.current_cell.shapes:
                s.selected = False

        if hit:
            hit.selected = not hit.selected if shift else True
            self._moving = True
            self._drag_offset = world

        self.selection_changed.emit()
        self.update()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Delete:
            self.delete_selected()
        elif event.key() == Qt.Key_Escape:
            self._draw_start = None
            self._draw_points = []
            for s in self.current_cell.shapes:
                s.selected = False
            self.selection_changed.emit()
            self.update()
        elif event.key() == Qt.Key_C and event.modifiers() & Qt.ControlModifier:
            self.copy_selected()

    def delete_selected(self):
        self.current_cell.shapes = [s for s in self.current_cell.shapes if not s.selected]
        self.selection_changed.emit()
        self.update()

    def copy_selected(self):
        new_shapes = []
        for s in self.current_cell.shapes:
            if s.selected:
                c = s.clone()
                c.translate(self.grid_spacing * 2, self.grid_spacing * 2)
                c.selected = True
                s.selected = False
                new_shapes.append(c)
        self.current_cell.shapes.extend(new_shapes)
        self.selection_changed.emit()
        self.update()

    def get_selected(self) -> List[Shape]:
        return [s for s in self.current_cell.shapes if s.selected]

    def run_drc(self):
        self.drc_violations = self.drc.check(self.current_cell.shapes)
        self.update()

    def zoom_fit(self):
        br = self.current_cell.bounding_rect()
        if br.width() <= 0 or br.height() <= 0:
            return
        margin = 50
        zx = (self.width() - 2 * margin) / br.width()
        zy = (self.height() - 2 * margin) / br.height()
        self._zoom = min(zx, zy)
        cx = br.center()
        self._pan_offset = QPointF(self.width() / 2 - cx.x() * self._zoom,
                                   self.height() / 2 - cx.y() * self._zoom)
        self.update()


# ---------------------------------------------------------------------------
# Layer Panel
# ---------------------------------------------------------------------------

class LayerPanel(QWidget):
    layer_changed = pyqtSignal(int)
    visibility_changed = pyqtSignal()

    def __init__(self, layers: List[LayerDef], parent=None):
        super().__init__(parent)
        self.layers = layers
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        lbl = QLabel("Layers")
        lbl.setStyleSheet("font-weight: bold; color: #ccc;")
        layout.addWidget(lbl)

        self.table = QTableWidget(len(layers), 3)
        self.table.setHorizontalHeaderLabels(["Vis", "Color", "Name"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.table.setColumnWidth(0, 30)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self.table.setColumnWidth(1, 40)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setVisible(False)

        for i, layer in enumerate(layers):
            cb = QCheckBox()
            cb.setChecked(layer.visible)
            cb.stateChanged.connect(lambda state, idx=i: self._toggle_vis(idx, state))
            self.table.setCellWidget(i, 0, cb)

            color_item = QTableWidgetItem()
            color_item.setBackground(layer.color)
            color_item.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(i, 1, color_item)

            name_item = QTableWidgetItem(layer.name)
            name_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table.setItem(i, 2, name_item)

        self.table.cellClicked.connect(self._on_cell_click)
        layout.addWidget(self.table)
        self.setMaximumWidth(220)

    def _toggle_vis(self, idx: int, state: int):
        self.layers[idx].visible = state == Qt.Checked
        self.visibility_changed.emit()

    def _on_cell_click(self, row: int, col: int):
        self.layer_changed.emit(row)


# ---------------------------------------------------------------------------
# Properties Panel
# ---------------------------------------------------------------------------

class PropertiesPanel(QWidget):
    property_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._shape: Optional[Shape] = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        lbl = QLabel("Properties")
        lbl.setStyleSheet("font-weight: bold; color: #ccc;")
        layout.addWidget(lbl)
        self.form_layout = QFormLayout()
        layout.addLayout(self.form_layout)
        layout.addStretch()
        self.setMaximumWidth(240)
        self._fields: Dict[str, QLineEdit] = {}

    def show_shape(self, shape: Optional[Shape], layers: List[LayerDef]):
        self._shape = shape
        # Clear form
        while self.form_layout.rowCount() > 0:
            self.form_layout.removeRow(0)
        self._fields.clear()

        if shape is None:
            return

        layer_name = layers[shape.layer_index].name if shape.layer_index < len(layers) else "?"
        self._add_field("ID", str(shape.id), readonly=True)
        self._add_field("Type", shape.__class__.__name__, readonly=True)
        self._add_field("Layer", layer_name, readonly=True)

        if isinstance(shape, RectShape):
            self._add_field("X", f"{shape.rect.x():.1f}")
            self._add_field("Y", f"{shape.rect.y():.1f}")
            self._add_field("Width", f"{shape.rect.width():.1f}")
            self._add_field("Height", f"{shape.rect.height():.1f}")
        elif isinstance(shape, PathShape):
            self._add_field("PathWidth", f"{shape.width:.1f}")
            self._add_field("Points", str(len(shape.points)), readonly=True)
        elif isinstance(shape, ViaShape):
            self._add_field("CX", f"{shape.center.x():.1f}")
            self._add_field("CY", f"{shape.center.y():.1f}")
            self._add_field("Size", f"{shape.size:.1f}")
        elif isinstance(shape, TextShape):
            self._add_field("Text", shape.text)
            self._add_field("FontSize", f"{shape.font_size:.1f}")
        elif isinstance(shape, PolygonShape):
            self._add_field("Vertices", str(len(shape.points)), readonly=True)

        for key, val in shape.properties.items():
            self._add_field(f"p:{key}", str(val))

    def _add_field(self, label: str, value: str, readonly: bool = False):
        le = QLineEdit(value)
        le.setReadOnly(readonly)
        if not readonly:
            le.editingFinished.connect(self._apply_changes)
        le.setStyleSheet("color: #ddd; background: #333; border: 1px solid #555;")
        self._fields[label] = le
        lbl = QLabel(label)
        lbl.setStyleSheet("color: #aaa;")
        self.form_layout.addRow(lbl, le)

    def _apply_changes(self):
        s = self._shape
        if s is None:
            return
        try:
            if isinstance(s, RectShape):
                if "X" in self._fields:
                    s.rect.moveLeft(float(self._fields["X"].text()))
                if "Y" in self._fields:
                    s.rect.moveTop(float(self._fields["Y"].text()))
                if "Width" in self._fields:
                    s.rect.setWidth(float(self._fields["Width"].text()))
                if "Height" in self._fields:
                    s.rect.setHeight(float(self._fields["Height"].text()))
            elif isinstance(s, PathShape):
                if "PathWidth" in self._fields:
                    s.width = float(self._fields["PathWidth"].text())
            elif isinstance(s, ViaShape):
                if "CX" in self._fields:
                    s.center.setX(float(self._fields["CX"].text()))
                if "CY" in self._fields:
                    s.center.setY(float(self._fields["CY"].text()))
                if "Size" in self._fields:
                    s.size = float(self._fields["Size"].text())
            elif isinstance(s, TextShape):
                if "Text" in self._fields:
                    s.text = self._fields["Text"].text()
                if "FontSize" in self._fields:
                    s.font_size = float(self._fields["FontSize"].text())
        except ValueError:
            pass
        self.property_changed.emit()


# ---------------------------------------------------------------------------
# Cell Tree Panel
# ---------------------------------------------------------------------------

class CellTreePanel(QWidget):
    cell_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        lbl = QLabel("Cells")
        lbl.setStyleSheet("font-weight: bold; color: #ccc;")
        layout.addWidget(lbl)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("Cell Hierarchy")
        self.tree.itemClicked.connect(self._on_click)
        layout.addWidget(self.tree)

        btn_layout = QHBoxLayout()
        self.btn_new = QPushButton("New Cell")
        self.btn_inst = QPushButton("Instance")
        btn_layout.addWidget(self.btn_new)
        btn_layout.addWidget(self.btn_inst)
        layout.addLayout(btn_layout)
        self.setMaximumWidth(220)

    def refresh(self, cells: Dict[str, Cell]):
        self.tree.clear()
        for name, cell in cells.items():
            item = QTreeWidgetItem([name])
            item.setData(0, Qt.UserRole, name)
            for inst_name, origin in cell.instances:
                child = QTreeWidgetItem([f"{inst_name} @ ({origin.x():.0f},{origin.y():.0f})"])
                item.addChild(child)
            self.tree.addTopLevelItem(item)
        self.tree.expandAll()

    def _on_click(self, item: QTreeWidgetItem, col: int):
        name = item.data(0, Qt.UserRole)
        if name:
            self.cell_selected.emit(name)


# ---------------------------------------------------------------------------
# Main ICLayoutWidget
# ---------------------------------------------------------------------------

class ICLayoutWidget(QWidget):
    """Top-level IC Layout Editor widget for integration in a PyQt5 suite."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._logger: Optional[Callable] = None
        self.layers = [LayerDef(l.name, l.gds_layer, l.gds_datatype,
                                l.color, l.pattern) for l in DEFAULT_LAYERS]

        self._init_ui()
        self._connect_signals()
        self._log("IC Layout Editor initialized")

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Create canvas first (toolbar references it)
        self.canvas = LayoutCanvas(self.layers)

        # Toolbar
        self.toolbar = QToolBar("Layout Tools")
        self._setup_toolbar()
        main_layout.addWidget(self.toolbar)

        # Central splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left: layer + cell panels
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.layer_panel = LayerPanel(self.layers)
        left_layout.addWidget(self.layer_panel)

        self.cell_panel = CellTreePanel()
        left_layout.addWidget(self.cell_panel)

        splitter.addWidget(left_panel)

        # Center: canvas
        splitter.addWidget(self.canvas)

        # Right: properties
        self.props_panel = PropertiesPanel()
        splitter.addWidget(self.props_panel)

        splitter.setSizes([200, 800, 220])
        main_layout.addWidget(splitter)

        # Status bar
        status = QHBoxLayout()
        self.coord_label = QLabel("X: 0  Y: 0")
        self.coord_label.setStyleSheet("color: #aaa; font-family: monospace;")
        self.tool_label = QLabel("Tool: Select")
        self.tool_label.setStyleSheet("color: #aaa;")
        self.layer_label = QLabel("Layer: Metal1")
        self.layer_label.setStyleSheet("color: #aaa;")
        self.drc_label = QLabel("DRC: -")
        self.drc_label.setStyleSheet("color: #aaa;")
        status.addWidget(self.coord_label)
        status.addStretch()
        status.addWidget(self.tool_label)
        status.addWidget(self.layer_label)
        status.addWidget(self.drc_label)
        main_layout.addLayout(status)

        self.cell_panel.refresh(self.canvas.cells)

    def _setup_toolbar(self):
        tool_group = QActionGroup(self)
        tool_group.setExclusive(True)

        tools = [
            ("Select", DrawTool.SELECT),
            ("Rect", DrawTool.RECTANGLE),
            ("Polygon", DrawTool.POLYGON),
            ("Path/Wire", DrawTool.PATH),
            ("Via", DrawTool.VIA),
            ("Text", DrawTool.TEXT),
        ]
        for name, tool in tools:
            act = QAction(name, self)
            act.setCheckable(True)
            act.setData(tool)
            act.triggered.connect(lambda checked, t=tool: self._set_tool(t))
            tool_group.addAction(act)
            self.toolbar.addAction(act)
            if tool == DrawTool.SELECT:
                act.setChecked(True)

        self.toolbar.addSeparator()

        # Grid spacing
        self.toolbar.addWidget(QLabel(" Grid:"))
        self.grid_spin = QDoubleSpinBox()
        self.grid_spin.setRange(1, 100000)
        self.grid_spin.setValue(100)
        self.grid_spin.setSuffix(" nm")
        self.grid_spin.valueChanged.connect(self._grid_changed)
        self.toolbar.addWidget(self.grid_spin)

        self.snap_cb = QCheckBox("Snap")
        self.snap_cb.setChecked(True)
        self.snap_cb.stateChanged.connect(
            lambda s: setattr(self.canvas, 'snap_enabled', s == Qt.Checked))
        self.toolbar.addWidget(self.snap_cb)

        self.grid_vis_cb = QCheckBox("Grid")
        self.grid_vis_cb.setChecked(True)
        self.grid_vis_cb.stateChanged.connect(
            lambda s: (setattr(self.canvas, 'grid_visible', s == Qt.Checked),
                       self.canvas.update()))
        self.toolbar.addWidget(self.grid_vis_cb)

        self.toolbar.addSeparator()

        # Path width
        self.toolbar.addWidget(QLabel(" Wire W:"))
        self.wire_spin = QDoubleSpinBox()
        self.wire_spin.setRange(1, 100000)
        self.wire_spin.setValue(200)
        self.wire_spin.setSuffix(" nm")
        self.wire_spin.valueChanged.connect(lambda v: setattr(self.canvas, 'path_width', v))
        self.toolbar.addWidget(self.wire_spin)

        self.toolbar.addSeparator()

        # Action buttons
        drc_act = QAction("Run DRC", self)
        drc_act.triggered.connect(self._run_drc)
        self.toolbar.addAction(drc_act)

        fit_act = QAction("Zoom Fit", self)
        fit_act.triggered.connect(self.canvas.zoom_fit)
        self.toolbar.addAction(fit_act)

        del_act = QAction("Delete", self)
        del_act.triggered.connect(self.canvas.delete_selected)
        self.toolbar.addAction(del_act)

        copy_act = QAction("Copy", self)
        copy_act.triggered.connect(self.canvas.copy_selected)
        self.toolbar.addAction(copy_act)

        export_act = QAction("Export GDS", self)
        export_act.triggered.connect(self._export_gds)
        self.toolbar.addAction(export_act)

        self.toolbar.addSeparator()

        # Generator actions
        pcell_act = QAction("PCell Gen", self)
        pcell_act.setToolTip("Parameterized Cell Generator (NMOS/PMOS)")
        pcell_act.triggered.connect(self._parameterized_cell_dialog)
        self.toolbar.addAction(pcell_act)

        route_act = QAction("Auto-Route", self)
        route_act.setToolTip("Manhattan auto-routing between two points")
        route_act.triggered.connect(self._auto_route_dialog)
        self.toolbar.addAction(route_act)

        array_act = QAction("Array Gen", self)
        array_act.setToolTip("Generate regular array of vias/contacts/cells")
        array_act.triggered.connect(self._array_generator_dialog)
        self.toolbar.addAction(array_act)

        xsect_act = QAction("X-Section", self)
        xsect_act.setToolTip("Cross-section viewer at a given X coordinate")
        xsect_act.triggered.connect(self._cross_section_dialog)
        self.toolbar.addAction(xsect_act)

        area_act = QAction("Area/Perim", self)
        area_act.setToolTip("Calculate area and perimeter of selected shapes")
        area_act.triggered.connect(self._calc_area_perimeter)
        self.toolbar.addAction(area_act)

        label_act = QAction("Auto-Label", self)
        label_act.setToolTip("Auto-label nets and pins")
        label_act.triggered.connect(self._auto_label)
        self.toolbar.addAction(label_act)

    def _connect_signals(self):
        self.canvas.coordinate_changed.connect(self._update_coords)
        self.canvas.selection_changed.connect(self._update_props)
        self.layer_panel.layer_changed.connect(self._layer_selected)
        self.layer_panel.visibility_changed.connect(self.canvas.update)
        self.props_panel.property_changed.connect(self.canvas.update)
        self.cell_panel.cell_selected.connect(self._switch_cell)
        self.cell_panel.btn_new.clicked.connect(self._new_cell)
        self.cell_panel.btn_inst.clicked.connect(self._add_instance)

    # ---- Slots ----

    def _set_tool(self, tool: DrawTool):
        self.canvas.current_tool = tool
        self.tool_label.setText(f"Tool: {tool.name}")
        self._log(f"Tool switched to {tool.name}")

    def _grid_changed(self, val: float):
        self.canvas.grid_spacing = val
        self.canvas.update()

    def _update_coords(self, x: float, y: float):
        self.coord_label.setText(f"X: {x:.1f}  Y: {y:.1f}")

    def _update_props(self):
        sel = self.canvas.get_selected()
        if len(sel) == 1:
            self.props_panel.show_shape(sel[0], self.layers)
        else:
            self.props_panel.show_shape(None, self.layers)

    def _layer_selected(self, idx: int):
        self.canvas.active_layer = idx
        name = self.layers[idx].name if idx < len(self.layers) else "?"
        self.layer_label.setText(f"Layer: {name}")
        self._log(f"Active layer: {name}")

    def _switch_cell(self, name: str):
        if name in self.canvas.cells:
            self.canvas.current_cell_name = name
            self.canvas.update()
            self._log(f"Switched to cell: {name}")

    def _new_cell(self):
        name, ok = QInputDialog.getText(self, "New Cell", "Cell name:")
        if ok and name and name not in self.canvas.cells:
            self.canvas.cells[name] = Cell(name)
            self.cell_panel.refresh(self.canvas.cells)
            self._log(f"Created cell: {name}")

    def _add_instance(self):
        names = [n for n in self.canvas.cells if n != self.canvas.current_cell_name]
        if not names:
            return
        name, ok = QInputDialog.getItem(self, "Insert Instance",
                                        "Select cell:", names, 0, False)
        if ok and name:
            self.canvas.current_cell.add_instance(name, QPointF(0, 0))
            self.cell_panel.refresh(self.canvas.cells)
            self.canvas.update()
            self._log(f"Instantiated {name} in {self.canvas.current_cell_name}")

    def _run_drc(self):
        self.canvas.run_drc()
        n = len(self.canvas.drc_violations)
        self.drc_label.setText(f"DRC: {n} violations" if n else "DRC: Clean")
        self.drc_label.setStyleSheet(
            "color: #f44;" if n else "color: #4f4;")
        self._log(f"DRC completed: {n} violations")

    def _export_gds(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export GDS-II",
                                              "", "GDS Files (*.gds)")
        if path:
            self.export(path)

    def _log(self, msg: str):
        if self._logger:
            self._logger(msg)

    # ---- Public API ----

    def set_logger(self, fn: Callable):
        """Set a logging callback function."""
        self._logger = fn

    def load_file(self, path: str):
        """Load layout from a JSON file."""
        if not os.path.isfile(path):
            self._log(f"File not found: {path}")
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.canvas.cells.clear()
            for cell_data in data.get("cells", []):
                cell = Cell(cell_data["name"])
                for sd in cell_data.get("shapes", []):
                    shape = self._shape_from_dict(sd)
                    if shape:
                        cell.shapes.append(shape)
                for inst in cell_data.get("instances", []):
                    cell.instances.append(
                        (inst["name"], QPointF(inst["x"], inst["y"])))
                self.canvas.cells[cell.name] = cell
            if not self.canvas.cells:
                self.canvas.cells["TOP"] = Cell("TOP")
            self.canvas.current_cell_name = list(self.canvas.cells.keys())[0]
            self.cell_panel.refresh(self.canvas.cells)
            self.canvas.zoom_fit()
            self._log(f"Loaded layout from {path}")
        except Exception as e:
            self._log(f"Load error: {e}")

    def export(self, path: str = ""):
        """Export layout to GDS-II. If path is empty, exports to 'layout.gds'."""
        if not path:
            path = "layout.gds"
        try:
            exporter = GDSExporter()
            exporter.export_cells(self.canvas.cells, path)
            self._log(f"Exported GDS-II to {path}")
        except Exception as e:
            self._log(f"Export error: {e}")

    def save_json(self, path: str):
        """Save layout as JSON for re-loading."""
        data = {"cells": []}
        for name, cell in self.canvas.cells.items():
            cd = {"name": name, "shapes": [], "instances": []}
            for s in cell.shapes:
                cd["shapes"].append(s.to_dict())
            for inst_name, origin in cell.instances:
                cd["instances"].append({"name": inst_name,
                                        "x": origin.x(), "y": origin.y()})
            data["cells"].append(cd)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self._log(f"Saved JSON to {path}")

    # ---- Parameterized Cell Generator ----

    def _parameterized_cell_dialog(self):
        """Create a transistor layout (NMOS/PMOS) with W/L parameters."""
        from PyQt5.QtWidgets import QDialog, QFormLayout, QDialogButtonBox
        dlg = QDialog(self)
        dlg.setWindowTitle("Parameterized Cell Generator")
        form = QFormLayout(dlg)

        type_combo = QComboBox()
        type_combo.addItems(["NMOS", "PMOS"])
        form.addRow("Transistor Type:", type_combo)

        w_spin = QDoubleSpinBox()
        w_spin.setRange(100, 100000)
        w_spin.setValue(2000)
        w_spin.setSuffix(" nm")
        form.addRow("Width (W):", w_spin)

        l_spin = QDoubleSpinBox()
        l_spin.setRange(50, 50000)
        l_spin.setValue(500)
        l_spin.setSuffix(" nm")
        form.addRow("Length (L):", l_spin)

        x_spin = QDoubleSpinBox()
        x_spin.setRange(-1e6, 1e6)
        x_spin.setValue(0)
        form.addRow("Origin X:", x_spin)

        y_spin = QDoubleSpinBox()
        y_spin.setRange(-1e6, 1e6)
        y_spin.setValue(0)
        form.addRow("Origin Y:", y_spin)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)

        if dlg.exec_() != QDialog.Accepted:
            return

        tx_type = type_combo.currentText()
        W = w_spin.value()
        L = l_spin.value()
        ox = x_spin.value()
        oy = y_spin.value()
        cell_name = f"{tx_type}_W{int(W)}_L{int(L)}"

        cell = Cell(cell_name)
        # Diffusion layer (layer 0): active area
        diff_margin = 200
        diff_rect = QRectF(ox - diff_margin, oy - diff_margin,
                           L + 2 * diff_margin, W + 2 * diff_margin)
        cell.add_shape(RectShape(0, diff_rect))

        # Poly layer (layer 1): gate
        poly_ext = 300
        poly_rect = QRectF(ox, oy - poly_ext, L, W + 2 * poly_ext)
        cell.add_shape(RectShape(1, poly_rect))

        # Contact layer (layer 2): source and drain contacts
        contact_size = 100
        contact_spacing = 300
        # Source contacts (left of gate)
        n_contacts = max(1, int(W / contact_spacing))
        for i in range(n_contacts):
            cy = oy + (i + 0.5) * W / n_contacts - contact_size / 2
            cell.add_shape(ViaShape(2, QPointF(ox - diff_margin / 2, cy + contact_size / 2), contact_size))

        # Drain contacts (right of gate)
        for i in range(n_contacts):
            cy = oy + (i + 0.5) * W / n_contacts - contact_size / 2
            cell.add_shape(ViaShape(2, QPointF(ox + L + diff_margin / 2, cy + contact_size / 2), contact_size))

        # Metal1 layer (layer 3): source and drain straps
        m1_width = 200
        cell.add_shape(RectShape(3, QRectF(ox - diff_margin - m1_width / 2, oy,
                                            m1_width, W)))  # source
        cell.add_shape(RectShape(3, QRectF(ox + L + diff_margin - m1_width / 2, oy,
                                            m1_width, W)))  # drain

        # Gate label
        cell.add_shape(TextShape(1, QPointF(ox + L / 2, oy - poly_ext - 100),
                                 "G", 150))
        cell.add_shape(TextShape(3, QPointF(ox - diff_margin, oy - 200),
                                 "S", 150))
        cell.add_shape(TextShape(3, QPointF(ox + L + diff_margin, oy - 200),
                                 "D", 150))

        self.canvas.cells[cell_name] = cell
        self.canvas.current_cell.add_instance(cell_name, QPointF(0, 0))
        self.cell_panel.refresh(self.canvas.cells)
        self.canvas.update()
        self._log(f"Generated {tx_type} pcell: W={W:.0f} L={L:.0f}")

    # ---- Auto-Routing ----

    def _auto_route_dialog(self):
        """Simple Manhattan routing between two points on a metal layer."""
        from PyQt5.QtWidgets import QDialog, QFormLayout, QDialogButtonBox
        dlg = QDialog(self)
        dlg.setWindowTitle("Manhattan Auto-Route")
        form = QFormLayout(dlg)

        x1 = QDoubleSpinBox(); x1.setRange(-1e6, 1e6); x1.setValue(0)
        y1 = QDoubleSpinBox(); y1.setRange(-1e6, 1e6); y1.setValue(0)
        x2 = QDoubleSpinBox(); x2.setRange(-1e6, 1e6); x2.setValue(2000)
        y2 = QDoubleSpinBox(); y2.setRange(-1e6, 1e6); y2.setValue(1000)
        form.addRow("Start X:", x1)
        form.addRow("Start Y:", y1)
        form.addRow("End X:", x2)
        form.addRow("End Y:", y2)

        layer_combo = QComboBox()
        layer_combo.addItems([l.name for l in self.layers])
        layer_combo.setCurrentIndex(3)  # Metal1
        form.addRow("Layer:", layer_combo)

        width_spin = QDoubleSpinBox()
        width_spin.setRange(50, 10000)
        width_spin.setValue(200)
        width_spin.setSuffix(" nm")
        form.addRow("Wire Width:", width_spin)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)

        if dlg.exec_() != QDialog.Accepted:
            return

        p1 = QPointF(x1.value(), y1.value())
        p2 = QPointF(x2.value(), y2.value())
        mid = QPointF(x2.value(), y1.value())  # L-shaped route
        li = layer_combo.currentIndex()
        w = width_spin.value()

        path = PathShape(li, [p1, mid, p2], w)
        self.canvas.current_cell.add_shape(path)
        self.canvas.update()
        self._log(f"Auto-routed: ({x1.value():.0f},{y1.value():.0f}) -> ({x2.value():.0f},{y2.value():.0f})")

    # ---- Array Generator ----

    def _array_generator_dialog(self):
        """Create a regular array of vias, contacts, or cells."""
        from PyQt5.QtWidgets import QDialog, QFormLayout, QDialogButtonBox
        dlg = QDialog(self)
        dlg.setWindowTitle("Array Generator")
        form = QFormLayout(dlg)

        elem_combo = QComboBox()
        elem_combo.addItems(["Via", "Contact", "Rectangle"])
        form.addRow("Element:", elem_combo)

        layer_combo = QComboBox()
        layer_combo.addItems([l.name for l in self.layers])
        layer_combo.setCurrentIndex(4)  # Via
        form.addRow("Layer:", layer_combo)

        cols_spin = QSpinBox(); cols_spin.setRange(1, 100); cols_spin.setValue(4)
        rows_spin = QSpinBox(); rows_spin.setRange(1, 100); rows_spin.setValue(4)
        form.addRow("Columns:", cols_spin)
        form.addRow("Rows:", rows_spin)

        pitch_x = QDoubleSpinBox(); pitch_x.setRange(1, 100000); pitch_x.setValue(300)
        pitch_y = QDoubleSpinBox(); pitch_y.setRange(1, 100000); pitch_y.setValue(300)
        form.addRow("Pitch X (nm):", pitch_x)
        form.addRow("Pitch Y (nm):", pitch_y)

        size_spin = QDoubleSpinBox(); size_spin.setRange(10, 10000); size_spin.setValue(100)
        form.addRow("Element Size:", size_spin)

        ox_spin = QDoubleSpinBox(); ox_spin.setRange(-1e6, 1e6); ox_spin.setValue(0)
        oy_spin = QDoubleSpinBox(); oy_spin.setRange(-1e6, 1e6); oy_spin.setValue(0)
        form.addRow("Origin X:", ox_spin)
        form.addRow("Origin Y:", oy_spin)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)

        if dlg.exec_() != QDialog.Accepted:
            return

        elem_type = elem_combo.currentText()
        li = layer_combo.currentIndex()
        nc, nr = cols_spin.value(), rows_spin.value()
        px, py = pitch_x.value(), pitch_y.value()
        sz = size_spin.value()
        ox, oy = ox_spin.value(), oy_spin.value()

        count = 0
        for r in range(nr):
            for c in range(nc):
                cx = ox + c * px
                cy = oy + r * py
                if elem_type == "Via" or elem_type == "Contact":
                    self.canvas.current_cell.add_shape(ViaShape(li, QPointF(cx, cy), sz))
                else:
                    self.canvas.current_cell.add_shape(
                        RectShape(li, QRectF(cx - sz / 2, cy - sz / 2, sz, sz)))
                count += 1

        self.canvas.update()
        self._log(f"Generated {nc}x{nr} array ({count} elements)")

    # ---- Cross-Section Viewer ----

    def _cross_section_dialog(self):
        """Show vertical stack cross-section at a given X coordinate."""
        from PyQt5.QtWidgets import QDialog, QFormLayout, QDialogButtonBox
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as _FC
        from matplotlib.figure import Figure as _Fig

        dlg = QDialog(self)
        dlg.setWindowTitle("Cross-Section Viewer")
        dlg.setMinimumSize(600, 400)
        main_lay = QVBoxLayout(dlg)

        # Input
        input_lay = QHBoxLayout()
        input_lay.addWidget(QLabel("X Coordinate:"))
        x_spin = QDoubleSpinBox()
        x_spin.setRange(-1e6, 1e6)
        x_spin.setValue(500)
        input_lay.addWidget(x_spin)
        draw_btn = QPushButton("Draw Cross-Section")
        input_lay.addWidget(draw_btn)
        main_lay.addLayout(input_lay)

        fig = _Fig(figsize=(8, 4), tight_layout=True)
        canvas = _FC(fig)
        main_lay.addWidget(canvas)

        # Layer stack definition (vertical positions)
        layer_stack = {
            0: {"name": "Diffusion", "y_base": 0, "height": 50, "color": "#FFFF00"},
            1: {"name": "Poly", "y_base": 50, "height": 30, "color": "#FF0000"},
            2: {"name": "Contact", "y_base": 80, "height": 40, "color": "#666666"},
            3: {"name": "Metal1", "y_base": 120, "height": 50, "color": "#0064FF"},
            4: {"name": "Via", "y_base": 170, "height": 40, "color": "#B400B4"},
            5: {"name": "Metal2", "y_base": 210, "height": 50, "color": "#00C864"},
        }

        def do_draw():
            ax = fig.gca()
            ax.clear()
            x_cut = x_spin.value()
            shapes = self.canvas.current_cell.shapes

            # Find shapes intersected by the vertical line x = x_cut
            for shape in shapes:
                br = shape.bounding_rect()
                if br.left() <= x_cut <= br.right():
                    li = shape.layer_index
                    if li not in layer_stack:
                        continue
                    ls = layer_stack[li]
                    y_lo = br.top()
                    y_hi = br.bottom()
                    ax.barh(ls["y_base"] + ls["height"] / 2, y_hi - y_lo,
                            height=ls["height"], left=y_lo,
                            color=ls["color"], edgecolor="black", alpha=0.7)

            # Draw substrate
            ax.axhspan(-20, 0, color="#C0A060", alpha=0.3, label="Substrate")
            for li, ls in layer_stack.items():
                ax.axhline(ls["y_base"], color="#ccc", lw=0.5, ls=":")
                ax.text(-100, ls["y_base"] + ls["height"] / 2, ls["name"],
                        fontsize=7, va="center", ha="right")

            ax.set_xlabel("Y Position (nm)")
            ax.set_ylabel("Layer Stack")
            ax.set_title(f"Cross-Section at X = {x_cut:.0f} nm")
            ax.set_yticks([])
            canvas.draw_idle()

        draw_btn.clicked.connect(do_draw)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        main_lay.addWidget(close_btn)
        do_draw()
        dlg.exec_()

    # ---- Area and Perimeter Calculation ----

    def _calc_area_perimeter(self):
        """Calculate area and perimeter of selected shapes."""
        selected = self.canvas.get_selected()
        if not selected:
            QMessageBox.information(self, "No Selection", "Select shapes first.")
            return

        lines = ["Area / Perimeter Calculation", "=" * 40]
        total_area = 0
        total_perim = 0

        for s in selected:
            layer_name = self.layers[s.layer_index].name if s.layer_index < len(self.layers) else "?"
            if isinstance(s, RectShape):
                w, h = s.rect.width(), s.rect.height()
                area = w * h
                perim = 2 * (w + h)
            elif isinstance(s, PolygonShape) and len(s.points) >= 3:
                # Shoelace formula
                pts = s.points
                n = len(pts)
                area = 0
                perim = 0
                for i in range(n):
                    j = (i + 1) % n
                    area += pts[i].x() * pts[j].y()
                    area -= pts[j].x() * pts[i].y()
                    dx = pts[j].x() - pts[i].x()
                    dy = pts[j].y() - pts[i].y()
                    perim += math.sqrt(dx * dx + dy * dy)
                area = abs(area) / 2.0
            elif isinstance(s, ViaShape):
                area = s.size * s.size
                perim = 4 * s.size
            else:
                area = 0
                perim = 0

            total_area += area
            total_perim += perim
            lines.append(f"  {s.__class__.__name__} (ID={s.id}, {layer_name}):")
            lines.append(f"    Area = {area:.1f} nm^2 ({area / 1e6:.4f} um^2)")
            lines.append(f"    Perimeter = {perim:.1f} nm")

        lines.append("-" * 40)
        lines.append(f"  Total Area = {total_area:.1f} nm^2 ({total_area / 1e6:.4f} um^2)")
        lines.append(f"  Total Perimeter = {total_perim:.1f} nm")

        QMessageBox.information(self, "Area / Perimeter", "\n".join(lines))
        self._log(f"Area calculation: {len(selected)} shapes, total area={total_area / 1e6:.4f} um^2")

    # ---- Auto-Label Generator ----

    def _auto_label(self):
        """Auto-label nets and pins on the current cell."""
        shapes = self.canvas.current_cell.shapes
        net_counter = {"Diffusion": 0, "Poly": 0, "Contact": 0,
                       "Metal1": 0, "Via": 0, "Metal2": 0}
        labels_added = 0

        for s in shapes:
            if isinstance(s, TextShape):
                continue  # skip existing labels
            li = s.layer_index
            if li >= len(self.layers):
                continue
            layer_name = self.layers[li].name
            if layer_name not in net_counter:
                continue
            net_counter[layer_name] += 1
            br = s.bounding_rect()
            label_text = f"{layer_name[0]}{net_counter[layer_name]}"
            pos = QPointF(br.center().x(), br.top() - 100)
            text_shape = TextShape(li, pos, label_text, 150)
            self.canvas.current_cell.add_shape(text_shape)
            labels_added += 1

        self.canvas.update()
        self._log(f"Auto-labeled {labels_added} shapes")

    def _shape_from_dict(self, d: dict) -> Optional[Shape]:
        t = d.get("type", "")
        li = d.get("layer", 0)
        if t == "RectShape":
            return RectShape(li, QRectF(d["x"], d["y"], d["w"], d["h"]))
        elif t == "PolygonShape":
            pts = [QPointF(p[0], p[1]) for p in d.get("points", [])]
            return PolygonShape(li, pts)
        elif t == "PathShape":
            pts = [QPointF(p[0], p[1]) for p in d.get("points", [])]
            return PathShape(li, pts, d.get("width", 100))
        elif t == "ViaShape":
            return ViaShape(li, QPointF(d["cx"], d["cy"]), d.get("size", 100))
        elif t == "TextShape":
            return TextShape(li, QPointF(d["x"], d["y"]),
                             d.get("text", ""), d.get("font_size", 200))
        return None
