"""
CAD2DWidget - A 2D CAD widget for PyQt5 scientific suite (AutoCAD-like).
Provides drawing tools, layers, snap-to-grid, measurements, and export.
"""
import math
import struct
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QToolBar, QAction, QActionGroup,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget, QListWidgetItem,
    QSplitter, QGroupBox, QFormLayout, QColorDialog, QFileDialog, QCheckBox,
    QInputDialog, QMenu
)
from PyQt5.QtCore import Qt, QPointF, QRectF, QLineF, pyqtSignal
from PyQt5.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QPolygonF,
    QPainterPath, QImage, QWheelEvent, QMouseEvent, QKeyEvent,
    QTransform
)
try:
    from PyQt5.QtPrintSupport import QPrinter
except ImportError:
    QPrinter = None


# ---------------------------------------------------------------------------
# Simple DXF writer
# ---------------------------------------------------------------------------
class SimpleDXF:
    """Minimal DXF file writer for 2D entities."""

    @staticmethod
    def _header():
        return "0\nSECTION\n2\nENTITIES\n"

    @staticmethod
    def _footer():
        return "0\nENDSEC\n0\nEOF\n"

    @staticmethod
    def line(x1, y1, x2, y2, layer="0", color=7):
        return (f"0\nLINE\n8\n{layer}\n62\n{color}\n"
                f"10\n{x1}\n20\n{y1}\n30\n0\n"
                f"11\n{x2}\n21\n{y2}\n31\n0\n")

    @staticmethod
    def circle(cx, cy, r, layer="0", color=7):
        return (f"0\nCIRCLE\n8\n{layer}\n62\n{color}\n"
                f"10\n{cx}\n20\n{cy}\n30\n0\n40\n{r}\n")

    @staticmethod
    def arc(cx, cy, r, start_angle, end_angle, layer="0", color=7):
        return (f"0\nARC\n8\n{layer}\n62\n{color}\n"
                f"10\n{cx}\n20\n{cy}\n30\n0\n40\n{r}\n"
                f"50\n{start_angle}\n51\n{end_angle}\n")

    @staticmethod
    def text(x, y, height, string, layer="0", color=7):
        return (f"0\nTEXT\n8\n{layer}\n62\n{color}\n"
                f"10\n{x}\n20\n{y}\n30\n0\n40\n{height}\n1\n{string}\n")

    @classmethod
    def write(cls, entities, path):
        with open(path, "w") as f:
            f.write(cls._header())
            for e in entities:
                f.write(e)
            f.write(cls._footer())


# ---------------------------------------------------------------------------
# Drawing entities
# ---------------------------------------------------------------------------
class CADEntity:
    _uid = 0

    def __init__(self, etype, points, **kw):
        CADEntity._uid += 1
        self.uid = CADEntity._uid
        self.etype = etype            # line, rect, circle, arc, polygon, polyline, ellipse, text
        self.points = list(points)    # list of (x,y) tuples
        self.color = kw.get("color", QColor(255, 255, 255))
        self.line_width = kw.get("line_width", 2)
        self.line_style = kw.get("line_style", Qt.SolidLine)
        self.layer = kw.get("layer", "Layer 1")
        self.text = kw.get("text", "")
        self.radius = kw.get("radius", 0.0)
        self.start_angle = kw.get("start_angle", 0.0)
        self.end_angle = kw.get("end_angle", 360.0)
        self.rx = kw.get("rx", 0.0)
        self.ry = kw.get("ry", 0.0)
        self.selected = False

    def bounding_rect(self):
        if not self.points:
            return QRectF()
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        margin = max(self.radius, self.rx, self.ry, 10)
        return QRectF(min(xs) - margin, min(ys) - margin,
                      max(xs) - min(xs) + 2 * margin,
                      max(ys) - min(ys) + 2 * margin)

    def hit_test(self, px, py, tol=8):
        if self.etype == "line" and len(self.points) >= 2:
            return _point_line_dist(px, py, *self.points[0], *self.points[1]) < tol
        if self.etype in ("rect", "polygon", "polyline"):
            for i in range(len(self.points) - 1):
                if _point_line_dist(px, py, *self.points[i], *self.points[i + 1]) < tol:
                    return True
            if self.etype in ("rect", "polygon") and len(self.points) >= 3:
                if _point_line_dist(px, py, *self.points[-1], *self.points[0]) < tol:
                    return True
            return False
        if self.etype == "circle" and self.points:
            cx, cy = self.points[0]
            d = math.hypot(px - cx, py - cy)
            return abs(d - self.radius) < tol
        if self.etype == "ellipse" and self.points:
            cx, cy = self.points[0]
            if self.rx > 0 and self.ry > 0:
                v = ((px - cx) / self.rx) ** 2 + ((py - cy) / self.ry) ** 2
                return abs(v - 1.0) < tol / min(self.rx, self.ry)
            return False
        if self.etype == "arc" and self.points:
            cx, cy = self.points[0]
            d = math.hypot(px - cx, py - cy)
            return abs(d - self.radius) < tol
        if self.etype == "text" and self.points:
            r = QRectF(self.points[0][0], self.points[0][1] - 14, 100, 20)
            return r.contains(QPointF(px, py))
        return False


def _point_line_dist(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


# ---------------------------------------------------------------------------
# Hatch pattern generators
# ---------------------------------------------------------------------------
HATCH_PATTERNS = {
    "lines": "lines",
    "crosshatch": "crosshatch",
    "dots": "dots",
    "diagonal": "diagonal",
    "solid": "solid",
}


def _generate_hatch_path(rect: QRectF, pattern: str, spacing: float = 8.0) -> QPainterPath:
    """Generate a QPainterPath for a hatch fill inside *rect*."""
    path = QPainterPath()
    x0, y0 = rect.left(), rect.top()
    x1, y1 = rect.right(), rect.bottom()
    if pattern == "solid":
        path.addRect(rect)
    elif pattern == "lines":
        y = y0
        while y <= y1:
            path.moveTo(x0, y)
            path.lineTo(x1, y)
            y += spacing
    elif pattern == "crosshatch":
        y = y0
        while y <= y1:
            path.moveTo(x0, y)
            path.lineTo(x1, y)
            y += spacing
        x = x0
        while x <= x1:
            path.moveTo(x, y0)
            path.lineTo(x, y1)
            x += spacing
    elif pattern == "diagonal":
        d = 0.0
        total = (x1 - x0) + (y1 - y0)
        while d <= total:
            sx = x0 + min(d, x1 - x0)
            sy = y0 + max(0, d - (x1 - x0))
            ex = x0 + max(0, d - (y1 - y0))
            ey = y0 + min(d, y1 - y0)
            path.moveTo(sx, sy)
            path.lineTo(ex, ey)
            d += spacing
    elif pattern == "dots":
        y = y0
        while y <= y1:
            x = x0
            while x <= x1:
                path.addEllipse(QPointF(x, y), 1.0, 1.0)
                x += spacing
            y += spacing
    return path


# ---------------------------------------------------------------------------
# Dimension annotation helpers
# ---------------------------------------------------------------------------

def _draw_linear_dimension(painter: QPainter, p1, p2, offset=20.0, zoom=1.0):
    """Draw a linear dimension annotation between two world points (already screen-converted)."""
    sx1, sy1 = p1
    sx2, sy2 = p2
    dist = math.hypot((sx2 - sx1), (sy2 - sy1)) / zoom
    # Direction perpendicular to line
    dx, dy = sx2 - sx1, sy2 - sy1
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return
    nx, ny = -dy / length, dx / length
    # Offset points for dimension line
    ox1, oy1 = sx1 + nx * offset, sy1 + ny * offset
    ox2, oy2 = sx2 + nx * offset, sy2 + ny * offset
    pen = QPen(QColor(0, 200, 255), 1, Qt.SolidLine)
    painter.setPen(pen)
    # Extension lines
    painter.drawLine(QPointF(sx1, sy1), QPointF(ox1, oy1))
    painter.drawLine(QPointF(sx2, sy2), QPointF(ox2, oy2))
    # Dimension line
    painter.drawLine(QPointF(ox1, oy1), QPointF(ox2, oy2))
    # Arrowheads
    arrow_len = 8
    ang = math.atan2(oy2 - oy1, ox2 - ox1)
    for tip, base_ang in [(QPointF(ox1, oy1), ang), (QPointF(ox2, oy2), ang + math.pi)]:
        a1x = tip.x() + arrow_len * math.cos(base_ang + 0.3)
        a1y = tip.y() + arrow_len * math.sin(base_ang + 0.3)
        a2x = tip.x() + arrow_len * math.cos(base_ang - 0.3)
        a2y = tip.y() + arrow_len * math.sin(base_ang - 0.3)
        painter.drawLine(tip, QPointF(a1x, a1y))
        painter.drawLine(tip, QPointF(a2x, a2y))
    # Text
    mid = QPointF((ox1 + ox2) / 2, (oy1 + oy2) / 2 - 4)
    painter.setFont(QFont("Consolas", 9))
    painter.drawText(mid, f"{dist:.1f}")


def _draw_angular_dimension(painter: QPainter, center, p1, p2, radius=40.0, zoom=1.0):
    """Draw angular dimension between two rays from *center*."""
    cx, cy = center
    a1 = math.degrees(math.atan2(p1[1] - cy, p1[0] - cx))
    a2 = math.degrees(math.atan2(p2[1] - cy, p2[0] - cx))
    sweep = (a2 - a1) % 360
    if sweep > 180:
        sweep -= 360
    pen = QPen(QColor(0, 200, 255), 1, Qt.SolidLine)
    painter.setPen(pen)
    rect = QRectF(cx - radius, cy - radius, 2 * radius, 2 * radius)
    painter.drawArc(rect, int(-a1 * 16), int(-sweep * 16))
    # Text at midpoint of arc
    mid_angle = math.radians(a1 + sweep / 2)
    tx = cx + (radius + 12) * math.cos(mid_angle)
    ty = cy + (radius + 12) * math.sin(mid_angle)
    painter.setFont(QFont("Consolas", 9))
    painter.drawText(QPointF(tx, ty), f"{abs(sweep):.1f}\u00b0")


def _draw_radius_dimension(painter: QPainter, center, radius_screen, zoom=1.0):
    """Draw a radius/diameter dimension for a circle."""
    cx, cy = center
    real_r = radius_screen / zoom
    pen = QPen(QColor(0, 200, 255), 1, Qt.SolidLine)
    painter.setPen(pen)
    # Radial line at 45 degrees
    ex = cx + radius_screen * math.cos(math.radians(45))
    ey = cy - radius_screen * math.sin(math.radians(45))
    painter.drawLine(QPointF(cx, cy), QPointF(ex, ey))
    painter.setFont(QFont("Consolas", 9))
    painter.drawText(QPointF(ex + 4, ey - 4), f"R{real_r:.1f}")


# ---------------------------------------------------------------------------
# Fillet / chamfer utilities
# ---------------------------------------------------------------------------

def _fillet_corner(p_prev, p_corner, p_next, radius):
    """Compute fillet arc replacing a corner. Returns (arc_center, arc_start, arc_end, start_pt, end_pt)."""
    dx1, dy1 = p_prev[0] - p_corner[0], p_prev[1] - p_corner[1]
    dx2, dy2 = p_next[0] - p_corner[0], p_next[1] - p_corner[1]
    l1 = math.hypot(dx1, dy1)
    l2 = math.hypot(dx2, dy2)
    if l1 < 1e-9 or l2 < 1e-9:
        return None
    ux1, uy1 = dx1 / l1, dy1 / l1
    ux2, uy2 = dx2 / l2, dy2 / l2
    # Half-angle between the two edges
    dot = ux1 * ux2 + uy1 * uy2
    dot = max(-1.0, min(1.0, dot))
    half_angle = math.acos(dot) / 2
    if abs(math.sin(half_angle)) < 1e-9:
        return None
    d = radius / math.sin(half_angle)
    tan_len = radius / math.tan(half_angle)
    if tan_len > l1 or tan_len > l2:
        return None  # radius too large
    # Bisector direction
    bx, by = ux1 + ux2, uy1 + uy2
    bl = math.hypot(bx, by)
    if bl < 1e-9:
        return None
    bx, by = bx / bl, by / bl
    center = (p_corner[0] + bx * d, p_corner[1] + by * d)
    start_pt = (p_corner[0] + ux1 * tan_len, p_corner[1] + uy1 * tan_len)
    end_pt = (p_corner[0] + ux2 * tan_len, p_corner[1] + uy2 * tan_len)
    return center, start_pt, end_pt, radius


def _chamfer_corner(p_prev, p_corner, p_next, dist):
    """Compute chamfer line replacing a corner. Returns (start_pt, end_pt)."""
    dx1, dy1 = p_prev[0] - p_corner[0], p_prev[1] - p_corner[1]
    dx2, dy2 = p_next[0] - p_corner[0], p_next[1] - p_corner[1]
    l1 = math.hypot(dx1, dy1)
    l2 = math.hypot(dx2, dy2)
    if l1 < 1e-9 or l2 < 1e-9 or dist > l1 or dist > l2:
        return None
    start_pt = (p_corner[0] + dx1 / l1 * dist, p_corner[1] + dy1 / l1 * dist)
    end_pt = (p_corner[0] + dx2 / l2 * dist, p_corner[1] + dy2 / l2 * dist)
    return start_pt, end_pt


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------
DEFAULT_LAYERS = {
    "Layer 1": QColor(255, 255, 255),
    "Layer 2": QColor(0, 200, 255),
    "Layer 3": QColor(255, 200, 0),
}


# ---------------------------------------------------------------------------
# Canvas widget (handles painting, zoom, pan, drawing)
# ---------------------------------------------------------------------------
class CADCanvas(QWidget):
    mouse_moved = pyqtSignal(float, float)
    selection_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.entities: list[CADEntity] = []
        self.current_tool = "select"
        self.current_layer = "Layer 1"
        self.current_color = QColor(255, 255, 255)
        self.current_line_width = 2
        self.current_line_style = Qt.SolidLine
        self.grid_size = 20
        self.snap_enabled = True
        self.show_grid = True
        # view transform
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self._last_pan_pos = None
        self._is_panning = False
        # drawing state
        self._draw_points = []
        self._cursor_pos = (0, 0)
        # measurement
        self._measure_start = None
        self._measure_end = None

    # -- coordinate transforms --
    def world_to_screen(self, wx, wy):
        sx = wx * self._zoom + self._pan.x()
        sy = wy * self._zoom + self._pan.y()
        return sx, sy

    def screen_to_world(self, sx, sy):
        wx = (sx - self._pan.x()) / self._zoom
        wy = (sy - self._pan.y()) / self._zoom
        return wx, wy

    def snap(self, wx, wy):
        if self.snap_enabled and self.grid_size > 0:
            g = self.grid_size
            return round(wx / g) * g, round(wy / g) * g
        return wx, wy

    # -- painting --
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        # grid
        if self.show_grid:
            self._draw_grid(painter)
        # entities
        for ent in self.entities:
            self._draw_entity(painter, ent)
        # in-progress drawing
        if self._draw_points:
            pen = QPen(self.current_color, self.current_line_width / self._zoom, self.current_line_style)
            painter.setPen(pen)
            pts = self._draw_points + [self._cursor_pos]
            for i in range(len(pts) - 1):
                sx1, sy1 = self.world_to_screen(*pts[i])
                sx2, sy2 = self.world_to_screen(*pts[i + 1])
                painter.drawLine(QPointF(sx1, sy1), QPointF(sx2, sy2))
        # measurement overlay
        if self._measure_start and self._measure_end:
            pen = QPen(QColor(0, 255, 0), 1, Qt.DashLine)
            painter.setPen(pen)
            s1 = self.world_to_screen(*self._measure_start)
            s2 = self.world_to_screen(*self._measure_end)
            painter.drawLine(QPointF(*s1), QPointF(*s2))
            dist = math.hypot(self._measure_end[0] - self._measure_start[0],
                              self._measure_end[1] - self._measure_start[1])
            angle = math.degrees(math.atan2(self._measure_end[1] - self._measure_start[1],
                                            self._measure_end[0] - self._measure_start[0]))
            mid = ((s1[0] + s2[0]) / 2, (s1[1] + s2[1]) / 2)
            painter.drawText(QPointF(mid[0], mid[1] - 10), f"D={dist:.1f}  A={angle:.1f}\u00b0")
        painter.end()

    def _draw_grid(self, painter):
        pen = QPen(QColor(50, 50, 50), 1)
        painter.setPen(pen)
        g = self.grid_size
        w, h = self.width(), self.height()
        x0, y0 = self.screen_to_world(0, 0)
        x1, y1 = self.screen_to_world(w, h)
        sx = int(x0 // g) * g
        while sx <= x1:
            scx, _ = self.world_to_screen(sx, 0)
            painter.drawLine(int(scx), 0, int(scx), h)
            sx += g
        sy = int(y0 // g) * g
        while sy <= y1:
            _, scy = self.world_to_screen(0, sy)
            painter.drawLine(0, int(scy), w, int(scy))
            sy += g

    def _draw_entity(self, painter, ent: CADEntity):
        pen = QPen(ent.color, ent.line_width, ent.line_style)
        if ent.selected:
            pen.setColor(QColor(255, 100, 100))
            pen.setWidth(ent.line_width + 1)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        if ent.etype == "line" and len(ent.points) >= 2:
            s1 = self.world_to_screen(*ent.points[0])
            s2 = self.world_to_screen(*ent.points[1])
            painter.drawLine(QPointF(*s1), QPointF(*s2))
        elif ent.etype == "rect" and len(ent.points) >= 2:
            s1 = self.world_to_screen(*ent.points[0])
            s2 = self.world_to_screen(*ent.points[1])
            painter.drawRect(QRectF(QPointF(*s1), QPointF(*s2)))
        elif ent.etype == "circle" and ent.points:
            cx, cy = self.world_to_screen(*ent.points[0])
            r = ent.radius * self._zoom
            painter.drawEllipse(QPointF(cx, cy), r, r)
        elif ent.etype == "ellipse" and ent.points:
            cx, cy = self.world_to_screen(*ent.points[0])
            painter.drawEllipse(QPointF(cx, cy), ent.rx * self._zoom, ent.ry * self._zoom)
        elif ent.etype == "arc" and ent.points:
            cx, cy = self.world_to_screen(*ent.points[0])
            r = ent.radius * self._zoom
            rect = QRectF(cx - r, cy - r, 2 * r, 2 * r)
            painter.drawArc(rect, int(ent.start_angle * 16), int((ent.end_angle - ent.start_angle) * 16))
        elif ent.etype in ("polygon", "polyline") and len(ent.points) >= 2:
            poly = QPolygonF([QPointF(*self.world_to_screen(*p)) for p in ent.points])
            if ent.etype == "polygon":
                painter.drawPolygon(poly)
            else:
                painter.drawPolyline(poly)
        elif ent.etype == "text" and ent.points:
            sx, sy = self.world_to_screen(*ent.points[0])
            painter.setFont(QFont("Consolas", max(8, int(12 * self._zoom))))
            painter.drawText(QPointF(sx, sy), ent.text)
        elif ent.etype == "dim_linear" and len(ent.points) >= 2:
            s1 = self.world_to_screen(*ent.points[0])
            s2 = self.world_to_screen(*ent.points[1])
            _draw_linear_dimension(painter, s1, s2, offset=25, zoom=self._zoom)
        elif ent.etype == "dim_angular" and len(ent.points) >= 3:
            sc = self.world_to_screen(*ent.points[0])
            s1 = self.world_to_screen(*ent.points[1])
            s2 = self.world_to_screen(*ent.points[2])
            _draw_angular_dimension(painter, sc, s1, s2, radius=40, zoom=self._zoom)
        elif ent.etype == "dim_radius" and ent.points:
            sc = self.world_to_screen(*ent.points[0])
            _draw_radius_dimension(painter, sc, ent.radius * self._zoom, zoom=self._zoom)
        elif ent.etype == "hatch" and len(ent.points) >= 3:
            screen_pts = [QPointF(*self.world_to_screen(*p)) for p in ent.points]
            poly = QPolygonF(screen_pts)
            brect = poly.boundingRect()
            pattern = ent.text if ent.text in HATCH_PATTERNS else "lines"
            if pattern == "solid":
                painter.setBrush(QBrush(ent.color, Qt.Dense4Pattern))
                painter.drawPolygon(poly)
                painter.setBrush(Qt.NoBrush)
            else:
                hatch_path = _generate_hatch_path(brect, pattern, spacing=8.0 * self._zoom)
                clip_path = QPainterPath()
                clip_path.addPolygon(poly)
                clip_path.closeSubpath()
                clipped = hatch_path.intersected(clip_path)
                painter.drawPath(clipped)
                painter.drawPolygon(poly)

    # -- mouse events --
    def mousePressEvent(self, event: QMouseEvent):
        wx, wy = self.screen_to_world(event.x(), event.y())
        wx, wy = self.snap(wx, wy)
        if event.button() == Qt.MiddleButton:
            self._is_panning = True
            self._last_pan_pos = event.pos()
            return
        if event.button() == Qt.RightButton:
            self._finish_drawing(wx, wy)
            return
        if self.current_tool == "select":
            self._handle_select(wx, wy, event.modifiers() & Qt.ShiftModifier)
        elif self.current_tool == "measure":
            if self._measure_start is None:
                self._measure_start = (wx, wy)
            else:
                self._measure_end = (wx, wy)
                self.update()
        elif self.current_tool in ("polygon", "polyline"):
            self._draw_points.append((wx, wy))
        elif self.current_tool == "dim_linear":
            self._draw_points.append((wx, wy))
            if len(self._draw_points) == 2:
                kw = dict(color=QColor(0, 200, 255), line_width=1, layer=self.current_layer)
                ent = CADEntity("dim_linear", list(self._draw_points), **kw)
                self.entities.append(ent)
                self._draw_points.clear()
                self.update()
        elif self.current_tool == "dim_angular":
            self._draw_points.append((wx, wy))
            if len(self._draw_points) == 3:
                kw = dict(color=QColor(0, 200, 255), line_width=1, layer=self.current_layer)
                ent = CADEntity("dim_angular", list(self._draw_points), **kw)
                self.entities.append(ent)
                self._draw_points.clear()
                self.update()
        elif self.current_tool == "dim_radius":
            self._draw_points.append((wx, wy))
            if len(self._draw_points) == 2:
                p1, p2 = self._draw_points
                r = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                kw = dict(color=QColor(0, 200, 255), line_width=1, layer=self.current_layer, radius=r)
                ent = CADEntity("dim_radius", [p1], **kw)
                self.entities.append(ent)
                self._draw_points.clear()
                self.update()
        elif self.current_tool in ("line", "rect", "circle", "ellipse", "arc"):
            self._draw_points.append((wx, wy))
            if len(self._draw_points) == 2:
                self._commit_shape()
        elif self.current_tool == "text":
            text, ok = QInputDialog.getText(self, "Text", "Enter text:")
            if ok and text:
                ent = CADEntity("text", [(wx, wy)], text=text,
                                color=QColor(self.current_color), line_width=self.current_line_width,
                                layer=self.current_layer)
                self.entities.append(ent)
                self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MiddleButton:
            self._is_panning = False

    def mouseMoveEvent(self, event: QMouseEvent):
        wx, wy = self.screen_to_world(event.x(), event.y())
        wx, wy = self.snap(wx, wy)
        self._cursor_pos = (wx, wy)
        self.mouse_moved.emit(wx, wy)
        if self._is_panning and self._last_pan_pos:
            delta = event.pos() - self._last_pan_pos
            self._pan += QPointF(delta.x(), delta.y())
            self._last_pan_pos = event.pos()
        self.update()

    def wheelEvent(self, event: QWheelEvent):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        mx, my = event.x(), event.y()
        wx, wy = self.screen_to_world(mx, my)
        self._zoom *= factor
        self._zoom = max(0.05, min(self._zoom, 50.0))
        nx, ny = self.world_to_screen(wx, wy)
        self._pan += QPointF(mx - nx, my - ny)
        self.update()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Delete:
            self.entities = [e for e in self.entities if not e.selected]
            self.selection_changed.emit()
            self.update()
        elif event.key() == Qt.Key_Escape:
            self._draw_points.clear()
            self._measure_start = self._measure_end = None
            self.update()

    # -- helpers --
    def _handle_select(self, wx, wy, shift):
        if not shift:
            for e in self.entities:
                e.selected = False
        for e in reversed(self.entities):
            if e.hit_test(wx, wy):
                e.selected = not e.selected
                break
        self.selection_changed.emit()
        self.update()

    def _commit_shape(self):
        p1, p2 = self._draw_points
        kw = dict(color=QColor(self.current_color), line_width=self.current_line_width,
                  line_style=self.current_line_style, layer=self.current_layer)
        if self.current_tool == "line":
            ent = CADEntity("line", [p1, p2], **kw)
        elif self.current_tool == "rect":
            ent = CADEntity("rect", [p1, p2], **kw)
        elif self.current_tool == "circle":
            r = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            ent = CADEntity("circle", [p1], radius=r, **kw)
        elif self.current_tool == "ellipse":
            rx = abs(p2[0] - p1[0])
            ry = abs(p2[1] - p1[1])
            ent = CADEntity("ellipse", [p1], rx=rx, ry=ry, **kw)
        elif self.current_tool == "arc":
            r = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            sa = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
            ent = CADEntity("arc", [p1], radius=r, start_angle=sa, end_angle=sa + 180, **kw)
        else:
            self._draw_points.clear()
            return
        self.entities.append(ent)
        self._draw_points.clear()
        self.update()

    def _finish_drawing(self, wx, wy):
        if self.current_tool in ("polygon", "polyline") and len(self._draw_points) >= 2:
            kw = dict(color=QColor(self.current_color), line_width=self.current_line_width,
                      line_style=self.current_line_style, layer=self.current_layer)
            ent = CADEntity(self.current_tool, list(self._draw_points), **kw)
            self.entities.append(ent)
        self._draw_points.clear()
        self.update()

    def move_selected(self, dx, dy):
        for e in self.entities:
            if e.selected:
                e.points = [(p[0] + dx, p[1] + dy) for p in e.points]
        self.update()

    def delete_selected(self):
        self.entities = [e for e in self.entities if not e.selected]
        self.selection_changed.emit()
        self.update()


# ---------------------------------------------------------------------------
# Main CAD2DWidget
# ---------------------------------------------------------------------------
class CAD2DWidget(QWidget):
    """Full 2D CAD widget with tools, layers, properties, and export."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._logger = None
        self._init_ui()

    def set_logger(self, fn):
        self._logger = fn

    def _log(self, msg):
        if self._logger:
            self._logger(msg)

    # -- UI setup --
    def _init_ui(self):
        main = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        # Left: canvas + toolbar
        left = QWidget()
        vbox = QVBoxLayout(left)
        vbox.setContentsMargins(0, 0, 0, 0)
        self.canvas = CADCanvas()
        self._toolbar = self._create_toolbar()
        vbox.addWidget(self._toolbar)
        self.canvas.mouse_moved.connect(self._on_mouse_moved)
        self.canvas.selection_changed.connect(self._on_selection_changed)
        vbox.addWidget(self.canvas)
        self._coord_label = QLabel("X: 0  Y: 0")
        self._coord_label.setStyleSheet("color: #0f0; background: #222; padding: 2px;")
        vbox.addWidget(self._coord_label)
        splitter.addWidget(left)
        # Right panel: layers + properties
        right = QWidget()
        right.setMaximumWidth(220)
        rvbox = QVBoxLayout(right)
        # Layers
        grp_layers = QGroupBox("Layers")
        fl = QVBoxLayout(grp_layers)
        self._layer_list = QListWidget()
        for name, color in DEFAULT_LAYERS.items():
            item = QListWidgetItem(name)
            item.setForeground(color)
            self._layer_list.addItem(item)
        self._layer_list.setCurrentRow(0)
        self._layer_list.currentTextChanged.connect(self._on_layer_changed)
        fl.addWidget(self._layer_list)
        rvbox.addWidget(grp_layers)
        # Properties
        grp_props = QGroupBox("Properties")
        form = QFormLayout(grp_props)
        self._color_btn = QLabel("  ")
        self._color_btn.setStyleSheet("background: white; border: 1px solid gray;")
        self._color_btn.mousePressEvent = lambda e: self._pick_color()
        form.addRow("Color:", self._color_btn)
        self._width_spin = QSpinBox()
        self._width_spin.setRange(1, 20)
        self._width_spin.setValue(2)
        self._width_spin.valueChanged.connect(lambda v: setattr(self.canvas, 'current_line_width', v))
        form.addRow("Width:", self._width_spin)
        self._style_combo = QComboBox()
        self._style_combo.addItems(["Solid", "Dash", "Dot", "DashDot"])
        self._style_combo.currentIndexChanged.connect(self._on_style_changed)
        form.addRow("Style:", self._style_combo)
        self._snap_cb = QCheckBox("Snap to grid")
        self._snap_cb.setChecked(True)
        self._snap_cb.toggled.connect(lambda v: setattr(self.canvas, 'snap_enabled', v))
        form.addRow(self._snap_cb)
        self._grid_spin = QSpinBox()
        self._grid_spin.setRange(5, 100)
        self._grid_spin.setValue(20)
        self._grid_spin.valueChanged.connect(lambda v: setattr(self.canvas, 'grid_size', v))
        form.addRow("Grid:", self._grid_spin)
        rvbox.addWidget(grp_props)
        rvbox.addStretch()
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        main.addWidget(splitter)

    def _create_toolbar(self):
        tb = QToolBar()
        group = QActionGroup(self)
        tools = [
            ("Select", "select"), ("Line", "line"), ("Rect", "rect"),
            ("Circle", "circle"), ("Arc", "arc"), ("Polygon", "polygon"),
            ("Polyline", "polyline"), ("Ellipse", "ellipse"), ("Text", "text"),
            ("Measure", "measure"),
        ]
        for label, tool in tools:
            act = QAction(label, self)
            act.setCheckable(True)
            act.setData(tool)
            act.triggered.connect(lambda checked, t=tool: self._set_tool(t))
            group.addAction(act)
            tb.addAction(act)
            if tool == "select":
                act.setChecked(True)
        tb.addSeparator()
        # Dimension tools
        dim_menu = QMenu("Dimensions", self)
        dim_menu.addAction("Linear Dim", lambda: self._set_tool("dim_linear"))
        dim_menu.addAction("Angular Dim", lambda: self._set_tool("dim_angular"))
        dim_menu.addAction("Radius Dim", lambda: self._set_tool("dim_radius"))
        dim_act = tb.addAction("Dimensions")
        dim_act.setMenu(dim_menu)
        # Creation tools
        create_menu = QMenu("Create", self)
        create_menu.addAction("Rect Array", self._create_rect_array)
        create_menu.addAction("Polar Array", self._create_polar_array)
        create_menu.addAction("Fillet", self._apply_fillet)
        create_menu.addAction("Chamfer", self._apply_chamfer)
        create_menu.addAction("Hatch/Fill", self._apply_hatch)
        create_menu.addAction("Title Block", self._generate_title_block)
        create_act = tb.addAction("Create")
        create_act.setMenu(create_menu)
        tb.addSeparator()
        tb.addAction("Delete", self.canvas.delete_selected)
        tb.addAction("Zoom Fit", self._zoom_fit)
        tb.addAction("Export DXF", lambda: self.export("dxf"))
        tb.addAction("Export PNG", lambda: self.export("png"))
        tb.addAction("Export SVG", lambda: self.export("svg"))
        # Print-scale export
        scale_menu = QMenu("Print Scale", self)
        for scale_label, scale_val in [("1:1", 1.0), ("1:2", 0.5), ("1:5", 0.2), ("1:10", 0.1), ("2:1", 2.0)]:
            scale_menu.addAction(scale_label, lambda s=scale_val, l=scale_label: self._export_print_scale(s, l))
        scale_act = tb.addAction("Print Scale")
        scale_act.setMenu(scale_menu)
        return tb

    # -- slots --
    def _set_tool(self, tool):
        self.canvas.current_tool = tool
        self.canvas._draw_points.clear()
        self._log(f"Tool: {tool}")

    def _on_mouse_moved(self, x, y):
        self._coord_label.setText(f"X: {x:.1f}  Y: {y:.1f}")

    def _on_layer_changed(self, name):
        self.canvas.current_layer = name
        if name in DEFAULT_LAYERS:
            c = DEFAULT_LAYERS[name]
            self.canvas.current_color = c
            self._color_btn.setStyleSheet(f"background: {c.name()}; border: 1px solid gray;")

    def _on_selection_changed(self):
        sel = [e for e in self.canvas.entities if e.selected]
        if sel:
            e = sel[0]
            self._color_btn.setStyleSheet(f"background: {e.color.name()}; border: 1px solid gray;")
            self._width_spin.setValue(e.line_width)

    def _pick_color(self):
        c = QColorDialog.getColor(self.canvas.current_color, self)
        if c.isValid():
            self.canvas.current_color = c
            self._color_btn.setStyleSheet(f"background: {c.name()}; border: 1px solid gray;")
            for e in self.canvas.entities:
                if e.selected:
                    e.color = QColor(c)
            self.canvas.update()

    def _on_style_changed(self, idx):
        styles = [Qt.SolidLine, Qt.DashLine, Qt.DotLine, Qt.DashDotLine]
        self.canvas.current_line_style = styles[idx] if idx < len(styles) else Qt.SolidLine

    # -- array operations --
    def _create_rect_array(self):
        """Create a rectangular array (NxM copies) of selected entities."""
        sel = [e for e in self.canvas.entities if e.selected]
        if not sel:
            self._log("Select entities first for rectangular array.")
            return
        cols, ok1 = QInputDialog.getInt(self, "Rect Array", "Columns:", 3, 1, 50)
        if not ok1:
            return
        rows, ok2 = QInputDialog.getInt(self, "Rect Array", "Rows:", 3, 1, 50)
        if not ok2:
            return
        dx, ok3 = QInputDialog.getDouble(self, "Rect Array", "Column spacing:", 40.0, 1.0, 5000.0, 1)
        if not ok3:
            return
        dy, ok4 = QInputDialog.getDouble(self, "Rect Array", "Row spacing:", 40.0, 1.0, 5000.0, 1)
        if not ok4:
            return
        new_entities = []
        for e in sel:
            for r in range(rows):
                for c in range(cols):
                    if r == 0 and c == 0:
                        continue
                    new_pts = [(p[0] + c * dx, p[1] + r * dy) for p in e.points]
                    ne = CADEntity(e.etype, new_pts, color=QColor(e.color), line_width=e.line_width,
                                   line_style=e.line_style, layer=e.layer, text=e.text,
                                   radius=e.radius, start_angle=e.start_angle, end_angle=e.end_angle,
                                   rx=e.rx, ry=e.ry)
                    new_entities.append(ne)
        self.canvas.entities.extend(new_entities)
        self.canvas.update()
        self._log(f"Rect array: {cols}x{rows}, {len(new_entities)} copies created")

    def _create_polar_array(self):
        """Create a polar array (N copies around a center point) of selected entities."""
        sel = [e for e in self.canvas.entities if e.selected]
        if not sel:
            self._log("Select entities first for polar array.")
            return
        count, ok1 = QInputDialog.getInt(self, "Polar Array", "Number of copies:", 6, 2, 100)
        if not ok1:
            return
        cx, ok2 = QInputDialog.getDouble(self, "Polar Array", "Center X:", 0.0, -10000, 10000, 1)
        if not ok2:
            return
        cy, ok3 = QInputDialog.getDouble(self, "Polar Array", "Center Y:", 0.0, -10000, 10000, 1)
        if not ok3:
            return
        total_angle, ok4 = QInputDialog.getDouble(self, "Polar Array", "Total angle (deg):", 360.0, 1, 360, 1)
        if not ok4:
            return
        new_entities = []
        for i in range(1, count):
            angle = math.radians(total_angle * i / count)
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            for e in sel:
                new_pts = []
                for px, py in e.points:
                    rx, ry = px - cx, py - cy
                    nx = cx + rx * cos_a - ry * sin_a
                    ny = cy + rx * sin_a + ry * cos_a
                    new_pts.append((nx, ny))
                ne = CADEntity(e.etype, new_pts, color=QColor(e.color), line_width=e.line_width,
                               line_style=e.line_style, layer=e.layer, text=e.text,
                               radius=e.radius, start_angle=e.start_angle, end_angle=e.end_angle,
                               rx=e.rx, ry=e.ry)
                new_entities.append(ne)
        self.canvas.entities.extend(new_entities)
        self.canvas.update()
        self._log(f"Polar array: {count} copies around ({cx}, {cy})")

    # -- fillet / chamfer --
    def _apply_fillet(self):
        """Apply fillet to corners of selected polygon/polyline/rect."""
        sel = [e for e in self.canvas.entities if e.selected and e.etype in ("polygon", "polyline", "rect")]
        if not sel:
            self._log("Select a polygon, polyline, or rect for fillet.")
            return
        radius, ok = QInputDialog.getDouble(self, "Fillet", "Fillet radius:", 5.0, 0.1, 500.0, 1)
        if not ok:
            return
        for e in sel:
            pts = list(e.points)
            if e.etype == "rect" and len(pts) == 2:
                x0, y0 = pts[0]
                x1, y1 = pts[1]
                pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            closed = e.etype in ("polygon", "rect")
            new_pts = []
            n = len(pts)
            for i in range(n):
                p_prev = pts[(i - 1) % n] if closed else pts[max(0, i - 1)]
                p_curr = pts[i]
                p_next = pts[(i + 1) % n] if closed else pts[min(n - 1, i + 1)]
                if (not closed and (i == 0 or i == n - 1)):
                    new_pts.append(p_curr)
                    continue
                result = _fillet_corner(p_prev, p_curr, p_next, radius)
                if result:
                    center, start_pt, end_pt, r = result
                    # Approximate arc with line segments
                    sa = math.atan2(start_pt[1] - center[1], start_pt[0] - center[0])
                    ea = math.atan2(end_pt[1] - center[1], end_pt[0] - center[0])
                    if ea < sa:
                        ea += 2 * math.pi
                    steps = max(4, int(abs(ea - sa) / 0.15))
                    for s in range(steps + 1):
                        t = sa + (ea - sa) * s / steps
                        new_pts.append((center[0] + r * math.cos(t), center[1] + r * math.sin(t)))
                else:
                    new_pts.append(p_curr)
            e.points = new_pts
            if e.etype == "rect":
                e.etype = "polygon"
        self.canvas.update()
        self._log(f"Fillet applied (r={radius})")

    def _apply_chamfer(self):
        """Apply chamfer to corners of selected polygon/polyline/rect."""
        sel = [e for e in self.canvas.entities if e.selected and e.etype in ("polygon", "polyline", "rect")]
        if not sel:
            self._log("Select a polygon, polyline, or rect for chamfer.")
            return
        dist, ok = QInputDialog.getDouble(self, "Chamfer", "Chamfer distance:", 5.0, 0.1, 500.0, 1)
        if not ok:
            return
        for e in sel:
            pts = list(e.points)
            if e.etype == "rect" and len(pts) == 2:
                x0, y0 = pts[0]
                x1, y1 = pts[1]
                pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            closed = e.etype in ("polygon", "rect")
            new_pts = []
            n = len(pts)
            for i in range(n):
                p_prev = pts[(i - 1) % n] if closed else pts[max(0, i - 1)]
                p_curr = pts[i]
                p_next = pts[(i + 1) % n] if closed else pts[min(n - 1, i + 1)]
                if (not closed and (i == 0 or i == n - 1)):
                    new_pts.append(p_curr)
                    continue
                result = _chamfer_corner(p_prev, p_curr, p_next, dist)
                if result:
                    new_pts.append(result[0])
                    new_pts.append(result[1])
                else:
                    new_pts.append(p_curr)
            e.points = new_pts
            if e.etype == "rect":
                e.etype = "polygon"
        self.canvas.update()
        self._log(f"Chamfer applied (d={dist})")

    # -- hatch / fill --
    def _apply_hatch(self):
        """Apply hatch/fill pattern to selected closed shapes."""
        sel = [e for e in self.canvas.entities if e.selected and e.etype in ("polygon", "rect", "circle", "ellipse")]
        if not sel:
            self._log("Select a closed shape (polygon, rect, circle, ellipse) for hatch.")
            return
        patterns = list(HATCH_PATTERNS.keys())
        pattern, ok = QInputDialog.getItem(self, "Hatch Pattern", "Pattern:", patterns, 0, False)
        if not ok:
            return
        for e in sel:
            pts = list(e.points)
            if e.etype == "rect" and len(pts) == 2:
                x0, y0 = pts[0]
                x1, y1 = pts[1]
                pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            elif e.etype == "circle" and pts:
                cx, cy = pts[0]
                r = e.radius
                n = 36
                pts = [(cx + r * math.cos(2 * math.pi * i / n),
                         cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]
            elif e.etype == "ellipse" and pts:
                cx, cy = pts[0]
                n = 36
                pts = [(cx + e.rx * math.cos(2 * math.pi * i / n),
                         cy + e.ry * math.sin(2 * math.pi * i / n)) for i in range(n)]
            hatch_ent = CADEntity("hatch", pts, color=QColor(e.color), line_width=1,
                                  line_style=Qt.SolidLine, layer=e.layer, text=pattern)
            self.canvas.entities.append(hatch_ent)
        self.canvas.update()
        self._log(f"Hatch '{pattern}' applied to {len(sel)} shape(s)")

    # -- title block --
    def _generate_title_block(self):
        """Generate a standard engineering drawing border with title block."""
        title, ok = QInputDialog.getText(self, "Title Block", "Drawing title:", text="UNTITLED")
        if not ok:
            return
        # A3-like border in drawing units (420mm x 297mm at scale)
        w, h = 840, 594
        border_margin = 20
        kw = dict(color=QColor(255, 255, 255), line_width=2, line_style=Qt.SolidLine, layer="Layer 1")
        kw_thin = dict(color=QColor(200, 200, 200), line_width=1, line_style=Qt.SolidLine, layer="Layer 1")
        entities = []
        # Outer border
        entities.append(CADEntity("rect", [(border_margin, border_margin),
                                           (w - border_margin, h - border_margin)], **kw))
        # Title block area (bottom-right, 180 wide x 60 tall)
        tb_x = w - border_margin - 180
        tb_y = h - border_margin - 60
        entities.append(CADEntity("rect", [(tb_x, tb_y), (w - border_margin, h - border_margin)], **kw))
        # Internal lines in title block
        entities.append(CADEntity("line", [(tb_x, tb_y + 20), (w - border_margin, tb_y + 20)], **kw_thin))
        entities.append(CADEntity("line", [(tb_x, tb_y + 40), (w - border_margin, tb_y + 40)], **kw_thin))
        entities.append(CADEntity("line", [(tb_x + 90, tb_y), (tb_x + 90, h - border_margin)], **kw_thin))
        # Title text
        entities.append(CADEntity("text", [(tb_x + 5, tb_y + 15)], text=f"TITLE: {title}",
                                  color=QColor(255, 255, 255), line_width=1, layer="Layer 1"))
        entities.append(CADEntity("text", [(tb_x + 5, tb_y + 35)], text="DRAWN BY:",
                                  color=QColor(200, 200, 200), line_width=1, layer="Layer 1"))
        entities.append(CADEntity("text", [(tb_x + 5, tb_y + 55)], text="DATE:",
                                  color=QColor(200, 200, 200), line_width=1, layer="Layer 1"))
        entities.append(CADEntity("text", [(tb_x + 95, tb_y + 15)], text="SCALE:",
                                  color=QColor(200, 200, 200), line_width=1, layer="Layer 1"))
        entities.append(CADEntity("text", [(tb_x + 95, tb_y + 35)], text="SHEET: 1 of 1",
                                  color=QColor(200, 200, 200), line_width=1, layer="Layer 1"))
        entities.append(CADEntity("text", [(tb_x + 95, tb_y + 55)], text="REV: A",
                                  color=QColor(200, 200, 200), line_width=1, layer="Layer 1"))
        self.canvas.entities.extend(entities)
        self.canvas.update()
        self._log(f"Title block generated: '{title}'")

    # -- print-scale export --
    def _export_print_scale(self, scale_factor, scale_label):
        """Export the drawing at a specific print scale to PDF or PNG."""
        path, _ = QFileDialog.getSaveFileName(self, f"Export at {scale_label}",
                                               "", "PDF Files (*.pdf);;PNG Files (*.png)")
        if not path:
            return
        if path.lower().endswith(".pdf"):
            printer = QPrinter(QPrinter.HighResolution)
            printer.setOutputFormat(QPrinter.PdfFormat)
            printer.setOutputFileName(path)
            painter = QPainter(printer)
            painter.setRenderHint(QPainter.Antialiasing)
            # Apply scale
            painter.scale(scale_factor, scale_factor)
            # Draw all entities
            painter.fillRect(QRectF(0, 0, 10000, 10000), QColor(255, 255, 255))
            old_zoom = self.canvas._zoom
            old_pan = self.canvas._pan
            self.canvas._zoom = scale_factor
            self.canvas._pan = QPointF(0, 0)
            for ent in self.canvas.entities:
                self.canvas._draw_entity(painter, ent)
            self.canvas._zoom = old_zoom
            self.canvas._pan = old_pan
            painter.end()
        else:
            # PNG export at scale
            base_w, base_h = 2000, 1414  # A3-like proportions
            img_w = int(base_w * scale_factor)
            img_h = int(base_h * scale_factor)
            img = QImage(img_w, img_h, QImage.Format_ARGB32)
            img.fill(QColor(255, 255, 255))
            painter = QPainter(img)
            painter.setRenderHint(QPainter.Antialiasing)
            old_zoom = self.canvas._zoom
            old_pan = self.canvas._pan
            self.canvas._zoom = scale_factor
            self.canvas._pan = QPointF(0, 0)
            for ent in self.canvas.entities:
                self.canvas._draw_entity(painter, ent)
            self.canvas._zoom = old_zoom
            self.canvas._pan = old_pan
            painter.end()
            img.save(path)
        self._log(f"Exported at scale {scale_label}: {path}")

    def _zoom_fit(self):
        self.canvas._zoom = 1.0
        self.canvas._pan = QPointF(0, 0)
        self.canvas.update()

    # -- public API --
    def load_file(self, path):
        """Load a simple DXF file (lines, circles, text)."""
        self._log(f"Loading: {path}")
        try:
            with open(path, "r") as f:
                lines = f.readlines()
            lines = [l.strip() for l in lines]
            i = 0
            while i < len(lines) - 1:
                if lines[i] == "0" and lines[i + 1] == "LINE":
                    x1 = y1 = x2 = y2 = 0.0
                    j = i + 2
                    while j < len(lines) - 1 and not (lines[j] == "0" and lines[j + 1] in ("LINE", "CIRCLE", "ARC", "TEXT", "ENDSEC")):
                        if lines[j] == "10": x1 = float(lines[j + 1])
                        elif lines[j] == "20": y1 = float(lines[j + 1])
                        elif lines[j] == "11": x2 = float(lines[j + 1])
                        elif lines[j] == "21": y2 = float(lines[j + 1])
                        j += 1
                    self.canvas.entities.append(CADEntity("line", [(x1, y1), (x2, y2)]))
                    i = j
                else:
                    i += 1
            self.canvas.update()
            self._log(f"Loaded {len(self.canvas.entities)} entities")
        except Exception as ex:
            self._log(f"Load error: {ex}")

    def export(self, fmt=None):
        """Export current drawing to DXF, PNG, or SVG."""
        if fmt is None:
            fmt = "png"
        if fmt == "dxf":
            path, _ = QFileDialog.getSaveFileName(self, "Export DXF", "", "DXF Files (*.dxf)")
            if path:
                dxf_ents = []
                for e in self.canvas.entities:
                    ci = 7
                    if e.etype == "line" and len(e.points) >= 2:
                        dxf_ents.append(SimpleDXF.line(*e.points[0], *e.points[1], e.layer, ci))
                    elif e.etype == "circle" and e.points:
                        dxf_ents.append(SimpleDXF.circle(*e.points[0], e.radius, e.layer, ci))
                    elif e.etype == "arc" and e.points:
                        dxf_ents.append(SimpleDXF.arc(*e.points[0], e.radius, e.start_angle, e.end_angle, e.layer, ci))
                    elif e.etype == "text" and e.points:
                        dxf_ents.append(SimpleDXF.text(*e.points[0], 10, e.text, e.layer, ci))
                SimpleDXF.write(dxf_ents, path)
                self._log(f"Exported DXF: {path}")
        elif fmt in ("png", "svg"):
            ext = fmt.upper()
            filt = f"{ext} Files (*.{fmt})"
            path, _ = QFileDialog.getSaveFileName(self, f"Export {ext}", "", filt)
            if path:
                if fmt == "png":
                    img = QImage(self.canvas.size(), QImage.Format_ARGB32)
                    img.fill(QColor(30, 30, 30))
                    painter = QPainter(img)
                    painter.setRenderHint(QPainter.Antialiasing)
                    self.canvas.render(painter)
                    painter.end()
                    img.save(path)
                else:
                    from PyQt5.QtSvg import QSvgGenerator
                    gen = QSvgGenerator()
                    gen.setFileName(path)
                    gen.setSize(self.canvas.size())
                    painter = QPainter(gen)
                    self.canvas.render(painter)
                    painter.end()
                self._log(f"Exported {ext}: {path}")
