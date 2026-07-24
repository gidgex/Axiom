"""
CAD3DWidget - A 3D CAD widget for PyQt5 scientific suite (SolidWorks-like basic).
Provides 3D viewport, primitive shapes, transforms, STL/OBJ loading, and export.
"""
import math
import struct
import os
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QToolBar, QAction, QLabel,
    QComboBox, QDoubleSpinBox, QListWidget, QListWidgetItem, QSplitter,
    QGroupBox, QFormLayout, QPushButton, QFileDialog, QCheckBox, QMenu,
    QInputDialog, QSpinBox, QColorDialog
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


# ---------------------------------------------------------------------------
# 3D Shape primitives
# ---------------------------------------------------------------------------
class Shape3D:
    _uid = 0

    def __init__(self, stype, name=None, **params):
        Shape3D._uid += 1
        self.uid = Shape3D._uid
        self.stype = stype
        self.name = name or f"{stype}_{self.uid}"
        self.params = params  # dimensions, resolution etc.
        self.position = np.array(params.get("position", [0.0, 0.0, 0.0]), dtype=float)
        self.rotation = np.array(params.get("rotation", [0.0, 0.0, 0.0]), dtype=float)  # degrees
        self.scale = np.array(params.get("scale", [1.0, 1.0, 1.0]), dtype=float)
        self.color = params.get("color", "#4488cc")
        self.alpha = params.get("alpha", 0.7)
        self.visible = True
        self.selected = False
        # mesh data: vertices (Nx3), faces (Mx3 indices)
        self.vertices = np.zeros((0, 3))
        self.faces = np.zeros((0, 3), dtype=int)
        self._generate_mesh()

    def _generate_mesh(self):
        gen = {
            "box": self._gen_box, "sphere": self._gen_sphere,
            "cylinder": self._gen_cylinder, "cone": self._gen_cone,
            "torus": self._gen_torus, "plane": self._gen_plane,
            "custom": lambda: None,
        }
        fn = gen.get(self.stype)
        if fn:
            fn()
        self._apply_transform()

    def _apply_transform(self):
        if len(self.vertices) == 0:
            return
        v = self.vertices.copy()
        v *= self.scale
        for axis, angle_deg in enumerate(self.rotation):
            if angle_deg != 0:
                v = _rotate_vertices(v, axis, math.radians(angle_deg))
        v += self.position
        self.vertices = v

    # -- mesh generators --
    def _gen_box(self):
        sx = self.params.get("sx", 1.0)
        sy = self.params.get("sy", 1.0)
        sz = self.params.get("sz", 1.0)
        hx, hy, hz = sx / 2, sy / 2, sz / 2
        self.vertices = np.array([
            [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
            [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
        ])
        self.faces = np.array([
            [0,1,2],[0,2,3],[4,6,5],[4,7,6],
            [0,4,5],[0,5,1],[2,6,7],[2,7,3],
            [0,3,7],[0,7,4],[1,5,6],[1,6,2],
        ])

    def _gen_sphere(self, n=16):
        r = self.params.get("radius", 0.5)
        verts, faces = [], []
        for i in range(n + 1):
            theta = math.pi * i / n
            for j in range(n + 1):
                phi = 2 * math.pi * j / n
                x = r * math.sin(theta) * math.cos(phi)
                y = r * math.sin(theta) * math.sin(phi)
                z = r * math.cos(theta)
                verts.append([x, y, z])
        for i in range(n):
            for j in range(n):
                a = i * (n + 1) + j
                b = a + n + 1
                faces.append([a, b, b + 1])
                faces.append([a, b + 1, a + 1])
        self.vertices = np.array(verts)
        self.faces = np.array(faces)

    def _gen_cylinder(self, n=24):
        r = self.params.get("radius", 0.5)
        h = self.params.get("height", 1.0)
        verts, faces = [], []
        for i in range(n):
            a = 2 * math.pi * i / n
            verts.append([r * math.cos(a), r * math.sin(a), -h / 2])
            verts.append([r * math.cos(a), r * math.sin(a), h / 2])
        # center caps
        bi = len(verts)
        verts.append([0, 0, -h / 2])
        verts.append([0, 0, h / 2])
        for i in range(n):
            j = (i + 1) % n
            faces.append([i * 2, j * 2, j * 2 + 1])
            faces.append([i * 2, j * 2 + 1, i * 2 + 1])
            faces.append([bi, j * 2, i * 2])
            faces.append([bi + 1, i * 2 + 1, j * 2 + 1])
        self.vertices = np.array(verts)
        self.faces = np.array(faces)

    def _gen_cone(self, n=24):
        r = self.params.get("radius", 0.5)
        h = self.params.get("height", 1.0)
        verts, faces = [], []
        for i in range(n):
            a = 2 * math.pi * i / n
            verts.append([r * math.cos(a), r * math.sin(a), 0])
        tip = len(verts)
        verts.append([0, 0, h])
        base_c = len(verts)
        verts.append([0, 0, 0])
        for i in range(n):
            j = (i + 1) % n
            faces.append([i, j, tip])
            faces.append([base_c, j, i])
        self.vertices = np.array(verts)
        self.faces = np.array(faces)

    def _gen_torus(self, n=20, m=12):
        R = self.params.get("major_radius", 0.7)
        rr = self.params.get("minor_radius", 0.25)
        verts = []
        for i in range(n):
            theta = 2 * math.pi * i / n
            for j in range(m):
                phi = 2 * math.pi * j / m
                x = (R + rr * math.cos(phi)) * math.cos(theta)
                y = (R + rr * math.cos(phi)) * math.sin(theta)
                z = rr * math.sin(phi)
                verts.append([x, y, z])
        faces = []
        for i in range(n):
            ni = (i + 1) % n
            for j in range(m):
                nj = (j + 1) % m
                a = i * m + j
                b = ni * m + j
                c = ni * m + nj
                d = i * m + nj
                faces.append([a, b, c])
                faces.append([a, c, d])
        self.vertices = np.array(verts)
        self.faces = np.array(faces)

    def _gen_plane(self):
        sx = self.params.get("sx", 2.0)
        sy = self.params.get("sy", 2.0)
        hx, hy = sx / 2, sy / 2
        self.vertices = np.array([[-hx, -hy, 0], [hx, -hy, 0], [hx, hy, 0], [-hx, hy, 0]])
        self.faces = np.array([[0, 1, 2], [0, 2, 3]])

    def get_face_vertices(self):
        """Return list of (N,3) arrays for Poly3DCollection."""
        if len(self.faces) == 0 or len(self.vertices) == 0:
            return []
        return [self.vertices[f] for f in self.faces]

    def get_edge_lines(self):
        """Return edge line segments for wireframe mode."""
        edges = set()
        for f in self.faces:
            for k in range(len(f)):
                a, b = int(f[k]), int(f[(k + 1) % len(f)])
                edges.add((min(a, b), max(a, b)))
        return [(self.vertices[a], self.vertices[b]) for a, b in edges]


def _rotate_vertices(v, axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    R = np.eye(3)
    if axis == 0:
        R[1, 1], R[1, 2], R[2, 1], R[2, 2] = c, -s, s, c
    elif axis == 1:
        R[0, 0], R[0, 2], R[2, 0], R[2, 2] = c, s, -s, c
    else:
        R[0, 0], R[0, 1], R[1, 0], R[1, 1] = c, -s, s, c
    return v @ R.T


# ---------------------------------------------------------------------------
# STL / OBJ loaders
# ---------------------------------------------------------------------------
def load_ascii_stl(path):
    """Parse ASCII STL file and return vertices, faces."""
    verts, faces = [], []
    with open(path, "r") as f:
        tri = []
        for line in f:
            line = line.strip()
            if line.startswith("vertex"):
                parts = line.split()
                tri.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if len(tri) == 3:
                    base = len(verts)
                    verts.extend(tri)
                    faces.append([base, base + 1, base + 2])
                    tri = []
    return np.array(verts) if verts else np.zeros((0, 3)), np.array(faces, dtype=int) if faces else np.zeros((0, 3), dtype=int)


def load_obj(path):
    """Parse basic OBJ file (v and f lines)."""
    verts, faces = [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                idx = [int(p.split("/")[0]) - 1 for p in parts]
                if len(idx) >= 3:
                    for i in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[i], idx[i + 1]])
    return np.array(verts) if verts else np.zeros((0, 3)), np.array(faces, dtype=int) if faces else np.zeros((0, 3), dtype=int)


def export_ascii_stl(shapes, path):
    """Write shapes to ASCII STL."""
    with open(path, "w") as f:
        f.write("solid model\n")
        for shape in shapes:
            fv = shape.get_face_vertices()
            for tri in fv:
                if len(tri) == 3:
                    n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                    nl = np.linalg.norm(n)
                    if nl > 0:
                        n = n / nl
                    f.write(f"  facet normal {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
                    f.write("    outer loop\n")
                    for v in tri:
                        f.write(f"      vertex {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                    f.write("    endloop\n  endfacet\n")
        f.write("endsolid model\n")


def export_obj(shapes, path):
    """Write shapes to OBJ."""
    with open(path, "w") as f:
        f.write("# OBJ export\n")
        offset = 0
        for shape in shapes:
            for v in shape.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in shape.faces:
                idx = " ".join(str(int(i) + 1 + offset) for i in face)
                f.write(f"f {idx}\n")
            offset += len(shape.vertices)


# ---------------------------------------------------------------------------
# Boolean operations (visual / CSG concept)
# ---------------------------------------------------------------------------
def boolean_union(a: Shape3D, b: Shape3D) -> Shape3D:
    """Union: merge vertices and faces of two shapes."""
    result = Shape3D("custom", name=f"Union_{a.uid}_{b.uid}")
    offset = len(a.vertices)
    result.vertices = np.vstack([a.vertices, b.vertices]) if len(a.vertices) and len(b.vertices) else a.vertices if len(a.vertices) else b.vertices
    b_faces = b.faces + offset if len(b.faces) else np.zeros((0, 3), dtype=int)
    result.faces = np.vstack([a.faces, b_faces]) if len(a.faces) and len(b_faces) else a.faces if len(a.faces) else b_faces
    result.color = a.color
    result.alpha = a.alpha
    return result


def boolean_subtract(a: Shape3D, b: Shape3D) -> Shape3D:
    """Subtract: visual approximation - keep A, mark subtracted region."""
    result = Shape3D("custom", name=f"Subtract_{a.uid}_{b.uid}")
    result.vertices = a.vertices.copy()
    result.faces = a.faces.copy()
    result.color = "#cc4444"
    result.alpha = 0.6
    return result


def boolean_intersect(a: Shape3D, b: Shape3D) -> Shape3D:
    """Intersect: visual approximation - show overlapping bounding region."""
    result = Shape3D("custom", name=f"Intersect_{a.uid}_{b.uid}")
    result.vertices = a.vertices.copy()
    result.faces = a.faces.copy()
    result.color = "#44cc44"
    result.alpha = 0.5
    return result


# ---------------------------------------------------------------------------
# Extrude / Revolve operations
# ---------------------------------------------------------------------------

def extrude_profile(profile_pts, height, n_segments=1):
    """Extrude a 2D profile (list of (x,y) points) along the Z axis.

    Returns a Shape3D with the extruded solid mesh.
    """
    n = len(profile_pts)
    if n < 3:
        return None
    verts = []
    faces = []
    # Create vertices for bottom and top faces and intermediate segments
    for seg in range(n_segments + 1):
        z = height * seg / n_segments
        for px, py in profile_pts:
            verts.append([px, py, z])
    # Side faces
    for seg in range(n_segments):
        base_lo = seg * n
        base_hi = (seg + 1) * n
        for i in range(n):
            j = (i + 1) % n
            faces.append([base_lo + i, base_lo + j, base_hi + j])
            faces.append([base_lo + i, base_hi + j, base_hi + i])
    # Bottom face (fan triangulation)
    for i in range(1, n - 1):
        faces.append([0, i + 1, i])
    # Top face
    top_base = n_segments * n
    for i in range(1, n - 1):
        faces.append([top_base, top_base + i, top_base + i + 1])
    shape = Shape3D("custom", name=f"Extrude_h{height:.1f}")
    shape.vertices = np.array(verts, dtype=float)
    shape.faces = np.array(faces, dtype=int)
    shape.color = "#44aa88"
    return shape


def revolve_profile(profile_pts, axis_origin=(0, 0), axis_dir=(0, 1),
                    angle_deg=360.0, n_steps=24):
    """Revolve a 2D profile around an axis to create a solid of revolution.

    *profile_pts* is a list of (r, z) pairs where r is the distance from the axis.
    The axis is along Z by default. *axis_origin* and *axis_dir* define the axis
    in the XZ plane for flexibility.
    """
    n = len(profile_pts)
    if n < 2:
        return None
    angle_rad = math.radians(angle_deg)
    closed = abs(angle_deg - 360.0) < 0.01
    steps = n_steps if closed else n_steps + 1
    verts = []
    for s in range(steps):
        theta = angle_rad * s / n_steps
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        for r, z in profile_pts:
            verts.append([r * cos_t, r * sin_t, z])
    faces = []
    for s in range(n_steps):
        s_next = (s + 1) % steps
        base = s * n
        base_next = s_next * n
        for i in range(n - 1):
            faces.append([base + i, base_next + i, base_next + i + 1])
            faces.append([base + i, base_next + i + 1, base + i + 1])
    shape = Shape3D("custom", name=f"Revolve_{angle_deg:.0f}deg")
    shape.vertices = np.array(verts, dtype=float)
    shape.faces = np.array(faces, dtype=int)
    shape.color = "#aa8844"
    return shape


def section_view(shape: Shape3D, plane_origin=None, plane_normal=None):
    """Cut *shape* with a plane and return a new shape showing the cross-section.

    The plane is defined by a point (*plane_origin*) and a normal vector.
    Default: XY plane at z=0.
    Returns the portion of the mesh on the negative side of the plane
    plus visible section-cut edges.
    """
    if plane_origin is None:
        plane_origin = np.array([0.0, 0.0, 0.0])
    else:
        plane_origin = np.array(plane_origin, dtype=float)
    if plane_normal is None:
        plane_normal = np.array([0.0, 0.0, 1.0])
    else:
        plane_normal = np.array(plane_normal, dtype=float)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    # Signed distance of each vertex to the plane
    dists = (shape.vertices - plane_origin) @ plane_normal
    # Keep faces entirely on negative side or straddling the plane
    kept_faces = []
    for f in shape.faces:
        d = dists[f]
        if np.all(d <= 0.01):
            kept_faces.append(f)
        elif np.any(d <= 0.0):
            kept_faces.append(f)  # partial -- keep for visual approximation
    result = Shape3D("custom", name=f"Section_{shape.name}")
    result.vertices = shape.vertices.copy()
    result.faces = np.array(kept_faces, dtype=int) if kept_faces else np.zeros((0, 3), dtype=int)
    result.color = "#cc8844"
    result.alpha = 0.8
    return result


def compute_mass_properties(shape: Shape3D, density=1.0):
    """Compute volume, surface area, and center of mass from a triangle mesh.

    Uses the divergence theorem for volume and simple face-area summation.
    Returns a dict with keys: volume, surface_area, center_of_mass.
    """
    verts = shape.vertices
    faces = shape.faces
    if len(faces) == 0 or len(verts) == 0:
        return {"volume": 0.0, "surface_area": 0.0, "center_of_mass": np.zeros(3), "mass": 0.0}
    total_volume = 0.0
    total_area = 0.0
    center_sum = np.zeros(3)
    for f in faces:
        v0, v1, v2 = verts[f[0]], verts[f[1]], verts[f[2]]
        # Signed volume of tetrahedron with origin
        cross = np.cross(v1, v2)
        vol = np.dot(v0, cross) / 6.0
        total_volume += vol
        # Triangle area
        edge_cross = np.cross(v1 - v0, v2 - v0)
        area = np.linalg.norm(edge_cross) / 2.0
        total_area += area
        # Weighted centroid
        centroid = (v0 + v1 + v2) / 3.0
        center_sum += centroid * vol
    if abs(total_volume) > 1e-12:
        center_of_mass = center_sum / total_volume
    else:
        center_of_mass = np.zeros(3)
    total_volume = abs(total_volume)
    mass = total_volume * density
    return {
        "volume": total_volume,
        "surface_area": total_area,
        "center_of_mass": center_of_mass,
        "mass": mass,
    }


# ---------------------------------------------------------------------------
# Matplotlib 3D Viewport
# ---------------------------------------------------------------------------
class Viewport3D(FigureCanvas):
    """Matplotlib-based 3D viewport."""

    def __init__(self, parent=None):
        self.fig = Figure(facecolor="#1e1e1e")
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111, projection="3d", facecolor="#1e1e1e")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.shapes: list[Shape3D] = []
        self.wireframe_mode = False
        self.render_mode = False  # basic lighting and shading
        self.setMinimumSize(400, 400)

    def refresh(self):
        self.ax.cla()
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_facecolor("#1e1e1e")
        for shape in self.shapes:
            if not shape.visible:
                continue
            if self.wireframe_mode:
                edges = shape.get_edge_lines()
                if edges:
                    lc = Line3DCollection(edges, colors=shape.color, linewidths=0.8 if not shape.selected else 2.0)
                    self.ax.add_collection3d(lc)
            elif self.render_mode:
                fv = shape.get_face_vertices()
                if fv:
                    # Basic diffuse lighting (Lambertian shading)
                    light_dir = np.array([0.5, 0.3, 1.0])
                    light_dir = light_dir / np.linalg.norm(light_dir)
                    base_color = np.array(QColor(shape.color).getRgbF()[:3])
                    face_colors = []
                    for tri in fv:
                        if len(tri) >= 3:
                            n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                            nl = np.linalg.norm(n)
                            if nl > 0:
                                n = n / nl
                            diffuse = max(0.15, abs(np.dot(n, light_dir)))
                            face_colors.append(base_color * diffuse)
                        else:
                            face_colors.append(base_color * 0.5)
                    face_colors = np.clip(face_colors, 0, 1)
                    ec = "#ff4444" if shape.selected else "#222222"
                    lw = 1.5 if shape.selected else 0.1
                    pc = Poly3DCollection(fv, alpha=shape.alpha, edgecolor=ec, linewidths=lw)
                    pc.set_facecolor(face_colors)
                    self.ax.add_collection3d(pc)
            else:
                fv = shape.get_face_vertices()
                if fv:
                    ec = "#ff4444" if shape.selected else "#333333"
                    lw = 1.5 if shape.selected else 0.3
                    pc = Poly3DCollection(fv, alpha=shape.alpha, facecolor=shape.color,
                                          edgecolor=ec, linewidths=lw)
                    self.ax.add_collection3d(pc)
        self._auto_limits()
        self.draw()

    def _auto_limits(self):
        all_v = [s.vertices for s in self.shapes if s.visible and len(s.vertices)]
        if not all_v:
            self.ax.set_xlim(-2, 2)
            self.ax.set_ylim(-2, 2)
            self.ax.set_zlim(-2, 2)
            return
        v = np.vstack(all_v)
        mn, mx = v.min(axis=0), v.max(axis=0)
        c = (mn + mx) / 2
        span = max((mx - mn).max(), 0.5) * 0.6
        self.ax.set_xlim(c[0] - span, c[0] + span)
        self.ax.set_ylim(c[1] - span, c[1] + span)
        self.ax.set_zlim(c[2] - span, c[2] + span)


# ---------------------------------------------------------------------------
# Main CAD3DWidget
# ---------------------------------------------------------------------------
class CAD3DWidget(QWidget):
    """Full 3D CAD widget with primitives, transforms, file I/O, and boolean ops."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._logger = None
        self._init_ui()

    def set_logger(self, fn):
        self._logger = fn

    def _log(self, msg):
        if self._logger:
            self._logger(msg)

    def _init_ui(self):
        main = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        # Left: viewport + toolbar
        left = QWidget()
        vbox = QVBoxLayout(left)
        vbox.setContentsMargins(0, 0, 0, 0)
        self._toolbar = self._create_toolbar()
        vbox.addWidget(self._toolbar)
        self.viewport = Viewport3D()
        vbox.addWidget(self.viewport)
        splitter.addWidget(left)
        # Right panel
        right = QWidget()
        right.setMaximumWidth(240)
        rvbox = QVBoxLayout(right)
        # Object list
        grp_obj = QGroupBox("Objects")
        ol = QVBoxLayout(grp_obj)
        self._obj_list = QListWidget()
        self._obj_list.currentRowChanged.connect(self._on_obj_selected)
        ol.addWidget(self._obj_list)
        btn_row = QHBoxLayout()
        btn_del = QPushButton("Delete")
        btn_del.clicked.connect(self._delete_selected)
        btn_dup = QPushButton("Duplicate")
        btn_dup.clicked.connect(self._duplicate_selected)
        btn_row.addWidget(btn_del)
        btn_row.addWidget(btn_dup)
        ol.addLayout(btn_row)
        rvbox.addWidget(grp_obj)
        # Transform
        grp_xform = QGroupBox("Transform")
        form = QFormLayout(grp_xform)
        self._pos_spins = []
        self._rot_spins = []
        self._scl_spins = []
        for label, arr, callback in [("Pos", self._pos_spins, self._apply_transform),
                                      ("Rot", self._rot_spins, self._apply_transform),
                                      ("Scale", self._scl_spins, self._apply_transform)]:
            row = QHBoxLayout()
            for axis in "XYZ":
                sp = QDoubleSpinBox()
                sp.setRange(-999, 999)
                sp.setSingleStep(0.1)
                sp.setDecimals(2)
                if label == "Scale":
                    sp.setValue(1.0)
                sp.valueChanged.connect(callback)
                row.addWidget(sp)
                arr.append(sp)
            form.addRow(f"{label}:", row)
        rvbox.addWidget(grp_xform)
        # Display
        grp_disp = QGroupBox("Display")
        df = QVBoxLayout(grp_disp)
        self._wireframe_cb = QCheckBox("Wireframe")
        self._wireframe_cb.toggled.connect(self._toggle_wireframe)
        df.addWidget(self._wireframe_cb)
        self._render_cb = QCheckBox("Render (lit)")
        self._render_cb.toggled.connect(self._toggle_render)
        df.addWidget(self._render_cb)
        self._color_btn = QPushButton("Color")
        self._color_btn.clicked.connect(self._pick_color)
        df.addWidget(self._color_btn)
        rvbox.addWidget(grp_disp)
        rvbox.addStretch()
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        main.addWidget(splitter)

    def _create_toolbar(self):
        tb = QToolBar()
        prims = [("Box", "box"), ("Sphere", "sphere"), ("Cylinder", "cylinder"),
                 ("Cone", "cone"), ("Torus", "torus"), ("Plane", "plane")]
        for label, stype in prims:
            tb.addAction(label, lambda st=stype: self._add_primitive(st))
        tb.addSeparator()
        tb.addAction("Load STL", lambda: self._load_dialog("stl"))
        tb.addAction("Load OBJ", lambda: self._load_dialog("obj"))
        tb.addSeparator()
        tb.addAction("Export STL", lambda: self.export("stl"))
        tb.addAction("Export OBJ", lambda: self.export("obj"))
        tb.addSeparator()
        bool_menu = QMenu("Boolean", self)
        bool_menu.addAction("Union", lambda: self._boolean_op("union"))
        bool_menu.addAction("Subtract", lambda: self._boolean_op("subtract"))
        bool_menu.addAction("Intersect", lambda: self._boolean_op("intersect"))
        bool_btn = tb.addAction("Boolean")
        bool_btn.setMenu(bool_menu)
        tb.addSeparator()
        # Creation / generation tools
        create_menu = QMenu("Create", self)
        create_menu.addAction("Extrude Profile", self._extrude_profile)
        create_menu.addAction("Revolve Profile", self._revolve_profile)
        create_menu.addAction("Section View", self._section_view)
        create_act = tb.addAction("Create")
        create_act.setMenu(create_menu)
        tb.addAction("Assembly", self._assembly_position)
        tb.addAction("Mass Props", self._show_mass_properties)
        return tb

    # -- primitive creation --
    def _add_primitive(self, stype):
        params = {}
        if stype == "box":
            params = {"sx": 1.0, "sy": 1.0, "sz": 1.0}
        elif stype == "sphere":
            params = {"radius": 0.5}
        elif stype == "cylinder":
            params = {"radius": 0.4, "height": 1.0}
        elif stype == "cone":
            params = {"radius": 0.5, "height": 1.0}
        elif stype == "torus":
            params = {"major_radius": 0.7, "minor_radius": 0.25}
        elif stype == "plane":
            params = {"sx": 2.0, "sy": 2.0}
        shape = Shape3D(stype, **params)
        self.viewport.shapes.append(shape)
        self._refresh_list()
        self.viewport.refresh()
        self._log(f"Added {stype}")

    # -- object list --
    def _refresh_list(self):
        self._obj_list.blockSignals(True)
        self._obj_list.clear()
        for s in self.viewport.shapes:
            item = QListWidgetItem(s.name)
            if not s.visible:
                item.setForeground(QColor(100, 100, 100))
            self._obj_list.addItem(item)
        self._obj_list.blockSignals(False)

    def _on_obj_selected(self, row):
        for i, s in enumerate(self.viewport.shapes):
            s.selected = (i == row)
        if 0 <= row < len(self.viewport.shapes):
            s = self.viewport.shapes[row]
            self._set_spin_values(self._pos_spins, s.position)
            self._set_spin_values(self._rot_spins, s.rotation)
            self._set_spin_values(self._scl_spins, s.scale)
        self.viewport.refresh()

    def _set_spin_values(self, spins, vals):
        for sp, v in zip(spins, vals):
            sp.blockSignals(True)
            sp.setValue(v)
            sp.blockSignals(False)

    def _get_selected(self):
        row = self._obj_list.currentRow()
        if 0 <= row < len(self.viewport.shapes):
            return row, self.viewport.shapes[row]
        return -1, None

    # -- transforms --
    def _apply_transform(self):
        row, shape = self._get_selected()
        if shape is None:
            return
        new_pos = np.array([sp.value() for sp in self._pos_spins])
        new_rot = np.array([sp.value() for sp in self._rot_spins])
        new_scl = np.array([sp.value() for sp in self._scl_spins])
        # Regenerate mesh with updated transform
        shape.position = new_pos
        shape.rotation = new_rot
        shape.scale = new_scl
        shape.vertices = np.zeros((0, 3))
        shape.faces = np.zeros((0, 3), dtype=int)
        shape._generate_mesh()
        self.viewport.refresh()

    # -- display --
    def _toggle_wireframe(self, on):
        self.viewport.wireframe_mode = on
        if on:
            self.viewport.render_mode = False
            self._render_cb.blockSignals(True)
            self._render_cb.setChecked(False)
            self._render_cb.blockSignals(False)
        self.viewport.refresh()

    def _toggle_render(self, on):
        self.viewport.render_mode = on
        if on:
            self.viewport.wireframe_mode = False
            self._wireframe_cb.blockSignals(True)
            self._wireframe_cb.setChecked(False)
            self._wireframe_cb.blockSignals(False)
        self.viewport.refresh()

    def _pick_color(self):
        _, shape = self._get_selected()
        if shape is None:
            return
        c = QColorDialog.getColor(QColor(shape.color), self)
        if c.isValid():
            shape.color = c.name()
            self.viewport.refresh()

    # -- delete / duplicate --
    def _delete_selected(self):
        row, shape = self._get_selected()
        if shape is not None:
            self.viewport.shapes.pop(row)
            self._refresh_list()
            self.viewport.refresh()
            self._log(f"Deleted {shape.name}")

    def _duplicate_selected(self):
        _, shape = self._get_selected()
        if shape is None:
            return
        dup = Shape3D(shape.stype, **shape.params)
        dup.name = shape.name + "_copy"
        dup.position = shape.position + np.array([0.5, 0.5, 0.0])
        dup.rotation = shape.rotation.copy()
        dup.scale = shape.scale.copy()
        dup.color = shape.color
        dup.vertices = np.zeros((0, 3))
        dup.faces = np.zeros((0, 3), dtype=int)
        dup._generate_mesh()
        self.viewport.shapes.append(dup)
        self._refresh_list()
        self.viewport.refresh()
        self._log(f"Duplicated {shape.name}")

    # -- extrude / revolve / section --
    def _extrude_profile(self):
        """Extrude a 2D profile along Z axis to create a 3D solid."""
        shape_type, ok = QInputDialog.getItem(
            self, "Extrude Profile", "Profile shape:",
            ["Square", "Circle", "Triangle", "L-shape", "Hexagon"], 0, False)
        if not ok:
            return
        height, ok2 = QInputDialog.getDouble(self, "Extrude", "Height:", 1.0, 0.01, 100.0, 2)
        if not ok2:
            return
        size, ok3 = QInputDialog.getDouble(self, "Extrude", "Profile size:", 0.5, 0.01, 50.0, 2)
        if not ok3:
            return
        # Generate 2D profile points
        if shape_type == "Square":
            h = size / 2
            profile = [(-h, -h), (h, -h), (h, h), (-h, h)]
        elif shape_type == "Circle":
            n = 24
            profile = [(size / 2 * math.cos(2 * math.pi * i / n),
                         size / 2 * math.sin(2 * math.pi * i / n)) for i in range(n)]
        elif shape_type == "Triangle":
            h = size / 2
            profile = [(0, h), (-h, -h), (h, -h)]
        elif shape_type == "L-shape":
            s = size
            profile = [(0, 0), (s, 0), (s, s * 0.3), (s * 0.3, s * 0.3), (s * 0.3, s), (0, s)]
        elif shape_type == "Hexagon":
            profile = [(size / 2 * math.cos(math.pi / 3 * i),
                         size / 2 * math.sin(math.pi / 3 * i)) for i in range(6)]
        else:
            return
        shape = extrude_profile(profile, height)
        if shape:
            self.viewport.shapes.append(shape)
            self._refresh_list()
            self.viewport.refresh()
            self._log(f"Extruded {shape_type} profile, h={height}")

    def _revolve_profile(self):
        """Revolve a 2D profile around the Z axis."""
        profile_type, ok = QInputDialog.getItem(
            self, "Revolve Profile", "Profile shape:",
            ["Rectangle", "Circle", "Triangle", "Custom trapezoid"], 0, False)
        if not ok:
            return
        angle, ok2 = QInputDialog.getDouble(self, "Revolve", "Angle (degrees):", 360.0, 10.0, 360.0, 1)
        if not ok2:
            return
        size, ok3 = QInputDialog.getDouble(self, "Revolve", "Profile size:", 0.3, 0.01, 50.0, 2)
        if not ok3:
            return
        offset, ok4 = QInputDialog.getDouble(self, "Revolve", "Axis offset (r):", 0.7, 0.0, 50.0, 2)
        if not ok4:
            return
        # Profile as (r, z) pairs
        h = size / 2
        if profile_type == "Rectangle":
            profile = [(offset - h, -h), (offset + h, -h), (offset + h, h), (offset - h, h)]
        elif profile_type == "Circle":
            n = 12
            profile = [(offset + h * math.cos(2 * math.pi * i / n),
                         h * math.sin(2 * math.pi * i / n)) for i in range(n)]
        elif profile_type == "Triangle":
            profile = [(offset, h), (offset + h, -h), (offset - h, -h)]
        elif profile_type == "Custom trapezoid":
            profile = [(offset - h, -h), (offset + h, -h), (offset + h * 0.6, h), (offset - h * 0.6, h)]
        else:
            return
        shape = revolve_profile(profile, angle_deg=angle)
        if shape:
            self.viewport.shapes.append(shape)
            self._refresh_list()
            self.viewport.refresh()
            self._log(f"Revolved {profile_type}, angle={angle}")

    def _section_view(self):
        """Cut a selected solid with a plane and show the cross-section."""
        _, shape = self._get_selected()
        if shape is None:
            self._log("Select an object first for section view.")
            return
        axis, ok = QInputDialog.getItem(self, "Section Plane", "Cut plane normal:",
                                         ["X (YZ plane)", "Y (XZ plane)", "Z (XY plane)"], 2, False)
        if not ok:
            return
        offset, ok2 = QInputDialog.getDouble(self, "Section", "Plane offset:", 0.0, -100, 100, 2)
        if not ok2:
            return
        normals = {"X (YZ plane)": [1, 0, 0], "Y (XZ plane)": [0, 1, 0], "Z (XY plane)": [0, 0, 1]}
        normal = normals[axis]
        origin = [n * offset for n in normal]
        result = section_view(shape, plane_origin=origin, plane_normal=normal)
        self.viewport.shapes.append(result)
        self._refresh_list()
        self.viewport.refresh()
        self._log(f"Section view created (plane {axis}, offset={offset})")

    def _assembly_position(self):
        """Position a selected part relative to another for assembly."""
        if len(self.viewport.shapes) < 2:
            self._log("Need at least 2 objects for assembly positioning.")
            return
        _, shape = self._get_selected()
        if shape is None:
            self._log("Select the object to reposition.")
            return
        names = [s.name for s in self.viewport.shapes if s is not shape]
        if not names:
            self._log("No other objects to position relative to.")
            return
        target_name, ok = QInputDialog.getItem(self, "Assembly", "Position relative to:", names, 0, False)
        if not ok:
            return
        target = next((s for s in self.viewport.shapes if s.name == target_name), None)
        if target is None:
            return
        mode, ok2 = QInputDialog.getItem(self, "Assembly", "Placement mode:",
                                          ["Align centers", "Stack on top (+Z)", "Place beside (+X)",
                                           "Place beside (+Y)", "Custom offset"], 0, False)
        if not ok2:
            return
        # Compute bounding boxes
        t_center = target.vertices.mean(axis=0) if len(target.vertices) else np.zeros(3)
        s_center = shape.vertices.mean(axis=0) if len(shape.vertices) else np.zeros(3)
        if mode == "Align centers":
            delta = t_center - s_center
        elif mode == "Stack on top (+Z)":
            t_max_z = target.vertices[:, 2].max() if len(target.vertices) else 0
            s_min_z = shape.vertices[:, 2].min() if len(shape.vertices) else 0
            delta = t_center - s_center
            delta[2] = t_max_z - s_min_z
        elif mode == "Place beside (+X)":
            t_max_x = target.vertices[:, 0].max() if len(target.vertices) else 0
            s_min_x = shape.vertices[:, 0].min() if len(shape.vertices) else 0
            delta = np.array([t_max_x - s_min_x + 0.1, 0, 0])
        elif mode == "Place beside (+Y)":
            t_max_y = target.vertices[:, 1].max() if len(target.vertices) else 0
            s_min_y = shape.vertices[:, 1].min() if len(shape.vertices) else 0
            delta = np.array([0, t_max_y - s_min_y + 0.1, 0])
        elif mode == "Custom offset":
            ox, ok_x = QInputDialog.getDouble(self, "Offset X", "X:", 0.0, -100, 100, 2)
            oy, ok_y = QInputDialog.getDouble(self, "Offset Y", "Y:", 0.0, -100, 100, 2)
            oz, ok_z = QInputDialog.getDouble(self, "Offset Z", "Z:", 0.0, -100, 100, 2)
            if not (ok_x and ok_y and ok_z):
                return
            delta = t_center - s_center + np.array([ox, oy, oz])
        else:
            return
        shape.vertices += delta
        shape.position += delta
        for sp, v in zip(self._pos_spins, shape.position):
            sp.blockSignals(True)
            sp.setValue(v)
            sp.blockSignals(False)
        self.viewport.refresh()
        self._log(f"Positioned {shape.name} relative to {target_name} ({mode})")

    def _show_mass_properties(self):
        """Compute and display mass properties of the selected object."""
        _, shape = self._get_selected()
        if shape is None:
            self._log("Select an object to compute mass properties.")
            return
        density, ok = QInputDialog.getDouble(self, "Mass Properties", "Material density:", 1.0, 0.001, 20000, 3)
        if not ok:
            return
        props = compute_mass_properties(shape, density=density)
        com = props["center_of_mass"]
        msg = (f"Mass Properties for '{shape.name}':\n"
               f"  Volume        : {props['volume']:.6f}\n"
               f"  Surface Area  : {props['surface_area']:.6f}\n"
               f"  Mass          : {props['mass']:.6f}\n"
               f"  Center of Mass: ({com[0]:.4f}, {com[1]:.4f}, {com[2]:.4f})\n"
               f"  Density       : {density}")
        self._log(msg)
        QInputDialog.getMultiLineText(self, "Mass Properties", "Results:", msg)

    # -- boolean operations --
    def _boolean_op(self, op):
        sel = [s for s in self.viewport.shapes if s.selected]
        if len(sel) < 2:
            self._log("Boolean requires 2 selected objects. Select via list with Ctrl.")
            return
        a, b = sel[0], sel[1]
        ops = {"union": boolean_union, "subtract": boolean_subtract, "intersect": boolean_intersect}
        result = ops[op](a, b)
        self.viewport.shapes = [s for s in self.viewport.shapes if s not in (a, b)]
        self.viewport.shapes.append(result)
        self._refresh_list()
        self.viewport.refresh()
        self._log(f"Boolean {op}: {a.name} & {b.name}")

    # -- file I/O --
    def _load_dialog(self, fmt):
        filt = "STL Files (*.stl)" if fmt == "stl" else "OBJ Files (*.obj)"
        path, _ = QFileDialog.getOpenFileName(self, f"Load {fmt.upper()}", "", filt)
        if path:
            self.load_file(path)

    def load_file(self, path):
        """Load STL or OBJ file."""
        self._log(f"Loading: {path}")
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".stl":
                verts, faces = load_ascii_stl(path)
            elif ext == ".obj":
                verts, faces = load_obj(path)
            else:
                self._log(f"Unsupported format: {ext}")
                return
            shape = Shape3D("custom", name=os.path.basename(path))
            shape.vertices = verts
            shape.faces = faces
            self.viewport.shapes.append(shape)
            self._refresh_list()
            self.viewport.refresh()
            self._log(f"Loaded {len(verts)} vertices, {len(faces)} faces")
        except Exception as ex:
            self._log(f"Load error: {ex}")

    def export(self, fmt=None):
        """Export scene to STL or OBJ."""
        if fmt is None:
            fmt = "stl"
        if fmt == "stl":
            path, _ = QFileDialog.getSaveFileName(self, "Export STL", "", "STL Files (*.stl)")
            if path:
                export_ascii_stl(self.viewport.shapes, path)
                self._log(f"Exported STL: {path}")
        elif fmt == "obj":
            path, _ = QFileDialog.getSaveFileName(self, "Export OBJ", "", "OBJ Files (*.obj)")
            if path:
                export_obj(self.viewport.shapes, path)
                self._log(f"Exported OBJ: {path}")
