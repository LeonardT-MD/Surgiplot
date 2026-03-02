"""
Surgiplot GUI — single-file app (copy-paste ready)

AI feature separation:
- The Single-image depth -> 3D reconstruction + point collection dialog is in:
    surgiplot/ai/single_image_depth.py

Start flow (IMPORTANT change):
- Start window now has 3 modes:
    1) Load annotation file
    2) Manual entry
    3) Single surgical image (AI depth → pick points → OK → workspace)

Feature deps (only needed if you use the dialog):
  pip install -U numpy pillow matplotlib transformers torch
"""

from __future__ import annotations

import numpy as np
from PySide6 import QtWidgets, QtCore

from surgiplot.core.io.loaders import load_dataset, load_points_from_manual
from surgiplot.metrics import VOM_VOA, AOA_SF, AOE, DISTANCE_3D, AREA_3D

# NEW: import the dialog from the separated module
try:
    from surgiplot.ai.single_image_depth import SingleImage3DDialog
    HAS_SINGLE_IMAGE_DIALOG = True
except Exception:
    SingleImage3DDialog = None  # type: ignore
    HAS_SINGLE_IMAGE_DIALOG = False


# -------------------------
# Matplotlib 3D plot panel
# -------------------------
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    HAS_MPL = True
except Exception:
    HAS_MPL = False
    FigureCanvas = None  # type: ignore
    Figure = None  # type: ignore
    Poly3DCollection = None  # type: ignore


def _dbg_get(dbg, key: str, default=None):
    """Robustly get debug field from dict-like or attribute-like debug."""
    if dbg is None:
        return default
    if isinstance(dbg, dict):
        return dbg.get(key, default)
    return getattr(dbg, key, default)


def _as_xyz(p):
    return float(p[0]), float(p[1]), float(p[2])


def _close_loop(xs, ys, zs):
    if not xs:
        return xs, ys, zs
    return xs + [xs[0]], ys + [ys[0]], zs + [zs[0]]


def _project_point_fixed_distance(p, p_ref, distance: float):
    """Project point p to be at a fixed distance from p_ref along vector p_ref->p."""
    p = np.asarray(p, dtype=float).reshape(3,)
    p_ref = np.asarray(p_ref, dtype=float).reshape(3,)
    d = p - p_ref
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return p_ref.copy()
    return p_ref + (d / n) * float(distance)


class PlotPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)

        if not HAS_MPL:
            msg = QtWidgets.QLabel(
                "<b>3D preview disabled</b><br>"
                "Install matplotlib to enable the 3D plot panel.<br><br>"
                "<code>pip install matplotlib</code>"
            )
            msg.setWordWrap(True)
            layout.addWidget(msg)
            self.fig = None
            self.canvas = None
            self.ax = None
            return

        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection="3d")
        layout.addWidget(self.canvas)

    def clear(self):
        if not HAS_MPL:
            return
        self.fig.clear()
        self.ax = self.fig.add_subplot(111, projection="3d")

    def _draw(self):
        if not HAS_MPL:
            return
        self.canvas.draw()

    def _scatter_all(self, ds):
        xs, ys, zs = [], [], []
        for _k, v in ds.points.items():
            xs.append(float(v[0]))
            ys.append(float(v[1]))
            zs.append(float(v[2]))
        self.ax.scatter(xs, ys, zs, s=10)

    def plot_points(self, ds, highlight: list[str] | None = None, title: str = ""):
        if not HAS_MPL:
            return

        self.clear()
        self._scatter_all(ds)

        if highlight:
            hx, hy, hz = [], [], []
            for nm in highlight:
                if nm not in ds.points:
                    continue
                p = ds.points[nm]
                hx.append(float(p[0]))
                hy.append(float(p[1]))
                hz.append(float(p[2]))
            if hx:
                self.ax.scatter(hx, hy, hz, s=70)

        self.ax.set_title(title)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self._draw()

    # -------------------------
    # VOM / sVOM / VoA plotting
    # -------------------------
    def plot_vom_voa(
        self,
        ds,
        entry: list[str],
        target: list[str],
        *,
        debug=None,
        show_polygons: bool = True,
        show_ellipses: bool = True,
        show_target_distance: bool = True,
        show_vom: bool = True,
        show_svom: bool = True,
        n_vom_slices: int = 40,
        n_svom_slices: int = 15,
        title="VOM_VOA view",
    ):
        if not HAS_MPL:
            return

        self.clear()
        self._scatter_all(ds)

        entry_pts = [ds.points[n] for n in entry]
        target_pts = [ds.points[n] for n in target]

        ex = [float(p[0]) for p in entry_pts]
        ey = [float(p[1]) for p in entry_pts]
        ez = [float(p[2]) for p in entry_pts]
        tx = [float(p[0]) for p in target_pts]
        ty = [float(p[1]) for p in target_pts]
        tz = [float(p[2]) for p in target_pts]

        ex2, ey2, ez2 = _close_loop(ex, ey, ez)
        tx2, ty2, tz2 = _close_loop(tx, ty, tz)

        if show_polygons:
            self.ax.plot(ex2, ey2, ez2, linewidth=2)
            self.ax.plot(tx2, ty2, tz2, linewidth=2)
            if Poly3DCollection is not None:
                self.ax.add_collection3d(Poly3DCollection([np.asarray(entry_pts, dtype=float)], alpha=0.10))
                self.ax.add_collection3d(Poly3DCollection([np.asarray(target_pts, dtype=float)], alpha=0.10))

        c_entry = _dbg_get(debug, "entry_centroid", None)
        c_target = _dbg_get(debug, "target_centroid", None)

        if c_entry is None:
            c_entry = np.mean(np.asarray(entry_pts, dtype=float), axis=0)
        if c_target is None:
            c_target = np.mean(np.asarray(target_pts, dtype=float), axis=0)

        cx1, cy1, cz1 = _as_xyz(c_entry)
        cx2, cy2, cz2 = _as_xyz(c_target)

        self.ax.scatter([cx1], [cy1], [cz1], s=70)
        self.ax.scatter([cx2], [cy2], [cz2], s=70)

        if show_target_distance:
            self.ax.plot([cx1, cx2], [cy1, cy2], [cz1, cz2], linestyle="--", linewidth=2)

        ell1 = _dbg_get(debug, "entry_ellipse_3d", None)
        ell2 = _dbg_get(debug, "target_ellipse_3d", None)
        ell3 = _dbg_get(debug, "cut_ellipse_3d", _dbg_get(debug, "svom_cut_ellipse_3d", None))

        def _plot_loop(arr, lw=1.8):
            if arr is None:
                return
            arr = np.asarray(arr, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 3:
                return
            self.ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], linewidth=lw)

        if show_ellipses:
            _plot_loop(ell1, lw=2.0)
            _plot_loop(ell2, lw=2.0)
            _plot_loop(ell3, lw=2.0)

        vom_slices = _dbg_get(debug, "vom_slices_3d", None)
        svom_slices = _dbg_get(debug, "svom_slices_3d", None)

        def _add_side_mesh(loop_a, loop_b, alpha=0.12):
            if Poly3DCollection is None:
                return
            a = np.asarray(loop_a, dtype=float)
            b = np.asarray(loop_b, dtype=float)
            if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
                return
            n = a.shape[0]
            if n < 3:
                return
            faces = []
            for i in range(n - 1):
                faces.append([a[i], a[i + 1], b[i + 1], b[i]])
            faces.append([a[-1], a[0], b[0], b[-1]])
            self.ax.add_collection3d(Poly3DCollection(faces, alpha=alpha, linewidths=0.1))

        def _interpolate_loops(loop_start, loop_end, n_slices):
            loop_start = np.asarray(loop_start, dtype=float)
            loop_end = np.asarray(loop_end, dtype=float)
            if loop_start.shape != loop_end.shape:
                return None
            if loop_start.ndim != 2 or loop_start.shape[1] != 3:
                return None
            slices = []
            for i in range(n_slices + 1):
                t = i / float(n_slices)
                slices.append((1 - t) * loop_start + t * loop_end)
            return slices

        if vom_slices is None and ell1 is not None and ell2 is not None:
            vom_slices = _interpolate_loops(ell1, ell2, max(2, int(n_vom_slices)))

        if svom_slices is None and ell3 is not None and ell2 is not None:
            svom_slices = _interpolate_loops(ell3, ell2, max(2, int(n_svom_slices)))

        if show_vom and vom_slices is not None:
            try:
                for i in range(len(vom_slices) - 1):
                    _add_side_mesh(vom_slices[i], vom_slices[i + 1], alpha=0.08)
            except Exception:
                pass

        if show_svom and svom_slices is not None:
            try:
                for i in range(len(svom_slices) - 1):
                    _add_side_mesh(svom_slices[i], svom_slices[i + 1], alpha=0.14)
            except Exception:
                pass

        self.ax.set_title(title)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self._draw()

    def plot_aoa_sf(
        self,
        ds,
        cranial: str,
        caudal: str,
        medial: str,
        lateral: str,
        pivot: str,
        constraints: list[str] | None = None,
        rescale_radius: float | None = None,
        title="AOA_SF view",
    ):
        if not HAS_MPL:
            return

        p_cr = ds.points[cranial]
        p_ca = ds.points[caudal]
        p_me = ds.points[medial]
        p_la = ds.points[lateral]
        p_pi = ds.points[pivot]

        self.clear()
        self._scatter_all(ds)

        def _plot_triangle(p1, p2, p3, alpha=0.25, linewidth=2, dashed_base=False):
            self.ax.plot([float(p2[0]), float(p3[0])], [float(p2[1]), float(p3[1])], [float(p2[2]), float(p3[2])], linewidth=linewidth)
            self.ax.plot([float(p1[0]), float(p3[0])], [float(p1[1]), float(p3[1])], [float(p1[2]), float(p3[2])], linewidth=linewidth)
            ls = "--" if dashed_base else "-"
            self.ax.plot([float(p1[0]), float(p2[0])], [float(p1[1]), float(p2[1])], [float(p1[2]), float(p2[2])], linestyle=ls, linewidth=linewidth)
            if Poly3DCollection is not None:
                verts = [[np.asarray(p1), np.asarray(p2), np.asarray(p3)]]
                self.ax.add_collection3d(Poly3DCollection(verts, alpha=alpha))

        def _plot_quad(points_4, alpha=0.18, linewidth=2):
            pts = [np.asarray(p) for p in points_4]
            xs = [float(p[0]) for p in pts] + [float(pts[0][0])]
            ys = [float(p[1]) for p in pts] + [float(pts[0][1])]
            zs = [float(p[2]) for p in pts] + [float(pts[0][2])]
            self.ax.plot(xs, ys, zs, linewidth=linewidth)
            if Poly3DCollection is not None:
                self.ax.add_collection3d(Poly3DCollection([pts], alpha=alpha))

        def _dotted(p_from, p_to):
            self.ax.plot([float(p_from[0]), float(p_to[0])], [float(p_from[1]), float(p_to[1])], [float(p_from[2]), float(p_to[2])], linestyle="--", linewidth=1.5)

        _plot_triangle(p_cr, p_ca, p_pi, alpha=0.22, dashed_base=True)
        _plot_triangle(p_me, p_la, p_pi, alpha=0.22, dashed_base=True)
        _plot_quad([p_cr, p_ca, p_me, p_la], alpha=0.16, linewidth=2)

        self.ax.scatter([float(p_pi[0])], [float(p_pi[1])], [float(p_pi[2])], s=90)

        if constraints:
            for nm in constraints:
                if nm in ds.points:
                    p = ds.points[nm]
                    self.ax.scatter([float(p[0])], [float(p[1])], [float(p[2])], s=55)

        if rescale_radius is not None and rescale_radius > 0:
            p_cr_r = _project_point_fixed_distance(p_cr, p_pi, rescale_radius)
            p_ca_r = _project_point_fixed_distance(p_ca, p_pi, rescale_radius)
            p_me_r = _project_point_fixed_distance(p_me, p_pi, rescale_radius)
            p_la_r = _project_point_fixed_distance(p_la, p_pi, rescale_radius)

            _dotted(p_cr, p_cr_r)
            _dotted(p_ca, p_ca_r)
            _dotted(p_me, p_me_r)
            _dotted(p_la, p_la_r)

            _plot_triangle(p_cr_r, p_ca_r, p_pi, alpha=0.08, dashed_base=False)
            _plot_triangle(p_me_r, p_la_r, p_pi, alpha=0.08, dashed_base=False)
            _plot_quad([p_cr_r, p_ca_r, p_me_r, p_la_r], alpha=0.06, linewidth=1.5)

        self.ax.set_title(title)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self._draw()

    def plot_aoe(self, ds, A: str, B: str, C: str, title="AOE view"):
        if not HAS_MPL:
            return

        self.clear()
        self._scatter_all(ds)

        a = ds.points[A]
        b = ds.points[B]
        c = ds.points[C]

        self.ax.plot([float(a[0]), float(b[0])], [float(a[1]), float(b[1])], [float(a[2]), float(b[2])], linewidth=3)
        self.ax.plot([float(b[0]), float(c[0])], [float(b[1]), float(c[1])], [float(b[2]), float(c[2])], linewidth=3)

        self.ax.set_title(title)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self._draw()

    def plot_distance(self, ds, A: str, B: str, title="DISTANCE_3D view"):
        if not HAS_MPL:
            return

        self.clear()
        self._scatter_all(ds)

        a = ds.points[A]
        b = ds.points[B]
        self.ax.scatter([float(a[0])], [float(a[1])], [float(a[2])], s=80)
        self.ax.scatter([float(b[0])], [float(b[1])], [float(b[2])], s=80)
        self.ax.plot([float(a[0]), float(b[0])], [float(a[1]), float(b[1])], [float(a[2]), float(b[2])], linewidth=3)

        self.ax.set_title(title)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self._draw()

    def plot_area(self, ds, poly: list[str], title="AREA_3D view"):
        if not HAS_MPL:
            return

        self.clear()
        self._scatter_all(ds)

        pts = [ds.points[n] for n in poly]
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        zs = [float(p[2]) for p in pts]
        xs2, ys2, zs2 = xs + [xs[0]], ys + [ys[0]], zs + [zs[0]]
        self.ax.plot(xs2, ys2, zs2, linewidth=2)

        if Poly3DCollection is not None:
            self.ax.add_collection3d(Poly3DCollection([list(map(np.asarray, pts))], alpha=0.12))

        self.ax.set_title(title)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self._draw()


# -------------------------
# Dataset table panel
# -------------------------
class DatasetPanel(QtWidgets.QWidget):
    """Always-visible dataset table with editable 'labels' column."""
    applied = QtCore.Signal()

    def __init__(self, ds, parent=None):
        super().__init__(parent)
        self.ds = ds

        layout = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QLabel("<b>Dataset</b> (edit labels, then click Apply)")
        layout.addWidget(header)

        self.source_cb = QtWidgets.QComboBox()
        self.source_cb.addItems(["navigation", "photogrammetry", "scanner", "single_image_depth"])
        self.source_cb.setCurrentText(self.ds.meta.get("source", "navigation"))
        layout.addWidget(self.source_cb)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["name", "x", "y", "z", "labels"])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_apply = QtWidgets.QPushButton("Apply labels")
        self.btn_apply.clicked.connect(self._apply)
        btn_row.addWidget(self.btn_apply)

        self.btn_refresh = QtWidgets.QPushButton("Refresh view")
        self.btn_refresh.clicked.connect(self.populate)
        btn_row.addWidget(self.btn_refresh)

        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.populate()

    def populate(self):
        rows = self.ds.to_table_rows()
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(r["name"])))
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(r["x"])))
            self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(r["y"])))
            self.table.setItem(i, 3, QtWidgets.QTableWidgetItem(str(r["z"])))
            self.table.setItem(i, 4, QtWidgets.QTableWidgetItem(str(r.get("labels", ""))))

        for row in range(self.table.rowCount()):
            for col in (0, 1, 2, 3):
                it = self.table.item(row, col)
                it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)

    def to_rows(self):
        rows = []
        for i in range(self.table.rowCount()):
            rows.append(
                {
                    "name": self.table.item(i, 0).text(),
                    "x": float(self.table.item(i, 1).text()),
                    "y": float(self.table.item(i, 2).text()),
                    "z": float(self.table.item(i, 3).text()),
                    "labels": self.table.item(i, 4).text().strip(),
                }
            )
        return rows

    def _apply(self):
        self.ds.meta["source"] = self.source_cb.currentText()
        rows = self.to_rows()
        self.ds.apply_labels_from_table(rows, overwrite=True)
        QtWidgets.QMessageBox.information(self, "Applied", "Labels applied to dataset.")
        self.applied.emit()


# -------------------------
# Metric panel
# -------------------------
class MetricPanel(QtWidgets.QWidget):
    run_requested = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)

        layout.addWidget(QtWidgets.QLabel("<b>Metrics</b>"))

        self.metric_cb = QtWidgets.QComboBox()
        self.metric_cb.addItems(["VOM_VOA", "AOA_SF", "AOE", "DISTANCE_3D", "AREA_3D"])
        layout.addWidget(self.metric_cb)

        self.form = QtWidgets.QFormLayout()

        # VOM_VOA fields
        self.vom_entry = QtWidgets.QLineEdit()
        self.vom_entry.setPlaceholderText("Entry polygon points (comma-separated)")
        self.form.addRow("VOM_VOA entry", self.vom_entry)

        self.vom_target = QtWidgets.QLineEdit()
        self.vom_target.setPlaceholderText("Target polygon points (comma-separated)")
        self.form.addRow("VOM_VOA target", self.vom_target)

        self.stand_dist = QtWidgets.QDoubleSpinBox()
        self.stand_dist.setRange(0.1, 1000.0)
        self.stand_dist.setValue(10.0)
        self.stand_dist.setSuffix(" mm")
        self.form.addRow("stand_dist (sVOM)", self.stand_dist)

        # VOM_VOA plot toggles
        self.vom_show_polygons = QtWidgets.QCheckBox()
        self.vom_show_polygons.setChecked(True)
        self.form.addRow("Plot polygons", self.vom_show_polygons)

        self.vom_show_ellipses = QtWidgets.QCheckBox()
        self.vom_show_ellipses.setChecked(True)
        self.form.addRow("Plot ellipses", self.vom_show_ellipses)

        self.vom_show_distance = QtWidgets.QCheckBox()
        self.vom_show_distance.setChecked(True)
        self.form.addRow("Plot target distance", self.vom_show_distance)

        self.vom_show_vom = QtWidgets.QCheckBox()
        self.vom_show_vom.setChecked(True)
        self.form.addRow("Plot VOM surface", self.vom_show_vom)

        self.vom_show_svom = QtWidgets.QCheckBox()
        self.vom_show_svom.setChecked(True)
        self.form.addRow("Plot sVOM surface", self.vom_show_svom)

        self.vom_slices = QtWidgets.QSpinBox()
        self.vom_slices.setRange(5, 200)
        self.vom_slices.setValue(40)
        self.form.addRow("VOM slices", self.vom_slices)

        self.svom_slices = QtWidgets.QSpinBox()
        self.svom_slices.setRange(5, 200)
        self.svom_slices.setValue(15)
        self.form.addRow("sVOM slices", self.svom_slices)

        # AOA_SF fields
        self.aoa_cranial = QtWidgets.QLineEdit(); self.aoa_cranial.setPlaceholderText("cranial entry point")
        self.aoa_caudal = QtWidgets.QLineEdit(); self.aoa_caudal.setPlaceholderText("caudal entry point")
        self.aoa_medial = QtWidgets.QLineEdit(); self.aoa_medial.setPlaceholderText("medial entry point")
        self.aoa_lateral = QtWidgets.QLineEdit(); self.aoa_lateral.setPlaceholderText("lateral entry point")
        self.aoa_pivot = QtWidgets.QLineEdit(); self.aoa_pivot.setPlaceholderText("pivot point")

        self.form.addRow("AOA_SF cranial", self.aoa_cranial)
        self.form.addRow("AOA_SF caudal", self.aoa_caudal)
        self.form.addRow("AOA_SF medial", self.aoa_medial)
        self.form.addRow("AOA_SF lateral", self.aoa_lateral)
        self.form.addRow("AOA_SF pivot", self.aoa_pivot)

        self.aoa_constraints = QtWidgets.QLineEdit()
        self.aoa_constraints.setPlaceholderText("Optional constraints (comma-separated)")
        self.form.addRow("AOA_SF constraints", self.aoa_constraints)

        self.aoa_rescale_enable = QtWidgets.QCheckBox("Enable SF rescaling (project entry points to fixed radius from pivot)")
        self.form.addRow("AOA_SF rescale", self.aoa_rescale_enable)

        self.aoa_rescale_radius = QtWidgets.QDoubleSpinBox()
        self.aoa_rescale_radius.setRange(0.1, 5000.0)
        self.aoa_rescale_radius.setValue(250.0)
        self.aoa_rescale_radius.setSuffix(" mm")
        self.form.addRow("AOA_SF radius", self.aoa_rescale_radius)

        # AOE fields
        self.A_edit = QtWidgets.QLineEdit(); self.A_edit.setPlaceholderText("A point name")
        self.B_edit = QtWidgets.QLineEdit(); self.B_edit.setPlaceholderText("B pivot name")
        self.C_edit = QtWidgets.QLineEdit(); self.C_edit.setPlaceholderText("C point name")
        self.form.addRow("AOE A", self.A_edit)
        self.form.addRow("AOE B", self.B_edit)
        self.form.addRow("AOE C", self.C_edit)

        # DISTANCE_3D fields
        self.dist_A = QtWidgets.QLineEdit(); self.dist_A.setPlaceholderText("A point name/label")
        self.dist_B = QtWidgets.QLineEdit(); self.dist_B.setPlaceholderText("B point name/label")
        self.form.addRow("DISTANCE A", self.dist_A)
        self.form.addRow("DISTANCE B", self.dist_B)

        # AREA_3D field
        self.area_poly = QtWidgets.QLineEdit(); self.area_poly.setPlaceholderText("Polygon points (>=3), comma-separated")
        self.form.addRow("AREA polygon", self.area_poly)

        layout.addLayout(self.form)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton("Run")
        self.btn_run.clicked.connect(self._emit_run)
        btn_row.addWidget(self.btn_run)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.out = QtWidgets.QPlainTextEdit()
        self.out.setReadOnly(True)
        layout.addWidget(self.out)

        self.metric_cb.currentTextChanged.connect(self._update_visibility)
        self.aoa_rescale_enable.stateChanged.connect(lambda _s: self._update_visibility(self.metric_cb.currentText()))
        self._update_visibility(self.metric_cb.currentText())

    def _emit_run(self):
        self.run_requested.emit(self.metric_cb.currentText())

    def _set_row_visible(self, widget: QtWidgets.QWidget, visible: bool):
        widget.setVisible(visible)
        for i in range(self.form.rowCount()):
            label_item = self.form.itemAt(i, QtWidgets.QFormLayout.LabelRole)
            field_item = self.form.itemAt(i, QtWidgets.QFormLayout.FieldRole)
            if field_item and field_item.widget() is widget:
                if label_item and label_item.widget():
                    label_item.widget().setVisible(visible)
                break

    def _update_visibility(self, metric: str):
        is_vom = metric == "VOM_VOA"
        is_aoa = metric == "AOA_SF"
        is_aoe = metric == "AOE"
        is_dist = metric == "DISTANCE_3D"
        is_area = metric == "AREA_3D"

        for w in (
            self.vom_entry, self.vom_target, self.stand_dist,
            self.vom_show_polygons, self.vom_show_ellipses, self.vom_show_distance,
            self.vom_show_vom, self.vom_show_svom, self.vom_slices, self.svom_slices
        ):
            self._set_row_visible(w, is_vom)

        for w in (
            self.aoa_cranial, self.aoa_caudal, self.aoa_medial, self.aoa_lateral,
            self.aoa_pivot, self.aoa_constraints, self.aoa_rescale_enable
        ):
            self._set_row_visible(w, is_aoa)
        self._set_row_visible(self.aoa_rescale_radius, is_aoa and self.aoa_rescale_enable.isChecked())

        for w in (self.A_edit, self.B_edit, self.C_edit):
            self._set_row_visible(w, is_aoe)

        for w in (self.dist_A, self.dist_B):
            self._set_row_visible(w, is_dist)

        self._set_row_visible(self.area_poly, is_area)

    @staticmethod
    def split_names(s: str):
        return [x.strip() for x in (s or "").split(",") if x.strip()]


# -------------------------
# Main window
# -------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, ds):
        super().__init__()
        self.ds = ds

        self.setWindowTitle("Surgiplot — Workspace")
        self.resize(1400, 720)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        self.dataset_panel = DatasetPanel(self.ds)
        self.metric_panel = MetricPanel()
        self.plot_panel = PlotPanel()

        splitter.addWidget(self.dataset_panel)
        splitter.addWidget(self.metric_panel)
        splitter.addWidget(self.plot_panel)
        splitter.setSizes([420, 520, 460])

        # Toolbar action for single-image points (optional "add more points" while already in workspace)
        tb = self.addToolBar("Tools")
        act_single = tb.addAction("Add points from single image…")
        act_single.triggered.connect(self._open_single_image_dialog)
        if not HAS_SINGLE_IMAGE_DIALOG:
            act_single.setEnabled(False)
            act_single.setToolTip("Install feature deps and ensure surgiplot/ai/single_image_depth.py is present.")

        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)
        lay.addWidget(splitter)
        self.setCentralWidget(central)

        self.dataset_panel.applied.connect(self._on_dataset_applied)
        self.metric_panel.run_requested.connect(self._run_metric)

        self.plot_panel.plot_points(self.ds, title="Dataset overview")

    def _open_single_image_dialog(self):
        if not HAS_SINGLE_IMAGE_DIALOG or SingleImage3DDialog is None:
            QtWidgets.QMessageBox.critical(
                self,
                "Feature unavailable",
                "Single-image 3D dialog is not available.\n\n"
                "1) Ensure file exists: surgiplot/ai/single_image_depth.py\n"
                "2) Install deps: pip install -U numpy pillow matplotlib transformers torch",
            )
            return

        dlg = SingleImage3DDialog(self.ds, parent=self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self.dataset_panel.populate()
            self.plot_panel.plot_points(self.ds, title="Dataset updated (image points committed)")

    def _on_dataset_applied(self):
        self.dataset_panel.populate()
        self.plot_panel.plot_points(self.ds, title="Dataset updated")

    def _resolve(self, name: str) -> str:
        name = (name or "").strip()
        if not name:
            return name
        if hasattr(self.ds, "resolve_name"):
            return self.ds.resolve_name(name)
        return name

    def _resolve_list(self, names: list[str]) -> list[str]:
        return [self._resolve(n) for n in names]

    def _run_metric(self, metric_name: str):
        try:
            if metric_name == "VOM_VOA":
                entry_raw = self.metric_panel.split_names(self.metric_panel.vom_entry.text())
                target_raw = self.metric_panel.split_names(self.metric_panel.vom_target.text())
                if len(entry_raw) < 3 or len(target_raw) < 3:
                    raise ValueError("VOM_VOA expects entry and target polygons (>= 3 points each).")

                entry = self._resolve_list(entry_raw)
                target = self._resolve_list(target_raw)

                res = VOM_VOA(
                    data=self.ds,
                    entry=entry,
                    target=target,
                    stand_dist=float(self.metric_panel.stand_dist.value()),
                    return_debug=True,
                )

                self.metric_panel.out.setPlainText(
                    f"VoA (deg): {res.voa_deg:.3f}\n"
                    f"VOM (mm^3): {res.vom_mm3:.3f}\n"
                    f"sVOM (mm^3): {res.svom_mm3:.3f}\n\n"
                    f"Debug:\n{getattr(res, 'debug', None)}"
                )

                self.plot_panel.plot_vom_voa(
                    self.ds,
                    entry,
                    target,
                    debug=getattr(res, "debug", None),
                    show_polygons=bool(self.metric_panel.vom_show_polygons.isChecked()),
                    show_ellipses=bool(self.metric_panel.vom_show_ellipses.isChecked()),
                    show_target_distance=bool(self.metric_panel.vom_show_distance.isChecked()),
                    show_vom=bool(self.metric_panel.vom_show_vom.isChecked()),
                    show_svom=bool(self.metric_panel.vom_show_svom.isChecked()),
                    n_vom_slices=int(self.metric_panel.vom_slices.value()),
                    n_svom_slices=int(self.metric_panel.svom_slices.value()),
                    title="VOM / sVOM / VoA (3D)",
                )

            elif metric_name == "AOA_SF":
                import inspect

                cran = self._resolve(self.metric_panel.aoa_cranial.text())
                caud = self._resolve(self.metric_panel.aoa_caudal.text())
                med = self._resolve(self.metric_panel.aoa_medial.text())
                lat = self._resolve(self.metric_panel.aoa_lateral.text())
                piv = self._resolve(self.metric_panel.aoa_pivot.text())

                if not all([cran, caud, med, lat, piv]):
                    raise ValueError("AOA_SF requires cranial, caudal, medial, lateral, and pivot.")

                entry = [cran, caud, med, lat]

                constraints_raw = self.metric_panel.split_names(self.metric_panel.aoa_constraints.text())
                constraints = self._resolve_list(constraints_raw) if constraints_raw else None

                enable_rescale = bool(self.metric_panel.aoa_rescale_enable.isChecked())
                rescale_radius = float(self.metric_panel.aoa_rescale_radius.value()) if enable_rescale else None

                kwargs = dict(data=self.ds, entry=entry, target=piv, constraints=constraints, return_debug=True)

                sig = None
                try:
                    sig = inspect.signature(AOA_SF)
                except Exception:
                    sig = None

                if sig is not None and rescale_radius is not None:
                    if "sf_rescale_radius_mm" in sig.parameters:
                        kwargs["sf_rescale_radius_mm"] = rescale_radius
                    elif "rescale_radius_mm" in sig.parameters:
                        kwargs["rescale_radius_mm"] = rescale_radius

                res = AOA_SF(**kwargs)

                lines = [
                    f"AoA vertical (deg): {getattr(res, 'aoa_vertical_deg', float('nan')):.3f}",
                    f"AoA horizontal (deg): {getattr(res, 'aoa_horizontal_deg', float('nan')):.3f}",
                ]
                if hasattr(res, "sf_entry_area_mm2"):
                    lines.append(f"SF entry area (mm^2): {float(getattr(res, 'sf_entry_area_mm2')):.3f}")
                if hasattr(res, "sf_entry_area_rescaled_mm2"):
                    lines.append(f"SF entry area RESCALED (mm^2): {float(getattr(res, 'sf_entry_area_rescaled_mm2')):.3f}")
                if hasattr(res, "sf_constraints_proxy_mm2"):
                    lines.append(f"SF constraints proxy (mm^2): {float(getattr(res, 'sf_constraints_proxy_mm2')):.3f}")
                if hasattr(res, "sf_proxy_mm2"):
                    lines.append(f"SF proxy area (mm^2): {float(getattr(res, 'sf_proxy_mm2')):.3f}")

                dbg = getattr(res, "debug", None)
                if dbg is not None:
                    lines.append("\nDebug:")
                    lines.append(str(dbg))

                self.metric_panel.out.setPlainText("\n".join(lines))

                self.plot_panel.plot_aoa_sf(
                    self.ds,
                    cranial=cran,
                    caudal=caud,
                    medial=med,
                    lateral=lat,
                    pivot=piv,
                    constraints=constraints,
                    rescale_radius=rescale_radius,
                    title="AOA_SF view (triangles + SF + optional rescale)",
                )

            elif metric_name == "AOE":
                A = self._resolve(self.metric_panel.A_edit.text())
                B = self._resolve(self.metric_panel.B_edit.text())
                C = self._resolve(self.metric_panel.C_edit.text())

                if not all([A, B, C]):
                    raise ValueError("AOE requires A, B, and C.")

                res = AOE(data=self.ds, A=A, B=B, C=C, return_debug=True)

                self.metric_panel.out.setPlainText(
                    f"AoE (deg): {res.aoe_deg:.3f}\n\n"
                    f"Debug:\n{getattr(res, 'debug', None)}"
                )
                self.plot_panel.plot_aoe(self.ds, A, B, C)

            elif metric_name == "DISTANCE_3D":
                A = self._resolve(self.metric_panel.dist_A.text())
                B = self._resolve(self.metric_panel.dist_B.text())

                if not all([A, B]):
                    raise ValueError("DISTANCE_3D requires A and B.")

                res = DISTANCE_3D(data=self.ds, A=A, B=B, return_debug=True)

                self.metric_panel.out.setPlainText(
                    f"Distance (mm): {res.distance_mm:.3f}\n\n"
                    f"Debug:\n{getattr(res, 'debug', None)}"
                )
                self.plot_panel.plot_distance(self.ds, A, B)

            else:  # AREA_3D
                poly_raw = self.metric_panel.split_names(self.metric_panel.area_poly.text())
                if len(poly_raw) < 3:
                    raise ValueError("AREA_3D requires at least 3 polygon points.")
                poly = self._resolve_list(poly_raw)

                res = AREA_3D(data=self.ds, polygon=poly, return_debug=True)

                self.metric_panel.out.setPlainText(
                    f"Area (mm^2): {res.area_mm2:.3f}\n\n"
                    f"Debug:\n{getattr(res, 'debug', None)}"
                )
                self.plot_panel.plot_area(self.ds, poly)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))


# -------------------------
# Start window (load)
# -------------------------
class StartWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Surgiplot")
        self.resize(760, 460)

        self.main_window: MainWindow | None = None

        layout = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel("<h2>Surgiplot</h2><p>Select an input mode, then open the workspace.</p>")
        title.setTextFormat(QtCore.Qt.RichText)
        layout.addWidget(title)

        mode_group = QtWidgets.QGroupBox("Input mode")
        mode_layout = QtWidgets.QVBoxLayout(mode_group)

        self.rb_file = QtWidgets.QRadioButton("Load annotation file (txt/csv/tsv/space-delimited)")
        self.rb_manual = QtWidgets.QRadioButton("Manual entry (paste coordinates)")
        self.rb_image = QtWidgets.QRadioButton("Single surgical image (AI depth → pick points → workspace)")

        self.rb_file.setChecked(True)
        mode_layout.addWidget(self.rb_file)
        mode_layout.addWidget(self.rb_manual)
        mode_layout.addWidget(self.rb_image)

        # Disable image mode if dialog missing
        if not HAS_SINGLE_IMAGE_DIALOG:
            self.rb_image.setEnabled(False)
            self.rb_image.setToolTip(
                "AI image mode unavailable.\n\n"
                "1) Ensure file exists: surgiplot/ai/single_image_depth.py\n"
                "2) Install deps: pip install -U numpy pillow matplotlib transformers torch"
            )

        layout.addWidget(mode_group)

        src_layout = QtWidgets.QHBoxLayout()
        src_layout.addWidget(QtWidgets.QLabel("Source of data:"))
        self.source = QtWidgets.QComboBox()
        self.source.addItems(["navigation", "photogrammetry", "scanner", "single_image_depth"])
        src_layout.addWidget(self.source)
        src_layout.addStretch(1)
        layout.addLayout(src_layout)

        file_layout = QtWidgets.QHBoxLayout()
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setPlaceholderText("Select an annotation file...")
        btn_browse = QtWidgets.QPushButton("Browse")
        btn_browse.clicked.connect(self._browse)
        file_layout.addWidget(self.path_edit)
        file_layout.addWidget(btn_browse)
        layout.addLayout(file_layout)

        self.manual_text = QtWidgets.QPlainTextEdit()
        self.manual_text.setPlaceholderText("Manual entry format:\nname x y z\nOR\nx y z")
        self.manual_text.setVisible(False)
        layout.addWidget(self.manual_text)

        btn = QtWidgets.QPushButton("Open workspace")
        btn.clicked.connect(self._continue)
        layout.addWidget(btn)

        self.rb_file.toggled.connect(self._toggle_mode)
        self.rb_manual.toggled.connect(self._toggle_mode)
        self.rb_image.toggled.connect(self._toggle_mode)
        self._toggle_mode()

    def _toggle_mode(self):
        is_file = self.rb_file.isChecked()
        is_manual = self.rb_manual.isChecked()
        is_image = self.rb_image.isChecked()

        self.path_edit.setVisible(is_file)
        self.manual_text.setVisible(is_manual)

        # Hide both in image mode
        if is_image:
            self.path_edit.setVisible(False)
            self.manual_text.setVisible(False)

        # Suggest correct source for image mode
        if is_image:
            self.source.setCurrentText("single_image_depth")

    def _browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select annotation file", "", "Text/CSV (*.txt *.csv *.tsv);;All files (*)"
        )
        if path:
            self.path_edit.setText(path)

    def _parse_manual(self, raw: str, source: str):
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        pts = []
        names = []
        for ln in lines:
            parts = ln.split()
            if len(parts) == 4:
                names.append(parts[0])
                pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif len(parts) == 3:
                pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
            else:
                raise ValueError(f"Bad line: '{ln}'")
        if names:
            return load_points_from_manual(pts, names=names, source=source, alias_points=True)
        return load_points_from_manual(pts, names=None, source=source, alias_points=False)

    def _continue(self):
        src = self.source.currentText()

        # 1) AI image workflow (modal dialog, then open workspace)
        if self.rb_image.isChecked():
            if not HAS_SINGLE_IMAGE_DIALOG or SingleImage3DDialog is None:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Feature unavailable",
                    "AI image mode is not available.\n\n"
                    "1) Ensure file exists: surgiplot/ai/single_image_depth.py\n"
                    "2) Install deps: pip install -U numpy pillow matplotlib transformers torch",
                )
                return

            # Start from an empty dataset
            ds = load_points_from_manual([], names=None, source="single_image_depth", alias_points=False)

            dlg = SingleImage3DDialog(ds, parent=self)
            if dlg.exec() != QtWidgets.QDialog.Accepted:
                return

            self.main_window = MainWindow(ds)
            self.main_window.show()
            self.close()
            return

        # 2) File or manual
        if self.rb_file.isChecked():
            path = self.path_edit.text().strip()
            if not path:
                QtWidgets.QMessageBox.warning(self, "Missing file", "Please select an annotation file.")
                return
            ds = load_dataset(path, source=src, alias_points=True)
        else:
            raw = self.manual_text.toPlainText().strip()
            if not raw:
                QtWidgets.QMessageBox.warning(self, "Missing input", "Please paste coordinates for manual entry.")
                return
            ds = self._parse_manual(raw, source=src)

        self.main_window = MainWindow(ds)
        self.main_window.show()
        self.close()


def main():
    app = QtWidgets.QApplication([])
    w = StartWindow()
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
