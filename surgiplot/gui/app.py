from __future__ import annotations

from PySide6 import QtWidgets, QtCore

from surgiplot.core.io.loaders import load_dataset, load_points_from_manual
from surgiplot.metrics import VOM_VOA, AOA_SF, AOE, DISTANCE_3D, AREA_3D

# -------------------------
# Matplotlib 3D plot panel
# -------------------------
# Create plotting if missing + degrade gracefully if matplotlib is not installed.
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


def _project_point_fixed_distance(p, p_ref, distance: float):
    """Project point p to be at a fixed distance from p_ref along vector p_ref->p."""
    import numpy as np

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

    def plot_vom_voa(self, ds, entry: list[str], target: list[str], title="VOM_VOA view"):
        if not HAS_MPL:
            return

        self.clear()
        self._scatter_all(ds)

        def poly_coords(names):
            pts = [ds.points[n] for n in names]
            return (
                [float(p[0]) for p in pts],
                [float(p[1]) for p in pts],
                [float(p[2]) for p in pts],
            )

        ex, ey, ez = poly_coords(entry)
        tx, ty, tz = poly_coords(target)

        ex2, ey2, ez2 = ex + [ex[0]], ey + [ey[0]], ez + [ez[0]]
        tx2, ty2, tz2 = tx + [tx[0]], ty + [ty[0]], tz + [tz[0]]

        self.ax.plot(ex2, ey2, ez2, linewidth=2)
        self.ax.plot(tx2, ty2, tz2, linewidth=2)

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
        """
        Mirrors the legacy AoA plot style:
          - two triangles (V-AoA and H-AoA) sharing pivot
          - SF quadrilateral (entry polygon) filled
          - optional rescaled overlay + dotted projection lines to rescaled points
        """
        if not HAS_MPL:
            return

        import numpy as np

        # canonical convention for plotting:
        #   V-AoA triangle uses (cranial, caudal, pivot)
        #   H-AoA triangle uses (medial, lateral, pivot)
        p_cr = ds.points[cranial]
        p_ca = ds.points[caudal]
        p_me = ds.points[medial]
        p_la = ds.points[lateral]
        p_pi = ds.points[pivot]

        self.clear()
        self._scatter_all(ds)

        # --- helpers
        def _plot_triangle(p1, p2, p3, alpha=0.25, linewidth=2, dashed_base=False):
            # edges
            self.ax.plot(
                [float(p2[0]), float(p3[0])],
                [float(p2[1]), float(p3[1])],
                [float(p2[2]), float(p3[2])],
                linewidth=linewidth,
            )
            self.ax.plot(
                [float(p1[0]), float(p3[0])],
                [float(p1[1]), float(p3[1])],
                [float(p1[2]), float(p3[2])],
                linewidth=linewidth,
            )
            ls = "--" if dashed_base else "-"
            self.ax.plot(
                [float(p1[0]), float(p2[0])],
                [float(p1[1]), float(p2[1])],
                [float(p1[2]), float(p2[2])],
                linestyle=ls,
                linewidth=linewidth,
            )
            # fill
            if Poly3DCollection is not None:
                verts = [[np.asarray(p1), np.asarray(p2), np.asarray(p3)]]
                self.ax.add_collection3d(
                    Poly3DCollection(verts, alpha=alpha)
                )

        def _plot_quad(points_4, alpha=0.18, linewidth=2):
            pts = [np.asarray(p) for p in points_4]
            xs = [float(p[0]) for p in pts] + [float(pts[0][0])]
            ys = [float(p[1]) for p in pts] + [float(pts[0][1])]
            zs = [float(p[2]) for p in pts] + [float(pts[0][2])]
            self.ax.plot(xs, ys, zs, linewidth=linewidth)
            if Poly3DCollection is not None:
                self.ax.add_collection3d(Poly3DCollection([pts], alpha=alpha))

        def _dotted(p_from, p_to):
            self.ax.plot(
                [float(p_from[0]), float(p_to[0])],
                [float(p_from[1]), float(p_to[1])],
                [float(p_from[2]), float(p_to[2])],
                linestyle="--",
                linewidth=1.5,
            )

        # --- original geometry
        # triangles
        _plot_triangle(p_cr, p_ca, p_pi, alpha=0.22, dashed_base=True)   # V-AoA
        _plot_triangle(p_me, p_la, p_pi, alpha=0.22, dashed_base=True)   # H-AoA

        # SF entry quad
        _plot_quad([p_cr, p_ca, p_me, p_la], alpha=0.16, linewidth=2)

        # pivot highlight
        self.ax.scatter([float(p_pi[0])], [float(p_pi[1])], [float(p_pi[2])], s=90)

        # constraints (optional)
        if constraints:
            for nm in constraints:
                if nm in ds.points:
                    p = ds.points[nm]
                    self.ax.scatter([float(p[0])], [float(p[1])], [float(p[2])], s=55)

        # --- rescaled overlay
        if rescale_radius is not None and rescale_radius > 0:
            p_cr_r = _project_point_fixed_distance(p_cr, p_pi, rescale_radius)
            p_ca_r = _project_point_fixed_distance(p_ca, p_pi, rescale_radius)
            p_me_r = _project_point_fixed_distance(p_me, p_pi, rescale_radius)
            p_la_r = _project_point_fixed_distance(p_la, p_pi, rescale_radius)

            # dotted links
            _dotted(p_cr, p_cr_r)
            _dotted(p_ca, p_ca_r)
            _dotted(p_me, p_me_r)
            _dotted(p_la, p_la_r)

            # rescaled triangles + quad (lighter)
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
            import numpy as np
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
    applied = QtCore.Signal()  # emitted when Apply pressed

    def __init__(self, ds, parent=None):
        super().__init__(parent)
        self.ds = ds

        layout = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QLabel("<b>Dataset</b> (edit labels, then click Apply)")
        layout.addWidget(header)

        self.source_cb = QtWidgets.QComboBox()
        self.source_cb.addItems(["navigation", "photogrammetry", "scanner"])
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
    run_requested = QtCore.Signal(str)  # emits metric name when run pressed

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

        # AOA_SF rescaling (SF rescaling requested)
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

        self._set_row_visible(self.vom_entry, is_vom)
        self._set_row_visible(self.vom_target, is_vom)
        self._set_row_visible(self.stand_dist, is_vom)

        self._set_row_visible(self.aoa_cranial, is_aoa)
        self._set_row_visible(self.aoa_caudal, is_aoa)
        self._set_row_visible(self.aoa_medial, is_aoa)
        self._set_row_visible(self.aoa_lateral, is_aoa)
        self._set_row_visible(self.aoa_pivot, is_aoa)
        self._set_row_visible(self.aoa_constraints, is_aoa)
        self._set_row_visible(self.aoa_rescale_enable, is_aoa)
        self._set_row_visible(self.aoa_rescale_radius, is_aoa and self.aoa_rescale_enable.isChecked())

        self._set_row_visible(self.A_edit, is_aoe)
        self._set_row_visible(self.B_edit, is_aoe)
        self._set_row_visible(self.C_edit, is_aoe)

        self._set_row_visible(self.dist_A, is_dist)
        self._set_row_visible(self.dist_B, is_dist)

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

        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)
        lay.addWidget(splitter)
        self.setCentralWidget(central)

        self.dataset_panel.applied.connect(self._on_dataset_applied)
        self.metric_panel.run_requested.connect(self._run_metric)
        self.metric_panel.aoa_rescale_enable.stateChanged.connect(lambda _s: self.metric_panel._update_visibility(self.metric_panel.metric_cb.currentText()))

        self.plot_panel.plot_points(self.ds, title="Dataset overview")

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
                    f"Debug:\n{res.debug}"
                )
                self.plot_panel.plot_vom_voa(self.ds, entry, target)

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

                # Call AOA_SF in a way that stays compatible if your signature changes.
                kwargs = dict(data=self.ds, entry=entry, target=piv, constraints=constraints, return_debug=True)
                sig = None
                try:
                    sig = inspect.signature(AOA_SF)
                except Exception:
                    sig = None

                if sig is not None:
                    if "sf_rescale_radius_mm" in sig.parameters and rescale_radius is not None:
                        kwargs["sf_rescale_radius_mm"] = rescale_radius
                    if "rescale_radius_mm" in sig.parameters and rescale_radius is not None:
                        kwargs["rescale_radius_mm"] = rescale_radius

                res = AOA_SF(**kwargs)

                # Robust printing: support both old and new AOASFResult fields
                lines = [
                    f"AoA vertical (deg): {getattr(res, 'aoa_vertical_deg', float('nan')):.3f}",
                    f"AoA horizontal (deg): {getattr(res, 'aoa_horizontal_deg', float('nan')):.3f}",
                ]

                # prefer new SF fields if present
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
                    f"Debug:\n{res.debug}"
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
                    f"Debug:\n{res.debug}"
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
                    f"Debug:\n{res.debug}"
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
        self.resize(720, 420)

        self.main_window: MainWindow | None = None

        layout = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel("<h2>Surgiplot</h2><p>Load points then use the workspace.</p>")
        title.setTextFormat(QtCore.Qt.RichText)
        layout.addWidget(title)

        mode_group = QtWidgets.QGroupBox("Input mode")
        mode_layout = QtWidgets.QVBoxLayout(mode_group)
        self.rb_file = QtWidgets.QRadioButton("Load annotation file (txt/csv/tsv/space-delimited)")
        self.rb_manual = QtWidgets.QRadioButton("Manual entry (paste coordinates)")
        self.rb_file.setChecked(True)
        mode_layout.addWidget(self.rb_file)
        mode_layout.addWidget(self.rb_manual)
        layout.addWidget(mode_group)

        src_layout = QtWidgets.QHBoxLayout()
        src_layout.addWidget(QtWidgets.QLabel("Source of data:"))
        self.source = QtWidgets.QComboBox()
        self.source.addItems(["navigation", "photogrammetry", "scanner"])
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

    def _toggle_mode(self):
        is_file = self.rb_file.isChecked()
        self.path_edit.setVisible(is_file)
        self.manual_text.setVisible(not is_file)

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
