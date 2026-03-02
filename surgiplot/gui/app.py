"""
surgiplot.gui.app

Surgiplot GUI — single-file app (copy-paste ready)

What this file contains:
- A working Qt (PySide6) GUI entrypoint with `main()` (fixes your ImportError).
- Dataset loading (CSV/JSON via your loader), manual point loading, and point table editing.
- Metric computation + plotting hooks for:
  - AoA/SF (AOA_SF)
  - VOM/VoA (VOM_VOA)
  - AoE (AOE)
  - Distance (DISTANCE_3D)
  - Area (AREA_3D)
- Optional “Single image → Depth → 3D point picking” dialog (kept separated in
  `surgiplot.ai.single_image_depth` as you intended). If that module/deps are missing,
  the GUI still runs and simply disables that feature.

Assumptions about your package (based on your repo history):
- `load_dataset`, `load_points_from_manual` exist in `surgiplot.core.io.loaders`
- Metrics classes/functions exist in `surgiplot.metrics`:
  `VOM_VOA, AOA_SF, AOE, DISTANCE_3D, AREA_3D`
- Your dataset object has at least:
  - `ds.points` : dict[str, np.ndarray shape (3,)]
  - optionally `ds.meta` : dict
  - optionally `ds.name` / `ds.path`

If your dataset class differs, adjust only the tiny adapter methods in DatasetPanel.
"""

from __future__ import annotations

import os
import sys
import traceback
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import numpy as np
from PySide6 import QtWidgets, QtCore

# -------------------------
# Core project imports
# -------------------------
from surgiplot.core.io.loaders import load_dataset, load_points_from_manual
from surgiplot.metrics import VOM_VOA, AOA_SF, AOE, DISTANCE_3D, AREA_3D

# Optional AI dialog (kept separate on purpose)
try:
    from surgiplot.ai.single_image_depth import SingleImage3DDialog  # feature-only
    HAS_SINGLE_IMAGE_3D = True
except Exception:
    SingleImage3DDialog = None  # type: ignore
    HAS_SINGLE_IMAGE_3D = False

# -------------------------
# Matplotlib (optional but strongly recommended)
# -------------------------
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    HAS_MPL = True
except Exception:
    FigureCanvas = None  # type: ignore
    Figure = None  # type: ignore
    HAS_MPL = False


# =============================================================================
# Small helpers
# =============================================================================
def _exc_text() -> str:
    return traceback.format_exc()


def _ensure_points_dict(ds) -> Dict[str, np.ndarray]:
    if not hasattr(ds, "points") or ds.points is None:
        ds.points = {}
    if not isinstance(ds.points, dict):
        raise TypeError("Dataset ds.points must be a dict[label -> xyz]")
    return ds.points


def _as_xyz(arr: Any) -> np.ndarray:
    a = np.asarray(arr, dtype=float).reshape(-1)
    if a.size != 3:
        raise ValueError("Point must be 3D (x,y,z).")
    return a.reshape(3,)


# =============================================================================
# Plot panel
# =============================================================================
class PlotPanel(QtWidgets.QWidget):
    """
    A simple matplotlib viewer with multiple plotting utilities.
    Your metric classes should provide their own plotting methods; this panel
    calls them if available, otherwise falls back to minimal visualization.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        lay = QtWidgets.QVBoxLayout(self)
        if not HAS_MPL:
            lay.addWidget(
                QtWidgets.QLabel(
                    "Matplotlib is not installed, so plotting is disabled.\n\n"
                    "Install it in your env:\n  pip install -U matplotlib"
                )
            )
            self.fig = None
            self.canvas = None
            self.ax = None
            return

        self.fig = Figure(figsize=(6, 5))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection="3d")
        lay.addWidget(self.canvas)

        self._title = QtWidgets.QLabel("")
        self._title.setWordWrap(True)
        lay.addWidget(self._title)

    def clear(self, title: str = ""):
        if not HAS_MPL:
            return
        self.fig.clear()
        self.ax = self.fig.add_subplot(111, projection="3d")
        self._title.setText(title or "")
        self.canvas.draw_idle()

    def plot_points(self, points: Dict[str, np.ndarray], title: str = "Dataset points"):
        if not HAS_MPL:
            return
        self.clear(title)

        if not points:
            self._title.setText("No points to plot.")
            self.canvas.draw_idle()
            return

        labels = list(points.keys())
        xyz = np.vstack([_as_xyz(points[k]) for k in labels])

        self.ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=20, alpha=0.85)
        for i, lb in enumerate(labels):
            p = xyz[i]
            self.ax.text(p[0], p[1], p[2], lb, fontsize=9)

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.canvas.draw_idle()

    def plot_metric_result(self, result: Any, title: str = "Metric result"):
        """
        Tries a few conventions:
        - if result has `plot(ax=...)` or `plot(ax)` method
        - else if result has `fig`/`ax`
        - else prints as text in title
        """
        if not HAS_MPL:
            return

        self.clear(title)

        # 1) result.plot(ax=...)
        try:
            if hasattr(result, "plot"):
                try:
                    result.plot(ax=self.ax)
                    self.canvas.draw_idle()
                    return
                except TypeError:
                    # maybe result.plot(ax) signature
                    result.plot(self.ax)
                    self.canvas.draw_idle()
                    return
        except Exception:
            pass

        # 2) If metric object returns a dict with points/lines
        if isinstance(result, dict):
            # common keys: "points", "lines"
            pts = result.get("points", None)
            if isinstance(pts, dict):
                self.plot_points(pts, title=title)
                return

        # 3) Fallback: show dataset text
        self._title.setText(f"{title}\n\n{result!r}")
        self.canvas.draw_idle()


# =============================================================================
# Dataset panel (load/edit points)
# =============================================================================
class DatasetPanel(QtWidgets.QWidget):
    dataset_changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.ds = None  # dataset object

        self.lbl_path = QtWidgets.QLabel("No dataset loaded.")
        self.lbl_path.setWordWrap(True)

        self.btn_load = QtWidgets.QPushButton("Load dataset…")
        self.btn_load_manual = QtWidgets.QPushButton("Import points (manual)…")
        self.btn_save_points = QtWidgets.QPushButton("Export points table…")
        self.btn_clear_points = QtWidgets.QPushButton("Clear all points")

        self.btn_add_point = QtWidgets.QPushButton("Add / update point…")
        self.btn_del_point = QtWidgets.QPushButton("Delete selected point")

        self.btn_single_image_3d = QtWidgets.QPushButton("Single image → 3D points…")
        self.btn_single_image_3d.setEnabled(HAS_SINGLE_IMAGE_3D)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Label", "X", "Y", "Z"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

        # Layout
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.btn_load)
        top.addWidget(self.btn_load_manual)
        top.addWidget(self.btn_single_image_3d)
        top.addStretch(1)
        top.addWidget(self.btn_save_points)
        top.addWidget(self.btn_clear_points)

        mid = QtWidgets.QHBoxLayout()
        mid.addWidget(self.btn_add_point)
        mid.addWidget(self.btn_del_point)
        mid.addStretch(1)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.lbl_path)
        lay.addLayout(top)
        lay.addLayout(mid)
        lay.addWidget(self.table, 1)

        # Wiring
        self.btn_load.clicked.connect(self._on_load_dataset)
        self.btn_load_manual.clicked.connect(self._on_load_manual_points)
        self.btn_save_points.clicked.connect(self._on_export_points)
        self.btn_clear_points.clicked.connect(self._on_clear_points)

        self.btn_add_point.clicked.connect(self._on_add_point)
        self.btn_del_point.clicked.connect(self._on_delete_selected)

        self.btn_single_image_3d.clicked.connect(self._on_single_image_3d)

        self._refresh()

    # ---- dataset IO
    def _on_load_dataset(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Surgiplot dataset",
            "",
            "Dataset files (*.json *.csv *.tsv *.xlsx *.pkl *.pickle);;All files (*)",
        )
        if not path:
            return
        try:
            self.ds = load_dataset(path)
            _ensure_points_dict(self.ds)
            if isinstance(getattr(self.ds, "meta", None), dict):
                self.ds.meta["loaded_from"] = path
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load dataset failed", f"{e}\n\n{_exc_text()}")
            return
        self._refresh()
        self.dataset_changed.emit()

    def _on_load_manual_points(self):
        if self.ds is None:
            QtWidgets.QMessageBox.warning(self, "No dataset", "Load a dataset first.")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import points from manual file",
            "",
            "Point files (*.csv *.tsv *.txt *.json);;All files (*)",
        )
        if not path:
            return
        try:
            pts = load_points_from_manual(path)
            if not isinstance(pts, dict):
                raise TypeError("load_points_from_manual must return dict[label->xyz]")
            _ensure_points_dict(self.ds).update({str(k): _as_xyz(v) for k, v in pts.items()})
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Import points failed", f"{e}\n\n{_exc_text()}")
            return
        self._refresh()
        self.dataset_changed.emit()

    def _on_export_points(self):
        if self.ds is None:
            QtWidgets.QMessageBox.warning(self, "No dataset", "Load a dataset first.")
            return
        points = _ensure_points_dict(self.ds)
        if not points:
            QtWidgets.QMessageBox.information(self, "No points", "No points to export.")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export points as CSV", "points.csv", "CSV (*.csv);;All files (*)"
        )
        if not path:
            return

        try:
            import csv

            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["label", "x", "y", "z"])
                for lb in sorted(points.keys()):
                    p = _as_xyz(points[lb])
                    w.writerow([lb, float(p[0]), float(p[1]), float(p[2])])
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export failed", f"{e}\n\n{_exc_text()}")
            return

        QtWidgets.QMessageBox.information(self, "Exported", f"Saved:\n{path}")

    def _on_clear_points(self):
        if self.ds is None:
            return
        resp = QtWidgets.QMessageBox.question(
            self,
            "Clear all points",
            "Clear ALL dataset points?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if resp != QtWidgets.QMessageBox.Yes:
            return
        _ensure_points_dict(self.ds).clear()
        self._refresh()
        self.dataset_changed.emit()

    # ---- point editing
    def _on_add_point(self):
        if self.ds is None:
            QtWidgets.QMessageBox.warning(self, "No dataset", "Load a dataset first.")
            return
        label, ok = QtWidgets.QInputDialog.getText(self, "Point label", "Label:")
        if not ok:
            return
        label = (label or "").strip()
        if not label:
            return

        xyz_text, ok = QtWidgets.QInputDialog.getText(
            self,
            "Point coordinates",
            "Enter x,y,z (comma-separated):",
        )
        if not ok:
            return
        try:
            parts = [float(x.strip()) for x in (xyz_text or "").replace(";", ",").split(",")]
            if len(parts) != 3:
                raise ValueError("Need exactly 3 values.")
            p = np.array(parts, dtype=float).reshape(3,)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Invalid coordinates", f"{e}")
            return

        _ensure_points_dict(self.ds)[label] = p
        self._refresh()
        self.dataset_changed.emit()

    def _selected_label(self) -> Optional[str]:
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            return None
        row = int(sel[0].row())
        it = self.table.item(row, 0)
        if it is None:
            return None
        s = it.text().strip()
        return s or None

    def _on_delete_selected(self):
        if self.ds is None:
            return
        lb = self._selected_label()
        if not lb:
            return
        resp = QtWidgets.QMessageBox.question(
            self,
            "Delete point",
            f"Delete point '{lb}'?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if resp != QtWidgets.QMessageBox.Yes:
            return
        _ensure_points_dict(self.ds).pop(lb, None)
        self._refresh()
        self.dataset_changed.emit()

    # ---- optional single-image 3D picker
    def _on_single_image_3d(self):
        if not HAS_SINGLE_IMAGE_3D:
            QtWidgets.QMessageBox.information(
                self,
                "Feature not available",
                "Single-image 3D feature is not available in this environment.\n"
                "Make sure `surgiplot.ai.single_image_depth` exists and its dependencies are installed.",
            )
            return
        if self.ds is None:
            QtWidgets.QMessageBox.warning(self, "No dataset", "Load a dataset first.")
            return

        dlg = SingleImage3DDialog(self.ds, on_dataset_changed_callback=self._on_ai_points_committed, parent=self)
        dlg.exec()

    def _on_ai_points_committed(self):
        self._refresh()
        self.dataset_changed.emit()

    # ---- refresh table
    def _refresh(self):
        if self.ds is None:
            self.lbl_path.setText("No dataset loaded.")
            self.table.setRowCount(0)
            return

        path = ""
        if isinstance(getattr(self.ds, "meta", None), dict):
            path = str(self.ds.meta.get("loaded_from", ""))

        if not path:
            path = str(getattr(self.ds, "path", "") or "")

        name = str(getattr(self.ds, "name", "") or "")
        hdr = "Loaded dataset"
        if name:
            hdr += f": {name}"
        if path:
            hdr += f"\n{path}"
        self.lbl_path.setText(hdr)

        pts = _ensure_points_dict(self.ds)
        labels = sorted(pts.keys())
        self.table.setRowCount(len(labels))

        for r, lb in enumerate(labels):
            p = _as_xyz(pts[lb])
            it0 = QtWidgets.QTableWidgetItem(lb)
            it1 = QtWidgets.QTableWidgetItem(f"{float(p[0]):.6f}")
            it2 = QtWidgets.QTableWidgetItem(f"{float(p[1]):.6f}")
            it3 = QtWidgets.QTableWidgetItem(f"{float(p[2]):.6f}")

            # read-only cells
            for it in (it0, it1, it2, it3):
                it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)

            self.table.setItem(r, 0, it0)
            self.table.setItem(r, 1, it1)
            self.table.setItem(r, 2, it2)
            self.table.setItem(r, 3, it3)


# =============================================================================
# Metric panel (compute & plot)
# =============================================================================
class MetricPanel(QtWidgets.QWidget):
    """
    UI to compute different metrics using your `surgiplot.metrics` classes.
    """

    def __init__(self, dataset_panel: DatasetPanel, plot_panel: PlotPanel, parent=None):
        super().__init__(parent)
        self.dataset_panel = dataset_panel
        self.plot_panel = plot_panel

        self.cmb_metric = QtWidgets.QComboBox()
        self.cmb_metric.addItems(["AoA / SF", "VOM / VoA", "AoE", "Distance", "Area", "Plot points only"])

        self.btn_compute = QtWidgets.QPushButton("Compute")
        self.btn_plot_points = QtWidgets.QPushButton("Plot points")

        # VOM/VoA options (kept minimal but compatible with your recent work)
        self.grp_vom = QtWidgets.QGroupBox("VOM / VoA options")
        self.grp_vom.setCheckable(True)
        self.grp_vom.setChecked(False)

        self.chk_polygons = QtWidgets.QCheckBox("Polygons")
        self.chk_polygons.setChecked(True)
        self.chk_ellipses = QtWidgets.QCheckBox("Ellipses")
        self.chk_ellipses.setChecked(True)
        self.chk_distance = QtWidgets.QCheckBox("Target distance vector")
        self.chk_distance.setChecked(True)
        self.chk_vom = QtWidgets.QCheckBox("VOM")
        self.chk_vom.setChecked(True)
        self.chk_svom = QtWidgets.QCheckBox("sVOM (10 mm)")
        self.chk_svom.setChecked(True)

        self.spin_slices = QtWidgets.QSpinBox()
        self.spin_slices.setRange(5, 400)
        self.spin_slices.setValue(80)

        g = QtWidgets.QGridLayout(self.grp_vom)
        g.addWidget(self.chk_polygons, 0, 0)
        g.addWidget(self.chk_ellipses, 0, 1)
        g.addWidget(self.chk_distance, 1, 0)
        g.addWidget(self.chk_vom, 1, 1)
        g.addWidget(self.chk_svom, 2, 0)
        g.addWidget(QtWidgets.QLabel("Slices:"), 2, 1)
        g.addWidget(self.spin_slices, 2, 2)

        lay = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Metric:"))
        top.addWidget(self.cmb_metric, 1)
        top.addWidget(self.btn_plot_points)
        top.addWidget(self.btn_compute)
        lay.addLayout(top)
        lay.addWidget(self.grp_vom)

        self.txt_out = QtWidgets.QPlainTextEdit()
        self.txt_out.setReadOnly(True)
        lay.addWidget(self.txt_out, 1)

        # wiring
        self.btn_compute.clicked.connect(self.compute_current)
        self.btn_plot_points.clicked.connect(self.plot_points)

    def _ds(self):
        return self.dataset_panel.ds

    def plot_points(self):
        ds = self._ds()
        if ds is None:
            QtWidgets.QMessageBox.warning(self, "No dataset", "Load a dataset first.")
            return
        pts = _ensure_points_dict(ds)
        self.plot_panel.plot_points(pts, title="Dataset points")

    def compute_current(self):
        ds = self._ds()
        if ds is None:
            QtWidgets.QMessageBox.warning(self, "No dataset", "Load a dataset first.")
            return

        metric_name = self.cmb_metric.currentText().strip()
        pts = _ensure_points_dict(ds)

        if metric_name == "Plot points only":
            self.plot_points()
            return

        try:
            if metric_name == "AoA / SF":
                metric = AOA_SF()
                result = metric.compute(pts) if hasattr(metric, "compute") else metric(pts)  # support both styles
                self._show_result("AOA_SF", result)
                self.plot_panel.plot_metric_result(result, title="AoA / SF")
                return

            if metric_name == "VOM / VoA":
                metric = VOM_VOA()
                toggles = dict(
                    show_polygons=bool(self.chk_polygons.isChecked()),
                    show_ellipses=bool(self.chk_ellipses.isChecked()),
                    show_distance=bool(self.chk_distance.isChecked()),
                    show_vom=bool(self.chk_vom.isChecked()),
                    show_svom=bool(self.chk_svom.isChecked()),
                    slices=int(self.spin_slices.value()),
                )
                # support either compute(points, **opts) or compute(ds, **opts)
                if hasattr(metric, "compute"):
                    try:
                        result = metric.compute(pts, **toggles)
                    except TypeError:
                        # maybe it wants dataset
                        result = metric.compute(ds, **toggles)
                else:
                    result = metric(pts, **toggles)
                self._show_result("VOM_VOA", result)
                self.plot_panel.plot_metric_result(result, title="VOM / VoA")
                return

            if metric_name == "AoE":
                metric = AOE()
                result = metric.compute(pts) if hasattr(metric, "compute") else metric(pts)
                self._show_result("AOE", result)
                self.plot_panel.plot_metric_result(result, title="AoE")
                return

            if metric_name == "Distance":
                metric = DISTANCE_3D()
                result = metric.compute(pts) if hasattr(metric, "compute") else metric(pts)
                self._show_result("DISTANCE_3D", result)
                self.plot_panel.plot_metric_result(result, title="Distance")
                return

            if metric_name == "Area":
                metric = AREA_3D()
                result = metric.compute(pts) if hasattr(metric, "compute") else metric(pts)
                self._show_result("AREA_3D", result)
                self.plot_panel.plot_metric_result(result, title="Area")
                return

            QtWidgets.QMessageBox.warning(self, "Unknown metric", f"Unhandled metric: {metric_name}")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Metric computation failed", f"{e}\n\n{_exc_text()}")

    def _show_result(self, title: str, result: Any):
        # Pretty-print common result shapes.
        out_lines = [f"{title} result:"]
        if isinstance(result, dict):
            for k in sorted(result.keys()):
                v = result[k]
                if isinstance(v, (float, int, np.floating, np.integer)):
                    out_lines.append(f"- {k}: {float(v):.6g}")
                else:
                    out_lines.append(f"- {k}: {v!r}")
        else:
            out_lines.append(repr(result))
        self.txt_out.setPlainText("\n".join(out_lines))


# =============================================================================
# Main window
# =============================================================================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Surgiplot")
        self.resize(1400, 850)

        self.dataset_panel = DatasetPanel()
        self.plot_panel = PlotPanel()
        self.metric_panel = MetricPanel(self.dataset_panel, self.plot_panel)

        # Update plot when dataset changes (optional)
        self.dataset_panel.dataset_changed.connect(self._on_dataset_changed)

        # Layout: left dataset + metrics, right plot
        left = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        left.addWidget(self.dataset_panel)
        left.addWidget(self.metric_panel)
        left.setSizes([480, 330])

        main = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main.addWidget(left)
        main.addWidget(self.plot_panel)
        main.setSizes([560, 840])

        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(main)
        self.setCentralWidget(w)

        self._build_menu()

    def _build_menu(self):
        m = self.menuBar()

        file_menu = m.addMenu("&File")

        act_quit = QtGuiAction("Quit", self, shortcut="Ctrl+Q", triggered=self.close)
        file_menu.addAction(act_quit)

        help_menu = m.addMenu("&Help")
        act_about = QtGuiAction("About", self, triggered=self._about)
        help_menu.addAction(act_about)

    def _about(self):
        msg = (
            "Surgiplot GUI\n\n"
            "- Load dataset + points\n"
            "- Compute AoA/SF, VOM/VoA, AoE, Distance, Area\n"
            "- Optional single-image 3D point collection (if installed)\n"
        )
        QtWidgets.QMessageBox.information(self, "About Surgiplot", msg)

    def _on_dataset_changed(self):
        # optional: auto-plot points whenever dataset changes
        ds = self.dataset_panel.ds
        if ds is None:
            return
        try:
            self.plot_panel.plot_points(_ensure_points_dict(ds), title="Dataset points")
        except Exception:
            pass


# Simple QAction wrapper to avoid QtGui import at top if you want
class QtGuiAction(QtWidgets.QAction):
    def __init__(self, text, parent=None, shortcut=None, triggered=None):
        super().__init__(text, parent)
        if shortcut:
            self.setShortcut(shortcut)
        if triggered:
            self.triggered.connect(triggered)


# =============================================================================
# Entry point (THIS fixes your ImportError)
# =============================================================================
def main() -> None:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    win = MainWindow()
    win.show()

    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
