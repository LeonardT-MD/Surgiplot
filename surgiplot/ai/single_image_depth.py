"""
surgiplot.ai.single_image_depth

Single-image depth -> 3D reconstruction + point picking dialog, separated from app.py.

Dependencies (feature-only):
  pip install -U numpy pillow matplotlib transformers torch

Notes:
- Uses HuggingFace transformers "depth-estimation" pipeline (Depth Anything V2).
- Depth is relative; scale can be set interactively from two 2D clicks (known distance in mm).
- Click 3D point cloud -> prompt label -> stores into a BUFFER table (edit/remove).
- Press OK -> commits buffered points into ds.points (optionally overwriting).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
from PySide6 import QtWidgets, QtCore

# Matplotlib is required for this feature. Keep it local to avoid breaking core GUI if missing.
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:
    HAS_MPL = False
    FigureCanvas = None  # type: ignore
    Figure = None  # type: ignore
    plt = None  # type: ignore


# -------------------------
# Core math helpers
# -------------------------
@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fov_deg: float = 60.0

    def __post_init__(self):
        self.width = int(self.width)
        self.height = int(self.height)
        fov_rad = math.radians(float(self.fov_deg))
        self.fx = 0.5 * self.width / math.tan(0.5 * fov_rad)
        self.fy = self.fx
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0


def robust_norm_depth(d: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Robustly normalize depth-like values to [0, 1] using 1st-99th percentiles."""
    d = np.asarray(d, dtype=np.float32)
    p1, p99 = np.percentile(d, [1, 99])
    d = (d - p1) / (p99 - p1 + eps)
    return np.clip(d, 0.0, 1.0)


def backproject(depth_m: np.ndarray, K: CameraIntrinsics, stride: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backproject a depth map into a point cloud (camera coordinates).

    Returns:
      xyz: (N,3)
      uv : (N,2) pixel coordinates for each 3D point
    """
    depth_m = np.asarray(depth_m, dtype=np.float32)
    h, w = depth_m.shape[:2]

    ys = np.arange(0, h, stride, dtype=np.int32)
    xs = np.arange(0, w, stride, dtype=np.int32)
    gx, gy = np.meshgrid(xs, ys)

    u = gx.reshape(-1).astype(np.float32)
    v = gy.reshape(-1).astype(np.float32)
    z = depth_m[gy, gx].reshape(-1).astype(np.float32)

    valid = np.isfinite(z) & (z > 1e-9)
    u, v, z = u[valid], v[valid], z[valid]

    x = (u - K.cx) * z / K.fx
    y = (v - K.cy) * z / K.fy

    xyz = np.stack([x, y, z], axis=1)
    uv = np.stack([u, v], axis=1)
    return xyz, uv


# -------------------------
# Depth estimation (AI)
# -------------------------
class DepthEstimator:
    """
    Thin wrapper around transformers depth-estimation pipeline.
    Lazily loads the model.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._pipe = None

    def _ensure(self):
        if self._pipe is not None:
            return

        try:
            from transformers import pipeline
        except Exception as e:
            raise RuntimeError("Missing dependency: transformers. Install with: pip install transformers") from e

        try:
            import torch  # noqa: F401
        except Exception as e:
            raise RuntimeError("Missing dependency: torch. Install CPU or CUDA torch first.") from e

        self._pipe = pipeline(task="depth-estimation", model=self.model_id)

    def predict_depth_raw(self, img_rgb: np.ndarray) -> np.ndarray:
        """Returns a 2D float32 array derived from the pipeline output depth image."""
        self._ensure()

        try:
            from PIL import Image
        except Exception as e:
            raise RuntimeError("Missing dependency: pillow. Install with: pip install pillow") from e

        img = Image.fromarray(np.asarray(img_rgb, dtype=np.uint8))
        out = self._pipe(img)
        depth_img = out["depth"]
        depth_raw = np.array(depth_img).astype(np.float32)
        return depth_raw


# -------------------------
# 3D picker widget
# -------------------------
class PointCloudPicker3D(QtWidgets.QWidget):
    """
    Embedded matplotlib 3D scatter that emits picked xyz.
    """
    picked = QtCore.Signal(object)  # np.ndarray shape (3,)

    def __init__(self, parent=None):
        super().__init__(parent)

        lay = QtWidgets.QVBoxLayout(self)

        if not HAS_MPL:
            lay.addWidget(QtWidgets.QLabel("Matplotlib missing; cannot show 3D picker."))
            self.fig = None
            self.canvas = None
            self.ax = None
            self._xyz = None
            return

        self.fig = Figure(figsize=(6, 5))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection="3d")
        lay.addWidget(self.canvas)

        self._xyz: Optional[np.ndarray] = None
        self._sc = None
        self.canvas.mpl_connect("pick_event", self._on_pick)

    def set_xyz(self, xyz: np.ndarray):
        if not HAS_MPL:
            return

        xyz = np.asarray(xyz, dtype=np.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must be (N,3)")

        self._xyz = xyz
        self.ax.clear()
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")

        # picker tolerance: increase if needed
        self._sc = self.ax.scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2],
            s=2, alpha=0.75, picker=6
        )
        self.ax.view_init(elev=20, azim=-60)
        self.canvas.draw_idle()

    def _on_pick(self, event):
        if self._xyz is None:
            return
        ind = getattr(event, "ind", None)
        if ind is None or len(ind) == 0:
            return
        i = int(ind[0])
        self.picked.emit(self._xyz[i].copy())


# -------------------------
# Picked points buffer panel (label/edit/remove before committing)
# -------------------------
class PickedPointsPanel(QtWidgets.QWidget):
    """
    Table showing buffered picked points (label, x, y, z). Label is editable; coords are read-only.
    """
    changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._points: Dict[str, np.ndarray] = {}

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(QtWidgets.QLabel("<b>Picked points (buffer)</b> — edit labels / remove, then OK to commit"))

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["label", "x", "y", "z"])
        self.table.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(self.table)

        btns = QtWidgets.QHBoxLayout()
        self.btn_remove = QtWidgets.QPushButton("Remove selected")
        self.btn_clear = QtWidgets.QPushButton("Clear all")
        btns.addWidget(self.btn_remove)
        btns.addWidget(self.btn_clear)
        btns.addStretch(1)
        lay.addLayout(btns)

        self.btn_remove.clicked.connect(self.remove_selected)
        self.btn_clear.clicked.connect(self.clear)
        self.table.itemChanged.connect(self._on_item_changed)

        self._refresh()

    def points(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self._points.items()}

    def upsert(self, label: str, xyz: np.ndarray):
        label = str(label).strip()
        if not label:
            return
        self._points[label] = np.asarray(xyz, dtype=float).reshape(3,)
        self._refresh()
        self.changed.emit()

    def remove_selected(self):
        rows = sorted({it.row() for it in self.table.selectedItems()}, reverse=True)
        if not rows:
            return

        labels = []
        for r in rows:
            it = self.table.item(r, 0)
            if it:
                labels.append(it.text().strip())

        for lb in labels:
            self._points.pop(lb, None)

        self._refresh()
        self.changed.emit()

    def clear(self):
        self._points.clear()
        self._refresh()
        self.changed.emit()

    def _refresh(self):
        self.table.blockSignals(True)

        items = list(self._points.items())
        self.table.setRowCount(len(items))

        for i, (label, xyz) in enumerate(items):
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(label)))
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{float(xyz[0]):.6f}"))
            self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{float(xyz[1]):.6f}"))
            self.table.setItem(i, 3, QtWidgets.QTableWidgetItem(f"{float(xyz[2]):.6f}"))

        # set editability: label editable, coords read-only
        for r in range(self.table.rowCount()):
            it0 = self.table.item(r, 0)
            it0.setFlags(it0.flags() | QtCore.Qt.ItemIsEditable)
            for c in (1, 2, 3):
                it = self.table.item(r, c)
                it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)

        self.table.blockSignals(False)

    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem):
        # only label edits
        if item.column() != 0:
            return

        rebuilt: Dict[str, np.ndarray] = {}
        labels = []

        for r in range(self.table.rowCount()):
            lb = self.table.item(r, 0).text().strip()
            if not lb:
                QtWidgets.QMessageBox.warning(self, "Invalid label", "Labels cannot be empty.")
                self._refresh()
                return
            labels.append(lb)

            x = float(self.table.item(r, 1).text())
            y = float(self.table.item(r, 2).text())
            z = float(self.table.item(r, 3).text())
            rebuilt[lb] = np.array([x, y, z], dtype=float)

        if len(set(labels)) != len(labels):
            QtWidgets.QMessageBox.warning(self, "Duplicate labels", "Two points share the same label. Please fix.")
            self._refresh()
            return

        self._points = rebuilt
        self.changed.emit()


# -------------------------
# Dialog: single-image -> 3D -> commit buffered points into dataset
# -------------------------
class SingleImage3DDialog(QtWidgets.QDialog):
    """
    Workflow:
      - Load image
      - Run depth estimation -> point cloud
      - Optional: set scale using 2D clicks + known distance (mm)
      - Click points in 3D -> prompt label -> add to BUFFER (editable/removable)
      - OK commits buffered points into ds.points
    """

    def __init__(self, ds, parent=None):
        super().__init__(parent)

        if not HAS_MPL:
            QtWidgets.QMessageBox.critical(
                self,
                "Missing dependency",
                "Matplotlib is required for Single-image 3D points.\n\nInstall: pip install matplotlib",
            )
            raise RuntimeError("Matplotlib missing")

        self.ds = ds

        self.setWindowTitle("Single Image → Depth → 3D → Pick points (OK to commit)")
        self.setModal(True)
        self.resize(1400, 760)

        self.image_path: Optional[str] = None
        self.img_rgb: Optional[np.ndarray] = None
        self.depth_m: Optional[np.ndarray] = None
        self.K: Optional[CameraIntrinsics] = None
        self.xyz: Optional[np.ndarray] = None

        # controls
        self.btn_load = QtWidgets.QPushButton("Load image…")
        self.btn_run = QtWidgets.QPushButton("Run depth + reconstruct")
        self.btn_scale = QtWidgets.QPushButton("Set scale (2D clicks)")
        self.btn_scale.setEnabled(False)

        self.model_cb = QtWidgets.QComboBox()
        self.model_cb.addItems([
            "depth-anything/Depth-Anything-V2-Small-hf",
            "depth-anything/Depth-Anything-V2-Base-hf",
            "depth-anything/Depth-Anything-V2-Large-hf",
        ])

        self.fov_spin = QtWidgets.QDoubleSpinBox()
        self.fov_spin.setRange(10.0, 140.0)
        self.fov_spin.setValue(60.0)
        self.fov_spin.setSuffix("°")

        self.stride_spin = QtWidgets.QSpinBox()
        self.stride_spin.setRange(1, 12)
        self.stride_spin.setValue(3)

        self.scale_mm_spin = QtWidgets.QDoubleSpinBox()
        self.scale_mm_spin.setRange(0.1, 500.0)
        self.scale_mm_spin.setValue(10.0)
        self.scale_mm_spin.setSuffix(" mm")

        self.status = QtWidgets.QLabel("Load an image, run reconstruction, click 3D points, then press OK to commit.")
        self.status.setWordWrap(True)

        self.picker = PointCloudPicker3D()
        self.picker.picked.connect(self._on_picked_xyz)

        self.points_panel = PickedPointsPanel()
        self.points_panel.changed.connect(self._update_status)

        top = QtWidgets.QGridLayout()
        top.addWidget(self.btn_load, 0, 0)
        top.addWidget(QtWidgets.QLabel("Model:"), 0, 1)
        top.addWidget(self.model_cb, 0, 2)
        top.addWidget(QtWidgets.QLabel("FOV:"), 0, 3)
        top.addWidget(self.fov_spin, 0, 4)

        top.addWidget(self.btn_run, 1, 0)
        top.addWidget(QtWidgets.QLabel("Stride:"), 1, 1)
        top.addWidget(self.stride_spin, 1, 2)
        top.addWidget(QtWidgets.QLabel("Scale:"), 1, 3)
        top.addWidget(self.scale_mm_spin, 1, 4)
        top.addWidget(self.btn_scale, 1, 5)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(self.picker)
        split.addWidget(self.points_panel)
        split.setSizes([950, 450])

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(split)
        layout.addWidget(self.status)
        layout.addWidget(buttons)

        self.btn_load.clicked.connect(self._load_image)
        self.btn_run.clicked.connect(self._run_depth)
        self.btn_scale.clicked.connect(self._set_scale)

        self._estimator: Optional[DepthEstimator] = None
        self._update_status()

    def buffered_points(self) -> Dict[str, np.ndarray]:
        """Return buffered points without committing."""
        return self.points_panel.points()

    def _update_status(self):
        n = len(self.points_panel.points())
        extra = ""
        if self.image_path:
            extra = f" | image: {os.path.basename(self.image_path)}"
        self.status.setText(f"Buffered points: {n}{extra}. Pick points in 3D; edit/remove in table; OK commits.")

    def _load_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select surgical image", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff);;All files (*)"
        )
        if not path:
            return

        try:
            from PIL import Image
        except Exception:
            QtWidgets.QMessageBox.critical(self, "Missing dependency", "Install pillow: pip install pillow")
            return

        self.image_path = path
        img = Image.open(path).convert("RGB")
        self.img_rgb = np.array(img)

        h, w = self.img_rgb.shape[:2]
        self.K = CameraIntrinsics(w, h, float(self.fov_spin.value()))
        self.depth_m = None
        self.xyz = None
        self.btn_scale.setEnabled(False)

        self.points_panel.clear()
        self.status.setText(f"Loaded: {os.path.basename(path)} ({w}×{h}). Now run depth + reconstruct.")

    def _run_depth(self):
        if self.img_rgb is None:
            QtWidgets.QMessageBox.warning(self, "Missing image", "Load an image first.")
            return

        h, w = self.img_rgb.shape[:2]
        self.K = CameraIntrinsics(w, h, float(self.fov_spin.value()))

        self.status.setText("Running depth estimation… (first time may download model)")
        QtWidgets.QApplication.processEvents()

        model_id = self.model_cb.currentText()
        try:
            self._estimator = DepthEstimator(model_id)
            depth_raw = self._estimator.predict_depth_raw(self.img_rgb)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Depth estimation error", str(e))
            return

        depth_rel = robust_norm_depth(depth_raw)
        self.depth_m = 0.2 + 0.8 * depth_rel  # baseline scaling; refined by user scale

        xyz, _uv = backproject(self.depth_m, self.K, stride=int(self.stride_spin.value()))
        self.xyz = xyz
        self.picker.set_xyz(xyz)

        self.btn_scale.setEnabled(True)
        self._update_status()

    def _set_scale(self):
        if self.img_rgb is None or self.depth_m is None or self.K is None:
            return

        if plt is None:
            QtWidgets.QMessageBox.critical(self, "Missing dependency", "Matplotlib is required for scaling.")
            return

        plt.figure(figsize=(10, 6))
        plt.imshow(self.img_rgb)
        plt.title("Click TWO points with known real distance (close window after selecting).")
        plt.axis("off")
        pts = plt.ginput(2, timeout=0)
        plt.close()

        if len(pts) != 2:
            return

        (u1, v1), (u2, v2) = pts
        try:
            p1 = self._pixel_to_xyz(u1, v1)
            p2 = self._pixel_to_xyz(u2, v2)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Scale error", str(e))
            return

        dist = float(np.linalg.norm(p1 - p2))
        if dist <= 1e-9:
            QtWidgets.QMessageBox.warning(self, "Scale error", "Selected points yield zero/invalid distance.")
            return

        desired_m = float(self.scale_mm_spin.value()) / 1000.0
        scale = desired_m / dist

        self.depth_m = self.depth_m * scale
        xyz, _uv = backproject(self.depth_m, self.K, stride=int(self.stride_spin.value()))
        self.xyz = xyz
        self.picker.set_xyz(xyz)

        self.status.setText(f"Scale applied: ×{scale:.6f}. Continue picking points; then press OK to commit.")

    def _pixel_to_xyz(self, u: float, v: float) -> np.ndarray:
        assert self.depth_m is not None and self.K is not None
        ui = int(np.clip(round(u), 0, self.K.width - 1))
        vi = int(np.clip(round(v), 0, self.K.height - 1))

        z = float(self.depth_m[vi, ui])
        if not np.isfinite(z) or z <= 0:
            raise ValueError("Invalid depth at selected pixel.")

        x = (ui - self.K.cx) * z / self.K.fx
        y = (vi - self.K.cy) * z / self.K.fy
        return np.array([x, y, z], dtype=float)

    def _on_picked_xyz(self, xyz):
        label, ok = QtWidgets.QInputDialog.getText(self, "Label point", "Label for this point (unique):")
        if not ok:
            return

        label = (label or "").strip()
        if not label:
            return

        if label in self.points_panel.points():
            resp = QtWidgets.QMessageBox.question(
                self,
                "Overwrite buffer point?",
                f"Point '{label}' already exists in the buffer. Overwrite it?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if resp != QtWidgets.QMessageBox.Yes:
                return

        self.points_panel.upsert(label, np.asarray(xyz, dtype=float).reshape(3,))
        self._update_status()

    def _on_ok(self):
        pts = self.points_panel.points()
        if not pts:
            resp = QtWidgets.QMessageBox.question(
                self,
                "No points selected",
                "You have not selected any points. Continue anyway?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if resp != QtWidgets.QMessageBox.Yes:
                return

        for label, xyz in pts.items():
            if label in self.ds.points:
                resp = QtWidgets.QMessageBox.question(
                    self,
                    "Overwrite dataset point?",
                    f"Point '{label}' exists in dataset. Overwrite it?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                )
                if resp != QtWidgets.QMessageBox.Yes:
                    continue
            self.ds.points[label] = np.asarray(xyz, dtype=float).reshape(3,)

        if isinstance(getattr(self.ds, "meta", None), dict):
            self.ds.meta["source"] = "single_image_depth"

        self.accept()
