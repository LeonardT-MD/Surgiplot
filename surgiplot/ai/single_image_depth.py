"""
surgiplot.ai.single_image_depth

Single-image depth -> 3D reconstruction + buffered point picking dialog (OK commits, Cancel discards),
separated from app.py.

Dependencies (feature-only):
  conda/pip install -U numpy pillow matplotlib transformers torch

iPhone formats (HEIC/HEIF):
  pip install -U pillow-heif

Notes:
- Uses HuggingFace transformers "depth-estimation" pipeline (Depth Anything V2).
- Depth is relative; scale can be set interactively from two 2D clicks (known distance in mm).
- Click 3D point cloud -> prompt label -> point is stored in a PENDING buffer.
- You can Rename/Delete/Undo/Clear pending points.
- ONLY when you click OK: pending points are committed into ds.points[label] = xyz.
- Cancel rejects without modifying ds.

Public API used by GUI:
- class SingleImage3DDialog(QtWidgets.QDialog)
"""

from __future__ import annotations

import math
import os
import time
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from PySide6 import QtWidgets, QtCore

LOG = logging.getLogger("surgiplot")

# Matplotlib is required for this feature. Keep it local to avoid breaking core GUI if missing.
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    HAS_MPL = True
except Exception:
    HAS_MPL = False
    FigureCanvas = None  # type: ignore
    Figure = None  # type: ignore


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
# iPhone image support helpers (HEIC/HEIF)
# -------------------------
def _try_register_heif_opener() -> bool:
    """Try to register an HEIC/HEIF opener for Pillow."""
    try:
        import pillow_heif  # type: ignore
        pillow_heif.register_heif_opener()
        return True
    except Exception:
        return False


def _open_image_rgb(path: str) -> np.ndarray:
    """Open an image and return RGB uint8 ndarray."""
    try:
        from PIL import Image
    except Exception as e:
        raise RuntimeError("Missing dependency: pillow. Install with: pip install pillow") from e

    _try_register_heif_opener()

    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        ext = os.path.splitext(path)[1].lower()
        msg = (
            "Cannot open this image with Pillow on your system.\n\n"
            f"File: {os.path.basename(path)}\n"
            f"Extension: {ext or '(none)'}\n"
            f"Error: {e}\n\n"
            "If this is an iPhone HEIC/HEIF image, install HEIF support:\n"
            "  pip install -U pillow-heif\n\n"
            "Alternatively, convert the photo to JPG/PNG and try again."
        )
        raise RuntimeError(msg) from e

    arr = np.array(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise RuntimeError("Loaded image is not RGB; unexpected format.")
    return arr


def _image_dialog_filter() -> str:
    """Qt file dialog filter."""
    return (
        "Images (*.png *.PNG *.jpg *.JPG *.jpeg *.JPEG *.tif *.TIF *.tiff *.TIFF *.bmp *.BMP *.webp *.WEBP "
        "*.heic *.HEIC *.heif *.HEIF);;All files (*)"
    )


# -------------------------
# Depth estimation (AI)
# -------------------------
class DepthEstimator:
    """
    Thin wrapper around transformers depth-estimation pipeline.
    Lazily loads the model.

    Apple Silicon:
      - If torch.backends.mps.is_available(): uses MPS by default.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._pipe = None
        self._torch = None
        self._device = None
        self._dtype = None

    @staticmethod
    def _torch_import_or_raise():
        """
        Import torch and raise an actionable error if torch fails due to numpy ABI mismatch
        (common with NumPy 2.x + torch built against NumPy 1.x).
        """
        try:
            import torch  # type: ignore
            return torch
        except Exception as e:
            msg = str(e)
            # Common symptom: _ARRAY_API not found / numpy ABI mismatch
            if "_ARRAY_API" in msg or "Failed to initialize NumPy" in msg or "compiled using NumPy 1.x" in msg:
                raise RuntimeError(
                    "PyTorch failed to import due to a NumPy ABI mismatch.\n\n"
                    "Your environment likely has NumPy 2.x but torch (or a torch dependency) "
                    "was compiled against NumPy 1.x.\n\n"
                    "Fix (recommended):\n"
                    "  pip install -U 'numpy<2'\n\n"
                    "Then reinstall torch (in the same env):\n"
                    "  pip install -U torch torchvision torchaudio\n\n"
                    f"Original error:\n{e}"
                ) from e
            raise RuntimeError(
                "PyTorch failed to import. Install torch for your platform.\n\n"
                "Try:\n"
                "  pip install -U torch torchvision torchaudio\n\n"
                f"Original error:\n{e}"
            ) from e

    @staticmethod
    def _select_device(torch_mod):
        """
        Device preference:
          1) MPS (Apple Silicon)
          2) CUDA
          3) CPU
        """
        # MPS
        try:
            if hasattr(torch_mod, "backends") and hasattr(torch_mod.backends, "mps"):
                if torch_mod.backends.mps.is_available():
                    return "mps"
        except Exception:
            pass

        # CUDA
        try:
            if torch_mod.cuda.is_available():
                return "cuda"
        except Exception:
            pass

        return "cpu"

    @staticmethod
    def _select_dtype(torch_mod, device: str):
        """
        Safe dtype choices:
          - On CPU: float32
          - On MPS/CUDA: float16 often faster, but some models might be safer in float32.
        We'll default to float16 on GPU backends, but allow fallback.
        """
        if device in ("cuda", "mps"):
            return getattr(torch_mod, "float16", None) or torch_mod.float32
        return torch_mod.float32

    def _ensure(self):
        if self._pipe is not None:
            return

        try:
            from transformers import pipeline
        except Exception as e:
            raise RuntimeError("Missing dependency: transformers. Install with: pip install transformers") from e

        torch_mod = self._torch_import_or_raise()
        self._torch = torch_mod

        device = self._select_device(torch_mod)
        dtype = self._select_dtype(torch_mod, device)

        self._device = device
        self._dtype = dtype

        # HF pipeline device:
        # - For CUDA: integer GPU index (0)
        # - For CPU: -1
        # - For MPS: transformers supports "mps" by passing device="mps" in recent versions,
        #   but compatibility varies. We'll handle both safely.
        model_kwargs = {"torch_dtype": dtype}

        # Logging
        try:
            mps_avail = bool(getattr(torch_mod.backends, "mps", None) and torch_mod.backends.mps.is_available())
        except Exception:
            mps_avail = False

        try:
            cuda_avail = bool(torch_mod.cuda.is_available())
        except Exception:
            cuda_avail = False

        LOG.info("Initializing depth pipeline. model_id=%s", self.model_id)
        LOG.info("Torch: version=%s | numpy=%s", getattr(torch_mod, "__version__", "?"), np.__version__)
        LOG.info("Backends: mps=%s cuda=%s -> device=%s", mps_avail, cuda_avail, device)
        LOG.info("Using torch_dtype=%s", "float16" if dtype == torch_mod.float16 else "float32")

        # Try device options in a robust order
        pipe = None
        last_err = None

        # 1) MPS explicit
        if device == "mps":
            for dev_arg in ("mps", 0, -1):
                try:
                    # Some transformers versions accept device="mps"; others only int.
                    pipe = pipeline(task="depth-estimation", model=self.model_id, device=dev_arg, model_kwargs=model_kwargs)
                    break
                except Exception as e:
                    last_err = e
                    pipe = None

        # 2) CUDA (device=0)
        elif device == "cuda":
            try:
                pipe = pipeline(task="depth-estimation", model=self.model_id, device=0, model_kwargs=model_kwargs)
            except Exception as e:
                last_err = e
                pipe = None

        # 3) CPU fallback
        if pipe is None:
            try:
                pipe = pipeline(task="depth-estimation", model=self.model_id, device=-1, model_kwargs={"torch_dtype": torch_mod.float32})
                self._device = "cpu"
                self._dtype = torch_mod.float32
                LOG.info("Fell back to CPU pipeline.")
            except Exception as e:
                last_err = e
                pipe = None

        if pipe is None:
            raise RuntimeError(
                "Failed to initialize transformers depth-estimation pipeline.\n\n"
                "This is usually due to an incompatible torch/transformers/numpy combination.\n"
                "Recommended fix:\n"
                "  pip install -U 'numpy<2'\n"
                "  pip install -U torch torchvision torchaudio transformers\n\n"
                f"Last error:\n{last_err}"
            )

        self._pipe = pipe

    def predict_depth_raw(self, img_rgb: np.ndarray) -> np.ndarray:
        """Returns a 2D float32 array derived from the pipeline output depth image."""
        self._ensure()

        try:
            from PIL import Image
        except Exception as e:
            raise RuntimeError("Missing dependency: pillow. Install with: pip install pillow") from e

        img = Image.fromarray(np.asarray(img_rgb, dtype=np.uint8))

        t0 = time.time()
        out = self._pipe(img)
        dt = time.time() - t0

        depth_img = out["depth"]
        depth_raw = np.array(depth_img).astype(np.float32)

        LOG.info("Depth inference done in %.2fs. depth_raw shape=%s dtype=%s", dt, depth_raw.shape, depth_raw.dtype)
        return depth_raw


# -------------------------
# 3D picker widget
# -------------------------
class PointCloudPicker3D(QtWidgets.QWidget):
    """Embedded matplotlib 3D scatter that emits picked xyz."""
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

    def set_cloud(self, xyz: np.ndarray, colors: Optional[np.ndarray] = None):
        """
        xyz: (N,3)
        colors: optional (N,3) float in [0,1] or uint8 in [0,255]
        """
        if not HAS_MPL:
            return

        xyz = np.asarray(xyz, dtype=np.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must be (N,3)")

        if colors is not None:
            colors = np.asarray(colors)
            if colors.ndim != 2 or colors.shape[1] != 3 or colors.shape[0] != xyz.shape[0]:
                raise ValueError("colors must be (N,3) matching xyz")
            if colors.dtype.kind in ("u", "i"):
                colors = colors.astype(np.float32) / 255.0
            else:
                colors = colors.astype(np.float32)
                colors = np.clip(colors, 0.0, 1.0)

        self._xyz = xyz
        self.ax.clear()
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")

        self._sc = self.ax.scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2],
            s=2, alpha=0.95,
            c=colors if colors is not None else None,
            picker=6
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
# Embedded 2D scale picker dialog (NO external matplotlib window)
# -------------------------
class TwoClickScaleDialog(QtWidgets.QDialog):
    """Embedded 2D image viewer: collect exactly 2 clicks in image pixel coords."""
    def __init__(self, img_rgb: np.ndarray, parent=None):
        super().__init__(parent)

        if not HAS_MPL:
            raise RuntimeError("Matplotlib missing")

        self.setWindowTitle("Set scale: click TWO points (then OK)")
        self.setModal(True)
        self.resize(900, 600)

        self.img_rgb = np.asarray(img_rgb, dtype=np.uint8)
        self.pts: list[tuple[float, float]] = []

        lay = QtWidgets.QVBoxLayout(self)

        self.fig = Figure(figsize=(8, 5))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        lay.addWidget(self.canvas, 1)

        btns = QtWidgets.QHBoxLayout()
        self.lbl = QtWidgets.QLabel("Clicks: 0 / 2")
        self.btn_clear = QtWidgets.QPushButton("Clear")
        self.btn_ok = QtWidgets.QPushButton("OK")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_ok.setEnabled(False)

        btns.addWidget(self.lbl, 1)
        btns.addWidget(self.btn_clear)
        btns.addWidget(self.btn_cancel)
        btns.addWidget(self.btn_ok)
        lay.addLayout(btns)

        self.btn_clear.clicked.connect(self._clear)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok.clicked.connect(self.accept)

        self._redraw_base()
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.draw_idle()

    def _redraw_base(self):
        self.ax.clear()
        self.ax.imshow(self.img_rgb)
        self.ax.set_title("Click two points with known distance")
        self.ax.axis("off")

    def _clear(self):
        self.pts.clear()
        self._redraw_base()
        self.lbl.setText("Clicks: 0 / 2")
        self.btn_ok.setEnabled(False)
        self.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        if len(self.pts) >= 2:
            return

        u, v = float(event.xdata), float(event.ydata)
        self.pts.append((u, v))
        self.ax.plot([u], [v], marker="o")
        self.lbl.setText(f"Clicks: {len(self.pts)} / 2")

        if len(self.pts) == 2:
            (u1, v1), (u2, v2) = self.pts
            self.ax.plot([u1, u2], [v1, v2], linewidth=2)
            self.btn_ok.setEnabled(True)

        self.canvas.draw_idle()


# -------------------------
# Dialog: single-image -> 3D -> buffered points -> OK commits
# -------------------------
class SingleImage3DDialog(QtWidgets.QDialog):
    """
    Workflow:
      - Load image
      - Run depth estimation -> point cloud
      - Optional: set scale using 2D clicks + known distance (mm)
      - Click points in 3D -> prompt label -> add to PENDING buffer (editable)
      - OK commits pending points into ds.points; Cancel discards
    """

    def __init__(self, ds, on_dataset_changed_callback=None, parent=None):
        super().__init__(parent)

        if not HAS_MPL:
            QtWidgets.QMessageBox.critical(
                self,
                "Missing dependency",
                "Matplotlib is required for Single-image 3D points.\n\nInstall: pip install matplotlib",
            )
            raise RuntimeError("Matplotlib missing")

        self.ds = ds
        self.on_dataset_changed_callback = on_dataset_changed_callback

        self.setWindowTitle("Single Image → Depth → 3D → Collect points (OK commits)")
        self.setModal(True)
        self.resize(1350, 780)

        self.image_path: Optional[str] = None
        self.img_rgb: Optional[np.ndarray] = None
        self.depth_m: Optional[np.ndarray] = None
        self.K: Optional[CameraIntrinsics] = None
        self.xyz: Optional[np.ndarray] = None

        self._estimator: Optional[DepthEstimator] = None

        # Buffered points (pending until OK)
        self._pending_points: Dict[str, np.ndarray] = {}
        self._pending_order: list[str] = []

        # Controls
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

        self.status = QtWidgets.QLabel(
            "Load an image, run reconstruction, then click 3D points to add them to the pending list. OK commits."
        )
        self.status.setWordWrap(True)

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

        # Main area
        self.picker = PointCloudPicker3D()
        self.picker.picked.connect(self._on_picked_xyz)

        self.pending_table = QtWidgets.QTableWidget()
        self.pending_table.setColumnCount(4)
        self.pending_table.setHorizontalHeaderLabels(["label", "x (m)", "y (m)", "z (m)"])
        self.pending_table.horizontalHeader().setStretchLastSection(True)
        self.pending_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.pending_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

        self.btn_rename = QtWidgets.QPushButton("Rename")
        self.btn_delete = QtWidgets.QPushButton("Delete")
        self.btn_undo = QtWidgets.QPushButton("Undo last")
        self.btn_clear = QtWidgets.QPushButton("Clear all")

        self.btn_rename.setEnabled(False)
        self.btn_delete.setEnabled(False)
        self.btn_undo.setEnabled(False)
        self.btn_clear.setEnabled(False)

        self.pending_table.itemSelectionChanged.connect(self._on_pending_selection_changed)
        self.btn_rename.clicked.connect(self._rename_selected)
        self.btn_delete.clicked.connect(self._delete_selected)
        self.btn_undo.clicked.connect(self._undo_last)
        self.btn_clear.clicked.connect(self._clear_all)

        right_btns = QtWidgets.QHBoxLayout()
        right_btns.addWidget(self.btn_rename)
        right_btns.addWidget(self.btn_delete)
        right_btns.addWidget(self.btn_undo)
        right_btns.addWidget(self.btn_clear)

        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("<b>Pending points (OK commits)</b>"))
        right.addWidget(self.pending_table)
        right.addLayout(right_btns)

        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(right)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self.picker)
        splitter.addWidget(right_widget)
        splitter.setSizes([900, 420])

        # OK/Cancel
        self.btn_ok = QtWidgets.QPushButton("OK (commit to dataset)")
        self.btn_cancel = QtWidgets.QPushButton("Cancel (discard)")
        self.btn_ok.setEnabled(False)

        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(self.status, 1)
        bottom.addWidget(self.btn_cancel)
        bottom.addWidget(self.btn_ok)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(splitter, 1)
        layout.addLayout(bottom)

        # Wiring
        self.btn_load.clicked.connect(self._load_image)
        self.btn_run.clicked.connect(self._run_depth)
        self.btn_scale.clicked.connect(self._set_scale)
        self.btn_ok.clicked.connect(self._commit_and_accept)
        self.btn_cancel.clicked.connect(self.reject)

        self._refresh_pending_table()

    # -------------------------
    # Image / depth / reconstruction
    # -------------------------
    def _load_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select surgical image", "", _image_dialog_filter()
        )
        if not path:
            return

        try:
            self.img_rgb = _open_image_rgb(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Cannot open image", str(e))
            return

        self.image_path = path
        h, w = self.img_rgb.shape[:2]
        self.K = CameraIntrinsics(w, h, float(self.fov_spin.value()))
        self.depth_m = None
        self.xyz = None
        self.btn_scale.setEnabled(False)

        LOG.info("Loaded image: %s (%s dtype=%s)", os.path.basename(path), self.img_rgb.shape, self.img_rgb.dtype)
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
            LOG.exception("Depth estimation failed.")
            QtWidgets.QMessageBox.critical(self, "Depth estimation error", str(e))
            return

        depth_rel = robust_norm_depth(depth_raw)
        depth_m = 0.2 + 0.8 * depth_rel
        self.depth_m = depth_m

        xyz, uv = backproject(depth_m, self.K, stride=int(self.stride_spin.value()))
        self.xyz = xyz

        # Color each point from the source image (so anatomy is interpretable)
        u = np.clip(uv[:, 0].round().astype(np.int32), 0, w - 1)
        v = np.clip(uv[:, 1].round().astype(np.int32), 0, h - 1)
        colors = self.img_rgb[v, u, :].astype(np.float32) / 255.0  # (N,3)

        self.picker.set_cloud(xyz, colors=colors)
        self.btn_scale.setEnabled(True)
        self.status.setText(
            "Reconstruction ready. Optional: set scale. Then click points in 3D; each click asks for a label and adds it to the pending list."
        )

    def _set_scale(self):
        if self.img_rgb is None or self.depth_m is None or self.K is None:
            return

        dlg = TwoClickScaleDialog(self.img_rgb, parent=self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        if len(dlg.pts) != 2:
            return

        (u1, v1), (u2, v2) = dlg.pts
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

        h, w = self.img_rgb.shape[:2]
        xyz, uv = backproject(self.depth_m, self.K, stride=int(self.stride_spin.value()))
        self.xyz = xyz

        u = np.clip(uv[:, 0].round().astype(np.int32), 0, w - 1)
        v = np.clip(uv[:, 1].round().astype(np.int32), 0, h - 1)
        colors = self.img_rgb[v, u, :].astype(np.float32) / 255.0

        self.picker.set_cloud(xyz, colors=colors)
        self.status.setText(f"Scale applied: ×{scale:.6f}. Continue clicking 3D points to add them to the pending list.")

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

    # -------------------------
    # Pending points editing
    # -------------------------
    def _on_picked_xyz(self, xyz):
        if self.xyz is None:
            return

        label, ok = QtWidgets.QInputDialog.getText(
            self, "Label point", "Label for this point (pending until OK):"
        )
        if not ok:
            return

        label = (label or "").strip()
        if not label:
            return

        if label in self._pending_points:
            resp = QtWidgets.QMessageBox.question(
                self,
                "Overwrite pending?",
                f"Pending point '{label}' already exists. Overwrite it?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if resp != QtWidgets.QMessageBox.Yes:
                return

        arr = np.asarray(xyz, dtype=float).reshape(3,)
        self._pending_points[label] = arr

        if label in self._pending_order:
            self._pending_order.remove(label)
        self._pending_order.append(label)

        self.status.setText(f"Pending: {label} -> ({arr[0]:.4f}, {arr[1]:.4f}, {arr[2]:.4f})")
        self._refresh_pending_table()

    def _refresh_pending_table(self):
        labels = sorted(self._pending_points.keys())
        self.pending_table.setRowCount(len(labels))

        for r, lb in enumerate(labels):
            p = self._pending_points[lb]
            it0 = QtWidgets.QTableWidgetItem(lb)
            it1 = QtWidgets.QTableWidgetItem(f"{float(p[0]):.6f}")
            it2 = QtWidgets.QTableWidgetItem(f"{float(p[1]):.6f}")
            it3 = QtWidgets.QTableWidgetItem(f"{float(p[2]):.6f}")

            for it in (it0, it1, it2, it3):
                it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)

            self.pending_table.setItem(r, 0, it0)
            self.pending_table.setItem(r, 1, it1)
            self.pending_table.setItem(r, 2, it2)
            self.pending_table.setItem(r, 3, it3)

        has_any = len(self._pending_points) > 0
        self.btn_clear.setEnabled(has_any)
        self.btn_undo.setEnabled(len(self._pending_order) > 0)
        self.btn_ok.setEnabled(has_any)

        self._on_pending_selection_changed()

    def _selected_label(self) -> Optional[str]:
        sel = self.pending_table.selectionModel().selectedRows()
        if not sel:
            return None
        row = int(sel[0].row())
        it = self.pending_table.item(row, 0)
        if it is None:
            return None
        return it.text().strip() or None

    def _on_pending_selection_changed(self):
        lb = self._selected_label()
        enabled = lb is not None and lb in self._pending_points
        self.btn_rename.setEnabled(enabled)
        self.btn_delete.setEnabled(enabled)

    def _rename_selected(self):
        old = self._selected_label()
        if not old or old not in self._pending_points:
            return

        new, ok = QtWidgets.QInputDialog.getText(self, "Rename point", f"New label for '{old}':")
        if not ok:
            return
        new = (new or "").strip()
        if not new or new == old:
            return

        if new in self._pending_points:
            QtWidgets.QMessageBox.warning(self, "Rename error", f"Label '{new}' already exists in pending points.")
            return

        self._pending_points[new] = self._pending_points.pop(old)

        if old in self._pending_order:
            idx = self._pending_order.index(old)
            self._pending_order[idx] = new

        self.status.setText(f"Renamed pending point: {old} → {new}")
        self._refresh_pending_table()

    def _delete_selected(self):
        lb = self._selected_label()
        if not lb or lb not in self._pending_points:
            return
        resp = QtWidgets.QMessageBox.question(
            self, "Delete pending point", f"Delete '{lb}' from pending points?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if resp != QtWidgets.QMessageBox.Yes:
            return

        self._pending_points.pop(lb, None)
        if lb in self._pending_order:
            self._pending_order = [x for x in self._pending_order if x != lb]

        self.status.setText(f"Deleted pending point: {lb}")
        self._refresh_pending_table()

    def _undo_last(self):
        if not self._pending_order:
            return
        last = self._pending_order.pop()
        if last in self._pending_points:
            self._pending_points.pop(last, None)
            self.status.setText(f"Undo last: removed '{last}'")
            self._refresh_pending_table()

    def _clear_all(self):
        resp = QtWidgets.QMessageBox.question(
            self, "Clear all", "Clear ALL pending points?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if resp != QtWidgets.QMessageBox.Yes:
            return
        self._pending_points.clear()
        self._pending_order.clear()
        self.status.setText("Cleared all pending points.")
        self._refresh_pending_table()

    # -------------------------
    # Commit / accept
    # -------------------------
    def _commit_and_accept(self):
        if not self._pending_points:
            QtWidgets.QMessageBox.information(self, "Nothing to commit", "No pending points to commit.")
            return

        overwriting = [lb for lb in self._pending_points.keys() if lb in getattr(self.ds, "points", {})]
        if overwriting:
            resp = QtWidgets.QMessageBox.question(
                self,
                "Overwrite existing dataset points?",
                "The following labels already exist in the dataset and will be overwritten:\n\n"
                + "\n".join(overwriting)
                + "\n\nProceed?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if resp != QtWidgets.QMessageBox.Yes:
                return

        for lb, p in self._pending_points.items():
            self.ds.points[lb] = np.asarray(p, dtype=float).reshape(3,)

        if isinstance(getattr(self.ds, "meta", None), dict):
            self.ds.meta["source"] = "single_image_depth"

        if callable(self.on_dataset_changed_callback):
            try:
                self.on_dataset_changed_callback()
            except Exception:
                pass

        self.accept()

