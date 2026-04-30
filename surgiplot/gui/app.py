"""
surgiplot.gui.app

Surgiplot GUI — updated import-aware app

Contains:
- Qt (PySide6) GUI entrypoint with `main()`
- Navigation import (Stryker / Medtronic / auto-detect)
- Formatted database import
- Generic point-file import into existing dataset
- Optional review / alias editor
- Metric computation hooks
- Optional imported 3D scene workspace
- Optional matplotlib 3D plotting
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, Optional, Any, List
import re

import numpy as np
from PySide6 import QtWidgets, QtCore, QtGui

from surgiplot.core.dataset import Dataset
from surgiplot.core.io.loaders import load_dataset, load_navigation_file
from surgiplot.metrics import VOM_VOA, AOA_SF, AOE, DISTANCE_3D, AREA_3D, VOLUME_3D
from surgiplot.gui.label_editor import run_label_editor
try:
    from surgiplot.ai.scene_picker import HAS_PYVISTA, QtInteractor, pv
except Exception:
    HAS_PYVISTA = False
    QtInteractor = None  # type: ignore
    pv = None  # type: ignore

try:
    from surgiplot.ai.scene_reconstruction import AISceneReconstructionDialog
    HAS_AI_SCENE = True
except Exception:
    AISceneReconstructionDialog = None  # type: ignore
    HAS_AI_SCENE = False

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MPL = True
except Exception:
    FigureCanvas = None  # type: ignore
    Figure = None  # type: ignore
    HAS_MPL = False


# =============================================================================
# Helpers
# =============================================================================
def _exc_text() -> str:
    return traceback.format_exc()


def _ensure_points_dict(ds) -> Dict[str, np.ndarray]:
    if not hasattr(ds, "points") or ds.points is None:
        ds.points = {}
    if not isinstance(ds.points, dict):
        raise TypeError("Dataset ds.points must be a dict[label -> xyz]")
    return ds.points


def _ensure_aliases_dict(ds) -> Dict[str, str]:
    if not hasattr(ds, "aliases") or ds.aliases is None:
        ds.aliases = {}
    if not isinstance(ds.aliases, dict):
        raise TypeError("Dataset ds.aliases must be a dict[alias -> canonical_name]")
    return ds.aliases


def _as_xyz(arr: Any) -> np.ndarray:
    a = np.asarray(arr, dtype=float).reshape(-1)
    if a.size != 3:
        raise ValueError("Point must be 3D (x,y,z).")
    return a.reshape(3,)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _dataset_to_rows(ds) -> List[dict]:
    pts = _ensure_points_dict(ds)
    aliases = getattr(ds, "aliases", {}) or {}

    reverse_aliases: Dict[str, List[str]] = {}
    for alias, canonical in aliases.items():
        reverse_aliases.setdefault(str(canonical), []).append(str(alias))

    rows = []
    for name in sorted(pts.keys()):
        p = _as_xyz(pts[name])
        labels = ", ".join(sorted(reverse_aliases.get(name, [])))
        rows.append({
            "name": name,
            "x": float(p[0]),
            "y": float(p[1]),
            "z": float(p[2]),
            "labels": labels,
        })
    return rows


def _apply_rows_to_dataset(ds, rows: List[dict], source: str = "") -> None:
    pts = _ensure_points_dict(ds)
    aliases = _ensure_aliases_dict(ds)

    pts.clear()
    aliases.clear()

    for row in rows:
        name = str(row["name"]).strip()
        xyz = np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=float)
        pts[name] = xyz

        labels = str(row.get("labels", "") or "").strip()
        if labels:
            for alias in [x.strip() for x in labels.split(",") if x.strip()]:
                aliases[alias] = name

    if isinstance(getattr(ds, "meta", None), dict) and source:
        ds.meta["source"] = source


def _merge_rows_into_dataset(ds, rows: List[dict], source: str = "") -> None:
    pts = _ensure_points_dict(ds)
    aliases = _ensure_aliases_dict(ds)

    for row in rows:
        name = str(row["name"]).strip()
        xyz = np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=float)
        pts[name] = xyz

        labels = str(row.get("labels", "") or "").strip()
        if labels:
            for alias in [x.strip() for x in labels.split(",") if x.strip()]:
                aliases[alias] = name

    if isinstance(getattr(ds, "meta", None), dict) and source:
        ds.meta["source"] = source


def _dataset_scene_meta(ds) -> Dict[str, Any]:
    if ds is None:
        return {}
    meta = getattr(ds, "meta", None)
    if not isinstance(meta, dict):
        return {}
    ai_scene = meta.get("ai_scene")
    if not isinstance(ai_scene, dict):
        return {}
    return {"ai_scene": _json_safe(ai_scene)}


# =============================================================================
# Plot panel
# =============================================================================
class PlotPanel(QtWidgets.QWidget):
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
            self._scene_payload = None
            self._interactive_points: List[dict] = []
            self._layer_artists: Dict[str, List[Any]] = {}
            self._title = QtWidgets.QLabel("")
            self._status = QtWidgets.QLabel("")
            lay.addWidget(self._title)
            lay.addWidget(self._status)
            return

        self._scene_payload: Optional[dict] = None
        self._interactive_points: List[dict] = []
        self._layer_artists: Dict[str, List[Any]] = {}
        self._selection_artist = None
        self._selected_point = None
        self._projection = "3d"
        self._layer_list_syncing = False
        self._model_opacity = 0.45
        self._model_show_edges = False
        self._model_point_size = 4.0
        self._model_use_source_colors = True
        self._metric_opacity = 1.0
        self._metric_emphasis = 1.0
        self._viewer_mode = "pyvista" if HAS_PYVISTA else "matplotlib"
        self._pv_selection_actor = None
        self._pv_pick_actor = None

        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(8)
        self.btn_reset_view = QtWidgets.QToolButton()
        self.btn_reset_view.setText("Reset View")
        self.btn_zoom_in = QtWidgets.QToolButton()
        self.btn_zoom_in.setText("Zoom In")
        self.btn_zoom_out = QtWidgets.QToolButton()
        self.btn_zoom_out.setText("Zoom Out")
        self.btn_export_image = QtWidgets.QToolButton()
        self.btn_export_image.setText("Export PNG")
        self.btn_export_scene = QtWidgets.QToolButton()
        self.btn_export_scene.setText("Save 3D Scene")
        self.btn_open_scene = QtWidgets.QToolButton()
        self.btn_open_scene.setText("Open 3D Scene")
        for btn in (
            self.btn_reset_view,
            self.btn_zoom_in,
            self.btn_zoom_out,
            self.btn_open_scene,
            self.btn_export_scene,
            self.btn_export_image,
        ):
            btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
            btn.setAutoRaise(False)
        controls.addWidget(self.btn_reset_view)
        controls.addWidget(self.btn_zoom_in)
        controls.addWidget(self.btn_zoom_out)
        controls.addStretch(1)
        controls.addWidget(self.btn_open_scene)
        controls.addWidget(self.btn_export_scene)
        controls.addWidget(self.btn_export_image)
        lay.addLayout(controls)

        self.fig = Figure(figsize=(6, 5), constrained_layout=True)
        self.fig.patch.set_facecolor("#f5f7fb")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(520)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self._apply_axes_style_3d()
        self.viewer_stack = QtWidgets.QStackedWidget()
        self.viewer_stack.addWidget(self.canvas)
        self.pv_plotter = None
        self._pv_widget = None
        if HAS_PYVISTA and QtInteractor is not None:
            self.pv_plotter = QtInteractor(self)
            self._pv_widget = self.pv_plotter.interactor if hasattr(self.pv_plotter, "interactor") else self.pv_plotter
            self.viewer_stack.addWidget(self._pv_widget)
            self._configure_pyvista_view()

        self.layer_panel = QtWidgets.QFrame()
        self.layer_panel.setObjectName("SceneLayerPanel")
        layer_lay = QtWidgets.QVBoxLayout(self.layer_panel)
        layer_lay.setContentsMargins(12, 12, 12, 12)
        layer_lay.setSpacing(8)
        self.lbl_layers = QtWidgets.QLabel("Scene Components")
        self.lbl_layers.setObjectName("SceneLayerTitle")
        self.render_panel = QtWidgets.QFrame()
        render_lay = QtWidgets.QGridLayout(self.render_panel)
        render_lay.setContentsMargins(0, 0, 0, 0)
        render_lay.setHorizontalSpacing(8)
        render_lay.setVerticalSpacing(6)
        self.lbl_render = QtWidgets.QLabel("3D Model Rendering")
        self.lbl_metric_render = QtWidgets.QLabel("Metric Overlay Rendering")
        self.slider_model_opacity = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_model_opacity.setRange(5, 100)
        self.slider_model_opacity.setValue(int(round(self._model_opacity * 100)))
        self.slider_metric_opacity = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_metric_opacity.setRange(15, 100)
        self.slider_metric_opacity.setValue(int(round(self._metric_opacity * 100)))
        self.chk_model_edges = QtWidgets.QCheckBox("Show mesh edges")
        self.chk_model_edges.setChecked(self._model_show_edges)
        self.chk_model_colors = QtWidgets.QCheckBox("Show source colors / texture")
        self.chk_model_colors.setChecked(self._model_use_source_colors)
        self.spin_model_point_size = QtWidgets.QDoubleSpinBox()
        self.spin_model_point_size.setRange(1.0, 12.0)
        self.spin_model_point_size.setDecimals(1)
        self.spin_model_point_size.setSingleStep(0.5)
        self.spin_model_point_size.setValue(self._model_point_size)
        self.spin_metric_emphasis = QtWidgets.QDoubleSpinBox()
        self.spin_metric_emphasis.setRange(0.5, 3.0)
        self.spin_metric_emphasis.setDecimals(2)
        self.spin_metric_emphasis.setSingleStep(0.1)
        self.spin_metric_emphasis.setValue(self._metric_emphasis)
        render_lay.addWidget(self.lbl_render, 0, 0, 1, 2)
        render_lay.addWidget(QtWidgets.QLabel("Opacity"), 1, 0)
        render_lay.addWidget(self.slider_model_opacity, 1, 1)
        render_lay.addWidget(QtWidgets.QLabel("Point size"), 2, 0)
        render_lay.addWidget(self.spin_model_point_size, 2, 1)
        render_lay.addWidget(self.chk_model_colors, 3, 0, 1, 2)
        render_lay.addWidget(self.chk_model_edges, 4, 0, 1, 2)
        render_lay.addWidget(self.lbl_metric_render, 5, 0, 1, 2)
        render_lay.addWidget(QtWidgets.QLabel("Opacity"), 6, 0)
        render_lay.addWidget(self.slider_metric_opacity, 6, 1)
        render_lay.addWidget(QtWidgets.QLabel("Emphasis"), 7, 0)
        render_lay.addWidget(self.spin_metric_emphasis, 7, 1)
        layer_actions = QtWidgets.QHBoxLayout()
        self.btn_show_all_layers = QtWidgets.QToolButton()
        self.btn_show_all_layers.setText("Show All")
        self.btn_hide_all_layers = QtWidgets.QToolButton()
        self.btn_hide_all_layers.setText("Hide All")
        layer_actions.addWidget(self.btn_show_all_layers)
        layer_actions.addWidget(self.btn_hide_all_layers)
        layer_actions.addStretch(1)
        self.list_layers = QtWidgets.QListWidget()
        self.list_layers.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.list_layers.itemChanged.connect(self._on_layer_item_changed)
        self.lbl_points = QtWidgets.QLabel("Metric Points")
        self.list_points = QtWidgets.QListWidget()
        self.list_points.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list_points.itemSelectionChanged.connect(self._on_point_item_selected)
        layer_lay.addWidget(self.lbl_layers)
        layer_lay.addWidget(self.render_panel)
        layer_lay.addLayout(layer_actions)
        layer_lay.addWidget(self.list_layers, 1)
        layer_lay.addWidget(self.lbl_points)
        layer_lay.addWidget(self.list_points, 1)
        self.layer_panel.setMinimumWidth(210)
        self.layer_panel.setMaximumWidth(280)

        scene_row = QtWidgets.QHBoxLayout()
        scene_row.setSpacing(10)
        scene_row.addWidget(self.viewer_stack, 1)
        scene_row.addWidget(self.layer_panel)
        lay.addLayout(scene_row, 1)

        self._title = QtWidgets.QLabel("")
        self._title.setObjectName("PlotTitle")
        self._title.setWordWrap(True)
        self._status = QtWidgets.QLabel("Hover over a point to inspect its coordinates. Click to pin the selection.")
        self._status.setObjectName("PlotStatus")
        lay.addWidget(self._title)
        lay.addWidget(self._status)

        self.btn_reset_view.clicked.connect(self.reset_view)
        self.btn_zoom_in.clicked.connect(lambda: self.zoom_view(0.82))
        self.btn_zoom_out.clicked.connect(lambda: self.zoom_view(1.22))
        self.btn_export_image.clicked.connect(self.export_png)
        self.btn_export_scene.clicked.connect(self.export_scene)
        self.btn_open_scene.clicked.connect(self.open_scene)
        self.btn_show_all_layers.clicked.connect(self.show_all_layers)
        self.btn_hide_all_layers.clicked.connect(self.hide_all_layers)
        self.slider_model_opacity.valueChanged.connect(self._on_render_controls_changed)
        self.chk_model_edges.toggled.connect(self._on_render_controls_changed)
        self.chk_model_colors.toggled.connect(self._on_render_controls_changed)
        self.spin_model_point_size.valueChanged.connect(self._on_render_controls_changed)
        self.slider_metric_opacity.valueChanged.connect(self._on_render_controls_changed)
        self.spin_metric_emphasis.valueChanged.connect(self._on_render_controls_changed)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("button_press_event", self._on_mouse_click)

    def _apply_axes_style_3d(self) -> None:
        if self.ax is None:
            return
        self.ax.set_facecolor("#f5f7fb")
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            try:
                axis.set_pane_color((1.0, 1.0, 1.0, 0.98))
                axis._axinfo["grid"]["color"] = (0.35, 0.41, 0.49, 0.16)
                axis._axinfo["grid"]["linewidth"] = 0.7
                axis._axinfo["axisline"]["color"] = (0.38, 0.43, 0.50, 0.45)
                axis._axinfo["tick"]["color"] = (0.34, 0.39, 0.46, 0.8)
            except Exception:
                pass
        self.ax.tick_params(colors="#445063", labelsize=9)
        self.ax.grid(True, alpha=0.18)
        self.ax.view_init(elev=18, azim=-58)

    def _apply_axes_style_polar(self) -> None:
        if self.ax is None:
            return
        self.ax.set_facecolor("#ffffff")
        self.ax.tick_params(colors="#445063", labelsize=9)
        self.ax.grid(True, alpha=0.25, color="#8e97a6")

    def _configure_pyvista_view(self) -> None:
        if self.pv_plotter is None:
            return
        try:
            self.pv_plotter.set_background("#f8fafc", top="#edf3f8")
        except Exception:
            self.pv_plotter.set_background("#f8fafc")
        try:
            self.pv_plotter.enable_anti_aliasing()
        except Exception:
            pass
        try:
            self.pv_plotter.enable_depth_peeling()
        except Exception:
            pass
        try:
            self.pv_plotter.enable_trackball_style()
        except Exception:
            pass

    def _enable_pyvista_metric_picking(self) -> None:
        if self.pv_plotter is None:
            return
        try:
            self.pv_plotter.enable_point_picking(
                callback=self._on_pyvista_metric_pick,
                show_message=False,
                use_picker=False,
                show_point=False,
                pickable_window=False,
                tolerance=0.03,
                left_clicking=True,
                clear_on_no_selection=False,
            )
        except Exception:
            pass

    def _on_pyvista_metric_pick(self, point: Any) -> None:
        arr = np.asarray(point, dtype=float).reshape(-1)
        if arr.size < 3 or not self._interactive_points:
            return
        xyz = arr[:3]
        nearest = min(
            self._interactive_points,
            key=lambda item: float(np.linalg.norm(_as_xyz(item["xyz"]) - xyz)),
        )
        self._select_interactive_point(nearest, focus=False)

    def _use_pyvista(self) -> bool:
        return bool(HAS_PYVISTA and self._projection == "3d" and self.pv_plotter is not None and self._pv_widget is not None)

    def _set_viewer_mode(self, projection: str) -> None:
        if projection == "3d" and self._use_pyvista():
            self._viewer_mode = "pyvista"
            if self._pv_widget is not None:
                self.viewer_stack.setCurrentWidget(self._pv_widget)
        else:
            self._viewer_mode = "matplotlib"
            self.viewer_stack.setCurrentWidget(self.canvas)

    def _pv_set_artist_visibility(self, artist: Any, visible: bool) -> None:
        try:
            if hasattr(artist, "set_visibility"):
                artist.set_visibility(visible)
            elif hasattr(artist, "SetVisibility"):
                artist.SetVisibility(1 if visible else 0)
            elif hasattr(artist, "VisibilityOn") and hasattr(artist, "VisibilityOff"):
                artist.VisibilityOn() if visible else artist.VisibilityOff()
        except Exception:
            pass

    def _pv_polyline(self, points: np.ndarray, color: str, width: float = 2.0, closed: bool = False, opacity: float = 1.0, name: Optional[str] = None, pickable: bool = False):
        if self.pv_plotter is None or pv is None:
            return None
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
            return None
        if closed:
            pts = np.vstack([pts, pts[0]])
        line = pv.lines_from_points(pts, close=False)
        return self.pv_plotter.add_mesh(line, color=color, line_width=width, opacity=opacity, name=name, reset_camera=False, pickable=pickable)

    def _pv_points(self, points: np.ndarray, color: Optional[str] = None, rgb: Optional[np.ndarray] = None, size: float = 10.0, opacity: float = 1.0, name: Optional[str] = None, pickable: bool = False):
        if self.pv_plotter is None or pv is None:
            return None
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
            return None
        cloud = pv.PolyData(pts)
        kwargs = dict(point_size=size, render_points_as_spheres=True, opacity=opacity, name=name, reset_camera=False, pickable=pickable)
        if rgb is not None:
            arr = np.asarray(rgb, dtype=float)
            if arr.ndim == 2 and arr.shape[0] == len(pts) and arr.shape[1] == 3:
                cloud.point_data["rgb"] = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
                return self.pv_plotter.add_mesh(cloud, scalars="rgb", rgb=True, **kwargs)
        return self.pv_plotter.add_mesh(cloud, color=color or "#94a3b8", **kwargs)

    def _pv_surface(self, triangles: np.ndarray, color: Optional[str] = None, rgb: Optional[np.ndarray] = None, opacity: float = 1.0, show_edges: bool = False, edge_color: str = "#64748b", line_width: float = 0.2, name: Optional[str] = None, pickable: bool = False):
        if self.pv_plotter is None or pv is None:
            return None
        tris = np.asarray(triangles, dtype=float)
        if tris.ndim != 3 or len(tris) == 0 or tris.shape[-1] != 3 or tris.shape[1] not in (3, 4):
            return None
        if tris.shape[1] == 4:
            quad_count = len(tris)
            tris = np.concatenate(
                [
                    tris[:, [0, 1, 2], :],
                    tris[:, [0, 2, 3], :],
                ],
                axis=0,
            )
            if rgb is not None:
                arr = np.asarray(rgb, dtype=float)
                if arr.ndim == 2 and arr.shape[0] == quad_count and arr.shape[1] == 3:
                    rgb = np.repeat(arr, 2, axis=0)
        verts = tris.reshape(-1, 3)
        face_count = len(tris)
        faces = np.hstack(
            [np.full((face_count, 1), 3, dtype=np.int64), np.arange(face_count * 3, dtype=np.int64).reshape(face_count, 3)]
        ).reshape(-1)
        surf = pv.PolyData(verts, faces)
        kwargs = dict(
            opacity=opacity,
            show_edges=show_edges,
            edge_color=edge_color,
            line_width=line_width,
            smooth_shading=True,
            ambient=0.25,
            diffuse=0.82,
            specular=0.10,
            specular_power=12,
            name=name,
            reset_camera=False,
            pickable=pickable,
        )
        if rgb is not None:
            arr = np.asarray(rgb, dtype=float)
            if arr.ndim == 2 and arr.shape[0] == face_count and arr.shape[1] == 3:
                surf.cell_data["rgb"] = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
                return self.pv_plotter.add_mesh(surf, scalars="rgb", rgb=True, **kwargs)
        return self.pv_plotter.add_mesh(surf, color=color or "#94a3b8", **kwargs)

    def _pv_polygon_surface(self, polygon: np.ndarray, color: str, opacity: float, name: Optional[str] = None, pickable: bool = False):
        poly = np.asarray(polygon, dtype=float)
        if poly.ndim != 2 or poly.shape[1] != 3 or len(poly) < 3:
            return None
        tris = np.array([[poly[0], poly[i], poly[i + 1]] for i in range(1, len(poly) - 1)], dtype=float)
        return self._pv_surface(tris, color=color, opacity=opacity, show_edges=False, name=name, pickable=pickable)

    def _pv_arrow(self, start: np.ndarray, direction: np.ndarray, scale: float, color: str, name: Optional[str] = None, pickable: bool = False):
        if self.pv_plotter is None or pv is None:
            return None
        start = np.asarray(start, dtype=float).reshape(3,)
        direction = np.asarray(direction, dtype=float).reshape(3,)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            return None
        arrow = pv.Arrow(start=start, direction=direction / norm, scale=scale)
        return self.pv_plotter.add_mesh(arrow, color=color, opacity=0.95, name=name, reset_camera=False, pickable=pickable)

    def _pv_set_bounds_from_points(self, arrays: List[np.ndarray]) -> None:
        if self.pv_plotter is None:
            return
        valid = []
        for arr in arrays:
            arr = np.asarray(arr, dtype=float)
            if arr.ndim == 2 and arr.shape[1] == 3 and len(arr) > 0:
                valid.append(arr)
        if not valid:
            return
        pts = np.vstack(valid)
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        spans = np.maximum(maxs - mins, 1e-6)
        pad = 0.08 * float(np.max(spans))
        bounds = (
            float(mins[0] - pad), float(maxs[0] + pad),
            float(mins[1] - pad), float(maxs[1] + pad),
            float(mins[2] - pad), float(maxs[2] + pad),
        )
        try:
            self.pv_plotter.show_bounds(
                bounds=bounds,
                grid="back",
                location="outer",
                all_edges=True,
                xtitle="X",
                ytitle="Y",
                ztitle="Z",
                font_size=9,
                color="#7b8aa0",
                fmt="%.3f",
                minor_ticks=False,
            )
        except Exception:
            pass
        try:
            self.pv_plotter.reset_camera(bounds=bounds)
        except Exception:
            try:
                self.pv_plotter.reset_camera()
            except Exception:
                pass

    def clear(self, title: str = "", projection: str = "3d"):
        if not HAS_MPL:
            self._title.setText(title or "")
            return
        self._projection = projection
        self._set_viewer_mode(projection)
        if self._viewer_mode == "matplotlib":
            self.fig.clear()
            if projection == "polar":
                self.ax = self.fig.add_subplot(111, projection="polar")
                self._apply_axes_style_polar()
            else:
                self.ax = self.fig.add_subplot(111, projection="3d")
                self._apply_axes_style_3d()
        else:
            if self.pv_plotter is not None:
                self.pv_plotter.clear()
                self._configure_pyvista_view()
        self._interactive_points = []
        self.list_points.clear()
        self._layer_artists = {}
        self._layer_list_syncing = True
        self.list_layers.clear()
        self._layer_list_syncing = False
        self._selection_artist = None
        self._selected_point = None
        self._pv_selection_actor = None
        self._pv_pick_actor = None
        self._title.setText(title or "")
        self._status.setText("Hover over a point to inspect its coordinates. Click to pin the selection." if self._viewer_mode == "matplotlib" else "Use the scene controls to inspect the 3D geometry.")
        if self._viewer_mode == "matplotlib":
            self.canvas.draw_idle()

    def _normalize_artists(self, artists: Any) -> List[Any]:
        if artists is None:
            return []
        if isinstance(artists, (list, tuple)):
            flat: List[Any] = []
            for artist in artists:
                flat.extend(self._normalize_artists(artist))
            return flat
        return [artists]

    def _register_layer(self, label: str, artists: Any, visible: bool = True) -> None:
        if not HAS_MPL or not label:
            return
        normalized = [artist for artist in self._normalize_artists(artists) if artist is not None]
        if not normalized:
            return
        for artist in normalized:
            self._set_artist_visible(artist, visible)
        self._layer_artists[label] = normalized
        self._layer_list_syncing = True
        item = QtWidgets.QListWidgetItem(label)
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
        item.setCheckState(QtCore.Qt.Checked if visible else QtCore.Qt.Unchecked)
        self.list_layers.addItem(item)
        self._layer_list_syncing = False

    def _on_layer_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        if self._layer_list_syncing:
            return
        label = item.text()
        visible = item.checkState() == QtCore.Qt.Checked
        for artist in self._layer_artists.get(label, []):
            self._set_artist_visible(artist, visible)
        if self._selection_artist is not None and self._selected_point is not None:
            self._set_artist_visible(self._selection_artist, True)
        if self._viewer_mode == "matplotlib":
            self.canvas.draw_idle()
        elif self.pv_plotter is not None:
            self.pv_plotter.render()

    def _set_all_layers(self, visible: bool) -> None:
        self._layer_list_syncing = True
        for i in range(self.list_layers.count()):
            item = self.list_layers.item(i)
            item.setCheckState(QtCore.Qt.Checked if visible else QtCore.Qt.Unchecked)
            label = item.text()
            for artist in self._layer_artists.get(label, []):
                self._set_artist_visible(artist, visible)
        self._layer_list_syncing = False
        if self._selection_artist is not None and self._selected_point is not None:
            self._set_artist_visible(self._selection_artist, True)
        if self._viewer_mode == "matplotlib":
            self.canvas.draw_idle()
        elif self.pv_plotter is not None:
            self.pv_plotter.render()

    def _set_artist_visible(self, artist: Any, visible: bool) -> None:
        try:
            if hasattr(artist, "set_visible"):
                artist.set_visible(visible)
            elif hasattr(artist, "SetVisibility"):
                artist.SetVisibility(1 if visible else 0)
            elif hasattr(artist, "VisibilityOn") and hasattr(artist, "VisibilityOff"):
                artist.VisibilityOn() if visible else artist.VisibilityOff()
        except Exception:
            pass

    def show_all_layers(self) -> None:
        self._set_all_layers(True)

    def hide_all_layers(self) -> None:
        self._set_all_layers(False)

    def plot_points(self, points: Dict[str, np.ndarray], title: str = "Dataset points", scene_meta: Optional[dict] = None):
        scene = {
            "scene_type": "dataset",
            "title": title,
            "points": {str(k): _json_safe(_as_xyz(v)) for k, v in points.items()},
            "scene_meta": _json_safe(scene_meta or {}),
        }
        self._render_scene(scene)

    def plot_metric_result(self, result: Any, title: str = "Metric result", scene_meta: Optional[dict] = None):
        payload = asdict(result) if is_dataclass(result) else result
        scene = {
            "scene_type": "metric",
            "metric": title,
            "title": title,
            "payload": _json_safe(payload),
            "scene_meta": _json_safe(scene_meta or {}),
        }
        self._render_scene(scene)

    def _render_scene(self, scene: dict) -> None:
        if not HAS_MPL:
            self._title.setText(f"{scene.get('title', 'Plot')}\n\n{scene!r}")
            return

        scene = _json_safe(scene)
        title = str(scene.get("title", "Plot"))
        scene_type = str(scene.get("scene_type", "") or "")
        payload = scene.get("payload", {}) if isinstance(scene.get("payload", {}), dict) else {}
        metric = str(scene.get("metric", "") or "")
        try:
            if scene_type == "dataset":
                self._render_dataset_scene(scene)
            elif scene_type == "metric" and metric == "VOM / VoA":
                self._render_vom_scene(title, payload, scene.get("scene_meta", {}))
            elif scene_type == "metric" and metric == "AoA / SF":
                self._render_aoa_scene(title, payload, scene.get("scene_meta", {}))
            elif scene_type == "metric" and metric == "AoE":
                self._render_aoe_scene(title, payload)
            elif scene_type == "metric" and metric == "Distance":
                self._render_distance_scene(title, payload, scene.get("scene_meta", {}))
            elif scene_type == "metric" and metric == "Area":
                self._render_area_scene(title, payload, scene.get("scene_meta", {}))
            elif scene_type == "metric" and metric == "Volume":
                self._render_volume_scene(title, payload, scene.get("scene_meta", {}))
            else:
                self.clear(title)
                self._title.setText(f"{title}\n\n{payload!r}")
        except Exception as e:
            self.clear(title)
            self._title.setText(f"{title}\n\nPlot failed: {e}")
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_zlabel("Z")

        view = scene.get("view", {}) if isinstance(scene.get("view"), dict) else {}
        if self._projection == "3d":
            if self._viewer_mode == "matplotlib":
                elev = float(view.get("elev", 18.0))
                azim = float(view.get("azim", -58.0))
                self.ax.view_init(elev=elev, azim=azim)
                xlim = view.get("xlim")
                ylim = view.get("ylim")
                zlim = view.get("zlim")
                try:
                    if isinstance(xlim, (list, tuple)) and len(xlim) == 2:
                        self.ax.set_xlim3d([float(xlim[0]), float(xlim[1])])
                    if isinstance(ylim, (list, tuple)) and len(ylim) == 2:
                        self.ax.set_ylim3d([float(ylim[0]), float(ylim[1])])
                    if isinstance(zlim, (list, tuple)) and len(zlim) == 2:
                        self.ax.set_zlim3d([float(zlim[0]), float(zlim[1])])
                except Exception:
                    pass
            elif self.pv_plotter is not None:
                camera_position = view.get("camera_position")
                if isinstance(camera_position, (list, tuple)) and len(camera_position) == 3:
                    try:
                        self.pv_plotter.camera_position = camera_position
                    except Exception:
                        pass
        self._scene_payload = scene
        if self._viewer_mode == "matplotlib":
            self.canvas.draw_idle()
        elif self.pv_plotter is not None:
            self.pv_plotter.render()

    def _current_view_state(self) -> dict:
        if self._projection != "3d":
            return {}
        if self._viewer_mode == "pyvista" and self.pv_plotter is not None:
            try:
                return {"camera_position": [list(map(float, p)) for p in self.pv_plotter.camera_position]}
            except Exception:
                return {}
        if self.ax is None:
            return {}
        return {
            "elev": float(self.ax.elev),
            "azim": float(self.ax.azim),
            "xlim": [float(v) for v in self.ax.get_xlim3d()],
            "ylim": [float(v) for v in self.ax.get_ylim3d()],
            "zlim": [float(v) for v in self.ax.get_zlim3d()],
        }

    def _rerender_current_scene(self) -> None:
        if self._scene_payload is None:
            return
        scene = dict(self._scene_payload)
        scene["view"] = self._current_view_state()
        self._render_scene(scene)

    def _on_render_controls_changed(self, *_args) -> None:
        self._model_opacity = max(0.05, float(self.slider_model_opacity.value()) / 100.0)
        self._model_show_edges = bool(self.chk_model_edges.isChecked())
        self._model_use_source_colors = bool(self.chk_model_colors.isChecked())
        self._model_point_size = float(self.spin_model_point_size.value())
        self._metric_opacity = max(0.15, float(self.slider_metric_opacity.value()) / 100.0)
        self._metric_emphasis = max(0.5, float(self.spin_metric_emphasis.value()))
        self._rerender_current_scene()

    def _metric_alpha(self, base: float) -> float:
        return float(max(0.05, min(1.0, float(base) * self._metric_opacity)))

    def _metric_size(self, base: float) -> float:
        return float(max(2.0, float(base) * self._metric_emphasis))

    def _metric_width(self, base: float) -> float:
        return float(max(0.5, float(base) * self._metric_emphasis))

    def _set_interactive_points(self, points: List[dict]) -> None:
        self._interactive_points = [
            {
                "label": str(p.get("label", "Point")),
                "xyz": _as_xyz(p.get("xyz")),
                "kind": str(p.get("kind", "point")),
            }
            for p in points
        ]
        self.list_points.clear()
        for point in self._interactive_points:
            item = QtWidgets.QListWidgetItem(str(point["label"]))
            self.list_points.addItem(item)
        if self._viewer_mode == "pyvista" and self.pv_plotter is not None:
            self._pv_selection_actor = None
            self._pv_pick_actor = None
            if self._interactive_points:
                coords = np.vstack([p["xyz"] for p in self._interactive_points])
                self._pv_pick_actor = self._pv_points(
                    coords,
                    color="#111827",
                    size=max(18.0, self._model_point_size + 10.0),
                    opacity=0.001,
                    name="interactive_pick_points",
                    pickable=True,
                )
                self._enable_pyvista_metric_picking()
            else:
                self._pv_pick_actor = None

    def _focus_pyvista_point(self, point: dict) -> None:
        if self.pv_plotter is None:
            return
        xyz = _as_xyz(point["xyz"])
        try:
            if self._pv_selection_actor is not None:
                self.pv_plotter.remove_actor(self._pv_selection_actor, render=False)
        except Exception:
            pass
        self._pv_selection_actor = self._pv_points(
            xyz.reshape(1, 3),
            color="#0f172a",
            size=max(18.0, self._model_point_size + 10.0),
            opacity=1.0,
            name="selected_metric_point",
            pickable=False,
        )
        try:
            position = np.asarray(self.pv_plotter.camera_position[0], dtype=float)
            focal = np.asarray(self.pv_plotter.camera_position[1], dtype=float)
            delta = xyz - focal
            self.pv_plotter.set_focus(xyz)
            self.pv_plotter.set_position(position + delta, reset=False, render=True)
        except Exception:
            try:
                self.pv_plotter.render()
            except Exception:
                pass

    def _select_interactive_point(self, point: dict, focus: bool = True) -> None:
        self._selected_point = point
        self._describe_point(point, "Selected")
        labels = [str(p["label"]) for p in self._interactive_points]
        try:
            idx = labels.index(str(point["label"]))
        except ValueError:
            idx = -1
        if idx >= 0:
            self.list_points.blockSignals(True)
            self.list_points.setCurrentRow(idx)
            self.list_points.blockSignals(False)
        if self._viewer_mode == "pyvista":
            self._focus_pyvista_point(point)
        elif focus:
            self._set_selected_point(point)

    def _on_point_item_selected(self) -> None:
        row = self.list_points.currentRow()
        if row < 0 or row >= len(self._interactive_points):
            return
        self._select_interactive_point(self._interactive_points[row], focus=True)

    def _render_dataset_scene(self, scene: dict) -> None:
        self.clear(str(scene.get("title", "Dataset points")))
        scene_meta = scene.get("scene_meta", {}) if isinstance(scene.get("scene_meta"), dict) else {}
        points = scene.get("points", {}) if isinstance(scene.get("points"), dict) else {}
        if not points:
            self._title.setText("No points to plot.")
            return

        labels = sorted(points.keys())
        xyz = np.vstack([_as_xyz(points[k]) for k in labels])
        if self._viewer_mode == "pyvista" and self.pv_plotter is not None:
            self._render_ai_scene_background(scene_meta, [])
            scatter = self._pv_points(
                xyz,
                color="#2f6fed",
                size=self._metric_size(12.0),
                opacity=self._metric_alpha(0.96),
                name="Dataset points",
            )
            self._register_layer("Dataset points", scatter)
            self._set_interactive_points(
                [{"label": lb, "xyz": xyz[i], "kind": "dataset"} for i, lb in enumerate(labels)]
            )
            self._title.setText(f"{len(labels)} points loaded")
            self._status.setText("Click a point in the scene or select it from the Metric Points list.")
            self._pv_set_bounds_from_points([xyz] + self._scene_point_arrays(scene_meta))
            return
        self._render_ai_scene_background(scene_meta, [])
        scatter = self.ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            s=58,
            c="#2f6fed",
            alpha=0.92,
            edgecolors="white",
            linewidths=0.9,
            depthshade=False,
        )
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title(f"{len(labels)} points loaded", fontsize=11, color="#233044", pad=12)
        self._register_layer("Dataset points", scatter)
        self._set_interactive_points(
            [{"label": lb, "xyz": xyz[i], "kind": "dataset"} for i, lb in enumerate(labels)]
        )
        self._set_axes_from_points([xyz] + self._scene_point_arrays(scene_meta))

    def _scene_point_arrays(self, scene_meta: dict) -> List[np.ndarray]:
        ai_scene = scene_meta.get("ai_scene") if isinstance(scene_meta, dict) else None
        if not isinstance(ai_scene, dict):
            return []
        arrays: List[np.ndarray] = []
        xyz = np.asarray(ai_scene.get("cloud_xyz", []), dtype=float)
        if xyz.ndim == 2 and xyz.shape[1] == 3 and len(xyz) > 0:
            arrays.append(xyz)
        tris = np.asarray(ai_scene.get("surface_triangles", []), dtype=float)
        if tris.ndim == 3 and tris.shape[-1] == 3 and len(tris) > 0:
            arrays.append(tris.reshape(-1, 3))
        return arrays

    def _render_ai_scene_background(self, scene_meta: dict, interactive: List[dict]) -> None:
        ai_scene = scene_meta.get("ai_scene") if isinstance(scene_meta, dict) else None
        if not isinstance(ai_scene, dict):
            return
        cloud_xyz = np.asarray(ai_scene.get("cloud_xyz", []), dtype=float)
        cloud_rgb = np.asarray(ai_scene.get("cloud_rgb", []), dtype=float)
        surface_triangles = np.asarray(ai_scene.get("surface_triangles", []), dtype=float)
        surface_rgb = np.asarray(ai_scene.get("surface_rgb", []), dtype=float)
        has_surface = surface_triangles.ndim == 3 and surface_triangles.shape[1:] == (3, 3) and len(surface_triangles) > 0
        show_surface = has_surface
        show_points = not has_surface
        if self._viewer_mode == "pyvista" and self.pv_plotter is not None:
            if has_surface:
                if (
                    not self._model_use_source_colors
                    or surface_rgb.ndim != 2
                    or surface_rgb.shape[0] != len(surface_triangles)
                    or surface_rgb.shape[1] != 3
                ):
                    surface_rgb = None
                surface_actor = self._pv_surface(
                    surface_triangles,
                    color="#94a3b8",
                    rgb=surface_rgb,
                    opacity=self._model_opacity,
                    show_edges=self._model_show_edges,
                    edge_color="#64748b",
                    line_width=0.2,
                    name="Rendered surface",
                )
                self._register_layer("Rendered surface", surface_actor, visible=show_surface)
            elif cloud_xyz.ndim == 2 and cloud_xyz.shape[1] == 3 and len(cloud_xyz) > 0:
                if (
                    not self._model_use_source_colors
                    or cloud_rgb.ndim != 2
                    or cloud_rgb.shape[0] != len(cloud_xyz)
                    or cloud_rgb.shape[1] != 3
                ):
                    cloud_rgb = None
                cloud_actor = self._pv_points(
                    cloud_xyz,
                    rgb=cloud_rgb,
                    color="#94a3b8",
                    size=self._model_point_size,
                    opacity=min(self._model_opacity + 0.10, 1.0),
                    name="Rendered imaging",
                )
                self._register_layer("Rendered imaging", cloud_actor, visible=show_points)
            calibration = str(ai_scene.get("calibration", "") or "")
            if calibration:
                self._status.setText(f"3D scene calibration: {calibration}.")
            return

        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        if surface_triangles.ndim == 3 and surface_triangles.shape[1:] == (3, 3) and len(surface_triangles) > 0:
            if (
                not self._model_use_source_colors
                or surface_rgb.ndim != 2
                or surface_rgb.shape[0] != len(surface_triangles)
                or surface_rgb.shape[1] != 3
            ):
                surface_rgb = None
            edgecolors = (0.40, 0.45, 0.52, 0.16) if self._model_show_edges else "none"
            linewidths = 0.10 if self._model_show_edges else 0.0
            surface_artist = self.ax.add_collection3d(
                Poly3DCollection(
                    list(surface_triangles),
                    facecolors=surface_rgb if surface_rgb is not None else (0.70, 0.74, 0.82, max(0.18, self._model_opacity)),
                    edgecolors=edgecolors,
                    linewidths=linewidths,
                    alpha=self._model_opacity,
                )
            )
            self._register_layer("Rendered surface", surface_artist, visible=show_surface)
        if cloud_xyz.ndim != 2 or cloud_xyz.shape[1] != 3 or len(cloud_xyz) == 0:
            return
        if cloud_rgb.ndim != 2 or cloud_rgb.shape[0] != len(cloud_xyz) or cloud_rgb.shape[1] != 3:
            cloud_rgb = None
        if not self._model_use_source_colors:
            cloud_rgb = None
        cloud_artist = self.ax.scatter(
            cloud_xyz[:, 0],
            cloud_xyz[:, 1],
            cloud_xyz[:, 2],
            s=self._model_point_size,
            c=cloud_rgb if cloud_rgb is not None else "#94a3b8",
            alpha=min(self._model_opacity + 0.10, 1.0),
            linewidths=0.0,
            depthshade=True,
        )
        self._register_layer("Rendered imaging", cloud_artist, visible=show_points)
        calibration = str(ai_scene.get("calibration", "") or "")
        if calibration:
            self._status.setText(
                "Hover over a point to inspect its coordinates. Click to pin the selection. "
                f"3D scene calibration: {calibration}."
            )

    def _render_vom_scene(self, title: str, payload: dict, scene_meta: dict) -> None:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        self.clear(title)
        dbg = payload.get("debug", {}) if isinstance(payload.get("debug"), dict) else {}
        entry_poly = np.asarray(dbg.get("entry_polygon", []), dtype=float)
        target_poly = np.asarray(dbg.get("target_polygon", []), dtype=float)
        entry_ellipse = np.asarray(dbg.get("entry_ellipse_3d", []), dtype=float)
        target_ellipse = np.asarray(dbg.get("target_ellipse_3d", []), dtype=float)
        cut_ellipse = np.asarray(dbg.get("svom_cut_ellipse_3d", []), dtype=float)
        full_surface = np.asarray(dbg.get("frustum_surface", []), dtype=float)
        svom_surface = np.asarray(dbg.get("svom_surface", []), dtype=float)
        cE = np.asarray(dbg.get("centroid_entry", []), dtype=float).reshape(-1)
        cT = np.asarray(dbg.get("centroid_target", []), dtype=float).reshape(-1)
        all_points: List[np.ndarray] = []
        interactive: List[dict] = []
        entry_names = list(scene_meta.get("entry_names", []) or [])
        target_names = list(scene_meta.get("target_names", []) or [])
        self._render_ai_scene_background(scene_meta, interactive)
        if self._viewer_mode == "pyvista" and self.pv_plotter is not None:
            if entry_poly.ndim == 2 and entry_poly.shape[1] == 3:
                all_points.append(entry_poly)
                self._register_layer("Entry polygon", [
                    self._pv_polyline(entry_poly, "tab:blue", width=self._metric_width(3.4), closed=True, opacity=self._metric_alpha(0.82), name="entry_polygon_line"),
                    self._pv_points(entry_poly, color="tab:blue", size=self._metric_size(15.0), opacity=self._metric_alpha(0.96), name="entry_polygon_points"),
                ])
                for i, pt in enumerate(entry_poly):
                    label = entry_names[i] if i < len(entry_names) else f"Entry {i + 1}"
                    interactive.append({"label": label, "xyz": pt, "kind": "entry"})
            if target_poly.ndim == 2 and target_poly.shape[1] == 3:
                all_points.append(target_poly)
                self._register_layer("Target polygon", [
                    self._pv_polyline(target_poly, "tab:red", width=self._metric_width(3.4), closed=True, opacity=self._metric_alpha(0.82), name="target_polygon_line"),
                    self._pv_points(target_poly, color="tab:red", size=self._metric_size(15.0), opacity=self._metric_alpha(0.96), name="target_polygon_points"),
                ])
                for i, pt in enumerate(target_poly):
                    label = target_names[i] if i < len(target_names) else f"Target {i + 1}"
                    interactive.append({"label": label, "xyz": pt, "kind": "target"})
            if entry_ellipse.ndim == 2 and entry_ellipse.shape[1] == 3:
                all_points.append(entry_ellipse)
                self._register_layer("Entry ellipse", self._pv_polyline(entry_ellipse, "tab:blue", width=self._metric_width(3.5), closed=False, opacity=self._metric_alpha(1.0), name="entry_ellipse"))
            if target_ellipse.ndim == 2 and target_ellipse.shape[1] == 3:
                all_points.append(target_ellipse)
                self._register_layer("Target ellipse", self._pv_polyline(target_ellipse, "tab:red", width=self._metric_width(3.5), closed=False, opacity=self._metric_alpha(1.0), name="target_ellipse"))
            if full_surface.ndim == 3 and full_surface.shape[-1] == 3 and len(full_surface) > 0:
                self._register_layer("Corridor shell", self._pv_surface(full_surface, color="#6b7280", opacity=self._metric_alpha(0.34), show_edges=True, edge_color="#a7b0bc", line_width=self._metric_width(0.45), name="corridor_shell"))
                all_points.append(full_surface.reshape(-1, 3))
            if svom_surface.ndim == 3 and svom_surface.shape[-1] == 3 and len(svom_surface) > 0:
                self._register_layer("sVOM region", self._pv_surface(svom_surface, color="tab:green", opacity=self._metric_alpha(0.34), show_edges=True, edge_color="#3b8f5d", line_width=self._metric_width(0.35), name="svom_region"))
                all_points.append(svom_surface.reshape(-1, 3))
            if cut_ellipse.ndim == 2 and cut_ellipse.shape[1] == 3:
                all_points.append(cut_ellipse)
                self._register_layer("sVOM cut ellipse", self._pv_polyline(cut_ellipse, "tab:green", width=self._metric_width(3.0), closed=False, opacity=self._metric_alpha(1.0), name="svom_cut_ellipse"))
            if cE.size == 3 and cT.size == 3:
                centers = np.vstack([cE, cT])
                all_points.append(centers)
                self._register_layer("Trajectory axis", [
                    self._pv_points(centers, color="#111827", size=self._metric_size(16.0), opacity=self._metric_alpha(0.98), name="trajectory_centers"),
                    self._pv_polyline(centers, "#111827", width=self._metric_width(2.8), closed=False, opacity=self._metric_alpha(0.88), name="trajectory_axis"),
                ])
                interactive.extend([
                    {"label": "Entry centroid", "xyz": cE, "kind": "centroid"},
                    {"label": "Target centroid", "xyz": cT, "kind": "centroid"},
                ])
            if cE.size == 3 and cT.size == 3:
                approach = cT - cE
                norm = float(np.linalg.norm(approach))
                if norm > 1e-9:
                    scale = max(float(dbg.get("distance_h", 0.0) or 0.0) * 0.25, 1.0)
                    direction = approach / norm
                    start = cT - direction * scale
                    all_points.append(np.vstack([start, cT]))
                    self._register_layer("Target approach", self._pv_arrow(start, direction, scale, "tab:purple", name="target_approach"))
            self._set_interactive_points(interactive)
            self._title.setText(
                f"VoA {float(payload.get('voa_deg', 0.0)):.2f}°  |  "
                f"VOM {float(payload.get('vom_mm3', 0.0)):.2f} mm³  |  "
                f"sVOM {float(payload.get('svom_mm3', 0.0)):.2f} mm³"
            )
            self._status.setText("Click a metric point in the scene or use the Metric Points list.")
            self._pv_set_bounds_from_points(all_points + self._scene_point_arrays(scene_meta))
            return

        if entry_poly.ndim == 2 and entry_poly.shape[1] == 3:
            closed = np.vstack([entry_poly, entry_poly[0]])
            all_points.append(entry_poly)
            entry_line = self.ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color="tab:blue", linewidth=1.2, alpha=0.48)
            entry_pts = self.ax.scatter(
                entry_poly[:, 0], entry_poly[:, 1], entry_poly[:, 2],
                color="tab:blue", s=24, alpha=0.7, edgecolors="white", linewidths=0.45, depthshade=False,
            )
            self._register_layer("Entry polygon", [entry_line, entry_pts])
            for i, pt in enumerate(entry_poly):
                label = entry_names[i] if i < len(entry_names) else f"Entry {i + 1}"
                interactive.append({"label": label, "xyz": pt, "kind": "entry"})

        if target_poly.ndim == 2 and target_poly.shape[1] == 3:
            closed = np.vstack([target_poly, target_poly[0]])
            all_points.append(target_poly)
            target_line = self.ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color="tab:red", linewidth=1.2, alpha=0.48)
            target_pts = self.ax.scatter(
                target_poly[:, 0], target_poly[:, 1], target_poly[:, 2],
                color="tab:red", s=24, alpha=0.7, edgecolors="white", linewidths=0.45, depthshade=False,
            )
            self._register_layer("Target polygon", [target_line, target_pts])
            for i, pt in enumerate(target_poly):
                label = target_names[i] if i < len(target_names) else f"Target {i + 1}"
                interactive.append({"label": label, "xyz": pt, "kind": "target"})

        if entry_ellipse.ndim == 2 and entry_ellipse.shape[1] == 3:
            all_points.append(entry_ellipse)
            entry_ellipse_line = self.ax.plot(entry_ellipse[:, 0], entry_ellipse[:, 1], entry_ellipse[:, 2], color="tab:blue", linewidth=2.2)
            self._register_layer("Entry ellipse", entry_ellipse_line)

        if target_ellipse.ndim == 2 and target_ellipse.shape[1] == 3:
            all_points.append(target_ellipse)
            target_ellipse_line = self.ax.plot(target_ellipse[:, 0], target_ellipse[:, 1], target_ellipse[:, 2], color="tab:red", linewidth=2.2)
            self._register_layer("Target ellipse", target_ellipse_line)

        if full_surface.ndim == 3 and full_surface.shape[-1] == 3 and len(full_surface) > 0:
            frustum_artist = self.ax.add_collection3d(
                Poly3DCollection(
                    list(full_surface),
                    facecolors="tab:gray",
                    alpha=0.045,
                    edgecolors=(0.65, 0.67, 0.70, 0.12),
                    linewidths=0.15,
                )
            )
            self._register_layer("Corridor shell", frustum_artist)

        if svom_surface.ndim == 3 and svom_surface.shape[-1] == 3 and len(svom_surface) > 0:
            svom_artist = self.ax.add_collection3d(
                Poly3DCollection(
                    list(svom_surface),
                    facecolors="tab:green",
                    alpha=0.18,
                    edgecolors=(0.14, 0.62, 0.32, 0.25),
                    linewidths=0.25,
                )
            )
            self._register_layer("sVOM region", svom_artist)

        if cut_ellipse.ndim == 2 and cut_ellipse.shape[1] == 3:
            all_points.append(cut_ellipse)
            cut_line = self.ax.plot(cut_ellipse[:, 0], cut_ellipse[:, 1], cut_ellipse[:, 2], color="tab:green", linewidth=1.9)
            self._register_layer("sVOM cut ellipse", cut_line)

        if cE.size == 3 and cT.size == 3:
            centers = np.vstack([cE, cT])
            all_points.append(centers)
            centers_pts = self.ax.scatter(
                centers[:, 0], centers[:, 1], centers[:, 2],
                color="black", s=44, depthshade=False, edgecolors="white", linewidths=0.6,
            )
            centers_line = self.ax.plot(
                [cE[0], cT[0]], [cE[1], cT[1]], [cE[2], cT[2]],
                color="black", linestyle="--", linewidth=1.35, alpha=0.75,
            )
            self._register_layer("Trajectory axis", [centers_pts, centers_line])
            interactive.extend([
                {"label": "Entry centroid", "xyz": cE, "kind": "centroid"},
                {"label": "Target centroid", "xyz": cT, "kind": "centroid"},
            ])

        if cE.size == 3 and cT.size == 3:
            approach = cT - cE
            norm = float(np.linalg.norm(approach))
            if norm > 1e-9:
                scale = max(float(dbg.get("distance_h", 0.0) or 0.0) * 0.25, 1.0)
                direction = approach / norm
                start = cT - direction * scale
                all_points.append(np.vstack([start, cT]))
                approach_artist = self.ax.quiver(
                    start[0], start[1], start[2],
                    direction[0], direction[1], direction[2],
                    length=scale,
                    color="tab:purple",
                    normalize=True,
                    linewidth=1.3,
                )
                self._register_layer("Target approach", approach_artist)

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title(
            f"VoA {float(payload.get('voa_deg', 0.0)):.2f}°  |  "
            f"VOM {float(payload.get('vom_mm3', 0.0)):.2f} mm³  |  "
            f"sVOM {float(payload.get('svom_mm3', 0.0)):.2f} mm³",
            fontsize=11,
            color="#233044",
            pad=12,
        )
        self._set_interactive_points(interactive)
        self._set_axes_from_points(all_points + self._scene_point_arrays(scene_meta))

    def _render_aoa_scene(self, title: str, payload: dict, scene_meta: dict) -> None:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        self.clear(title)
        dbg = payload.get("debug", {}) if isinstance(payload.get("debug"), dict) else {}
        entry = np.asarray(dbg.get("entry_points", []), dtype=float)
        pivot = np.asarray(dbg.get("pivot", []), dtype=float).reshape(-1)
        rescaled = np.asarray(dbg.get("rescaled_entry_points", []), dtype=float)
        vertical_triangle = np.asarray(dbg.get("vertical_triangle", []), dtype=float)
        horizontal_triangle = np.asarray(dbg.get("horizontal_triangle", []), dtype=float)
        vertical_triangle_rescaled = np.asarray(dbg.get("vertical_triangle_rescaled", []), dtype=float)
        horizontal_triangle_rescaled = np.asarray(dbg.get("horizontal_triangle_rescaled", []), dtype=float)
        sf_surface = np.asarray(dbg.get("sf_surface_3d", []), dtype=float)
        sf_surface_rescaled = np.asarray(dbg.get("sf_surface_rescaled_3d", []), dtype=float)

        all_points: List[np.ndarray] = []
        interactive: List[dict] = []
        entry_names = list(scene_meta.get("entry_names", []) or [])
        vertical_props = dbg.get("vertical_triangle_properties", {}) if isinstance(dbg.get("vertical_triangle_properties"), dict) else {}
        horizontal_props = dbg.get("horizontal_triangle_properties", {}) if isinstance(dbg.get("horizontal_triangle_properties"), dict) else {}
        self._render_ai_scene_background(scene_meta, interactive)
        if self._viewer_mode == "pyvista" and self.pv_plotter is not None:
            def _pv_triangle(tri: np.ndarray, color: str, label: str, face_alpha: float) -> None:
                if tri.ndim != 2 or tri.shape != (3, 3):
                    return
                self._register_layer(label, [
                    self._pv_polyline(tri, color, width=self._metric_width(3.2), closed=True, opacity=self._metric_alpha(0.95), name=f"{label}_line"),
                    self._pv_points(tri, color=color, size=self._metric_size(15.0), opacity=self._metric_alpha(0.96), name=f"{label}_points"),
                    self._pv_polygon_surface(tri, color=color, opacity=self._metric_alpha(face_alpha), name=f"{label}_surface"),
                ])
                all_points.append(tri)

            _pv_triangle(vertical_triangle, "tab:blue", "V-AoA", 0.18)
            _pv_triangle(horizontal_triangle, "tab:green", "H-AoA", 0.18)
            if sf_surface.ndim == 2 and sf_surface.shape == (4, 3):
                self._register_layer("SF surface", [
                    self._pv_polyline(sf_surface, "#0f172a", width=2.4, closed=True, opacity=0.75, name="sf_surface_line"),
                    self._pv_polygon_surface(sf_surface, color="#94a3b8", opacity=self._metric_alpha(0.14), name="sf_surface_fill"),
                ])
                all_points.append(sf_surface)
            if sf_surface_rescaled.ndim == 2 and sf_surface_rescaled.shape == (4, 3):
                self._register_layer("Standardized SF surface", [
                    self._pv_polyline(sf_surface_rescaled, "tab:orange", width=self._metric_width(2.8), closed=True, opacity=self._metric_alpha(0.90), name="sf_surface_std_line"),
                    self._pv_polygon_surface(sf_surface_rescaled, color="tab:orange", opacity=self._metric_alpha(0.10), name="sf_surface_std_fill"),
                ])
                all_points.append(sf_surface_rescaled)
            if pivot.size == 3:
                self._register_layer("Pivot", self._pv_points(pivot.reshape(1, 3), color="tab:red", size=self._metric_size(18.0), opacity=self._metric_alpha(0.98), name="pivot_point"))
                all_points.append(pivot.reshape(1, 3))
            if rescaled.ndim == 2 and rescaled.shape[1] == 3 and len(rescaled) >= 4:
                proj_actors: List[Any] = [self._pv_points(rescaled, color="tab:orange", size=self._metric_size(14.0), opacity=self._metric_alpha(0.94), name="projected_points")]
                for original, projected in zip(entry[:4], rescaled[:4]):
                    proj_actors.append(self._pv_polyline(np.vstack([original, projected]), "black", width=self._metric_width(2.0), closed=False, opacity=self._metric_alpha(0.72)))
                self._register_layer("Standardization projection", proj_actors)
                all_points.append(rescaled)
            if entry.ndim == 2 and entry.shape[1] == 3:
                for i, pt in enumerate(entry[:4]):
                    label = entry_names[i] if i < len(entry_names) else f"Entry {i + 1}"
                    interactive.append({"label": label, "xyz": pt, "kind": "entry"})
            if pivot.size == 3:
                interactive.append({"label": scene_meta.get("pivot_name", "Pivot"), "xyz": pivot, "kind": "pivot"})
            self._set_interactive_points(interactive)
            self._title.setText(
                f"AoA(v) {float(payload.get('aoa_vertical_deg', 0.0)):.2f}°  |  "
                f"AoA(h) {float(payload.get('aoa_horizontal_deg', 0.0)):.2f}°  |  "
                f"SF {float(payload.get('sf_entry_area_mm2', 0.0)):.2f} mm²"
            )
            self._status.setText(
                f"V-AoA γ: {float(vertical_props.get('angles_deg', {}).get('gamma', payload.get('aoa_vertical_deg', 0.0))):.2f}°  |  "
                f"H-AoA γ: {float(horizontal_props.get('angles_deg', {}).get('gamma', payload.get('aoa_horizontal_deg', 0.0))):.2f}°  |  "
                "click a point in the scene or use the Metric Points list"
            )
            self._pv_set_bounds_from_points(all_points + self._scene_point_arrays(scene_meta))
            return

        def _plot_triangle(tri: np.ndarray, color: str, face_alpha: float, label: str, linestyle: str = "-") -> None:
            if tri.ndim != 2 or tri.shape != (3, 3):
                return
            closed = np.vstack([tri, tri[0]])
            tri_line = self.ax.plot(
                closed[:, 0], closed[:, 1], closed[:, 2],
                color=color, linewidth=2.2, alpha=0.95, linestyle=linestyle, label=label,
            )
            tri_pts = self.ax.scatter(
                tri[:, 0], tri[:, 1], tri[:, 2],
                color=color, s=52, edgecolors="white", linewidths=0.6, depthshade=False,
            )
            tri_face = self.ax.add_collection3d(
                Poly3DCollection([tri], facecolors=color, alpha=face_alpha, edgecolors="none")
            )
            self._register_layer(label, [tri_line, tri_pts, tri_face])
            all_points.append(tri)

        _plot_triangle(vertical_triangle, "tab:blue", 0.18, "V-AoA")
        _plot_triangle(horizontal_triangle, "tab:green", 0.18, "H-AoA")

        if sf_surface.ndim == 2 and sf_surface.shape == (4, 3):
            closed = np.vstack([sf_surface, sf_surface[0]])
            sf_line = self.ax.plot(
                closed[:, 0], closed[:, 1], closed[:, 2],
                color="#0f172a", linewidth=1.5, alpha=0.7, label="SF surface",
            )
            sf_face = self.ax.add_collection3d(
                Poly3DCollection([sf_surface], facecolors="#94a3b8", alpha=0.14, edgecolors="none")
            )
            self._register_layer("SF surface", [sf_line, sf_face])
            all_points.append(sf_surface)

        if sf_surface_rescaled.ndim == 2 and sf_surface_rescaled.shape == (4, 3):
            closed = np.vstack([sf_surface_rescaled, sf_surface_rescaled[0]])
            sf_std_line = self.ax.plot(
                closed[:, 0], closed[:, 1], closed[:, 2],
                color="tab:orange", linewidth=1.7, alpha=0.92, label="Standardized SF surface",
            )
            sf_std_face = self.ax.add_collection3d(
                Poly3DCollection([sf_surface_rescaled], facecolors="tab:orange", alpha=0.10, edgecolors="none")
            )
            self._register_layer("Standardized SF surface", [sf_std_line, sf_std_face])
            all_points.append(sf_surface_rescaled)

        if pivot.size == 3:
            all_points.append(pivot.reshape(1, 3))
            pivot_pt = self.ax.scatter(
                [pivot[0]], [pivot[1]], [pivot[2]],
                color="tab:red", s=68, edgecolors="white", linewidths=0.7, depthshade=False,
            )
            pivot_text = self.ax.text(pivot[0], pivot[1], pivot[2], " Target Point", color="tab:red", fontsize=9)
            self._register_layer("Pivot", [pivot_pt, pivot_text])
            interactive.append({"label": scene_meta.get("pivot_name", "Pivot"), "xyz": pivot, "kind": "pivot"})

        if entry.ndim == 2 and entry.shape[1] == 3:
            all_points.append(entry)
            for i, pt in enumerate(entry[:4]):
                label = entry_names[i] if i < len(entry_names) else f"Entry {i + 1}"
                interactive.append({"label": label, "xyz": pt, "kind": "entry"})
            entry_pts_artist = self.ax.scatter(
                entry[:, 0], entry[:, 1], entry[:, 2],
                color="#2563eb", s=18, alpha=0.0, edgecolors="none", depthshade=False,
            )
            self._register_layer("Entry points", entry_pts_artist)

        if rescaled.ndim == 2 and rescaled.shape[1] == 3 and len(rescaled) >= 4:
            all_points.append(rescaled)
            projected_pts = self.ax.scatter(
                rescaled[:, 0], rescaled[:, 1], rescaled[:, 2],
                color="tab:orange", s=42, edgecolors="white", linewidths=0.55, depthshade=False, label="Projected points",
            )
            projection_artists: List[Any] = [projected_pts]
            for original, projected in zip(entry[:4], rescaled[:4]):
                projection_artists.extend(self.ax.plot(
                    [original[0], projected[0]],
                    [original[1], projected[1]],
                    [original[2], projected[2]],
                    color="black",
                    linestyle=":",
                    linewidth=1.15,
                    alpha=0.85,
                ))
            for tri in (vertical_triangle_rescaled, horizontal_triangle_rescaled):
                if tri.ndim == 2 and tri.shape == (3, 3):
                    closed = np.vstack([tri, tri[0]])
                    projection_artists.extend(self.ax.plot(
                        closed[:, 0], closed[:, 1], closed[:, 2],
                        color="black", linewidth=1.3, alpha=0.82, linestyle=":", label="_nolegend_",
                    ))
            self._register_layer("Standardization projection", projection_artists)

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title(
            f"AoA(v) {float(payload.get('aoa_vertical_deg', 0.0)):.2f}°  |  "
            f"AoA(h) {float(payload.get('aoa_horizontal_deg', 0.0)):.2f}°  |  "
            f"SF {float(payload.get('sf_entry_area_mm2', 0.0)):.2f} mm²",
            fontsize=11,
            color="#233044",
            pad=12,
        )
        annotation = (
            f"V-AoA γ: {float(vertical_props.get('angles_deg', {}).get('gamma', payload.get('aoa_vertical_deg', 0.0))):.2f}°\n"
            f"H-AoA γ: {float(horizontal_props.get('angles_deg', {}).get('gamma', payload.get('aoa_horizontal_deg', 0.0))):.2f}°"
        )
        self.ax.text2D(
            0.03,
            0.95,
            annotation,
            transform=self.ax.transAxes,
            fontsize=10.5,
            color="#c81e1e",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#d5d9e2", alpha=0.92),
        )
        self._set_interactive_points(interactive)
        self._set_axes_from_points(all_points + self._scene_point_arrays(scene_meta))

    def _render_aoe_scene(self, title: str, payload: dict) -> None:
        angle = max(0.0, min(float(payload.get("aoe_deg", 0.0)), 360.0))
        remaining = max(0.0, 360.0 - angle)
        self.clear(title, projection="polar")
        self.ax.set_theta_zero_location("N")
        self.ax.set_theta_direction(-1)
        self.ax.set_ylim(0, 1)
        self.ax.set_yticks([])
        self.ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
        self.ax.bar(
            x=np.deg2rad(angle / 2.0),
            height=1.0,
            width=np.deg2rad(angle),
            bottom=0.0,
            color="tab:green",
            alpha=0.88,
            edgecolor="white",
            linewidth=1.0,
        )
        if remaining > 0:
            self.ax.bar(
                x=np.deg2rad(angle + remaining / 2.0),
                height=1.0,
                width=np.deg2rad(remaining),
                bottom=0.0,
                color="0.87",
                alpha=0.78,
                edgecolor="white",
                linewidth=0.8,
            )
        self.ax.text(
            0.0, 0.0, f"AoE\n{angle:.2f}°",
            ha="center", va="center", fontsize=13, fontweight="bold", color="#223045",
        )
        self.ax.set_title("AoE · Angle of Exposure", va="bottom", color="#233044", pad=18)

    def _render_distance_scene(self, title: str, payload: dict, scene_meta: dict) -> None:
        self.clear(title)
        self._render_ai_scene_background(scene_meta, [])
        dbg = payload.get("debug", {}) if isinstance(payload.get("debug"), dict) else {}
        A = np.asarray(dbg.get("A", []), dtype=float).reshape(-1)
        B = np.asarray(dbg.get("B", []), dtype=float).reshape(-1)
        if A.size != 3 or B.size != 3:
            raise ValueError("Distance plot requires debug geometry.")

        pts = np.vstack([A, B])
        if self._viewer_mode == "pyvista" and self.pv_plotter is not None:
            self._register_layer("Distance segment", [
                self._pv_polyline(pts, "tab:cyan", width=self._metric_width(3.0), closed=False, opacity=self._metric_alpha(0.95), name="distance_segment"),
                self._pv_points(pts, color="tab:cyan", size=self._metric_size(16.0), opacity=self._metric_alpha(0.98), name="distance_points"),
            ])
            self._set_interactive_points(
                [
                    {"label": scene_meta.get("A_name", "Point A"), "xyz": A, "kind": "distance"},
                    {"label": scene_meta.get("B_name", "Point B"), "xyz": B, "kind": "distance"},
                ]
            )
            self._title.setText(f"Distance {float(payload.get('distance_mm', 0.0)):.2f} mm")
            self._status.setText("Click a point in the scene or use the Metric Points list.")
            self._pv_set_bounds_from_points([pts] + self._scene_point_arrays(scene_meta))
            return
        dist_line = self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="tab:cyan", linewidth=2)
        dist_pts = self.ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            color="tab:cyan", s=50, edgecolors="white", linewidths=0.55, depthshade=False,
        )
        self._register_layer("Distance segment", [dist_line, dist_pts])
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title(f"Distance {float(payload.get('distance_mm', 0.0)):.2f} mm", fontsize=11, color="#233044", pad=12)
        self._set_interactive_points(
            [
                {"label": scene_meta.get("A_name", "Point A"), "xyz": A, "kind": "distance"},
                {"label": scene_meta.get("B_name", "Point B"), "xyz": B, "kind": "distance"},
            ]
        )
        self._set_axes_from_points([pts] + self._scene_point_arrays(scene_meta))

    def _render_area_scene(self, title: str, payload: dict, scene_meta: dict) -> None:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        self.clear(title)
        self._render_ai_scene_background(scene_meta, [])
        dbg = payload.get("debug", {}) if isinstance(payload.get("debug"), dict) else {}
        poly = np.asarray(dbg.get("polygon", []), dtype=float)
        coplanar = np.asarray(dbg.get("coplanar_polygon_3d", []), dtype=float)
        if poly.ndim != 2 or poly.shape[1] != 3 or len(poly) < 3:
            raise ValueError("Area plot requires debug geometry.")
        if self._viewer_mode == "pyvista" and self.pv_plotter is not None:
            self._register_layer("Original polygon", [
                self._pv_polyline(poly, "tab:orange", width=self._metric_width(3.0), closed=True, opacity=self._metric_alpha(0.92), name="area_polygon_line"),
                self._pv_points(poly, color="tab:orange", size=self._metric_size(15.0), opacity=self._metric_alpha(0.96), name="area_polygon_points"),
            ])
            all_points = [poly]
            if coplanar.ndim == 2 and coplanar.shape == poly.shape:
                self._register_layer("PCA fitted coplanar surface", [
                    self._pv_polyline(coplanar, "#0f172a", width=self._metric_width(2.5), closed=True, opacity=self._metric_alpha(0.75), name="area_fit_line"),
                    self._pv_polygon_surface(coplanar, color="#94a3b8", opacity=self._metric_alpha(0.16), name="area_fit_surface"),
                ])
                all_points.append(coplanar)
            polygon_names = list(scene_meta.get("polygon_names", []) or [])
            self._set_interactive_points(
                [
                    {"label": polygon_names[i] if i < len(polygon_names) else f"Polygon {i + 1}", "xyz": pt, "kind": "area"}
                    for i, pt in enumerate(poly)
                ]
            )
            self._title.setText(f"Area {float(payload.get('area_mm2', 0.0)):.2f} mm²")
            self._status.setText("Click a point in the scene or use the Metric Points list.")
            self._pv_set_bounds_from_points(all_points + self._scene_point_arrays(scene_meta))
            return

        closed = np.vstack([poly, poly[0]])
        poly_line = self.ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color="tab:orange", linewidth=2, alpha=0.92, label="Original polygon")
        poly_pts = self.ax.scatter(
            poly[:, 0], poly[:, 1], poly[:, 2],
            color="tab:orange", s=46, edgecolors="white", linewidths=0.55, depthshade=False,
        )
        self._register_layer("Original polygon", [poly_line, poly_pts])
        all_points = [poly]

        if coplanar.ndim == 2 and coplanar.shape == poly.shape:
            coplanar_closed = np.vstack([coplanar, coplanar[0]])
            fit_line = self.ax.plot(
                coplanar_closed[:, 0], coplanar_closed[:, 1], coplanar_closed[:, 2],
                color="#0f172a", linewidth=1.8, alpha=0.75, label="PCA fitted coplanar surface",
            )
            fit_face = self.ax.add_collection3d(
                Poly3DCollection([coplanar], facecolors="#94a3b8", alpha=0.16, edgecolors="none")
            )
            self._register_layer("PCA fitted coplanar surface", [fit_line, fit_face])
            all_points.append(coplanar)

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title(f"Area {float(payload.get('area_mm2', 0.0)):.2f} mm²", fontsize=11, color="#233044", pad=12)
        polygon_names = list(scene_meta.get("polygon_names", []) or [])
        self._set_interactive_points(
            [
                {"label": polygon_names[i] if i < len(polygon_names) else f"Polygon {i + 1}", "xyz": pt, "kind": "area"}
                for i, pt in enumerate(poly)
            ]
        )
        self._set_axes_from_points(all_points + self._scene_point_arrays(scene_meta))

    def _render_volume_scene(self, title: str, payload: dict, scene_meta: dict) -> None:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        self.clear(title)
        self._render_ai_scene_background(scene_meta, [])
        dbg = payload.get("debug", {}) if isinstance(payload.get("debug"), dict) else {}
        points = np.asarray(dbg.get("points", []), dtype=float)
        surface = np.asarray(dbg.get("surface_triangles", []), dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
            raise ValueError("Volume plot requires debug geometry.")
        if self._viewer_mode == "pyvista" and self.pv_plotter is not None:
            self._register_layer("Input points", self._pv_points(points, color="#2563eb", size=self._metric_size(16.0), opacity=self._metric_alpha(0.98), name="volume_input_points"))
            all_points = [points]
            if surface.ndim == 3 and surface.shape[-1] == 3 and len(surface) > 0:
                self._register_layer("Reconstructed surface", self._pv_surface(surface, color="#94a3b8", opacity=self._metric_alpha(0.22), show_edges=True, edge_color="#334155", line_width=self._metric_width(0.4), name="volume_surface"))
                all_points.append(surface.reshape(-1, 3))
            point_names = list(scene_meta.get("point_names", []) or [])
            self._set_interactive_points(
                [
                    {"label": point_names[i] if i < len(point_names) else f"Point {i + 1}", "xyz": pt, "kind": "volume"}
                    for i, pt in enumerate(points)
                ]
            )
            self._title.setText(f"Volume {float(payload.get('volume_mm3', 0.0)):.2f} mm³")
            self._status.setText("Click a point in the scene or use the Metric Points list.")
            self._pv_set_bounds_from_points(all_points + self._scene_point_arrays(scene_meta))
            return

        vol_pts = self.ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            color="#2563eb", s=50, edgecolors="white", linewidths=0.6, depthshade=False, label="Input points",
        )
        self._register_layer("Input points", vol_pts)
        all_points = [points]

        if surface.ndim == 3 and surface.shape[-1] == 3 and len(surface) > 0:
            vol_surface = self.ax.add_collection3d(
                Poly3DCollection(
                    list(surface),
                    facecolors="#94a3b8",
                    alpha=0.22,
                    edgecolors=(0.20, 0.25, 0.31, 0.28),
                    linewidths=0.35,
                )
            )
            self._register_layer("Reconstructed surface", vol_surface)
            all_points.append(surface.reshape(-1, 3))

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title(
            f"Volume {float(payload.get('volume_mm3', 0.0)):.2f} mm³",
            fontsize=11,
            color="#233044",
            pad=12,
        )
        point_names = list(scene_meta.get("point_names", []) or [])
        self._set_interactive_points(
            [
                {"label": point_names[i] if i < len(point_names) else f"Point {i + 1}", "xyz": pt, "kind": "volume"}
                for i, pt in enumerate(points)
            ]
        )
        self._set_axes_from_points(all_points + self._scene_point_arrays(scene_meta))

    def _set_axes_from_points(self, arrays: List[np.ndarray]) -> None:
        valid = []
        for arr in arrays:
            arr = np.asarray(arr, dtype=float)
            if arr.ndim == 2 and arr.shape[1] == 3 and len(arr) > 0:
                valid.append(arr)
        if valid:
            self._set_axes_equal_from_points(np.vstack(valid))

    def _set_axes_equal_from_points(self, points: np.ndarray) -> None:
        pts = np.asarray(points, dtype=float)
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        mids = (mins + maxs) * 0.5
        spans = np.maximum(maxs - mins, 1.0)
        radius = 0.6 * float(np.max(spans))
        self.ax.set_xlim3d([mids[0] - radius, mids[0] + radius])
        self.ax.set_ylim3d([mids[1] - radius, mids[1] + radius])
        self.ax.set_zlim3d([mids[2] - radius, mids[2] + radius])

    def _nearest_interactive_point(self, event) -> Optional[dict]:
        if self._projection != "3d" or not self._interactive_points or event.x is None or event.y is None:
            return None
        try:
            from mpl_toolkits.mplot3d import proj3d
        except Exception:
            return None

        projected: List[tuple[float, dict]] = []
        for point in self._interactive_points:
            xyz = point["xyz"]
            x2, y2, _ = proj3d.proj_transform(xyz[0], xyz[1], xyz[2], self.ax.get_proj())
            sx, sy = self.ax.transData.transform((x2, y2))
            dist = float(np.hypot(sx - event.x, sy - event.y))
            projected.append((dist, point))
        if not projected:
            return None
        dist, point = min(projected, key=lambda item: item[0])
        if dist > 14.0:
            return None
        return point

    def _set_selected_point(self, point: Optional[dict]) -> None:
        self._selected_point = point
        if self._projection != "3d" or self.ax is None:
            return
        if self._selection_artist is not None:
            try:
                self._selection_artist.remove()
            except Exception:
                pass
            self._selection_artist = None
        if point is None:
            self.canvas.draw_idle()
            return
        xyz = point["xyz"]
        self._selection_artist = self.ax.scatter(
            [xyz[0]], [xyz[1]], [xyz[2]],
            s=150,
            facecolors="none",
            edgecolors="#0f172a",
            linewidths=1.4,
            depthshade=False,
        )
        self.canvas.draw_idle()

    def _describe_point(self, point: dict, prefix: str) -> None:
        xyz = _as_xyz(point["xyz"])
        self._status.setText(
            f"{prefix}: {point['label']}  |  x={xyz[0]:.3f}, y={xyz[1]:.3f}, z={xyz[2]:.3f}"
        )

    def _on_mouse_move(self, event) -> None:
        point = self._nearest_interactive_point(event)
        if point is None:
            if self._selected_point is None:
                self._status.setText("Hover over a point to inspect its coordinates. Click to pin the selection.")
            return
        self._describe_point(point, "Hover")

    def _on_mouse_click(self, event) -> None:
        point = self._nearest_interactive_point(event)
        if point is None:
            if self._selected_point is not None:
                self._set_selected_point(None)
            self._status.setText("Hover over a point to inspect its coordinates. Click to pin the selection.")
            return
        self._set_selected_point(point)
        self._describe_point(point, "Selected")

    def reset_view(self) -> None:
        if self._projection != "3d":
            return
        if self._viewer_mode == "pyvista" and self.pv_plotter is not None:
            try:
                self.pv_plotter.view_isometric()
                self.pv_plotter.reset_camera()
                self.pv_plotter.render()
            except Exception:
                pass
            return
        if not HAS_MPL or self.ax is None:
            return
        if self._projection == "3d":
            self.ax.view_init(elev=18, azim=-58)
            self.canvas.draw_idle()

    def zoom_view(self, factor: float) -> None:
        if self._projection != "3d":
            return
        if self._viewer_mode == "pyvista" and self.pv_plotter is not None:
            try:
                self.pv_plotter.camera.zoom(1.0 / float(factor))
                self.pv_plotter.render()
            except Exception:
                pass
            return
        if not HAS_MPL or self.ax is None:
            return
        for getter, setter in (
            (self.ax.get_xlim3d, self.ax.set_xlim3d),
            (self.ax.get_ylim3d, self.ax.set_ylim3d),
            (self.ax.get_zlim3d, self.ax.set_zlim3d),
        ):
            lo, hi = [float(v) for v in getter()]
            mid = 0.5 * (lo + hi)
            radius = 0.5 * (hi - lo) * float(factor)
            setter([mid - radius, mid + radius])
            self.canvas.draw_idle()

    def export_png(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export plot as PNG",
            "surgiplot-plot.png",
            "PNG image (*.png);;All files (*)",
        )
        if not path:
            return
        if self._viewer_mode == "pyvista" and self.pv_plotter is not None:
            self.pv_plotter.screenshot(path)
        elif HAS_MPL and self.fig is not None:
            self.fig.savefig(path, dpi=220, facecolor=self.fig.get_facecolor(), bbox_inches="tight")
        else:
            return
        QtWidgets.QMessageBox.information(self, "Plot exported", f"Saved plot image:\n{path}")

    def export_scene(self) -> None:
        if self._scene_payload is None:
            QtWidgets.QMessageBox.information(self, "No scene", "Create a plot first.")
            return
        if self._projection != "3d":
            QtWidgets.QMessageBox.information(self, "2D plot", "3D scene export is only available for 3D plots.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export 3D scene",
            "surgiplot-scene.json",
            "Surgiplot Scene (*.json);;All files (*)",
        )
        if not path:
            return
        scene = dict(self._scene_payload)
        scene["view"] = self._current_view_state()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(scene, f, indent=2, sort_keys=True)
        QtWidgets.QMessageBox.information(self, "Scene exported", f"Saved interactive 3D scene:\n{path}")

    def open_scene(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open saved 3D scene",
            "",
            "Surgiplot Scene (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                scene = json.load(f)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Open failed", f"{e}\n\n{_exc_text()}")
            return
        self._render_scene(scene)


# =============================================================================
# Dataset panel
# =============================================================================
class DatasetPanel(QtWidgets.QWidget):
    dataset_changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.ds = None
        self._table_refreshing = False

        self.lbl_path = QtWidgets.QLabel("No dataset loaded.")
        self.lbl_path.setWordWrap(True)

        self.btn_import_nav = QtWidgets.QPushButton("Import navigation file…")
        self.btn_import_db = QtWidgets.QPushButton("Import formatted database…")
        self.btn_import_generic = QtWidgets.QPushButton("Import generic point file…")
        self.btn_save_points = QtWidgets.QPushButton("Export points table…")
        self.btn_clear_points = QtWidgets.QPushButton("Clear all points")

        self.btn_add_point = QtWidgets.QPushButton("Add / update point…")
        self.btn_del_point = QtWidgets.QPushButton("Delete selected point")

        self.btn_ai_scene = QtWidgets.QPushButton("3D Scene Workspace…")
        self.btn_ai_scene.setEnabled(HAS_AI_SCENE)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Point", "Labels", "X", "Y", "Z"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.itemChanged.connect(self._on_table_item_changed)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.btn_import_nav)
        top.addWidget(self.btn_import_db)
        top.addWidget(self.btn_import_generic)
        top.addWidget(self.btn_ai_scene)
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

        self.btn_import_nav.clicked.connect(self._on_import_navigation)
        self.btn_import_db.clicked.connect(self._on_import_database)
        self.btn_import_generic.clicked.connect(self._on_import_generic_points)
        self.btn_save_points.clicked.connect(self._on_export_points)
        self.btn_clear_points.clicked.connect(self._on_clear_points)
        self.btn_add_point.clicked.connect(self._on_add_point)
        self.btn_del_point.clicked.connect(self._on_delete_selected)
        self.btn_ai_scene.clicked.connect(self._on_ai_scene_reconstruction)

        self._refresh()

    def _choose_navigation_format(self) -> Optional[str]:
        items = ["auto", "stryker", "medtronic"]
        fmt, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Navigation format",
            "Choose import format:",
            items,
            0,
            False,
        )
        if not ok:
            return None
        return str(fmt).strip().lower()

    def _review_imported_dataset(self, imported_ds, default_source: str) -> Optional[tuple[list[dict], str]]:
        rows = _dataset_to_rows(imported_ds)
        return run_label_editor(rows=rows, source=default_source, parent=self)

    def _on_import_navigation(self):
        fmt = self._choose_navigation_format()
        if fmt is None:
            return

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import neuronavigation file",
            "",
            "Navigation files (*.txt *.dat *.json *.csv *.tsv);;All files (*)",
        )
        if not path:
            return

        try:
            imported_ds = load_navigation_file(path, navigation_format=fmt)
            reviewed = self._review_imported_dataset(imported_ds, default_source="navigation")
            if reviewed is None:
                return

            rows, source = reviewed
            if self.ds is None:
                self.ds = imported_ds
                _apply_rows_to_dataset(self.ds, rows, source=source)
            else:
                _merge_rows_into_dataset(self.ds, rows, source=source)

            if isinstance(getattr(self.ds, "meta", None), dict):
                self.ds.meta["loaded_from"] = path
                self.ds.meta["last_import_kind"] = "navigation"

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Navigation import failed", f"{e}\n\n{_exc_text()}")
            return

        self._refresh()
        self.dataset_changed.emit()

    def _on_import_database(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import formatted database",
            "",
            "Database files (*.xlsx *.xls *.csv *.tsv);;All files (*)",
        )
        if not path:
            return

        try:
            imported_ds = load_dataset(path, kind="database", source="database")
            reviewed = self._review_imported_dataset(imported_ds, default_source="database")
            if reviewed is None:
                return

            rows, source = reviewed
            self.ds = imported_ds
            _apply_rows_to_dataset(self.ds, rows, source=source)

            if isinstance(getattr(self.ds, "meta", None), dict):
                self.ds.meta["loaded_from"] = path
                self.ds.meta["last_import_kind"] = "database"

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Database import failed", f"{e}\n\n{_exc_text()}")
            return

        self._refresh()
        self.dataset_changed.emit()

    def _on_import_generic_points(self):
        if self.ds is None:
            QtWidgets.QMessageBox.warning(self, "No dataset", "Load or import a dataset first.")
            return

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import generic point file",
            "",
            "Point files (*.csv *.tsv *.txt);;All files (*)",
        )
        if not path:
            return

        try:
            imported_ds = load_dataset(path, kind="generic", source="navigation")
            reviewed = self._review_imported_dataset(imported_ds, default_source="navigation")
            if reviewed is None:
                return

            rows, source = reviewed
            _merge_rows_into_dataset(self.ds, rows, source=source)

            if isinstance(getattr(self.ds, "meta", None), dict):
                self.ds.meta["last_import_path"] = path
                self.ds.meta["last_import_kind"] = "generic"

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Generic point import failed", f"{e}\n\n{_exc_text()}")
            return

        self._refresh()
        self.dataset_changed.emit()

    def _on_export_points(self):
        if self.ds is None:
            QtWidgets.QMessageBox.warning(self, "No dataset", "No dataset loaded.")
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
                w.writerow(["name", "x", "y", "z"])
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
        _ensure_aliases_dict(self.ds).clear()
        self._refresh()
        self.dataset_changed.emit()

    def _on_add_point(self):
        if self.ds is None:
            QtWidgets.QMessageBox.warning(self, "No dataset", "Load or import a dataset first.")
            return

        label, ok = QtWidgets.QInputDialog.getText(self, "Point label", "Label:")
        if not ok:
            return
        label = (label or "").strip()
        if not label:
            return

        xyz_text, ok = QtWidgets.QInputDialog.getText(self, "Point coordinates", "Enter x,y,z (comma-separated):")
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

        pts = _ensure_points_dict(self.ds)
        aliases = _ensure_aliases_dict(self.ds)
        if label in aliases and label not in pts:
            QtWidgets.QMessageBox.warning(
                self,
                "Alias conflict",
                f"'{label}' is currently used as an alias for '{aliases[label]}'. "
                "Choose a different canonical point name or edit the labels first.",
            )
            return

        pts[label] = p
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
            self, "Delete point", f"Delete point '{lb}'?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if resp != QtWidgets.QMessageBox.Yes:
            return
        _ensure_points_dict(self.ds).pop(lb, None)

        aliases = _ensure_aliases_dict(self.ds)
        for alias in list(aliases.keys()):
            if aliases[alias] == lb:
                aliases.pop(alias, None)

        self._refresh()
        self.dataset_changed.emit()

    def _on_ai_scene_reconstruction(self):
        if not HAS_AI_SCENE:
            QtWidgets.QMessageBox.information(
                self,
                "Feature not available",
                "The 3D scene workspace is not available.\n"
                "Ensure `surgiplot.ai.scene_reconstruction` exists and optional dependencies are installed.",
            )
            return
        if self.ds is None:
            self.ds = Dataset(meta={"source": "scene_workspace"})
        dlg = AISceneReconstructionDialog(self.ds, on_dataset_changed_callback=self._on_ai_points_committed, parent=self)
        dlg.exec()

    def _on_ai_points_committed(self):
        self._refresh()
        self.dataset_changed.emit()

    def _on_table_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._table_refreshing or self.ds is None or item is None:
            return
        if item.column() != 1:
            return

        canonical_item = self.table.item(item.row(), 0)
        if canonical_item is None:
            return
        canonical = canonical_item.text().strip()
        if not canonical:
            return

        pts = _ensure_points_dict(self.ds)
        aliases = _ensure_aliases_dict(self.ds)
        if canonical not in pts:
            return

        requested = []
        for part in item.text().split(","):
            alias = part.strip()
            if alias and alias not in requested:
                requested.append(alias)

        conflicts: List[str] = []
        clean_aliases: List[str] = []
        for alias in requested:
            if alias == canonical:
                continue
            if alias in pts and alias != canonical:
                conflicts.append(alias)
                continue
            owner = aliases.get(alias)
            if owner is not None and owner != canonical:
                conflicts.append(alias)
                continue
            clean_aliases.append(alias)

        for alias in list(aliases.keys()):
            if aliases[alias] == canonical:
                aliases.pop(alias, None)
        for alias in clean_aliases:
            aliases[alias] = canonical

        self._refresh()
        self.dataset_changed.emit()

        if conflicts:
            QtWidgets.QMessageBox.warning(
                self,
                "Alias conflict",
                "Some labels were not applied because they are already used elsewhere:\n"
                + ", ".join(conflicts),
            )

    def _refresh(self):
        if self.ds is None:
            self.lbl_path.setText("No dataset loaded.")
            self.table.setRowCount(0)
            return

        self._table_refreshing = True

        path = ""
        if isinstance(getattr(self.ds, "meta", None), dict):
            path = str(self.ds.meta.get("loaded_from", ""))

        if not path:
            path = str(getattr(self.ds, "path", "") or "")

        vendor = ""
        source = ""
        if isinstance(getattr(self.ds, "meta", None), dict):
            vendor = str(self.ds.meta.get("vendor", "") or "")
            source = str(self.ds.meta.get("source", "") or "")

        hdr = "Loaded dataset"
        if vendor:
            hdr += f" | vendor: {vendor}"
        if source:
            hdr += f" | source: {source}"
        if path:
            hdr += f"\n{path}"
        self.lbl_path.setText(hdr)

        pts = _ensure_points_dict(self.ds)
        aliases = _ensure_aliases_dict(self.ds)
        labels = sorted(pts.keys())
        self.table.setRowCount(len(labels))
        reverse_aliases: Dict[str, List[str]] = {}
        for alias, canonical in aliases.items():
            reverse_aliases.setdefault(str(canonical), []).append(str(alias))

        for r, lb in enumerate(labels):
            p = _as_xyz(pts[lb])
            label_text = ", ".join(sorted(reverse_aliases.get(lb, [])))
            it0 = QtWidgets.QTableWidgetItem(lb)
            it1 = QtWidgets.QTableWidgetItem(label_text)
            it2 = QtWidgets.QTableWidgetItem(f"{float(p[0]):.6f}")
            it3 = QtWidgets.QTableWidgetItem(f"{float(p[1]):.6f}")
            it4 = QtWidgets.QTableWidgetItem(f"{float(p[2]):.6f}")
            for it in (it0, it2, it3, it4):
                it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(r, 0, it0)
            self.table.setItem(r, 1, it1)
            self.table.setItem(r, 2, it2)
            self.table.setItem(r, 3, it3)
            self.table.setItem(r, 4, it4)
        self._table_refreshing = False


# =============================================================================
# Metric panel
# =============================================================================
class MetricPanel(QtWidgets.QWidget):
    analysis_updated = QtCore.Signal(str, str, str, object)

    def __init__(self, dataset_panel: DatasetPanel, plot_panel: PlotPanel, parent=None):
        super().__init__(parent)
        self.dataset_panel = dataset_panel
        self.plot_panel = plot_panel

        self.cmb_metric = QtWidgets.QComboBox()
        self.cmb_metric.addItems(["AoA / SF", "VOM / VoA", "AoE", "Distance", "Area", "Volume", "Dataset overview"])

        self.btn_compute = QtWidgets.QPushButton("Compute")
        self.btn_plot_points = QtWidgets.QPushButton("Plot Dataset")
        self.btn_copy_json = QtWidgets.QPushButton("Copy Result JSON")

        self._last_result_payload = None
        self._last_debug_text = ""

        self._point_selectors: List[QtWidgets.QComboBox] = []
        self._name_list_edits: List[QtWidgets.QLineEdit] = []
        self._point_combo_buttons: Dict[QtWidgets.QComboBox, QtWidgets.QToolButton] = {}
        self._name_edit_buttons: Dict[QtWidgets.QLineEdit, QtWidgets.QToolButton] = {}

        self.metric_forms = QtWidgets.QStackedWidget()
        self.metric_form_indexes: Dict[str, int] = {}

        self.form_plot_only = QtWidgets.QLabel(
            "Display the current dataset in 3D without running a quantitative analysis."
        )
        self.form_plot_only.setWordWrap(True)
        self.metric_form_indexes["Dataset overview"] = self.metric_forms.addWidget(self.form_plot_only)

        self.form_aoa = QtWidgets.QWidget()
        aoa_layout = QtWidgets.QFormLayout(self.form_aoa)
        self.cmb_cranial = self._make_point_combo()
        self.cmb_caudal = self._make_point_combo()
        self.cmb_medial = self._make_point_combo()
        self.cmb_lateral = self._make_point_combo()
        self.cmb_pivot = self._make_point_combo()
        self.btn_toggle_standardization = QtWidgets.QPushButton("Standardization")
        self.btn_toggle_standardization.setCheckable(True)
        self.spin_sf_radius_cm = QtWidgets.QDoubleSpinBox()
        self.spin_sf_radius_cm.setRange(0.01, 1000.0)
        self.spin_sf_radius_cm.setDecimals(2)
        self.spin_sf_radius_cm.setValue(20.00)
        self.spin_sf_radius_cm.setSuffix(" cm")
        self.spin_sf_radius_cm.hide()
        aoa_layout.addRow("Cranial:", self._point_field_widget(self.cmb_cranial))
        aoa_layout.addRow("Caudal:", self._point_field_widget(self.cmb_caudal))
        aoa_layout.addRow("Medial:", self._point_field_widget(self.cmb_medial))
        aoa_layout.addRow("Lateral:", self._point_field_widget(self.cmb_lateral))
        aoa_layout.addRow("Pivot:", self._point_field_widget(self.cmb_pivot))
        aoa_layout.addRow("Standardization:", self.btn_toggle_standardization)
        aoa_layout.addRow("Distance:", self.spin_sf_radius_cm)
        self.metric_form_indexes["AoA / SF"] = self.metric_forms.addWidget(self.form_aoa)

        self.form_vom = QtWidgets.QWidget()
        vom_layout = QtWidgets.QFormLayout(self.form_vom)
        self.edit_vom_entry = self._make_name_list_edit(
            "Enter entry polygon labels, separated by commas"
        )
        self.edit_vom_target = self._make_name_list_edit(
            "Enter target polygon labels, separated by commas"
        )
        self.spin_stand_dist = QtWidgets.QDoubleSpinBox()
        self.spin_stand_dist.setRange(0.001, 1000.0)
        self.spin_stand_dist.setDecimals(3)
        self.spin_stand_dist.setValue(10.0)
        self.spin_stand_dist.setSuffix(" mm")
        vom_layout.addRow("Entry Polygon:", self._name_list_field_widget(self.edit_vom_entry))
        vom_layout.addRow("Target Polygon:", self._name_list_field_widget(self.edit_vom_target))
        vom_layout.addRow("Standard Distance:", self.spin_stand_dist)
        self.metric_form_indexes["VOM / VoA"] = self.metric_forms.addWidget(self.form_vom)

        self.form_aoe = QtWidgets.QWidget()
        aoe_layout = QtWidgets.QFormLayout(self.form_aoe)
        self.cmb_aoe_a = self._make_point_combo()
        self.cmb_aoe_b = self._make_point_combo()
        self.cmb_aoe_c = self._make_point_combo()
        aoe_layout.addRow("Point A:", self._point_field_widget(self.cmb_aoe_a))
        aoe_layout.addRow("Pivot B:", self._point_field_widget(self.cmb_aoe_b))
        aoe_layout.addRow("Point C:", self._point_field_widget(self.cmb_aoe_c))
        self.metric_form_indexes["AoE"] = self.metric_forms.addWidget(self.form_aoe)

        self.form_distance = QtWidgets.QWidget()
        distance_layout = QtWidgets.QFormLayout(self.form_distance)
        self.cmb_distance_a = self._make_point_combo()
        self.cmb_distance_b = self._make_point_combo()
        distance_layout.addRow("Point A:", self._point_field_widget(self.cmb_distance_a))
        distance_layout.addRow("Point B:", self._point_field_widget(self.cmb_distance_b))
        self.metric_form_indexes["Distance"] = self.metric_forms.addWidget(self.form_distance)

        self.form_area = QtWidgets.QWidget()
        area_layout = QtWidgets.QFormLayout(self.form_area)
        self.edit_area_polygon = self._make_name_list_edit(
            "Enter polygon labels, separated by commas"
        )
        area_layout.addRow("Polygon:", self._name_list_field_widget(self.edit_area_polygon))
        self.metric_form_indexes["Area"] = self.metric_forms.addWidget(self.form_area)

        self.form_volume = QtWidgets.QWidget()
        volume_layout = QtWidgets.QFormLayout(self.form_volume)
        self.edit_volume_points = self._make_name_list_edit(
            "Enter boundary point labels, separated by commas"
        )
        volume_layout.addRow("Boundary Points:", self._name_list_field_widget(self.edit_volume_points))
        self.metric_form_indexes["Volume"] = self.metric_forms.addWidget(self.form_volume)

        lay = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Metric:"))
        top.addWidget(self.cmb_metric, 1)
        top.addWidget(self.btn_plot_points)
        top.addWidget(self.btn_compute)
        top.addStretch(1)
        top.addWidget(self.btn_copy_json)
        lay.addLayout(top)
        lay.addWidget(self.metric_forms)

        self.txt_out = QtWidgets.QPlainTextEdit()
        self.txt_out.setReadOnly(True)
        self.txt_out.setPlaceholderText("Quantitative results and methodological notes will appear here.")
        self.txt_out.setMinimumHeight(280)
        lay.addWidget(self.txt_out, 1)

        self.btn_compute.clicked.connect(self.compute_current)
        self.btn_plot_points.clicked.connect(self.plot_points)
        self.btn_copy_json.clicked.connect(self.copy_result_json)
        self.btn_toggle_standardization.toggled.connect(self._sync_standardization_state)
        self.cmb_metric.currentTextChanged.connect(self._sync_metric_form)
        self.dataset_panel.dataset_changed.connect(self.refresh_point_selectors)

        self._sync_metric_form()
        self._sync_standardization_state()
        self.refresh_point_selectors()

    def _make_point_combo(self) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox()
        combo.setEditable(True)
        combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(16)
        self._point_selectors.append(combo)
        return combo

    def _make_name_list_edit(self, placeholder: str) -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit()
        edit.setPlaceholderText(placeholder)
        self._name_list_edits.append(edit)
        return edit

    def _point_field_widget(self, combo: QtWidgets.QComboBox) -> QtWidgets.QWidget:
        wrapper = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(wrapper)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        btn = QtWidgets.QToolButton()
        btn.setText("Select…")
        btn.clicked.connect(lambda: self._pick_single_point(combo))
        self._point_combo_buttons[combo] = btn
        lay.addWidget(combo, 1)
        lay.addWidget(btn)
        return wrapper

    def _name_list_field_widget(self, edit: QtWidgets.QLineEdit) -> QtWidgets.QWidget:
        wrapper = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(wrapper)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        btn = QtWidgets.QToolButton()
        btn.setText("Select…")
        btn.clicked.connect(lambda: self._pick_multiple_points(edit))
        self._name_edit_buttons[edit] = btn
        lay.addWidget(edit, 1)
        lay.addWidget(btn)
        return wrapper

    def _dataset_names(self) -> List[str]:
        ds = self._ds()
        if ds is None:
            return []
        pts = sorted(_ensure_points_dict(ds).keys())
        aliases = sorted(
            a for a, tgt in _ensure_aliases_dict(ds).items()
            if a not in pts and tgt in _ensure_points_dict(ds)
        )
        return pts + aliases

    def refresh_point_selectors(self):
        names = self._dataset_names()

        for combo in self._point_selectors:
            current = combo.currentText().strip()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("")
            combo.addItems(names)
            if current:
                combo.setCurrentText(current)
            combo.blockSignals(False)

        completer = QtWidgets.QCompleter(names, self)
        completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        completer.setFilterMode(QtCore.Qt.MatchContains)
        for edit in self._name_list_edits:
            edit.setCompleter(completer)

    def _sync_metric_form(self):
        metric_name = self.cmb_metric.currentText().strip()
        idx = self.metric_form_indexes.get(metric_name)
        if idx is not None:
            self.metric_forms.setCurrentIndex(idx)

    def _sync_standardization_state(self):
        enabled = bool(self.btn_toggle_standardization.isChecked())
        self.btn_toggle_standardization.setText("Standardization ON" if enabled else "Standardization")
        self.spin_sf_radius_cm.setVisible(enabled)

    def _ds(self):
        return self.dataset_panel.ds

    def _pick_single_point(self, combo: QtWidgets.QComboBox) -> None:
        ds = self._ds()
        if ds is None:
            QtWidgets.QMessageBox.information(self, "No dataset", "Load a dataset before selecting points.")
            return
        dlg = PointPickerDialog(ds, multi_select=False, parent=self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        selected = dlg.selected_points()
        if selected:
            combo.setCurrentText(selected[0])

    def _pick_multiple_points(self, edit: QtWidgets.QLineEdit) -> None:
        ds = self._ds()
        if ds is None:
            QtWidgets.QMessageBox.information(self, "No dataset", "Load a dataset before selecting points.")
            return
        dlg = PointPickerDialog(ds, multi_select=True, parent=self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        selected = dlg.selected_points()
        if not selected:
            return
        existing = [part.strip() for part in edit.text().split(",") if part.strip()]
        merged: List[str] = []
        seen = set()
        for name in existing + selected:
            if name not in seen:
                seen.add(name)
                merged.append(name)
        edit.setText(", ".join(merged))

    def plot_points(self):
        ds = self._ds()
        if ds is None:
            QtWidgets.QMessageBox.warning(self, "No dataset", "Load or import a dataset first.")
            return
        self.plot_panel.plot_points(_ensure_points_dict(ds), title="Dataset points", scene_meta=_dataset_scene_meta(ds))
        self.analysis_updated.emit("Dataset points", "Dataset visualization updated.", "", None)

    def compute_current(self):
        ds = self._ds()
        if ds is None:
            QtWidgets.QMessageBox.warning(self, "No dataset", "Load or import a dataset first.")
            return

        metric_name = self.cmb_metric.currentText().strip()

        if metric_name == "Dataset overview":
            self.plot_points()
            return

        try:
            if metric_name == "AoA / SF":
                entry_names = [
                    self._require_text(self.cmb_cranial, "Cranial"),
                    self._require_text(self.cmb_caudal, "Caudal"),
                    self._require_text(self.cmb_medial, "Medial"),
                    self._require_text(self.cmb_lateral, "Lateral"),
                ]
                pivot_name = self._require_text(self.cmb_pivot, "Pivot")
                result = AOA_SF(
                    data=ds,
                    entry=entry_names,
                    target=pivot_name,
                    sf_rescale_radius_mm=self._aoa_standardization_radius_mm(),
                    return_debug=True,
                )
                self._show_result("AOA_SF", result)
                self.plot_panel.plot_metric_result(
                    result,
                    title="AoA / SF",
                    scene_meta={
                        "entry_names": entry_names,
                        "pivot_name": pivot_name,
                        **_dataset_scene_meta(ds),
                    },
                )
                return

            if metric_name == "VOM / VoA":
                entry_names = self._parse_name_list(self.edit_vom_entry, minimum=3)
                target_names = self._parse_name_list(self.edit_vom_target, minimum=3)
                result = VOM_VOA(
                    data=ds,
                    entry=entry_names,
                    target=target_names,
                    stand_dist=float(self.spin_stand_dist.value()),
                    return_debug=True,
                )
                self._show_result("VOM_VOA", result)
                self.plot_panel.plot_metric_result(
                    result,
                    title="VOM / VoA",
                    scene_meta={
                        "entry_names": entry_names,
                        "target_names": target_names,
                        "stand_dist_mm": float(self.spin_stand_dist.value()),
                        **_dataset_scene_meta(ds),
                    },
                )
                return

            if metric_name == "AoE":
                name_a = self._require_text(self.cmb_aoe_a, "Point A")
                name_b = self._require_text(self.cmb_aoe_b, "Pivot B")
                name_c = self._require_text(self.cmb_aoe_c, "Point C")
                result = AOE(
                    data=ds,
                    A=name_a,
                    B=name_b,
                    C=name_c,
                    return_debug=True,
                )
                self._show_result("AOE", result)
                self.plot_panel.plot_metric_result(
                    result,
                    title="AoE",
                    scene_meta={"A_name": name_a, "B_name": name_b, "C_name": name_c, **_dataset_scene_meta(ds)},
                )
                return

            if metric_name == "Distance":
                name_a = self._require_text(self.cmb_distance_a, "Point A")
                name_b = self._require_text(self.cmb_distance_b, "Point B")
                result = DISTANCE_3D(
                    data=ds,
                    A=name_a,
                    B=name_b,
                    return_debug=True,
                )
                self._show_result("DISTANCE_3D", result)
                self.plot_panel.plot_metric_result(
                    result,
                    title="Distance",
                    scene_meta={"A_name": name_a, "B_name": name_b, **_dataset_scene_meta(ds)},
                )
                return

            if metric_name == "Area":
                polygon_names = self._parse_name_list(self.edit_area_polygon, minimum=3)
                result = AREA_3D(
                    data=ds,
                    polygon=polygon_names,
                    return_debug=True,
                )
                self._show_result("AREA_3D", result)
                self.plot_panel.plot_metric_result(
                    result,
                    title="Area",
                    scene_meta={"polygon_names": polygon_names, **_dataset_scene_meta(ds)},
                )
                return

            if metric_name == "Volume":
                point_names = self._parse_name_list(self.edit_volume_points, minimum=4)
                result = VOLUME_3D(
                    data=ds,
                    points=point_names,
                    return_debug=True,
                )
                self._show_result("VOLUME_3D", result)
                self.plot_panel.plot_metric_result(
                    result,
                    title="Volume",
                    scene_meta={"point_names": point_names, **_dataset_scene_meta(ds)},
                )
                return

            QtWidgets.QMessageBox.warning(self, "Unknown metric", f"Unhandled metric: {metric_name}")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Metric computation failed", f"{e}\n\n{_exc_text()}")

    def _require_text(self, widget: QtWidgets.QComboBox, label: str) -> str:
        value = widget.currentText().strip()
        if not value:
            raise ValueError(f"{label} is required.")
        return value

    def _parse_name_list(self, widget: QtWidgets.QLineEdit, minimum: int = 1) -> Optional[List[str]]:
        raw = widget.text().strip()
        names = self._expand_name_tokens(raw)
        if not names:
            if minimum <= 0:
                return None
            raise ValueError("Enter one or more point names or labels, separated by commas.")
        if len(names) < minimum:
            raise ValueError(f"Enter at least {minimum} point names or labels.")
        return names

    def _expand_name_tokens(self, raw: str) -> List[str]:
        normalized = (
            raw.replace("\u2010", "-")
               .replace("\u2011", "-")
               .replace("\u2012", "-")
               .replace("\u2013", "-")
               .replace("\u2014", "-")
               .replace("\u2212", "-")
        )
        tokens = [part.strip() for part in normalized.split(",") if part.strip()]
        expanded: List[str] = []
        for token in tokens:
            expanded.extend(self._expand_token_range(token))
        return expanded

    def _expand_token_range(self, token: str) -> List[str]:
        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", token)
        if not match:
            return [token]
        start = int(match.group(1))
        end = int(match.group(2))
        step = 1 if end >= start else -1
        return [str(i) for i in range(start, end + step, step)]

    def _aoa_standardization_radius_mm(self) -> Optional[float]:
        if not self.btn_toggle_standardization.isChecked():
            return None
        return float(self.spin_sf_radius_cm.value()) * 10.0

    def _show_result(self, title: str, result: Any):
        payload = asdict(result) if is_dataclass(result) else result
        self._last_result_payload = _json_safe(payload)
        result_text = self._build_result_text(title, payload)
        debug_text = self._build_debug_text(payload)
        self._last_debug_text = debug_text
        self.txt_out.setPlainText(result_text)
        self.analysis_updated.emit(title, result_text, debug_text, self._last_result_payload)

    def _build_result_text(self, title: str, payload: Any) -> str:
        if not isinstance(payload, dict):
            return f"{title}\n\n{repr(payload)}"

        title_map = {
            "AOA_SF": "AoA / SF",
            "VOM_VOA": "VoA / VOM / sVOM",
            "AOE": "AoE · Angle of Exposure",
            "DISTANCE_3D": "3D Distance",
            "AREA_3D": "AE · Area of Exposure",
            "VOLUME_3D": "Volume",
        }
        heading = title_map.get(title, title.replace("_", " "))

        if title == "AOA_SF":
            lines = [
                heading,
                "",
                "Primary Measurements",
                f"- Vertical angle of attack: {float(payload.get('aoa_vertical_deg', 0.0)):.2f} deg",
                f"- Horizontal angle of attack: {float(payload.get('aoa_horizontal_deg', 0.0)):.2f} deg",
                f"- Surgical freedom area: {float(payload.get('sf_entry_area_mm2', 0.0)):.2f} mm^2",
            ]
            if payload.get("rescale_radius_mm") is not None:
                lines.append(f"- Standardization distance: {float(payload['rescale_radius_mm']) / 10.0:.2f} cm")
                lines.append(f"- Standardized surgical freedom area: {float(payload.get('sf_entry_area_rescaled_mm2', 0.0)):.2f} mm^2")
            lines.extend([
                "",
                "Interpretation",
                "Attack angles are measured at the selected pivot point.",
                "Surgical freedom is reported as the PCA-fitted coplanar quadrilateral area.",
            ])
            return "\n".join(lines)

        if title == "VOM_VOA":
            return "\n".join([
                heading,
                "",
                "Primary Measurements",
                f"- VoA (Visuooperative Angle): {float(payload.get('voa_deg', 0.0)):.2f} deg",
                f"- VOM (Volume of Operative Maneuverability): {float(payload.get('vom_mm3', 0.0)):.2f} mm^3",
                f"- sVOM (standardized VOM): {float(payload.get('svom_mm3', 0.0)):.2f} mm^3",
                "",
                "Interpretation",
                "Reported volume corresponds to the physical 3D lofted corridor model.",
            ])

        if title == "AOE":
            return "\n".join([
                heading,
                "",
                "Primary Measurement",
                f"- AoE (Angle of Exposure): {float(payload.get('aoe_deg', 0.0)):.2f} deg",
                "",
                "Interpretation",
                "The angle is measured at the selected pivot between the two input rays.",
            ])

        if title == "DISTANCE_3D":
            return "\n".join([
                heading,
                "",
                "Primary Measurement",
                f"- Euclidean distance: {float(payload.get('distance_mm', 0.0)):.2f} mm",
            ])

        if title == "AREA_3D":
            return "\n".join([
                heading,
                "",
                "Primary Measurement",
                f"- AE (Area of Exposure): {float(payload.get('area_mm2', 0.0)):.2f} mm^2",
                "",
                "Method",
                "The polygon is projected to its PCA best-fit plane and measured in 2D using the shoelace formula.",
            ])

        if title == "VOLUME_3D":
            method = "automatic alpha-shape reconstruction"
            dbg = payload.get("debug", {}) if isinstance(payload.get("debug"), dict) else {}
            if isinstance(dbg, dict):
                if dbg.get("reconstruction_method") == "convex_hull":
                    method = "convex hull fallback reconstruction"
            return "\n".join([
                heading,
                "",
                "Primary Measurement",
                f"- Estimated enclosed volume: {float(payload.get('volume_mm3', 0.0)):.2f} mm^3",
                "",
                "Method",
                f"The solid was reconstructed from the sparse boundary points using {method}.",
            ])

        out_lines = [heading, "", "Results"]
        for k in sorted(payload.keys()):
            if k == "debug":
                continue
            v = payload[k]
            if isinstance(v, dict):
                out_lines.append(f"- {k}:")
                out_lines.extend(self._format_mapping(v, indent="  "))
            elif isinstance(v, (float, int, np.floating, np.integer)):
                out_lines.append(f"- {k}: {float(v):.6g}")
            else:
                out_lines.append(f"- {k}: {self._format_result_value(v)}")
        return "\n".join(out_lines)

    def _build_debug_text(self, payload: Any) -> str:
        if not isinstance(payload, dict):
            return "No debug payload available."
        debug = payload.get("debug")
        if not isinstance(debug, dict):
            return "No debug payload available."
        lines = ["Debugger Console", "", "Secondary parameters, intermediate geometry, and methodological details"]
        lines.extend(self._format_mapping(debug, indent=""))
        return "\n".join(lines)

    def copy_result_json(self):
        if self._last_result_payload is None:
            QtWidgets.QMessageBox.information(self, "No result", "Compute a metric first.")
            return
        text = json.dumps(self._last_result_payload, indent=2, sort_keys=True)
        QtWidgets.QApplication.clipboard().setText(text)
        QtWidgets.QMessageBox.information(self, "Copied", "Structured result JSON copied to the clipboard.")

    def _format_mapping(self, value: Dict[str, Any], indent: str = "") -> List[str]:
        lines: List[str] = []
        for key in sorted(value.keys()):
            item = value[key]
            prefix = f"{indent}- {key}:"
            if isinstance(item, dict):
                lines.append(prefix)
                lines.extend(self._format_mapping(item, indent=indent + "  "))
            elif isinstance(item, (float, int, np.floating, np.integer)):
                lines.append(f"{prefix} {float(item):.6g}")
            else:
                lines.append(f"{prefix} {self._format_result_value(item)}")
        return lines

    def _format_result_value(self, value: Any) -> str:
        if isinstance(value, np.ndarray):
            arr = np.asarray(value)
            if arr.ndim > 1:
                return f"array(shape={arr.shape})"
            return np.array2string(arr, precision=4, suppress_small=True)
        return repr(value)

    def _json_safe(self, value: Any) -> Any:
        return _json_safe(value)


# =============================================================================
# Analysis support widgets
# =============================================================================
class DatasetPreviewPanel(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PreviewCard")

        self.lbl_summary = QtWidgets.QLabel("No dataset loaded.")
        self.lbl_summary.setWordWrap(True)
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Point", "Labels", "X", "Y", "Z"])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.lbl_summary)
        lay.addWidget(self.table, 1)

    def set_dataset(self, ds) -> None:
        if ds is None:
            self.lbl_summary.setText("No dataset loaded.")
            self.table.setRowCount(0)
            return

        rows = _dataset_to_rows(ds)
        vendor = ""
        source = ""
        path = ""
        if isinstance(getattr(ds, "meta", None), dict):
            vendor = str(ds.meta.get("vendor", "") or "")
            source = str(ds.meta.get("source", "") or "")
            path = str(ds.meta.get("loaded_from", "") or "")

        summary = f"{len(rows)} points"
        if vendor:
            summary += f" | {vendor}"
        if source:
            summary += f" | {source}"
        if path:
            summary += f"\n{path}"
        self.lbl_summary.setText(summary)

        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            values = [
                str(row["name"]),
                str(row["labels"]),
                f"{float(row['x']):.6f}",
                f"{float(row['y']):.6f}",
                f"{float(row['z']):.6f}",
            ]
            for c, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                self.table.setItem(r, c, item)


class DebugConsole(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DebugCard")

        self.btn_toggle = QtWidgets.QToolButton()
        self.btn_toggle.setText("Debugger Console")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setChecked(False)
        self.btn_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.btn_toggle.setArrowType(QtCore.Qt.RightArrow)

        self.txt_debug = QtWidgets.QPlainTextEdit()
        self.txt_debug.setReadOnly(True)
        self.txt_debug.setPlaceholderText("Secondary parameters, intermediate geometry, and methodological details will appear here.")
        self.txt_debug.hide()

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.btn_toggle)
        lay.addWidget(self.txt_debug, 1)

        self.btn_toggle.toggled.connect(self._set_expanded)

    def _set_expanded(self, expanded: bool) -> None:
        self.btn_toggle.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        self.txt_debug.setVisible(expanded)

    def set_text(self, text: str) -> None:
        self.txt_debug.setPlainText(text or "No debug payload available.")

    def set_expanded(self, expanded: bool) -> None:
        self.btn_toggle.setChecked(expanded)


class CollapsibleCard(QtWidgets.QFrame):
    def __init__(self, title: str, content: QtWidgets.QWidget, expanded: bool = True, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self._content = content

        self.btn_toggle = QtWidgets.QToolButton()
        self.btn_toggle.setText(title)
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setChecked(expanded)
        self.btn_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.btn_toggle.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        self.btn_toggle.setObjectName("CardToggle")

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)
        lay.addWidget(self.btn_toggle)
        lay.addWidget(self._content, 1)

        self.btn_toggle.toggled.connect(self.set_expanded)
        self.set_expanded(expanded)

    def set_expanded(self, expanded: bool) -> None:
        self.btn_toggle.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        self._content.setVisible(expanded)

    def is_expanded(self) -> bool:
        return bool(self.btn_toggle.isChecked())


class ToolWindow(QtWidgets.QDialog):
    def __init__(self, title: str, content: QtWidgets.QWidget, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self.setWindowFlag(QtCore.Qt.Tool, True)
        self.resize(760, 520)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(10)
        lay.addWidget(content, 1)

    def open_and_raise(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()


class DebugConsoleWindow(ToolWindow):
    def __init__(self, parent=None):
        self.txt_debug = QtWidgets.QPlainTextEdit()
        self.txt_debug.setReadOnly(True)
        self.txt_debug.setPlaceholderText("Secondary parameters, intermediate geometry, and methodological details will appear here.")
        super().__init__("Debugger Console", self.txt_debug, parent=parent)
        self.resize(820, 420)

    def set_text(self, text: str) -> None:
        self.txt_debug.setPlainText(text or "No debug payload available.")


class DatasetPreviewWindow(ToolWindow):
    def __init__(self, parent=None):
        self.preview = DatasetPreviewPanel()
        super().__init__("Dataset Preview", self.preview, parent=parent)
        self.resize(860, 560)

    def set_dataset(self, ds) -> None:
        self.preview.set_dataset(ds)


class PointPickerDialog(QtWidgets.QDialog):
    def __init__(self, ds, multi_select: bool, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Dataset Points")
        self.resize(760, 540)
        self._multi_select = multi_select
        self._ds = ds

        self.lbl_info = QtWidgets.QLabel(
            "Select one or more points from the current dataset. Double-click a row to confirm."
        )
        self.lbl_info.setWordWrap(True)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Point", "Labels", "X", "Y", "Z"])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.MultiSelection if multi_select
            else QtWidgets.QAbstractItemView.SingleSelection
        )
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.cellDoubleClicked.connect(self._accept_if_possible)

        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_ok = QtWidgets.QPushButton("Use Selection")

        actions = QtWidgets.QHBoxLayout()
        actions.addStretch(1)
        actions.addWidget(self.btn_cancel)
        actions.addWidget(self.btn_ok)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.lbl_info)
        lay.addWidget(self.table, 1)
        lay.addLayout(actions)

        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok.clicked.connect(self.accept)

        self._populate()

    def _populate(self) -> None:
        if self._ds is None:
            self.table.setRowCount(0)
            return
        rows = _dataset_to_rows(self._ds)
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            values = [
                str(row["name"]),
                str(row["labels"]),
                f"{float(row['x']):.6f}",
                f"{float(row['y']):.6f}",
                f"{float(row['z']):.6f}",
            ]
            for c, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                self.table.setItem(r, c, item)

    def _accept_if_possible(self, _row: int, _column: int) -> None:
        if not self._multi_select:
            self.accept()

    def selected_points(self) -> List[str]:
        names: List[str] = []
        seen = set()
        for model_index in self.table.selectionModel().selectedRows():
            item = self.table.item(model_index.row(), 0)
            if item is None:
                continue
            name = item.text().strip()
            if name and name not in seen:
                seen.add(name)
                names.append(name)
        return names


# =============================================================================
# Main window
# =============================================================================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Surgiplot")
        self.resize(1560, 940)

        self.dataset_panel = DatasetPanel()
        self.plot_panel = PlotPanel()
        self.metric_panel = MetricPanel(self.dataset_panel, self.plot_panel)
        self.dataset_preview_window = DatasetPreviewWindow(self)
        self.debug_console_window = DebugConsoleWindow(self)
        self.debug_console_window.set_text("No debug payload available.")

        self.pages = QtWidgets.QStackedWidget()
        self.btn_proceed = QtWidgets.QPushButton("Proceed to Analysis")
        self.btn_proceed.setEnabled(False)
        self.btn_go_database = QtWidgets.QPushButton("Go to Database")
        self.btn_open_dataset_preview = QtWidgets.QPushButton("Open Dataset Preview")
        self.btn_open_debug_console = QtWidgets.QPushButton("Open Debug Console")
        self.lbl_analysis_context = QtWidgets.QLabel("No dataset loaded.")
        self.lbl_analysis_context.setObjectName("AnalysisContext")
        self.lbl_analysis_context.setWordWrap(True)

        self.dataset_panel.dataset_changed.connect(self._on_dataset_changed)
        self.metric_panel.analysis_updated.connect(self._on_analysis_updated)
        self.btn_proceed.clicked.connect(self._go_to_analysis)
        self.btn_go_database.clicked.connect(self._go_to_database)
        self.btn_open_dataset_preview.clicked.connect(self._open_dataset_preview)
        self.btn_open_debug_console.clicked.connect(self._open_debug_console)

        self.pages.addWidget(self._build_database_page())
        self.pages.addWidget(self._build_analysis_page())

        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(self.pages)
        self.setCentralWidget(w)

        self._build_menu()
        self._apply_theme()
        self._on_dataset_changed()

    def _build_database_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.setSpacing(14)

        hero = self._make_header_card(
            "1. Dataset Ingestion & Curation",
            "Import neuronavigation exports or formatted datasets, review the point table, refine semantic labels, and curate the dataset before analysis.",
        )
        lay.addWidget(hero)

        actions = QtWidgets.QHBoxLayout()
        actions.addStretch(1)
        actions.addWidget(self.btn_proceed)
        lay.addLayout(actions)

        lay.addWidget(self.dataset_panel, 1)
        return page

    def _build_analysis_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.setSpacing(14)

        header = self._make_header_card(
            "2. Quantitative Analysis",
            "Compute corridor metrics, inspect the live 3D geometry, and open auxiliary scientific views when you need deeper methodological detail.",
        )
        header.layout().addWidget(self.lbl_analysis_context)
        lay.addWidget(header)

        top_actions = QtWidgets.QHBoxLayout()
        top_actions.addWidget(self.btn_go_database)
        top_actions.addWidget(self.btn_open_dataset_preview)
        top_actions.addWidget(self.btn_open_debug_console)
        top_actions.addStretch(1)
        lay.addLayout(top_actions)

        main_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_split.setChildrenCollapsible(False)

        analysis_card = self._make_card("Analysis Workspace")
        analysis_card.layout().addWidget(self.metric_panel)

        right_split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right_split.setChildrenCollapsible(False)

        plot_card = self._make_card("3D Scene")
        plot_card.layout().addWidget(self.plot_panel)

        right_split.addWidget(plot_card)
        right_split.setStretchFactor(0, 1)
        right_split.setSizes([980])

        main_split.addWidget(analysis_card)
        main_split.addWidget(right_split)
        main_split.setStretchFactor(0, 3)
        main_split.setStretchFactor(1, 4)
        main_split.setSizes([720, 980])

        lay.addWidget(main_split, 1)
        return page

    def _make_header_card(self, title: str, subtitle: str) -> QtWidgets.QFrame:
        card = QtWidgets.QFrame()
        card.setObjectName("HeaderCard")
        lay = QtWidgets.QVBoxLayout(card)
        lay.setContentsMargins(18, 18, 18, 18)
        lay.setSpacing(6)
        title_label = QtWidgets.QLabel(title)
        title_label.setObjectName("HeaderTitle")
        subtitle_label = QtWidgets.QLabel(subtitle)
        subtitle_label.setObjectName("HeaderSubtitle")
        subtitle_label.setWordWrap(True)
        lay.addWidget(title_label)
        lay.addWidget(subtitle_label)
        return card

    def _make_card(self, title: str) -> QtWidgets.QFrame:
        card = QtWidgets.QFrame()
        card.setObjectName("Card")
        lay = QtWidgets.QVBoxLayout(card)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)
        title_label = QtWidgets.QLabel(title)
        title_label.setObjectName("CardTitle")
        lay.addWidget(title_label)
        return card

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #edf2f7;
                color: #1f2937;
            }
            QFrame#HeaderCard, QFrame#Card, QFrame#PreviewCard, QFrame#DebugCard, QFrame#SceneLayerPanel {
                background: #ffffff;
                border: 1px solid #dce3ef;
                border-radius: 18px;
            }
            QLabel#HeaderTitle {
                font-size: 24px;
                font-weight: 700;
                color: #0f172a;
            }
            QLabel#HeaderSubtitle {
                font-size: 13px;
                color: #526073;
            }
            QLabel#CardTitle {
                font-size: 15px;
                font-weight: 700;
                color: #0f172a;
            }
            QLabel#SceneLayerTitle {
                font-size: 13px;
                font-weight: 700;
                color: #0f172a;
            }
            QLabel#PlotTitle {
                font-size: 14px;
                font-weight: 600;
                color: #1f2937;
            }
            QLabel#PlotStatus, QLabel#AnalysisContext {
                color: #526073;
                font-size: 12px;
            }
            QPushButton, QToolButton {
                background: #f7f9fc;
                border: 1px solid #d6dfeb;
                border-radius: 10px;
                padding: 8px 12px;
                font-weight: 600;
            }
            QPushButton:hover, QToolButton:hover {
                background: #eef4ff;
                border-color: #a7c1ff;
            }
            QPushButton:disabled {
                color: #98a2b3;
                background: #f4f6f8;
            }
            QComboBox, QLineEdit, QDoubleSpinBox, QPlainTextEdit, QTableWidget {
                background: #fbfcfe;
                border: 1px solid #d6dfeb;
                border-radius: 10px;
                padding: 6px 8px;
            }
            QHeaderView::section {
                background: #f1f5fb;
                border: none;
                border-bottom: 1px solid #d6dfeb;
                padding: 8px;
                color: #435168;
                font-weight: 600;
            }
            QListWidget {
                background: #fbfcfe;
                border: 1px solid #d6dfeb;
                border-radius: 12px;
                padding: 6px;
            }
            QSplitter::handle {
                background: #dce3ef;
                margin: 3px;
                border-radius: 3px;
            }
            """
        )

    def _build_menu(self):
        m = self.menuBar()
        file_menu = m.addMenu("&File")
        file_menu.addAction(QtGuiAction("Quit", self, shortcut="Ctrl+Q", triggered=self.close))

        help_menu = m.addMenu("&Help")
        help_menu.addAction(QtGuiAction("About", self, triggered=self._about))

    def _about(self):
        msg = (
            "Surgiplot GUI\n\n"
            "- Import navigation files (Stryker / Medtronic)\n"
            "- Import formatted databases\n"
            "- Import generic point files into an existing dataset\n"
            "- Compute AoA/SF, VOM/VoA, AoE, Distance, Area\n"
            "- Optional imported 3D scene workspace (if installed)\n"
        )
        QtWidgets.QMessageBox.information(self, "About Surgiplot", msg)

    def _on_dataset_changed(self):
        ds = self.dataset_panel.ds
        self.btn_proceed.setEnabled(ds is not None and bool(_ensure_points_dict(ds)) if ds is not None else False)
        self.dataset_preview_window.set_dataset(ds)
        if ds is None:
            self.lbl_analysis_context.setText("No dataset loaded.")
            return

        rows = _dataset_to_rows(ds)
        vendor = ""
        source = ""
        if isinstance(getattr(ds, "meta", None), dict):
            vendor = str(ds.meta.get("vendor", "") or "")
            source = str(ds.meta.get("source", "") or "")
        context = f"{len(rows)} points ready for analysis"
        if vendor:
            context += f" | {vendor}"
        if source:
            context += f" | {source}"
        self.lbl_analysis_context.setText(context)
        try:
            self.plot_panel.plot_points(_ensure_points_dict(ds), title="Dataset points", scene_meta=_dataset_scene_meta(ds))
        except Exception:
            pass

    def _on_analysis_updated(self, title: str, _result_text: str, debug_text: str, _payload: object) -> None:
        self.debug_console_window.set_text(debug_text)
        if title != "Dataset points":
            self.pages.setCurrentIndex(1)

    def _go_to_analysis(self) -> None:
        if self.dataset_panel.ds is None or not _ensure_points_dict(self.dataset_panel.ds):
            QtWidgets.QMessageBox.information(self, "No dataset", "Load a dataset before proceeding to analysis.")
            return
        self.pages.setCurrentIndex(1)

    def _go_to_database(self) -> None:
        self.pages.setCurrentIndex(0)

    def _open_dataset_preview(self) -> None:
        self.dataset_preview_window.open_and_raise()

    def _open_debug_console(self) -> None:
        self.debug_console_window.open_and_raise()


# =============================================================================
# QAction helper
# =============================================================================
class QtGuiAction(QtGui.QAction):
    def __init__(self, text, parent=None, shortcut: Optional[str] = None, triggered=None):
        super().__init__(text, parent)
        if shortcut:
            self.setShortcut(shortcut)
        if triggered:
            self.triggered.connect(triggered)


# =============================================================================
# Entry point
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
