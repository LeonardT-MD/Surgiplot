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
- Optional single-image 3D dialog
- Optional matplotlib 3D plotting
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, Optional, Any, List

import numpy as np
from PySide6 import QtWidgets, QtCore, QtGui

from surgiplot.core.io.loaders import load_dataset, load_navigation_file
from surgiplot.metrics import VOM_VOA, AOA_SF, AOE, DISTANCE_3D, AREA_3D
from surgiplot.gui.label_editor import run_label_editor

try:
    from surgiplot.ai.single_image_depth import SingleImage3DDialog
    HAS_SINGLE_IMAGE_3D = True
except Exception:
    SingleImage3DDialog = None  # type: ignore
    HAS_SINGLE_IMAGE_3D = False

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    HAS_MPL = True
except Exception:
    FigureCanvas = None  # type: ignore
    NavigationToolbar = None  # type: ignore
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
            self.toolbar = None
            self._scene_payload = None
            self._interactive_points: List[dict] = []
            self._title = QtWidgets.QLabel("")
            self._status = QtWidgets.QLabel("")
            lay.addWidget(self._title)
            lay.addWidget(self._status)
            return

        self._scene_payload: Optional[dict] = None
        self._interactive_points: List[dict] = []
        self._selection_artist = None
        self._selected_point = None
        self._projection = "3d"

        controls = QtWidgets.QHBoxLayout()
        self.btn_reset_view = QtWidgets.QPushButton("Reset View")
        self.btn_export_image = QtWidgets.QPushButton("Export PNG…")
        self.btn_export_scene = QtWidgets.QPushButton("Export 3D Scene…")
        self.btn_open_scene = QtWidgets.QPushButton("Open Scene…")
        controls.addWidget(self.btn_reset_view)
        controls.addStretch(1)
        controls.addWidget(self.btn_open_scene)
        controls.addWidget(self.btn_export_scene)
        controls.addWidget(self.btn_export_image)
        lay.addLayout(controls)

        self.fig = Figure(figsize=(6, 5), constrained_layout=True)
        self.fig.patch.set_facecolor("#f5f7fb")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(520)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self._apply_axes_style_3d()
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas)

        self._title = QtWidgets.QLabel("")
        self._title.setObjectName("PlotTitle")
        self._title.setWordWrap(True)
        self._status = QtWidgets.QLabel("Hover over a point to inspect its coordinates. Click to pin the selection.")
        self._status.setObjectName("PlotStatus")
        lay.addWidget(self._title)
        lay.addWidget(self._status)

        self.btn_reset_view.clicked.connect(self.reset_view)
        self.btn_export_image.clicked.connect(self.export_png)
        self.btn_export_scene.clicked.connect(self.export_scene)
        self.btn_open_scene.clicked.connect(self.open_scene)
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

    def clear(self, title: str = "", projection: str = "3d"):
        if not HAS_MPL:
            self._title.setText(title or "")
            return
        self.fig.clear()
        self._projection = projection
        if projection == "polar":
            self.ax = self.fig.add_subplot(111, projection="polar")
            self._apply_axes_style_polar()
        else:
            self.ax = self.fig.add_subplot(111, projection="3d")
            self._apply_axes_style_3d()
        self._interactive_points = []
        self._selection_artist = None
        self._selected_point = None
        self._title.setText(title or "")
        self._status.setText("Hover over a point to inspect its coordinates. Click to pin the selection.")
        self.canvas.draw_idle()

    def plot_points(self, points: Dict[str, np.ndarray], title: str = "Dataset points"):
        scene = {
            "scene_type": "dataset",
            "title": title,
            "points": {str(k): _json_safe(_as_xyz(v)) for k, v in points.items()},
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
            elev = float(view.get("elev", 18.0))
            azim = float(view.get("azim", -58.0))
            self.ax.view_init(elev=elev, azim=azim)
        self._scene_payload = scene
        self.canvas.draw_idle()

    def _render_dataset_scene(self, scene: dict) -> None:
        self.clear(str(scene.get("title", "Dataset points")))
        points = scene.get("points", {}) if isinstance(scene.get("points"), dict) else {}
        if not points:
            self._title.setText("No points to plot.")
            return

        labels = sorted(points.keys())
        xyz = np.vstack([_as_xyz(points[k]) for k in labels])
        self.ax.scatter(
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
        self._set_interactive_points(
            [{"label": lb, "xyz": xyz[i], "kind": "dataset"} for i, lb in enumerate(labels)]
        )
        self._set_axes_from_points([xyz])

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
        normal = np.asarray(dbg.get("target_normal", []), dtype=float).reshape(-1)

        all_points: List[np.ndarray] = []
        interactive: List[dict] = []
        entry_names = list(scene_meta.get("entry_names", []) or [])
        target_names = list(scene_meta.get("target_names", []) or [])

        if entry_poly.ndim == 2 and entry_poly.shape[1] == 3:
            closed = np.vstack([entry_poly, entry_poly[0]])
            all_points.append(entry_poly)
            self.ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color="tab:blue", linewidth=1.2, alpha=0.48)
            self.ax.scatter(
                entry_poly[:, 0], entry_poly[:, 1], entry_poly[:, 2],
                color="tab:blue", s=24, alpha=0.7, edgecolors="white", linewidths=0.45, depthshade=False,
            )
            for i, pt in enumerate(entry_poly):
                label = entry_names[i] if i < len(entry_names) else f"Entry {i + 1}"
                interactive.append({"label": label, "xyz": pt, "kind": "entry"})

        if target_poly.ndim == 2 and target_poly.shape[1] == 3:
            closed = np.vstack([target_poly, target_poly[0]])
            all_points.append(target_poly)
            self.ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color="tab:red", linewidth=1.2, alpha=0.48)
            self.ax.scatter(
                target_poly[:, 0], target_poly[:, 1], target_poly[:, 2],
                color="tab:red", s=24, alpha=0.7, edgecolors="white", linewidths=0.45, depthshade=False,
            )
            for i, pt in enumerate(target_poly):
                label = target_names[i] if i < len(target_names) else f"Target {i + 1}"
                interactive.append({"label": label, "xyz": pt, "kind": "target"})

        if entry_ellipse.ndim == 2 and entry_ellipse.shape[1] == 3:
            all_points.append(entry_ellipse)
            self.ax.plot(entry_ellipse[:, 0], entry_ellipse[:, 1], entry_ellipse[:, 2], color="tab:blue", linewidth=2.2)

        if target_ellipse.ndim == 2 and target_ellipse.shape[1] == 3:
            all_points.append(target_ellipse)
            self.ax.plot(target_ellipse[:, 0], target_ellipse[:, 1], target_ellipse[:, 2], color="tab:red", linewidth=2.2)

        if full_surface.ndim == 3 and full_surface.shape[-1] == 3 and len(full_surface) > 0:
            self.ax.add_collection3d(
                Poly3DCollection(
                    list(full_surface),
                    facecolors="tab:gray",
                    alpha=0.045,
                    edgecolors=(0.65, 0.67, 0.70, 0.12),
                    linewidths=0.15,
                )
            )

        if svom_surface.ndim == 3 and svom_surface.shape[-1] == 3 and len(svom_surface) > 0:
            self.ax.add_collection3d(
                Poly3DCollection(
                    list(svom_surface),
                    facecolors="tab:green",
                    alpha=0.18,
                    edgecolors=(0.14, 0.62, 0.32, 0.25),
                    linewidths=0.25,
                )
            )

        if cut_ellipse.ndim == 2 and cut_ellipse.shape[1] == 3:
            all_points.append(cut_ellipse)
            self.ax.plot(cut_ellipse[:, 0], cut_ellipse[:, 1], cut_ellipse[:, 2], color="tab:green", linewidth=1.9)

        if cE.size == 3 and cT.size == 3:
            centers = np.vstack([cE, cT])
            all_points.append(centers)
            self.ax.scatter(
                centers[:, 0], centers[:, 1], centers[:, 2],
                color="black", s=44, depthshade=False, edgecolors="white", linewidths=0.6,
            )
            self.ax.plot(
                [cE[0], cT[0]], [cE[1], cT[1]], [cE[2], cT[2]],
                color="black", linestyle="--", linewidth=1.35, alpha=0.75,
            )
            interactive.extend([
                {"label": "Entry centroid", "xyz": cE, "kind": "centroid"},
                {"label": "Target centroid", "xyz": cT, "kind": "centroid"},
            ])

        if cT.size == 3 and normal.size == 3:
            scale = max(float(dbg.get("distance_h", 0.0) or 0.0) * 0.25, 1.0)
            tip = cT + normal * scale
            all_points.append(np.vstack([cT, tip]))
            self.ax.quiver(
                cT[0], cT[1], cT[2],
                normal[0], normal[1], normal[2],
                length=scale,
                color="tab:purple",
                normalize=True,
                linewidth=1.3,
            )

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
        self._set_axes_from_points(all_points)

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

        def _plot_triangle(tri: np.ndarray, color: str, face_alpha: float, label: str, linestyle: str = "-") -> None:
            if tri.ndim != 2 or tri.shape != (3, 3):
                return
            closed = np.vstack([tri, tri[0]])
            self.ax.plot(
                closed[:, 0], closed[:, 1], closed[:, 2],
                color=color, linewidth=2.2, alpha=0.95, linestyle=linestyle, label=label,
            )
            self.ax.scatter(
                tri[:, 0], tri[:, 1], tri[:, 2],
                color=color, s=52, edgecolors="white", linewidths=0.6, depthshade=False,
            )
            self.ax.add_collection3d(
                Poly3DCollection([tri], facecolors=color, alpha=face_alpha, edgecolors="none")
            )
            all_points.append(tri)

        _plot_triangle(vertical_triangle, "tab:blue", 0.18, "V-AoA")
        _plot_triangle(horizontal_triangle, "tab:green", 0.18, "H-AoA")

        if sf_surface.ndim == 2 and sf_surface.shape == (4, 3):
            closed = np.vstack([sf_surface, sf_surface[0]])
            self.ax.plot(
                closed[:, 0], closed[:, 1], closed[:, 2],
                color="#0f172a", linewidth=1.5, alpha=0.7, label="SF surface",
            )
            self.ax.add_collection3d(
                Poly3DCollection([sf_surface], facecolors="#94a3b8", alpha=0.14, edgecolors="none")
            )
            all_points.append(sf_surface)

        if sf_surface_rescaled.ndim == 2 and sf_surface_rescaled.shape == (4, 3):
            closed = np.vstack([sf_surface_rescaled, sf_surface_rescaled[0]])
            self.ax.plot(
                closed[:, 0], closed[:, 1], closed[:, 2],
                color="tab:orange", linewidth=1.7, alpha=0.92, label="Standardized SF surface",
            )
            self.ax.add_collection3d(
                Poly3DCollection([sf_surface_rescaled], facecolors="tab:orange", alpha=0.10, edgecolors="none")
            )
            all_points.append(sf_surface_rescaled)

        if pivot.size == 3:
            all_points.append(pivot.reshape(1, 3))
            self.ax.scatter(
                [pivot[0]], [pivot[1]], [pivot[2]],
                color="tab:red", s=68, edgecolors="white", linewidths=0.7, depthshade=False,
            )
            interactive.append({"label": scene_meta.get("pivot_name", "Pivot"), "xyz": pivot, "kind": "pivot"})
            self.ax.text(pivot[0], pivot[1], pivot[2], " Target Point", color="tab:red", fontsize=9)

        if entry.ndim == 2 and entry.shape[1] == 3:
            all_points.append(entry)
            for i, pt in enumerate(entry[:4]):
                label = entry_names[i] if i < len(entry_names) else f"Entry {i + 1}"
                interactive.append({"label": label, "xyz": pt, "kind": "entry"})

        if rescaled.ndim == 2 and rescaled.shape[1] == 3 and len(rescaled) >= 4:
            all_points.append(rescaled)
            self.ax.scatter(
                rescaled[:, 0], rescaled[:, 1], rescaled[:, 2],
                color="tab:orange", s=42, edgecolors="white", linewidths=0.55, depthshade=False, label="Projected points",
            )
            for original, projected in zip(entry[:4], rescaled[:4]):
                self.ax.plot(
                    [original[0], projected[0]],
                    [original[1], projected[1]],
                    [original[2], projected[2]],
                    color="black",
                    linestyle=":",
                    linewidth=1.15,
                    alpha=0.85,
                )
            for tri in (vertical_triangle_rescaled, horizontal_triangle_rescaled):
                if tri.ndim == 2 and tri.shape == (3, 3):
                    closed = np.vstack([tri, tri[0]])
                    self.ax.plot(
                        closed[:, 0], closed[:, 1], closed[:, 2],
                        color="black", linewidth=1.3, alpha=0.82, linestyle=":", label="_nolegend_",
                    )

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
        self._set_axes_from_points(all_points)

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
        dbg = payload.get("debug", {}) if isinstance(payload.get("debug"), dict) else {}
        A = np.asarray(dbg.get("A", []), dtype=float).reshape(-1)
        B = np.asarray(dbg.get("B", []), dtype=float).reshape(-1)
        if A.size != 3 or B.size != 3:
            raise ValueError("Distance plot requires debug geometry.")

        pts = np.vstack([A, B])
        self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="tab:cyan", linewidth=2)
        self.ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            color="tab:cyan", s=50, edgecolors="white", linewidths=0.55, depthshade=False,
        )
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
        self._set_axes_from_points([pts])

    def _render_area_scene(self, title: str, payload: dict, scene_meta: dict) -> None:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        self.clear(title)
        dbg = payload.get("debug", {}) if isinstance(payload.get("debug"), dict) else {}
        poly = np.asarray(dbg.get("polygon", []), dtype=float)
        coplanar = np.asarray(dbg.get("coplanar_polygon_3d", []), dtype=float)
        if poly.ndim != 2 or poly.shape[1] != 3 or len(poly) < 3:
            raise ValueError("Area plot requires debug geometry.")

        closed = np.vstack([poly, poly[0]])
        self.ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color="tab:orange", linewidth=2, alpha=0.92, label="Original polygon")
        self.ax.scatter(
            poly[:, 0], poly[:, 1], poly[:, 2],
            color="tab:orange", s=46, edgecolors="white", linewidths=0.55, depthshade=False,
        )
        all_points = [poly]

        if coplanar.ndim == 2 and coplanar.shape == poly.shape:
            coplanar_closed = np.vstack([coplanar, coplanar[0]])
            self.ax.plot(
                coplanar_closed[:, 0], coplanar_closed[:, 1], coplanar_closed[:, 2],
                color="#0f172a", linewidth=1.8, alpha=0.75, label="PCA fitted coplanar surface",
            )
            self.ax.add_collection3d(
                Poly3DCollection([coplanar], facecolors="#94a3b8", alpha=0.16, edgecolors="none")
            )
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
        self._set_axes_from_points(all_points)
        handles, labels = self.ax.get_legend_handles_labels()
        if labels:
            self.ax.legend(loc="best")

    def _set_interactive_points(self, points: List[dict]) -> None:
        self._interactive_points = [
            {
                "label": str(p.get("label", "Point")),
                "xyz": _as_xyz(p.get("xyz")),
                "kind": str(p.get("kind", "point")),
            }
            for p in points
        ]

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
        if not HAS_MPL or self.ax is None:
            return
        if self._projection == "3d":
            self.ax.view_init(elev=18, azim=-58)
            self.canvas.draw_idle()

    def export_png(self) -> None:
        if not HAS_MPL or self.fig is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export plot as PNG",
            "surgiplot-plot.png",
            "PNG image (*.png);;All files (*)",
        )
        if not path:
            return
        self.fig.savefig(path, dpi=220, facecolor=self.fig.get_facecolor(), bbox_inches="tight")
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
        scene["view"] = {"elev": float(self.ax.elev), "azim": float(self.ax.azim)}
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

        self.btn_single_image_3d = QtWidgets.QPushButton("Single image → 3D points…")
        self.btn_single_image_3d.setEnabled(HAS_SINGLE_IMAGE_3D)

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

        self.btn_import_nav.clicked.connect(self._on_import_navigation)
        self.btn_import_db.clicked.connect(self._on_import_database)
        self.btn_import_generic.clicked.connect(self._on_import_generic_points)
        self.btn_save_points.clicked.connect(self._on_export_points)
        self.btn_clear_points.clicked.connect(self._on_clear_points)
        self.btn_add_point.clicked.connect(self._on_add_point)
        self.btn_del_point.clicked.connect(self._on_delete_selected)
        self.btn_single_image_3d.clicked.connect(self._on_single_image_3d)

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

    def _on_single_image_3d(self):
        if not HAS_SINGLE_IMAGE_3D:
            QtWidgets.QMessageBox.information(
                self,
                "Feature not available",
                "Single-image 3D feature is not available.\n"
                "Ensure `surgiplot.ai.single_image_depth` exists and deps are installed.",
            )
            return
        if self.ds is None:
            QtWidgets.QMessageBox.warning(self, "No dataset", "Load or import a dataset first.")
            return

        dlg = SingleImage3DDialog(self.ds, on_dataset_changed_callback=self._on_ai_points_committed, parent=self)
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
        self.cmb_metric.addItems(["AoA / SF", "VOM / VoA", "AoE", "Distance", "Area", "Dataset overview"])

        self.btn_compute = QtWidgets.QPushButton("Compute")
        self.btn_plot_points = QtWidgets.QPushButton("Plot Dataset")
        self.btn_copy_json = QtWidgets.QPushButton("Copy Result JSON")

        self._last_result_payload = None
        self._last_debug_text = ""

        self._point_selectors: List[QtWidgets.QComboBox] = []
        self._name_list_edits: List[QtWidgets.QLineEdit] = []

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
        aoa_layout.addRow("Cranial:", self.cmb_cranial)
        aoa_layout.addRow("Caudal:", self.cmb_caudal)
        aoa_layout.addRow("Medial:", self.cmb_medial)
        aoa_layout.addRow("Lateral:", self.cmb_lateral)
        aoa_layout.addRow("Pivot:", self.cmb_pivot)
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
        vom_layout.addRow("Entry Polygon:", self.edit_vom_entry)
        vom_layout.addRow("Target Polygon:", self.edit_vom_target)
        vom_layout.addRow("Standard Distance:", self.spin_stand_dist)
        self.metric_form_indexes["VOM / VoA"] = self.metric_forms.addWidget(self.form_vom)

        self.form_aoe = QtWidgets.QWidget()
        aoe_layout = QtWidgets.QFormLayout(self.form_aoe)
        self.cmb_aoe_a = self._make_point_combo()
        self.cmb_aoe_b = self._make_point_combo()
        self.cmb_aoe_c = self._make_point_combo()
        aoe_layout.addRow("Point A:", self.cmb_aoe_a)
        aoe_layout.addRow("Pivot B:", self.cmb_aoe_b)
        aoe_layout.addRow("Point C:", self.cmb_aoe_c)
        self.metric_form_indexes["AoE"] = self.metric_forms.addWidget(self.form_aoe)

        self.form_distance = QtWidgets.QWidget()
        distance_layout = QtWidgets.QFormLayout(self.form_distance)
        self.cmb_distance_a = self._make_point_combo()
        self.cmb_distance_b = self._make_point_combo()
        distance_layout.addRow("Point A:", self.cmb_distance_a)
        distance_layout.addRow("Point B:", self.cmb_distance_b)
        self.metric_form_indexes["Distance"] = self.metric_forms.addWidget(self.form_distance)

        self.form_area = QtWidgets.QWidget()
        area_layout = QtWidgets.QFormLayout(self.form_area)
        self.edit_area_polygon = self._make_name_list_edit(
            "Enter polygon labels, separated by commas"
        )
        area_layout.addRow("Polygon:", self.edit_area_polygon)
        self.metric_form_indexes["Area"] = self.metric_forms.addWidget(self.form_area)

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

    def plot_points(self):
        ds = self._ds()
        if ds is None:
            QtWidgets.QMessageBox.warning(self, "No dataset", "Load or import a dataset first.")
            return
        self.plot_panel.plot_points(_ensure_points_dict(ds), title="Dataset points")
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
                    scene_meta={"A_name": name_a, "B_name": name_b, "C_name": name_c},
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
                    scene_meta={"A_name": name_a, "B_name": name_b},
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
                    scene_meta={"polygon_names": polygon_names},
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
        names = [part.strip() for part in raw.split(",") if part.strip()]
        if not names:
            if minimum <= 0:
                return None
            raise ValueError("Enter one or more point names or labels, separated by commas.")
        if len(names) < minimum:
            raise ValueError(f"Enter at least {minimum} point names or labels.")
        return names

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
            QFrame#HeaderCard, QFrame#Card, QFrame#PreviewCard, QFrame#DebugCard {
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
            "- Optional single-image 3D point collection (if installed)\n"
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
            self.plot_panel.plot_points(_ensure_points_dict(ds), title="Dataset points")
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
