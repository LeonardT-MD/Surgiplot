from __future__ import annotations

import os
import platform
from typing import Optional

import numpy as np
from PySide6 import QtCore, QtWidgets

def _should_enable_pyvista() -> bool:
    backend = os.environ.get("SURGIPLOT_3D_BACKEND", "auto").strip().lower()
    if backend in {"matplotlib", "mpl", "safe"}:
        return False
    if backend == "pyvista":
        return True
    # Native Qt/VTK startup is still fragile on macOS in some local setups.
    return platform.system().lower() != "darwin"


if _should_enable_pyvista():
    try:
        import pyvista as pv  # type: ignore
        from pyvistaqt import QtInteractor  # type: ignore

        HAS_PYVISTA = True
    except Exception:
        pv = None  # type: ignore
        QtInteractor = None  # type: ignore
        HAS_PYVISTA = False
else:
    pv = None  # type: ignore
    QtInteractor = None  # type: ignore
    HAS_PYVISTA = False

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure

    HAS_MPL = True
except Exception:
    HAS_MPL = False
    FigureCanvas = None  # type: ignore
    NavigationToolbar = None  # type: ignore
    Figure = None  # type: ignore


def _surface_triangles_to_indexed_mesh(
    triangles: np.ndarray,
    colors: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    tris = np.asarray(triangles, dtype=np.float32)
    if tris.ndim != 3 or tris.shape[1:] != (3, 3) or len(tris) == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0, 3), dtype=np.int64),
            colors,
            None,
        )

    flat = tris.reshape(-1, 3)
    # Quantize before deduplication to collapse numerically identical triangle-soup vertices.
    quantized = np.round(flat, decimals=5)
    unique_verts, inverse = np.unique(quantized, axis=0, return_inverse=True)
    face_count = len(tris)
    tri_indices = inverse.reshape(face_count, 3).astype(np.int64)
    faces = np.hstack(
        [
            np.full((face_count, 1), 3, dtype=np.int64),
            tri_indices,
        ]
    ).reshape(-1)
    face_colors = None
    vertex_colors = None
    if colors is not None:
        arr = np.asarray(colors)
        if arr.ndim == 2 and arr.shape[0] == face_count and arr.shape[1] == 3:
            face_colors = arr
            expanded = np.repeat(arr.astype(np.float32, copy=False), 3, axis=0)
            color_sums = np.zeros((len(unique_verts), 3), dtype=np.float32)
            color_counts = np.zeros((len(unique_verts), 1), dtype=np.float32)
            np.add.at(color_sums, inverse, expanded)
            np.add.at(color_counts[:, 0], inverse, 1.0)
            vertex_colors = np.clip(color_sums / np.maximum(color_counts, 1.0), 0.0, 1.0)
    return unique_verts.astype(np.float32, copy=False), faces, tri_indices, face_colors, vertex_colors


def _smooth_vertex_colors(
    vertex_colors: Optional[np.ndarray],
    tri_indices: np.ndarray,
    iterations: int = 1,
    blend: float = 0.30,
) -> Optional[np.ndarray]:
    cols = None if vertex_colors is None else np.asarray(vertex_colors, dtype=np.float32)
    faces = np.asarray(tri_indices, dtype=np.int64)
    if cols is None or cols.ndim != 2 or cols.shape[1] != 3 or len(cols) == 0:
        return cols
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        return cols
    out = cols.copy()
    weight = float(np.clip(blend, 0.0, 1.0))
    if weight <= 0.0:
        return out
    for _ in range(max(int(iterations), 0)):
        sums = out.copy()
        counts = np.ones((len(out), 1), dtype=np.float32)
        for a, b in ((0, 1), (1, 2), (2, 0)):
            src = faces[:, a]
            dst = faces[:, b]
            np.add.at(sums, src, out[dst])
            np.add.at(sums, dst, out[src])
            np.add.at(counts[:, 0], src, 1.0)
            np.add.at(counts[:, 0], dst, 1.0)
        neighbor_avg = sums / np.maximum(counts, 1.0)
        out = np.clip((1.0 - weight) * out + weight * neighbor_avg, 0.0, 1.0)
    return out


def _project_vertex_colors_from_cloud(
    verts: np.ndarray,
    cloud_xyz: Optional[np.ndarray],
    cloud_colors: Optional[np.ndarray],
    max_cloud_points: int = 120000,
) -> Optional[np.ndarray]:
    xyz = None if cloud_xyz is None else np.asarray(cloud_xyz, dtype=np.float32)
    cols = None if cloud_colors is None else np.asarray(cloud_colors, dtype=np.float32)
    vtx = np.asarray(verts, dtype=np.float32)
    if (
        xyz is None
        or cols is None
        or xyz.ndim != 2
        or cols.ndim != 2
        or xyz.shape[1] != 3
        or cols.shape[1] != 3
        or len(xyz) == 0
        or len(cols) != len(xyz)
        or vtx.ndim != 2
        or vtx.shape[1] != 3
        or len(vtx) == 0
    ):
        return None

    if len(xyz) > max_cloud_points:
        idx = np.linspace(0, len(xyz) - 1, num=max_cloud_points, dtype=int)
        xyz = xyz[idx]
        cols = cols[idx]

    mins = np.min(xyz, axis=0)
    maxs = np.max(xyz, axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    base_res = max(10, int(np.ceil(np.cbrt(max(len(xyz), 64)) * 0.55)))
    cell_size = spans / float(base_res)
    grid_coords = np.floor((xyz - mins) / np.maximum(cell_size, 1e-6)).astype(np.int32)
    grid: dict[tuple[int, int, int], list[int]] = {}
    for idx, key_arr in enumerate(grid_coords):
        key = (int(key_arr[0]), int(key_arr[1]), int(key_arr[2]))
        grid.setdefault(key, []).append(idx)

    result = np.zeros((len(vtx), 3), dtype=np.float32)
    coarse_idx = np.linspace(0, len(xyz) - 1, num=min(len(xyz), 4096), dtype=int)
    coarse_xyz = xyz[coarse_idx]
    coarse_cols = cols[coarse_idx]

    for i, vert in enumerate(vtx):
        key_arr = np.floor((vert - mins) / np.maximum(cell_size, 1e-6)).astype(np.int32)
        candidates: list[int] = []
        for radius in (0, 1, 2):
            if candidates:
                break
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dz in range(-radius, radius + 1):
                        key = (int(key_arr[0] + dx), int(key_arr[1] + dy), int(key_arr[2] + dz))
                        pts = grid.get(key)
                        if pts:
                            candidates.extend(pts)
        if candidates:
            cand_idx = np.asarray(candidates, dtype=np.int32)
            cand_xyz = xyz[cand_idx]
            cand_cols = cols[cand_idx]
        else:
            cand_xyz = coarse_xyz
            cand_cols = coarse_cols
        d2 = np.sum((cand_xyz - vert) ** 2, axis=1)
        if len(d2) == 0:
            result[i] = np.array([0.7, 0.74, 0.82], dtype=np.float32)
            continue
        take = min(4, len(d2))
        nn = np.argpartition(d2, take - 1)[:take]
        w = 1.0 / np.maximum(d2[nn], 1e-8)
        result[i] = np.sum(cand_cols[nn] * w[:, None], axis=0) / np.sum(w)
    return np.clip(result, 0.0, 1.0)


def _prepare_surface_dataset(
    verts: np.ndarray,
    faces: np.ndarray,
    face_colors: Optional[np.ndarray],
    vertex_colors: Optional[np.ndarray],
):
    if pv is None:
        return None
    surf = pv.PolyData(verts, faces)
    if face_colors is not None:
        surf.cell_data["rgb"] = np.clip(face_colors * 255.0, 0, 255).astype(np.uint8)
    if vertex_colors is not None:
        surf.point_data["rgb"] = np.clip(vertex_colors * 255.0, 0, 255).astype(np.uint8)
    try:
        surf = surf.compute_normals(
            cell_normals=False,
            point_normals=True,
            split_vertices=False,
            consistent_normals=True,
            auto_orient_normals=False,
            non_manifold_traversal=True,
            inplace=False,
        )
    except Exception:
        pass
    return surf


def _build_textured_surface_dataset(surface_texture_payload: Optional[dict]):
    if pv is None or not isinstance(surface_texture_payload, dict):
        return None, None
    verts = np.asarray(surface_texture_payload.get("vertices", []), dtype=np.float32)
    faces_idx = np.asarray(surface_texture_payload.get("faces", []), dtype=np.int64)
    uv = np.asarray(surface_texture_payload.get("uv", []), dtype=np.float32)
    texture_path = str(surface_texture_payload.get("texture_path", "") or "").strip()
    if (
        verts.ndim != 2
        or verts.shape[1] != 3
        or faces_idx.ndim != 2
        or faces_idx.shape[1] != 3
        or uv.ndim != 2
        or uv.shape[1] != 2
        or len(verts) == 0
        or len(faces_idx) == 0
        or len(uv) != len(verts)
        or not texture_path
        or not os.path.exists(texture_path)
    ):
        return None, None
    faces = np.hstack(
        [np.full((len(faces_idx), 1), 3, dtype=np.int64), faces_idx.astype(np.int64)]
    ).reshape(-1)
    surf = pv.PolyData(verts, faces)
    try:
        surf.active_texture_coordinates = uv.astype(np.float32, copy=False)
    except Exception:
        try:
            surf.point_data["Texture Coordinates"] = uv.astype(np.float32, copy=False)
        except Exception:
            pass
    try:
        surf = surf.compute_normals(
            cell_normals=False,
            point_normals=True,
            split_vertices=False,
            consistent_normals=True,
            auto_orient_normals=False,
            non_manifold_traversal=True,
            inplace=False,
        )
    except Exception:
        pass
    try:
        texture = pv.read_texture(texture_path)
    except Exception:
        texture = None
    return surf, texture


class PointCloudPicker3D(QtWidgets.QWidget):
    """3D scene viewer with picking. Uses PyVista when available, otherwise Matplotlib."""

    picked = QtCore.Signal(object)  # np.ndarray shape (3,)
    landmark_selected = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.backend = "pyvista" if HAS_PYVISTA else "matplotlib"
        self._xyz: Optional[np.ndarray] = None
        self._sc = None
        self._plotter = None
        self._point_actor = None
        self._surface_actor = None
        self._unit_label = "scene"
        self._center_on_click = False
        self._corner_axes_visible = True
        self._render_profile = "balanced"
        self._interaction_lod_active = False
        self._interaction_observers_added = False
        self._current_point_colors: Optional[np.ndarray] = None
        self._current_surface_triangles: Optional[np.ndarray] = None
        self._current_surface_colors: Optional[np.ndarray] = None
        self._current_surface_texture_payload: Optional[dict] = None
        self._current_show_points = True
        self._current_show_surface = True
        self._scale_preview_active = False
        self._scale_anchor: Optional[np.ndarray] = None
        self._mouse_move_observer_added = False
        self._cached_surface_dataset = None
        self._cached_surface_interaction_dataset = None
        self._cached_surface_texture_dataset = None
        self._cached_surface_texture = None
        self._cached_point_dataset = None
        self._surface_visual_state: Optional[dict[str, float]] = None
        self._point_visual_state: Optional[dict[str, float]] = None
        self._hybrid_surface_points = False
        self._landmark_points = np.empty((0, 3), dtype=np.float32)
        self._landmark_labels: list[str] = []
        self._selected_landmark_label: Optional[str] = None
        self._landmark_actor = None
        self._selected_landmark_actor = None
        self._landmark_label_actor = None

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(self._build_controls())

        if HAS_PYVISTA:
            self._plotter = QtInteractor(self)
            self._configure_pyvista_scene()
            lay.addWidget(self._plotter.interactor if hasattr(self._plotter, "interactor") else self._plotter)
            return

        if not HAS_MPL:
            lay.addWidget(QtWidgets.QLabel("No 3D viewer backend is available."))
            self.fig = None
            self.canvas = None
            self.ax = None
            return

        self.fig = Figure(figsize=(6, 5))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.toolbar.setToolTip("Use the toolbar or mouse to zoom, pan, and reset the 3D scene.")
        self.toolbar.setIconSize(QtCore.QSize(18, 18))
        self.toolbar.setStyleSheet(
            """
            QToolBar {
                background: #eef3f8;
                border: 1px solid #d7e0ea;
                border-radius: 10px;
                spacing: 4px;
                padding: 6px 8px;
            }
            QToolButton {
                background: #ffffff;
                border: 1px solid #d9e2ec;
                border-radius: 8px;
                padding: 6px;
                margin: 1px;
            }
            QToolButton:hover {
                background: #f4f8fc;
                border-color: #b8c7d8;
            }
            QToolButton:pressed {
                background: #e6eef7;
                border-color: #9db2c8;
            }
            """
        )
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas)
        self.canvas.mpl_connect("pick_event", self._on_mpl_pick)

    def _build_controls(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QFrame(self)
        panel.setStyleSheet(
            """
            QFrame {
                background: #edf3f8;
                border: 1px solid #d7e1eb;
                border-radius: 12px;
            }
            QPushButton, QToolButton {
                background: #ffffff;
                color: #203044;
                border: 1px solid #c9d6e2;
                border-radius: 9px;
                padding: 6px 10px;
                font-weight: 600;
            }
            QPushButton:hover, QToolButton:hover {
                background: #f6f9fc;
                border-color: #aebfd0;
            }
            QPushButton:pressed, QToolButton:pressed {
                background: #e6eef7;
                border-color: #95abc2;
            }
            QLabel {
                color: #53657a;
                font-size: 12px;
            }
            """
        )
        row = QtWidgets.QHBoxLayout(panel)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(8)

        specs = [
            ("Reset View", self.reset_view),
            ("Zoom In", lambda: self.zoom_view(0.82)),
            ("Zoom Out", lambda: self.zoom_view(1.22)),
            ("Fit", self.fit_scene),
            ("Iso", self.view_isometric),
            ("Top", self.view_top),
            ("Front", self.view_front),
            ("Left", self.view_left),
        ]
        for label, handler in specs:
            btn = QtWidgets.QToolButton()
            btn.setText(label)
            btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
            btn.setAutoRaise(False)
            btn.clicked.connect(handler)
            row.addWidget(btn)
        self.btn_center_on_click = QtWidgets.QToolButton()
        self.btn_center_on_click.setText("Center View on Click")
        self.btn_center_on_click.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.btn_center_on_click.setAutoRaise(False)
        self.btn_center_on_click.setCheckable(True)
        self.btn_center_on_click.toggled.connect(self._set_center_on_click)
        row.addWidget(self.btn_center_on_click)
        self.btn_corner_axes = QtWidgets.QToolButton()
        self.btn_corner_axes.setText("Corner XYZ")
        self.btn_corner_axes.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.btn_corner_axes.setAutoRaise(False)
        self.btn_corner_axes.setCheckable(True)
        self.btn_corner_axes.setChecked(True)
        self.btn_corner_axes.toggled.connect(self._set_corner_axes_visible)
        row.addWidget(self.btn_corner_axes)
        row.addStretch(1)
        row.addWidget(QtWidgets.QLabel("Rotate: drag  |  Zoom: wheel  |  Pan: shift-drag"))
        return panel

    def _configure_pyvista_scene(self) -> None:
        if self._plotter is None:
            return
        try:
            self._plotter.set_background("#f8fafc", top="#edf3f8")
        except Exception:
            self._plotter.set_background("#f8fafc")
        if self._render_profile == "quality":
            try:
                self._plotter.enable_anti_aliasing()
            except Exception:
                pass
            try:
                self._plotter.enable_depth_peeling()
            except Exception:
                pass
        try:
            self._plotter.enable_trackball_style()
        except Exception:
            pass
        self._ensure_pyvista_mouse_tracking()
        self._ensure_pyvista_interaction_observers()
        self._add_pyvista_overlays()

    def _add_pyvista_overlays(self) -> None:
        if self._plotter is None:
            return
        try:
            self._plotter.add_text(
                "Surgiplot 3D Scene",
                position="upper_left",
                font_size=11,
                color="#203044",
                name="scene_title",
            )
        except Exception:
            pass
        try:
            self._plotter.add_text(
                "Rotate: drag | Zoom: wheel/right-drag | Pan: shift-drag",
                position="lower_left",
                font_size=9,
                color="#475569",
                name="scene_help",
            )
        except Exception:
            pass
        self._set_pyvista_bounds()
        try:
            self._plotter.add_axes(
                interactive=False,
                line_width=2,
                color="#334155",
                xlabel="X",
                ylabel="Y",
                zlabel="Z",
                viewport=(0.0, 0.0, 0.18, 0.18),
            )
            if self._corner_axes_visible:
                self._plotter.show_axes()
            else:
                self._plotter.hide_axes()
        except Exception:
            pass

    def _set_pyvista_bounds(self) -> None:
        if self._plotter is None:
            return
        try:
            self._plotter.show_bounds(
                grid="back",
                location="outer",
                all_edges=True,
                xtitle=f"X ({self._unit_label})",
                ytitle=f"Y ({self._unit_label})",
                ztitle=f"Z ({self._unit_label})",
                font_size=9,
                color="#7b8aa0",
                fmt="%.3f",
                minor_ticks=False,
            )
        except Exception:
            pass

    def set_render_profile(self, profile: str) -> None:
        profile = str(profile or "balanced").strip().lower()
        if profile not in {"fast", "balanced", "quality"}:
            profile = "balanced"
        self._render_profile = profile

    def _ensure_pyvista_mouse_tracking(self) -> None:
        if self._plotter is None or self._mouse_move_observer_added:
            return
        interactor = getattr(self._plotter, "iren", None)
        if interactor is None:
            interactor = getattr(self._plotter, "interactor", None)
        if interactor is None:
            return
        try:
            interactor.add_observer("MouseMoveEvent", self._on_pyvista_mouse_move)
            self._mouse_move_observer_added = True
        except Exception:
            pass

    def _ensure_pyvista_interaction_observers(self) -> None:
        if self._plotter is None or self._interaction_observers_added:
            return
        interactor = getattr(self._plotter, "iren", None)
        if interactor is None:
            interactor = getattr(self._plotter, "interactor", None)
        if interactor is None:
            return
        try:
            interactor.add_observer("StartInteractionEvent", self._on_pyvista_interaction_start)
            interactor.add_observer("EndInteractionEvent", self._on_pyvista_interaction_end)
            self._interaction_observers_added = True
        except Exception:
            pass

    def _enable_pyvista_surface_picking(self) -> None:
        if self._plotter is None:
            return
        try:
            self._plotter.enable_surface_point_picking(
                callback=self._on_pyvista_pick,
                show_message=False,
                show_point=False,
                tolerance=0.0025,
                pickable_window=False,
                left_clicking=True,
                use_picker=False,
                clear_on_no_selection=False,
            )
        except Exception:
            pass

    def _enable_pyvista_point_picking(self) -> None:
        if self._plotter is None:
            return
        try:
            self._plotter.enable_point_picking(
                callback=self._on_pyvista_pick,
                show_message=False,
                use_picker=False,
                show_point=False,
                pickable_window=False,
                tolerance=0.025,
                left_clicking=True,
                clear_on_no_selection=False,
            )
        except Exception:
            pass

    def _style_axes(self, unit_label: str) -> None:
        self.ax.set_xlabel(f"X ({unit_label})", labelpad=10)
        self.ax.set_ylabel(f"Y ({unit_label})", labelpad=10)
        self.ax.set_zlabel(f"Z ({unit_label})", labelpad=10)
        self.ax.grid(True, alpha=0.10, linewidth=0.5, color="#94a3b8")
        try:
            for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
                axis.pane.set_facecolor((0.985, 0.989, 0.995, 0.92))
                axis.pane.set_edgecolor((0.88, 0.91, 0.95, 0.45))
        except Exception:
            pass

    def _normalize_colors(self, colors: Optional[np.ndarray], count: int) -> Optional[np.ndarray]:
        if colors is None:
            return None
        arr = np.asarray(colors)
        if arr.ndim != 2 or arr.shape[0] != count or arr.shape[1] != 3:
            raise ValueError("colors must be (N,3) matching the geometry")
        if arr.dtype.kind in ("u", "i"):
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = np.clip(arr.astype(np.float32), 0.0, 1.0)
        return arr

    def _build_cached_pyvista_datasets(self) -> None:
        self._cached_surface_dataset = None
        self._cached_surface_interaction_dataset = None
        self._cached_surface_texture_dataset = None
        self._cached_surface_texture = None
        self._cached_point_dataset = None
        if pv is None:
            return

        if self._current_surface_texture_payload is not None:
            textured_dataset, texture = _build_textured_surface_dataset(self._current_surface_texture_payload)
            self._cached_surface_texture_dataset = textured_dataset
            self._cached_surface_texture = texture

        if self._current_surface_triangles is not None:
            tris = np.asarray(self._current_surface_triangles, dtype=np.float32)
            if tris.ndim == 3 and tris.shape[1:] == (3, 3) and len(tris) > 0:
                verts, faces, tri_indices, face_cols, vertex_cols = _surface_triangles_to_indexed_mesh(tris, self._current_surface_colors)
                face_cols = self._normalize_colors(face_cols, len(tris)) if face_cols is not None else None
                vertex_cols = self._normalize_colors(vertex_cols, len(verts)) if vertex_cols is not None else None
                projected_cols = _project_vertex_colors_from_cloud(verts, self._xyz, self._current_point_colors)
                if projected_cols is not None:
                    if vertex_cols is None:
                        vertex_cols = projected_cols
                    else:
                        vertex_cols = np.clip(0.25 * vertex_cols + 0.75 * projected_cols, 0.0, 1.0)
                vertex_cols = _smooth_vertex_colors(vertex_cols, tri_indices, iterations=2, blend=0.34)
                surf = _prepare_surface_dataset(verts, faces, face_cols, vertex_cols)
                self._cached_surface_dataset = surf
                lod_tris, lod_cols = self._downsample_surface_for_interaction(tris, self._current_surface_colors)
                if len(lod_tris) > 0 and len(lod_tris) < len(tris):
                    lod_verts, lod_faces, lod_tri_indices, lod_face_cols, lod_vertex_cols = _surface_triangles_to_indexed_mesh(lod_tris, lod_cols)
                    lod_face_cols = self._normalize_colors(lod_face_cols, len(lod_tris)) if lod_face_cols is not None else None
                    lod_vertex_cols = self._normalize_colors(lod_vertex_cols, len(lod_verts)) if lod_vertex_cols is not None else None
                    lod_projected_cols = _project_vertex_colors_from_cloud(lod_verts, self._xyz, self._current_point_colors, max_cloud_points=80000)
                    if lod_projected_cols is not None:
                        if lod_vertex_cols is None:
                            lod_vertex_cols = lod_projected_cols
                        else:
                            lod_vertex_cols = np.clip(0.20 * lod_vertex_cols + 0.80 * lod_projected_cols, 0.0, 1.0)
                    lod_vertex_cols = _smooth_vertex_colors(lod_vertex_cols, lod_tri_indices, iterations=1, blend=0.24)
                    lod_surf = _prepare_surface_dataset(lod_verts, lod_faces, lod_face_cols, lod_vertex_cols)
                    self._cached_surface_interaction_dataset = lod_surf

        if self._current_show_points and self._xyz is not None:
            cloud = pv.PolyData(self._xyz)
            if self._current_point_colors is not None:
                cloud.point_data["rgb"] = np.clip(self._current_point_colors * 255.0, 0, 255).astype(np.uint8)
            self._cached_point_dataset = cloud

    def _swap_surface_dataset(self, interaction: bool) -> None:
        if self._surface_actor is None:
            return
        mapper = getattr(self._surface_actor, "mapper", None)
        if mapper is None and hasattr(self._surface_actor, "GetMapper"):
            try:
                mapper = self._surface_actor.GetMapper()
            except Exception:
                mapper = None
        if mapper is None:
            return
        dataset = self._cached_surface_interaction_dataset if interaction else self._cached_surface_dataset
        if dataset is None:
            return
        try:
            if hasattr(mapper, "SetInputDataObject"):
                mapper.SetInputDataObject(dataset)
            elif hasattr(mapper, "SetInputData"):
                mapper.SetInputData(dataset)
            elif hasattr(mapper, "dataset"):
                mapper.dataset = dataset
            if hasattr(mapper, "Modified"):
                mapper.Modified()
        except Exception:
            pass

    def _apply_interaction_render_state(self, active: bool) -> None:
        if self._plotter is None:
            return
        try:
            interactor = getattr(self._plotter, "iren", None) or getattr(self._plotter, "interactor", None)
            if interactor is not None and hasattr(interactor, "SetDesiredUpdateRate"):
                interactor.SetDesiredUpdateRate(30.0 if active else 0.1)
        except Exception:
            pass

        if self._surface_actor is not None:
            try:
                prop = self._surface_actor.GetProperty()
                if active:
                    self._surface_visual_state = {
                        "ambient": float(prop.GetAmbient()),
                        "diffuse": float(prop.GetDiffuse()),
                        "specular": float(prop.GetSpecular()),
                        "specular_power": float(prop.GetSpecularPower()),
                        "opacity": float(prop.GetOpacity()),
                        "interpolation": float(prop.GetInterpolation()),
                    }
                    prop.SetInterpolationToFlat()
                    prop.SetAmbient(0.12)
                    prop.SetDiffuse(0.88)
                    prop.SetSpecular(0.0)
                    prop.SetSpecularPower(1.0)
                    prop.SetOpacity(min(float(prop.GetOpacity()), 0.92))
                elif self._surface_visual_state is not None:
                    prop.SetAmbient(self._surface_visual_state["ambient"])
                    prop.SetDiffuse(self._surface_visual_state["diffuse"])
                    prop.SetSpecular(self._surface_visual_state["specular"])
                    prop.SetSpecularPower(self._surface_visual_state["specular_power"])
                    prop.SetOpacity(self._surface_visual_state["opacity"])
                    try:
                        prop.SetInterpolation(int(self._surface_visual_state["interpolation"]))
                    except Exception:
                        pass
            except Exception:
                pass

        if self._point_actor is not None:
            try:
                prop = self._point_actor.GetProperty()
                if active:
                    self._point_visual_state = {
                        "opacity": float(prop.GetOpacity()),
                        "point_size": float(prop.GetPointSize()),
                    }
                    prop.SetOpacity(min(float(prop.GetOpacity()), 0.85))
                    prop.SetPointSize(max(2.0, float(prop.GetPointSize()) * 0.9))
                    if hasattr(prop, "SetRenderPointsAsSpheres"):
                        prop.SetRenderPointsAsSpheres(False)
                elif self._point_visual_state is not None:
                    prop.SetOpacity(self._point_visual_state["opacity"])
                    prop.SetPointSize(self._point_visual_state["point_size"])
                    if hasattr(prop, "SetRenderPointsAsSpheres"):
                        prop.SetRenderPointsAsSpheres(True)
            except Exception:
                pass

        try:
            self._plotter.render()
        except Exception:
            pass

    def set_cloud(
        self,
        xyz: np.ndarray,
        colors: Optional[np.ndarray] = None,
        surface_triangles: Optional[np.ndarray] = None,
        surface_colors: Optional[np.ndarray] = None,
        surface_texture_payload: Optional[dict] = None,
        show_points: bool = True,
        show_surface: bool = True,
        unit_label: str = "m",
    ) -> None:
        xyz = np.asarray(xyz, dtype=np.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must be (N,3)")
        self._xyz = xyz
        self._unit_label = unit_label
        point_colors = self._normalize_colors(colors, len(xyz))
        self._current_point_colors = point_colors
        self._current_surface_triangles = None if surface_triangles is None else np.asarray(surface_triangles, dtype=np.float32)
        self._current_surface_colors = None if surface_colors is None else np.asarray(surface_colors)
        self._current_surface_texture_payload = surface_texture_payload if isinstance(surface_texture_payload, dict) else None
        self._current_show_points = bool(show_points)
        self._current_show_surface = bool(show_surface)
        self._interaction_lod_active = False
        self._hybrid_surface_points = False

        if HAS_PYVISTA and self._plotter is not None:
            self._build_cached_pyvista_datasets()
            self._set_cloud_pyvista(
                self._xyz,
                self._current_point_colors,
                self._current_surface_triangles,
                self._current_surface_colors,
                show_points,
                show_surface,
                unit_label,
            )
            return

        if not HAS_MPL:
            return

        self._set_cloud_matplotlib(
            xyz,
            point_colors,
            surface_triangles,
            surface_colors,
            show_points,
            show_surface,
            unit_label,
        )

    def _set_cloud_pyvista(
        self,
        xyz: np.ndarray,
        point_colors: Optional[np.ndarray],
        surface_triangles: Optional[np.ndarray],
        surface_colors: Optional[np.ndarray],
        show_points: bool,
        show_surface: bool,
        unit_label: str,
        preserve_camera: bool = False,
    ) -> None:
        assert self._plotter is not None and pv is not None
        camera_position = None
        if preserve_camera:
            try:
                camera_position = self._plotter.camera_position
            except Exception:
                camera_position = None
        self._plotter.clear()
        self._configure_pyvista_scene()

        if (
            show_surface
            and self._cached_surface_texture_dataset is not None
            and self._cached_surface_texture is not None
        ):
            self._surface_actor = self._plotter.add_mesh(
                self._cached_surface_texture_dataset,
                texture=self._cached_surface_texture,
                opacity=0.99 if self._render_profile == "quality" else 0.97,
                show_edges=False,
                smooth_shading=True,
                ambient=0.24,
                diffuse=0.86,
                specular=0.02,
                specular_power=8,
                interpolate_before_map=True,
                name="scene_surface",
            )
        elif show_surface and surface_triangles is not None:
            tris = np.asarray(surface_triangles, dtype=np.float32)
            if tris.ndim == 3 and tris.shape[1:] == (3, 3) and len(tris) > 0:
                face_cols = None
                vertex_cols = None
                if surface_triangles is self._current_surface_triangles and self._cached_surface_dataset is not None:
                    surf = self._cached_surface_dataset
                    if "rgb" in surf.point_data:
                        vertex_cols = np.asarray(surf.point_data["rgb"], dtype=np.uint8)
                    if "rgb" in surf.cell_data:
                        face_cols = np.asarray(surf.cell_data["rgb"], dtype=np.uint8)
                else:
                    verts, faces, tri_indices, face_cols_arr, vertex_cols_arr = _surface_triangles_to_indexed_mesh(tris, surface_colors)
                    face_cols = self._normalize_colors(face_cols_arr, len(tris)) if face_cols_arr is not None else None
                    vertex_cols = self._normalize_colors(vertex_cols_arr, len(verts)) if vertex_cols_arr is not None else None
                    projected_cols = _project_vertex_colors_from_cloud(verts, xyz, point_colors)
                    if projected_cols is not None:
                        if vertex_cols is None:
                            vertex_cols = projected_cols
                        else:
                            vertex_cols = np.clip(0.25 * vertex_cols + 0.75 * projected_cols, 0.0, 1.0)
                    vertex_cols = _smooth_vertex_colors(vertex_cols, tri_indices, iterations=2, blend=0.34)
                    surf = _prepare_surface_dataset(verts, faces, face_cols, vertex_cols)
                if vertex_cols is not None:
                    smooth = True
                    opacity = 0.98 if self._render_profile == "quality" else 0.95
                    self._surface_actor = self._plotter.add_mesh(
                        surf,
                        scalars="rgb",
                        rgb=True,
                        opacity=opacity,
                        show_edges=False,
                        smooth_shading=smooth,
                        ambient=0.24 if self._render_profile == "fast" else 0.30,
                        diffuse=0.82 if self._render_profile == "fast" else 0.86,
                        specular=0.00 if self._render_profile == "fast" else 0.03,
                        specular_power=4 if self._render_profile == "fast" else 10,
                        interpolate_before_map=True,
                        preference="point",
                        name="scene_surface",
                    )
                elif face_cols is not None:
                    if "rgb" not in surf.cell_data:
                        surf.cell_data["rgb"] = np.clip(face_cols * 255.0, 0, 255).astype(np.uint8)
                    smooth = True
                    opacity = 0.98 if self._render_profile == "quality" else 0.95
                    self._surface_actor = self._plotter.add_mesh(
                        surf,
                        scalars="rgb",
                        rgb=True,
                        opacity=opacity,
                        show_edges=False,
                        smooth_shading=smooth,
                        ambient=0.20 if self._render_profile == "fast" else 0.26,
                        diffuse=0.78 if self._render_profile == "fast" else 0.82,
                        specular=0.00 if self._render_profile == "fast" else 0.04,
                        specular_power=6 if self._render_profile == "fast" else 12,
                        interpolate_before_map=False,
                        preference="cell",
                        name="scene_surface",
                    )
                else:
                    smooth = self._render_profile == "quality"
                    self._surface_actor = self._plotter.add_mesh(
                        surf,
                        color="#b7c0d6",
                        opacity=0.96 if self._render_profile == "quality" else 0.92,
                        show_edges=False,
                        smooth_shading=smooth,
                        ambient=0.20 if self._render_profile == "fast" else 0.26,
                        diffuse=0.78 if self._render_profile == "fast" else 0.82,
                        specular=0.00 if self._render_profile == "fast" else 0.04,
                        specular_power=6 if self._render_profile == "fast" else 12,
                        name="scene_surface",
                    )

        if show_points:
            if xyz is self._xyz and self._cached_point_dataset is not None:
                cloud = self._cached_point_dataset
            else:
                cloud = pv.PolyData(xyz)
            if show_surface:
                point_size = 3.2 if self._render_profile == "fast" else 4.0
                point_opacity = 0.72 if self._render_profile == "fast" else 0.86
                render_as_spheres = True
            else:
                point_size = 6.5 if self._render_profile == "fast" else (7.8 if self._render_profile == "balanced" else 9.0)
                point_opacity = 1.0
                render_as_spheres = True
            if point_colors is not None:
                if "rgb" not in cloud.point_data or xyz is not self._xyz:
                    cloud.point_data["rgb"] = np.clip(point_colors * 255.0, 0, 255).astype(np.uint8)
                self._point_actor = self._plotter.add_mesh(
                    cloud,
                    scalars="rgb",
                    rgb=True,
                    point_size=point_size,
                    render_points_as_spheres=render_as_spheres,
                    opacity=point_opacity,
                    name="scene_points",
                )
            else:
                self._point_actor = self._plotter.add_mesh(
                    cloud,
                    color="#94a3b8",
                    point_size=point_size,
                    render_points_as_spheres=render_as_spheres,
                    opacity=point_opacity,
                    name="scene_points",
                )
        self._set_pyvista_bounds()
        if self._hybrid_surface_points and self._point_actor is not None:
            try:
                self._point_actor.VisibilityOff()
            except Exception:
                pass
        if show_surface:
            self._enable_pyvista_surface_picking()
        elif show_points:
            self._enable_pyvista_point_picking()
        if camera_position is not None:
            try:
                self._plotter.camera_position = camera_position
                self._refresh_landmark_overlays()
                self._plotter.render()
                return
            except Exception:
                pass
        self.view_isometric()
        self.fit_scene()
        self._refresh_landmark_overlays()

    def _interaction_point_limit(self) -> int:
        if self._render_profile == "quality":
            return 30000
        if self._render_profile == "balanced":
            return 18000
        return 10000

    def _interaction_face_limit(self) -> int:
        if self._render_profile == "quality":
            return 14000
        if self._render_profile == "balanced":
            return 8000
        return 5000

    def _downsample_points_for_interaction(self, xyz: np.ndarray, colors: Optional[np.ndarray]) -> tuple[np.ndarray, Optional[np.ndarray]]:
        pts = np.asarray(xyz, dtype=np.float32)
        cols = None if colors is None else np.asarray(colors)
        limit = self._interaction_point_limit()
        if len(pts) <= limit:
            return pts, cols
        idx = np.linspace(0, len(pts) - 1, num=limit, dtype=int)
        return pts[idx], cols[idx] if cols is not None and len(cols) == len(pts) else cols

    def _downsample_surface_for_interaction(self, tris: np.ndarray, colors: Optional[np.ndarray]) -> tuple[np.ndarray, Optional[np.ndarray]]:
        surf = np.asarray(tris, dtype=np.float32)
        cols = None if colors is None else np.asarray(colors)
        limit = self._interaction_face_limit()
        if len(surf) <= limit:
            return surf, cols
        idx = np.linspace(0, len(surf) - 1, num=limit, dtype=int)
        return surf[idx], cols[idx] if cols is not None and len(cols) == len(surf) else cols

    def _on_pyvista_interaction_start(self, *_args) -> None:
        if self._plotter is None or self._interaction_lod_active:
            return
        if self._current_surface_triangles is None and self._xyz is None:
            return
        try:
            self._interaction_lod_active = True
            if self._hybrid_surface_points and self._surface_actor is not None and self._point_actor is not None:
                try:
                    self._surface_actor.VisibilityOff()
                    self._point_actor.VisibilityOn()
                except Exception:
                    pass
            elif self._current_show_surface and not self._current_show_points and self._current_surface_texture_payload is None:
                self._swap_surface_dataset(True)
            self._apply_interaction_render_state(True)
        except Exception:
            self._interaction_lod_active = False

    def _on_pyvista_interaction_end(self, *_args) -> None:
        if self._plotter is None or not self._interaction_lod_active:
            return
        try:
            self._apply_interaction_render_state(False)
            if self._hybrid_surface_points and self._surface_actor is not None and self._point_actor is not None:
                try:
                    self._point_actor.VisibilityOff()
                    self._surface_actor.VisibilityOn()
                except Exception:
                    pass
            elif self._current_show_surface and not self._current_show_points and self._current_surface_texture_payload is None:
                self._swap_surface_dataset(False)
            self._interaction_lod_active = False
        except Exception:
            pass

    def _set_cloud_matplotlib(
        self,
        xyz: np.ndarray,
        point_colors: Optional[np.ndarray],
        surface_triangles: Optional[np.ndarray],
        surface_colors: Optional[np.ndarray],
        show_points: bool,
        show_surface: bool,
        unit_label: str,
    ) -> None:
        self.ax.clear()
        self._style_axes(unit_label)

        if show_surface and surface_triangles is not None:
            tris = np.asarray(surface_triangles, dtype=np.float32)
            if tris.ndim == 3 and tris.shape[1:] == (3, 3) and len(tris) > 0:
                try:
                    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

                    facecolors = None
                    if surface_colors is not None:
                        sc = self._normalize_colors(surface_colors, len(tris))
                        if sc is not None:
                            facecolors = sc
                    mesh = Poly3DCollection(
                        list(tris),
                        facecolors=facecolors if facecolors is not None else (0.72, 0.76, 0.84, 0.92),
                        edgecolors=(0.36, 0.40, 0.48, 0.10),
                        linewidths=0.03,
                        alpha=0.82,
                    )
                    self.ax.add_collection3d(mesh)
                except Exception:
                    pass

        point_alpha = 0.82 if show_points else 0.0
        point_size = 7.5 if show_points else 18
        self._sc = self.ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            s=point_size,
            alpha=point_alpha,
            c=point_colors if point_colors is not None else None,
            linewidths=0.0,
            depthshade=True,
            picker=6,
        )
        self.ax.view_init(elev=20, azim=-60)
        if len(self._landmark_points) > 0:
            self.ax.scatter(
                self._landmark_points[:, 0],
                self._landmark_points[:, 1],
                self._landmark_points[:, 2],
                s=46,
                c="#1d4ed8",
                edgecolors="white",
                linewidths=0.8,
                depthshade=False,
            )
            if self._selected_landmark_label and self._selected_landmark_label in self._landmark_labels:
                idx = self._landmark_labels.index(self._selected_landmark_label)
                pt = self._landmark_points[idx]
                self.ax.scatter(
                    [pt[0]],
                    [pt[1]],
                    [pt[2]],
                    s=78,
                    c="#f97316",
                    edgecolors="white",
                    linewidths=1.0,
                    depthshade=False,
                )
        try:
            mins = xyz.min(axis=0)
            maxs = xyz.max(axis=0)
            mids = (mins + maxs) * 0.5
            spans = np.maximum(maxs - mins, 1e-6)
            radius = 0.55 * float(np.max(spans))
            self.ax.set_xlim3d([mids[0] - radius, mids[0] + radius])
            self.ax.set_ylim3d([mids[1] - radius, mids[1] + radius])
            self.ax.set_zlim3d([mids[2] - radius, mids[2] + radius])
        except Exception:
            pass
        self.canvas.draw_idle()

    def _on_pyvista_pick(self, *args) -> None:
        if not args:
            return
        point = np.asarray(args[0], dtype=float).reshape(-1)
        if point.size >= 3:
            self._maybe_select_existing_landmark(point[:3])
            if self._center_on_click:
                self._center_pyvista_view(point[:3])
            self.picked.emit(point[:3].copy())

    def _on_pyvista_mouse_move(self, *_args) -> None:
        if not self._scale_preview_active or self._scale_anchor is None or self._plotter is None or pv is None:
            return
        try:
            point = np.asarray(self._plotter.pick_mouse_position(), dtype=float).reshape(-1)
        except Exception:
            return
        if point.size < 3 or not np.all(np.isfinite(point[:3])):
            return
        self._draw_scale_preview_line(point[:3])

    def _on_mpl_pick(self, event) -> None:
        if self._xyz is None:
            return
        ind = getattr(event, "ind", None)
        if ind is None or len(ind) == 0:
            return
        point = self._xyz[int(ind[0])].copy()
        if self._center_on_click:
            self._center_mpl_view(point)
        self.picked.emit(point)

    def _set_center_on_click(self, enabled: bool) -> None:
        self._center_on_click = bool(enabled)

    def _set_corner_axes_visible(self, enabled: bool) -> None:
        self._corner_axes_visible = bool(enabled)
        if HAS_PYVISTA and self._plotter is not None:
            try:
                if enabled:
                    self._plotter.show_axes()
                else:
                    self._plotter.hide_axes()
                self._plotter.render()
            except Exception:
                pass

    def begin_scale_preview(self) -> None:
        self._scale_preview_active = True
        self._scale_anchor = None
        self._clear_scale_preview_line()

    def set_scale_preview_anchor(self, point: np.ndarray) -> None:
        self._scale_preview_active = True
        self._scale_anchor = np.asarray(point, dtype=float).reshape(3,)
        self._draw_scale_preview_line(self._scale_anchor)

    def clear_scale_preview(self) -> None:
        self._scale_preview_active = False
        self._scale_anchor = None
        self._clear_scale_preview_line()

    def set_landmark_points(
        self,
        points: np.ndarray,
        labels: list[str],
        selected_label: Optional[str] = None,
    ) -> None:
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) != len(labels):
            pts = np.empty((0, 3), dtype=np.float32)
            labels = []
        self._landmark_points = pts
        self._landmark_labels = [str(lb) for lb in labels]
        self._selected_landmark_label = str(selected_label) if selected_label else None
        self._refresh_landmark_overlays()

    def _refresh_landmark_overlays(self) -> None:
        if HAS_PYVISTA and self._plotter is not None:
            self._clear_landmark_actors()
            if len(self._landmark_points) == 0 or pv is None:
                return
            try:
                cloud = pv.PolyData(self._landmark_points)
                self._landmark_actor = self._plotter.add_mesh(
                    cloud,
                    color="#1d4ed8",
                    point_size=14.0,
                    render_points_as_spheres=True,
                    opacity=1.0,
                    pickable=False,
                    name="scene_landmarks",
                )
                if self._selected_landmark_label and self._selected_landmark_label in self._landmark_labels:
                    idx = self._landmark_labels.index(self._selected_landmark_label)
                    selected_cloud = pv.PolyData(self._landmark_points[idx : idx + 1])
                    self._selected_landmark_actor = self._plotter.add_mesh(
                        selected_cloud,
                        color="#f97316",
                        point_size=20.0,
                        render_points_as_spheres=True,
                        opacity=1.0,
                        pickable=False,
                        name="scene_selected_landmark",
                    )
                self._plotter.render()
            except Exception:
                pass
            return
        if HAS_MPL and getattr(self, "canvas", None) is not None:
            self.canvas.draw_idle()

    def _clear_landmark_actors(self) -> None:
        if self._plotter is None:
            return
        for actor in (self._landmark_actor, self._selected_landmark_actor, self._landmark_label_actor):
            if actor is None:
                continue
            try:
                self._plotter.remove_actor(actor, render=False)
            except Exception:
                pass
        self._landmark_actor = None
        self._selected_landmark_actor = None
        self._landmark_label_actor = None

    def _maybe_select_existing_landmark(self, point: np.ndarray) -> None:
        if len(self._landmark_points) == 0 or not self._landmark_labels:
            return
        query = np.asarray(point, dtype=float).reshape(3,)
        d2 = np.sum((self._landmark_points.astype(float) - query) ** 2, axis=1)
        idx = int(np.argmin(d2))
        mins = np.min(self._landmark_points, axis=0)
        maxs = np.max(self._landmark_points, axis=0)
        diag = float(np.linalg.norm(maxs - mins))
        tol = max(0.75, 0.015 * diag)
        if float(np.sqrt(d2[idx])) <= tol:
            self.landmark_selected.emit(self._landmark_labels[idx])

    def _draw_scale_preview_line(self, point: np.ndarray) -> None:
        if self._plotter is None or pv is None or self._scale_anchor is None:
            return
        try:
            line = pv.Line(self._scale_anchor, np.asarray(point, dtype=float).reshape(3,))
            self._plotter.add_mesh(
                line,
                color="#ef4444",
                line_width=6,
                opacity=0.95,
                render_lines_as_tubes=True,
                name="scale_preview_line",
                reset_camera=False,
                render=True,
            )
        except Exception:
            pass

    def _clear_scale_preview_line(self) -> None:
        if self._plotter is None:
            return
        try:
            self._plotter.remove_actor("scale_preview_line", render=True)
        except Exception:
            try:
                self._plotter.render()
            except Exception:
                pass

    def _center_pyvista_view(self, point: np.ndarray) -> None:
        if self._plotter is None:
            return
        try:
            position = np.asarray(self._plotter.camera_position[0], dtype=float)
            focal = np.asarray(self._plotter.camera_position[1], dtype=float)
            delta = np.asarray(point, dtype=float).reshape(3,) - focal
            self._plotter.set_focus(point)
            self._plotter.set_position(position + delta, reset=False, render=True)
        except Exception:
            try:
                self._plotter.set_focus(point)
                self._plotter.render()
            except Exception:
                pass

    def _center_mpl_view(self, point: np.ndarray) -> None:
        if not HAS_MPL or self.ax is None or self.canvas is None:
            return
        try:
            xlim = np.asarray(self.ax.get_xlim3d(), dtype=float)
            ylim = np.asarray(self.ax.get_ylim3d(), dtype=float)
            zlim = np.asarray(self.ax.get_zlim3d(), dtype=float)
            radius = 0.5 * max(xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0])
            p = np.asarray(point, dtype=float).reshape(3,)
            self.ax.set_xlim3d([p[0] - radius, p[0] + radius])
            self.ax.set_ylim3d([p[1] - radius, p[1] + radius])
            self.ax.set_zlim3d([p[2] - radius, p[2] + radius])
            self.canvas.draw_idle()
        except Exception:
            pass

    def _reset_bounds_matplotlib(self) -> None:
        if self._xyz is None or not HAS_MPL or self.ax is None:
            return
        try:
            mins = self._xyz.min(axis=0)
            maxs = self._xyz.max(axis=0)
            mids = (mins + maxs) * 0.5
            spans = np.maximum(maxs - mins, 1e-6)
            radius = 0.55 * float(np.max(spans))
            self.ax.set_xlim3d([mids[0] - radius, mids[0] + radius])
            self.ax.set_ylim3d([mids[1] - radius, mids[1] + radius])
            self.ax.set_zlim3d([mids[2] - radius, mids[2] + radius])
        except Exception:
            pass

    def _set_mpl_view(self, elev: float, azim: float) -> None:
        if not HAS_MPL or self.ax is None or self.canvas is None:
            return
        self.ax.view_init(elev=elev, azim=azim)
        self._style_axes(self._unit_label)
        self.canvas.draw_idle()

    def reset_view(self) -> None:
        if HAS_PYVISTA and self._plotter is not None:
            self.view_isometric()
            self.fit_scene()
            return
        self._reset_bounds_matplotlib()
        self._set_mpl_view(20, -60)

    def fit_scene(self) -> None:
        if HAS_PYVISTA and self._plotter is not None:
            try:
                self._plotter.reset_camera()
            except Exception:
                pass
            return
        self._reset_bounds_matplotlib()
        if HAS_MPL and self.canvas is not None:
            self.canvas.draw_idle()

    def view_isometric(self) -> None:
        if HAS_PYVISTA and self._plotter is not None:
            try:
                self._plotter.view_isometric()
            except Exception:
                pass
            return
        self._set_mpl_view(20, -60)

    def view_top(self) -> None:
        if HAS_PYVISTA and self._plotter is not None:
            try:
                self._plotter.view_xy()
            except Exception:
                pass
            return
        self._set_mpl_view(90, -90)

    def view_front(self) -> None:
        if HAS_PYVISTA and self._plotter is not None:
            try:
                self._plotter.view_xz()
            except Exception:
                pass
            return
        self._set_mpl_view(0, -90)

    def view_left(self) -> None:
        if HAS_PYVISTA and self._plotter is not None:
            try:
                self._plotter.view_yz()
            except Exception:
                pass
            return
        self._set_mpl_view(0, 180)

    def zoom_view(self, factor: float) -> None:
        factor = float(max(factor, 1e-3))
        if HAS_PYVISTA and self._plotter is not None:
            try:
                camera = getattr(self._plotter, "camera", None)
                if camera is not None and hasattr(camera, "Zoom"):
                    camera.Zoom(1.0 / factor)
                    self._plotter.render()
                    return
            except Exception:
                pass
            try:
                self._plotter.reset_camera_clipping_range()
                self._plotter.render()
            except Exception:
                pass
            return
        if not HAS_MPL or self.ax is None or self.canvas is None:
            return
        try:
            xlim = np.asarray(self.ax.get_xlim3d(), dtype=float)
            ylim = np.asarray(self.ax.get_ylim3d(), dtype=float)
            zlim = np.asarray(self.ax.get_zlim3d(), dtype=float)
            mids = np.array([xlim.mean(), ylim.mean(), zlim.mean()], dtype=float)
            xrad = 0.5 * (xlim[1] - xlim[0]) * factor
            yrad = 0.5 * (ylim[1] - ylim[0]) * factor
            zrad = 0.5 * (zlim[1] - zlim[0]) * factor
            self.ax.set_xlim3d([mids[0] - xrad, mids[0] + xrad])
            self.ax.set_ylim3d([mids[1] - yrad, mids[1] + yrad])
            self.ax.set_zlim3d([mids[2] - zrad, mids[2] + zrad])
            self.canvas.draw_idle()
        except Exception:
            pass
