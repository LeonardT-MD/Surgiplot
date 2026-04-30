"""
surgiplot.ai.scene_reconstruction

Import-only 3D scene workspace for the GUI.

Goals:
- Import a local point cloud, mesh, or 3D model.
- Rescale the entire scene from two picked 3D points.
- Collect 3D landmarks into the shared Dataset.
- Store a lightweight rendered scene in dataset metadata so downstream
  metric plots can overlay geometry on top of the imported scene.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import logging
import os
import platform
import struct
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PySide6 import QtCore, QtWidgets

from surgiplot.ai.scene_picker import HAS_MPL, HAS_PYVISTA, PointCloudPicker3D

LOG = logging.getLogger("surgiplot")


@dataclass
class RuntimeCheck:
    available: bool
    device: str
    missing_modules: List[str]
    message: str


@dataclass
class HardwareProfile:
    device: str
    cpu_count: int
    memory_gb: float
    machine: str
    platform_summary: str


@dataclass
class ReconstructionProfile:
    render_limit: int
    face_limit: int
    description: str
    render_mode: str = "balanced"


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _local_device_name() -> str:
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"} and platform.system().lower() == "darwin":
        return "apple-silicon"
    return "local"


def _hardware_profile() -> HardwareProfile:
    cpu_count = int(os.cpu_count() or 1)
    memory_gb = 0.0
    try:
        import psutil  # type: ignore

        memory_gb = float(psutil.virtual_memory().total) / (1024.0 ** 3)
    except Exception:
        memory_gb = 0.0
    machine = platform.machine()
    summary = f"{platform.system()} {platform.release()} | {machine} | {cpu_count} CPU"
    if memory_gb > 0:
        summary += f" | {memory_gb:.0f} GB RAM"
    return HardwareProfile(
        device=_local_device_name(),
        cpu_count=cpu_count,
        memory_gb=memory_gb,
        machine=machine,
        platform_summary=summary,
    )


def _recommended_profile(hardware: HardwareProfile) -> ReconstructionProfile:
    high_mem = hardware.memory_gb >= 32.0 if hardware.memory_gb > 0 else False
    if HAS_PYVISTA:
        if high_mem:
            return ReconstructionProfile(
                render_limit=180000,
                face_limit=0,
                description="PyVista preview is active. Balanced mode preserves the imported mesh by default while keeping interaction stable for typical local use.",
                render_mode="balanced",
            )
        return ReconstructionProfile(
            render_limit=90000,
            face_limit=0,
            description="PyVista preview is active. Balanced mode preserves the imported mesh by default while keeping interaction stable for typical local use.",
            render_mode="balanced",
        )
    if high_mem:
        return ReconstructionProfile(
            render_limit=60000,
            face_limit=30000,
            description="High-memory local preview stays closer to the original scene while still trimming the heaviest geometry for stable interaction.",
            render_mode="balanced",
        )
    return ReconstructionProfile(
        render_limit=36000,
        face_limit=18000,
        description="Standard local preview preserves more of the imported scene while still reducing the render load for reliable interaction.",
        render_mode="balanced",
    )


def check_runtime() -> RuntimeCheck:
    required_modules = ["numpy", "matplotlib"]
    missing = [name for name in required_modules if not _module_available(name)]
    device = _local_device_name()
    if missing:
        return RuntimeCheck(
            available=False,
            device=device,
            missing_modules=missing,
            message="The 3D scene workspace requires the local plotting stack.",
        )
    return RuntimeCheck(
        available=True,
        device=device,
        missing_modules=[],
        message="Imported 3D scene mode is ready for local scaling, landmark collection, and metric analysis.",
    )


def _downsample_cloud(points: np.ndarray, colors: np.ndarray, limit: int) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    cols = np.asarray(colors, dtype=float)
    if limit <= 0 or len(pts) <= limit:
        return pts, cols
    idx = np.linspace(0, len(pts) - 1, num=limit, dtype=int)
    return pts[idx], cols[idx]


def _has_meaningful_colors(colors: np.ndarray) -> bool:
    cols = np.asarray(colors, dtype=float)
    if cols.ndim != 2 or cols.shape[1] != 3 or len(cols) == 0:
        return False
    finite = np.all(np.isfinite(cols), axis=1)
    cols = cols[finite]
    if len(cols) == 0:
        return False
    spread = np.std(cols, axis=0)
    return bool(np.max(spread) >= 0.035)


def _geometry_colormap(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float).reshape(-1)
    if len(vals) == 0:
        return np.empty((0, 3), dtype=float)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-9:
        t = np.full(len(vals), 0.5, dtype=float)
    else:
        t = np.clip((vals - vmin) / (vmax - vmin), 0.0, 1.0)
    # Blue -> teal -> sand -> warm red
    anchors = np.array(
        [
            [0.16, 0.27, 0.64],
            [0.13, 0.60, 0.62],
            [0.86, 0.80, 0.56],
            [0.78, 0.38, 0.30],
        ],
        dtype=float,
    )
    bins = np.array([0.0, 0.38, 0.72, 1.0], dtype=float)
    out = np.empty((len(t), 3), dtype=float)
    for i, value in enumerate(t):
        if value <= bins[1]:
            local = (value - bins[0]) / max(bins[1] - bins[0], 1e-9)
            out[i] = anchors[0] * (1.0 - local) + anchors[1] * local
        elif value <= bins[2]:
            local = (value - bins[1]) / max(bins[2] - bins[1], 1e-9)
            out[i] = anchors[1] * (1.0 - local) + anchors[2] * local
        else:
            local = (value - bins[2]) / max(bins[3] - bins[2], 1e-9)
            out[i] = anchors[2] * (1.0 - local) + anchors[3] * local
    return np.clip(out, 0.0, 1.0)


def _apply_geometry_color_fallback(
    xyz: np.ndarray,
    colors: np.ndarray,
    surface: np.ndarray,
    surface_colors: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if _has_meaningful_colors(colors):
        tri_cols = np.asarray(surface_colors, dtype=float)
        if tri_cols.ndim == 2 and len(tri_cols) > 0 and _has_meaningful_colors(tri_cols):
            return colors, tri_cols
        if surface.ndim == 3 and len(surface) > 0:
            face_values = np.mean(surface[:, :, 2], axis=1)
            return colors, _geometry_colormap(face_values)
        return colors, surface_colors

    xyz = np.asarray(xyz, dtype=float)
    point_values = xyz[:, 2] if xyz.ndim == 2 and xyz.shape[1] == 3 else np.zeros((0,), dtype=float)
    point_colors = _geometry_colormap(point_values)
    if surface.ndim == 3 and len(surface) > 0:
        face_values = np.mean(surface[:, :, 2], axis=1)
        face_colors = _geometry_colormap(face_values)
    else:
        face_colors = np.empty((0, 3), dtype=float)
    return point_colors, face_colors


def _downsample_triangles(
    triangles: np.ndarray,
    colors: np.ndarray,
    limit: int,
) -> Tuple[np.ndarray, np.ndarray]:
    tris = np.asarray(triangles, dtype=float)
    cols = np.asarray(colors, dtype=float)
    if tris.ndim != 3 or tris.shape[1:] != (3, 3) or limit <= 0 or len(tris) <= limit:
        return tris, cols
    idx = np.linspace(0, len(tris) - 1, num=limit, dtype=int)
    return tris[idx], cols[idx] if len(cols) == len(tris) else cols


def _filter_scene_cloud(points: np.ndarray, colors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    cols = np.asarray(colors, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 32:
        return pts, cols

    finite = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite]
    cols = cols[finite]
    if len(pts) < 32:
        return pts, cols

    center = np.median(pts, axis=0)
    radii = np.linalg.norm(pts - center, axis=1)
    r_med = float(np.median(radii))
    mad = float(np.median(np.abs(radii - r_med))) + 1e-9
    robust_keep = radii <= (r_med + 6.0 * 1.4826 * mad)

    keep = robust_keep
    if int(np.count_nonzero(keep)) < max(24, len(pts) // 5):
        keep = np.ones(len(pts), dtype=bool)
    return pts[keep], cols[keep]


def _optimize_scene_preview(
    xyz: np.ndarray,
    colors: np.ndarray,
    surface: np.ndarray,
    surface_colors: np.ndarray,
    profile: ReconstructionProfile,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    xyz = np.asarray(xyz, dtype=float)
    colors = np.asarray(colors, dtype=float)
    surface = np.asarray(surface, dtype=float)
    surface_colors = np.asarray(surface_colors, dtype=float)
    colors, surface_colors = _apply_geometry_color_fallback(xyz, colors, surface, surface_colors)

    xyz, colors = _filter_scene_cloud(xyz, colors)
    raw_points = int(len(xyz))
    raw_faces = int(len(surface)) if surface.ndim == 3 else 0

    if raw_faces > 0:
        surface, surface_colors = _downsample_triangles(surface, surface_colors, limit=profile.face_limit)
        if HAS_PYVISTA:
            xyz = np.asarray(xyz, dtype=float)
            colors = np.asarray(colors, dtype=float)
        else:
            xyz, colors = _downsample_cloud(xyz, colors, limit=profile.render_limit)
    else:
        xyz, colors = _downsample_cloud(xyz, colors, limit=profile.render_limit)
        surface = np.empty((0, 3, 3), dtype=float)
        surface_colors = np.empty((0, 3), dtype=float)

    meta = {
        "raw_point_count": raw_points,
        "preview_point_count": int(len(xyz)),
        "raw_face_count": raw_faces,
        "preview_face_count": int(len(surface)) if surface.ndim == 3 else 0,
    }
    return xyz, colors, surface, surface_colors, meta


def _parse_point_rows(lines: Sequence[str]) -> np.ndarray:
    rows: List[List[float]] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p for p in line.replace(",", " ").split() if p]
        nums: List[float] = []
        for token in parts:
            try:
                nums.append(float(token))
            except Exception:
                nums = []
                break
        if len(nums) >= 3:
            rows.append(nums[:3])
    if len(rows) < 2:
        raise RuntimeError("Could not read enough 3D points from the selected file.")
    return np.asarray(rows, dtype=float)


def _load_xyz_scene(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        xyz = _parse_point_rows(f.readlines())
    colors = np.tile(np.array([[0.60, 0.67, 0.82]], dtype=float), (len(xyz), 1))
    return xyz, colors, np.empty((0, 3, 3), dtype=float), np.empty((0, 3), dtype=float)


def _load_image_rgb(path: str) -> Optional[np.ndarray]:
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return None
    try:
        img = Image.open(path).convert("RGB")
        return np.asarray(img, dtype=np.uint8)
    except Exception:
        return None


def _sample_texture_rgb(texture_rgb: np.ndarray, uv: Sequence[float]) -> np.ndarray:
    tex = np.asarray(texture_rgb, dtype=np.uint8)
    if tex.ndim != 3 or tex.shape[2] != 3:
        return np.array([0.74, 0.77, 0.83], dtype=float)
    u = float(uv[0]) if len(uv) > 0 else 0.0
    v = float(uv[1]) if len(uv) > 1 else 0.0
    u = u % 1.0
    v = v % 1.0
    h, w = tex.shape[:2]
    x = int(np.clip(round(u * (w - 1)), 0, w - 1))
    y = int(np.clip(round((1.0 - v) * (h - 1)), 0, h - 1))
    return tex[y, x, :].astype(float) / 255.0


def _parse_mtl_file(path: str) -> Dict[str, Dict[str, str]]:
    materials: Dict[str, Dict[str, str]] = {}
    current: Optional[str] = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            key = parts[0]
            value = parts[1].strip() if len(parts) > 1 else ""
            if key == "newmtl":
                current = value
                materials.setdefault(current, {})
            elif current is not None and key in {"map_Kd", "Kd"}:
                materials[current][key] = value
    return materials


def _triangulate_face(indices: List[int]) -> List[Tuple[int, int, int]]:
    if len(indices) < 3:
        return []
    if len(indices) == 3:
        return [(indices[0], indices[1], indices[2])]
    return [(indices[0], indices[i], indices[i + 1]) for i in range(1, len(indices) - 1)]


def _valid_face_indices(indices: Sequence[int], vertex_count: int) -> bool:
    if vertex_count <= 0:
        return False
    return all(0 <= idx < vertex_count for idx in indices)


def _ply_scalar_dtype(name: str) -> str:
    mapping = {
        "char": "i1",
        "int8": "i1",
        "uchar": "u1",
        "uint8": "u1",
        "short": "<i2",
        "int16": "<i2",
        "ushort": "<u2",
        "uint16": "<u2",
        "int": "<i4",
        "int32": "<i4",
        "uint": "<u4",
        "uint32": "<u4",
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "float64": "<f8",
    }
    if name not in mapping:
        raise RuntimeError(f"Unsupported PLY scalar type: {name}")
    return mapping[name]


def _load_obj_scene(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertices: List[List[float]] = []
    texcoords: List[List[float]] = []
    faces: List[Tuple[int, int, int]] = []
    face_texcoords: List[Optional[Tuple[int, int, int]]] = []
    mtl_libraries: List[str] = []
    active_material: Optional[str] = None
    face_materials: List[Optional[str]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("mtllib "):
                parts = line.split(maxsplit=1)
                if len(parts) > 1:
                    mtl_libraries.append(parts[1].strip())
            elif line.startswith("usemtl "):
                parts = line.split(maxsplit=1)
                active_material = parts[1].strip() if len(parts) > 1 else None
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("vt "):
                parts = line.split()
                if len(parts) >= 3:
                    texcoords.append([float(parts[1]), float(parts[2])])
            elif line.startswith("f "):
                face_indices = []
                face_uv_indices: List[int] = []
                for token in line.split()[1:]:
                    chunks = token.split("/")
                    head = chunks[0]
                    if not head:
                        continue
                    idx = int(head)
                    idx = len(vertices) + idx if idx < 0 else idx - 1
                    face_indices.append(idx)
                    if len(chunks) >= 2 and chunks[1]:
                        uv_idx = int(chunks[1])
                        uv_idx = len(texcoords) + uv_idx if uv_idx < 0 else uv_idx - 1
                        face_uv_indices.append(uv_idx)
                    else:
                        face_uv_indices.append(-1)
                tris = _triangulate_face(face_indices)
                faces.extend(tris)
                if len(face_uv_indices) == len(face_indices):
                    tri_uvs = _triangulate_face(face_uv_indices)
                    face_texcoords.extend([tuple(t) for t in tri_uvs])
                else:
                    face_texcoords.extend([None] * len(tris))
                face_materials.extend([active_material] * len(tris))
    if len(vertices) < 2:
        raise RuntimeError("The OBJ file does not contain enough vertices.")
    xyz = np.asarray(vertices, dtype=float)
    colors = np.tile(np.array([[0.75, 0.77, 0.82]], dtype=float), (len(xyz), 1))

    material_defs: Dict[str, Dict[str, str]] = {}
    obj_dir = os.path.dirname(path)
    for rel in mtl_libraries:
        mtl_path = os.path.join(obj_dir, rel)
        if os.path.exists(mtl_path):
            try:
                material_defs.update(_parse_mtl_file(mtl_path))
            except Exception:
                continue

    material_textures: Dict[str, np.ndarray] = {}
    material_kd: Dict[str, np.ndarray] = {}
    for name, meta in material_defs.items():
        kd = meta.get("Kd")
        if kd:
            parts = kd.split()
            if len(parts) >= 3:
                try:
                    material_kd[name] = np.clip(np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float), 0.0, 1.0)
                except Exception:
                    pass
        tex_rel = meta.get("map_Kd")
        if tex_rel:
            tex_path = os.path.join(obj_dir, tex_rel)
            tex_rgb = _load_image_rgb(tex_path)
            if tex_rgb is not None:
                material_textures[name] = tex_rgb

    valid_faces: List[Tuple[int, int, int]] = []
    valid_face_uvs: List[Optional[Tuple[int, int, int]]] = []
    valid_face_materials: List[Optional[str]] = []
    for i, face in enumerate(faces):
        if _valid_face_indices(face, len(xyz)):
            valid_faces.append(face)
            valid_face_uvs.append(face_texcoords[i] if i < len(face_texcoords) else None)
            valid_face_materials.append(face_materials[i] if i < len(face_materials) else None)

    tris = np.asarray([[xyz[a], xyz[b], xyz[c]] for a, b, c in valid_faces], dtype=float) if valid_faces else np.empty((0, 3, 3), dtype=float)
    tri_colors_list: List[np.ndarray] = []
    vertex_color_sum = np.zeros((len(xyz), 3), dtype=float)
    vertex_color_count = np.zeros((len(xyz), 1), dtype=float)
    for face, uv_face, material_name in zip(valid_faces, valid_face_uvs, valid_face_materials):
        face_color = None
        if material_name and material_name in material_textures and uv_face is not None and len(texcoords) > 0:
            tex = material_textures[material_name]
            samples: List[np.ndarray] = []
            valid_uv_face = True
            for uv_idx in uv_face:
                if uv_idx < 0 or uv_idx >= len(texcoords):
                    valid_uv_face = False
                    break
                samples.append(_sample_texture_rgb(tex, texcoords[uv_idx]))
            if valid_uv_face and samples:
                face_color = np.mean(np.vstack(samples), axis=0)
        if face_color is None and material_name and material_name in material_kd:
            face_color = material_kd[material_name]
        if face_color is None:
            face_color = np.array([0.74, 0.77, 0.83], dtype=float)
        tri_colors_list.append(np.asarray(face_color, dtype=float))
        for vid in face:
            vertex_color_sum[vid] += face_color
            vertex_color_count[vid, 0] += 1.0

    nonzero = vertex_color_count[:, 0] > 0
    if np.any(nonzero):
        colors[nonzero] = vertex_color_sum[nonzero] / vertex_color_count[nonzero]
    tri_colors = np.vstack(tri_colors_list) if tri_colors_list else np.empty((0, 3), dtype=float)
    return xyz, colors, tris, tri_colors


def _load_stl_scene(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        f.read(80)
        tri_count_bytes = f.read(4)
        if len(tri_count_bytes) == 4:
            tri_count = struct.unpack("<I", tri_count_bytes)[0]
            expected = 84 + tri_count * 50
            try:
                size = os.path.getsize(path)
            except Exception:
                size = -1
            if size == expected and tri_count > 0:
                tris = []
                for _ in range(tri_count):
                    block = f.read(50)
                    if len(block) < 50:
                        break
                    vals = struct.unpack("<12fH", block)
                    tris.append(
                        np.array(
                            [
                                [vals[3], vals[4], vals[5]],
                                [vals[6], vals[7], vals[8]],
                                [vals[9], vals[10], vals[11]],
                            ],
                            dtype=float,
                        )
                    )
                surface = np.asarray(tris, dtype=float)
                xyz = surface.reshape(-1, 3)
                colors = np.tile(np.array([[0.74, 0.77, 0.83]], dtype=float), (len(xyz), 1))
                tri_colors = np.tile(np.array([[0.74, 0.77, 0.83]], dtype=float), (len(surface), 1))
                return xyz, colors, surface, tri_colors

    vertices: List[List[float]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip().lower()
            if line.startswith("vertex "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if len(vertices) < 3:
        raise RuntimeError("The STL file does not contain enough geometry.")
    xyz = np.asarray(vertices, dtype=float)
    n = (len(xyz) // 3) * 3
    surface = xyz[:n].reshape(-1, 3, 3)
    colors = np.tile(np.array([[0.74, 0.77, 0.83]], dtype=float), (len(xyz), 1))
    tri_colors = np.tile(np.array([[0.74, 0.77, 0.83]], dtype=float), (len(surface), 1))
    return xyz, colors, surface, tri_colors


def _load_ply_scene(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError("Invalid PLY file header.")
            header_lines.append(line.decode("utf-8", errors="ignore").strip())
            if header_lines[-1] == "end_header":
                break
        fmt = "ascii"
        vertex_count = 0
        face_count = 0
        props: List[str] = []
        in_vertex = False
        face_count_type = "uchar"
        face_index_type = "int"
        for line in header_lines:
            if line.startswith("format "):
                fmt = line.split()[1]
            elif line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
                in_vertex = True
            elif line.startswith("element face "):
                face_count = int(line.split()[-1])
                in_vertex = False
            elif line.startswith("property ") and in_vertex:
                props.append(line.split()[-1])
            elif line.startswith("property list ") and not in_vertex:
                parts = line.split()
                if len(parts) >= 5:
                    face_count_type = parts[2]
                    face_index_type = parts[3]

        if vertex_count <= 0:
            raise RuntimeError("The PLY file does not contain vertices.")

        if fmt == "ascii":
            body = f.read().decode("utf-8", errors="ignore").splitlines()
            vertex_rows = body[:vertex_count]
            face_rows = body[vertex_count: vertex_count + face_count]
            data = [row.split()[: len(props)] for row in vertex_rows]
            arr = np.asarray(data)
            xyz = arr[:, :3].astype(float)
            colors = np.tile(np.array([[0.70, 0.74, 0.82]], dtype=float), (len(xyz), 1))
            if len(props) >= 6 and props[3:6] == ["red", "green", "blue"]:
                colors = arr[:, 3:6].astype(float) / 255.0
            tris = []
            for row in face_rows:
                parts = row.split()
                if not parts:
                    continue
                count = int(parts[0])
                idxs = [int(v) for v in parts[1:1 + count]]
                if not _valid_face_indices(idxs, len(xyz)):
                    continue
                for a, b, c in _triangulate_face(idxs):
                    tris.append([xyz[a], xyz[b], xyz[c]])
            surface = np.asarray(tris, dtype=float) if tris else np.empty((0, 3, 3), dtype=float)
            tri_colors = np.tile(np.array([[0.70, 0.74, 0.82]], dtype=float), (len(surface), 1)) if len(surface) else np.empty((0, 3), dtype=float)
            return xyz, colors, surface, tri_colors

        if fmt != "binary_little_endian":
            raise RuntimeError("Only ASCII and binary little-endian PLY are currently supported.")

        dtype_fields = []
        for prop in props:
            if prop in {"x", "y", "z"}:
                dtype_fields.append((prop, "<f4"))
            elif prop in {"red", "green", "blue"}:
                dtype_fields.append((prop, "u1"))
            else:
                dtype_fields.append((prop, "<f4"))
        vertex_dtype = np.dtype(dtype_fields)
        vertex_array = np.fromfile(f, dtype=vertex_dtype, count=vertex_count)
        xyz = np.column_stack([vertex_array["x"], vertex_array["y"], vertex_array["z"]]).astype(float)
        colors = np.tile(np.array([[0.70, 0.74, 0.82]], dtype=float), (len(xyz), 1))
        if all(name in vertex_array.dtype.names for name in ("red", "green", "blue")):
            colors = np.column_stack([vertex_array["red"], vertex_array["green"], vertex_array["blue"]]).astype(float) / 255.0
        tris = []
        count_fmt = np.dtype(_ply_scalar_dtype(face_count_type))
        index_fmt = np.dtype(_ply_scalar_dtype(face_index_type))
        for _ in range(face_count):
            count_bytes = f.read(count_fmt.itemsize)
            if len(count_bytes) != count_fmt.itemsize:
                break
            count = int(np.frombuffer(count_bytes, dtype=count_fmt, count=1)[0])
            if count < 0:
                continue
            index_bytes = f.read(index_fmt.itemsize * count)
            if len(index_bytes) != index_fmt.itemsize * count:
                break
            idxs = np.frombuffer(index_bytes, dtype=index_fmt, count=count).astype(int).tolist()
            if not _valid_face_indices(idxs, len(xyz)):
                continue
            for a, b, c in _triangulate_face(idxs):
                tris.append([xyz[a], xyz[b], xyz[c]])
        surface = np.asarray(tris, dtype=float) if tris else np.empty((0, 3, 3), dtype=float)
        tri_colors = np.tile(np.array([[0.70, 0.74, 0.82]], dtype=float), (len(surface), 1)) if len(surface) else np.empty((0, 3), dtype=float)
        return xyz, colors, surface, tri_colors


def _load_scene_file(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".xyz", ".pts", ".csv", ".txt"}:
        xyz, colors, surface, surface_colors = _load_xyz_scene(path)
    elif ext == ".obj":
        xyz, colors, surface, surface_colors = _load_obj_scene(path)
    elif ext == ".stl":
        xyz, colors, surface, surface_colors = _load_stl_scene(path)
    elif ext == ".ply":
        xyz, colors, surface, surface_colors = _load_ply_scene(path)
    else:
        raise RuntimeError("Unsupported 3D scene format. Supported: PLY, OBJ, STL, XYZ, CSV, TXT.")
    return xyz, colors, surface, surface_colors, {
        "backend": "imported_scene",
        "source_file": path,
        "format": ext.lstrip("."),
    }


class AISceneReconstructionDialog(QtWidgets.QDialog):
    def __init__(self, ds, on_dataset_changed_callback=None, parent=None):
        super().__init__(parent)
        if not HAS_MPL:
            raise RuntimeError("Matplotlib is required for the 3D scene workspace.")

        self.ds = ds
        self.on_dataset_changed_callback = on_dataset_changed_callback

        self.setWindowTitle("3D Scene Workspace")
        self.resize(1460, 900)
        self.setModal(True)

        self._merged_xyz_m: Optional[np.ndarray] = None
        self._merged_colors: Optional[np.ndarray] = None
        self._surface_triangles_mm: Optional[np.ndarray] = None
        self._surface_colors: Optional[np.ndarray] = None
        self._raw_xyz_m: Optional[np.ndarray] = None
        self._raw_colors: Optional[np.ndarray] = None
        self._raw_surface_triangles_m: Optional[np.ndarray] = None
        self._raw_surface_colors: Optional[np.ndarray] = None
        self._pending_points_m: Dict[str, np.ndarray] = {}
        self._pending_order: List[str] = []
        self._scene_scale_to_mm = 1.0
        self._calibration_summary = "Not calibrated"
        self._collect_mode = False
        self._hardware = _hardware_profile()
        self._active_profile = _recommended_profile(self._hardware)
        self._scene_backend = "imported_scene"
        self._reconstruction_meta: Dict[str, Any] = {}
        self._source_scene_paths: List[str] = []
        self._scale_pick_mode = False
        self._scale_pick_points: List[np.ndarray] = []
        self._last_picked_xyz_mm: Optional[np.ndarray] = None
        self._last_pick_time_s: float = 0.0

        self.btn_select_scene = QtWidgets.QPushButton("Select Scene…")
        self.btn_run = QtWidgets.QPushButton("Load Scene")
        self.btn_optimize = QtWidgets.QPushButton("Reload Preview")
        self.btn_set_scale = QtWidgets.QPushButton("Set Scale")
        self.btn_set_scale.setCheckable(True)
        self.btn_set_scale.setEnabled(False)
        self.btn_collect = QtWidgets.QPushButton("Collect Point")
        self.btn_collect.setCheckable(True)
        self.btn_collect.setEnabled(False)

        self.scale_cm_spin = QtWidgets.QDoubleSpinBox()
        self.scale_cm_spin.setRange(0.01, 500.0)
        self.scale_cm_spin.setDecimals(2)
        self.scale_cm_spin.setValue(1.00)
        self.scale_cm_spin.setSuffix(" cm")
        self.points_limit_spin = QtWidgets.QSpinBox()
        self.points_limit_spin.setRange(0, 500000)
        self.points_limit_spin.setSingleStep(5000)
        self.points_limit_spin.setValue(int(self._active_profile.render_limit))
        self.faces_limit_spin = QtWidgets.QSpinBox()
        self.faces_limit_spin.setRange(0, 250000)
        self.faces_limit_spin.setSingleStep(5000)
        self.faces_limit_spin.setValue(int(self._active_profile.face_limit))
        self.preview_mode_combo = QtWidgets.QComboBox()
        self.preview_mode_combo.addItems(["Fast", "Balanced", "Quality"])
        self.preview_mode_combo.setCurrentText("Balanced")

        self.lbl_runtime = QtWidgets.QLabel(
            "Load a point cloud, mesh, or 3D model, rescale it from two picked points, then collect landmarks for downstream analysis."
        )
        self.lbl_runtime.setWordWrap(True)

        self.list_sources = QtWidgets.QListWidget()
        self.list_sources.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list_sources.currentRowChanged.connect(self._on_source_selection_changed)

        self.lbl_source_info = QtWidgets.QLabel("No scene loaded.")
        self.lbl_source_info.setWordWrap(True)

        self.picker = PointCloudPicker3D()
        self.picker.picked.connect(self._on_picked_xyz)

        self.pending_table = QtWidgets.QTableWidget()
        self.pending_table.setColumnCount(4)
        self.pending_table.setHorizontalHeaderLabels(["Label", "X (scene)", "Y (scene)", "Z (scene)"])
        self.pending_table.horizontalHeader().setStretchLastSection(True)
        self.pending_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.pending_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

        self.btn_rename = QtWidgets.QPushButton("Rename")
        self.btn_delete = QtWidgets.QPushButton("Delete")
        self.btn_undo = QtWidgets.QPushButton("Undo Last")
        self.btn_clear = QtWidgets.QPushButton("Clear All")
        self.btn_ok = QtWidgets.QPushButton("Commit to Dataset")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_ok.setEnabled(False)

        controls = QtWidgets.QGridLayout()
        controls.addWidget(self.btn_select_scene, 0, 0)
        controls.addWidget(self.btn_run, 0, 1)
        controls.addWidget(self.btn_optimize, 0, 2)
        controls.addWidget(QtWidgets.QLabel("Point limit:"), 0, 3)
        controls.addWidget(self.points_limit_spin, 0, 4)
        controls.addWidget(QtWidgets.QLabel("Face limit:"), 0, 5)
        controls.addWidget(self.faces_limit_spin, 0, 6)
        controls.addWidget(QtWidgets.QLabel("Preview:"), 0, 7)
        controls.addWidget(self.preview_mode_combo, 0, 8)
        controls.addWidget(QtWidgets.QLabel("Known Distance:"), 1, 0)
        controls.addWidget(self.scale_cm_spin, 1, 1)
        controls.addWidget(self.btn_set_scale, 1, 2)
        controls.addWidget(self.btn_collect, 1, 3)

        left_panel = QtWidgets.QWidget()
        left_lay = QtWidgets.QVBoxLayout(left_panel)
        left_lay.addWidget(QtWidgets.QLabel("<b>Imported Scene</b>"))
        left_lay.addWidget(self.list_sources, 1)
        left_lay.addWidget(self.lbl_source_info)

        right_panel = QtWidgets.QWidget()
        right_lay = QtWidgets.QVBoxLayout(right_panel)
        right_lay.addWidget(QtWidgets.QLabel("<b>Pending Landmark Dataset</b>"))
        right_lay.addWidget(self.pending_table, 1)
        pending_actions = QtWidgets.QHBoxLayout()
        for btn in (self.btn_rename, self.btn_delete, self.btn_undo, self.btn_clear):
            pending_actions.addWidget(btn)
        pending_actions.addStretch(1)
        right_lay.addLayout(pending_actions)

        main_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_split.addWidget(left_panel)
        main_split.addWidget(self.picker)
        main_split.addWidget(right_panel)
        main_split.setStretchFactor(0, 1)
        main_split.setStretchFactor(1, 4)
        main_split.setStretchFactor(2, 2)
        main_split.setSizes([260, 840, 420])

        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(self.lbl_runtime, 1)
        bottom.addWidget(self.btn_cancel)
        bottom.addWidget(self.btn_ok)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(main_split, 1)
        layout.addLayout(bottom)

        self.btn_select_scene.clicked.connect(self._select_scene)
        self.btn_run.clicked.connect(self._run_reconstruction)
        self.btn_optimize.clicked.connect(self._apply_preview_settings)
        self.preview_mode_combo.currentTextChanged.connect(self._on_preview_mode_changed)
        self.btn_set_scale.toggled.connect(self._on_scale_mode_toggled)
        self.btn_collect.toggled.connect(self._set_collect_mode)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok.clicked.connect(self._commit_and_accept)
        self.btn_rename.clicked.connect(self._rename_selected)
        self.btn_delete.clicked.connect(self._delete_selected)
        self.btn_undo.clicked.connect(self._undo_last)
        self.btn_clear.clicked.connect(self._clear_all)
        self.pending_table.itemSelectionChanged.connect(self._on_pending_selection_changed)

        self._refresh_runtime_status()
        self._refresh_pending_table()

    def _coord_unit_label(self) -> str:
        return "mm" if self._calibration_summary != "Not calibrated" else "scene"

    def _refresh_runtime_status(self) -> None:
        status = check_runtime()
        hardware = self._hardware.platform_summary
        preview_mode = self.preview_mode_combo.currentText().strip()
        if status.available:
            self.lbl_runtime.setText(
                f"{status.message}\n"
                f"Source: Imported 3D Scene | Device: {status.device}\n"
                f"Viewer backend: {self.picker.backend}\n"
                f"Hardware: {hardware}\n"
                f"Preview mode: {preview_mode} | point limit: {int(self.points_limit_spin.value())} | face limit: {int(self.faces_limit_spin.value())}\n"
                f"{self._active_profile.description}"
            )
        else:
            extra = ""
            if status.missing_modules:
                extra = "\nMissing modules: " + ", ".join(status.missing_modules)
            self.lbl_runtime.setText(f"{status.message}\nSource: Imported 3D Scene | Device: {status.device}{extra}")

    def _apply_recommended_profile(self) -> None:
        self._hardware = _hardware_profile()
        self._active_profile = _recommended_profile(self._hardware)
        self.points_limit_spin.setValue(int(self._active_profile.render_limit))
        self.faces_limit_spin.setValue(int(self._active_profile.face_limit))
        self.preview_mode_combo.setCurrentText(self._active_profile.render_mode.capitalize())
        self._refresh_runtime_status()

    def _current_profile(self) -> ReconstructionProfile:
        return ReconstructionProfile(
            render_limit=int(self.points_limit_spin.value()),
            face_limit=int(self.faces_limit_spin.value()),
            description=self._active_profile.description,
            render_mode=self.preview_mode_combo.currentText().strip().lower(),
        )

    def _on_preview_mode_changed(self, mode: str) -> None:
        mode_norm = str(mode or "Balanced").strip().lower()
        if mode_norm == "fast":
            self.points_limit_spin.setValue(35000 if HAS_PYVISTA else 24000)
            self.faces_limit_spin.setValue(18000 if HAS_PYVISTA else 12000)
        elif mode_norm == "quality":
            self.points_limit_spin.setValue(180000 if HAS_PYVISTA else 60000)
            self.faces_limit_spin.setValue(0 if HAS_PYVISTA else 30000)
        else:
            self.points_limit_spin.setValue(int(self._active_profile.render_limit))
            self.faces_limit_spin.setValue(int(self._active_profile.face_limit))
        self._refresh_runtime_status()

    def _apply_preview_settings(self) -> None:
        self._refresh_runtime_status()
        if self._raw_xyz_m is None or self._raw_colors is None:
            return
        progress = QtWidgets.QProgressDialog("Reloading 3D preview…", "", 0, 1, self)
        progress.setWindowTitle("3D Scene Workspace")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setCancelButton(None)
        progress.show()
        QtWidgets.QApplication.processEvents()
        try:
            self._rebuild_preview_from_raw()
        finally:
            progress.setValue(1)
            progress.close()

    def _select_scene(self) -> None:
        self.list_sources.clear()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select 3D Scene",
            "",
            "3D scenes (*.ply *.obj *.stl *.xyz *.csv *.txt);;All files (*)",
        )
        if not path:
            return
        self.list_sources.addItem(path)
        self._on_source_selection_changed(0)

    def _selected_scene_paths(self) -> List[str]:
        return [
            self.list_sources.item(i).text().strip()
            for i in range(self.list_sources.count())
            if self.list_sources.item(i).text().strip()
        ]

    def _on_source_selection_changed(self, row: int) -> None:
        paths = self._selected_scene_paths()
        if row < 0 or row >= len(paths):
            noun = "scene file" if len(paths) == 1 else "scene files"
            self.lbl_source_info.setText(f"{len(paths)} {noun} selected.")
            return
        self.lbl_source_info.setText(f"Imported scene source:\n{os.path.basename(paths[row])}")

    def _ensure_ready(self) -> bool:
        status = check_runtime()
        if status.available:
            return True
        QtWidgets.QMessageBox.information(self, "3D scene workspace not ready", status.message)
        return False

    def _on_scale_mode_toggled(self, enabled: bool) -> None:
        self._scale_pick_mode = bool(enabled)
        self._scale_pick_points = []
        self.btn_set_scale.setText("Pick 2 Scale Points" if enabled else "Set Scale")
        if enabled:
            self.btn_collect.setChecked(False)
            if hasattr(self.picker, "begin_scale_preview"):
                self.picker.begin_scale_preview()
            self.lbl_runtime.setText(
                "Scale mode is active. Click two points in the 3D scene, then the scene will be rescaled to the known distance."
            )
        else:
            if hasattr(self.picker, "clear_scale_preview"):
                self.picker.clear_scale_preview()

    def _apply_scale_from_picks(self) -> None:
        if len(self._scale_pick_points) != 2:
            return
        p1, p2 = [np.asarray(p, dtype=float).reshape(3,) for p in self._scale_pick_points]
        measured = float(np.linalg.norm(p1 - p2))
        if measured <= 1e-9:
            QtWidgets.QMessageBox.warning(self, "Calibration failed", "The selected points yielded zero distance.")
            self.btn_set_scale.setChecked(False)
            return
        desired_mm = float(self.scale_cm_spin.value()) * 10.0
        factor = desired_mm / measured
        self._scene_scale_to_mm *= factor
        if self._pending_points_m:
            for label in list(self._pending_points_m.keys()):
                self._pending_points_m[label] = np.asarray(self._pending_points_m[label], dtype=float) * factor
        self._calibration_summary = f"{desired_mm:.2f} mm from 3D picked reference"
        self._update_picker()
        self._refresh_pending_table()
        self.btn_set_scale.setChecked(False)
        if hasattr(self.picker, "clear_scale_preview"):
            self.picker.clear_scale_preview()
        self.lbl_runtime.setText(
            f"3D scale calibration applied.\nReference distance: {desired_mm / 10.0:.2f} cm.\n"
            "You can now collect landmarks in calibrated scene coordinates."
        )

    def _run_reconstruction(self) -> None:
        paths = self._selected_scene_paths()
        if not paths:
            QtWidgets.QMessageBox.warning(self, "Need a scene file", "Select a point cloud, mesh, or 3D model file first.")
            return
        if not self._ensure_ready():
            return

        self._refresh_runtime_status()
        progress = QtWidgets.QProgressDialog("Loading 3D scene…", "", 0, 1, self)
        progress.setWindowTitle("3D Scene Workspace")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setCancelButton(None)
        progress.show()
        QtWidgets.QApplication.processEvents()

        try:
            progress.setLabelText("Importing external 3D scene…")
            QtWidgets.QApplication.processEvents()
            raw_xyz, raw_colors, raw_surface_triangles, raw_surface_colors, reconstruction_meta = _load_scene_file(paths[0])
        except Exception as e:
            progress.close()
            LOG.exception("3D scene loading failed.")
            QtWidgets.QMessageBox.critical(self, "3D scene loading failed", str(e))
            return
        finally:
            progress.setValue(1)
            progress.close()

        self._raw_xyz_m = np.asarray(raw_xyz, dtype=float)
        self._raw_colors = np.asarray(raw_colors, dtype=float)
        self._raw_surface_triangles_m = np.asarray(raw_surface_triangles, dtype=float)
        self._raw_surface_colors = np.asarray(raw_surface_colors, dtype=float)
        self._reconstruction_meta = reconstruction_meta
        self._source_scene_paths = list(paths)
        self._scene_scale_to_mm = 1.0
        self._calibration_summary = "Not calibrated"
        self._pending_points_m.clear()
        self._pending_order.clear()
        self._refresh_pending_table()
        self._rebuild_preview_from_raw()
        self.btn_set_scale.setEnabled(True)
        self.btn_collect.setEnabled(True)
        self._set_preview_ready_text()

    def _rebuild_preview_from_raw(self) -> None:
        if self._raw_xyz_m is None or self._raw_colors is None:
            return
        merged_xyz, merged_colors, surface_triangles, surface_colors, preview_meta = _optimize_scene_preview(
            self._raw_xyz_m,
            self._raw_colors,
            self._raw_surface_triangles_m if self._raw_surface_triangles_m is not None else np.empty((0, 3, 3), dtype=float),
            self._raw_surface_colors if self._raw_surface_colors is not None else np.empty((0, 3), dtype=float),
            self._current_profile(),
        )
        self._merged_xyz_m = merged_xyz
        self._merged_colors = merged_colors
        self._surface_triangles_mm = surface_triangles
        self._surface_colors = surface_colors
        self._reconstruction_meta.update(preview_meta)
        self._update_picker()
        self._set_preview_ready_text()

    def _set_preview_ready_text(self) -> None:
        if self._merged_xyz_m is None:
            return
        face_count = len(self._surface_triangles_mm) if self._surface_triangles_mm is not None else 0
        self.lbl_runtime.setText(
            "3D scene ready.\n"
            f"Preview vertices: {len(self._merged_xyz_m)}"
            + (f" | preview faces: {face_count}\n" if face_count > 0 else "\n")
            + f"Current preview mode: {self.preview_mode_combo.currentText()} | points {int(self.points_limit_spin.value())} | faces {int(self.faces_limit_spin.value())}.\n"
            + "Use 0 to keep the original geometry for that layer. Adjust the limits if needed, click Reload Preview, then set scale from two 3D points and collect landmarks for analysis."
        )

    def _update_picker(self) -> None:
        if self._merged_xyz_m is None or self._merged_colors is None:
            return
        cloud = self._merged_xyz_m * self._scene_scale_to_mm
        surface = None
        if self._surface_triangles_mm is not None and len(self._surface_triangles_mm) > 0:
            surface = self._surface_triangles_mm * self._scene_scale_to_mm
        show_surface = surface is not None and len(surface) > 0
        show_points = not show_surface
        if hasattr(self.picker, "set_render_profile"):
            self.picker.set_render_profile(self.preview_mode_combo.currentText().strip().lower())
        self.picker.set_cloud(
            cloud,
            colors=self._merged_colors,
            surface_triangles=surface,
            surface_colors=self._surface_colors,
            show_points=show_points,
            show_surface=show_surface,
            unit_label=self._coord_unit_label(),
        )

    def _set_collect_mode(self, enabled: bool) -> None:
        self._collect_mode = bool(enabled)
        self.btn_collect.setText("Collect Point ON" if enabled else "Collect Point")

    def _on_picked_xyz(self, xyz_mm: np.ndarray) -> None:
        arr = np.asarray(xyz_mm, dtype=float).reshape(3,)
        now = time.monotonic()
        if (
            self._last_picked_xyz_mm is not None
            and np.linalg.norm(arr - self._last_picked_xyz_mm) <= 1e-6
            and (now - self._last_pick_time_s) <= 0.35
        ):
            return
        self._last_picked_xyz_mm = arr.copy()
        self._last_pick_time_s = now

        if self._scale_pick_mode:
            if self._scale_pick_points:
                if np.linalg.norm(arr - self._scale_pick_points[0]) <= 1e-6:
                    self.lbl_runtime.setText(
                        "Scale mode: second click matched the first point. Please click a distinct second point."
                    )
                    return
            self._scale_pick_points.append(arr)
            if len(self._scale_pick_points) == 1:
                if hasattr(self.picker, "set_scale_preview_anchor"):
                    self.picker.set_scale_preview_anchor(self._scale_pick_points[0])
                self.lbl_runtime.setText(
                    "Scale mode: first 3D point stored. Click the second point to complete local rescaling."
                )
                return
            if len(self._scale_pick_points) >= 2:
                self._apply_scale_from_picks()
                return
        if not self._collect_mode:
            self.lbl_runtime.setText(
                "Point selected in the 3D scene. Turn on 'Collect Point' to add it to the landmark dataset."
            )
            return

        label, ok = QtWidgets.QInputDialog.getText(self, "Collect point", "Label for this 3D landmark:")
        if not ok:
            return
        label = (label or "").strip()
        if not label:
            return
        self._pending_points_m[label] = arr
        if label in self._pending_order:
            self._pending_order.remove(label)
        self._pending_order.append(label)
        self._refresh_pending_table()
        self.lbl_runtime.setText(
            f"Collected {label}: x={arr[0]:.2f}, y={arr[1]:.2f}, z={arr[2]:.2f} {self._coord_unit_label()}"
        )

    def _refresh_pending_table(self) -> None:
        labels = sorted(self._pending_points_m.keys())
        unit = self._coord_unit_label()
        self.pending_table.setHorizontalHeaderLabels(["Label", f"X ({unit})", f"Y ({unit})", f"Z ({unit})"])
        self.pending_table.setRowCount(len(labels))
        for row, label in enumerate(labels):
            point = self._pending_points_m[label]
            values = [label, f"{float(point[0]):.4f}", f"{float(point[1]):.4f}", f"{float(point[2]):.4f}"]
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                self.pending_table.setItem(row, col, item)
        has_any = bool(labels)
        for btn in (self.btn_rename, self.btn_delete, self.btn_undo, self.btn_clear, self.btn_ok):
            btn.setEnabled(has_any)
        self._on_pending_selection_changed()

    def _selected_pending_label(self) -> Optional[str]:
        sel = self.pending_table.selectionModel().selectedRows()
        if not sel:
            return None
        item = self.pending_table.item(sel[0].row(), 0)
        return item.text().strip() if item is not None else None

    def _on_pending_selection_changed(self) -> None:
        selected = self._selected_pending_label()
        self.btn_rename.setEnabled(selected is not None)
        self.btn_delete.setEnabled(selected is not None)

    def _rename_selected(self) -> None:
        old = self._selected_pending_label()
        if not old or old not in self._pending_points_m:
            return
        new, ok = QtWidgets.QInputDialog.getText(self, "Rename point", f"New label for '{old}':")
        if not ok:
            return
        new = (new or "").strip()
        if not new or new == old:
            return
        if new in self._pending_points_m:
            QtWidgets.QMessageBox.warning(self, "Duplicate label", f"'{new}' already exists in the pending landmark set.")
            return
        self._pending_points_m[new] = self._pending_points_m.pop(old)
        if old in self._pending_order:
            self._pending_order[self._pending_order.index(old)] = new
        self._refresh_pending_table()

    def _delete_selected(self) -> None:
        label = self._selected_pending_label()
        if not label or label not in self._pending_points_m:
            return
        self._pending_points_m.pop(label, None)
        self._pending_order = [name for name in self._pending_order if name != label]
        self._refresh_pending_table()

    def _undo_last(self) -> None:
        if not self._pending_order:
            return
        label = self._pending_order.pop()
        self._pending_points_m.pop(label, None)
        self._refresh_pending_table()

    def _clear_all(self) -> None:
        self._pending_points_m.clear()
        self._pending_order.clear()
        self._refresh_pending_table()

    def _build_scene_meta(self) -> Dict[str, Any]:
        if self._merged_xyz_m is None or self._merged_colors is None:
            return {}
        xyz_mm = self._merged_xyz_m * self._scene_scale_to_mm
        colors = self._merged_colors
        surface_triangles = np.empty((0, 3, 3), dtype=float)
        surface_colors = np.empty((0, 3), dtype=float)
        if self._surface_triangles_mm is not None and len(self._surface_triangles_mm) > 0:
            surface_triangles = self._surface_triangles_mm * self._scene_scale_to_mm
            if self._surface_colors is not None:
                surface_colors = self._surface_colors
        return {
            "kind": "scene_import",
            "backend": self._scene_backend,
            "source_paths": list(self._source_scene_paths),
            "cloud_xyz": xyz_mm.tolist(),
            "cloud_rgb": colors.tolist(),
            "surface_triangles": surface_triangles.tolist(),
            "surface_rgb": surface_colors.tolist(),
            "calibration": self._calibration_summary,
            "scale_to_mm": float(self._scene_scale_to_mm),
            "preview_point_limit": int(self.points_limit_spin.value()),
            "preview_face_limit": int(self.faces_limit_spin.value()),
            "reconstruction_meta": dict(self._reconstruction_meta),
        }

    def _commit_and_accept(self) -> None:
        if not self._pending_points_m:
            QtWidgets.QMessageBox.information(self, "No landmarks", "Collect at least one landmark before committing.")
            return
        if self._calibration_summary == "Not calibrated":
            reply = QtWidgets.QMessageBox.question(
                self,
                "Uncalibrated scene",
                "This scene has not been rescaled to a known world distance yet.\n\n"
                "Landmarks can still be committed, but downstream measurements will remain in arbitrary scene units.\n\n"
                "Commit anyway?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return

        for label, point_mm in self._pending_points_m.items():
            self.ds.points[label] = np.asarray(point_mm, dtype=float).reshape(3,)

        if isinstance(getattr(self.ds, "meta", None), dict):
            self.ds.meta["source"] = "scene_import"
            self.ds.meta["ai_scene"] = self._build_scene_meta()

        if callable(self.on_dataset_changed_callback):
            try:
                self.on_dataset_changed_callback()
            except Exception:
                pass
        self.accept()
