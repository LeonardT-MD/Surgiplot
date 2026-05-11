from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import vtk


class SurgiplotMetricBridge:
    """Thin adapter from Slicer markups nodes to the shared Surgiplot backend."""

    AOA_ENTRY_ROLES = ("cranial", "caudal", "medial", "lateral")

    def __init__(self) -> None:
        self._sp = None

    def _repo_root(self) -> Path:
        here = Path(__file__).resolve()
        for candidate in here.parents:
            if (candidate / "surgiplot" / "__init__.py").exists():
                return candidate
        raise RuntimeError("Could not locate Surgiplot repository root from SurgiplotToolkit.")

    def _candidate_backend_roots(self) -> list[Path]:
        here = Path(__file__).resolve()
        candidates: list[Path] = []

        vendored = here.parent / "External" / "surgiplot_backend"
        if (vendored / "surgiplot" / "__init__.py").exists():
            candidates.append(vendored)

        for candidate in here.parents:
            if (candidate / "surgiplot" / "__init__.py").exists():
                candidates.append(candidate)

        unique: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate.resolve())
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique

    def backend(self):
        if self._sp is None:
            try:
                import surgiplot as sp  # type: ignore
            except ImportError:
                sp = None
                for root in self._candidate_backend_roots():
                    if str(root) not in sys.path:
                        sys.path.insert(0, str(root))
                    try:
                        import surgiplot as _sp  # type: ignore

                        sp = _sp
                        break
                    except ImportError:
                        continue
                if sp is None:
                    raise
            self._sp = sp
        return self._sp

    def markups_count(self, node) -> int:
        return int(node.GetNumberOfControlPoints()) if node is not None else 0

    def markups_to_points(self, node, *, expected: Optional[int] = None, minimum: Optional[int] = None) -> np.ndarray:
        if node is None:
            raise ValueError("A markups node is required.")
        count = int(node.GetNumberOfControlPoints())
        if expected is not None and count != expected:
            raise ValueError(f"Expected exactly {expected} control points, got {count}.")
        if minimum is not None and count < minimum:
            raise ValueError(f"Expected at least {minimum} control points, got {count}.")
        pts = np.zeros((count, 3), dtype=float)
        for index in range(count):
            p = [0.0, 0.0, 0.0]
            node.GetNthControlPointPositionWorld(index, p)
            pts[index, :] = np.asarray(p, dtype=float)
        return pts

    def markups_to_point(self, node) -> np.ndarray:
        return self.markups_to_points(node, expected=1)[0]

    def markups_labels(self, node) -> list[str]:
        if node is None:
            raise ValueError("A markups node is required.")
        labels: list[str] = []
        count = int(node.GetNumberOfControlPoints())
        for index in range(count):
            labels.append(str(node.GetNthControlPointLabel(index) or "").strip())
        return labels

    def count_summary(self, node, *, expected: Optional[int] = None, minimum: Optional[int] = None) -> dict[str, Any]:
        count = self.markups_count(node)
        valid = True
        warnings: list[str] = []
        if node is None:
            valid = False
            warnings.append("No markups node selected.")
        if expected is not None and count != expected:
            valid = False
            warnings.append(f"Expected exactly {expected} control points, got {count}.")
        if minimum is not None and count < minimum:
            valid = False
            warnings.append(f"Expected at least {minimum} control points, got {count}.")
        return {"valid": valid, "count": count, "warnings": warnings}

    @staticmethod
    def _normalize_role_label(label: str) -> str:
        text = "".join(ch for ch in str(label or "").strip().lower() if ch.isalnum())
        if not text:
            return ""
        if "cranial" in text or "superior" in text:
            return "cranial"
        if "caudal" in text or "inferior" in text:
            return "caudal"
        if "medial" in text:
            return "medial"
        if "lateral" in text:
            return "lateral"
        return ""

    def resolve_aoa_entry(self, entry_node) -> dict[str, Any]:
        points = self.markups_to_points(entry_node, expected=4)
        labels = self.markups_labels(entry_node)

        role_to_index: dict[str, int] = {}
        duplicate_roles: list[str] = []
        for index, label in enumerate(labels):
            role = self._normalize_role_label(label)
            if not role:
                continue
            if role in role_to_index:
                duplicate_roles.append(role)
                continue
            role_to_index[role] = index

        if all(role in role_to_index for role in self.AOA_ENTRY_ROLES):
            ordered_indices = [role_to_index[role] for role in self.AOA_ENTRY_ROLES]
            return {
                "points": points[ordered_indices],
                "labels": labels,
                "ordered_labels": [labels[idx] or self.AOA_ENTRY_ROLES[pos] for pos, idx in enumerate(ordered_indices)],
                "role_order": list(self.AOA_ENTRY_ROLES),
                "ordering_strategy": "label_inferred",
                "warnings": [],
            }

        warnings: list[str] = []
        if duplicate_roles:
            warnings.append(
                "Duplicate AoA/SF role labels were detected; falling back to control-point order."
            )
        unlabeled_roles = [role for role in self.AOA_ENTRY_ROLES if role not in role_to_index]
        if unlabeled_roles:
            warnings.append(
                "Could not infer full AoA/SF role order from point labels; expected labels matching "
                "'cranial', 'caudal', 'medial', and 'lateral'. Falling back to control-point order."
            )

        return {
            "points": points,
            "labels": labels,
            "ordered_labels": labels,
            "role_order": list(self.AOA_ENTRY_ROLES),
            "ordering_strategy": "control_point_order",
            "warnings": warnings,
        }

    def describe_aoa_entry(self, entry_node) -> dict[str, Any]:
        if entry_node is None:
            return {
                "valid": False,
                "point_count": 0,
                "labels": [],
                "role_order": list(self.AOA_ENTRY_ROLES),
                "ordering_strategy": "missing_node",
                "warnings": ["No entry markups node selected."],
            }

        count = self.markups_count(entry_node)
        labels = self.markups_labels(entry_node)
        if count != 4:
            return {
                "valid": False,
                "point_count": count,
                "labels": labels,
                "role_order": list(self.AOA_ENTRY_ROLES),
                "ordering_strategy": "invalid_point_count",
                "warnings": [f"AoA / SF entry requires exactly 4 control points; current node has {count}."],
            }

        resolved = self.resolve_aoa_entry(entry_node)
        return {
            "valid": True,
            "point_count": count,
            "labels": labels,
            "role_order": list(resolved["role_order"]),
            "ordered_labels": list(resolved["ordered_labels"]),
            "ordering_strategy": str(resolved["ordering_strategy"]),
            "warnings": list(resolved["warnings"]),
        }

    def dataset_from_named_markups(self, named_nodes: dict[str, Any]):
        sp = self.backend()
        points: list[np.ndarray] = []
        names: list[str] = []
        for prefix, node in named_nodes.items():
            if node is None:
                continue
            node_points = self.markups_to_points(node, minimum=1)
            if len(node_points) == 1:
                points.append(node_points[0])
                names.append(prefix)
            else:
                for index, point in enumerate(node_points, start=1):
                    points.append(point)
                    names.append(f"{prefix}_{index}")
        return sp.from_points(points, names=names, source="slicer")

    def compute_vom_family(self, entry_node, target_node, *, stand_dist: float, return_debug: bool = False):
        sp = self.backend()
        entry = self.markups_to_points(entry_node, minimum=3)
        target = self.markups_to_points(target_node, minimum=3)
        return sp.VOM(data=None, entry=entry, target=target, stand_dist=stand_dist, return_debug=return_debug)

    def compute_aoa_family(self, entry_node, pivot_node, *, sf_rescale_radius_mm: Optional[float], return_debug: bool = False):
        sp = self.backend()
        resolved_entry = self.resolve_aoa_entry(entry_node)
        entry = resolved_entry["points"]
        pivot = self.markups_to_point(pivot_node)
        radius = None if sf_rescale_radius_mm is None or sf_rescale_radius_mm <= 0 else float(sf_rescale_radius_mm)
        result = sp.AOA(data=None, entry=entry, target=pivot, sf_rescale_radius_mm=radius, return_debug=return_debug)
        debug = dict(result.debug or {})
        debug["input_labels"] = resolved_entry["labels"]
        debug["ordered_entry_labels"] = resolved_entry["ordered_labels"]
        debug["entry_role_order"] = resolved_entry["role_order"]
        debug["ordering_strategy"] = resolved_entry["ordering_strategy"]
        debug["warnings"] = list(resolved_entry["warnings"])
        result.debug = debug
        return result

    def compute_aoe(self, a_node, b_node, c_node):
        sp = self.backend()
        return sp.AOE(data=None, A=self.markups_to_point(a_node), B=self.markups_to_point(b_node), C=self.markups_to_point(c_node), return_debug=True)

    def compute_distance(self, a_node, b_node):
        sp = self.backend()
        return sp.DISTANCE_3D(data=None, A=self.markups_to_point(a_node), B=self.markups_to_point(b_node), return_debug=True)

    def compute_area(self, polygon_node):
        sp = self.backend()
        return sp.AREA_3D(data=None, polygon=self.markups_to_points(polygon_node, minimum=3), return_debug=True)

    def compute_volume(self, volume_node):
        sp = self.backend()
        return sp.VOLUME_3D(data=None, points=self.markups_to_points(volume_node, minimum=4), return_debug=True)

    def triangles_to_polydata(self, triangles: Iterable[Any]):
        tri_array = np.asarray(list(triangles), dtype=float)
        if tri_array.size == 0:
            return None
        if tri_array.ndim == 2 and tri_array.shape[1] == 3 and tri_array.shape[0] >= 3:
            polygon = tri_array
            tri_array = np.stack(
                [np.vstack([polygon[0], polygon[index], polygon[index + 1]]) for index in range(1, polygon.shape[0] - 1)],
                axis=0,
            )
        elif tri_array.ndim == 3 and tri_array.shape[2] == 3 and tri_array.shape[1] >= 3 and tri_array.shape[1] != 3:
            polygons = tri_array
            triangulated = []
            for polygon in polygons:
                triangulated.extend(
                    np.vstack([polygon[0], polygon[index], polygon[index + 1]])
                    for index in range(1, polygon.shape[0] - 1)
                )
            tri_array = np.asarray(triangulated, dtype=float)
        if tri_array.ndim != 3 or tri_array.shape[1:] != (3, 3):
            raise ValueError("Expected triangles shaped (N, 3, 3), stacked polygons shaped (N, M, 3), or a polygon shaped (M, 3).")

        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        for triangle in tri_array:
            ids = []
            for point in triangle:
                ids.append(points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2])))
            tri = vtk.vtkTriangle()
            for idx, pid in enumerate(ids):
                tri.GetPointIds().SetId(idx, pid)
            cells.InsertNextCell(tri)

        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetPolys(cells)
        return poly_data

    def polyline_to_polydata(self, points_like: Iterable[Any], *, closed: bool = False):
        pts = np.asarray(points_like, dtype=float)
        if pts.size == 0:
            return None
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 2:
            raise ValueError("Expected polyline points shaped (N, 3) with N >= 2.")

        vtk_points = vtk.vtkPoints()
        for point in pts:
            vtk_points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))

        line_count = pts.shape[0] + (1 if closed else 0)
        poly_line = vtk.vtkPolyLine()
        poly_line.GetPointIds().SetNumberOfIds(line_count)
        for index in range(pts.shape[0]):
            poly_line.GetPointIds().SetId(index, index)
        if closed:
            poly_line.GetPointIds().SetId(pts.shape[0], 0)

        cells = vtk.vtkCellArray()
        cells.InsertNextCell(poly_line)

        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(vtk_points)
        poly_data.SetLines(cells)
        return poly_data
