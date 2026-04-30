from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict

import numpy as np
from scipy.spatial import Delaunay, ConvexHull, cKDTree


@dataclass
class Volume3DResult:
    volume_mm3: float
    debug: Optional[Dict[str, Any]] = None

    def plot(self, ax) -> None:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        dbg = self.debug or {}
        points = np.asarray(dbg.get("points", []), dtype=float)
        surface = np.asarray(dbg.get("surface_triangles", []), dtype=float)

        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
            raise ValueError("Volume plot requires a valid 3D point cloud.")

        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            color="#2563eb", s=48, edgecolors="white", linewidths=0.6, depthshade=False, label="Input points",
        )

        if surface.ndim == 3 and surface.shape[-1] == 3 and len(surface) > 0:
            ax.add_collection3d(
                Poly3DCollection(
                    list(surface),
                    facecolors="#94a3b8",
                    alpha=0.22,
                    edgecolors=(0.20, 0.25, 0.31, 0.30),
                    linewidths=0.35,
                )
            )

        ax.set_title(f"Volume={self.volume_mm3:.2f} mm^3")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        mids = (mins + maxs) * 0.5
        spans = np.maximum(maxs - mins, 1.0)
        radius = 0.6 * float(np.max(spans))
        ax.set_xlim3d([mids[0] - radius, mids[0] + radius])
        ax.set_ylim3d([mids[1] - radius, mids[1] + radius])
        ax.set_zlim3d([mids[2] - radius, mids[2] + radius])


def _resolve_cloud(data, spec) -> np.ndarray:
    if data is not None:
        if isinstance(spec, (list, tuple)) and spec and all(isinstance(x, str) for x in spec):
            if hasattr(data, "resolve_names"):
                spec = data.resolve_names(spec)
        pts = data.resolve_points(spec)
    else:
        pts = np.asarray(spec, dtype=float)

    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("Point cloud must be an (N,3) array-like.")
    if pts.shape[0] < 4:
        raise ValueError("Need at least 4 non-coplanar points to estimate a volume.")
    return pts


def _tetra_volume(tet: np.ndarray) -> float:
    a, b, c, d = np.asarray(tet, dtype=float)
    return float(abs(np.linalg.det(np.vstack([b - a, c - a, d - a]))) / 6.0)


def _circumsphere_radius(tet: np.ndarray) -> float:
    tet = np.asarray(tet, dtype=float)
    a = tet[0]
    M = 2.0 * np.vstack([tet[1] - a, tet[2] - a, tet[3] - a])
    rhs = np.array([
        np.dot(tet[1], tet[1]) - np.dot(a, a),
        np.dot(tet[2], tet[2]) - np.dot(a, a),
        np.dot(tet[3], tet[3]) - np.dot(a, a),
    ], dtype=float)
    if abs(np.linalg.det(M)) < 1e-12:
        return float("inf")
    center = np.linalg.solve(M, rhs)
    return float(np.linalg.norm(center - a))


def _tetra_connectivity(selected_tets: np.ndarray) -> int:
    if len(selected_tets) == 0:
        return 0
    face_to_tets: Dict[tuple[int, int, int], list[int]] = {}
    for idx, tet in enumerate(selected_tets):
        faces = [
            tuple(sorted((tet[0], tet[1], tet[2]))),
            tuple(sorted((tet[0], tet[1], tet[3]))),
            tuple(sorted((tet[0], tet[2], tet[3]))),
            tuple(sorted((tet[1], tet[2], tet[3]))),
        ]
        for face in faces:
            face_to_tets.setdefault(face, []).append(idx)

    adjacency = {i: set() for i in range(len(selected_tets))}
    for tet_ids in face_to_tets.values():
        if len(tet_ids) < 2:
            continue
        for i in tet_ids:
            adjacency[i].update(j for j in tet_ids if j != i)

    components = 0
    unseen = set(range(len(selected_tets)))
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            cur = stack.pop()
            for nb in adjacency[cur]:
                if nb in unseen:
                    unseen.remove(nb)
                    stack.append(nb)
    return components


def _boundary_triangles_from_tets(points: np.ndarray, selected_tets: np.ndarray) -> np.ndarray:
    if len(selected_tets) == 0:
        return np.empty((0, 3, 3), dtype=float)

    face_counts: Dict[tuple[int, int, int], list[int]] = {}
    for tet in selected_tets:
        faces = [
            (tet[0], tet[1], tet[2]),
            (tet[0], tet[1], tet[3]),
            (tet[0], tet[2], tet[3]),
            (tet[1], tet[2], tet[3]),
        ]
        for face in faces:
            face_counts.setdefault(tuple(sorted(face)), []).append(face)

    overall_center = np.asarray(points, dtype=float).mean(axis=0)
    tris: list[np.ndarray] = []
    for sorted_face, originals in face_counts.items():
        if len(originals) != 1:
            continue
        tri = np.asarray(points[list(originals[0])], dtype=float)
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        tri_center = tri.mean(axis=0)
        if np.dot(normal, tri_center - overall_center) < 0:
            tri = tri[[0, 2, 1]]
        tris.append(tri)
    return np.asarray(tris, dtype=float)


def _auto_alpha_shape(points: np.ndarray) -> Dict[str, Any]:
    delaunay = Delaunay(points, qhull_options="QJ")
    simplices = np.asarray(delaunay.simplices, dtype=int)
    tetra_points = points[simplices]
    tetra_volumes = np.asarray([_tetra_volume(tet) for tet in tetra_points], dtype=float)
    circumradii = np.asarray([_circumsphere_radius(tet) for tet in tetra_points], dtype=float)

    finite = np.isfinite(circumradii) & (tetra_volumes > 1e-12)
    if not np.any(finite):
        raise ValueError("Automatic alpha-shape reconstruction failed: degenerate tetrahedralization.")

    tree = cKDTree(points)
    k = min(4, len(points))
    dists, _ = tree.query(points, k=k)
    if dists.ndim == 1:
        local_spacing = float(np.median(dists))
    else:
        local_spacing = float(np.median(dists[:, 1:]))

    candidate_alphas = np.unique(np.concatenate([
        np.percentile(circumradii[finite], [35, 45, 55, 65, 75, 85, 95]),
        np.array([local_spacing, 1.5 * local_spacing, 2.0 * local_spacing, 3.0 * local_spacing], dtype=float),
    ]))
    candidate_alphas = candidate_alphas[np.isfinite(candidate_alphas) & (candidate_alphas > 0)]
    candidate_alphas.sort()

    best: Optional[Dict[str, Any]] = None
    for alpha in candidate_alphas:
        mask = finite & (circumradii <= alpha)
        if not np.any(mask):
            continue
        selected_tets = simplices[mask]
        boundary = _boundary_triangles_from_tets(points, selected_tets)
        if len(boundary) < 4:
            continue
        used_vertices = np.unique(selected_tets.reshape(-1))
        coverage = float(len(used_vertices) / len(points))
        components = _tetra_connectivity(selected_tets)
        volume = float(np.sum(tetra_volumes[mask]))
        candidate = {
            "alpha_mm": float(alpha),
            "selected_tets": selected_tets,
            "surface_triangles": boundary,
            "coverage_ratio": coverage,
            "components": components,
            "volume_mm3": volume,
        }
        if components == 1 and coverage >= 0.8:
            return candidate
        if best is None:
            best = candidate
        else:
            best_score = (best["components"] != 1, 1.0 - best["coverage_ratio"], best["alpha_mm"])
            cand_score = (components != 1, 1.0 - coverage, float(alpha))
            if cand_score < best_score:
                best = candidate

    if best is None:
        raise ValueError("Automatic alpha-shape reconstruction could not identify a stable closed solid.")
    return best


def _convex_hull_reconstruction(points: np.ndarray) -> Dict[str, Any]:
    hull = ConvexHull(points)
    faces = np.asarray([points[simplex] for simplex in hull.simplices], dtype=float)
    return {
        "alpha_mm": None,
        "selected_tets": None,
        "surface_triangles": faces,
        "coverage_ratio": 1.0,
        "components": 1,
        "volume_mm3": float(hull.volume),
    }


def VOLUME_3D(
    data=None,
    points=None,
    space: str = "native",
    transforms=None,
    return_debug: bool = False,
) -> Volume3DResult:
    """Estimate the enclosed volume of a closed solid reconstructed from sparse 3D boundary points.

    Default behavior uses an automatic alpha-shape style tetrahedral reconstruction, which is
    intended to approximate a segmentation-like closed solid from 10-20 anatomically sampled points.
    If the alpha-shape inference is unstable, the computation falls back to the convex hull.
    """
    if points is None:
        raise ValueError("Provide a point cloud (>=4 points or names/labels).")

    P = _resolve_cloud(data, points)
    centered = P - P.mean(axis=0)
    if np.linalg.matrix_rank(centered) < 3:
        raise ValueError("Input points are coplanar or nearly coplanar; a 3D volume cannot be estimated reliably.")

    convex = _convex_hull_reconstruction(P)
    reconstruction_method = "convex_hull"
    alpha_result = None
    try:
        alpha_result = _auto_alpha_shape(P)
        result = alpha_result
        reconstruction_method = "alpha_shape_auto"
    except Exception:
        result = convex

    dbg = None
    if return_debug:
        dbg = {
            "points": P,
            "surface_triangles": result["surface_triangles"],
            "reconstruction_method": reconstruction_method,
            "alpha_mm": result["alpha_mm"],
            "coverage_ratio": result["coverage_ratio"],
            "connected_components": result["components"],
            "convex_hull_volume_mm3": float(convex["volume_mm3"]),
            "alpha_shape_volume_mm3": float(alpha_result["volume_mm3"]) if alpha_result is not None else None,
            "space": space,
        }

    return Volume3DResult(volume_mm3=float(result["volume_mm3"]), debug=dbg)
