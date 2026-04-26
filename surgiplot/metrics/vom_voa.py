from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict

import numpy as np

from surgiplot.core.geometry.pca_plane import best_fit_plane_pca, project_to_plane_2d
from surgiplot.core.geometry.polygon_area import shoelace_area


@dataclass
class VOMVOAResult:
    voa_deg: float
    vom_mm3: float
    svom_mm3: float
    debug: Optional[Dict[str, Any]] = None

    def plot(self, ax) -> None:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        dbg = self.debug or {}
        all_plot_points: list[np.ndarray] = []
        entry_poly = dbg.get("entry_polygon")
        target_poly = dbg.get("target_polygon")
        entry_ellipse = dbg.get("entry_ellipse_3d")
        target_ellipse = dbg.get("target_ellipse_3d")
        cut_ellipse = dbg.get("svom_cut_ellipse_3d")
        cE = dbg.get("centroid_entry")
        cT = dbg.get("centroid_target")
        normal = dbg.get("target_normal")
        full_surface = dbg.get("frustum_surface")
        svom_surface = dbg.get("svom_surface")

        if entry_poly is not None:
            E = np.asarray(entry_poly, dtype=float)
            if E.ndim == 2 and E.shape[1] == 3:
                all_plot_points.append(E)
                closed = np.vstack([E, E[0]])
                ax.plot(
                    closed[:, 0], closed[:, 1], closed[:, 2],
                    color="tab:blue", linewidth=1.2, alpha=0.5, label="Entry polygon",
                )
                ax.scatter(E[:, 0], E[:, 1], E[:, 2], color="tab:blue", s=18, alpha=0.65)

        if target_poly is not None:
            T = np.asarray(target_poly, dtype=float)
            if T.ndim == 2 and T.shape[1] == 3:
                all_plot_points.append(T)
                closed = np.vstack([T, T[0]])
                ax.plot(
                    closed[:, 0], closed[:, 1], closed[:, 2],
                    color="tab:red", linewidth=1.2, alpha=0.5, label="Target polygon",
                )
                ax.scatter(T[:, 0], T[:, 1], T[:, 2], color="tab:red", s=18, alpha=0.65)

        if entry_ellipse is not None:
            Eel = np.asarray(entry_ellipse, dtype=float)
            all_plot_points.append(Eel)
            ax.plot(Eel[:, 0], Eel[:, 1], Eel[:, 2], color="tab:blue", linewidth=2.2, label="Entry ellipse")

        if target_ellipse is not None:
            Tel = np.asarray(target_ellipse, dtype=float)
            all_plot_points.append(Tel)
            ax.plot(Tel[:, 0], Tel[:, 1], Tel[:, 2], color="tab:red", linewidth=2.2, label="Target ellipse")

        if full_surface is not None:
            side_faces = list(np.asarray(full_surface, dtype=float))
            if side_faces:
                ax.add_collection3d(
                    Poly3DCollection(
                        side_faces,
                        facecolors="tab:gray",
                        alpha=0.07,
                        edgecolors="0.7",
                        linewidths=0.2,
                    )
                )

        if svom_surface is not None:
            side_faces = list(np.asarray(svom_surface, dtype=float))
            if side_faces:
                ax.add_collection3d(
                    Poly3DCollection(
                        side_faces,
                        facecolors="tab:green",
                        alpha=0.22,
                        edgecolors="tab:green",
                        linewidths=0.4,
                    )
                )

        if cut_ellipse is not None:
            Cel = np.asarray(cut_ellipse, dtype=float)
            all_plot_points.append(Cel)
            ax.plot(Cel[:, 0], Cel[:, 1], Cel[:, 2], color="tab:green", linewidth=2.0, label="sVOM cut ellipse")

        if cE is not None and cT is not None:
            cE = np.asarray(cE, dtype=float).reshape(3,)
            cT = np.asarray(cT, dtype=float).reshape(3,)
            all_plot_points.append(np.vstack([cE, cT]))
            ax.scatter([cE[0], cT[0]], [cE[1], cT[1]], [cE[2], cT[2]], color="black", s=40)
            ax.plot([cE[0], cT[0]], [cE[1], cT[1]], [cE[2], cT[2]], color="black", linestyle="--", linewidth=1.4)

        if cT is not None and normal is not None:
            cT = np.asarray(cT, dtype=float).reshape(3,)
            normal = np.asarray(normal, dtype=float).reshape(3,)
            scale = max(float(dbg.get("distance_h", 0.0) or 0.0) * 0.25, 1.0)
            ax.quiver(
                cT[0], cT[1], cT[2],
                normal[0], normal[1], normal[2],
                length=scale,
                color="tab:purple",
                normalize=True,
            )

        ax.set_title(
            f"VoA={self.voa_deg:.2f} deg | "
            f"VOM={self.vom_mm3:.2f} mm^3 | "
            f"sVOM={self.svom_mm3:.2f} mm^3"
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        if all_plot_points:
            _set_axes_equal_from_points(ax, np.vstack(all_plot_points))
        else:
            _set_axes_equal(ax)
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc="upper right")


def _as_poly(data, spec) -> np.ndarray:
    if data is not None:
        if isinstance(spec, (list, tuple)) and spec and all(isinstance(x, str) for x in spec):
            if hasattr(data, "resolve_names"):
                spec = data.resolve_names(spec)
        arr = data.resolve_points(spec)
    else:
        arr = np.asarray(spec, dtype=float)

    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 3:
        raise ValueError("Polygon must be an (N,3) array with N>=3.")
    return arr


def _centroid(poly3d: np.ndarray) -> np.ndarray:
    return np.asarray(poly3d, dtype=float).mean(axis=0)


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=float).reshape(3,)
    v = np.asarray(v, dtype=float).reshape(3,)
    den = float(np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
    cosang = float(np.dot(u, v) / den)
    cosang = max(-1.0, min(1.0, cosang))
    return float(np.degrees(np.arccos(cosang)))


def _polygon_area_3d(poly3d: np.ndarray) -> float:
    pts2, _c, _basis = project_to_plane_2d(poly3d)
    return float(shoelace_area(pts2))


def _polygon_normal(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 3:
        raise ValueError("points must be (N,3) with N>=3")

    c = pts.mean(axis=0)
    normal = np.zeros(3, dtype=float)
    for i in range(pts.shape[0]):
        u = pts[i] - c
        v = pts[(i + 1) % pts.shape[0]] - c
        normal += np.cross(u, v)

    n = float(np.linalg.norm(normal))
    if n < 1e-12:
        _c, fallback, _basis = best_fit_plane_pca(pts)
        return fallback
    return normal / n


def _fit_area_matched_ellipse(poly3d: np.ndarray, n_samples: int = 120) -> Dict[str, Any]:
    pts2, centroid, basis = project_to_plane_2d(poly3d)
    area_polygon = float(shoelace_area(pts2))

    spread = np.max(np.abs(pts2), axis=0)
    spread = np.where(spread > 1e-9, spread, np.std(pts2, axis=0))
    spread = np.where(spread > 1e-9, spread, 1.0)

    # Preserve the PCA-derived aspect ratio and scale uniformly so that
    # the fitted ellipse area matches the polygon area.
    uniform_scale = float(np.sqrt(max(area_polygon, 1e-12) / (np.pi * spread[0] * spread[1])))
    a = float(uniform_scale * spread[0])
    b = float(uniform_scale * spread[1])

    theta = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    ellipse2d = np.column_stack([a * np.cos(theta), b * np.sin(theta)])
    ellipse3d = ellipse2d @ basis[:, :2].T + centroid

    return {
        "polygon_area_2d": area_polygon,
        "polygon_area_3d": _polygon_area_3d(poly3d),
        "centroid": centroid,
        "basis": basis,
        "ellipse_a": a,
        "ellipse_b": b,
        "ellipse_area": float(np.pi * a * b),
        "ellipse_2d": ellipse2d,
        "ellipse_3d": ellipse3d,
        "projected_polygon_2d": pts2,
    }


def _ellipse_points_3d(center: np.ndarray, basis: np.ndarray, a: float, b: float, n_samples: int = 120) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    pts2 = np.column_stack([a * np.cos(theta), b * np.sin(theta)])
    return pts2 @ basis[:, :2].T + center


def _interpolate_centers(target_center: np.ndarray, entry_center: np.ndarray, ratio: float) -> np.ndarray:
    return target_center + (entry_center - target_center) * float(ratio)


def _interpolate_basis(target_basis: np.ndarray, entry_basis: np.ndarray, ratio: float) -> np.ndarray:
    B = (1.0 - ratio) * np.asarray(target_basis, dtype=float) + ratio * np.asarray(entry_basis, dtype=float)
    u = B[:, 0]
    u = u / (np.linalg.norm(u) + 1e-12)
    v = B[:, 1] - u * np.dot(B[:, 1], u)
    v = v / (np.linalg.norm(v) + 1e-12)
    n = np.cross(u, v)
    n = n / (np.linalg.norm(n) + 1e-12)
    return np.column_stack([u, v, n])


def _tilt_deviation_deg(voa_deg: float) -> float:
    return float(max(0.0, 90.0 - float(voa_deg)))


def _supplementary_volume_weight_endpoint(voa_deg: float) -> float:
    return float(np.sin(np.radians(float(voa_deg))))


def _supplementary_svom_adjustment_factor(voa_deg: float) -> float:
    tilt_rad = np.radians(_tilt_deviation_deg(voa_deg))
    return float(0.1 * (1.0 - np.cos(tilt_rad)))


def _legacy_frustum_volume(area_start: float, area_end: float, height: float) -> float:
    if height <= 1e-12:
        return 0.0
    return float((height / 3.0) * (area_start + area_end + np.sqrt(max(area_start * area_end, 0.0))))


def _legacy_tilt_weighted_elliptical_volume(
    a_start: float,
    b_start: float,
    a_end: float,
    b_end: float,
    H: float,
    voa_deg: float,
    n_steps: int = 400,
) -> float:
    if H <= 1e-12:
        return 0.0
    hs = np.linspace(0.0, float(H), int(max(n_steps, 32)))
    ratio = hs / float(H)
    a = float(a_start) + (float(a_end) - float(a_start)) * ratio
    b = float(b_start) + (float(b_end) - float(b_start)) * ratio
    area = np.pi * a * b
    voa_rad = np.radians(float(voa_deg))
    weight = 1.0 + (np.sin(voa_rad) - 1.0) * ratio
    return float(_trapezoid(area * weight, hs))


def _build_side_faces(ring0: np.ndarray, ring1: np.ndarray) -> list[list[np.ndarray]]:
    A = np.asarray(ring0, dtype=float)
    B = np.asarray(ring1, dtype=float)
    n = min(A.shape[0], B.shape[0])
    if n < 2:
        return []
    if A.shape[0] != n:
        A = A[:n]
    if B.shape[0] != n:
        B = B[:n]

    faces: list[list[np.ndarray]] = []
    for i in range(n - 1):
        faces.append([A[i], A[i + 1], B[i + 1], B[i]])
    return faces


def _build_frustum_surface(entry_ellipse: np.ndarray, target_ellipse: np.ndarray) -> np.ndarray:
    return np.asarray(_build_side_faces(target_ellipse, entry_ellipse), dtype=float)


def _build_section_surface(rings: list[np.ndarray]) -> np.ndarray:
    faces: list[list[np.ndarray]] = []
    for i in range(len(rings) - 1):
        faces.extend(_build_side_faces(rings[i], rings[i + 1]))
    return np.asarray(faces, dtype=float)


def _loft_ring_stack(
    target_fit: Dict[str, Any],
    entry_fit: Dict[str, Any],
    ratio_end: float,
    n_sections: int = 64,
) -> list[np.ndarray]:
    ratio_end = float(max(0.0, min(1.0, ratio_end)))
    if ratio_end <= 0.0:
        return [np.asarray(target_fit["ellipse_3d"], dtype=float)]

    rings: list[np.ndarray] = []
    for frac in np.linspace(0.0, ratio_end, int(max(n_sections, 2))):
        center = _interpolate_centers(target_fit["centroid"], entry_fit["centroid"], frac)
        basis = _interpolate_basis(target_fit["basis"], entry_fit["basis"], frac)
        a = float(target_fit["ellipse_a"]) + (float(entry_fit["ellipse_a"]) - float(target_fit["ellipse_a"])) * frac
        b = float(target_fit["ellipse_b"]) + (float(entry_fit["ellipse_b"]) - float(target_fit["ellipse_b"])) * frac
        rings.append(
            _ellipse_points_3d(
                center,
                basis,
                a,
                b,
                n_samples=np.asarray(target_fit["ellipse_3d"]).shape[0],
            )
        )
    return rings


def _orient_triangle_outward(tri: np.ndarray, overall_center: np.ndarray) -> np.ndarray:
    tri = np.asarray(tri, dtype=float)
    a, b, c = tri
    normal = np.cross(b - a, c - a)
    tri_center = tri.mean(axis=0)
    if np.dot(normal, tri_center - overall_center) < 0:
        return np.asarray([a, c, b], dtype=float)
    return tri


def _triangulate_ring_cap(ring: np.ndarray, reverse: bool, overall_center: np.ndarray) -> list[np.ndarray]:
    ring = np.asarray(ring, dtype=float)
    center = ring.mean(axis=0)
    tris: list[np.ndarray] = []
    for i in range(ring.shape[0]):
        j = (i + 1) % ring.shape[0]
        tri = np.asarray([center, ring[i], ring[j]], dtype=float)
        if reverse:
            tri = tri[[0, 2, 1]]
        tris.append(_orient_triangle_outward(tri, overall_center))
    return tris


def _triangulate_ring_stack(rings: list[np.ndarray]) -> list[np.ndarray]:
    if not rings:
        return []
    all_points = np.vstack(rings)
    overall_center = all_points.mean(axis=0)

    tris: list[np.ndarray] = []
    tris.extend(_triangulate_ring_cap(rings[0], reverse=False, overall_center=overall_center))
    tris.extend(_triangulate_ring_cap(rings[-1], reverse=True, overall_center=overall_center))

    for k in range(len(rings) - 1):
        r0 = np.asarray(rings[k], dtype=float)
        r1 = np.asarray(rings[k + 1], dtype=float)
        n = min(r0.shape[0], r1.shape[0])
        for i in range(n):
            j = (i + 1) % n
            t1 = _orient_triangle_outward(np.asarray([r0[i], r0[j], r1[j]], dtype=float), overall_center)
            t2 = _orient_triangle_outward(np.asarray([r0[i], r1[j], r1[i]], dtype=float), overall_center)
            tris.extend([t1, t2])
    return tris


def _mesh_volume(triangles: list[np.ndarray]) -> float:
    vol = 0.0
    for tri in triangles:
        a, b, c = np.asarray(tri, dtype=float)
        vol += np.dot(a, np.cross(b, c)) / 6.0
    return float(abs(vol))


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _set_axes_equal(ax) -> None:
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    radius = 0.5 * max(x_range, y_range, z_range, 1.0)

    x_mid = sum(x_limits) * 0.5
    y_mid = sum(y_limits) * 0.5
    z_mid = sum(z_limits) * 0.5

    ax.set_xlim3d([x_mid - radius, x_mid + radius])
    ax.set_ylim3d([y_mid - radius, y_mid + radius])
    ax.set_zlim3d([z_mid - radius, z_mid + radius])


def _set_axes_equal_from_points(ax, points: np.ndarray) -> None:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        _set_axes_equal(ax)
        return

    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    mids = (mins + maxs) * 0.5
    spans = np.maximum(maxs - mins, 1.0)
    radius = 0.6 * float(np.max(spans))

    ax.set_xlim3d([mids[0] - radius, mids[0] + radius])
    ax.set_ylim3d([mids[1] - radius, mids[1] + radius])
    ax.set_zlim3d([mids[2] - radius, mids[2] + radius])


def VOM_VOA(
    data=None,
    entry=None,
    target=None,
    stand_dist: float = 10.0,
    space: str = "native",
    transforms=None,
    return_debug: bool = False,
) -> VOMVOAResult:
    """Compute VoA, VOM, and sVOM using the supplementary ellipse-based method.

    Method summary
    --------------
    1. Project irregular entry and target polygons into their PCA planes.
    2. Fit area-matched reference ellipses in PCA space and reconstruct them in 3D.
    3. Compute VoA from the centroid-to-centroid trajectory vs the target-plane normal.
    4. Build a physically defined 3D loft from interpolated elliptical sections.
    5. Compute VOM and sVOM as the enclosed volume of that solid and its distal segment.

    Notes on tilt and physical validity
    -----------------------------------
    A separate multiplicative “tilting adjustment” on volume is not physically rigorous.
    The physically valid approach is to define the corridor as an actual 3D solid and
    compute its enclosed volume directly. In this implementation, VoA remains an important
    orientation descriptor, but it does not artificially scale the volume. The supplementary
    heuristic tilt factors are still reported in debug output for comparison.
    """
    if entry is None or target is None:
        raise ValueError("Both entry and target must be provided.")

    E = _as_poly(data, entry)
    T = _as_poly(data, target)

    entry_fit = _fit_area_matched_ellipse(E)
    target_fit = _fit_area_matched_ellipse(T)

    cE = entry_fit["centroid"]
    cT = target_fit["centroid"]
    v1 = cT - cE
    H = float(np.linalg.norm(v1))
    if H < 1e-9:
        raise ValueError("Entry and target centroids are identical (zero target distance).")

    target_normal = _polygon_normal(T)
    theta_vs_normal = _angle_deg(v1, target_normal)
    voa = float(90.0 - abs(theta_vs_normal))
    voa = max(0.0, min(90.0, voa))

    a_target = float(target_fit["ellipse_a"])
    b_target = float(target_fit["ellipse_b"])
    a_entry = float(entry_fit["ellipse_a"])
    b_entry = float(entry_fit["ellipse_b"])
    area_target_poly = float(target_fit["polygon_area_3d"])
    area_entry_poly = float(entry_fit["polygon_area_3d"])
    area_target_ellipse = float(target_fit["ellipse_area"])
    area_entry_ellipse = float(entry_fit["ellipse_area"])

    full_rings = _loft_ring_stack(target_fit, entry_fit, ratio_end=1.0, n_sections=72)
    vom = _mesh_volume(_triangulate_ring_stack(full_rings))

    sd = float(stand_dist)
    if sd <= 0:
        raise ValueError("stand_dist must be > 0")
    sd = min(sd, H)
    cut_ratio = sd / H

    a_cut = a_target + (a_entry - a_target) * cut_ratio
    b_cut = b_target + (b_entry - b_target) * cut_ratio
    svom_rings = _loft_ring_stack(target_fit, entry_fit, ratio_end=cut_ratio, n_sections=48)
    svom = _mesh_volume(_triangulate_ring_stack(svom_rings))

    supplementary_svom_adjustment_factor = _supplementary_svom_adjustment_factor(voa)
    a_cut_adjusted = a_cut * (1.0 - supplementary_svom_adjustment_factor)
    b_cut_adjusted = b_cut * (1.0 - supplementary_svom_adjustment_factor)
    area_cut_poly_linear = area_target_poly + (area_entry_poly - area_target_poly) * cut_ratio
    area_cut_ellipse_linear = np.pi * a_cut * b_cut
    area_cut_ellipse_adjusted = np.pi * a_cut_adjusted * b_cut_adjusted

    legacy_vom_polygon_frustum_mm3 = _legacy_frustum_volume(area_target_poly, area_entry_poly, H)
    legacy_svom_polygon_frustum_mm3 = _legacy_frustum_volume(area_target_poly, area_cut_poly_linear, sd)

    legacy_vom_ellipse_frustum_mm3 = _legacy_frustum_volume(area_target_ellipse, area_entry_ellipse, H)
    legacy_svom_ellipse_frustum_mm3 = _legacy_frustum_volume(area_target_ellipse, area_cut_ellipse_linear, sd)
    legacy_svom_ellipse_adjusted_frustum_mm3 = _legacy_frustum_volume(
        area_target_ellipse,
        area_cut_ellipse_adjusted,
        sd,
    )

    supplementary_vom_tilt_weighted_mm3 = _legacy_tilt_weighted_elliptical_volume(
        a_start=a_target,
        b_start=b_target,
        a_end=a_entry,
        b_end=b_entry,
        H=H,
        voa_deg=voa,
    )
    supplementary_svom_tilt_weighted_mm3 = _legacy_tilt_weighted_elliptical_volume(
        a_start=a_target,
        b_start=b_target,
        a_end=a_cut_adjusted,
        b_end=b_cut_adjusted,
        H=sd,
        voa_deg=voa,
    )

    # Plot geometry
    target_ellipse_3d = target_fit["ellipse_3d"]
    entry_ellipse_3d = entry_fit["ellipse_3d"]

    cut_center = _interpolate_centers(cT, cE, cut_ratio)
    cut_basis = _interpolate_basis(target_fit["basis"], entry_fit["basis"], cut_ratio)
    cut_ellipse_3d = _ellipse_points_3d(
        cut_center,
        cut_basis,
        a_cut,
        b_cut,
        n_samples=target_ellipse_3d.shape[0],
    )

    frustum_surface = _build_section_surface(full_rings)
    svom_surface = _build_section_surface(svom_rings)

    dbg = None
    if return_debug:
        dbg = {
            "entry_polygon": E,
            "target_polygon": T,
            "entry_ellipse_3d": entry_ellipse_3d,
            "target_ellipse_3d": target_ellipse_3d,
            "svom_cut_ellipse_3d": cut_ellipse_3d,
            "frustum_surface": frustum_surface,
            "svom_surface": svom_surface,
            "centroid_entry": cE,
            "centroid_target": cT,
            "target_normal": target_normal,
            "distance_h": H,
            "voa_deg": voa,
            "angle_vs_normal_deg": theta_vs_normal,
            "tilt_deviation_deg": _tilt_deviation_deg(voa),
            "supplementary_volume_weight_endpoint": _supplementary_volume_weight_endpoint(voa),
            "supplementary_svom_adjustment_factor": supplementary_svom_adjustment_factor,
            "tilt_adjustment_applied_to_volume": False,
            "physical_volume_model": "3d_lofted_elliptical_frustum",
            "entry_polygon_area_mm2": area_entry_poly,
            "target_polygon_area_mm2": area_target_poly,
            "entry_ellipse_area_mm2": area_entry_ellipse,
            "target_ellipse_area_mm2": area_target_ellipse,
            "entry_ellipse_a_mm": a_entry,
            "entry_ellipse_b_mm": b_entry,
            "target_ellipse_a_mm": a_target,
            "target_ellipse_b_mm": b_target,
            "cut_ellipse_a_mm": a_cut,
            "cut_ellipse_b_mm": b_cut,
            "supplementary_cut_ellipse_a_mm": a_cut_adjusted,
            "supplementary_cut_ellipse_b_mm": b_cut_adjusted,
            "stand_dist_mm": sd,
            "comparison_metrics": {
                "physical_vom_mm3": float(vom),
                "physical_svom_mm3": float(svom),
                "legacy_polygon_frustum_vom_mm3": legacy_vom_polygon_frustum_mm3,
                "legacy_polygon_frustum_svom_mm3": legacy_svom_polygon_frustum_mm3,
                "legacy_ellipse_frustum_vom_mm3": legacy_vom_ellipse_frustum_mm3,
                "legacy_ellipse_frustum_svom_mm3": legacy_svom_ellipse_frustum_mm3,
                "legacy_ellipse_adjusted_svom_mm3": legacy_svom_ellipse_adjusted_frustum_mm3,
                "supplementary_tilt_weighted_vom_mm3": supplementary_vom_tilt_weighted_mm3,
                "supplementary_tilt_weighted_svom_mm3": supplementary_svom_tilt_weighted_mm3,
            },
            "space": space,
        }

    return VOMVOAResult(
        voa_deg=float(voa),
        vom_mm3=float(vom),
        svom_mm3=float(svom),
        debug=dbg,
    )
