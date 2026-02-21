from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict, Iterable, Union, Sequence
import numpy as np

PointSpec = Union[str, Sequence[str], np.ndarray, Sequence[Sequence[float]]]


@dataclass
class AOASFResult:
    aoa_vertical_deg: float
    aoa_horizontal_deg: float

    # SF metrics
    sf_entry_area_mm2: float                  # area of entry quad projected to best-fit plane
    sf_entry_area_rescaled_mm2: float         # same area after rescaling entry points to fixed radius from pivot (0 if not requested)
    sf_constraints_proxy_mm2: float           # legacy/optional: convex hull area of constraints projected to plane

    # Rescale metadata
    rescale_radius_mm: Optional[float] = None

    debug: Optional[Dict[str, Any]] = None


def _resolve_points(data, spec: PointSpec) -> np.ndarray:
    if data is not None:
        return data.resolve_points(spec)
    arr = np.asarray(spec, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, 3)
    return arr


def _resolve_point(data, spec: Union[str, Iterable[float], np.ndarray]) -> np.ndarray:
    pts = _resolve_points(data, spec)
    if pts.shape[0] != 1:
        raise ValueError("Expected a single point.")
    return pts[0]


def _angle_deg(u, v) -> float:
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    den = float(np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
    cosang = float(np.dot(u, v) / den)
    cosang = max(-1.0, min(1.0, cosang))
    return float(np.degrees(np.arccos(cosang)))


def _project_to_fixed_radius(p: np.ndarray, pivot: np.ndarray, radius: float) -> np.ndarray:
    """Project point p along pivot->p direction to be exactly `radius` away from pivot."""
    v = p - pivot
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return p.copy()
    return pivot + (v / n) * float(radius)


def _polygon_area_3d(points3d: np.ndarray) -> float:
    """
    Area of a 3D polygon (not necessarily planar) by projecting to best-fit PCA plane and applying shoelace.
    Requires your existing geometry helpers.
    """
    from surgiplot.core.geometry.pca_plane import project_to_plane_2d
    from surgiplot.core.geometry.polygon_area import shoelace_area

    pts2, _c, _basis = project_to_plane_2d(points3d)
    return float(shoelace_area(pts2))


def AOA_SF(
    data=None,
    entry=None,
    target=None,
    constraints=None,
    technique: Optional[str] = None,
    space: str = "native",
    transforms=None,
    return_debug: bool = False,
    sf_rescale_radius_mm: Optional[float] = None,  # NEW: rescale distance from pivot for SF + AoA triangles (geometry-only)
) -> AOASFResult:
    """
    AoA + SF (entry-area) with optional rescaling about the pivot.

    Recommended GUI mapping:
      cranial, caudal, medial, lateral -> entry (4 points)
      pivot -> target

    AoA components:
      - Vertical AoA: triangle (cranial, caudal, pivot)
      - Horizontal AoA: triangle (medial, lateral, pivot)

    SF:
      - Entry SF area: polygon area of (cranial, caudal, medial, lateral) projected to best-fit plane.
      - Rescaled entry SF area: same, but with the 4 entry points projected to a fixed radius from pivot.

    Optional legacy:
      - constraints (>=3 points): convex hull area projected to best-fit plane (kept as additional proxy).
    """
    if entry is None or target is None:
        raise ValueError("Provide entry and target/pivot.")

    E = _resolve_points(data, entry)
    if E.shape[0] < 2:
        raise ValueError("Entry must contain at least 2 points.")
    T = _resolve_point(data, target)

    # Expect 4 entry points for the new AoA/SF convention, but keep generic support:
    # If 4+ points: use first 2 for V-AoA endpoints, last 2 for H-AoA endpoints (best for GUI)
    if E.shape[0] < 4:
        # fall back to centroid-based approach for AoA if user passes generic polygon
        cE = E.mean(axis=0)
        d = T - cE
        d_xy = np.array([d[0], d[1], 0.0])
        d_yz = np.array([0.0, d[1], d[2]])
        aoa_h = 0.0 if np.linalg.norm(d_xy) < 1e-12 else _angle_deg(d_xy, np.array([0.0, 1.0, 0.0]))
        aoa_v = 0.0 if np.linalg.norm(d_yz) < 1e-12 else _angle_deg(d_yz, np.array([0.0, 0.0, 1.0]))
        sf_entry_area = 0.0
        sf_entry_area_rescaled = 0.0
    else:
        # interpret entry as [cranial, caudal, medial, lateral] (GUI order)
        p_cran = E[0]
        p_caud = E[1]
        p_med = E[2]
        p_lat = E[3]

        # AoA = included angle at pivot between vectors
        # Vertical: between (cranial - pivot) and (caudal - pivot)
        u_v = p_cran - T
        v_v = p_caud - T
        aoa_v = _angle_deg(u_v, v_v)

        # Horizontal: between (medial - pivot) and (lateral - pivot)
        u_h = p_med - T
        v_h = p_lat - T
        aoa_h = _angle_deg(u_h, v_h)

        # SF area from entry quadrilateral
        entry_quad = np.vstack([p_cran, p_caud, p_med, p_lat])
        sf_entry_area = _polygon_area_3d(entry_quad)

        sf_entry_area_rescaled = 0.0
        if sf_rescale_radius_mm is not None:
            r = float(sf_rescale_radius_mm)
            if r <= 0:
                raise ValueError("sf_rescale_radius_mm must be > 0")
            q2 = np.vstack([
                _project_to_fixed_radius(p_cran, T, r),
                _project_to_fixed_radius(p_caud, T, r),
                _project_to_fixed_radius(p_med,  T, r),
                _project_to_fixed_radius(p_lat,  T, r),
            ])
            sf_entry_area_rescaled = _polygon_area_3d(q2)

    # Optional constraints proxy (kept)
    sf_constraints_proxy = 0.0
    if constraints is not None:
        from scipy.spatial import ConvexHull
        from surgiplot.core.geometry.pca_plane import project_to_plane_2d

        C = _resolve_points(data, constraints)
        if C.shape[0] >= 3:
            pts2, _c, _basis = project_to_plane_2d(C)
            try:
                hull = ConvexHull(pts2)
                sf_constraints_proxy = float(hull.volume)  # area in 2D
            except Exception:
                sf_constraints_proxy = 0.0

    dbg = None
    if return_debug:
        dbg = {
            "pivot": T,
            "entry_points": E,
            "technique": technique,
            "space": space,
            "sf_rescale_radius_mm": sf_rescale_radius_mm,
        }

    return AOASFResult(
        aoa_vertical_deg=float(aoa_v),
        aoa_horizontal_deg=float(aoa_h),
        sf_entry_area_mm2=float(sf_entry_area),
        sf_entry_area_rescaled_mm2=float(sf_entry_area_rescaled),
        sf_constraints_proxy_mm2=float(sf_constraints_proxy),
        rescale_radius_mm=float(sf_rescale_radius_mm) if sf_rescale_radius_mm is not None else None,
        debug=dbg,
    )
