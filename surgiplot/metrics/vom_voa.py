from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict, Union, Iterable
import numpy as np
from surgiplot.core.geometry.pca_plane import best_fit_plane_pca, project_to_plane_2d
from surgiplot.core.geometry.polygon_area import shoelace_area

@dataclass
class VOMVOAResult:
    voa_deg: float
    vom_mm3: float
    svom_mm3: float
    debug: Optional[Dict[str, Any]] = None

def _as_poly(data, spec) -> np.ndarray:
    if data is not None:
        return data.resolve_points(spec)
    arr = np.asarray(spec, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("Polygon must be (N,3)")
    return arr

def _poly_area_3d(poly3d: np.ndarray) -> float:
    pts2, _c, _basis = project_to_plane_2d(poly3d)
    return float(shoelace_area(pts2))

def _centroid(poly3d: np.ndarray) -> np.ndarray:
    return np.asarray(poly3d, dtype=float).mean(axis=0)

def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    num = float(np.dot(u, v))
    den = float(np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
    cosang = max(-1.0, min(1.0, num/den))
    return float(np.degrees(np.arccos(cosang)))

def VOM_VOA(
    data=None,
    entry=None,
    target=None,
    stand_dist: float = 10.0,
    space: str = "native",
    transforms=None,
    return_debug: bool = False,
) -> VOMVOAResult:
    """Compute VoA + VOM + sVOM from entry & target polygons.

    Notes
    -----
    - VoA definition implemented per your spec:
        v1 = centroid(entry) -> centroid(target) (trajectory / target distance vector)
        v2 = normal of target plane (PCA best-fit)
        VoA = 90° − | angle(v1, v2) |
      => 0° = trajectory parallel to target plane (poor), 90° = perpendicular (optimal)

    - VOM: corridor modeled as a frustum-like volume between entry area A1 and target area A2.
      Here we use the *frustum volume formula* (assumes similar cross-sections):
        V = h/3 * (A1 + A2 + sqrt(A1*A2))
      This is a pragmatic, stable default; you can later swap in the full ellipsoidal-cone integration.

    - sVOM: standardized VOM for a fixed height (stand_dist, mm) from target toward entry.
      We compute the interpolated area A(h) along the centroid-to-centroid axis and compute
      volume of the smaller frustum segment with height stand_dist ending at target.
    """
    if entry is None or target is None:
        raise ValueError("Both entry and target must be provided.")

    E = _as_poly(data, entry)
    T = _as_poly(data, target)

    cE = _centroid(E)
    cT = _centroid(T)
    v1 = cT - cE
    h = float(np.linalg.norm(v1))
    if h < 1e-9:
        raise ValueError("Entry and target centroids are identical (zero target distance).")

    _c, nT, _basis = best_fit_plane_pca(T)
    ang = _angle_deg(v1, nT)
    voa = 90.0 - abs(ang)

    A1 = _poly_area_3d(E)
    A2 = _poly_area_3d(T)

    vom = (h/3.0) * (A1 + A2 + np.sqrt(max(A1*A2, 0.0)))

    # sVOM: segment of height stand_dist ending at target
    sd = float(stand_dist)
    if sd <= 0:
        raise ValueError("stand_dist must be > 0")
    if sd > h:
        sd = h  # cap
    # linear interpolation of area vs distance (simple default)
    # distance from target toward entry: 0..h
    # area at distance d from target: A(d) = A2 + (A1-A2)*(d/h)
    A_at_sd = A2 + (A1 - A2) * (sd / h)
    svom = (sd/3.0) * (A2 + A_at_sd + np.sqrt(max(A2*A_at_sd, 0.0)))

    dbg = None
    if return_debug:
        dbg = {
            "centroid_entry": cE,
            "centroid_target": cT,
            "target_normal": nT,
            "distance_h": h,
            "A_entry": A1,
            "A_target": A2,
            "A_at_stand_dist": A_at_sd,
            "angle_v1_vs_normal_deg": ang,
            "space": space,
        }

    return VOMVOAResult(voa_deg=float(voa), vom_mm3=float(vom), svom_mm3=float(svom), debug=dbg)
