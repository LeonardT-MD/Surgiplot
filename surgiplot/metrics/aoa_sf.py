from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict
import numpy as np

@dataclass
class AOASFResult:
    aoa_vertical_deg: float
    aoa_horizontal_deg: float
    sf_proxy_mm2: float
    debug: Optional[Dict[str, Any]] = None

def _resolve_points(data, spec) -> np.ndarray:
    if data is not None:
        return data.resolve_points(spec)
    arr = np.asarray(spec, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1,3)
    return arr

def _resolve_point(data, spec) -> np.ndarray:
    pts = _resolve_points(data, spec)
    if pts.shape[0] != 1:
        raise ValueError("Expected a single point.")
    return pts[0]

def _angle_deg(u, v) -> float:
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    den = float(np.linalg.norm(u)*np.linalg.norm(v) + 1e-12)
    cosang = float(np.dot(u, v)/den)
    cosang = max(-1.0, min(1.0, cosang))
    return float(np.degrees(np.arccos(cosang)))

def AOA_SF(
    data=None,
    entry=None,
    target=None,
    constraints=None,
    technique: Optional[str] = None,
    space: str = "native",
    transforms=None,
    return_debug: bool = False,
) -> AOASFResult:
    """Compute AoA + a pragmatic SF proxy.

    This function is intentionally **generic** so the package runs end-to-end out of the box.
    You can later swap in your project-specific AoA/SF implementation without changing the API.

    Generic definitions used here
    -----------------------------
    - Let `cE` be the centroid of the entry polygon (or mean of provided entry points).
    - Let `T` be the target pivot point.
    - Direction vector: d = T - cE

    AoA components (in degrees):
    - Horizontal AoA: angle between projection of d onto axial plane (XY) and +Y axis
    - Vertical AoA:   angle between projection of d onto sagittal plane (YZ) and +Z axis

    SF proxy (mm^2):
    - If `constraints` are provided (points around the proximal workspace),
      we approximate SF as the area of the convex hull of constraints projected to the best-fit plane.
    - Otherwise returns 0.

    Parameters
    ----------
    entry:
      Entry polygon points or names (N>=1). If N>=3 behaves as polygon; if N<3 treated as mean point.
    target:
      Pivot/target point (single).
    constraints:
      Optional constraint/workspace points (N>=3) for an SF proxy area.
    """
    if entry is None or target is None:
        raise ValueError("Provide entry and target.")

    E = _resolve_points(data, entry)
    T = _resolve_point(data, target)

    cE = E.mean(axis=0)
    d = T - cE
    if np.linalg.norm(d) < 1e-9:
        raise ValueError("Entry centroid and target are identical.")

    # Horizontal: projection to XY plane vs +Y axis
    d_xy = np.array([d[0], d[1], 0.0])
    if np.linalg.norm(d_xy) < 1e-12:
        aoa_h = 0.0
    else:
        aoa_h = _angle_deg(d_xy, np.array([0.0, 1.0, 0.0]))

    # Vertical: projection to YZ plane vs +Z axis
    d_yz = np.array([0.0, d[1], d[2]])
    if np.linalg.norm(d_yz) < 1e-12:
        aoa_v = 0.0
    else:
        aoa_v = _angle_deg(d_yz, np.array([0.0, 0.0, 1.0]))

    sf_proxy = 0.0
    dbg = None

    if constraints is not None:
        from scipy.spatial import ConvexHull
        from surgiplot.core.geometry.pca_plane import project_to_plane_2d
        C = _resolve_points(data, constraints)
        if C.shape[0] >= 3:
            pts2, _c, _basis = project_to_plane_2d(C)
            try:
                hull = ConvexHull(pts2)
                sf_proxy = float(hull.volume)  # in 2D ConvexHull.volume == area
            except Exception:
                sf_proxy = 0.0

    if return_debug:
        dbg = {
            "centroid_entry": cE,
            "target": T,
            "direction": d,
            "technique": technique,
            "space": space,
        }

    return AOASFResult(
        aoa_vertical_deg=float(aoa_v),
        aoa_horizontal_deg=float(aoa_h),
        sf_proxy_mm2=float(sf_proxy),
        debug=dbg,
    )
