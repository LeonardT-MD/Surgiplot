from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict
import numpy as np

from surgiplot.core.geometry.pca_plane import project_to_plane_2d
from surgiplot.core.geometry.polygon_area import shoelace_area


@dataclass
class Area3DResult:
    area_mm2: float
    debug: Optional[Dict[str, Any]] = None


def _resolve_poly(data, spec) -> np.ndarray:
    if data is not None:
        # harden lists of names
        if isinstance(spec, (list, tuple)) and spec and all(isinstance(x, str) for x in spec):
            if hasattr(data, "resolve_names"):
                spec = data.resolve_names(spec)
        poly = data.resolve_points(spec)
    else:
        poly = np.asarray(spec, dtype=float)

    if poly.ndim != 2 or poly.shape[1] != 3:
        raise ValueError("Polygon must be an (N,3) array-like.")
    if poly.shape[0] < 3:
        raise ValueError("Need at least 3 points to compute an area.")
    return poly


def AREA_3D(
    data=None,
    polygon=None,
    space: str = "native",
    transforms=None,
    return_debug: bool = False,
) -> Area3DResult:
    """Area (mm^2) of a 3D polygon defined by >=3 points.

    Uses best-fit plane projection + shoelace.
    """
    if polygon is None:
        raise ValueError("Provide polygon (>=3 points or names/labels).")

    P = _resolve_poly(data, polygon)
    pts2, c, basis = project_to_plane_2d(P)
    area = float(shoelace_area(pts2))

    dbg = None
    if return_debug:
        dbg = {
            "centroid": c,
            "basis": basis,
            "space": space,
            "n_points": int(P.shape[0]),
        }

    return Area3DResult(area_mm2=area, debug=dbg)
