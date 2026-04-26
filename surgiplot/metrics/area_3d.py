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

    def plot(self, ax) -> None:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        dbg = self.debug or {}
        poly = dbg.get("polygon")
        if poly is None:
            raise ValueError("Area plot requires debug geometry.")

        P = np.asarray(poly, dtype=float)
        if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] < 3:
            raise ValueError("Invalid polygon debug geometry.")

        fitted = np.asarray(dbg.get("coplanar_polygon_3d", []), dtype=float)
        closed = np.vstack([P, P[0]])
        ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color="tab:orange", linewidth=2, alpha=0.92, label="Original polygon")
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], color="tab:orange", s=40, edgecolors="white", linewidths=0.55, depthshade=False)

        if fitted.ndim == 2 and fitted.shape == P.shape:
            fitted_closed = np.vstack([fitted, fitted[0]])
            ax.plot(
                fitted_closed[:, 0], fitted_closed[:, 1], fitted_closed[:, 2],
                color="#0f172a", linewidth=1.8, alpha=0.75, label="PCA fitted coplanar surface",
            )
            ax.add_collection3d(
                Poly3DCollection([fitted], facecolors="#94a3b8", alpha=0.16, edgecolors="none")
            )

        ax.set_title(f"Area={self.area_mm2:.2f} mm^2")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        pts = P if fitted.size == 0 else np.vstack([P, fitted])
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        mids = (mins + maxs) * 0.5
        spans = np.maximum(maxs - mins, 1.0)
        radius = 0.6 * float(np.max(spans))
        ax.set_xlim3d([mids[0] - radius, mids[0] + radius])
        ax.set_ylim3d([mids[1] - radius, mids[1] + radius])
        ax.set_zlim3d([mids[2] - radius, mids[2] + radius])
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc="best")


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
    coplanar_polygon_3d = pts2 @ basis[:, :2].T + c

    dbg = None
    if return_debug:
        dbg = {
            "polygon": P,
            "projected_polygon_2d": pts2,
            "coplanar_polygon_3d": coplanar_polygon_3d,
            "centroid": c,
            "basis": basis,
            "space": space,
            "n_points": int(P.shape[0]),
            "area_formula": "shoelace_on_pca_best_fit_plane_projection",
        }

    return Area3DResult(area_mm2=area, debug=dbg)
