from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict, Iterable, Union, Sequence
import numpy as np
from surgiplot.core.geometry.pca_plane import project_to_plane_2d
from surgiplot.core.geometry.polygon_area import shoelace_area

PointSpec = Union[str, Sequence[str], np.ndarray, Sequence[Sequence[float]]]


@dataclass
class AOASFResult:
    aoa_vertical_deg: float
    aoa_horizontal_deg: float

    # SF metrics
    sf_entry_area_mm2: float                  # area of entry quad projected to best-fit plane
    sf_entry_area_rescaled_mm2: float         # same area after rescaling entry points to fixed radius from pivot (0 if not requested)

    # Rescale metadata
    rescale_radius_mm: Optional[float] = None

    debug: Optional[Dict[str, Any]] = None

    def plot(self, ax) -> None:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        dbg = self.debug or {}
        pivot = dbg.get("pivot")
        entry = dbg.get("entry_points")
        rescaled = dbg.get("rescaled_entry_points")
        vertical_triangle = dbg.get("vertical_triangle")
        horizontal_triangle = dbg.get("horizontal_triangle")
        vertical_triangle_rescaled = dbg.get("vertical_triangle_rescaled")
        horizontal_triangle_rescaled = dbg.get("horizontal_triangle_rescaled")

        all_points: list[np.ndarray] = []

        def _plot_triangle(points: np.ndarray, color: str, face_alpha: float, label: str, linestyle: str = "-") -> None:
            tri = np.asarray(points, dtype=float)
            closed = np.vstack([tri, tri[0]])
            ax.plot(
                closed[:, 0], closed[:, 1], closed[:, 2],
                color=color, linewidth=2.1, alpha=0.95, linestyle=linestyle, label=label,
            )
            ax.scatter(
                tri[:, 0], tri[:, 1], tri[:, 2],
                color=color, s=44, edgecolors="white", linewidths=0.55, depthshade=False,
            )
            ax.add_collection3d(
                Poly3DCollection([tri], facecolors=color, alpha=face_alpha, edgecolors="none")
            )
            all_points.append(tri)

        if vertical_triangle is not None:
            _plot_triangle(np.asarray(vertical_triangle, dtype=float), "tab:blue", 0.18, "V-AoA")
        if horizontal_triangle is not None:
            _plot_triangle(np.asarray(horizontal_triangle, dtype=float), "tab:green", 0.18, "H-AoA")

        if entry is not None:
            E = np.asarray(entry, dtype=float)
            if E.ndim == 2 and E.shape[1] == 3:
                all_points.append(E)

        if rescaled is not None:
            R = np.asarray(rescaled, dtype=float)
            if R.ndim == 2 and R.shape[1] == 3 and R.shape[0] >= 4:
                all_points.append(R)
                ax.scatter(
                    R[:, 0], R[:, 1], R[:, 2],
                    color="tab:orange", s=40, edgecolors="white", linewidths=0.55, depthshade=False, label="Projected points",
                )
                if entry is not None:
                    E = np.asarray(entry, dtype=float)
                    for original, projected in zip(E[:4], R[:4]):
                        ax.plot(
                            [original[0], projected[0]],
                            [original[1], projected[1]],
                            [original[2], projected[2]],
                            color="black",
                            linestyle=":",
                            linewidth=1.1,
                            alpha=0.85,
                        )
                for tri in (vertical_triangle_rescaled, horizontal_triangle_rescaled):
                    if tri is not None:
                        tri_arr = np.asarray(tri, dtype=float)
                        closed_tri = np.vstack([tri_arr, tri_arr[0]])
                        ax.plot(
                            closed_tri[:, 0], closed_tri[:, 1], closed_tri[:, 2],
                            color="black", linewidth=1.25, alpha=0.82, linestyle=":",
                        )
                sf_surface = dbg.get("sf_surface_rescaled_3d")
                surface = np.asarray(sf_surface, dtype=float) if sf_surface is not None else R[:4]
                closed = np.vstack([surface[:4], surface[0]])
                ax.plot(
                    closed[:, 0], closed[:, 1], closed[:, 2],
                    color="tab:orange", linewidth=1.7, alpha=0.92, label="Standardized SF surface",
                )
                if surface.ndim == 2 and surface.shape == (4, 3):
                    ax.add_collection3d(
                        Poly3DCollection([surface], facecolors="tab:orange", alpha=0.11, edgecolors="none")
                    )
                    all_points.append(surface)

        sf_surface = dbg.get("sf_surface_3d")
        if sf_surface is not None:
            S = np.asarray(sf_surface, dtype=float)
            if S.ndim == 2 and S.shape == (4, 3):
                closed = np.vstack([S, S[0]])
                ax.plot(
                    closed[:, 0], closed[:, 1], closed[:, 2],
                    color="#0f172a", linewidth=1.5, alpha=0.72, label="SF surface",
                )
                ax.add_collection3d(
                    Poly3DCollection([S], facecolors="#94a3b8", alpha=0.12, edgecolors="none")
                )
                all_points.append(S)

        if pivot is not None:
            T = np.asarray(pivot, dtype=float).reshape(3,)
            all_points.append(T.reshape(1, 3))
            ax.scatter(
                [T[0]], [T[1]], [T[2]],
                color="tab:red", s=74, label="Pivot", edgecolors="white", linewidths=0.7, depthshade=False,
            )

        if all_points:
            pts = np.vstack(all_points)
            mins = pts.min(axis=0)
            maxs = pts.max(axis=0)
            mids = (mins + maxs) * 0.5
            spans = np.maximum(maxs - mins, 1.0)
            radius = 0.6 * float(np.max(spans))
            ax.set_xlim3d([mids[0] - radius, mids[0] + radius])
            ax.set_ylim3d([mids[1] - radius, mids[1] + radius])
            ax.set_zlim3d([mids[2] - radius, mids[2] + radius])

        annotation = (
            f"V-AoA γ: {self.aoa_vertical_deg:.2f}°\n"
            f"H-AoA γ: {self.aoa_horizontal_deg:.2f}°"
        )
        ax.text2D(
            0.03,
            0.95,
            annotation,
            transform=ax.transAxes,
            fontsize=10.5,
            color="#c81e1e",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#d5d9e2", alpha=0.92),
        )
        ax.set_title(
            f"AoA(v)={self.aoa_vertical_deg:.2f} deg | "
            f"AoA(h)={self.aoa_horizontal_deg:.2f} deg | "
            f"SF={self.sf_entry_area_mm2:.2f} mm^2"
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc="best")


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
    pts2, _c, _basis = project_to_plane_2d(points3d)
    return float(shoelace_area(pts2))


def _coplanar_polygon(points3d: np.ndarray) -> tuple[np.ndarray, float]:
    pts = np.asarray(points3d, dtype=float)
    pts2, centroid, basis = project_to_plane_2d(pts)
    coplanar = pts2 @ basis[:, :2].T + centroid
    area = float(shoelace_area(pts2))
    return coplanar, area


def _triangle_sides(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> tuple[float, float, float]:
    a = float(np.linalg.norm(p2 - p3))
    b = float(np.linalg.norm(p1 - p3))
    c = float(np.linalg.norm(p1 - p2))
    return a, b, c


def _herons_area(a: float, b: float, c: float) -> float:
    s = 0.5 * (a + b + c)
    val = max(s * (s - a) * (s - b) * (s - c), 0.0)
    return float(np.sqrt(val))


def _triangle_debug(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Dict[str, Any]:
    a, b, c = _triangle_sides(p1, p2, p3)
    alpha = _angle_deg(p2 - p1, p3 - p1)
    beta = _angle_deg(p1 - p2, p3 - p2)
    gamma = _angle_deg(p1 - p3, p2 - p3)
    return {
        "points": np.vstack([p1, p2, p3]),
        "sides_mm": {"a": a, "b": b, "c": c},
        "angles_deg": {"alpha": alpha, "beta": beta, "gamma": gamma},
        "area_mm2": _herons_area(a, b, c),
    }


def AOA_SF(
    data=None,
    entry=None,
    target=None,
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

    """
    if entry is None or target is None:
        raise ValueError("Provide entry and target/pivot.")

    E = _resolve_points(data, entry)
    if E.shape[0] < 2:
        raise ValueError("Entry must contain at least 2 points.")
    T = _resolve_point(data, target)
    vertical_triangle = None
    horizontal_triangle = None
    vertical_triangle_rescaled = None
    horizontal_triangle_rescaled = None
    sf_surface_3d = None
    sf_surface_rescaled_3d = None

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
        q2 = None
    else:
        # interpret entry as [cranial, caudal, medial, lateral] (GUI order)
        p_cran = E[0]
        p_caud = E[1]
        p_med = E[2]
        p_lat = E[3]

        vertical_triangle = _triangle_debug(p_cran, p_caud, T)
        horizontal_triangle = _triangle_debug(p_med, p_lat, T)
        aoa_v = float(vertical_triangle["angles_deg"]["gamma"])
        aoa_h = float(horizontal_triangle["angles_deg"]["gamma"])

        # SF surface uses the anatomical non-crossing order:
        # cranial -> lateral -> caudal -> medial
        entry_quad_ordered = np.vstack([p_cran, p_lat, p_caud, p_med])
        sf_surface_3d, sf_entry_area = _coplanar_polygon(entry_quad_ordered)

        sf_entry_area_rescaled = 0.0
        q2 = None
        sf_surface_rescaled_3d = None
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
            vertical_triangle_rescaled = _triangle_debug(q2[0], q2[1], T)
            horizontal_triangle_rescaled = _triangle_debug(q2[2], q2[3], T)
            rescaled_quad_ordered = np.vstack([q2[0], q2[3], q2[1], q2[2]])
            sf_surface_rescaled_3d, sf_entry_area_rescaled = _coplanar_polygon(rescaled_quad_ordered)

    dbg = None
    if return_debug:
        dbg = {
            "pivot": T,
            "entry_points": E,
            "rescaled_entry_points": q2,
            "technique": technique,
            "space": space,
            "sf_rescale_radius_mm": sf_rescale_radius_mm,
            "entry_quad_area_mm2": sf_entry_area,
            "standardized_entry_quad_area_mm2": sf_entry_area_rescaled,
            "sf_surface_vertex_order": ["cranial", "lateral", "caudal", "medial"] if E.shape[0] >= 4 else None,
            "sf_surface_3d": sf_surface_3d,
            "sf_surface_rescaled_3d": sf_surface_rescaled_3d,
            "vertical_triangle": vertical_triangle["points"] if E.shape[0] >= 4 else None,
            "horizontal_triangle": horizontal_triangle["points"] if E.shape[0] >= 4 else None,
            "vertical_triangle_properties": vertical_triangle if E.shape[0] >= 4 else None,
            "horizontal_triangle_properties": horizontal_triangle if E.shape[0] >= 4 else None,
            "vertical_triangle_rescaled": vertical_triangle_rescaled["points"] if vertical_triangle_rescaled is not None else None,
            "horizontal_triangle_rescaled": horizontal_triangle_rescaled["points"] if horizontal_triangle_rescaled is not None else None,
            "vertical_triangle_rescaled_properties": vertical_triangle_rescaled,
            "horizontal_triangle_rescaled_properties": horizontal_triangle_rescaled,
        }

    return AOASFResult(
        aoa_vertical_deg=float(aoa_v),
        aoa_horizontal_deg=float(aoa_h),
        sf_entry_area_mm2=float(sf_entry_area),
        sf_entry_area_rescaled_mm2=float(sf_entry_area_rescaled),
        rescale_radius_mm=float(sf_rescale_radius_mm) if sf_rescale_radius_mm is not None else None,
        debug=dbg,
    )
