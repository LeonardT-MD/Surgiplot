from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict

import numpy as np

from surgiplot.core.geometry.pca_plane import project_to_plane_2d
from surgiplot.core.geometry.polygon_area import shoelace_area


ELLIPSE_POLYGON_AREA_RTOL = 1e-4
NEAR_CIRCLE_AXIS_RATIO = 1.05
LOFT_VOLUME_RTOL = 1e-4
LOFT_VOLUME_ATOL_MM3 = 1e-6
LOFT_MAX_SECTIONS = 257
VOM_METHOD_VERSION = "3d-loft-2.0"


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

        if cE is not None and cT is not None:
            cE = np.asarray(cE, dtype=float).reshape(3,)
            cT = np.asarray(cT, dtype=float).reshape(3,)
            approach = cT - cE
            norm = float(np.linalg.norm(approach))
            if norm > 1e-9:
                scale = max(float(dbg.get("distance_h", 0.0) or 0.0) * 0.25, 1.0)
                direction = approach / norm
                start = cT - direction * scale
                all_plot_points.append(np.vstack([start, cT]))
                ax.quiver(
                    start[0], start[1], start[2],
                    direction[0], direction[1], direction[2],
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
    if not np.all(np.isfinite(arr)):
        raise ValueError("Polygon coordinates must be finite.")
    if np.unique(arr, axis=0).shape[0] < 3:
        raise ValueError("Polygon requires at least three distinct points.")
    return arr


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=float).reshape(3,)
    v = np.asarray(v, dtype=float).reshape(3,)
    u_norm = float(np.linalg.norm(u))
    v_norm = float(np.linalg.norm(v))
    if u_norm <= 1e-12 or v_norm <= 1e-12:
        raise ValueError("Cannot compute an angle from a zero-length vector.")
    den = u_norm * v_norm
    cosang = float(np.dot(u, v) / den)
    cosang = max(-1.0, min(1.0, cosang))
    return float(np.degrees(np.arccos(cosang)))


def _conic_to_ellipse_geometry(coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert Ax2+Bxy+Cy2+Dx+Ey+F=0 to center, semiaxes, and a 2D rotation."""
    A, B, C, D, E, F = np.asarray(coefficients, dtype=float).reshape(6,)
    Q = np.asarray([[A, 0.5 * B], [0.5 * B, C]], dtype=float)
    linear = np.asarray([D, E], dtype=float)
    eigenvalues = np.linalg.eigvalsh(Q)
    if np.all(eigenvalues < 0.0):
        Q = -Q
        linear = -linear
        F = -F
        eigenvalues = -eigenvalues[::-1]
    if np.min(eigenvalues) <= np.finfo(float).eps:
        raise ValueError("Fitted conic is not a positive-definite ellipse.")

    center = -0.5 * np.linalg.solve(Q, linear)
    level = -(float(center @ Q @ center) + float(linear @ center) + float(F))
    if not np.isfinite(level) or level <= np.finfo(float).eps:
        raise ValueError("Fitted ellipse has a non-positive scale.")

    values, vectors = np.linalg.eigh(Q)
    axes = np.sqrt(level / values)
    order = np.argsort(axes)[::-1]
    axes = axes[order]
    vectors = vectors[:, order]
    if np.linalg.det(vectors) < 0.0:
        vectors[:, 1] *= -1.0
    return center, axes, vectors


def _direct_ellipse_fit(poly2d: np.ndarray, weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Numerically scaled Fitzgibbon/Halir-Flusser direct least-squares ellipse fit."""
    points = np.asarray(poly2d, dtype=float)
    if points.shape[0] < 5:
        raise ValueError("At least five points are required for direct ellipse fitting.")
    if weights is None:
        weights = np.ones(points.shape[0], dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(points.shape[0],)
    weights = np.clip(weights, 0.0, np.inf)
    if np.count_nonzero(weights > 1e-8) < 5:
        raise ValueError("At least five weighted inliers are required for direct ellipse fitting.")

    total_weight = float(weights.sum())
    origin = np.sum(points * weights[:, None], axis=0) / total_weight
    centered = points - origin
    scale = np.sqrt(np.sum(centered * centered * weights[:, None], axis=0) / total_weight)
    scale = np.where(scale > 1e-9, scale, 1.0)
    normalized = centered / scale
    x = normalized[:, 0]
    y = normalized[:, 1]
    D1 = np.column_stack([x * x, x * y, y * y])
    D2 = np.column_stack([x, y, np.ones_like(x)])
    root_weight = np.sqrt(weights)[:, None]
    D1w = D1 * root_weight
    D2w = D2 * root_weight
    S1 = D1w.T @ D1w
    S2 = D1w.T @ D2w
    S3 = D2w.T @ D2w
    T = -np.linalg.pinv(S3, rcond=1e-12) @ S2.T
    constraint = np.asarray([[0.0, 0.0, 2.0], [0.0, -1.0, 0.0], [2.0, 0.0, 0.0]])
    matrix = np.linalg.solve(constraint, S1 + S2 @ T)
    _, vectors = np.linalg.eig(matrix)

    candidates: list[tuple[float, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    transform = np.asarray(
        [
            [1.0 / scale[0], 0.0, -origin[0] / scale[0]],
            [0.0, 1.0 / scale[1], -origin[1] / scale[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    for column in range(vectors.shape[1]):
        quadratic = np.real_if_close(vectors[:, column], tol=1000)
        if np.iscomplexobj(quadratic):
            continue
        quadratic = np.asarray(quadratic, dtype=float)
        if 4.0 * quadratic[0] * quadratic[2] - quadratic[1] ** 2 <= 0.0:
            continue
        normalized_coefficients = np.concatenate([quadratic, T @ quadratic])
        A, B, C, D, E, F = normalized_coefficients
        conic_normalized = np.asarray(
            [[A, 0.5 * B, 0.5 * D], [0.5 * B, C, 0.5 * E], [0.5 * D, 0.5 * E, F]],
            dtype=float,
        )
        conic = transform.T @ conic_normalized @ transform
        coefficients = np.asarray(
            [conic[0, 0], 2.0 * conic[0, 1], conic[1, 1], 2.0 * conic[0, 2], 2.0 * conic[1, 2], conic[2, 2]],
            dtype=float,
        )
        try:
            geometry = _conic_to_ellipse_geometry(coefficients)
        except (ValueError, np.linalg.LinAlgError):
            continue
        values = np.column_stack(
            [
                points[:, 0] ** 2,
                points[:, 0] * points[:, 1],
                points[:, 1] ** 2,
                points[:, 0],
                points[:, 1],
                np.ones(points.shape[0]),
            ]
        ) @ coefficients
        gradient_x = 2.0 * coefficients[0] * points[:, 0] + coefficients[1] * points[:, 1] + coefficients[3]
        gradient_y = coefficients[1] * points[:, 0] + 2.0 * coefficients[2] * points[:, 1] + coefficients[4]
        residuals = np.abs(values) / np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y + 1e-18)
        score = float(np.sum(weights * residuals * residuals) / total_weight)
        candidates.append((score, coefficients, geometry))
    if not candidates:
        raise ValueError("Direct least-squares fitting did not produce a valid ellipse.")
    score, coefficients, geometry = min(candidates, key=lambda item: item[0])
    center, axes, rotation = geometry
    return {
        "coefficients": coefficients,
        "center": center,
        "axes": axes,
        "rotation": rotation,
        "score": score,
    }


def _ellipse_residuals(points: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    x = np.asarray(points, dtype=float)[:, 0]
    y = np.asarray(points, dtype=float)[:, 1]
    A, B, C, D, E, F = np.asarray(coefficients, dtype=float)
    values = A * x * x + B * x * y + C * y * y + D * x + E * y + F
    gradient_x = 2.0 * A * x + B * y + D
    gradient_y = B * x + 2.0 * C * y + E
    return np.abs(values) / np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y + 1e-18)


def _robust_direct_ellipse_fit(poly2d: np.ndarray, max_iterations: int = 8) -> Dict[str, Any]:
    """Fit an ellipse with Tukey-biweight IRLS and report contour outliers explicitly."""
    points = np.asarray(poly2d, dtype=float)
    weights = np.ones(points.shape[0], dtype=float)
    fit = _direct_ellipse_fit(points, weights)

    # Deterministic RANSAC-style initialization prevents one misplaced contour
    # point from dominating the algebraic least-squares solution. Five points
    # are the minimum for an ellipse-specific conic fit.
    rng = np.random.default_rng(0)
    subsets: list[np.ndarray] = []
    if points.shape[0] == 5:
        subsets.append(np.arange(5, dtype=int))
    else:
        for _ in range(min(128, 8 * points.shape[0])):
            subsets.append(np.sort(rng.choice(points.shape[0], size=5, replace=False)))
    best_score = float(np.median(_ellipse_residuals(points, fit["coefficients"])))
    best_fit = fit
    for subset in subsets:
        subset_weights = np.zeros(points.shape[0], dtype=float)
        subset_weights[subset] = 1.0
        try:
            candidate = _direct_ellipse_fit(points, subset_weights)
        except (ValueError, np.linalg.LinAlgError):
            continue
        candidate_residuals = _ellipse_residuals(points, candidate["coefficients"])
        candidate_score = float(np.median(candidate_residuals))
        if candidate_score < best_score:
            best_score = candidate_score
            best_fit = candidate
    fit = best_fit
    initial_residuals = _ellipse_residuals(points, fit["coefficients"])
    initial_median = float(np.median(initial_residuals))
    initial_mad = float(np.median(np.abs(initial_residuals - initial_median)))
    initial_scale = max(1.4826 * initial_mad, 1e-9 * max(float(np.ptp(points, axis=0).max()), 1.0))
    initial_u = np.maximum(0.0, initial_residuals - initial_median) / (4.685 * initial_scale)
    weights = np.where(initial_u < 1.0, (1.0 - initial_u * initial_u) ** 2, 0.0)
    if np.count_nonzero(weights > 1e-8) < 5:
        weights[:] = 0.0
        weights[np.argsort(initial_residuals)[:5]] = 1.0
    fit = _direct_ellipse_fit(points, weights)
    for _ in range(int(max_iterations)):
        residuals = _ellipse_residuals(points, fit["coefficients"])
        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median)))
        robust_scale = max(1.4826 * mad, np.finfo(float).eps * max(float(np.ptp(points, axis=0).max()), 1.0))
        u = np.maximum(0.0, residuals - median) / (4.685 * robust_scale)
        new_weights = np.where(u < 1.0, (1.0 - u * u) ** 2, 0.0)
        if np.count_nonzero(new_weights > 1e-8) < 5:
            keep = np.argsort(residuals)[:5]
            new_weights[:] = 0.0
            new_weights[keep] = 1.0
        if np.max(np.abs(new_weights - weights)) < 1e-5:
            weights = new_weights
            break
        weights = new_weights
        fit = _direct_ellipse_fit(points, weights)

    fit = _direct_ellipse_fit(points, weights)
    residuals = _ellipse_residuals(points, fit["coefficients"])
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    robust_scale = max(1.4826 * mad, np.finfo(float).eps * max(float(np.ptp(points, axis=0).max()), 1.0))
    inliers = residuals <= median + 4.685 * robust_scale
    if np.count_nonzero(inliers) < 5:
        inliers[:] = False
        inliers[np.argsort(residuals)[:5]] = True
    fit.update({"weights": weights, "residuals": residuals, "inlier_mask": inliers})
    return fit


def _ellipse_sample_count(max_relative_area_error: float = ELLIPSE_POLYGON_AREA_RTOL) -> int:
    """Choose the smallest regular-polygon sample count satisfying an ellipse area bound."""
    tolerance = float(max_relative_area_error)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("max_relative_area_error must be finite and positive")
    for count in range(8, 4097):
        relative_error = 1.0 - count * np.sin(2.0 * np.pi / count) / (2.0 * np.pi)
        if relative_error <= tolerance:
            return count
    raise RuntimeError("Requested ellipse polygonization tolerance requires more than 4096 samples")


def _fit_area_matched_ellipse(poly3d: np.ndarray, n_samples: Optional[int] = None) -> Dict[str, Any]:
    if n_samples is None:
        n_samples = _ellipse_sample_count()
    pts2, centroid, basis = project_to_plane_2d(poly3d)
    raw_area_polygon = float(shoelace_area(pts2))
    scale = max(float(np.ptp(pts2, axis=0).max()), 1.0)
    if raw_area_polygon <= np.finfo(float).eps * scale * scale * 100.0:
        raise ValueError("Polygon is degenerate or its vertex order produces zero area.")

    fit_method = "direct_least_squares_tukey_irls"
    outlier_indices: list[int] = []
    fit_center_2d = np.zeros(2, dtype=float)
    fit_rotation_2d = np.eye(2, dtype=float)
    try:
        robust_fit = _robust_direct_ellipse_fit(pts2)
        fit_center_2d = np.asarray(robust_fit["center"], dtype=float)
        provisional_axes = np.asarray(robust_fit["axes"], dtype=float)
        fit_rotation_2d = np.asarray(robust_fit["rotation"], dtype=float)
        inlier_mask = np.asarray(robust_fit["inlier_mask"], dtype=bool)
        outlier_indices = np.flatnonzero(~inlier_mask).astype(int).tolist()
        inlier_polygon = pts2[inlier_mask]
        area_polygon = float(shoelace_area(inlier_polygon)) if inlier_polygon.shape[0] >= 3 else raw_area_polygon
        if area_polygon <= np.finfo(float).eps * scale * scale * 100.0:
            area_polygon = raw_area_polygon
            outlier_indices = []
    except (ValueError, np.linalg.LinAlgError):
        fit_method = "svd_extent_fallback"
        spread = np.max(np.abs(pts2), axis=0)
        spread = np.where(spread > 1e-9, spread, np.std(pts2, axis=0))
        provisional_axes = np.where(spread > 1e-9, spread, 1.0)
        area_polygon = raw_area_polygon

    # Preserve the fitted aspect ratio while imposing exact equality between
    # the analytical ellipse area and the robust ordered-contour area.
    uniform_scale = float(np.sqrt(max(area_polygon, 1e-12) / (np.pi * provisional_axes[0] * provisional_axes[1])))
    a = float(uniform_scale * provisional_axes[0])
    b = float(uniform_scale * provisional_axes[1])
    near_circular = bool(max(a, b) / min(a, b) <= NEAR_CIRCLE_AXIS_RATIO)
    if near_circular:
        a = b = float(np.sqrt(area_polygon / np.pi))
        fit_rotation_2d = np.eye(2, dtype=float)

    in_plane_basis = basis[:, :2] @ fit_rotation_2d
    u = in_plane_basis[:, 0] / np.linalg.norm(in_plane_basis[:, 0])
    v = in_plane_basis[:, 1] - u * np.dot(in_plane_basis[:, 1], u)
    v = v / np.linalg.norm(v)
    normal = np.cross(u, v)
    if np.dot(normal, basis[:, 2]) < 0.0:
        v *= -1.0
        normal *= -1.0
    ellipse_basis = np.column_stack([u, v, normal])
    ellipse_center = centroid + basis[:, :2] @ fit_center_2d

    theta = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    ellipse2d_local = np.column_stack([a * np.cos(theta), b * np.sin(theta)])
    ellipse2d = ellipse2d_local @ fit_rotation_2d.T + fit_center_2d
    ellipse3d = ellipse2d_local @ ellipse_basis[:, :2].T + ellipse_center

    return {
        "polygon_area_2d": area_polygon,
        "polygon_area_3d": area_polygon,
        "raw_polygon_area_3d": raw_area_polygon,
        "centroid": ellipse_center,
        "vertex_centroid": centroid,
        "basis": ellipse_basis,
        "ellipse_a": a,
        "ellipse_b": b,
        "ellipse_area": float(np.pi * a * b),
        "ellipse_2d": ellipse2d,
        "ellipse_3d": ellipse3d,
        "projected_polygon_2d": pts2,
        "ellipse_fit_method": fit_method,
        "ellipse_fit_outlier_indices": outlier_indices,
        "ellipse_fit_near_circular_regularization": near_circular,
    }


def _ellipse_points_3d(center: np.ndarray, basis: np.ndarray, a: float, b: float, n_samples: int = 120) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    pts2 = np.column_stack([a * np.cos(theta), b * np.sin(theta)])
    return pts2 @ basis[:, :2].T + center


def _interpolate_centers(target_center: np.ndarray, entry_center: np.ndarray, ratio: float) -> np.ndarray:
    return target_center + (entry_center - target_center) * float(ratio)


def _rotation_angle(first: np.ndarray, second: np.ndarray) -> float:
    cosine = (float(np.trace(np.asarray(first).T @ np.asarray(second))) - 1.0) * 0.5
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _align_ellipse_frame(reference_fit: Dict[str, Any], moving_fit: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve PCA sign/axis ambiguity using the smallest entry-target rotation."""
    reference = np.asarray(reference_fit["basis"], dtype=float)
    moving = np.asarray(moving_fit["basis"], dtype=float).copy()
    a = float(moving_fit["ellipse_a"])
    b = float(moving_fit["ellipse_b"])

    if np.dot(reference[:, 2], moving[:, 2]) < 0.0:
        moving[:, 1] *= -1.0
        moving[:, 2] *= -1.0

    if bool(moving_fit.get("ellipse_fit_near_circular_regularization")):
        normal = moving[:, 2]
        u = reference[:, 0] - normal * np.dot(reference[:, 0], normal)
        if np.linalg.norm(u) <= 1e-10:
            u = reference[:, 1] - normal * np.dot(reference[:, 1], normal)
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)
        candidates = [(np.column_stack([u, v, normal]), a, b, "circular_parallel_transport")]
    else:
        u, v, normal = moving[:, 0], moving[:, 1], moving[:, 2]
        candidates = [
            (np.column_stack([u, v, normal]), a, b, "identity"),
            (np.column_stack([-u, -v, normal]), a, b, "sign_flip_uv"),
            (np.column_stack([v, -u, normal]), b, a, "swap_v_minus_u"),
            (np.column_stack([-v, u, normal]), b, a, "swap_minus_v_u"),
        ]

    aligned_basis, aligned_a, aligned_b, operation = min(
        candidates,
        key=lambda candidate: _rotation_angle(reference, candidate[0]),
    )
    aligned = dict(moving_fit)
    aligned["basis"] = aligned_basis
    aligned["ellipse_a"] = float(aligned_a)
    aligned["ellipse_b"] = float(aligned_b)
    aligned["ellipse_3d"] = _ellipse_points_3d(
        np.asarray(aligned["centroid"], dtype=float),
        aligned_basis,
        float(aligned_a),
        float(aligned_b),
        n_samples=np.asarray(moving_fit["ellipse_3d"]).shape[0],
    )
    aligned["frame_alignment_operation"] = operation
    aligned["frame_alignment_rotation_deg"] = float(np.degrees(_rotation_angle(reference, aligned_basis)))
    return aligned


def _rotation_matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [0.25 * s, (matrix[2, 1] - matrix[1, 2]) / s, (matrix[0, 2] - matrix[2, 0]) / s, (matrix[1, 0] - matrix[0, 1]) / s]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            s = np.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 0.0)) * 2.0
            quaternion = np.asarray([ (matrix[2, 1] - matrix[1, 2]) / s, 0.25 * s, (matrix[0, 1] + matrix[1, 0]) / s, (matrix[0, 2] + matrix[2, 0]) / s ])
        elif index == 1:
            s = np.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 0.0)) * 2.0
            quaternion = np.asarray([ (matrix[0, 2] - matrix[2, 0]) / s, (matrix[0, 1] + matrix[1, 0]) / s, 0.25 * s, (matrix[1, 2] + matrix[2, 1]) / s ])
        else:
            s = np.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 0.0)) * 2.0
            quaternion = np.asarray([ (matrix[1, 0] - matrix[0, 1]) / s, (matrix[0, 2] + matrix[2, 0]) / s, (matrix[1, 2] + matrix[2, 1]) / s, 0.25 * s ])
    return quaternion / np.linalg.norm(quaternion)


def _quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=float) / np.linalg.norm(quaternion)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _slerp_rotation(first: np.ndarray, second: np.ndarray, ratio: float) -> np.ndarray:
    q0 = _rotation_matrix_to_quaternion(first)
    q1 = _rotation_matrix_to_quaternion(second)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        q = q0 + float(ratio) * (q1 - q0)
        return _quaternion_to_rotation_matrix(q / np.linalg.norm(q))
    angle = float(np.arccos(dot))
    sin_angle = float(np.sin(angle))
    q = np.sin((1.0 - float(ratio)) * angle) / sin_angle * q0 + np.sin(float(ratio) * angle) / sin_angle * q1
    return _quaternion_to_rotation_matrix(q)


def _interpolate_basis(target_basis: np.ndarray, entry_basis: np.ndarray, ratio: float) -> np.ndarray:
    return _slerp_rotation(np.asarray(target_basis, dtype=float), np.asarray(entry_basis, dtype=float), float(ratio))


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
    for i in range(n):
        j = (i + 1) % n
        faces.append([A[i], A[j], B[j], B[i]])
    return faces


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
    """Return a closed triangle mesh volume in a local numerical frame.

    Subtracting a reference point makes the signed-tetrahedron sum stable and
    translation invariant even for navigation coordinates far from the origin.
    """
    if not triangles:
        return 0.0
    mesh = np.asarray(triangles, dtype=float)
    reference = mesh.reshape(-1, 3).mean(axis=0)
    shifted = mesh - reference
    contributions = np.einsum("ij,ij->i", shifted[:, 0], np.cross(shifted[:, 1], shifted[:, 2])) / 6.0
    return float(abs(np.sum(contributions)))


def _loft_volume_converged(
    target_fit: Dict[str, Any],
    entry_fit: Dict[str, Any],
    ratio_end: float,
    *,
    relative_tolerance: float = LOFT_VOLUME_RTOL,
    absolute_tolerance_mm3: float = LOFT_VOLUME_ATOL_MM3,
    maximum_sections: int = LOFT_MAX_SECTIONS,
) -> tuple[float, list[np.ndarray], Dict[str, Any]]:
    """Refine longitudinal rings until successive closed-mesh volumes converge."""
    previous_volume: Optional[float] = None
    sections = 9
    history: list[Dict[str, float]] = []
    while True:
        rings = _loft_ring_stack(target_fit, entry_fit, ratio_end=ratio_end, n_sections=sections)
        volume = _mesh_volume(_triangulate_ring_stack(rings))
        history.append({"sections": float(sections), "volume_mm3": float(volume)})
        if previous_volume is not None:
            difference = abs(volume - previous_volume)
            threshold = float(absolute_tolerance_mm3) + float(relative_tolerance) * max(abs(volume), abs(previous_volume))
            if difference <= threshold:
                return volume, rings, {
                    "converged": True,
                    "sections": sections,
                    "relative_tolerance": float(relative_tolerance),
                    "absolute_tolerance_mm3": float(absolute_tolerance_mm3),
                    "history": history,
                }
        if sections >= int(maximum_sections):
            return volume, rings, {
                "converged": False,
                "sections": sections,
                "relative_tolerance": float(relative_tolerance),
                "absolute_tolerance_mm3": float(absolute_tolerance_mm3),
                "history": history,
            }
        previous_volume = volume
        sections = min(2 * (sections - 1) + 1, int(maximum_sections))


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
    2. Robustly fit direct least-squares ellipses and rescale them to the inlier polygon areas.
    3. Compute VoA from the centroid-to-centroid trajectory vs the target-plane normal.
    4. Align ellipse frames and interpolate orientation with quaternion SLERP.
    5. Build a physically defined 3D loft with convergence-controlled longitudinal refinement.
    6. Compute VOM and sVOM as the enclosed volume of that solid and its distal segment.

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

    target_fit = _fit_area_matched_ellipse(T)
    entry_fit = _align_ellipse_frame(target_fit, _fit_area_matched_ellipse(E))

    cE = entry_fit["centroid"]
    cT = target_fit["centroid"]
    v1 = cT - cE
    H = float(np.linalg.norm(v1))
    if H < 1e-9:
        raise ValueError("Entry and target centroids are identical (zero target distance).")

    # VoA uses the normalized target SVD-plane normal. Its sign is immaterial
    # because the acute trajectory-normal angle is used below.
    target_normal = np.asarray(target_fit["basis"], dtype=float)[:, 2]
    target_normal = target_normal / np.linalg.norm(target_normal)
    theta_vs_normal = _angle_deg(v1, target_normal)
    acute_theta_vs_normal = min(float(theta_vs_normal), float(180.0 - theta_vs_normal))
    voa = float(90.0 - acute_theta_vs_normal)
    voa = max(0.0, min(90.0, voa))

    a_target = float(target_fit["ellipse_a"])
    b_target = float(target_fit["ellipse_b"])
    a_entry = float(entry_fit["ellipse_a"])
    b_entry = float(entry_fit["ellipse_b"])
    area_target_poly = float(target_fit["polygon_area_3d"])
    area_entry_poly = float(entry_fit["polygon_area_3d"])
    area_target_ellipse = float(target_fit["ellipse_area"])
    area_entry_ellipse = float(entry_fit["ellipse_area"])

    vom, full_rings, vom_convergence = _loft_volume_converged(target_fit, entry_fit, ratio_end=1.0)

    requested_sd = float(stand_dist)
    if not np.isfinite(requested_sd) or requested_sd <= 0:
        raise ValueError("stand_dist must be a finite value > 0")
    sd = min(requested_sd, H)
    cut_ratio = sd / H

    a_cut = a_target + (a_entry - a_target) * cut_ratio
    b_cut = b_target + (b_entry - b_target) * cut_ratio
    if cut_ratio >= 1.0 - 1e-12:
        svom = float(vom)
        svom_rings = full_rings
        svom_convergence = dict(vom_convergence)
        svom_convergence["reused_full_vom_mesh"] = True
    else:
        svom, svom_rings, svom_convergence = _loft_volume_converged(
            target_fit,
            entry_fit,
            ratio_end=cut_ratio,
        )
        svom_convergence["reused_full_vom_mesh"] = False

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
            "acute_angle_vs_normal_deg": acute_theta_vs_normal,
            "tilt_deviation_deg": _tilt_deviation_deg(voa),
            "supplementary_volume_weight_endpoint": _supplementary_volume_weight_endpoint(voa),
            "supplementary_svom_adjustment_factor": supplementary_svom_adjustment_factor,
            "tilt_adjustment_applied_to_volume": False,
            "physical_volume_model": "3d_lofted_elliptical_frustum",
            "method_version": VOM_METHOD_VERSION,
            "orientation_interpolation": "aligned_quaternion_slerp",
            "target_normal_method": "normalized_svd_plane_normal",
            "entry_frame_alignment_operation": entry_fit.get("frame_alignment_operation"),
            "entry_frame_alignment_rotation_deg": entry_fit.get("frame_alignment_rotation_deg"),
            "ellipse_polygon_samples": int(np.asarray(target_fit["ellipse_3d"]).shape[0]),
            "ellipse_polygon_max_relative_area_error": ELLIPSE_POLYGON_AREA_RTOL,
            "near_circle_axis_ratio_threshold": NEAR_CIRCLE_AXIS_RATIO,
            "vom_longitudinal_convergence": vom_convergence,
            "svom_longitudinal_convergence": svom_convergence,
            "entry_polygon_area_mm2": area_entry_poly,
            "target_polygon_area_mm2": area_target_poly,
            "entry_raw_polygon_area_mm2": float(entry_fit["raw_polygon_area_3d"]),
            "target_raw_polygon_area_mm2": float(target_fit["raw_polygon_area_3d"]),
            "entry_ellipse_area_mm2": area_entry_ellipse,
            "target_ellipse_area_mm2": area_target_ellipse,
            "entry_ellipse_fit_method": entry_fit.get("ellipse_fit_method"),
            "target_ellipse_fit_method": target_fit.get("ellipse_fit_method"),
            "entry_ellipse_fit_outlier_indices": entry_fit.get("ellipse_fit_outlier_indices", []),
            "target_ellipse_fit_outlier_indices": target_fit.get("ellipse_fit_outlier_indices", []),
            "entry_near_circular_regularization": entry_fit.get("ellipse_fit_near_circular_regularization", False),
            "target_near_circular_regularization": target_fit.get("ellipse_fit_near_circular_regularization", False),
            "entry_ellipse_a_mm": a_entry,
            "entry_ellipse_b_mm": b_entry,
            "target_ellipse_a_mm": a_target,
            "target_ellipse_b_mm": b_target,
            "cut_ellipse_a_mm": a_cut,
            "cut_ellipse_b_mm": b_cut,
            "supplementary_cut_ellipse_a_mm": a_cut_adjusted,
            "supplementary_cut_ellipse_b_mm": b_cut_adjusted,
            "stand_dist_mm": sd,
            "requested_stand_dist_mm": requested_sd,
            "stand_dist_clipped_to_corridor": bool(requested_sd > H),
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
