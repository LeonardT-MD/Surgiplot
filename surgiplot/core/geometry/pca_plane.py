from __future__ import annotations

import numpy as np


def best_fit_plane_pca(points: np.ndarray):
    """Return centroid and normal vector of best-fit plane via PCA.

    points: (N,3)
    Returns: centroid (3,), normal (3,), basis (3x3) where basis[:,0:2] span plane, basis[:,2]=normal
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 3:
        raise ValueError("points must be (N,3) with N>=3")
    if not np.all(np.isfinite(pts)):
        raise ValueError("points must contain only finite coordinates")
    if np.unique(pts, axis=0).shape[0] < 3:
        raise ValueError("at least three distinct points are required")

    c = pts.mean(axis=0)
    X = pts - c
    if np.linalg.matrix_rank(X) < 2:
        raise ValueError("points are collinear; a stable plane cannot be estimated")

    # Use NumPy SVD instead of sklearn PCA so the geometry core remains lightweight
    # and importable in runtimes such as 3D Slicer.
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    # Right singular vectors correspond to principal axes; the smallest-variance
    # direction is the last axis and becomes the plane normal.
    raw_basis = vh.T  # columns
    u = raw_basis[:, 0]
    u = u / np.linalg.norm(u)
    v = raw_basis[:, 1] - u * np.dot(raw_basis[:, 1], u)
    v = v / np.linalg.norm(v)
    # SVD vector signs are arbitrary. Reconstruct the third axis from the
    # first two so every returned plane frame is an orthonormal, right-handed
    # rotation matrix suitable for interpolation.
    normal = np.cross(u, v)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= np.finfo(float).eps:
        raise ValueError("best-fit plane normal is numerically degenerate")
    normal = normal / normal_norm
    basis = np.column_stack([u, v, normal])
    return c, normal, basis


def project_to_plane_2d(points: np.ndarray):
    """Project 3D points to 2D in best-fit PCA plane and return (pts2, centroid, basis)."""
    c, n, basis = best_fit_plane_pca(points)
    X = np.asarray(points, dtype=float) - c
    # take first two basis vectors (in-plane)
    u = basis[:, 0]
    v = basis[:, 1]
    pts2 = np.stack([X @ u, X @ v], axis=1)
    return pts2, c, basis
