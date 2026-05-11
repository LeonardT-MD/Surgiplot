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

    c = pts.mean(axis=0)
    X = pts - c

    # Use NumPy SVD instead of sklearn PCA so the geometry core remains lightweight
    # and importable in runtimes such as 3D Slicer.
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    # Right singular vectors correspond to principal axes; the smallest-variance
    # direction is the last axis and becomes the plane normal.
    basis = vh.T  # columns
    normal = basis[:, 2]
    # normalize
    normal = normal / (np.linalg.norm(normal) + 1e-12)
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
