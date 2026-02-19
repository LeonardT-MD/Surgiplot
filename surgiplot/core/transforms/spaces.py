from __future__ import annotations
import numpy as np

def apply_affine(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Apply 4x4 affine to (N,3) points."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1,3)
    A = np.asarray(affine, dtype=float)
    if A.shape != (4,4):
        raise ValueError("affine must be 4x4")
    homog = np.c_[pts, np.ones((pts.shape[0],1))]
    out = (homog @ A.T)[:, :3]
    return out
