from __future__ import annotations
import numpy as np

def shoelace_area(poly2d: np.ndarray) -> float:
    """Area of a 2D polygon given ordered vertices (N,2)."""
    P = np.asarray(poly2d, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2 or P.shape[0] < 3:
        raise ValueError("poly2d must be (N,2) with N>=3")
    x = P[:,0]
    y = P[:,1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
