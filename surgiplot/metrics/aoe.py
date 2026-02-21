from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict
import numpy as np


@dataclass
class AOEResult:
    aoe_deg: float
    debug: Optional[Dict[str, Any]] = None


def _resolve_point(data, spec) -> np.ndarray:
    """
    Resolve a single 3D point to shape (3,).

    - If `data` (Dataset) is provided:
        - string -> resolves via Dataset.get() (supports aliases via resolve_name inside Dataset)
        - array-like -> numeric
    - If `data` is None:
        - array-like -> numeric
    """
    if data is not None:
        if isinstance(spec, str):
            # Dataset.get() should resolve aliases -> canonical
            return data.get(spec).reshape(3,)
        # numeric point
        arr = np.asarray(spec, dtype=float).reshape(3,)
        return arr

    return np.asarray(spec, dtype=float).reshape(3,)


def AOE(
    data=None,
    A=None,
    B=None,
    C=None,
    space: str = "native",
    transforms=None,
    return_debug: bool = False,
) -> AOEResult:
    """Angle of Exposure (AoE) from three 3D points A, B, C.

    Implementation
    --------------
    AoE is taken as the included angle at pivot B between vectors (A-B) and (C-B):
      AoE = arccos( (u·v) / (||u|| ||v||) )   in degrees

    Note: If you need the specific Porto et al. construction (AoE = 2*(a+b)),
    you can replace this function body while keeping the same signature.
    """
    if A is None or B is None or C is None:
        raise ValueError("Provide A, B, C (names or coordinates).")

    A_pt = _resolve_point(data, A)
    B_pt = _resolve_point(data, B)
    C_pt = _resolve_point(data, C)

    u = A_pt - B_pt
    v = C_pt - B_pt
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < 1e-12 or nv < 1e-12:
        raise ValueError("Degenerate vectors: check that A,B,C are distinct.")

    cosang = float(np.dot(u, v) / (nu * nv))
    cosang = max(-1.0, min(1.0, cosang))
    aoe = float(np.degrees(np.arccos(cosang)))

    dbg = None
    if return_debug:
        dbg = {"A": A_pt, "B": B_pt, "C": C_pt, "space": space, "cos": cosang}

    return AOEResult(aoe_deg=aoe, debug=dbg)
