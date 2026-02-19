from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict
import numpy as np

@dataclass
class AOEResult:
    aoe_deg: float
    debug: Optional[Dict[str, Any]] = None

def _resolve_point(data, spec) -> np.ndarray:
    if data is not None and isinstance(spec, str):
        return data.get(spec).reshape(3,)
    arr = np.asarray(spec, dtype=float).reshape(3,)
    return arr

def AOE(A=None, B=None, C=None, data=None, space: str = "native", transforms=None, return_debug: bool = False) -> AOEResult:
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

    A = _resolve_point(data, A)
    B = _resolve_point(data, B)
    C = _resolve_point(data, C)

    u = A - B
    v = C - B
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < 1e-12 or nv < 1e-12:
        raise ValueError("Degenerate vectors: check that A,B,C are distinct.")

    cosang = float(np.dot(u, v) / (nu*nv))
    cosang = max(-1.0, min(1.0, cosang))
    aoe = float(np.degrees(np.arccos(cosang)))

    dbg = None
    if return_debug:
        dbg = {"A": A, "B": B, "C": C, "space": space, "cos": cosang}
    return AOEResult(aoe_deg=aoe, debug=dbg)
