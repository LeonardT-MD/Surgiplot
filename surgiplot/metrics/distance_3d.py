from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict
import numpy as np


@dataclass
class Distance3DResult:
    distance_mm: float
    debug: Optional[Dict[str, Any]] = None

    def plot(self, ax) -> None:
        dbg = self.debug or {}
        A = dbg.get("A")
        B = dbg.get("B")
        if A is None or B is None:
            raise ValueError("Distance plot requires debug geometry.")

        A = np.asarray(A, dtype=float).reshape(3,)
        B = np.asarray(B, dtype=float).reshape(3,)
        ax.plot([A[0], B[0]], [A[1], B[1]], [A[2], B[2]], color="tab:cyan", linewidth=2)
        ax.scatter([A[0], B[0]], [A[1], B[1]], [A[2], B[2]], color="tab:cyan", s=40)
        ax.set_title(f"Distance={self.distance_mm:.2f} mm")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")


def _resolve_point(data, spec) -> np.ndarray:
    if data is not None:
        # string name/label supported
        if isinstance(spec, str):
            return data.get(spec).reshape(3,)
        # numeric
        return np.asarray(spec, dtype=float).reshape(3,)
    return np.asarray(spec, dtype=float).reshape(3,)


def DISTANCE_3D(
    data=None,
    A=None,
    B=None,
    space: str = "native",
    transforms=None,
    return_debug: bool = False,
) -> Distance3DResult:
    """Linear distance in 3D between points A and B (mm).

    Accepts:
      - names/labels (if data is Dataset)
      - numeric (x,y,z)
    """
    if A is None or B is None:
        raise ValueError("Provide A and B (names/labels or coordinates).")

    A_pt = _resolve_point(data, A)
    B_pt = _resolve_point(data, B)

    d = float(np.linalg.norm(B_pt - A_pt))

    dbg = None
    if return_debug:
        dbg = {"A": A_pt, "B": B_pt, "space": space}

    return Distance3DResult(distance_mm=d, debug=dbg)
