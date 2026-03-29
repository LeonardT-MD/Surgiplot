from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np


def is_medtronic_json(text: str) -> bool:
    """
    Detect Medtronic export JSON.

    Expected main key:
      annotations
    """
    try:
        obj = json.loads(text)
    except Exception:
        return False

    return isinstance(obj, dict) and "annotations" in obj


def parse_medtronic_annotations(text: str, coordinate_system: str = "LPS") -> List[Dict[str, Any]]:
    """
    Parse Medtronic JSON export.

    We currently support LPS only by design.

    Expected structure:
      annotations -> LPS -> [{name, point:{x,y,z}}, ...]

    Returns a list of records:
      {
        "name": "top nostril",
        "xyz": np.ndarray(shape=(3,)),
        "coordinate_system": "LPS"
      }
    """
    coord = (coordinate_system or "LPS").upper()
    if coord != "LPS":
        raise ValueError("Only LPS coordinate system is currently supported for Medtronic import.")

    obj = json.loads(text)
    annotations = obj.get("annotations", {})
    items = annotations.get("LPS", []) or []

    out: List[Dict[str, Any]] = []
    for i, item in enumerate(items, start=1):
        name = str(item.get("name", "")).strip() or f"point_{i}"
        point = item.get("point", {}) or {}

        try:
            xyz = np.array([
                float(point["x"]),
                float(point["y"]),
                float(point["z"]),
            ], dtype=float)
        except Exception:
            continue

        out.append({
            "name": name,
            "xyz": xyz,
            "coordinate_system": "LPS",
        })

    return out


def load_medtronic_file(path: str | Path, coordinate_system: str = "LPS") -> List[Dict[str, Any]]:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore")
    return parse_medtronic_annotations(text, coordinate_system=coordinate_system)