from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np


_MEDTRONIC_COORDINATE_SYSTEMS = ("LPS", "RPI", "DCM")


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


def _select_medtronic_coordinate_system(obj: Dict[str, Any], coordinate_system: str = "auto") -> str:
    coord = (coordinate_system or "auto").upper()
    if coord != "AUTO":
        if coord not in _MEDTRONIC_COORDINATE_SYSTEMS:
            raise ValueError(
                f"Unsupported Medtronic coordinate system '{coordinate_system}'. "
                f"Expected one of: auto, {', '.join(_MEDTRONIC_COORDINATE_SYSTEMS)}."
            )
        return coord

    for key in ("LPS", "RPI", "DCM"):
        items = obj.get("annotations", {}).get(key, []) or []
        if items:
            return key

    for key in ("LPS", "RPI", "DCM"):
        items = obj.get("plans", {}).get(key, []) or []
        if items:
            return key

    return "LPS"


def _parse_xyz_point(point: Dict[str, Any]) -> np.ndarray:
    return np.array([
        float(point["x"]),
        float(point["y"]),
        float(point["z"]),
    ], dtype=float)


def parse_medtronic_annotations(
    text: str,
    coordinate_system: str = "auto",
    include_plans: bool = True,
) -> List[Dict[str, Any]]:
    """
    Parse Medtronic JSON export.

    Expected structure:
      annotations -> {LPS|RPI|DCM} -> [{name, point:{x,y,z}}, ...]
      plans -> {LPS|RPI|DCM} -> [{name, entry:{...}, target:{...}}, ...]

    Returns a list of records:
      {
        "name": "top nostril",
        "xyz": np.ndarray(shape=(3,)),
        "coordinate_system": "LPS"
      }
    """
    obj = json.loads(text)
    coord = _select_medtronic_coordinate_system(obj, coordinate_system=coordinate_system)

    annotations = obj.get("annotations", {}) or {}
    items = annotations.get(coord, []) or []

    out: List[Dict[str, Any]] = []
    for i, item in enumerate(items, start=1):
        name = str(item.get("name", "")).strip() or f"point_{i}"
        point = item.get("point", {}) or {}

        try:
            xyz = _parse_xyz_point(point)
        except Exception:
            continue

        out.append({
            "name": name,
            "xyz": xyz,
            "coordinate_system": coord,
        })

    if include_plans:
        plans = obj.get("plans", {}) or {}
        plan_items = plans.get(coord, []) or []
        for i, item in enumerate(plan_items, start=1):
            base_name = str(item.get("name", "")).strip() or f"plan_{i}"
            entry = item.get("entry", {}) or {}
            target = item.get("target", {}) or {}

            try:
                entry_xyz = _parse_xyz_point(entry)
                out.append({
                    "name": f"{base_name} entry",
                    "xyz": entry_xyz,
                    "coordinate_system": coord,
                    "kind": "plan_entry",
                })
            except Exception:
                pass

            try:
                target_xyz = _parse_xyz_point(target)
                out.append({
                    "name": f"{base_name} target",
                    "xyz": target_xyz,
                    "coordinate_system": coord,
                    "kind": "plan_target",
                })
            except Exception:
                pass

    return out


def load_medtronic_file(
    path: str | Path,
    coordinate_system: str = "auto",
    include_plans: bool = True,
) -> List[Dict[str, Any]]:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore")
    return parse_medtronic_annotations(
        text,
        coordinate_system=coordinate_system,
        include_plans=include_plans,
    )
