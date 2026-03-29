from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict, Any
import numpy as np


_STRYKER_BLOCK_RE = re.compile(r"^\[(ANNOTATIONPOINT[^\]]+)\]$", re.IGNORECASE)


def is_stryker_annotation_text(text: str) -> bool:
    """
    Detect Stryker annotation export text.

    Typical markers:
      [ANNOTATIONPOINT_...]
      name=#138
      point=x,y,z
    """
    return (
        "point=" in text
        and "name=" in text
        and bool(_STRYKER_BLOCK_RE.search(text))
    )


def parse_stryker_points(text: str) -> List[Dict[str, Any]]:
    """
    Parse Stryker annotation point export (.txt or .dat).

    Expected repeated block structure like:
      [ANNOTATIONPOINT_...]
      COLOR=...
      desc=...
      name=#138
      point=x,y,z
      ...

    Returns a list of raw normalized point records:
      {
        "name": "#138",
        "xyz": np.ndarray(shape=(3,)),
        "color": "255,255,0" | None,
        "desc": "...",
        "raw_block": "ANNOTATIONPOINT_..."
      }
    """
    lines = text.splitlines()
    records: List[Dict[str, Any]] = []

    current_block = None
    current: Dict[str, Any] = {}

    def flush_current() -> None:
        nonlocal current, current_block, records

        if not current:
            return

        name = str(current.get("name", "")).strip()
        point = str(current.get("point", "")).strip()

        if not point:
            current = {}
            current_block = None
            return

        parts = [p.strip() for p in point.split(",")]
        if len(parts) != 3:
            current = {}
            current_block = None
            return

        try:
            xyz = np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float)
        except ValueError:
            current = {}
            current_block = None
            return

        records.append({
            "name": name,
            "xyz": xyz,
            "color": current.get("COLOR"),
            "desc": current.get("desc"),
            "raw_block": current_block,
        })

        current = {}
        current_block = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        block_match = _STRYKER_BLOCK_RE.match(line)
        if block_match:
            flush_current()
            current_block = block_match.group(1)
            continue

        if "=" in line:
            key, value = line.split("=", 1)
            current[key.strip()] = value.strip()

    flush_current()
    return records


def load_stryker_file(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore")
    return parse_stryker_points(text)