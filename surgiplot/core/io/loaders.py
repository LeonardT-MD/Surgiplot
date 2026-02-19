from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Iterable
import re
import numpy as np

from ..dataset import Dataset

_SOURCE_ALLOWED = {"navigation", "photogrammetry", "scanner", ""}

def _infer_delimiter(line: str) -> str:
    # prefer comma, then tab, then whitespace
    if "," in line:
        return ","
    if "\t" in line:
        return "\t"
    return None  # whitespace

def _parse_generic_points(text: str) -> List[Tuple[str, np.ndarray]]:
    """Parse a simple point file.

    Accepts lines like:
      name, x, y, z
      name\tx\ty\tz
      name x y z

    Ignores empty lines and comment lines starting with # or //.
    """
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and (not ln.startswith("#")) and (not ln.startswith("//"))]
    if not lines:
        return []

    delim = _infer_delimiter(lines[0])
    out = []
    for ln in lines:
        parts = [p.strip() for p in (ln.split(delim) if delim else re.split(r"\s+", ln)) if p.strip()]
        if len(parts) < 4:
            continue
        name = parts[0]
        try:
            xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)
        except ValueError:
            continue
        out.append((name, xyz))
    return out

def load_dataset(path: str | Path,
                 source: str = "",
                 alias_points: bool = True,
                 point_prefix: str = "point_",
                 meta: Optional[Dict[str, Any]] = None) -> Dataset:
    """Load a dataset from an annotation file.

    Parameters
    ----------
    path:
      File path (txt/csv/tsv/space-delimited supported by generic parser)
    source:
      One of: navigation | photogrammetry | scanner (stored in ds.meta)
    alias_points:
      If True, create stable aliases point_1..point_N (in file order)
      while preserving original point names.
    """
    source = (source or "").strip().lower()
    if source not in _SOURCE_ALLOWED:
        raise ValueError(f"Invalid source '{source}'. Allowed: navigation, photogrammetry, scanner.")
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore")
    pairs = _parse_generic_points(text)

    ds = Dataset(meta=dict(meta or {}))
    if source:
        ds.meta["source"] = source
    ds.meta["path"] = str(p)

    for name, xyz in pairs:
        ds.add_point(name, xyz, overwrite=False)

    # Create point_1.. aliases
    if alias_points:
        for i, (name, _xyz) in enumerate(pairs, start=1):
            alias = f"{point_prefix}{i}"
            # allow both: point_i -> original name
            if alias not in ds.points and alias not in ds.aliases:
                ds.add_alias(alias, name, overwrite=False)

    return ds

def load_points_from_manual(points: Iterable[Iterable[float]],
                            names: Optional[Iterable[str]] = None,
                            source: str = "",
                            alias_points: bool = True,
                            point_prefix: str = "point_") -> Dataset:
    """Create Dataset from manually provided coordinates.

    points: iterable of (x,y,z)
    names: optional iterable of names; if omitted, uses point_1.. as canonical names.
    """
    pts = [np.asarray(p, dtype=float).reshape(3,) for p in points]
    if names is None:
        names = [f"{point_prefix}{i}" for i in range(1, len(pts)+1)]
        alias_points = False  # already canonical
    names = [str(n).strip() for n in names]
    if len(names) != len(pts):
        raise ValueError("names and points must have same length.")

    ds = Dataset(meta={})
    source = (source or "").strip().lower()
    if source:
        ds.meta["source"] = source

    for n, xyz in zip(names, pts):
        ds.add_point(n, xyz)

    if alias_points:
        for i, n in enumerate(names, start=1):
            alias = f"{point_prefix}{i}"
            if alias not in ds.points and alias not in ds.aliases:
                ds.add_alias(alias, n, overwrite=False)
    return ds
