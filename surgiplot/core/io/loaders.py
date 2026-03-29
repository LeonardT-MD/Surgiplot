from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Iterable
import re
import numpy as np

from ..dataset import Dataset
from .normalized import load_formatted_database
from .parsers import (
    is_stryker_annotation_text,
    parse_stryker_points,
    is_medtronic_json,
    parse_medtronic_annotations,
)


_SOURCE_ALLOWED = {"navigation", "photogrammetry", "scanner", "database", ""}
_KIND_ALLOWED = {"auto", "navigation", "database", "generic"}
_NAVIGATION_FORMATS = {"auto", "stryker", "medtronic", "brainlab", "generic"}


def _infer_delimiter(line: str) -> str | None:
    if "," in line:
        return ","
    if "\t" in line:
        return "\t"
    return None


def _parse_generic_points(text: str) -> List[Tuple[str, np.ndarray]]:
    """
    Parse a simple point file.

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


def _records_to_dataset(
    records: List[Dict[str, Any]],
    source: str,
    path: str,
    meta: Optional[Dict[str, Any]] = None,
    alias_points: bool = True,
    point_prefix: str = "point_",
) -> Dataset:
    """
    Generic record -> Dataset conversion.

    Used for non-Stryker vendor imports where point_1, point_2, ...
    is the intended canonical naming.
    """
    ds = Dataset(meta=dict(meta or {}))
    if source:
        ds.meta["source"] = source
    ds.meta["path"] = str(path)
    ds.meta.setdefault("coordinate_system", "LPS")

    original_labels: Dict[str, str] = {}

    for i, rec in enumerate(records, start=1):
        canonical = f"{point_prefix}{i}"
        original = str(rec["name"]).strip()
        xyz = np.asarray(rec["xyz"], dtype=float).reshape(3,)

        ds.add_point(canonical, xyz, overwrite=False)

        if original and original not in ds.aliases:
            ds.add_alias(original, canonical, overwrite=False)

        original_labels[canonical] = original

    if alias_points:
        for i in range(1, len(records) + 1):
            alias = f"{point_prefix}{i}"
            canonical = f"{point_prefix}{i}"
            if alias not in ds.points and alias not in ds.aliases:
                ds.add_alias(alias, canonical, overwrite=False)

    ds.meta["original_labels"] = original_labels
    return ds


def _stryker_records_to_dataset(
    records: List[Dict[str, Any]],
    path: str,
    meta: Optional[Dict[str, Any]] = None,
    point_prefix: str = "point_",
) -> Dataset:
    """
    Stryker-specific conversion.

    Decision:
      #138 -> point_138
      #1   -> point_1

    So we preserve navigation numbering instead of reindexing sequentially.
    """
    ds = Dataset(meta=dict(meta or {}))
    ds.meta["source"] = "navigation"
    ds.meta["vendor"] = "stryker"
    ds.meta["coordinate_system"] = "LPS"
    ds.meta["path"] = str(path)

    original_labels: Dict[str, str] = {}

    for i, rec in enumerate(records, start=1):
        original = str(rec.get("name", "")).strip()
        xyz = np.asarray(rec["xyz"], dtype=float).reshape(3,)

        match = re.search(r"\d+", original)
        if match:
            idx = int(match.group())
            canonical = f"{point_prefix}{idx}"
        else:
            canonical = f"{point_prefix}{i}"

        if canonical in ds.points:
            continue

        ds.add_point(canonical, xyz, overwrite=False)

        if original and original not in ds.aliases:
            ds.add_alias(original, canonical, overwrite=False)

        original_labels[canonical] = original

    ds.meta["original_labels"] = original_labels
    return ds


def detect_navigation_format(path: str | Path) -> str:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore")

    if is_medtronic_json(text):
        return "medtronic"
    if is_stryker_annotation_text(text):
        return "stryker"
    return "generic"


def load_navigation_file(
    path: str | Path,
    navigation_format: str = "auto",
    alias_points: bool = True,
    point_prefix: str = "point_",
    meta: Optional[Dict[str, Any]] = None,
) -> Dataset:
    """
    Load a vendor-native neuronavigation file.

    Current support:
      - Stryker (.txt/.dat)
      - Medtronic (.json, LPS only)
      - generic fallback
    """
    fmt = (navigation_format or "auto").strip().lower()
    if fmt not in _NAVIGATION_FORMATS:
        raise ValueError(f"Invalid navigation_format '{navigation_format}'.")

    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore")

    if fmt == "auto":
        fmt = detect_navigation_format(p)

    if fmt == "stryker":
        records = parse_stryker_points(text)
        return _stryker_records_to_dataset(
            records,
            path=str(p),
            meta={
                **(meta or {}),
                "raw_format": p.suffix.lower(),
            },
            point_prefix=point_prefix,
        )

    if fmt == "medtronic":
        records = parse_medtronic_annotations(text, coordinate_system="LPS")
        ds = _records_to_dataset(
            records,
            source="navigation",
            path=str(p),
            meta={
                **(meta or {}),
                "vendor": "medtronic",
                "coordinate_system": "LPS",
                "raw_format": "json",
            },
            alias_points=alias_points,
            point_prefix=point_prefix,
        )
        return ds

    if fmt == "brainlab":
        raise NotImplementedError("Brainlab import not implemented yet.")

    pairs = _parse_generic_points(text)
    records = [{"name": name, "xyz": xyz} for name, xyz in pairs]
    ds = _records_to_dataset(
        records,
        source="navigation",
        path=str(p),
        meta={
            **(meta or {}),
            "vendor": "unknown",
            "coordinate_system": "LPS",
            "raw_format": p.suffix.lower(),
        },
        alias_points=alias_points,
        point_prefix=point_prefix,
    )
    return ds


def load_dataset(
    path: str | Path,
    source: str = "",
    alias_points: bool = True,
    point_prefix: str = "point_",
    meta: Optional[Dict[str, Any]] = None,
    kind: str = "auto",
    navigation_format: str = "auto",
) -> Dataset:
    """
    Unified entrypoint.

    Parameters
    ----------
    path
      Input file path.
    source
      One of: navigation | photogrammetry | scanner | database
    kind
      One of:
        - auto
        - navigation
        - database
        - generic
    navigation_format
      One of:
        - auto
        - stryker
        - medtronic
        - brainlab
        - generic
    """
    source = (source or "").strip().lower()
    if source not in _SOURCE_ALLOWED:
        raise ValueError(
            f"Invalid source '{source}'. Allowed: navigation, photogrammetry, scanner, database."
        )

    kind = (kind or "auto").strip().lower()
    if kind not in _KIND_ALLOWED:
        raise ValueError(f"Invalid kind '{kind}'. Allowed: {sorted(_KIND_ALLOWED)}")

    p = Path(path)

    if kind == "database":
        return load_formatted_database(
            p,
            source=source or "database",
            meta=meta,
            alias_points=alias_points,
            point_prefix=point_prefix,
        )

    if kind == "navigation":
        return load_navigation_file(
            p,
            navigation_format=navigation_format,
            alias_points=alias_points,
            point_prefix=point_prefix,
            meta=meta,
        )

    if kind == "generic":
        text = p.read_text(encoding="utf-8", errors="ignore")
        pairs = _parse_generic_points(text)
        records = [{"name": name, "xyz": xyz} for name, xyz in pairs]
        return _records_to_dataset(
            records,
            source=source,
            path=str(p),
            meta={
                **(meta or {}),
                "coordinate_system": "LPS",
            },
            alias_points=alias_points,
            point_prefix=point_prefix,
        )

    try:
        detected = detect_navigation_format(p)
        if detected in {"stryker", "medtronic"}:
            return load_navigation_file(
                p,
                navigation_format=detected,
                alias_points=alias_points,
                point_prefix=point_prefix,
                meta=meta,
            )
    except Exception:
        pass

    if p.suffix.lower() in {".xlsx", ".xls", ".csv", ".tsv"}:
        try:
            return load_formatted_database(
                p,
                source=source or "database",
                meta=meta,
                alias_points=alias_points,
                point_prefix=point_prefix,
            )
        except Exception:
            pass

    text = p.read_text(encoding="utf-8", errors="ignore")
    pairs = _parse_generic_points(text)
    records = [{"name": name, "xyz": xyz} for name, xyz in pairs]
    return _records_to_dataset(
        records,
        source=source,
        path=str(p),
        meta={
            **(meta or {}),
            "coordinate_system": "LPS",
        },
        alias_points=alias_points,
        point_prefix=point_prefix,
    )


def load_points_from_manual(
    points: Iterable[Iterable[float]],
    names: Optional[Iterable[str]] = None,
    source: str = "",
    alias_points: bool = True,
    point_prefix: str = "point_",
) -> Dataset:
    """
    Create Dataset from manually provided coordinates.

    Parameters
    ----------
    points
      Iterable of (x, y, z)
    names
      Optional iterable of names. If omitted, uses point_1..point_N as canonical names.
    """
    pts = [np.asarray(p, dtype=float).reshape(3,) for p in points]

    if names is None:
        names = [f"{point_prefix}{i}" for i in range(1, len(pts) + 1)]
        alias_points = False

    names = [str(n).strip() for n in names]
    if len(names) != len(pts):
        raise ValueError("names and points must have same length.")

    ds = Dataset(meta={})
    source = (source or "").strip().lower()
    if source:
        ds.meta["source"] = source
    ds.meta["coordinate_system"] = "LPS"

    for n, xyz in zip(names, pts):
        ds.add_point(n, xyz, overwrite=False)

    if alias_points:
        for i, n in enumerate(names, start=1):
            alias = f"{point_prefix}{i}"
            if alias not in ds.points and alias not in ds.aliases:
                ds.add_alias(alias, n, overwrite=False)

    return ds