from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from ..dataset import Dataset


_REQUIRED_COLUMNS = {"name", "x", "y", "z"}


def load_normalized_json(path: str | Path) -> Dataset:
    """Load a dataset previously written by :meth:`Dataset.export_json`."""
    p = Path(path)
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid Surgiplot dataset JSON: {p}") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("points"), dict):
        raise ValueError("Normalized JSON must contain a 'points' object.")

    raw_meta = payload.get("meta", {})
    if raw_meta is not None and not isinstance(raw_meta, dict):
        raise ValueError("Normalized JSON field 'meta' must be an object.")
    ds = Dataset(meta=dict(raw_meta or {}))
    ds.meta.setdefault("path", str(p))

    for raw_name, raw_xyz in payload["points"].items():
        ds.add_point(str(raw_name), raw_xyz, overwrite=False)
    if not ds.points:
        raise ValueError("Normalized JSON dataset contains no points.")

    raw_aliases = payload.get("aliases", {})
    if raw_aliases is not None and not isinstance(raw_aliases, dict):
        raise ValueError("Normalized JSON field 'aliases' must be an object.")
    for raw_alias, raw_target in (raw_aliases or {}).items():
        alias = str(raw_alias).strip()
        target = str(raw_target).strip()
        if alias == target and target in ds.points:
            ds.aliases[target] = target
            continue
        ds.add_alias(alias, target, overwrite=False)
    return ds


def load_formatted_database(
    path: str | Path,
    source: str = "",
    meta: Optional[Dict[str, Any]] = None,
    alias_points: bool = True,
    point_prefix: str = "point_",
) -> Dataset:
    """
    Load an already-normalized point database.

    Required columns:
      name, x, y, z

    Supported file types:
      .xlsx, .xls, .csv, .tsv

    Optional future columns:
      label, role, group, coordinate_system, vendor, note, color
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {p}")

    if p.suffix.lower() == ".json":
        return load_normalized_json(p)

    if p.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(p)
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    elif p.suffix.lower() == ".tsv":
        df = pd.read_csv(p, sep="\t")
    else:
        raise ValueError(f"Unsupported formatted database file type: {p.suffix}")

    cols = {c.lower().strip(): c for c in df.columns}
    missing = [c for c in _REQUIRED_COLUMNS if c not in cols]
    if missing:
        raise ValueError(f"Formatted database missing required columns: {missing}")

    ds = Dataset(meta=dict(meta or {}))
    ds.meta["path"] = str(p)
    ds.meta["source"] = source or "database"
    ds.meta["vendor"] = "formatted_database"
    ds.meta["coordinate_system"] = "LPS"

    for idx, row in df.iterrows():
        raw_name = row[cols["name"]]
        name = "" if pd.isna(raw_name) else str(raw_name).strip()
        if not name:
            name = f"{point_prefix}{idx + 1}"

        try:
            xyz = np.array([
                float(row[cols["x"]]),
                float(row[cols["y"]]),
                float(row[cols["z"]]),
            ], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid coordinates in formatted dataset row {idx + 2}.") from exc

        ds.add_point(name, xyz, overwrite=False)

        if "labels" in cols and not pd.isna(row[cols["labels"]]):
            labels = [
                label.strip()
                for label in str(row[cols["labels"]]).split(",")
                if label.strip()
            ]
            if labels:
                ds.set_labels(name, labels, overwrite=True)

    if not ds.points:
        raise ValueError("Formatted database contains no point rows.")

    if alias_points:
        for i, name in enumerate(ds.points.keys(), start=1):
            alias = f"{point_prefix}{i}"
            if alias not in ds.points and alias not in ds.aliases:
                ds.add_alias(alias, name, overwrite=False)

    return ds
