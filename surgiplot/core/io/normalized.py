from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from ..dataset import Dataset


_REQUIRED_COLUMNS = {"name", "x", "y", "z"}


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
        name = str(row[cols["name"]]).strip()
        if not name:
            name = f"{point_prefix}{idx + 1}"

        xyz = np.array([
            float(row[cols["x"]]),
            float(row[cols["y"]]),
            float(row[cols["z"]]),
        ], dtype=float)

        ds.add_point(name, xyz, overwrite=False)

    if alias_points:
        for i, name in enumerate(ds.points.keys(), start=1):
            alias = f"{point_prefix}{i}"
            if alias not in ds.points and alias not in ds.aliases:
                ds.add_alias(alias, name, overwrite=False)

    return ds