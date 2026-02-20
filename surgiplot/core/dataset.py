from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Iterable, Any, Tuple, List, Union
import numpy as np

PointLike = Union[np.ndarray, Iterable[float]]

@dataclass
class Dataset:
    """Container for 3D points with aliasing (e.g., point_1 <-> clinoid).

    - `points` stores coordinates by key.
    - `aliases` maps alias -> canonical key (or directly to a key in `points`).
    - `meta` stores provenance (source, units, etc.).

    Resolution rule:
      - if name exists in points: use it
      - else if name exists in aliases: resolve to target key in points
      - else KeyError
    """
    points: Dict[str, np.ndarray] = field(default_factory=dict)
    aliases: Dict[str, str] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def add_point(self, name: str, xyz: PointLike, overwrite: bool = False) -> None:
        name = str(name).strip()
        if (not overwrite) and (name in self.points):
            raise KeyError(f"Point '{name}' already exists.")
        arr = np.asarray(xyz, dtype=float).reshape(3,)
        self.points[name] = arr

    def add_alias(self, alias: str, target: str, overwrite: bool = False) -> None:
        alias = str(alias).strip()
        target = str(target).strip()
        if target not in self.points:
            raise KeyError(f"Cannot alias to missing point '{target}'.")
        if (not overwrite) and (alias in self.points or alias in self.aliases):
            raise KeyError(f"Alias '{alias}' already exists (as point or alias).")
        self.aliases[alias] = target

    def resolve_name(self, name: str) -> str:
        name = str(name).strip()
        if name in self.points:
            return name
        if name in self.aliases:
            target = self.aliases[name]
            if target not in self.points:
                raise KeyError(f"Alias '{name}' points to missing key '{target}'.")
            return target
        raise KeyError(f"Unknown point name/alias: '{name}'.")

    def get(self, name: str) -> np.ndarray:
        key = self.resolve_name(name)
        return self.points[key]

    def resolve_points(self, spec: Union[str, Iterable[str], np.ndarray, Iterable[PointLike]]) -> np.ndarray:
        """Resolve a point specification to an (N,3) ndarray.

        Accepted:
          - single string name -> (1,3)
          - list of string names -> (N,3)
          - ndarray shape (3,) -> (1,3)
          - ndarray shape (N,3) -> (N,3)
          - list of xyz iterables -> (N,3)
        """
        if isinstance(spec, str):
            return self.get(spec).reshape(1,3)
        arr = np.asarray(spec, dtype=float)
        if arr.ndim == 1 and arr.shape[0] == 3:
            return arr.reshape(1,3)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return arr
        # maybe list of names
        if isinstance(spec, Iterable):
            items = list(spec)
            if len(items) == 0:
                raise ValueError("Empty point specification.")
            if all(isinstance(x, str) for x in items):
                return np.vstack([self.get(x) for x in items])
            # list of xyz
            return np.vstack([np.asarray(x, dtype=float).reshape(3,) for x in items])
        raise TypeError("Unsupported point specification type.")

    def to_table_rows(self) -> List[dict]:
        """Rows for label editor: includes derived 'labels' column."""
        rows = []
        # canonical points first, then aliases shown as labels
        inv_alias = {}
        for a, tgt in self.aliases.items():
            inv_alias.setdefault(tgt, []).append(a)
        for name, xyz in self.points.items():
            labels = ", ".join(sorted(inv_alias.get(name, [])))
            rows.append({
                "name": name,
                "x": float(xyz[0]),
                "y": float(xyz[1]),
                "z": float(xyz[2]),
                "labels": labels,
            })
        return rows

    def apply_labels_from_table(self, rows: List[dict], overwrite: bool = True) -> None:
        """Update aliases from the `labels` column of label-editor rows.

        Each row's `labels` can contain comma-separated labels. Whitespace trimmed.
        """
        # Remove existing aliases that target any of the involved points (if overwrite)
        if overwrite:
            involved = {r["name"] for r in rows if r.get("name") in self.points}
            self.aliases = {a:t for a,t in self.aliases.items() if t not in involved}

        for r in rows:
            name = str(r.get("name", "")).strip()
            if name not in self.points:
                continue
            labels_str = str(r.get("labels", "") or "")
            labels = [s.strip() for s in labels_str.split(",") if s.strip()]
            for lab in labels:
                # avoid alias == canonical name collisions
                if lab == name:
                    continue
                self.add_alias(lab, name, overwrite=overwrite)


    def export_csv(self, path: str) -> None:
        """Export points + labels to CSV."""
        import csv
        rows = self.to_table_rows()
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["name","x","y","z","labels"])
            w.writeheader()
            for r in rows:
                w.writerow(r)

    def export_json(self, path: str) -> None:
        """Export points + aliases + meta to JSON."""
        import json
        payload = {
            "meta": self.meta,
            "points": {k: v.tolist() for k, v in self.points.items()},
            "aliases": self.aliases,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def edit_labels(self) -> "Dataset":
        """Open the Qt labeling grid and return self (mutated)."""
        from surgiplot.gui.label_editor import run_label_editor
        rows = self.to_table_rows()
        edited = run_label_editor(rows, source=self.meta.get("source", ""))
        if edited is None:
            return self  # cancelled
        # meta/source update may be returned
        rows2, source = edited
        if source:
            self.meta["source"] = source
        self.apply_labels_from_table(rows2, overwrite=True)
        return self
