from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Any, List, Union
import numpy as np

PointLike = Union[np.ndarray, Iterable[float]]


@dataclass
class Dataset:
    """Container for 3D points with aliasing (e.g., point_1 <-> clinoid).

    - `points` stores coordinates by key.
    - `aliases` maps alias -> canonical key in points.
      (We also store canonical -> preferred label as a convenience, but canonical always resolves to itself.)

    Resolution rule:
      - if name exists in points: use it
      - else if name exists in aliases: resolve to target key in points
      - else case-insensitive check in aliases
      - else KeyError
    """
    points: Dict[str, np.ndarray] = field(default_factory=dict)
    aliases: Dict[str, str] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    # ----------------------------
    # Core point/alias management
    # ----------------------------
    def add_point(self, name: str, xyz: PointLike, overwrite: bool = False) -> None:
        name = str(name).strip()
        if (not overwrite) and (name in self.points):
            raise KeyError(f"Point '{name}' already exists.")
        arr = np.asarray(xyz, dtype=float).reshape(3,)
        self.points[name] = arr

    def add_alias(self, alias: str, target: str, overwrite: bool = False) -> None:
        alias = str(alias).strip()
        target = str(target).strip()
        if not alias:
            raise ValueError("Alias cannot be empty.")
        if target not in self.points:
            raise KeyError(f"Cannot alias to missing point '{target}'.")

        # Do not allow alias to shadow an existing canonical point name (unless overwrite and it's an alias already)
        if alias in self.points:
            raise KeyError(f"Alias '{alias}' conflicts with an existing point name.")

        if (not overwrite) and (alias in self.aliases):
            raise KeyError(f"Alias '{alias}' already exists.")
        self.aliases[alias] = target

    def add_bidirectional_alias(self, target: str, label: str, overwrite: bool = True) -> None:
        """Store BOTH:
        - label -> target (for lookup by label)
        - target -> label (preferred display label; does not affect resolution because points wins)
        """
        target = str(target).strip()
        label = str(label).strip()
        if not label or not target:
            return
        if target not in self.points:
            raise KeyError(f"Cannot alias to missing point '{target}'.")

        # forward
        if label != target:
            self.add_alias(label, target, overwrite=overwrite)

        # reverse (display preference). This is safe because resolve_name checks points first.
        # Still: do NOT allow reverse key to be a point name other than the same target.
        self.aliases[target] = target  # ensure exists as canonical mapping (harmless)
        # store preferred label separately to avoid polluting alias logic:
        self.meta.setdefault("preferred_labels", {})
        self.meta["preferred_labels"][target] = label

    # ----------------------------
    # Name resolution
    # ----------------------------
    def resolve_name(self, name: str) -> str:
        name = str(name).strip()
        if not name:
            raise KeyError("Empty point name/alias.")

        # canonical wins
        if name in self.points:
            return name

        # direct alias
        if name in self.aliases:
            target = self.aliases[name]
            if target not in self.points:
                raise KeyError(f"Alias '{name}' points to missing key '{target}'.")
            return target

        # case-insensitive alias match
        low = name.lower()
        for a, tgt in self.aliases.items():
            if isinstance(a, str) and a.lower() == low:
                if tgt not in self.points:
                    raise KeyError(f"Alias '{a}' points to missing key '{tgt}'.")
                return tgt

        raise KeyError(f"Unknown point name/alias: '{name}'.")

    def resolve_names(self, names: Iterable[str]) -> List[str]:
        return [self.resolve_name(n) for n in names]

    def get(self, name: str) -> np.ndarray:
        key = self.resolve_name(name)
        return self.points[key]

    # ----------------------------
    # Point specification resolver
    # ----------------------------
    def resolve_points(
        self,
        spec: Union[str, Iterable[str], np.ndarray, Iterable[PointLike]]
    ) -> np.ndarray:
        """Resolve a point specification to an (N,3) ndarray.

        Accepted:
          - single string name -> (1,3)
          - list/tuple of string names -> (N,3)
          - ndarray shape (3,) -> (1,3)
          - ndarray shape (N,3) -> (N,3)
          - list of xyz iterables -> (N,3)
        """
        # single name
        if isinstance(spec, str):
            return self.get(spec).reshape(1, 3)

        # if it's an ndarray already
        if isinstance(spec, np.ndarray):
            arr = np.asarray(spec, dtype=float)
            if arr.ndim == 1 and arr.shape[0] == 3:
                return arr.reshape(1, 3)
            if arr.ndim == 2 and arr.shape[1] == 3:
                return arr
            raise ValueError(f"Unsupported ndarray shape: {arr.shape}")

        # iterable: could be list of names OR list of xyz
        if isinstance(spec, Iterable):
            items = list(spec)
            if len(items) == 0:
                raise ValueError("Empty point specification.")

            # list of names
            if all(isinstance(x, str) for x in items):
                return np.vstack([self.get(x) for x in items])

            # list of xyz
            return np.vstack([np.asarray(x, dtype=float).reshape(3,) for x in items])

        raise TypeError("Unsupported point specification type.")

    # ----------------------------
    # Table IO (labels)
    # ----------------------------
    def to_table_rows(self) -> List[dict]:
        """Rows for label editor: includes derived 'labels' column.

        We show labels that map to each canonical point.
        """
        rows = []
        inv_alias: Dict[str, List[str]] = {}
        for a, tgt in self.aliases.items():
            # skip canonical self-alias entries (if any)
            if a == tgt:
                continue
            # also skip entries that are just display-preferences stored in meta
            if a in self.points:
                continue
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

        Behavior:
          - remove prior labels targeting these points (if overwrite)
          - add new labels (bidirectional: label->point AND store preferred label for point)
        """
        involved = {str(r.get("name", "")).strip() for r in rows if str(r.get("name", "")).strip() in self.points}

        if overwrite and involved:
            self.aliases = {a: t for a, t in self.aliases.items() if t not in involved and a not in involved}
            # also clean preferred labels
            if "preferred_labels" in self.meta:
                self.meta["preferred_labels"] = {k: v for k, v in self.meta["preferred_labels"].items() if k not in involved}

        for r in rows:
            name = str(r.get("name", "")).strip()
            if name not in self.points:
                continue

            labels_str = str(r.get("labels", "") or "")
            labels = [s.strip() for s in labels_str.split(",") if s.strip()]

            for lab in labels:
                if lab == name:
                    continue
                self.add_bidirectional_alias(name, lab, overwrite=overwrite)

    # ----------------------------
    # Export
    # ----------------------------
    def export_csv(self, path: str) -> None:
        import csv
        rows = self.to_table_rows()
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["name", "x", "y", "z", "labels"])
            w.writeheader()
            for r in rows:
                w.writerow(r)

    def export_json(self, path: str) -> None:
        import json
        payload = {
            "meta": self.meta,
            "points": {k: v.tolist() for k, v in self.points.items()},
            "aliases": self.aliases,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    # ----------------------------
    # GUI helper (optional)
    # ----------------------------
    def edit_labels(self) -> "Dataset":
        """Open the Qt labeling grid and return self (mutated)."""
        from surgiplot.gui.label_editor import run_label_editor
        rows = self.to_table_rows()
        edited = run_label_editor(rows, source=self.meta.get("source", ""))
        if edited is None:
            return self  # cancelled
        rows2, source = edited
        if source:
            self.meta["source"] = source
        self.apply_labels_from_table(rows2, overwrite=True)
        return self
