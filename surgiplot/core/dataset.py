from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Any, List, Union, Sequence
import re
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

    def labels_for_point(self, name: str) -> List[str]:
        canonical = self.resolve_name(name)
        labels: List[str] = []
        for alias, target in self.aliases.items():
            if alias == target or alias in self.points:
                continue
            if target == canonical:
                labels.append(alias)
        preferred = self.meta.get("preferred_labels", {}).get(canonical)
        if preferred and preferred not in labels:
            labels.insert(0, str(preferred))
        return sorted(dict.fromkeys(labels), key=str.lower)

    def preferred_label(self, name: str) -> str:
        canonical = self.resolve_name(name)
        preferred = self.meta.get("preferred_labels", {}).get(canonical)
        if isinstance(preferred, str) and preferred.strip():
            return preferred.strip()
        labels = self.labels_for_point(canonical)
        return labels[0] if labels else canonical

    def set_labels(
        self,
        name: str,
        labels: Sequence[str],
        overwrite: bool = True,
    ) -> None:
        canonical = self.resolve_name(name)
        normalized = [str(label).strip() for label in labels if str(label).strip()]
        unique_labels = [label for label in dict.fromkeys(normalized) if label != canonical]

        if overwrite:
            self.aliases = {
                alias: target
                for alias, target in self.aliases.items()
                if not (target == canonical and alias != target and alias not in self.points)
            }
            preferred_labels = self.meta.get("preferred_labels", {})
            if canonical in preferred_labels:
                preferred_labels.pop(canonical, None)

        for idx, label in enumerate(unique_labels):
            self.add_alias(label, canonical, overwrite=overwrite)
            if idx == 0:
                self.meta.setdefault("preferred_labels", {})
                self.meta["preferred_labels"][canonical] = label

    def rename_point(self, name: str, new_name: str, overwrite: bool = False) -> str:
        canonical = self.resolve_name(name)
        target = str(new_name).strip()
        if not target:
            raise ValueError("New point name cannot be empty.")
        if target == canonical:
            return canonical
        if target in self.points and not overwrite:
            raise KeyError(f"Point '{target}' already exists.")
        if target in self.aliases and self.aliases.get(target) != canonical:
            raise KeyError(f"Name '{target}' conflicts with an existing alias.")

        point = self.points.pop(canonical)
        self.points[target] = point

        remapped_aliases: Dict[str, str] = {}
        for alias, alias_target in self.aliases.items():
            if alias == canonical:
                continue
            remapped_aliases[alias] = target if alias_target == canonical else alias_target
        self.aliases = remapped_aliases

        preferred = self.meta.get("preferred_labels", {}).pop(canonical, None)
        if preferred:
            self.meta.setdefault("preferred_labels", {})
            self.meta["preferred_labels"][target] = preferred

        original_labels = self.meta.get("original_labels")
        if isinstance(original_labels, dict) and canonical in original_labels:
            original_labels[target] = original_labels.pop(canonical)
        return target

    # ----------------------------
    # Name resolution
    # ----------------------------
    @staticmethod
    def _expand_range_token(token: str) -> List[str]:
        text = str(token or "").strip()
        if not text:
            return []

        patterns = [
            r"^(?P<prefix>[A-Za-z_]+[_-]?)(?P<start>\d+)\s*[-:]\s*(?P<end>\d+)$",
            r"^(?P<prefix>[A-Za-z_]+[_-]?)(?P<start>\d+)\s*[-:]\s*(?P=prefix)?(?P<end>\d+)$",
            r"^(?P<start>\d+)\s*[-:]\s*(?P<end>\d+)$",
        ]
        for pattern in patterns:
            match = re.fullmatch(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            prefix = match.groupdict().get("prefix") or ""
            start = int(match.group("start"))
            end = int(match.group("end"))
            step = 1 if end >= start else -1
            values = range(start, end + step, step)
            if prefix:
                return [f"{prefix}{idx}" for idx in values]
            return [str(idx) for idx in values]
        return [text]

    def expand_name_spec(self, spec: Union[str, Iterable[str]]) -> List[str]:
        if isinstance(spec, str):
            tokens = [spec]
        else:
            tokens = list(spec)

        expanded: List[str] = []
        for token in tokens:
            if not isinstance(token, str):
                raise TypeError("Name specifications must be strings.")
            expanded.extend(self._expand_range_token(token))
        return [item for item in expanded if item]

    def resolve_name(self, name: str) -> str:
        name = str(name).strip()
        if not name:
            raise KeyError("Empty point name/alias.")

        # canonical wins
        if name in self.points:
            return name

        # case-insensitive canonical match
        low = name.lower()
        for key in self.points:
            if isinstance(key, str) and key.lower() == low:
                return key

        # common shorthand for canonical point names:
        #   1        -> point_1
        #   Point_1  -> point_1
        #   point-1  -> point_1
        point_idx = None
        if name.isdigit():
            point_idx = int(name)
        else:
            match = re.fullmatch(r"point[\s_-]*(\d+)", name, flags=re.IGNORECASE)
            if match:
                point_idx = int(match.group(1))

        if point_idx is not None:
            shorthand_key = f"point_{point_idx}"
            if shorthand_key in self.points:
                return shorthand_key

        # direct alias
        if name in self.aliases:
            target = self.aliases[name]
            if target not in self.points:
                raise KeyError(f"Alias '{name}' points to missing key '{target}'.")
            return target

        # case-insensitive alias match
        for a, tgt in self.aliases.items():
            if isinstance(a, str) and a.lower() == low:
                if tgt not in self.points:
                    raise KeyError(f"Alias '{a}' points to missing key '{tgt}'.")
                return tgt

        raise KeyError(f"Unknown point name/alias: '{name}'.")

    def resolve_names(self, names: Iterable[str]) -> List[str]:
        return [self.resolve_name(n) for n in self.expand_name_spec(names)]

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
            names = self.expand_name_spec(spec)
            if len(names) == 1:
                return self.get(names[0]).reshape(1, 3)
            return np.vstack([self.get(name) for name in names])

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
                expanded = self.expand_name_spec(items)
                return np.vstack([self.get(x) for x in expanded])

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

    # ----------------------------
    # Scientific metric shortcuts
    # ----------------------------
    def VOM(
        self,
        *,
        entry,
        target,
        stand_dist: float = 10.0,
        return_debug: bool = False,
        **kwargs,
    ):
        from surgiplot.metrics import VOM_VOA

        return VOM_VOA(
            data=self,
            entry=entry,
            target=target,
            stand_dist=stand_dist,
            return_debug=return_debug,
            **kwargs,
        )

    def VOA(self, *, entry, target, stand_dist: float = 10.0, **kwargs) -> float:
        return float(self.VOM(entry=entry, target=target, stand_dist=stand_dist, **kwargs).voa_deg)

    def SVOM(self, *, entry, target, stand_dist: float = 10.0, **kwargs) -> float:
        return float(self.VOM(entry=entry, target=target, stand_dist=stand_dist, **kwargs).svom_mm3)

    def AOA(
        self,
        *,
        entry,
        target,
        sf_rescale_radius_mm=None,
        return_debug: bool = False,
        **kwargs,
    ):
        from surgiplot.metrics import AOA_SF

        return AOA_SF(
            data=self,
            entry=entry,
            target=target,
            sf_rescale_radius_mm=sf_rescale_radius_mm,
            return_debug=return_debug,
            **kwargs,
        )

    def SF(self, *, entry, target, sf_rescale_radius_mm=None, **kwargs) -> float:
        return float(
            self.AOA(
                entry=entry,
                target=target,
                sf_rescale_radius_mm=sf_rescale_radius_mm,
                **kwargs,
            ).sf_entry_area_mm2
        )

    def AOE(self, *, A, B, C, return_debug: bool = False, **kwargs):
        from surgiplot.metrics import AOE

        return AOE(data=self, A=A, B=B, C=C, return_debug=return_debug, **kwargs)

    def DISTANCE_3D(self, *, A, B, return_debug: bool = False, **kwargs):
        from surgiplot.metrics import DISTANCE_3D

        return DISTANCE_3D(data=self, A=A, B=B, return_debug=return_debug, **kwargs)

    def AREA_3D(self, *, polygon, return_debug: bool = False, **kwargs):
        from surgiplot.metrics import AREA_3D

        return AREA_3D(data=self, polygon=polygon, return_debug=return_debug, **kwargs)

    def VOLUME_3D(self, *, points, return_debug: bool = False, **kwargs):
        from surgiplot.metrics import VOLUME_3D

        return VOLUME_3D(data=self, points=points, return_debug=return_debug, **kwargs)
