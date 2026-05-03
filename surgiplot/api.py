from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

from surgiplot.core.dataset import Dataset
from surgiplot.core.io.loaders import load_dataset, load_points_from_manual
from surgiplot.metrics import (
    AOASFResult,
    AOA_SF,
    AOE,
    AREA_3D,
    DISTANCE_3D,
    VOLUME_3D,
    VOMVOAResult,
    VOM_VOA,
)


def load(
    path: str,
    *,
    source: str = "",
    alias_points: bool = True,
    point_prefix: str = "point_",
    meta: Optional[dict[str, Any]] = None,
    kind: str = "auto",
    navigation_format: str = "auto",
) -> Dataset:
    return load_dataset(
        path,
        source=source,
        alias_points=alias_points,
        point_prefix=point_prefix,
        meta=meta,
        kind=kind,
        navigation_format=navigation_format,
    )


def from_points(
    points: Iterable[Iterable[float]],
    *,
    names: Optional[Iterable[str]] = None,
    source: str = "",
    alias_points: bool = True,
    point_prefix: str = "point_",
) -> Dataset:
    return load_points_from_manual(
        points,
        names=names,
        source=source,
        alias_points=alias_points,
        point_prefix=point_prefix,
    )


def apply_labels(dataset: Dataset, mapping: Mapping[str, Sequence[str] | str]) -> Dataset:
    for name, labels in mapping.items():
        dataset.set_labels(name, [labels] if isinstance(labels, str) else list(labels), overwrite=True)
    return dataset


def rename_points(dataset: Dataset, mapping: Mapping[str, str]) -> Dataset:
    for old_name, new_name in mapping.items():
        dataset.rename_point(old_name, new_name, overwrite=False)
    return dataset


def AOA(*args, **kwargs) -> AOASFResult:
    return AOA_SF(*args, **kwargs)


def SF(*args, **kwargs) -> float:
    result = AOA_SF(*args, **kwargs)
    return result.sf_entry_area_mm2


def VOM(*args, **kwargs) -> VOMVOAResult:
    return VOM_VOA(*args, **kwargs)


def VOA(*args, **kwargs) -> float:
    result = VOM_VOA(*args, **kwargs)
    return result.voa_deg


def SVOM(*args, **kwargs) -> float:
    result = VOM_VOA(*args, **kwargs)
    return result.svom_mm3


__all__ = [
    "AOA",
    "AOA_SF",
    "AOASFResult",
    "AOE",
    "AREA_3D",
    "DISTANCE_3D",
    "Dataset",
    "SF",
    "SVOM",
    "VOA",
    "VOM",
    "VOM_VOA",
    "VOMVOAResult",
    "VOLUME_3D",
    "apply_labels",
    "from_points",
    "load",
    "load_dataset",
    "load_points_from_manual",
    "rename_points",
]
