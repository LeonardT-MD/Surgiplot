from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

from surgiplot.core.dataset import Dataset
from surgiplot.core.io.loaders import load_dataset, load_points_from_manual
from surgiplot.metrics import (
    AOA_SF,
    AOE,
    AREA_3D,
    DISTANCE_3D,
    VOLUME_3D,
    VOM_VOA,
)


@dataclass(frozen=True)
class AOAResult:
    vertical_deg: float
    horizontal_deg: float
    source: Any


@dataclass(frozen=True)
class SFResult:
    area_mm2: float
    standardized_area_mm2: float
    source: Any


@dataclass(frozen=True)
class VOMResult:
    volume_mm3: float
    source: Any


@dataclass(frozen=True)
class VOAResult:
    angle_deg: float
    source: Any


@dataclass(frozen=True)
class SVOMResult:
    volume_mm3: float
    source: Any


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


def AOA(*args, **kwargs) -> AOAResult:
    result = AOA_SF(*args, **kwargs)
    return AOAResult(
        vertical_deg=result.aoa_vertical_deg,
        horizontal_deg=result.aoa_horizontal_deg,
        source=result,
    )


def SF(*args, **kwargs) -> SFResult:
    result = AOA_SF(*args, **kwargs)
    return SFResult(
        area_mm2=result.sf_entry_area_mm2,
        standardized_area_mm2=result.sf_entry_area_rescaled_mm2,
        source=result,
    )


def VOM(*args, **kwargs) -> VOMResult:
    result = VOM_VOA(*args, **kwargs)
    return VOMResult(volume_mm3=result.vom_mm3, source=result)


def VOA(*args, **kwargs) -> VOAResult:
    result = VOM_VOA(*args, **kwargs)
    return VOAResult(angle_deg=result.voa_deg, source=result)


def SVOM(*args, **kwargs) -> SVOMResult:
    result = VOM_VOA(*args, **kwargs)
    return SVOMResult(volume_mm3=result.svom_mm3, source=result)


__all__ = [
    "AOA",
    "AOAResult",
    "AOA_SF",
    "AOE",
    "AREA_3D",
    "DISTANCE_3D",
    "Dataset",
    "SF",
    "SFResult",
    "SVOM",
    "SVOMResult",
    "VOA",
    "VOAResult",
    "VOM",
    "VOMResult",
    "VOM_VOA",
    "VOLUME_3D",
    "apply_labels",
    "from_points",
    "load",
    "load_dataset",
    "load_points_from_manual",
    "rename_points",
]
