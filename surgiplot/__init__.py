from importlib.metadata import version as _pkg_version

from .api import (
    AOA,
    AOA_SF,
    AOE,
    AREA_3D,
    DISTANCE_3D,
    Dataset,
    SF,
    SVOM,
    VOA,
    VOM,
    VOM_VOA,
    VOLUME_3D,
    apply_labels,
    from_points,
    load,
    load_dataset,
    load_points_from_manual,
    rename_points,
)

try:
    __version__ = _pkg_version("surgiplot")
except Exception:
    __version__ = "0.1.0"

__all__ = [
    "AOA",
    "AOA_SF",
    "AOE",
    "AREA_3D",
    "DISTANCE_3D",
    "Dataset",
    "SF",
    "SVOM",
    "VOA",
    "VOM",
    "VOM_VOA",
    "VOLUME_3D",
    "apply_labels",
    "from_points",
    "load",
    "load_dataset",
    "load_points_from_manual",
    "rename_points",
    "__version__",
]
