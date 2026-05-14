from importlib.metadata import version as _pkg_version

from .api import (
    AOA,
    AOA_SF,
    AOE,
    AREA_3D,
    AcquisitionProvenance,
    DISTANCE_3D,
    Dataset,
    MetricRunRecord,
    SF,
    SVOM,
    VOA,
    VOM,
    VOM_VOA,
    VOLUME_3D,
    apply_labels,
    build_metric_run_record,
    comparison_rows,
    from_points,
    load,
    load_dataset,
    load_points_from_manual,
    rename_points,
    repeatability_rows,
)

try:
    __version__ = _pkg_version("surgiplot")
except Exception:
    __version__ = "0.1.0"

__all__ = [
    "AOA",
    "AOA_SF",
    "AOE",
    "AcquisitionProvenance",
    "AREA_3D",
    "DISTANCE_3D",
    "Dataset",
    "MetricRunRecord",
    "SF",
    "SVOM",
    "VOA",
    "VOM",
    "VOM_VOA",
    "VOLUME_3D",
    "apply_labels",
    "build_metric_run_record",
    "comparison_rows",
    "from_points",
    "load",
    "load_dataset",
    "load_points_from_manual",
    "rename_points",
    "repeatability_rows",
    "__version__",
]
