from __future__ import annotations

import json
import sys
from pathlib import Path

import vtk


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    module_dir = repo_root / "extensions" / "SurgiplotToolkit" / "SurgiplotToolkit"
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

    import SurgiplotToolkit as st  # type: ignore

    logic = st.SurgiplotToolkitLogic()
    entry, _ = logic.create_entry_markups_node(reuse_existing=False)
    pivot, _ = logic.create_pivot_markups_node(reuse_existing=False)
    target, _ = logic.create_target_markups_node(reuse_existing=False)

    entry_points = ((0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 1.0))
    target_points = ((0.0, 0.0, 3.0), (3.0, 0.0, 3.0), (0.0, 2.0, 3.0), (0.5, 0.5, 3.5))

    for point in entry_points:
        entry.AddControlPointWorld(vtk.vtkVector3d(*point))
    for point in target_points:
        target.AddControlPointWorld(vtk.vtkVector3d(*point))
    pivot.AddControlPointWorld(vtk.vtkVector3d(0.5, 0.5, 2.5))

    logic.assign_aoa_roles_by_order(entry)
    aoa = logic.compute_aoa_family(entry, pivot, sf_rescale_radius_mm=10.0)
    vom = logic.compute_vom_family(entry, target, stand_dist=10.0)

    summary = {
        "aoa_keys": sorted(aoa.keys()),
        "aoa_ordering": aoa.get("ordering_strategy"),
        "entry_count": aoa.get("entry_point_count"),
        "target_count": vom.get("target_point_count"),
        "vom_keys": sorted(vom.keys()),
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
