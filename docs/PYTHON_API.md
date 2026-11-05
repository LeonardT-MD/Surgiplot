# Python API Reference

## 1. Import surface

The supported public interface is exported from `surgiplot`:

```python
import surgiplot as sp

print(sp.__version__)
```

The principal public objects are:

- data: `Dataset`, `load`, `load_dataset`, `from_points`, `load_points_from_manual`;
- labeling: `apply_labels`, `rename_points`;
- metrics: `DISTANCE_3D`, `AREA_3D`, `VOLUME_3D`, `AOE`, `AOA_SF`, `VOM_VOA`;
- convenience wrappers: `AOA`, `SF`, `VOM`, `VOA`, `SVOM`;
- study records: `AcquisitionProvenance`, `MetricRunRecord`, `build_metric_run_record`, `comparison_rows`, `repeatability_rows`.

Importing the core package does not initialize Qt, VTK, PyVista, or Matplotlib.

## 2. Constructing datasets

### 2.1 Direct coordinates

```python
import surgiplot as sp

dataset = sp.from_points(
    points=[
        [0.0, 10.0, 0.0],
        [0.0, -10.0, 0.0],
        [-10.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 0.0, 20.0],
    ],
    names=["cranial", "caudal", "medial", "lateral", "pivot"],
    source="database",
)
```

Coordinates must be finite triples. Point names must be nonempty and unambiguous under case-insensitive comparison.

### 2.2 File loading

```python
dataset = sp.load(
    "points.json",
    source="database",
    kind="auto",
    navigation_format="auto",
)
```

Parameters:

| Parameter | Accepted values | Meaning |
|---|---|---|
| `path` | existing file | source dataset |
| `source` | `navigation`, `photogrammetry`, `scanner`, `database`, empty | provenance category; no coordinate transform is implied |
| `kind` | `auto`, `navigation`, `database`, `generic` | loader family |
| `navigation_format` | `auto`, `stryker`, `medtronic`, `brainlab`, `generic` | navigation parser; Brainlab is reserved but not implemented |
| `alias_points` | Boolean | create sequential convenience aliases where applicable |
| `point_prefix` | string | canonical prefix, default `point_` |
| `meta` | mapping or `None` | additional metadata |

`load_dataset` is an equivalent explicit name. See [Data Formats](DATA_FORMATS.md) for schema behavior.

## 3. `Dataset`

### 3.1 Core state

```python
dataset.points   # dict[str, numpy.ndarray]
dataset.aliases  # dict[str, str], alias -> canonical name
dataset.meta     # dict[str, Any]
```

Coordinates are mutable NumPy arrays. For controlled workflows, treat the three dictionaries as analysis state: copy or export before changing them in place.

### 3.2 Point operations

```python
dataset.add_point("point_6", [1, 2, 3])
dataset.add_alias("target", "point_6")
dataset.set_labels("point_6", ["target", "tumor center"])
dataset.rename_point("point_6", "target_center")

canonical = dataset.resolve_name("TARGET")
coordinates = dataset.get("target")
matrix = dataset.resolve_points(["point_1", "target_center"])
```

`resolve_points` accepts a name, an iterable of names, a `(3,)` array, an `(N,3)` array, or an iterable of coordinate triples. Name ranges are inclusive:

```python
dataset.resolve_names(["point_1-4"])
dataset.resolve_points("point_4:1")
```

Range expansion preserves the requested direction. It does not sort a polygon geometrically.

### 3.3 Label helpers

```python
sp.apply_labels(dataset, {
    "point_1": ["cranial", "superior"],
    "point_2": "caudal",
})

sp.rename_points(dataset, {
    "point_3": "medial_limit",
})
```

The first label passed to `set_labels` or `apply_labels` is the preferred display label. Aliases are lookup keys; they do not duplicate coordinates.

### 3.4 Export

```python
dataset.export_json("normalized.json")
dataset.export_csv("normalized.csv")
dataset.export_csv("normalized.tsv")
```

JSON retains JSON-safe metadata and aliases. CSV/TSV retain points and comma-separated labels but not arbitrary nested metadata.

### 3.5 Provenance metadata

```python
dataset.set_provenance(
    acquisition_modality="navigation",
    operator="operator-01",
    session_id="session-02",
)
provenance = dataset.get_provenance()
```

Avoid placing direct patient identifiers in source repositories, examples, issue reports, or screenshots.

## 4. Metric calling conventions

All metrics support direct coordinates:

```python
result = sp.DISTANCE_3D(A=[0, 0, 0], B=[3, 4, 0])
assert result.distance_mm == 5.0
```

They also support names resolved through `data`:

```python
result = sp.DISTANCE_3D(data=dataset, A="entry", B="target")
```

Several signatures retain `space` and `transforms` compatibility fields. Version `0.3.0` records `space` in debug output where applicable but does not automatically apply `transforms`. Normalize coordinates before calling a metric.

## 5. Result objects

### 5.1 Distance

```python
result = sp.DISTANCE_3D(data=dataset, A="A", B="B", return_debug=True)
result.distance_mm
result.debug
```

Type: `Distance3DResult`.

### 5.2 Projected 3D area

```python
result = sp.AREA_3D(
    data=dataset,
    polygon=["boundary_1", "boundary_2", "boundary_3", "boundary_4"],
    return_debug=True,
)
result.area_mm2
```

Type: `Area3DResult`.

### 5.3 Sparse-point volume

```python
result = sp.VOLUME_3D(
    data=dataset,
    points=["surface_1-24"],
    return_debug=True,
)
result.volume_mm3
result.debug["reconstruction_method"]
```

Type: `Volume3DResult`. Retain the reconstruction method with the result.

### 5.4 Principal included AoE

```python
result = sp.AOE(data=dataset, A="limit_a", B="pivot", C="limit_c")
result.aoe_deg
```

Type: `AOEResult`. The output is the principal included angle and cannot exceed 180 degrees.

### 5.5 AoA and SF

```python
result = sp.AOA_SF(
    data=dataset,
    entry=["cranial", "caudal", "medial", "lateral"],
    target="pivot",
    sf_rescale_radius_mm=20.0,
    return_debug=True,
)

result.aoa_vertical_deg
result.aoa_horizontal_deg
result.sf_entry_area_mm2
result.sf_entry_area_rescaled_mm2
```

Type: `AOASFResult`. Exactly four distinct cardinal entry points are required in the documented role order.

### 5.6 VOM, sVOM, and VoA

```python
result = sp.VOM_VOA(
    data=dataset,
    entry=["entry_1-8"],
    target=["target_1-8"],
    stand_dist=10.0,
    return_debug=True,
)

result.vom_mm3
result.svom_mm3
result.voa_deg
```

Type: `VOMVOAResult`. Entry and target inputs are ordered polygon perimeters, not unordered point clouds.

With `return_debug=True`, the VOM payload includes `ellipse_frame_correspondence_policy`, `entry_frame_alignment_operation`, `entry_frame_alignment_swapped_axes`, and `entry_frame_alignment_rotation_deg`. The policy evaluates equivalent signed and axis-swapped ellipse frames and chooses the minimum-rotation correspondence before quaternion SLERP; semiaxes follow the selected swap.

## 6. Convenience wrappers

```python
combined_aoa = sp.AOA(data=dataset, entry=roles, target="pivot")
native_sf = sp.SF(data=dataset, entry=roles, target="pivot")

combined_vom = sp.VOM(data=dataset, entry=entry, target=target)
voa_degrees = sp.VOA(data=dataset, entry=entry, target=target)
standardized_volume = sp.SVOM(data=dataset, entry=entry, target=target)
```

`AOA` and `VOM` return their full composite dataclasses. `SF`, `VOA`, and `SVOM` return floats.

Dataset-bound equivalents are available:

```python
dataset.AOA(entry=roles, target="pivot")
dataset.SF(entry=roles, target="pivot")
dataset.VOM(entry=entry, target=target)
dataset.VOA(entry=entry, target=target)
dataset.SVOM(entry=entry, target=target)
dataset.AOE(A="A", B="B", C="C")
dataset.DISTANCE_3D(A="A", B="B")
dataset.AREA_3D(polygon=polygon)
dataset.VOLUME_3D(points=cloud)
```

## 7. Debug geometry and plotting

Set `return_debug=True` to request intermediate geometry. Result dataclasses expose a `plot(ax)` method:

```python
import matplotlib.pyplot as plt

result = sp.VOM_VOA(
    data=dataset,
    entry=entry,
    target=target,
    return_debug=True,
)

figure = plt.figure()
axis = figure.add_subplot(111, projection="3d")
result.plot(axis)
figure.savefig("vom_verification.png", dpi=200)
```

Plotting is a verification aid. Debug dictionary keys are intentionally richer and less stable than the scalar result fields; do not make a long-term database schema depend on every debug key.

## 8. Study records and repeatability

```python
record = sp.build_metric_run_record(
    subject_id="study-subject-001",
    anatomical_target="CPA",
    metric_family="VOM / sVOM / VoA",
    payload={
        "VOM_mm3": result.vom_mm3,
        "sVOM_mm3": result.svom_mm3,
        "VoA_deg": result.voa_deg,
    },
    acquisition_modality="navigation",
    operator="operator-01",
    session_id="session-01",
    source_context="post-dissection",
    reference_anatomy="target contour",
    source_file="normalized-session-01.json",
)

comparison = sp.comparison_rows([record])
repeatability = sp.repeatability_rows([record])
```

Repeatability groups records by subject ID, anatomical target, and metric family. It reports finite numeric values only, using population standard deviation and `abs(mean)` for coefficient of variation. A zero-mean group reports `CV_percent = 0.0` because relative CV is undefined near zero; interpret the absolute dispersion fields instead.

## 9. Exception contract

Typical exceptions are:

| Exception | Meaning |
|---|---|
| `FileNotFoundError` | input path is missing |
| `KeyError` | point/alias cannot resolve or conflicts with another name |
| `ValueError` | malformed data, invalid role assignment, non-finite coordinate, or degenerate geometry |
| `NotImplementedError` | explicitly requested Brainlab parser |

Do not replace errors with zero-valued measurements in an analysis pipeline. Record the failed case and correct the acquisition or protocol.

## 10. Minimal reproducible analysis pattern

```python
from dataclasses import asdict
import json
import surgiplot as sp

dataset = sp.load("normalized.json")
result = sp.AOA_SF(
    data=dataset,
    entry=["cranial", "caudal", "medial", "lateral"],
    target="pivot",
    sf_rescale_radius_mm=20.0,
    return_debug=False,
)

payload = {
    "surgiplot_version": sp.__version__,
    "coordinate_system": dataset.meta.get("coordinate_system"),
    "result": asdict(result),
}

with open("result.json", "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2)
```

For formal work, also record the Git commit, environment lock/export, operating system, input checksum, role mapping, and study protocol version.
