# Surgiplot Extended Guide

## Scientific, Computational, and Practical Guide for Coordinate-Based Neurosurgical Analysis

This document is a methods-oriented companion to the main repository [README](/Users/leonardo/Projects/Surgiplot/README.md). It is intended for:

- internal lab documentation
- reproducible research workflows
- manuscript methods sections
- supplementary technical notes
- onboarding collaborators who need both the conceptual and practical picture

The guide explains:

- the conceptual model of Surgiplot
- the mathematical approach behind each metric family
- the dataset abstraction and point-label system
- Python library usage
- CLI workflows for scripting and batch studies
- GUI workflows for tabular data and rendered 3D scene analysis
- recommended research patterns for reproducibility

## 1. Conceptual Overview

Surgiplot is a local coordinate-based analysis environment for neurosurgical anatomical and skull-base research. Its central assumption is that once landmarks are represented as explicit 3D coordinates in a shared frame, operative geometry can be studied using transparent mathematical constructs rather than opaque application state.

In practice, Surgiplot separates a study workflow into three layers:

1. `data acquisition`
   Landmarks are obtained from navigation exports, normalized tables, generic point files, or direct picking on imported 3D scenes.

2. `dataset normalization`
   All points are converted into a common `Dataset` representation with canonical point identifiers and optional semantic aliases.

3. `metric computation`
   Geometric metrics are computed from those explicit coordinates using inspectable mathematical procedures.

This design is valuable in research because it makes the measurement pipeline:

- reproducible
- scriptable
- inspectable
- vendor-agnostic at the coordinate level

## 2. Dataset Model

The canonical container in Surgiplot is the `Dataset` object, defined in [dataset.py](/Users/leonardo/Projects/Surgiplot/surgiplot/core/dataset.py).

Each dataset stores:

- `points`
  Canonical 3D coordinates keyed by names such as `point_1`, `point_2`, etc.

- `aliases`
  Human-readable labels such as `cranial`, `caudal`, `pivot`, `medial`, `lateral`, or source-native names imported from navigation systems.

- `meta`
  Metadata such as source type, coordinate system, file provenance, calibration, imported scene payload, and preferred display labels.

This abstraction allows the same metric to be called using:

- canonical identifiers
- source-native labels
- semantic aliases

For example, if `point_4` has been labeled as `pivot`, both of the following are valid:

```python
ds.DISTANCE_3D(A="point_1", B="point_4")
ds.DISTANCE_3D(A="point_1", B="pivot")
```

### 2.1 Canonical names and aliases

Canonical point names are stable internal keys. Aliases exist to make scripts and analyses anatomically legible. A typical workflow is:

1. import a navigation or scene-derived dataset
2. inspect the canonical point list
3. assign anatomical labels
4. reuse those labels in repeated metric calls

### 2.2 Range-aware point specifications

For multi-point metrics, Surgiplot supports compact range specifications. These are particularly useful for area, volume, and corridor-style metrics.

Accepted examples include:

- `point_1-9`
- `point_1:9`
- `1-9`

These range expressions are expanded consistently in both CLI and Python wherever a multi-point specification is expected.

## 3. Supported Inputs

Surgiplot supports several input classes through the loaders in [loaders.py](/Users/leonardo/Projects/Surgiplot/surgiplot/core/io/loaders.py).

### 3.1 Navigation exports

Current support includes:

- Stryker annotation-style exports
- Medtronic JSON annotation exports
- generic navigation-like point lists

The loader detects or infers the source format, then converts points into the canonical dataset representation.

### 3.2 Normalized tables

Supported tabular formats include:

- `.csv`
- `.tsv`
- `.xlsx`
- `.xls`

The required normalized columns are:

- `name`
- `x`
- `y`
- `z`

An optional `labels` column can be used to store aliases.

### 3.3 Generic point text files

Simple text-based point files are also supported. Typical accepted lines include:

```text
point_1, 1.0, 2.0, 3.0
point_2 4.0 5.0 6.0
point_3    7.0    8.0    9.0
```

### 3.4 GUI-derived 3D scene landmarks

Landmarks collected from the 3D scene workspace are committed back into the same `Dataset` abstraction. This means GUI-derived landmarks and imported tabular landmarks can be processed identically afterward.

## 4. Mathematical Foundations

Surgiplot’s metric implementations are not black boxes. Each metric family is computed from explicit geometric operations defined in the `surgiplot/metrics/` module set.

### 4.1 PCA best-fit plane

Several metrics require projecting a 3D polygon into a 2D local plane before area-related computations can be performed. This is handled in [pca_plane.py](/Users/leonardo/Projects/Surgiplot/surgiplot/core/geometry/pca_plane.py).

Method:

1. compute the centroid of the input 3D points
2. center the coordinates
3. perform PCA on the centered cloud
4. use the first two principal components as an in-plane basis
5. use the smallest-variance component as the plane normal

This produces:

- a best-fit local 2D coordinate frame
- a local plane normal
- a reversible projection between 3D and 2D

This is used by:

- `AREA_3D`
- `AOA_SF`
- `VOM_VOA`

### 4.2 Shoelace area

Whenever a planar polygon area must be computed after PCA projection, Surgiplot uses the shoelace formula on the 2D projected coordinates. This is the basis of:

- polygonal area estimation
- entry-area computation in surgical freedom
- area-matching in the VOM/VoA ellipse workflow

## 5. Metric Families

## 5.1 Distance in 3D

Implemented in [distance_3d.py](/Users/leonardo/Projects/Surgiplot/surgiplot/metrics/distance_3d.py).

Inputs:

- point `A`
- point `B`

Computation:

\[
d = \lVert B - A \rVert_2
\]

Interpretation:

- linear Euclidean distance in 3D space
- appropriate for straightforward point-to-point anatomical measurements

## 5.2 Area in 3D

Implemented in [area_3d.py](/Users/leonardo/Projects/Surgiplot/surgiplot/metrics/area_3d.py).

Inputs:

- a polygon with at least 3 3D points

Computation:

1. project the polygon into its PCA best-fit plane
2. compute 2D polygon area using the shoelace formula
3. report the area in square millimeters

Interpretation:

- area of an anatomical contour represented by a sparse 3D polygon
- useful when the contour is approximately planar or when a best-fit plane is the desired local representation

## 5.3 Volume in 3D

Implemented in [volume_3d.py](/Users/leonardo/Projects/Surgiplot/surgiplot/metrics/volume_3d.py).

Inputs:

- a sparse set of at least 4 non-coplanar 3D points

Computation:

1. test whether the point set spans 3D space
2. attempt an automatic alpha-shape-like tetrahedral reconstruction
3. if alpha-shape inference is unstable, fall back to the convex hull
4. return the enclosed volume

The code explicitly documents that the default intention is to estimate a closed solid from sparse boundary points rather than to perform full segmentation.

Interpretation:

- appropriate for sparse anatomical approximations of a closed volume
- more expressive than a convex hull when the alpha-shape succeeds
- still transparent because the fallback behavior is explicit

## 5.4 AoE: Angle of exposure

Implemented in [aoe.py](/Users/leonardo/Projects/Surgiplot/surgiplot/metrics/aoe.py).

Inputs:

- points `A`, `B`, `C`

Computation:

AoE is defined as the included angle at pivot `B` between the vectors:

\[
u = A - B
\]
\[
v = C - B
\]

and

\[
\text{AoE} = \arccos \left(\frac{u \cdot v}{\|u\| \|v\|}\right)
\]

Interpretation:

- point-based exposure angle
- useful for directional or aperture-style geometric assessment around a pivot

## 5.5 AoA and SF

Implemented in [aoa_sf.py](/Users/leonardo/Projects/Surgiplot/surgiplot/metrics/aoa_sf.py).

This family combines:

- `AoA` — Angle of Attack
- `SF` — surgical freedom

The code uses a pragmatic anatomical convention:

- entry points are expected in the order
  `cranial, caudal, medial, lateral`
- target is expected to be a single `pivot`

### AoA

Two triangles are formed:

- vertical triangle: `(cranial, caudal, pivot)`
- horizontal triangle: `(medial, lateral, pivot)`

For each triangle, Surgiplot computes its interior angle at the pivot. These are reported as:

- `aoa_vertical_deg`
- `aoa_horizontal_deg`

Interpretation:

- directional attack freedom in two orthogonal anatomical planes
- useful when a corridor is better described by paired angular descriptors than by one scalar

### SF

Surgical freedom is represented by the entry quadrilateral:

- the points are reordered into a non-crossing surface representation
- the quadrilateral is projected to a PCA best-fit plane
- area is computed in that local plane

Reported values:

- `sf_entry_area_mm2`
- `sf_entry_area_rescaled_mm2`

The rescaled SF option projects the entry points onto a sphere centered at the pivot with user-specified radius. This standardizes the effective distance from the target and makes entry-area comparisons less dependent on corridor depth.

Interpretation:

- `SF` measures the effective entry aperture
- standardized `SF` is useful when comparing cases with different pivot distances

## 5.6 VOM, sVOM, and VoA

Implemented in [vom_voa.py](/Users/leonardo/Projects/Surgiplot/surgiplot/metrics/vom_voa.py).

This family combines:

- `VoA` — Visuo-operative Angle
- `VOM` — Volume of Operative Maneuverability
- `sVOM` — standardized VOM

The implementation is explicitly documented in the code as a supplementary ellipse-based method.

### Conceptual motivation

Real surgical entry and target contours are often irregular polygons rather than ideal circles or ellipses. Surgiplot addresses this by:

1. preserving the original polygonal coordinates
2. projecting them into local PCA planes
3. fitting area-matched reference ellipses
4. lofting a 3D corridor between the target and entry reference sections

This creates a smooth geometric solid that is still tied to the measured polygon areas.

### Step-by-step method

1. `entry` and `target` polygons are resolved as 3D point sets
2. each polygon is projected into its PCA plane
3. the polygon area is computed using the shoelace formula
4. an area-matched ellipse is fitted while preserving the PCA-derived aspect ratio
5. the ellipse is reconstructed in 3D
6. the centroids of entry and target are linked into a corridor axis
7. intermediate elliptical rings are interpolated between target and entry
8. the stack of rings is triangulated into a closed 3D mesh
9. the enclosed mesh volume is computed

This yields `VOM`.

### VoA

VoA is computed from the centroid-to-centroid trajectory relative to the target polygon normal.

In the current code:

- the target-plane normal is estimated from the target polygon
- the angle between the corridor direction and that normal is computed
- VoA is derived as a bounded angle in degrees

Interpretation:

- orientation descriptor of the operative corridor
- complements volume rather than replacing it

### sVOM

sVOM is computed by truncating the full corridor at a standardized distal distance `stand_dist`, then computing the enclosed volume of that distal segment.

Interpretation:

- standardizes part of the corridor geometry for more comparable inter-case measurements
- especially useful when full corridor depth differs substantially between cases

### Important methodological note

The code explicitly avoids applying an arbitrary tilt-based multiplicative volume correction as the primary result. Instead, volume is computed from a physically defined 3D solid. This is an important design decision because it keeps the reported VOM and sVOM tied to an explicit geometric object rather than to an abstract angle-dependent scaling heuristic.

## 6. Python Library Usage

The main scientific API is exposed from [api.py](/Users/leonardo/Projects/Surgiplot/surgiplot/api.py).

### 6.1 Loading data

```python
import surgiplot as sp

ds = sp.load("case_01.csv", source="database")
```

Manual construction is also supported:

```python
ds = sp.from_points(
    points=[
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
    ],
    names=["cranial", "caudal", "medial", "pivot"],
    source="research",
)
```

### 6.2 Label management

```python
sp.apply_labels(
    ds,
    {
        "point_1": ["cranial"],
        "point_2": ["caudal"],
        "point_3": ["medial"],
        "point_4": ["lateral"],
        "point_5": ["pivot"],
    },
)
```

Canonical names can also be renamed:

```python
sp.rename_points(ds, {"point_120": "entry_anchor"})
```

### 6.3 Recommended metric-calling style

The most readable research style is to call metrics directly from the dataset:

```python
result = ds.VOM(entry="point_1-4", target="point_10-13", stand_dist=10.0)
print(result.vom_mm3, result.svom_mm3, result.voa_deg)

result = ds.AOA(
    entry=["cranial", "caudal", "medial", "lateral"],
    target="pivot",
    sf_rescale_radius_mm=10.0,
)
print(result.aoa_vertical_deg, result.aoa_horizontal_deg)
print(result.sf_entry_area_mm2, result.sf_entry_area_rescaled_mm2)
```

Other metric families:

```python
distance = ds.DISTANCE_3D(A="point_1", B="point_2")
area = ds.AREA_3D(polygon="point_1-4")
volume = ds.VOLUME_3D(points="point_1-9")
aoe = ds.AOE(A="point_10", B="point_11", C="point_12")
```

### 6.4 Scalar shortcuts

When only one scalar is needed:

```python
voa_deg = ds.VOA(entry="point_1-4", target="point_10-13")
svom_mm3 = ds.SVOM(entry="point_1-4", target="point_10-13")
sf_mm2 = ds.SF(entry=["cranial", "caudal", "medial", "lateral"], target="pivot")
```

### 6.5 Batch scripting pattern

A typical research script processes repeated target sets against a single entry set:

```python
import surgiplot as sp

ds = sp.load("case_07_labeled.json", source="database")

targets = {
    "set_A": ["t1", "t2", "t3", "t4"],
    "set_B": ["u1", "u2", "u3", "u4"],
}

rows = []
for name, target in targets.items():
    result = ds.VOM(
        entry=["cranial", "caudal", "medial", "lateral"],
        target=target,
        stand_dist=10.0,
    )
    rows.append(
        {
            "target": name,
            "vom_mm3": result.vom_mm3,
            "svom_mm3": result.svom_mm3,
            "voa_deg": result.voa_deg,
        }
    )
```

This style is well suited to notebooks, manuscript tables, and workflow engines.

## 7. CLI Guide

The CLI is intended for stateless, reproducible terminal-based workflows.

### 7.1 Main structure

```bash
surgiplot version
surgiplot dataset summary FILE
surgiplot dataset labels FILE [--label ...] [--rename ...] [--out ...]
surgiplot dataset export FILE OUT
surgiplot metric vom ...
surgiplot metric voa ...
surgiplot metric svom ...
surgiplot metric aoa ...
surgiplot metric sf ...
surgiplot metric aoe ...
surgiplot metric distance ...
surgiplot metric area ...
surgiplot metric volume ...
```

### 7.2 Dataset normalization

```bash
surgiplot dataset export raw_navigation.txt normalized_case.json --source navigation
```

### 7.3 Labeling

```bash
surgiplot dataset labels normalized_case.json \
  --label point_1=cranial \
  --label point_2=caudal \
  --label point_3=medial \
  --label point_4=lateral \
  --label point_5=pivot \
  --out labeled_case.json
```

### 7.4 Grouped corridor metrics

`VOM`, `sVOM`, and `VoA` are reported together:

```bash
surgiplot metric vom labeled_case.json \
  --entry point_1-4 \
  --target point_10-13 \
  --stand-dist 10.0
```

`AoA(v)`, `AoA(h)`, and `SF` are reported together:

```bash
surgiplot metric aoa labeled_case.json \
  --cranial cranial \
  --caudal caudal \
  --medial medial \
  --lateral lateral \
  --pivot pivot \
  --sf-rescale-radius-mm 10.0
```

### 7.5 Scalar calls

```bash
surgiplot metric voa labeled_case.json --entry point_1-4 --target point_10-13
surgiplot metric svom labeled_case.json --entry point_1-4 --target point_10-13
surgiplot metric sf labeled_case.json --cranial cranial --caudal caudal --medial medial --lateral lateral --pivot pivot
```

### 7.6 Area, distance, and volume

```bash
surgiplot metric distance labeled_case.json --A point_1 --B point_2
surgiplot metric area labeled_case.json --polygon point_1-4
surgiplot metric volume labeled_case.json --points point_1-9
```

CLI outputs are emitted as structured JSON-like text. This makes them easy to redirect into files, capture in shell pipelines, or parse downstream in automated research workflows.

## 8. GUI Guide

The GUI is most useful when the study begins from visual anatomy rather than from an already finalized point table.

Launch with:

```bash
surgiplot_gui
```

## 8.1 Tabular dataset workflow

The GUI can be used for:

- reviewing loaded point tables
- inspecting coordinates and labels
- refining aliases
- preparing a dataset before script-based analysis

This is useful when a dataset exists already but requires curation before formal metric computation.

## 8.2 3D rendered model and scene workflow

The GUI also supports a scene-driven workflow in which landmarks are collected directly from an imported 3D model.

### Step 1: open the 3D scene workspace

This workspace is designed for:

- importing a rendered scene bundle
- calibrating local scale
- collecting labeled landmarks
- committing those points into the active dataset

### Step 2: import the scene bundle

Use `Select Scene…` and choose:

- the primary geometry file
- any companion files such as `MTL` or texture images, if present

Supported scene classes include:

- `OBJ`
- `PLY`
- `STL`
- `XYZ / PTS / CSV / TXT`

### Step 3: load the scene

Click `Load Scene`.

The current default render strategy shows the imported scene as a combined anatomical layer:

- mesh
- rendered imaging / points

This combination was chosen because it preserves realism while maintaining practical interactive performance in the current rendering stack.

### Step 4: calibrate physical scale

To convert scene units into millimetric units:

1. enter the known anatomical distance in the `Known Distance` field
2. activate `Set Scale`
3. click two points in the rendered scene

Surgiplot rescales the imported scene so that the selected distance matches the specified physical value.

This calibration matters because downstream coordinates and metric outputs are only anatomically meaningful in physical units.

### Step 5: collect landmarks

Activate `Collect Point`, then click on the rendered anatomy.

For each selected point:

- a label is requested
- the point is added to the pending landmark table
- the point is drawn back into the 3D viewer
- the point remains selectable for validation

### Step 6: refine the pending landmark set

Before committing, the pending set can be:

- renamed
- pruned
- undone
- cleared

This lets the scene workspace function as a true landmark-acquisition interface rather than a one-shot picker.

### Step 7: commit to the dataset

When satisfied, click `Commit to Dataset`.

This stores:

- the collected landmarks
- scene provenance
- calibration metadata
- imported-scene visualization payload for later review

## 8.3 3D analysis after landmark acquisition

Once committed, the main analysis/editor viewer becomes the place where geometry is interpreted.

At that stage, Surgiplot can:

- display the imported scene as anatomical context
- overlay the landmark dataset
- render metric constructs
- support direct visual review of corridor geometry

This is particularly useful for:

- confirming that entry and target sets are anatomically sensible
- checking whether point order is correct
- visually comparing multiple target definitions
- reviewing how computed constructs sit relative to the original rendered scene

## 9. Recommended Research Workflows

## 9.1 CLI-first workflow

Best when:

- the point table already exists
- the study is large
- reproducibility and batch processing are primary concerns

Recommended pattern:

1. normalize the source export
2. assign stable labels
3. compute metrics in scripts
4. save structured outputs

## 9.2 GUI-first workflow

Best when:

- the landmark set must be acquired visually
- the anatomy is easier to interpret in a rendered 3D scene
- metric validity depends on iterative anatomical inspection

Recommended pattern:

1. import the rendered model
2. calibrate it
3. collect landmarks
4. commit them to the dataset
5. export the dataset
6. run repeated metrics later in CLI or Python

## 9.3 Hybrid workflow

For many research programs, the most practical pattern is hybrid:

- `GUI`
  visual acquisition and validation

- `Python / CLI`
  repeated metric computation, aggregation, and manuscript-ready reproducibility

## 10. Reproducibility Recommendations

For publication-facing work, it is good practice to preserve:

1. the original source export
2. the normalized Surgiplot dataset
3. the label mapping
4. the metric call parameters
5. the Surgiplot version

The package version can be retrieved with:

```bash
surgiplot version
```

or:

```python
import surgiplot as sp
print(sp.__version__)
```

## 11. Suggested Methods-Section Description

The following paragraph can be adapted for manuscripts or internal protocols:

> Surgiplot is a local coordinate-based analysis framework for neurosurgical anatomical research. Heterogeneous point sources, including neuronavigation exports, normalized tabular landmark datasets, and landmarks acquired directly on rendered 3D models, are converted into a common dataset abstraction containing explicit 3D coordinates and semantic labels. Operative metrics are computed from transparent geometric procedures, including Euclidean distance, PCA-plane polygon area, sparse-point enclosed volume, angle-of-exposure, angle-of-attack and surgical-freedom constructs, and an ellipse-based visuo-operative corridor model yielding VoA, VOM, and standardized VOM. The same dataset can be processed interactively in the GUI or reproducibly through Python and CLI workflows.

## 12. Scope and Boundaries

Surgiplot is intentionally designed as a coordinate-analysis system rather than as a general-purpose surgical planning platform or full-featured mesh-editing environment.

Its strengths are:

- explicit coordinate geometry
- transparent metric implementations
- scriptability
- integration of visual landmark acquisition with reproducible downstream computation

Its current 3D tooling should be understood as:

- a landmark-acquisition and metric-context viewer
- not a replacement for dedicated high-end mesh workstations or full surgical planning suites

## 13. Related Repository References

- project overview: [README](/Users/leonardo/Projects/Surgiplot/README.md)
- public API: [api.py](/Users/leonardo/Projects/Surgiplot/surgiplot/api.py)
- dataset model: [dataset.py](/Users/leonardo/Projects/Surgiplot/surgiplot/core/dataset.py)
- VOM / VoA / sVOM implementation: [vom_voa.py](/Users/leonardo/Projects/Surgiplot/surgiplot/metrics/vom_voa.py)
- AoA / SF implementation: [aoa_sf.py](/Users/leonardo/Projects/Surgiplot/surgiplot/metrics/aoa_sf.py)
- AoE implementation: [aoe.py](/Users/leonardo/Projects/Surgiplot/surgiplot/metrics/aoe.py)
- 3D area implementation: [area_3d.py](/Users/leonardo/Projects/Surgiplot/surgiplot/metrics/area_3d.py)
- 3D volume implementation: [volume_3d.py](/Users/leonardo/Projects/Surgiplot/surgiplot/metrics/volume_3d.py)
- CLI: [app.py](/Users/leonardo/Projects/Surgiplot/surgiplot/cli/app.py)
- GUI: [app.py](/Users/leonardo/Projects/Surgiplot/surgiplot/gui/app.py)
- 3D scene workflow: [scene_reconstruction.py](/Users/leonardo/Projects/Surgiplot/surgiplot/ai/scene_reconstruction.py)
