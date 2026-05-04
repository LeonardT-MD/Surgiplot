<p align="center">
  <img src="docs/assets/SURGIPLOT_BANNER.png" alt="Surgiplot banner" width="100%">
</p>

<p align="center">
  <img src="docs/assets/LOGO_SURGIPLOT.png" alt="Surgiplot logo" width="180">
</p>

# Surgiplot

For a longer publication-oriented methods and usage manual, see [docs/SURGIPLOT_PUBLICATION_GUIDE.md](/Users/leonardo/Projects/Surgiplot/docs/SURGIPLOT_PUBLICATION_GUIDE.md).

Surgiplot is a local scientific software environment for quantitative neuroanatomical and skull-base analysis. It provides:

1. a desktop GUI for dataset curation, imported 3D scene landmark acquisition, and interactive visualization
2. a Python library and CLI for reproducible scripting, batch studies, and research pipelines

Its core design principle is simple: once landmarks are represented as explicit 3D coordinates in a common frame, operative metrics should be computed from inspectable geometric constructs rather than opaque software state.

## Scope

Surgiplot currently supports:

- `VoA` — Visuo-operative Angle
- `VOM` — Volume of Operative Maneuverability
- `sVOM` — standardized VOM
- `AoA` — Angle of Attack
- `SF` — Surgical Freedom
- `AoE` — Angle of Exposure
- `3D Distance`
- `3D Area`
- `3D Volume`

The same metric logic can be applied to landmarks originating from:

- neuronavigation exports
- normalized tabular datasets
- generic 3D point files
- manually assembled point sets
- imported 3D scenes curated through the GUI

## Installation

### Core package

```bash
pip install -e .
```

### With 3D scene rendering support

```bash
pip install -e .[scene3d]
```

### Main entry points

```bash
surgiplot
surgiplot version
surgiplot_gui
```

## Conceptual model

Surgiplot uses a single editable `Dataset` object as its canonical data container.

Each dataset stores:

- canonical point identifiers such as `point_1`, `point_2`, ...
- optional semantic labels or aliases such as `apex`, `cranial`, `pivot`, `medial`
- metadata such as acquisition source, vendor, and coordinate system

This allows the same metric function to be called using either canonical point names or human-readable labels.

In practice, Surgiplot now supports two equivalent scripting idioms:

1. module-level metric calls, such as `sp.VOM(data=ds, ...)`
2. dataset-bound metric calls, such as `ds.VOM(...)`

The second style is often more natural in research scripts where one dataset is reused across many metric calls.

## Python library

### Minimal example

```python
import surgiplot as sp

ds = sp.load("annotations.csv", source="database")

result = ds.DISTANCE_3D(A="point_1", B="point_2")
print(result.distance_mm)
```

### Manual dataset construction

```python
import surgiplot as sp

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

### Relabeling and canonical point management

```python
import surgiplot as sp

ds = sp.load("navigation_export.txt", source="navigation")

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

sp.rename_points(ds, {"point_120": "entry_anchor"})
```

### Research-facing convenience API

The full metric kernels remain available:

```python
sp.VOM_VOA(...)
sp.AOA_SF(...)
sp.AOE(...)
sp.DISTANCE_3D(...)
sp.AREA_3D(...)
sp.VOLUME_3D(...)
```

For script ergonomics, Surgiplot also exposes convenience wrappers:

```python
sp.VOM(...)
sp.VOA(...)
sp.SVOM(...)
sp.AOA(...)
sp.SF(...)
```

These wrappers do not change the underlying computation.

- `sp.VOM(...)` is the grouped convenience call for the `VOM / sVOM / VoA` family and returns the full `VOMVOAResult`
- `sp.AOA(...)` is the grouped convenience call for the `AoA / SF` family and returns the full `AOASFResult`
- `sp.VOA(...)`, `sp.SVOM(...)`, and `sp.SF(...)` remain scalar shortcuts when only one quantity is needed

### Dataset-bound metric methods

Every major metric can also be called directly from a `Dataset` instance:

```python
ds.VOM(entry=[...], target=[...], stand_dist=10.0)
ds.AOA(entry=[...], target="pivot", sf_rescale_radius_mm=10.0)
ds.AOE(A="point_1", B="point_2", C="point_3")
ds.DISTANCE_3D(A="point_1", B="point_2")
ds.AREA_3D(polygon=["point_1", "point_2", "point_3", "point_4"])
ds.VOLUME_3D(points=["point_1", "point_2", "point_3", "point_4", "point_5", "point_6"])
```

This is the recommended style when:

- one dataset is loaded once and reused many times
- multiple targets or polygons are tested in sequence
- notebook or pipeline code should remain compact and readable

### Example: VOM / VoA workflow

```python
import surgiplot as sp

ds = sp.load("case_07.csv", source="database")

result = sp.VOM(
    data=ds,
    entry=["entry_a", "entry_b", "entry_c", "entry_d"],
    target=["target_a", "target_b", "target_c", "target_d"],
    stand_dist=10.0,
    return_debug=True,
)

print(result.voa_deg)
print(result.vom_mm3)
print(result.svom_mm3)
```

The same grouped call is available directly from the dataset object, which is often the simplest scripting style for repeated measurements:

```python
import surgiplot as sp

ds = sp.load("case_07.csv", source="database")

result = ds.VOM(
    entry=["entry_a", "entry_b", "entry_c", "entry_d"],
    target=["target_a", "target_b", "target_c", "target_d"],
    stand_dist=10.0,
)

print(result.vom_mm3, result.svom_mm3, result.voa_deg)
```

This makes repeated target testing straightforward:

```python
targets = {
    "target_set_a": ["t1", "t2", "t3", "t4"],
    "target_set_b": ["u1", "u2", "u3", "u4"],
}

for name, target in targets.items():
    result = ds.VOM(
        entry=["cranial", "caudal", "medial", "lateral"],
        target=target,
        stand_dist=10.0,
    )
    print(name, result.vom_mm3, result.svom_mm3, result.voa_deg)
```

### Example: AoA / SF workflow

```python
import surgiplot as sp

ds = sp.load("case_07.csv", source="database")

result = ds.AOA(
    entry=["cranial", "caudal", "medial", "lateral"],
    target="pivot",
    sf_rescale_radius_mm=10.0,
    return_debug=True,
)

print(result.aoa_vertical_deg)
print(result.aoa_horizontal_deg)
print(result.sf_entry_area_mm2)
print(result.sf_entry_area_rescaled_mm2)
```

### Example: additional metric families

```python
aoe = ds.AOE(A="point_10", B="point_11", C="point_12")
distance = ds.DISTANCE_3D(A="point_1", B="point_2")
area = ds.AREA_3D(polygon=["point_1", "point_2", "point_3", "point_4"])
volume = ds.VOLUME_3D(points=["point_1", "point_2", "point_3", "point_4", "point_5", "point_6"])

print(aoe.aoe_deg)
print(distance.distance_mm)
print(area.area_mm2)
print(volume.volume_mm3)
```

### Scalar shortcuts

When only one scalar quantity is needed:

```python
voa_deg = ds.VOA(
    entry=["cranial", "caudal", "medial", "lateral"],
    target=["t1", "t2", "t3", "t4"],
)

svom_mm3 = ds.SVOM(
    entry=["cranial", "caudal", "medial", "lateral"],
    target=["t1", "t2", "t3", "t4"],
)

sf_mm2 = ds.SF(
    entry=["cranial", "caudal", "medial", "lateral"],
    target="pivot",
)
```

## CLI guide

The CLI is designed for stateless research workflows: load a dataset or annotation file, optionally normalize/relabel it, and emit structured results.

## CLI overview

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

### Dataset inspection

```bash
surgiplot dataset summary case_01.csv --source database
```

### Relabeling for downstream scripts

```bash
surgiplot dataset labels case_01.csv \
  --label point_1=cranial \
  --label point_2=caudal \
  --label point_3=medial \
  --label point_4=lateral \
  --label point_5=pivot \
  --out case_01_labeled.json
```

### Renaming canonical points

```bash
surgiplot dataset labels case_01.csv \
  --rename point_120=entry_anchor \
  --rename point_121=target_anchor \
  --out case_01_renamed.json
```

### Export to normalized JSON

```bash
surgiplot dataset export raw_navigation.txt normalized_case.json --source navigation
```

### VOM / VoA from the terminal

```bash
surgiplot metric vom case_01_labeled.json \
  --entry cranial,caudal,medial,lateral \
  --target target_a,target_b,target_c,target_d \
  --stand-dist 10.0
```

This grouped command emits `VOM`, `sVOM`, and `VoA` together.

If you only need one scalar:

```bash
surgiplot metric voa case_01_labeled.json \
  --entry cranial,caudal,medial,lateral \
  --target target_a,target_b,target_c,target_d

surgiplot metric svom case_01_labeled.json \
  --entry cranial,caudal,medial,lateral \
  --target target_a,target_b,target_c,target_d
```

### AoA / SF from the terminal

```bash
surgiplot metric aoa case_01_labeled.json \
  --cranial cranial \
  --caudal caudal \
  --medial medial \
  --lateral lateral \
  --pivot pivot \
  --sf-rescale-radius-mm 10.0
```

This grouped command emits `AoA(v)`, `AoA(h)`, and `SF`.

If only the SF scalar is needed:

```bash
surgiplot metric sf case_01_labeled.json \
  --cranial cranial \
  --caudal caudal \
  --medial medial \
  --lateral lateral \
  --pivot pivot
```

### Area, distance, and volume

```bash
surgiplot metric distance case_01_labeled.json --A point_1 --B point_2
surgiplot metric area case_01_labeled.json --polygon point_1,point_2,point_3,point_4
surgiplot metric volume case_01_labeled.json --points point_1,point_2,point_3,point_4,point_5,point_6
```

Point-list arguments also support compact numeric ranges, which is especially useful for polygonal area and sparse volume calls:

```bash
surgiplot metric area case_01_labeled.json --polygon point_1-4
surgiplot metric volume case_01_labeled.json --points point_1-9
```

Accepted range styles include:

- `point_1-9`
- `point_1:9`
- `1-9`

The same range syntax is also supported by multi-point Python calls such as:

```python
result = ds.VOM(entry="point_1-4", target="point_10-13", stand_dist=10.0)
area = ds.AREA_3D(polygon="point_1-4")
volume = ds.VOLUME_3D(points="point_1-9")
```

CLI outputs are emitted as structured JSON-like text, making them easy to capture in shell pipelines, notebooks, Snakemake rules, or institutional batch-processing environments.

### Recommended scripting pattern

For reproducible academic workflows, the recommended pattern is:

1. normalize or export the raw annotations once
2. assign stable anatomical labels
3. reuse the labeled dataset in repeated metric calls
4. capture outputs as JSON in notebooks, shell scripts, or workflow engines

For example:

```bash
surgiplot dataset export raw_navigation.txt normalized_case.json --source navigation

surgiplot dataset labels normalized_case.json \
  --label point_1=cranial \
  --label point_2=caudal \
  --label point_3=medial \
  --label point_4=lateral \
  --label point_5=pivot \
  --out labeled_case.json

surgiplot metric aoa labeled_case.json \
  --cranial cranial \
  --caudal caudal \
  --medial medial \
  --lateral lateral \
  --pivot pivot
```

## Supported inputs

### Navigation imports

- Stryker annotation exports
- Medtronic JSON annotation exports
- generic navigation-like point lists

### Tabular / normalized inputs

- `.csv`
- `.tsv`
- `.xlsx`
- `.xls`

Required normalized columns:

- `name`
- `x`
- `y`
- `z`

### Generic point text

Accepted line formats include:

```text
point_1, 1.0, 2.0, 3.0
point_2 4.0 5.0 6.0
point_3\t7.0\t8.0\t9.0
```

## GUI guide

The GUI is the preferred environment when the workflow begins from visual anatomy rather than from a pre-existing table of coordinates.

It is particularly useful for:

- reviewing and editing point datasets interactively
- importing a rendered 3D model or scene bundle
- calibrating a scene to real-world scale
- collecting landmarks directly on the rendered anatomy
- inspecting metric geometry visually after computation

Launch with:

```bash
surgiplot_gui
```

### Main GUI roles

The GUI has two complementary roles:

1. `dataset curation`
   Load or edit a landmark dataset, assign or refine labels, and prepare the point set for scripting or metric computation.

2. `scene-driven landmark acquisition`
   Import a 3D scene, calibrate it, pick landmarks directly on the rendered model, then commit those landmarks back into the dataset for analysis.

### Typical GUI workflow

The usual end-to-end workflow is:

1. open `surgiplot_gui`
2. import or create a dataset, or start from an empty dataset if landmarks will be acquired from a 3D scene
3. open the 3D scene workspace
4. import the rendered scene or model bundle
5. calibrate the scene scale from two known points
6. collect labeled landmarks on the rendered model
7. commit those landmarks to the dataset
8. run the desired metric family in the main analysis viewer

### 3D rendered scene workflow

The scene-import workspace is designed for anatomical landmark acquisition on an imported 3D model.

#### 1. Select the model bundle

Use `Select Scene…` to choose the primary geometry file and any companion assets in one pass.

Typical supported bundles include:

- `OBJ + MTL + texture images`
- `PLY`
- `STL`
- `XYZ / PTS / CSV / TXT` point-based scene files

If companion files are present, Surgiplot keeps them associated with the primary geometry during import.

#### 2. Load the scene

Click `Load Scene`.

At load time Surgiplot:

- imports the scene geometry and colors
- prepares the combined default rendering
- keeps the imported scene available as measurement reference geometry

The current default display target is a combined anatomical rendering:

- `mesh`
- `rendered imaging / points`

This combination is shown together because it provides the best compromise between realism and interactive usability for the current renderer path.

#### 3. Calibrate scale

Before collecting final landmarks, calibrate the model to physical units.

Use:

- `Known Distance`
- `Set Scale`

Then:

1. enter the known distance in centimeters
2. activate `Set Scale`
3. click two anatomical points in the rendered scene

Surgiplot rescales the imported scene so that the picked distance matches the specified physical distance. After calibration, downstream coordinates are stored in millimetric space rather than arbitrary scene units.

#### 4. Collect landmarks

Turn on `Collect Point`, then click directly on the rendered anatomy.

For each pick:

- Surgiplot asks for a landmark label
- the point is added to the pending landmark table
- the point is shown back in the 3D viewer
- the point remains selectable in both the table and the rendered scene

This is useful for constructing datasets directly from 3D anatomical scenes before any CLI or Python analysis.

#### 5. Refine the pending landmark set

Before committing, the right-side pending table can be used to:

- rename landmarks
- delete a selected landmark
- undo the last collected point
- clear the pending set

This lets the scene workspace function as a geometric acquisition step before dataset finalization.

#### 6. Commit landmarks to the dataset

When the pending landmark set is ready, click `Commit to Dataset`.

Surgiplot stores:

- the collected landmark coordinates
- the scene-import provenance
- scene calibration metadata
- imported scene payload for later visualization in the analysis viewer

### 3D picking behavior

The current 3D picking workflow is designed so that landmark acquisition remains spatially faithful to the rendered model:

- when a surface mesh is present, picks are resolved on the mesh surface
- collected points remain visible in the 3D scene
- selecting a point in the table highlights it in the rendered view
- clicking near an existing rendered landmark can reselect it

This makes the scene workspace usable not only for acquisition but also for immediate spatial verification.

### Analysis after 3D acquisition

After landmarks are committed, the main analysis/editor viewer becomes the place for metric interpretation.

In that stage, Surgiplot can:

- show the imported scene as background context
- overlay landmarks and metric constructs on top of it
- compute and inspect `VOM / sVOM / VoA`
- compute and inspect `AoA / SF`
- compute and inspect `AoE`, `distance`, `area`, and `volume`

The imported scene is retained as a combined layer so that anatomical context remains visible during metric review.

### Recommended GUI-to-CLI workflow

For many research projects, the most practical pattern is:

1. acquire or refine landmarks in the GUI
2. commit and save the resulting dataset
3. run repeated metrics in Python or through the CLI
4. return to the GUI only when additional landmark refinement or visual verification is needed

This creates a good separation between:

- `GUI`: visual acquisition, curation, validation
- `CLI / Python`: reproducible repeated computation, batch analysis, and manuscript-grade pipelines

## Package layout

```text
surgiplot/
  core/
    dataset.py
    io/
    geometry/
    transforms/
  metrics/
    aoa_sf.py
    aoe.py
    vom_voa.py
    distance_3d.py
    area_3d.py
    volume_3d.py
  cli/
    app.py
  gui/
    app.py
  ai/
    scene_reconstruction.py
    scene_picker.py
```

## Reproducibility notes

For research use, the recommended pattern is:

1. preserve the original export file
2. normalize to a Surgiplot dataset
3. record label mappings explicitly
4. run metrics from a script or CLI command
5. save metric outputs together with dataset version and package version

You can retrieve the installed package version with:

```bash
surgiplot version
```

or in Python:

```python
import surgiplot as sp
print(sp.__version__)
```

## Current design boundaries

Surgiplot is intentionally oriented around explicit coordinate-based geometry. It is not a general-purpose mesh editing system, and it does not attempt to replace specialized surgical planning or scientific visualization platforms.

For large-scale interactive 3D scene handling, the GUI is best regarded as a landmark acquisition and metric-integration environment rather than a full native mesh workstation.

## Citation-style description

If you need a concise methodological description for internal notes or manuscript methods sections:

> Surgiplot is a local coordinate-based analysis environment for neurosurgical anatomical research. It normalizes heterogeneous 3D landmark sources into a common dataset abstraction and computes explicit geometric metrics including AoA, SF, VoA, VOM, sVOM, AoE, linear distance, polygonal area, and sparse point-derived volume.
