# Surgiplot

Surgiplot is a local quantitative neuroanatomical measurement platform for reproducible geometric analysis of surgical corridors, operative angles, exposure metrics, point-derived surfaces, and landmark-based volumetry.

The project is designed for workflows in which the user has a set of 3D landmarks coming from neuronavigation, a manually curated coordinate table, or a reconstructed 3D scene, and wants to turn those landmarks into interpretable quantitative measurements with transparent geometry.

Surgiplot currently supports:

- **VoA**: Visuooperative Angle
- **VOM**: Volume of Operative Maneuverability
- **sVOM**: standardized VOM
- **AoA**: Angle of Attack
- **SF**: Surgical Freedom
- **AoE**: Angle of Exposure
- **AE**: Area of Exposure
- **3D Distance**
- **Volume** from sparse 3D boundary points

It can be used as:

1. a **desktop GUI** for dataset curation, 3D scene exploration, landmark picking, and metric visualization
2. a **local Python toolkit / CLI** for scripting, auditing, and reproducible batch-style analysis

## Why Surgiplot exists

Modern skull base and neuroanatomical analysis often depends on custom spreadsheets, one-off notebooks, or platform-specific navigation exports that are difficult to standardize across users, cases, and publications. Surgiplot was built to provide:

- a **shared data model** for anatomically meaningful 3D points
- **transparent metric computation** with inspectable geometric intermediates
- **interactive 3D visualization** of both the anatomical scene and the measured constructs
- a **bridge between navigation-derived and image/mesh-derived workflows**

The core idea is simple: once a set of 3D points is available in a common coordinate frame, multiple operative measurements can be defined, reproduced, visualized, and audited from the same dataset.

## Core data model

Regardless of acquisition source, Surgiplot normalizes data into a shared editable dataset.

Each dataset includes:

- canonical coordinate names such as `point_1`, `point_2`, ...
- an editable `labels` field for semantic aliases such as `opticocarotid_point`, `apc`, `cranial`, etc.

All metric calls can reference either:

- canonical point IDs
- semantic labels
- compact numeric shorthand where appropriate

Examples:

```python
entry = ["point_1", "point_2", "point_3", "point_4"]
```

```python
entry = ["clinoid_A", "clinoid_B", "clinoid_C", "clinoid_D"]
```

```text
1-4
```

In the GUI, numeric range shorthand such as `1-5` is expanded to `1, 2, 3, 4, 5` in multi-point metric fields.

## Supported data sources

Surgiplot is intended to work locally with heterogeneous sources of 3D geometry, including:

- neuronavigation exports
- structured point tables (`CSV`, `TSV`, `XLSX`)
- manually entered coordinates
- external 3D meshes / point clouds imported into the 3D Scene Workspace

The imported or collected points are ultimately mapped into the same dataset abstraction, which means downstream metric logic does not need to care whether the landmarks came from a navigation platform or from a 3D mesh-based scene.

## Main workflows

### 1. Navigation / table workflow

1. Import a navigation export or point table
2. Inspect and curate the dataset
3. Edit labels or add supplementary points
4. Proceed to analysis
5. Compute one or more quantitative metrics
6. Visualize the resulting geometry in 3D

### 2. 3D scene workflow

1. Import a mesh or point cloud into the **3D Scene Workspace**
2. Visually inspect the anatomy in a GPU-backed 3D viewer
3. Rescale the scene by selecting two points and entering the known real-world distance
4. Collect landmarks directly on the scene
5. Commit those landmarks into the Surgiplot dataset
6. Move to the analysis workspace and compute the same metrics used for navigation-derived points

This second workflow is particularly useful when the user has an external anatomical reconstruction and wants to treat it as a quantitative landmark source.

## Metric overview

### VoA: Visuooperative Angle

**Concept**

VoA describes the alignment between the operative trajectory and the target base. In the current implementation it is derived from the centroid-to-centroid direction between entry and target constructs relative to the target plane normal.

**Rationale**

A purely volumetric corridor description is incomplete if trajectory orientation is ignored. Two corridors can have similar cross-sectional dimensions but substantially different approach geometry relative to the target.

**Use cases**

- comparing approaches to the same target
- quantifying directional alignment of an operative corridor
- reporting orientation together with VOM / sVOM

### VOM: Volume of Operative Maneuverability

**Concept**

VOM models the operative corridor volume between entry and target regions. In the current full methodology, irregular entry and target polygons are:

1. projected to PCA best-fit planes
2. converted into area-matched reference ellipses
3. reconstructed in 3D
4. lofted into a geometric corridor solid

The principal output is the enclosed volume of that 3D solid.

**Rationale**

This provides a corridor-oriented volume metric that is more anatomically and procedurally meaningful than generic Euclidean bounding volumes.

**Use cases**

- corridor comparison across approaches
- target-specific access-volume analysis
- methodological studies of operative maneuverability

### sVOM: standardized VOM

**Concept**

sVOM represents a standardized distal segment of the corridor, rather than the full entry-to-target extent. It is useful when the user wants a normalized terminal portion of the corridor for cross-case comparability.

**Rationale**

A standardized distal segment may be more robust for comparing effective working space near a target than a full-length corridor whose proximal extent varies substantially by case or approach definition.

**Use cases**

- near-target corridor comparison
- standard-distance methodological studies
- analyses focused on distal working volume rather than full exposure

### AoA: Angle of Attack

**Concept**

AoA is computed in vertical and horizontal components:

- **vertical AoA** from the cranial-caudal-pivot triangle
- **horizontal AoA** from the medial-lateral-pivot triangle

Optional standardization projects each entry point along its pivot ray to a fixed distance before recalculating standardized geometry.

**Rationale**

Approach angles help quantify directional access independently from pure area or volume. Vertical and horizontal decomposition makes the metric more interpretable in anatomical and surgical terms.

**Use cases**

- comparing working direction across approaches
- quantifying access angle at a target
- combining directional and areal corridor analysis

### SF: Surgical Freedom

**Concept**

SF is represented as the quadrilateral surface defined by the ordered entry points:

- cranial
- lateral
- caudal
- medial

The software also supports a standardized version after projecting the entry points to a fixed pivot-centered distance.

**Rationale**

SF provides a planarized operational summary of the entry maneuvering envelope, complementing the AoA triangles and the volumetric corridor metrics.

**Use cases**

- comparing entry working area
- standardized access-window analysis
- reporting proximal maneuverability independently of corridor length

### AoE: Angle of Exposure

**Concept**

AoE describes a 3-point angular construct and is visualized in the GUI as a 360 degree circular plot with the measured exposed angle highlighted.

**Rationale**

This is useful when the relevant question is angular exposure around a pivot or viewpoint rather than 3D corridor volume.

**Use cases**

- angular target exposure analysis
- comparing visible or exposed sectors between approaches

### AE: Area of Exposure

**Concept**

AE is computed from an ordered 3D polygon. The polygon is:

1. projected onto a PCA best-fit plane
2. converted into 2D plane coordinates
3. measured with the shoelace formula

The GUI can also display a reconstructed coplanar fitted surface in native 3D space.

**Rationale**

Anatomical point sets are often nearly, but not perfectly, coplanar. PCA-based plane fitting provides a reproducible way to define an idealized exposure surface.

**Use cases**

- measuring exposure windows from ordered landmarks
- comparing planarized surface extents
- methodological studies of exposure area

### 3D Distance

**Concept**

This metric computes Euclidean distance between two 3D landmarks.

**Rationale**

Simple distances remain fundamental for validation, scaling checks, landmark separation, and descriptive anatomy.

**Use cases**

- linear anatomical measurements
- scale confirmation
- target-entry separation

### Volume

**Concept**

This metric estimates the volume of a closed solid reconstructed from a sparse boundary point cloud. The implementation attempts a locally inferred alpha-shape-like reconstruction and falls back to a convex hull only when necessary.

**Rationale**

Users may have a set of points sampled from the perimeter or boundary of an anatomical compartment without a ready-made segmentation. This function provides a volumetric reconstruction from those sparse 3D coordinates.

**Use cases**

- point-based surrogate segmentation
- cavity or compartment volume estimation from sparse samples
- exploratory volumetric reconstruction in anatomical research

## Methodological principles

Surgiplot is built around a few explicit methodological choices:

- **shared coordinate frame first**: all metrics assume the landmarks live in one coherent 3D frame
- **inspectable geometry**: important intermediate constructs are available for plotting and debugging
- **explicit geometric models**: corridor or surface measurements are tied to defined geometric constructs rather than opaque black-box outputs
- **local execution**: users can run the software on their own machine without requiring a cloud workflow

For example:

- `AE` is not computed as a warped mesh area; it is computed as a PCA-plane-projected polygon area
- `VOM` is not reported as a heuristic shading artifact; it is computed from an explicit lofted corridor model
- `AoA / SF` standardization does not globally scale the case; it projects each entry point along its pivot ray to a chosen fixed distance

## 3D visualization philosophy

Surgiplot separates:

- the **background anatomical scene** (imported mesh / point cloud)
- the **metric geometry** (points, polygons, ellipses, corridor shells, triangles, surfaces, arrows)

This distinction matters because users often need to:

- fade the anatomy back
- turn textures off and inspect mesh shape only
- strengthen metric overlays for publication screenshots or methodological review

The current analysis viewer therefore provides separate controls for:

- **3D model rendering**
  - opacity
  - point size
  - mesh edges
  - source colors / texture on-off
- **metric overlay rendering**
  - opacity
  - emphasis

The metric scenes are rendered directly in the same 3D coordinate system as the imported anatomy, so the overlay is geometrically meaningful rather than schematic.

## Installation

### Recommended local setup

Surgiplot is designed to run locally. A conda or Miniforge environment is recommended for a stable scientific stack.

```bash
git clone https://github.com/LeonardT-MD/Surgiplot.git
cd Surgiplot

conda create -n surgiplot python=3.11 pip -y
conda activate surgiplot

pip install -e .
```

### Optional GPU-backed 3D scene backend

For better imported mesh / point-cloud visualization and PyVista-based 3D analysis scenes:

```bash
pip install -e ".[scene3d]"
```

This installs:

- `pyvista`
- `pyvistaqt`
- `vtk`

If `.[scene3d]` is not installed, Surgiplot falls back to the Matplotlib-based 3D viewer where possible.

### Local installer script

```bash
bash surgiplot/scripts/install_conda.sh
```

## Running Surgiplot

### GUI mode

```bash
surgiplot_gui
```

The GUI supports:

- dataset import and curation
- direct label editing
- 3D scene import
- local scene rescaling from two picked 3D points
- landmark collection into the shared dataset
- metric configuration and analysis
- 3D overlay plotting
- result inspection and debug review

### CLI / library mode

```bash
surgiplot
```

Surgiplot can also be used from scripts or notebooks where the user wants explicit control of data loading, metric calls, and downstream reproducibility.

## Example use cases

### Navigation-derived operative corridor study

A user exports landmarks from a neuronavigation platform, assigns anatomical aliases, computes AoA, SF, VoA, VOM, and sVOM, then compares results across approaches or specimens.

### Mesh-based anatomical landmark collection

A user imports a textured anatomical mesh, rescales it from a known ruler or reference distance, picks points directly on the model, commits them into a dataset, and computes the same metrics as if the points had come from neuronavigation.

### Point-based boundary reconstruction

A user samples points around the perimeter of an anatomical compartment and uses the `Volume` metric to obtain an inferred enclosed volume from sparse coordinates.

### Reproducible methods development

A user develops a landmark scheme in the GUI, exports the curated dataset, and reruns the exact measurements through scripted Python code for publication or validation.

## Reproducibility and transparency

Surgiplot is intended to make quantitative workflows easier to inspect and re-run. The platform supports:

- explicit named landmarks
- editable labels and aliases
- debug payloads containing intermediate geometric data
- 3D visualization of the measured constructs
- export of datasets and result payloads

The goal is not merely to provide a number, but to make the geometry that produced that number understandable.

## Current scope and limitations

Surgiplot is intended for quantitative geometric analysis, not for autonomous diagnosis or clinical decision-making.

Important practical limitations include:

- results depend on the quality and consistency of the chosen landmarks
- polygon-based metrics depend on meaningful point ordering
- point-derived volume reconstruction remains a model of a likely solid, not a true image segmentation
- imported meshes and point clouds are only as metrically valid as their own reconstruction pipeline and scaling

These limitations are not bugs; they are part of the epistemology of landmark-based anatomical measurement and should be acknowledged in study design and reporting.

## Project status

Surgiplot is an actively evolving local research software project. The codebase currently contains both mature metric logic and rapidly iterated visualization / scene-analysis tooling.

The project direction emphasizes:

- local-first execution
- transparent geometry
- shared metrics across navigation-derived and mesh-derived workflows
- increasingly integrated 3D scene analysis

## License

This repository currently does **not** include a final `LICENSE` file. Add the intended open-source license before formal public distribution or reuse terms are asserted.

## Citation

If Surgiplot is used in academic work, cite:

- the Surgiplot repository
- and the associated methodological publications for the operative metrics when available or formally linked

If you want the GitHub front page to look more like a formal software paper landing page, the next natural step would be to add:

- a project logo or banner
- example screenshots / figure panels
- a concise “Quick Start” section near the top
- a “Methods and Validation” section linked to manuscripts or supplements
