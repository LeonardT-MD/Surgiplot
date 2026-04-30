<p align="center">
  <img src="docs/assets/SURGIPLOT_BANNER.png" alt="Surgiplot banner" width="100%">
</p>

<p align="center">
  <img src="docs/assets/LOGO_SURGIPLOT.png" alt="Surgiplot logo" width="180">
</p>

# Surgiplot

Surgiplot is a local software environment for quantitative neuroanatomical and skull base analysis. It was developed to support reproducible geometric study of operative corridors, approach angles, exposure metrics, landmark-defined surfaces, and point-derived volumetric constructs from three-dimensional anatomical data.

The software is intended for workflows in which the primary observations are represented by 3D coordinates, whether derived from neuronavigation, structured tabular annotation, manual landmark selection, or an imported three-dimensional scene. These coordinates are subsequently used to construct explicit geometric models from which quantitative operative descriptors can be derived, visualized, and audited.

Surgiplot currently supports the following metrics:

- **VoA**: Visuooperative Angle
- **VOM**: Volume of Operative Maneuverability
- **sVOM**: standardized VOM
- **AoA**: Angle of Attack
- **SF**: Surgical Freedom
- **AoE**: Angle of Exposure
- **AE**: Area of Exposure
- **3D Distance**
- **Volume** from sparse 3D boundary points

Surgiplot can be used either as:

1. a **desktop GUI** for dataset curation, three-dimensional scene exploration, landmark acquisition, and interactive visualization
2. a **local Python toolkit / CLI** for scripted analysis, reproducibility studies, and methodological audit

## Overview

Quantitative anatomical analysis in neurosurgical research frequently relies on ad hoc spreadsheets, case-specific notebooks, or proprietary navigation exports that are difficult to harmonize across investigators, laboratories, and publications. Surgiplot was conceived to provide a unified computational framework in which:

- anatomically meaningful landmarks are stored in a common data structure
- geometric constructs are explicitly defined rather than implicitly assumed
- intermediate computational objects remain inspectable
- the same metric logic can be applied across heterogeneous acquisition sources

The central premise of the software is that, once a set of reliable 3D landmarks has been defined in a shared coordinate frame, multiple measurements of operative access, exposure, directionality, and spatial extent can be derived from the same annotated dataset in a reproducible manner.

## Core data model

Irrespective of acquisition source, Surgiplot normalizes imported or collected coordinates into a shared editable dataset. Each dataset is composed of:

- canonical coordinate identifiers such as `point_1`, `point_2`, ...
- an editable label field for semantic aliases such as `apc`, `cranial`, `medial`, or other anatomy-specific terms

Metric definitions may therefore refer to:

- canonical point identifiers
- semantic aliases
- compact numeric shorthand where appropriate in the GUI

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

In the GUI, numeric shorthand such as `1-5` is expanded to `1, 2, 3, 4, 5` in multi-point fields.

## Supported data sources

Surgiplot is designed to operate locally on heterogeneous sources of 3D anatomical information, including:

- neuronavigation exports
- structured point tables (`CSV`, `TSV`, `XLSX`)
- manually entered coordinates
- external 3D meshes or point clouds imported into the 3D Scene Workspace

All imported or collected coordinates are ultimately mapped into the same dataset abstraction. Accordingly, downstream metric computation is independent of whether the landmark set originated from a navigation system, a curated spreadsheet, or a mesh-based three-dimensional scene.

## Principal workflows

### 1. Navigation or tabular annotation workflow

1. Import a navigation export or point table
2. Inspect and curate the dataset
3. Edit labels or add supplementary points
4. Proceed to analysis
5. Compute one or more quantitative metrics
6. Visualize the resulting geometry in 3D

### 2. Three-dimensional scene workflow

1. Import a mesh or point cloud into the **3D Scene Workspace**
2. Visually inspect the anatomy in the three-dimensional viewer
3. Rescale the scene by selecting two points and entering the known real-world distance
4. Collect landmarks directly on the scene
5. Commit those landmarks into the Surgiplot dataset
6. Move to the analysis workspace and compute the same metrics used for navigation-derived points

This workflow is particularly relevant when a previously reconstructed anatomical model is to be treated as a landmark source for the same downstream measurements used in navigation-derived studies.

## Metric definitions

### VoA: Visuooperative Angle

**Definition**

VoA describes the alignment between the operative trajectory and the target construct. In the present implementation it is derived from the centroid-to-centroid trajectory between entry and target geometries relative to the target-plane normal.

**Methodological rationale**

A purely volumetric corridor description is incomplete if directional alignment is ignored. Two corridors may exhibit similar cross-sectional dimensions while differing substantially in their approach relationship to the target.

**Typical applications**

- comparison of approaches to a common target
- quantification of corridor orientation
- paired reporting with VOM and sVOM

### VOM: Volume of Operative Maneuverability

**Definition**

VOM models the operative corridor volume between entry and target regions. In the current methodology, irregular entry and target polygons are:

1. projected to PCA-derived best-fit planes
2. converted into area-matched reference ellipses
3. reconstructed in 3D
4. lofted into a geometric corridor solid

The primary output is the enclosed volume of the resulting three-dimensional corridor solid.

**Methodological rationale**

This yields a corridor-oriented volumetric descriptor that is more anatomically and procedurally meaningful than generic enclosing volumes or purely linear measurements.

**Typical applications**

- corridor comparison across approaches
- target-specific access-volume analysis
- methodological study of operative maneuverability

### sVOM: standardized VOM

**Definition**

sVOM represents a standardized distal segment of the corridor rather than its full entry-to-target extent.

**Methodological rationale**

A standardized distal segment may provide a more robust substrate for cross-case comparison of near-target working space than a full-length corridor whose proximal extent varies by case definition or surgical approach.

**Typical applications**

- near-target corridor comparison
- standard-distance methodological studies
- analyses focused on distal rather than full-length corridor volume

### AoA: Angle of Attack

**Definition**

AoA is computed in vertical and horizontal components:

- **vertical AoA** from the cranial-caudal-pivot triangle
- **horizontal AoA** from the medial-lateral-pivot triangle

Optional standardization projects each entry point along its pivot-directed ray to a fixed distance before recalculation of standardized geometry.

**Methodological rationale**

Approach angles quantify directionality independently from pure area or volume. Vertical and horizontal decomposition improves anatomical and operative interpretability.

**Typical applications**

- comparison of working direction across approaches
- quantification of access angle at a target
- combined directional and areal corridor analysis

### SF: Surgical Freedom

**Definition**

SF is represented as the quadrilateral surface defined by the ordered entry points:

- cranial
- lateral
- caudal
- medial

The software also supports a standardized version obtained by projecting the entry points to a fixed pivot-centered distance.

**Methodological rationale**

SF provides a planarized summary of the proximal maneuvering envelope and complements the directional information given by the AoA triangles and the distal information captured by volumetric corridor metrics.

**Typical applications**

- comparison of entry working area
- standardized access-window analysis
- reporting of proximal maneuverability independently of corridor length

### AoE: Angle of Exposure

**Definition**

AoE describes a three-point angular construct and is visualized in the GUI as a 360 degree circular plot with the measured angle highlighted.

**Methodological rationale**

This metric is useful when the relevant quantity is angular exposure about a pivot or viewpoint rather than three-dimensional corridor extent.

**Typical applications**

- angular target exposure analysis
- comparison of visible or exposed sectors between approaches

### AE: Area of Exposure

**Definition**

AE is computed from an ordered 3D polygon. The polygon is:

1. projected onto a PCA-derived best-fit plane
2. converted into 2D plane coordinates
3. measured with the shoelace formula

The GUI may also display the corresponding coplanar fitted surface reconstructed in native three-dimensional space.

**Methodological rationale**

Anatomical point sets are often nearly, but not perfectly, coplanar. PCA-based plane fitting provides a reproducible formalization of an idealized exposure surface.

**Typical applications**

- measurement of exposure windows from ordered landmarks
- comparison of planarized surface extents
- methodological studies of exposure area

### 3D Distance

**Definition**

This metric computes Euclidean distance between two 3D landmarks.

**Methodological rationale**

Simple distances remain fundamental for anatomical description, validation, scaling checks, and target-entry separation.

**Typical applications**

- linear anatomical measurements
- scale confirmation
- target-entry separation

### Volume

**Definition**

This metric estimates the volume of a closed solid reconstructed from a sparse boundary point cloud. The implementation attempts a local alpha-shape-like reconstruction and falls back to a convex hull only when necessary.

**Methodological rationale**

In some studies, the available observation is not a dense segmentation but a sparse set of perimeter or boundary samples. This metric provides a principled volumetric reconstruction from those sparse coordinates.

**Typical applications**

- point-based surrogate segmentation
- cavity or compartment volume estimation from sparse samples
- exploratory volumetric reconstruction in anatomical research

## Methodological principles

Surgiplot is organized around several methodological principles:

- **shared coordinate frame first**: all metrics assume that the landmarks exist in a coherent 3D frame
- **inspectable geometry**: intermediate constructs remain available for plotting and debugging
- **explicit geometric models**: corridor and surface measurements are tied to defined geometric constructs rather than opaque black-box outputs
- **local execution**: the software is designed to run on the user’s machine without requiring a cloud workflow

Representative examples include:

- `AE` is not computed as the area of a warped mesh, but as the area of an ordered polygon projected onto its PCA-derived best-fit plane
- `VOM` is not inferred from a schematic graphic, but from an explicit lofted corridor model
- `AoA / SF` standardization does not globally scale the case; rather, it projects each entry point along its pivot ray to a selected fixed distance

## Three-dimensional visualization

Surgiplot separates:

- the **background anatomical scene** (imported mesh or point cloud)
- the **metric geometry** (points, polygons, ellipses, corridor shells, triangles, surfaces, arrows)

This distinction is useful in practice because investigators may need to:

- reduce the visual dominance of the anatomical scene
- disable source texture and inspect mesh shape alone
- strengthen metric overlays for figure preparation or methodological review

The analysis viewer therefore provides separate controls for:

- **3D model rendering**
  - opacity
  - point size
  - mesh edges
  - source colors / texture on-off
- **metric overlay rendering**
  - opacity
  - emphasis

Metric scenes are rendered directly within the same three-dimensional coordinate system as the imported anatomy, so the overlay remains geometrically concordant with the underlying model rather than purely illustrative.

## Installation

### Recommended local setup

Surgiplot is intended for local execution. A conda or Miniforge environment is recommended for a stable scientific Python stack.

```bash
git clone https://github.com/LeonardT-MD/Surgiplot.git
cd Surgiplot

conda create -n surgiplot python=3.11 pip -y
conda activate surgiplot

pip install -e .
```

### Optional GPU-backed 3D scene backend

For improved mesh and point-cloud visualization, including PyVista-based three-dimensional analysis scenes:

```bash
pip install -e ".[scene3d]"
```

This installs:

- `pyvista`
- `pyvistaqt`
- `vtk`

If `.[scene3d]` is not installed, Surgiplot falls back to the Matplotlib-based viewer where applicable.

### Local installer script

```bash
bash surgiplot/scripts/install_conda.sh
```

## Running Surgiplot

### GUI mode

```bash
surgiplot_gui
```

The graphical interface supports:

- dataset import and curation
- direct label editing
- 3D scene import
- local scene rescaling from two selected 3D points
- landmark collection into the shared dataset
- metric configuration and analysis
- 3D overlay plotting
- result inspection and debug review

### CLI / library mode

```bash
surgiplot
```

Surgiplot may also be used from scripts or notebooks when explicit control of data loading, metric definition, and downstream reproducibility is preferred.

## Representative use cases

### Navigation-derived operative corridor study

Landmarks are exported from a neuronavigation platform, assigned anatomical aliases, and used to compute AoA, SF, VoA, VOM, and sVOM for comparative analysis across approaches or specimens.

### Mesh-based anatomical landmark collection

An external anatomical mesh is imported, rescaled from a known reference distance, annotated directly in three-dimensional space, and analyzed using the same metric framework employed for navigation-derived coordinates.

### Point-based boundary reconstruction

A sparse set of boundary points is sampled around an anatomical compartment and used to estimate an enclosed volume from an inferred surface reconstruction.

### Reproducible methods development

A landmarking scheme is developed in the GUI, exported as a curated dataset, and then re-executed through scripted analysis for validation, methods development, or manuscript preparation.

## Reproducibility and auditability

Surgiplot is intended to facilitate inspection, re-execution, and methodological audit of quantitative workflows. The platform supports:

- explicit named landmarks
- editable labels and aliases
- debug payloads containing intermediate geometric data
- 3D visualization of the measured constructs
- export of datasets and result payloads

The purpose is not only to generate a numerical output, but also to preserve the geometric pathway by which that output was obtained.

## Scope and limitations

Surgiplot is intended for quantitative geometric analysis. It is not a diagnostic system and should not be interpreted as a substitute for clinical judgment.

Important practical limitations include:

- results depend on the quality and consistency of the selected landmarks
- polygon-based metrics depend on meaningful point ordering
- point-derived volume reconstruction remains a model of a likely solid rather than a true image segmentation
- imported meshes and point clouds are only as metrically valid as their own reconstruction pipeline and scaling

These limitations are inherent to landmark-based anatomical measurement and should be explicitly acknowledged in study design, interpretation, and reporting.

## Repository purpose

This repository serves as the software release and methodological implementation site for Surgiplot. Its purpose is to provide:

- a reference implementation of the operative metrics supported by the platform
- an inspectable local toolchain for three-dimensional anatomical measurement
- a shared computational environment for navigation-derived and mesh-derived quantitative workflows

## License

This repository currently does **not** include a final `LICENSE` file. The intended open-source license should be added before formal public distribution or reuse terms are asserted.

## Citation

If Surgiplot is used in academic work, cite both:

- the Surgiplot repository
- the associated methodological publications for the operative metrics when available or formally linked
