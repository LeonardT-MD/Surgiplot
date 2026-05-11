# Supplementary Methods

## Technical description of the Surgiplot framework

### 1. Software architecture

`Surgiplot` is a coordinate-based morphometric framework for neurosurgical anatomical analysis. The project consists of a shared computational core and multiple user-facing front ends. The same metric kernels are exposed through:

- a Python library
- a command-line interface
- a desktop GUI
- a 3D Slicer extension (`SurgiplotToolkit`)

The purpose of this design is to preserve identical metric definitions across interactive and scripted workflows.

### 2. Dataset model

All workflows use a common `Dataset` abstraction in which 3D landmarks are stored as named coordinates with optional aliases and metadata. Canonical names (e.g. `point_1`) are retained for internal consistency, whereas aliases (e.g. `pivot`, `cranial`) permit anatomically meaningful calls. Multi-point inputs also support compact range syntax such as `point_1-9`.

### 3. Geometric primitives

#### 3.1 Best-fit plane estimation

Best-fit planes are estimated by singular value decomposition of the centered point cloud. The first two principal directions define the in-plane basis, while the least-variance direction defines the plane normal. This approach replaced an earlier machine-learning dependency in order to keep the geometry core lightweight and portable, including within the Python environment bundled with 3D Slicer.

#### 3.2 Polygon area

After projection into the local best-fit 2D frame, polygon area is computed using the shoelace formula.

#### 3.3 Directional rescaling

Standardized surgical freedom is obtained by projecting each cardinal entry point to a fixed radius from the pivot along the original pivot-to-point direction.

### 4. Metric-specific implementation notes

#### 4.1 Distance

`DISTANCE_3D` is the Euclidean distance between two points. It accepts either names/aliases or direct numeric coordinates.

#### 4.2 Area

`AREA_3D` uses best-fit-plane projection and the shoelace formula to compute exposure area from an ordered landmark contour. The implementation explicitly stores both the original polygon and its fitted coplanar representation for debug and plotting purposes.

#### 4.3 Volume

`VOLUME_3D` reconstructs a sparse closed solid from boundary points. The implementation first attempts an alpha-shape-like tetrahedral selection based on Delaunay tetrahedralization and circumsphere radii. If a stable closed solid cannot be identified, it falls back to the convex hull. Debug outputs include the reconstructed surface triangles.

#### 4.4 AoE

`AOE` is the included angle at point `B` between rays `BA` and `BC`. In interactive Slicer use, the result is displayed as a triangle and curved arc to facilitate immediate interpretation of the angle in the 3D scene.

#### 4.5 AoA / SF

`AOA_SF` uses four ordered entry points and a pivot. The implementation computes:

- vertical angle of attack
- horizontal angle of attack
- native surgical freedom (`SF`) area
- standardized surgical freedom (`standardized SF`) area when a fixed rescaling radius is provided

The debug payload retains ordered labels, point-role order, warnings, pivot, original entry points, standardized entry points, and the corresponding triangle and surface constructs required for scene visualization.

#### 4.6 VOM / sVOM / VoA

`VOM_VOA` replaces the entry and target polygons with area-matched ellipses constructed on the PCA best-fit plane of each polygon. This preserves the measured polygon area while imposing an analytically stable cross-sectional geometry. The corridor volume is then modeled as the shell between the entry and target elliptical sections. The standardized distal component (`sVOM`) is defined by a cut ellipse at a protocol-defined distance from the target.

The implementation also derives:

- entry polygon area
- target polygon area
- entry ellipse area
- target ellipse area
- distal cut ellipse area
- target distance

`VoA` is computed from the corridor vector relative to the target plane by first reducing the direction-to-normal relation to the acute case.

### 5. Desktop GUI and 3D scene workflow

The desktop GUI supports both normalized tabular datasets and direct point collection on imported 3D scenes. In the scene workflow, the user can calibrate scale, collect landmarks, commit them into the dataset, and compute metrics subsequently. The GUI was explicitly designed so that the rendered scene is a coordinate acquisition interface and not merely a visualization target.

### 6. 3D Slicer integration

The `SurgiplotToolkit` extension brings the same metric families into a multimodal Slicer scene. It supports:

- model, segmentation, and image-based context selection
- direct landmark collection in 3D or slice views
- a persistent shared landmark dataset
- metric-specific workspaces
- plotting of derived geometry back into the Slicer scene
- export of case-level and point-level datasets
- export of metric-specific 3D scene products

Raw surface models can be rescaled by two-point calibration if they do not already encode meaningful physical spacing. In model-based workflows, picked landmarks are snapped to the visible reference model surface to maintain scene coherence.

### 7. Output datasets

Two complementary export datasets are produced:

1. `Case results dataset`
   one row per subject, with target-prefixed columns representing scalar result outputs

2. `Points reproducibility dataset`
   one row per used landmark, recording subject, target, metric family, point group, point label, and coordinates

These datasets are intended to preserve both statistical usability and landmark-level traceability.

### 8. Caveats and intended use

The framework is optimized for coordinate-defined morphometric studies rather than for full image segmentation replacement. The main sources of uncertainty remain landmark definition, polygon ordering, and the interpretation of simplified analytic constructs such as area-matched ellipses or sparse boundary volumes. These limitations should be stated explicitly whenever the methods are reported.
