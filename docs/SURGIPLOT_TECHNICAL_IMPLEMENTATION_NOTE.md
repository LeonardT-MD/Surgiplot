# Technical Implementation Note

## Reviewer-oriented summary of the Surgiplot framework

This note is intended for readers who need a concise but precise account of how the Surgiplot system is implemented, what each metric actually computes, and which simplifications are deliberate rather than accidental.

## 1. Core principle

Surgiplot is not organized around a proprietary viewing state. It is organized around explicit 3D coordinates. Every metric in the system ultimately resolves to a set of named points in a shared coordinate frame and applies a documented geometric operation to them. This is the main reason why the same metric definitions can be exposed through a Python API, a CLI, a desktop GUI, and a 3D Slicer extension without changing the numerical output.

## 2. Numerical consistency across interfaces

The front ends differ in interaction style, not in metric implementation. The same underlying functions are called from:

- Python
- CLI
- desktop GUI
- `SurgiplotToolkit` in 3D Slicer

This was an explicit project goal because anatomical research often begins with interactive scene work and later moves into batch analysis. A framework that changed its definitions between those stages would not be scientifically defensible.

## 3. Best-fit plane implementation

Area-related metrics depend on planarization. The framework uses singular value decomposition on the centered point cloud to obtain a PCA-like local basis. The first two axes define the best-fit plane, and the smallest-variance axis defines the normal. The implementation was intentionally rewritten to use NumPy SVD rather than `scikit-learn`, both to reduce dependency weight and to keep the geometry core compatible with 3D Slicer’s Python environment.

## 4. Metric families and implementation choices

### 4.1 Distance

Distance is a direct Euclidean norm. There is no projection, smoothing, or coordinate warping.

### 4.2 Area

Area is computed after best-fit-plane projection by the shoelace formula. In practice this is an exposure-area computation from an ordered landmark contour. The framework stores the original polygon and the fitted coplanar polygon for debug and plotting. This makes the planar approximation inspectable rather than hidden.

### 4.3 Volume

Volume is reconstructed from sparse boundary points. The preferred path is an alpha-shape-like tetrahedral reconstruction; the fallback is the convex hull. This should be read as a sparse geometric estimator, not as a segmentation-derived anatomical gold standard.

### 4.4 AoE

AoE is the included angle at the pivot point between two rays. In the Slicer scene this is rendered both as a triangle and as an arc because the latter is usually the more anatomically legible representation.

### 4.5 AoA / SF

The `AoA / SF` family uses four ordered entry landmarks and one pivot. The implementation distinguishes vertical and horizontal angle-of-attack constructs rather than attempting to reduce the corridor to a single scalar angle. Surgical freedom (`SF`) is represented as a quadrilateral area in the best-fit plane of the entry set.

Standardized SF is not a normalization by arbitrary coefficient. It is a true geometric projection of the entry points onto a sphere centered at the pivot with radius fixed by the user.

### 4.6 VOM / sVOM / VoA

The `VOM / sVOM / VoA` family is the most model-specific part of the framework. It assumes that corridor geometry can be summarized by entry and target polygons whose areas are preserved while their cross-sections are regularized into ellipses. This is an analytical design choice. It sacrifices local contour irregularity in exchange for stable corridor comparison and interpretable longitudinal geometry.

The `VoA` implementation was explicitly corrected so that angle-to-plane is derived from the acute angle to the target normal. Without that correction, opposite normal orientations could generate misleading zero-degree values even in clearly oblique anatomical scenarios.

## 5. Why the shared landmark dataset matters

Both the desktop GUI and the Slicer extension retain a shared landmark dataset independent of any one metric. This is important for three reasons:

1. the same anatomical points can be reused across metrics
2. the user may collect landmarks before deciding which metric subsets to analyze
3. the actual points used in a result remain inspectable and exportable

This design is central to reproducibility. It allows the project to output not only final scalar values but also the landmarks that generated them.

## 6. Slicer-specific implementation notes

The 3D Slicer extension adds scene-native behaviors that are not relevant to the Python or CLI layers:

- model-surface snapping for model-based picks
- optional two-point scale calibration for raw models
- CT/MRI background-volume guidance for slice-based picking
- plotting of debug geometry back into the 3D scene
- saving metric-specific 3D result scenes for later use

These scene features do not redefine the metrics. They exist to make the coordinate acquisition and result verification anatomically meaningful inside Slicer.

## 7. Export structure

Two export classes are maintained:

- a case-level results dataset with one row per subject and target-prefixed scalar columns
- a point-level reproducibility dataset that records the actual landmarks used for each metric call

This distinction is methodologically useful. The first table is suited to statistical analysis; the second is suited to audit and replication.

## 8. Main caveats for review

The most important caveats are not software-specific bugs but methodological conditions:

1. landmark definition is the dominant source of measurement variability
2. point order matters for polygon-based metrics
3. sparse volume reconstruction is an approximation
4. corridor ellipses are a formalization, not a literal preservation of contour irregularity
5. standardization parameters such as radius or distal cut distance must be declared explicitly in study methods

## 9. Bottom-line interpretation

Surgiplot should be understood as a coordinate-native morphometric framework whose strength lies in transparent geometry, interface-independent metric definitions, and explicit retention of the landmarks that generated each reported result. That combination is what makes it suitable for both exploratory anatomical work and formal reproducible analysis.
