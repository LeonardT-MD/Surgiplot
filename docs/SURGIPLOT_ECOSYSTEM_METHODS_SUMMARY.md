# Surgiplot Ecosystem Methods Summary

## Computational rationale, mathematical methods, workflows, and caveats across CLI, Python, GUI, and 3D Slicer

This document is intended as a technical synthesis of the entire Surgiplot project. It is not a user quickstart. It is a structured methods note that can later be mined for manuscript writing, supplementary methods, internal protocols, and reviewer responses.

The emphasis here is on four questions:

1. what Surgiplot is trying to formalize
2. how each method is actually computed
3. how the same computational core is exposed across the Python library, CLI, GUI, and Slicer extension
4. what caveats should be reported or kept in mind during scientific use

## 1. General project architecture

Surgiplot is built around the idea that anatomical and operative measurements can be represented as explicit operations on 3D coordinates rather than as application-specific state hidden inside a viewer. The project therefore separates:

- `coordinate acquisition`
- `dataset normalization`
- `metric computation`
- `visualization and reporting`

The same computation core is used by four front ends:

- the Python API
- the CLI
- the desktop GUI
- the `SurgiplotToolkit` extension for 3D Slicer

This is methodologically important. If two different user interfaces yielded different metric definitions, the project would not support reproducible analysis. The software has therefore been developed so that the front ends differ in ergonomics, not in the numerical implementation of the metric kernels.

## 2. Data model

### 2.1 Dataset abstraction

The common container is the `Dataset` object in [dataset.py](/Users/leonardo/Projects/Surgiplot/surgiplot/core/dataset.py).

It stores:

- canonical point names
- 3D coordinates
- aliases and anatomical labels
- metadata

This allows one to distinguish between:

- a stable internal identifier such as `point_17`
- a semantic alias such as `pivot`
- a role-based label such as `entry_4`

This distinction is crucial in repeated anatomical work. The canonical point protects internal consistency; the alias preserves interpretability.

### 2.2 Point resolution

Whenever a metric is called, Surgiplot resolves user input to coordinates. Depending on context, those inputs may be:

- canonical names
- aliases
- explicit coordinate arrays
- compact ranges such as `point_1-9`

The range expansion feature is especially relevant for multi-point metrics such as area, volume, and corridor analyses. Its scientific significance is modest, but its practical significance is large: it reduces clerical error in repeated scripts and batch calls.

## 3. Mathematical primitives

### 3.1 Euclidean distance

The simplest primitive is the Euclidean norm:

\[
\|x\|_2 = \sqrt{x_1^2 + x_2^2 + x_3^2}
\]

It is used directly in `Distance`, in axis lengths, in radius projection, and in target-distance calculations.

### 3.2 SVD-based best-fit plane

Area-related constructs are handled through a best-fit plane estimated by singular value decomposition in [pca_plane.py](/Users/leonardo/Projects/Surgiplot/surgiplot/core/geometry/pca_plane.py).

Given a point cloud \(P_i\):

1. compute the centroid \(c\)
2. center the cloud \(X_i = P_i - c\)
3. perform SVD on \(X\)
4. use the right singular vectors as orthogonal principal directions
5. define the plane normal as the smallest-variance axis

This replaced an earlier `scikit-learn` dependency so that the geometry core would remain importable in environments such as 3D Slicer without requiring a heavy external ML stack.

### 3.3 2D projection and shoelace area

Once a local plane basis is known, points are projected into 2D coordinates in that basis. Polygonal area is then computed with the shoelace formula:

\[
A = \frac{1}{2}\left|\sum_{i=1}^{n} x_i y_{i+1} - y_i x_{i+1}\right|
\]

This underlies:

- `AREA_3D`
- the surgical freedom area in `AOA_SF`
- polygon area components in `VOM_VOA`

### 3.4 Directional rescaling

Standardized SF uses a radial projection from the pivot. A point \(P\) is projected to radius \(r\) from pivot \(T\):

\[
P^{*} = T + r \frac{P - T}{\|P - T\|}
\]

This is an explicit geometric reparameterization rather than an empirical normalization.

## 4. Metric families

### 4.1 DISTANCE_3D

Implemented in [distance_3d.py](/Users/leonardo/Projects/Surgiplot/surgiplot/metrics/distance_3d.py).

Inputs:

- point `A`
- point `B`

Output:

\[
d = \|B-A\|_2
\]

Caveat:

- Scientifically trivial but sensitive to point definition and to whether the coordinates are already in millimeters.

### 4.2 AREA_3D

Implemented in [area_3d.py](/Users/leonardo/Projects/Surgiplot/surgiplot/metrics/area_3d.py).

Inputs:

- a polygon with at least 3 points

Method:

1. best-fit plane from SVD
2. 2D projection
3. shoelace area

Interpretation:

- planarized exposure area of an anatomical contour

Caveat:

- the result is meaningful only if the contour is reasonably represented by a best-fit plane
- polygon order matters

### 4.3 VOLUME_3D

Implemented in [volume_3d.py](/Users/leonardo/Projects/Surgiplot/surgiplot/metrics/volume_3d.py).

Inputs:

- at least four non-coplanar points describing a sparse closed boundary

Method:

1. Delaunay tetrahedralization
2. automatic alpha-shape-like selection using circumsphere radii
3. if stable reconstruction fails, fallback to convex hull

Outputs:

- volume estimate
- debug reconstruction geometry

Caveat:

- the method is suitable for sparse morphometric approximation
- it is not a replacement for segmentation-derived ground-truth volume

### 4.4 AOE

Implemented in [aoe.py](/Users/leonardo/Projects/Surgiplot/surgiplot/metrics/aoe.py).

Inputs:

- three points `A`, `B`, `C`

Method:

\[
u = A-B,\qquad v=C-B
\]

\[
\mathrm{AoE} = \cos^{-1}\left(\frac{u\cdot v}{\|u\|\|v\|}\right)
\]

Interpretation:

- included angle at `B`

Caveat:

- this is a direct geometric angle, not a broader solid-angle construct

### 4.5 AOA_SF

Implemented in [aoa_sf.py](/Users/leonardo/Projects/Surgiplot/surgiplot/metrics/aoa_sf.py).

Inputs:

- four ordered cardinal entry points
- one pivot

The metric family yields:

- vertical AoA
- horizontal AoA
- native surgical freedom (`SF`) area
- standardized surgical freedom (`standardized SF`) area when requested

Interpretation:

- `AoA` quantifies angular opening toward the pivot along two anatomically meaningful planes
- `SF` quantifies the area of the entry quadrilateral
- `standardized SF` quantifies the same construct after radial normalization to a fixed pivot distance

Caveats:

- cardinal assignment is anatomically meaningful and cannot be treated as a mere sorting problem
- comparison across cases requires a declared standardization radius when standardized SF is used

### 4.6 VOM_VOA

Implemented in [vom_voa.py](/Users/leonardo/Projects/Surgiplot/surgiplot/metrics/vom_voa.py).

Inputs:

- entry polygon
- target polygon
- standard distance for the distal cut

Outputs:

- `VoA`
- `VOM`
- `sVOM`
- polygon and ellipse area descriptors
- target distance

Method summary:

1. fit best-fit planes to entry and target polygons
2. compute polygon areas
3. fit area-matched ellipses preserving PCA-derived anisotropy
4. connect the ellipses to form a frustum-like corridor shell
5. compute full corridor volume
6. define a distal cut ellipse at standard distance for `sVOM`
7. compute the distal truncated volume
8. compute `VoA` from corridor direction relative to the target plane

Caveats:

- the ellipse replacement is deliberate and should be reported as such
- the method expresses corridor geometry in a stabilized analytic form, not as raw surface morphology

## 5. Python library layer

The Python API is the cleanest expression of the computational core. It supports:

- dataset loading
- alias assignment
- direct metric calls
- dataset-bound convenience methods

Typical usage now follows one of two styles.

### 5.1 Module-level calls

```python
import surgiplot as sp

ds = sp.load("case.csv", source="database")
result = sp.VOM(data=ds, entry="point_1-4", target="point_10-13", stand_dist=10.0)
```

### 5.2 Dataset-bound calls

```python
import surgiplot as sp

ds = sp.load("case.csv", source="database")
result = ds.VOM(entry="point_1-4", target="point_10-13", stand_dist=10.0)
```

The second style is particularly suitable for repeated analyses in notebooks or batch pipelines because it keeps the dataset explicit while reducing syntactic overhead.

## 6. CLI layer

The CLI exists for scripting, shell-based workflows, and study automation. It is not a reduced toy interface. It is a serious batch front end.

Important features include:

- direct loading of normalized datasets
- grouped outputs for `VOM` and `AOA`
- range-based point expansion
- dataset relabeling workflows

The crucial design change in the current generation of the CLI is that the grouped metric families return grouped outputs together rather than forcing users to reconstruct families from isolated scalar commands.

For example:

- `vom` returns `VOM`, `sVOM`, and `VoA`
- `aoa` returns `AoA(v)`, `AoA(h)`, and `SF`

This matches the scientific logic of the metrics better than atomized commands would.

## 7. Desktop GUI layer

The desktop GUI serves two different user populations:

- users working from tabular or navigation-derived point datasets
- users working from rendered 3D scenes and direct point picking

### 7.1 Tabular / normalized dataset use

The GUI allows point inspection, relabeling, metric computation, and visualization without requiring scripting knowledge.

### 7.2 3D scene workflow

The 3D scene workflow was developed because purely tabular landmark management is insufficient when the anatomical meaning of a point is inseparable from the rendered scene from which it is selected.

The core steps are:

1. load the scene
2. calibrate scale when needed
3. pick points on the rendered model or imaging representation
4. store them into the same dataset abstraction
5. compute metrics afterward

One important design result of the GUI work was the decision to preserve both:

- a shared dataset of landmarks
- metric-specific export or reuse of subsets

That design later influenced the Slicer extension directly.

## 8. 3D Slicer layer

`SurgiplotToolkit` extends the same logic into a Slicer-native environment. It is not merely another GUI wrapper. Its distinct value lies in multimodal scene context.

The extension brings together:

- models
- segmentations
- CT and MRI slice views
- direct markups placement
- metric-specific 3D overlays
- exportable case results and reproducibility tables

The rationale is straightforward: for many skull-base studies, the interpretation of a landmark is inseparable from the 3D scene and from the slice planes in which it was chosen.

## 9. Output, reporting, and reproducibility

The project now produces two parallel output types:

### 9.1 Scalar result datasets

These are study-ready tables in which:

- each patient occupies one row
- each target-specific measurement family contributes additional columns

This is suitable for direct statistical work or later manuscript table preparation.

### 9.2 Point reproducibility datasets

These store the actual points used, their labels, their grouping, and the associated target and metric family. This is essential if a reviewer or collaborator later asks which points produced a given scalar result.

## 10. Caveats by layer

### 10.1 Across all layers

- coordinate quality dominates result quality
- label semantics matter
- geometric simplifications should be described explicitly

### 10.2 CLI / Python

- the library is only as trustworthy as the dataset fed into it
- range-based convenience should not obscure the need to verify point ordering

### 10.3 GUI

- direct picking on rendered scenes is powerful but requires a defensible visual workflow
- scale calibration should be documented in methods if it was applied

### 10.4 3D Slicer

- scene complexity and visibility management can influence usability but should not influence the numerical result
- markups collected in imaging context should still be checked in 3D context when anatomical interpretation is critical

## 11. Why the project is structured this way

The key methodological strength of Surgiplot is not that it computes a large number of quantities. The strength is that the same quantities can be computed consistently across several modes of use:

- scripted Python analysis
- command-line batch execution
- local desktop GUI interaction
- scene-native 3D Slicer analysis

That is what makes the project suitable for research use. It allows:

- interactive pilot analysis
- later scripted repetition
- transparent metric definition
- explicit reproducibility of landmarks and derived geometry

## 12. Points for manuscript drafting

When the manuscript is written, the following points are worth preserving.

1. `Surgiplot is coordinate-native`
   The software operates on explicit 3D coordinates and derived local geometric frames rather than on opaque proprietary measurement states.

2. `The metric families are inspectable`
   Each metric corresponds to a documented geometric procedure, not an undocumented viewer tool.

3. `The same computation core is reused across all interfaces`
   Numerical consistency across Python, CLI, GUI, and 3D Slicer is a major design feature.

4. `Standardized constructs are explicit`
   Standardized SF and sVOM are not vague normalized values; they are defined by declared geometric transformations and distances.

5. `Reproducibility is point-level, not only result-level`
   The project stores not only the scalar outputs but also the points that generated them.

## 13. Practical summary

If one had to summarize the ecosystem in a single sentence suitable for methods framing, it would be this:

Surgiplot is a coordinate-based morphometric framework for neurosurgical anatomical analysis in which landmark acquisition, dataset normalization, geometric computation, visual verification, and reproducible export are preserved as a single coherent pipeline across scripting and interactive environments.
