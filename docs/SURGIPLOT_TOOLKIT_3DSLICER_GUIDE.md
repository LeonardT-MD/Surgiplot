# SurgiplotToolkit for 3D Slicer

## Detailed User and Methods Guide

This document describes the use of `SurgiplotToolkit` within 3D Slicer. It is written as a technical guide for collaborators, reviewers, and academic users who need to understand not only how the module is operated, but also why each metric exists and what each computation represents.

The focus of the extension is not to replace the underlying Surgiplot computation core. Rather, the purpose of the Slicer integration is to place those computations inside a scene-native environment where models, segmentations, CT, MRI, and user-defined landmarks can be handled within the same session. The resulting workflow is intended for studies in skull-base and microsurgical anatomy in which measurement, geometry, and 3D visualization must remain coupled.

## 1. Scope of the extension

`SurgiplotToolkit` provides a Slicer-facing interface to the same metric families available in Surgiplot itself:

- `Distance`
- `Area`
- `Volume`
- `AoA / SF`
- `AoE`
- `VOM / sVOM / VoA`

In this guide, the terms are used in a strict sense:

- `Area` denotes polygonal exposure area computed from an ordered landmark contour
- `AoE` denotes `angle of exposure`
- `AoA / SF` denotes the combined `angle of attack / surgical freedom` family
- `VOM / sVOM / VoA` denotes the corridor family composed of volume of maneuverability, standardized volume of maneuverability, and volume of attack

The extension is designed around four assumptions:

1. the user may work from different anatomical carriers, including surface models, segmentations, and image volumes
2. points may be collected prospectively while a metric is being assembled, or retrospectively from a reusable landmark dataset
3. the measurement workflow must remain visible in 3D, not only as a scalar report
4. derived geometry should be exportable and study results should be serializable into tabular outputs suitable for later analysis

## 2. Conceptual structure of the module

The module is organized into three functional layers.

### 2.1 Scene context and landmark dataset

This layer defines the anatomical frame in which points are being interpreted.

It includes:

- `Analysis context`
  the main scene object under analysis, typically a model, but potentially also a segmentation or an image-derived context

- `CT / MRI slice background`
  an optional scalar volume used to drive slice-based point placement

- `Landmark dataset`
  a persistent shared markups node in which collected landmarks are accumulated, relabeled, reordered, and reused across metrics

- `Scale node`
  an auxiliary two-point calibration object used only when a surface model requires unit correction

### 2.2 Metric workspaces

Each metric has its own collapsible workspace. That was an intentional design decision. A generic measurement panel becomes hard to read very quickly once entry polygons, pivot points, single-point angular pivots, and sparse volume clouds are mixed together. For this reason, each workspace exposes only the landmarks and actions relevant to that metric.

### 2.3 Result layer

The result layer has three roles:

- show structured numerical output
- control scene visibility and opacity
- accumulate case-level and point-level datasets for downstream export

## 3. Rationale for the shared landmark dataset

The landmark dataset is the central working memory of the extension.

The dataset exists for three reasons.

First, in anatomical work, the same point frequently participates in more than one computation. A landmark placed as a cranial limit for `AoA / SF` may later be reused as part of an exposure polygon or to define the boundary of a corridor entry. Requiring the user to recollect it for each metric would degrade both efficiency and reproducibility.

Second, metric selection often occurs after point collection, not before it. During exploratory work, one may collect a dense set of landmarks around an approach, then later decide which subset should define the formal measurement.

Third, the dataset creates an auditable trace. The measurements are not merely scalar outcomes; they are attached to explicit labeled points that can be reviewed, exported, and checked by another observer.

## 4. Analysis context and background volume

### 4.1 Analysis context

The `Analysis context` selector should be understood as the anatomical object against which the extension will display derived geometry. In most model-based workflows this is the imported 3D model. In segmentation- or image-based work, it can serve as the nearest equivalent context for organizing results in the Slicer scene.

The context is used for:

- snapping model-based picks to the visible surface
- linking result nodes to the correct subject-hierarchy folder
- applying reference-model opacity
- optional model-only rescaling

### 4.2 Background volume

The `CT / MRI slice background` selector does not alter measurements. It only determines which scalar volume is shown in slice viewers while the user places points from imaging data. In image-native workflows, this allows precise placement without changing the underlying geometry of the measurement routines.

## 5. Scale calibration

### 5.1 Why calibration is needed

CT and MRI volumes generally arrive with meaningful physical spacing already encoded. A raw surface model does not always do so. In that situation, coordinate differences can still be computed, but their unit may not correspond to millimeters. The scale tool exists to correct that problem when a known anatomical or physical reference length is available.

### 5.2 How the calibration is defined

The user places two scale points on the model surface and provides the known real-world distance between them. Let:

\[
d_{model}
\]

be the Euclidean distance between the two points in the model’s current coordinate frame, and let

\[
d_{real}
\]

be the known physical distance, expressed in centimeters in the user interface and converted internally to millimeters.

The scale factor is:

\[
s = \frac{d_{real}}{d_{model}}
\]

The reference model is then rescaled uniformly by \(s\). The calibration points are updated with the model so that their anatomical locations remain coherent after the transformation.

### 5.3 Practical consequence

Once scaling has been applied successfully, subsequent picks and metric computations are interpreted directly in the corrected world coordinate system. There should be no need to apply a secondary scalar multiplier during later measurement steps.

## 6. Distance

### 6.1 Measurement concept

`Distance` is the simplest metric in the toolkit. It measures the straight-line Euclidean separation between two 3D points:

\[
d = \|B - A\|_2
\]

This is appropriate for point-to-point anatomical distances when the meaning of the measurement does not require projection onto a surface or plane.

### 6.2 Workflow in Slicer

1. Open the `Distance` workspace.
2. Either:
   - select two existing landmarks from the shared dataset and assign them to `A` and `B`, or
   - place new points directly through `Pick A` and `Pick B`.
3. Review the assigned nodes.
4. Click `Compute Distance`.

### 6.3 3D output

The extension plots:

- a line segment between `A` and `B`
- labeled endpoint landmarks
- a numerical result payload in the panel

## 7. Area

### 7.1 Measurement concept

`Area` is a polygonal exposure-area metric. The user supplies an ordered set of boundary landmarks. These points are not assumed to be perfectly coplanar. Instead, Surgiplot derives a best-fit plane from the point cloud by singular value decomposition and projects the polygon into that local 2D basis.

If \(P_i \in \mathbb{R}^3\) are the polygon vertices, the procedure is:

1. compute the centroid \(c\)
2. center the coordinates \(X_i = P_i - c\)
3. compute the best-fit plane by SVD of the centered cloud
4. take the first two principal directions as the in-plane basis
5. project the vertices into 2D
6. compute the area with the shoelace formula

The resulting area is therefore the area of the polygon expressed on its PCA best-fit plane, not the area of a triangulated 3D mesh patch.

### 7.2 Workflow in Slicer

1. Open `Area`.
2. Either:
   - collect points sequentially with `Start Polygon Picking`, or
   - select pre-existing dataset points in the intended polygon order and assign them to `Polygon`.
3. Stop picking when the polygon is complete.
4. Compute the metric.

### 7.3 Interpretation

The validity of the result depends on the anatomical rationale of the chosen polygon and on the ordering of its points. Self-crossing sequences should be avoided during data collection. The toolkit attempts to keep the display rational, but the scientific responsibility still lies with the observer to choose the contour meaningfully.

### 7.4 3D output

The extension plots:

- the original polygon
- the fitted coplanar surface
- the input landmarks

## 8. Volume

### 8.1 Measurement concept

`Volume` estimates the enclosed volume of a sparse 3D boundary point set. It is intended for exploratory morphometric reconstruction from manually collected landmarks rather than for formal segmentation replacement.

The implementation attempts:

1. Delaunay tetrahedralization
2. alpha-shape-like tetrahedral selection through circumsphere-radius filtering
3. automatic selection of a stable alpha candidate
4. if no stable alpha reconstruction is found, fallback to the convex hull

The result is therefore an estimate of enclosed volume under an explicit sparse-boundary model.

### 8.2 Workflow in Slicer

1. Open `Volume`.
2. Collect points with `Start Volume Picking` or assign a selected set from the shared dataset to `Volume`.
3. Compute the metric.

### 8.3 Caveat

The volume result depends strongly on sampling density and topological plausibility. Sparse points work best when they describe the surface of a single closed solid. If points are too few, nearly coplanar, or distributed across multiple disconnected regions, the fallback behavior may dominate the result.

### 8.4 3D output

The extension plots:

- the reconstructed surface
- the input landmarks

## 9. AoE

### 9.1 Measurement concept

`AoE` is the included angle at point `B` between the rays \(BA\) and \(BC\). If:

\[
u = A - B, \qquad v = C - B
\]

then

\[
\theta = \cos^{-1}\left(\frac{u \cdot v}{\|u\|\|v\|}\right)
\]

The result is reported in degrees.

### 9.2 Workflow in Slicer

1. Open `AoE`.
2. Define points `A`, `B`, and `C` either by dataset assignment or by direct picking.
3. Compute the metric.

### 9.3 3D output

The extension plots:

- the triangle \(ABC\)
- a curved arc at the pivot expressing the measured angle more explicitly than the triangle alone
- the three landmarks

This representation is preferable in practice because it separates the angular construct from the supporting triangle.

## 10. AoA / SF

### 10.1 Measurement concept

The `AoA / SF` family groups two related but distinct constructs:

- `Angle of Attack`
- `Surgical Freedom`

The entry plane is defined by four cardinal landmarks:

- cranial
- caudal
- medial
- lateral

and a single target pivot.

### 10.2 Angle of Attack

The implementation computes vertical and horizontal angular opening using triangles built from the pivot and the relevant cardinal pairs. In effect, the procedure evaluates the opening angles seen from the target pivot toward the superior-inferior and medial-lateral extents of the entry.

The vertical and horizontal angles are therefore not generic solid angles. They are plane-specific angular descriptors of corridor width relative to the pivot.

### 10.3 Surgical freedom

The same four entry points define a quadrilateral. As for polygonal area elsewhere in Surgiplot, the quadrilateral is projected into a best-fit plane and its area is measured in square millimeters. This gives the native `SF_area_mm2`.

### 10.4 Standardized SF

The standardized surgical freedom construct exists to compare cases at a fixed radius from the pivot. Each cardinal point is projected outward along the line joining the pivot to that point until it lies at the chosen radius:

\[
P_i^{*} = T + r \cdot \frac{P_i - T}{\|P_i - T\|}
\]

where \(T\) is the pivot and \(r\) is the chosen standardization radius. The standardized area is then computed on the rescaled quadrilateral.

### 10.5 Workflow in Slicer

1. Open `AoA / SF`.
2. Collect or assign:
   - cranial
   - caudal
   - medial
   - lateral
   - pivot
3. Confirm that the role assignment is correct.
4. Compute `AoA / SF`.
5. If standardization is required:
   - enable standardized SF
   - set the standardization radius
   - run `Compute Standardized SF`

### 10.6 3D output

The extension plots:

- vertical AoA triangle
- horizontal AoA triangle
- SF quadrilateral
- optionally the standardized SF quadrilateral
- dotted guide lines from the pivot through each cardinal point to the standardized vertices when the standardized computation is requested

### 10.7 Caveat

The scientific validity of the metric depends on correct identification of the four cardinal roles. This is not a purely geometric issue. The anatomical assignment of cranial, caudal, medial, and lateral determines the meaning of the angles and of the quadrilateral itself.

## 11. VOM / sVOM / VoA

### 11.1 Measurement concept

This family models a corridor between two polygonal apertures:

- an `entry` polygon
- a `target` polygon

The polygons are converted into area-matched ellipses on their local best-fit planes. The rationale is to obtain a smooth, parameterizable representation of corridor extent while preserving gross areal scale.

For each polygon:

1. project to the PCA best-fit plane
2. compute polygon area
3. preserve the PCA-derived aspect ratio
4. scale the principal radii uniformly so that the ellipse area matches the polygon area

This yields:

- entry polygon area
- target polygon area
- entry ellipse
- target ellipse

The corridor between the two ellipses defines the `VOM` construct.

### 11.2 VOM

`VOM` is the volume between the entry and target elliptical sections. Computationally, the extension represents this as a frustum-like surface between matched parameter samples on the two ellipses.

### 11.3 sVOM

`sVOM` is the distal standardized volume obtained at a fixed distance from the target along the corridor axis. The extension reports both the standardized cut ellipse and the volume of the distal truncated segment.

### 11.4 VoA

`VoA` is the angle between the corridor direction and the target plane. The implementation derives the target normal, computes the angle between the corridor vector and the target normal, reduces it to the acute case, and then converts it into the plane angle:

\[
\mathrm{VoA} = 90^\circ - \theta_{\text{acute to target normal}}
\]

This prevents the trivial zero-degree behavior that would arise if the normal orientation were treated naively.

### 11.5 Workflow in Slicer

1. Open `VOM / sVOM / VoA`.
2. Collect entry points with `Start Entry Picking` or assign them from the dataset.
3. Collect target points with `Start Target Picking` or assign them from the dataset.
4. Set the standard distance parameter if required by the study protocol.
5. Compute the metric family.

### 11.6 Reported outputs

The extension reports:

- `VoA_deg`
- `VOM_mm3`
- `sVOM_mm3`
- `target_distance_mm`
- `entry_polygon_area_mm2`
- `target_polygon_area_mm2`
- `entry_ellipse_area_mm2`
- `target_ellipse_area_mm2`
- `svom_cut_ellipse_area_mm2`
- `stand_distance_mm`

### 11.7 3D output

The extension plots:

- entry polygon
- target polygon
- entry ellipse
- target ellipse
- standardized cut ellipse
- full corridor shell
- standardized distal shell
- side guide lines between entry and target sections
- centroid axis
- target-direction arrow used to visualize the VoA directionality

### 11.8 Caveat

This is a model-based corridor abstraction. It is appropriate when the anatomical problem is one of corridor geometry rather than of fine-grained free-form segmentation. The fitted ellipses are a deliberate simplification. They make the cross-sections comparable and computationally stable, but they do not claim to reproduce arbitrary local contour irregularity.

## 12. Prospective versus retrospective workflows

The module is designed to support both.

### 12.1 Prospective workflow

The user opens a metric workspace and collects points as the metric is being assembled. This is appropriate when the measurement target is clear from the outset.

### 12.2 Retrospective workflow

The user collects a large landmark dataset first, then later selects subsets for individual metrics. This is appropriate in exploratory anatomical studies, in re-analysis work, and when inter-rater checking is planned.

## 13. Visibility, plotting, and scene persistence

Each metric workspace contains controls to:

- show or hide the computed overlay
- show or hide the input points for that metric
- save the plotted 3D scene for that metric

The extension also provides global control of all point layers so the anatomical model can be inspected either with or without landmarks.

## 14. Case output datasets

The toolkit provides two distinct export datasets.

### 14.1 Case results dataset

This dataset is organized with:

- one row per subject
- one column per target-specific returned field

For example:

- `IAC_VOM_mm3`
- `IAC_VoA_deg`
- `IAC_entry_polygon_area_mm2`

This flat structure is easier to analyze statistically than a long-form relational export when the intended downstream use is spreadsheet-based morphometric comparison.

### 14.2 Points reproducibility dataset

This dataset stores:

- subject ID
- anatomical target
- metric family
- point group
- point label
- X, Y, Z

Its role is not convenience. Its role is reproducibility. It records the actual landmarks used to generate each exported result.

## 15. Recommended use in academic studies

The extension is especially appropriate when:

- the study endpoint is defined by explicit anatomical landmarks
- corridor geometry must be understood in 3D
- both scalar results and visual reconstructions need to be retained
- the same computations will later be repeated in Python or CLI form for batch analysis

In that sense, the toolkit should be viewed as the interactive front end of a larger methodological framework rather than as a closed monolithic application.

## 16. Practical cautions

Several issues should be documented in any serious use of the toolkit.

1. `Landmark definition remains the dominant source of variability`
   The software standardizes the computations. It does not standardize the anatomy. Inter-observer differences in point definition remain critical.

2. `Area metrics depend on point ordering`
   This is especially important for exposure polygons and corridor entries or targets.

3. `Volume is an estimate from sparse points`
   It is not a substitute for full segmentation when segmentation is scientifically required.

4. `Standardized SF and sVOM are protocol-dependent`
   The chosen radius or standard distance must be reported consistently across cases.

5. `Rescaling should be used only when justified`
   Native DICOM- or segmentation-based mm coordinates should not be recalibrated unless there is a documented reason to do so.

## 17. Final perspective

`SurgiplotToolkit` is best understood as a scene-native anatomical measurement environment that inherits its numerical behavior from the Surgiplot core. Its real contribution lies in bringing coordinate-defined morphometry into direct contact with the 3D scene in which landmarks are actually chosen. For skull-base and microsurgical studies, that is often the difference between a numerically correct result and a result that can actually be interpreted anatomically.
