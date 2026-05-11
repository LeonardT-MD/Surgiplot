# Methods Draft

## Surgiplot framework for coordinate-based morphometric analysis

Morphometric analyses were performed with `Surgiplot`, a local coordinate-based analysis framework developed for neurosurgical anatomical and skull-base research. The framework was designed to preserve numerical consistency across four operational layers: a Python library, a command-line interface, a desktop graphical user interface, and a 3D Slicer extension (`SurgiplotToolkit`). Across all front ends, metric computations were performed by the same shared computation core.

The general workflow consisted of four stages: coordinate acquisition, dataset normalization, metric computation, and result export. Landmark coordinates were obtained either from tabular/navigation-derived datasets or by direct point placement on rendered three-dimensional scenes. All landmarks were converted into a common `Dataset` representation, in which each point was stored as a 3D coordinate together with a stable canonical identifier and optional semantic aliases. This structure allowed the same coordinate set to be addressed by internal point names or by anatomical labels.

## Coordinate handling and local geometric frames

Several metric families required the analysis of polygons defined by 3D points that were not assumed to be strictly coplanar. In these cases, a best-fit plane was estimated from the point cloud by singular value decomposition (SVD). Briefly, the centroid of the polygon was computed, the point cloud was centered, and the first two principal axes of the centered cloud were used as the in-plane basis. The least-variance axis was taken as the plane normal. This yielded a reversible local 2D frame for planarized computations.

Polygonal areas were then computed in the local 2D frame using the shoelace formula. This approach was used for the generic `AREA_3D` metric and for the area-based components of the `AoA / SF` and `VOM / sVOM / VoA` families.

## Metric definitions

### Distance

The `Distance` metric was defined as the Euclidean norm between two points \(A\) and \(B\):

\[
d = \|B-A\|_2
\]

and reported in millimeters.

### Area

`Area` was defined as the exposure area enclosed by a user-specified polygon in 3D space. The polygon vertices were projected onto their SVD-derived best-fit plane, and area was computed with the shoelace formula in that 2D frame. The reported value therefore corresponds to the area of the best-fit planar representation of the polygon rather than to an arbitrarily triangulated surface patch.

### Volume

`Volume` was estimated from sparse boundary points. The implementation attempted an automatic alpha-shape-like reconstruction using Delaunay tetrahedralization and tetrahedron selection based on circumsphere radius. If no stable closed reconstruction could be identified, the method fell back to a convex hull estimate. Volumes were reported in cubic millimeters. This metric was used as a sparse geometric approximation rather than as a replacement for full segmentation-based volumetry.

### AoE

`AoE` was defined as the included angle at a pivot point \(B\) between rays \(BA\) and \(BC\):

\[
\mathrm{AoE} = \cos^{-1}\left(\frac{(A-B)\cdot(C-B)}{\|A-B\|\|C-B\|}\right)
\]

and reported in degrees.

### AoA and surgical freedom

The `AoA / SF` family used four anatomically ordered entry landmarks (`cranial`, `caudal`, `medial`, `lateral`) together with a pivot. Vertical and horizontal `AoA` were computed from pivot-based angular constructs defined by the corresponding cardinal limits. `Surgical freedom` (`SF`) was defined as the area of the quadrilateral delimited by the four entry points after projection to the best-fit plane of the entry set.

For standardized surgical freedom, each entry point \(P_i\) was projected radially from the pivot \(T\) to a fixed radius \(r\):

\[
P_i^{*} = T + r \frac{P_i-T}{\|P_i-T\|}
\]

and the area of the rescaled quadrilateral was then recomputed. Native and standardized SF were reported in square millimeters.

### VOM, sVOM, and VoA

The `VOM / sVOM / VoA` family modeled the operative corridor between an entry polygon and a target polygon. Each polygon was projected into its best-fit plane and replaced by an area-matched ellipse that preserved the PCA-derived anisotropy of the original polygon while matching its area. A frustum-like shell was then generated between entry and target ellipses to define `VOM`. A distal cut at a predefined standard distance from the target defined the standardized corridor segment (`sVOM`).

`VoA` was computed from the corridor direction relative to the target plane. Specifically, the acute angle between the corridor vector and the target-plane normal was computed first, and the angle to the plane was then defined as:

\[
\mathrm{VoA} = 90^\circ - \theta_{\text{acute to target normal}}
\]

The following outputs were retained from this family: `VoA_deg`, `VOM_mm3`, `sVOM_mm3`, `target_distance_mm`, `entry_polygon_area_mm2`, `target_polygon_area_mm2`, `entry_ellipse_area_mm2`, `target_ellipse_area_mm2`, and `svom_cut_ellipse_area_mm2`.

## Interactive workflows

The same metric definitions were accessible through both direct scripting and interactive interfaces. In the desktop GUI and in `SurgiplotToolkit` for 3D Slicer, users could either compute metrics prospectively while collecting landmarks or retrospectively from a persistent shared landmark dataset. This design allowed the same landmark set to be reused across multiple metrics without recollection.

Within 3D Slicer, a dedicated analysis context could be selected from a model, segmentation, or image-based scene. When necessary, raw 3D models were rescaled by selecting two calibration points on the model surface and entering a known real-world distance. If the original model distance was \(d_{model}\) and the known physical distance was \(d_{real}\), the uniform scale factor was:

\[
s = \frac{d_{real}}{d_{model}}
\]

The reference model was then rescaled by \(s\), and later picks were interpreted directly in the corrected coordinate frame.

## Export and reproducibility

Two output datasets were generated. The first was a case-level results table containing one row per subject and target-specific result columns, for example `IAC_VOM_mm3` or `CN_VII_AoA_vertical_deg`. The second was a point-level reproducibility table containing, for each metric call, the subject identifier, anatomical target, metric family, point grouping, point label, and 3D coordinates. This second table was used to retain a direct record of the landmarks that generated each scalar output.

## Methodological caveats

Several caveats should be noted. First, the dominant source of variability remains landmark definition rather than numerical implementation. Second, polygonal metrics depend on point ordering and on the appropriateness of a best-fit plane approximation. Third, `Volume` is an estimate derived from sparse points and should not be interpreted as a segmentation-derived ground truth. Fourth, the use of standardized constructs such as standardized SF or sVOM requires explicit reporting of the chosen radius or standard distance. Finally, the ellipse substitution in the VOM family should be understood as an intentional analytical simplification introduced to stabilize corridor comparison rather than as an attempt to preserve every contour irregularity of the original polygons.
