# Metric Reference

## 1. Scope and notation

This document defines the version `0.3.0` software contract. It describes what the code computes, independently of how a historical study may have implemented a similarly named construct.

Notation:

- points are column vectors in \(\mathbb{R}^3\);
- input spatial units are assumed to be millimeters;
- \(\|\cdot\|_2\) is the Euclidean norm;
- ordered polygon indices wrap modulo \(n\);
- all inverse-cosine arguments are clipped to `[-1, 1]` after normalization;
- all reported angles use degrees.

## 2. Shared validation

The metric boundary rejects:

- missing required inputs;
- coordinates with NaN or infinity;
- arrays with an invalid dimensional shape;
- too few distinct points;
- zero-length angular arms;
- collinear polygons requiring a best-fit plane;
- coplanar point clouds requiring a 3D volume;
- nonpositive or non-finite standardization values.

Vertex order is semantic. Surgiplot does not infer a polygon perimeter from an unordered point cloud.

## 3. Best-fit plane

For ordered points (P_i), the centroid is:

\[
c = \frac{1}{n}\sum_{i=1}^{n}P_i
\]

The centered matrix \(X\) contains rows \(P_i-c\). Singular value decomposition gives:

\[
X = U\Sigma V^T
\]

The first two columns of \(V\) define the local in-plane axes \(u,v\); the third defines the normal \(n\). Projected coordinates are:

\[
x_i=(P_i-c)\cdot u,\qquad y_i=(P_i-c)\cdot v
\]

At least three distinct, non-collinear points are required.

## 4. `DISTANCE_3D`

### Inputs

- `A`: one point or resolvable name;
- `B`: one point or resolvable name.

### Definition

\[
d = \|B-A\|_2
\]

### Result

`Distance3DResult.distance_mm`

### Debug fields

When requested: `A`, `B`, and the declared `space` string.

### Properties

- invariant under rigid translation and rotation;
- scales linearly under uniform scale;
- zero is valid when A and B coincide.

The function does not apply the optional `transforms` argument in this release.

## 5. `AREA_3D`

### Inputs

- `polygon`: at least three distinct ordered 3D points.

### Definition

Points are projected to the SVD best-fit plane. The projected polygon area is:

\[
A = \frac{1}{2}\left|\sum_{i=1}^{n}x_i y_{i+1}-y_i x_{i+1}\right|
\]

### Result

`Area3DResult.area_mm2`

### Debug fields

- original polygon;
- projected 2D polygon;
- reconstructed coplanar 3D polygon;
- centroid and SVD basis;
- point count and formula identifier.

### Interpretation

The result is the area of the polygon’s best-fit planar projection. It is not the area of an arbitrary triangulated nonplanar surface.

### Failure and bias conditions

- collinear or zero-area polygons fail;
- self-intersecting ordering may cancel area and fail or yield an unintended polygon;
- highly nonplanar contours are compressed to one plane;
- point density and vertex order determine the represented boundary.

## 6. `VOLUME_3D`

### Inputs

- `points`: at least four distinct, non-coplanar boundary samples.

### Primary reconstruction

The code computes a Delaunay tetrahedralization with Qhull joggling (`QJ`). For each tetrahedron, it computes:

- absolute tetrahedral volume;
- circumsphere radius.

Candidate alpha thresholds combine selected percentiles of finite circumsphere radii with multiples of median local nearest-neighbor spacing. For each threshold, tetrahedra with radius below the threshold are retained. The candidate records:

- boundary triangle count;
- fraction of input vertices used;
- connected tetrahedral components;
- sum of retained tetrahedral volumes.

The first connected candidate with at least 80% vertex coverage is accepted. Otherwise the best candidate is chosen by component count, coverage, and alpha.

### Fallback

If automatic alpha reconstruction fails with a recognized geometric or numerical error, the result falls back to the SciPy convex hull.

### Result

`Volume3DResult.volume_mm3`

### Debug fields

- points and surface triangles;
- `reconstruction_method`: `alpha_shape_auto` or `convex_hull`;
- alpha threshold;
- vertex coverage;
- connected-component count;
- convex-hull volume;
- alpha-shape volume when available.

### Interpretation

This is a sparse-point morphometric estimate. It is not segmentation-derived ground truth. Analyses must retain the reconstruction method because alpha and convex-hull estimates have different bias characteristics.

## 7. `AOE` — principal included Angle of Exposure

### Inputs

- `A`, `B`, `C`, with `B` as pivot.

### Definition

\[
u=A-B,\qquad v=C-B
\]

\[
\mathrm{AoE}=\cos^{-1}\left(\frac{u\cdot v}{\|u\|_2\|v\|_2}\right)
\]

### Result

`AOEResult.aoe_deg`, bounded to `[0, 180]` by the principal inverse cosine.

### Critical scope statement

This is a three-point included angle. It does not reconstruct a reflex angle greater than 180° and is not, by itself, the complete vectorial circular Angle of Exposure protocol described by Porto et al. A study using that published protocol must not substitute this principal-angle output without methodological justification.

### Debug fields

`A`, `B`, `C`, `space`, and normalized dot-product cosine.

## 8. `AOA_SF` — Angle of Attack and Surgical Freedom

### Inputs

Exactly four distinct entry points in this order:

1. cranial;
2. caudal;
3. medial;
4. lateral.

The target is one distinct pivot point. Optional `sf_rescale_radius_mm` must be finite and positive.

### Vertical and horizontal AoA

Vertical AoA is the interior angle at the pivot of the triangle `(cranial, caudal, pivot)`. Horizontal AoA uses `(medial, lateral, pivot)`.

For endpoints (P_1,P_2) and pivot (T):

\[
\gamma=\cos^{-1}\left(\frac{(P_1-T)\cdot(P_2-T)}{\|P_1-T\|_2\|P_2-T\|_2}\right)
\]

### Native SF

The non-crossing perimeter is:

```text
cranial → lateral → caudal → medial
```

This quadrilateral is projected to its best-fit plane and evaluated with the shoelace formula.

### Standardized SF

Each entry point (P_i) is projected along its pivot ray to radius (r):

\[
P_i^*=T+r\frac{P_i-T}{\|P_i-T\|_2}
\]

The standardized quadrilateral uses the same anatomical perimeter order and the same best-fit-plane area procedure.

### Result

- `aoa_vertical_deg`;
- `aoa_horizontal_deg`;
- `sf_entry_area_mm2`;
- `sf_entry_area_rescaled_mm2` (`0.0` when no radius is requested);
- `rescale_radius_mm`.

### Version change

Before `0.2.0`, fewer than four entry points triggered a generic centroid fallback unrelated to the cardinal AoA/SF contract. That fallback has been removed. Incomplete or duplicate role input now fails explicitly.

## 9. `VOM_VOA` — VOM, sVOM, and VoA

### Terminology

- VOM: Volume of Operative Maneuverability;
- sVOM: standardized Volume of Operative Maneuverability;
- VoA: Visuo-operative Angle.

### Inputs

- ordered entry polygon, at least three points;
- ordered target polygon, at least three points;
- positive finite standard distance `stand_dist`.

Both polygons must have finite, distinct, non-collinear vertices and nonzero shoelace area after projection.

### 9.1 Robust area-constrained ellipse cross-sections

Each polygon is projected to its own right-handed SVD plane. With five or more points, Surgiplot fits an ellipse-specific conic using the numerically scaled direct least-squares method of Fitzgibbon, Pilu, and Fisher, implemented using the stable block decomposition described by Halíř and Flusser. Tukey-biweight iteratively reweighted least squares is initialized by deterministic five-point consensus fits. Points outside the robust residual threshold are reported explicitly and are excluded from the ordered inlier polygon used for area matching. Three- or four-point contours retain the deterministic SVD-extent fallback.

Let the robust direct fit return provisional semiaxes \(\hat a,\hat b\), and let \(A_p\) be the ordered inlier-polygon area. A single area-constraint scale is applied:

\[
k=\sqrt{\frac{A_p}{\pi\hat a\hat b}},\qquad
a=k\hat a,\qquad b=k\hat b
\]

Therefore \(\pi ab=A_p\) while the robust fitted aspect ratio is preserved. If \(\max(a,b)/\min(a,b)\leq1.05\), the section is regularized to a circle of radius \(\sqrt{A_p/\pi}\), because the major-axis direction of a nearly circular fit is not identifiable.

### 9.2 Corridor axis and target distance

Let \(c_E\) and \(c_T\) be the fitted ellipse centers:

\[
v=c_T-c_E,\qquad H=\|v\|_2
\]

Identical centroids fail because no trajectory is defined.

### 9.3 VoA

The target normal \(n_T\) is the normalized SVD best-fit-plane normal. Its arbitrary sign is removed by taking the acute trajectory-normal angle.

First:

\[
\theta=\cos^{-1}\left(\frac{v\cdot n_T}{\|v\|_2\|n_T\|_2}\right)
\]

Normal orientation is removed:

\[
\theta_a=\min(\theta,180^\circ-\theta)
\]

The angle to the target plane is:

\[
\mathrm{VoA}=90^\circ-\theta_a
\]

The result is bounded to `[0°, 90°]`.

This convention defines an approach perpendicular to the target plane as ideal (`VoA = 90°`) and a tangential approach as worst (`VoA = 0°`). VoA is reported separately and does not multiply the geometric volume.

### 9.4 Loft construction

Target-to-entry interpolation parameter \(q\in[0,1]\) controls:

- center: linear interpolation;
- semiaxes: linear interpolation;
- orientation: quaternion spherical linear interpolation (SLERP).

Before interpolation, target and entry normals are aligned. An ellipse does not possess a directed material major axis: simultaneous sign reversal leaves its frame unchanged, and a 90° frame rotation combined with exchanging the two semiaxes describes the same outline. Surgiplot therefore evaluates every valid right-handed in-plane sign and axis permutation. When an axis-swapped representation minimizes the target-to-entry rotation, both the frame and its associated semiaxes are swapped before interpolation. Quaternion SLERP then follows this minimum-rotation correspondence. This is the formal corridor rule sometimes illustrated as “Option B”; it prevents an artificial large twist such as forcing `+70°` when the same endpoint ellipse admits an equivalent `−20°` swapped representation. Nearly circular sections use minimum-twist parallel transport because their in-plane major axis is not identifiable.

Endpoint ellipses alone do not identify a physical material trajectory through anatomy. Minimum rotation is therefore a declared regularization assumption chosen for geometric continuity, not proof that tissue fibers or surgical instruments follow the interpolated frame.

Longitudinal resolution is not a fixed ring count. Refinement starts with 9 rings and doubles the number of intervals (`9, 17, 33, ...`) until successive closed-mesh volumes differ by no more than `1e-4` relative plus `1e-6 mm³` absolute, with a 257-ring safety limit. Angular resolution is similarly criterion-based: the smallest regular polygon whose area differs from its analytical ellipse by at most `1e-4` is used (currently 257 vertices). Adjacent rings are triangulated and both ends are capped.

### 9.5 Mesh volume and numerical frame

For oriented triangles with vertices (a,b,c), closed-mesh volume uses signed tetrahedral contributions:

\[
V=\left|\sum \frac{a\cdot(b\times c)}{6}\right|
\]

Before evaluation, Surgiplot subtracts the mean mesh vertex position from every triangle vertex. This local reference does not change the solid but prevents catastrophic cancellation when global coordinates are large. Translation invariance is regression-tested at million-unit offsets.

The debug payload records angular tolerance, selected longitudinal sections, convergence history, and whether the maximum refinement was reached. It also reports `ellipse_frame_correspondence_policy`, `entry_frame_alignment_operation`, `entry_frame_alignment_swapped_axes`, and `entry_frame_alignment_rotation_deg` so the selected correspondence is auditable.

### 9.6 sVOM

The effective standard distance is:

\[
d_s=\min(d_{requested},H)
\]

and the distal ratio is \(q_s=d_s/H\). A second convergence-controlled closed loft spans target to \(q_s\). When the requested distance reaches or exceeds the corridor length, the full VOM mesh and value are reused so sVOM equals VOM exactly.

Debug output distinguishes:

- `requested_stand_dist_mm`;
- effective `stand_dist_mm`;
- `stand_dist_clipped_to_corridor`.

### Results

- `voa_deg`;
- `vom_mm3`;
- `svom_mm3`.

Debug output identifies this revised formulation as method version `3d-loft-2.0`; CLI and Toolkit composite exports retain that identifier.

### Debug comparison metrics

The debug payload also provides legacy polygon-frustum, ellipse-frustum, adjusted-ellipse, and supplementary tilt-weighted values. These are comparison fields, not the primary `vom_mm3` or `svom_mm3` outputs.

### Methodological interpretation

The primary software outputs come from an explicit 3D loft. No separate angle-dependent multiplier is applied to the primary volume. Robust ellipse fitting, outlier exclusion, near-circle regularization, minimum-rotation axis correspondence, SLERP, and convergence tolerances are analytical modeling choices and must be reported. Because an axis swap also changes which semiaxis is interpolated to which endpoint semiaxis, the choice can change VOM and sVOM numerically; it is not merely a display correction.

References:

- Fitzgibbon AW, Pilu M, Fisher RB. Direct least square fitting of ellipses. *IEEE Transactions on Pattern Analysis and Machine Intelligence*. 1999;21(5):476-480. doi:10.1109/34.765658.
- Halíř R, Flusser J. Numerically stable direct least squares fitting of ellipses. *WSCG*. 1998.
- Shoemake K. Animating rotation with quaternion curves. *Computer Graphics*. 1985;19(3):245-254. doi:10.1145/325165.325242.
- Alyassin AM, Lancaster JL, Downs JH, Fox PT. Evaluation of new algorithms for the interactive measurement of surface area and volume. *Medical Physics*. 1994;21(6):741-752. doi:10.1118/1.597333.

## 10. Composite wrappers

| Wrapper | Behavior |
|---|---|
| `AOA(...)` | returns the full `AOASFResult` |
| `SF(...)` | returns native SF scalar |
| `VOM(...)` | returns the full `VOMVOAResult` |
| `VOA(...)` | returns VoA scalar |
| `SVOM(...)` | returns sVOM scalar |

## 11. Scaling laws and invariants

For uniform positive scale \(s\):

- distances multiply by \(s\);
- areas multiply by \(s^2\);
- volumes multiply by \(s^3\);
- AoE, AoA, and VoA remain unchanged;
- standardized SF changes according to the explicit chosen radius, not the original scale, if the radius is held fixed.

All metrics should be invariant under rigid translation and rotation, subject to floating-point and discretization tolerance.

## 12. Scientific references

- VOM and sVOM context: [PMID 39952396](https://pubmed.ncbi.nlm.nih.gov/39952396/).
- VoA context: [PMID 41791114](https://pubmed.ncbi.nlm.nih.gov/41791114/).
- Vectorial Angle of Exposure context: [PMID 38364224](https://pubmed.ncbi.nlm.nih.gov/38364224/).

The source and tests define this software release. Publication language should describe the exact version and implementation used.
