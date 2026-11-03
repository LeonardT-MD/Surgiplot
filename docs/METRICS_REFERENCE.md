# Metric Reference

## 1. Scope and notation

This document defines the version `0.2.0` software contract. It describes what the code computes, independently of how a historical study may have implemented a similarly named construct.

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

### 9.1 Area-matched cross-sections

Each polygon is projected to its own SVD plane. Let its polygon area be \(A_p\). Along the two in-plane SVD axes, the implementation calculates coordinate extents:

\[
s_x=\max_i|x_i|,\qquad s_y=\max_i|y_i|
\]

Degenerate extents fall back to standard deviation and finally to `1.0`. A uniform factor is chosen:

\[
k=\sqrt{\frac{A_p}{\pi s_xs_y}}
\]

Ellipse semiaxes are:

\[
a=ks_x,\qquad b=ks_y
\]

Therefore the analytical ellipse area is \(\pi ab=A_p\), subject to floating-point precision.

### 9.2 Corridor axis and target distance

Let \(c_E\) and \(c_T\) be entry and target centroids:

\[
v=c_T-c_E,\qquad H=\|v\|_2
\]

Identical centroids fail because no trajectory is defined.

### 9.3 VoA

The target normal \(n_T\) is computed from ordered target edges around the target centroid. A best-fit-plane normal is used if the accumulated edge normal is numerically degenerate.

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

### 9.4 Loft construction

Target-to-entry interpolation parameter \(q\in[0,1]\) controls:

- center: linear interpolation;
- semiaxes: linear interpolation;
- in-plane basis: elementwise interpolation followed by orthonormalization.

The full VOM uses 72 longitudinal sections. Each section contains 120 sampled ellipse points. Adjacent rings are triangulated and both ends are capped, producing a closed mesh.

### 9.5 Mesh volume and numerical frame

For oriented triangles with vertices (a,b,c), closed-mesh volume uses signed tetrahedral contributions:

\[
V=\left|\sum \frac{a\cdot(b\times c)}{6}\right|
\]

Before evaluation, Surgiplot subtracts the mean mesh vertex position from every triangle vertex. This local reference does not change the solid but prevents catastrophic cancellation when global coordinates are large. Translation invariance is regression-tested at million-unit offsets.

The sampled polygonal ellipse introduces a small discretization error relative to the analytical ellipse area. Tests permit a relative error of `5e-4` for a simple parallel constant-section fixture.

### 9.6 sVOM

The effective standard distance is:

\[
d_s=\min(d_{requested},H)
\]

and the distal ratio is \(q_s=d_s/H\). A second closed loft from target to \(q_s\) uses 48 longitudinal sections. When the requested distance exceeds the corridor length, sVOM equals VOM.

Debug output distinguishes:

- `requested_stand_dist_mm`;
- effective `stand_dist_mm`;
- `stand_dist_clipped_to_corridor`.

### Results

- `voa_deg`;
- `vom_mm3`;
- `svom_mm3`.

### Debug comparison metrics

The debug payload also provides legacy polygon-frustum, ellipse-frustum, adjusted-ellipse, and supplementary tilt-weighted values. These are comparison fields, not the primary `vom_mm3` or `svom_mm3` outputs.

### Methodological interpretation

The primary software outputs come from an explicit 3D loft. No separate angle-dependent multiplier is applied to the primary volume. The ellipse substitution and basis interpolation are analytical modeling choices and must be reported.

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
