# Desktop GUI Guide

## 1. Purpose and launch

The desktop application provides point-dataset curation, metric configuration, result inspection, and 2D/3D visualization over the same numerical functions used by the Python and command-line interfaces.

Install a GUI profile and launch:

```bash
python -m pip install '.[gui]'
surgiplot-gui
```

For interactive mesh rendering:

```bash
python -m pip install '.[scene3d]'
surgiplot-gui
```

The legacy executable name `surgiplot_gui` remains available for compatibility.

## 2. Application structure

The main window separates data preparation from analysis:

- **Database**: import, review, edit, add, delete, clear, and export points;
- **Analysis**: choose a metric, assign point roles, compute, inspect results, and visualize debug geometry;
- **Dataset Preview**: detachable summary of current points and labels;
- **Debug Console**: detachable technical output for the current computation;
- **Visualizer**: metric and scene layers with camera and visibility controls.

Dataset and analysis panels share one in-memory `Dataset`. Edits become available to metric selectors after the dataset-change signal refreshes the analysis view.

## 3. Backend selection

The application chooses a rendering path from `SURGIPLOT_3D_BACKEND`:

| Value | Behavior |
|---|---|
| `auto` | select an available backend, preferring the interactive path where supported |
| `pyvista` | request PyVista/VTK |
| `safe`, `matplotlib`, `mpl` | use Matplotlib rendering |

Example:

```bash
SURGIPLOT_3D_BACKEND=safe surgiplot-gui
```

Changing the viewer backend does not change metric formulas.

## 4. Importing data

### Navigation export

Use the navigation import action and select the vendor format when prompted. Auto detection recognizes current Stryker annotation and Medtronic JSON structures. Brainlab is displayed as a reserved format but is not implemented.

### Normalized database

Import CSV, TSV, XLSX, or Surgiplot JSON. Review the preview before accepting it into the active dataset.

### Generic points

Use generic import for simple `name x y z`, comma-separated, or tab-separated point files. Original names become aliases when records are normalized to sequential canonical identifiers.

### Rendered-scene workspace

The rendered-scene workflow loads geometry and textures into a local acquisition workspace. It can optimize scene representation and commit picked points to the active dataset. This feature is a scene/landmark tool; the standalone repository does not ship segmentation models or weights.

## 5. Editing and labeling

The dataset table exposes canonical names, coordinates, and semantic labels. Use labels for anatomical roles while preserving stable canonical identifiers.

Recommended pattern:

```text
canonical name: point_17
labels: cranial, superior limit
preferred label: cranial
```

Validation rejects non-finite coordinates, blank names, duplicate canonical identifiers, and ambiguous aliases. Review the dataset preview after bulk edits.

## 6. Metric forms

### VOM / sVOM / VoA

Select an ordered entry perimeter, an ordered target perimeter, and the standard distance. Lists accept comma-separated names and ranges. Verify that polygon traversal follows the anatomical boundary without crossing it.

### AoA / SF

Assign all five visible roles:

1. cranial;
2. caudal;
3. medial;
4. lateral;
5. pivot/target.

The four entry roles must be distinct. Optional SF standardization projects each entry-pivot ray to the specified positive radius, entered in millimeters.

### AoE

Assign endpoints A and C and pivot B. The displayed value is the principal included three-point angle, not a reflex-capable full vectorial reconstruction.

### Distance, area, and volume

- distance uses two points;
- area uses an ordered polygon with at least three distinct vertices;
- volume uses at least four distinct non-coplanar samples.

## 7. Point picker

Single-role controls open a single-selection picker. Polygon and cloud controls open multi-selection. The picker works from the current canonical point names and labels.

For polygons, the selection text determines the computational order. A visually correct set with the wrong order may define a self-intersecting perimeter.

## 8. Results and debug output

After a successful computation, the application shows:

- primary scalar result fields with units;
- method-specific secondary fields where applicable;
- warnings or standardization information;
- structured debug geometry in the technical console;
- a visual reconstruction of the points and modeled geometry.

Use **Copy result JSON** when transferring a result into an analysis record. Store the normalized input separately; copied result JSON is not a substitute for source data provenance.

## 9. Visualizer controls

The visualizer supports:

- resetting and zooming the camera;
- selecting interactive points;
- toggling registered layers individually;
- showing or hiding all layers;
- changing metric geometry opacity, size, and line width;
- exporting a PNG;
- saving or opening supported scene payloads.

Layer visibility changes presentation only. Hidden geometry continues to exist in the scene unless explicitly removed, and visibility does not change metric computation over already selected point coordinates.

## 10. Visual verification protocol

Before accepting a result:

1. inspect all selected landmarks against the source anatomy;
2. confirm coordinate frame and scale;
3. verify role labels;
4. inspect polygon closure and absence of crossing edges;
5. inspect target and entry plane orientation;
6. confirm the standard distance or radius;
7. note any fallback reconstruction method;
8. export or record the software version and input dataset.

The visualizer is a quality-control aid; it does not validate anatomical correctness automatically.

## 11. Failure recovery

### GUI starts but no 3D scene appears

Restart with the safe backend. If that works, inspect the PyVista/VTK/OpenGL installation rather than changing the metric core.

### Point selectors are empty

Return to Database, confirm that the import was accepted, and inspect Dataset Preview. Empty input files are rejected.

### AoA/SF refuses to compute

Confirm that cranial, caudal, medial, lateral, and pivot are all assigned, that cardinal entries are distinct, and that no entry coincides with the pivot.

### Area or VOM reports a degenerate polygon

Check for duplicate points, collinearity, wrong range expansion, or self-intersecting order.

### Volume falls back to convex hull

Review sample density and surface coverage. The method is exposed in debug output because a convex hull can overestimate concave morphology.

### Rendered-scene interaction is slow

Reduce scene complexity in the reconstruction workspace, use the safe backend, hide nonessential layers, and confirm available memory. Optimization changes scene representation; verify picked landmarks after any aggressive simplification.

## 12. Controlled-use recommendations

- Work on pseudonymized copies of source data.
- Keep the application and source datasets outside shared source-control directories.
- Record the commit/package version before a measurement session.
- Use one coordinate frame per active dataset.
- Export normalized JSON before and after material label edits.
- Repeat a subset of measurements to quantify intra- and inter-operator variability.
- Do not treat the software as a clinical navigation or treatment-decision system.
