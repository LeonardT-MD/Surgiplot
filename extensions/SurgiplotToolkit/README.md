# SurgiplotToolkit

This directory contains the first scripted 3D Slicer extension scaffold for integrating Surgiplot into Slicer.

Current goals of the scaffold:

- load markups from the Slicer scene
- call Surgiplot metric families from Slicer UI
- push results back into the Slicer scene
- create a clean adapter layer between Slicer MRML nodes and the `surgiplot` Python backend

## Development mode

This scaffold lives inside the main Surgiplot repository for now as an incubation area:

```text
Surgiplot/
  extensions/
    SurgiplotToolkit/
```

Later, it can be moved into its own repository with minimal structural change.

## Loading in Slicer

This is a Python scripted extension scaffold. Based on the 3D Slicer developer guide, the easiest development workflow is to add the module source directory to Slicer’s additional module paths.

Use this directory in Slicer:

```text
/Users/leonardo/Projects/Surgiplot/extensions/SurgiplotToolkit/SurgiplotToolkit
```

Relevant official Slicer documentation:

- [Developer guide: Extensions](https://slicer.readthedocs.io/en/latest/developer_guide/extensions.html)
- [Developer guide: Module overview](https://slicer.readthedocs.io/en/5.8/developer_guide/module_overview.html)
- [User guide: Extension Wizard](https://slicer.readthedocs.io/en/5.6/user_guide/modules/extensionwizard.html)

## First milestone

The first module prototype currently focuses on:

- `VOM / sVOM / VoA`
- `AoA / SF`
- `AoE`
- `3D Distance`
- `3D Area`
- `3D Volume`

using:

- Slicer markups nodes as point containers
- a thin adapter that converts MRML markups to NumPy arrays
- the existing `surgiplot` backend for all actual computations

Current implemented prototype behavior:

- `Workflow overview and validation`
  the module now keeps a live validation summary for the currently selected nodes and reports whether:
  - `AoA / SF`
  - `VOM / sVOM / VoA`
  - `AoE`
  - `Distance`
  - `Area`
  - `Volume`
  are ready to compute.
  It also includes a `Clear Result Nodes` action for scene cleanup during iterative testing.

- `Landmark dataset workflow`
  the module now includes a Surgiplot-style shared landmark dataset workflow:
  - collect labeled landmarks progressively into a reusable markups dataset
  - support direct role buttons such as `cranial`, `caudal`, `medial`, `lateral`, and `pivot`
  - support progressive `entry_*` and `target_*` point collection for corridor polygons
  - export selected collected landmarks into the metric-specific `Entry`, `Target`, and `Pivot` nodes
  - auto-fill `AoA / SF` inputs from collected labels when the standard labels are present
  - support optional model-only scale calibration using two picked scale points and a known distance

- `Reference anatomy`
  the module can optionally link generated result nodes to a selected model, scalar volume, or segmentation context. Results are grouped under a context-specific folder in the Slicer subject hierarchy and the selected anatomy name is also written into grouped result tables.

- `CT/MRI-linked workflow helpers`
  the module can:
  - set a selected scalar volume as slice background
  - create standard entry, target, and pivot markups nodes
  - arm direct placement into entry, target, or pivot nodes from the module UI
  - arm the next expected point automatically during guided sessions
  - jump all slice views to entry, target, or pivot locations
  - support a more direct image-linked markup workflow before metric computation

- `Guided acquisition`
  the module can start:
  - an `AoA / SF` session, which prepares entry and pivot nodes and tracks progress toward `4 + 1` required points
  - a `VOM / sVOM / VoA` session, which prepares entry and target nodes and tracks progress toward the minimum contour requirements
  - indicate the current guided target and next expected point
  - jump slices to the active guided target
  - switch Slicer into markup placement mode for the next expected point

- `VOM / sVOM / VoA`
  reads entry and target markups, computes the corridor family through Surgiplot, and adds:
  - result surfaces as Slicer model nodes
  - corridor guide curves
  - centroid fiducials
  - a Slicer table node with the grouped report
  - a text report node with the latest structured result payload

- `AoA / SF`
  computes the family from entry markups plus a pivot markup and adds:
  - triangle/surface debug models
  - ordered entry-landmark fiducials
  - a Slicer table node with grouped outputs
  - a text report node with the latest structured result payload
  - ordering metadata showing whether anatomical labels were inferred or raw control-point order was used
  - a role-assignment assistant that can stamp `cranial / caudal / medial / lateral` labels onto the current 4-point entry node in control-point order

- `AoE`, `Distance`, `Area`, and `Volume`
  currently return structured scalar outputs through the module panel, update a scene text report node, and provide a first integration path for future scene plotting.

## Architectural direction

Long term, this extension is expected to evolve toward:

1. full markups-driven metric workflows
2. rendered debug geometry plotted directly in Slicer 3D views
3. image-aware workflows using CT, MRI, DICOM, and surface models
4. scene serialization and report export

The implementation intentionally keeps the computational backend in the main `surgiplot` package so that:

- metric logic remains shared
- GUI/CLI/Python and Slicer stay numerically aligned
- research scripts and Slicer workflows use the same computation core

## Local testing workflow

You do not need to package or install the extension formally to test it during development.

Recommended loop:

1. Open `3D Slicer` from `/Applications/Slicer.app`.
2. Open:
   `Edit -> Application Settings -> Modules`
3. In `Additional module paths`, add:

```text
/Users/leonardo/Projects/Surgiplot/extensions/SurgiplotToolkit/SurgiplotToolkit
```

4. Restart Slicer.
5. Search for and open `SurgiplotToolkit`.

During active development:

- after Python-only code changes, use the module reload workflow in Slicer if developer mode is enabled
- otherwise restart Slicer to guarantee a clean reload
- re-open `SurgiplotToolkit` and rerun the same acquisition/metric sequence

Suggested manual smoke test checklist:

1. Load a scalar volume or anatomical model into Slicer.
2. In `Imaging & Markups Workflow`, select the CT/MRI in `Slice background volume`.
3. Click `Set Background Volume`.
4. Click `Create Entry Node`, `Create Target Node`, and `Create Pivot Node`.
5. Click `Place Into Entry` and add one or more points in slice or 3D views.
6. Click `Jump Slices To Entry` and confirm slice centering updates.
7. Start `AoA / SF Session` and confirm:
   - status shows `4 / 1` style progress
   - `Place Next Guided Point` arms placement into the expected node
   - `Jump To Guided Target` centers on the next acquisition context
8. Start `VOM Session` and confirm:
   - status switches to entry/target contour guidance
   - the active guided target changes from entry to target after the minimum entry contour is reached
9. Run `Compute AoA / SF` or `Compute VOM / sVOM / VoA`.
10. Inspect:
   - the `Results` text panel
   - created model nodes in the 3D view
   - created fiducial and table nodes
   - the `SurgiplotToolkit Results` folder in Subject Hierarchy

Recommended interactive workflow for parity with the Surgiplot GUI scene workspace:

1. Create or reuse the `Landmark Dataset`.
2. If working on a raw 3D model that is not already in mm, create the `Scale` node and pick 2 scale points.
3. Use the labeled collection buttons to progressively acquire:
   - `cranial`, `caudal`, `medial`, `lateral`, `pivot`
   - `entry_*` points
   - `target_*` points
   - custom labels when needed
4. Select the collected rows you want to use for a given metric.
5. Export them into:
   - `Entry`
   - `Target`
   - `Pivot`
6. Run the metric from the existing compute section.

For careful validation while coding, it is best to keep one small reference scene and repeat the same smoke test after each change rather than changing both code and test data at the same time.

There is also a headless runtime smoke test for the local installed Slicer app:

```bash
/Applications/Slicer.app/Contents/MacOS/Slicer \
  --no-splash \
  --no-main-window \
  --python-script /Users/leonardo/Projects/Surgiplot/extensions/SurgiplotToolkit/scripts/headless_smoke_test.py \
  --exit-after-startup
```

That script currently validates:

- module import inside Slicer Python
- markups node creation
- AoA / SF grouped computation
- VOM / sVOM / VoA grouped computation
- result payload integrity at a basic smoke-test level

## Standalone extension packaging

This incubator already contains the core pieces of a standalone Slicer extension repository:

- [CMakeLists.txt](/Users/leonardo/Projects/Surgiplot/extensions/SurgiplotToolkit/CMakeLists.txt)
- [SurgiplotToolkit.s4ext](/Users/leonardo/Projects/Surgiplot/extensions/SurgiplotToolkit/SurgiplotToolkit.s4ext)
- [SurgiplotToolkit/CMakeLists.txt](/Users/leonardo/Projects/Surgiplot/extensions/SurgiplotToolkit/SurgiplotToolkit/CMakeLists.txt)

To export it as a separate repository scaffold, run:

```bash
python /Users/leonardo/Projects/Surgiplot/extensions/SurgiplotToolkit/scripts/export_standalone_repo.py
```

By default this creates:

```text
/Users/leonardo/Projects/SurgiplotToolkit
```

The export script copies:

- the Slicer extension scaffold
- the module source
- the standalone metadata
- a vendored copy of the `surgiplot` backend into:
  `SurgiplotToolkit/SurgiplotToolkitLib/External/surgiplot_backend/surgiplot`

That vendored path is already supported by the bridge loader, so the exported extension can evolve independently from the incubator repository.
