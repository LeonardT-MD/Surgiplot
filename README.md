<p align="center">
  <img src="docs/assets/SURGIPLOT_BANNER.png" alt="Surgiplot banner" width="100%">
</p>

<p align="center">
  <img src="docs/assets/LOGO_SURGIPLOT.png" alt="Surgiplot logo" width="180">
</p>

# Surgiplot

Surgiplot is a coordinate-based morphometric framework for neurosurgical anatomical analysis. The repository contains the shared computation core together with the Python API, the command-line interface, and the desktop GUI for dataset curation, rendered-scene landmark acquisition, and interactive analysis.

The guiding principle is explicit geometry. Once anatomical landmarks are represented as 3D coordinates in a common frame, operative metrics should be derived from inspectable geometric procedures rather than opaque viewer state.

## Repository scope

This repository is the primary home of:

- the `surgiplot` Python package
- the CLI for scripted and batch workflows
- the desktop GUI for tabular and rendered-scene analysis
- the shared metric and geometry core used by all interfaces
- the project’s publication-oriented methods documentation

The `3D Slicer` extension is now conceptually a companion project:

- `SurgiplotToolkit`

An incubator copy may still exist in this repository for coordinated development, but the long-term public-facing distinction is:

- `Surgiplot`: computation core, Python, CLI, desktop GUI
- `SurgiplotToolkit`: scene-native `3D Slicer` integration

## Supported metric families

The computation core currently supports:

- `Distance`
- `Area`
- `Volume`
- `AoE`
- `AoA / SF`
- `VOM / sVOM / VoA`

The same metric logic can be applied to landmarks originating from:

- normalized tabular datasets
- neuronavigation exports
- generic point lists
- manually assembled coordinate arrays
- landmarks collected interactively on imported 3D scenes

## Installation

### Core package

```bash
pip install -e .
```

### With rendered-scene support

```bash
pip install -e .[scene3d]
```

### Main entry points

```bash
surgiplot
surgiplot version
surgiplot_gui
```

## Project architecture

Surgiplot is organized around four layers:

1. `coordinate acquisition`
2. `dataset normalization`
3. `metric computation`
4. `visualization and export`

All user-facing layers call the same computation core. This is critical methodologically, because it keeps Python scripts, CLI pipelines, GUI work, and Slicer integration numerically aligned.

## Python and CLI use

The canonical data object is the `Dataset`, which stores:

- stable internal point identifiers
- optional anatomical labels and aliases
- dataset metadata

Two complementary scripting styles are supported.

### Module-level style

```python
import surgiplot as sp

ds = sp.load("case.csv", source="database")
result = sp.VOM(data=ds, entry="point_1-4", target="point_10-13", stand_dist=10.0)
```

### Dataset-bound style

```python
import surgiplot as sp

ds = sp.load("case.csv", source="database")
result = ds.VOM(entry="point_1-4", target="point_10-13", stand_dist=10.0)
```

The CLI mirrors the same metric families for reproducible shell-based workflows and supports compact multi-point range syntax such as:

- `point_1-9`
- `point_1:9`
- `1-9`

## Desktop GUI use

The desktop GUI is intended for:

- interactive dataset inspection and relabeling
- point collection on imported rendered 3D scenes
- scale calibration when raw models are not already expressed in meaningful physical units
- post-hoc metric computation from a persistent landmark dataset

The GUI is not a separate numerical system. It is an acquisition and visualization layer for the same underlying coordinate model.

## Documentation

Core documentation for this repository is listed in:

- [docs/DOCUMENTATION_INDEX.md](/Users/leonardo/Projects/Surgiplot/docs/DOCUMENTATION_INDEX.md)

The most important documents are:

- [docs/SURGIPLOT_PUBLICATION_GUIDE.md](/Users/leonardo/Projects/Surgiplot/docs/SURGIPLOT_PUBLICATION_GUIDE.md)
- [docs/SURGIPLOT_ECOSYSTEM_METHODS_SUMMARY.md](/Users/leonardo/Projects/Surgiplot/docs/SURGIPLOT_ECOSYSTEM_METHODS_SUMMARY.md)
- [docs/SURGIPLOT_MANUSCRIPT_METHODS_DRAFT.md](/Users/leonardo/Projects/Surgiplot/docs/SURGIPLOT_MANUSCRIPT_METHODS_DRAFT.md)
- [docs/SURGIPLOT_SUPPLEMENTARY_METHODS.md](/Users/leonardo/Projects/Surgiplot/docs/SURGIPLOT_SUPPLEMENTARY_METHODS.md)
- [docs/SURGIPLOT_TECHNICAL_IMPLEMENTATION_NOTE.md](/Users/leonardo/Projects/Surgiplot/docs/SURGIPLOT_TECHNICAL_IMPLEMENTATION_NOTE.md)

For the `3D Slicer` extension specifically, the corresponding public documentation should live in the separate `SurgiplotToolkit` repository.

## Scientific orientation

Surgiplot is intended for anatomically explicit and reproducible quantitative work. The framework is best suited for studies in which:

- landmarks can be defined meaningfully in 3D
- derived geometric constructs should remain inspectable
- the same analyses must be repeatable across interactive and scripted environments
- scalar outputs alone are insufficient without retained point-level provenance

The project therefore emphasizes:

- transparent mathematics
- stable dataset semantics
- interface-independent metric definitions
- explicit reproducibility of the landmarks used to generate each reported result

## Provenance, comparison, and repeatability

The current `Surgiplot` core now also includes a first study-level layer for:

- acquisition provenance
- metric-run registration
- cross-modality comparison tables
- repeatability summaries across repeated runs

These capabilities are exposed most concretely in `SurgiplotToolkit`, but the underlying data model now exists in the shared core as well through:

- [surgiplot/core/studies.py](/Users/leonardo/Projects/Surgiplot/surgiplot/core/studies.py)

The intent is methodological rather than cosmetic. A reported metric should be traceable not only to a point set, but also to:

- `subject_id`
- `anatomical_target`
- `acquisition_modality`
- `operator`
- `session_id`
- interface context

This allows the same metric family to be compared across:

- different acquisition modalities
- different observers
- repeated sessions

while preserving a stable run registry suitable for later statistical analysis.
