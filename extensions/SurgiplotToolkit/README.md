# SurgiplotToolkit Incubator Copy

This directory is the `SurgiplotToolkit` development copy that lives inside the main `Surgiplot` repository for coordinated work on:

- the Slicer module code
- the bridge layer to the shared Surgiplot backend
- release preparation before vendoring the updated backend into the separate extension repository

## Role inside the main repository

This copy should be understood as an integration and release workspace rather than as the preferred public home of the extension.

The primary project split is:

- [Surgiplot](/Users/leonardo/Projects/Surgiplot)
  computation core, Python API, CLI, desktop GUI, and framework-level methods documentation

- `SurgiplotToolkit`
  the separate `3D Slicer` extension repository intended for scene-native use, packaging, and public-facing Slicer documentation

## When to work here

This incubator copy is useful when:

- a backend change in `surgiplot` must be tested immediately against the Slicer layer
- a coordinated release of the extension is being prepared
- the vendored backend snapshot for the standalone extension repository is being refreshed

## Public-facing toolkit repository

The standalone extension repository is expected to live separately at:

- [SurgiplotToolkit](/Users/leonardo/Projects/SurgiplotToolkit)

That repository should contain:

- the Slicer extension scaffold
- the vendored backend snapshot used by the extension
- the public README
- the Slicer-specific methods and user documentation

## Practical note

If you are looking for the user-facing documentation for the extension itself, use the standalone toolkit repository rather than this incubator copy. This directory exists primarily to keep coordinated development against the main Surgiplot codebase straightforward.
