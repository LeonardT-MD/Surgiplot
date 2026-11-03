# Changelog

All notable changes to Surgiplot are documented here. The project is pre-1.0; validation may become stricter between minor releases.

## 0.2.0 — 2026-11-03

### Corrected

- made VOM/sVOM closed-mesh volume invariant to large global coordinate translations by evaluating signed tetrahedral contributions in a local numerical frame;
- made sparse-volume Delaunay, nearest-neighbor spacing, circumsphere, and convex-hull calculations translation invariant by evaluating them in a shared local coordinate frame;
- required all four distinct AoA/SF cardinal entry roles and removed the unrelated incomplete-input centroid fallback;
- corrected Stryker multiline auto-detection;
- restored loadable Surgiplot JSON export and CSV/TSV label round trips;
- made `.tsv` export genuinely tab-delimited;
- rejected empty imports, non-finite coordinates, ambiguous case-insensitive names, and degenerate metric geometry;
- prevented negative repeatability CV for negative means and excluded non-finite observations;
- emitted CLI metric JSON without Rich markup processing.

### Added

- 32 regression tests spanning metrics, transforms, I/O, parser detection, CLI, and study statistics;
- explicit package extras for GUI, rendered-scene, and development profiles;
- `surgiplot-gui` executable alias while retaining `surgiplot_gui` compatibility;
- extensive installation, architecture, data, metric, API, CLI, GUI, validation, development, privacy, and publication documentation;
- explicit primary-source links and software-versus-publication scope statements.
- an MIT software license and automated GitHub CI across Python 3.10–3.13, including lint, tests, package builds, and distribution metadata checks.

### Changed

- standardized public terminology to VOM, sVOM, VoA (Visuo-operative Angle), AoA, SF, and AoE;
- documented AoE as the implemented principal three-point included angle, not a reflex-capable full vectorial protocol;
- reduced mandatory dependencies to core/CLI requirements and moved graphical packages to optional extras;
- updated project metadata, author attribution, classifiers, and repository links;
- promoted the public package classifier from Alpha to Beta for the externally installable `0.2.0` release;
- simplified the Conda environment and installer for the standalone terminal application.

### Repository scope

- the standalone repository now excludes the separately distributed SurgiplotToolkit 3D Slicer extension, model runtimes, checkpoints, generated documentation archives, and workstation caches from the release tree.

## 0.1.0

- initial standalone dataset, metric, CLI, GUI, vendor-import, rendered-scene, and study-provenance implementation.
