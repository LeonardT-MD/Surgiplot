# Development and Release Guide

## 1. Repository scope

This repository is the authoritative home of the standalone Surgiplot Python package, command-line application, and desktop GUI. The 3D Slicer extension is developed and distributed separately in the `SurgiplotToolkit` repository.

Do not vendor a mutable extension checkout, model runtime, weights, patient data, or a virtual environment into this repository.

## 2. Development environment

```bash
git clone https://github.com/LeonardT-MD/Surgiplot.git
cd Surgiplot
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[gui,dev]'
```

Use `.[scene3d,dev]` when modifying PyVista paths. Keep the core testable without GUI extras.

## 3. Source boundaries

| Directory | Responsibility |
|---|---|
| `surgiplot/core` | dataset, geometry primitives, I/O, transforms, study records |
| `surgiplot/metrics` | numerical definitions and result dataclasses |
| `surgiplot/cli` | argument parsing and JSON/table presentation |
| `surgiplot/gui` | Qt state, forms, visualization, user interaction |
| `surgiplot/ai` | legacy-named rendered-scene workspace; no model inference in this release |
| `tests` | deterministic unit and interface regression tests |
| `docs` | public software and scientific contract |

Formulas belong in `metrics`, not in CLI or GUI callbacks. Interfaces should resolve inputs, call the public function, and present its result.

## 4. Coding conventions

- target Python 3.10+ syntax and standard library;
- use type annotations for public and nontrivial internal functions;
- keep units in field names (`_mm`, `_mm2`, `_mm3`, `_deg`);
- validate finite values at public boundaries;
- reject degenerate geometry rather than inventing fallback measurements;
- use explicit anatomical role names in user interfaces;
- do not silently transform coordinate frames;
- attach method/fallback information to debug output;
- prefer small numerical helpers with analytical fixtures.

Run Ruff before committing:

```bash
ruff check surgiplot tests
```

## 5. Tests

```bash
python -m pytest -q
python -m compileall -q surgiplot tests
```

Every metric change should add or update tests for:

- a known analytical geometry;
- invalid or degenerate input;
- rigid translation;
- rotation where relevant;
- scaling law;
- result/dataclass field contract;
- an affected CLI or I/O path.

Bug fixes require a regression fixture that fails before the fix. For numerical bugs, include coordinates that exercise the failure magnitude, not only a minimal nominal case.

## 6. Documentation contract

When a public signature, output field, algorithm, validation rule, or dependency changes, update:

1. docstrings;
2. [Metric Reference](METRICS_REFERENCE.md) or [Data Formats](DATA_FORMATS.md);
3. Python/CLI/GUI documentation as applicable;
4. validation notes and tests;
5. `CHANGELOG.md`.

Canonical terminology is:

- VOM — Volume of Operative Maneuverability;
- sVOM — standardized Volume of Operative Maneuverability;
- VoA — Visuo-operative Angle;
- AoA — Angle of Attack;
- SF — Surgical Freedom;
- AoE — Angle of Exposure.

Do not use an expanded metric name unless it is defined by the cited method or by an explicitly documented software construct.

## 7. GUI development

Test both backends where available:

```bash
SURGIPLOT_3D_BACKEND=safe surgiplot-gui
SURGIPLOT_3D_BACKEND=pyvista surgiplot-gui
```

For CI or headless smoke checks, set the platform plugin supported by the environment, for example:

```bash
QT_QPA_PLATFORM=offscreen SURGIPLOT_3D_BACKEND=safe \
  python -c 'from PySide6.QtWidgets import QApplication; from surgiplot.gui.app import MainWindow; app=QApplication([]); window=MainWindow(); print(window.windowTitle())'
```

GUI changes must not introduce eager Qt/VTK imports into `import surgiplot`.

## 8. Dependency policy

Runtime dependencies in `pyproject.toml` must be directly imported by supported core/CLI paths. Heavy graphical packages belong in extras. Add a dependency only after checking:

- supported Python versions and platforms;
- binary-wheel availability;
- license compatibility;
- maintenance status and advisories;
- import-time and installation cost;
- whether a standard-library or existing dependency solution is sufficient.

Do not add segmentation model weights to package data.

## 9. Versioning

The project uses semantic-version intent while pre-1.0:

- patch: compatible bug/documentation corrections;
- minor: new features or strengthened validation that may reject previously accepted invalid input;
- major/1.0: stable public contract and declared compatibility policy.

The version in `pyproject.toml` is authoritative for built distributions. The source fallback in `surgiplot/__init__.py` must match it.

## 10. Build and wheel inspection

```bash
python -m build
python -m twine check dist/*
python -m zipfile -l dist/surgiplot-*.whl
```

Inspect both wheel and source distribution for:

- unintended binaries;
- local absolute paths;
- caches and editor files;
- clinical data or screenshots;
- companion-repository code;
- missing documentation or package modules.

Package discovery is intentionally restricted to `surgiplot` and `surgiplot.*`. `MANIFEST.in` includes the public documentation and tests in source distributions while pruning the separate toolkit, model-runtime, checkpoint, cache, and local-workspace trees. Do not broaden these patterns without inspecting the resulting archive member list.

## 11. Release sequence

1. update source, tests, docs, version, and changelog;
2. run tests, Ruff, compile checks, and GUI smoke test;
3. build distributions in a clean tree;
4. install the wheel into an isolated environment;
5. run CLI and import smoke tests outside the checkout;
6. review `git diff` and exact staged file list;
7. run privacy, path, credential, and large-file audits;
8. commit with a release-oriented message;
9. push the reviewed branch;
10. create an annotated tag and release only after owner approval;
11. archive the resolved environment and validation output.

## 12. Git hygiene

Before staging:

```bash
git status --short
git diff --stat
```

Stage exact paths rather than `git add .` when the workspace contains local research assets. Then inspect:

```bash
git diff --cached --name-status
git diff --cached --check
git diff --cached
```

Never commit generated clinical outputs. Removing a file in a new commit does not purge it from existing history.

## 13. Pull-request review checklist

- [ ] code belongs to this repository’s scope;
- [ ] mathematical change is documented;
- [ ] regression test demonstrates the change;
- [ ] coordinate and unit assumptions are explicit;
- [ ] failure behavior is explicit;
- [ ] optional dependencies remain optional;
- [ ] CLI JSON remains machine-readable;
- [ ] GUI labels match canonical terminology;
- [ ] no identifying or local-machine content;
- [ ] package builds and wheel installs;
- [ ] changelog updated.

## 14. Contribution and licensing status

At version `0.2.0`, the repository does not yet contain a software license. Copyright therefore remains reserved by default. External contributors and redistributors need an explicit license decision from the owner before reuse terms can be considered clear. Add a reviewed `LICENSE` file and contribution policy before inviting broad external contributions.
