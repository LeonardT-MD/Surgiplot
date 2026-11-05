# Validation and Quality Assurance

## 1. Validation scope

Surgiplot validation separates four questions:

1. **software correctness** — does the implementation satisfy its documented formulas and invariants?;
2. **numerical stability** — do equivalent geometries remain equivalent under coordinate translation, rotation, and scale?;
3. **workflow integrity** — do import, labeling, export, and CLI paths preserve the intended dataset?;
4. **scientific validity** — does the modeled quantity answer the research question under the acquisition protocol?

Automated tests address the first three. Scientific validity additionally requires anatomical ground truth, protocol-specific experiments, and operator-repeatability studies.

## 2. Automated suite

From a development install:

```bash
python -m pytest -q
ruff check surgiplot tests
python -m compileall -q surgiplot tests
```

The version `0.3.0` regression suite contains 37 tests covering metrics, transforms, data I/O, CLI behavior, vendor detection, study statistics, and the revised VOM/sVOM/VoA geometry.

## 3. Analytical metric fixtures

### Distance

A 3-4-5 coordinate fixture verifies exact Euclidean distance:

```text
A = (0, 0, 0)
B = (3, 4, 0)
expected = 5 mm
```

### Area

An ordered rectangle verifies the best-fit-plane shoelace result. Degenerate collinear input must raise `ValueError` rather than return a plausible zero.

### Volume

A tetrahedron verifies the determinant-derived reference volume. Coplanar and duplicate-only clouds must fail before Delaunay reconstruction.

### AoE

Orthogonal arms verify a 90-degree principal included angle. A coincident endpoint and pivot must fail because its arm has zero length.

### AoA and SF

A symmetric four-cardinal fixture verifies vertical/horizontal angles and quadrilateral area. Tests also require:

- exactly four cardinal entries;
- four distinct entry coordinates;
- no entry coincident with pivot;
- a finite positive standardization radius.

### VOM, sVOM, and VoA

Parallel, constant-section polygons provide an analytical prism comparison. Angular ring resolution is selected to limit regular-polygon area error to `1e-4`. Longitudinal refinement must satisfy `1e-4` relative plus `1e-6 mm³` absolute volume convergence.

The suite also verifies:

- rigid translation to million-unit coordinate offsets does not materially change volume;
- standard distance larger than the corridor is clipped;
- clipped sVOM reuses the full VOM mesh and equals VOM exactly;
- perpendicular and tangential approaches return `90°` and `0°` VoA, respectively;
- contour reversal and cyclic reindexing do not change VOM after frame alignment;
- a `+70°` entry major-axis fit selects the geometrically equivalent `−20°` axis-swapped frame, records a 20° correspondence rotation, and avoids the forced large-twist loft;
- a single gross contour outlier is identified and excluded from area-constrained ellipse fitting;
- degenerate or coincident cross-sections fail explicitly.

## 4. I/O fixtures

The test suite round-trips one dataset through:

- Surgiplot JSON;
- CSV;
- true tab-delimited TSV.

It checks canonical names, coordinates, labels, and metadata semantics appropriate to each format. Additional tests cover:

- Stryker multiline block recognition;
- empty-file rejection;
- generic point parsing;
- JSON schema loading;
- non-finite coordinate rejection;
- case-insensitive alias ambiguity.

CSV/TSV cannot round-trip arbitrary nested metadata by design. This is an interface limitation, not a failed test.

## 5. CLI fixtures

CLI tests use Typer's test runner and verify:

- metric output parses as JSON without Rich markup alteration;
- range expansion is inclusive and preserves direction;
- invalid ranges or missing role inputs fail;
- dataset commands call the same loader and metric code as the public Python API.

## 6. Study-statistics fixtures

Repeatability tests verify grouping and field-wise statistics:

- group key: subject, anatomical target, metric family;
- population standard deviation;
- coefficient of variation based on absolute mean;
- non-finite values excluded;
- nonnumeric fields excluded from numerical summaries.

For a mean near zero, relative CV is not interpretable. The implementation reports zero for `CV_percent`; users must examine SD and range.

## 7. Numerical tolerances

Tests use exact comparison only where integer-representable geometry permits it. Otherwise tolerances must be justified by the numerical method:

| Operation | Principal error source | Validation approach |
|---|---|---|
| distance/angles | floating-point normalization | strict absolute tolerance |
| projected area | SVD basis and shoelace arithmetic | relative/absolute tolerance |
| VOM/sVOM | sampled ring discretization and triangulation | `5e-4` relative fixture tolerance |
| alpha-shape volume | Delaunay/Qhull and threshold selection | structural/method assertions plus bounded fixtures |

Tolerances must not be widened solely to make a failing regression pass. Investigate the source of change and document any deliberate algorithm revision.

## 8. Manual GUI acceptance test

Before release, perform a clean-environment GUI check:

1. launch with the safe backend;
2. import normalized JSON;
3. inspect and edit a label;
4. compute every metric family;
5. verify that all AoA cardinal controls are visible;
6. inspect debug geometry and layer visibility;
7. copy result JSON;
8. export a visualization PNG;
9. open and close detachable windows;
10. repeat with PyVista when supported.

The safe-backend test separates GUI logic from graphics-driver failures.

## 9. Packaging acceptance test

Build and inspect distributions:

```bash
python -m build
python -m twine check dist/*
```

Install the wheel into a new environment, not the source checkout:

```bash
python -m venv /tmp/surgiplot-release-test
/tmp/surgiplot-release-test/bin/python -m pip install dist/surgiplot-*.whl
/tmp/surgiplot-release-test/bin/surgiplot version
```

On Windows, use an explicit temporary environment path and the corresponding `Scripts` executables.

Acceptance requires:

- package imports outside the repository;
- console entry points exist;
- version matches release metadata;
- no source-only files are required at runtime;
- no patient data, local paths, caches, or model weights are present in the archive.

## 10. Scientific validation recommendations

For each metric and anatomical application, a formal validation study should include:

- a prespecified landmark and perimeter protocol;
- physical or synthetic phantoms with known geometry;
- comparison to an independent reference implementation;
- sensitivity to point density and polygon order;
- sensitivity to coordinate noise;
- inter- and intra-operator repeatability;
- cross-device or cross-modality agreement;
- reconstruction-method stratification for sparse volume;
- explicit treatment of excluded or failed cases.

VOM/sVOM should additionally evaluate ellipse substitution and loft sampling. AoA/SF should evaluate cardinal-role placement. The three-point AoE implementation must be validated as a principal-angle measurement and must not be generalized to a reflex-angle protocol.

## 11. Release acceptance criteria

A release is acceptable when:

- tests, lint, compile checks, and package build pass;
- a wheel installs and runs in an isolated location;
- public documentation matches the source signatures;
- canonical metric names are consistent;
- current tracked files pass privacy and credential scans;
- the working tree contains no accidentally staged local assets;
- algorithm changes are recorded in the changelog;
- known limitations are not hidden by interface language.

## 12. Known validation boundary

Passing this suite does not certify clinical safety, establish regulatory compliance, prove anatomical landmark correctness, or validate use for treatment decisions. Surgiplot is research software and requires application-specific validation.
