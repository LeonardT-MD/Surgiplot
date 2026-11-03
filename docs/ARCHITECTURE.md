# Architecture and Execution Model

## 1. Design goals

Surgiplot is organized around explicit coordinates rather than viewer state. The architecture prioritizes:

- one numerical implementation per metric;
- inspectable intermediate geometry;
- stable point naming and alias resolution;
- deterministic scripted workflows;
- optional, isolated GUI and rendering dependencies;
- auditable study provenance;
- local processing of research data.

## 2. Package map

```text
surgiplot/
├── __init__.py                 public top-level exports and version
├── api.py                      convenience API and scalar wrappers
├── cli/app.py                  Typer command tree
├── core/
│   ├── dataset.py              point/alias/meta container and metric methods
│   ├── studies.py              provenance, run records, comparison, repeatability
│   ├── geometry/               SVD plane and polygon primitives
│   ├── io/                     normalized, generic, and vendor loaders
│   └── transforms/             affine point transformation
├── metrics/
│   ├── distance_3d.py
│   ├── area_3d.py
│   ├── volume_3d.py
│   ├── aoe.py
│   ├── aoa_sf.py
│   └── vom_voa.py
├── gui/
│   ├── app.py                  desktop application and result renderer
│   └── label_editor.py         point-label editor
├── ai/                         legacy internal namespace for rendered-scene tools
│   ├── scene_picker.py
│   └── scene_reconstruction.py
├── env/                        Conda environment definition
└── scripts/                    installation helpers
```

The `surgiplot.ai` namespace name is retained for compatibility. Its current tracked functionality is a local rendered-scene import, optimization, calibration, and landmark-acquisition workspace; it does not download or execute segmentation weights.

## 3. Dependency boundaries

### Core boundary

The metric core depends on NumPy and SciPy. Pandas and OpenPyXL support normalized tabular I/O. Typer and Rich support the CLI.

The core must remain importable without Qt, VTK, PyVista, or Matplotlib. Plot methods import Matplotlib components lazily and are used only when explicitly called.

### GUI boundary

`surgiplot.gui` imports PySide6. It is installed through the `gui` or `scene3d` extra. GUI code calls metric functions directly and converts result dataclasses into display payloads.

### Rendered-scene boundary

The scene workspace can use:

- PyVista/VTK for native interactive meshes;
- Matplotlib as a safe fallback;
- Pillow for texture loading;
- psutil for optional hardware-profile reporting.

Backend availability affects rendering and interaction performance, not metric formulas.

## 4. Data-flow contract

### 4.1 Acquisition

Inputs enter through:

- a normalized table or JSON dataset;
- a Stryker or Medtronic export;
- generic point text;
- manual arrays;
- a rendered-scene pick.

### 4.2 Normalization

Every path produces a `Dataset` containing:

- `points: dict[str, ndarray(3,)]`;
- `aliases: dict[str, str]` mapping alias to canonical point;
- `meta: dict[str, Any]` for source, coordinate, provenance, and scene metadata.

Point coordinates are finite `float64` values. Canonical names and aliases are case-insensitively unambiguous.

### 4.3 Resolution

Metric functions accept either direct coordinates or dataset names. Dataset resolution supports:

- exact canonical names;
- case-insensitive canonical names;
- aliases;
- numeric shorthand such as `1` → `point_1`;
- range expressions such as `point_1-4`.

The resolved coordinates are passed to the same numerical function regardless of interface.

### 4.4 Computation

Each metric returns a dataclass. Scalar fields constitute the stable public result. When `return_debug=True`, a debug dictionary contains intermediate arrays, method indicators, and plot geometry.

Debug payloads are designed for verification and visualization. They are not guaranteed to be a minimal or permanently frozen serialization API.

### 4.5 Export and study aggregation

Dataset exports preserve points and aliases. Study helpers flatten result payloads into `MetricRunRecord` rows, then generate comparison or repeatability tables without altering the source `Dataset`.

## 5. Public API layers

### Top-level package

`import surgiplot as sp` exposes the canonical user API. This includes all metric functions, `Dataset`, I/O helpers, and study helpers.

### Dataset-bound API

`Dataset` methods delegate to the metric modules. For example, `dataset.AREA_3D(...)` and `surgiplot.AREA_3D(data=dataset, ...)` execute the same function.

### Scalar wrappers

`SF`, `VOA`, and `SVOM` return a scalar selected from the corresponding composite result. `AOA` and `VOM` retain their composite result objects for access to related outputs.

### CLI

The CLI resolves files and point names, calls the public API, and emits JSON for metric commands. No formula is reimplemented in the command functions.

### GUI

The GUI owns UI state, input selection, scene rendering, and result presentation. It calls the same public metric functions and uses debug geometry for visualization.

## 6. Coordinate semantics

Surgiplot does not automatically transform between RAS, LPS, RPI, scanner, navigation, and model coordinates. Vendor parsers retain the selected coordinate-system label in metadata. All points used in one metric must already share a frame.

The `space` and `transforms` parameters present in several metric signatures are reserved compatibility fields. The current metric functions do not apply `transforms` automatically. Callers requiring a transformation must use `apply_affine` or otherwise normalize coordinates before computation.

## 7. Numerical stability strategy

The code applies the following guards:

- finite-coordinate validation at dataset and metric boundaries;
- uniqueness checks for polygon and volume inputs;
- rank checks before plane or volume reconstruction;
- explicit zero-length-vector rejection for angular metrics;
- clipping of inverse-cosine arguments to `[-1, 1]`;
- local-reference evaluation of closed-mesh volume for translation stability;
- explicit alpha-shape fallback method reporting.

The local-reference volume calculation is essential for navigation or image coordinates far from the global origin. Signed tetrahedral contributions are computed after subtracting a mesh reference centroid, limiting catastrophic cancellation without changing the geometry.

## 8. Error model

Public computation and loading functions raise:

- `FileNotFoundError` for missing files;
- `KeyError` for unresolved or conflicting point names and aliases;
- `ValueError` for malformed files, invalid roles, non-finite coordinates, degenerate geometry, or unsupported values;
- `NotImplementedError` for an explicitly requested Brainlab parser, which is declared but not implemented.

The GUI catches user-facing errors at interaction boundaries and displays them. Library and CLI layers allow errors to propagate so failed analyses are not silently converted into plausible values.

## 9. Backward compatibility

The project is pre-1.0 and may strengthen validation between minor releases. Version `0.2.0` intentionally rejects incomplete AoA/SF cardinal inputs that older code converted into a generic centroid fallback.

The GUI accepts the legacy rendered-result label `VOM / VoA` when reading an existing scene payload, while new UI and output labels use `VOM / sVOM / VoA`.

## 10. Extension boundary

SurgiplotToolkit may vendor a snapshot of the core so it can operate inside 3D Slicer’s Python runtime. The standalone repository remains authoritative for terminal software development. Synchronization between repositories must be explicit, versioned, and followed by the test suite in both environments.
