# Getting Started: Installation, Terminal Syntax, and Complete Workflows

This guide takes a new user from download to a verified result. It covers the command-line interface (CLI), desktop application, and Python API using a synthetic dataset that contains no patient information.

For exact formulas and validation rules, use the [Metric Reference](METRICS_REFERENCE.md). For every command option, use the [CLI Reference](CLI_REFERENCE.md).

## 1. Choose an interface

All interfaces call the same numerical metric implementation.

| Interface | Best suited to | Launch |
|---|---|---|
| Terminal/CLI | reproducible commands, batch work, scripts | `surgiplot --help` |
| Desktop GUI | interactive import, role selection, and visual review | `surgiplot-gui` |
| Python API | notebooks, custom pipelines, and study automation | `import surgiplot as sp` |

Surgiplot is research software, not a medical device. Work with pseudonymized data and verify every landmark, coordinate frame, unit, and polygon order before accepting a result.

## 2. Requirements

- CPython 3.10 or newer;
- a terminal: Terminal on macOS/Linux or PowerShell on Windows;
- a graphical desktop session for the optional GUI;
- input coordinates in one consistent spatial frame.

Surgiplot assumes coordinate values are millimeters and consequently reports distances in `mm`, areas in `mm²`, and volumes in `mm³`. It does not infer or convert units automatically.

Check Python before installing:

```bash
python3 --version
```

On Windows, use this when `python3` is unavailable:

```powershell
py --version
```

## 3. Install Surgiplot

Use an isolated environment so Surgiplot does not change unrelated Python software.

### 3.1 macOS or Linux

```bash
python3 -m venv surgiplot-env
source surgiplot-env/bin/activate
python -m pip install --upgrade pip
python -m pip install "https://github.com/LeonardT-MD/Surgiplot/releases/download/v0.2.0/surgiplot-0.2.0-py3-none-any.whl"
```

### 3.2 Windows PowerShell

```powershell
py -3.11 -m venv surgiplot-env
.\surgiplot-env\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "https://github.com/LeonardT-MD/Surgiplot/releases/download/v0.2.0/surgiplot-0.2.0-py3-none-any.whl"
```

If local PowerShell policy prevents activation, invoke the environment interpreter directly instead of weakening machine-wide security controls:

```powershell
.\surgiplot-env\Scripts\python.exe -m surgiplot.cli.app --help
```

### 3.3 Install the desktop GUI

The release wheel installs the core and CLI. To install an optional interface profile from the source checkout:

```bash
git clone https://github.com/LeonardT-MD/Surgiplot.git
cd Surgiplot
python -m pip install ".[gui]"
```

For the GUI plus PyVista/VTK rendered-scene support:

```bash
python -m pip install ".[scene3d]"
```

The full rendered-scene profile is substantially larger than the core installation. Start with `gui` unless interactive mesh rendering is required.

### 3.4 Verify the installation

```bash
surgiplot version
surgiplot --help
python -c "import surgiplot; print(surgiplot.__version__)"
```

The expected release version is `0.2.0`. If the shell cannot find `surgiplot`, confirm that `surgiplot-env` is active or use:

```bash
python -m surgiplot.cli.app --help
```

## 4. Understand terminal syntax

The general command structure is:

```text
surgiplot <command-group> <command> <input-file> [required options] [optional options]
```

Examples:

```bash
surgiplot dataset summary points.csv
surgiplot metric distance points.csv --A entry --B target
```

Syntax rules:

- words without `--` are positional arguments and their order matters;
- options begin with `--` and are followed by their value;
- point names containing spaces must be quoted, for example `--pivot "target centroid"`;
- `--A`, `--B`, and `--C` are intentionally uppercase for distance and AoE;
- lists accept comma-separated names such as `entry_1,entry_2,entry_3`;
- inclusive numeric ranges such as `entry_1-4` expand to `entry_1,entry_2,entry_3,entry_4`;
- range order is retained, which is essential for polygon perimeters;
- a backslash at the end of a Bash line continues the same command on the next line;
- run any command with `--help` to see its current options.

Help is hierarchical:

```bash
surgiplot --help
surgiplot dataset --help
surgiplot dataset export --help
surgiplot metric --help
surgiplot metric vom-voa --help
```

The command tree is:

```text
surgiplot
├── version
├── dataset
│   ├── summary
│   ├── labels
│   └── export
└── metric
    ├── distance
    ├── area
    ├── volume
    ├── aoe
    ├── aoa-sf
    ├── aoa
    ├── sf
    ├── vom-voa
    ├── vom
    ├── voa
    └── svom
```

## 5. Obtain the synthetic tutorial dataset

When working in a cloned repository, the file is already available at:

```text
examples/tutorial_points.csv
```

To download only that file on macOS or Linux:

```bash
curl -L "https://raw.githubusercontent.com/LeonardT-MD/Surgiplot/main/examples/tutorial_points.csv" -o tutorial_points.csv
```

On Windows PowerShell:

```powershell
Invoke-WebRequest "https://raw.githubusercontent.com/LeonardT-MD/Surgiplot/main/examples/tutorial_points.csv" -OutFile tutorial_points.csv
```

In the commands below, replace `examples/tutorial_points.csv` with `tutorial_points.csv` if the file was downloaded separately.

## 6. Complete terminal walkthrough

Run these commands from the cloned repository root. Each metric example uses a known part of the synthetic geometry.

### Step 1 — Inspect the dataset

```bash
surgiplot dataset summary examples/tutorial_points.csv
```

Expected essentials:

```text
Points: 19
Source: database
```

List canonical point names and aliases:

```bash
surgiplot dataset labels examples/tutorial_points.csv
```

Do this before a computation when the incoming names are unfamiliar.

### Step 2 — Export a normalized reusable dataset

```bash
surgiplot dataset export examples/tutorial_points.csv tutorial_normalized.json
```

The normalized JSON preserves point names, coordinates, labels, metadata, and provenance fields supported by the public dataset model. It can be used in every following command in place of the CSV.

### Step 3 — Compute a three-dimensional distance

```bash
surgiplot metric distance examples/tutorial_points.csv --A entry_center --B target_center
```

Expected JSON:

```json
{
  "distance_mm": 10.0
}
```

### Step 4 — Compute an ordered polygon area

```bash
surgiplot metric area examples/tutorial_points.csv --polygon entry_1-4
```

Expected JSON:

```json
{
  "area_mm2": 100.0
}
```

The point sequence must traverse the boundary. An unordered or crossing list is not corrected automatically.

### Step 5 — Compute a sparse-point enclosed volume

```bash
surgiplot metric volume examples/tutorial_points.csv --points volume_1-4
```

Expected JSON:

```json
{
  "volume_mm3": 1.3333333333333335
}
```

This is a sparse-point reconstruction estimate, not a segmentation-derived ground-truth volume.

### Step 6 — Compute the principal three-point AoE

```bash
surgiplot metric aoe examples/tutorial_points.csv --A medial --B pivot --C lateral
```

Expected result:

```json
{
  "AoE_deg": 53.13010235415599
}
```

`B` is the pivot. This implementation reports the principal included angle in `[0°, 180°]`; it is not a reflex-capable full circular reconstruction.

### Step 7 — Compute AoA and Surgical Freedom

```bash
surgiplot metric aoa-sf examples/tutorial_points.csv \
  --cranial cranial \
  --caudal caudal \
  --medial medial \
  --lateral lateral \
  --pivot pivot \
  --sf-rescale-radius-mm 10
```

Expected result:

```json
{
  "AoA_horizontal_deg": 53.13010235415599,
  "AoA_vertical_deg": 53.13010235415599,
  "SF_area_mm2": 50.0,
  "SF_area_standardized_mm2": 40.00000000000001
}
```

All four cardinal entry roles are mandatory and must resolve to distinct points. The standardized SF radius must be finite and positive.

### Step 8 — Compute VOM, sVOM, and VoA

```bash
surgiplot metric vom-voa examples/tutorial_points.csv \
  --entry entry_1-4 \
  --target target_1-4 \
  --stand-dist 5
```

Expected result, within ordinary floating-point tolerance:

```json
{
  "VOM_mm3": 519.7624309792972,
  "VoA_deg": 90.0,
  "sVOM_mm3": 154.92918615729863
}
```

Entry and target values are ordered polygon perimeters. `--stand-dist` is the positive distal standardization distance used for sVOM.

### Step 9 — Save results for a reproducible pipeline

Shell redirection writes the JSON output to a file:

```bash
surgiplot metric vom-voa examples/tutorial_points.csv \
  --entry entry_1-4 \
  --target target_1-4 \
  --stand-dist 5 \
  > tutorial_vom_result.json
```

Check that failures stop a Bash pipeline:

```bash
set -euo pipefail
surgiplot dataset summary examples/tutorial_points.csv
surgiplot metric distance examples/tutorial_points.csv --A entry_center --B target_center > distance_result.json
```

Preserve the input dataset, command, result JSON, Surgiplot version, Python version, units, coordinate frame, and standardization parameters together.

## 7. Use your own dataset

The smallest normalized CSV has this form:

```csv
name,x,y,z
entry,12.5,-4.2,31.0
target,9.1,-8.7,19.4
```

An optional `labels` column provides comma-separated aliases. For complete schemas, vendor formats, coordinate semantics, and round trips, see [Data Formats](DATA_FORMATS.md).

Recommended sequence:

1. copy the source data outside the repository and pseudonymize it;
2. determine the coordinate frame and units from the source system;
3. run `surgiplot dataset summary SOURCE`;
4. run `surgiplot dataset labels SOURCE`;
5. normalize it with `surgiplot dataset export SOURCE normalized.json`;
6. review names and coordinates;
7. compute one metric with explicit point roles;
8. visually verify geometry in the GUI or Python debug output;
9. save the normalized input, exact command, result, and software version.

For a navigation export, declare or detect its format:

```bash
surgiplot dataset summary navigation-export.txt \
  --source navigation \
  --kind navigation \
  --navigation-format auto
```

Supported automatic vendor paths currently include Stryker text and Medtronic JSON. Brainlab is reserved but not implemented in `0.2.0`.

## 8. Desktop GUI walkthrough

Install the `gui` or `scene3d` profile, then launch:

```bash
surgiplot-gui
```

Use this sequence:

1. open **Database**;
2. choose the import action matching the source: normalized dataset, navigation export, or generic points;
3. inspect the table, names, labels, coordinates, and dataset preview;
4. edit labels only after confirming the intended point identity;
5. open **Analysis** and select a metric family;
6. assign every displayed role; preserve perimeter order for area and VOM polygons;
7. enter the standard distance or radius when the metric requires one;
8. compute the metric;
9. inspect scalar fields, warnings, debug geometry, and the visual reconstruction;
10. use **Copy result JSON** and separately retain the normalized source dataset.

For AoA/SF, assign cranial, caudal, medial, lateral, and pivot. For VOM/sVOM/VoA, provide ordered entry and target perimeters. Hidden visual layers remain present in the scene and do not change the coordinates already selected for a metric.

If PyVista/VTK has a graphics problem, start the GUI with the safe Matplotlib backend:

```bash
SURGIPLOT_3D_BACKEND=safe surgiplot-gui
```

Windows PowerShell equivalent:

```powershell
$env:SURGIPLOT_3D_BACKEND = "safe"
surgiplot-gui
```

See the [Desktop GUI Guide](GUI_GUIDE.md) for rendered-scene acquisition, visualizer controls, and failure recovery.

## 9. Python walkthrough

Create `tutorial.py` beside the `examples` directory:

```python
import surgiplot as sp

dataset = sp.load("examples/tutorial_points.csv", kind="database")

distance = sp.DISTANCE_3D(
    data=dataset,
    A="entry_center",
    B="target_center",
)

aoa_sf = sp.AOA_SF(
    data=dataset,
    entry=["cranial", "caudal", "medial", "lateral"],
    target="pivot",
    sf_rescale_radius_mm=10.0,
    return_debug=True,
)

corridor = sp.VOM_VOA(
    data=dataset,
    entry="entry_1-4",
    target="target_1-4",
    stand_dist=5.0,
    return_debug=True,
)

print(distance.distance_mm)
print(
    aoa_sf.aoa_vertical_deg,
    aoa_sf.aoa_horizontal_deg,
    aoa_sf.sf_entry_area_mm2,
)
print(corridor.vom_mm3, corridor.svom_mm3, corridor.voa_deg)
```

Run it:

```bash
python tutorial.py
```

Public functions return typed result dataclasses. `return_debug=True` adds the resolved geometry and method information required for inspection. See the [Python API](PYTHON_API.md) for constructors, convenience methods, result fields, and study records.

## 10. Common failures

### `surgiplot: command not found`

Activate the installation environment, then inspect:

```bash
python -m pip show surgiplot
python -c "import sys; print(sys.executable)"
python -m surgiplot.cli.app --help
```

### A point cannot be resolved

Run `surgiplot dataset labels FILE` and use a canonical name or unambiguous alias. Quote names containing spaces.

### Area or VOM is degenerate

Check duplicates, collinearity, self-intersection, and perimeter order. Plot or inspect the selected points rather than rearranging them until a command happens to succeed.

### AoA/SF refuses to compute

Supply all four distinct cardinal entry roles and a pivot. No entry may coincide with the pivot.

### Sparse volume uses a convex hull

Review point density and surface coverage. A convex hull can overestimate a concave structure; retain the reported reconstruction method with the result.

### GUI starts without a working 3D scene

Use `SURGIPLOT_3D_BACKEND=safe`, then inspect the PyVista, VTK, OpenGL, and graphics-driver installation separately from the metric core.

## 11. Where to continue

| Need | Document |
|---|---|
| every CLI option | [CLI Reference](CLI_REFERENCE.md) |
| exact formulas and required roles | [Metric Reference](METRICS_REFERENCE.md) |
| input schemas and vendor imports | [Data Formats](DATA_FORMATS.md) |
| Python objects and return fields | [Python API](PYTHON_API.md) |
| complete GUI controls | [Desktop GUI Guide](GUI_GUIDE.md) |
| installation variants and removal | [Installation and Deployment](INSTALLATION.md) |
| analytical fixtures and tolerances | [Validation](VALIDATION.md) |
| privacy and PHI handling | [Security and Privacy](SECURITY_AND_PRIVACY.md) |

When documentation and a tagged implementation differ, the tagged source code and automated tests are authoritative for that release.
