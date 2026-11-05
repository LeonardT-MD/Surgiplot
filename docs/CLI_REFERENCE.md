# Command-Line Interface Reference

## 1. Invocation

After installation:

```bash
surgiplot --help
surgiplot version
```

Module invocation is equivalent:

```bash
python -m surgiplot.cli.app --help
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
    ├── vom-voa
    ├── vom
    ├── voa
    ├── svom
    ├── aoa-sf
    ├── aoa
    ├── sf
    ├── aoe
    ├── distance
    ├── area
    └── volume
```

The hidden legacy `launch` command intentionally fails with migration guidance. Use `surgiplot-gui` for the desktop interface.

## 2. Shared loader options

Dataset and metric commands accept these option families:

| Option | Default | Values |
|---|---|---|
| `--source` | empty | `navigation`, `photogrammetry`, `scanner`, `database` |
| `--kind` | `auto` | `auto`, `navigation`, `database`, `generic` |
| `--navigation-format` | `auto` | `auto`, `stryker`, `medtronic`, `brainlab`, `generic` |

`source` records provenance and does not transform coordinates. `brainlab` is a reserved explicit selection that is not implemented in version `0.3.0`.

## 3. Name-list syntax

Comma-separated role lists accept names, aliases, and inclusive ranges:

```text
entry_1,entry_2,entry_3
point_1-8
point_8:1
1-4
```

Range order is retained. This is important for polygon perimeters.

Shell quoting is recommended whenever an argument contains spaces:

```bash
--target "tumor centroid"
```

## 4. Dataset commands

### 4.1 Summary

```bash
surgiplot dataset summary points.json
surgiplot dataset summary annotations.txt \
  --source navigation \
  --kind navigation \
  --navigation-format stryker
```

The human-readable table reports point count, alias count, source, coordinate-system label, and input path.

### 4.2 Labels and renaming

```bash
surgiplot dataset labels points.json \
  --rename point_1=cranial_limit \
  --label cranial_limit=cranial,superior \
  --label point_2=caudal,inferior \
  --out labeled.json
```

Options may be repeated:

- `--rename OLD=NEW` renames a canonical point;
- `--label POINT=label1,label2` replaces the labels for a point;
- `--out` writes JSON, CSV, or TSV.

The first listed label becomes the preferred display label. Conflicting case-insensitive aliases fail.

### 4.3 Export

```bash
surgiplot dataset export vendor-export.txt normalized.json \
  --source navigation \
  --kind navigation \
  --navigation-format auto
```

Output suffix determines the format: `.json`, `.csv`, or `.tsv`.

## 5. Metric commands

Metric commands emit machine-readable, indented JSON to standard output. JSON bypasses Rich markup processing so brackets and strings remain unmodified.

### 5.1 VOM, sVOM, and VoA

Combined:

```bash
surgiplot metric vom-voa points.json \
  --entry entry_1-8 \
  --target target_1-8 \
  --stand-dist 10
```

Output keys:

```json
{
  "VoA_deg": 0.0,
  "VOM_mm3": 1000.0,
  "sVOM_mm3": 500.0,
  "method_version": "3d-loft-2.0",
  "ellipse_frame_correspondence_policy": "minimum_rotation_with_axis_permutation",
  "entry_frame_alignment_operation": "swap_v_minus_u",
  "entry_frame_alignment_swapped_axes": true,
  "entry_frame_alignment_rotation_deg": 20.0
}
```

The `vom` command currently returns the same composite payload. Retain the method version and frame-correspondence fields with exported results because the selected axis correspondence can change both the rendered loft and its numerical volume. Scalar selectors are:

```bash
surgiplot metric voa points.json --entry entry_1-8 --target target_1-8
surgiplot metric svom points.json --entry entry_1-8 --target target_1-8 --stand-dist 10
```

The requested standard distance must be finite and positive. It is clipped to the entry-target centroid distance when longer than the corridor.

### 5.2 AoA and SF

```bash
surgiplot metric aoa-sf points.json \
  --cranial cranial \
  --caudal caudal \
  --medial medial \
  --lateral lateral \
  --pivot target \
  --sf-rescale-radius-mm 20
```

Output keys:

```json
{
  "AoA_horizontal_deg": 53.1,
  "AoA_vertical_deg": 47.8,
  "SF_area_mm2": 245.0,
  "SF_area_standardized_mm2": 180.0
}
```

All four cardinal options are mandatory and must resolve to distinct entry points. `aoa` returns the same composite payload. `sf` returns only `SF_area_mm2`.

### 5.3 Principal included AoE

```bash
surgiplot metric aoe points.json --A limit_a --B pivot --C limit_c
```

`--A`, `--B`, and `--C` use uppercase option spelling. The result is bounded to `[0, 180]` degrees and is not a complete reflex-capable vectorial AoE reconstruction.

### 5.4 Distance

```bash
surgiplot metric distance points.json --A entry --B target
```

Output: `distance_mm`.

### 5.5 Area

```bash
surgiplot metric area points.json --polygon boundary_1-12
```

Output: `area_mm2`. The list must follow the intended perimeter.

### 5.6 Volume

```bash
surgiplot metric volume points.json --points surface_1-40
```

Output: `volume_mm3`. At least four distinct non-coplanar samples are required.

## 6. Automation examples

### 6.1 Capture one result

```bash
surgiplot metric distance points.json --A entry --B target > result.json
```

### 6.2 Extract a scalar with `jq`

```bash
VOLUME=$(surgiplot metric vom-voa points.json \
  --entry entry_1-8 \
  --target target_1-8 | jq -r '.VOM_mm3')
```

### 6.3 Fail-fast shell script

```bash
#!/usr/bin/env bash
set -euo pipefail

surgiplot dataset export raw-points.txt normalized.json --kind generic
surgiplot metric area normalized.json --polygon point_1-8 > area.json
```

Metric or loader validation failures return a nonzero process status through Typer/Python exception handling. A pipeline should stop and investigate rather than substitute a missing result.

## 7. Reproducibility record

For each scripted run, preserve:

```bash
surgiplot version
python --version
git rev-parse HEAD
python -m pip freeze
```

Also retain the normalized input, coordinate-system declaration, units, exact role arguments, standardization parameters, and generated JSON.

## 8. Limitations

- CLI metric commands do not expose debug geometry; use the Python API or GUI for visual verification.
- The CLI does not transform coordinate frames.
- The CLI does not infer polygon order.
- Brainlab parsing is not implemented.
- Study-level record aggregation is currently a Python API feature.
