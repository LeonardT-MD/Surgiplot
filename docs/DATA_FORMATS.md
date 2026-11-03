# Data Formats and Coordinate Semantics

## 1. Dataset object

The canonical data structure is `surgiplot.core.dataset.Dataset`:

```python
Dataset(
    points={"point_1": np.array([x, y, z])},
    aliases={"cranial": "point_1"},
    meta={"source": "navigation", "coordinate_system": "LPS"},
)
```

### Invariants

- every canonical point name is non-empty;
- canonical names are unique under case-insensitive comparison;
- every coordinate is finite and has exactly three components;
- every alias resolves to an existing canonical point;
- aliases cannot shadow canonical names under case-insensitive comparison;
- one metric call operates in one coordinate frame.

The `Dataset` does not infer units or transform coordinate systems.

## 2. Canonical names, labels, and aliases

Canonical names are stable internal identifiers. Aliases are semantic labels used for lookup. A point can have multiple aliases and one preferred label.

```python
dataset.set_labels("point_1", ["cranial", "superior limit"])
dataset.get("cranial")
dataset.get("CRANIAL")
```

Both calls resolve to `point_1`. Ambiguous labels are rejected rather than resolved by insertion order.

### Name shorthand

When the canonical point exists:

- `1` resolves to `point_1`;
- `Point-1`, `point 1`, and `point_1` resolve to `point_1`;
- matching is case-insensitive.

### Range syntax

The Python dataset resolver and CLI accept inclusive ranges:

```text
point_1-4
point_1:4
1-4
point_4-1
```

Descending ranges preserve descending order. Polygon metrics use the resolved order; range expansion does not geometrically sort vertices.

## 3. Normalized CSV and TSV

Required columns are case-insensitive:

| Column | Type | Meaning |
|---|---|---|
| `name` | string | canonical point name |
| `x` | finite float | first coordinate |
| `y` | finite float | second coordinate |
| `z` | finite float | third coordinate |

Optional:

| Column | Type | Meaning |
|---|---|---|
| `labels` | comma-separated string | semantic aliases; first label is preferred |

Example:

```csv
name,x,y,z,labels
point_1,12.1,-4.8,31.0,"cranial,superior limit"
point_2,11.7,-9.2,29.9,"caudal,inferior limit"
```

`Dataset.export_csv("file.tsv")` writes a real tab-delimited file. `.csv` uses commas. Both are loadable with `kind="database"` or auto detection.

Empty tables, missing required columns, duplicate names, non-finite coordinates, and invalid numeric values are rejected.

## 4. XLSX

XLSX uses the same columns as normalized CSV. The first worksheet read by Pandas is used. Formula evaluation depends on the value stored in the workbook; Surgiplot does not run Microsoft Excel to recalculate formulas.

For controlled research ingestion:

- convert formulas to stable values before import;
- avoid merged cells and multi-row headers;
- use one row per landmark;
- retain the original workbook outside the repository;
- export a normalized JSON copy for computational provenance.

## 5. Surgiplot JSON

`Dataset.export_json` writes:

```json
{
  "schema": "surgiplot.dataset",
  "schema_version": 1,
  "meta": {
    "source": "database",
    "coordinate_system": "LPS"
  },
  "points": {
    "point_1": [12.1, -4.8, 31.0]
  },
  "aliases": {
    "cranial": "point_1"
  }
}
```

NumPy arrays and scalar types in metadata are converted to JSON-compatible lists and numbers. Non-finite metadata numbers and unknown metadata objects are converted to strings so the file remains strict JSON. Point coordinates themselves must be finite and are never stringified.

The loader validates that `points` is an object and that aliases target existing points. `schema` and `schema_version` are currently informational; future incompatible schemas must increment the version.

## 6. Generic point text

Accepted line forms:

```text
name,1.0,2.0,3.0
name<TAB>1.0<TAB>2.0<TAB>3.0
name 1.0 2.0 3.0
```

Blank lines and lines beginning with `#` or `//` are ignored. A header row is skipped because its coordinate fields cannot be parsed as floats. At least one valid point is required.

Generic records are normalized to sequential canonical names (`point_1`, `point_2`, …); original names become aliases. This makes heterogeneous input naming predictable while retaining semantic lookup.

## 7. Stryker annotations

The parser recognizes repeated annotation blocks such as:

```text
[ANNOTATIONPOINT_1]
name=#138
point=1.0,2.0,3.0
```

Numeric annotation identifiers are preserved:

```text
#138 → point_138
```

Metadata records:

- `source = navigation`;
- `vendor = stryker`;
- `coordinate_system = LPS`;
- source path and raw suffix.

Files containing vendor markers but no valid annotation points fail with a specific error. Duplicate numeric identifiers are not silently overwritten; the first parsed point is retained by the current Stryker conversion logic. Such files should be reviewed because duplicate IDs indicate an ambiguous export.

## 8. Medtronic annotations and plans

The parser expects JSON with `annotations` grouped by coordinate system:

```json
{
  "annotations": {
    "LPS": [
      {"name": "Target", "point": {"x": 1, "y": 2, "z": 3}}
    ]
  }
}
```

Supported coordinate-system keys are `LPS`, `RPI`, and `DCM`. Auto selection uses the first populated group in that order. Optional plans are imported as two points named `<plan> entry` and `<plan> target`.

Medtronic records become sequential canonical points; original annotation or plan names become aliases. The selected coordinate-system key is retained in metadata.

## 9. Brainlab status

`navigation_format="brainlab"` is reserved but raises `NotImplementedError`. Auto detection does not claim Brainlab support. Brainlab exports must be converted to a normalized table or generic point file using a validated external procedure before import.

## 10. Coordinate systems and units

### Coordinate-system label

Metadata may state `LPS`, `RPI`, `DCM`, or a custom frame. The label does not perform a transformation. Mixing frames in one metric can produce numerically valid but anatomically meaningless results.

### Units

Metric output labels assume millimeters. If coordinates are in meters, centimeters, voxels, or arbitrary model units, scale them before computation.

Scaling consequences:

- distance scales by (s);
- area scales by (s^2);
- volume scales by (s^3);
- angles are invariant under positive uniform scale.

### Affine transformation

Use:

```python
from surgiplot.core.transforms.spaces import apply_affine

transformed = apply_affine(points, affine_4x4)
```

The matrix is applied to homogeneous row vectors using `homogeneous @ affine.T`. Validate orientation conventions before use.

## 11. Source and provenance metadata

Loader-level `source` values are:

- `navigation`;
- `photogrammetry`;
- `scanner`;
- `database`;
- empty/unspecified.

Rendered-scene workflows may store additional internal source values directly in metadata. Study-level provenance is separate and should include subject, target, modality, operator, session, source context, reference anatomy, and source file.

## 12. Round-trip guarantees and exclusions

### Guaranteed public semantics

- finite point coordinates;
- canonical point names;
- alias-to-point mappings;
- JSON-safe metadata for JSON export;
- labels for normalized CSV/TSV export.

### Not guaranteed in delimited exports

- arbitrary nested metadata;
- rendered-scene geometry;
- NumPy arrays stored only in metadata;
- exact alias preference if an external tool reorders a comma-separated labels cell.

Use Surgiplot JSON when full metadata retention is required. Use CSV/TSV for interoperable point tables.

## 13. Data validation checklist

Before computing metrics:

1. confirm that every point belongs to the intended subject and target;
2. confirm coordinate frame and units;
3. inspect aliases and role assignment;
4. check polygon order visually;
5. verify that standardization distances are in millimeters;
6. retain the normalized dataset and Git/software version;
7. do not place identifiable clinical data in the software repository.
