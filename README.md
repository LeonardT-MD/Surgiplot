# Surgiplot

Surgiplot is a quantitative neuroanatomical measurement toolkit designed
for reproducible geometric analysis of surgical corridors, operative
angles, and exposure constraints in skull base and microsurgical
approaches.

It supports both:

1.  **Library mode** -- programmatic use in Python (terminal, IDE,
    Jupyter)
2.  **GUI mode** -- interactive PySide6/Qt application

The framework standardizes advanced operative metrics including:

-   **VOM** -- Volume of Operative Maneuverability\
-   **sVOM** -- Standardized Volume of Operative Maneuverability\
-   **VoA** -- Visuo-Operative Angle\
-   **AoA** -- Angle of Attack\
-   **SF** -- Surgical Freedom\
-   **AOE** -- Angle of Exposure

Surgiplot integrates coordinate datasets from:

-   Neuronavigation systems\
-   Photogrammetry reconstructions\
-   3D segmentation pipelines (e.g., 3D Slicer)\
-   Manual coordinate entry

------------------------------------------------------------------------

# Core Concept: Interactive Labeling Grid

Regardless of input source (file import or manual), Surgiplot provides
an interactive editable labeling grid.

Each dataset includes:

-   Raw coordinate points (`point_1`, `point_2`, ...)\
-   An editable `labels` column

Users may assign semantic anatomical labels such as:

    clinoid_A
    clinoid_B
    ICA_M1
    BA_apex

These labels can be referenced interchangeably with raw point IDs in all
metric calls.

Example:

``` python
entry=["point_1","point_2","point_3","point_4"]
```

or

``` python
entry=["clinoid_A","clinoid_B","clinoid_C","clinoid_D"]
```

------------------------------------------------------------------------

# Installation

## Recommended: Conda (Reproducible Scientific Environment)

Surgiplot depends on a controlled scientific stack (NumPy, SciPy,
PySide6/Qt, PyTorch).\
To avoid platform inconsistencies, **Conda installation is strongly
recommended**.

### Quick Install

``` bash
git clone https://github.com/LeonardT-MD/Surgiplot.git
cd Surgiplot

conda env create -f env/surgiplot_master.yml
conda activate surgiplot_master

surgiplot_gui
```

The provided environment:

-   Uses `conda-forge`
-   Pins compatible versions of Qt, NumPy, SciPy, and PyTorch
-   Installs Surgiplot in editable mode
-   Optionally installs AI depth estimation dependencies

------------------------------------------------------------------------

## Editable Development Install (Advanced Users)

``` bash
pip install -e ".[ai]"
```

The `[ai]` extra installs optional AI modules for depth estimation.

------------------------------------------------------------------------

# Running Surgiplot

## CLI / Library Mode

``` bash
surgiplot
```

## GUI Mode

``` bash
surgiplot_gui
```

The GUI supports:

-   Interactive point labeling
-   3D plotting
-   Metric configuration panels
-   Export to CSV/JSON
-   Automatic generation of reproducible Python snippets

------------------------------------------------------------------------

# Library Usage

``` python
from surgiplot import load_dataset
from surgiplot.metrics import VOM_VOA, AOA_SF, AOE

ds = load_dataset("annotations.txt", source="navigation")
ds = ds.edit_labels()

res = VOM_VOA(
    data=ds,
    entry=["clinoid_A","clinoid_B","clinoid_C","clinoid_D"],
    target=["BA_1","BA_2","BA_3","BA_4"],
    stand_dist=10
)

aoe = AOE(data=ds, A="p1", B="pivot", C="p3")
```

------------------------------------------------------------------------

# Scientific Methodology

## Volume of Operative Maneuverability (VOM)

The surgical corridor is modeled as a truncated ellipsoidal cone between
entry and target polygons.\
Polygons are projected via PCA, areas computed with the Shoelace
formula, reconstructed in 3D, and approximated via ellipse fitting.\
Volume is obtained via numerical integration.

sVOM standardizes corridor height (default 10 mm).

## Visuo-Operative Angle (VoA)

VoA = 90° − \|acos( (v1·v2) / (\|\|v1\|\| \|\|v2\|\|) )\|

Where:

-   v1 = centroid-to-centroid trajectory vector\
-   v2 = target plane normal

Interpretation:

-   0° → trajectory parallel to target plane\
-   90° → perpendicular trajectory

------------------------------------------------------------------------

# AI Module (Optional)

Surgiplot includes an optional single-image depth estimation module
based on HuggingFace Transformers.

If AI dependencies are not installed:

-   The GUI remains fully functional\
-   Depth estimation is disabled gracefully

To enable AI:

``` bash
pip install -e ".[ai]"
```

------------------------------------------------------------------------

# Reproducibility

The GUI enables:

-   Export of labeled datasets (CSV / JSON)\
-   Copying of exact Python code corresponding to metric runs\
-   Deterministic re-execution in scripts or notebooks

------------------------------------------------------------------------

# License

Specify your chosen license (e.g., MIT, BSD-3-Clause).

------------------------------------------------------------------------

# Citation

If using Surgiplot in academic work, please cite the associated
methodological publication.
