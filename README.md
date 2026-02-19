# Surgiplot (dual-mode)

Surgiplot is designed to be used:

1) **As a Python library** (terminal/IDE/Jupyter)  
2) **As a GUI application** (PySide6/Qt)

A key feature is the **interactive labeling grid**: regardless of how points are loaded
(file or manual), Surgiplot opens an editable table with an extra `labels` column.
You can assign human-readable labels (e.g., `clinoid`) and then reference the same
point in syntax either as `point_1` **or** `clinoid`.

## Install (editable)
```bash
pip install -e .
```

## Run
```bash
surgiplot
surgiplot_gui
```

## Library usage
```python
from surgiplot import load_dataset
from surgiplot.metrics import VOM_VOA, AOA_SF, AOE

ds = load_dataset("annotations.txt", source="navigation")
# Labeling grid can be opened any time:
ds = ds.edit_labels()

res = VOM_VOA(data=ds, entry=["point_1","point_2","point_3","point_4"],
                    target=["clinoid_A","clinoid_B","clinoid_C","clinoid_D"],
                    stand_dist=10)

aoe = AOE(data=ds, A="p1", B="p2", C="p3")
```

## GUI extras
- Export labeled dataset (CSV/JSON)
- Copy reproducible Python snippet for the current metric run
