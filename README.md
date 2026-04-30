# Surgiplot

Surgiplot is a local quantitative neuroanatomical measurement toolkit for reproducible geometric analysis of surgical corridors, operative angles, exposure metrics, and landmark-derived volumes.

It supports both:

1. **Library mode** for Python scripting, terminals, and notebooks
2. **GUI mode** for interactive dataset curation, 3D scene exploration, and metric analysis

The framework standardizes advanced operative metrics including:

- **VoA** — Visuooperative Angle
- **VOM** — Volume of Operative Maneuverability
- **sVOM** — standardized VOM
- **AoA** — Angle of Attack
- **SF** — Surgical Freedom
- **AoE** — Angle of Exposure
- **AE** — Area of Exposure
- **3D Distance**
- **Volume**

Surgiplot integrates coordinate datasets from:

- neuronavigation systems
- imported point tables
- external 3D meshes / point clouds
- manual coordinate entry

## Core concept

Regardless of input source, Surgiplot normalizes data into a shared editable dataset.

Each dataset includes:

- canonical coordinate points (`point_1`, `point_2`, ...)
- an editable `labels` column for anatomical aliases

Users may reference either the canonical point IDs or their semantic labels in all metric calls.

Example:

```python
entry = ["point_1", "point_2", "point_3", "point_4"]
```

or

```python
entry = ["clinoid_A", "clinoid_B", "clinoid_C", "clinoid_D"]
```

## Installation

### Recommended local setup

Surgiplot is designed to run locally. A conda or Miniforge environment is recommended for a stable scientific desktop stack.

```bash
git clone https://github.com/LeonardT-MD/Surgiplot.git
cd Surgiplot

conda create -n surgiplot python=3.11 pip -y
conda activate surgiplot

pip install -e .
```

### Optional GPU-backed 3D scene backend

For better imported mesh / point-cloud visualization in the 3D Scene Workspace, install the optional PyVista backend:

```bash
pip install -e ".[scene3d]"
```

This installs:

- `pyvista`
- `pyvistaqt`
- `vtk`

If `.[scene3d]` is not installed, Surgiplot falls back to the Matplotlib-based 3D viewer.

### Local installer script

For users who prefer a single command, the bundled installer script creates a conda environment and installs the optional 3D scene backend:

```bash
bash surgiplot/scripts/install_conda.sh
```

## Running Surgiplot

### GUI mode

```bash
surgiplot_gui
```

The GUI supports:

- interactive point labeling
- external 3D scene import
- local rescaling from two picked 3D points
- landmark collection into the shared dataset
- metric configuration panels
- 3D plotting and overlay rendering
- export to CSV / JSON

### CLI / library mode

```bash
surgiplot
```

## 3D Scene Workspace

The optional 3D Scene Workspace allows local import of:

- `PLY`
- `OBJ`
- `STL`
- `XYZ`
- `CSV`
- `TXT`

Supported workflow:

1. import a point cloud or mesh
2. preview it locally in 3D
3. rescale the scene by clicking two 3D points and entering the real-world distance
4. collect landmarks directly in the scene
5. commit those landmarks into the Surgiplot dataset
6. run the standard metrics and visualize them over the same coordinate system

## Scientific methodology

### VOM / sVOM / VoA

Entry and target polygons are projected to best-fit PCA planes, converted into reference ellipses, reconstructed in 3D, and used to compute the corridor geometry. VoA is derived from the corridor trajectory relative to the target plane normal. The main VOM / sVOM output uses the physical 3D corridor geometry.

### AoA / SF

Vertical and horizontal AoA are computed from the cranial-caudal-pivot and medial-lateral-pivot triangles. Optional standardization projects each entry point along its pivot ray to a fixed distance before recomputing standardized SF.

### AE

Area is computed by projecting the ordered 3D polygon onto its PCA best-fit plane and applying the shoelace formula to that 2D polygon.

## Reproducibility

The GUI enables:

- export of labeled datasets
- copying exact result payloads
- deterministic re-execution in scripts or notebooks

## License

Specify your chosen license here.

## Citation

If you use Surgiplot in academic work, please cite the associated methodological publication.
