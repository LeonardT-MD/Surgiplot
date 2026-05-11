# Vendored backend location

If `SurgiplotToolkit` is split into a standalone Slicer extension repository, the shared
`surgiplot` Python backend can be vendored here:

```text
SurgiplotToolkit/
  SurgiplotToolkit/
    SurgiplotToolkitLib/
      External/
        surgiplot_backend/
          surgiplot/
```

The bridge loader in `surgiplot_bridge.py` will prefer this vendored backend when present,
and will otherwise fall back to the surrounding Surgiplot repository during incubator
development.
