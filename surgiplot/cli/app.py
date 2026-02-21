from __future__ import annotations

import typer
from rich import print
from rich.prompt import Prompt

from surgiplot.core.io.loaders import load_dataset, load_points_from_manual
from surgiplot.metrics import VOM_VOA, AOA_SF, AOE, DISTANCE_3D, AREA_3D

app = typer.Typer(help="Surgiplot CLI (loads points, opens labeling grid, then runs metrics).")


def _prompt_source() -> str:
    return Prompt.ask(
        "Source of data",
        choices=["navigation", "photogrammetry", "scanner"],
        default="navigation",
    )


def _label_grid(ds):
    # Always open interactive grid as requested
    ds.edit_labels()
    return ds


def _split_csv(s: str) -> list[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def _resolve_list(ds, names: list[str]) -> list[str]:
    if hasattr(ds, "resolve_names"):
        return ds.resolve_names(names)
    return names


def _resolve_one(ds, name: str) -> str:
    name = (name or "").strip()
    if hasattr(ds, "resolve_name"):
        return ds.resolve_name(name)
    return name


@app.command()
def launch():
    """Interactive launcher (file/manual) -> labeling grid."""
    mode = Prompt.ask("Input mode", choices=["file", "manual"], default="file")
    source = _prompt_source()

    if mode == "file":
        path = Prompt.ask("Path to annotation file")
        ds = load_dataset(path, source=source, alias_points=True)
    else:
        print("Paste points line-by-line. End with an empty line.")
        lines = []
        while True:
            ln = input()
            if not ln.strip():
                break
            lines.append(ln.strip())

        pts = []
        names = []
        for ln in lines:
            parts = ln.split()
            if len(parts) == 4:
                names.append(parts[0])
                pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif len(parts) == 3:
                pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
            else:
                raise typer.BadParameter(f"Bad line: {ln}")

        ds = load_points_from_manual(
            pts,
            names=names if names else None,
            source=source,
            alias_points=True,
        )

    _label_grid(ds)
    print(f"Loaded {len(ds.points)} points with {len(ds.aliases)} aliases. Source={ds.meta.get('source','')}")
    print("Run CLI metric commands (vom-voa / aoa-sf / aoe / distance-3d / area-3d), or use the GUI via `surgiplot_gui`.")


@app.command("vom-voa")
def vom_voa(
    file: str = typer.Option(None, "--file", help="Annotation file path"),
    source: str = typer.Option(None, "--source", help="navigation|photogrammetry|scanner"),
    entry: str = typer.Option(..., "--entry", help="Comma-separated entry point names/labels"),
    target: str = typer.Option(..., "--target", help="Comma-separated target point names/labels"),
    stand_dist: float = typer.Option(10.0, "--stand-dist", help="Standard distance for sVOM"),
):
    """Compute VOM_VOA via CLI. Opens labeling grid before computation."""
    if file is None:
        raise typer.BadParameter("--file is required for this command (use `launch` for manual mode).")

    src = (source or "navigation").strip().lower()
    ds = load_dataset(file, source=src, alias_points=True)
    _label_grid(ds)

    entry_names = _resolve_list(ds, _split_csv(entry))
    target_names = _resolve_list(ds, _split_csv(target))

    res = VOM_VOA(
        data=ds,
        entry=entry_names,
        target=target_names,
        stand_dist=stand_dist,
        return_debug=False,
    )
    print({"VoA_deg": res.voa_deg, "VOM_mm3": res.vom_mm3, "sVOM_mm3": res.svom_mm3})


@app.command("aoa-sf")
def aoa_sf(
    file: str = typer.Option(None, "--file", help="Annotation file path"),
    source: str = typer.Option(None, "--source", help="navigation|photogrammetry|scanner"),

    # New preferred interface (matches GUI spec)
    cranial: str = typer.Option("", "--cranial", help="Cranial entry point name/label"),
    caudal: str = typer.Option("", "--caudal", help="Caudal entry point name/label"),
    medial: str = typer.Option("", "--medial", help="Medial entry point name/label"),
    lateral: str = typer.Option("", "--lateral", help="Lateral entry point name/label"),
    pivot: str = typer.Option("", "--pivot", help="Pivot point name/label"),

    # Backward-compatible legacy interface
    entry: str = typer.Option("", "--entry", help="Legacy: comma-separated entry points"),
    target: str = typer.Option("", "--target", help="Legacy: single pivot point name/label"),

    constraints: str = typer.Option("", "--constraints", help="Optional constraints (comma-separated)"),
):
    """Compute AOA_SF via CLI. Opens labeling grid before computation."""
    if file is None:
        raise typer.BadParameter("--file is required for this command (use `launch` for manual mode).")

    src = (source or "navigation").strip().lower()
    ds = load_dataset(file, source=src, alias_points=True)
    _label_grid(ds)

    # Determine which mode to use
    use_new = all([
        (cranial or "").strip(),
        (caudal or "").strip(),
        (medial or "").strip(),
        (lateral or "").strip(),
        (pivot or "").strip(),
    ])

    if use_new:
        cran = _resolve_one(ds, cranial)
        caud = _resolve_one(ds, caudal)
        med = _resolve_one(ds, medial)
        lat = _resolve_one(ds, lateral)
        piv = _resolve_one(ds, pivot)
        entry_names = [cran, caud, med, lat]
        target_name = piv
    else:
        if not entry.strip() or not target.strip():
            raise typer.BadParameter(
                "Provide either --cranial --caudal --medial --lateral --pivot, "
                "or legacy --entry and --target."
            )
        entry_names = _resolve_list(ds, _split_csv(entry))
        target_name = _resolve_one(ds, target)

    cons = _split_csv(constraints)
    cons = _resolve_list(ds, cons) if cons else None

    res = AOA_SF(
        data=ds,
        entry=entry_names,
        target=target_name,
        constraints=cons,
        return_debug=False,
    )
    print({
        "AoA_vertical_deg": res.aoa_vertical_deg,
        "AoA_horizontal_deg": res.aoa_horizontal_deg,
        "SF_proxy_mm2": res.sf_proxy_mm2,
    })


@app.command("aoe")
def aoe(
    file: str = typer.Option(None, "--file", help="Annotation file path"),
    source: str = typer.Option(None, "--source", help="navigation|photogrammetry|scanner"),
    a: str = typer.Option(..., "--A", help="Point A name/label"),
    b: str = typer.Option(..., "--B", help="Point B (pivot) name/label"),
    c: str = typer.Option(..., "--C", help="Point C name/label"),
):
    """Compute AOE via CLI. Opens labeling grid before computation."""
    if file is None:
        raise typer.BadParameter("--file is required for this command (use `launch` for manual mode).")

    src = (source or "navigation").strip().lower()
    ds = load_dataset(file, source=src, alias_points=True)
    _label_grid(ds)

    A = _resolve_one(ds, a)
    B = _resolve_one(ds, b)
    C = _resolve_one(ds, c)

    res = AOE(data=ds, A=A, B=B, C=C, return_debug=False)
    print({"AoE_deg": res.aoe_deg})


@app.command("distance-3d")
def distance_3d(
    file: str = typer.Option(None, "--file", help="Annotation file path"),
    source: str = typer.Option(None, "--source", help="navigation|photogrammetry|scanner"),
    a: str = typer.Option(..., "--A", help="Point A name/label"),
    b: str = typer.Option(..., "--B", help="Point B name/label"),
):
    """Compute 3D linear distance between points A and B (mm). Opens labeling grid."""
    if file is None:
        raise typer.BadParameter("--file is required for this command (use `launch` for manual mode).")

    src = (source or "navigation").strip().lower()
    ds = load_dataset(file, source=src, alias_points=True)
    _label_grid(ds)

    A = _resolve_one(ds, a)
    B = _resolve_one(ds, b)

    res = DISTANCE_3D(data=ds, A=A, B=B, return_debug=False)
    print({"distance_mm": res.distance_mm})


@app.command("area-3d")
def area_3d(
    file: str = typer.Option(None, "--file", help="Annotation file path"),
    source: str = typer.Option(None, "--source", help="navigation|photogrammetry|scanner"),
    polygon: str = typer.Option(..., "--polygon", help="Comma-separated polygon point names/labels (>=3)"),
):
    """Compute 3D polygon area (mm^2) from >=3 points. Opens labeling grid."""
    if file is None:
        raise typer.BadParameter("--file is required for this command (use `launch` for manual mode).")

    src = (source or "navigation").strip().lower()
    ds = load_dataset(file, source=src, alias_points=True)
    _label_grid(ds)

    poly_names = _resolve_list(ds, _split_csv(polygon))

    res = AREA_3D(data=ds, polygon=poly_names, return_debug=False)
    print({"area_mm2": res.area_mm2})


def main():
    app()
