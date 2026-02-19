from __future__ import annotations

import typer
from rich import print
from rich.prompt import Prompt
from pathlib import Path
import numpy as np

from surgiplot.core.io.loaders import load_dataset, load_points_from_manual
from surgiplot.metrics import VOM_VOA, AOA_SF, AOE

app = typer.Typer(help="Surgiplot CLI (loads points, opens labeling grid, then runs metrics).")

def _prompt_source() -> str:
    return Prompt.ask("Source of data", choices=["navigation", "photogrammetry", "scanner"], default="navigation")

def _label_grid(ds):
    # Always open interactive grid as requested
    ds.edit_labels()
    return ds

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
        ds = load_points_from_manual(pts, names=names if names else None, source=source, alias_points=True)

    _label_grid(ds)
    print(f"Loaded {len(ds.points)} points with {len(ds.aliases)} aliases. Source={ds.meta.get('source','')}")
    print("Now run a metric command (vom-voa / aoa-sf / aoe) using the same dataset in Python, or use GUI via surgiplot_gui.")

@app.command()
def vom_voa(file: str = typer.Option(None, "--file", help="Annotation file path"), 
            source: str = typer.Option(None, "--source", help="navigation|photogrammetry|scanner"), 
            entry: str = typer.Option(..., "--entry", help="Comma-separated entry point names"), 
            target: str = typer.Option(..., "--target", help="Comma-separated target point names"), 
            stand_dist: float = typer.Option(10.0, "--stand-dist", help="Standard distance for sVOM")):
    """Compute VOM_VOA via CLI. Opens labeling grid before computation."""
    if file is None:
        raise typer.BadParameter("--file is required for this command (use `launch` for manual mode).")

    src = (source or "navigation").strip().lower()
    ds = load_dataset(file, source=src, alias_points=True)
    _label_grid(ds)

    entry_names = [x.strip() for x in entry.split(",") if x.strip()]
    target_names = [x.strip() for x in target.split(",") if x.strip()]

    res = VOM_VOA(data=ds, entry=entry_names, target=target_names, stand_dist=stand_dist, return_debug=False)
    print({"VoA_deg": res.voa_deg, "VOM_mm3": res.vom_mm3, "sVOM_mm3": res.svom_mm3})

@app.command()
def aoa_sf(file: str = typer.Option(None, "--file"), 
           source: str = typer.Option(None, "--source"), 
           entry: str = typer.Option(..., "--entry"), 
           target: str = typer.Option(..., "--target", help="Single pivot point name"), 
           constraints: str = typer.Option("", "--constraints")):
    """Compute AOA_SF via CLI. Opens labeling grid before computation."""
    if file is None:
        raise typer.BadParameter("--file is required for this command (use `launch` for manual mode).")

    src = (source or "navigation").strip().lower()
    ds = load_dataset(file, source=src, alias_points=True)
    _label_grid(ds)

    entry_names = [x.strip() for x in entry.split(",") if x.strip()]
    cons = [x.strip() for x in constraints.split(",") if x.strip()] or None

    res = AOA_SF(data=ds, entry=entry_names, target=target.strip(), constraints=cons, return_debug=False)
    print({"AoA_vertical_deg": res.aoa_vertical_deg, "AoA_horizontal_deg": res.aoa_horizontal_deg, "SF_proxy_mm2": res.sf_proxy_mm2})

@app.command()
def aoe(file: str = typer.Option(None, "--file"), 
        source: str = typer.Option(None, "--source"), 
        a: str = typer.Option(..., "--A"), 
        b: str = typer.Option(..., "--B"), 
        c: str = typer.Option(..., "--C")):
    """Compute AOE via CLI. Opens labeling grid before computation."""
    if file is None:
        raise typer.BadParameter("--file is required for this command (use `launch` for manual mode).")

    src = (source or "navigation").strip().lower()
    ds = load_dataset(file, source=src, alias_points=True)
    _label_grid(ds)

    res = AOE(data=ds, A=a, B=b, C=c, return_debug=False)
    print({"AoE_deg": res.aoe_deg})

def main():
    app()
