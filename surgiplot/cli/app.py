from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

import typer
from rich import print as rich_print
from rich.table import Table

from surgiplot import (
    AOA,
    AOA_SF,
    AOE,
    AREA_3D,
    DISTANCE_3D,
    SF,
    SVOM,
    VOA,
    VOM,
    VOM_VOA,
    VOLUME_3D,
    __version__,
    apply_labels,
    load,
    rename_points,
)

app = typer.Typer(
    help="Surgiplot CLI for reproducible neurosurgical coordinate analysis.",
    no_args_is_help=True,
)
dataset_app = typer.Typer(help="Dataset inspection, relabeling, and export.", no_args_is_help=True)
metric_app = typer.Typer(help="Metric computation commands for scripting pipelines.", no_args_is_help=True)
app.add_typer(dataset_app, name="dataset")
app.add_typer(metric_app, name="metric")


@app.command("version")
def version() -> None:
    _emit_json({"package": "surgiplot", "version": __version__})


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in (raw or "").split(",") if item.strip()]


def _pairs(values: Iterable[str], option_name: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for value in values:
        text = str(value or "").strip()
        if "=" not in text:
            raise typer.BadParameter(f"{option_name} expects KEY=VALUE entries.")
        key, val = text.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not key or not val:
            raise typer.BadParameter(f"{option_name} expects non-empty KEY=VALUE entries.")
        mapping[key] = val
    return mapping


def _load_dataset_for_cli(
    file: str,
    *,
    source: str = "",
    kind: str = "auto",
    navigation_format: str = "auto",
    edit_labels: bool = False,
):
    dataset = load(
        file,
        source=source,
        kind=kind,
        navigation_format=navigation_format,
        alias_points=True,
    )
    if edit_labels:
        dataset.edit_labels()
    return dataset


def _resolve_names(dataset, names: Iterable[str]) -> list[str]:
    return dataset.resolve_names(names) if hasattr(dataset, "resolve_names") else list(names)


def _resolve_name(dataset, name: str) -> str:
    return dataset.resolve_name(name) if hasattr(dataset, "resolve_name") else name


def _emit_json(payload: dict[str, Any]) -> None:
    rich_print_json = json.dumps(payload, indent=2, sort_keys=True)
    rich_print(rich_print_json)


def _save_dataset(dataset, out: str) -> None:
    out_path = Path(out)
    suffix = out_path.suffix.lower()
    if suffix == ".json":
        dataset.export_json(str(out_path))
        return
    if suffix in {".csv", ".tsv"}:
        dataset.export_csv(str(out_path))
        return
    raise typer.BadParameter("Output dataset file must end in .json, .csv, or .tsv.")


@dataset_app.command("summary")
def dataset_summary(
    file: str = typer.Argument(..., help="Input dataset or annotation file."),
    source: str = typer.Option("", "--source", help="navigation|photogrammetry|scanner|database"),
    kind: str = typer.Option("auto", "--kind", help="auto|navigation|database|generic"),
    navigation_format: str = typer.Option("auto", "--navigation-format", help="auto|stryker|medtronic|brainlab|generic"),
):
    dataset = _load_dataset_for_cli(file, source=source, kind=kind, navigation_format=navigation_format)
    table = Table(title="Surgiplot Dataset Summary")
    table.add_column("Field", style="bold cyan")
    table.add_column("Value", style="white")
    table.add_row("Points", str(len(dataset.points)))
    table.add_row("Aliases", str(len([a for a, tgt in dataset.aliases.items() if a != tgt and a not in dataset.points])))
    table.add_row("Source", str(dataset.meta.get("source", "")))
    table.add_row("Coordinate system", str(dataset.meta.get("coordinate_system", "")))
    table.add_row("Path", str(dataset.meta.get("path", file)))
    rich_print(table)


@dataset_app.command("labels")
def dataset_labels(
    file: str = typer.Argument(..., help="Input dataset or annotation file."),
    source: str = typer.Option("", "--source", help="navigation|photogrammetry|scanner|database"),
    kind: str = typer.Option("auto", "--kind", help="auto|navigation|database|generic"),
    navigation_format: str = typer.Option("auto", "--navigation-format", help="auto|stryker|medtronic|brainlab|generic"),
    label: list[str] = typer.Option(None, "--label", help="Set labels with POINT=label1,label2"),
    rename: list[str] = typer.Option(None, "--rename", help="Rename canonical points with OLD=NEW"),
    out: Optional[str] = typer.Option(None, "--out", help="Write updated dataset to .json/.csv/.tsv"),
):
    dataset = _load_dataset_for_cli(file, source=source, kind=kind, navigation_format=navigation_format)
    rename_map = _pairs(rename or [], "--rename")
    if rename_map:
        rename_points(dataset, rename_map)
    label_map = {
        name: _split_csv(labels)
        for name, labels in _pairs(label or [], "--label").items()
    }
    if label_map:
        apply_labels(dataset, label_map)

    table = Table(title="Surgiplot Point Labels")
    table.add_column("Point", style="bold cyan")
    table.add_column("Preferred label", style="green")
    table.add_column("Aliases", style="white")
    for name in sorted(dataset.points):
        aliases = ", ".join(dataset.labels_for_point(name))
        table.add_row(name, dataset.preferred_label(name), aliases)
    rich_print(table)

    if out:
        _save_dataset(dataset, out)
        rich_print(f"[green]Saved[/green] updated dataset to {out}")


@dataset_app.command("export")
def dataset_export(
    file: str = typer.Argument(..., help="Input dataset or annotation file."),
    out: str = typer.Argument(..., help="Output path (.json/.csv/.tsv)."),
    source: str = typer.Option("", "--source", help="navigation|photogrammetry|scanner|database"),
    kind: str = typer.Option("auto", "--kind", help="auto|navigation|database|generic"),
    navigation_format: str = typer.Option("auto", "--navigation-format", help="auto|stryker|medtronic|brainlab|generic"),
):
    dataset = _load_dataset_for_cli(file, source=source, kind=kind, navigation_format=navigation_format)
    _save_dataset(dataset, out)
    rich_print(f"[green]Saved[/green] normalized dataset to {out}")


@metric_app.command("vom-voa")
def metric_vom_voa(
    file: str = typer.Argument(..., help="Input dataset or annotation file."),
    entry: str = typer.Option(..., "--entry", help="Comma-separated entry point names/labels."),
    target: str = typer.Option(..., "--target", help="Comma-separated target point names/labels."),
    stand_dist: float = typer.Option(10.0, "--stand-dist", help="Standard distance for sVOM."),
    source: str = typer.Option("", "--source", help="navigation|photogrammetry|scanner|database"),
    kind: str = typer.Option("auto", "--kind", help="auto|navigation|database|generic"),
    navigation_format: str = typer.Option("auto", "--navigation-format", help="auto|stryker|medtronic|brainlab|generic"),
):
    dataset = _load_dataset_for_cli(file, source=source, kind=kind, navigation_format=navigation_format)
    result = VOM_VOA(
        data=dataset,
        entry=_resolve_names(dataset, _split_csv(entry)),
        target=_resolve_names(dataset, _split_csv(target)),
        stand_dist=stand_dist,
        return_debug=False,
    )
    _emit_json(
        {
            "VoA_deg": result.voa_deg,
            "VOM_mm3": result.vom_mm3,
            "sVOM_mm3": result.svom_mm3,
        }
    )


@metric_app.command("vom")
def metric_vom(
    file: str = typer.Argument(...),
    entry: str = typer.Option(..., "--entry"),
    target: str = typer.Option(..., "--target"),
    stand_dist: float = typer.Option(10.0, "--stand-dist"),
    source: str = typer.Option("", "--source"),
    kind: str = typer.Option("auto", "--kind"),
    navigation_format: str = typer.Option("auto", "--navigation-format"),
):
    dataset = _load_dataset_for_cli(file, source=source, kind=kind, navigation_format=navigation_format)
    result = VOM(data=dataset, entry=_resolve_names(dataset, _split_csv(entry)), target=_resolve_names(dataset, _split_csv(target)), stand_dist=stand_dist)
    _emit_json(
        {
            "VoA_deg": result.voa_deg,
            "VOM_mm3": result.vom_mm3,
            "sVOM_mm3": result.svom_mm3,
        }
    )


@metric_app.command("voa")
def metric_voa(
    file: str = typer.Argument(...),
    entry: str = typer.Option(..., "--entry"),
    target: str = typer.Option(..., "--target"),
    stand_dist: float = typer.Option(10.0, "--stand-dist"),
    source: str = typer.Option("", "--source"),
    kind: str = typer.Option("auto", "--kind"),
    navigation_format: str = typer.Option("auto", "--navigation-format"),
):
    dataset = _load_dataset_for_cli(file, source=source, kind=kind, navigation_format=navigation_format)
    result = VOA(data=dataset, entry=_resolve_names(dataset, _split_csv(entry)), target=_resolve_names(dataset, _split_csv(target)), stand_dist=stand_dist)
    _emit_json({"VoA_deg": result})


@metric_app.command("svom")
def metric_svom(
    file: str = typer.Argument(...),
    entry: str = typer.Option(..., "--entry"),
    target: str = typer.Option(..., "--target"),
    stand_dist: float = typer.Option(10.0, "--stand-dist"),
    source: str = typer.Option("", "--source"),
    kind: str = typer.Option("auto", "--kind"),
    navigation_format: str = typer.Option("auto", "--navigation-format"),
):
    dataset = _load_dataset_for_cli(file, source=source, kind=kind, navigation_format=navigation_format)
    result = SVOM(data=dataset, entry=_resolve_names(dataset, _split_csv(entry)), target=_resolve_names(dataset, _split_csv(target)), stand_dist=stand_dist)
    _emit_json({"sVOM_mm3": result})


@metric_app.command("aoa-sf")
def metric_aoa_sf(
    file: str = typer.Argument(..., help="Input dataset or annotation file."),
    cranial: str = typer.Option(..., "--cranial"),
    caudal: str = typer.Option(..., "--caudal"),
    medial: str = typer.Option(..., "--medial"),
    lateral: str = typer.Option(..., "--lateral"),
    pivot: str = typer.Option(..., "--pivot"),
    sf_rescale_radius_mm: Optional[float] = typer.Option(None, "--sf-rescale-radius-mm"),
    source: str = typer.Option("", "--source"),
    kind: str = typer.Option("auto", "--kind"),
    navigation_format: str = typer.Option("auto", "--navigation-format"),
):
    dataset = _load_dataset_for_cli(file, source=source, kind=kind, navigation_format=navigation_format)
    entry = [
        _resolve_name(dataset, cranial),
        _resolve_name(dataset, caudal),
        _resolve_name(dataset, medial),
        _resolve_name(dataset, lateral),
    ]
    result = AOA_SF(
        data=dataset,
        entry=entry,
        target=_resolve_name(dataset, pivot),
        sf_rescale_radius_mm=sf_rescale_radius_mm,
        return_debug=False,
    )
    _emit_json(
        {
            "AoA_vertical_deg": result.aoa_vertical_deg,
            "AoA_horizontal_deg": result.aoa_horizontal_deg,
            "SF_area_mm2": result.sf_entry_area_mm2,
            "SF_area_standardized_mm2": result.sf_entry_area_rescaled_mm2,
        }
    )


@metric_app.command("aoa")
def metric_aoa(
    file: str = typer.Argument(...),
    cranial: str = typer.Option(..., "--cranial"),
    caudal: str = typer.Option(..., "--caudal"),
    medial: str = typer.Option(..., "--medial"),
    lateral: str = typer.Option(..., "--lateral"),
    pivot: str = typer.Option(..., "--pivot"),
    sf_rescale_radius_mm: Optional[float] = typer.Option(None, "--sf-rescale-radius-mm"),
    source: str = typer.Option("", "--source"),
    kind: str = typer.Option("auto", "--kind"),
    navigation_format: str = typer.Option("auto", "--navigation-format"),
):
    dataset = _load_dataset_for_cli(file, source=source, kind=kind, navigation_format=navigation_format)
    result = AOA(
        data=dataset,
        entry=[
            _resolve_name(dataset, cranial),
            _resolve_name(dataset, caudal),
            _resolve_name(dataset, medial),
            _resolve_name(dataset, lateral),
        ],
        target=_resolve_name(dataset, pivot),
        sf_rescale_radius_mm=sf_rescale_radius_mm,
        return_debug=False,
    )
    _emit_json(
        {
            "AoA_vertical_deg": result.aoa_vertical_deg,
            "AoA_horizontal_deg": result.aoa_horizontal_deg,
            "SF_area_mm2": result.sf_entry_area_mm2,
            "SF_area_standardized_mm2": result.sf_entry_area_rescaled_mm2,
        }
    )


@metric_app.command("sf")
def metric_sf(
    file: str = typer.Argument(...),
    cranial: str = typer.Option(..., "--cranial"),
    caudal: str = typer.Option(..., "--caudal"),
    medial: str = typer.Option(..., "--medial"),
    lateral: str = typer.Option(..., "--lateral"),
    pivot: str = typer.Option(..., "--pivot"),
    sf_rescale_radius_mm: Optional[float] = typer.Option(None, "--sf-rescale-radius-mm"),
    source: str = typer.Option("", "--source"),
    kind: str = typer.Option("auto", "--kind"),
    navigation_format: str = typer.Option("auto", "--navigation-format"),
):
    dataset = _load_dataset_for_cli(file, source=source, kind=kind, navigation_format=navigation_format)
    result = SF(
        data=dataset,
        entry=[
            _resolve_name(dataset, cranial),
            _resolve_name(dataset, caudal),
            _resolve_name(dataset, medial),
            _resolve_name(dataset, lateral),
        ],
        target=_resolve_name(dataset, pivot),
        sf_rescale_radius_mm=sf_rescale_radius_mm,
        return_debug=False,
    )
    _emit_json({"SF_area_mm2": result})


@metric_app.command("aoe")
def metric_aoe(
    file: str = typer.Argument(...),
    a: str = typer.Option(..., "--A"),
    b: str = typer.Option(..., "--B"),
    c: str = typer.Option(..., "--C"),
    source: str = typer.Option("", "--source"),
    kind: str = typer.Option("auto", "--kind"),
    navigation_format: str = typer.Option("auto", "--navigation-format"),
):
    dataset = _load_dataset_for_cli(file, source=source, kind=kind, navigation_format=navigation_format)
    result = AOE(
        data=dataset,
        A=_resolve_name(dataset, a),
        B=_resolve_name(dataset, b),
        C=_resolve_name(dataset, c),
        return_debug=False,
    )
    _emit_json({"AoE_deg": result.aoe_deg})


@metric_app.command("distance")
def metric_distance(
    file: str = typer.Argument(...),
    a: str = typer.Option(..., "--A"),
    b: str = typer.Option(..., "--B"),
    source: str = typer.Option("", "--source"),
    kind: str = typer.Option("auto", "--kind"),
    navigation_format: str = typer.Option("auto", "--navigation-format"),
):
    dataset = _load_dataset_for_cli(file, source=source, kind=kind, navigation_format=navigation_format)
    result = DISTANCE_3D(data=dataset, A=_resolve_name(dataset, a), B=_resolve_name(dataset, b), return_debug=False)
    _emit_json({"distance_mm": result.distance_mm})


@metric_app.command("area")
def metric_area(
    file: str = typer.Argument(...),
    polygon: str = typer.Option(..., "--polygon"),
    source: str = typer.Option("", "--source"),
    kind: str = typer.Option("auto", "--kind"),
    navigation_format: str = typer.Option("auto", "--navigation-format"),
):
    dataset = _load_dataset_for_cli(file, source=source, kind=kind, navigation_format=navigation_format)
    result = AREA_3D(data=dataset, polygon=_resolve_names(dataset, _split_csv(polygon)), return_debug=False)
    _emit_json({"area_mm2": result.area_mm2})


@metric_app.command("volume")
def metric_volume(
    file: str = typer.Argument(...),
    points: str = typer.Option(..., "--points"),
    source: str = typer.Option("", "--source"),
    kind: str = typer.Option("auto", "--kind"),
    navigation_format: str = typer.Option("auto", "--navigation-format"),
):
    dataset = _load_dataset_for_cli(file, source=source, kind=kind, navigation_format=navigation_format)
    result = VOLUME_3D(data=dataset, points=_resolve_names(dataset, _split_csv(points)), return_debug=False)
    _emit_json({"volume_mm3": result.volume_mm3})


@app.command("launch", hidden=True)
def launch():
    """Deprecated interactive launcher retained for backward compatibility."""
    raise typer.BadParameter(
        "The interactive CLI launcher has been retired for research scripting. "
        "Use `surgiplot dataset ...`, `surgiplot metric ...`, or `surgiplot_gui`."
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
