from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from surgiplot.cli.app import app


runner = CliRunner()


def test_version_is_machine_readable_json() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["package"] == "surgiplot"


def test_distance_command_accepts_normalized_csv(tmp_path) -> None:
    path = tmp_path / "points.csv"
    path.write_text("name,x,y,z\nA,0,0,0\nB,3,4,12\n", encoding="utf-8")
    result = runner.invoke(
        app,
        ["metric", "distance", str(path), "--A", "A", "--B", "B"],
    )
    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout)["distance_mm"] == 13.0


def test_range_expansion_in_area_command(tmp_path) -> None:
    path = tmp_path / "points.csv"
    path.write_text(
        "name,x,y,z\n"
        "point_1,0,0,0\n"
        "point_2,2,0,0\n"
        "point_3,2,2,0\n"
        "point_4,0,2,0\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["metric", "area", str(path), "--polygon", "point_1-4"])
    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout)["area_mm2"] == pytest.approx(4.0)
