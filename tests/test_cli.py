from __future__ import annotations

import json
import math

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


def test_vom_command_reports_minimum_rotation_axis_swap(tmp_path) -> None:
    path = tmp_path / "vom_points.csv"
    rows = ["name,x,y,z"]
    angle = math.radians(70.0)
    for index in range(32):
        theta = 2.0 * math.pi * index / 32.0
        rows.append(f"target_{index + 1},{4.0 * math.cos(theta)},{math.sin(theta)},0")
        local_x = 5.0 * math.cos(theta)
        local_y = 1.5 * math.sin(theta)
        x = local_x * math.cos(angle) - local_y * math.sin(angle)
        y = local_x * math.sin(angle) + local_y * math.cos(angle)
        rows.append(f"entry_{index + 1},{x},{y},100")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "metric",
            "vom-voa",
            str(path),
            "--entry",
            "entry_1-32",
            "--target",
            "target_1-32",
            "--stand-dist",
            "50",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ellipse_frame_correspondence_policy"] == "minimum_rotation_with_axis_permutation"
    assert payload["entry_frame_alignment_swapped_axes"] is True
    assert payload["entry_frame_alignment_operation"].startswith("swap_")
    assert payload["entry_frame_alignment_rotation_deg"] == pytest.approx(20.0, abs=1e-8)
