from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from surgiplot import Dataset, from_points, load


def test_dataset_rejects_empty_nonfinite_and_case_colliding_points() -> None:
    dataset = Dataset()
    with pytest.raises(ValueError, match="empty"):
        dataset.add_point("", [0, 0, 0])
    with pytest.raises(ValueError, match="finite"):
        dataset.add_point("A", [0, np.nan, 0])
    dataset.add_point("Point_A", [0, 0, 0])
    with pytest.raises(KeyError, match="case-insensitively"):
        dataset.add_point("point_a", [1, 0, 0])


def test_label_preference_and_case_insensitive_resolution() -> None:
    dataset = from_points([[1, 2, 3]], names=["point_1"], alias_points=False)
    dataset.set_labels("point_1", ["Clinoid", "Anterior landmark"])
    assert dataset.preferred_label("point_1") == "Clinoid"
    assert dataset.labels_for_point("point_1")[0] == "Clinoid"
    assert dataset.resolve_name("clinoid") == "point_1"


def test_rename_rejects_case_collision_and_can_promote_own_alias() -> None:
    dataset = from_points([[1, 2, 3], [4, 5, 6]], names=["Point_A", "Point_B"], alias_points=False)
    dataset.set_labels("Point_A", ["entry"])
    assert dataset.rename_point("Point_A", "ENTRY") == "ENTRY"
    assert dataset.resolve_name("entry") == "ENTRY"
    with pytest.raises(KeyError, match="conflicts"):
        dataset.rename_point("ENTRY", "point_b")


def test_json_round_trip_preserves_points_aliases_and_scientific_metadata(tmp_path) -> None:
    dataset = from_points([[1, 2, 3], [4, 5, 6]], names=["A", "B"], alias_points=False)
    dataset.set_labels("A", ["entry"])
    dataset.meta["matrix"] = np.eye(2)
    dataset.meta["nonfinite_note"] = np.nan
    path = tmp_path / "dataset.json"
    dataset.export_json(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "surgiplot.dataset"
    loaded = load(str(path))
    np.testing.assert_allclose(loaded.get("entry"), [1, 2, 3])
    np.testing.assert_allclose(loaded.get("B"), [4, 5, 6])
    assert loaded.meta["matrix"] == [[1.0, 0.0], [0.0, 1.0]]
    assert loaded.meta["nonfinite_note"] == "nan"


@pytest.mark.parametrize("suffix,separator", [(".csv", ","), (".tsv", "\t")])
def test_delimited_round_trip_preserves_labels(tmp_path, suffix: str, separator: str) -> None:
    dataset = from_points([[1, 2, 3], [4, 5, 6]], names=["A", "B"], alias_points=False)
    dataset.set_labels("A", ["entry", "cranial"])
    path = tmp_path / f"dataset{suffix}"
    dataset.export_csv(path)

    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert separator in header
    loaded = load(str(path), kind="database")
    assert loaded.resolve_name("entry") == "A"
    assert loaded.resolve_name("cranial") == "A"


def test_formatted_database_rejects_nonfinite_coordinates(tmp_path) -> None:
    path = tmp_path / "invalid.csv"
    pd.DataFrame([{"name": "A", "x": 1, "y": np.nan, "z": 3}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="finite"):
        load(str(path), kind="database")


def test_empty_generic_file_is_rejected(tmp_path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("# no points\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No importable"):
        load(str(path), kind="generic")


def test_stryker_and_medtronic_detection(tmp_path) -> None:
    stryker = tmp_path / "stryker.txt"
    stryker.write_text(
        "[ANNOTATIONPOINT_1]\nname=#138\npoint=1,2,3\n",
        encoding="utf-8",
    )
    stryker_dataset = load(str(stryker), kind="navigation")
    np.testing.assert_allclose(stryker_dataset.get("point_138"), [1, 2, 3])

    medtronic = tmp_path / "medtronic.json"
    medtronic.write_text(
        json.dumps(
            {
                "annotations": {
                    "LPS": [{"name": "Target", "point": {"x": 4, "y": 5, "z": 6}}]
                }
            }
        ),
        encoding="utf-8",
    )
    medtronic_dataset = load(str(medtronic), kind="navigation")
    np.testing.assert_allclose(medtronic_dataset.get("Target"), [4, 5, 6])
