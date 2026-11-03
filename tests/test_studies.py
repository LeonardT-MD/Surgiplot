from __future__ import annotations

from surgiplot import build_metric_run_record, repeatability_rows


def test_repeatability_uses_nonnegative_cv_and_ignores_nonfinite_values() -> None:
    records = [
        build_metric_run_record("S1", "target", "test", {"signed": -2.0}),
        build_metric_run_record("S1", "target", "test", {"signed": -4.0}),
        build_metric_run_record("S1", "target", "test", {"signed": float("inf")}),
    ]
    rows = repeatability_rows(records)
    assert len(rows) == 1
    assert rows[0]["Replicate Count"] == "2"
    assert float(rows[0]["CV_percent"]) >= 0.0
