"""Tests for CSV export behavior in `evals.outputs`."""

from __future__ import annotations

import csv
from pathlib import Path

from evals.logging import ExperimentRun
from evals.outputs import save_run


def test_csv_export_does_not_corrupt_reserved_index_field(tmp_path: Path) -> None:
    """
    Guard against silent CSV corruption when a sweep dimension is named "index".

    Previously, `variables["index"]` overwrote the result's true `index` column.
    The exporter should keep the real result index in `index` and store the
    variable value under `var_index`.
    """
    run = ExperimentRun(
        name="test_csv_reserved_index",
        config={},
        git_hash="test",
        git_branch="test",
        python_version="test",
    )
    run.results = [
        {
            "index": 7,
            "variables": {"index": 999, "label": "varlabel", "x": 1},
            "prompt": "prompt",
            "response": "response",
            "judgment": {"label": "ok"},
        }
    ]

    outputs = save_run(run, tmp_path, formats=["csv"])
    csv_path = outputs["csv"]

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    row = rows[0]
    assert row["index"] == "7"
    assert row["var_index"] == "999"
    assert row["var_label"] == "varlabel"
    assert row["x"] == "1"
    assert row["label"] == "ok"


