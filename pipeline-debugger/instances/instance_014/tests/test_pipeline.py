from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "fact_orders.csv"
REJECTED_PATH = PROJECT_ROOT / "outputs" / "rejected_rows.csv"


def run_pipeline() -> subprocess.CompletedProcess[str]:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    return subprocess.run(
        [sys.executable, "run_pipeline.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture(scope="module")
def pipeline_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    result = run_pipeline()
    assert result.returncode == 0, result.stderr
    assert OUTPUT_PATH.exists(), "fact_orders.csv was not created"
    assert REJECTED_PATH.exists(), "rejected_rows.csv was not created"
    return pd.read_csv(OUTPUT_PATH), pd.read_csv(REJECTED_PATH)


def test_run_pipeline_exit_code() -> None:
    result = run_pipeline()
    assert result.returncode == 0, result.stderr


def test_output_schema(pipeline_outputs: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    output_df, _ = pipeline_outputs
    assert list(output_df.columns) == [
        "order_id",
        "customer_id",
        "customer_name",
        "segment",
        "order_date",
        "amount",
        "discount",
        "net_amount",
        "is_high_value",
        "processing_status",
    ]


def test_row_count_and_completeness(
    pipeline_outputs: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    output_df, _ = pipeline_outputs
    assert len(output_df) == 4
    assert set(output_df["order_id"].tolist()) == {100, 101, 102, 103}


def test_transformation_logic(pipeline_outputs: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    output_df, _ = pipeline_outputs
    target = output_df.loc[output_df["order_id"] == 101].iloc[0]
    assert target["customer_name"] == "Northwind"
    assert target["segment"] == "enterprise"
    assert target["amount"] == pytest.approx(1200.0)
    assert target["discount"] == pytest.approx(120.0)
    assert target["net_amount"] == pytest.approx(1080.0)
    assert bool(target["is_high_value"]) is True


def test_error_handling_for_malformed_rows(
    pipeline_outputs: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    _, rejected_df = pipeline_outputs
    assert len(rejected_df) == 1
    assert str(rejected_df.iloc[0]["order_id"]) == "104"
    assert "could not convert" in str(rejected_df.iloc[0]["reason"]).lower()
