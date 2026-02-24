import os
import tempfile

import pandas as pd

from processor.pipeline import load_config, run_pipeline


def test_load_config_defaults() -> None:
    cfg = load_config()
    assert isinstance(cfg, dict)


def test_run_pipeline_csv(tmp_path) -> None:
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df.to_csv(input_file, index=False)
    count = run_pipeline(str(input_file), str(output_file))
    assert count == 3
    assert output_file.exists()


def test_run_pipeline_filters_empty_rows(tmp_path) -> None:
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"
    df = pd.DataFrame({"A": [1, None, 3], "B": [4, None, 6]})
    df.to_csv(input_file, index=False)
    count = run_pipeline(str(input_file), str(output_file))
    assert count == 2
