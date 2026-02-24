"""Tests for data cleaning transformations."""

import pytest
import pandas as pd


def test_clean_data_removes_duplicates():
    from pipelines.transform.clean import clean_data
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    result = clean_data(df)
    assert len(result) == 2


def test_clean_data_fills_missing():
    from pipelines.transform.clean import clean_data
    df = pd.DataFrame({"a": [1.0, None], "b": ["hello", None]})
    result = clean_data(df)
    assert result["a"].isna().sum() == 0
    assert result["b"].isna().sum() == 0


def test_clean_data_normalizes_strings():
    from pipelines.transform.clean import clean_data
    df = pd.DataFrame({"a": [1], "b": ["  HELLO  "]})
    result = clean_data(df)
    assert result["b"].iloc[0] == "hello"
