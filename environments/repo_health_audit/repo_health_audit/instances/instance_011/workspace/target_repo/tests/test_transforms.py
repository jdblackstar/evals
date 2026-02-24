import pandas as pd

from processor.transforms import add_computed_column, filter_rows, normalize_columns


def test_normalize_columns_lowercases() -> None:
    df = pd.DataFrame({"Name": [1], " Age ": [2]})
    result = normalize_columns(df)
    assert list(result.columns) == ["name", "age"]


def test_filter_rows_removes_all_null() -> None:
    df = pd.DataFrame({"a": [1, None], "b": [2, None]})
    result = filter_rows(df)
    assert len(result) == 1


def test_add_computed_column() -> None:
    df = pd.DataFrame({"x": [1, 2], "y": [10, 20]})
    result = add_computed_column(df, "x", "y", "z")
    assert list(result["z"]) == [11, 22]
