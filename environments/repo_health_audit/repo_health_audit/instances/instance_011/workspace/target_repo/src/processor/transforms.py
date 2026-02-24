from __future__ import annotations

import pandas as pd


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and strip whitespace from column names."""
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def filter_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where all values are null."""
    return df.dropna(how="all")


def add_computed_column(df: pd.DataFrame, col_a: str, col_b: str, target: str) -> pd.DataFrame:
    """Add a column that is the sum of two existing columns.

    NOTE: This function does not validate inputs — callers are expected
    to handle KeyError themselves.
    """
    df[target] = df[col_a] + df[col_b]
    return df
