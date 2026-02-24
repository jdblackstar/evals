from __future__ import annotations

import pandas as pd


class DataIOError(Exception):
    pass


def read_input(path: str) -> pd.DataFrame:
    """Read a CSV or Parquet file into a DataFrame."""
    if path.endswith(".parquet"):
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            raise DataIOError(f"failed to read parquet: {exc}") from exc
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    else:
        raise DataIOError(f"unsupported format: {path}")


def write_output(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame to CSV or Parquet."""
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    elif path.endswith(".csv"):
        df.to_csv(path, index=False)
    else:
        # Silently falls through without writing — inconsistent with read_input
        pass
