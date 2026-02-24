import pandas as pd
from lib.db import get_engine


def load_to_warehouse(df: pd.DataFrame, table_name: str) -> int:
    engine = get_engine()
    rows = df.to_sql(table_name, engine, if_exists="append", index=False)
    return rows or 0


def load_to_parquet(df: pd.DataFrame, path: str) -> None:
    # No error handling — will crash on write failure
    df.to_parquet(path, engine="pyarrow")
