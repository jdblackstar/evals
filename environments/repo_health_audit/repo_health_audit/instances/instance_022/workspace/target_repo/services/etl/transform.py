import pandas as pd
import numpy as np


def clean_nulls(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()


def normalize_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = df[col].str.lower().str.strip()
    return df


def aggregate_daily(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    return df.groupby(pd.Grouper(key=date_col, freq="D"))[value_col].sum().reset_index()
