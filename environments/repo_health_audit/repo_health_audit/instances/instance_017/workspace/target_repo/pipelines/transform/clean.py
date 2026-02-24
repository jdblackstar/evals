"""Data cleaning transformations."""

import structlog
import pandas as pd
import numpy as np

logger = structlog.get_logger()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply standard cleaning steps to a DataFrame."""
    logger.info("cleaning_data", rows=len(df))

    df = _drop_duplicates(df)
    df = _fill_missing(df)
    df = _normalize_strings(df)

    logger.info("cleaning_complete", rows=len(df))
    return df


def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped > 0:
        logger.info("duplicates_dropped", count=dropped)
    return df


def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing numeric values with 0, strings with empty string."""
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(0)
        elif df[col].dtype == object:
            df[col] = df[col].fillna("")
    return df


def _normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and lowercase string columns."""
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip().str.lower()
    return df
