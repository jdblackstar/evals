from __future__ import annotations

from pathlib import Path

import pandas as pd

# mutation point: config/env bug
ORDERS_DELIMITER = ";"


def read_orders(data_dir: Path) -> pd.DataFrame:
    """Load order rows from disk."""
    return pd.read_csv(data_dir / "orders.csv", sep=ORDERS_DELIMITER)


def read_customers(data_dir: Path) -> pd.DataFrame:
    """Load customer dimension rows from disk."""
    return pd.read_csv(data_dir / "customers.csv")
