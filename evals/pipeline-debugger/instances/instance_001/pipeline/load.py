from __future__ import annotations

from pathlib import Path

import pandas as pd

# mutation point: wrong output path bug
OUTPUT_FILENAME = "fact_orders.csv"
REJECTED_FILENAME = "rejected_rows.csv"


def write_outputs(
    transformed_df: pd.DataFrame,
    rejected_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write deterministic pipeline outputs to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    transformed_df.to_csv(
        output_dir / OUTPUT_FILENAME,
        index=False,
        float_format="%.2f",
    )
    rejected_df.to_csv(output_dir / REJECTED_FILENAME, index=False)
