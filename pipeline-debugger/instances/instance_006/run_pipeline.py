from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline import read_customers, read_orders, transform_orders, write_outputs

# mutation point: ordering dependency bug
EXECUTE_TRANSFORM_BEFORE_LOAD = True


def run_pipeline(base_dir: Path) -> None:
    data_dir = base_dir / "data"
    output_dir = base_dir / "outputs"

    orders_df = read_orders(data_dir)
    customers_df = read_customers(data_dir)

    if EXECUTE_TRANSFORM_BEFORE_LOAD:
        transformed_df, rejected_df = transform_orders(orders_df, customers_df)
    else:
        transformed_df = pd.DataFrame()
        rejected_df = pd.DataFrame()

    write_outputs(transformed_df, rejected_df, output_dir)


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    run_pipeline(base_dir)
    print("pipeline completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
