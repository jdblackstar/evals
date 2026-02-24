from __future__ import annotations

from datetime import datetime

import pandas as pd

# mutation point: schema drift / join key bug
JOIN_KEY = "customer_id"
# mutation point: join logic bug
JOIN_HOW = "left"
# mutation point: silent data loss bug
DROP_VALID_ROWS_BELOW = None
# mutation point: error handling bug
CRASH_ON_BAD_ROW = True

SEGMENT_DISCOUNT_RATES = {
    "enterprise": 0.10,
    "smb": 0.05,
    "consumer": 0.00,
}

OUTPUT_COLUMNS = [
    "order_id",
    "customer_id",
    "customer_name",
    "segment",
    "order_date",
    "amount",
    "discount",
    "net_amount",
    "is_high_value",
    "processing_status",
]


def transform_orders(
    orders_df: pd.DataFrame,
    customers_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean orders, enrich with customers, and compute derived fields."""
    cleaned_rows: list[dict] = []
    rejected_rows: list[dict] = []

    for row_index, raw in orders_df.iterrows():
        try:
            order_id = int(raw["order_id"])
            customer_id = int(raw["customer_id"])
            order_date = datetime.strptime(str(raw["order_date"]), "%Y-%m-%d").date()
            # mutation point: type coercion bug
            amount = float(raw["amount"])
            if amount < 0:
                raise ValueError("amount must be non-negative")
        except Exception as exc:  # malformed rows are rejected, not fatal
            if CRASH_ON_BAD_ROW:
                raise
            rejected_rows.append(
                {
                    "row_number": int(row_index),
                    "order_id": str(raw.get("order_id", "")),
                    "reason": str(exc),
                }
            )
            continue

        cleaned_rows.append(
            {
                "order_id": order_id,
                "customer_id": customer_id,
                "order_date": order_date.isoformat(),
                "amount": amount,
            }
        )

    cleaned_df = pd.DataFrame(cleaned_rows)
    if cleaned_df.empty:
        cleaned_df = pd.DataFrame(
            columns=["order_id", "customer_id", "order_date", "amount"]
        )

    if DROP_VALID_ROWS_BELOW is not None:
        cleaned_df = cleaned_df[cleaned_df["amount"] >= DROP_VALID_ROWS_BELOW]

    merged = cleaned_df.merge(customers_df, how=JOIN_HOW, on=JOIN_KEY, sort=False)
    merged["customer_name"] = merged["customer_name"].fillna("unknown")
    merged["segment"] = merged["segment"].fillna("consumer")

    discount_rates = merged["segment"].map(SEGMENT_DISCOUNT_RATES).fillna(0.0)
    merged["discount"] = (merged["amount"] * discount_rates).round(2)
    merged["net_amount"] = (merged["amount"] - merged["discount"]).round(2)
    merged["is_high_value"] = merged["net_amount"] >= 1000.0
    merged["processing_status"] = "ok"

    transformed_df = (
        merged[OUTPUT_COLUMNS].sort_values("order_id").reset_index(drop=True)
    )

    rejected_df = pd.DataFrame(rejected_rows)
    if rejected_df.empty:
        rejected_df = pd.DataFrame(columns=["row_number", "order_id", "reason"])

    return transformed_df, rejected_df
