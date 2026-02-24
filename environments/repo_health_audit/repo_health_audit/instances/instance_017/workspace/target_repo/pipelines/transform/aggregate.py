"""Aggregation transformations for metrics computation."""

import structlog
import pandas as pd

logger = structlog.get_logger()


def aggregate_metrics(df: pd.DataFrame, group_by: str, metrics: list[str]) -> pd.DataFrame:
    """Aggregate numeric columns by a grouping key.

    Args:
        df: Input DataFrame.
        group_by: Column name to group by.
        metrics: List of numeric columns to aggregate.

    Returns:
        Aggregated DataFrame with sum, mean, count for each metric.
    """
    logger.info("aggregating", group_by=group_by, metrics=metrics)

    try:
        agg_funcs = {m: ["sum", "mean", "count"] for m in metrics}
        result = df.groupby(group_by).agg(agg_funcs)
        result.columns = ["_".join(col) for col in result.columns]
        result = result.reset_index()
    except KeyError as e:
        logger.error("aggregation_failed", missing_column=str(e))
        raise

    logger.info("aggregation_complete", output_rows=len(result))
    return result
