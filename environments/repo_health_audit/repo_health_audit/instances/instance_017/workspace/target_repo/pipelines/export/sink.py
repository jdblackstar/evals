"""Data sink connectors for the export pipeline."""

import structlog
import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = structlog.get_logger()


def write_sink(df: pd.DataFrame, sink_type: str, path: str) -> int:
    """Write a DataFrame to the specified sink.

    Returns the number of rows written.
    """
    logger.info("writing_sink", sink_type=sink_type, path=path, rows=len(df))

    try:
        if sink_type == "s3":
            return _write_s3(df, path)
        elif sink_type == "local":
            return _write_local(df, path)
        else:
            raise ValueError(f"Unsupported sink type: {sink_type}")
    except Exception as e:
        logger.error("sink_write_failed", sink_type=sink_type, path=path, error=str(e))
        raise


def _write_s3(df: pd.DataFrame, path: str) -> int:
    """Write a parquet file to S3."""
    table = pa.Table.from_pandas(df)
    bucket, key = path.replace("s3://", "").split("/", 1)
    s3 = boto3.client("s3")
    buf = pa.BufferOutputStream()
    pq.write_table(table, buf)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue().to_pybytes())
    return len(df)


def _write_local(df: pd.DataFrame, path: str) -> int:
    """Write a parquet file locally."""
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)
    return len(df)
