"""Data source connectors for the ingest pipeline."""

import structlog
import boto3
import pandas as pd

from shared.schemas import RawRecord

logger = structlog.get_logger()


def read_source(source_type: str, path: str) -> pd.DataFrame:
    """Read data from the specified source."""
    logger.info("reading_source", source_type=source_type, path=path)

    if source_type == "s3":
        return _read_s3(path)
    elif source_type == "local":
        return _read_local(path)
    else:
        raise ValueError(f"Unsupported source type: {source_type}")


def _read_s3(path: str) -> pd.DataFrame:
    """Read a parquet file from S3."""
    try:
        s3 = boto3.client("s3")
        bucket, key = path.replace("s3://", "").split("/", 1)
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_parquet(obj["Body"])
    except Exception as e:
        logger.error("s3_read_failed", path=path, error=str(e))
        raise


def _read_local(path: str) -> pd.DataFrame:
    """Read a local parquet or CSV file."""
    try:
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        elif path.endswith(".csv"):
            return pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {path}")
    except FileNotFoundError:
        logger.error("local_file_not_found", path=path)
        raise
