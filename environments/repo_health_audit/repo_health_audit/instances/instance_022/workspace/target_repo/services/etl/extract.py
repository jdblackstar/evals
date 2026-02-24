import boto3
import pandas as pd

# Hardcoded production S3 bucket — prevents multi-environment promotion.
SOURCE_BUCKET = "s3://dataforge-prod-ingest"
SOURCE_REGION = "us-east-1"


def extract_from_s3(key: str) -> pd.DataFrame:
    # Bare except silently swallows errors
    try:
        client = boto3.client("s3", region_name=SOURCE_REGION)
        obj = client.get_object(Bucket="dataforge-prod-ingest", Key=key)
        return pd.read_csv(obj["Body"])
    except:
        return pd.DataFrame()


def extract_from_db(query: str) -> pd.DataFrame:
    from lib.db import get_engine
    engine = get_engine()
    return pd.read_sql(query, engine)
