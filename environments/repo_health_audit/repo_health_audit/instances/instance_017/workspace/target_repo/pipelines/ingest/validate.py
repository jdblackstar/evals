"""Validation logic for ingested records."""

import structlog
import pandas as pd
from pydantic import ValidationError

from shared.schemas import RawRecord

logger = structlog.get_logger()


def validate_records(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Validate each row against the RawRecord schema.

    Returns a tuple of (valid_df, errors).
    """
    valid_rows = []
    errors = []

    for idx, row in df.iterrows():
        try:
            record = RawRecord(**row.to_dict())
            valid_rows.append(record.model_dump())
        except ValidationError as e:
            logger.warning("validation_failed", row_index=idx, errors=e.error_count())
            errors.append({"index": idx, "errors": e.errors()})

    valid_df = pd.DataFrame(valid_rows) if valid_rows else pd.DataFrame()
    logger.info("validation_complete", valid=len(valid_rows), invalid=len(errors))
    return valid_df, errors
