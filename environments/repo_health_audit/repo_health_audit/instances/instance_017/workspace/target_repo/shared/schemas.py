"""Pydantic schemas shared across pipeline stages."""

from datetime import datetime
from pydantic import BaseModel, Field


class RawRecord(BaseModel):
    """Schema for raw ingested records before cleaning."""
    id: str
    timestamp: datetime
    category: str
    value: float
    source: str
    metadata: dict = Field(default_factory=dict)


class CleanedRecord(BaseModel):
    """Schema for records after cleaning and validation."""
    id: str
    timestamp: datetime
    category: str
    value: float
    source: str
    is_valid: bool = True


class AggregatedMetric(BaseModel):
    """Schema for aggregated output metrics."""
    group_key: str
    metric_name: str
    total: float
    mean: float
    count: int
