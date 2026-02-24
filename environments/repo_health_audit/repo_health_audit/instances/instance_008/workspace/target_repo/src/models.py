"""Data models."""

from pydantic import BaseModel


class SyncRecord(BaseModel):
    source: str
    target: str
    status: str = "pending"
