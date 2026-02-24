"""Shared data models used across packages."""

from pydantic import BaseModel


class BaseItem(BaseModel):
    name: str
    category: str
    value: float = 0.0


class ProcessingResult(BaseModel):
    item_name: str
    status: str
    error: str | None = None
