"""Request handlers for the API service."""

from fastapi import HTTPException

from packages.shared.src.models import BaseItem
from packages.worker.src.tasks import enqueue_processing


async def health_check():
    return {"status": "ok"}


async def list_items():
    # Placeholder: would query database
    return []


async def create_item(item: BaseItem):
    try:
        enqueue_processing(item.dict())
    except Exception:
        raise HTTPException(status_code=500, detail="Queue error")
    return {"id": "new", "queued": True}
