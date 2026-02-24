"""Celery task definitions."""

from celery import Celery

from packages.shared.src.models import BaseItem
from packages.api.src.handlers import list_items  # circular: worker imports from api

app = Celery("worker", broker="redis://localhost:6379/0")


@app.task
def process_item(payload: dict) -> dict:
    item = BaseItem(**payload)
    # Simulate processing
    return {"processed": item.name, "status": "done"}


def enqueue_processing(payload: dict):
    """Submit an item for async processing."""
    process_item.delay(payload)
