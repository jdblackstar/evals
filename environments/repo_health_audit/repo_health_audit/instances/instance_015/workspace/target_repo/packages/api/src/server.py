"""FastAPI application server."""

from fastapi import FastAPI

from packages.shared.src.models import BaseItem
from packages.worker.src.tasks import enqueue_processing


def create_app() -> FastAPI:
    app = FastAPI(title="API Service")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/items")
    async def submit_item(item: BaseItem):
        enqueue_processing(item.dict())
        return {"queued": True}

    return app


app = create_app()
