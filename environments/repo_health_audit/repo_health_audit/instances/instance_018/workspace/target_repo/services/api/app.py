import os

from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok", "env": os.getenv("APP_ENV", "dev")}


@app.get("/items")
def list_items():
    return []
