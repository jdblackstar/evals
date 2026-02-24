from fastapi import FastAPI
from services.api.schemas import QueryRequest

app = FastAPI(title="DataForge API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def run_query(req: QueryRequest):
    # Placeholder — no input validation beyond Pydantic
    return {"result": [], "query": req.sql}


@app.get("/tables")
def list_tables():
    return {"tables": ["events", "users", "transactions"]}
