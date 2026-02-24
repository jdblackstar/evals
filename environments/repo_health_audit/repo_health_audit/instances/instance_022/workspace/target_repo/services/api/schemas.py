from pydantic import BaseModel


class QueryRequest(BaseModel):
    sql: str
    limit: int = 100


class QueryResponse(BaseModel):
    result: list[dict]
    query: str
    row_count: int = 0
