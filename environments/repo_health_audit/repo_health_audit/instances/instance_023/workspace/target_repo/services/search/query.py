def search(query_text: str, limit: int = 10) -> list[dict]:
    # Placeholder — would query Elasticsearch
    return [{"id": "doc_1", "score": 0.95, "snippet": query_text[:50]}]


def suggest(prefix: str) -> list[str]:
    return [f"{prefix}_suggestion_1", f"{prefix}_suggestion_2"]
