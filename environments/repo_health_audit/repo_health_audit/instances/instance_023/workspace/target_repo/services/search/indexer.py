from services.common.events import publish_event


def index_document(doc_id: str, content: str) -> bool:
    # Placeholder — would call Elasticsearch
    publish_event("doc.indexed", {"doc_id": doc_id})
    return True


def bulk_index(documents: list[dict]) -> int:
    indexed = 0
    for doc in documents:
        if index_document(doc["id"], doc["content"]):
            indexed += 1
    return indexed
