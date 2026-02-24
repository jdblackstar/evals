import structlog

# Circular dependency: common/events.py imports from auth/store.py,
# and auth/handler.py imports from common/events.py.
from services.auth.store import persist_event

logger = structlog.get_logger()


def publish_event(event_type: str, payload: dict) -> None:
    """Publish an event to the event bus and persist it."""
    logger.info("event_published", event_type=event_type, payload=payload)
    persist_event(event_type, payload)


def replay_events(since: str) -> list[dict]:
    """Replay events from the persistence store."""
    return []
