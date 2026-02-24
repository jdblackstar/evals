import redis

# This module is imported by services/common/events.py for event persistence,
# creating a circular dependency: auth -> common -> auth
_redis = redis.Redis(host="localhost", port=6379, db=0)


def get_user(username: str) -> dict | None:
    data = _redis.hgetall(f"user:{username}")
    if not data:
        return None
    return {"id": username, "data": data}


def save_session(user_id: str) -> str:
    import uuid
    session_id = str(uuid.uuid4())
    _redis.setex(f"session:{session_id}", 3600, user_id)
    return session_id


def persist_event(event_type: str, payload: dict) -> None:
    """Store event in Redis for durability. Called by common/events.py."""
    import json
    _redis.lpush("events:log", json.dumps({"type": event_type, **payload}))
