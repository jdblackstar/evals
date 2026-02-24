"""Queue management utilities."""

import redis


_connection = None


def get_queue(url: str = "redis://localhost:6379/0"):
    global _connection
    if _connection is None:
        _connection = redis.from_url(url)
    return _connection


def queue_length(queue_name: str = "celery") -> int:
    conn = get_queue()
    return conn.llen(queue_name)
