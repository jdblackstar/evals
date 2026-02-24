"""Notification sender using Celery and Redis."""

import structlog
from celery import Celery

from libs.common.config import get_redis_url

logger = structlog.get_logger()

app = Celery("notifications", broker=get_redis_url())


@app.task
def send_notification(user_id: str, message: str, channel: str = "email"):
    """Send a notification to a user via the specified channel."""
    logger.info("sending_notification", user_id=user_id, channel=channel)
    # Placeholder: would integrate with email/SMS/push provider
    return {"user_id": user_id, "channel": channel, "status": "sent"}


@app.task
def send_batch_notifications(user_ids: list[str], message: str):
    """Send the same notification to multiple users."""
    results = []
    for uid in user_ids:
        result = send_notification(uid, message)
        results.append(result)
    return results
