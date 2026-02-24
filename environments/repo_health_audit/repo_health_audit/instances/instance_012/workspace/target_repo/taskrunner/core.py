from __future__ import annotations

import structlog

logger = structlog.get_logger()


class TaskError(Exception):
    """Raised when a task fails to execute."""


class Task:
    def __init__(self, name: str, payload: dict) -> None:
        self.name = name
        self.payload = payload
        self.status = "pending"

    def validate(self) -> bool:
        if not self.name:
            raise TaskError("task name must not be empty")
        if not isinstance(self.payload, dict):
            raise TaskError("payload must be a dict")
        return True

    def mark_complete(self) -> None:
        self.status = "complete"
        logger.info("task_completed", task=self.name)
