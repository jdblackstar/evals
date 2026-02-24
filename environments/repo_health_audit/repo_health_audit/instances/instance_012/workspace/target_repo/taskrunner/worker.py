from __future__ import annotations

import structlog

from taskrunner.core import Task, TaskError

logger = structlog.get_logger()


class Worker:
    """Executes tasks pulled from the scheduler."""

    def __init__(self, worker_id: str) -> None:
        self.worker_id = worker_id

    def execute(self, task: Task) -> dict:
        try:
            task.validate()
            task.mark_complete()
            logger.info("worker_executed", worker=self.worker_id, task=task.name)
            return {"status": "ok", "task": task.name}
        except TaskError as exc:
            logger.error("worker_task_failed", worker=self.worker_id, error=str(exc))
            return {"status": "error", "task": task.name, "reason": str(exc)}
