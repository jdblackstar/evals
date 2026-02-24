from __future__ import annotations

from collections import deque

from taskrunner.core import Task, TaskError


class Scheduler:
    """Simple FIFO task scheduler backed by a deque."""

    def __init__(self) -> None:
        self._queue: deque[Task] = deque()

    def enqueue(self, task: Task) -> None:
        try:
            task.validate()
        except TaskError:
            raise
        self._queue.append(task)

    def next(self) -> Task | None:
        if self._queue:
            return self._queue.popleft()
        return None

    @property
    def pending_count(self) -> int:
        return len(self._queue)
