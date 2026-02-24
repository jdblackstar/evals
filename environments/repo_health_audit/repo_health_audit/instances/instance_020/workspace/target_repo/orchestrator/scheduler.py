import time
from datetime import datetime


class PipelineScheduler:
    def __init__(self, interval_seconds: int = 300):
        self.interval = interval_seconds
        self.last_run = None

    def should_run(self) -> bool:
        if self.last_run is None:
            return True
        elapsed = (datetime.now() - self.last_run).total_seconds()
        return elapsed >= self.interval

    def mark_complete(self):
        self.last_run = datetime.now()

    def wait_for_next(self):
        time.sleep(self.interval)
