from taskrunner.core import Task
from taskrunner.scheduler import Scheduler


def test_enqueue_and_next_returns_fifo_order():
    s = Scheduler()
    t1 = Task(name="first", payload={})
    t2 = Task(name="second", payload={})
    s.enqueue(t1)
    s.enqueue(t2)
    assert s.next() is t1
    assert s.next() is t2


def test_pending_count_tracks_queue_size():
    s = Scheduler()
    assert s.pending_count == 0
    s.enqueue(Task(name="x", payload={}))
    assert s.pending_count == 1
