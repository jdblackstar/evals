from taskrunner.core import Task, TaskError


def test_task_validate_accepts_valid_task():
    t = Task(name="send_email", payload={"to": "user@example.com"})
    assert t.validate() is True


def test_task_validate_rejects_empty_name():
    t = Task(name="", payload={})
    try:
        t.validate()
        assert False, "expected TaskError"
    except TaskError:
        assert True


def test_task_mark_complete_sets_status():
    t = Task(name="cleanup", payload={"dry_run": True})
    t.mark_complete()
    assert t.status == "complete"
