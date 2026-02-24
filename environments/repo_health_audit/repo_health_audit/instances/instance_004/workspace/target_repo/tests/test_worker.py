from services.worker.runner import run_job


def test_run_job_ok() -> None:
    assert run_job({"id": 1})["ok"] is True


def test_run_job_returns_id() -> None:
    assert run_job({"id": 9})["id"] == 9


def test_run_job_requires_id() -> None:
    try:
        run_job({})
        assert False, "expected ValueError"
    except ValueError:
        assert True


def test_run_job_bool_ok() -> None:
    assert isinstance(run_job({"id": 5})["ok"], bool)
