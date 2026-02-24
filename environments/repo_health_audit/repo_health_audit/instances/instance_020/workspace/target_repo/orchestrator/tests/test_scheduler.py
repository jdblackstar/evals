from orchestrator.scheduler import PipelineScheduler


def test_should_run_initially():
    scheduler = PipelineScheduler()
    assert scheduler.should_run() is True


def test_should_not_run_after_mark():
    scheduler = PipelineScheduler(interval_seconds=9999)
    scheduler.mark_complete()
    assert scheduler.should_run() is False
