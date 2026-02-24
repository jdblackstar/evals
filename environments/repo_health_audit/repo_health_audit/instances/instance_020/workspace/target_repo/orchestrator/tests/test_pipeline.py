from orchestrator.pipeline import validate_output


def test_validate_output_missing_file():
    assert validate_output("/nonexistent/path") is False


def test_validate_output_callable():
    assert callable(validate_output)


def test_run_engine_importable():
    from orchestrator.pipeline import run_engine
    assert callable(run_engine)
