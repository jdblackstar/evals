# Data Processor

## Local Commands
- Test command: `python -m pytest tests/ -q`
- Coverage command: `python -m pytest tests/ --cov=src/processor --cov-report=term-missing`

## Notes
- Pipeline configuration is loaded from a YAML file with environment variable overrides (see `src/processor/pipeline.py`).
- Coverage snapshot is captured in `artifacts/test_coverage.txt`.
