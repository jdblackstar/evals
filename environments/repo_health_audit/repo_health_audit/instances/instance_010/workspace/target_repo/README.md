# CLI Tool

## Local Commands
- Test command: `python -m pytest tests/`
- Coverage command: `python -m pytest tests/ --cov=src/cli --cov-report=term`

## Notes
- Entry point is `src/cli/main.py`.
- Configuration is loaded from a TOML file with environment variable overrides (see `src/cli/commands.py`).
- Coverage snapshot is captured in `artifacts/coverage_report.txt`.
