# Acme Ledger

## Local Commands
- Test command: `python -m pytest tests -q`
- Coverage command: `python -m pytest tests --cov=src --cov-report=term-missing`

## Notes
- Service is configured through environment variables in `src/ledger/config.py`.
- Coverage snapshot is captured in `artifacts/coverage.txt`.
