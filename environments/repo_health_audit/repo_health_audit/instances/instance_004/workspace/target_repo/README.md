# Northstar Platform

## Commands
- Test command: `./scripts/run_tests.sh`
- Coverage command: `poetry run pytest services --cov=services --cov-report=term`
- Security scan command: `trivy fs .`

If a command cannot run due to missing tooling, use the latest internal artifact snapshots in `artifacts/` and document the limitation.
