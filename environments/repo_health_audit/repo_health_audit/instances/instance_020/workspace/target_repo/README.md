# Data Pipeline

Python orchestrator with Rust compute engine for high-throughput data processing.

## Structure
- `orchestrator/` — Python scheduling and pipeline logic
- `engine/` — Rust binary for data transformation

## Commands
- Test command: `make test`
- Coverage command: `make coverage`

## Configuration
Environment-based configuration via `orchestrator/config.py`. Supports dev, staging, and prod environments through environment variables.

## Notes
If a command cannot run due to missing tooling, check `artifacts/` for the latest vulnerability report.
