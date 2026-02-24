# DataForge Platform

Makefile-orchestrated data platform with ETL, API, and scheduler services.

## Canonical command entrypoints
- Test command: `make test`
- Coverage command: `make coverage`

## Service layout
- `services/etl/` - Extract-transform-load pipeline
- `services/api/` - REST API for data access
- `services/scheduler/` - Cron-based job scheduler
- `lib/` - Shared database and utility library

## Migrations
SQL migration files are in `migrations/`. Run with `make migrate`.

## Security scanning
- Static analysis: `semgrep --config=auto` (requires semgrep)
- Dependency audit: `bandit -r services/` (requires bandit)

## CI coverage
Combined coverage is reported in `artifacts/coverage_combined.txt`.
Per-service coverage files (e.g., `artifacts/coverage_etl_only.txt`) are for
debugging and do not reflect full-platform coverage.
