# Beacon Search Platform

CI-driven monorepo for search, ingest, and auth services.

## Canonical command entrypoints
- Test command: `python .ci/run_tests.py --suite all`
- Coverage command: `python .ci/run_tests.py --suite all --coverage`

## Service layout
- `services/auth/` - Authentication and session management
- `services/search/` - Elasticsearch-backed search service
- `services/ingest/` - Document ingestion and validation
- `services/common/` - Shared event bus and utilities

## Diagnostics
- Cycle check: `./scripts/check_cycles.sh` (requires `pydeps`)
- Security scan: `./scripts/vuln_scan.sh` (requires `trivy`)

## Coverage artifacts
Three coverage reports are generated per CI run:
- `artifacts/coverage_unit.txt` — unit tests only
- `artifacts/coverage_integration.txt` — integration tests only
- `artifacts/coverage_combined.txt` — canonical combined coverage
