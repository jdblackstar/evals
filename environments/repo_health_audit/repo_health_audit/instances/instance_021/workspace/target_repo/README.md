# Nexus Monorepo

Multi-package Python monorepo with custom build orchestration.

## Canonical command entrypoints
- Test command: `python build/buildtool.py test --all`
- Coverage command: `python build/buildtool.py test --all --coverage`

## Package layout
- `packages/auth/` - Authentication service
- `packages/billing/` - Billing and payments
- `packages/gateway/` - API gateway and routing
- `packages/shared/` - Shared types and utilities

## Diagnostics
- Cycle check: `./scripts/check_cycles.sh` (requires `pydeps`)
- Security scan: `./scripts/security_scan.sh` (requires `trivy`)

## CI coverage
CI coverage reports are generated in `artifacts/coverage_ci.txt`.
Local developer coverage (artifacts/coverage_local.txt) may differ due to test fixtures.
