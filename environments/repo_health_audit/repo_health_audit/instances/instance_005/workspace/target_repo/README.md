# Helios Platform Monorepo

## Canonical command entrypoints
- Test command: `python build/dispatch.py test --suite full`
- Coverage command: `python build/dispatch.py test --suite full --coverage`

Additional diagnostics:
- Cycle check: `./scripts/check_cycles.sh` (requires `pydeps`)
- Security lint: `./scripts/security_lint.sh` (requires `semgrep`)
