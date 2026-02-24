# Atlas Monorepo

## Command policy
Two command families exist. For audits, use the `nox` sessions because they include integration checks.

- Unit-only tests: `python -m pytest packages -q`
- Canonical test command for audits: `nox -s monorepo_tests`

- Unit-only coverage: `python -m pytest packages --cov=packages --cov=libs --cov-report=term`
- Canonical coverage command for audits: `nox -s coverage_gate`
