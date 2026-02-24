# Microservices Platform

Python+Go microservices architecture with ECS deployment.

## Services
- `services/api/` — FastAPI-based REST API (Python)
- `services/worker/` — Background task worker (Go)

## Commands
- Test command: `./scripts/run_tests.sh`
- Coverage command: `pytest services/api --cov`

## Configuration
Environment variables are used per-service with `.env` files for each deployment stage (dev, staging, prod).

## Notes
If a command cannot run due to missing tooling, check `artifacts/` for the latest scan results.
