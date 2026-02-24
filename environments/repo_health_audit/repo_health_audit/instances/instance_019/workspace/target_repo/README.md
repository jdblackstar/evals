# Full-Stack Application

Python backend (Flask) with Node.js frontend (React).

## Structure
- `backend/` — Flask REST API with SQLAlchemy
- `frontend/` — React single-page application

## Commands
- Test command: `./scripts/test_backend.sh`
- Coverage command: `coverage run -m pytest backend/tests && coverage report`

## Deployment
Docker containers for both backend and frontend. Infrastructure managed via Terraform.

## Notes
If a command cannot run due to missing tooling, check `artifacts/` for the latest audit results.
