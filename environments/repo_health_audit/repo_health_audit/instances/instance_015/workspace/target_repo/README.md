# Nx-Style Workspace

Monorepo containing API and Worker services with shared models.

## Structure

- `packages/api/` - FastAPI-based HTTP API
- `packages/worker/` - Celery-based background worker
- `packages/shared/` - Shared models and utilities

## Testing

Run all tests:

```bash
python -m pytest packages/ -v
```

## Coverage

Generate a combined coverage report:

```bash
python -m pytest packages/ --cov=packages --cov-report=term
```

## Deployment

Each service has its own Dockerfile. Infrastructure is managed via Terraform in `infra/`.
