# DataSync

A full-featured data synchronization platform.

## Features

- High-performance sync engine
- Comprehensive test suite with 95% coverage
- Docker-based deployment
- Terraform infrastructure modules for AWS

## Testing

```bash
pytest tests/ -v --cov=src
```

## Deployment

```bash
docker compose up -d
terraform apply -auto-approve
```
