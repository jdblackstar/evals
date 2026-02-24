# Data Platform

A modular data pipeline with ingest, transform, and export stages.

## Structure

- `pipelines/ingest/` - Source connectors and validation
- `pipelines/transform/` - Cleaning and aggregation
- `pipelines/export/` - Sink connectors (S3, local)
- `shared/` - Pydantic schemas shared across stages
- `infra/` - Terraform modules for AWS Glue

## Setup

```bash
pip install -e ".[dev]"
```

## Testing

Run the full test suite:

```bash
nox -s test
```

## Coverage

The canonical coverage report is generated in CI. To run locally:

```bash
nox -s coverage
```

The CI coverage report is stored at `artifacts/coverage_ci.txt` and is the authoritative source.
Local coverage (`artifacts/coverage_local.txt`) may differ due to environment differences.

## Deployment

A single Dockerfile is provided at the repo root for containerized execution.
Infrastructure is provisioned via Terraform (AWS Glue) in `infra/`.
