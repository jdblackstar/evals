# Deployment Checklist

## Auth Service
- Dockerfile present.
- EKS deployment target configured.
- Health endpoint: not implemented (missing /health route).
- Secrets: hardcoded production endpoint in handler.py.

## Search Service
- Dockerfile present.
- Elasticsearch cluster provisioned via Terraform.
- Health check: Elasticsearch cluster health API available, but no application-level probe.

## Ingest Service
- Dockerfile present.
- Runs as batch process, no HTTP endpoint.
- No readiness probe applicable.

## Overall
- Three services containerized (partial readiness).
- No rolling update or canary strategy defined.
- No rollback playbook documented.
- Database/Redis migrations: not automated.
- Auth hardcoded endpoint prevents environment promotion.
