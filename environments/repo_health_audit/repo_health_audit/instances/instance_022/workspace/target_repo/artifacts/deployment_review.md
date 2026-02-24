# Deployment Review

## ETL
- Dockerfile present for containerized deployment.
- Lambda trigger configured in Terraform.
- No health check mechanism (batch workload).

## API
- Dockerfile present with EXPOSE directive.
- No readiness or liveness probes configured.
- No load balancer or ingress definition in Terraform.

## Scheduler
- No Dockerfile — runs as a Celery beat process.
- No container orchestration for scheduler service.

## Overall
- Two of three services are containerized (partial).
- No rollback strategy documented.
- No blue/green or canary deployment mechanism.
- Database migrations are manual (`make migrate`), not automated in CI.
