# Deployment Status

## Gateway
- Dockerfile present with EXPOSE directive.
- ECS service definition exists in infra/modules/ecs.
- Health check endpoint at /health.
- No readiness probe configured in task definition.

## Auth
- Dockerfile present.
- No health check endpoint exposed.
- Secrets management: hardcoded key in source (not using Secrets Manager).

## Billing
- Dockerfile present.
- No Kubernetes/ECS health probe.
- Database migration strategy: not documented.

## Overall
- Rollback procedure: not documented.
- Blue/green or canary deployment: not configured.
- Partial readiness — gateway is closest to production-ready but auth and billing lack health probes.
