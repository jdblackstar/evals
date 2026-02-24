# Observability Audit

## Logging
- structlog is listed as a dependency but not imported in any service file.
- All services use bare print() statements for output.
- No structured logging in practice.

## Metrics
- No Prometheus, StatsD, or CloudWatch metrics client found.
- No custom metrics exported from any service.

## Tracing
- No distributed tracing library in dependencies.
- No trace context propagation between ETL, API, or scheduler.

## Alerting
- No alerting configuration or runbook references.

## Assessment
- Weak observability maturity: dependency declared but not used, no metrics, no tracing.
