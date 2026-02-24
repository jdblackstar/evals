# Observability Review

## Logging
- structlog is listed as a dependency but only imported in auth/service.py.
- Gateway and billing use print statements, not structured logging.
- Mixed approach: moderate maturity.

## Metrics
- No Prometheus/StatsD client found in any package.
- No custom metrics exported.

## Tracing
- No distributed tracing library (opentelemetry, jaeger) in dependencies.
- No trace context propagation between services.

## Alerting
- No alerting rules or runbook references found.

## Assessment
- Structured logging partially adopted.
- No metrics or tracing infrastructure.
- Moderate maturity overall due to partial structured logging adoption.
