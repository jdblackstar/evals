# Observability Review

## Logging
- structlog is a declared dependency and actively used in services/common/events.py.
- Auth and search services use structlog indirectly through the events module.
- Ingest service does not use structured logging.

## Metrics
- No Prometheus or StatsD client found.
- No custom metrics exported from any service.

## Tracing
- No distributed tracing library (opentelemetry, jaeger) in dependencies.
- No trace context propagation headers between services.
- Elasticsearch has built-in request logging but no application-level correlation.

## Alerting
- No alerting rules or PagerDuty/OpsGenie integration found.

## Assessment
- Moderate maturity: structured logging is partially adopted via structlog in the
  event bus, but no metrics or distributed tracing infrastructure exists.
