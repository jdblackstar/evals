"""Application configuration — hardcoded defaults, no multi-env support."""

REDIS_URL = "redis://localhost:6379/0"
BROKER_URL = "redis://localhost:6379/1"
MAX_RETRIES = 3
WORKER_CONCURRENCY = 4
LOG_LEVEL = "INFO"
