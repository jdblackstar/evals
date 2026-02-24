"""Environment-based configuration for all services."""

import os

ENV = os.environ.get("APP_ENV", "development")

_DEFAULTS = {
    "development": {
        "database_url": "sqlite:///dev.db",
        "redis_url": "redis://localhost:6379/0",
        "gateway_host": "localhost",
        "gateway_port": "5000",
        "service_urls": {
            "users": "http://localhost:5001",
            "notifications": "http://localhost:5002",
        },
    },
    "staging": {
        "database_url": "postgresql://staging-db:5432/app",
        "redis_url": "redis://staging-redis:6379/0",
        "gateway_host": "0.0.0.0",
        "gateway_port": "8080",
        "service_urls": {
            "users": "http://users-svc:8080",
            "notifications": "http://notifications-svc:8080",
        },
    },
    "production": {
        "database_url": os.environ.get("DATABASE_URL", ""),
        "redis_url": os.environ.get("REDIS_URL", ""),
        "gateway_host": "0.0.0.0",
        "gateway_port": "8080",
        "service_urls": {
            "users": "http://users-svc:8080",
            "notifications": "http://notifications-svc:8080",
        },
    },
}


def _get_config() -> dict:
    return _DEFAULTS.get(ENV, _DEFAULTS["development"])


def get_database_url() -> str:
    return os.environ.get("DATABASE_URL", _get_config()["database_url"])


def get_redis_url() -> str:
    return os.environ.get("REDIS_URL", _get_config()["redis_url"])


def get_service_url(service_name: str) -> str:
    config = _get_config()
    return config["service_urls"].get(service_name, f"http://{service_name}:8080")
