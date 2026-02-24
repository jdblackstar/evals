import os


def get_runtime_config() -> dict[str, str]:
    return {
        "api_env": os.getenv("API_ENV", "dev"),
        "db_url": os.getenv("DATABASE_URL", "sqlite:///service.db"),
    }
