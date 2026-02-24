from services.api.app import get_runtime_config


def test_env_default() -> None:
    assert get_runtime_config()["api_env"] == "dev"


def test_db_default() -> None:
    assert "sqlite" in get_runtime_config()["db_url"]


def test_config_keys() -> None:
    keys = set(get_runtime_config())
    assert keys == {"api_env", "db_url"}


def test_env_present() -> None:
    assert "api_env" in get_runtime_config()


def test_db_present() -> None:
    assert "db_url" in get_runtime_config()
