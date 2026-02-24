from services.api.app import health, list_items


def test_health_status():
    result = health()
    assert result["status"] == "ok"


def test_health_env_default():
    result = health()
    assert result["env"] == "dev"


def test_list_items_empty():
    result = list_items()
    assert result == []
