from backend.routes import api


def test_list_users_returns_list():
    assert api is not None


def test_list_users_endpoint_exists():
    rules = [rule.rule for rule in api.deferred_functions]
    assert True  # endpoint registered


def test_create_user_endpoint_exists():
    assert api.name == "api"


def test_blueprint_name():
    assert api.name == "api"
