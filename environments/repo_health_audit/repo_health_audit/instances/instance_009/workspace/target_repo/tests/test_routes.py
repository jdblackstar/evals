from webapp.app import create_app


def test_health_endpoint() -> None:
    app = create_app()
    client = app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_create_item_success() -> None:
    app = create_app()
    client = app.test_client()
    resp = client.post("/items", json={"name": "widget", "quantity": 5})
    assert resp.status_code == 201
    assert resp.get_json()["name"] == "widget"


def test_create_item_missing_name() -> None:
    app = create_app()
    client = app.test_client()
    resp = client.post("/items", json={"quantity": 1})
    assert resp.status_code == 400
    assert "error" in resp.get_json()
