from backend.db import DATABASE_URL, get_session


def test_database_url_set():
    assert "sqlite" in DATABASE_URL


def test_get_session_callable():
    assert callable(get_session)
