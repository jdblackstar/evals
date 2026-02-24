import pytest


@pytest.fixture
def mock_redis():
    """Shared fixture providing a mock Redis connection for all test suites."""
    class MockRedis:
        def __init__(self):
            self._store = {}

        def hgetall(self, key):
            return self._store.get(key, {})

        def setex(self, key, ttl, value):
            self._store[key] = value

        def lpush(self, key, value):
            self._store.setdefault(key, []).append(value)

    return MockRedis()
