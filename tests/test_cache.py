"""Tests for evals.cache module."""

import asyncio
import tempfile
from pathlib import Path

from evals.cache import CacheStore, get_cache


class TestGetCache:
    """Tests for the get_cache function."""

    def test_same_path_returns_same_instance(self):
        """Test that calling get_cache with the same path returns the same instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            cache1 = get_cache(db_path)
            cache2 = get_cache(db_path)

            assert cache1 is cache2
            assert cache1.db_path == cache2.db_path

    def test_different_paths_return_different_instances(self):
        """Test that calling get_cache with different paths returns different instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path1 = Path(tmpdir) / "cache1.db"
            db_path2 = Path(tmpdir) / "cache2.db"

            cache1 = get_cache(db_path1)
            cache2 = get_cache(db_path2)

            assert cache1 is not cache2
            assert cache1.db_path != cache2.db_path
            assert cache1.db_path == db_path1
            assert cache2.db_path == db_path2

    def test_path_normalization(self):
        """Test that paths are normalized correctly (resolved absolute paths)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            db_path1 = tmpdir_path / "test.db"
            db_path2 = tmpdir_path / ".." / tmpdir_path.name / "test.db"

            cache1 = get_cache(db_path1)
            cache2 = get_cache(db_path2)

            # Should return the same instance since paths resolve to the same location
            assert cache1 is cache2

    def test_default_path(self):
        """Test that default path works correctly."""
        cache1 = get_cache()
        cache2 = get_cache(".cache/responses.db")

        # Should return the same instance if paths resolve to the same location
        # Compare resolved paths since CacheStore stores paths as-is
        assert cache1.db_path.resolve() == cache2.db_path.resolve()
        assert cache1 is cache2

    def test_string_and_path_equivalence(self):
        """Test that string and Path objects for the same path return the same instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path_str = str(Path(tmpdir) / "test.db")
            db_path_path = Path(tmpdir) / "test.db"

            cache1 = get_cache(db_path_str)
            cache2 = get_cache(db_path_path)

            assert cache1 is cache2


class TestCacheStoreEventLoop:
    """Tests for CacheStore event loop handling."""

    def test_multiple_event_loops(self):
        """Test that cache works correctly across multiple asyncio.run() calls."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            cache = CacheStore(db_path)

            async def _use_cache():
                """Use cache in an async context."""
                await cache.set("test-model", "test prompt", {}, {"result": "test"})
                result = await cache.get("test-model", "test prompt", {})
                assert result is not None
                assert result.response == {"result": "test"}
                return True

            # Each asyncio.run() creates a new event loop
            # This should not raise RuntimeError about locks attached to different loops
            result1 = asyncio.run(_use_cache())
            result2 = asyncio.run(_use_cache())

            assert result1 is True
            assert result2 is True

            # Verify the cache persists across event loops
            async def _verify_cache():
                """Verify cache entry persists."""
                result = await cache.get("test-model", "test prompt", {})
                assert result is not None
                return result.response

            result = asyncio.run(_verify_cache())
            assert result == {"result": "test"}
