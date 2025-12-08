"""
Caching layer for API responses.

Uses SQLite for persistent, async-compatible caching of model completions.
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite


@dataclass
class CacheEntry:
    """A cached API response."""

    key: str
    response: dict[str, Any]
    model: str
    created_at: float
    metadata: dict[str, Any] | None = None


class CacheStore:
    """
    SQLite-based cache for model API responses.

    Cache keys are hashes of (model, prompt, params) to ensure
    identical requests return cached results.
    """

    def __init__(self, db_path: Path | str = ".cache/responses.db") -> None:
        """
        Initialize the cache store.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def _ensure_initialized(self) -> None:
        """Initialize the database schema if needed."""
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        response TEXT NOT NULL,
                        model TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        metadata TEXT
                    )
                """)
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_cache_model ON cache(model)
                """)
                await db.commit()

            self._initialized = True

    @staticmethod
    def _make_key(model: str, prompt: str, params: dict[str, Any]) -> str:
        """
        Generate a cache key from request parameters.

        Args:
            model: Model identifier.
            prompt: The prompt text.
            params: Additional parameters (temperature, etc.).

        Returns:
            SHA256 hash as hex string.
        """
        # Sort params for consistent hashing
        sorted_params = json.dumps(params, sort_keys=True)
        content = f"{model}|{prompt}|{sorted_params}"
        return hashlib.sha256(content.encode()).hexdigest()

    async def get(
        self,
        model: str,
        prompt: str,
        params: dict[str, Any],
    ) -> CacheEntry | None:
        """
        Retrieve a cached response.

        Args:
            model: Model identifier.
            prompt: The prompt text.
            params: Additional parameters.

        Returns:
            CacheEntry if found, None otherwise.
        """
        await self._ensure_initialized()
        key = self._make_key(model, prompt, params)

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM cache WHERE key = ?", (key,))
            row = await cursor.fetchone()

            if row is None:
                return None

            return CacheEntry(
                key=row["key"],
                response=json.loads(row["response"]),
                model=row["model"],
                created_at=row["created_at"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            )

    async def set(
        self,
        model: str,
        prompt: str,
        params: dict[str, Any],
        response: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Store a response in the cache.

        Args:
            model: Model identifier.
            prompt: The prompt text.
            params: Additional parameters.
            response: The API response to cache.
            metadata: Optional metadata to store.

        Returns:
            The cache key.
        """
        await self._ensure_initialized()
        key = self._make_key(model, prompt, params)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO cache (key, response, model, created_at, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    key,
                    json.dumps(response),
                    model,
                    time.time(),
                    json.dumps(metadata) if metadata else None,
                ),
            )
            await db.commit()

        return key

    async def has(
        self,
        model: str,
        prompt: str,
        params: dict[str, Any],
    ) -> bool:
        """
        Check if a response is cached.

        Args:
            model: Model identifier.
            prompt: The prompt text.
            params: Additional parameters.

        Returns:
            True if cached, False otherwise.
        """
        entry = await self.get(model, prompt, params)
        return entry is not None

    async def delete(self, key: str) -> bool:
        """
        Delete a cached entry.

        Args:
            key: The cache key to delete.

        Returns:
            True if deleted, False if not found.
        """
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("DELETE FROM cache WHERE key = ?", (key,))
            await db.commit()
            return cursor.rowcount > 0

    async def clear(self, model: str | None = None) -> int:
        """
        Clear cached entries.

        Args:
            model: If provided, only clear entries for this model.

        Returns:
            Number of entries deleted.
        """
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            if model:
                cursor = await db.execute("DELETE FROM cache WHERE model = ?", (model,))
            else:
                cursor = await db.execute("DELETE FROM cache")
            await db.commit()
            return cursor.rowcount

    async def stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats.
        """
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM cache")
            total = (await cursor.fetchone())[0]

            cursor = await db.execute(
                "SELECT model, COUNT(*) FROM cache GROUP BY model"
            )
            by_model = {row[0]: row[1] for row in await cursor.fetchall()}

            size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0

        return {
            "total_entries": total,
            "by_model": by_model,
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 2),
        }


_cache_instances: dict[str, CacheStore] = {}


def get_cache(db_path: Path | str = ".cache/responses.db") -> CacheStore:
    """
    Get or create a cache instance for the given database path.

    Each unique db_path gets its own cache instance. This allows
    multiple cache instances to coexist with different database paths.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        CacheStore instance for the given path.
    """
    global _cache_instances
    # Normalize path to string for consistent key lookup
    path_str = str(Path(db_path).resolve())

    if path_str not in _cache_instances:
        _cache_instances[path_str] = CacheStore(db_path)

    return _cache_instances[path_str]
