"""
Model runner for executing completions via OpenRouter.

Provides async-first execution with caching, rate limiting, and batch processing.
Supports both single-turn and multi-turn conversations.
"""

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from evals.cache import CacheStore, get_cache
from evals.config import ModelConfig
from evals.primitives import Turn

load_dotenv()


@dataclass
class Completion:
    """A single model completion result."""

    content: str
    model: str
    prompt: str
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    cached: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_used(self) -> int:
        """Total tokens used for this completion."""
        if self.usage:
            return self.usage.get("total_tokens", 0)
        return 0


@dataclass
class BatchResult:
    """Results from a batch of completions."""

    completions: list[Completion]
    total_tokens: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Proportion of successful completions."""
        total = len(self.completions) + len(self.errors)
        if total == 0:
            return 0.0
        return len(self.completions) / total


@dataclass
class ConversationTurnResult:
    """Result from a single turn in a conversation."""

    turn_index: int
    role: Literal["user", "assistant", "system"]
    content: str
    usage: dict[str, int] | None = None
    cached: bool = False


@dataclass
class ConversationResult:
    """Results from a multi-turn conversation."""

    turns: list[ConversationTurnResult]
    total_tokens: int = 0
    cache_hits: int = 0
    model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def last_response(self) -> str:
        """Get the last assistant response."""
        for turn in reversed(self.turns):
            if turn.role == "assistant":
                return turn.content
        return ""

    @property
    def all_responses(self) -> list[str]:
        """Get all assistant responses in order."""
        return [t.content for t in self.turns if t.role == "assistant"]

    def to_turns(self) -> list[Turn]:
        """Convert back to Turn objects."""
        return [Turn(role=t.role, content=t.content) for t in self.turns]


class ModelRunner:
    """
    Runner for executing model completions via OpenRouter.

    Features:
    - Async-first with sync wrapper
    - Response caching
    - Automatic retries with exponential backoff
    - Batch processing with rate limiting
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        config: ModelConfig | None = None,
        api_key: str | None = None,
        cache: CacheStore | None = None,
        max_concurrent: int = 10,
    ) -> None:
        """
        Initialize the model runner.

        Args:
            config: Model configuration. Can be set later via set_config().
            api_key: OpenRouter API key. Defaults to OPENROUTER_API_KEY env var.
            cache: Cache store for responses. Uses default if not provided.
            max_concurrent: Maximum concurrent API requests.
        """
        self.config = config
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self._cache = cache
        self._max_concurrent = max_concurrent
        self._client: AsyncOpenAI | None = None
        self._semaphore: asyncio.Semaphore | None = None

    def set_config(self, config: ModelConfig) -> None:
        """Set or update the model configuration."""
        self.config = config

    @property
    def cache(self) -> CacheStore:
        """Get the cache store, creating default if needed."""
        if self._cache is None:
            self._cache = get_cache()
        return self._cache

    @property
    def client(self) -> AsyncOpenAI:
        """Get the OpenAI client, creating if needed."""
        if self._client is None:
            if not self._api_key:
                raise ValueError(
                    "No API key provided. Set OPENROUTER_API_KEY environment variable "
                    "or pass api_key to ModelRunner."
                )
            self._client = AsyncOpenAI(
                base_url=self.OPENROUTER_BASE_URL,
                api_key=self._api_key,
            )
        return self._client

    @property
    def semaphore(self) -> asyncio.Semaphore:
        """Get the concurrency semaphore."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
        return self._semaphore

    def _get_cache_params(self, **overrides: Any) -> dict[str, Any]:
        """Get parameters to use for cache key generation."""
        if self.config is None:
            raise ValueError("Model config not set")

        params = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.seed is not None:
            params["seed"] = self.config.seed
        if self.config.top_p is not None:
            params["top_p"] = self.config.top_p

        params.update(overrides)
        return params

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _call_api(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Make an API call with retries.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            **kwargs: Additional API parameters.

        Returns:
            Raw API response as dict.
        """
        if self.config is None:
            raise ValueError("Model config not set")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        params = {
            "model": self.config.name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        if self.config.seed is not None:
            params["seed"] = self.config.seed
        if self.config.top_p is not None:
            params["top_p"] = self.config.top_p

        params.update(kwargs)

        response = await self.client.chat.completions.create(**params)
        return response.model_dump()

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> Completion:
        """
        Get a single completion.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            use_cache: Whether to use cached responses.
            **kwargs: Additional API parameters.

        Returns:
            Completion result.
        """
        if self.config is None:
            raise ValueError("Model config not set")

        cache_params = self._get_cache_params(**kwargs)
        # Include system_prompt in cache key to avoid collisions
        if system_prompt:
            cache_params["system_prompt"] = system_prompt

        if use_cache:
            cached = await self.cache.get(self.config.name, prompt, cache_params)
            if cached:
                choice = cached.response["choices"][0]
                return Completion(
                    content=choice["message"]["content"],
                    model=cached.response.get("model", self.config.name),
                    prompt=prompt,
                    finish_reason=choice.get("finish_reason"),
                    usage=cached.response.get("usage"),
                    cached=True,
                )

        async with self.semaphore:
            response = await self._call_api(prompt, system_prompt, **kwargs)

        if use_cache:
            await self.cache.set(self.config.name, prompt, cache_params, response)

        choice = response["choices"][0]
        return Completion(
            content=choice["message"]["content"],
            model=response.get("model", self.config.name),
            prompt=prompt,
            finish_reason=choice.get("finish_reason"),
            usage=response.get("usage"),
            cached=False,
        )

    async def batch_complete(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> BatchResult:
        """
        Execute completions for multiple prompts in parallel.

        Args:
            prompts: List of prompts to complete.
            system_prompt: Optional system prompt for all completions.
            use_cache: Whether to use cached responses.
            **kwargs: Additional API parameters.

        Returns:
            BatchResult with all completions.
        """
        tasks = [
            self.complete(prompt, system_prompt, use_cache, **kwargs)
            for prompt in prompts
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        completions: list[Completion] = []
        errors: list[dict[str, Any]] = []
        cache_hits = 0
        cache_misses = 0
        total_tokens = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append(
                    {
                        "index": i,
                        "prompt": prompts[i],
                        "error": str(result),
                        "type": type(result).__name__,
                    }
                )
            else:
                completions.append(result)
                total_tokens += result.tokens_used
                if result.cached:
                    cache_hits += 1
                else:
                    cache_misses += 1

        return BatchResult(
            completions=completions,
            total_tokens=total_tokens,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            errors=errors,
        )

    def complete_sync(
        self,
        prompt: str,
        system_prompt: str | None = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> Completion:
        """
        Synchronous wrapper for complete().

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            use_cache: Whether to use cached responses.
            **kwargs: Additional API parameters.

        Returns:
            Completion result.
        """
        return asyncio.run(self.complete(prompt, system_prompt, use_cache, **kwargs))

    def batch_complete_sync(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> BatchResult:
        """
        Synchronous wrapper for batch_complete().

        Args:
            prompts: List of prompts to complete.
            system_prompt: Optional system prompt for all completions.
            use_cache: Whether to use cached responses.
            **kwargs: Additional API parameters.

        Returns:
            BatchResult with all completions.
        """
        return asyncio.run(
            self.batch_complete(prompts, system_prompt, use_cache, **kwargs)
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _call_api_messages(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Make an API call with a full messages list.

        Args:
            messages: List of message dicts with role and content.
            **kwargs: Additional API parameters.

        Returns:
            Raw API response as dict.
        """
        if self.config is None:
            raise ValueError("Model config not set")

        params = {
            "model": self.config.name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        if self.config.seed is not None:
            params["seed"] = self.config.seed
        if self.config.top_p is not None:
            params["top_p"] = self.config.top_p

        params.update(kwargs)

        response = await self.client.chat.completions.create(**params)
        return response.model_dump()

    def _get_conversation_cache_key(
        self,
        messages: list[dict[str, str]],
        **params: Any,
    ) -> str:
        """Generate a cache key for a conversation."""
        key_data = {
            "messages": messages,
            "params": params,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _serialize_messages_for_cache(
        self,
        messages: list[dict[str, str]],
        **params: Any,
    ) -> str:
        """Serialize messages and params for cache lookup."""
        key_data = {
            "messages": messages,
            "params": params,
        }
        return json.dumps(key_data, sort_keys=True)

    async def complete_conversation(
        self,
        turns: list[Turn],
        system_prompt: str | None = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> ConversationResult:
        """
        Execute a multi-turn conversation.

        Runs through user turns, getting assistant responses for each.
        Maintains conversation history between turns.

        Args:
            turns: List of Turn objects (typically user turns to respond to).
            system_prompt: Optional system prompt for the conversation.
            use_cache: Whether to cache responses.
            **kwargs: Additional API parameters.

        Returns:
            ConversationResult with all turns and responses.
        """
        if self.config is None:
            raise ValueError("Model config not set")

        result_turns: list[ConversationTurnResult] = []
        messages: list[dict[str, str]] = []
        total_tokens = 0
        cache_hits = 0

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            result_turns.append(
                ConversationTurnResult(
                    turn_index=0,
                    role="system",
                    content=system_prompt,
                )
            )

        for i, turn in enumerate(turns):
            messages.append(turn.to_message_dict())
            result_turns.append(
                ConversationTurnResult(
                    turn_index=len(result_turns),
                    role=turn.role,
                    content=turn.content,
                )
            )

            if turn.role != "user":
                continue

            cache_params = self._get_cache_params(**kwargs)
            cache_prompt = self._serialize_messages_for_cache(messages, **cache_params)

            cached_response = None
            if use_cache:
                cached_response = await self.cache.get(
                    self.config.name,
                    cache_prompt,
                    cache_params,
                )

            if cached_response:
                choice = cached_response.response["choices"][0]
                assistant_content = choice["message"]["content"]
                usage = cached_response.response.get("usage")
                cache_hits += 1
            else:
                async with self.semaphore:
                    response = await self._call_api_messages(messages, **kwargs)

                choice = response["choices"][0]
                assistant_content = choice["message"]["content"]
                usage = response.get("usage")

                if use_cache:
                    await self.cache.set(
                        self.config.name,
                        cache_prompt,
                        cache_params,
                        response,
                    )

            messages.append({"role": "assistant", "content": assistant_content})
            result_turns.append(
                ConversationTurnResult(
                    turn_index=len(result_turns),
                    role="assistant",
                    content=assistant_content,
                    usage=usage,
                    cached=cached_response is not None,
                )
            )

            if usage:
                total_tokens += usage.get("total_tokens", 0)

        return ConversationResult(
            turns=result_turns,
            total_tokens=total_tokens,
            cache_hits=cache_hits,
            model=self.config.name,
        )

    def complete_conversation_sync(
        self,
        turns: list[Turn],
        system_prompt: str | None = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> ConversationResult:
        """
        Synchronous wrapper for complete_conversation().

        Args:
            turns: List of Turn objects.
            system_prompt: Optional system prompt.
            use_cache: Whether to cache responses.
            **kwargs: Additional API parameters.

        Returns:
            ConversationResult with all turns.
        """
        return asyncio.run(
            self.complete_conversation(turns, system_prompt, use_cache, **kwargs)
        )


def create_runner(
    config: ModelConfig | None = None,
    api_key: str | None = None,
) -> ModelRunner:
    """
    Create a model runner instance.

    Args:
        config: Model configuration.
        api_key: OpenRouter API key.

    Returns:
        Configured ModelRunner instance.
    """
    return ModelRunner(config=config, api_key=api_key)
