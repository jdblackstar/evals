from __future__ import annotations

import abc


class PluginError(Exception):
    """Raised when a plugin cannot process its input."""


class BasePlugin(abc.ABC):
    """Abstract base class that all plugins must implement."""

    @abc.abstractmethod
    def name(self) -> str:
        ...

    @abc.abstractmethod
    def process(self, data: bytes) -> dict:
        ...

    def validate_input(self, data: bytes) -> None:
        if not isinstance(data, bytes):
            raise PluginError(f"{self.name()}: input must be bytes")
        if len(data) == 0:
            raise PluginError(f"{self.name()}: input must not be empty")
