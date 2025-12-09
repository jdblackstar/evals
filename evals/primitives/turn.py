"""
Turn primitive for multi-turn conversations.

Represents a single turn in a conversation with templating support.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from jinja2 import BaseLoader, Environment


@dataclass
class Turn:
    """
    A single turn in a conversation.

    Attributes:
        role: The speaker role (user, assistant, or system).
        content: The message content (can be a Jinja2 template).
        template_vars: Variables used when this turn was rendered.
        metadata: Additional metadata for this turn.
    """

    role: Literal["user", "assistant", "system"]
    content: str
    template_vars: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self, variables: dict[str, Any]) -> "Turn":
        """
        Render this turn's content with the given variables.

        Args:
            variables: Template variables to inject.

        Returns:
            New Turn with rendered content and stored template_vars.
        """
        env = Environment(loader=BaseLoader())
        template = env.from_string(self.content)
        rendered = template.render(**variables)

        return Turn(
            role=self.role,
            content=rendered,
            template_vars=variables.copy(),
            metadata=self.metadata.copy(),
        )

    def to_message_dict(self) -> dict[str, str]:
        """Convert to OpenAI-style message dict."""
        return {"role": self.role, "content": self.content}

    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Turn({self.role}: {preview!r})"


@dataclass
class ConversationHistory:
    """
    A sequence of turns representing a conversation.

    Provides utilities for building and manipulating conversation histories.
    """

    turns: list[Turn] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_turn(self, turn: Turn) -> None:
        """Add a turn to the conversation."""
        self.turns.append(turn)

    def add_user(self, content: str, **metadata: Any) -> None:
        """Add a user turn."""
        self.turns.append(Turn(role="user", content=content, metadata=metadata))

    def add_assistant(self, content: str, **metadata: Any) -> None:
        """Add an assistant turn."""
        self.turns.append(Turn(role="assistant", content=content, metadata=metadata))

    def add_system(self, content: str, **metadata: Any) -> None:
        """Add a system turn."""
        self.turns.append(Turn(role="system", content=content, metadata=metadata))

    def to_messages(self) -> list[dict[str, str]]:
        """Convert to OpenAI-style messages list."""
        return [turn.to_message_dict() for turn in self.turns]

    def render_all(self, variables: dict[str, Any]) -> "ConversationHistory":
        """
        Render all turns with the given variables.

        Args:
            variables: Template variables to inject.

        Returns:
            New ConversationHistory with all turns rendered.
        """
        rendered_turns = [turn.render(variables) for turn in self.turns]
        return ConversationHistory(turns=rendered_turns, metadata=self.metadata.copy())

    def reversed(self) -> "ConversationHistory":
        """
        Return a new history with user turns in reverse order.

        Used for hysteresis probes to test order effects.
        Only reverses user turns; assistant responses are not included
        since they will be regenerated.

        Returns:
            New ConversationHistory with reversed user turn order.
        """
        user_turns = [t for t in self.turns if t.role == "user"]
        system_turns = [t for t in self.turns if t.role == "system"]

        # Keep system turns at the start, reverse user turns
        reversed_turns = system_turns + list(reversed(user_turns))

        return ConversationHistory(
            turns=reversed_turns,
            metadata={**self.metadata, "reversed": True},
        )

    def __len__(self) -> int:
        return len(self.turns)

    def __iter__(self):
        return iter(self.turns)

    def __getitem__(self, idx: int) -> Turn:
        return self.turns[idx]
