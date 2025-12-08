"""
Shared primitives for evaluation tasks.

Provides foundational building blocks like Turn, Agent, and other
reusable types used across sequences, multi-agent, and probe modules.
"""

from evals.primitives.turn import ConversationHistory, Turn

__all__ = ["Turn", "ConversationHistory"]
