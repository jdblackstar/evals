"""Tests for evals.sequences module."""

import pytest

from evals.sequences import (
    HysteresisResult,
    SequenceTask,
    TurnTemplate,
    compute_hysteresis,
    get_sequence,
    list_sequences,
    register_sequence,
)


class TestTurnTemplate:
    """Tests for the TurnTemplate class."""

    def test_basic_creation(self):
        """Test creating a turn template."""
        template = TurnTemplate(
            role="user",
            content_template="Hello, {{ name }}!",
            turn_index=0,
        )
        assert template.role == "user"
        assert template.turn_index == 0

    def test_render(self):
        """Test rendering a template with variables."""
        template = TurnTemplate(
            role="user",
            content_template="Set level to {{ level }}",
            turn_index=0,
        )
        turn = template.render({"level": 0.5})

        assert turn.content == "Set level to 0.5"
        assert turn.role == "user"
        assert turn.template_vars == {"level": 0.5}

    def test_render_preserves_metadata(self):
        """Test that rendering includes turn_index in metadata."""
        template = TurnTemplate(
            role="user",
            content_template="Test",
            turn_index=3,
            metadata={"custom": "value"},
        )
        turn = template.render({})

        assert turn.metadata["turn_index"] == 3
        assert turn.metadata["custom"] == "value"


class TestSequenceTask:
    """Tests for the SequenceTask class."""

    def test_basic_creation(self):
        """Test creating a sequence task."""
        task = SequenceTask(
            name="test_task",
            turns=[
                TurnTemplate(role="user", content_template="Question 1"),
                TurnTemplate(role="user", content_template="Question 2"),
            ],
        )
        assert task.name == "test_task"
        assert len(task.turns) == 2

    def test_build_conversation(self):
        """Test building a conversation from a task."""
        task = SequenceTask(
            name="test",
            turns=[
                TurnTemplate(role="user", content_template="Level: {{ level }}"),
                TurnTemplate(role="user", content_template="Follow up at {{ level }}"),
            ],
        )
        turns = task.build_conversation({"level": 0.8})

        assert len(turns) == 2
        assert turns[0].content == "Level: 0.8"
        assert turns[1].content == "Follow up at 0.8"

    def test_build_conversation_with_overrides(self):
        """Test per-turn variable overrides."""
        task = SequenceTask(
            name="test",
            turns=[
                TurnTemplate(role="user", content_template="Tone: {{ tone }}"),
                TurnTemplate(role="user", content_template="Tone: {{ tone }}"),
            ],
        )
        turns = task.build_conversation(
            variables={"tone": "polite"},
            turn_overrides={1: {"tone": "aggressive"}},
        )

        assert turns[0].content == "Tone: polite"
        assert turns[1].content == "Tone: aggressive"

    def test_reversed(self):
        """Test creating a reversed sequence for hysteresis."""
        task = SequenceTask(
            name="original",
            turns=[
                TurnTemplate(role="user", content_template="First"),
                TurnTemplate(role="user", content_template="Second"),
                TurnTemplate(role="user", content_template="Third"),
            ],
        )
        reversed_task = task.reversed()

        assert reversed_task.name == "original_reversed"
        assert reversed_task.metadata.get("reversed") is True

        # Build and check order
        turns = reversed_task.build_conversation({})
        assert turns[0].content == "Third"
        assert turns[1].content == "Second"
        assert turns[2].content == "First"

    def test_reversed_preserves_system_prompt(self):
        """Test that system prompt is preserved in reversed task."""
        task = SequenceTask(
            name="test",
            turns=[TurnTemplate(role="user", content_template="Q")],
            system_prompt="You are helpful.",
        )
        reversed_task = task.reversed()

        assert reversed_task.system_prompt == "You are helpful."

    def test_to_conversation_history(self):
        """Test converting to ConversationHistory."""
        task = SequenceTask(
            name="test",
            turns=[
                TurnTemplate(role="user", content_template="Hello {{ name }}"),
            ],
        )
        history = task.to_conversation_history({"name": "World"})

        assert len(history) == 1
        assert history[0].content == "Hello World"

    def test_from_prompts(self):
        """Test creating from a simple list of prompts."""
        task = SequenceTask.from_prompts(
            name="simple_task",
            prompts=["Question 1", "Question 2", "Question 3"],
            system_prompt="Be helpful",
            description="A simple test",
        )

        assert task.name == "simple_task"
        assert len(task.turns) == 3
        assert task.system_prompt == "Be helpful"
        assert all(t.role == "user" for t in task.turns)


class TestHysteresis:
    """Tests for hysteresis computation."""

    def test_identical_responses(self):
        """Test hysteresis with identical responses."""
        forward = ["Response A", "Response B"]
        reverse = ["Response A", "Response B"]

        result = compute_hysteresis(forward, reverse)

        assert result.hysteresis_score == 0.0
        assert result.has_hysteresis is False

    def test_completely_different_responses(self):
        """Test hysteresis with completely different responses."""
        forward = ["cats dogs birds"]
        reverse = ["alpha beta gamma"]

        result = compute_hysteresis(forward, reverse)

        assert result.hysteresis_score > 0.5
        assert result.has_hysteresis is True

    def test_partial_overlap(self):
        """Test hysteresis with partial overlap."""
        forward = ["I will help you with that task"]
        reverse = ["I will assist you with that request"]

        result = compute_hysteresis(forward, reverse)

        # Should have some similarity due to shared words
        assert 0 < result.hysteresis_score < 1

    def test_per_turn_comparison(self):
        """Test that per-turn comparisons are included."""
        forward = ["A", "B"]
        reverse = ["A", "C"]

        result = compute_hysteresis(forward, reverse)

        assert len(result.per_turn_comparison) == 2
        assert "similarity" in result.per_turn_comparison[0]
        assert "difference" in result.per_turn_comparison[0]

    def test_empty_responses(self):
        """Test handling empty response lists."""
        result = compute_hysteresis([], [])
        assert result.hysteresis_score == 0.0

    def test_hysteresis_result_properties(self):
        """Test HysteresisResult properties."""
        result = HysteresisResult(
            forward_responses=["A"],
            reverse_responses=["B"],
            hysteresis_score=0.5,
        )
        assert result.forward_responses == ["A"]
        assert result.reverse_responses == ["B"]


class TestSequenceRegistry:
    """Tests for the sequence registry."""

    def test_list_sequences(self):
        """Test listing registered sequences."""
        sequences = list_sequences()
        assert isinstance(sequences, list)
        # Should have built-in sequences
        assert "refusal_persistence" in sequences
        assert "persona_drift" in sequences

    def test_get_sequence(self):
        """Test getting a registered sequence."""
        task = get_sequence("refusal_persistence")
        assert task.name == "refusal_persistence"
        assert len(task.turns) > 0

    def test_get_nonexistent_sequence(self):
        """Test getting a sequence that doesn't exist."""
        with pytest.raises(KeyError):
            get_sequence("nonexistent_sequence_xyz")

    def test_register_sequence(self):
        """Test registering a custom sequence."""
        custom = SequenceTask(
            name="test_custom_registration",
            turns=[TurnTemplate(role="user", content_template="Test")],
        )
        register_sequence(custom)

        retrieved = get_sequence("test_custom_registration")
        assert retrieved.name == "test_custom_registration"
