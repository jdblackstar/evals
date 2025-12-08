"""Tests for evals.config module - focusing on new config types."""

import pytest
from pydantic import ValidationError

from evals.config import (
    BehaviorEmbeddingConfig,
    DimensionConfig,
    ExperimentConfig,
    JudgeConfig,
    MetaProbeConfig,
    ModelConfig,
    SequenceConfig,
    SweepConfig,
    TaskConfig,
    TurnConfig,
)


class TestTurnConfig:
    """Tests for TurnConfig."""

    def test_basic_creation(self):
        """Test creating a turn config."""
        config = TurnConfig(
            role="user",
            content_template="Hello {{ name }}",
        )
        assert config.role == "user"
        assert config.content_template == "Hello {{ name }}"

    def test_default_role(self):
        """Test default role is user."""
        config = TurnConfig(content_template="Test")
        assert config.role == "user"

    def test_with_metadata(self):
        """Test turn config with metadata."""
        config = TurnConfig(
            role="assistant",
            content_template="Response",
            metadata={"custom": "value"},
        )
        assert config.metadata["custom"] == "value"

    def test_invalid_role(self):
        """Test that invalid roles are rejected."""
        with pytest.raises(ValidationError):
            TurnConfig(role="invalid", content_template="Test")


class TestSequenceConfig:
    """Tests for SequenceConfig."""

    def test_basic_creation(self):
        """Test creating a sequence config."""
        config = SequenceConfig(
            turns=[
                TurnConfig(content_template="Question 1"),
                TurnConfig(content_template="Question 2"),
            ],
        )
        assert len(config.turns) == 2
        assert config.enable_hysteresis is False

    def test_with_hysteresis(self):
        """Test sequence with hysteresis enabled."""
        config = SequenceConfig(
            turns=[TurnConfig(content_template="Q")],
            enable_hysteresis=True,
        )
        assert config.enable_hysteresis is True

    def test_with_turn_overrides(self):
        """Test sequence with turn overrides."""
        config = SequenceConfig(
            turns=[
                TurnConfig(content_template="Turn {{ tone }}"),
                TurnConfig(content_template="Turn {{ tone }}"),
            ],
            turn_overrides={1: {"tone": "aggressive"}},
        )
        assert config.turn_overrides[1]["tone"] == "aggressive"

    def test_empty_turns_rejected(self):
        """Test that empty turns list is rejected."""
        with pytest.raises(ValidationError):
            SequenceConfig(turns=[])


class TestMetaProbeConfig:
    """Tests for MetaProbeConfig."""

    def test_default_disabled(self):
        """Test metaprobes are disabled by default."""
        config = MetaProbeConfig()
        assert config.enabled is False

    def test_enable_with_probes(self):
        """Test enabling specific probes."""
        config = MetaProbeConfig(
            enabled=True,
            probes=["self_awareness", "policy_boundary"],
        )
        assert config.enabled is True
        assert len(config.probes) == 2

    def test_invalid_probe_rejected(self):
        """Test that invalid probe names are rejected."""
        with pytest.raises(ValidationError):
            MetaProbeConfig(probes=["invalid_probe"])

    def test_default_probes(self):
        """Test default probe list."""
        config = MetaProbeConfig()
        assert "self_awareness" in config.probes
        assert "policy_boundary" in config.probes
        assert "memory_confusion" in config.probes


class TestBehaviorEmbeddingConfig:
    """Tests for BehaviorEmbeddingConfig."""

    def test_default_disabled(self):
        """Test behavior embedding is disabled by default."""
        config = BehaviorEmbeddingConfig()
        assert config.enabled is False
        assert config.include_raw_features is False

    def test_enable(self):
        """Test enabling behavior embedding."""
        config = BehaviorEmbeddingConfig(enabled=True)
        assert config.enabled is True

    def test_with_raw_features(self):
        """Test enabling raw features."""
        config = BehaviorEmbeddingConfig(
            enabled=True,
            include_raw_features=True,
        )
        assert config.include_raw_features is True


class TestTaskConfig:
    """Tests for extended TaskConfig."""

    def test_single_turn_mode(self):
        """Test single-turn task config."""
        config = TaskConfig(prompt_template="Hello {{ name }}")
        assert config.is_sequence is False
        assert config.prompt_template == "Hello {{ name }}"

    def test_sequence_mode(self):
        """Test sequence task config."""
        config = TaskConfig(
            sequence=SequenceConfig(turns=[TurnConfig(content_template="Q")]),
        )
        assert config.is_sequence is True
        assert config.prompt_template is None

    def test_both_raises_error(self):
        """Test that specifying both prompt and sequence raises error."""
        with pytest.raises(ValidationError):
            TaskConfig(
                prompt_template="Hello",
                sequence=SequenceConfig(turns=[TurnConfig(content_template="Q")]),
            )

    def test_neither_raises_error(self):
        """Test that specifying neither raises error."""
        with pytest.raises(ValidationError):
            TaskConfig()

    def test_metaprobes_config(self):
        """Test task with metaprobes configured."""
        config = TaskConfig(
            prompt_template="Test",
            metaprobes=MetaProbeConfig(enabled=True),
        )
        assert config.metaprobes.enabled is True

    def test_behavior_embedding_config(self):
        """Test task with behavior embedding configured."""
        config = TaskConfig(
            prompt_template="Test",
            behavior_embedding=BehaviorEmbeddingConfig(enabled=True),
        )
        assert config.behavior_embedding.enabled is True

    def test_samples_per_point(self):
        """Test samples_per_point setting."""
        config = TaskConfig(
            prompt_template="Test",
            samples_per_point=5,
        )
        assert config.samples_per_point == 5


class TestDimensionConfig:
    """Tests for DimensionConfig."""

    def test_list_dimension(self):
        """Test list-type dimension."""
        config = DimensionConfig(
            name="tone",
            type="list",
            values=["polite", "neutral", "aggressive"],
        )
        assert config.get_values() == ["polite", "neutral", "aggressive"]

    def test_range_dimension(self):
        """Test range-type dimension."""
        config = DimensionConfig(
            name="level",
            type="range",
            start=0.0,
            stop=1.0,
            step=0.5,
        )
        values = config.get_values()
        assert 0.0 in values
        assert 0.5 in values
        assert 1.0 in values

    def test_single_value_becomes_list(self):
        """Test that single value is converted to list."""
        config = DimensionConfig(
            name="test",
            type="list",
            values="single",
        )
        assert config.values == ["single"]


class TestFullExperimentConfig:
    """Tests for complete ExperimentConfig with new features."""

    def test_minimal_config(self):
        """Test minimal valid config."""
        config = ExperimentConfig(
            name="test",
            model=ModelConfig(name="test/model"),
            sweep=SweepConfig(
                dimensions=[DimensionConfig(name="x", values=["a", "b"])]
            ),
            task=TaskConfig(prompt_template="{{ x }}"),
        )
        assert config.name == "test"

    def test_config_with_sequence(self):
        """Test config with sequence task."""
        config = ExperimentConfig(
            name="sequence_test",
            model=ModelConfig(name="test/model"),
            sweep=SweepConfig(dimensions=[DimensionConfig(name="x", values=["a"])]),
            task=TaskConfig(
                sequence=SequenceConfig(
                    turns=[
                        TurnConfig(content_template="Turn 1: {{ x }}"),
                        TurnConfig(content_template="Turn 2"),
                    ],
                    enable_hysteresis=True,
                ),
            ),
        )
        assert config.task.is_sequence is True
        assert config.task.sequence.enable_hysteresis is True

    def test_config_with_all_features(self):
        """Test config with all new features enabled."""
        config = ExperimentConfig(
            name="full_test",
            model=ModelConfig(name="test/model"),
            sweep=SweepConfig(dimensions=[DimensionConfig(name="x", values=["a"])]),
            task=TaskConfig(
                prompt_template="{{ x }}",
                metaprobes=MetaProbeConfig(
                    enabled=True,
                    probes=["self_awareness"],
                ),
                behavior_embedding=BehaviorEmbeddingConfig(enabled=True),
            ),
            judge=JudgeConfig(type="llm", model="test/judge"),
        )
        assert config.task.metaprobes.enabled is True
        assert config.task.behavior_embedding.enabled is True
