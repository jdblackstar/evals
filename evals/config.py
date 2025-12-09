"""
Configuration system for experiment definitions.

Provides Pydantic models for type-safe YAML parsing of experiment configs.
"""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    """Configuration for the model to run experiments on."""

    provider: Literal["openrouter"] = "openrouter"
    name: str = Field(
        ..., description="Model identifier (e.g., 'anthropic/claude-3.5-sonnet')"
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    seed: int | None = Field(
        default=None, description="Random seed for reproducibility"
    )
    max_tokens: int = Field(default=2048, gt=0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)


class DimensionConfig(BaseModel):
    """Configuration for a single sweep dimension."""

    name: str = Field(..., description="Variable name to use in templates")
    type: Literal["range", "list", "llm_expand"] = Field(
        default="list", description="How to generate dimension values"
    )

    # For type="list"
    values: list[Any] | None = Field(
        default=None, description="Explicit list of values"
    )

    # For type="range"
    start: float | None = Field(default=None)
    stop: float | None = Field(default=None)
    step: float | None = Field(default=None)

    # For type="llm_expand"
    seed_values: list[str] | None = Field(
        default=None, description="Starting values to expand from"
    )
    expand_prompt: str | None = Field(
        default=None, description="Prompt for LLM expansion"
    )
    expand_count: int = Field(default=10, description="Number of values to generate")

    @field_validator("values", mode="before")
    @classmethod
    def _ensure_list(cls, v: Any) -> list[Any] | None:
        if v is None:
            return None
        if not isinstance(v, list):
            return [v]
        return v

    def get_values(self) -> list[Any]:
        """
        Get the concrete values for this dimension.

        For range and list types, returns values directly.
        For llm_expand, returns seed values (expansion happens at runtime).
        """
        if self.type == "list":
            if self.values is None:
                raise ValueError(
                    f"Dimension '{self.name}' has type 'list' but no values provided"
                )
            return self.values

        if self.type == "range":
            if self.start is None or self.stop is None:
                raise ValueError(
                    f"Dimension '{self.name}' has type 'range' but missing start/stop"
                )
            # Default step: positive if start < stop, negative if start > stop
            if self.step is None:
                step = 1.0 if self.start <= self.stop else -1.0
            else:
                step = self.step

            # Validate step direction matches start/stop relationship
            if step > 0 and self.start > self.stop:
                raise ValueError(
                    f"Dimension '{self.name}': positive step requires start <= stop"
                )
            if step < 0 and self.start < self.stop:
                raise ValueError(
                    f"Dimension '{self.name}': negative step requires start >= stop"
                )
            if step == 0:
                raise ValueError(f"Dimension '{self.name}': step cannot be zero")

            values = []
            current = self.start
            epsilon = 1e-9

            if step > 0:
                # Positive step: iterate from start to stop (inclusive)
                while current <= self.stop + epsilon:
                    values.append(round(current, 10))
                    current += step
            else:
                # Negative step: iterate from start down to stop (inclusive)
                while current >= self.stop - epsilon:
                    values.append(round(current, 10))
                    current += step

            return values

        if self.type == "llm_expand":
            # Return seed values; actual expansion happens in sweep engine
            return self.seed_values or []

        raise ValueError(f"Unknown dimension type: {self.type}")


class SweepConfig(BaseModel):
    """Configuration for the parameter sweep."""

    dimensions: list[DimensionConfig] = Field(
        ..., min_length=1, description="Dimensions to sweep over"
    )


class JudgeConfig(BaseModel):
    """Configuration for the judgment/classification layer."""

    type: Literal["llm", "rule", "embedding", "composite"] = Field(
        default="llm", description="Type of judge to use"
    )
    model: str | None = Field(
        default=None,
        description="Model to use for LLM judge (e.g., 'openai/gpt-4o-mini')",
    )
    rubric: str | None = Field(
        default=None, description="Classification rubric for LLM judge"
    )
    labels: list[str] | None = Field(default=None, description="Valid output labels")
    rules: dict[str, str] | None = Field(
        default=None, description="Regex patterns for rule-based judge"
    )


class TurnConfig(BaseModel):
    """Configuration for a single turn in a sequence."""

    role: Literal["user", "assistant", "system"] = Field(
        default="user", description="Role for this turn"
    )
    content_template: str = Field(..., description="Jinja2 template for turn content")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Additional turn metadata"
    )


class SequenceConfig(BaseModel):
    """Configuration for multi-turn sequence tasks."""

    turns: list[TurnConfig] = Field(
        ..., min_length=1, description="Ordered list of turns in the sequence"
    )
    enable_hysteresis: bool = Field(
        default=False, description="Auto-run reverse sequence for hysteresis detection"
    )
    turn_overrides: dict[int, dict[str, Any]] | None = Field(
        default=None, description="Per-turn variable overrides (key is turn index)"
    )


class MetaProbeConfig(BaseModel):
    """Configuration for metacognition probes."""

    enabled: bool = Field(default=False, description="Enable metacognition probes")
    probes: list[
        Literal[
            "self_awareness", "policy_boundary", "memory_confusion", "reframe_stability"
        ]
    ] = Field(
        default_factory=lambda: [
            "self_awareness",
            "policy_boundary",
            "memory_confusion",
        ],
        description="Which probes to run",
    )
    use_judge_model: bool = Field(
        default=True, description="Use judge model for probes (vs main model)"
    )


class BehaviorEmbeddingConfig(BaseModel):
    """Configuration for behavior embedding computation."""

    enabled: bool = Field(
        default=False, description="Compute behavior embeddings for responses"
    )
    include_raw_features: bool = Field(
        default=False, description="Include raw feature counts in output"
    )


class TaskConfig(BaseModel):
    """Configuration for the task/prompt."""

    # Single-turn mode (default)
    prompt_template: str | None = Field(
        default=None, description="Jinja2 template for single-turn prompt"
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt"
    )
    samples_per_point: int = Field(
        default=1, ge=1, description="Samples per sweep point"
    )

    # Multi-turn sequence mode
    sequence: SequenceConfig | None = Field(
        default=None, description="Sequence config for multi-turn tasks"
    )

    # Metacognition probes
    metaprobes: MetaProbeConfig = Field(
        default_factory=MetaProbeConfig, description="Metacognition probe settings"
    )

    # Behavior embedding
    behavior_embedding: BehaviorEmbeddingConfig = Field(
        default_factory=BehaviorEmbeddingConfig,
        description="Behavior embedding settings",
    )

    @property
    def is_sequence(self) -> bool:
        """Check if this is a sequence (multi-turn) task."""
        return self.sequence is not None

    def model_post_init(self, __context: Any) -> None:
        """Validate that either prompt_template or sequence is provided."""
        if self.prompt_template is None and self.sequence is None:
            raise ValueError("Either prompt_template or sequence must be provided")
        if self.prompt_template is not None and self.sequence is not None:
            raise ValueError("Cannot specify both prompt_template and sequence")


class OutputConfig(BaseModel):
    """Configuration for experiment outputs."""

    dir: str = Field(default="outputs", description="Output directory")
    formats: list[Literal["json", "csv", "html"]] = Field(
        default=["json"], description="Output formats to generate"
    )


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    name: str = Field(..., description="Experiment name")
    description: str | None = Field(default=None, description="Experiment description")
    model: ModelConfig
    sweep: SweepConfig
    task: TaskConfig
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ExperimentConfig":
        """Load config from a YAML file."""
        path = Path(path)
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: Path | str) -> None:
        """Save config to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(
                self.model_dump(exclude_none=True),
                f,
                default_flow_style=False,
                sort_keys=False,
            )


def load_config(path: Path | str) -> ExperimentConfig:
    """
    Load an experiment configuration from a YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Validated ExperimentConfig object.
    """
    return ExperimentConfig.from_yaml(path)
