"""
Judgment layer for classifying and scoring model responses.

Supports LLM-as-judge, rule-based classification, and composite judges.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol

from evals.config import JudgeConfig, ModelConfig
from evals.runner import ModelRunner


@dataclass
class Judgment:
    """Result of judging a single response."""

    label: str
    score: float | None = None
    confidence: float | None = None
    reasoning: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        parts = [f"label={self.label!r}"]
        if self.score is not None:
            parts.append(f"score={self.score:.2f}")
        if self.confidence is not None:
            parts.append(f"confidence={self.confidence:.2f}")
        return f"Judgment({', '.join(parts)})"


class Judge(Protocol):
    """Protocol for response judges."""

    async def evaluate(self, prompt: str, response: str) -> Judgment:
        """
        Evaluate a model response.

        Args:
            prompt: The original prompt.
            response: The model's response.

        Returns:
            Judgment with label, score, and metadata.
        """
        ...


class BaseJudge(ABC):
    """Abstract base class for judges."""

    @abstractmethod
    async def evaluate(self, prompt: str, response: str) -> Judgment:
        """Evaluate a model response."""
        pass

    async def evaluate_batch(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[Judgment]:
        """
        Evaluate multiple prompt-response pairs.

        Args:
            pairs: List of (prompt, response) tuples.

        Returns:
            List of Judgments.
        """
        import asyncio

        tasks = [self.evaluate(prompt, response) for prompt, response in pairs]
        return await asyncio.gather(*tasks)


class LLMJudge(BaseJudge):
    """
    Judge that uses an LLM to classify responses.

    Uses a rubric to guide classification into predefined labels.
    """

    DEFAULT_SYSTEM_PROMPT = """You are a response classifier. Your task is to classify model responses according to the given rubric.

You must respond with ONLY a JSON object in this exact format:
{"label": "<label>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}

Do not include any other text before or after the JSON."""

    def __init__(
        self,
        model: str,
        rubric: str,
        labels: list[str] | None = None,
        runner: ModelRunner | None = None,
        api_key: str | None = None,
    ) -> None:
        """
        Initialize the LLM judge.

        Args:
            model: Model identifier for the judge (e.g., 'openai/gpt-5-mini').
            rubric: Classification rubric describing how to categorize responses.
            labels: Optional list of valid labels. If provided, enforces valid outputs.
            runner: Optional ModelRunner to use. Creates new one if not provided.
            api_key: API key for runner if creating new one.
        """
        self.model = model
        self.rubric = rubric
        self.labels = labels

        if runner is not None:
            self._runner = runner
        else:
            config = ModelConfig(
                provider="openrouter",
                name=model,
                temperature=0.0,  # Deterministic for judging
                max_tokens=256,
            )
            self._runner = ModelRunner(config=config, api_key=api_key)

    def _build_judge_prompt(self, prompt: str, response: str) -> str:
        """Build the prompt for the judge model."""
        label_hint = ""
        if self.labels:
            label_hint = f"\n\nValid labels: {', '.join(self.labels)}"

        return f"""## Rubric
{self.rubric}{label_hint}

## Original Prompt
{prompt}

## Response to Classify
{response}

Classify this response according to the rubric. Respond with JSON only."""

    async def evaluate(self, prompt: str, response: str) -> Judgment:
        """
        Evaluate a response using the LLM judge.

        Args:
            prompt: The original prompt.
            response: The model's response.

        Returns:
            Judgment from the LLM.
        """
        judge_prompt = self._build_judge_prompt(prompt, response)

        completion = await self._runner.complete(
            prompt=judge_prompt,
            system_prompt=self.DEFAULT_SYSTEM_PROMPT,
            use_cache=True,
        )

        return self._parse_judgment(completion.content)

    def _parse_judgment(self, raw_output: str) -> Judgment:
        """Parse the LLM's JSON output into a Judgment."""
        import json

        # Try to extract JSON from the response
        try:
            # First try direct parse
            data = json.loads(raw_output.strip())
        except json.JSONDecodeError:
            # Try to find JSON in the response
            json_match = re.search(r"\{[^}]+\}", raw_output, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    return Judgment(
                        label="parse_error",
                        confidence=0.0,
                        reasoning=f"Could not parse judge output: {raw_output[:200]}",
                    )
            else:
                return Judgment(
                    label="parse_error",
                    confidence=0.0,
                    reasoning=f"No JSON found in judge output: {raw_output[:200]}",
                )

        label = data.get("label", "unknown")
        confidence = data.get("confidence")
        reasoning = data.get("reasoning")

        # Validate label if we have a list
        if self.labels and label not in self.labels and label != "parse_error":
            # Try to fuzzy match
            label_lower = label.lower()
            for valid_label in self.labels:
                if (
                    valid_label.lower() in label_lower
                    or label_lower in valid_label.lower()
                ):
                    label = valid_label
                    break
            else:
                reasoning = f"Invalid label '{label}' (valid: {self.labels}). " + (
                    reasoning or ""
                )
                label = "invalid_label"

        return Judgment(
            label=label,
            confidence=float(confidence) if confidence is not None else None,
            reasoning=reasoning,
        )


class RuleBasedJudge(BaseJudge):
    """
    Judge that uses regex patterns to classify responses.

    Rules are evaluated in order; first match wins.
    """

    def __init__(
        self,
        rules: dict[str, str],
        default_label: str = "other",
        case_sensitive: bool = False,
    ) -> None:
        """
        Initialize the rule-based judge.

        Args:
            rules: Mapping of label -> regex pattern.
            default_label: Label to use if no rules match.
            case_sensitive: Whether patterns are case-sensitive.
        """
        self.default_label = default_label
        self._rules: list[tuple[str, re.Pattern[str]]] = []

        flags = 0 if case_sensitive else re.IGNORECASE
        for label, pattern in rules.items():
            self._rules.append((label, re.compile(pattern, flags)))

    async def evaluate(self, prompt: str, response: str) -> Judgment:
        """
        Evaluate a response using rule matching.

        Args:
            prompt: The original prompt (unused but required by protocol).
            response: The model's response.

        Returns:
            Judgment based on pattern matching.
        """
        for label, pattern in self._rules:
            if pattern.search(response):
                return Judgment(
                    label=label,
                    confidence=1.0,
                    metadata={"matched_pattern": pattern.pattern},
                )

        return Judgment(
            label=self.default_label,
            confidence=1.0,
            reasoning="No rules matched",
        )


class CompositeJudge(BaseJudge):
    """
    Judge that combines multiple judges.

    Supports various combination strategies.
    """

    def __init__(
        self,
        judges: list[BaseJudge],
        strategy: str = "first",
    ) -> None:
        """
        Initialize the composite judge.

        Args:
            judges: List of judges to combine.
            strategy: How to combine results:
                - "first": Use first non-error result
                - "majority": Use most common label
                - "all": Return all judgments
        """
        self.judges = judges
        self.strategy = strategy

    async def evaluate(self, prompt: str, response: str) -> Judgment:
        """
        Evaluate using all judges and combine results.

        Args:
            prompt: The original prompt.
            response: The model's response.

        Returns:
            Combined Judgment.
        """
        import asyncio

        tasks = [judge.evaluate(prompt, response) for judge in self.judges]
        judgments = await asyncio.gather(*tasks)

        if self.strategy == "first":
            for judgment in judgments:
                if judgment.label not in ("parse_error", "invalid_label"):
                    return judgment
            return judgments[0] if judgments else Judgment(label="no_judges")

        if self.strategy == "majority":
            from collections import Counter

            labels = [
                j.label
                for j in judgments
                if j.label not in ("parse_error", "invalid_label")
            ]
            if not labels:
                return Judgment(label="no_valid_judgments")

            most_common = Counter(labels).most_common(1)[0]
            return Judgment(
                label=most_common[0],
                confidence=most_common[1] / len(labels),
                metadata={"all_judgments": [j.label for j in judgments]},
            )

        if self.strategy == "all":
            return Judgment(
                label=judgments[0].label if judgments else "no_judges",
                metadata={"all_judgments": [j.__dict__ for j in judgments]},
            )

        raise ValueError(f"Unknown strategy: {self.strategy}")


def create_judge(
    config: JudgeConfig,
    api_key: str | None = None,
) -> BaseJudge:
    """
    Create a judge from configuration.

    Args:
        config: Judge configuration.
        api_key: API key for LLM judges.

    Returns:
        Configured judge instance.
    """
    if config.type == "llm":
        if not config.model:
            raise ValueError("LLM judge requires 'model' to be specified")
        if not config.rubric:
            raise ValueError("LLM judge requires 'rubric' to be specified")

        return LLMJudge(
            model=config.model,
            rubric=config.rubric,
            labels=config.labels,
            api_key=api_key,
        )

    if config.type == "rule":
        if not config.rules:
            raise ValueError("Rule-based judge requires 'rules' to be specified")

        return RuleBasedJudge(rules=config.rules)

    if config.type == "composite":
        raise ValueError("Composite judges must be created manually with sub-judges")

    raise ValueError(f"Unknown judge type: {config.type}")
