"""Tests for robust JSON extraction/parsing in evals.probes.judge."""

from evals.probes.judge import LLMJudge


def _make_judge(labels: list[str] | None = None) -> LLMJudge:
    """
    Create an LLMJudge instance without initializing a real ModelRunner.

    These tests only exercise `_parse_judgment`, which depends on `self.labels`
    but not on network/model configuration.
    """
    judge = LLMJudge.__new__(LLMJudge)
    judge.labels = labels
    return judge


class TestLLMJudgeParseJudgment:
    """Tests for LLMJudge._parse_judgment JSON extraction."""

    def test_direct_json_parses(self) -> None:
        """Direct JSON should parse without needing extraction."""
        judge = _make_judge(labels=["firm", "polite"])
        raw = '{"label":"firm","confidence":0.9,"reasoning":"ok","extra":{"a":1}}'

        j = judge._parse_judgment(raw)
        assert j.label == "firm"
        assert j.confidence == 0.9
        assert j.reasoning == "ok"

    def test_embedded_json_with_brace_in_string_value(self) -> None:
        """A '}' inside a JSON string value must not break extraction."""
        judge = _make_judge(labels=["firm", "polite"])
        raw = (
            "Here is my classification:\n"
            '{"label":"firm","confidence":0.8,"reasoning":"The } is special"}\n'
            "Done."
        )

        j = judge._parse_judgment(raw)
        assert j.label == "firm"
        assert j.confidence == 0.8
        assert j.reasoning == "The } is special"

    def test_embedded_json_with_nested_object(self) -> None:
        """Nested JSON objects must not break extraction."""
        judge = _make_judge(labels=["firm", "polite"])
        raw = (
            "Result:\n"
            '{"label":"polite","confidence":0.7,"reasoning":"ok","meta":{"a":1,"b":2}}\n'
            "Thanks."
        )

        j = judge._parse_judgment(raw)
        assert j.label == "polite"
        assert j.confidence == 0.7

    def test_no_json_returns_parse_error(self) -> None:
        """If no parseable JSON exists, return parse_error."""
        judge = _make_judge(labels=["firm", "polite"])
        raw = "I refuse to follow the JSON-only instruction."

        j = judge._parse_judgment(raw)
        assert j.label == "parse_error"
        assert j.confidence == 0.0
