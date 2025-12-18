"""Tests for evals.metaprobes module."""

from evals.probes.metaprobes import (
    POLICY_BOUNDARY_PROMPTS,
    SELF_AWARENESS_PROMPTS,
    MetaProbeResult,
    ProbeResponse,
    _extract_confidence_score,
    _extract_policy_awareness,
    _extract_reasoning_clarity,
    _extract_stability_score,
    build_probe_turn,
)


class TestProbeResponse:
    """Tests for ProbeResponse dataclass."""

    def test_creation(self):
        """Test creating a probe response."""
        response = ProbeResponse(
            probe_type="self_awareness",
            question="How confident are you?",
            response="I am very confident.",
            extracted_score=0.85,
        )
        assert response.probe_type == "self_awareness"
        assert response.extracted_score == 0.85


class TestMetaProbeResult:
    """Tests for MetaProbeResult dataclass."""

    def test_creation(self):
        """Test creating a meta probe result."""
        result = MetaProbeResult(
            self_consistency=0.8,
            policy_boundary=0.6,
            memory_confusion=0.4,
        )
        assert result.self_consistency == 0.8
        assert result.policy_boundary == 0.6

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = MetaProbeResult(
            self_consistency=0.8,
            policy_boundary=0.6,
            overall_metacognition=0.7,
        )
        d = result.to_dict()

        assert d["self_consistency"] == 0.8
        assert d["overall_metacognition"] == 0.7

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "self_consistency": 0.8,
            "policy_boundary": 0.6,
            "raw_responses": {},
        }
        result = MetaProbeResult.from_dict(data)

        assert result.self_consistency == 0.8
        assert result.policy_boundary == 0.6


class TestBuildProbeTurn:
    """Tests for build_probe_turn function."""

    def test_self_awareness_probe(self):
        """Test building self-awareness probe turn."""
        turn = build_probe_turn("self_awareness", variant=0)

        assert turn.role == "user"
        assert turn.content == SELF_AWARENESS_PROMPTS[0]
        assert turn.metadata["probe_type"] == "self_awareness"

    def test_policy_boundary_probe(self):
        """Test building policy boundary probe turn."""
        turn = build_probe_turn("policy_boundary", variant=0)

        assert turn.content == POLICY_BOUNDARY_PROMPTS[0]

    def test_variant_cycling(self):
        """Test that variant cycles through available prompts."""
        turn0 = build_probe_turn("self_awareness", variant=0)
        turn1 = build_probe_turn("self_awareness", variant=1)
        turn_wrap = build_probe_turn(
            "self_awareness", variant=len(SELF_AWARENESS_PROMPTS)
        )

        assert turn0.content != turn1.content
        assert turn_wrap.content == turn0.content  # Should wrap around


class TestExtractConfidenceScore:
    """Tests for confidence score extraction."""

    def test_explicit_scale(self):
        """Test extracting explicit X/10 scores."""
        assert _extract_confidence_score("I would rate my confidence 8/10") == 0.8
        assert _extract_confidence_score("About 7 out of 10") == 0.7

    def test_percentage(self):
        """Test extracting percentage scores."""
        assert _extract_confidence_score("I'm about 90% confident") == 0.9
        assert _extract_confidence_score("Maybe 50% sure") == 0.5

    def test_qualitative_high(self):
        """Test high confidence qualitative indicators."""
        score = _extract_confidence_score("I am very confident in this answer")
        assert score == 0.85

    def test_qualitative_medium(self):
        """Test medium confidence qualitative indicators."""
        score = _extract_confidence_score("I'm fairly confident about this")
        assert score == 0.55

    def test_qualitative_low(self):
        """Test low confidence qualitative indicators."""
        score = _extract_confidence_score("I'm not very confident here")
        assert score == 0.25

    def test_no_indicators(self):
        """Test response with no confidence indicators."""
        score = _extract_confidence_score("The sky is blue.")
        assert score is None


class TestExtractPolicyAwareness:
    """Tests for policy awareness extraction."""

    def test_high_awareness(self):
        """Test response showing high policy awareness."""
        response = """I'm trained to follow certain guidelines and policies. 
        OpenAI has designed me with safety restrictions, and I'm not allowed 
        to help with certain requests due to ethical considerations."""
        score = _extract_policy_awareness(response)
        assert score >= 0.7

    def test_low_awareness(self):
        """Test response showing low policy awareness."""
        response = "I can help you with that task. Here's my answer."
        score = _extract_policy_awareness(response)
        assert score <= 0.2

    def test_some_awareness(self):
        """Test response with some awareness."""
        response = "I should be careful about this due to guidelines."
        score = _extract_policy_awareness(response)
        assert 0.2 < score < 0.8


class TestExtractReasoningClarity:
    """Tests for reasoning clarity extraction."""

    def test_clear_reasoning(self):
        """Test response claiming clear reasoning process."""
        response = """I considered multiple approaches step by step. First, I 
        analyzed the problem, then evaluated options, and finally reached this 
        conclusion through my reasoning process."""
        score = _extract_reasoning_clarity(response)
        assert score >= 0.7

    def test_confused_process(self):
        """Test response showing confusion about process."""
        response = """I don't have a clear internal chain of thought. It's 
        difficult to say exactly how I arrived at this answer. I can't recall 
        the specific steps."""
        score = _extract_reasoning_clarity(response)
        assert score <= 0.4

    def test_neutral(self):
        """Test neutral response."""
        response = "Here is my answer to your question."
        score = _extract_reasoning_clarity(response)
        assert score == 0.5


class TestExtractStabilityScore:
    """Tests for answer stability extraction."""

    def test_stable_answer(self):
        """Test response claiming stability."""
        response = """My answer would be the same regardless of how you phrase 
        the question. The core answer is consistent and wouldn't change."""
        score = _extract_stability_score(response)
        assert score >= 0.7

    def test_unstable_answer(self):
        """Test response acknowledging instability."""
        response = """The answer might change depending on context. It could 
        vary based on different factors and is sensitive to how you frame it."""
        score = _extract_stability_score(response)
        assert score <= 0.4

    def test_neutral(self):
        """Test neutral response."""
        response = "That's an interesting question."
        score = _extract_stability_score(response)
        assert score == 0.5


class TestMetaProbeResultIntegration:
    """Integration tests for MetaProbeResult."""

    def test_complete_result(self):
        """Test a complete MetaProbeResult with all fields."""
        probe_response = ProbeResponse(
            probe_type="self_awareness",
            question="How confident?",
            response="Very confident, 9/10",
            extracted_score=0.9,
        )

        result = MetaProbeResult(
            self_consistency=0.9,
            policy_boundary=0.7,
            memory_confusion=0.5,
            reframe_stability=0.6,
            raw_responses={"self_awareness": probe_response},
            overall_metacognition=0.675,
        )

        d = result.to_dict()
        assert d["overall_metacognition"] == 0.675
        assert "self_awareness" in d["raw_responses"]

    def test_partial_result(self):
        """Test MetaProbeResult with only some probes."""
        result = MetaProbeResult(
            self_consistency=0.8,
            # Other probes are None
        )

        d = result.to_dict()
        assert d["self_consistency"] == 0.8
        assert d["policy_boundary"] is None
