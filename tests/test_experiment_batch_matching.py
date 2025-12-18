"""Tests for matching batch completions to their sample indices."""

from evals.experiment import _match_completions_to_samples
from evals.runner import Completion
from evals.sweep import SweepPoint


def test_match_completions_skips_failed_duplicate_prompt() -> None:
    """Later duplicate prompt should keep its original sample index."""
    point = SweepPoint(variables={"v": 1}, prompt="same", index=0)
    batch = [(point, 0), (point, 1)]

    completion = Completion(
        content="ok",
        model="test-model",
        prompt="same",
        request_index=1,
    )
    completion.metadata["request_index"] = 1

    matched = _match_completions_to_samples(batch, [completion])

    assert len(matched) == 1
    matched_point, sample_idx, _ = matched[0]
    assert sample_idx == 1
    assert matched_point is batch[1][0]


def test_match_completions_falls_back_to_prompt_order() -> None:
    """If no request index is set, fall back to prompt FIFO ordering."""
    point = SweepPoint(variables={"v": 1}, prompt="same", index=0)
    batch = [(point, 0)]

    completion = Completion(
        content="ok",
        model="test-model",
        prompt="same",
    )

    matched = _match_completions_to_samples(batch, [completion])

    assert len(matched) == 1
    matched_point, sample_idx, _ = matched[0]
    assert sample_idx == 0
    assert matched_point is batch[0][0]
