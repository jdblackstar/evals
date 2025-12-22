"""Tests for summary statistics across multiple sweep dimensions."""

from evals.analysis import ExperimentResult, ExperimentResults, summarize_results


def test_summarize_results_includes_dimension_names() -> None:
    """
    Summary rows should indicate which dimension each value belongs to.
    """
    results = [
        ExperimentResult(
            index=0,
            variables={"assertiveness": 0.0, "tone": "polite"},
            prompt="p0",
            response="r0",
            judgment={"label": "polite"},
        ),
        ExperimentResult(
            index=1,
            variables={"assertiveness": 0.5, "tone": "firm"},
            prompt="p1",
            response="r1",
            judgment={"label": "firm"},
        ),
    ]
    experiment_results = ExperimentResults(
        name="multi_dim_test",
        results=results,
        config={"sweep": {"dimensions": [{"name": "assertiveness"}, {"name": "tone"}]}},
    )

    summary = summarize_results(experiment_results)

    dimensions = {row["dimension"] for row in summary}
    assert dimensions == {"assertiveness", "tone"}

    assert {
        row["dimension_value"] for row in summary if row["dimension"] == "assertiveness"
    } == {0.0, 0.5}
    assert {
        row["dimension_value"] for row in summary if row["dimension"] == "tone"
    } == {
        "polite",
        "firm",
    }

    for row in summary:
        assert row["count"] == 1
        assert row["proportion"] == 1.0
