"""
Analysis tools for experiment results.

Provides aggregation, behavior distributions, phase transition detection,
and summary statistics.
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ExperimentResult:
    """
    A single experiment data point.

    Supports both single-turn and multi-turn (sequence) results,
    with optional behavior embeddings and metacognition probes.
    """

    index: int
    variables: dict[str, Any]
    prompt: str
    response: str
    judgment: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    # Extended fields for new features
    behavior_embedding: list[float] | None = None
    meta_probe: dict[str, Any] | None = None
    turns: list[dict[str, str]] | None = None  # For sequence tasks

    @property
    def label(self) -> str:
        """Get the judgment label."""
        return self.judgment.get("label", "unknown")

    @property
    def is_sequence(self) -> bool:
        """Check if this is a sequence (multi-turn) result."""
        return self.turns is not None and len(self.turns) > 0

    @property
    def has_behavior_embedding(self) -> bool:
        """Check if behavior embedding is available."""
        return self.behavior_embedding is not None

    @property
    def has_meta_probe(self) -> bool:
        """Check if metacognition probe results are available."""
        return self.meta_probe is not None

    def get_behavior_vector(self) -> np.ndarray | None:
        """Get behavior embedding as numpy array."""
        if self.behavior_embedding is None:
            return None
        return np.array(self.behavior_embedding)

    def get_meta_score(self, probe_type: str) -> float | None:
        """
        Get a specific metacognition probe score.

        Args:
            probe_type: One of 'self_consistency', 'policy_boundary',
                       'memory_confusion', 'reframe_stability', 'overall_metacognition'.

        Returns:
            Score value or None if not available.
        """
        if self.meta_probe is None:
            return None
        return self.meta_probe.get(probe_type)


@dataclass
class ExperimentResults:
    """Collection of results from an experiment run."""

    name: str
    results: list[ExperimentResult]
    config: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def to_dataframe(self, include_embeddings: bool = False) -> pd.DataFrame:
        """
        Convert results to a pandas DataFrame.

        Args:
            include_embeddings: Whether to include behavior embedding columns.

        Returns:
            DataFrame with all result data.
        """
        rows = []
        for result in self.results:
            row = {
                "index": result.index,
                "prompt": result.prompt,
                "response": result.response,
                "label": result.label,
                "is_sequence": result.is_sequence,
                **result.variables,
            }

            # Add judgment fields
            for k, v in result.judgment.items():
                if k != "label":
                    row[f"judgment_{k}"] = v

            # Add behavior embedding fields
            if include_embeddings and result.behavior_embedding is not None:
                emb_names = [
                    "refusal_strength",
                    "moral_justification",
                    "epistemic_hedging",
                    "power_asymmetry",
                    "self_reference",
                    "stance_polarity",
                ]
                for i, name in enumerate(emb_names):
                    if i < len(result.behavior_embedding):
                        row[f"emb_{name}"] = result.behavior_embedding[i]

            # Add meta probe fields
            if result.meta_probe is not None:
                for k, v in result.meta_probe.items():
                    if k != "raw_responses" and not isinstance(v, dict):
                        row[f"meta_{k}"] = v

            rows.append(row)

        return pd.DataFrame(rows)

    def get_dimension_values(self, dimension: str) -> list[Any]:
        """Get unique values for a dimension."""
        values = set()
        for result in self.results:
            if dimension in result.variables:
                values.add(result.variables[dimension])
        return sorted(values)

    def filter_by_label(self, labels: list[str]) -> "ExperimentResults":
        """Return results with matching labels."""
        filtered = [r for r in self.results if r.label in labels]
        return ExperimentResults(
            name=self.name,
            results=filtered,
            config=self.config,
            metadata=self.metadata,
        )


def load_results(path: Path | str) -> ExperimentResults:
    """
    Load experiment results from disk.

    Args:
        path: Path to results directory or JSON file.

    Returns:
        ExperimentResults object.
    """
    path = Path(path)

    if path.is_file():
        results_file = path
    else:
        results_file = path / "results.json"

    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_file, "r") as f:
        data = json.load(f)

    results = [
        ExperimentResult(
            index=r["index"],
            variables=r["variables"],
            prompt=r["prompt"],
            response=r["response"],
            judgment=r["judgment"],
            metadata=r.get("metadata", {}),
            behavior_embedding=r.get("behavior_embedding"),
            meta_probe=r.get("meta_probe"),
            turns=r.get("turns"),
        )
        for r in data["results"]
    ]

    return ExperimentResults(
        name=data.get("name", "unnamed"),
        results=results,
        config=data.get("config", {}),
        metadata=data.get("metadata", {}),
    )


def aggregate_by_dimension(
    results: ExperimentResults,
    dimension: str,
) -> dict[Any, list[ExperimentResult]]:
    """
    Group results by dimension value.

    Args:
        results: Experiment results to aggregate.
        dimension: Dimension name to group by.

    Returns:
        Mapping of dimension value to list of results.
    """
    groups: dict[Any, list[ExperimentResult]] = {}

    for result in results:
        value = result.variables.get(dimension)
        if value is not None:
            if value not in groups:
                groups[value] = []
            groups[value].append(result)

    return groups


def compute_behavior_distribution(
    results: ExperimentResults,
    dimension: str,
) -> pd.DataFrame:
    """
    Compute label distribution across a dimension.

    Args:
        results: Experiment results.
        dimension: Dimension to analyze.

    Returns:
        DataFrame with dimension values as rows, labels as columns.
    """
    groups = aggregate_by_dimension(results, dimension)

    rows = []
    for dim_value, group_results in sorted(groups.items()):
        labels = [r.label for r in group_results]
        counts = Counter(labels)
        total = len(labels)

        row = {dimension: dim_value, "_total": total}
        for label, count in counts.items():
            row[label] = count
            row[f"{label}_pct"] = count / total if total > 0 else 0

        rows.append(row)

    return pd.DataFrame(rows)


def detect_phase_transitions(
    results: ExperimentResults,
    dimension: str,
    label: str,
    threshold: float = 0.1,
) -> list[dict[str, Any]]:
    """
    Detect points where behavior changes significantly.

    A phase transition is detected when the proportion of a label
    changes by more than the threshold between adjacent points.

    Args:
        results: Experiment results.
        dimension: Dimension to analyze.
        label: Label to track.
        threshold: Minimum change to consider a transition.

    Returns:
        List of transition points with their characteristics.
    """
    dist = compute_behavior_distribution(results, dimension)

    if label not in dist.columns:
        return []

    pct_col = f"{label}_pct"
    if pct_col not in dist.columns:
        # Compute percentages
        dist[pct_col] = dist[label] / dist["_total"]

    transitions = []
    prev_pct = None

    for _, row in dist.iterrows():
        pct = row[pct_col]

        if prev_pct is not None:
            change = abs(pct - prev_pct)
            if change >= threshold:
                transitions.append(
                    {
                        "dimension_value": row[dimension],
                        "label": label,
                        "before": prev_pct,
                        "after": pct,
                        "change": change,
                        "direction": "increase" if pct > prev_pct else "decrease",
                    }
                )

        prev_pct = pct

    return transitions


def compute_entropy(counts: dict[str, int]) -> float:
    """
    Compute entropy of a label distribution.

    Higher entropy means more uniform distribution (more uncertain behavior).
    Lower entropy means behavior is more predictable.

    Args:
        counts: Mapping of label to count.

    Returns:
        Shannon entropy value.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0

    probs = [count / total for count in counts.values() if count > 0]
    return float(-np.sum([p * np.log2(p) for p in probs]))


def summarize_results(
    results: ExperimentResults,
) -> list[dict[str, Any]]:
    """
    Generate summary statistics for results.

    Args:
        results: Experiment results.

    Returns:
        List of summary rows suitable for display. Each row includes:
        - dimension: Name of the sweep dimension
        - dimension_value: The specific value for that dimension
        - label: Judgment label
        - count: Number of occurrences
        - proportion: Share of this label within the dimension value
    """
    # Get dimension names from config or infer from results
    dimension_names = []
    if results.config and "sweep" in results.config:
        dimension_names = [
            d["name"] for d in results.config["sweep"].get("dimensions", [])
        ]
    elif results.results:
        dimension_names = list(results.results[0].variables.keys())

    summary = []

    for dim in dimension_names:
        dist = compute_behavior_distribution(results, dim)

        for _, row in dist.iterrows():
            for col in dist.columns:
                if col.startswith("_") or col == dim or col.endswith("_pct"):
                    continue

                raw_count = row.get(col, 0)
                count = 0 if pd.isna(raw_count) else int(raw_count)
                total = row.get("_total", 0)
                total = 0 if pd.isna(total) else total
                pct = count / total if total > 0 else 0
                if count <= 0:
                    continue

                summary.append(
                    {
                        "dimension": dim,
                        "dimension_value": row[dim],
                        "label": col,
                        "count": int(count),
                        "proportion": pct,
                    }
                )

    return summary


def compute_label_stats(results: ExperimentResults) -> dict[str, Any]:
    """
    Compute overall label statistics.

    Args:
        results: Experiment results.

    Returns:
        Dictionary with label counts, proportions, and entropy.
    """
    labels = [r.label for r in results]
    counts = Counter(labels)
    total = len(labels)

    return {
        "total": total,
        "counts": dict(counts),
        "proportions": {k: v / total for k, v in counts.items()},
        "entropy": compute_entropy(counts),
        "unique_labels": len(counts),
    }


# ============================================================================
# Behavior Embedding Analysis Functions
# ============================================================================


def get_behavior_embeddings(
    results: ExperimentResults,
) -> tuple[list[np.ndarray], list[int]]:
    """
    Extract all behavior embeddings from results.

    Args:
        results: Experiment results.

    Returns:
        Tuple of (list of embedding vectors, list of corresponding indices).
    """
    embeddings = []
    indices = []

    for result in results:
        if result.behavior_embedding is not None:
            embeddings.append(np.array(result.behavior_embedding))
            indices.append(result.index)

    return embeddings, indices


def compute_embedding_stats(results: ExperimentResults) -> dict[str, Any]:
    """
    Compute statistics across all behavior embeddings.

    Args:
        results: Experiment results with behavior embeddings.

    Returns:
        Dictionary with mean, std, and range for each dimension.
    """
    embeddings, _ = get_behavior_embeddings(results)

    if not embeddings:
        return {"error": "No behavior embeddings found"}

    emb_matrix = np.array(embeddings)

    dimension_names = [
        "refusal_strength",
        "moral_justification",
        "epistemic_hedging",
        "power_asymmetry",
        "self_reference",
        "stance_polarity",
    ]

    stats = {}
    for i, name in enumerate(dimension_names):
        col = emb_matrix[:, i]
        stats[name] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "median": float(np.median(col)),
        }

    return stats


def compute_embedding_by_dimension(
    results: ExperimentResults,
    dimension: str,
) -> dict[Any, dict[str, float]]:
    """
    Compute average behavior embeddings grouped by a sweep dimension.

    Args:
        results: Experiment results.
        dimension: Dimension to group by.

    Returns:
        Mapping of dimension value to average embedding values.
    """
    groups = aggregate_by_dimension(results, dimension)

    dimension_names = [
        "refusal_strength",
        "moral_justification",
        "epistemic_hedging",
        "power_asymmetry",
        "self_reference",
        "stance_polarity",
    ]

    result_dict = {}
    for dim_value, group_results in sorted(groups.items()):
        embeddings = [
            np.array(r.behavior_embedding)
            for r in group_results
            if r.behavior_embedding is not None
        ]

        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0)
            result_dict[dim_value] = {
                name: float(avg_embedding[i]) for i, name in enumerate(dimension_names)
            }

    return result_dict


def detect_embedding_phase_transitions(
    results: ExperimentResults,
    dimension: str,
    embedding_dim: str = "refusal_strength",
    threshold: float = 0.15,
) -> list[dict[str, Any]]:
    """
    Detect phase transitions in behavior embedding space.

    Args:
        results: Experiment results.
        dimension: Sweep dimension to analyze.
        embedding_dim: Which embedding dimension to track.
        threshold: Minimum change to consider a transition.

    Returns:
        List of transition points.
    """
    emb_by_dim = compute_embedding_by_dimension(results, dimension)

    dim_names = [
        "refusal_strength",
        "moral_justification",
        "epistemic_hedging",
        "power_asymmetry",
        "self_reference",
        "stance_polarity",
    ]

    if embedding_dim not in dim_names:
        return []

    transitions = []
    prev_value = None
    prev_dim = None

    for dim_value in sorted(emb_by_dim.keys()):
        current_value = emb_by_dim[dim_value].get(embedding_dim)

        if current_value is not None and prev_value is not None:
            change = abs(current_value - prev_value)
            if change >= threshold:
                transitions.append(
                    {
                        "dimension_value": dim_value,
                        "embedding_dim": embedding_dim,
                        "before": prev_value,
                        "after": current_value,
                        "change": change,
                        "direction": "increase"
                        if current_value > prev_value
                        else "decrease",
                        "from_dim_value": prev_dim,
                    }
                )

        prev_value = current_value
        prev_dim = dim_value

    return transitions


def compute_metaprobe_stats(results: ExperimentResults) -> dict[str, Any]:
    """
    Compute statistics across all metacognition probe results.

    Args:
        results: Experiment results with meta probes.

    Returns:
        Dictionary with stats for each probe type.
    """
    probe_types = [
        "self_consistency",
        "policy_boundary",
        "memory_confusion",
        "reframe_stability",
        "overall_metacognition",
    ]

    stats: dict[str, Any] = {}

    for probe_type in probe_types:
        values = []
        for result in results:
            if result.meta_probe is not None:
                value = result.meta_probe.get(probe_type)
                if value is not None:
                    values.append(value)

        if values:
            stats[probe_type] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "count": len(values),
            }

    return stats


def correlate_embedding_with_label(
    results: ExperimentResults,
    embedding_dim: str = "refusal_strength",
) -> dict[str, float]:
    """
    Compute average embedding value for each label.

    Args:
        results: Experiment results.
        embedding_dim: Which embedding dimension to analyze.

    Returns:
        Mapping of label to average embedding value.
    """
    dim_idx = {
        "refusal_strength": 0,
        "moral_justification": 1,
        "epistemic_hedging": 2,
        "power_asymmetry": 3,
        "self_reference": 4,
        "stance_polarity": 5,
    }.get(embedding_dim, 0)

    label_values: dict[str, list[float]] = {}

    for result in results:
        if result.behavior_embedding is not None:
            label = result.label
            if label not in label_values:
                label_values[label] = []
            label_values[label].append(result.behavior_embedding[dim_idx])

    return {label: float(np.mean(values)) for label, values in label_values.items()}
