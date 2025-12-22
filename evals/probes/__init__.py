"""
Probes for evaluating and classifying model responses.

This subpackage contains:
- judge: LLM and rule-based response classification
- metaprobes: Metacognition and self-awareness testing
- behavior_vectorizer: Response fingerprinting via embeddings
"""

# Judge components
# Behavior embeddings
from evals.probes.behavior_vectorizer import (
    BehaviorEmbedding,
    compute_behavior_embedding,
    compute_embedding_centroid,
    compute_embedding_variance,
    detect_embedding_outliers,
)
from evals.probes.judge import (
    BaseJudge,
    CompositeJudge,
    Judgment,
    LLMJudge,
    RuleBasedJudge,
    create_judge,
)

# Metacognition probes
from evals.probes.metaprobes import (
    MEMORY_CONFUSION_PROMPTS,
    POLICY_BOUNDARY_PROMPTS,
    REFRAME_STABILITY_PROMPTS,
    SELF_AWARENESS_PROMPTS,
    MetaProbeResult,
    ProbeResponse,
    build_probe_turn,
    run_all_metaprobes,
    run_memory_confusion_probe,
    run_policy_boundary_probe,
    run_self_awareness_probe,
)

__all__ = [
    # Judge
    "Judgment",
    "BaseJudge",
    "LLMJudge",
    "RuleBasedJudge",
    "CompositeJudge",
    "create_judge",
    # Metaprobes
    "ProbeResponse",
    "MetaProbeResult",
    "build_probe_turn",
    "run_all_metaprobes",
    "run_self_awareness_probe",
    "run_policy_boundary_probe",
    "run_memory_confusion_probe",
    "SELF_AWARENESS_PROMPTS",
    "POLICY_BOUNDARY_PROMPTS",
    "MEMORY_CONFUSION_PROMPTS",
    "REFRAME_STABILITY_PROMPTS",
    # Behavior embeddings
    "BehaviorEmbedding",
    "compute_behavior_embedding",
    "compute_embedding_centroid",
    "compute_embedding_variance",
    "detect_embedding_outliers",
]
