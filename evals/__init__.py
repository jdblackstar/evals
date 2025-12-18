"""
Behavioral Research Substrate - A modular harness for LLM experiments.

This package provides:
- Sweep engine for parameter exploration
- Model runner with caching and multi-turn support
- Judgment layer for response classification
- Sequence tasks for path-dependent evaluation
- Behavior embeddings for response fingerprinting
- Metacognition probes for self-awareness testing
- Analysis and visualization tools
- Reproducible experiment logging
"""

__version__ = "0.1.0"

# Core experiment components
from evals.analysis import (
    ExperimentResult,
    ExperimentResults,
    compute_behavior_distribution,
    detect_phase_transitions,
    load_results,
)
from evals.config import ExperimentConfig, load_config
from evals.experiment import run_experiment, run_experiment_batch

# Primitives
from evals.primitives import ConversationHistory, Turn

# Behavior embeddings
from evals.probes.behavior_vectorizer import (
    BehaviorEmbedding,
    compute_behavior_embedding,
    compute_embedding_centroid,
)

# Metacognition probes
from evals.probes.metaprobes import (
    MetaProbeResult,
    run_all_metaprobes,
    run_memory_confusion_probe,
    run_policy_boundary_probe,
    run_self_awareness_probe,
)
from evals.runner import Completion, ConversationResult, ModelRunner

# Sequence tasks
from evals.sequences import (
    HysteresisResult,
    SequenceTask,
    TurnTemplate,
    compute_hysteresis,
)

__all__ = [
    # Version
    "__version__",
    # Config
    "ExperimentConfig",
    "load_config",
    # Experiment
    "run_experiment",
    "run_experiment_batch",
    # Runner
    "ModelRunner",
    "Completion",
    "ConversationResult",
    # Analysis
    "ExperimentResult",
    "ExperimentResults",
    "load_results",
    "compute_behavior_distribution",
    "detect_phase_transitions",
    # Sequences
    "SequenceTask",
    "TurnTemplate",
    "compute_hysteresis",
    "HysteresisResult",
    # Behavior embeddings
    "BehaviorEmbedding",
    "compute_behavior_embedding",
    "compute_embedding_centroid",
    # Metaprobes
    "MetaProbeResult",
    "run_all_metaprobes",
    "run_self_awareness_probe",
    "run_policy_boundary_probe",
    "run_memory_confusion_probe",
    # Primitives
    "Turn",
    "ConversationHistory",
]
