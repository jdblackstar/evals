"""Tests for evals.behavior_vectorizer module."""

import numpy as np
import pytest

from evals.probes.behavior_vectorizer import (
    BehaviorEmbedding,
    compute_behavior_embedding,
    compute_embedding_centroid,
    compute_embedding_variance,
    detect_embedding_outliers,
)


class TestBehaviorEmbedding:
    """Tests for the BehaviorEmbedding dataclass."""

    def test_creation(self):
        """Test creating a behavior embedding."""
        emb = BehaviorEmbedding(
            refusal_strength=0.5,
            moral_justification=0.3,
            epistemic_hedging=0.2,
            power_asymmetry=0.1,
            self_reference=0.4,
            stance_polarity=-0.2,
        )
        assert emb.refusal_strength == 0.5
        assert emb.stance_polarity == -0.2

    def test_to_vector(self):
        """Test conversion to numpy vector."""
        emb = BehaviorEmbedding(
            refusal_strength=0.1,
            moral_justification=0.2,
            epistemic_hedging=0.3,
            power_asymmetry=0.4,
            self_reference=0.5,
            stance_polarity=0.6,
        )
        vec = emb.to_vector()

        assert isinstance(vec, np.ndarray)
        assert len(vec) == 6
        assert vec[0] == 0.1
        assert vec[5] == 0.6

    def test_to_dict(self):
        """Test conversion to dictionary."""
        emb = BehaviorEmbedding(
            refusal_strength=0.5,
            moral_justification=0.3,
            epistemic_hedging=0.2,
            power_asymmetry=0.1,
            self_reference=0.4,
            stance_polarity=-0.2,
        )
        d = emb.to_dict()

        assert d["refusal_strength"] == 0.5
        assert d["stance_polarity"] == -0.2
        assert len(d) == 6

    def test_to_list(self):
        """Test conversion to list."""
        emb = BehaviorEmbedding(
            refusal_strength=0.1,
            moral_justification=0.2,
            epistemic_hedging=0.3,
            power_asymmetry=0.4,
            self_reference=0.5,
            stance_polarity=0.6,
        )
        lst = emb.to_list()

        assert lst == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    def test_from_vector(self):
        """Test creating from a vector."""
        vec = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        emb = BehaviorEmbedding.from_vector(vec)

        assert emb.refusal_strength == 0.1
        assert emb.stance_polarity == 0.6

    def test_from_vector_wrong_size(self):
        """Test that wrong-sized vectors raise error."""
        with pytest.raises(ValueError):
            BehaviorEmbedding.from_vector([0.1, 0.2, 0.3])

    def test_distance(self):
        """Test Euclidean distance between embeddings."""
        emb1 = BehaviorEmbedding.from_vector([0, 0, 0, 0, 0, 0])
        emb2 = BehaviorEmbedding.from_vector([1, 0, 0, 0, 0, 0])

        assert emb1.distance(emb2) == 1.0

    def test_distance_identical(self):
        """Test distance between identical embeddings."""
        emb = BehaviorEmbedding.from_vector([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        assert emb.distance(emb) == 0.0

    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical embeddings."""
        emb = BehaviorEmbedding.from_vector([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        assert emb.cosine_similarity(emb) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal embeddings."""
        emb1 = BehaviorEmbedding.from_vector([1, 0, 0, 0, 0, 0])
        emb2 = BehaviorEmbedding.from_vector([0, 1, 0, 0, 0, 0])
        assert emb1.cosine_similarity(emb2) == pytest.approx(0.0)


class TestComputeBehaviorEmbedding:
    """Tests for the compute_behavior_embedding function."""

    def test_empty_response(self):
        """Test embedding of empty response."""
        emb = compute_behavior_embedding("")
        assert emb.refusal_strength == 0.0

    def test_refusal_detection(self):
        """Test detection of refusal patterns."""
        refusal = "I cannot help with that request. I'm unable to assist."
        emb = compute_behavior_embedding(refusal)

        assert emb.refusal_strength > 0.3

    def test_no_refusal(self):
        """Test that helpful responses have low refusal."""
        helpful = "Sure, I'd be happy to help you with that task!"
        emb = compute_behavior_embedding(helpful)

        assert emb.refusal_strength < 0.2

    def test_moral_justification(self):
        """Test detection of moral/ethical language."""
        moral = "This raises ethical concerns. We should consider the moral implications and potential harm."
        emb = compute_behavior_embedding(moral)

        assert emb.moral_justification > 0.2

    def test_epistemic_hedging(self):
        """Test detection of hedging language."""
        hedged = "I think perhaps this might be the case, though I'm not entirely sure."
        emb = compute_behavior_embedding(hedged)

        assert emb.epistemic_hedging > 0.2

    def test_deference(self):
        """Test detection of deferential language."""
        deferential = (
            "If you'd like, I could help. It's up to you, whatever you prefer."
        )
        emb = compute_behavior_embedding(deferential)

        # Positive power_asymmetry = deferential
        assert emb.power_asymmetry > 0

    def test_assertive(self):
        """Test detection of assertive language."""
        assertive = (
            "You should definitely do this. The best approach is clearly option A."
        )
        emb = compute_behavior_embedding(assertive)

        # Negative power_asymmetry = assertive
        assert emb.power_asymmetry < 0

    def test_self_reference(self):
        """Test detection of self-referential language."""
        self_ref = "I think I can help. My understanding is that I should assist you."
        emb = compute_behavior_embedding(self_ref)

        assert emb.self_reference > 0.1

    def test_raw_features_stored(self):
        """Test that raw features are stored."""
        emb = compute_behavior_embedding("I cannot help with that.")
        assert "word_count" in emb.raw_features
        assert "refusal_matches" in emb.raw_features


class TestEmbeddingAnalysis:
    """Tests for embedding analysis functions."""

    def test_compute_centroid(self):
        """Test computing centroid of embeddings."""
        embeddings = [
            BehaviorEmbedding.from_vector([0, 0, 0, 0, 0, 0]),
            BehaviorEmbedding.from_vector([1, 1, 1, 1, 1, 1]),
        ]
        centroid = compute_embedding_centroid(embeddings)

        assert centroid.refusal_strength == pytest.approx(0.5)
        assert centroid.stance_polarity == pytest.approx(0.5)

    def test_compute_centroid_empty(self):
        """Test centroid with empty list raises error."""
        with pytest.raises(ValueError):
            compute_embedding_centroid([])

    def test_compute_variance(self):
        """Test computing variance across embeddings."""
        embeddings = [
            BehaviorEmbedding.from_vector([0, 0, 0, 0, 0, 0]),
            BehaviorEmbedding.from_vector([1, 0, 0, 0, 0, 0]),
        ]
        variance = compute_embedding_variance(embeddings)

        assert variance["refusal_strength"] > 0  # Has variance
        assert variance["moral_justification"] == 0  # No variance

    def test_compute_variance_single(self):
        """Test variance with single embedding."""
        embeddings = [BehaviorEmbedding.from_vector([0.5] * 6)]
        variance = compute_embedding_variance(embeddings)

        # All zeros for single embedding
        assert all(v == 0 for v in variance.values())

    def test_detect_outliers(self):
        """Test outlier detection."""
        # Create embeddings with one outlier
        normal = BehaviorEmbedding.from_vector([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        outlier = BehaviorEmbedding.from_vector([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])

        embeddings = [normal, normal, normal, normal, outlier]
        outliers = detect_embedding_outliers(embeddings, threshold=2.0)

        assert 4 in outliers  # The outlier is at index 4

    def test_detect_outliers_none(self):
        """Test when there are no outliers."""
        embeddings = [
            BehaviorEmbedding.from_vector([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
            for _ in range(5)
        ]
        outliers = detect_embedding_outliers(embeddings)

        assert len(outliers) == 0

    def test_detect_outliers_few_samples(self):
        """Test outlier detection with too few samples."""
        embeddings = [BehaviorEmbedding.from_vector([0.5] * 6)]
        outliers = detect_embedding_outliers(embeddings)

        assert outliers == []


class TestRealWorldResponses:
    """Integration tests with realistic response patterns."""

    def test_typical_refusal(self):
        """Test embedding of a typical AI refusal."""
        response = """I cannot assist with that request. While I understand you may 
        have reasons for asking, I need to decline as it could potentially cause harm. 
        Perhaps I could help you with something else instead?"""

        emb = compute_behavior_embedding(response)

        assert emb.refusal_strength > 0.2
        assert emb.moral_justification > 0  # "harm" mentioned

    def test_typical_compliance(self):
        """Test embedding of a typical helpful response."""
        response = """Absolutely! Here's how you can do that:

        1. First, open the settings menu
        2. Navigate to the preferences section
        3. Enable the feature you want

        Let me know if you need any clarification!"""

        emb = compute_behavior_embedding(response)

        assert emb.refusal_strength < 0.1
        assert emb.stance_polarity > -0.5  # Reasonably assertive

    def test_hedged_response(self):
        """Test embedding of a heavily hedged response."""
        response = """I think this might work, but I'm not entirely sure. Perhaps 
        you could try it and see? It's possible that the outcome could vary. 
        Generally speaking, it tends to work, but your mileage may vary."""

        emb = compute_behavior_embedding(response)

        assert emb.epistemic_hedging > 0.3
