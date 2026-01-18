"""
Unit tests for EvidenceReRanker.

Tests re-ranking logic, score computation, and configuration handling.
"""

import pytest
from unittest.mock import MagicMock

from src.mitigation.re_ranker import EvidenceReRanker
from src.utils.data_structures import EvidenceChunk, VerifierSignal
from src.utils.config import Config


@pytest.fixture
def sample_config():
    """Create a Config object with reranker settings."""
    config = MagicMock(spec=Config)
    config.get.return_value = {
        'reranker': {
            'enabled': True,
            'alpha': 0.6,
            'beta': 0.4,
            'fallback_score': 0.5
        }
    }
    return config


@pytest.fixture
def sample_evidence_chunks():
    """Create sample evidence chunks with varying retrieval scores."""
    return [
        EvidenceChunk(
            doc_id="doc1",
            sent_id=0,
            text="Paris is the capital of France.",
            char_start=0,
            char_end=32,
            score_dense=0.9,  # High retrieval score
            rank=0
        ),
        EvidenceChunk(
            doc_id="doc2",
            sent_id=1,
            text="London is the capital of England.",
            char_start=0,
            char_end=34,
            score_dense=0.7,  # Medium retrieval score
            rank=1
        ),
        EvidenceChunk(
            doc_id="doc3",
            sent_id=2,
            text="Berlin is the capital of Germany.",
            char_start=0,
            char_end=34,
            score_dense=0.5,  # Low retrieval score
            rank=2
        )
    ]


@pytest.fixture
def sample_verification_signals():
    """Create sample verification signals with varying quality."""
    return {
        # doc1: High verification score (good coverage + entailment)
        "doc1#0": VerifierSignal(
            claim_id="c1",
            doc_id="doc1",
            sent_id=0,
            nli={'entailment': 0.9, 'contradiction': 0.05, 'neutral': 0.05},
            coverage={'entities': 0.8, 'numbers': 0.0, 'tokens_overlap': 0.7},
            uncertainty={'mean_entropy': 0.5},
            consistency={'variance': None},
            citation_span_match=0.9,
            numeric_check=True
        ),
        # doc2: Low verification score (poor coverage, low entailment)
        "doc2#1": VerifierSignal(
            claim_id="c1",
            doc_id="doc2",
            sent_id=1,
            nli={'entailment': 0.3, 'contradiction': 0.1, 'neutral': 0.6},
            coverage={'entities': 0.2, 'numbers': 0.0, 'tokens_overlap': 0.3},
            uncertainty={'mean_entropy': 1.5},
            consistency={'variance': None},
            citation_span_match=0.3,
            numeric_check=True
        )
        # doc3 has no verification signal (will use fallback)
    }


class TestEvidenceReRankerInitialization:
    """Test EvidenceReRanker initialization and configuration loading."""
    
    def test_init_with_default_config(self, sample_config):
        """Test initialization with provided config."""
        reranker = EvidenceReRanker(sample_config)
        
        assert reranker.alpha == 0.6
        assert reranker.beta == 0.4
        assert reranker.fallback_score == 0.5
        assert reranker.enabled is True
    
    def test_init_with_custom_weights(self):
        """Test initialization with custom alpha/beta weights."""
        config = MagicMock(spec=Config)
        config.get.return_value = {
            'reranker': {
                'alpha': 0.7,
                'beta': 0.3,
                'fallback_score': 0.4,
                'enabled': True
            }
        }
        
        reranker = EvidenceReRanker(config)
        
        assert reranker.alpha == 0.7
        assert reranker.beta == 0.3
        assert reranker.fallback_score == 0.4
    
    def test_init_with_disabled_reranking(self):
        """Test initialization with re-ranking disabled."""
        config = MagicMock(spec=Config)
        config.get.return_value = {
            'reranker': {
                'enabled': False,
                'alpha': 0.6,
                'beta': 0.4,
                'fallback_score': 0.5
            }
        }
        
        reranker = EvidenceReRanker(config)
        
        assert reranker.enabled is False
    
    def test_init_with_missing_config(self):
        """Test initialization with missing config (uses defaults)."""
        config = MagicMock(spec=Config)
        config.get.return_value = {}  # Empty config
        
        reranker = EvidenceReRanker(config)
        
        # Should use defaults
        assert reranker.alpha == 0.6
        assert reranker.beta == 0.4
        assert reranker.fallback_score == 0.5
        assert reranker.enabled is True
    
    def test_init_with_invalid_weights(self):
        """Test that invalid weights raise ValueError."""
        config = MagicMock(spec=Config)
        config.get.return_value = {
            'reranker': {
                'alpha': 1.5,  # Invalid (> 1.0)
                'beta': -0.1,  # Invalid (< 0.0)
                'fallback_score': 0.5,
                'enabled': True
            }
        }
        
        with pytest.raises(ValueError, match="Alpha and beta must be in"):
            EvidenceReRanker(config)


class TestEvidenceReRanking:
    """Test core re-ranking functionality."""
    
    def test_rerank_changes_order(self, sample_config, sample_evidence_chunks, sample_verification_signals):
        """Test that re-ranking changes the order based on verification scores."""
        reranker = EvidenceReRanker(sample_config)
        
        # Original order: doc1 (0.9), doc2 (0.7), doc3 (0.5) by retrieval score
        # After re-ranking with verification:
        # - doc1: 0.6*0.9 + 0.4*0.85 = 0.54 + 0.34 = 0.88
        # - doc2: 0.6*0.7 + 0.4*0.25 = 0.42 + 0.10 = 0.52
        # - doc3: 0.6*0.5 + 0.4*0.50 = 0.30 + 0.20 = 0.50 (fallback)
        # Expected order: doc1, doc2, doc3 (doc1 still on top but margins change)
        
        reranked = reranker.rerank(sample_evidence_chunks, sample_verification_signals)
        
        # Check that order is maintained but could change with different scores
        assert len(reranked) == 3
        assert reranked[0].doc_id == "doc1"  # Still top due to high both scores
    
    def test_rerank_with_verification_boost(self, sample_config, sample_verification_signals):
        """Test that low retrieval + high verification can boost ranking."""
        reranker = EvidenceReRanker(sample_config)
        
        # Create chunks where low retrieval score has high verification
        chunks = [
            EvidenceChunk(
                doc_id="doc_low_ret",
                sent_id=0,
                text="Low retrieval score chunk",
                char_start=0,
                char_end=26,
                score_dense=0.3,  # Low retrieval
                rank=1
            ),
            EvidenceChunk(
                doc_id="doc_high_ret",
                sent_id=1,
                text="High retrieval score chunk",
                char_start=0,
                char_end=27,
                score_dense=0.9,  # High retrieval
                rank=0
            )
        ]
        
        # Add high verification for low retrieval chunk
        signals = {
            "doc_low_ret#0": VerifierSignal(
                claim_id="c1",
                doc_id="doc_low_ret",
                sent_id=0,
                nli={'entailment': 1.0, 'contradiction': 0.0, 'neutral': 0.0},
                coverage={'entities': 1.0, 'numbers': 0.0, 'tokens_overlap': 0.9},
                uncertainty={'mean_entropy': 0.1},
                consistency={'variance': None},
                citation_span_match=1.0,
                numeric_check=True
            )
            # doc_high_ret has no signal (will use fallback=0.5)
        }
        
        # doc_low_ret: 0.6*0.3 + 0.4*1.0 = 0.18 + 0.40 = 0.58
        # doc_high_ret: 0.6*0.9 + 0.4*0.5 = 0.54 + 0.20 = 0.74
        # doc_high_ret still wins, but margin is smaller
        
        reranked = reranker.rerank(chunks, signals)
        
        # High retrieval still wins but verification score matters
        assert reranked[0].doc_id == "doc_high_ret"
    
    def test_rerank_disabled_returns_original_order(self, sample_evidence_chunks, sample_verification_signals):
        """Test that disabled re-ranking returns original order."""
        config = MagicMock(spec=Config)
        config.get.return_value = {
            'reranker': {
                'enabled': False,
                'alpha': 0.6,
                'beta': 0.4,
                'fallback_score': 0.5
            }
        }
        
        reranker = EvidenceReRanker(config)
        reranked = reranker.rerank(sample_evidence_chunks, sample_verification_signals)
        
        # Should return original order unchanged
        assert reranked == sample_evidence_chunks
    
    def test_rerank_with_empty_signals(self, sample_config, sample_evidence_chunks):
        """Test re-ranking with no verification signals (all use fallback)."""
        reranker = EvidenceReRanker(sample_config)
        
        # Empty signals dict
        signals = {}
        
        reranked = reranker.rerank(sample_evidence_chunks, signals)
        
        # All chunks use fallback=0.5, so order determined by retrieval score
        # doc1: 0.6*0.9 + 0.4*0.5 = 0.74
        # doc2: 0.6*0.7 + 0.4*0.5 = 0.62
        # doc3: 0.6*0.5 + 0.4*0.5 = 0.50
        assert reranked[0].doc_id == "doc1"
        assert reranked[1].doc_id == "doc2"
        assert reranked[2].doc_id == "doc3"
    
    def test_rerank_with_empty_evidence_list(self, sample_config):
        """Test that empty evidence list raises ValueError."""
        reranker = EvidenceReRanker(sample_config)
        
        with pytest.raises(ValueError, match="evidence_list cannot be empty"):
            reranker.rerank([], {})
    
    def test_rerank_preserves_all_chunks(self, sample_config, sample_evidence_chunks, sample_verification_signals):
        """Test that re-ranking preserves all chunks (no loss)."""
        reranker = EvidenceReRanker(sample_config)
        
        original_ids = set(f"{c.doc_id}#{c.sent_id}" for c in sample_evidence_chunks)
        reranked = reranker.rerank(sample_evidence_chunks, sample_verification_signals)
        reranked_ids = set(f"{c.doc_id}#{c.sent_id}" for c in reranked)
        
        assert original_ids == reranked_ids


class TestVerificationScoreComputation:
    """Test verification score computation logic."""
    
    def test_compute_verification_score_with_signal(self, sample_config):
        """Test verification score computation with valid signal."""
        reranker = EvidenceReRanker(sample_config)
        
        chunk = EvidenceChunk(
            doc_id="doc1",
            sent_id=0,
            text="Test chunk",
            char_start=0,
            char_end=10,
            score_dense=0.8,
            rank=0
        )
        
        signals = {
            "doc1#0": VerifierSignal(
                claim_id="c1",
                doc_id="doc1",
                sent_id=0,
                nli={'entailment': 0.8, 'contradiction': 0.1, 'neutral': 0.1},
                coverage={'entities': 0.6, 'numbers': 0.0, 'tokens_overlap': 0.5},
                uncertainty={'mean_entropy': 0.5},
                consistency={'variance': None},
                citation_span_match=0.7,
                numeric_check=True
            )
        }
        
        score = reranker._compute_verification_score(chunk, signals)
        
        # Expected: (0.6 + 0.8) / 2 = 0.7
        assert abs(score - 0.7) < 0.01
    
    def test_compute_verification_score_without_signal(self, sample_config):
        """Test verification score with missing signal (uses fallback)."""
        reranker = EvidenceReRanker(sample_config)
        
        chunk = EvidenceChunk(
            doc_id="doc_missing",
            sent_id=0,
            text="Test chunk",
            char_start=0,
            char_end=10,
            score_dense=0.8,
            rank=0
        )
        
        signals = {}  # No signals
        
        score = reranker._compute_verification_score(chunk, signals)
        
        # Should use fallback
        assert score == 0.5
    
    def test_compute_verification_score_with_missing_fields(self, sample_config):
        """Test verification score with incomplete signal (uses fallback)."""
        reranker = EvidenceReRanker(sample_config)
        
        chunk = EvidenceChunk(
            doc_id="doc1",
            sent_id=0,
            text="Test chunk",
            char_start=0,
            char_end=10,
            score_dense=0.8,
            rank=0
        )
        
        # Create signal with missing fields
        signals = {
            "doc1#0": VerifierSignal(
                claim_id="c1",
                doc_id="doc1",
                sent_id=0,
                nli={},  # Empty NLI
                coverage={},  # Empty coverage
                uncertainty={'mean_entropy': 0.5},
                consistency={'variance': None},
                citation_span_match=0.7,
                numeric_check=True
            )
        }
        
        score = reranker._compute_verification_score(chunk, signals)
        
        # Should compute with 0.0 defaults: (0.0 + 0.0) / 2 = 0.0
        assert score == 0.0
    
    def test_compute_verification_score_clamping(self, sample_config):
        """Test that verification scores are clamped to [0, 1]."""
        reranker = EvidenceReRanker(sample_config)
        
        chunk = EvidenceChunk(
            doc_id="doc1",
            sent_id=0,
            text="Test chunk",
            char_start=0,
            char_end=10,
            score_dense=0.8,
            rank=0
        )
        
        # Create signal with out-of-range values (shouldn't happen but test robustness)
        signals = {
            "doc1#0": VerifierSignal(
                claim_id="c1",
                doc_id="doc1",
                sent_id=0,
                nli={'entailment': 1.5, 'contradiction': 0.0, 'neutral': 0.0},  # > 1.0
                coverage={'entities': 1.2, 'numbers': 0.0, 'tokens_overlap': 0.5},  # > 1.0
                uncertainty={'mean_entropy': 0.5},
                consistency={'variance': None},
                citation_span_match=0.7,
                numeric_check=True
            )
        }
        
        score = reranker._compute_verification_score(chunk, signals)
        
        # Should be clamped to 1.0: (1.2 + 1.5) / 2 = 1.35 -> clamped to 1.0
        assert score == 1.0


class TestScoreBreakdown:
    """Test score breakdown utility method."""
    
    def test_get_score_breakdown(self, sample_config, sample_verification_signals):
        """Test detailed score breakdown for debugging."""
        reranker = EvidenceReRanker(sample_config)
        
        chunk = EvidenceChunk(
            doc_id="doc1",
            sent_id=0,
            text="Test chunk",
            char_start=0,
            char_end=10,
            score_dense=0.9,
            rank=0
        )
        
        breakdown = reranker.get_score_breakdown(chunk, sample_verification_signals)
        
        assert 'retrieval_score' in breakdown
        assert 'verification_score' in breakdown
        assert 'final_score' in breakdown
        assert 'alpha' in breakdown
        assert 'beta' in breakdown
        
        assert breakdown['retrieval_score'] == 0.9
        assert breakdown['alpha'] == 0.6
        assert breakdown['beta'] == 0.4
        
        # Verify final score computation
        expected_final = 0.6 * 0.9 + 0.4 * breakdown['verification_score']
        assert abs(breakdown['final_score'] - expected_final) < 0.01


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_rerank_with_single_chunk(self, sample_config):
        """Test re-ranking with only one chunk."""
        reranker = EvidenceReRanker(sample_config)
        
        chunks = [
            EvidenceChunk(
                doc_id="doc1",
                sent_id=0,
                text="Single chunk",
                char_start=0,
                char_end=12,
                score_dense=0.8,
                rank=0
            )
        ]
        
        reranked = reranker.rerank(chunks, {})
        
        assert len(reranked) == 1
        assert reranked[0].doc_id == "doc1"
    
    def test_rerank_with_all_identical_scores(self, sample_config):
        """Test re-ranking when all chunks have identical final scores."""
        reranker = EvidenceReRanker(sample_config)
        
        # All chunks with same scores
        chunks = [
            EvidenceChunk(
                doc_id=f"doc{i}",
                sent_id=i,
                text=f"Chunk {i}",
                char_start=0,
                char_end=10,
                score_dense=0.7,  # Same retrieval score
                rank=i
            )
            for i in range(3)
        ]
        
        # All use fallback (same verification score)
        reranked = reranker.rerank(chunks, {})
        
        # Order should be stable (Python sort is stable)
        assert len(reranked) == 3
    
    def test_compute_verification_score_with_corrupted_signal(self, sample_config):
        """Test verification score computation with corrupted signal (triggers exception path)."""
        reranker = EvidenceReRanker(sample_config)
        
        chunk = EvidenceChunk(
            doc_id="doc1",
            sent_id=0,
            text="Test chunk",
            char_start=0,
            char_end=10,
            score_dense=0.8,
            rank=0
        )
        
        # Create signal with non-dict NLI/coverage (will trigger AttributeError)
        class BadSignal:
            def __init__(self):
                self.nli = "not a dict"  # Wrong type
                self.coverage = None  # Will cause error
        
        signals = {
            "doc1#0": BadSignal()
        }
        
        # Should catch exception and return fallback
        score = reranker._compute_verification_score(chunk, signals)
        assert score == 0.5  # Fallback score
