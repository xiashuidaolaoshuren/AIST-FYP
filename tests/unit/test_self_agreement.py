"""
Unit tests for Self-Agreement Detector.

Tests the consistency measurement functionality for hallucination detection.
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch
from src.verification.self_agreement import SelfAgreementDetector
from src.utils.data_structures import EvidenceChunk


@pytest.fixture
def sample_config():
    """Sample configuration for SelfAgreementDetector."""
    return {
        'verification': {
            'self_agreement': {
                'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                'k_samples': 3,  # Use 3 for faster tests
                'temperature': 1.5,
                'device': 'cpu'  # Use CPU for tests
            }
        }
    }


@pytest.fixture
def mock_generator():
    """Mock GeneratorWrapper for testing."""
    generator = Mock()
    
    # Mock generate_with_metadata to return different responses
    def generate_side_effect(prompt, evidence_chunks=None, **kwargs):
        # Simulate stochastic generation with slight variations
        import random
        responses = [
            "Machine learning is a subset of artificial intelligence.",
            "ML is a part of AI technology.",
            "Machine learning belongs to the field of AI.",
            "ML is a subfield of artificial intelligence.",
            "Machine learning is an AI technique."
        ]
        return {
            'text': random.choice(responses),
            'tokens': [],
            'token_ids': [],
            'logits': [],
            'scores': [],
            'evidence_used': []
        }
    
    generator.generate_with_metadata = Mock(side_effect=generate_side_effect)
    return generator


@pytest.fixture
def sample_evidence():
    """Sample evidence chunks."""
    return [
        EvidenceChunk(
            doc_id="doc1",
            sent_id=0,
            text="Machine learning is a subset of artificial intelligence that focuses on data.",
            char_start=0,
            char_end=77,
            score_dense=0.95,
            rank=0
        ),
        EvidenceChunk(
            doc_id="doc1",
            sent_id=1,
            text="AI encompasses machine learning, deep learning, and other techniques.",
            char_start=78,
            char_end=148,
            score_dense=0.90,
            rank=1
        )
    ]


class TestSelfAgreementDetector:
    """Test suite for SelfAgreementDetector class."""
    
    def test_initialization(self, sample_config, mock_generator):
        """Test detector initialization."""
        detector = SelfAgreementDetector(sample_config, mock_generator)
        
        assert detector.config == sample_config
        assert detector.generator == mock_generator
        assert detector.k_samples == 3
        assert detector.temperature == 1.5
        assert detector.model_name == 'sentence-transformers/all-MiniLM-L6-v2'
        assert detector.similarity_model is not None
        assert detector.device == 'cpu'
    
    def test_initialization_default_values(self, mock_generator):
        """Test initialization with default configuration values."""
        config = {'verification': {}}
        detector = SelfAgreementDetector(config, mock_generator)
        
        # Should use defaults
        assert detector.k_samples == 5
        assert detector.temperature == 1.5
        assert 'all-MiniLM-L6-v2' in detector.model_name
    
    def test_generate_samples(self, sample_config, mock_generator, sample_evidence):
        """Test sample generation."""
        detector = SelfAgreementDetector(sample_config, mock_generator)
        
        query = "What is machine learning?"
        samples = detector.generate_samples(query, sample_evidence, k=3)
        
        assert len(samples) == 3
        assert all(isinstance(s, str) for s in samples)
        assert all(len(s) > 0 for s in samples)
        
        # Verify generator was called 3 times
        assert mock_generator.generate_with_metadata.call_count == 3
    
    def test_generate_samples_empty_query(self, sample_config, mock_generator):
        """Test generate_samples raises error with empty query."""
        detector = SelfAgreementDetector(sample_config, mock_generator)
        
        with pytest.raises(ValueError, match="query cannot be empty"):
            detector.generate_samples("")
    
    def test_generate_samples_none_query(self, sample_config, mock_generator):
        """Test generate_samples raises error with None query."""
        detector = SelfAgreementDetector(sample_config, mock_generator)
        
        with pytest.raises(ValueError, match="query cannot be empty"):
            detector.generate_samples(None)
    
    def test_generate_samples_whitespace_query(self, sample_config, mock_generator):
        """Test generate_samples raises error with whitespace-only query."""
        detector = SelfAgreementDetector(sample_config, mock_generator)
        
        with pytest.raises(ValueError, match="query cannot be empty"):
            detector.generate_samples("   ")
    
    def test_generate_samples_custom_k(self, sample_config, mock_generator, sample_evidence):
        """Test generate_samples with custom k value."""
        detector = SelfAgreementDetector(sample_config, mock_generator)
        
        samples = detector.generate_samples("What is AI?", sample_evidence, k=5)
        
        assert len(samples) == 5
        assert mock_generator.generate_with_metadata.call_count == 5
    
    def test_measure_consistency_high_similarity(self, sample_config, mock_generator):
        """Test consistency measurement with similar samples."""
        detector = SelfAgreementDetector(sample_config, mock_generator)
        
        original = "Machine learning is a subset of AI"
        samples = [
            "ML is a subset of artificial intelligence",
            "Machine learning is part of AI",
            "ML belongs to AI"
        ]
        
        result = detector.measure_consistency(original, samples)
        
        assert 'consistency_score' in result
        assert 'variance' in result
        assert 'individual_scores' in result
        assert 'min_score' in result
        assert 'max_score' in result
        
        # High similarity should give high consistency score
        assert result['consistency_score'] > 0.7
        assert len(result['individual_scores']) == 3
    
    def test_measure_consistency_low_similarity(self, sample_config, mock_generator):
        """Test consistency measurement with dissimilar samples."""
        detector = SelfAgreementDetector(sample_config, mock_generator)
        
        original = "Machine learning is a subset of AI"
        samples = [
            "The weather is sunny today",
            "Python is a programming language",
            "The cat sits on the mat"
        ]
        
        result = detector.measure_consistency(original, samples)
        
        # Low similarity should give low consistency score
        assert result['consistency_score'] < 0.3
        assert result['variance'] > 0  # Should have variance in scores
    
    def test_measure_consistency_empty_original(self, sample_config, mock_generator):
        """Test measure_consistency raises error with empty original answer."""
        detector = SelfAgreementDetector(sample_config, mock_generator)
        
        with pytest.raises(ValueError, match="original_answer cannot be empty"):
            detector.measure_consistency("", ["sample1", "sample2"])
    
    def test_measure_consistency_empty_samples(self, sample_config, mock_generator):
        """Test measure_consistency raises error with empty samples list."""
        detector = SelfAgreementDetector(sample_config, mock_generator)
        
        with pytest.raises(ValueError, match="samples list cannot be empty"):
            detector.measure_consistency("original answer", [])
    
    def test_measure_consistency_empty_sample_in_list(self, sample_config, mock_generator):
        """Test measure_consistency handles empty samples gracefully."""
        detector = SelfAgreementDetector(sample_config, mock_generator)

        result = detector.measure_consistency("original", ["sample1", "", "sample3"])
        assert 'consistency_score' in result
    
    def test_detect_integration(self, sample_config, mock_generator, sample_evidence):
        """Test full detect() method integration."""
        detector = SelfAgreementDetector(sample_config, mock_generator)
        
        claim_text = "Machine learning is a subset of AI"
        query = "What is machine learning?"
        
        result = detector.detect(claim_text, query, sample_evidence)
        
        assert 'variance' in result
        assert 'score' in result
        assert 'samples_generated' in result
        
        assert result['samples_generated'] == 3
        assert isinstance(result['score'], float)
        assert isinstance(result['variance'], float)
        assert 0.0 <= result['score'] <= 1.0
    
    def test_detect_empty_claim(self, sample_config, mock_generator, sample_evidence):
        """Test detect raises error with empty claim."""
        detector = SelfAgreementDetector(sample_config, mock_generator)
        
        with pytest.raises(ValueError, match="claim_text cannot be empty"):
            detector.detect("", "query", sample_evidence)
    
    def test_detect_empty_query(self, sample_config, mock_generator, sample_evidence):
        """Test detect raises error with empty query."""
        detector = SelfAgreementDetector(sample_config, mock_generator)
        
        with pytest.raises(ValueError, match="query cannot be empty"):
            detector.detect("claim", "", sample_evidence)
    
    def test_detect_none_evidence(self, sample_config, mock_generator):
        """Test detect handles None evidence gracefully."""
        detector = SelfAgreementDetector(sample_config, mock_generator)
        
        result = detector.detect("claim text", "query", None)
        
        # Should not raise error, but generate samples without evidence
        assert result is not None
        assert 'score' in result
    
    def test_consistency_score_properties(self, sample_config, mock_generator):
        """Test that consistency score has expected mathematical properties."""
        detector = SelfAgreementDetector(sample_config, mock_generator)
        
        # Identical samples should give perfect consistency
        original = "Machine learning is a branch of AI"
        identical_samples = [original, original, original]
        
        result = detector.measure_consistency(original, identical_samples)
        
        # Should be close to 1.0 (perfect similarity)
        assert result['consistency_score'] > 0.95
        assert result['min_score'] > 0.95
        assert result['max_score'] > 0.95
        assert result['variance'] < 0.01  # Very low variance
    
    def test_generation_failure_handling(self, sample_config, sample_evidence):
        """Test handling of generation failures."""
        # Create generator that raises exception
        failing_generator = Mock()
        failing_generator.generate_with_metadata = Mock(
            side_effect=RuntimeError("Generation failed")
        )
        
        detector = SelfAgreementDetector(sample_config, failing_generator)
        
        with pytest.raises(RuntimeError, match="All 3 generation attempts produced empty samples"):
            detector.generate_samples("query", sample_evidence)
    
    def test_similarity_model_error_handling(self, sample_config, mock_generator):
        """Test handling of similarity model errors."""
        detector = SelfAgreementDetector(sample_config, mock_generator)
        
        # Patch the similarity model to raise exception
        with patch.object(detector.similarity_model, 'encode', side_effect=Exception("Model error")):
            result = detector.measure_consistency("original", ["sample1", "sample2"])
            
            # Should return fallback values
            assert result['consistency_score'] == 0.5
            assert result['variance'] == 0.0
    
    def test_detect_error_handling(self, sample_config, sample_evidence):
        """Test detect() error handling."""
        # Create generator that fails
        failing_generator = Mock()
        failing_generator.generate_with_metadata = Mock(
            side_effect=Exception("Generation error")
        )
        
        detector = SelfAgreementDetector(sample_config, failing_generator)
        
        result = detector.detect("claim", "query", sample_evidence)
        
        # Should return fallback indicating failure
        assert result['variance'] is None
        assert result['score'] is None
        assert result['samples_generated'] == 0

    def test_detect_batch_uses_batched_generation(self, sample_config, sample_evidence):
        """detect_batch should call generator.generate_batch_n_samples once for misses."""
        generator = Mock()
        generator.generate_batch_n_samples = Mock(
            return_value=[
                ["a1", "a2", "a3"],
                ["b1", "b2", "b3"],
            ]
        )
        detector = SelfAgreementDetector(sample_config, generator)

        with patch.object(
            detector,
            'measure_consistency',
            side_effect=[
                {'consistency_score': 0.8, 'variance': 0.01},
                {'consistency_score': 0.6, 'variance': 0.03},
            ],
        ):
            results = detector.detect_batch(
                claim_texts=["claim 1", "claim 2"],
                queries=["q1", "q2"],
                evidence_chunks_list=[[sample_evidence[0]], [sample_evidence[1]]],
            )

        generator.generate_batch_n_samples.assert_called_once()
        assert len(results) == 2
        assert results[0]['samples_generated'] == 3
        assert results[1]['samples_generated'] == 3

    def test_detect_batch_cache_hit_skips_generation(self, sample_config, sample_evidence):
        """detect_batch should reuse cached samples and avoid generator calls."""
        generator = Mock()
        generator.generate_batch_n_samples = Mock(return_value=[])
        detector = SelfAgreementDetector(sample_config, generator)

        query = "cached query"
        evidence = [sample_evidence[0]]
        cache_key = detector._cache_key(query, evidence)
        detector._sample_cache[cache_key] = ["s1", "s2", "s3"]

        with patch.object(
            detector,
            'measure_consistency',
            return_value={'consistency_score': 0.9, 'variance': 0.0},
        ):
            results = detector.detect_batch(
                claim_texts=["claim"],
                queries=[query],
                evidence_chunks_list=[evidence],
            )

        generator.generate_batch_n_samples.assert_not_called()
        assert len(results) == 1
        assert results[0]['samples_generated'] == 3
