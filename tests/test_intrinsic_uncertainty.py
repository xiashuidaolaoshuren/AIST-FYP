"""
Unit tests for IntrinsicUncertaintyDetector.

Tests entropy calculation, token-claim alignment, edge cases, and numerical stability.
Comprehensive coverage for the intrinsic uncertainty detection functionality.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.verification.intrinsic_uncertainty import IntrinsicUncertaintyDetector
from src.utils.data_structures import Claim, EvidenceChunk
from src.utils.config import Config


class TestIntrinsicUncertaintyDetector:
    """Test suite for IntrinsicUncertaintyDetector class."""
    
    @pytest.fixture
    def sample_config(self):
        """Create a test configuration with verification settings."""
        config = Config()
        # Ensure verification settings exist
        if not hasattr(config, 'verification'):
            config._config['verification'] = {
                'intrinsic': {'epsilon': 1e-10}
            }
        return config
    
    @pytest.fixture
    def sample_claim(self):
        """Create a sample claim for testing."""
        return Claim(
            claim_id='test_c1',
            answer_id='test_a1',
            text='Machine learning is a subset of AI.',
            answer_char_span=[0, 35],
            extraction_method='test'
        )
    
    @pytest.fixture
    def sample_evidence(self):
        """Create a sample evidence chunk for testing."""
        return EvidenceChunk(
            doc_id='test_doc',
            sent_id=1,
            text='Machine learning is a subset of artificial intelligence.',
            char_start=0,
            char_end=56,
            score_dense=0.95,
            rank=1
        )
    
    @pytest.fixture
    def sample_metadata(self):
        """
        Create sample metadata matching generator_wrapper.py structure.
        
        Returns metadata dict with:
        - text: Generated response
        - tokens: List of token strings
        - logits: List of numpy arrays (one per token)
        - token_scores: List of probability scores
        """
        text = 'Machine learning is a subset of AI.'
        tokens = ['Machine', '▁learning', '▁is', '▁a', '▁subset', '▁of', '▁AI', '.']
        
        # Create realistic logits (vocab_size = 100 for testing)
        vocab_size = 100
        logits = []
        for i in range(len(tokens)):
            # Create logits with different distributions
            token_logits = np.random.randn(vocab_size)
            logits.append(token_logits)
        
        token_scores = [0.85, 0.92, 0.95, 0.88, 0.90, 0.93, 0.87, 0.91]
        
        return {
            'text': text,
            'tokens': tokens,
            'logits': logits,
            'token_scores': token_scores
        }
    
    def test_initialization(self, sample_config):
        """Test that detector initializes correctly with config."""
        detector = IntrinsicUncertaintyDetector(sample_config)
        
        assert detector is not None
        assert detector.config == sample_config
        assert detector.epsilon == 1e-10
        assert detector.logger is not None
    
    def test_entropy_uniform_distribution(self, sample_config):
        """
        Test entropy calculation with uniform distribution (maximum entropy).
        
        Uniform distribution should yield high entropy (~log(vocab_size)).
        """
        detector = IntrinsicUncertaintyDetector(sample_config)
        
        # Create uniform logits (all same value)
        vocab_size = 100
        uniform_logits = np.ones(vocab_size)
        
        entropy = detector._calculate_entropy(uniform_logits, detector.epsilon)
        
        # Uniform distribution over 100 tokens: H ≈ log(100) ≈ 4.6
        assert entropy > 4.5
        assert entropy < 4.7
        assert isinstance(entropy, float)
    
    def test_entropy_peaked_distribution(self, sample_config):
        """
        Test entropy calculation with peaked distribution (low entropy).
        
        Peaked distribution (one very high logit) should yield low entropy.
        """
        detector = IntrinsicUncertaintyDetector(sample_config)
        
        # Create peaked logits (one dominant token)
        vocab_size = 100
        peaked_logits = np.ones(vocab_size) * -10.0  # Low baseline
        peaked_logits[0] = 10.0  # One dominant peak
        
        entropy = detector._calculate_entropy(peaked_logits, detector.epsilon)
        
        # Peaked distribution should have very low entropy
        assert entropy < 1.0
        assert entropy >= 0.0
        assert isinstance(entropy, float)
    
    def test_token_claim_alignment_exact(self, sample_config, sample_claim, sample_evidence, sample_metadata):
        """
        Test exact token-claim alignment.
        
        Claim spans exactly match token boundaries in generated text.
        """
        detector = IntrinsicUncertaintyDetector(sample_config)
        
        # Test alignment for exact match
        token_indices = detector._align_claim_tokens(sample_claim, sample_metadata)
        
        # Should find tokens corresponding to the claim
        assert isinstance(token_indices, list)
        assert len(token_indices) > 0
        # Should have tokens for "Machine learning is a subset of AI"
        assert len(token_indices) >= 6  # At least 6-8 tokens (with fuzzy ±1)
    
    def test_token_claim_alignment_fuzzy(self, sample_config, sample_evidence):
        """
        Test fuzzy token-claim alignment with 1-token difference.
        
        Claim boundaries don't exactly match tokens, but fuzzy matching
        should handle it with ±1 token tolerance.
        """
        detector = IntrinsicUncertaintyDetector(sample_config)
        
        # Create claim with slightly offset span
        claim = Claim(
            claim_id='test_c2',
            answer_id='test_a2',
            text='learning is a subset',
            answer_char_span=[8, 28],  # Offset from token boundaries
            extraction_method='test'
        )
        
        metadata = {
            'text': 'Machine learning is a subset of AI.',
            'tokens': ['Machine', '▁learning', '▁is', '▁a', '▁subset', '▁of', '▁AI', '.'],
            'logits': [np.random.randn(100) for _ in range(8)]
        }
        
        token_indices = detector._align_claim_tokens(claim, metadata)
        
        # Should still find tokens with fuzzy matching
        assert isinstance(token_indices, list)
        assert len(token_indices) > 0
        # Should capture tokens around "learning is a subset"
        assert len(token_indices) >= 3  # At least 3-5 tokens
    
    def test_edge_case_empty_claim(self, sample_config, sample_evidence, sample_metadata):
        """
        Test edge case: empty claim returns 0.0 entropy.
        
        Empty claim should be handled gracefully with zero entropy.
        """
        detector = IntrinsicUncertaintyDetector(sample_config)
        
        empty_claim = Claim(
            claim_id='test_c3',
            answer_id='test_a3',
            text='',
            answer_char_span=[0, 0],
            extraction_method='test'
        )
        
        signal = detector.compute_signal(empty_claim, sample_evidence, sample_metadata)
        
        assert signal == {'mean_entropy': 0.0}
    
    def test_edge_case_single_token(self, sample_config, sample_evidence):
        """
        Test edge case: single token claim returns correct entropy.
        
        Single token claim should compute entropy for that one token.
        """
        detector = IntrinsicUncertaintyDetector(sample_config)
        
        single_token_claim = Claim(
            claim_id='test_c4',
            answer_id='test_a4',
            text='Machine',
            answer_char_span=[0, 7],
            extraction_method='test'
        )
        
        metadata = {
            'text': 'Machine',
            'tokens': ['Machine'],
            'logits': [np.random.randn(100)]
        }
        
        signal = detector.compute_signal(single_token_claim, sample_evidence, metadata)
        
        assert 'mean_entropy' in signal
        assert signal['mean_entropy'] > 0.0  # Should have positive entropy
        assert isinstance(signal['mean_entropy'], float)
    
    def test_numerical_stability_extreme_logits(self, sample_config):
        """
        Test numerical stability with very large/small logits.
        
        Extreme logit values should not cause overflow/underflow errors.
        """
        detector = IntrinsicUncertaintyDetector(sample_config)
        
        # Test with very large logits
        large_logits = np.array([1000.0] + [0.0] * 99)
        entropy_large = detector._calculate_entropy(large_logits, detector.epsilon)
        assert np.isfinite(entropy_large)
        assert entropy_large >= 0.0
        
        # Test with very small logits
        small_logits = np.array([-1000.0] * 100)
        entropy_small = detector._calculate_entropy(small_logits, detector.epsilon)
        assert np.isfinite(entropy_small)
        assert entropy_small >= 0.0
        
        # Test with mixed extreme values
        mixed_logits = np.array([1000.0, -1000.0] + [0.0] * 98)
        entropy_mixed = detector._calculate_entropy(mixed_logits, detector.epsilon)
        assert np.isfinite(entropy_mixed)
        assert entropy_mixed >= 0.0
    
    def test_compute_signal_output_format(self, sample_config, sample_claim, sample_evidence, sample_metadata):
        """
        Test that compute_signal returns correct output format.
        
        Output should be {'mean_entropy': float} with valid entropy value.
        """
        detector = IntrinsicUncertaintyDetector(sample_config)
        
        signal = detector.compute_signal(sample_claim, sample_evidence, sample_metadata)
        
        # Check output format
        assert isinstance(signal, dict)
        assert 'mean_entropy' in signal
        assert len(signal) == 1  # Only mean_entropy field
        
        # Check value type and range
        assert isinstance(signal['mean_entropy'], float)
        assert signal['mean_entropy'] >= 0.0
        assert signal['mean_entropy'] <= 15.0  # Reasonable upper bound
    
    def test_alignment_failure_fallback(self, sample_config, sample_evidence):
        """
        Test fallback behavior when token-claim alignment fails.
        
        When alignment fails (e.g., text mismatch), detector should use
        full response entropy as fallback rather than crashing.
        """
        detector = IntrinsicUncertaintyDetector(sample_config)
        
        # Create claim with text that doesn't match metadata
        mismatched_claim = Claim(
            claim_id='test_c5',
            answer_id='test_a5',
            text='This text does not appear in metadata',
            answer_char_span=[0, 38],
            extraction_method='test'
        )
        
        metadata = {
            'text': 'Completely different text here.',
            'tokens': ['Completely', '▁different', '▁text', '▁here', '.'],
            'logits': [np.random.randn(100) for _ in range(5)]
        }
        
        # Should not crash, should use fallback
        signal = detector.compute_signal(mismatched_claim, sample_evidence, metadata)
        
        assert 'mean_entropy' in signal
        assert isinstance(signal['mean_entropy'], float)
        assert signal['mean_entropy'] >= 0.0  # Valid fallback entropy
    
    def test_missing_logits_in_metadata(self, sample_config, sample_claim, sample_evidence):
        """
        Test edge case: metadata missing logits field.
        
        Should handle gracefully and return 0.0 entropy.
        """
        detector = IntrinsicUncertaintyDetector(sample_config)
        
        metadata_no_logits = {
            'text': 'Some text',
            'tokens': ['Some', 'text'],
            'logits': []  # Empty logits
        }
        
        signal = detector.compute_signal(sample_claim, sample_evidence, metadata_no_logits)
        
        assert signal == {'mean_entropy': 0.0}
    
    def test_entropy_decreases_with_confidence(self, sample_config):
        """
        Test that entropy decreases as distribution becomes more peaked.
        
        This validates the entropy calculation is working correctly.
        """
        detector = IntrinsicUncertaintyDetector(sample_config)
        
        vocab_size = 100
        
        # Distribution 1: Uniform (maximum entropy)
        uniform = np.ones(vocab_size)
        entropy_uniform = detector._calculate_entropy(uniform, detector.epsilon)
        
        # Distribution 2: Slightly peaked
        slightly_peaked = np.ones(vocab_size)
        slightly_peaked[0] = 2.0
        entropy_slightly = detector._calculate_entropy(slightly_peaked, detector.epsilon)
        
        # Distribution 3: Very peaked (minimum entropy)
        very_peaked = np.ones(vocab_size) * -10.0
        very_peaked[0] = 10.0
        entropy_peaked = detector._calculate_entropy(very_peaked, detector.epsilon)
        
        # Entropy should decrease as distribution becomes more peaked
        assert entropy_uniform > entropy_slightly
        assert entropy_slightly > entropy_peaked
        assert entropy_peaked >= 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
