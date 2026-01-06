"""
Unit tests for NLIDetector (Task 3).

Tests the zero-shot NLI detector that classifies claim-evidence relationships
as entailment, neutral, or contradiction.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.verification.nli_detector import NLIDetector
from src.utils.config import Config


class TestNLIDetector:
    """Test suite for NLI Detector class."""
    
    @pytest.fixture
    def sample_config(self):
        """Create a test configuration with NLI settings."""
        config = Config()
        config.verification = type('obj', (object,), {
            'enabled': True,
            'nli': type('obj', (object,), {
                'model_name': 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli',
                'device': 'cpu',  # Use CPU for tests to avoid CUDA issues
                'batch_size': 1
            })()
        })()
        return config
    
    @pytest.fixture
    def detector(self, sample_config):
        """Create an NLIDetector instance for testing."""
        return NLIDetector(sample_config)
    
    def test_initialization(self, detector):
        """Test that NLI detector initializes correctly."""
        assert detector is not None
        assert detector.model is not None
        assert detector.tokenizer is not None
        assert detector.device is not None
        assert detector.label_mapping is not None
        
        # Verify label mapping has all three categories
        assert 'entailment' in detector.label_mapping
        assert 'neutral' in detector.label_mapping
        assert 'contradiction' in detector.label_mapping
        assert None not in detector.label_mapping.values()
    
    def test_detect_entailment(self, detector):
        """Test detection of clear entailment case."""
        evidence = "Albert Einstein won the Nobel Prize in Physics in 1921."
        claim = "Einstein received a Nobel Prize."
        
        scores = detector.detect(claim, evidence)
        
        # Verify output structure
        assert isinstance(scores, dict)
        assert 'entailment' in scores
        assert 'neutral' in scores
        assert 'contradiction' in scores
        
        # Verify probabilities
        assert all(0 <= scores[k] <= 1 for k in scores)
        assert abs(sum(scores.values()) - 1.0) < 0.01  # Should sum to ~1.0
        
        # Entailment should be highest
        assert scores['entailment'] > scores['neutral']
        assert scores['entailment'] > scores['contradiction']
    
    def test_detect_contradiction(self, detector):
        """Test detection of clear contradiction case."""
        evidence = "Albert Einstein won the Nobel Prize in Physics in 1921."
        claim = "Einstein never won any Nobel Prize."
        
        scores = detector.detect(claim, evidence)
        
        # Verify output structure
        assert isinstance(scores, dict)
        assert len(scores) == 3
        
        # Verify probabilities
        assert all(0 <= scores[k] <= 1 for k in scores)
        assert abs(sum(scores.values()) - 1.0) < 0.01
        
        # Contradiction should be highest
        assert scores['contradiction'] > scores['entailment']
        assert scores['contradiction'] > scores['neutral']
    
    def test_detect_neutral(self, detector):
        """Test detection of neutral (unrelated) case."""
        evidence = "Albert Einstein won the Nobel Prize in Physics in 1921."
        claim = "The weather today is sunny."
        
        scores = detector.detect(claim, evidence)
        
        # Verify output structure
        assert isinstance(scores, dict)
        assert len(scores) == 3
        
        # Verify probabilities
        assert all(0 <= scores[k] <= 1 for k in scores)
        assert abs(sum(scores.values()) - 1.0) < 0.01
        
        # Neutral should be highest (or at least not contradiction/entailment)
        assert scores['neutral'] > min(scores['entailment'], scores['contradiction'])
    
    def test_detect_empty_claim(self, detector):
        """Test that empty claim raises ValueError."""
        evidence = "Albert Einstein won the Nobel Prize."
        claim = ""
        
        with pytest.raises(ValueError, match="claim_text cannot be empty"):
            detector.detect(claim, evidence)
    
    def test_detect_empty_evidence(self, detector):
        """Test that empty evidence raises ValueError."""
        evidence = ""
        claim = "Einstein won a Nobel Prize."
        
        with pytest.raises(ValueError, match="evidence_text cannot be empty"):
            detector.detect(claim, evidence)
    
    def test_detect_whitespace_only(self, detector):
        """Test that whitespace-only strings are treated as empty."""
        evidence = "Albert Einstein won the Nobel Prize."
        claim = "   "  # Only whitespace
        
        with pytest.raises(ValueError):
            detector.detect(claim, evidence)
    
    def test_detect_long_texts(self, detector):
        """Test detection with long texts (truncation handling)."""
        # Create long texts that exceed typical model max length
        evidence = "Albert Einstein won the Nobel Prize in Physics in 1921. " * 100
        claim = "Einstein received a Nobel Prize in the early 20th century."
        
        # Should handle gracefully with truncation
        scores = detector.detect(claim, evidence)
        
        assert isinstance(scores, dict)
        assert len(scores) == 3
        assert all(0 <= scores[k] <= 1 for k in scores)
        assert abs(sum(scores.values()) - 1.0) < 0.01
    
    def test_detect_special_characters(self, detector):
        """Test detection with special characters and punctuation."""
        evidence = "Einstein (1879-1955) won the Nobel Prize in Physics!"
        claim = "Einstein won a Nobel Prize."
        
        scores = detector.detect(claim, evidence)
        
        assert isinstance(scores, dict)
        assert len(scores) == 3
        # Should still detect entailment despite special characters
        assert scores['entailment'] > scores['contradiction']
    
    def test_probabilities_sum_to_one(self, detector):
        """Test that probabilities always sum to approximately 1.0."""
        test_cases = [
            ("Einstein won a Nobel Prize.", "Albert Einstein received the Nobel Prize in Physics."),
            ("The sky is blue.", "Grass is green."),
            ("Dogs are mammals.", "Dogs are not mammals."),
        ]
        
        for claim, evidence in test_cases:
            scores = detector.detect(claim, evidence)
            total = sum(scores.values())
            assert abs(total - 1.0) < 0.01, f"Probabilities sum to {total}, expected ~1.0"
    
    def test_detect_batch_single_pair(self, detector):
        """Test batch detection with a single pair."""
        claims = ["Einstein won a Nobel Prize."]
        evidence_list = ["Albert Einstein received the Nobel Prize in Physics in 1921."]
        
        results = detector.detect_batch(claims, evidence_list)
        
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], dict)
        assert 'entailment' in results[0]
    
    def test_detect_batch_multiple_pairs(self, detector):
        """Test batch detection with multiple pairs."""
        claims = [
            "Einstein won a Nobel Prize.",
            "The Earth is flat.",
            "Paris is the capital of France."
        ]
        evidence_list = [
            "Albert Einstein received the Nobel Prize in Physics in 1921.",
            "The Earth is approximately spherical in shape.",
            "Paris is the capital and largest city of France."
        ]
        
        results = detector.detect_batch(claims, evidence_list)
        
        assert isinstance(results, list)
        assert len(results) == 3
        
        # First should be entailment
        assert results[0]['entailment'] > results[0]['contradiction']
        
        # Second should be contradiction
        assert results[1]['contradiction'] > results[1]['entailment']
        
        # Third should be entailment
        assert results[2]['entailment'] > results[2]['contradiction']
    
    def test_detect_batch_length_mismatch(self, detector):
        """Test that batch detection fails with mismatched lengths."""
        claims = ["Claim 1", "Claim 2"]
        evidence_list = ["Evidence 1"]  # Length mismatch
        
        with pytest.raises(ValueError, match="Length mismatch"):
            detector.detect_batch(claims, evidence_list)
    
    def test_detect_batch_empty_lists(self, detector):
        """Test that batch detection fails with empty lists."""
        claims = []
        evidence_list = []
        
        with pytest.raises(ValueError, match="Empty input lists"):
            detector.detect_batch(claims, evidence_list)
    
    def test_label_mapping_correctness(self, detector):
        """Test that label mapping is correctly loaded from model config."""
        # All indices should be valid (0, 1, or 2 for 3-class NLI)
        for label, idx in detector.label_mapping.items():
            assert isinstance(idx, int)
            assert 0 <= idx <= 2
        
        # All indices should be unique
        indices = list(detector.label_mapping.values())
        assert len(indices) == len(set(indices)), "Duplicate indices in label mapping"
