"""
Unit tests for VerifierHub.

Tests the centralized orchestration of verification detectors.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.verification.verifier_hub import VerifierHub
from src.utils.data_structures import Claim, EvidenceChunk
from src.utils.config import Config


class TestVerifierHub:
    """Test suite for VerifierHub class."""
    
    @pytest.fixture
    def sample_config(self):
        """Create a test configuration with verification enabled."""
        config = Config()
        config.verification = type('obj', (object,), {
            'enabled': True,
            'spacy_model': 'en_core_web_sm',
            'intrinsic': type('obj', (object,), {
                'epsilon': 1e-10,
                'method': 'entropy'
            })(),
            'grounded': type('obj', (object,), {
                'entity_types': ["PERSON", "ORG", "GPE", "DATE", "NORP"],
                'fuzzy_matching': True,
                'min_token_length': 2,
                'rouge_method': 'rouge-l'
            })()
        })()
        return config
    
    @pytest.fixture
    def sample_config_disabled(self):
        """Create a config with verification disabled."""
        config = Config()
        config.verification = type('obj', (object,), {
            'enabled': False
        })()
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
        """Create sample generation metadata."""
        return {
            'text': 'Machine learning is a subset of AI.',
            'tokens': ['Machine', 'learning', 'is', 'a', 'subset', 'of', 'AI', '.'],
            'logits': [[0.1] * 100 for _ in range(8)]  # Mock logits
        }
    
    def test_initialization_enabled(self, sample_config):
        """Test VerifierHub initialization with verification enabled."""
        hub = VerifierHub(sample_config)
        
        assert hub.enabled is True
        assert hub.uncertainty_detector is not None
        assert hub.grounded_detector is not None
        assert hub.nli_detector is not None  # Implemented in Month 4
        assert hub.self_agreement_detector is None  # Not implemented yet (Month 4)
    
    def test_initialization_disabled(self, sample_config_disabled):
        """Test VerifierHub initialization with verification disabled."""
        hub = VerifierHub(sample_config_disabled)
        
        assert hub.enabled is False
    
    def test_is_enabled(self, sample_config, sample_config_disabled):
        """Test is_enabled() method."""
        hub_enabled = VerifierHub(sample_config)
        hub_disabled = VerifierHub(sample_config_disabled)
        
        assert hub_enabled.is_enabled() is True
        assert hub_disabled.is_enabled() is False
    
    def test_get_detector_status_enabled(self, sample_config):
        """Test get_detector_status() with enabled hub."""
        hub = VerifierHub(sample_config)
        status = hub.get_detector_status()
        
        assert status['enabled'] is True
        assert status['intrinsic'] is True
        assert status['grounded'] is True
        assert status['nli'] is True  # Implemented in Month 4
        assert status['self_agreement'] is False  # Not implemented yet (Month 4)
    
    def test_get_detector_status_disabled(self, sample_config_disabled):
        """Test get_detector_status() with disabled hub."""
        hub = VerifierHub(sample_config_disabled)
        status = hub.get_detector_status()
        
        assert status['enabled'] is False
        assert status['intrinsic'] is False
        assert status['grounded'] is False
    
    def test_verify_claim_success(self, sample_config, sample_claim, sample_evidence, sample_metadata):
        """Test successful claim verification."""
        hub = VerifierHub(sample_config)
        signal = hub.verify_claim(sample_claim, sample_evidence, sample_metadata)
        
        assert signal is not None
        assert signal.claim_id == 'test_c1'
        assert signal.doc_id == 'test_doc'
        assert signal.sent_id == 1
        
        # Check Month 3 signals are present
        assert signal.uncertainty is not None
        assert 'mean_entropy' in signal.uncertainty
        assert signal.coverage is not None
        assert 'entities' in signal.coverage
        assert 'numbers' in signal.coverage
        assert 'tokens_overlap' in signal.coverage
        
        # Check Month 4 NLI signal is present (implemented)
        assert signal.nli is not None
        assert 'entailment' in signal.nli
        assert 'neutral' in signal.nli
        assert 'contradiction' in signal.nli
        
        # Check self-agreement is None (not implemented yet)
        assert signal.consistency == {'variance': None}
    
    def test_verify_claim_disabled(self, sample_config_disabled, sample_claim, sample_evidence, sample_metadata):
        """Test verify_claim returns None when verification is disabled."""
        hub = VerifierHub(sample_config_disabled)
        signal = hub.verify_claim(sample_claim, sample_evidence, sample_metadata)
        
        assert signal is None
    
    def test_verify_claim_none_claim(self, sample_config, sample_evidence, sample_metadata):
        """Test verify_claim raises error with None claim."""
        hub = VerifierHub(sample_config)
        
        with pytest.raises(ValueError, match="claim cannot be None"):
            hub.verify_claim(None, sample_evidence, sample_metadata)
    
    def test_verify_claim_none_evidence(self, sample_config, sample_claim, sample_metadata):
        """Test verify_claim raises error with None evidence."""
        hub = VerifierHub(sample_config)
        
        with pytest.raises(ValueError, match="evidence cannot be None"):
            hub.verify_claim(sample_claim, None, sample_metadata)
    
    def test_verify_claim_none_metadata(self, sample_config, sample_claim, sample_evidence):
        """Test verify_claim handles None metadata gracefully."""
        hub = VerifierHub(sample_config)
        
        # Should log warning but not crash
        signal = hub.verify_claim(sample_claim, sample_evidence, None)
        
        # Signal should still be created (with fallback values)
        assert signal is not None
