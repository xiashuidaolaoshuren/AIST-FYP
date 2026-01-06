"""
Comprehensive Integration Tests for Month 4 Verifier Module.

Tests full integration of all Month 4 components:
- VerifierHub with all 4 detectors (Tasks 1-4)
- NLIDetector (Task 3)
- SelfAgreementDetector (Task 4)
- Multi-evidence verification with aggregation (Task 2)
- End-to-end verification pipeline
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.verification.verifier_hub import VerifierHub
from src.utils.data_structures import Claim, EvidenceChunk, VerifierSignal
from src.utils.config import Config


class TestMonth4Integration:
    """Comprehensive integration tests for Month 4 verifier features."""
    
    @pytest.fixture
    def full_config(self):
        """Create complete configuration with all Month 4 features enabled."""
        config = Config()
        config.verification = type('obj', (object,), {
            'enabled': True,
            'verify_all_evidence': True,
            'aggregation_method': 'mean',
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
            })(),
            'nli': type('obj', (object,), {
                'model_name': 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli',
                'device': 'cpu',
                'batch_size': 1
            })(),
            'self_agreement': type('obj', (object,), {
                'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                'k_samples': 3,  # Use 3 for faster tests
                'temperature': 1.5,
                'device': 'cpu',
                'aggregation_method': 'inherit'
            })()
        })()
        return config
    
    @pytest.fixture
    def mock_generator(self):
        """Mock GeneratorWrapper for self-agreement testing."""
        generator = Mock()
        
        # Mock to return similar but slightly different responses
        responses = [
            "Machine learning is a subset of artificial intelligence.",
            "ML is a part of AI technology.",
            "Machine learning belongs to the AI field."
        ]
        response_idx = [0]
        
        def generate_side_effect(prompt, evidence_chunks=None, **kwargs):
            idx = response_idx[0] % len(responses)
            response_idx[0] += 1
            return {
                'text': responses[idx],
                'tokens': responses[idx].split(),
                'token_ids': [],
                'logits': [],
                'scores': [],
                'evidence_used': []
            }
        
        generator.generate_with_metadata = Mock(side_effect=generate_side_effect)
        return generator
    
    @pytest.fixture
    def sample_claim(self):
        """Sample claim for testing."""
        return Claim(
            claim_id='m4_c1',
            answer_id='m4_a1',
            text='Machine learning is a subset of artificial intelligence.',
            answer_char_span=[0, 57],
            extraction_method='test'
        )
    
    @pytest.fixture
    def sample_evidence_single(self):
        """Single evidence chunk."""
        return EvidenceChunk(
            doc_id='doc1',
            sent_id=0,
            text='Machine learning (ML) is a subset of artificial intelligence (AI) that focuses on data.',
            char_start=0,
            char_end=88,
            score_dense=0.95,
            rank=0
        )
    
    @pytest.fixture
    def sample_evidence_multiple(self):
        """Multiple evidence chunks."""
        return [
            EvidenceChunk(
                doc_id='doc1',
                sent_id=0,
                text='Machine learning (ML) is a subset of artificial intelligence (AI).',
                char_start=0,
                char_end=67,
                score_dense=0.95,
                rank=0
            ),
            EvidenceChunk(
                doc_id='doc1',
                sent_id=1,
                text='AI encompasses machine learning, deep learning, and neural networks.',
                char_start=68,
                char_end=136,
                score_dense=0.90,
                rank=1
            ),
            EvidenceChunk(
                doc_id='doc2',
                sent_id=0,
                text='Artificial intelligence is a broad field in computer science.',
                char_start=0,
                char_end=61,
                score_dense=0.85,
                rank=2
            )
        ]
    
    @pytest.fixture
    def sample_metadata(self):
        """Sample generation metadata with query for self-agreement."""
        return {
            'text': 'Machine learning is a subset of artificial intelligence.',
            'tokens': ['Machine', 'learning', 'is', 'a', 'subset', 'of', 'artificial', 'intelligence', '.'],
            'logits': [[0.1] * 100 for _ in range(9)],
            'original_query': 'What is machine learning?'
        }
    
    def test_all_detectors_initialized(self, full_config, mock_generator):
        """Test all 4 detectors are initialized correctly."""
        hub = VerifierHub(full_config, mock_generator)
        
        assert hub.enabled is True
        assert hub.uncertainty_detector is not None  # Task 1 (Month 3)
        assert hub.grounded_detector is not None     # Task 1 (Month 3)
        assert hub.nli_detector is not None          # Task 3 (Month 4)
        assert hub.self_agreement_detector is not None  # Task 4 (Month 4)
        
        # Check detector status
        status = hub.get_detector_status()
        assert status['enabled'] is True
        assert status['intrinsic'] is True
        assert status['grounded'] is True
        assert status['nli'] is True
        assert status['self_agreement'] is True
    
    def test_single_chunk_all_signals(self, full_config, mock_generator, sample_claim,
                                     sample_evidence_single, sample_metadata):
        """Test all signals populated for single-chunk verification."""
        hub = VerifierHub(full_config, mock_generator)
        signal = hub.verify_claim(sample_claim, sample_evidence_single, sample_metadata)
        
        assert signal is not None
        assert signal.claim_id == 'm4_c1'
        
        # Check Month 3 signals
        assert signal.uncertainty is not None
        assert 'mean_entropy' in signal.uncertainty
        assert signal.coverage is not None
        assert all(k in signal.coverage for k in ['entities', 'numbers', 'tokens_overlap'])
        
        # Check Month 4 signals
        assert signal.nli is not None
        assert all(k in signal.nli for k in ['entailment', 'neutral', 'contradiction'])
        
        assert signal.consistency is not None
        # Should have either score or variance (or both)
        assert 'score' in signal.consistency or 'variance' in signal.consistency
    
    def test_multi_chunk_all_signals(self, full_config, mock_generator, sample_claim,
                                    sample_evidence_multiple, sample_metadata):
        """Test all signals populated for multi-chunk verification."""
        hub = VerifierHub(full_config, mock_generator)
        signal = hub.verify_claim(sample_claim, sample_evidence_multiple, sample_metadata)
        
        assert signal is not None
        
        # Check aggregated signals
        assert signal.uncertainty is not None
        assert signal.coverage is not None
        assert signal.nli is not None
        assert signal.consistency is not None
        
        # Check per-chunk signals
        assert signal.per_chunk_signals is not None
        assert len(signal.per_chunk_signals) == 3
        
        for chunk_signal in signal.per_chunk_signals:
            assert 'doc_id' in chunk_signal
            assert 'uncertainty' in chunk_signal
            assert 'coverage' in chunk_signal
            assert 'nli' in chunk_signal
    
    def test_nli_scores_valid(self, full_config, mock_generator, sample_claim,
                             sample_evidence_single, sample_metadata):
        """Test NLI scores are valid probabilities."""
        hub = VerifierHub(full_config, mock_generator)
        signal = hub.verify_claim(sample_claim, sample_evidence_single, sample_metadata)
        
        nli = signal.nli
        assert nli is not None
        
        # All three scores present
        assert 'entailment' in nli
        assert 'neutral' in nli
        assert 'contradiction' in nli
        
        # All are probabilities
        for score in nli.values():
            assert 0.0 <= score <= 1.0
        
        # Sum to ~1.0
        total = sum(nli.values())
        assert 0.95 <= total <= 1.05
    
    def test_self_agreement_executed(self, full_config, mock_generator, sample_claim,
                                    sample_evidence_single, sample_metadata):
        """Test self-agreement detector is executed."""
        hub = VerifierHub(full_config, mock_generator)
        signal = hub.verify_claim(sample_claim, sample_evidence_single, sample_metadata)
        
        # Generator should have been called k times (3 in test config)
        assert mock_generator.generate_with_metadata.call_count >= 3
        
        # Consistency should be populated
        assert signal.consistency is not None
        consistency = signal.consistency
        
        if consistency.get('score') is not None:
            assert 0.0 <= consistency['score'] <= 1.0
        
        if consistency.get('variance') is not None:
            assert consistency['variance'] >= 0.0
    
    def test_backward_compatibility_no_generator(self, full_config, sample_claim,
                                                sample_evidence_single, sample_metadata):
        """Test system works without generator (no self-agreement)."""
        hub = VerifierHub(full_config, generator=None)
        
        # Self-agreement should be disabled
        assert hub.self_agreement_detector is None
        
        # Other detectors should work
        signal = hub.verify_claim(sample_claim, sample_evidence_single, sample_metadata)
        assert signal is not None
        
        # Month 3 signals should be present
        assert signal.uncertainty is not None
        assert signal.coverage is not None
        
        # NLI should work (doesn't need generator)
        assert signal.nli is not None
        
        # Consistency should be None or empty
        assert signal.consistency is not None
        consistency_empty = (
            signal.consistency.get('score') is None and
            signal.consistency.get('variance') is None
        )
        assert consistency_empty
    
    def test_aggregation_max_method(self, full_config, mock_generator, sample_claim,
                                   sample_evidence_multiple, sample_metadata):
        """Test max aggregation method."""
        full_config.verification.aggregation_method = 'max'
        hub = VerifierHub(full_config, mock_generator)
        
        signal = hub.verify_claim(sample_claim, sample_evidence_multiple, sample_metadata)
        
        assert signal is not None
        assert signal.per_chunk_signals is not None
        
        # Aggregated scores should reflect max logic
        # (actual verification would require checking against per-chunk values)
        assert signal.coverage is not None
        assert signal.uncertainty is not None
        assert signal.nli is not None
    
    def test_aggregation_mean_method(self, full_config, mock_generator, sample_claim,
                                    sample_evidence_multiple, sample_metadata):
        """Test mean aggregation method."""
        full_config.verification.aggregation_method = 'mean'
        hub = VerifierHub(full_config, mock_generator)
        
        signal = hub.verify_claim(sample_claim, sample_evidence_multiple, sample_metadata)
        
        assert signal is not None
        assert signal.per_chunk_signals is not None
        
        # Aggregated scores should reflect mean logic
        assert signal.coverage is not None
        assert signal.uncertainty is not None
        assert signal.nli is not None
    
    def test_contradictory_claim_detected(self, full_config, mock_generator):
        """Test NLI detects contradictory claim."""
        hub = VerifierHub(full_config, mock_generator)
        
        claim = Claim(
            claim_id='bad_c1',
            answer_id='bad_a1',
            text='Machine learning is not related to artificial intelligence.',
            answer_char_span=[0, 59],
            extraction_method='test'
        )
        
        evidence = EvidenceChunk(
            doc_id='doc1',
            sent_id=0,
            text='Machine learning is a subset of artificial intelligence.',
            char_start=0,
            char_end=57,
            score_dense=0.95,
            rank=0
        )
        
        metadata = {
            'text': claim.text,
            'tokens': claim.text.split(),
            'logits': [[0.1] * 100 for _ in range(len(claim.text.split()))],
            'original_query': 'What is machine learning?'
        }
        
        signal = hub.verify_claim(claim, evidence, metadata)
        
        # Contradiction should be detected
        assert signal.nli['contradiction'] > signal.nli['entailment']
    
    def test_missing_query_graceful_handling(self, full_config, mock_generator, sample_claim,
                                            sample_evidence_single):
        """Test graceful handling when query missing."""
        # Metadata without original_query
        metadata_no_query = {
            'text': 'Test text',
            'tokens': ['Test'],
            'logits': []
        }
        
        hub = VerifierHub(full_config, mock_generator)
        signal = hub.verify_claim(sample_claim, sample_evidence_single, metadata_no_query)
        
        # Should not crash
        assert signal is not None
        
        # Other signals should still work
        assert signal.uncertainty is not None
        assert signal.coverage is not None
        assert signal.nli is not None
    
    def test_end_to_end_high_quality_claim(self, full_config, mock_generator):
        """Test high-quality supported claim."""
        hub = VerifierHub(full_config, mock_generator)
        
        claim = Claim(
            claim_id='good_c1',
            answer_id='good_a1',
            text='Einstein won the Nobel Prize in Physics in 1921.',
            answer_char_span=[0, 50],
            extraction_method='test'
        )
        
        evidence = EvidenceChunk(
            doc_id='einstein',
            sent_id=0,
            text='Albert Einstein was awarded the Nobel Prize in Physics in 1921 for his photoelectric effect work.',
            char_start=0,
            char_end=100,
            score_dense=0.98,
            rank=0
        )
        
        metadata = {
            'text': claim.text,
            'tokens': claim.text.split(),
            'logits': [[0.1] * 100 for _ in range(len(claim.text.split()))],
            'original_query': 'When did Einstein win the Nobel Prize?'
        }
        
        signal = hub.verify_claim(claim, evidence, metadata)
        
        # Should have high entailment
        assert signal.nli['entailment'] > 0.5
        
        # Should have good coverage
        assert signal.coverage['tokens_overlap'] > 0.3
    
    def test_verifier_signal_serialization(self, full_config, mock_generator, sample_claim,
                                          sample_evidence_single, sample_metadata):
        """Test VerifierSignal can be serialized to dict."""
        hub = VerifierHub(full_config, mock_generator)
        signal = hub.verify_claim(sample_claim, sample_evidence_single, sample_metadata)
        
        # Convert to dict
        signal_dict = signal.to_dict()
        
        # Check all fields present
        assert 'claim_id' in signal_dict
        assert 'doc_id' in signal_dict
        assert 'sent_id' in signal_dict
        assert 'nli' in signal_dict
        assert 'coverage' in signal_dict
        assert 'uncertainty' in signal_dict
        assert 'consistency' in signal_dict
        assert 'citation_span_match' in signal_dict
        assert 'numeric_check' in signal_dict
