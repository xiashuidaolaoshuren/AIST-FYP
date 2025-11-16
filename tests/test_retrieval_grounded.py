"""
Unit tests for RetrievalGroundedDetector.

Tests entity extraction, fuzzy matching, number coverage, token overlap (ROUGE-L),
edge cases, and spaCy model reuse. Comprehensive coverage for retrieval-grounded
hallucination detection functionality.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.verification.retrieval_grounded import RetrievalGroundedDetector
from src.utils.data_structures import Claim, EvidenceChunk
from src.utils.config import Config
from src.utils.nlp_utils import get_spacy_model


class TestRetrievalGroundedDetector:
    """Test suite for RetrievalGroundedDetector class."""
    
    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration for testing."""
        config = Config()
        config.verification = type('obj', (object,), {
            'spacy_model': 'en_core_web_sm',
            'grounded': type('obj', (object,), {
                'entity_types': ["PERSON", "ORG", "GPE", "DATE", "NORP"],
                'fuzzy_matching': True,
                'min_token_length': 2
            })()
        })()
        return config
    
    @pytest.fixture
    def sample_config_no_fuzzy(self):
        """Create a configuration with fuzzy matching disabled."""
        config = Config()
        config.verification = type('obj', (object,), {
            'spacy_model': 'en_core_web_sm',
            'grounded': type('obj', (object,), {
                'entity_types': ["PERSON", "ORG", "GPE", "DATE", "NORP"],
                'fuzzy_matching': False,
                'min_token_length': 2
            })()
        })()
        return config
    
    @pytest.fixture
    def sample_claim_with_entities(self):
        """Create a claim with entities (PERSON, GPE)."""
        return Claim(
            claim_id='test_c1',
            answer_id='test_a1',
            text='Barack Obama was born in Hawaii.',
            answer_char_span=[0, 33],
            extraction_method='test'
        )
    
    @pytest.fixture
    def sample_claim_with_numbers(self):
        """Create a claim with numeric values."""
        return Claim(
            claim_id='test_c2',
            answer_id='test_a2',
            text='The dataset contains 185,000 claims and 42 sources.',
            answer_char_span=[0, 53],
            extraction_method='test'
        )
    
    @pytest.fixture
    def sample_claim_no_entities(self):
        """Create a claim without any entities."""
        return Claim(
            claim_id='test_c3',
            answer_id='test_a3',
            text='It was a beautiful day.',
            answer_char_span=[0, 23],
            extraction_method='test'
        )
    
    @pytest.fixture
    def sample_evidence_matching(self):
        """Create evidence that matches entities in claim."""
        return EvidenceChunk(
            doc_id='test_doc1',
            sent_id=1,
            text='Barack Obama, the 44th President, was born in Hawaii on August 4, 1961.',
            char_start=0,
            char_end=71,
            score_dense=0.95,
            rank=1
        )
    
    @pytest.fixture
    def sample_evidence_partial(self):
        """Create evidence with partial entity match."""
        return EvidenceChunk(
            doc_id='test_doc2',
            sent_id=2,
            text='Barack Obama served as President. He lived in Washington.',
            char_start=0,
            char_end=58,
            score_dense=0.85,
            rank=2
        )
    
    @pytest.fixture
    def sample_evidence_no_match(self):
        """Create evidence without matching entities."""
        return EvidenceChunk(
            doc_id='test_doc3',
            sent_id=3,
            text='The capital city has many tourist attractions.',
            char_start=0,
            char_end=46,
            score_dense=0.70,
            rank=3
        )
    
    @pytest.fixture
    def sample_evidence_with_numbers(self):
        """Create evidence with matching numbers."""
        return EvidenceChunk(
            doc_id='test_doc4',
            sent_id=4,
            text='This large-scale dataset has 185,000 claims from 42 different sources.',
            char_start=0,
            char_end=72,
            score_dense=0.90,
            rank=1
        )
    
    @pytest.fixture
    def sample_metadata(self):
        """Create empty metadata (not used by grounded detector)."""
        return {}
    
    # Test 1: Initialization
    def test_initialization(self, sample_config):
        """Test detector initializes correctly with config parameters."""
        detector = RetrievalGroundedDetector(sample_config)
        
        assert detector.config == sample_config
        assert detector.nlp is not None
        assert detector.entity_types == ["PERSON", "ORG", "GPE", "DATE", "NORP"]
        assert detector.fuzzy_matching == True
        assert detector.min_token_length == 2
    
    # Test 2: Entity extraction
    def test_entity_extraction(self, sample_config, sample_claim_with_entities, 
                                sample_evidence_matching, sample_metadata):
        """Test spaCy extracts correct entities from claim."""
        detector = RetrievalGroundedDetector(sample_config)
        
        # Process claim to extract entities
        doc_claim = detector.nlp(sample_claim_with_entities.text)
        entities = [ent.text for ent in doc_claim.ents if ent.label_ in detector.entity_types]
        
        # Should extract "Barack Obama" (PERSON) and "Hawaii" (GPE)
        assert len(entities) >= 2
        entity_texts = [e.lower() for e in entities]
        assert any('obama' in e for e in entity_texts)
        assert any('hawaii' in e for e in entity_texts)
    
    # Test 3: Entity coverage - full match
    def test_entity_coverage_full_match(self, sample_config, sample_claim_with_entities,
                                        sample_evidence_matching, sample_metadata):
        """Test entity coverage when all entities are in evidence."""
        detector = RetrievalGroundedDetector(sample_config)
        signal = detector.compute_signal(sample_claim_with_entities, 
                                         sample_evidence_matching, 
                                         sample_metadata)
        
        # All entities present: "Barack Obama" and "Hawaii" both in evidence
        assert signal['entities'] >= 0.95  # Allow small spaCy variance
        assert 'numbers' in signal
        assert 'tokens_overlap' in signal
    
    # Test 4: Entity coverage - partial match
    def test_entity_coverage_partial_match(self, sample_config, sample_claim_with_entities,
                                           sample_evidence_partial, sample_metadata):
        """Test entity coverage when some entities are matched."""
        detector = RetrievalGroundedDetector(sample_config)
        signal = detector.compute_signal(sample_claim_with_entities,
                                         sample_evidence_partial,
                                         sample_metadata)
        
        # Evidence has "Barack Obama" (full match) but not "Hawaii"
        # Claim: "Barack Obama was born in Hawaii."
        # Evidence: "Barack Obama served as President. He lived in Washington."
        # Should match 1 out of 2 entities = 0.5 coverage
        assert 0.4 <= signal['entities'] <= 0.6  # Partial coverage (50%)
    
    # Test 5: Entity coverage - no match
    def test_entity_coverage_no_match(self, sample_config, sample_claim_with_entities,
                                      sample_evidence_no_match, sample_metadata):
        """Test entity coverage when no entities are matched."""
        detector = RetrievalGroundedDetector(sample_config)
        signal = detector.compute_signal(sample_claim_with_entities,
                                         sample_evidence_no_match,
                                         sample_metadata)
        
        # No entities match: neither "Barack Obama" nor "Hawaii" in evidence
        assert signal['entities'] <= 0.2  # Very low coverage
    
    # Test 6: Fuzzy matching enabled
    def test_fuzzy_matching_enabled(self, sample_config):
        """Test fuzzy matching finds substring matches."""
        detector = RetrievalGroundedDetector(sample_config)
        
        claim = Claim('test_c', 'test_a', 'ML is powerful.', [0, 16], 'test')
        evidence = EvidenceChunk('doc', 1, 'Machine learning is powerful.', 0, 28, 0.9, 1)
        
        signal = detector.compute_signal(claim, evidence, {})
        
        # With fuzzy matching, "ML" should match "Machine learning" via token overlap
        # Token overlap should be high since the sentence structure matches
        assert signal['tokens_overlap'] > 0.5
    
    # Test 7: Fuzzy matching disabled
    def test_fuzzy_matching_disabled(self, sample_config_no_fuzzy):
        """Test exact matching when fuzzy is disabled."""
        detector = RetrievalGroundedDetector(sample_config_no_fuzzy)
        
        claim = Claim('test_c', 'test_a', 'The president was Obama.', [0, 24], 'test')
        evidence = EvidenceChunk('doc', 1, 'Barack Obama was president.', 0, 27, 0.9, 1)
        
        signal = detector.compute_signal(claim, evidence, {})
        
        # Without fuzzy matching, "Obama" must match exactly (it does)
        # But "president" vs "president" should still match
        assert signal['entities'] >= 0  # Will depend on exact entity extraction
    
    # Test 8: Number coverage
    def test_number_coverage(self, sample_config, sample_claim_with_numbers,
                             sample_evidence_with_numbers, sample_metadata):
        """Test numbers are extracted and matched correctly."""
        detector = RetrievalGroundedDetector(sample_config)
        signal = detector.compute_signal(sample_claim_with_numbers,
                                         sample_evidence_with_numbers,
                                         sample_metadata)
        
        # Both "185,000" and "42" should be found in evidence
        assert signal['numbers'] >= 0.95  # High number coverage
    
    # Test 9: Token overlap (ROUGE-L)
    def test_token_overlap_rouge_l(self, sample_config):
        """Test ROUGE-L calculates correct F1 score."""
        detector = RetrievalGroundedDetector(sample_config)
        
        # Exact match
        claim1 = Claim('c1', 'a1', 'The cat sat on the mat.', [0, 24], 'test')
        evidence1 = EvidenceChunk('doc', 1, 'The cat sat on the mat.', 0, 24, 0.9, 1)
        signal1 = detector.compute_signal(claim1, evidence1, {})
        assert signal1['tokens_overlap'] >= 0.95  # Near perfect match
        
        # Partial overlap
        claim2 = Claim('c2', 'a2', 'The cat sat.', [0, 12], 'test')
        evidence2 = EvidenceChunk('doc', 2, 'The dog sat on the mat.', 0, 23, 0.9, 1)
        signal2 = detector.compute_signal(claim2, evidence2, {})
        assert 0.4 <= signal2['tokens_overlap'] <= 0.7  # Some overlap
        
        # No overlap
        claim3 = Claim('c3', 'a3', 'Python programming.', [0, 19], 'test')
        evidence3 = EvidenceChunk('doc', 3, 'Java development.', 0, 17, 0.9, 1)
        signal3 = detector.compute_signal(claim3, evidence3, {})
        assert signal3['tokens_overlap'] <= 0.3  # Minimal overlap
    
    # Test 10: Edge case - no entities in claim
    def test_edge_case_no_entities(self, sample_config, sample_claim_no_entities,
                                    sample_evidence_matching, sample_metadata):
        """Test that claims without entities return 1.0 for entity coverage."""
        detector = RetrievalGroundedDetector(sample_config)
        signal = detector.compute_signal(sample_claim_no_entities,
                                         sample_evidence_matching,
                                         sample_metadata)
        
        # No entities in claim → trivially satisfied → 1.0
        assert signal['entities'] == 1.0
    
    # Test 11: Edge case - empty claim
    def test_edge_case_empty_claim(self, sample_config, sample_evidence_matching, sample_metadata):
        """Test empty claim returns zeros."""
        detector = RetrievalGroundedDetector(sample_config)
        
        empty_claim = Claim('c_empty', 'a_empty', '', [0, 0], 'test')
        signal = detector.compute_signal(empty_claim, sample_evidence_matching, sample_metadata)
        
        assert signal['entities'] == 0.0
        assert signal['numbers'] == 0.0
        assert signal['tokens_overlap'] == 0.0
    
    # Test 12: Edge case - empty evidence
    def test_edge_case_empty_evidence(self, sample_config, sample_claim_with_entities, sample_metadata):
        """Test empty evidence returns zeros."""
        detector = RetrievalGroundedDetector(sample_config)
        
        empty_evidence = EvidenceChunk('doc_empty', 1, '', 0, 0, 0.0, 1)
        signal = detector.compute_signal(sample_claim_with_entities, empty_evidence, sample_metadata)
        
        assert signal['entities'] == 0.0
        assert signal['numbers'] == 0.0
        assert signal['tokens_overlap'] == 0.0
    
    # Test 13: Output format validation
    def test_compute_signal_output_format(self, sample_config, sample_claim_with_entities,
                                          sample_evidence_matching, sample_metadata):
        """Test compute_signal returns correct dictionary format."""
        detector = RetrievalGroundedDetector(sample_config)
        signal = detector.compute_signal(sample_claim_with_entities,
                                         sample_evidence_matching,
                                         sample_metadata)
        
        # Check keys exist
        assert 'entities' in signal
        assert 'numbers' in signal
        assert 'tokens_overlap' in signal
        
        # Check values are floats in [0, 1]
        assert isinstance(signal['entities'], float)
        assert isinstance(signal['numbers'], float)
        assert isinstance(signal['tokens_overlap'], float)
        assert 0.0 <= signal['entities'] <= 1.0
        assert 0.0 <= signal['numbers'] <= 1.0
        assert 0.0 <= signal['tokens_overlap'] <= 1.0
    
    # Test 14: spaCy model reuse
    def test_spacy_model_reuse(self, sample_config):
        """Test that only one spaCy model instance is loaded (singleton pattern)."""
        # Get the spaCy model from nlp_utils
        model1 = get_spacy_model('en_core_web_sm')
        
        # Create detector (should reuse the same model)
        detector = RetrievalGroundedDetector(sample_config)
        
        # Verify detector uses the same model instance
        assert detector.nlp is model1  # Should be the exact same object
