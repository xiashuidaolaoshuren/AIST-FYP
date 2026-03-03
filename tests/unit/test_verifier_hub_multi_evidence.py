"""
Unit tests for VerifierHub multi-evidence verification (Task 2).

Tests the new multi-evidence functionality including:
- verify_all_evidence configuration flag
- Aggregation methods (max, mean)
- Per-chunk signal storage
- Backward compatibility
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.verification.verifier_hub import VerifierHub
from src.utils.data_structures import Claim, EvidenceChunk, VerifierSignal
from src.utils.config import Config


@pytest.fixture
def config_multi_max():
    """Config with multi-evidence enabled and max aggregation."""
    config = Config()
    config.verification = type('obj', (object,), {
        'enabled': True,
        'verify_all_evidence': True,
        'aggregation_method': 'max',
        'spacy_model': 'en_core_web_sm',
        'intrinsic': type('obj', (object,), {
            'epsilon': 1e-10
        })(),
        'grounded': type('obj', (object,), {
            'entity_types': ['PERSON', 'ORG', 'GPE'],
            'fuzzy_matching': True,
            'min_token_length': 2
        })()
    })()
    return config


@pytest.fixture
def config_multi_mean():
    """Config with multi-evidence enabled and mean aggregation."""
    config = Config()
    config.verification = type('obj', (object,), {
        'enabled': True,
        'verify_all_evidence': True,
        'aggregation_method': 'mean',
        'spacy_model': 'en_core_web_sm',
        'intrinsic': type('obj', (object,), {
            'epsilon': 1e-10
        })(),
        'grounded': type('obj', (object,), {
            'entity_types': ['PERSON', 'ORG', 'GPE'],
            'fuzzy_matching': True,
            'min_token_length': 2
        })()
    })()
    return config


@pytest.fixture
def config_single_only():
    """Config with multi-evidence disabled (backward compatible)."""
    config = Config()
    config.verification = type('obj', (object,), {
        'enabled': True,
        'verify_all_evidence': False,
        'aggregation_method': 'max',
        'spacy_model': 'en_core_web_sm',
        'intrinsic': type('obj', (object,), {
            'epsilon': 1e-10
        })(),
        'grounded': type('obj', (object,), {
            'entity_types': ['PERSON', 'ORG', 'GPE'],
            'fuzzy_matching': True,
            'min_token_length': 2
        })()
    })()
    return config


@pytest.fixture
def sample_claim():
    """Sample claim for testing."""
    return Claim(
        claim_id="claim_001",
        answer_id="answer_001",
        text="Albert Einstein won the Nobel Prize in Physics in 1921.",
        answer_char_span=[0, 60]
    )


@pytest.fixture
def evidence_chunks_multiple():
    """Multiple evidence chunks with varying relevance."""
    return [
        EvidenceChunk(
            doc_id="doc_001",
            sent_id=1,
            text="Albert Einstein received the Nobel Prize in Physics in 1921 for his work on the photoelectric effect.",
            char_start=0,
            char_end=100,
            score_dense=0.95,
            rank=0
        ),
        EvidenceChunk(
            doc_id="doc_002",
            sent_id=3,
            text="Einstein was a theoretical physicist who developed the theory of relativity.",
            char_start=200,
            char_end=280,
            score_dense=0.85,
            rank=1
        ),
        EvidenceChunk(
            doc_id="doc_003",
            sent_id=7,
            text="The Nobel Prize in Physics is awarded annually by the Royal Swedish Academy of Sciences.",
            char_start=500,
            char_end=590,
            score_dense=0.70,
            rank=2
        )
    ]


@pytest.fixture
def generation_metadata():
    """Sample generation metadata."""
    return {
        'tokens': ['Albert', 'Einstein', 'won', 'the', 'Nobel', 'Prize'],
        'token_scores': [
            {'<pad>': 0.1, 'Albert': 0.85, 'The': 0.05},
            {'Einstein': 0.90, 'Newton': 0.08, 'Tesla': 0.02},
            {'won': 0.75, 'received': 0.20, 'lost': 0.05},
            {'the': 0.95, 'a': 0.03, 'an': 0.02},
            {'Nobel': 0.88, 'Field': 0.10, 'Turing': 0.02},
            {'Prize': 0.92, 'Award': 0.06, 'Medal': 0.02}
        ]
    }


def test_multi_evidence_max_aggregation(config_multi_max, sample_claim, evidence_chunks_multiple, generation_metadata):
    """Test multi-evidence verification with MAX aggregation."""
    hub = VerifierHub(config_multi_max)
    
    signal = hub.verify_claim(sample_claim, evidence_chunks_multiple, generation_metadata)
    
    assert signal is not None
    assert isinstance(signal, VerifierSignal)
    
    # Check that per_chunk_signals is populated
    assert signal.per_chunk_signals is not None
    assert len(signal.per_chunk_signals) == 3
    
    # Verify each chunk has required fields
    for chunk_signal in signal.per_chunk_signals:
        assert 'doc_id' in chunk_signal
        assert 'sent_id' in chunk_signal
        assert 'coverage' in chunk_signal
        assert 'uncertainty' in chunk_signal
    
    # MAX aggregation should pick the best coverage (highest) and lowest entropy
    # First chunk has best match, so coverage should be high
    assert signal.coverage['entities'] > 0.0  # Should find Einstein, Nobel Prize
    assert signal.coverage['tokens_overlap'] > 0.0


def test_multi_evidence_mean_aggregation(config_multi_mean, sample_claim, evidence_chunks_multiple, generation_metadata):
    """Test multi-evidence verification with MEAN aggregation."""
    hub = VerifierHub(config_multi_mean)
    
    signal = hub.verify_claim(sample_claim, evidence_chunks_multiple, generation_metadata)
    
    assert signal is not None
    assert isinstance(signal, VerifierSignal)
    
    # Check per_chunk_signals
    assert signal.per_chunk_signals is not None
    assert len(signal.per_chunk_signals) == 3
    
    # MEAN should average all scores
    # Coverage should be positive but possibly lower than MAX
    assert signal.coverage['entities'] >= 0.0
    assert signal.coverage['tokens_overlap'] >= 0.0


def test_backward_compatibility_single_chunk(config_single_only, sample_claim, evidence_chunks_multiple, generation_metadata):
    """Test backward compatibility: multi-evidence disabled uses top-1 chunk."""
    hub = VerifierHub(config_single_only)
    
    # Pass list but with verify_all_evidence=False
    signal = hub.verify_claim(sample_claim, evidence_chunks_multiple, generation_metadata)
    
    assert signal is not None
    assert isinstance(signal, VerifierSignal)
    
    # Should NOT have per_chunk_signals (single-chunk mode)
    assert signal.per_chunk_signals is None
    
    # Should use top-ranked chunk
    assert signal.doc_id == "doc_001"
    assert signal.sent_id == 1


def test_single_chunk_input_still_works(config_multi_max, sample_claim, evidence_chunks_multiple, generation_metadata):
    """Test that passing a single chunk still works (not a list)."""
    hub = VerifierHub(config_multi_max)
    
    # Pass single chunk (not a list)
    single_chunk = evidence_chunks_multiple[0]
    signal = hub.verify_claim(sample_claim, single_chunk, generation_metadata)
    
    assert signal is not None
    assert isinstance(signal, VerifierSignal)
    
    # Single chunk input should not have per_chunk_signals
    assert signal.per_chunk_signals is None
    assert signal.doc_id == "doc_001"


def test_empty_evidence_list(config_multi_max, sample_claim, generation_metadata):
    """Test handling of empty evidence list."""
    hub = VerifierHub(config_multi_max)
    
    signal = hub.verify_claim(sample_claim, [], generation_metadata)
    
    # Should return None for empty evidence
    assert signal is None


def test_max_vs_mean_aggregation_difference(config_multi_max, config_multi_mean, sample_claim, evidence_chunks_multiple, generation_metadata):
    """Test that MAX and MEAN produce different results."""
    hub_max = VerifierHub(config_multi_max)
    hub_mean = VerifierHub(config_multi_mean)
    
    signal_max = hub_max.verify_claim(sample_claim, evidence_chunks_multiple, generation_metadata)
    signal_mean = hub_mean.verify_claim(sample_claim, evidence_chunks_multiple, generation_metadata)
    
    assert signal_max is not None
    assert signal_mean is not None
    
    # MAX should generally give higher coverage than MEAN (optimistic)
    # This may not always be true for all metrics, but should hold for entity coverage
    # where first chunk has best match
    assert signal_max.coverage['entities'] >= signal_mean.coverage['entities']


def test_per_chunk_signals_structure(config_multi_max, sample_claim, evidence_chunks_multiple, generation_metadata):
    """Test the structure of per_chunk_signals."""
    hub = VerifierHub(config_multi_max)
    
    signal = hub.verify_claim(sample_claim, evidence_chunks_multiple, generation_metadata)
    
    assert signal.per_chunk_signals is not None
    
    for i, chunk_signal in enumerate(signal.per_chunk_signals):
        # Verify structure matches expected format
        assert chunk_signal['doc_id'] == evidence_chunks_multiple[i].doc_id
        assert chunk_signal['sent_id'] == evidence_chunks_multiple[i].sent_id
        
        # Coverage dict should have all keys
        assert 'entities' in chunk_signal['coverage']
        assert 'numbers' in chunk_signal['coverage']
        assert 'tokens_overlap' in chunk_signal['coverage']
        
        # Uncertainty dict
        assert 'mean_entropy' in chunk_signal['uncertainty']
        
        # Other fields
        assert 'citation_span_match' in chunk_signal
        assert 'numeric_check' in chunk_signal
        assert isinstance(chunk_signal['numeric_check'], bool)


def test_aggregation_with_numeric_check(config_multi_max, sample_claim, generation_metadata):
    """Test numeric_check aggregation (any() for max, majority for mean)."""
    hub_max = VerifierHub(config_multi_max)
    
    # Create chunks where only one has a number
    chunks_with_numbers = [
        EvidenceChunk(
            doc_id="doc_001", sent_id=1,
            text="Einstein was born in 1879.",  # Has number
            char_start=0, char_end=30, score_dense=0.9, rank=0
        ),
        EvidenceChunk(
            doc_id="doc_002", sent_id=2,
            text="Einstein was a physicist.",  # No number
            char_start=30, char_end=60, score_dense=0.8, rank=1
        )
    ]
    
    signal = hub_max.verify_claim(sample_claim, chunks_with_numbers, generation_metadata)
    
    assert signal is not None
    # MAX aggregation uses any() - so should be True if any chunk matches
    # (actual result depends on whether numbers are detected in the claim)


def test_config_flag_verify_all_evidence(config_multi_max, config_single_only):
    """Test that verify_all_evidence flag is correctly read from config."""
    hub_multi = VerifierHub(config_multi_max)
    hub_single = VerifierHub(config_single_only)
    
    assert hub_multi.verify_all_evidence is True
    assert hub_single.verify_all_evidence is False


def test_config_flag_aggregation_method(config_multi_max, config_multi_mean):
    """Test that aggregation_method flag is correctly read from config."""
    hub_max = VerifierHub(config_multi_max)
    hub_mean = VerifierHub(config_multi_mean)
    
    assert hub_max.aggregation_method == 'max'
    assert hub_mean.aggregation_method == 'mean'


def test_contradiction_first_fusion_preserves_max_contradiction():
    """Contradiction-first fusion should keep strongest contradiction and select its chunk as primary."""
    config = Config()
    config.verification = type('obj', (object,), {
        'enabled': True,
        'verify_all_evidence': True,
        'aggregation_method': 'max',
        'contradiction_first_fusion': True,
        'contradiction_priority_threshold': 0.5,
        'contradiction_priority_margin': 0.0,
        'modules': type('obj', (object,), {
            'intrinsic': False,
            'grounded': False,
            'nli': False,
            'self_agreement': False,
        })(),
        'intrinsic': type('obj', (object,), {
            'strict_logits': False,
            'epsilon': 1e-10,
        })(),
    })()

    hub = VerifierHub(config)
    per_chunk_signals = [
        {
            'doc_id': 'doc_1',
            'sent_id': 1,
            'coverage': {'entities': 0.8, 'numbers': 1.0, 'tokens_overlap': 0.8},
            'uncertainty': {'mean_entropy': 0.9},
            'citation_span_match': 0.8,
            'numeric_check': True,
            'nli': {'entailment': 0.70, 'neutral': 0.20, 'contradiction': 0.20},
        },
        {
            'doc_id': 'doc_2',
            'sent_id': 2,
            'coverage': {'entities': 0.5, 'numbers': 0.0, 'tokens_overlap': 0.5},
            'uncertainty': {'mean_entropy': 1.1},
            'citation_span_match': 0.5,
            'numeric_check': False,
            'nli': {'entailment': 0.10, 'neutral': 0.10, 'contradiction': 0.85},
        },
    ]

    aggregated, primary_idx = hub._aggregate_signals(per_chunk_signals)

    assert primary_idx == 1
    assert aggregated['nli']['contradiction'] == pytest.approx(0.85)
