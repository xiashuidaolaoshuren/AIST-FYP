"""
Unit tests for RuleBasedAggregator class.

Tests hierarchical classification rules, confidence breakdown calculation,
and ClaimDecision structure validation with comprehensive edge case coverage.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.verification.rule_based_aggregator import SignalNormalizer, RuleBasedAggregator
from src.utils.config import Config
from src.utils.data_structures import VerifierSignal, ClaimDecision


class TestRuleBasedAggregator:
    """Test suite for RuleBasedAggregator class."""
    
    @pytest.fixture
    def sample_config(self):
        """Create a test configuration with aggregator settings."""
        config = Config()
        # Add aggregator config for testing
        if not hasattr(config, 'verification'):
            config._config['verification'] = {}
        config._config['verification']['aggregator'] = {
            'entropy_threshold': 2.0,
            'k_entropy': 2.0,
            'contradiction_threshold': 0.5,
            'entailment_threshold': 0.7,
            'coverage_threshold': 0.6,
            'entropy_confidence_threshold': 0.4,
            'consistency_confidence_threshold': 0.4,
            'low_coverage_threshold': 0.3,
            'coverage_weights': {
                'entities': 0.4,
                'numbers': 0.3,
                'tokens_overlap': 0.3
            }
        }
        return config
    
    @pytest.fixture
    def aggregator(self, sample_config):
        """Create a RuleBasedAggregator instance for testing."""
        return RuleBasedAggregator(sample_config)
    
    @pytest.fixture
    def mock_signal_base(self) -> Dict[str, Any]:
        """Base mock signal with neutral values."""
        return {
            'claim_id': 'test_claim_1',
            'doc_id': 'doc_123',
            'sent_id': 5,
            'nli': {
                'entailment': 0.5,
                'contradiction': 0.3,
                'neutral': 0.2
            },
            'coverage': {
                'entities': 0.5,
                'numbers': 0.0,
                'tokens_overlap': 0.5
            },
            'uncertainty': {
                'mean_entropy': 1.5
            },
            'consistency': {
                'variance': 0.5
            },
            'citation_span_match': 0.8,
            'numeric_check': True
        }
    
    @pytest.fixture
    def mock_signal_contradictory_nli(self, mock_signal_base):
        """Mock signal with high NLI contradiction."""
        signal_dict = mock_signal_base.copy()
        signal_dict['nli'] = {
            'entailment': 0.1,
            'contradiction': 0.8,  # High contradiction
            'neutral': 0.1
        }
        return VerifierSignal(**signal_dict)
    
    @pytest.fixture
    def mock_signal_contradictory_numeric(self, mock_signal_base):
        """Mock signal with numeric mismatch."""
        signal_dict = mock_signal_base.copy()
        signal_dict['coverage'] = {
            'entities': 0.6,
            'numbers': 0.5,  # Claim has numbers
            'tokens_overlap': 0.6
        }
        signal_dict['numeric_check'] = False  # Numeric mismatch
        return VerifierSignal(**signal_dict)
    
    @pytest.fixture
    def mock_signal_supported(self, mock_signal_base):
        """Mock signal with strong support."""
        signal_dict = mock_signal_base.copy()
        signal_dict['nli'] = {
            'entailment': 0.9,  # High entailment
            'contradiction': 0.05,
            'neutral': 0.05
        }
        signal_dict['coverage'] = {
            'entities': 0.9,  # High coverage (weighted avg needs to be >0.6)
            'numbers': 0.0,
            'tokens_overlap': 0.9
        }
        signal_dict['uncertainty'] = {'mean_entropy': 0.8}  # Low entropy
        signal_dict['consistency'] = {'variance': 0.2}  # High consistency
        return VerifierSignal(**signal_dict)
    
    @pytest.fixture
    def mock_signal_low_confidence(self, mock_signal_base):
        """Mock signal with weak signals across all dimensions."""
        signal_dict = mock_signal_base.copy()
        signal_dict['nli'] = {
            'entailment': 0.4,  # Weak entailment
            'contradiction': 0.3,  # Low contradiction
            'neutral': 0.3
        }
        signal_dict['coverage'] = {
            'entities': 0.25,  # Low coverage
            'numbers': 0.0,
            'tokens_overlap': 0.2
        }
        signal_dict['uncertainty'] = {'mean_entropy': 3.5}  # High entropy
        signal_dict['consistency'] = {'variance': 1.5}  # Low consistency
        return VerifierSignal(**signal_dict)
    
    # ==================== Initialization Tests ====================
    
    def test_aggregator_initialization(self, sample_config):
        """Test that RuleBasedAggregator initializes correctly."""
        aggregator = RuleBasedAggregator(sample_config)
        
        assert aggregator.config == sample_config
        assert isinstance(aggregator.normalizer, SignalNormalizer)
        assert aggregator.thresholds['contradiction'] == 0.5
        assert aggregator.thresholds['entailment'] == 0.7
        assert aggregator.thresholds['coverage'] == 0.6
        assert aggregator.thresholds['entropy_conf'] == 0.4
        assert aggregator.thresholds['consistency_conf'] == 0.4
        assert aggregator.thresholds['low_coverage'] == 0.3
    
    def test_aggregator_initialization_no_config(self):
        """Test that aggregator uses defaults when config missing."""
        config = Config()
        # No verification.aggregator section
        aggregator = RuleBasedAggregator(config)
        
        # Should use research-backed defaults
        assert aggregator.thresholds['contradiction'] == 0.5
        assert aggregator.thresholds['entailment'] == 0.65
        assert aggregator.thresholds['coverage'] == 0.6
    
    # ==================== Rule 1: Contradictory Classification Tests ====================
    
    def test_aggregate_contradictory_nli(self, aggregator, mock_signal_contradictory_nli):
        """Test contradictory classification with high NLI contradiction."""
        decision = aggregator.aggregate(mock_signal_contradictory_nli)
        
        assert isinstance(decision, ClaimDecision)
        assert decision.claim_id == 'test_claim_1'
        assert decision.status == 'Contradictory'
        assert 'contradiction' in decision.rationale.lower()
        assert 'nli' in decision.rationale.lower()
        assert decision.primary_evidence == 'doc_123#5'
        assert len(decision.signals_ref) > 0
    
    def test_aggregate_contradictory_numeric(self, aggregator, mock_signal_contradictory_numeric):
        """Test numeric mismatch defaults to Low Confidence without contradiction corroboration."""
        decision = aggregator.aggregate(mock_signal_contradictory_numeric)
        
        assert decision.status == 'Low Confidence'
        assert 'numeric' in decision.rationale.lower()
        assert 'mismatch' in decision.rationale.lower()

    def test_aggregate_contradictory_numeric_with_corroboration(self, aggregator, mock_signal_contradictory_numeric):
        """Numeric mismatch becomes Contradictory when contradiction signal corroborates it."""
        signal_dict = mock_signal_contradictory_numeric.to_dict()
        signal_dict['nli']['contradiction'] = 0.45
        signal_dict['nli']['entailment'] = 0.1
        signal = VerifierSignal(**signal_dict)

        decision = aggregator.aggregate(signal)

        assert decision.status == 'Contradictory'
        assert 'numeric' in decision.rationale.lower()
        assert 'corroboration' in decision.rationale.lower()

    def test_aggregate_ambiguous_mode_suppresses_contradictory(self, aggregator, mock_signal_base):
        """Ambiguous primary NLI mode should suppress contradictory rule firing."""
        signal_dict = mock_signal_base.copy()
        signal_dict['nli'] = {
            'entailment': 0.93,
            'contradiction': 0.97,
            'neutral': 0.01,
        }
        signal_dict['primary_nli_mode'] = 'ambiguous'
        signal = VerifierSignal(**signal_dict)

        decision = aggregator.aggregate(signal)

        assert decision.status == 'Low Confidence'
    
    def test_contradictory_confidence_breakdown(self, aggregator, mock_signal_contradictory_nli):
        """Test confidence breakdown for contradictory classification."""
        decision = aggregator.aggregate(mock_signal_contradictory_nli)
        
        assert 'confidence' in dir(decision)
        conf = decision.confidence
        
        # Check all required keys
        assert 'support_prob' in conf
        assert 'contradict_prob' in conf
        assert 'coverage_score' in conf
        assert 'entropy_conf' in conf
        assert 'consistency_conf' in conf
        assert 'overall_confidence' in conf
        assert 'band' in conf
        
        # Contradictory should have high contradiction probability
        assert conf['contradict_prob'] > 0.5
        # Overall confidence should be reasonable (0-100 scale)
        assert 0 <= conf['overall_confidence'] <= 100
    
    # ==================== Rule 2: Supported Classification Tests ====================
    
    def test_aggregate_supported(self, aggregator, mock_signal_supported):
        """Test supported classification with high entailment and coverage."""
        decision = aggregator.aggregate(mock_signal_supported)
        
        assert decision.status == 'Supported'
        assert 'support' in decision.rationale.lower() or 'entailment' in decision.rationale.lower()
        assert 'coverage' in decision.rationale.lower()
    
    def test_supported_requires_both_conditions(self, aggregator, mock_signal_base):
        """Test that supported requires BOTH high entailment AND high coverage."""
        # High entailment but low coverage
        signal_dict = mock_signal_base.copy()
        signal_dict['nli'] = {
            'entailment': 0.9,
            'contradiction': 0.05,
            'neutral': 0.05
        }
        signal_dict['coverage'] = {
            'entities': 0.2,  # Low coverage
            'numbers': 0.0,
            'tokens_overlap': 0.2
        }
        signal = VerifierSignal(**signal_dict)
        decision = aggregator.aggregate(signal)
        
        # Should NOT be supported (needs both conditions)
        assert decision.status != 'Supported'
    
    def test_supported_confidence_breakdown(self, aggregator, mock_signal_supported):
        """Test confidence breakdown for supported classification."""
        decision = aggregator.aggregate(mock_signal_supported)
        
        conf = decision.confidence
        
        # Supported should have high support probability
        assert conf['support_prob'] > 0.7
        # Coverage should be high (weighted average)
        assert conf['coverage_score'] >= 0.6, f"Coverage {conf['coverage_score']} should be >= 0.6"
        # Overall confidence should be high
        assert conf['overall_confidence'] > 60.0
        # Band should be High or Medium
        assert conf['band'] in ['High', 'Medium']
    
    # ==================== Rule 3: Low Confidence Classification Tests ====================
    
    def test_aggregate_low_confidence(self, aggregator, mock_signal_low_confidence):
        """Test low confidence fallback with weak signals."""
        decision = aggregator.aggregate(mock_signal_low_confidence)
        
        assert decision.status == 'Low Confidence'
        assert 'low confidence' in decision.rationale.lower()
    
    def test_low_confidence_rationale_lists_reasons(self, aggregator, mock_signal_low_confidence):
        """Test that low confidence rationale lists specific weak signals."""
        decision = aggregator.aggregate(mock_signal_low_confidence)
        
        rationale = decision.rationale.lower()
        
        # Should mention multiple weak signals
        # At least 2 of: entropy, consistency, coverage, support
        weak_signals = [
            'entropy' in rationale or 'uncertainty' in rationale,
            'consistency' in rationale,
            'coverage' in rationale,
            'support' in rationale
        ]
        assert sum(weak_signals) >= 2
    
    def test_low_confidence_band(self, aggregator, mock_signal_low_confidence):
        """Test that low confidence gets 'Low' band."""
        decision = aggregator.aggregate(mock_signal_low_confidence)
        
        assert decision.confidence['band'] == 'Low'
        # Overall confidence should be around 50
        assert 45.0 <= decision.confidence['overall_confidence'] <= 55.0
    
    # ==================== Confidence Breakdown Tests ====================
    
    def test_confidence_breakdown_structure(self, aggregator, mock_signal_base):
        """Test that confidence breakdown has all required fields."""
        signal = VerifierSignal(**mock_signal_base)
        decision = aggregator.aggregate(signal)
        
        conf = decision.confidence
        
        # Check all required keys
        required_keys = [
            'support_prob', 'contradict_prob', 'coverage_score',
            'entropy_conf', 'consistency_conf', 'overall_confidence', 'band'
        ]
        for key in required_keys:
            assert key in conf, f"Missing key: {key}"
    
    def test_confidence_breakdown_ranges(self, aggregator, mock_signal_base):
        """Test that confidence values are in valid ranges."""
        signal = VerifierSignal(**mock_signal_base)
        decision = aggregator.aggregate(signal)
        
        conf = decision.confidence
        
        # All probabilities should be in [0, 1]
        assert 0.0 <= conf['support_prob'] <= 1.0
        assert 0.0 <= conf['contradict_prob'] <= 1.0
        assert 0.0 <= conf['coverage_score'] <= 1.0
        assert 0.0 <= conf['entropy_conf'] <= 1.0
        assert 0.0 <= conf['consistency_conf'] <= 1.0
        
        # Overall confidence should be in [0, 100]
        assert 0.0 <= conf['overall_confidence'] <= 100.0
        
        # Band should be one of three values
        assert conf['band'] in ['High', 'Medium', 'Low']
    
    def test_confidence_band_high(self, aggregator, mock_signal_contradictory_nli):
        """Test that high contradiction/support gets 'High' band."""
        decision = aggregator.aggregate(mock_signal_contradictory_nli)
        
        # High contradiction should get 'High' band
        assert decision.confidence['band'] == 'High'
    
    # ==================== Helper Method Tests ====================
    
    def test_has_numeric_claims_true(self, aggregator, mock_signal_contradictory_numeric):
        """Test _has_numeric_claims returns True when claim has numbers."""
        result = aggregator._has_numeric_claims(mock_signal_contradictory_numeric)
        
        assert result is True
    
    def test_has_numeric_claims_false(self, aggregator, mock_signal_base):
        """Test _has_numeric_claims returns False when no numbers."""
        signal = VerifierSignal(**mock_signal_base)
        result = aggregator._has_numeric_claims(signal)
        
        assert result is False
    
    def test_has_numeric_claims_missing_key(self, aggregator, mock_signal_base):
        """Test _has_numeric_claims handles missing 'numbers' key gracefully."""
        signal_dict = mock_signal_base.copy()
        # Remove 'numbers' key
        signal_dict['coverage'] = {
            'entities': 0.5,
            'tokens_overlap': 0.5
        }
        signal = VerifierSignal(**signal_dict)
        
        result = aggregator._has_numeric_claims(signal)
        
        # Should return False (no error)
        assert result is False
    
    # ==================== Edge Case Tests ====================
    
    def test_aggregate_with_none_entropy(self, aggregator, mock_signal_base):
        """Test aggregation with None entropy value."""
        signal_dict = mock_signal_base.copy()
        signal_dict['uncertainty'] = {'mean_entropy': None}
        signal = VerifierSignal(**signal_dict)
        
        # Should not raise error
        decision = aggregator.aggregate(signal)
        
        assert isinstance(decision, ClaimDecision)
        assert decision.status in ['Supported', 'Contradictory', 'Low Confidence']
    
    def test_aggregate_with_none_consistency(self, aggregator, mock_signal_base):
        """Test aggregation with None consistency variance."""
        signal_dict = mock_signal_base.copy()
        signal_dict['consistency'] = {'variance': None}
        signal = VerifierSignal(**signal_dict)
        
        # Should not raise error
        decision = aggregator.aggregate(signal)
        
        assert isinstance(decision, ClaimDecision)
    
    def test_aggregate_with_nan_values(self, aggregator, mock_signal_base):
        """Test aggregation with NaN values in signals."""
        signal_dict = mock_signal_base.copy()
        signal_dict['nli'] = {
            'entail': float('nan'),
            'contradict': float('nan'),
            'neutral': float('nan')
        }
        signal = VerifierSignal(**signal_dict)
        
        # Should handle gracefully (normalizer returns 0.5 for invalid)
        decision = aggregator.aggregate(signal)
        
        assert isinstance(decision, ClaimDecision)
    
    def test_aggregate_with_empty_coverage(self, aggregator, mock_signal_base):
        """Test aggregation with empty coverage dict."""
        signal_dict = mock_signal_base.copy()
        signal_dict['coverage'] = {}
        signal = VerifierSignal(**signal_dict)
        
        # Should handle gracefully
        decision = aggregator.aggregate(signal)
        
        assert isinstance(decision, ClaimDecision)
        assert decision.confidence['coverage_score'] == 0.0
    
    # ==================== Threshold Sensitivity Tests ====================
    
    @pytest.mark.parametrize("contradiction_score,expected_status", [
        (0.49, 'Low Confidence'),  # Just below threshold
        (0.51, 'Contradictory'),   # Just above threshold
        (0.7, 'Contradictory'),    # Well above threshold
    ])
    def test_contradiction_threshold_sensitivity(self, sample_config, mock_signal_base,
                                                 contradiction_score, expected_status):
        """Test classification is sensitive to contradiction threshold."""
        aggregator = RuleBasedAggregator(sample_config)
        
        signal_dict = mock_signal_base.copy()
        signal_dict['nli'] = {
            'entailment': 0.1,
            'contradiction': contradiction_score,
            'neutral': 0.1
        }
        # Ensure it doesn't hit supported condition
        signal_dict['coverage'] = {
            'entities': 0.3,
            'numbers': 0.0,
            'tokens_overlap': 0.3
        }
        signal = VerifierSignal(**signal_dict)
        
        decision = aggregator.aggregate(signal)
        
        assert decision.status == expected_status
    
    @pytest.mark.parametrize("entailment_score,coverage_score,expected_status", [
        (0.9, 0.95, 'Supported'),      # Both high (weighted avg: 0.95*0.4 + 0*0.3 + 0.95*0.3 = 0.665 > 0.6)
        (0.9, 0.5, 'Low Confidence'),  # High entailment, low coverage
        (0.5, 0.95, 'Low Confidence'), # Low entailment, high coverage
        (0.5, 0.5, 'Low Confidence'),  # Both low
    ])
    def test_supported_requires_both_thresholds(self, sample_config, mock_signal_base,
                                               entailment_score, coverage_score, expected_status):
        """Test that supported classification requires both conditions."""
        aggregator = RuleBasedAggregator(sample_config)
        
        signal_dict = mock_signal_base.copy()
        signal_dict['nli'] = {
            'entailment': entailment_score,
            'contradiction': 0.1,
            'neutral': 0.1
        }
        signal_dict['coverage'] = {
            'entities': coverage_score,
            'numbers': 0.0,
            'tokens_overlap': coverage_score
        }
        signal = VerifierSignal(**signal_dict)
        
        decision = aggregator.aggregate(signal)
        
        assert decision.status == expected_status
    
    # ==================== ClaimDecision Structure Validation Tests ====================
    
    def test_claim_decision_has_all_fields(self, aggregator, mock_signal_base):
        """Test that ClaimDecision has all required fields."""
        signal = VerifierSignal(**mock_signal_base)
        decision = aggregator.aggregate(signal)
        
        # Required fields
        assert hasattr(decision, 'claim_id')
        assert hasattr(decision, 'status')
        assert hasattr(decision, 'rationale')
        assert hasattr(decision, 'primary_evidence')
        assert hasattr(decision, 'signals_ref')
        assert hasattr(decision, 'confidence')
        
        # Field types
        assert isinstance(decision.claim_id, str)
        assert isinstance(decision.status, str)
        assert isinstance(decision.rationale, str)
        assert isinstance(decision.primary_evidence, str)
        assert isinstance(decision.signals_ref, list)
        assert isinstance(decision.confidence, dict)
    
    def test_claim_decision_status_values(self, aggregator, mock_signal_base):
        """Test that status is one of the three valid values."""
        signal = VerifierSignal(**mock_signal_base)
        decision = aggregator.aggregate(signal)
        
        valid_statuses = ['Supported', 'Contradictory', 'Low Confidence']
        assert decision.status in valid_statuses
    
    def test_claim_decision_rationale_non_empty(self, aggregator, mock_signal_base):
        """Test that rationale is non-empty string."""
        signal = VerifierSignal(**mock_signal_base)
        decision = aggregator.aggregate(signal)
        
        assert len(decision.rationale) > 0
        assert isinstance(decision.rationale, str)
    
    def test_claim_decision_primary_evidence_format(self, aggregator, mock_signal_base):
        """Test that primary_evidence follows 'doc_id#sent_id' format."""
        signal = VerifierSignal(**mock_signal_base)
        decision = aggregator.aggregate(signal)
        
        # Should have format "doc_id#sent_id"
        assert '#' in decision.primary_evidence
        parts = decision.primary_evidence.split('#')
        assert len(parts) == 2
        assert parts[0] == 'doc_123'
        assert parts[1] == '5'
    
    # ==================== Integration Tests ====================
    
    def test_aggregate_end_to_end_contradictory(self, aggregator):
        """Test full aggregation pipeline for contradictory case."""
        signal = VerifierSignal(
            claim_id='claim_e2e_1',
            doc_id='doc_999',
            sent_id=10,
            nli={'entailment': 0.05, 'contradiction': 0.9, 'neutral': 0.05},
            coverage={'entities': 0.5, 'numbers': 0.0, 'tokens_overlap': 0.5},
            uncertainty={'mean_entropy': 1.5},
            consistency={'variance': 0.5},
            citation_span_match=0.7,
            numeric_check=True
        )
        
        decision = aggregator.aggregate(signal)
        
        assert decision.claim_id == 'claim_e2e_1'
        assert decision.status == 'Contradictory'
        assert decision.primary_evidence == 'doc_999#10'
        assert decision.confidence['contradict_prob'] > 0.7
        assert decision.confidence['band'] == 'High'
    
    def test_aggregate_end_to_end_supported(self, aggregator):
        """Test full aggregation pipeline for supported case."""
        signal = VerifierSignal(
            claim_id='claim_e2e_2',
            doc_id='doc_888',
            sent_id=20,
            nli={'entailment': 0.95, 'contradiction': 0.02, 'neutral': 0.03},
            coverage={'entities': 0.95, 'numbers': 0.0, 'tokens_overlap': 0.95},
            uncertainty={'mean_entropy': 0.5},
            consistency={'variance': 0.1},
            citation_span_match=0.95,
            numeric_check=True
        )
        
        decision = aggregator.aggregate(signal)
        
        assert decision.claim_id == 'claim_e2e_2'
        assert decision.status == 'Supported'
        assert decision.primary_evidence == 'doc_888#20'
        assert decision.confidence['support_prob'] > 0.8
        assert decision.confidence['coverage_score'] > 0.6
        assert decision.confidence['overall_confidence'] > 70.0
    
    def test_aggregate_end_to_end_low_confidence(self, aggregator):
        """Test full aggregation pipeline for low confidence case."""
        signal = VerifierSignal(
            claim_id='claim_e2e_3',
            doc_id='doc_777',
            sent_id=30,
            nli={'entailment': 0.4, 'contradiction': 0.3, 'neutral': 0.3},
            coverage={'entities': 0.2, 'numbers': 0.0, 'tokens_overlap': 0.15},
            uncertainty={'mean_entropy': 4.0},
            consistency={'variance': 2.0},
            citation_span_match=0.3,
            numeric_check=True
        )
        
        decision = aggregator.aggregate(signal)
        
        assert decision.claim_id == 'claim_e2e_3'
        assert decision.status == 'Low Confidence'
        assert decision.confidence['band'] == 'Low'
        assert 45.0 <= decision.confidence['overall_confidence'] <= 55.0
    
    # ==================== Error Handling Tests ====================
    
    def test_aggregate_handles_exception_gracefully(self, aggregator):
        """Test that aggregate returns fallback decision on error."""
        # Create invalid signal (will cause error during processing)
        signal = VerifierSignal(
            claim_id='error_claim',
            doc_id='doc_err',
            sent_id=-1,
            nli=None,  # Invalid: None instead of dict
            coverage=None,  # Invalid: None instead of dict
            uncertainty=None,  # Invalid: None instead of dict
            consistency=None,  # Invalid: None instead of dict
            citation_span_match=0.0,
            numeric_check=True
        )
        
        # Should not raise exception
        decision = aggregator.aggregate(signal)
        
        # Should return fallback decision
        assert decision.status == 'Low Confidence'
        assert 'error' in decision.rationale.lower()
        assert decision.confidence['overall_confidence'] == 0.0
