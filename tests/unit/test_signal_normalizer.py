"""
Unit tests for SignalNormalizer class.

Tests normalization edge cases (None, 0, inf, nan), all normalization methods,
and ensures output is always in [0, 1] range.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.verification.rule_based_aggregator import SignalNormalizer
from src.utils.config import Config


class TestSignalNormalizer:
    """Test suite for SignalNormalizer class."""
    
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
            'coverage_weights': {
                'entities': 0.4,
                'numbers': 0.3,
                'tokens_overlap': 0.3
            }
        }
        return config
    
    @pytest.fixture
    def normalizer(self, sample_config):
        """Create a SignalNormalizer instance for testing."""
        return SignalNormalizer(sample_config)
    
    # ==================== Entropy Normalization Tests ====================
    
    def test_normalize_entropy_range(self, normalizer):
        """Test that entropy normalization returns values in [0, 1] range."""
        test_values = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 50.0]
        
        for entropy in test_values:
            confidence = normalizer.normalize_entropy(entropy)
            assert 0.0 <= confidence <= 1.0, (
                f"Entropy {entropy} produced confidence {confidence} outside [0, 1]"
            )
    
    def test_normalize_entropy_threshold_midpoint(self, normalizer):
        """Test that entropy at threshold gives ~0.5 confidence."""
        confidence = normalizer.normalize_entropy(2.0)
        assert abs(confidence - 0.5) < 0.01, (
            f"Entropy at threshold should give ~0.5, got {confidence}"
        )
    
    def test_normalize_entropy_monotonic_decrease(self, normalizer):
        """Test that higher entropy gives lower confidence (monotonic decrease)."""
        entropies = [0.5, 1.0, 2.0, 3.0, 5.0]
        confidences = [normalizer.normalize_entropy(e) for e in entropies]
        
        for i in range(len(confidences) - 1):
            assert confidences[i] > confidences[i + 1], (
                f"Entropy normalization not monotonically decreasing: "
                f"{confidences[i]} <= {confidences[i + 1]}"
            )
    
    def test_normalize_entropy_none(self, normalizer):
        """Test that None entropy returns neutral 0.5."""
        confidence = normalizer.normalize_entropy(None)
        assert confidence == 0.5, f"None entropy should return 0.5, got {confidence}"
    
    def test_normalize_entropy_nan(self, normalizer):
        """Test that NaN entropy returns neutral 0.5."""
        confidence = normalizer.normalize_entropy(np.nan)
        assert confidence == 0.5, f"NaN entropy should return 0.5, got {confidence}"
    
    def test_normalize_entropy_positive_inf(self, normalizer):
        """Test that +Inf entropy returns 0.0 (very uncertain)."""
        confidence = normalizer.normalize_entropy(np.inf)
        assert confidence == 0.0, f"+Inf entropy should return 0.0, got {confidence}"
    
    def test_normalize_entropy_negative_inf(self, normalizer):
        """Test that -Inf entropy returns 1.0 (impossible but handle gracefully)."""
        confidence = normalizer.normalize_entropy(-np.inf)
        assert confidence == 1.0, f"-Inf entropy should return 1.0, got {confidence}"
    
    def test_normalize_entropy_extreme_values(self, normalizer):
        """Test entropy normalization with extreme but valid values."""
        # Very low entropy (very confident)
        assert normalizer.normalize_entropy(0.01) > 0.9
        
        # Very high entropy (very uncertain)
        assert normalizer.normalize_entropy(100.0) < 0.01
    
    def test_normalize_entropy_negative_value(self, normalizer):
        """Test that negative entropy (impossible) is handled gracefully."""
        confidence = normalizer.normalize_entropy(-1.0)
        # Should still return a value in [0, 1]
        assert 0.0 <= confidence <= 1.0
    
    # ==================== Consistency Normalization Tests ====================
    
    def test_normalize_consistency_range(self, normalizer):
        """Test that consistency normalization returns values in [0, 1] range."""
        test_values = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        
        for variance in test_values:
            confidence = normalizer.normalize_consistency(variance)
            assert 0.0 <= confidence <= 1.0, (
                f"Variance {variance} produced confidence {confidence} outside [0, 1]"
            )
    
    def test_normalize_consistency_zero_variance(self, normalizer):
        """Test that zero variance (perfect consistency) gives 1.0."""
        confidence = normalizer.normalize_consistency(0.0)
        assert confidence == 1.0, (
            f"Zero variance should give 1.0 confidence, got {confidence}"
        )
    
    def test_normalize_consistency_monotonic_decrease(self, normalizer):
        """Test that higher variance gives lower confidence (monotonic decrease)."""
        variances = [0.0, 0.5, 1.0, 2.0, 5.0]
        confidences = [normalizer.normalize_consistency(v) for v in variances]
        
        for i in range(len(confidences) - 1):
            assert confidences[i] > confidences[i + 1], (
                f"Consistency normalization not monotonically decreasing: "
                f"{confidences[i]} <= {confidences[i + 1]}"
            )
    
    def test_normalize_consistency_none(self, normalizer):
        """Test that None variance returns neutral 0.5."""
        confidence = normalizer.normalize_consistency(None)
        assert confidence == 0.5, f"None variance should return 0.5, got {confidence}"
    
    def test_normalize_consistency_nan(self, normalizer):
        """Test that NaN variance returns neutral 0.5."""
        confidence = normalizer.normalize_consistency(np.nan)
        assert confidence == 0.5, f"NaN variance should return 0.5, got {confidence}"
    
    def test_normalize_consistency_inf(self, normalizer):
        """Test that Inf variance returns 0.0 (very inconsistent)."""
        confidence = normalizer.normalize_consistency(np.inf)
        assert confidence == 0.0, f"Inf variance should return 0.0, got {confidence}"
    
    def test_normalize_consistency_negative_variance(self, normalizer):
        """Test that negative variance (impossible) is handled as 0."""
        confidence = normalizer.normalize_consistency(-1.0)
        # Should treat as 0 variance (log warning and use 0.0)
        assert confidence == 1.0, (
            f"Negative variance should be treated as 0.0, got {confidence}"
        )
    
    def test_normalize_consistency_expected_values(self, normalizer):
        """Test consistency normalization with expected variance values."""
        # exp(-0.5) ≈ 0.606
        assert abs(normalizer.normalize_consistency(0.5) - 0.606) < 0.01
        
        # exp(-1.0) ≈ 0.368
        assert abs(normalizer.normalize_consistency(1.0) - 0.368) < 0.01
        
        # exp(-2.0) ≈ 0.135
        assert abs(normalizer.normalize_consistency(2.0) - 0.135) < 0.01
    
    # ==================== Coverage Normalization Tests ====================
    
    def test_normalize_coverage_range(self, normalizer):
        """Test that coverage normalization returns values in [0, 1] range."""
        test_cases = [
            {'entities': 0.0, 'numbers': 0.0, 'tokens_overlap': 0.0},
            {'entities': 0.5, 'numbers': 0.5, 'tokens_overlap': 0.5},
            {'entities': 1.0, 'numbers': 1.0, 'tokens_overlap': 1.0},
            {'entities': 0.8, 'numbers': 0.6, 'tokens_overlap': 0.7},
        ]
        
        for coverage in test_cases:
            confidence = normalizer.normalize_coverage(coverage)
            assert 0.0 <= confidence <= 1.0, (
                f"Coverage {coverage} produced confidence {confidence} outside [0, 1]"
            )
    
    def test_normalize_coverage_weighted_average(self, normalizer):
        """Test that coverage uses correct weighted average formula."""
        coverage = {'entities': 0.8, 'numbers': 0.6, 'tokens_overlap': 0.7}
        expected = 0.8 * 0.4 + 0.6 * 0.3 + 0.7 * 0.3  # = 0.71
        
        confidence = normalizer.normalize_coverage(coverage)
        assert abs(confidence - expected) < 0.001, (
            f"Coverage calculation incorrect: expected {expected}, got {confidence}"
        )
    
    def test_normalize_coverage_all_zero(self, normalizer):
        """Test that zero coverage returns 0.0."""
        coverage = {'entities': 0.0, 'numbers': 0.0, 'tokens_overlap': 0.0}
        confidence = normalizer.normalize_coverage(coverage)
        assert confidence == 0.0, f"Zero coverage should return 0.0, got {confidence}"
    
    def test_normalize_coverage_all_one(self, normalizer):
        """Test that perfect coverage returns 1.0."""
        coverage = {'entities': 1.0, 'numbers': 1.0, 'tokens_overlap': 1.0}
        confidence = normalizer.normalize_coverage(coverage)
        assert confidence == 1.0, f"Perfect coverage should return 1.0, got {confidence}"
    
    def test_normalize_coverage_missing_keys(self, normalizer):
        """Test that missing coverage keys are treated as 0.0."""
        # Only entities present
        coverage1 = {'entities': 1.0}
        confidence1 = normalizer.normalize_coverage(coverage1)
        expected1 = 1.0 * 0.4  # Only entities contributes
        assert abs(confidence1 - expected1) < 0.001, (
            f"Partial coverage (entities only) incorrect: "
            f"expected {expected1}, got {confidence1}"
        )
        
        # Only numbers and tokens
        coverage2 = {'numbers': 0.6, 'tokens_overlap': 0.8}
        confidence2 = normalizer.normalize_coverage(coverage2)
        expected2 = 0.6 * 0.3 + 0.8 * 0.3  # entities=0
        assert abs(confidence2 - expected2) < 0.001
    
    def test_normalize_coverage_empty_dict(self, normalizer):
        """Test that empty coverage dict returns 0.0."""
        confidence = normalizer.normalize_coverage({})
        assert confidence == 0.0, f"Empty coverage dict should return 0.0, got {confidence}"
    
    def test_normalize_coverage_nan_values(self, normalizer):
        """Test that NaN coverage values are treated as 0.0."""
        coverage = {'entities': np.nan, 'numbers': 0.5, 'tokens_overlap': 0.7}
        confidence = normalizer.normalize_coverage(coverage)
        expected = 0.0 * 0.4 + 0.5 * 0.3 + 0.7 * 0.3  # entities NaN -> 0
        assert abs(confidence - expected) < 0.001
    
    def test_normalize_coverage_out_of_range(self, normalizer):
        """Test that out-of-range coverage values are clipped to [0, 1]."""
        # Values > 1.0 should be clipped to 1.0
        coverage1 = {'entities': 1.5, 'numbers': 0.5, 'tokens_overlap': 0.5}
        confidence1 = normalizer.normalize_coverage(coverage1)
        expected1 = 1.0 * 0.4 + 0.5 * 0.3 + 0.5 * 0.3  # entities clipped to 1.0
        assert abs(confidence1 - expected1) < 0.001
        
        # Negative values should be clipped to 0.0
        coverage2 = {'entities': -0.5, 'numbers': 0.5, 'tokens_overlap': 0.5}
        confidence2 = normalizer.normalize_coverage(coverage2)
        expected2 = 0.0 * 0.4 + 0.5 * 0.3 + 0.5 * 0.3  # entities clipped to 0.0
        assert abs(confidence2 - expected2) < 0.001
    
    # ==================== NLI Normalization Tests ====================
    
    def test_normalize_nli_extraction(self, normalizer):
        """Test that NLI extraction returns correct tuple."""
        nli = {'entail': 0.8, 'contradict': 0.1, 'neutral': 0.1}
        support, contradict = normalizer.normalize_nli(nli)
        
        assert support == 0.8, f"Expected support 0.8, got {support}"
        assert contradict == 0.1, f"Expected contradict 0.1, got {contradict}"
    
    def test_normalize_nli_range(self, normalizer):
        """Test that NLI extraction returns values in [0, 1] range."""
        test_cases = [
            {'entail': 0.0, 'contradict': 0.0, 'neutral': 1.0},
            {'entail': 0.5, 'contradict': 0.3, 'neutral': 0.2},
            {'entail': 1.0, 'contradict': 0.0, 'neutral': 0.0},
            {'entail': 0.2, 'contradict': 0.7, 'neutral': 0.1},
        ]
        
        for nli in test_cases:
            support, contradict = normalizer.normalize_nli(nli)
            assert 0.0 <= support <= 1.0, (
                f"NLI entailment {support} outside [0, 1]"
            )
            assert 0.0 <= contradict <= 1.0, (
                f"NLI contradiction {contradict} outside [0, 1]"
            )
    
    def test_normalize_nli_missing_keys(self, normalizer):
        """Test that missing NLI keys return neutral 0.5."""
        # Missing entail
        nli1 = {'contradict': 0.3, 'neutral': 0.7}
        support1, contradict1 = normalizer.normalize_nli(nli1)
        assert support1 == 0.5, f"Missing entail should give 0.5, got {support1}"
        assert contradict1 == 0.3
        
        # Missing contradict
        nli2 = {'entail': 0.8, 'neutral': 0.2}
        support2, contradict2 = normalizer.normalize_nli(nli2)
        assert support2 == 0.8
        assert contradict2 == 0.5, f"Missing contradict should give 0.5, got {contradict2}"
    
    def test_normalize_nli_empty_dict(self, normalizer):
        """Test that empty NLI dict returns (0.5, 0.5)."""
        support, contradict = normalizer.normalize_nli({})
        assert support == 0.5, f"Empty NLI should return support 0.5, got {support}"
        assert contradict == 0.5, f"Empty NLI should return contradict 0.5, got {contradict}"
    
    def test_normalize_nli_nan_values(self, normalizer):
        """Test that NaN NLI values return neutral 0.5."""
        nli = {'entail': np.nan, 'contradict': 0.3, 'neutral': 0.7}
        support, contradict = normalizer.normalize_nli(nli)
        assert support == 0.5, f"NaN entail should return 0.5, got {support}"
        assert contradict == 0.3
    
    def test_normalize_nli_out_of_range(self, normalizer):
        """Test that out-of-range NLI values are clipped to [0, 1]."""
        # Values > 1.0 should be clipped
        nli1 = {'entail': 1.5, 'contradict': 0.3, 'neutral': -0.8}
        support1, contradict1 = normalizer.normalize_nli(nli1)
        assert support1 == 1.0, f"Entail > 1.0 should be clipped to 1.0, got {support1}"
        assert contradict1 == 0.3
        
        # Negative values should be clipped to 0.0
        nli2 = {'entail': 0.8, 'contradict': -0.5, 'neutral': 0.7}
        support2, contradict2 = normalizer.normalize_nli(nli2)
        assert support2 == 0.8
        assert contradict2 == 0.0, f"Negative contradict should be clipped to 0.0, got {contradict2}"
    
    # ==================== Integration Tests ====================
    
    def test_normalizer_with_default_config(self):
        """Test that normalizer works with default config (no aggregator section)."""
        config = Config()
        # Don't add aggregator config - test defaults
        normalizer = SignalNormalizer(config)
        
        # Should use default thresholds
        assert normalizer.entropy_threshold == 2.0
        assert normalizer.k_entropy == 2.0
        assert normalizer.coverage_weights['entities'] == 0.4
        
        # Test that it still works
        assert 0.0 <= normalizer.normalize_entropy(3.0) <= 1.0
        assert 0.0 <= normalizer.normalize_consistency(1.0) <= 1.0
    
    def test_normalizer_custom_thresholds(self):
        """Test that normalizer respects custom config thresholds."""
        config = Config()
        config._config['verification'] = {
            'aggregator': {
                'entropy_threshold': 3.0,
                'k_entropy': 1.5,
                'coverage_weights': {
                    'entities': 0.5,
                    'numbers': 0.3,
                    'tokens_overlap': 0.2
                }
            }
        }
        
        normalizer = SignalNormalizer(config)
        
        assert normalizer.entropy_threshold == 3.0
        assert normalizer.k_entropy == 1.5
        assert normalizer.coverage_weights['entities'] == 0.5
        
        # Verify behavior changes with new threshold
        confidence = normalizer.normalize_entropy(3.0)
        assert abs(confidence - 0.5) < 0.01  # Should be at midpoint now


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
