"""
Rule-Based Aggregator for Multi-Signal Hallucination Detection.

This module implements the rule-based aggregator system for Month 5, which combines
multiple verification signals (entropy, consistency, coverage, NLI) into final
claim decisions. It consists of two main components:

1. SignalNormalizer: Transforms heterogeneous signals to unified 0-1 confidence scale
2. RuleBasedAggregator: Applies hierarchical rules to classify claims

The normalization ensures that different signal types (entropy, variance, coverage, etc.)
are comparable and can be combined using consistent thresholds.

Key Concepts:
- Entropy normalization: Sigmoid transformation to map uncertainty to confidence
- Consistency normalization: Exponential decay to map variance to consistency
- Coverage normalization: Weighted average of entity/number/token overlap
- NLI extraction: Direct extraction of entailment/contradiction probabilities

Reference:
- SelfCheckGPT: Entropy threshold ~2.0 for hallucination detection
- Coverage weights: Based on empirical importance (entities > numbers > tokens)
"""

import numpy as np
from typing import Dict, Optional, Tuple
import warnings

from src.utils.config import Config
from src.utils.logger import setup_logger


class SignalNormalizer:
    """
    Normalizes heterogeneous verification signals to unified 0-1 confidence scale.
    
    This class transforms raw detector signals into normalized confidence scores:
    - Entropy (high=uncertain) → Confidence (high=certain)
    - Variance (high=inconsistent) → Consistency (high=consistent)
    - Coverage metrics → Weighted confidence score
    - NLI probabilities → Support/Contradict confidence
    
    All normalized scores are in [0, 1] range where:
    - 0.0 = Very low confidence (likely hallucination)
    - 0.5 = Neutral/Unknown (missing data)
    - 1.0 = Very high confidence (likely factual)
    
    Attributes:
        config: Configuration object
        entropy_threshold: Entropy value at sigmoid midpoint (default: 2.0)
        k_entropy: Sigmoid steepness parameter (default: 2.0)
        coverage_weights: Weights for coverage components (entities, numbers, tokens)
        logger: Logger instance
    
    Example:
        >>> config = Config()
        >>> normalizer = SignalNormalizer(config)
        >>> entropy_conf = normalizer.normalize_entropy(3.5)
        >>> print(f"Entropy confidence: {entropy_conf:.3f}")  # ~0.18 (uncertain)
    """
    
    def __init__(self, config: Config):
        """
        Initialize the SignalNormalizer with configuration parameters.
        
        Loads normalization parameters from config.verification.aggregator.
        If not present, uses research-backed defaults.
        
        Args:
            config: Configuration object with aggregator settings
        """
        self.config = config
        self.logger = setup_logger(__name__)
        
        # Load entropy normalization parameters
        if (hasattr(config, 'verification') and 
            hasattr(config.verification, 'aggregator')):
            aggregator_config = config.verification.aggregator
            self.entropy_threshold = float(getattr(aggregator_config, 'entropy_threshold', 2.0))
            self.k_entropy = float(getattr(aggregator_config, 'k_entropy', 2.0))
            
            # Load coverage weights
            if hasattr(aggregator_config, 'coverage_weights'):
                weights = aggregator_config.coverage_weights
                self.coverage_weights = {
                    'entities': float(getattr(weights, 'entities', 0.4)),
                    'numbers': float(getattr(weights, 'numbers', 0.3)),
                    'tokens_overlap': float(getattr(weights, 'tokens_overlap', 0.3))
                }
            else:
                # Default weights based on empirical importance
                self.coverage_weights = {
                    'entities': 0.4,
                    'numbers': 0.3,
                    'tokens_overlap': 0.3
                }
        else:
            # Defaults from SelfCheckGPT research and empirical analysis
            self.entropy_threshold = 2.0
            self.k_entropy = 2.0
            self.coverage_weights = {
                'entities': 0.4,
                'numbers': 0.3,
                'tokens_overlap': 0.3
            }
            self.logger.warning(
                "No aggregator config found, using defaults: "
                f"entropy_threshold={self.entropy_threshold}, k_entropy={self.k_entropy}"
            )
        
        self.logger.info(
            f"SignalNormalizer initialized: "
            f"entropy_threshold={self.entropy_threshold}, "
            f"k_entropy={self.k_entropy}, "
            f"coverage_weights={self.coverage_weights}"
        )
    
    def normalize_entropy(self, mean_entropy: Optional[float]) -> float:
        """
        Normalize entropy to confidence score using sigmoid transformation.
        
        Converts model uncertainty (entropy) to confidence score:
        - Low entropy (< threshold) → High confidence (> 0.5)
        - High entropy (> threshold) → Low confidence (< 0.5)
        
        Formula: confidence = 1 / (1 + exp(k * (entropy - threshold)))
        
        This is a logistic sigmoid centered at entropy_threshold. The k parameter
        controls steepness (higher k = sharper transition).
        
        Args:
            mean_entropy: Mean token-level entropy from generator (or None)
        
        Returns:
            Normalized confidence score in [0, 1]
            Returns 0.5 (neutral) if mean_entropy is None or invalid
        
        Edge Cases:
            - None → 0.5 (neutral, missing data)
            - NaN → 0.5 (neutral, invalid data)
            - Inf → 0.0 (very uncertain)
            - Negative → Treated as 0.0 (unexpected but valid mathematically)
        
        Example:
            >>> normalizer.normalize_entropy(1.0)  # Low entropy
            0.731...  # High confidence
            >>> normalizer.normalize_entropy(5.0)  # High entropy
            0.047...  # Low confidence
        """
        # Handle None (missing data)
        if mean_entropy is None:
            self.logger.debug("Entropy is None, returning neutral 0.5")
            return 0.5
        
        # Handle NaN (invalid data)
        if np.isnan(mean_entropy):
            self.logger.warning("Entropy is NaN, returning neutral 0.5")
            return 0.5
        
        # Handle Inf (overflow or extreme uncertainty)
        if np.isinf(mean_entropy):
            if mean_entropy > 0:
                self.logger.debug("Entropy is +Inf, returning 0.0 (very uncertain)")
                return 0.0
            else:
                # -Inf is theoretically impossible for entropy, but handle gracefully
                self.logger.warning("Entropy is -Inf (impossible), returning 1.0")
                return 1.0
        
        try:
            # Calculate sigmoid: 1 / (1 + exp(k * (x - threshold)))
            exponent = self.k_entropy * (mean_entropy - self.entropy_threshold)
            
            # Clip exponent to prevent overflow in exp()
            # exp(709) ≈ 8.2e307 (near float64 max)
            # exp(-709) ≈ 1.2e-308 (near float64 min)
            exponent = np.clip(exponent, -700, 700)
            
            confidence = 1.0 / (1.0 + np.exp(exponent))
            
            # Ensure result is in valid range (numerical stability)
            confidence = float(np.clip(confidence, 0.0, 1.0))
            
            return confidence
            
        except (ValueError, OverflowError, FloatingPointError) as e:
            self.logger.error(
                f"Error normalizing entropy {mean_entropy}: {e}, returning 0.5"
            )
            return 0.5
    
    def normalize_consistency(self, variance: Optional[float]) -> float:
        """
        Normalize consistency variance to confidence score using exponential decay.
        
        Converts self-agreement variance to consistency confidence:
        - Low variance (consistent samples) → High confidence
        - High variance (inconsistent samples) → Low confidence
        
        Formula: confidence = exp(-variance)
        
        This exponential decay naturally maps variance [0, ∞) to confidence (1, 0].
        
        Args:
            variance: Variance of semantic similarity across stochastic samples (or None)
        
        Returns:
            Normalized consistency confidence in [0, 1]
            Returns 0.5 (neutral) if variance is None
        
        Edge Cases:
            - None → 0.5 (neutral, self-agreement not computed)
            - NaN → 0.5 (neutral, invalid data)
            - Inf → 0.0 (very inconsistent)
            - Negative → Treated as 0.0 (invalid variance, but handle gracefully)
        
        Example:
            >>> normalizer.normalize_consistency(0.0)   # Perfect consistency
            1.0
            >>> normalizer.normalize_consistency(1.0)   # Some variance
            0.367...
            >>> normalizer.normalize_consistency(5.0)   # High variance
            0.006...
        """
        # Handle None (self-agreement not computed)
        if variance is None:
            self.logger.debug("Variance is None, returning neutral 0.5")
            return 0.5
        
        # Handle NaN (invalid data)
        if np.isnan(variance):
            self.logger.warning("Variance is NaN, returning neutral 0.5")
            return 0.5
        
        # Handle negative variance (theoretically impossible, but be defensive)
        if variance < 0:
            self.logger.warning(
                f"Variance is negative ({variance}), treating as 0.0"
            )
            variance = 0.0
        
        # Handle Inf (extreme inconsistency)
        if np.isinf(variance):
            self.logger.debug("Variance is Inf, returning 0.0 (very inconsistent)")
            return 0.0
        
        try:
            # Calculate exponential decay: exp(-variance)
            # Clip variance to prevent underflow (exp(-700) ≈ 0)
            variance_clipped = min(variance, 700)
            
            confidence = np.exp(-variance_clipped)
            
            # Ensure result is in valid range
            confidence = float(np.clip(confidence, 0.0, 1.0))
            
            return confidence
            
        except (ValueError, OverflowError, FloatingPointError) as e:
            self.logger.error(
                f"Error normalizing variance {variance}: {e}, returning 0.5"
            )
            return 0.5
    
    def normalize_coverage(self, coverage_dict: Dict[str, float]) -> float:
        """
        Normalize coverage metrics to weighted confidence score.
        
        Combines entity, number, and token overlap into single coverage confidence
        using empirically-determined weights:
        - Entities: 0.4 (most important for factual accuracy)
        - Numbers: 0.3 (critical for quantitative claims)
        - Tokens: 0.3 (general semantic overlap)
        
        Formula: confidence = entities*0.4 + numbers*0.3 + tokens_overlap*0.3
        
        Args:
            coverage_dict: Dictionary with keys 'entities', 'numbers', 'tokens_overlap'
                          Each value should be in [0, 1] range
        
        Returns:
            Weighted coverage confidence in [0, 1]
            Returns 0.0 if all components are missing
        
        Edge Cases:
            - Missing keys → Treated as 0.0 (no coverage)
            - NaN values → Treated as 0.0
            - Out of range values → Clipped to [0, 1]
        
        Example:
            >>> coverage = {'entities': 0.8, 'numbers': 0.6, 'tokens_overlap': 0.7}
            >>> normalizer.normalize_coverage(coverage)
            0.71  # Weighted average
        """
        if not coverage_dict:
            self.logger.warning("Coverage dict is empty, returning 0.0")
            return 0.0
        
        try:
            # Extract components with defaults
            entities = coverage_dict.get('entities', 0.0)
            numbers = coverage_dict.get('numbers', 0.0)
            tokens_overlap = coverage_dict.get('tokens_overlap', 0.0)
            
            # Handle NaN values
            if np.isnan(entities):
                self.logger.debug("Entities coverage is NaN, using 0.0")
                entities = 0.0
            if np.isnan(numbers):
                self.logger.debug("Numbers coverage is NaN, using 0.0")
                numbers = 0.0
            if np.isnan(tokens_overlap):
                self.logger.debug("Tokens overlap is NaN, using 0.0")
                tokens_overlap = 0.0
            
            # Clip to valid range [0, 1]
            entities = float(np.clip(entities, 0.0, 1.0))
            numbers = float(np.clip(numbers, 0.0, 1.0))
            tokens_overlap = float(np.clip(tokens_overlap, 0.0, 1.0))
            
            # Calculate weighted average
            confidence = (
                entities * self.coverage_weights['entities'] +
                numbers * self.coverage_weights['numbers'] +
                tokens_overlap * self.coverage_weights['tokens_overlap']
            )
            
            # Ensure result is in valid range (should already be, but be defensive)
            confidence = float(np.clip(confidence, 0.0, 1.0))
            
            return confidence
            
        except (ValueError, TypeError) as e:
            self.logger.error(
                f"Error normalizing coverage {coverage_dict}: {e}, returning 0.0"
            )
            return 0.0
    
    def normalize_nli(self, nli_dict: Dict[str, float]) -> Tuple[float, float]:
        """
        Extract NLI support and contradiction confidence scores.
        
        Extracts the entailment and contradiction probabilities from NLI model output.
        These are already in [0, 1] range from the softmax output, so no transformation
        is needed - just extraction and validation.
        
        Args:
            nli_dict: Dictionary with keys 'entail', 'contradict', 'neutral'
                     Each value is a probability in [0, 1]
        
        Returns:
            Tuple of (support_confidence, contradict_confidence)
            Both values in [0, 1] range
            Returns (0.5, 0.5) if data is missing or invalid
        
        Edge Cases:
            - Missing keys → Return (0.5, 0.5) neutral
            - NaN values → Return (0.5, 0.5) neutral
            - Out of range values → Clip to [0, 1]
        
        Example:
            >>> nli = {'entail': 0.8, 'contradict': 0.1, 'neutral': 0.1}
            >>> normalizer.normalize_nli(nli)
            (0.8, 0.1)
        """
        if not nli_dict:
            self.logger.warning("NLI dict is empty, returning neutral (0.5, 0.5)")
            return (0.5, 0.5)
        
        try:
            # Extract components
            entail = nli_dict.get('entail', None)
            contradict = nli_dict.get('contradict', None)
            
            # Handle missing values
            if entail is None:
                self.logger.debug("NLI entailment is None, using 0.5")
                entail = 0.5
            if contradict is None:
                self.logger.debug("NLI contradiction is None, using 0.5")
                contradict = 0.5
            
            # Handle NaN values
            if np.isnan(entail):
                self.logger.warning("NLI entailment is NaN, using 0.5")
                entail = 0.5
            if np.isnan(contradict):
                self.logger.warning("NLI contradiction is NaN, using 0.5")
                contradict = 0.5
            
            # Clip to valid range [0, 1]
            entail = float(np.clip(entail, 0.0, 1.0))
            contradict = float(np.clip(contradict, 0.0, 1.0))
            
            return (entail, contradict)
            
        except (ValueError, TypeError) as e:
            self.logger.error(
                f"Error extracting NLI scores {nli_dict}: {e}, returning (0.5, 0.5)"
            )
            return (0.5, 0.5)
