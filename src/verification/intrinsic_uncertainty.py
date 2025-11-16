"""
Intrinsic Uncertainty Detector for LLM Hallucination Detection.

This module implements entropy-based uncertainty detection by analyzing the
probability distributions over tokens during generation. High entropy indicates
model uncertainty, which correlates with potential hallucinations.

Key concepts:
- Token-level entropy: H = -Σ(p * log(p)) for each token's probability distribution
- Claim-level entropy: Average entropy across tokens in a claim
- Numerical stability: Uses log-sum-exp trick and epsilon to prevent log(0)

Reference:
- SelfCheckGPT paper (Section 3.1): Token-level probability analysis
- Shannon Entropy: Information theory measure of uncertainty
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings

from src.utils.data_structures import Claim, EvidenceChunk
from src.utils.logger import setup_logger
from src.utils.config import Config


class IntrinsicUncertaintyDetector:
    """
    Detector that measures model confidence through token-level entropy.
    
    Analyzes the probability distribution over vocabulary tokens at each
    generation step. High entropy suggests the model is uncertain about
    which token to generate, indicating potential hallucination.
    
    The detector:
    1. Aligns claim text with generated tokens
    2. Extracts logits for aligned tokens from metadata
    3. Calculates entropy for each token's probability distribution
    4. Returns mean entropy across the claim
    
    Attributes:
        config: Configuration object
        epsilon: Small constant for numerical stability (prevents log(0))
        logger: Logger instance
    
    Example:
        >>> config = Config()
        >>> detector = IntrinsicUncertaintyDetector(config)
        >>> signal = detector.compute_signal(claim, evidence, metadata)
        >>> print(f"Mean entropy: {signal['mean_entropy']:.3f}")
    """
    
    def __init__(self, config: Config):
        """
        Initialize the intrinsic uncertainty detector.
        
        Args:
            config: Configuration object with verification settings
        """
        self.config = config
        self.logger = setup_logger(__name__)
        
        # Load epsilon from config for numerical stability
        self.epsilon = float(
            config.verification.intrinsic.epsilon 
            if hasattr(config, 'verification') and hasattr(config.verification, 'intrinsic')
            else 1e-10
        )
        
        self.logger.info(
            f"IntrinsicUncertaintyDetector initialized with epsilon={self.epsilon}"
        )
    
    def compute_signal(
        self,
        claim: Claim,
        evidence: EvidenceChunk,
        metadata: Dict
    ) -> Dict[str, float]:
        """
        Compute intrinsic uncertainty signal for a claim-evidence pair.
        
        Calculates the mean entropy across tokens in the claim by:
        1. Aligning claim text to generated tokens
        2. Extracting logits for aligned tokens
        3. Computing entropy for each token
        4. Averaging across the claim
        
        Args:
            claim: Claim object with text and char span
            evidence: Evidence chunk (not used for intrinsic detection, but required for API)
            metadata: Generator metadata with 'text', 'tokens', 'logits'
        
        Returns:
            Dictionary with 'mean_entropy' key (float value)
            Returns {'mean_entropy': 0.0} for edge cases
        
        Example:
            >>> signal = detector.compute_signal(claim, evidence, metadata)
            >>> if signal['mean_entropy'] > 5.0:
            ...     print("High uncertainty detected!")
        """
        # Edge case: empty claim
        if not claim.text or not claim.text.strip():
            self.logger.warning(f"Empty claim {claim.claim_id}, returning 0.0 entropy")
            return {'mean_entropy': 0.0}
        
        # Edge case: no logits in metadata
        if 'logits' not in metadata or not metadata['logits']:
            self.logger.warning(
                f"No logits in metadata for claim {claim.claim_id}, returning 0.0 entropy"
            )
            return {'mean_entropy': 0.0}
        
        try:
            # Align claim to token indices
            token_indices = self._align_claim_tokens(claim, metadata)
            
            if not token_indices:
                # Alignment failed - use full response as fallback
                self.logger.debug(
                    f"Token alignment failed for claim {claim.claim_id}, "
                    f"using all tokens as fallback"
                )
                token_indices = list(range(len(metadata['logits'])))
            
            # Calculate entropy for each aligned token
            entropies = []
            for token_idx in token_indices:
                if token_idx < len(metadata['logits']):
                    logits = metadata['logits'][token_idx]
                    entropy = self._calculate_entropy(logits, self.epsilon)
                    entropies.append(entropy)
            
            # Edge case: no valid entropies calculated
            if not entropies:
                self.logger.warning(
                    f"No entropies calculated for claim {claim.claim_id}, "
                    f"returning 0.0"
                )
                return {'mean_entropy': 0.0}
            
            # Calculate mean entropy
            mean_entropy = float(np.mean(entropies))
            
            self.logger.debug(
                f"Claim {claim.claim_id}: {len(entropies)} tokens, "
                f"mean_entropy={mean_entropy:.3f}"
            )
            
            return {'mean_entropy': mean_entropy}
        
        except Exception as e:
            self.logger.error(
                f"Error computing uncertainty for claim {claim.claim_id}: {e}",
                exc_info=True
            )
            return {'mean_entropy': 0.0}
    
    def _align_claim_tokens(
        self,
        claim: Claim,
        metadata: Dict
    ) -> List[int]:
        """
        Align claim text to token indices in the generated sequence.
        
        Maps the claim's character span to token indices by:
        1. Extracting claim substring from generated text
        2. Reconstructing token positions from token strings
        3. Finding overlap between claim span and token positions
        
        This is complex because:
        - Tokenizer may split words differently than text boundaries
        - Special tokens (▁ for spaces in SentencePiece) need handling
        - One-off errors in alignment are common
        
        Args:
            claim: Claim with answer_char_span [start, end]
            metadata: Metadata with 'text' and 'tokens'
        
        Returns:
            List of token indices corresponding to the claim
            Empty list if alignment fails
        
        Note:
            Uses approximate matching with ±1 token tolerance for robustness
        """
        try:
            generated_text = metadata['text']
            tokens = metadata['tokens']
            
            # Extract claim substring using character span
            claim_start, claim_end = claim.answer_char_span
            
            # Validate span
            if claim_start < 0 or claim_end > len(generated_text):
                self.logger.warning(
                    f"Claim span [{claim_start}, {claim_end}] out of bounds "
                    f"for text length {len(generated_text)}"
                )
                return []
            
            claim_text = generated_text[claim_start:claim_end].strip()
            
            # Edge case: claim text doesn't match (extraction inconsistency)
            if claim_text.lower() != claim.text.strip().lower():
                self.logger.debug(
                    f"Claim text mismatch. Expected: '{claim.text[:50]}...', "
                    f"Got: '{claim_text[:50]}...'"
                )
                # Try to find claim in generated text
                claim_start = generated_text.lower().find(claim.text.lower())
                if claim_start == -1:
                    return []
                claim_end = claim_start + len(claim.text)
            
            # Build character position map for each token
            token_char_positions = []
            char_pos = 0
            
            for token_idx, token_str in enumerate(tokens):
                # Clean token (remove SentencePiece markers)
                clean_token = token_str.replace('▁', ' ').replace('<pad>', '').replace('</s>', '')
                
                if not clean_token:
                    continue
                
                # Find this token in the generated text
                token_start = generated_text.find(clean_token, char_pos)
                
                if token_start != -1:
                    token_end = token_start + len(clean_token)
                    token_char_positions.append((token_idx, token_start, token_end))
                    char_pos = token_end
            
            # Find tokens that overlap with claim span
            aligned_indices = []
            
            for token_idx, token_start, token_end in token_char_positions:
                # Check if token overlaps with claim span
                # Token overlaps if: token_start < claim_end AND token_end > claim_start
                if token_start < claim_end and token_end > claim_start:
                    aligned_indices.append(token_idx)
            
            # Apply fuzzy matching: if we got close but not exact, expand by ±1 token
            if aligned_indices:
                first_idx = aligned_indices[0]
                last_idx = aligned_indices[-1]
                
                # Expand range to include boundary tokens
                if first_idx > 0:
                    aligned_indices.insert(0, first_idx - 1)
                if last_idx < len(tokens) - 1:
                    aligned_indices.append(last_idx + 1)
            
            self.logger.debug(
                f"Aligned claim '{claim.text[:30]}...' to {len(aligned_indices)} tokens "
                f"(indices {aligned_indices[:5]}...)" if len(aligned_indices) > 5
                else f"(indices {aligned_indices})"
            )
            
            return aligned_indices
        
        except Exception as e:
            self.logger.error(f"Error in token alignment: {e}", exc_info=True)
            return []
    
    def _calculate_entropy(
        self,
        logits: np.ndarray,
        epsilon: float
    ) -> float:
        """
        Calculate Shannon entropy from logits.
        
        Computes H = -Σ(p * log(p)) where p is the probability distribution
        over vocabulary tokens. Uses numerically stable softmax and adds
        epsilon to prevent log(0) errors.
        
        Higher entropy = more uncertainty = higher hallucination risk
        
        Args:
            logits: Numpy array of shape (vocab_size,) with raw logits
            epsilon: Small constant for numerical stability
        
        Returns:
            Entropy value (float), typically in range [0, 10]
            Returns 0.0 for invalid inputs
        
        Example:
            >>> logits = np.array([2.0, 1.0, 0.5, 0.1])
            >>> entropy = detector._calculate_entropy(logits, 1e-10)
            >>> print(f"Entropy: {entropy:.3f}")
        
        Note:
            Uses log-sum-exp trick for numerical stability:
            softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
        """
        try:
            # Convert to numpy if needed (handle torch tensors)
            if not isinstance(logits, np.ndarray):
                logits = np.array(logits)
            
            # Edge case: empty logits
            if logits.size == 0:
                return 0.0
            
            # Apply softmax with log-sum-exp trick for numerical stability
            # softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
            max_logit = np.max(logits)
            exp_logits = np.exp(logits - max_logit)
            probs = exp_logits / np.sum(exp_logits)
            
            # Calculate entropy: H = -sum(p * log(p + epsilon))
            # Add epsilon to prevent log(0)
            log_probs = np.log(probs + epsilon)
            entropy = -np.sum(probs * log_probs)
            
            # Sanity check: entropy should be non-negative
            if entropy < 0:
                self.logger.warning(
                    f"Negative entropy detected: {entropy}, clamping to 0.0"
                )
                entropy = 0.0
            
            return float(entropy)
        
        except Exception as e:
            self.logger.error(f"Error calculating entropy: {e}", exc_info=True)
            return 0.0
