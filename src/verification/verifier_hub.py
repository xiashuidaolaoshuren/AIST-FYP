"""
VerifierHub - Centralized Orchestration for Hallucination Detectors.

This module implements the VerifierHub class that manages initialization and
execution of all verification detectors (Intrinsic Uncertainty, Retrieval-Grounded,
NLI, Self-Agreement). It provides a clean interface for baseline_rag.py and
supports extensibility for future detectors.

Month 4 Architecture:
- Centralizes detector management
- Provides single verify_claim() interface
- Supports gradual feature rollout (Month 3: Intrinsic + Grounded, Month 4: + NLI + Self-Agreement)
"""

from typing import Dict, Optional
import traceback

from src.utils.data_structures import Claim, EvidenceChunk, VerifierSignal
from src.utils.config import Config
from src.utils.logger import setup_logger
from src.verification.intrinsic_uncertainty import IntrinsicUncertaintyDetector
from src.verification.retrieval_grounded import RetrievalGroundedDetector


class VerifierHub:
    """
    Central hub for orchestrating all verification detectors.
    
    Manages the lifecycle of verification components:
    - Initializes detectors based on config
    - Provides unified verify_claim() interface
    - Handles errors gracefully with logging
    - Supports future extensibility (NLI, Self-Agreement in Month 4)
    
    Current detectors (Month 3):
    - IntrinsicUncertaintyDetector: Entropy-based model confidence
    - RetrievalGroundedDetector: Evidence coverage (entities, numbers, tokens)
    
    Future detectors (Month 4):
    - NLIDetector: Entailment/contradiction via zero-shot NLI
    - SelfAgreementDetector: Consistency across stochastic samples
    
    Attributes:
        config: Configuration object
        enabled: Whether verification is enabled
        uncertainty_detector: Intrinsic uncertainty detector instance
        grounded_detector: Retrieval-grounded detector instance
        logger: Logger instance
    
    Example:
        >>> config = Config()
        >>> hub = VerifierHub(config)
        >>> signal = hub.verify_claim(claim, evidence, metadata)
        >>> print(f"Uncertainty: {signal.uncertainty['mean_entropy']:.3f}")
    """
    
    def __init__(self, config: Config):
        """
        Initialize the VerifierHub with all detectors.
        
        Args:
            config: Configuration object with verification settings
        
        Raises:
            ValueError: If config is invalid or missing required fields
            RuntimeError: If detector initialization fails
        """
        self.config = config
        self.logger = setup_logger(__name__)
        
        # Check if verification is enabled
        self.enabled = (
            hasattr(config, 'verification') and 
            hasattr(config.verification, 'enabled') and 
            config.verification.enabled
        )
        
        if not self.enabled:
            self.logger.warning("VerifierHub initialized but verification is disabled")
            return
        
        # Initialize Month 3 detectors
        try:
            self.logger.info("Initializing verification detectors...")
            
            # Initialize Intrinsic Uncertainty Detector
            self.uncertainty_detector = IntrinsicUncertaintyDetector(config)
            self.logger.info("✓ IntrinsicUncertaintyDetector initialized")
            
            # Initialize Retrieval-Grounded Detector
            self.grounded_detector = RetrievalGroundedDetector(config)
            self.logger.info("✓ RetrievalGroundedDetector initialized")
            
            # Future: NLI and Self-Agreement detectors (Month 4)
            self.nli_detector = None  # TODO: Initialize in Task 3
            self.self_agreement_detector = None  # TODO: Initialize in Task 4
            
            self.logger.info("VerifierHub initialization complete")
            
        except Exception as e:
            error_msg = f"Failed to initialize VerifierHub detectors: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            raise RuntimeError(error_msg) from e
    
    def verify_claim(
        self,
        claim: Claim,
        evidence: EvidenceChunk,
        metadata: Dict
    ) -> Optional[VerifierSignal]:
        """
        Verify a single claim against evidence using all enabled detectors.
        
        Executes all active detectors and combines their signals into a
        VerifierSignal object. Handles errors gracefully and logs warnings
        for failed detectors.
        
        Args:
            claim: Claim object to verify
            evidence: Evidence chunk to check against
            metadata: Generation metadata (contains tokens, logits, etc.)
        
        Returns:
            VerifierSignal object with all detector scores, or None if verification
            is disabled or fails critically
        
        Raises:
            ValueError: If inputs are invalid (e.g., None claim or evidence)
        
        Example:
            >>> signal = hub.verify_claim(claim, top_evidence, gen_metadata)
            >>> if signal:
            ...     print(f"Entity coverage: {signal.coverage['entities']:.2f}")
        """
        # Early return if verification disabled
        if not self.enabled:
            self.logger.debug("Verification disabled, skipping verify_claim")
            return None
        
        # Validate inputs
        if claim is None:
            self.logger.error("verify_claim called with None claim")
            raise ValueError("claim cannot be None")
        
        if evidence is None:
            self.logger.error("verify_claim called with None evidence")
            raise ValueError("evidence cannot be None")
        
        if metadata is None:
            self.logger.warning("verify_claim called with None metadata, some detectors may fail")
            metadata = {}
        
        try:
            # Compute intrinsic uncertainty signal
            uncertainty_signal = None
            try:
                uncertainty_signal = self.uncertainty_detector.compute_signal(
                    claim, evidence, metadata
                )
                self.logger.debug(
                    f"Uncertainty signal computed for claim {claim.claim_id}: "
                    f"mean_entropy={uncertainty_signal.get('mean_entropy', 0.0):.3f}"
                )
            except Exception as e:
                self.logger.error(
                    f"IntrinsicUncertaintyDetector failed for claim {claim.claim_id}: {str(e)}"
                )
                self.logger.debug(traceback.format_exc())
                # Use default fallback value
                uncertainty_signal = {'mean_entropy': 0.0}
            
            # Compute retrieval-grounded signal
            grounded_signal = None
            try:
                grounded_signal = self.grounded_detector.compute_signal(
                    claim, evidence, metadata
                )
                self.logger.debug(
                    f"Grounded signal computed for claim {claim.claim_id}: "
                    f"entities={grounded_signal.get('entities', 0.0):.2f}, "
                    f"numbers={grounded_signal.get('numbers', 0.0):.2f}, "
                    f"tokens={grounded_signal.get('tokens_overlap', 0.0):.2f}"
                )
            except Exception as e:
                self.logger.error(
                    f"RetrievalGroundedDetector failed for claim {claim.claim_id}: {str(e)}"
                )
                self.logger.debug(traceback.format_exc())
                # Use default fallback values
                grounded_signal = {
                    'entities': 0.0,
                    'numbers': 0.0,
                    'tokens_overlap': 0.0
                }
            
            # Future: Compute NLI signal (Month 4, Task 3)
            nli_signal = None  # TODO: self.nli_detector.compute_signal(claim, evidence)
            
            # Future: Compute self-agreement signal (Month 4, Task 4)
            consistency_signal = {'variance': None}  # TODO: self.self_agreement_detector.compute_signal(...)
            
            # Construct VerifierSignal
            signal = VerifierSignal(
                claim_id=claim.claim_id,
                doc_id=evidence.doc_id,
                sent_id=evidence.sent_id,
                nli=nli_signal,  # None for Month 3
                coverage=grounded_signal,
                uncertainty=uncertainty_signal,
                consistency=consistency_signal,
                citation_span_match=grounded_signal.get('tokens_overlap', 0.0),
                numeric_check=grounded_signal.get('numbers', 0.0) == 1.0
            )
            
            self.logger.debug(f"VerifierSignal created for claim {claim.claim_id}")
            return signal
            
        except Exception as e:
            error_msg = f"Critical error in verify_claim for claim {claim.claim_id}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            # Return None to allow pipeline to continue without this signal
            return None
    
    def is_enabled(self) -> bool:
        """
        Check if verification is enabled.
        
        Returns:
            True if verification is enabled, False otherwise
        """
        return self.enabled
    
    def get_detector_status(self) -> Dict[str, bool]:
        """
        Get status of all detectors.
        
        Returns:
            Dictionary mapping detector names to availability status
        
        Example:
            >>> status = hub.get_detector_status()
            >>> print(f"Intrinsic detector: {status['intrinsic']}")
        """
        if not self.enabled:
            return {
                'enabled': False,
                'intrinsic': False,
                'grounded': False,
                'nli': False,
                'self_agreement': False
            }
        
        return {
            'enabled': True,
            'intrinsic': self.uncertainty_detector is not None,
            'grounded': self.grounded_detector is not None,
            'nli': self.nli_detector is not None,
            'self_agreement': self.self_agreement_detector is not None
        }
