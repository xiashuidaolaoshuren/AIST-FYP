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

from typing import Dict, Optional, Union, List
import traceback

from src.utils.data_structures import Claim, EvidenceChunk, VerifierSignal
from src.utils.config import Config
from src.utils.logger import setup_logger
from src.verification.intrinsic_uncertainty import IntrinsicUncertaintyDetector
from src.verification.retrieval_grounded import RetrievalGroundedDetector
from src.verification.nli_detector import NLIDetector
from src.verification.self_agreement import SelfAgreementDetector


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
    
    def __init__(self, config: Config, generator=None):
        """
        Initialize the VerifierHub with all detectors.
        
        Args:
            config: Configuration object with verification settings
            generator: Optional GeneratorWrapper for self-agreement detection
        
        Raises:
            ValueError: If config is invalid or missing required fields
            RuntimeError: If detector initialization fails
        """
        self.config = config
        self.generator = generator
        self.logger = setup_logger(__name__)
        
        # Check if verification is enabled
        self.enabled = (
            hasattr(config, 'verification') and 
            hasattr(config.verification, 'enabled') and 
            config.verification.enabled
        )
        
        # Read Task 2 config flags
        if hasattr(config, 'verification'):
            self.verify_all_evidence = getattr(config.verification, 'verify_all_evidence', False)
            self.aggregation_method = getattr(config.verification, 'aggregation_method', 'max')
            self.strict_logits = bool(
                getattr(getattr(config.verification, 'intrinsic', None), 'strict_logits', False)
            )
        else:
            self.verify_all_evidence = False
            self.aggregation_method = 'max'
            self.strict_logits = False
        
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
            
            # Initialize NLI Detector (Month 4, Task 3)
            try:
                self.nli_detector = NLIDetector(config)
                self.logger.info("✓ NLIDetector initialized")
            except Exception as e:
                self.logger.warning(f"NLIDetector initialization failed: {str(e)}")
                self.logger.warning("Continuing without NLI detector")
                self.nli_detector = None
            
            # Initialize Self-Agreement Detector (Month 4, Task 4)
            try:
                if generator is not None:
                    self.self_agreement_detector = SelfAgreementDetector(config, generator)
                    self.logger.info("✓ SelfAgreementDetector initialized")
                else:
                    self.self_agreement_detector = None
                    self.logger.warning("Generator not provided, Self-Agreement detector disabled")
            except Exception as e:
                self.logger.warning(f"SelfAgreementDetector initialization failed: {str(e)}")
                self.logger.warning("Continuing without Self-Agreement detector")
                self.self_agreement_detector = None
            
            self.logger.info("VerifierHub initialization complete")
            
        except Exception as e:
            error_msg = f"Failed to initialize VerifierHub detectors: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            raise RuntimeError(error_msg) from e
    
    def verify_claim(
        self,
        claim: Claim,
        evidence: Union[EvidenceChunk, List[EvidenceChunk]],
        metadata: Dict
    ) -> Optional[VerifierSignal]:
        """
        Verify a claim against evidence using all enabled detectors.
        
        Supports both single-chunk and multi-chunk verification:
        - Single chunk: Verify claim against one evidence chunk (backward compatible)
        - Multi-chunk: Verify against all chunks and aggregate (if verify_all_evidence=True)
        
        Args:
            claim: Claim object to verify
            evidence: Single EvidenceChunk or list of EvidenceChunks
            metadata: Generation metadata (contains tokens, logits, etc.)
        
        Returns:
            VerifierSignal with aggregated scores (if multi-chunk) or single scores,
            or None if verification is disabled or fails critically
        
        Raises:
            ValueError: If inputs are invalid (e.g., None claim or evidence)
        
        Example:
            >>> # Single chunk
            >>> signal = hub.verify_claim(claim, top_chunk, metadata)
            >>> # Multi-chunk
            >>> signal = hub.verify_claim(claim, all_chunks, metadata)
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
        
        # Determine if evidence is a list
        is_multi_evidence = isinstance(evidence, list)
        
        # Multi-evidence verification path
        if is_multi_evidence and self.verify_all_evidence and len(evidence) > 1:
            self.logger.debug(
                f"Multi-evidence verification: {len(evidence)} chunks with {self.aggregation_method} aggregation"
            )
            return self._verify_claim_multi(claim, evidence, metadata)
        
        # Single-evidence verification path (backward compatible)
        if is_multi_evidence:
            if len(evidence) == 0:
                self.logger.warning("Empty evidence list provided, returning None")
                return None
            # Use first chunk if list provided but multi-verification disabled
            single_evidence = evidence[0]
            self.logger.debug("Multi-evidence disabled or single chunk, using top-ranked evidence")
        else:
            single_evidence = evidence
        
        return self._verify_single_chunk(claim, single_evidence, metadata)
    
    def _verify_single_chunk(
        self,
        claim: Claim,
        evidence: EvidenceChunk,
        metadata: Dict
    ) -> Optional[VerifierSignal]:
        """
        Verify claim against a single evidence chunk.
        
        Internal method used by verify_claim for single-chunk verification.
        
        Args:
            claim: Claim object
            evidence: Single evidence chunk
            metadata: Generation metadata
        
        Returns:
            VerifierSignal or None
        """
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
                self.logger.info(
                    "verifier_uncertainty",
                    extra={
                        "event": "verifier_uncertainty",
                        "data": {
                            "claim_id": claim.claim_id,
                            "mean_entropy": uncertainty_signal.get('mean_entropy', 0.0)
                        }
                    }
                )
            except Exception as e:
                self.logger.error(
                    f"IntrinsicUncertaintyDetector failed for claim {claim.claim_id}: {str(e)}"
                )
                self.logger.debug(traceback.format_exc())
                if self.strict_logits:
                    raise
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
                self.logger.info(
                    "verifier_grounded",
                    extra={
                        "event": "verifier_grounded",
                        "data": {
                            "claim_id": claim.claim_id,
                            "entities": grounded_signal.get('entities', 0.0),
                            "numbers": grounded_signal.get('numbers', 0.0),
                            "tokens_overlap": grounded_signal.get('tokens_overlap', 0.0)
                        }
                    }
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
            
            # Compute NLI signal (Month 4, Task 3)
            nli_signal = None
            if self.nli_detector is not None:
                try:
                    nli_scores = self.nli_detector.detect(
                        claim_text=claim.text,
                        evidence_text=evidence.text
                    )
                    nli_signal = nli_scores  # Dict with entailment, neutral, contradiction
                    self.logger.debug(
                        f"NLI signal computed for claim {claim.claim_id}: "
                        f"entailment={nli_signal.get('entailment', 0.0):.2f}, "
                        f"contradiction={nli_signal.get('contradiction', 0.0):.2f}, "
                        f"neutral={nli_signal.get('neutral', 0.0):.2f}"
                    )
                    self.logger.info(
                        "verifier_nli",
                        extra={
                            "event": "verifier_nli",
                            "data": {
                                "claim_id": claim.claim_id,
                                "entailment": nli_signal.get('entailment', 0.0),
                                "contradiction": nli_signal.get('contradiction', 0.0),
                                "neutral": nli_signal.get('neutral', 0.0)
                            }
                        }
                    )
                except Exception as e:
                    self.logger.error(
                        f"NLIDetector failed for claim {claim.claim_id}: {str(e)}"
                    )
                    self.logger.debug(traceback.format_exc())
                    # Use default fallback values (neutral)
                    nli_signal = {
                        'entailment': 0.33,
                        'neutral': 0.34,
                        'contradiction': 0.33
                    }
            else:
                self.logger.debug("NLI detector not available, skipping NLI signal")
            
            # Compute self-agreement signal (Month 4, Task 4)
            consistency_signal = {'variance': None}
            if self.self_agreement_detector is not None:
                try:
                    # Extract query from metadata
                    query = metadata.get('original_query', None)
                    if query:
                        self.logger.debug(f"Computing self-agreement for claim {claim.claim_id}")
                        sa_result = self.self_agreement_detector.detect(
                            claim_text=claim.text,
                            query=query,
                            evidence_chunks=[evidence] if isinstance(evidence, EvidenceChunk) else evidence
                        )
                        consistency_signal = sa_result
                        self.logger.debug(
                            f"Self-agreement signal computed for claim {claim.claim_id}: "
                            f"score={sa_result.get('score', 0.0):.3f}, "
                            f"variance={sa_result.get('variance', 0.0):.3f}"
                        )
                        self.logger.info(
                            "verifier_self_agreement",
                            extra={
                                "event": "verifier_self_agreement",
                                "data": {
                                    "claim_id": claim.claim_id,
                                    "score": sa_result.get('score', None),
                                    "variance": sa_result.get('variance', None),
                                    "samples_generated": sa_result.get('samples_generated', None)
                                }
                            }
                        )
                    else:
                        self.logger.warning(f"No original_query in metadata for claim {claim.claim_id}, skipping self-agreement")
                except Exception as e:
                    self.logger.error(
                        "SelfAgreementDetector failed for claim %s: %s",
                        claim.claim_id, str(e)
                    )
                    self.logger.debug(traceback.format_exc())
                    consistency_signal = {'variance': None}
            else:
                self.logger.debug("Self-agreement detector not available, skipping consistency signal")
            
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
            error_msg = f"Critical error in _verify_single_chunk for claim {claim.claim_id}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            # Return None to allow pipeline to continue without this signal
            return None
    
    def _verify_claim_multi(
        self,
        claim: Claim,
        evidence_list: List[EvidenceChunk],
        metadata: Dict
    ) -> Optional[VerifierSignal]:
        """
        Verify claim against multiple evidence chunks and aggregate results.
        
        Internal method that iterates over all evidence chunks, collects signals,
        and aggregates them using the configured method (max or mean).
        
        Args:
            claim: Claim object
            evidence_list: List of evidence chunks
            metadata: Generation metadata
        
        Returns:
            Aggregated VerifierSignal or None
        """
        try:
            per_chunk_signals = []
            
            # Collect signals from all chunks
            for chunk in evidence_list:
                try:
                    # Compute uncertainty and grounded signals for this chunk
                    uncertainty_signal = self.uncertainty_detector.compute_signal(
                        claim, chunk, metadata
                    )
                    grounded_signal = self.grounded_detector.compute_signal(
                        claim, chunk, metadata
                    )
                    
                    # Compute NLI signal if detector available
                    nli_signal = None
                    if self.nli_detector is not None:
                        try:
                            nli_signal = self.nli_detector.detect(
                                claim_text=claim.text,
                                evidence_text=chunk.text
                            )
                        except Exception as e:
                            self.logger.warning(f"NLI detection failed for chunk: {str(e)}")
                            nli_signal = {'entailment': 0.33, 'neutral': 0.34, 'contradiction': 0.33}
                    
                    # Store per-chunk details
                    chunk_data = {
                        'doc_id': chunk.doc_id,
                        'sent_id': chunk.sent_id,
                        'coverage': grounded_signal,
                        'uncertainty': uncertainty_signal,
                        'citation_span_match': grounded_signal.get('tokens_overlap', 0.0),
                        'numeric_check': grounded_signal.get('numbers', 0.0) == 1.0
                    }
                    if nli_signal is not None:
                        chunk_data['nli'] = nli_signal
                    
                    per_chunk_signals.append(chunk_data)
                    
                except Exception as e:
                    self.logger.warning(
                        f"Failed to compute signal for chunk {chunk.doc_id}#{chunk.sent_id}: {str(e)}"
                    )
                    # Add fallback values for failed chunk
                    fallback = {
                        'doc_id': chunk.doc_id,
                        'sent_id': chunk.sent_id,
                        'coverage': {'entities': 0.0, 'numbers': 0.0, 'tokens_overlap': 0.0},
                        'uncertainty': {'mean_entropy': 0.0},
                        'citation_span_match': 0.0,
                        'numeric_check': False
                    }
                    if self.nli_detector is not None:
                        fallback['nli'] = {'entailment': 0.33, 'neutral': 0.34, 'contradiction': 0.33}
                    per_chunk_signals.append(fallback)
            
            # Aggregate signals
            if not per_chunk_signals:
                self.logger.error("No valid signals collected from evidence chunks")
                return None
            
            aggregated = self._aggregate_signals(per_chunk_signals)
            
            # Compute self-agreement consistency (Task 4)
            consistency_signal = {'variance': None}
            if self.self_agreement_detector is not None:
                try:
                    query = metadata.get('original_query', None)
                    if query:
                        self.logger.debug(f"Computing self-agreement for claim {claim.claim_id} (multi-evidence)")
                        sa_result = self.self_agreement_detector.detect(
                            claim_text=claim.text,
                            query=query,
                            evidence_chunks=evidence_list
                        )
                        consistency_signal = sa_result
                        self.logger.debug(
                            f"Self-agreement computed: score={sa_result.get('score', 0.0):.3f}, "
                            f"variance={sa_result.get('variance', 0.0):.3f}"
                        )
                    else:
                        self.logger.warning(f"No original_query in metadata, skipping self-agreement")
                except Exception as e:
                    self.logger.error(f"Self-agreement failed: {str(e)}")
                    self.logger.debug(traceback.format_exc())
            
            # Use the top-ranked chunk's identifiers for the aggregated signal
            top_chunk = evidence_list[0]
            
            # Construct aggregated VerifierSignal
            signal = VerifierSignal(
                claim_id=claim.claim_id,
                doc_id=top_chunk.doc_id,
                sent_id=top_chunk.sent_id,
                nli=aggregated.get('nli', None),  # Task 3: Include aggregated NLI scores
                coverage=aggregated['coverage'],
                uncertainty=aggregated['uncertainty'],
                consistency=consistency_signal,  # Task 4: Self-agreement consistency
                citation_span_match=aggregated['citation_span_match'],
                numeric_check=aggregated['numeric_check'],
                per_chunk_signals=per_chunk_signals  # Store detailed breakdown
            )
            
            self.logger.info(
                f"Multi-evidence signal aggregated for claim {claim.claim_id}: "
                f"{len(per_chunk_signals)} chunks, method={self.aggregation_method}"
            )
            return signal
            
        except Exception as e:
            error_msg = f"Critical error in _verify_claim_multi for claim {claim.claim_id}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            return None
    
    def _aggregate_signals(self, per_chunk_signals: List[Dict]) -> Dict:
        """
        Aggregate per-chunk signals using configured method.
        
        Aggregation semantics:
        - MAX method (optimistic): Take best-case values
          * Coverage (higher=better): MAX
          * Entropy (lower=better): MIN
        - MEAN method: Average all values
        
        Args:
            per_chunk_signals: List of per-chunk signal dicts
        
        Returns:
            Aggregated signal dict with coverage, uncertainty, etc.
        """
        # Extract values for aggregation
        entities = [s['coverage'].get('entities', 0.0) for s in per_chunk_signals]
        numbers = [s['coverage'].get('numbers', 0.0) for s in per_chunk_signals]
        tokens = [s['coverage'].get('tokens_overlap', 0.0) for s in per_chunk_signals]
        entropies = [s['uncertainty'].get('mean_entropy', 0.0) for s in per_chunk_signals]
        citations = [s['citation_span_match'] for s in per_chunk_signals]
        numeric_checks = [s['numeric_check'] for s in per_chunk_signals]
        
        # Extract NLI scores if available
        nli_available = any('nli' in s for s in per_chunk_signals)
        if nli_available:
            entailments = [s.get('nli', {}).get('entailment', 0.33) for s in per_chunk_signals]
            neutrals = [s.get('nli', {}).get('neutral', 0.33) for s in per_chunk_signals]
            contradictions = [s.get('nli', {}).get('contradiction', 0.33) for s in per_chunk_signals]
        
        if self.aggregation_method == 'max':
            # Optimistic: best coverage, lowest uncertainty, highest entailment
            result = {
                'coverage': {
                    'entities': max(entities),
                    'numbers': max(numbers),
                    'tokens_overlap': max(tokens)
                },
                'uncertainty': {
                    'mean_entropy': min(entropies)  # Lower entropy = more confident
                },
                'citation_span_match': max(citations),
                'numeric_check': any(numeric_checks)  # True if any chunk has numeric match
            }
            if nli_available:
                result['nli'] = {
                    'entailment': max(entailments),  # Best entailment
                    'neutral': min(neutrals),  # Least neutral (most decisive)
                    'contradiction': min(contradictions)  # Least contradiction
                }
            return result
        else:  # mean
            # Average all scores
            result = {
                'coverage': {
                    'entities': sum(entities) / len(entities),
                    'numbers': sum(numbers) / len(numbers),
                    'tokens_overlap': sum(tokens) / len(tokens)
                },
                'uncertainty': {
                    'mean_entropy': sum(entropies) / len(entropies)
                },
                'citation_span_match': sum(citations) / len(citations),
                'numeric_check': sum(numeric_checks) / len(numeric_checks) >= 0.5  # True if majority match
            }
            if nli_available:
                result['nli'] = {
                    'entailment': sum(entailments) / len(entailments),
                    'neutral': sum(neutrals) / len(neutrals),
                    'contradiction': sum(contradictions) / len(contradictions)
                }
            return result
    
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
