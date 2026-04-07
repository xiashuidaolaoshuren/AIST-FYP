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

from typing import Dict, Optional, Union, List, Tuple, Any
import traceback
import time
from dataclasses import dataclass

from src.utils.data_structures import Claim, EvidenceChunk, VerifierSignal
from src.utils.config import Config
from src.utils.logger import setup_logger
from src.verification.intrinsic_uncertainty import IntrinsicUncertaintyDetector
from src.verification.retrieval_grounded import RetrievalGroundedDetector
from src.verification.nli_detector import NLIDetector
from src.verification.lettuce_detector import LettuceDetectDetector
from src.verification.self_agreement import SelfAgreementDetector


@dataclass
class _BatchPreparedState:
    """Prepared verifier batch state before NLI inference."""
    results: List[Optional[VerifierSignal]]
    prepared_items: List[Dict[str, Any]]
    nli_pending: List[Tuple[int, str, str]]


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
            self.contradiction_first_fusion = bool(
                getattr(config.verification, 'contradiction_first_fusion', False)
            )
            sentence_retrieval_cfg = getattr(config.verification, 'sentence_retrieval', None)
            agg_config = getattr(config.verification, 'aggregator', None)
            default_contradiction_threshold = float(
                getattr(agg_config, 'contradiction_threshold', 0.5)
            )
            self.contradiction_priority_threshold = float(
                getattr(
                    config.verification,
                    'contradiction_priority_threshold',
                    default_contradiction_threshold
                )
            )
            self.contradiction_priority_margin = float(
                getattr(config.verification, 'contradiction_priority_margin', 0.0)
            )
            self.coherence_threshold = float(
                getattr(config.verification, 'coherence_threshold', 0.6)
            )
            self.contradiction_dominance_factor = float(
                getattr(config.verification, 'contradiction_dominance_factor', 1.5)
            )
            self.min_entailment_for_dominance = float(
                getattr(config.verification, 'min_entailment_for_dominance', 0.0)
            )
            self.global_contradiction_entailment_floor = float(
                getattr(config.verification, 'global_contradiction_entailment_floor', 0.003)
            )
            self.artifact_coverage_floor = float(
                getattr(config.verification, 'artifact_coverage_floor', 0.5)
            )
            self.nli_ambiguity_threshold = float(
                getattr(config.verification, 'nli_ambiguity_threshold', 0.85)
            )
            self.min_entailment_context_threshold = float(
                getattr(config.verification, 'min_entailment_context_threshold', 0.0)
            )
            self.cross_chunk_conflict_entailment_threshold = float(
                getattr(config.verification, 'cross_chunk_conflict_entailment_threshold', 0.42)
            )
            self.min_dense_score_for_contradiction = float(
                getattr(sentence_retrieval_cfg, 'min_dense_score_for_contradiction', 0.0)
            )
            self.strict_logits = bool(
                getattr(getattr(config.verification, 'intrinsic', None), 'strict_logits', False)
            )
            nli_cfg = getattr(config.verification, 'nli', None)
            self.bidirectional_nli = bool(getattr(nli_cfg, 'bidirectional', False))
            self.nli_backend = str(getattr(nli_cfg, 'backend', 'deberta')).strip().lower()
        else:
            self.verify_all_evidence = False
            self.aggregation_method = 'max'
            self.contradiction_first_fusion = False
            self.contradiction_priority_threshold = 0.5
            self.contradiction_priority_margin = 0.0
            self.coherence_threshold = 0.6
            self.contradiction_dominance_factor = 1.5
            self.min_entailment_for_dominance = 0.0
            self.global_contradiction_entailment_floor = 0.003
            self.artifact_coverage_floor = 0.5
            self.nli_ambiguity_threshold = 0.85
            self.min_entailment_context_threshold = 0.0
            self.cross_chunk_conflict_entailment_threshold = 0.42
            self.min_dense_score_for_contradiction = 0.0
            self.strict_logits = False
            self.bidirectional_nli = False
            self.nli_backend = 'deberta'

        # Per-module enable flags (default: all enabled)
        self.module_flags = self._resolve_module_flags()

        # Detector placeholders for consistent status/introspection
        self.uncertainty_detector = None
        self.grounded_detector = None
        self.nli_detector = None
        self.self_agreement_detector = None
        
        if not self.enabled:
            self.logger.warning("VerifierHub initialized but verification is disabled")
            return

        # Initialize Month 3 detectors
        try:
            self.logger.info("Initializing verification detectors...")
            
            # Initialize Intrinsic Uncertainty Detector
            if self.module_flags['intrinsic']:
                self.uncertainty_detector = IntrinsicUncertaintyDetector(config)
                self.logger.info("✓ IntrinsicUncertaintyDetector initialized")
            else:
                self.logger.info("IntrinsicUncertaintyDetector disabled by config")
            
            # Initialize Retrieval-Grounded Detector
            if self.module_flags['grounded']:
                self.grounded_detector = RetrievalGroundedDetector(config)
                self.logger.info("✓ RetrievalGroundedDetector initialized")
            else:
                self.logger.info("RetrievalGroundedDetector disabled by config")
            
            # Initialize NLI Detector (Month 4, Task 3)
            if self.module_flags['nli']:
                try:
                    if self.nli_backend == 'lettucedetect':
                        self.nli_detector = LettuceDetectDetector(config)
                        self.logger.info("✓ LettuceDetectDetector initialized")
                    else:
                        self.nli_detector = NLIDetector(config)
                        self.logger.info("✓ NLIDetector initialized")
                except Exception as e:
                    error_msg = (
                        "NLI detector initialization failed for backend "
                        f"'{self.nli_backend}': {str(e)}"
                    )
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg) from e
            else:
                self.logger.info("NLIDetector disabled by config")
            
            # Initialize Self-Agreement Detector (Month 4, Task 4)
            if self.module_flags['self_agreement']:
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
            else:
                self.logger.info("SelfAgreementDetector disabled by config")
            
            self.logger.info("VerifierHub initialization complete")
            
        except Exception as e:
            error_msg = f"Failed to initialize VerifierHub detectors: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            raise RuntimeError(error_msg) from e

    def detect_nli_batch(self, claim_texts: List[str], evidence_texts: List[str]) -> List[Dict[str, float]]:
        """Run NLI batch inference using configured directionality policy."""
        if self.nli_detector is None:
            return []
        if self.bidirectional_nli and hasattr(self.nli_detector, 'detect_batch_bidirectional'):
            return self.nli_detector.detect_batch_bidirectional(claim_texts, evidence_texts)
        return self.nli_detector.detect_batch(claim_texts, evidence_texts)

    def _detect_nli_single(self, claim_text: str, evidence_text: str) -> Dict[str, float]:
        """Run single-pair NLI using configured directionality policy."""
        if self.nli_detector is None:
            return {'entailment': 0.33, 'neutral': 0.34, 'contradiction': 0.33}
        if self.bidirectional_nli and hasattr(self.nli_detector, 'detect_bidirectional'):
            return self.nli_detector.detect_bidirectional(claim_text=claim_text, evidence_text=evidence_text)
        return self.nli_detector.detect(claim_text=claim_text, evidence_text=evidence_text)

    def _resolve_module_flags(self) -> Dict[str, bool]:
        """Resolve per-detector enable flags from config with safe defaults."""
        defaults = {
            'intrinsic': True,
            'grounded': True,
            'nli': True,
            'self_agreement': True,
        }

        if not hasattr(self.config, 'verification'):
            return defaults

        verification = self.config.verification
        modules_cfg = getattr(verification, 'modules', None)

        if modules_cfg is not None:
            return {
                name: bool(getattr(modules_cfg, name, defaults[name]))
                for name in defaults
            }

        # Backward-compatible fallback for older config style
        return {
            'intrinsic': bool(getattr(getattr(verification, 'intrinsic', None), 'enabled', True)),
            'grounded': bool(getattr(getattr(verification, 'grounded', None), 'enabled', True)),
            'nli': bool(getattr(getattr(verification, 'nli', None), 'enabled', True)),
            'self_agreement': bool(getattr(getattr(verification, 'self_agreement', None), 'enabled', True)),
        }
    
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

    def verify_claims_batch(self, claim_records: List[Dict[str, Any]]) -> List[Optional[VerifierSignal]]:
        """
        Verify multiple claim records in one batch.

        Each record should include:
            - claim: Claim
            - evidence: EvidenceChunk or List[EvidenceChunk]
            - metadata: dict (optional)

        The method batches NLI calls for single-evidence records when possible,
        while preserving existing fallback behavior and output ordering.
        """
        if not self.enabled:
            return [None] * len(claim_records)

        if not claim_records:
            return []

        start_ts = time.perf_counter()
        prepared_state = self.prepare_verification_collect_nli(claim_records)

        nli_scores: List[Dict[str, float]] = []
        if self.nli_detector is not None and prepared_state.nli_pending:
            try:
                nli_scores = self.detect_nli_batch(
                    [item[1] for item in prepared_state.nli_pending],
                    [item[2] for item in prepared_state.nli_pending],
                )
            except Exception as e:
                self.logger.warning("Batch NLI failed in verify_claims_batch: %s", str(e))
                nli_scores = []

        results = self.finalize_from_nli_scores(prepared_state, nli_scores)
        elapsed_ms = (time.perf_counter() - start_ts) * 1000
        self.logger.debug(
            "verify_claims_batch complete: total=%d, single=%d, nli_batched=%d, elapsed_ms=%.2f",
            len(claim_records),
            len(prepared_state.prepared_items),
            len(prepared_state.nli_pending),
            elapsed_ms,
        )
        return results

    def prepare_verification_collect_nli(self, claim_records: List[Dict[str, Any]]) -> _BatchPreparedState:
        """Prepare non-NLI verifier signals and collect pending NLI pairs."""
        results: List[Optional[VerifierSignal]] = [None] * len(claim_records)

        single_items: List[Dict[str, Any]] = []
        multi_items: List[Dict[str, Any]] = []

        for idx, record in enumerate(claim_records):
            claim = record.get('claim')
            evidence = record.get('evidence')
            metadata = record.get('metadata') or {}

            if claim is None or evidence is None:
                continue

            is_multi_evidence = isinstance(evidence, list)
            if is_multi_evidence and self.verify_all_evidence and len(evidence) > 1:
                multi_items.append({
                    'index': idx,
                    'claim': claim,
                    'evidence': evidence,
                    'metadata': metadata,
                })
                continue

            if is_multi_evidence:
                if len(evidence) == 0:
                    continue
                evidence_chunk = evidence[0]
            else:
                evidence_chunk = evidence

            if evidence_chunk is None:
                continue

            single_items.append({
                'index': idx,
                'claim': claim,
                'evidence': evidence_chunk,
                'metadata': metadata,
            })

        for item in multi_items:
            try:
                results[item['index']] = self._verify_claim_multi(
                    item['claim'],
                    item['evidence'],
                    item['metadata'],
                )
            except Exception as e:
                self.logger.error(
                    "Batch verify multi-evidence failed for claim %s: %s",
                    getattr(item['claim'], 'claim_id', 'unknown'),
                    str(e),
                )
                self.logger.debug(traceback.format_exc())

        prepared_items: List[Dict[str, Any]] = []
        nli_pending: List[Tuple[int, str, str]] = []

        # Pass 1: Prepare non-SA signals and collect SA batch payload.
        precomputed_single: List[Dict[str, Any]] = []
        sa_claim_texts: List[str] = []
        sa_queries: List[str] = []
        sa_evidence_list: List[List[EvidenceChunk]] = []
        sa_pending_indices: List[int] = []

        for item in single_items:
            claim = item['claim']
            evidence = item['evidence']
            metadata = item['metadata']

            try:
                disable_intrinsic = bool(metadata.get('disable_intrinsic_uncertainty'))
                if self.uncertainty_detector is None or disable_intrinsic:
                    uncertainty_signal = {'mean_entropy': 0.0}
                else:
                    try:
                        uncertainty_signal = self.uncertainty_detector.compute_signal(
                            claim, evidence, metadata
                        )
                    except Exception:
                        if self.strict_logits:
                            raise
                        uncertainty_signal = {'mean_entropy': 0.0}

                if self.grounded_detector is None:
                    grounded_signal = {
                        'entities': 0.0,
                        'numbers': 0.0,
                        'tokens_overlap': 0.0,
                    }
                else:
                    try:
                        grounded_signal = self.grounded_detector.compute_signal(
                            claim, evidence, metadata
                        )
                    except Exception:
                        grounded_signal = {
                            'entities': 0.0,
                            'numbers': 0.0,
                            'tokens_overlap': 0.0,
                        }

                precomputed_single.append(
                    {
                        'index': item['index'],
                        'claim': claim,
                        'evidence': evidence,
                        'metadata': metadata,
                        'uncertainty': uncertainty_signal,
                        'grounded': grounded_signal,
                    }
                )

                query = metadata.get('original_query', None)
                if self.self_agreement_detector is not None and query:
                    sa_pending_indices.append(item['index'])
                    sa_claim_texts.append(claim.text)
                    sa_queries.append(query)
                    sa_evidence_list.append([evidence])
            except Exception as e:
                self.logger.error(
                    "Batch verify prepare failed for claim %s: %s",
                    getattr(claim, 'claim_id', 'unknown'),
                    str(e),
                )
                self.logger.debug(traceback.format_exc())

        # Pass 1.5: Batch self-agreement detection for single-evidence claims.
        sa_results_by_index: Dict[int, Dict[str, float]] = {}
        if self.self_agreement_detector is not None and sa_pending_indices:
            try:
                if hasattr(self.self_agreement_detector, 'detect_batch'):
                    batch_consistency = self.self_agreement_detector.detect_batch(
                        claim_texts=sa_claim_texts,
                        queries=sa_queries,
                        evidence_chunks_list=sa_evidence_list,
                    )
                    for idx, result in zip(sa_pending_indices, batch_consistency):
                        sa_results_by_index[idx] = result
                else:
                    for idx, claim_text, query, evidence_chunks in zip(
                        sa_pending_indices,
                        sa_claim_texts,
                        sa_queries,
                        sa_evidence_list,
                    ):
                        sa_results_by_index[idx] = self.self_agreement_detector.detect(
                            claim_text=claim_text,
                            query=query,
                            evidence_chunks=evidence_chunks,
                        )
            except Exception:
                for idx in sa_pending_indices:
                    sa_results_by_index[idx] = {'variance': None}

        # Pass 2: Build prepared items and NLI pending tuples.
        for item in precomputed_single:
            claim = item['claim']
            evidence = item['evidence']
            consistency_signal = sa_results_by_index.get(item['index'], {'variance': None})

            prepared_item = {
                'index': item['index'],
                'claim': claim,
                'evidence': evidence,
                'uncertainty': item['uncertainty'],
                'grounded': item['grounded'],
                'consistency': consistency_signal,
                'needs_nli': self.nli_detector is not None,
            }
            prepared_items.append(prepared_item)

            if prepared_item['needs_nli']:
                nli_pending.append((item['index'], claim.text, evidence.text))

        return _BatchPreparedState(
            results=results,
            prepared_items=prepared_items,
            nli_pending=nli_pending,
        )

    def finalize_from_nli_scores(
        self,
        prepared_state: _BatchPreparedState,
        nli_scores: List[Dict[str, float]],
    ) -> List[Optional[VerifierSignal]]:
        """Finalize prepared verifier items with provided NLI scores."""
        nli_idx = 0
        for item in prepared_state.prepared_items:
            if item.get('needs_nli', False):
                if nli_idx < len(nli_scores):
                    nli_signal = nli_scores[nli_idx]
                else:
                    nli_signal = {'entailment': 0.33, 'neutral': 0.34, 'contradiction': 0.33}
                nli_idx += 1
            else:
                nli_signal = None

            signal = VerifierSignal(
                claim_id=item['claim'].claim_id,
                doc_id=item['evidence'].doc_id,
                sent_id=item['evidence'].sent_id,
                nli=nli_signal,
                coverage=item['grounded'],
                uncertainty=item['uncertainty'],
                consistency=item['consistency'],
                citation_span_match=item['grounded'].get('tokens_overlap', 0.0),
                numeric_check=item['grounded'].get('numbers', 0.0) >= 0.999,
            )
            prepared_state.results[item['index']] = signal

        return prepared_state.results
    
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
            disable_intrinsic = bool(metadata.get('disable_intrinsic_uncertainty'))
            if self.uncertainty_detector is None:
                uncertainty_signal = {'mean_entropy': 0.0}
            elif disable_intrinsic:
                uncertainty_signal = {'mean_entropy': 0.0}
            else:
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
            if self.grounded_detector is None:
                grounded_signal = {
                    'entities': 0.0,
                    'numbers': 0.0,
                    'tokens_overlap': 0.0
                }
            else:
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
                    nli_scores = self._detect_nli_single(
                        claim_text=claim.text,
                        evidence_text=evidence.text,
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
                numeric_check=grounded_signal.get('numbers', 0.0) >= 0.999
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
            nli_batch_scores = None

            # Precompute NLI scores in batch for all evidence chunks when available.
            if self.nli_detector is not None:
                try:
                    claim_texts = [claim.text] * len(evidence_list)
                    evidence_texts = [chunk.text for chunk in evidence_list]
                    nli_batch_scores = self.detect_nli_batch(claim_texts, evidence_texts)
                except Exception as e:
                    self.logger.warning(f"Batch NLI precompute failed, falling back to per-chunk NLI: {str(e)}")
                    nli_batch_scores = None
            
            # Collect signals from all chunks
            for idx, chunk in enumerate(evidence_list):
                try:
                    # Compute uncertainty and grounded signals for this chunk
                    if self.uncertainty_detector is None or metadata.get('disable_intrinsic_uncertainty'):
                        uncertainty_signal = {'mean_entropy': 0.0}
                    else:
                        uncertainty_signal = self.uncertainty_detector.compute_signal(
                            claim, chunk, metadata
                        )
                    if self.grounded_detector is None:
                        grounded_signal = {'entities': 0.0, 'numbers': 0.0, 'tokens_overlap': 0.0}
                    else:
                        grounded_signal = self.grounded_detector.compute_signal(
                            claim, chunk, metadata
                        )
                    
                    # Compute NLI signal if detector available
                    nli_signal = None
                    if self.nli_detector is not None:
                        try:
                            if nli_batch_scores is not None and idx < len(nli_batch_scores):
                                nli_signal = nli_batch_scores[idx]
                            else:
                                nli_signal = self._detect_nli_single(
                                    claim_text=claim.text,
                                    evidence_text=chunk.text,
                                )
                        except Exception as e:
                            self.logger.warning(f"NLI detection failed for chunk: {str(e)}")
                            nli_signal = {'entailment': 0.33, 'neutral': 0.34, 'contradiction': 0.33}
                    
                    # Store per-chunk details
                    chunk_data = {
                        'doc_id': chunk.doc_id,
                        'sent_id': chunk.sent_id,
                        'score_dense': getattr(chunk, 'score_dense', None),
                        'coverage': grounded_signal,
                        'uncertainty': uncertainty_signal,
                        'citation_span_match': grounded_signal.get('tokens_overlap', 0.0),
                        'numeric_check': grounded_signal.get('numbers', 0.0) >= 0.999
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
                        'score_dense': None,
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
            
            aggregated, primary_chunk_idx = self._aggregate_signals(per_chunk_signals)
            
            # Determine which chunk to use for doc_id/sent_id stamping
            # Use primary evidence chunk if available (max method), otherwise use top-ranked
            if primary_chunk_idx is not None:
                source_chunk = evidence_list[primary_chunk_idx]
                self.logger.info(
                    f"Using primary evidence chunk {primary_chunk_idx} "
                    f"(doc_id={source_chunk.doc_id}, sent_id={source_chunk.sent_id})"
                )
            else:
                source_chunk = evidence_list[0]
                self.logger.debug("Using top-ranked chunk (mean aggregation method)")
            
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
                        self.logger.warning("No original_query in metadata, skipping self-agreement")
                except Exception as e:
                    self.logger.error(f"Self-agreement failed: {str(e)}")
                    self.logger.debug(traceback.format_exc())
            
            # Construct aggregated VerifierSignal using source chunk's identifiers
            signal = VerifierSignal(
                claim_id=claim.claim_id,
                doc_id=source_chunk.doc_id,
                sent_id=source_chunk.sent_id,
                nli=aggregated.get('nli', None),  # Task 3: Include aggregated NLI scores
                coverage=aggregated['coverage'],
                uncertainty=aggregated['uncertainty'],
                consistency=consistency_signal,  # Task 4: Self-agreement consistency
                citation_span_match=aggregated['citation_span_match'],
                numeric_check=aggregated['numeric_check'],
                per_chunk_signals=per_chunk_signals,  # Store detailed breakdown
                primary_nli_mode=aggregated.get('primary_nli_mode'),
                max_entailment_chunk_idx=aggregated.get('max_entailment_chunk_idx'),
                max_contradiction_chunk_idx=aggregated.get('max_contradiction_chunk_idx'),
                nli_coherence_score=aggregated.get('nli_coherence_score'),
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
    
    def _aggregate_signals(self, per_chunk_signals: List[Dict]) -> Tuple[Dict, Optional[int]]:
        """
        Aggregate per-chunk signals using configured method.
        
        Aggregation semantics:
        - MAX method (optimistic): Take best-case values
          * Coverage (higher=better): MAX
          * Entropy (lower=better): MIN
          * Tracks which chunk contributed the max entailment (primary evidence)
        - MEAN method: Average all values (no primary evidence tracking)
        
        Args:
            per_chunk_signals: List of per-chunk signal dicts
        
        Returns:
            Tuple of (aggregated_dict, primary_chunk_index)
            - aggregated_dict: Aggregated signal dict with coverage, uncertainty, etc.
            - primary_chunk_index: Index of chunk that contributed max entailment (MAX method only),
                                   or None for MEAN method
        """
        # Extract values for aggregation
        entities = [s['coverage'].get('entities', 0.0) for s in per_chunk_signals]
        numbers = [s['coverage'].get('numbers', 0.0) for s in per_chunk_signals]
        tokens = [s['coverage'].get('tokens_overlap', 0.0) for s in per_chunk_signals]
        coverage_scores = [
            (float(entity) * 0.4) + (float(number) * 0.3) + (float(token) * 0.3)
            for entity, number, token in zip(entities, numbers, tokens)
        ]
        entropies = [s['uncertainty'].get('mean_entropy', 0.0) for s in per_chunk_signals]
        citations = [s['citation_span_match'] for s in per_chunk_signals]
        numeric_checks = [s['numeric_check'] for s in per_chunk_signals]
        
        # Extract NLI scores if available
        nli_available = any('nli' in s for s in per_chunk_signals)
        if nli_available:
            entailments = [s.get('nli', {}).get('entailment', 0.33) for s in per_chunk_signals]
            neutrals = [s.get('nli', {}).get('neutral', 0.33) for s in per_chunk_signals]
            raw_contradictions = [s.get('nli', {}).get('contradiction', 0.33) for s in per_chunk_signals]
            reverse_entailments = [s.get('nli', {}).get('reverse_entailment', 0.0) for s in per_chunk_signals]
            contradictions = [
                max(0.0, float(c) * (1.0 - float(rev_e)))
                for c, rev_e in zip(raw_contradictions, reverse_entailments)
            ]
        
        primary_chunk_idx = None
        primary_nli_mode = None
        max_entailment_chunk_idx = None
        max_contradiction_chunk_idx = None
        nli_coherence_score = None
        
        if self.aggregation_method == 'max':
            # Optimistic: best coverage, lowest uncertainty, highest entailment
            # Track primary evidence chunk (entailment-first by default,
            # contradiction-first when explicitly enabled)
            if nli_available:
                max_entailment = max(entailments)
                entailment_chunk_idx = entailments.index(max_entailment)
                max_coverage_score = max(coverage_scores) if coverage_scores else 0.0

                # Optional guard: ignore low-ranked dense retrieval evidence when
                # selecting contradiction peaks to reduce noise-induced false positives.
                eligible_contradictions: List[Tuple[int, float]] = []
                for idx, contradiction in enumerate(contradictions):
                    score_dense = per_chunk_signals[idx].get('score_dense')
                    if score_dense is None:
                        eligible_contradictions.append((idx, contradiction))
                        continue
                    try:
                        if float(score_dense) >= self.min_dense_score_for_contradiction:
                            eligible_contradictions.append((idx, contradiction))
                    except (TypeError, ValueError):
                        eligible_contradictions.append((idx, contradiction))

                if eligible_contradictions:
                    contradiction_chunk_idx, max_contradiction = max(
                        eligible_contradictions,
                        key=lambda item: item[1],
                    )
                else:
                    # Respect dense-score gating: if no chunk is eligible, do not
                    # promote contradiction from low-ranked retrieval noise.
                    max_contradiction = 0.0
                    contradiction_chunk_idx = contradictions.index(max(contradictions))

                contradiction_peak_neutral = neutrals[contradiction_chunk_idx]
                contradiction_peak_entailment = entailments[contradiction_chunk_idx]
                contradiction_peak_reverse_entailment = reverse_entailments[contradiction_chunk_idx]

                max_entailment_chunk_idx = entailment_chunk_idx
                max_contradiction_chunk_idx = contradiction_chunk_idx
                nli_coherence_score = self._compute_nli_coherence_score(per_chunk_signals)

                use_contradiction_primary = (
                    self.contradiction_first_fusion
                    and max_contradiction >= self.contradiction_priority_threshold
                    and max_contradiction >= (max_entailment + self.contradiction_priority_margin)
                    and max_entailment >= self.min_entailment_context_threshold
                )

                if use_contradiction_primary:
                    same_chunk_signal = contradiction_chunk_idx == entailment_chunk_idx
                    contradiction_doc_id = str(
                        per_chunk_signals[contradiction_chunk_idx].get('doc_id', '')
                    )
                    entailment_doc_id = str(
                        per_chunk_signals[entailment_chunk_idx].get('doc_id', '')
                    )
                    same_paragraph_signal = (
                        bool(contradiction_doc_id)
                        and contradiction_doc_id == entailment_doc_id
                    )
                    coherent_contradiction = (
                        nli_coherence_score is not None
                        and nli_coherence_score >= self.coherence_threshold
                    )
                    dominant_contradiction = (
                        max_contradiction >= (max_entailment * self.contradiction_dominance_factor)
                        and max_entailment >= self.min_entailment_for_dominance
                    )
                    ambiguous_same_chunk = (
                        (same_chunk_signal or same_paragraph_signal)
                        and max_entailment >= self.nli_ambiguity_threshold
                        and max_contradiction >= self.nli_ambiguity_threshold
                    )

                    if ambiguous_same_chunk:
                        primary_chunk_idx = entailment_chunk_idx
                        primary_nli_mode = 'ambiguous'
                    elif same_chunk_signal or coherent_contradiction or dominant_contradiction:
                        primary_chunk_idx = contradiction_chunk_idx
                        primary_nli_mode = 'contradiction'
                    else:
                        primary_chunk_idx = entailment_chunk_idx
                        primary_nli_mode = 'entailment'

                    # Option C: cross-chunk conflict guard.
                    # When contradiction and entailment peaks come from DIFFERENT evidence
                    # chunks (cross-source) and substantial entailment exists elsewhere,
                    # the contradiction may be a cross-predicate / cross-event artifact
                    # rather than a genuine claim error. Demote to 'ambiguous' so the
                    # aggregator suppression guard prevents a false-positive verdict.
                    if (
                        primary_nli_mode == 'contradiction'
                        and not same_chunk_signal
                        and not same_paragraph_signal
                        and max_entailment >= self.cross_chunk_conflict_entailment_threshold
                    ):
                        primary_chunk_idx = entailment_chunk_idx
                        primary_nli_mode = 'ambiguous'
                        self.logger.debug(
                            "Option C: cross-chunk conflict detected — "
                            f"contradiction_chunk={contradiction_chunk_idx} "
                            f"(contradiction={max_contradiction:.3f}), "
                            f"entailment_chunk={entailment_chunk_idx} "
                            f"(entailment={max_entailment:.3f}) — "
                            "demoting to ambiguous"
                        )

                    # Global entailment floor: when ALL evidence chunks give near-zero
                    # entailment, any contradiction verdict (same_chunk, coherent, or
                    # dominant path) is unreliable — the claim likely adds extra info
                    # that DeBERTa cannot entail rather than containing a genuine error.
                    # Override to 'entailment' to suppress FP verdicts (e.g. the Blue
                    # Bell "third time ... which might be linked" pattern).
                    if (
                        primary_nli_mode == 'contradiction'
                        and max_entailment < self.global_contradiction_entailment_floor
                        and max_coverage_score >= self.artifact_coverage_floor
                    ):
                        primary_chunk_idx = entailment_chunk_idx
                        primary_nli_mode = 'entailment'
                        self.logger.debug(
                            "Global entailment floor: max_entailment=%.4f < %.4f and "
                            "max_coverage_score=%.4f >= %.4f — overriding contradiction to entailment",
                            max_entailment,
                            self.global_contradiction_entailment_floor,
                            max_coverage_score,
                            self.artifact_coverage_floor,
                        )
                else:
                    primary_chunk_idx = entailment_chunk_idx
                    primary_nli_mode = 'entailment'

                if use_contradiction_primary:
                    if primary_nli_mode == 'contradiction':
                        self.logger.debug(
                            "Primary evidence switched to contradiction-first: "
                            f"chunk {primary_chunk_idx} with contradiction={max_contradiction:.3f}, "
                            f"entailment_max={max_entailment:.3f}, coherence={nli_coherence_score:.3f}"
                        )
                    elif primary_nli_mode == 'ambiguous':
                        self.logger.debug(
                            "Contradiction-first candidate suppressed due to ambiguous same-source NLI: "
                            f"chunk={primary_chunk_idx}, contradiction={max_contradiction:.3f}, "
                            f"entailment={max_entailment:.3f}, "
                            f"ambiguity_threshold={self.nli_ambiguity_threshold:.3f}"
                        )
                    else:
                        self.logger.debug(
                            "Contradiction-first candidate suppressed due to incoherence: "
                            f"contradiction_chunk={contradiction_chunk_idx}, "
                            f"entailment_chunk={entailment_chunk_idx}, "
                            f"contradiction={max_contradiction:.3f}, entailment={max_entailment:.3f}, "
                            f"coherence={nli_coherence_score:.3f}, "
                            f"coherence_threshold={self.coherence_threshold:.3f}, "
                            f"dominance_factor={self.contradiction_dominance_factor:.3f}"
                        )
                else:
                    self.logger.debug(
                        f"Primary evidence: chunk {primary_chunk_idx} "
                        f"with entailment={max_entailment:.3f}"
                    )
            
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
                    'entailment': max_entailment,  # Best entailment
                    'neutral': min(neutrals),  # Least neutral (most decisive)
                    # Neutral/entailment at the contradiction peak chunk. These are
                    # used by downstream guards to judge contradiction quality.
                    'neutral_contradiction_peak': contradiction_peak_neutral,
                    'entailment_contradiction_peak': contradiction_peak_entailment,
                    'reverse_entailment_contradiction_peak': contradiction_peak_reverse_entailment,
                    'contradiction': (
                        max_contradiction
                        if self.contradiction_first_fusion
                        else min(contradictions)
                    )
                }
                result['primary_nli_mode'] = primary_nli_mode
                result['max_entailment_chunk_idx'] = max_entailment_chunk_idx
                result['max_contradiction_chunk_idx'] = max_contradiction_chunk_idx
                result['nli_coherence_score'] = nli_coherence_score
            return result, primary_chunk_idx
        else:  # mean
            # Average all scores (no primary evidence tracking for mean method)
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
                result['primary_nli_mode'] = None
                result['max_entailment_chunk_idx'] = None
                result['max_contradiction_chunk_idx'] = None
                result['nli_coherence_score'] = self._compute_nli_coherence_score(per_chunk_signals)
            return result, None

    def _compute_nli_coherence_score(self, per_chunk_signals: List[Dict]) -> float:
        """Compute [0,1] coherence of NLI verdict tendency across evidence chunks."""
        if not per_chunk_signals:
            return 0.5

        if len(per_chunk_signals) == 1:
            return 1.0

        votes = []
        for chunk_signal in per_chunk_signals:
            nli = chunk_signal.get('nli')
            if not nli:
                continue
            entail = float(nli.get('entailment', 0.0))
            contradict = float(nli.get('contradiction', 0.0))
            if contradict > entail:
                votes.append('contradiction')
            elif entail > contradict:
                votes.append('entailment')
            else:
                votes.append('neutral')

        if not votes:
            return 0.5

        counts = {}
        for vote in votes:
            counts[vote] = counts.get(vote, 0) + 1
        majority_ratio = max(counts.values()) / len(votes)

        return float(majority_ratio)
    
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
            'configured_intrinsic': self.module_flags['intrinsic'],
            'configured_grounded': self.module_flags['grounded'],
            'configured_nli': self.module_flags['nli'],
            'configured_self_agreement': self.module_flags['self_agreement'],
            'intrinsic': self.uncertainty_detector is not None,
            'grounded': self.grounded_detector is not None,
            'nli': self.nli_detector is not None,
            'self_agreement': self.self_agreement_detector is not None
        }
