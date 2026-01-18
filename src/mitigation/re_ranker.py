"""
Evidence Re-Ranker for Mitigation.

This module implements EvidenceReRanker, which re-orders retrieved evidence
chunks by combining retrieval scores with verification feedback scores.

The re-ranking improves evidence quality by promoting chunks that:
1. Have high semantic similarity (high retrieval score)
2. Show strong verification signals (high coverage + NLI entailment)

Formula:
    final_score = α × retrieval_score + β × verification_score
    
Where:
    - retrieval_score: Dense retrieval score from FAISS (score_dense)
    - verification_score: (coverage_entities + nli_entailment) / 2
    - α, β: Configurable weights (default: 0.6, 0.4)
"""

from typing import List, Dict, Optional
import logging

from src.utils.data_structures import EvidenceChunk, VerifierSignal
from src.utils.config import Config


logger = logging.getLogger(__name__)


class EvidenceReRanker:
    """
    Re-ranks evidence chunks by combining retrieval and verification scores.
    
    This class implements a hybrid re-ranking strategy that balances:
    - Retrieval quality (semantic similarity from FAISS)
    - Verification feedback (coverage and NLI entailment signals)
    
    The re-ranking helps surface evidence that is both semantically relevant
    and factually supportive, potentially improving overall answer quality.
    
    Attributes:
        alpha: Weight for retrieval score (0-1), default 0.6
        beta: Weight for verification score (0-1), default 0.4
        fallback_score: Score used when verification signal is missing
        enabled: Whether re-ranking is enabled
    
    Example:
        ```python
        config = Config()
        reranker = EvidenceReRanker(config)
        
        # Re-rank evidence using verification signals
        reranked = reranker.rerank(
            evidence_list=retrieved_chunks,
            verification_signals=signals_dict
        )
        
        # Top evidence now reflects both retrieval + verification quality
        best_evidence = reranked[0]
        ```
    """
    
    def __init__(self, config: Config):
        """
        Initialize the EvidenceReRanker.
        
        Args:
            config: Configuration object containing mitigation.reranker settings
                   - alpha: Weight for retrieval score (default: 0.6)
                   - beta: Weight for verification score (default: 0.4)
                   - fallback_score: Score for missing signals (default: 0.5)
                   - enabled: Enable/disable re-ranking (default: True)
        """
        # Load configuration with defaults
        reranker_config = config.get('mitigation', {}).get('reranker', {})
        
        self.alpha = reranker_config.get('alpha', 0.6)
        self.beta = reranker_config.get('beta', 0.4)
        self.fallback_score = reranker_config.get('fallback_score', 0.5)
        self.enabled = reranker_config.get('enabled', True)
        
        # Validate weights
        if not (0 <= self.alpha <= 1 and 0 <= self.beta <= 1):
            raise ValueError(
                f"Alpha and beta must be in [0, 1]. Got alpha={self.alpha}, beta={self.beta}"
            )
        
        if abs(self.alpha + self.beta - 1.0) > 0.01:
            logger.warning(
                f"Alpha + beta = {self.alpha + self.beta:.3f} (not 1.0). "
                f"Scores may not be normalized."
            )
        
        logger.info(
            f"EvidenceReRanker initialized: alpha={self.alpha}, beta={self.beta}, "
            f"fallback={self.fallback_score}, enabled={self.enabled}"
        )
    
    def rerank(
        self,
        evidence_list: List[EvidenceChunk],
        verification_signals: Dict[str, VerifierSignal]
    ) -> List[EvidenceChunk]:
        """
        Re-rank evidence chunks using retrieval + verification scores.
        
        This method computes a weighted final score for each evidence chunk:
            final_score = α × retrieval_score + β × verification_score
        
        Where verification_score is computed from the associated VerifierSignal:
            verification_score = (coverage_entities + nli_entailment) / 2
        
        If no verification signal exists for a chunk, fallback_score is used.
        
        Args:
            evidence_list: List of EvidenceChunk objects to re-rank
            verification_signals: Dict mapping "doc_id#sent_id" to VerifierSignal
                                 Key format must match f"{chunk.doc_id}#{chunk.sent_id}"
        
        Returns:
            List[EvidenceChunk]: Re-ranked evidence chunks, sorted by final_score
                                (highest score first)
        
        Raises:
            ValueError: If evidence_list is empty
        
        Example:
            ```python
            # Verification signals keyed by doc_id#sent_id
            signals = {
                "doc123#0": VerifierSignal(
                    coverage={'entities': 0.8, ...},
                    nli={'entailment': 0.9, ...},
                    ...
                ),
                "doc456#1": VerifierSignal(...)
            }
            
            reranked = reranker.rerank(evidence_list, signals)
            # Evidence with high verification scores move to top
            ```
        """
        if not evidence_list:
            raise ValueError("evidence_list cannot be empty")
        
        if not self.enabled:
            logger.debug("Re-ranking is disabled, returning original order")
            return evidence_list
        
        logger.debug(
            f"Re-ranking {len(evidence_list)} evidence chunks using "
            f"{len(verification_signals)} verification signals"
        )
        
        # Compute final scores for each chunk
        scored_evidence = []
        
        for chunk in evidence_list:
            # Get retrieval score
            retrieval_score = chunk.score_dense
            
            # Get verification score
            verification_score = self._compute_verification_score(
                chunk, verification_signals
            )
            
            # Compute weighted final score
            final_score = (
                self.alpha * retrieval_score +
                self.beta * verification_score
            )
            
            scored_evidence.append((chunk, final_score))
            
            logger.debug(
                f"Chunk {chunk.doc_id}#{chunk.sent_id}: "
                f"retrieval={retrieval_score:.3f}, "
                f"verification={verification_score:.3f}, "
                f"final={final_score:.3f}"
            )
        
        # Sort by final score (descending)
        scored_evidence.sort(key=lambda x: x[1], reverse=True)
        
        # Extract re-ranked chunks
        reranked_list = [chunk for chunk, score in scored_evidence]
        
        logger.info(
            f"Re-ranking complete. Top chunk: {reranked_list[0].doc_id}#{reranked_list[0].sent_id} "
            f"(score={scored_evidence[0][1]:.3f})"
        )
        
        return reranked_list
    
    def _compute_verification_score(
        self,
        chunk: EvidenceChunk,
        verification_signals: Dict[str, VerifierSignal]
    ) -> float:
        """
        Compute verification score for an evidence chunk.
        
        Formula:
            verification_score = (coverage_entities + nli_entailment) / 2
        
        If no verification signal exists, returns fallback_score.
        
        Args:
            chunk: EvidenceChunk to compute score for
            verification_signals: Dict of VerifierSignal objects
        
        Returns:
            float: Verification score in [0, 1]
        """
        # Build key: doc_id#sent_id
        key = f"{chunk.doc_id}#{chunk.sent_id}"
        
        # Check if verification signal exists
        if key not in verification_signals:
            logger.debug(
                f"No verification signal for {key}, using fallback={self.fallback_score}"
            )
            return self.fallback_score
        
        signal = verification_signals[key]
        
        # Extract coverage and NLI scores
        try:
            coverage_entities = signal.coverage.get('entities', 0.0)
            nli_entailment = signal.nli.get('entailment', 0.0)
            
            # Compute average
            verification_score = (coverage_entities + nli_entailment) / 2.0
            
            # Clamp to [0, 1]
            verification_score = max(0.0, min(1.0, verification_score))
            
            return verification_score
            
        except (AttributeError, TypeError) as e:
            logger.warning(
                f"Error extracting verification scores from signal {key}: {e}. "
                f"Using fallback={self.fallback_score}"
            )
            return self.fallback_score
    
    def get_score_breakdown(
        self,
        chunk: EvidenceChunk,
        verification_signals: Dict[str, VerifierSignal]
    ) -> Dict[str, float]:
        """
        Get detailed score breakdown for a single evidence chunk.
        
        Useful for debugging and understanding re-ranking decisions.
        
        Args:
            chunk: EvidenceChunk to analyze
            verification_signals: Dict of VerifierSignal objects
        
        Returns:
            Dict with keys: retrieval_score, verification_score, final_score
        
        Example:
            ```python
            breakdown = reranker.get_score_breakdown(chunk, signals)
            print(f"Retrieval: {breakdown['retrieval_score']:.3f}")
            print(f"Verification: {breakdown['verification_score']:.3f}")
            print(f"Final: {breakdown['final_score']:.3f}")
            ```
        """
        retrieval_score = chunk.score_dense
        verification_score = self._compute_verification_score(chunk, verification_signals)
        final_score = self.alpha * retrieval_score + self.beta * verification_score
        
        return {
            'retrieval_score': retrieval_score,
            'verification_score': verification_score,
            'final_score': final_score,
            'alpha': self.alpha,
            'beta': self.beta
        }
