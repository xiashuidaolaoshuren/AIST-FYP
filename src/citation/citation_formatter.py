"""
CitationFormatter for CiteEval-compatible citation injection.

This module implements citation formatting that aligns with the CiteEval benchmark format.
Citations are injected as bracketed indices [1][2][3] into the answer text based on
claim character spans, with support for deduplication and proper punctuation handling.
"""

from typing import List, Dict, Tuple, Optional, Any
from src.utils.data_structures import Claim, EvidenceChunk, VerifierSignal
from src.utils.config import Config
from src.utils.logger import setup_logger


class CitationFormatter:
    """
    Formats answer text with bracketed citations aligned to claims for CiteEval evaluation.
    
    This class takes an answer string, extracted claims, and evidence mappings, then:
    1. Builds a global deduplicated passage list
    2. Injects citation indices [1][2][3] after each claim
    3. Handles punctuation correctly (inserts before .!? if present)
    4. Supports export to CiteEval JSON format
    
    Citation format follows CiteEval standard: consecutive brackets with no spaces [1][2][3]
    
    Attributes:
        config: Configuration object
        max_citations_per_claim: Maximum number of citations to add per claim (default: 3)
        min_score_dense_for_citation: Minimum dense score required to attach citations
        logger: Logger instance
    
    Example:
        >>> config = Config()
        >>> formatter = CitationFormatter(config)
        >>> claims = [Claim(...), Claim(...)]
        >>> evidence_map = {'claim_1': [EvidenceChunk(...), ...], ...}
        >>> result = formatter.format_with_citations(answer_text, claims, evidence_map)
        >>> print(result['formatted_text'])
        'The cat sat on the mat[1][2]. It was very comfortable[3].'
    """
    
    def __init__(self, config: Config):
        """
        Initialize CitationFormatter with configuration.
        
        Loads max_citations_per_claim from config if available, otherwise uses default (3).
        
        Args:
            config: Configuration object with citation settings
        """
        self.config = config
        self.logger = setup_logger(__name__)

        citation_cfg = getattr(config, 'citation', None)
        if citation_cfg is None and hasattr(config, 'verification'):
            citation_cfg = getattr(config.verification, 'citation', None)
        
        # Load max citations per claim from config
        if citation_cfg is not None and hasattr(citation_cfg, 'max_citations_per_claim'):
            self.max_citations_per_claim = citation_cfg.max_citations_per_claim
        else:
            self.max_citations_per_claim = 3
            self.logger.info(
                "No citation.max_citations_per_claim in config, using default: 3"
            )

        if citation_cfg is not None and hasattr(citation_cfg, 'min_score_dense_for_citation'):
            self.min_score_dense_for_citation = float(citation_cfg.min_score_dense_for_citation)
        else:
            self.min_score_dense_for_citation = 0.0
            self.logger.info(
                "No citation.min_score_dense_for_citation in config, using default: 0.0"
            )

        if citation_cfg is not None and hasattr(citation_cfg, 'min_nli_entailment_for_citation'):
            self.min_nli_entailment_for_citation = float(citation_cfg.min_nli_entailment_for_citation)
        else:
            self.min_nli_entailment_for_citation = 0.0
            self.logger.info(
                "No citation.min_nli_entailment_for_citation in config, using default: 0.0"
            )
        
        self.logger.info(
            "CitationFormatter initialized with "
            f"max_citations_per_claim={self.max_citations_per_claim}, "
            f"min_score_dense_for_citation={self.min_score_dense_for_citation:.3f}, "
            f"min_nli_entailment_for_citation={self.min_nli_entailment_for_citation:.3f}"
        )
    
    def format_with_citations(
        self,
        answer_text: str,
        claims: List[Claim],
        evidence_map: Dict[str, List[EvidenceChunk]],
        verifier_signals: Optional[List[VerifierSignal]] = None,
    ) -> Dict[str, Any]:
        """
        Inject bracketed citations into answer text aligned to claim positions.
        
        Process:
        1. Build global passage list (deduplicated by doc_id#sent_id)
        2. Sort claims by position in reverse order (to avoid index shifting)
        3. For each claim, get top-N evidence and format as [1][2][3]
        4. Insert citations after claim text (before punctuation if present)
        
        Args:
            answer_text: Original answer string without citations
            claims: List of Claim objects with char_span positions
            evidence_map: Dictionary mapping claim_id to list of EvidenceChunk objects
        
        Returns:
            Dictionary with keys:
            - formatted_text: Answer with citations injected
            - citation_map: Dict mapping claim_id to list of passage indices
            - passage_list: List of deduplicated passages [{text, title}, ...]
        
        Example:
            >>> result = formatter.format_with_citations(
            ...     "The cat sat. It was happy.",
            ...     [claim1, claim2],
            ...     {'c1': [evidence1, evidence2], 'c2': [evidence3]}
            ... )
            >>> print(result['formatted_text'])
            'The cat sat[1][2]. It was happy[3].'
        """
        # Step 1: Build global passage list and index mapping
        passage_list, doc_to_index = self._build_passage_list(evidence_map)
        
        # Step 2: Sort claims by position (reverse to avoid index shifting)
        sorted_claims = sorted(claims, key=lambda c: c.answer_char_span[0], reverse=True)
        
        # Step 3: Check for overlapping spans (data quality warning)
        self._check_overlapping_spans(sorted_claims)
        
        # Step 4: Build citation map and insert citations
        citation_map = {}
        formatted_text = answer_text
        nli_lookup: Dict[Tuple[str, str, int], float] = {}

        if verifier_signals:
            for signal in verifier_signals:
                # Support both VerifierSignal dataclass objects and plain dicts
                # (dicts occur when signals are serialised via .to_dict() before
                # being stored in pipeline_output["verifier_signals"]).
                if isinstance(signal, dict):
                    nli_scores = signal.get("nli", {}) or {}
                    signal_claim_id = signal.get("claim_id", "")
                    signal_doc_id = signal.get("doc_id", "")
                    signal_sent_id = signal.get("sent_id", 0)
                else:
                    nli_scores = getattr(signal, "nli", {}) or {}
                    signal_claim_id = signal.claim_id
                    signal_doc_id = signal.doc_id
                    signal_sent_id = signal.sent_id
                entailment_score = nli_scores.get("entailment", nli_scores.get("entail", 0.0))
                nli_lookup[(signal_claim_id, signal_doc_id, signal_sent_id)] = float(entailment_score)
        
        for claim in sorted_claims:
            claim_id = claim.claim_id

            # Get top-N evidence for this claim
            evidence_chunks = evidence_map.get(claim_id, [])
            
            if not evidence_chunks:
                self.logger.warning(
                    f"No evidence found for claim {claim.claim_id}, skipping citation"
                )
                continue

            def _evidence_sort_key(chunk: EvidenceChunk, current_claim_id: str = claim_id) -> Tuple[float, float]:
                return (
                    nli_lookup.get(
                        (current_claim_id, chunk.doc_id, chunk.sent_id),
                        -1.0 if nli_lookup else 0.0,
                    ),
                    float(getattr(chunk, "score_dense", 0.0)),
                )
            
            # Rank evidence per-claim to keep the strongest support first.
            evidence_chunks = sorted(
                evidence_chunks,
                key=_evidence_sort_key,
                reverse=True,
            )

            best_dense_score = float(getattr(evidence_chunks[0], "score_dense", 0.0)) if evidence_chunks else 0.0
            best_nli_entailment = nli_lookup.get(
                (claim.claim_id, evidence_chunks[0].doc_id, evidence_chunks[0].sent_id),
                None,
            ) if evidence_chunks else None

            if best_nli_entailment is not None:
                if best_nli_entailment < self.min_nli_entailment_for_citation:
                    self.logger.debug(
                        "Skipping citations for claim %s because best NLI entailment %.4f is below threshold %.4f",
                        claim.claim_id,
                        best_nli_entailment,
                        self.min_nli_entailment_for_citation,
                    )
                    continue
            elif best_dense_score < self.min_score_dense_for_citation:
                self.logger.debug(
                    "Skipping citations for claim %s because best dense score %.4f is below threshold %.4f",
                    claim.claim_id,
                    best_dense_score,
                    self.min_score_dense_for_citation,
                )
                continue

            # Limit to max citations per claim
            evidence_chunks = evidence_chunks[:self.max_citations_per_claim]
            
            # Get citation indices from doc_to_index mapping
            citation_indices = []
            for evidence in evidence_chunks:
                evidence_key = f"{evidence.doc_id}#{evidence.sent_id}"
                if evidence_key in doc_to_index:
                    citation_indices.append(doc_to_index[evidence_key])
                else:
                    self.logger.warning(
                        f"Evidence {evidence_key} not found in passage list for claim {claim.claim_id}"
                    )
            
            if not citation_indices:
                self.logger.warning(
                    f"No valid citations for claim {claim.claim_id}, skipping"
                )
                continue

            # Deduplicate while preserving order so we do not emit repeated [i][i].
            citation_indices = list(dict.fromkeys(citation_indices))
            
            # Format citation string: [1][2][3] (no spaces)
            citation_str = ''.join(f'[{idx}]' for idx in citation_indices)
            
            # Insert citation at claim position
            start, end = claim.answer_char_span
            
            # Validate span boundaries
            if end > len(formatted_text):
                self.logger.error(
                    f"Claim {claim.claim_id} span [{start}, {end}] exceeds text length {len(formatted_text)}, skipping"
                )
                continue
            
            # Check if claim ends with punctuation
            if end > 0 and formatted_text[end-1:end] in '.!?':
                # Insert before punctuation
                formatted_text = (
                    formatted_text[:end-1] + citation_str + formatted_text[end-1:]
                )
            else:
                # Append after claim text
                formatted_text = (
                    formatted_text[:end] + citation_str + formatted_text[end:]
                )
            
            # Store citation mapping
            citation_map[claim.claim_id] = citation_indices
            
            self.logger.debug(
                f"Inserted citations {citation_indices} for claim {claim.claim_id} at position [{start}, {end}]"
            )
        
        return {
            'formatted_text': formatted_text,
            'citation_map': citation_map,
            'passage_list': passage_list
        }
    
    def _build_passage_list(
        self,
        evidence_map: Dict[str, List[EvidenceChunk]]
    ) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
        """
        Build global deduplicated passage list from all evidence chunks.
        
        Process:
        1. Collect all evidence chunks from all claims
        2. Sort by score_dense (descending) for quality-based ordering
        3. Deduplicate using doc_id#sent_id as unique key
        4. Build passage list with 1-based indexing
        5. Create doc_id#sent_id -> index mapping
        
        Args:
            evidence_map: Dictionary mapping claim_id to list of EvidenceChunk objects
        
        Returns:
            Tuple of:
            - passage_list: List of dicts with 'text' and 'title' keys
            - doc_to_index: Dict mapping 'doc_id#sent_id' to 1-based index
        
        Example:
            >>> passage_list, doc_to_index = formatter._build_passage_list(evidence_map)
            >>> print(passage_list[0])
            {'text': 'Evidence sentence...', 'title': 'enwiki_12345'}
            >>> print(doc_to_index['enwiki_12345#2'])
            1
        """
        # Step 1: Collect all evidence chunks
        all_chunks = []
        for claim_id, evidence_list in evidence_map.items():
            all_chunks.extend(evidence_list)
        
        if not all_chunks:
            self.logger.warning("No evidence chunks found in evidence_map")
            return ([], {})
        
        # Step 2: Sort by score_dense (descending)
        all_chunks.sort(key=lambda e: e.score_dense, reverse=True)
        
        # Step 3: Deduplicate using doc_id#sent_id as unique key
        seen = set()
        unique_chunks = []
        for chunk in all_chunks:
            evidence_key = f"{chunk.doc_id}#{chunk.sent_id}"
            if evidence_key not in seen:
                seen.add(evidence_key)
                unique_chunks.append(chunk)
        
        self.logger.info(
            f"Collected {len(all_chunks)} total chunks, "
            f"deduplicated to {len(unique_chunks)} unique passages"
        )
        
        # Step 4: Build passage list (1-based indexing for CiteEval format)
        passage_list = []
        doc_to_index = {}
        
        for idx, chunk in enumerate(unique_chunks, start=1):
            # CiteEval format: {text, title}
            passage_list.append({
                'text': chunk.text,
                'title': chunk.doc_id
            })
            
            # Map doc_id#sent_id to 1-based index
            evidence_key = f"{chunk.doc_id}#{chunk.sent_id}"
            doc_to_index[evidence_key] = idx
        
        return (passage_list, doc_to_index)
    
    def export_citeeval_format(
        self,
        query: str,
        formatted_output: Dict[str, Any],
        answer_id: str
    ) -> Dict[str, Any]:
        """
        Export formatted output to CiteEval benchmark JSON format.
        
        CiteEval format specification:
        {
            "id": "sample identifier",
            "query": "user query string",
            "passages": [
                {"text": "passage content", "title": "document id"},
                ...
            ],
            "pred": "model-generated response with citations in brackets"
        }
        
        Args:
            query: Original user query
            formatted_output: Output from format_with_citations()
            answer_id: Unique identifier for this answer
        
        Returns:
            Dictionary in CiteEval format
        
        Example:
            >>> citeeval_output = formatter.export_citeeval_format(
            ...     "What is a cat?",
            ...     formatted_output,
            ...     "ans_001"
            ... )
            >>> print(citeeval_output.keys())
            dict_keys(['id', 'query', 'passages', 'pred'])
        """
        return {
            'id': answer_id,
            'query': query,
            'passages': formatted_output['passage_list'],
            'pred': formatted_output['formatted_text']
        }
    
    def _check_overlapping_spans(self, sorted_claims: List[Claim]) -> None:
        """
        Check for overlapping claim spans and log warnings for data quality.
        
        Overlapping spans indicate potential issues with claim extraction.
        This is a validation check - we still proceed with reverse-order insertion.
        
        Args:
            sorted_claims: List of claims sorted by position (reverse order)
        """
        for i in range(len(sorted_claims) - 1):
            claim_curr = sorted_claims[i]
            claim_next = sorted_claims[i + 1]
            
            # Check if current claim starts before next claim ends (overlap)
            # Remember: sorted in reverse, so curr.start < next.end means overlap
            if claim_curr.answer_char_span[0] < claim_next.answer_char_span[1]:
                self.logger.warning(
                    f"Overlapping spans detected: "
                    f"Claim {claim_curr.claim_id} [{claim_curr.answer_char_span[0]}, {claim_curr.answer_char_span[1]}] "
                    f"overlaps with Claim {claim_next.claim_id} [{claim_next.answer_char_span[0]}, {claim_next.answer_char_span[1]}]. "
                    f"This may indicate claim extraction issues."
                )
