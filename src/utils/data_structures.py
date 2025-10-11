"""
Data structures for the Baseline RAG Module.

This module defines dataclasses that match the System_Architecture_Design.md
specifications for claim-evidence pairs and related structures.
"""

from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any


@dataclass
class EvidenceChunk:
    """
    A single piece of text retrieved from the knowledge corpus.
    
    Represents a sentence-level chunk from Wikipedia with retrieval scores
    and metadata for tracking provenance.
    
    Attributes:
        doc_id: Document identifier (e.g., "enwiki_12345")
        sent_id: Sentence identifier within the document
        text: The actual text content of the evidence
        char_start: Character offset where this chunk starts in the document
        char_end: Character offset where this chunk ends in the document
        score_dense: Dense retrieval score from semantic similarity
        rank: Ranking position in the retrieval results
        source: Source corpus name (default: "wikipedia")
        version: Version identifier for the corpus (default: "wiki_sent_v1")
        score_bm25: Optional BM25 sparse retrieval score
        score_hybrid: Optional hybrid retrieval score combining dense + sparse
    """
    doc_id: str
    sent_id: int
    text: str
    char_start: int
    char_end: int
    score_dense: float
    rank: int
    source: str = 'wikipedia'
    version: str = 'wiki_sent_v1'
    score_bm25: Optional[float] = None
    score_hybrid: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the dataclass to a dictionary."""
        return asdict(self)


@dataclass
class Claim:
    """
    An atomic, verifiable statement extracted from the LLM's draft response.
    
    Represents a single factual claim that can be verified against evidence.
    
    Attributes:
        claim_id: Unique identifier for this claim (e.g., "c_0007")
        answer_id: Identifier of the answer this claim belongs to
        text: The actual text content of the claim
        answer_char_span: Character offsets [start, end] in the answer text
        extraction_method: Method used to extract this claim (default: "rule_sentence_split_v1")
    """
    claim_id: str
    answer_id: str
    text: str
    answer_char_span: List[int]  # [start, end] - exactly 2 elements
    extraction_method: str = 'rule_sentence_split_v1'
    
    def __post_init__(self):
        """Validate that answer_char_span has exactly 2 elements."""
        if len(self.answer_char_span) != 2:
            raise ValueError(
                f"answer_char_span must have exactly 2 elements [start, end], "
                f"got {len(self.answer_char_span)}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the dataclass to a dictionary."""
        return asdict(self)


@dataclass
class ClaimEvidencePair:
    """
    Associates a claim with its corresponding retrieved evidence.
    
    Links a single claim to multiple evidence candidates, with the top-ranked
    evidence highlighted and detailed evidence spans provided.
    
    Attributes:
        claim_id: Identifier of the claim being verified
        evidence_candidates: List of evidence IDs (format: "doc_id#sent_id")
        top_evidence: The highest-ranked evidence ID
        evidence_spans: List of detailed EvidenceChunk dictionaries
    """
    claim_id: str
    evidence_candidates: List[str]
    top_evidence: str
    evidence_spans: List[Dict[str, Any]]  # List of EvidenceChunk dicts
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the dataclass to a dictionary."""
        return asdict(self)


@dataclass
class VerifierSignal:
    """
    The raw output of a single detector signal for a given claim-evidence pair.
    
    Contains all confidence signals from the Verifier Module's trainless detectors.
    
    Attributes:
        claim_id: Identifier of the claim being verified
        doc_id: Document identifier of the evidence
        sent_id: Sentence identifier of the evidence
        nli: Natural Language Inference scores (entail, contradict, neutral)
        coverage: Coverage metrics (entities, numbers, token overlap)
        uncertainty: Uncertainty metrics (mean entropy from generator)
        consistency: Self-agreement metrics (variance across samples)
        citation_span_match: Citation integrity score (0.0 to 1.0)
        numeric_check: Boolean indicating if numeric facts match
    """
    claim_id: str
    doc_id: str
    sent_id: int
    nli: Dict[str, float]  # {"entail": float, "contradict": float, "neutral": float}
    coverage: Dict[str, float]  # {"entities": float, "numbers": float, "tokens_overlap": float}
    uncertainty: Dict[str, float]  # {"mean_entropy": float}
    consistency: Dict[str, Optional[float]]  # {"variance": Optional[float]}
    citation_span_match: float
    numeric_check: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the dataclass to a dictionary."""
        return asdict(self)


@dataclass
class ClaimDecision:
    """
    The final, aggregated verdict for a single claim after verification.
    
    Contains the final decision from the rule-based aggregator combining
    all verifier signals.
    
    Attributes:
        claim_id: Identifier of the claim
        status: Final verdict (e.g., "Supported", "Contradictory", "Low Confidence")
        rationale: Human-readable explanation of the decision
        primary_evidence: The main evidence ID used for this decision
        signals_ref: List of signal IDs used in this decision
        confidence: Confidence breakdown with probabilities and band
    """
    claim_id: str
    status: str
    rationale: str
    primary_evidence: str
    signals_ref: List[str]
    confidence: Dict[str, Any]  # {"support_prob", "contradict_prob", "overall_confidence", "band"}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the dataclass to a dictionary."""
        return asdict(self)


@dataclass
class AnnotatedAnswer:
    """
    The final output object with the full answer annotated with claim decisions.
    
    Represents the complete pipeline output with all claims verified and
    mitigation actions applied.
    
    Attributes:
        answer_id: Unique identifier for this answer
        query_id: Identifier of the original query
        raw_answer: The unmodified draft response from the generator
        claims: List of ClaimDecision objects
        summary_stats: Statistics about claim verification results
        mitigation_actions: List of actions applied (e.g., "removed_contradicted_claims")
        version: Pipeline version identifier
    """
    answer_id: str
    query_id: str
    raw_answer: str
    claims: List[Dict[str, Any]]  # List of ClaimDecision dicts
    summary_stats: Dict[str, Any]
    mitigation_actions: List[str]
    version: str = "pipeline_v0.3"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the dataclass to a dictionary."""
        return asdict(self)
