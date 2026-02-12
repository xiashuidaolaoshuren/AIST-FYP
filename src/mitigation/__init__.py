"""
Mitigation module for hallucination detection.

This module provides strategies for mitigating hallucinations in LLM outputs:
- Evidence re-ranking: Improve retrieval quality using verification feedback
- Claim filtering: Remove contradictory claims from final output
"""

from .re_ranker import EvidenceReRanker
from .claim_filter import ClaimFilter

__all__ = ['EvidenceReRanker', 'ClaimFilter']
