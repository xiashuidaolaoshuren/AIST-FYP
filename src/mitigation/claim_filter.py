"""
Claim Filter for Mitigation.

This module implements ClaimFilter, which removes contradictory claims
from the final answer text by replacing them with placeholder text.

The filtering strategy:
1. Identifies claims with status="Contradictory" from ClaimDecision
2. Processes claims in reverse order by character span (to avoid index corruption)
3. Replaces contradictory claim text with configurable placeholder
4. Returns filtered text and count of removed claims

This approach ensures transparency (users see that claims were removed)
while preventing harmful hallucinations from reaching the final output.
"""

from typing import List, Tuple, Dict
import logging

from src.utils.data_structures import Claim, ClaimDecision
from src.utils.config import Config


logger = logging.getLogger(__name__)


class ClaimFilter:
    """
    Filters contradictory claims from answer text.
    
    This class implements a safe claim removal strategy that:
    - Identifies contradictory claims based on ClaimDecision verdicts
    - Replaces them with a transparent placeholder text
    - Preserves supported and low-confidence claims unchanged
    
    The filtering is done in reverse order (by character span) to avoid
    index corruption when replacing text segments.
    
    Attributes:
        placeholder: Text to replace contradictory claims with
        enabled: Whether filtering is enabled
    
    Example:
        ```python
        config = Config()
        filter = ClaimFilter(config)
        
        # Filter contradictory claims from answer
        filtered_text, removed_count = filter.filter_answer(
            answer_text="Paris is the capital. Berlin is in Asia.",
            claims=[claim1, claim2],
            decisions=[decision1, decision2]  # decision2 is Contradictory
        )
        
        # Output: "Paris is the capital. [CLAIM REMOVED: Contradictory]"
        # removed_count: 1
        ```
    """
    
    def __init__(self, config: Config):
        """
        Initialize the ClaimFilter.
        
        Args:
            config: Configuration object containing mitigation.filter settings
                   - placeholder: Text for removed claims (default: "[CLAIM REMOVED: Contradictory]")
                   - enabled: Enable/disable filtering (default: True)
        """
        # Load configuration with defaults
        filter_config = config.get('mitigation', {}).get('filter', {})
        
        self.placeholder = filter_config.get('placeholder', '[CLAIM REMOVED: Contradictory]')
        self.enabled = filter_config.get('enabled', True)
        
        logger.info(
            f"ClaimFilter initialized: enabled={self.enabled}, "
            f"placeholder='{self.placeholder}'"
        )
    
    def filter_answer(
        self,
        answer_text: str,
        claims: List[Claim],
        decisions: List[ClaimDecision]
    ) -> Tuple[str, int]:
        """
        Filter contradictory claims from answer text.
        
        This method identifies claims marked as "Contradictory" and replaces
        their text spans with the configured placeholder.
        
        Claims are processed in REVERSE order (by answer_char_span start position)
        to avoid index corruption when modifying the text. This ensures that
        earlier character spans remain valid after later replacements.
        
        Args:
            answer_text: Original answer text containing all claims
            claims: List of Claim objects with character span information
            decisions: List of ClaimDecision objects with verdict status
                      Must correspond 1-to-1 with claims list
        
        Returns:
            Tuple[str, int]: (filtered_text, removed_count)
                - filtered_text: Answer with contradictory claims replaced
                - removed_count: Number of claims removed
        
        Raises:
            ValueError: If claims and decisions lists have different lengths
        
        Example:
            ```python
            answer = "Paris is in France. London is in Asia."
            claims = [
                Claim(claim_id="c1", text="Paris is in France.", answer_char_span=[0, 19], ...),
                Claim(claim_id="c2", text="London is in Asia.", answer_char_span=[20, 38], ...)
            ]
            decisions = [
                ClaimDecision(claim_id="c1", status="Supported", ...),
                ClaimDecision(claim_id="c2", status="Contradictory", ...)
            ]
            
            filtered, count = filter.filter_answer(answer, claims, decisions)
            # filtered: "Paris is in France. [CLAIM REMOVED: Contradictory]"
            # count: 1
            ```
        """
        if not self.enabled:
            logger.debug("Claim filtering is disabled, returning original text")
            return answer_text, 0
        
        if len(claims) != len(decisions):
            raise ValueError(
                f"Claims and decisions must have same length. "
                f"Got {len(claims)} claims and {len(decisions)} decisions"
            )
        
        if not claims:
            logger.debug("No claims to filter, returning original text")
            return answer_text, 0
        
        logger.debug(
            f"Filtering {len(claims)} claims from answer (length={len(answer_text)})"
        )
        
        # Build claim_id -> claim mapping for quick lookup
        claim_dict = {claim.claim_id: claim for claim in claims}
        
        # Identify contradictory decisions
        contradictory_decisions = [
            decision for decision in decisions
            if decision.status == 'Contradictory'
        ]
        
        if not contradictory_decisions:
            logger.info("No contradictory claims found, returning original text")
            return answer_text, 0
        
        logger.info(
            f"Found {len(contradictory_decisions)} contradictory claims to remove"
        )
        
        # Sort decisions by claim char_span in REVERSE order
        # This prevents index corruption when replacing text
        sorted_decisions = sorted(
            contradictory_decisions,
            key=lambda d: claim_dict[d.claim_id].answer_char_span[0],
            reverse=True
        )
        
        # Apply filtering
        filtered_text = answer_text
        removed_count = 0
        
        for decision in sorted_decisions:
            claim = claim_dict[decision.claim_id]
            start, end = claim.answer_char_span
            
            # Validate span
            if start < 0 or end > len(filtered_text) or start >= end:
                logger.warning(
                    f"Invalid char_span [{start}, {end}] for claim {claim.claim_id}. "
                    f"Text length: {len(filtered_text)}. Skipping."
                )
                continue
            
            # Replace claim text with placeholder
            filtered_text = (
                filtered_text[:start] +
                self.placeholder +
                filtered_text[end:]
            )
            
            removed_count += 1
            
            logger.debug(
                f"Removed claim {claim.claim_id} at span [{start}, {end}]: "
                f"'{claim.text[:50]}...'"
            )
        
        logger.info(
            f"Filtering complete: removed {removed_count} claims. "
            f"Original length: {len(answer_text)}, "
            f"Filtered length: {len(filtered_text)}"
        )
        
        return filtered_text, removed_count
    
    def get_filtering_summary(
        self,
        claims: List[Claim],
        decisions: List[ClaimDecision]
    ) -> Dict[str, int]:
        """
        Get summary statistics about claims without modifying text.
        
        Useful for analysis and reporting.
        
        Args:
            claims: List of Claim objects
            decisions: List of ClaimDecision objects
        
        Returns:
            Dict with keys:
                - total_claims: Total number of claims
                - supported: Number of supported claims
                - contradictory: Number of contradictory claims
                - low_confidence: Number of low-confidence claims
                - would_remove: Number of claims that would be removed if filtering enabled
        
        Example:
            ```python
            summary = filter.get_filtering_summary(claims, decisions)
            print(f"Would remove: {summary['would_remove']}/{summary['total_claims']}")
            ```
        """
        if len(claims) != len(decisions):
            raise ValueError(
                f"Claims and decisions must have same length. "
                f"Got {len(claims)} claims and {len(decisions)} decisions"
            )
        
        status_counts = {
            'Supported': 0,
            'Contradictory': 0,
            'Low Confidence': 0
        }
        
        for decision in decisions:
            status = decision.status
            if status in status_counts:
                status_counts[status] += 1
        
        return {
            'total_claims': len(claims),
            'supported': status_counts['Supported'],
            'contradictory': status_counts['Contradictory'],
            'low_confidence': status_counts['Low Confidence'],
            'would_remove': status_counts['Contradictory']
        }
