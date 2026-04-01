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

from typing import Any, List, Optional, Tuple, Dict
import logging
import numpy as np

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
        self.lc_soft_filter_prob_threshold = float(
            filter_config.get('lc_soft_filter_prob_threshold', 0.0)
        )
        self.lc_soft_filter_excluded_tasks = {
            str(task).upper()
            for task in filter_config.get('lc_soft_filter_excluded_tasks', ['QA'])
        }
        self.lc_soft_filter_lc_avg_contradict_threshold = float(
            filter_config.get(
                'lc_soft_filter_lc_avg_contradict_threshold',
                self.lc_soft_filter_prob_threshold,
            )
        )
        self.lc_soft_filter_lc_avg_contradict_min_ratio = float(
            filter_config.get('lc_soft_filter_lc_avg_contradict_min_ratio', 0.30)
        )
        self.lc_soft_filter_lc_avg_contradict_min_avg_contradict_prob = float(
            filter_config.get('lc_soft_filter_lc_avg_contradict_min_avg_contradict_prob', 0.25)
        )
        self.lc_soft_filter_lc_avg_contradict_min_claims = int(
            filter_config.get('lc_soft_filter_lc_avg_contradict_min_claims', 3)
        )
        self.lc_soft_filter_lc_avg_contradict_excluded_tasks = {
            str(task).upper()
            for task in filter_config.get('lc_soft_filter_lc_avg_contradict_excluded_tasks', ['QA'])
        }
        self.lc_placeholder = filter_config.get(
            'lc_placeholder', '[CLAIM UNCERTAIN: Low Confidence]'
        )
        
        logger.info(
            f"ClaimFilter initialized: enabled={self.enabled}, "
            f"placeholder='{self.placeholder}'"
        )
    
    def filter_answer(
        self,
        answer_text: str,
        claims: List[Claim],
        decisions: List[ClaimDecision],
        sample_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, int, Dict[str, Any]]:
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
            return answer_text, 0, {
                'mode': 'disabled',
                'lc_soft_filter_threshold_applied': 0.0,
                'lc_soft_filter_escalated': False,
            }
        
        if len(claims) != len(decisions):
            raise ValueError(
                f"Claims and decisions must have same length. "
                f"Got {len(claims)} claims and {len(decisions)} decisions"
            )
        
        if not claims:
            logger.debug("No claims to filter, returning original text")
            return answer_text, 0, {
                'mode': 'no_claims',
                'lc_soft_filter_threshold_applied': 0.0,
                'lc_soft_filter_escalated': False,
            }
        
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
            if self.lc_soft_filter_prob_threshold > 0.0:
                lc_threshold, escalated = self._resolve_lc_soft_filter_threshold(
                    decisions,
                    sample_context,
                )
                if lc_threshold <= 0.0:
                    logger.info("LC soft-filter skipped for this sample (threshold=%.2f)", lc_threshold)
                    return answer_text, 0, {
                        'mode': 'none',
                        'lc_soft_filter_threshold_applied': 0.0,
                        'lc_soft_filter_escalated': False,
                    }
                filtered_text, removed_count = self._filter_lc_claims(
                    answer_text,
                    claim_dict,
                    decisions,
                    lc_threshold,
                )
                return filtered_text, removed_count, {
                    'mode': 'low_confidence',
                    'lc_soft_filter_threshold_applied': lc_threshold,
                    'lc_soft_filter_escalated': escalated,
                }
            logger.info("No contradictory claims found, returning original text")
            return answer_text, 0, {
                'mode': 'none',
                'lc_soft_filter_threshold_applied': 0.0,
                'lc_soft_filter_escalated': False,
            }
        
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
        
        return filtered_text, removed_count, {
            'mode': 'contradictory',
            'lc_soft_filter_threshold_applied': self.lc_soft_filter_prob_threshold,
            'lc_soft_filter_escalated': False,
        }

    def _resolve_lc_soft_filter_threshold(
        self,
        decisions: List[ClaimDecision],
        sample_context: Optional[Dict[str, Any]],
    ) -> Tuple[float, bool]:
        """Return (threshold, escalated) for LC soft-filtering in this sample."""
        base_threshold = self.lc_soft_filter_prob_threshold
        aggressive_threshold = self.lc_soft_filter_lc_avg_contradict_threshold

        if base_threshold <= 0.0:
            return 0.0, False

        task_type = str((sample_context or {}).get('task_type', '')).upper()
        if task_type and task_type in self.lc_soft_filter_excluded_tasks:
            return 0.0, False

        if aggressive_threshold <= 0.0 or aggressive_threshold >= base_threshold:
            return base_threshold, False

        if task_type and task_type in self.lc_soft_filter_lc_avg_contradict_excluded_tasks:
            return base_threshold, False

        # Only escalate when lc_avg_contradict is the primary detection trigger.
        # If low_confidence_coverage, contradictory, or data2txt_low_confidence is the
        # primary trigger, the aggressive threshold over-removes non-gold claims.
        detection_trigger_path = str((sample_context or {}).get('detection_trigger_path', ''))
        if detection_trigger_path and detection_trigger_path != 'lc_avg_contradict':
            return base_threshold, False

        if len(decisions) < self.lc_soft_filter_lc_avg_contradict_min_claims:
            return base_threshold, False

        low_confidence_decisions = [
            d for d in decisions if d.status == 'Low Confidence'
        ]
        if not low_confidence_decisions:
            return base_threshold, False

        low_confidence_ratio = len(low_confidence_decisions) / max(len(decisions), 1)
        if low_confidence_ratio < self.lc_soft_filter_lc_avg_contradict_min_ratio:
            return base_threshold, False

        avg_contradict_prob_low_conf = float(np.mean([
            float(d.confidence.get('contradict_prob', 0.0))
            for d in low_confidence_decisions
        ]))
        if avg_contradict_prob_low_conf < self.lc_soft_filter_lc_avg_contradict_min_avg_contradict_prob:
            return base_threshold, False

        return aggressive_threshold, True

    def _filter_lc_claims(
        self,
        answer_text: str,
        claim_dict: dict,
        decisions: List[ClaimDecision],
        lc_threshold: float,
    ) -> Tuple[str, int]:
        """Soft-filter Low Confidence claims whose contradict_prob exceeds the configured threshold.

        Only called when no Contradictory claims were found, targeting the lc_avg_contradict path.
        """
        lc_decisions = [
            d for d in decisions
            if d.status == 'Low Confidence'
            and float(d.confidence.get('contradict_prob', 0.0)) >= lc_threshold
            and d.claim_id in claim_dict
        ]
        if not lc_decisions:
            logger.info(
                "No LC claims above soft-filter threshold (%.2f), returning original text",
                lc_threshold,
            )
            return answer_text, 0

        logger.info(
            "Soft-filtering %d Low Confidence claims (contradict_prob >= %.2f)",
            len(lc_decisions),
            lc_threshold,
        )

        sorted_decisions = sorted(
            lc_decisions,
            key=lambda d: claim_dict[d.claim_id].answer_char_span[0],
            reverse=True,
        )

        filtered_text = answer_text
        removed_count = 0

        for decision in sorted_decisions:
            claim = claim_dict[decision.claim_id]
            start, end = claim.answer_char_span

            if start < 0 or end > len(filtered_text) or start >= end:
                logger.warning(
                    "Invalid char_span [%d, %d] for LC claim %s. Skipping.",
                    start, end, decision.claim_id,
                )
                continue

            filtered_text = filtered_text[:start] + self.lc_placeholder + filtered_text[end:]
            removed_count += 1

        logger.info(
            "LC soft-filter complete: removed %d claims. "
            "Original length: %d, Filtered length: %d",
            removed_count, len(answer_text), len(filtered_text),
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
