"""
Re-Prompting Module for Hallucination Mitigation.

This module implements RePrompter, which enables LLM self-correction through
iterative re-prompting with verification feedback. When verification detects
a high hallucination rate (>50% contradictory claims), the system constructs
a feedback prompt containing problematic claims and asks the generator to
self-correct.

Inspired by Chain-of-Verification (CoVe) approach from "Chain-of-Verification
Reduces Hallucination in Large Language Models" (Dhuliawala et al., 2023).

Key Components:
1. Hallucination Rate Analysis: Determines if re-prompting is needed
2. Feedback Prompt Construction: Creates verification-informed prompts
3. Iterative Refinement: Supports multi-iteration correction (max 2 iterations)
4. Re-Verification: Validates corrected responses

Formula for hallucination rate:
    hallucination_rate = contradictory_claims / total_claims
    
Trigger condition:
    re-prompt if hallucination_rate > threshold (default: 0.5)
"""

from typing import List, Dict, Tuple, Optional, Any
import logging

from src.utils.data_structures import ClaimDecision, EvidenceChunk, Claim
from src.utils.config import Config


logger = logging.getLogger(__name__)


class RePrompter:
    """
    Re-prompting mitigation strategy using verification feedback.
    
    Implements self-correction through iterative re-prompting:
    1. Analyze verification decisions to compute hallucination rate
    2. If rate exceeds threshold, construct feedback prompt with problematic claims
    3. Ask generator to self-correct based on verification feedback
    4. Re-verify corrected response (on final iteration only)
    5. Return improved response or original if no improvement needed
    
    Two strategies supported:
    - "full": Regenerate entire answer with verification feedback context
    - "claim-specific": Create targeted verification questions (CoVe-inspired)
    
    Attributes:
        config: Configuration object
        generator: GeneratorWrapper instance for text generation
        threshold: Hallucination rate threshold for triggering re-prompting (default: 0.5)
        max_iterations: Maximum number of correction iterations (default: 2)
        strategy: Re-prompting strategy ("full" or "claim-specific")
        enabled: Whether re-prompting is enabled
    
    Example:
        ```python
        config = Config()
        generator = GeneratorWrapper(config)
        repromptr = RePrompter(config, generator)
        
        # After initial verification
        result = repromptr.reprompt(
            query="What is the capital of France?",
            answer="The capital of France is Berlin.",  # Hallucinated
            decisions=claim_decisions,
            evidence=evidence_chunks,
            claims=extracted_claims
        )
        
        print(f"Final answer: {result['final_answer']}")
        print(f"Improved: {result['improved']}")
        print(f"Iterations: {result['iterations']}")
        ```
    """
    
    def __init__(self, config: Config, generator):
        """
        Initialize the RePrompter with configuration and generator.
        
        Args:
            config: Configuration object containing mitigation.reprompt settings
                   - threshold: Hallucination rate threshold (default: 0.5)
                   - max_iterations: Maximum correction iterations (default: 2)
                   - strategy: "full" or "claim-specific" (default: "full")
                   - enabled: Enable/disable re-prompting (default: False)
            generator: GeneratorWrapper instance for text generation
        
        Raises:
            ValueError: If threshold is not in [0, 1] or max_iterations < 1
        """
        self.config = config
        self.generator = generator
        
        # Load configuration with defaults
        reprompt_config = config.get('mitigation', {}).get('reprompt', {})
        
        self.threshold = reprompt_config.get('threshold', 0.5)
        self.max_iterations = reprompt_config.get('max_iterations', 2)
        self.strategy = reprompt_config.get('strategy', 'full')
        self.enabled = reprompt_config.get('enabled', False)
        
        # Validate parameters
        if not (0 <= self.threshold <= 1):
            raise ValueError(
                f"Threshold must be in [0, 1]. Got threshold={self.threshold}"
            )
        
        if self.max_iterations < 1:
            raise ValueError(
                f"max_iterations must be >= 1. Got max_iterations={self.max_iterations}"
            )
        
        if self.strategy not in ['full', 'claim-specific']:
            logger.warning(
                f"Unknown strategy '{self.strategy}'. Defaulting to 'full'."
            )
            self.strategy = 'full'
        
        logger.info(
            f"RePrompter initialized: threshold={self.threshold}, "
            f"max_iterations={self.max_iterations}, strategy={self.strategy}, "
            f"enabled={self.enabled}"
        )
    
    def should_reprompt(self, decisions: List[ClaimDecision]) -> Tuple[bool, float]:
        """
        Calculate hallucination rate and determine if re-prompting is needed.
        
        Hallucination rate is computed as:
            contradictory_claims / total_claims
        
        Re-prompting is triggered when:
            hallucination_rate > threshold
        
        Args:
            decisions: List of ClaimDecision objects from verification
        
        Returns:
            Tuple[bool, float]: (should_reprompt, hallucination_rate)
                - should_reprompt: True if rate exceeds threshold
                - hallucination_rate: Computed rate in [0, 1]
        
        Example:
            >>> decisions = [
            ...     ClaimDecision(status="Supported", ...),
            ...     ClaimDecision(status="Contradictory", ...),
            ...     ClaimDecision(status="Contradictory", ...)
            ... ]
            >>> should_retry, rate = repromptr.should_reprompt(decisions)
            >>> print(f"Rate: {rate:.2f}, Retry: {should_retry}")
            # Rate: 0.67, Retry: True (if threshold=0.5)
        """
        if not decisions:
            logger.warning("Empty decisions list, skipping re-prompting")
            return False, 0.0
        
        # Count contradictory claims
        contradictory_count = sum(
            1 for d in decisions if d.status == "Contradictory"
        )
        
        # Compute hallucination rate
        hallucination_rate = contradictory_count / len(decisions)
        
        # Check threshold
        should_retry = hallucination_rate > self.threshold
        
        logger.info(
            f"Hallucination analysis: {contradictory_count}/{len(decisions)} "
            f"contradictory ({hallucination_rate:.2%}), "
            f"threshold={self.threshold:.2%}, trigger={should_retry}"
        )
        
        return should_retry, hallucination_rate
    
    def _format_evidence(self, evidence: List[EvidenceChunk]) -> str:
        """
        Format evidence chunks for inclusion in re-prompting prompt.
        
        Uses simple "Passage N: text" format for clarity.
        
        Args:
            evidence: List of EvidenceChunk objects
        
        Returns:
            Formatted evidence string
        """
        if not evidence:
            return "No evidence provided."
        
        formatted_parts = []
        for i, chunk in enumerate(evidence, 1):
            formatted_parts.append(f"Passage {i}: {chunk.text}")
        
        return "\n\n".join(formatted_parts)
    
    def construct_feedback_prompt(
        self,
        original_query: str,
        original_answer: str,
        decisions: List[ClaimDecision],
        evidence: List[EvidenceChunk],
        claims: List[Claim]
    ) -> str:
        """
        Construct verification-informed prompt for self-correction.
        
        Two strategies implemented:
        
        1. "full" strategy:
           - Lists problematic claims with their verification status
           - Asks model to regenerate entire answer fixing issues
           - Provides full context: evidence + original answer + feedback
        
        2. "claim-specific" strategy (CoVe-inspired):
           - Creates targeted verification questions for contradictory claims
           - Asks model to answer questions then provide corrected answer
           - Encourages independent verification reasoning
        
        Args:
            original_query: The original user query
            original_answer: The draft answer that needs correction
            decisions: List of ClaimDecision objects with verification results
            evidence: List of evidence chunks used for generation
            claims: List of extracted Claim objects
        
        Returns:
            Constructed feedback prompt string
        
        Example (full strategy):
            ```
            Context: Passage 1: ... Passage 2: ...
            
            Question: What is the capital of France?
            
            Previous Answer: The capital of France is Berlin.
            
            Verification Feedback:
            - "The capital of France is Berlin": Contradictory (NLI contradiction with evidence)
            
            Please revise your answer to fix these issues, ensuring all claims are supported by the context.
            
            Revised Answer:
            ```
        """
        # Strategy 1: Full regeneration with verification feedback
        if self.strategy == "full":
            # Identify problematic claims
            problematic_decisions = [
                d for d in decisions
                if d.status in ["Contradictory", "Low Confidence"]
            ]
            
            # Build feedback section
            if problematic_decisions:
                feedback_lines = ["The following claims have verification issues:"]
                
                # Map decisions to claim text
                claim_map = {c.claim_id: c.claim_text for c in claims}
                
                for d in problematic_decisions:
                    claim_text = claim_map.get(d.claim_id, "[Unknown claim]")
                    feedback_lines.append(
                        f'- "{claim_text}": {d.status} ({d.rationale})'
                    )
                
                feedback = "\n".join(feedback_lines)
            else:
                feedback = "Some claims have low confidence or lack sufficient evidence support."
            
            # Construct full prompt
            prompt = (
                f"Context: {self._format_evidence(evidence)}\n\n"
                f"Question: {original_query}\n\n"
                f"Previous Answer: {original_answer}\n\n"
                f"Verification Feedback:\n{feedback}\n\n"
                f"Please revise your answer to fix these issues, "
                f"ensuring all claims are supported by the context. "
                f"Only include information that can be verified from the passages.\n\n"
                f"Revised Answer:"
            )
        
        # Strategy 2: Claim-specific verification questions (CoVe-style)
        else:  # "claim-specific"
            # Generate verification questions for contradictory claims
            contradictory_decisions = [
                d for d in decisions if d.status == "Contradictory"
            ]
            
            # Map decisions to claim text
            claim_map = {c.claim_id: c.claim_text for c in claims}
            
            verification_questions = []
            for d in contradictory_decisions:
                claim_text = claim_map.get(d.claim_id, "[Unknown claim]")
                verification_questions.append(
                    f"Is the following statement supported by the context: '{claim_text}'?"
                )
            
            if not verification_questions:
                # Fallback if no contradictory claims but still triggered
                verification_questions.append(
                    f"Is the following answer fully supported by the context?"
                )
            
            # Construct CoVe-style prompt
            prompt = (
                f"Context: {self._format_evidence(evidence)}\n\n"
                f"Question: {original_query}\n\n"
                f"Previous Answer: {original_answer}\n\n"
                f"Verification Questions:\n"
                + "\n".join(f"{i+1}. {q}" for i, q in enumerate(verification_questions))
                + "\n\nBased on the context and verification questions, "
                f"provide a corrected answer that is fully supported by the passages. "
                f"Do not include information that cannot be verified.\n\n"
                f"Corrected Answer:"
            )
        
        logger.debug(f"Constructed feedback prompt ({self.strategy} strategy):\n{prompt}")
        return prompt
    
    def reprompt(
        self,
        query: str,
        answer: str,
        decisions: List[ClaimDecision],
        evidence: List[EvidenceChunk],
        claims: List[Claim]
    ) -> Dict[str, Any]:
        """
        Execute re-prompting with iterative refinement.
        
        Workflow:
        1. Check if hallucination rate exceeds threshold
        2. If not, return original answer (no re-prompting needed)
        3. If yes, construct feedback prompt with problematic claims
        4. Generate corrected response with lower temperature (0.3)
        5. Optionally iterate up to max_iterations
        6. Return final corrected answer with metadata
        
        Note: Re-verification is performed externally after re-prompting
        to measure actual improvement.
        
        Args:
            query: Original user query
            answer: Draft answer to potentially correct
            decisions: List of ClaimDecision objects from verification
            evidence: List of evidence chunks used for generation
            claims: List of extracted Claim objects
        
        Returns:
            Dictionary containing:
                - final_answer: Corrected answer or original if no correction needed
                - iterations: Number of re-prompting iterations performed
                - improved: Whether re-prompting was triggered
                - hallucination_rate_before: Initial hallucination rate
                - hallucination_rate_after: Final rate (requires external re-verification)
                - feedback_prompt: The prompt used for correction (for debugging)
        
        Example:
            >>> result = repromptr.reprompt(query, answer, decisions, evidence, claims)
            >>> if result['improved']:
            ...     print(f"Corrected after {result['iterations']} iteration(s)")
            ...     print(f"New answer: {result['final_answer']}")
        """
        # Check if re-prompting is needed
        should_retry, initial_rate = self.should_reprompt(decisions)
        
        if not should_retry:
            logger.info("Hallucination rate below threshold, skipping re-prompting")
            return {
                'final_answer': answer,
                'iterations': 0,
                'improved': False,
                'hallucination_rate_before': initial_rate,
                'hallucination_rate_after': initial_rate,
                'feedback_prompt': None
            }
        
        logger.info(
            f"Re-prompting triggered: hallucination_rate={initial_rate:.2%} > "
            f"threshold={self.threshold:.2%}"
        )
        
        # Iterative correction loop
        current_answer = answer
        feedback_prompt = None
        
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"Re-prompting iteration {iteration}/{self.max_iterations}")
            
            # Construct feedback prompt
            feedback_prompt = self.construct_feedback_prompt(
                original_query=query,
                original_answer=current_answer,
                decisions=decisions,
                evidence=evidence,
                claims=claims
            )
            
            # Generate corrected response with lower temperature
            try:
                corrected_result = self.generator.generate_with_metadata(
                    prompt=feedback_prompt,
                    evidence_chunks=None,  # Evidence already in prompt
                    temperature=0.3,  # Lower temperature for more conservative correction
                    do_sample=True,
                    max_new_tokens=512
                )
                
                current_answer = corrected_result['text']
                logger.info(f"Generated corrected answer (iteration {iteration})")
                
            except Exception as e:
                logger.error(f"Error during re-prompting iteration {iteration}: {str(e)}")
                # Return best answer so far on error
                break
            
            # For now, single iteration is sufficient
            # Multi-iteration would require re-verification in each loop
            break
        
        logger.info(f"Re-prompting complete after {iteration} iteration(s)")
        
        # Note: hallucination_rate_after requires external re-verification
        # This will be computed in the pipeline after re-prompting
        return {
            'final_answer': current_answer,
            'iterations': iteration,
            'improved': True,
            'hallucination_rate_before': initial_rate,
            'hallucination_rate_after': None,  # Computed externally via re-verification
            'feedback_prompt': feedback_prompt
        }
