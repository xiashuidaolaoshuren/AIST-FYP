"""
Retrieval-Grounded Detector for LLM Hallucination Detection.

This module implements retrieval-grounded heuristics for detecting hallucinations
by measuring how well claims are supported by retrieved evidence. Uses three
complementary metrics: entity coverage, number coverage, and token overlap.

Key concepts:
- Entity coverage: Named entities (PERSON, ORG, GPE, etc.) in claim must appear in evidence
- Number coverage: Numeric values in claim must match evidence
- Token overlap: ROUGE-L F1 score measuring lexical similarity

Reference:
- SelfCheckGPT paper: Retrieval-based consistency checking
- RAGTruth benchmark: Citation span integrity metrics
"""

import re
import numpy as np
from typing import Dict, List, Optional

from src.utils.nlp_utils import get_spacy_model
from src.utils.data_structures import Claim, EvidenceChunk
from src.utils.logger import setup_logger
from src.utils.config import Config


class RetrievalGroundedDetector:
    """
    Detector that measures claim groundedness through evidence alignment.
    
    Analyzes how well a generated claim is supported by retrieved evidence
    using three complementary signals:
    1. Entity coverage: Are named entities in the claim present in evidence?
    2. Number coverage: Are numeric values in the claim found in evidence?
    3. Token overlap: How much lexical overlap exists (ROUGE-L F1)?
    
    The detector:
    1. Extracts entities and numbers from claim using spaCy
    2. Checks presence in evidence text (with optional fuzzy matching)
    3. Calculates token-level overlap using longest common subsequence
    4. Returns three grounding scores in [0.0, 1.0]
    
    Attributes:
        config: Configuration object
        nlp: Shared spaCy model from nlp_utils
        entity_types: List of spaCy NER labels to check (e.g., PERSON, ORG)
        fuzzy_matching: Whether to use case-insensitive substring matching
        min_token_length: Minimum token length for overlap calculation
        logger: Logger instance
    
    Example:
        >>> config = Config()
        >>> detector = RetrievalGroundedDetector(config)
        >>> signal = detector.compute_signal(claim, evidence, metadata)
        >>> print(f"Entity coverage: {signal['entities']:.2f}")
    """
    
    def __init__(self, config: Config):
        """
        Initialize the retrieval-grounded detector.
        
        Args:
            config: Configuration object with verification settings
        """
        self.config = config
        self.logger = setup_logger(__name__)
        
        # Load shared spaCy model (singleton pattern, no double-loading)
        spacy_model = (
            config.verification.spacy_model
            if hasattr(config, 'verification') and hasattr(config.verification, 'spacy_model')
            else 'en_core_web_sm'
        )
        self.nlp = get_spacy_model(spacy_model)
        
        # Load configuration parameters
        if hasattr(config, 'verification') and hasattr(config.verification, 'grounded'):
            self.entity_types = config.verification.grounded.entity_types
            self.fuzzy_matching = config.verification.grounded.fuzzy_matching
            self.min_token_length = config.verification.grounded.min_token_length
        else:
            # Defaults
            self.entity_types = ["PERSON", "ORG", "GPE", "DATE", "NORP"]
            self.fuzzy_matching = True
            self.min_token_length = 2
        
        self.logger.info(
            f"RetrievalGroundedDetector initialized with entity_types={self.entity_types}, "
            f"fuzzy_matching={self.fuzzy_matching}, min_token_length={self.min_token_length}"
        )
    
    def compute_signal(
        self,
        claim: Claim,
        evidence: EvidenceChunk,
        metadata: Dict
    ) -> Dict[str, float]:
        """
        Compute retrieval-grounded signal for a claim-evidence pair.
        
        Calculates three complementary grounding metrics:
        1. Entity coverage: Proportion of claim entities found in evidence
        2. Number coverage: Proportion of claim numbers found in evidence
        3. Token overlap: ROUGE-L F1 score between claim and evidence
        
        Args:
            claim: Claim object with text to verify
            evidence: Evidence chunk to check against
            metadata: Generator metadata (not used, but required for API consistency)
        
        Returns:
            Dictionary with three keys:
            - 'entities': float in [0.0, 1.0] (1.0 if no entities in claim)
            - 'numbers': float in [0.0, 1.0] (1.0 if no numbers in claim)
            - 'tokens_overlap': float in [0.0, 1.0]
        
        Example:
            >>> signal = detector.compute_signal(claim, evidence, metadata)
            >>> if signal['entities'] < 0.5:
            ...     print("Warning: Poor entity coverage!")
        """
        # Edge case: empty claim
        if not claim.text or not claim.text.strip():
            self.logger.warning(f"Empty claim {claim.claim_id}, returning zeros")
            return {'entities': 0.0, 'numbers': 0.0, 'tokens_overlap': 0.0}
        
        # Edge case: empty evidence
        if not evidence.text or not evidence.text.strip():
            self.logger.warning(
                f"Empty evidence for claim {claim.claim_id}, returning zeros"
            )
            return {'entities': 0.0, 'numbers': 0.0, 'tokens_overlap': 0.0}
        
        try:
            # Calculate three grounding metrics
            entities_score = self._calculate_entity_coverage(claim, evidence)
            numbers_score = self._calculate_number_coverage(claim, evidence)
            overlap_score = self._calculate_token_overlap(claim, evidence)
            
            self.logger.debug(
                f"Claim {claim.claim_id}: entities={entities_score:.2f}, "
                f"numbers={numbers_score:.2f}, tokens_overlap={overlap_score:.2f}"
            )
            
            return {
                'entities': entities_score,
                'numbers': numbers_score,
                'tokens_overlap': overlap_score
            }
        
        except Exception as e:
            self.logger.error(
                f"Error computing grounded signal for claim {claim.claim_id}: {e}",
                exc_info=True
            )
            return {'entities': 0.0, 'numbers': 0.0, 'tokens_overlap': 0.0}
    
    def _calculate_entity_coverage(
        self,
        claim: Claim,
        evidence: EvidenceChunk
    ) -> float:
        """
        Calculate what proportion of claim entities appear in evidence.
        
        Extracts named entities from the claim using spaCy NER, filters by
        configured entity types, and checks if each entity appears in the
        evidence text. Uses fuzzy matching if enabled.
        
        Args:
            claim: Claim with text to extract entities from
            evidence: Evidence to search for entities
        
        Returns:
            Float in [0.0, 1.0]: matched_entities / total_entities
            Returns 1.0 if no entities found (trivially satisfied)
        
        Example:
            >>> claim = Claim(..., text="Barack Obama was born in Hawaii")
            >>> # Extracts: ["Barack Obama", "Hawaii"] (PERSON, GPE)
            >>> # Checks if both appear in evidence
        """
        try:
            # Parse claim with spaCy
            doc_claim = self.nlp(claim.text)
            
            # Extract entities of configured types
            entities = [
                ent.text
                for ent in doc_claim.ents
                if ent.label_ in self.entity_types
            ]
            
            # Edge case: no entities in claim
            if not entities:
                self.logger.debug(
                    f"No entities found in claim {claim.claim_id}, returning 1.0"
                )
                return 1.0
            
            # Check each entity against evidence
            matched = 0
            evidence_text = evidence.text
            
            for entity in entities:
                if self._fuzzy_match(entity, evidence_text):
                    matched += 1
                    self.logger.debug(f"Entity '{entity}' found in evidence")
                else:
                    self.logger.debug(f"Entity '{entity}' NOT found in evidence")
            
            coverage = matched / len(entities)
            
            self.logger.debug(
                f"Claim {claim.claim_id}: {matched}/{len(entities)} entities matched "
                f"(coverage={coverage:.2f})"
            )
            
            return float(coverage)
        
        except Exception as e:
            self.logger.error(
                f"Error calculating entity coverage for claim {claim.claim_id}: {e}",
                exc_info=True
            )
            return 0.0
    
    def _calculate_number_coverage(
        self,
        claim: Claim,
        evidence: EvidenceChunk
    ) -> float:
        """
        Calculate what proportion of claim numbers appear in evidence.
        
        Extracts numeric tokens from the claim using spaCy's like_num property,
        then checks if each number appears exactly in the evidence text.
        
        Args:
            claim: Claim with text to extract numbers from
            evidence: Evidence to search for numbers
        
        Returns:
            Float in [0.0, 1.0]: matched_numbers / total_numbers
            Returns 1.0 if no numbers found (trivially satisfied)
        
        Example:
            >>> claim = Claim(..., text="The tower is 324 meters tall")
            >>> # Extracts: ["324"]
            >>> # Checks if "324" appears in evidence
        
        Note:
            Uses exact string matching for numbers. Does not handle
            written numbers (e.g., "three hundred") or unit variations
            (e.g., "324m" vs "324 meters").
        """
        try:
            # Parse claim with spaCy
            doc_claim = self.nlp(claim.text)
            
            # Extract numeric tokens
            numbers = [
                token.text
                for token in doc_claim
                if token.like_num
            ]
            
            # Edge case: no numbers in claim
            if not numbers:
                self.logger.debug(
                    f"No numbers found in claim {claim.claim_id}, returning 1.0"
                )
                return 1.0
            
            # Check each number against evidence (exact match)
            matched = 0
            evidence_text = evidence.text
            
            for number in numbers:
                if number in evidence_text:
                    matched += 1
                    self.logger.debug(f"Number '{number}' found in evidence")
                else:
                    self.logger.debug(f"Number '{number}' NOT found in evidence")
            
            coverage = matched / len(numbers)
            
            self.logger.debug(
                f"Claim {claim.claim_id}: {matched}/{len(numbers)} numbers matched "
                f"(coverage={coverage:.2f})"
            )
            
            return float(coverage)
        
        except Exception as e:
            self.logger.error(
                f"Error calculating number coverage for claim {claim.claim_id}: {e}",
                exc_info=True
            )
            return 0.0
    
    def _calculate_token_overlap(
        self,
        claim: Claim,
        evidence: EvidenceChunk
    ) -> float:
        """
        Calculate ROUGE-L F1 score between claim and evidence tokens.
        
        Computes lexical similarity using longest common subsequence (LCS)
        between tokenized claim and evidence. LCS preserves word order and
        measures how much of the claim is covered by the evidence.
        
        Algorithm:
        1. Tokenize claim and evidence (lowercase, alphanumeric only)
        2. Filter tokens by min_token_length
        3. Compute LCS length using dynamic programming
        4. Calculate precision = LCS / len(evidence_tokens)
        5. Calculate recall = LCS / len(claim_tokens)
        6. Return F1 = 2 * P * R / (P + R)
        
        Args:
            claim: Claim with text to tokenize
            evidence: Evidence with text to tokenize
        
        Returns:
            Float in [0.0, 1.0]: ROUGE-L F1 score
            Returns 0.0 for edge cases (empty tokens, division by zero)
        
        Example:
            >>> claim = "The Eiffel Tower is in Paris"
            >>> evidence = "The famous Eiffel Tower stands in Paris, France"
            >>> # LCS: ["the", "eiffel", "tower", "in", "paris"]
            >>> # Precision = 5/8, Recall = 5/6, F1 ≈ 0.69
        
        Note:
            This is a simplified ROUGE-L implementation. For production,
            consider using the rouge-score library for strict evaluation.
        """
        try:
            # Tokenize both texts
            claim_tokens = self._tokenize(claim.text)
            evidence_tokens = self._tokenize(evidence.text)
            
            # Edge case: empty tokens
            if not claim_tokens or not evidence_tokens:
                self.logger.debug(
                    f"Empty tokens for claim {claim.claim_id}, returning 0.0"
                )
                return 0.0
            
            # Compute LCS length
            lcs_length = self._compute_lcs_length(claim_tokens, evidence_tokens)
            
            # Calculate precision and recall
            precision = lcs_length / len(evidence_tokens)
            recall = lcs_length / len(claim_tokens)
            
            # Calculate F1 score
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
            
            self.logger.debug(
                f"Claim {claim.claim_id}: LCS={lcs_length}, "
                f"P={precision:.2f}, R={recall:.2f}, F1={f1:.2f}"
            )
            
            return float(f1)
        
        except Exception as e:
            self.logger.error(
                f"Error calculating token overlap for claim {claim.claim_id}: {e}",
                exc_info=True
            )
            return 0.0
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into lowercase alphanumeric tokens.
        
        Extracts alphanumeric words, converts to lowercase, and filters
        by minimum token length.
        
        Args:
            text: Text to tokenize
        
        Returns:
            List of lowercase tokens meeting length requirement
        
        Example:
            >>> tokens = detector._tokenize("The Eiffel Tower is 324m tall!")
            >>> # Returns: ["the", "eiffel", "tower", "is", "324m", "tall"]
            >>> # (assuming min_token_length=2)
        """
        # Extract alphanumeric tokens
        tokens = re.findall(r'\w+', text.lower())
        
        # Filter by minimum length
        tokens = [t for t in tokens if len(t) >= self.min_token_length]
        
        return tokens
    
    def _compute_lcs_length(
        self,
        tokens1: List[str],
        tokens2: List[str]
    ) -> int:
        """
        Compute longest common subsequence (LCS) length using dynamic programming.
        
        Uses standard DP algorithm with O(n*m) time and space complexity.
        LCS preserves order but allows gaps (unlike longest common substring).
        
        Args:
            tokens1: First token sequence
            tokens2: Second token sequence
        
        Returns:
            Integer length of the longest common subsequence
        
        Example:
            >>> tokens1 = ["the", "tower", "is", "tall"]
            >>> tokens2 = ["the", "famous", "tower", "is", "very", "tall"]
            >>> lcs_length = 4  # ["the", "tower", "is", "tall"]
        
        Algorithm:
            dp[i][j] = LCS length for tokens1[:i] and tokens2[:j]
            dp[i][j] = dp[i-1][j-1] + 1  if tokens1[i-1] == tokens2[j-1]
                     = max(dp[i-1][j], dp[i][j-1])  otherwise
        """
        n, m = len(tokens1), len(tokens2)
        
        # Initialize DP table
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        
        # Fill DP table
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if tokens1[i - 1] == tokens2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[n][m]
    
    def _fuzzy_match(self, entity: str, text: str) -> bool:
        """
        Check if entity appears in text using fuzzy matching.
        
        Currently implements simple case-insensitive substring matching.
        Future enhancement: Support edit distance, acronym expansion,
        partial token matching.
        
        Args:
            entity: Entity string to search for
            text: Text to search in
        
        Returns:
            True if entity found in text (case-insensitive), False otherwise
        
        Example:
            >>> detector._fuzzy_match("Barack Obama", "barack obama was president")
            True
            >>> detector._fuzzy_match("Obama", "The president was Barack Obama")
            True
        
        Note:
            If fuzzy_matching is disabled in config, uses case-sensitive matching.
            
            Future enhancement could add:
            - Edit distance threshold (handle typos: "Obamma" matches "Obama")
            - Acronym expansion ("ML" matches "machine learning")
            - Partial ratio ("New York City" matches "NYC")
        """
        entity_normalized = entity.strip()
        
        if self.fuzzy_matching:
            # Case-insensitive substring matching
            return entity_normalized.lower() in text.lower()
        else:
            # Case-sensitive exact substring matching
            return entity_normalized in text
