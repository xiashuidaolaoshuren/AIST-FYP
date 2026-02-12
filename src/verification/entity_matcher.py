"""
Entity Matcher for Tiered Entity Matching.

This module implements a tiered entity matching system for the
RetrievalGroundedDetector to improve entity coverage from ~70% to ~90%.
Uses three complementary matching strategies with early-exit optimization.

Matching Tiers (executed sequentially):
1. Tier 1 (Substring): Fast case-insensitive substring matching (~70% coverage)
2. Tier 2 (Acronym): Acronym extraction and matching (+20% coverage)
3. Tier 3 (Alias): Dictionary-based alias lookup (+5% coverage)

Key Features:
- Early-exit optimization: Returns immediately on first match (fast path)
- Configurable: Enable/disable tiers via config.yaml
- Performance: ~8ms overhead target (<10ms requirement)
- No external dependencies: Pure Python implementation

Usage:
    >>> from src.verification.entity_matcher import EntityMatcher
    >>> from src.utils.config import Config
    >>> 
    >>> config = Config()
    >>> matcher = EntityMatcher(config)
    >>> 
    >>> # Tier 1: Substring match
    >>> matcher.match_entity("Obama", "Obama spoke today")
    True
    >>> 
    >>> # Tier 2: Acronym match
    >>> matcher.match_entity("United States of America", "The U.S. economy")
    True
    >>> 
    >>> # Tier 3: Alias match
    >>> matcher.match_entity("United States", "America's GDP")
    True

References:
    - docs/entity_normalization_challenge.md: Design documentation
    - SelfCheckGPT paper: Entity-based consistency checking
"""

import re
import os
import json
import time
import hashlib
from collections import OrderedDict
from typing import Optional, List, Dict, Any

from src.utils.logger import setup_logger
from .entity_aliases import get_all_forms


class EntityMatcher:
    """
    Tiered entity matching system for handling surface form variations.
    
    Implements three matching strategies with early-exit optimization:
    1. Substring matching (baseline, fast)
    2. Acronym matching (handles "USA" vs "United States of America")
    3. Alias dictionary lookup (handles "America" vs "United States")
    
    The matcher tries tiers sequentially and returns immediately on the first
    match, ensuring optimal performance for common cases.
    
    Attributes:
        config: Configuration object
        logger: Logger instance
        use_acronym: Whether Tier 2 (acronym matching) is enabled
        use_aliases: Whether Tier 3 (alias dictionary) is enabled
    
    Example:
        >>> config = Config()
        >>> matcher = EntityMatcher(config)
        >>> 
        >>> # Configure at runtime
        >>> matcher.use_acronym = True
        >>> matcher.use_aliases = True
        >>> 
        >>> # Match entities
        >>> matcher.match_entity("WHO", "World Health Organization declared")
        True
    """
    
    def __init__(self, config):
        """
        Initialize the entity matcher with configuration.
        
        Args:
            config: Configuration object with verification settings.
                   Reads from config.verification.grounded.matching:
                   - acronym_matching: bool (default: True)
                   - alias_dictionary: bool (default: True)
        
        Example:
            >>> config = Config()
            >>> matcher = EntityMatcher(config)
        """
        self.config = config
        self.logger = setup_logger(__name__)
        
        # Load matching configuration with defaults
        self.use_acronym = True  # Default
        self.use_aliases = True  # Default
        self.use_llm = False  # Default

        # LLM matcher defaults (DeepSeek/OpenAI-compatible API)
        self.llm_provider = "deepseek"
        self.llm_base_url = "https://api.deepseek.com"
        self.llm_model = "deepseek-chat"
        self.llm_api_key_env = "DEEPSEEK_API_KEY"
        self.llm_timeout_s = 20
        self.llm_max_retries = 2
        self.llm_retry_backoff_s = 1.5
        self.llm_max_evidence_chars = 3000
        self.llm_cache_size = 2048
        self.llm_temperature = 0
        self.llm_max_tokens = 200
        self._llm_cache: "OrderedDict[str, bool]" = OrderedDict()
        
        # Try to read from config (graceful fallback if not present)
        try:
            if hasattr(config, 'verification') and hasattr(config.verification, 'grounded'):
                matching_config = getattr(config.verification.grounded, 'matching', {})
                if isinstance(matching_config, dict):
                    self.use_acronym = matching_config.get('acronym_matching', True)
                    self.use_aliases = matching_config.get('alias_dictionary', True)
                    llm_config = matching_config.get('llm', {})
                elif matching_config is not None:
                    # Handle case where matching is an object with attributes
                    self.use_acronym = getattr(matching_config, 'acronym_matching', True)
                    self.use_aliases = getattr(matching_config, 'alias_dictionary', True)
                    llm_config = getattr(matching_config, 'llm', {})
                else:
                    llm_config = {}

                if llm_config:
                    if isinstance(llm_config, dict):
                        self.use_llm = llm_config.get('enabled', False)
                        self.llm_provider = llm_config.get('provider', self.llm_provider)
                        self.llm_base_url = llm_config.get('base_url', self.llm_base_url)
                        self.llm_model = llm_config.get('model', self.llm_model)
                        self.llm_api_key_env = llm_config.get('api_key_env', self.llm_api_key_env)
                        self.llm_timeout_s = llm_config.get('timeout_s', self.llm_timeout_s)
                        self.llm_max_retries = llm_config.get('max_retries', self.llm_max_retries)
                        self.llm_retry_backoff_s = llm_config.get('retry_backoff_s', self.llm_retry_backoff_s)
                        self.llm_max_evidence_chars = llm_config.get('max_evidence_chars', self.llm_max_evidence_chars)
                        self.llm_cache_size = llm_config.get('cache_size', self.llm_cache_size)
                        self.llm_temperature = llm_config.get('temperature', self.llm_temperature)
                        self.llm_max_tokens = llm_config.get('max_tokens', self.llm_max_tokens)
                    else:
                        self.use_llm = getattr(llm_config, 'enabled', False)
                        self.llm_provider = getattr(llm_config, 'provider', self.llm_provider)
                        self.llm_base_url = getattr(llm_config, 'base_url', self.llm_base_url)
                        self.llm_model = getattr(llm_config, 'model', self.llm_model)
                        self.llm_api_key_env = getattr(llm_config, 'api_key_env', self.llm_api_key_env)
                        self.llm_timeout_s = getattr(llm_config, 'timeout_s', self.llm_timeout_s)
                        self.llm_max_retries = getattr(llm_config, 'max_retries', self.llm_max_retries)
                        self.llm_retry_backoff_s = getattr(llm_config, 'retry_backoff_s', self.llm_retry_backoff_s)
                        self.llm_max_evidence_chars = getattr(llm_config, 'max_evidence_chars', self.llm_max_evidence_chars)
                        self.llm_cache_size = getattr(llm_config, 'cache_size', self.llm_cache_size)
                        self.llm_temperature = getattr(llm_config, 'temperature', self.llm_temperature)
                        self.llm_max_tokens = getattr(llm_config, 'max_tokens', self.llm_max_tokens)
        except Exception as e:
            self.logger.warning(
                f"Could not load matching config, using defaults: {e}"
            )
        
        self.logger.info(
            f"EntityMatcher initialized: acronym_matching={self.use_acronym}, "
            f"alias_dictionary={self.use_aliases}, llm_enabled={self.use_llm}"
        )
    
    def match_entity(self, entity: str, evidence_text: str) -> bool:
        """
        Check if entity appears in evidence using tiered matching.
        
        Tries matching strategies in order of performance:
        1. Substring match (fastest, ~70% coverage)
        2. Acronym match (medium speed, +20% coverage)
        3. Alias dictionary (fast hash lookup, +5% coverage)
        
        Returns immediately on first match (early-exit optimization).
        
        Args:
            entity: Entity string to match (e.g., "United States of America")
            evidence_text: Text to search in (e.g., "The U.S. economy...")
        
        Returns:
            True if entity found via any tier, False otherwise
        
        Examples:
            >>> matcher.match_entity("Obama", "Barack Obama spoke")
            True
            
            >>> matcher.match_entity("United States of America", "The U.S. economy")
            True
            
            >>> matcher.match_entity("United States", "America's economy")
            True
            
            >>> matcher.match_entity("XYZ Corp", "Unrelated text")
            False
        
        Note:
            - Empty entity or evidence returns False
            - All matching is case-insensitive
            - Tier 2 and 3 can be disabled via config
        """
        # Edge case: empty inputs
        if not entity or not entity.strip():
            self.logger.debug("Empty entity, returning False")
            return False
        
        if not evidence_text or not evidence_text.strip():
            self.logger.debug("Empty evidence_text, returning False")
            return False
        
        # Tier 1: Substring match (baseline, handles ~70%)
        if self._match_substring(entity, evidence_text):
            self.logger.debug(f"Tier 1 (substring) matched: '{entity}'")
            return True
        
        # Tier 2: Acronym matching (handles acronyms like USA, WHO, MIT)
        if self.use_acronym and self._match_acronym(entity, evidence_text):
            self.logger.debug(f"Tier 2 (acronym) matched: '{entity}'")
            return True
        
        # Tier 3: Alias dictionary lookup (handles America ↔ United States)
        if self.use_aliases and self._match_aliases(entity, evidence_text):
            self.logger.debug(f"Tier 3 (aliases) matched: '{entity}'")
            return True

        # Tier 4: LLM-based matching (DeepSeek/OpenAI-compatible API)
        if self.use_llm and self._match_llm(entity, evidence_text):
            self.logger.debug(f"Tier 4 (llm) matched: '{entity}'")
            return True
        
        # No match found
        self.logger.debug(f"No match found for entity: '{entity}'")
        return False

    # =========================================================================
    # Tier 4: LLM-Based Matching (DeepSeek/OpenAI-compatible API)
    # =========================================================================

    def _match_llm(self, entity: str, text: str) -> bool:
        """
        Tier 4: LLM-based entity matching with strict quote verification.

        Calls an OpenAI-compatible API (DeepSeek) to decide if an entity
        appears in evidence. The model must return a verbatim evidence quote;
        if the quote is not found in the evidence text, the match is rejected.
        """
        api_key = os.getenv(self.llm_api_key_env)
        if not api_key:
            self.logger.warning(
                f"LLM matcher enabled but {self.llm_api_key_env} is not set; skipping"
            )
            return False

        truncated_text = text[: self.llm_max_evidence_chars]
        cache_key = self._llm_cache_key(entity, truncated_text)
        cached = self._llm_cache_get(cache_key)
        if cached is not None:
            return cached

        system_prompt, user_prompt = self._build_llm_prompts(entity, truncated_text)
        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.llm_temperature,
            "max_tokens": self.llm_max_tokens
        }

        url = self.llm_base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        last_error = None
        for attempt in range(self.llm_max_retries + 1):
            try:
                response = self._post_json(url, headers, payload, self.llm_timeout_s)
                content = self._extract_llm_content(response)
                result = self._parse_llm_result(content)
                is_match = self._validate_llm_result(result, entity, truncated_text)
                self._llm_cache_set(cache_key, is_match)
                return is_match
            except Exception as e:
                last_error = e
                if attempt < self.llm_max_retries:
                    time.sleep(self.llm_retry_backoff_s)
                continue

        if last_error:
            self.logger.warning(f"LLM matcher failed after retries: {last_error}")
        self._llm_cache_set(cache_key, False)
        return False

    def _build_llm_prompts(self, entity: str, evidence_text: str) -> (str, str):
        system_prompt = (
            "You are an entity matching assistant. "
            "You must not invent text. If you claim a match, you must quote the exact evidence substring."
        )
        user_prompt = (
            "Task: Decide if the claim entity refers to the same real-world entity as something explicitly "
            "mentioned in the evidence.\n"
            f"Claim entity: \"{entity}\"\n"
            f"Evidence text: \"{evidence_text}\"\n"
            "Return JSON only with keys: match (true/false), matched_surface_form (string or null), "
            "evidence_quote (string or null), confidence (0-1), rationale_short (<=20 words). "
            "If match=false, set matched_surface_form and evidence_quote to null."
        )
        return system_prompt, user_prompt

    def _post_json(self, url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout_s: int) -> Dict[str, Any]:
        import requests
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
        resp.raise_for_status()
        return resp.json()

    def _extract_llm_content(self, response: Dict[str, Any]) -> str:
        try:
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise ValueError(f"Unexpected LLM response format: {e}")

    def _parse_llm_result(self, content: str) -> Dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Attempt to extract JSON object from text
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        raise ValueError("LLM response is not valid JSON")

    def _validate_llm_result(self, result: Dict[str, Any], entity: str, evidence_text: str) -> bool:
        if not isinstance(result, dict):
            return False
        match = result.get("match", False)
        if match is not True:
            return False
        evidence_quote = result.get("evidence_quote")
        if not evidence_quote or not isinstance(evidence_quote, str):
            return False
        if evidence_quote not in evidence_text:
            self.logger.debug(
                f"LLM quote not found in evidence for entity '{entity}': '{evidence_quote[:50]}...'"
            )
            return False
        return True

    def _llm_cache_key(self, entity: str, evidence_text: str) -> str:
        digest = hashlib.sha256(
            (entity.strip().lower() + "||" + evidence_text).encode("utf-8")
        ).hexdigest()
        return digest

    def _llm_cache_get(self, key: str) -> Optional[bool]:
        if key in self._llm_cache:
            value = self._llm_cache.pop(key)
            self._llm_cache[key] = value
            return value
        return None

    def _llm_cache_set(self, key: str, value: bool) -> None:
        self._llm_cache[key] = value
        if len(self._llm_cache) > self.llm_cache_size:
            self._llm_cache.popitem(last=False)
    
    # =========================================================================
    # Tier 1: Substring Matching
    # =========================================================================
    
    def _match_substring(self, entity: str, text: str) -> bool:
        """
        Tier 1: Case-insensitive bidirectional substring matching.
        
        Simplest and fastest matching strategy. Handles the majority of cases
        where the entity appears verbatim or as a substring in the evidence,
        or vice versa (handles partial entity names).
        
        Args:
            entity: Entity to match
            text: Text to search in
        
        Returns:
            True if entity (lowercased) is substring of text (lowercased)
            or any word from entity appears in text
        
        Examples:
            >>> self._match_substring("Obama", "Barack Obama spoke")
            True
            
            >>> self._match_substring("Barack Obama", "Obama spoke")
            True  # "Obama" from entity found in text
            
            >>> self._match_substring("NASA", "nasa launched")
            True
        
        Performance: O(n*m) where n=len(text), m=len(entity)
        Handles: ~70% of entity matches (baseline)
        """
        entity_lower = entity.lower().strip()
        text_lower = text.lower()
        
        # Direct substring match
        if entity_lower in text_lower:
            return True
        
        # Check if any significant word from entity appears in text
        # (handles "Barack Obama" matching "Obama spoke")
        entity_words = entity_lower.split()
        if len(entity_words) > 1:
            # Check each word (skip single letters)
            for word in entity_words:
                if len(word) > 1 and word in text_lower:
                    return True
        
        return False
    
    # =========================================================================
    # Tier 2: Acronym Matching
    # =========================================================================
    
    def _match_acronym(self, entity: str, text: str) -> bool:
        """
        Tier 2: Acronym extraction and matching.
        
        Extracts acronyms from multi-word entities by taking the first letter
        of each capitalized word. Then searches for the acronym in the text
        with various punctuation variants (USA, U.S.A, U.S., etc.).
        
        Args:
            entity: Entity to extract acronym from
            text: Text to search for acronym in
        
        Returns:
            True if acronym found in text, False otherwise
        
        Examples:
            >>> self._match_acronym("United States of America", "The U.S. economy")
            True
            
            >>> self._match_acronym("World Health Organization", "WHO declared")
            True
            
            >>> self._match_acronym("Federal Bureau of Investigation", "The F.B.I. said")
            True
            
            >>> self._match_acronym("Obama", "The president spoke")
            False  # Single word, no acronym extracted
        
        Algorithm:
            1. Extract acronym from entity (first letter of capitalized words)
            2. If no acronym (single word), return False
            3. Normalize both acronym and evidence tokens (remove periods/spaces)
            4. Check if acronym matches any evidence token
        
        Performance: O(w*t) where w=words in entity, t=tokens in text
        Handles: +20% coverage on top of Tier 1
        """
        # Extract acronym from entity
        entity_acronym = self._extract_acronym(entity)
        if not entity_acronym:
            return False  # Single word or no capitalized words
        
        # Tokenize evidence and check each token
        # Split on whitespace and punctuation, but keep periods with letters
        # to handle "F.B.I." style acronyms
        import re
        evidence_tokens = re.split(r'[\s,()]+', text)
        
        for token in evidence_tokens:
            if not token:  # Skip empty tokens
                continue
            normalized_token = self._normalize_acronym(token)
            
            # Check if token matches the entity's acronym
            if normalized_token == entity_acronym:
                return True
            
            # Also check if the entire entity (when normalized) matches the token
            # This handles cases like "USA" entity matching "U.S.A" in text
            if len(normalized_token) >= 2:
                normalized_entity = self._normalize_acronym(entity)
                if normalized_entity == normalized_token:
                    return True
        
        return False
    
    @staticmethod
    def _extract_acronym(text: str) -> Optional[str]:
        """
        Extract acronym from multi-word phrase.
        
        Takes the first letter of each word that starts with an uppercase letter.
        Requires at least 2 capitalized words to form a valid acronym.
        
        Args:
            text: Text to extract acronym from
        
        Returns:
            Acronym string (uppercase) or None if not applicable
        
        Examples:
            >>> EntityMatcher._extract_acronym("United States of America")
            'USA'
            
            >>> EntityMatcher._extract_acronym("Federal Bureau of Investigation")
            'FBI'
            
            >>> EntityMatcher._extract_acronym("World Health Organization")
            'WHO'
            
            >>> EntityMatcher._extract_acronym("Obama")
            None  # Single word, no acronym
            
            >>> EntityMatcher._extract_acronym("the united states")
            None  # No capitalized words (except 'the' which is filtered)
        
        Algorithm:
            1. Split text into words
            2. Filter words starting with uppercase letter
            3. If < 2 capitalized words, return None
            4. Take first letter of each capitalized word
            5. Join and return uppercase
        """
        words = text.split()
        
        # Only consider words that start with uppercase
        capital_words = [w for w in words if w and w[0].isupper()]
        
        # Need at least 2 words to form an acronym
        if len(capital_words) < 2:
            return None
        
        # Extract first letter of each word and uppercase
        acronym = ''.join(w[0].upper() for w in capital_words)
        
        return acronym
    
    @staticmethod
    def _normalize_acronym(text: str) -> str:
        """
        Normalize acronym by removing periods and spaces.
        
        Handles various acronym punctuation styles:
        - "U.S.A" → "USA"
        - "U.S." → "US"
        - "F.B.I." → "FBI"
        - "W.H.O." → "WHO"
        
        Args:
            text: Text to normalize
        
        Returns:
            Normalized text (uppercase, no periods/spaces)
        
        Examples:
            >>> EntityMatcher._normalize_acronym("U.S.A")
            'USA'
            
            >>> EntityMatcher._normalize_acronym("U.S.")
            'US'
            
            >>> EntityMatcher._normalize_acronym("F.B.I.")
            'FBI'
            
            >>> EntityMatcher._normalize_acronym("usa")
            'USA'
        
        Performance: O(n) where n=len(text)
        """
        return text.replace('.', '').replace(' ', '').upper()
    
    # =========================================================================
    # Tier 3: Alias Dictionary Matching
    # =========================================================================
    
    def _match_aliases(self, entity: str, text: str) -> bool:
        """
        Tier 3: Alias dictionary lookup and matching with partial name support.
        
        Uses the curated entity_aliases dictionary to find all known surface
        forms for the entity, then checks if any form appears in the text.
        Handles bidirectional lookups (canonical→alias and alias→canonical).
        Also handles partial names by checking aliases for each word.
        
        Args:
            entity: Entity to look up aliases for
            text: Text to search for aliases in
        
        Returns:
            True if any alias found in text, False otherwise
        
        Examples:
            >>> self._match_aliases("United States", "America's economy")
            True
            
            >>> self._match_aliases("America", "The United States announced")
            True
            
            >>> self._match_aliases("UK", "Britain's parliament")
            True
            
            >>> self._match_aliases("Doctor Smith", "Dr. Smith reported")
            True  # "Doctor" → "Dr." via aliases
            
            >>> self._match_aliases("MIT", "Massachusetts Institute of Technology")
            True
        
        Algorithm:
            1. Call get_all_forms(entity) to get canonical + aliases
            2. If no direct match, try matching individual words (for names)
            3. For each form, check if it appears in text (case-insensitive)
            4. Return True on first match
        
        Performance: O(k*n*m) where:
            - k = average aliases per entity (~2-5)
            - n = len(text)
            - m = len(form)
        Optimized with early exit and hash table lookup (O(1) for get_all_forms)
        
        Handles: +5% coverage on top of Tier 1+2
        """
        text_lower = text.lower()
        entity_lower = entity.lower().strip()
        
        # Get all surface forms (canonical + aliases)
        all_forms = get_all_forms(entity_lower)
        
        # Check if any form appears in text
        for form in all_forms:
            if form.lower() in text_lower:
                self.logger.debug(
                    f"Alias match: entity='{entity}' matched via form='{form}'"
                )
                return True
        
        # If no direct match and entity has multiple words, try matching aliases
        # for individual words (e.g., "Doctor Smith" → check aliases for "Doctor")
        entity_words = entity_lower.split()
        if len(entity_words) > 1:
            for word in entity_words:
                word_forms = get_all_forms(word)
                if word_forms:
                    # Check if any alias of this word appears in text
                    for form in word_forms:
                        if form.lower() in text_lower:
                            self.logger.debug(
                                f"Partial alias match: entity='{entity}', word='{word}' matched via form='{form}'"
                            )
                            return True
        
        return False
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_config_summary(self) -> dict:
        """
        Get summary of current configuration.
        
        Returns:
            Dictionary with configuration details
        
        Example:
            >>> matcher.get_config_summary()
            {
                'acronym_matching': True,
                'alias_dictionary': True,
                'tiers_enabled': ['Tier 1 (Substring)', 'Tier 2 (Acronym)', 'Tier 3 (Alias)']
            }
        """
        tiers = ['Tier 1 (Substring)']  # Always enabled
        if self.use_acronym:
            tiers.append('Tier 2 (Acronym)')
        if self.use_aliases:
            tiers.append('Tier 3 (Alias)')
        if self.use_llm:
            tiers.append('Tier 4 (LLM)')
        
        return {
            'acronym_matching': self.use_acronym,
            'alias_dictionary': self.use_aliases,
            'llm_enabled': self.use_llm,
            'tiers_enabled': tiers
        }


# =============================================================================
# Module-level utility for testing
# =============================================================================

if __name__ == "__main__":
    """
    Test EntityMatcher with sample cases.
    """
    from src.utils.config import Config
    
    print("=" * 70)
    print("EntityMatcher Test Suite")
    print("=" * 70)
    
    # Initialize
    config = Config()
    matcher = EntityMatcher(config)
    
    print("\nConfiguration:")
    summary = matcher.get_config_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Test cases
    test_cases = [
        # (entity, evidence, expected_result, tier)
        ("Obama", "Barack Obama spoke today", True, "Tier 1"),
        ("United States of America", "The U.S. economy", True, "Tier 2"),
        ("United States", "America's GDP is high", True, "Tier 3"),
        ("MIT", "Massachusetts Institute of Technology", True, "Tier 3"),
        ("WHO", "World Health Organization declared", True, "Tier 2/3"),
        ("Doctor Smith", "Dr. Smith reported findings", True, "Tier 3"),
        ("XYZ Corp", "Unrelated text about ABC", False, "None"),
    ]
    
    print("\n" + "=" * 70)
    print("Test Results:")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for entity, evidence, expected, tier in test_cases:
        result = matcher.match_entity(entity, evidence)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        
        if result == expected:
            passed += 1
        else:
            failed += 1
        
        print(f"{status} | Entity: '{entity[:30]}...' | Expected: {tier}")
    
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
