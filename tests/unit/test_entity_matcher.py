"""
Comprehensive Unit Tests for EntityMatcher.

This test suite covers all three matching tiers, edge cases, configuration
options, and integration scenarios for the EntityMatcher class.

Test Organization:
- TestInitialization: Config loading and initialization
- TestTier1Substring: Basic substring matching
- TestTier2Acronym: Acronym extraction and matching
- TestTier3Aliases: Dictionary-based alias matching
- TestIntegration: Tiered fallback and configuration
- TestEdgeCases: Empty strings, special characters, boundary conditions
- TestPerformance: Benchmark tests (<10ms target)

Usage:
    pytest tests/unit/test_entity_matcher.py -v
    pytest tests/unit/test_entity_matcher.py -v --benchmark-only
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.verification.entity_matcher import EntityMatcher
from src.verification.entity_aliases import get_all_forms, get_canonical_form
from src.utils.config import Config


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_config():
    """Create a sample configuration with all tiers enabled."""
    config = Config()
    # Add matching config (graceful handling if not in config.yaml yet)
    if not hasattr(config, 'verification'):
        config.verification = type('obj', (object,), {})()
    if not hasattr(config.verification, 'grounded'):
        config.verification.grounded = type('obj', (object,), {})()
    
    config.verification.grounded.matching = {
        'acronym_matching': True,
        'alias_dictionary': True
    }
    return config


@pytest.fixture
def config_no_acronym():
    """Configuration with acronym matching disabled."""
    # Create a mock config object with disabled acronym matching
    class MockConfig:
        def __init__(self):
            self.verification = type('obj', (object,), {})()
            self.verification.grounded = type('obj', (object,), {})()
            self.verification.grounded.matching = {
                'acronym_matching': False,
                'alias_dictionary': True
            }
    return MockConfig()


@pytest.fixture
def config_no_aliases():
    """Configuration with alias dictionary disabled."""
    class MockConfig:
        def __init__(self):
            self.verification = type('obj', (object,), {})()
            self.verification.grounded = type('obj', (object,), {})()
            self.verification.grounded.matching = {
                'acronym_matching': True,
                'alias_dictionary': False
            }
    return MockConfig()


@pytest.fixture
def config_substring_only():
    """Configuration with only Tier 1 (substring) enabled."""
    class MockConfig:
        def __init__(self):
            self.verification = type('obj', (object,), {})()
            self.verification.grounded = type('obj', (object,), {})()
            self.verification.grounded.matching = {
                'acronym_matching': False,
                'alias_dictionary': False
            }
    return MockConfig()


@pytest.fixture
def matcher(sample_config):
    """Create EntityMatcher instance with default config."""
    return EntityMatcher(sample_config)


# =============================================================================
# Test Class 1: Initialization and Configuration
# =============================================================================

class TestInitialization:
    """Test EntityMatcher initialization and configuration loading."""
    
    def test_initialization_with_config(self, sample_config):
        """Test that EntityMatcher initializes correctly with config."""
        matcher = EntityMatcher(sample_config)
        
        assert matcher.config == sample_config
        assert matcher.use_acronym == True
        assert matcher.use_aliases == True
    
    def test_initialization_defaults(self):
        """Test initialization with minimal config (should use defaults)."""
        config = Config()
        matcher = EntityMatcher(config)
        
        # Should default to True even if config missing
        assert matcher.use_acronym == True
        assert matcher.use_aliases == True
    
    def test_config_acronym_disabled(self, config_no_acronym):
        """Test that acronym matching can be disabled."""
        matcher = EntityMatcher(config_no_acronym)
        assert matcher.use_acronym == False
        assert matcher.use_aliases == True
    
    def test_config_aliases_disabled(self, config_no_aliases):
        """Test that alias dictionary can be disabled."""
        matcher = EntityMatcher(config_no_aliases)
        assert matcher.use_acronym == True
        assert matcher.use_aliases == False
    
    def test_config_all_disabled(self, config_substring_only):
        """Test with only Tier 1 (substring) enabled."""
        matcher = EntityMatcher(config_substring_only)
        assert matcher.use_acronym == False
        assert matcher.use_aliases == False
    
    def test_get_config_summary(self, matcher):
        """Test configuration summary method."""
        summary = matcher.get_config_summary()
        
        assert 'acronym_matching' in summary
        assert 'alias_dictionary' in summary
        assert 'tiers_enabled' in summary
        assert summary['acronym_matching'] == True
        assert summary['alias_dictionary'] == True
        assert len(summary['tiers_enabled']) == 4  # All tiers enabled


# =============================================================================
# Test Class 2: Tier 1 - Substring Matching
# =============================================================================

class TestTier1Substring:
    """Test Tier 1 (substring) matching functionality."""
    
    def test_exact_match(self, matcher):
        """Test exact substring match."""
        assert matcher.match_entity("Obama", "Obama spoke today") == True
    
    def test_partial_match_longer_entity(self, matcher):
        """Test partial match where entity is longer than occurrence."""
        assert matcher.match_entity("Barack Obama", "Obama spoke") == True
    
    def test_partial_match_entity_in_middle(self, matcher):
        """Test entity appearing in middle of text."""
        assert matcher.match_entity("Obama", "President Obama visited") == True
    
    def test_case_insensitive_upper_to_lower(self, matcher):
        """Test case-insensitive matching (upper entity, lower text)."""
        assert matcher.match_entity("NASA", "nasa launched") == True
    
    def test_case_insensitive_lower_to_upper(self, matcher):
        """Test case-insensitive matching (lower entity, upper text)."""
        assert matcher.match_entity("nasa", "NASA launched") == True
    
    def test_case_insensitive_mixed(self, matcher):
        """Test case-insensitive matching with mixed case."""
        assert matcher.match_entity("Barack Obama", "BARACK OBAMA spoke") == True
    
    def test_no_match(self, matcher):
        """Test that non-matching entities return False."""
        assert matcher.match_entity("Trump", "Obama spoke today") == False
    
    def test_with_punctuation(self, matcher):
        """Test matching with punctuation in text."""
        assert matcher.match_entity("Obama", "Obama, the president, spoke") == True
    
    def test_multiword_entity(self, matcher):
        """Test matching multi-word entities."""
        assert matcher.match_entity("Barack Obama", "Barack Obama spoke") == True


# =============================================================================
# Test Class 3: Tier 2 - Acronym Matching
# =============================================================================

class TestTier2Acronym:
    """Test Tier 2 (acronym) matching functionality."""
    
    def test_usa_variants_dots(self, matcher):
        """Test USA acronym with periods (U.S.A)."""
        entity = "United States of America"
        assert matcher.match_entity(entity, "The U.S.A declared war") == True
    
    def test_usa_variants_no_dots(self, matcher):
        """Test USA acronym without periods."""
        entity = "United States of America"
        assert matcher.match_entity(entity, "USA announced today") == True
    
    def test_usa_variants_us_dots(self, matcher):
        """Test US acronym with periods (U.S.)."""
        entity = "United States of America"
        assert matcher.match_entity(entity, "The U.S. economy is strong") == True
    
    def test_who_match(self, matcher):
        """Test WHO acronym matching."""
        entity = "World Health Organization"
        assert matcher.match_entity(entity, "WHO declared a pandemic") == True
    
    def test_fbi_with_periods(self, matcher):
        """Test FBI with periods (F.B.I.)."""
        entity = "Federal Bureau of Investigation"
        assert matcher.match_entity(entity, "The F.B.I. investigated") == True
    
    def test_fbi_without_periods(self, matcher):
        """Test FBI without periods."""
        entity = "Federal Bureau of Investigation"
        assert matcher.match_entity(entity, "FBI agents arrived") == True
    
    def test_nato_match(self, matcher):
        """Test NATO acronym."""
        entity = "North Atlantic Treaty Organization"
        assert matcher.match_entity(entity, "NATO members agreed") == True
    
    def test_mit_match(self, matcher):
        """Test MIT acronym."""
        entity = "Massachusetts Institute of Technology"
        assert matcher.match_entity(entity, "MIT researchers published") == True
    
    def test_single_word_no_acronym(self, matcher):
        """Test that single words don't create acronyms."""
        # Should not match via acronym (would need substring)
        assert matcher._match_acronym("Obama", "The president spoke") == False
    
    def test_lowercase_phrase_no_acronym(self, matcher):
        """Test that lowercase phrases don't create acronyms."""
        assert matcher._extract_acronym("the united states") == None
    
    def test_acronym_disabled(self, config_no_acronym):
        """Test that acronym matching can be disabled."""
        matcher = EntityMatcher(config_no_acronym)
        # Use entity that's NOT in alias dictionary to isolate acronym tier
        entity = "Federal Aviation Administration"
        # When acronym matching disabled, FAA should NOT match
        assert matcher.match_entity(entity, "The FAA announced") == False  # Acronym disabled
        # But substring should still work
        assert matcher.match_entity(entity, "Federal Aviation Administration announced") == True


# =============================================================================
# Test Class 4: Tier 3 - Alias Matching
# =============================================================================

class TestTier3Aliases:
    """Test Tier 3 (alias dictionary) matching functionality."""
    
    def test_usa_to_america(self, matcher):
        """Test United States matches America."""
        assert matcher.match_entity("United States", "America's economy is strong") == True
    
    def test_america_to_usa(self, matcher):
        """Test bidirectional: America matches United States."""
        assert matcher.match_entity("America", "The United States announced") == True
    
    def test_uk_to_britain(self, matcher):
        """Test UK matches Britain."""
        assert matcher.match_entity("United Kingdom", "Britain's parliament voted") == True
    
    def test_britain_to_uk(self, matcher):
        """Test bidirectional: Britain matches UK."""
        assert matcher.match_entity("Britain", "The UK government said") == True
    
    def test_doctor_abbreviation(self, matcher):
        """Test Doctor matches Dr."""
        assert matcher.match_entity("Doctor Smith", "Dr. Smith reported") == True
    
    def test_dr_to_doctor(self, matcher):
        """Test bidirectional: Dr. matches Doctor."""
        assert matcher.match_entity("Dr. Jones", "Doctor Jones examined") == True
    
    def test_professor_abbreviation(self, matcher):
        """Test Professor matches Prof."""
        assert matcher.match_entity("Professor Wang", "Prof. Wang teaches") == True
    
    def test_mit_full_name(self, matcher):
        """Test MIT matches full name."""
        assert matcher.match_entity("MIT", "Massachusetts Institute of Technology published") == True
    
    def test_full_name_to_mit(self, matcher):
        """Test bidirectional: full name matches MIT."""
        assert matcher.match_entity("Massachusetts Institute of Technology", "MIT researchers") == True
    
    def test_who_full_name(self, matcher):
        """Test WHO matches full name."""
        assert matcher.match_entity("WHO", "World Health Organization declared") == True
    
    def test_aliases_disabled(self, config_no_aliases):
        """Test that alias matching can be disabled."""
        matcher = EntityMatcher(config_no_aliases)
        # Should not match via aliases
        assert matcher.match_entity("United States", "America's economy") == False


# =============================================================================
# Test Class 5: Integration Tests
# =============================================================================

class TestIntegration:
    """Test integration of all tiers and tiered fallback."""
    
    def test_tiered_fallback_tier1(self, matcher):
        """Test that Tier 1 matches and exits early."""
        entity = "United States of America"
        # Full name in evidence -> Tier 1 match
        assert matcher.match_entity(entity, "United States of America is large") == True
    
    def test_tiered_fallback_tier2(self, matcher):
        """Test that Tier 2 matches when Tier 1 fails."""
        entity = "United States of America"
        # Acronym in evidence -> Tier 2 match
        assert matcher.match_entity(entity, "The USA is large") == True
    
    def test_tiered_fallback_tier3(self, matcher):
        """Test that Tier 3 matches when Tier 1 and 2 fail."""
        entity = "United States of America"
        # Alias in evidence -> Tier 3 match
        assert matcher.match_entity(entity, "America is large") == True
    
    def test_multiple_tiers_match_first_wins(self, matcher):
        """Test that first matching tier wins (early exit)."""
        entity = "United States"
        # Both substring and alias would match, but substring is faster
        result = matcher.match_entity(entity, "The United States of America")
        assert result == True
    
    def test_no_tier_matches(self, matcher):
        """Test that False returned when no tier matches."""
        assert matcher.match_entity("XYZ Corporation", "ABC Company announced") == False
    
    def test_academic_entity_mit(self, matcher):
        """Test academic entity matching (MIT)."""
        # Tier 3 (alias) should match
        assert matcher.match_entity("MIT", "Massachusetts Institute of Technology") == True
        # Tier 2 (acronym) should also work
        assert matcher.match_entity("Massachusetts Institute of Technology", "MIT published") == True
    
    def test_academic_entity_phd(self, matcher):
        """Test academic title matching (PhD)."""
        assert matcher.match_entity("Doctor of Philosophy", "PhD student") == True
        assert matcher.match_entity("PhD", "Doctor of Philosophy degree") == True


# =============================================================================
# Test Class 6: Tier 4 - LLM Matching
# =============================================================================

class TestTier4LLM:
    """Test Tier 4 (LLM) matching functionality with mocked API calls."""

    def test_llm_match_success_with_quote(self, matcher, monkeypatch):
        """Test LLM match when evidence quote exists."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        matcher.use_llm = True
        matcher.llm_max_retries = 0
        matcher.llm_cache_size = 4

        def fake_post_json(self, url, headers, payload, timeout_s):
            return {
                "choices": [
                    {"message": {"content": '{"match": true, "matched_surface_form": "Zeta Corporation", "evidence_quote": "Zeta Corporation", "confidence": 0.9, "rationale_short": "Exact mention"}'}}
                ]
            }

        monkeypatch.setattr(EntityMatcher, "_post_json", fake_post_json)

        entity = "ZetaCorp"
        evidence = "The merger involved Zeta Corporation and Alpha LLC."
        assert matcher.match_entity(entity, evidence) == True

    def test_llm_match_fails_when_quote_missing(self, matcher, monkeypatch):
        """Test LLM match rejected if quoted evidence not found."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        matcher.use_llm = True
        matcher.llm_max_retries = 0

        def fake_post_json(self, url, headers, payload, timeout_s):
            return {
                "choices": [
                    {"message": {"content": '{"match": true, "matched_surface_form": "Zeta Corporation", "evidence_quote": "Nonexistent Quote", "confidence": 0.9, "rationale_short": "Exact mention"}'}}
                ]
            }

        monkeypatch.setattr(EntityMatcher, "_post_json", fake_post_json)

        entity = "ZetaCorp"
        evidence = "The merger involved Zeta Corporation and Alpha LLC."
        assert matcher.match_entity(entity, evidence) == False

    def test_llm_invalid_json_response(self, matcher, monkeypatch):
        """Test invalid JSON in LLM response returns False."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        matcher.use_llm = True
        matcher.llm_max_retries = 0

        def fake_post_json(self, url, headers, payload, timeout_s):
            return {"choices": [{"message": {"content": "not-json"}}]}

        monkeypatch.setattr(EntityMatcher, "_post_json", fake_post_json)

        entity = "ZetaCorp"
        evidence = "The merger involved Zeta Corporation and Alpha LLC."
        assert matcher.match_entity(entity, evidence) == False

    def test_llm_cache_hit(self, matcher, monkeypatch):
        """Test repeated LLM requests are served from cache."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        matcher.use_llm = True
        matcher.llm_max_retries = 0
        matcher.llm_cache_size = 4

        call_count = {"n": 0}

        def fake_post_json(self, url, headers, payload, timeout_s):
            call_count["n"] += 1
            return {
                "choices": [
                    {"message": {"content": '{"match": true, "matched_surface_form": "Zeta Corporation", "evidence_quote": "Zeta Corporation", "confidence": 0.9, "rationale_short": "Exact mention"}'}}
                ]
            }

        monkeypatch.setattr(EntityMatcher, "_post_json", fake_post_json)

        entity = "ZetaCorp"
        evidence = "The merger involved Zeta Corporation and Alpha LLC."
        assert matcher.match_entity(entity, evidence) == True
        assert matcher.match_entity(entity, evidence) == True
        assert call_count["n"] == 1


# =============================================================================
# Test Class 7: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_entity(self, matcher):
        """Test empty entity string."""
        assert matcher.match_entity("", "Some text") == False
    
    def test_empty_evidence(self, matcher):
        """Test empty evidence string."""
        assert matcher.match_entity("Entity", "") == False
    
    def test_both_empty(self, matcher):
        """Test both entity and evidence empty."""
        assert matcher.match_entity("", "") == False
    
    def test_whitespace_only_entity(self, matcher):
        """Test entity with only whitespace."""
        assert matcher.match_entity("   ", "Some text") == False
    
    def test_whitespace_only_evidence(self, matcher):
        """Test evidence with only whitespace."""
        assert matcher.match_entity("Entity", "   ") == False
    
    def test_special_characters_in_entity(self, matcher):
        """Test entity with special characters."""
        assert matcher.match_entity("C++", "C++ programming language") == True
    
    def test_special_characters_in_evidence(self, matcher):
        """Test evidence with special characters."""
        assert matcher.match_entity("Obama", "Obama's speech (2008) was...") == True
    
    def test_numbers_in_entity(self, matcher):
        """Test entity with numbers."""
        assert matcher.match_entity("F-16", "The F-16 fighter jet") == True
    
    def test_unicode_characters(self, matcher):
        """Test Unicode characters (Chinese, accents)."""
        # Chinese characters
        assert matcher.match_entity("北京", "北京是中国的首都") == True
        # Accented characters
        assert matcher.match_entity("café", "The café was open") == True
    
    def test_very_long_entity(self, matcher):
        """Test with very long entity name."""
        long_entity = "The International Organization for Standardization of Educational Materials"
        evidence = long_entity + " announced new standards"
        assert matcher.match_entity(long_entity, evidence) == True
    
    def test_very_long_evidence(self, matcher):
        """Test with very long evidence text."""
        entity = "Obama"
        evidence = "Lorem ipsum " * 100 + "Obama spoke" + " dolor sit amet" * 100
        assert matcher.match_entity(entity, evidence) == True


# =============================================================================
# Test Class 7: Helper Methods
# =============================================================================

class TestHelperMethods:
    """Test internal helper methods."""
    
    def test_extract_acronym_valid(self):
        """Test acronym extraction with valid multi-word phrase."""
        assert EntityMatcher._extract_acronym("United States of America") == "USA"
        assert EntityMatcher._extract_acronym("World Health Organization") == "WHO"
        assert EntityMatcher._extract_acronym("Federal Bureau of Investigation") == "FBI"
    
    def test_extract_acronym_single_word(self):
        """Test acronym extraction with single word (should return None)."""
        assert EntityMatcher._extract_acronym("Obama") is None
        assert EntityMatcher._extract_acronym("America") is None
    
    def test_extract_acronym_no_capitals(self):
        """Test acronym extraction with no capitalized words."""
        assert EntityMatcher._extract_acronym("the united states") is None
        assert EntityMatcher._extract_acronym("of the people") is None
    
    def test_normalize_acronym_with_dots(self):
        """Test acronym normalization with periods."""
        assert EntityMatcher._normalize_acronym("U.S.A") == "USA"
        assert EntityMatcher._normalize_acronym("U.S.") == "US"
        assert EntityMatcher._normalize_acronym("F.B.I.") == "FBI"
    
    def test_normalize_acronym_with_spaces(self):
        """Test acronym normalization with spaces."""
        assert EntityMatcher._normalize_acronym("U S A") == "USA"
        assert EntityMatcher._normalize_acronym("W H O") == "WHO"
    
    def test_normalize_acronym_lowercase(self):
        """Test acronym normalization converts to uppercase."""
        assert EntityMatcher._normalize_acronym("usa") == "USA"
        assert EntityMatcher._normalize_acronym("who") == "WHO"
    
    def test_normalize_acronym_mixed(self):
        """Test acronym normalization with mixed formatting."""
        assert EntityMatcher._normalize_acronym("u.s.a") == "USA"
        assert EntityMatcher._normalize_acronym("U.s.A.") == "USA"


# =============================================================================
# Test Class 8: Performance Benchmarks
# =============================================================================

class TestPerformance:
    """Performance benchmarks for entity matching (requires pytest-benchmark)."""
    
    def test_performance_tier1_substring(self, matcher, benchmark):
        """Benchmark Tier 1 (substring) matching."""
        entity = "Barack Obama"
        evidence = "President Barack Obama spoke at the United Nations today about climate change"
        
        result = benchmark(matcher.match_entity, entity, evidence)
        assert result == True
        # Should complete in <10ms (benchmark will show actual time)
    
    def test_performance_tier2_acronym(self, matcher, benchmark):
        """Benchmark Tier 2 (acronym) matching."""
        entity = "United States of America"
        evidence = "The U.S. economy grew by 2.5% last quarter according to new data"
        
        result = benchmark(matcher.match_entity, entity, evidence)
        assert result == True
    
    def test_performance_tier3_aliases(self, matcher, benchmark):
        """Benchmark Tier 3 (alias) matching."""
        entity = "United States"
        evidence = "America's foreign policy has shifted under the new administration"
        
        result = benchmark(matcher.match_entity, entity, evidence)
        assert result == True
    
    def test_performance_no_match(self, matcher, benchmark):
        """Benchmark performance when no tier matches (worst case)."""
        entity = "XYZ Corporation"
        evidence = "The ABC Company announced record profits this quarter in a surprise"
        
        result = benchmark(matcher.match_entity, entity, evidence)
        assert result == False
    
    def test_performance_long_evidence(self, matcher, benchmark):
        """Benchmark with long evidence text."""
        entity = "Obama"
        evidence = ("Lorem ipsum dolor sit amet " * 50 + 
                   "President Obama spoke today " + 
                   "consectetur adipiscing elit " * 50)
        
        result = benchmark(matcher.match_entity, entity, evidence)
        assert result == True


# =============================================================================
# Test Class 9: Alias Dictionary Integration
# =============================================================================

class TestAliasDictionaryIntegration:
    """Test integration with entity_aliases module."""
    
    def test_get_all_forms_used_correctly(self, matcher):
        """Test that get_all_forms is used correctly in alias matching."""
        # This entity should have multiple forms in dictionary
        entity = "United States"
        evidence = "America announced"
        
        # Should match via alias
        assert matcher.match_entity(entity, evidence) == True
    
    def test_bidirectional_lookup_works(self, matcher):
        """Test bidirectional alias lookup."""
        # Canonical → alias
        assert matcher.match_entity("united states of america", "USA today") == True
        # Alias → canonical
        assert matcher.match_entity("usa", "United States of America") == True
    
    def test_dictionary_coverage_academic(self, matcher):
        """Test that academic entities from dictionary work."""
        assert matcher.match_entity("MIT", "Massachusetts Institute of Technology") == True
        assert matcher.match_entity("Stanford University", "Stanford researchers") == True
        assert matcher.match_entity("PhD", "Doctor of Philosophy") == True


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    """Run tests with pytest."""
    import subprocess
    import os
    
    # Change to project root
    os.chdir(Path(__file__).parent.parent.parent)
    
    # Run pytest with verbose output
    result = subprocess.run(
        ["pytest", "tests/unit/test_entity_matcher.py", "-v", "--tb=short"],
        capture_output=False
    )
    
    sys.exit(result.returncode)
