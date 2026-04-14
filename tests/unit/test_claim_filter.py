"""
Unit tests for ClaimFilter.

Tests claim filtering logic, text replacement, and configuration handling.
"""

import pytest
from unittest.mock import MagicMock

from src.mitigation.claim_filter import ClaimFilter
from src.utils.data_structures import Claim, ClaimDecision
from src.utils.config import Config


@pytest.fixture
def sample_config():
    """Create a Config object with filter settings."""
    config = MagicMock(spec=Config)
    config.get.return_value = {
        'filter': {
            'enabled': True,
            'placeholder': '[CLAIM REMOVED: Contradictory]'
        }
    }
    return config


@pytest.fixture
def sample_answer_text():
    """Create sample answer text with multiple claims."""
    return "Paris is the capital of France. London is in Asia. Berlin is the capital of Germany."


@pytest.fixture
def sample_claims():
    """Create sample claims with character spans."""
    # Text: "Paris is the capital of France. London is in Asia. Berlin is the capital of Germany."
    # Positions: [0-31] is "Paris..." sentence, [32-50] is "London..." sentence, [51-84] is "Berlin..." sentence
    return [
        Claim(
            claim_id="c1",
            answer_id="a1",
            text="Paris is the capital of France.",
            answer_char_span=[0, 32]  # Includes the period and space after
        ),
        Claim(
            claim_id="c2",
            answer_id="a1",
            text="London is in Asia.",
            answer_char_span=[32, 51]  # From 'L' to period+space
        ),
        Claim(
            claim_id="c3",
            answer_id="a1",
            text="Berlin is the capital of Germany.",
            answer_char_span=[51, 84]  # From 'B' to end
        )
    ]


@pytest.fixture
def sample_decisions_mixed():
    """Create sample decisions with mixed verdicts."""
    return [
        ClaimDecision(
            claim_id="c1",
            status="Supported",
            rationale="High entailment and coverage",
            primary_evidence="doc1#0",
            signals_ref=["signal1"],
            confidence={'overall_confidence': 85, 'band': 'High'}
        ),
        ClaimDecision(
            claim_id="c2",
            status="Contradictory",
            rationale="High NLI contradiction score",
            primary_evidence="doc2#1",
            signals_ref=["signal2"],
            confidence={'overall_confidence': 90, 'band': 'High'}
        ),
        ClaimDecision(
            claim_id="c3",
            status="Supported",
            rationale="Good coverage",
            primary_evidence="doc3#2",
            signals_ref=["signal3"],
            confidence={'overall_confidence': 80, 'band': 'High'}
        )
    ]


class TestClaimFilterInitialization:
    """Test ClaimFilter initialization and configuration loading."""
    
    def test_init_with_default_config(self, sample_config):
        """Test initialization with provided config."""
        filter = ClaimFilter(sample_config)
        
        assert filter.placeholder == '[CLAIM REMOVED: Contradictory]'
        assert filter.enabled is True
    
    def test_init_with_custom_placeholder(self):
        """Test initialization with custom placeholder text."""
        config = MagicMock(spec=Config)
        config.get.return_value = {
            'filter': {
                'placeholder': '[REDACTED]',
                'enabled': True
            }
        }
        
        filter = ClaimFilter(config)
        
        assert filter.placeholder == '[REDACTED]'
    
    def test_init_with_disabled_filtering(self):
        """Test initialization with filtering disabled."""
        config = MagicMock(spec=Config)
        config.get.return_value = {
            'filter': {
                'enabled': False,
                'placeholder': '[CLAIM REMOVED: Contradictory]'
            }
        }
        
        filter = ClaimFilter(config)
        
        assert filter.enabled is False
    
    def test_init_with_missing_config(self):
        """Test initialization with missing config (uses defaults)."""
        config = MagicMock(spec=Config)
        config.get.return_value = {}  # Empty config
        
        filter = ClaimFilter(config)
        
        # Should use defaults
        assert filter.placeholder == '[CLAIM REMOVED: Contradictory]'
        assert filter.enabled is True


class TestClaimFiltering:
    """Test core claim filtering functionality."""
    
    def test_filter_single_contradictory_claim(
        self, sample_config, sample_answer_text, sample_claims, sample_decisions_mixed
    ):
        """Test filtering a single contradictory claim."""
        filter = ClaimFilter(sample_config)
        
        filtered_text, removed_count, _ = filter.filter_answer(
            sample_answer_text, sample_claims, sample_decisions_mixed
        )
        
        # Should replace "London is in Asia." with placeholder
        # Note: The space between claims is part of claim spans, so it gets replaced too
        expected = "Paris is the capital of France. [CLAIM REMOVED: Contradictory]Berlin is the capital of Germany."
        
        assert filtered_text == expected
        assert removed_count == 1
    
    def test_filter_multiple_contradictory_claims(self, sample_config):
        """Test filtering multiple contradictory claims."""
        filter = ClaimFilter(sample_config)
        
        answer = "Claim 1. Claim 2. Claim 3."
        claims = [
            Claim(
                claim_id="c1",
                answer_id="a1",
                text="Claim 1.",
                answer_char_span=[0, 8]
            ),
            Claim(
                claim_id="c2",
                answer_id="a1",
                text="Claim 2.",
                answer_char_span=[9, 17]
            ),
            Claim(
                claim_id="c3",
                answer_id="a1",
                text="Claim 3.",
                answer_char_span=[18, 26]
            )
        ]
        
        decisions = [
            ClaimDecision(
                claim_id="c1",
                status="Contradictory",
                rationale="",
                primary_evidence="",
                signals_ref=[],
                confidence={}
            ),
            ClaimDecision(
                claim_id="c2",
                status="Supported",
                rationale="",
                primary_evidence="",
                signals_ref=[],
                confidence={}
            ),
            ClaimDecision(
                claim_id="c3",
                status="Contradictory",
                rationale="",
                primary_evidence="",
                signals_ref=[],
                confidence={}
            )
        ]
        
        filtered_text, removed_count, _ = filter.filter_answer(answer, claims, decisions)
        
        # Should replace claims 1 and 3
        expected = "[CLAIM REMOVED: Contradictory] Claim 2. [CLAIM REMOVED: Contradictory]"
        
        assert filtered_text == expected
        assert removed_count == 2
    
    def test_filter_no_contradictory_claims(
        self, sample_config, sample_answer_text, sample_claims
    ):
        """Test filtering when no contradictory claims exist."""
        filter = ClaimFilter(sample_config)
        
        # All claims supported
        decisions = [
            ClaimDecision(
                claim_id=f"c{i+1}",
                status="Supported",
                rationale="",
                primary_evidence="",
                signals_ref=[],
                confidence={}
            )
            for i in range(3)
        ]
        
        filtered_text, removed_count, _ = filter.filter_answer(
            sample_answer_text, sample_claims, decisions
        )
        
        # Should return unchanged
        assert filtered_text == sample_answer_text
        assert removed_count == 0
    
    def test_filter_disabled_returns_original(
        self, sample_answer_text, sample_claims, sample_decisions_mixed
    ):
        """Test that disabled filtering returns original text."""
        config = MagicMock(spec=Config)
        config.get.return_value = {
            'filter': {
                'enabled': False,
                'placeholder': '[CLAIM REMOVED: Contradictory]'
            }
        }
        
        filter = ClaimFilter(config)
        
        filtered_text, removed_count, _ = filter.filter_answer(
            sample_answer_text, sample_claims, sample_decisions_mixed
        )
        
        # Should return original unchanged
        assert filtered_text == sample_answer_text
        assert removed_count == 0
    
    def test_filter_with_empty_claims(self, sample_config):
        """Test filtering with empty claims list."""
        filter = ClaimFilter(sample_config)
        
        filtered_text, removed_count, _ = filter.filter_answer("Test", [], [])
        
        assert filtered_text == "Test"
        assert removed_count == 0
    
    def test_filter_with_mismatched_lengths(
        self, sample_config, sample_answer_text, sample_claims
    ):
        """Test that mismatched claims/decisions raises ValueError."""
        filter = ClaimFilter(sample_config)
        
        # Only 2 decisions for 3 claims
        decisions = [
            ClaimDecision(
                claim_id="c1",
                status="Supported",
                rationale="",
                primary_evidence="",
                signals_ref=[],
                confidence={}
            )
        ]
        
        with pytest.raises(ValueError, match="Claims and decisions must have same length"):
            filter.filter_answer(sample_answer_text, sample_claims, decisions)
    
    def test_filter_preserves_text_integrity(self, sample_config):
        """Test that filtering doesn't corrupt surrounding text."""
        filter = ClaimFilter(sample_config)
        
        answer = "Before. Contradictory claim. After."
        claims = [
            Claim(
                claim_id="c1",
                answer_id="a1",
                text="Before.",
                answer_char_span=[0, 7]
            ),
            Claim(
                claim_id="c2",
                answer_id="a1",
                text="Contradictory claim.",
                answer_char_span=[8, 28]
            ),
            Claim(
                claim_id="c3",
                answer_id="a1",
                text="After.",
                answer_char_span=[29, 35]
            )
        ]
        
        decisions = [
            ClaimDecision(claim_id="c1", status="Supported", rationale="", primary_evidence="", signals_ref=[], confidence={}),
            ClaimDecision(claim_id="c2", status="Contradictory", rationale="", primary_evidence="", signals_ref=[], confidence={}),
            ClaimDecision(claim_id="c3", status="Supported", rationale="", primary_evidence="", signals_ref=[], confidence={})
        ]
        
        filtered_text, removed_count, _ = filter.filter_answer(answer, claims, decisions)
        
        # "Before." and "After." should be intact
        assert "Before." in filtered_text
        assert "After." in filtered_text
        assert "Contradictory claim." not in filtered_text
        assert removed_count == 1


class TestReverseOrderProcessing:
    """Test that claims are processed in reverse order to avoid index corruption."""
    
    def test_reverse_order_prevents_index_corruption(self, sample_config):
        """Test that reverse-order processing maintains correct spans."""
        filter = ClaimFilter(sample_config)
        
        # Three consecutive claims
        answer = "First. Second. Third."
        claims = [
            Claim(claim_id="c1", answer_id="a1", text="First.", answer_char_span=[0, 6]),
            Claim(claim_id="c2", answer_id="a1", text="Second.", answer_char_span=[7, 14]),
            Claim(claim_id="c3", answer_id="a1", text="Third.", answer_char_span=[15, 21])
        ]
        
        # All contradictory (will test reverse processing)
        decisions = [
            ClaimDecision(claim_id="c1", status="Contradictory", rationale="", primary_evidence="", signals_ref=[], confidence={}),
            ClaimDecision(claim_id="c2", status="Contradictory", rationale="", primary_evidence="", signals_ref=[], confidence={}),
            ClaimDecision(claim_id="c3", status="Contradictory", rationale="", primary_evidence="", signals_ref=[], confidence={})
        ]
        
        filtered_text, removed_count, _ = filter.filter_answer(answer, claims, decisions)
        
        # All should be replaced correctly
        placeholder = '[CLAIM REMOVED: Contradictory]'
        expected = f"{placeholder} {placeholder} {placeholder}"
        
        assert filtered_text == expected
        assert removed_count == 3


class TestInvalidSpanHandling:
    """Test handling of invalid character spans."""
    
    def test_filter_with_invalid_span_negative_start(self, sample_config):
        """Test filtering with negative start index (should skip)."""
        filter = ClaimFilter(sample_config)
        
        answer = "Valid text."
        claims = [
            Claim(
                claim_id="c1",
                answer_id="a1",
                text="Invalid",
                answer_char_span=[-5, 5]  # Negative start
            )
        ]
        
        decisions = [
            ClaimDecision(
                claim_id="c1",
                status="Contradictory",
                rationale="",
                primary_evidence="",
                signals_ref=[],
                confidence={}
            )
        ]
        
        # Should skip invalid span and log warning
        filtered_text, removed_count, _ = filter.filter_answer(answer, claims, decisions)
        
        assert filtered_text == answer  # Unchanged
        assert removed_count == 0
    
    def test_filter_with_invalid_span_exceeds_length(self, sample_config):
        """Test filtering with end index exceeding text length (should skip)."""
        filter = ClaimFilter(sample_config)
        
        answer = "Short text."
        claims = [
            Claim(
                claim_id="c1",
                answer_id="a1",
                text="Beyond",
                answer_char_span=[5, 100]  # End exceeds length
            )
        ]
        
        decisions = [
            ClaimDecision(
                claim_id="c1",
                status="Contradictory",
                rationale="",
                primary_evidence="",
                signals_ref=[],
                confidence={}
            )
        ]
        
        # Should skip invalid span
        filtered_text, removed_count, _ = filter.filter_answer(answer, claims, decisions)
        
        assert filtered_text == answer
        assert removed_count == 0
    
    def test_filter_with_invalid_span_start_after_end(self, sample_config):
        """Test filtering with start >= end (should skip)."""
        filter = ClaimFilter(sample_config)
        
        answer = "Valid text."
        claims = [
            Claim(
                claim_id="c1",
                answer_id="a1",
                text="Invalid",
                answer_char_span=[5, 3]  # start > end
            )
        ]
        
        decisions = [
            ClaimDecision(
                claim_id="c1",
                status="Contradictory",
                rationale="",
                primary_evidence="",
                signals_ref=[],
                confidence={}
            )
        ]
        
        # Should skip invalid span
        filtered_text, removed_count, _ = filter.filter_answer(answer, claims, decisions)
        
        assert filtered_text == answer
        assert removed_count == 0


class TestFilteringSummary:
    """Test filtering summary utility method."""
    
    def test_get_filtering_summary(self, sample_config, sample_claims, sample_decisions_mixed):
        """Test filtering summary statistics."""
        filter = ClaimFilter(sample_config)
        
        summary = filter.get_filtering_summary(sample_claims, sample_decisions_mixed)
        
        assert summary['total_claims'] == 3
        assert summary['supported'] == 2
        assert summary['contradictory'] == 1
        assert summary['low_confidence'] == 0
        assert summary['would_remove'] == 1
    
    def test_get_filtering_summary_all_contradictory(self, sample_config):
        """Test summary when all claims are contradictory."""
        filter = ClaimFilter(sample_config)
        
        claims = [
            Claim(claim_id=f"c{i}", answer_id="a1", text=f"Claim {i}", answer_char_span=[i*10, i*10+9])
            for i in range(5)
        ]
        
        decisions = [
            ClaimDecision(
                claim_id=f"c{i}",
                status="Contradictory",
                rationale="",
                primary_evidence="",
                signals_ref=[],
                confidence={}
            )
            for i in range(5)
        ]
        
        summary = filter.get_filtering_summary(claims, decisions)
        
        assert summary['total_claims'] == 5
        assert summary['contradictory'] == 5
        assert summary['would_remove'] == 5
    
    def test_get_filtering_summary_with_low_confidence(self, sample_config):
        """Test summary with low-confidence claims."""
        filter = ClaimFilter(sample_config)
        
        claims = [
            Claim(claim_id="c1", answer_id="a1", text="Claim 1", answer_char_span=[0, 7]),
            Claim(claim_id="c2", answer_id="a1", text="Claim 2", answer_char_span=[8, 15])
        ]
        
        decisions = [
            ClaimDecision(claim_id="c1", status="Supported", rationale="", primary_evidence="", signals_ref=[], confidence={}),
            ClaimDecision(claim_id="c2", status="Low Confidence", rationale="", primary_evidence="", signals_ref=[], confidence={})
        ]
        
        summary = filter.get_filtering_summary(claims, decisions)
        
        assert summary['supported'] == 1
        assert summary['low_confidence'] == 1
        assert summary['would_remove'] == 0  # Only contradictory are removed
    
    def test_get_filtering_summary_with_mismatched_lengths(self, sample_config, sample_claims):
        """Test that summary with mismatched lengths raises ValueError."""
        filter = ClaimFilter(sample_config)
        
        decisions = [
            ClaimDecision(claim_id="c1", status="Supported", rationale="", primary_evidence="", signals_ref=[], confidence={})
        ]
        
        with pytest.raises(ValueError, match="Claims and decisions must have same length"):
            filter.get_filtering_summary(sample_claims, decisions)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_filter_with_empty_answer_text(self, sample_config):
        """Test filtering with empty answer text."""
        filter = ClaimFilter(sample_config)
        
        claims = [
            Claim(claim_id="c1", answer_id="a1", text="Phantom", answer_char_span=[0, 7])
        ]
        
        decisions = [
            ClaimDecision(claim_id="c1", status="Contradictory", rationale="", primary_evidence="", signals_ref=[], confidence={})
        ]
        
        # Should handle gracefully (invalid span)
        filtered_text, removed_count, _ = filter.filter_answer("", claims, decisions)
        
        assert filtered_text == ""
        assert removed_count == 0
    
    def test_filter_with_single_character_claim(self, sample_config):
        """Test filtering single-character claim."""
        filter = ClaimFilter(sample_config)
        
        answer = "A B C"
        claims = [
            Claim(claim_id="c1", answer_id="a1", text="B", answer_char_span=[2, 3])
        ]
        
        decisions = [
            ClaimDecision(claim_id="c1", status="Contradictory", rationale="", primary_evidence="", signals_ref=[], confidence={})
        ]
        
        filtered_text, removed_count, _ = filter.filter_answer(answer, claims, decisions)
        
        assert "A" in filtered_text
        assert "C" in filtered_text
        assert "B" not in filtered_text
        assert removed_count == 1
    
    def test_filter_claim_at_text_boundaries(self, sample_config):
        """Test filtering claims at start and end of text."""
        filter = ClaimFilter(sample_config)
        
        answer = "Start. Middle. End."
        claims = [
            Claim(claim_id="c1", answer_id="a1", text="Start.", answer_char_span=[0, 6]),
            Claim(claim_id="c2", answer_id="a1", text="Middle.", answer_char_span=[7, 14]),
            Claim(claim_id="c3", answer_id="a1", text="End.", answer_char_span=[15, 19])
        ]
        
        decisions = [
            ClaimDecision(claim_id="c1", status="Contradictory", rationale="", primary_evidence="", signals_ref=[], confidence={}),
            ClaimDecision(claim_id="c2", status="Supported", rationale="", primary_evidence="", signals_ref=[], confidence={}),
            ClaimDecision(claim_id="c3", status="Contradictory", rationale="", primary_evidence="", signals_ref=[], confidence={})
        ]
        
        filtered_text, removed_count, _ = filter.filter_answer(answer, claims, decisions)
        
        # Start and End should be replaced
        placeholder = '[CLAIM REMOVED: Contradictory]'
        expected = f"{placeholder} Middle. {placeholder}"
        
        assert filtered_text == expected
        assert removed_count == 2


class TestPlaceholderDetection:
    """Test placeholder marker detection used by post-mitigation re-verification."""

    def test_is_placeholder_detects_contradictory_marker(self, sample_config):
        filter = ClaimFilter(sample_config)
        assert filter.is_placeholder("[CLAIM REMOVED: Contradictory]")

    def test_is_placeholder_detects_lc_marker(self, sample_config):
        filter = ClaimFilter(sample_config)
        assert filter.is_placeholder("Prefix [CLAIM UNCERTAIN: Low Confidence] suffix")


class TestPlaceholderSpanFiltering:
    """Regression tests for placeholder span-overlap filtering."""

    def test_filter_placeholder_claims_drops_fragmented_marker_claims(self, sample_config):
        filter = ClaimFilter(sample_config)
        filtered_text = (
            "[CLAIM REMOVED: Contradictory] It is Turkey's largest city. "
            "Ankara is Turkey's second-largest city."
        )

        claims = [
            Claim(
                claim_id="c1",
                answer_id="a1",
                text="[CLAIM REMOVED:",
                answer_char_span=[0, 15],
            ),
            Claim(
                claim_id="c2",
                answer_id="a1",
                text="Contradictory]",
                answer_char_span=[16, 30],
            ),
            Claim(
                claim_id="c3",
                answer_id="a1",
                text="It is Turkey's largest city.",
                answer_char_span=[31, 58],
            ),
        ]

        kept = filter.filter_placeholder_claims(claims, filtered_text)

        assert len(kept) == 1
        assert kept[0].claim_id == "c3"

    def test_filter_placeholder_claims_keeps_claims_when_no_marker(self, sample_config):
        filter = ClaimFilter(sample_config)
        text = "It is Turkey's largest city."
        claims = [
            Claim(
                claim_id="c1",
                answer_id="a1",
                text="It is Turkey's largest city.",
                answer_char_span=[0, len(text)],
            )
        ]

        kept = filter.filter_placeholder_claims(claims, text)
        assert len(kept) == 1
        assert kept[0].claim_id == "c1"


class TestPronounSubstitution:
    """Regression tests for pronoun substitution after removal placeholders."""

    def test_pronoun_substituted_after_contradictory_removal(self, sample_config):
        filter = ClaimFilter(sample_config)
        answer = "Istanbul is Turkey's largest city. It is also the country's cultural center."
        claims = [
            Claim(
                claim_id="c1",
                answer_id="a1",
                text="Istanbul is Turkey's largest city.",
                answer_char_span=[0, 33],
            ),
            Claim(
                claim_id="c2",
                answer_id="a1",
                text="It is also the country's cultural center.",
                answer_char_span=[34, len(answer)],
            ),
        ]
        decisions = [
            ClaimDecision(
                claim_id="c1",
                status="Contradictory",
                rationale="",
                primary_evidence="",
                signals_ref=[],
                confidence={},
            ),
            ClaimDecision(
                claim_id="c2",
                status="Supported",
                rationale="",
                primary_evidence="",
                signals_ref=[],
                confidence={},
            ),
        ]

        filtered_text, removed_count, _ = filter.filter_answer(answer, claims, decisions)

        assert removed_count == 1
        assert filtered_text.startswith("[CLAIM REMOVED: Contradictory]. Istanbul is also")

    def test_no_substitution_when_removed_subject_is_pronoun(self, sample_config):
        filter = ClaimFilter(sample_config)
        answer = "It is unknown. It remains uncertain."
        claims = [
            Claim(
                claim_id="c1",
                answer_id="a1",
                text="It is unknown.",
                answer_char_span=[0, 14],
            ),
            Claim(
                claim_id="c2",
                answer_id="a1",
                text="It remains uncertain.",
                answer_char_span=[14, len(answer)],
            ),
        ]
        decisions = [
            ClaimDecision(
                claim_id="c1",
                status="Contradictory",
                rationale="",
                primary_evidence="",
                signals_ref=[],
                confidence={},
            ),
            ClaimDecision(
                claim_id="c2",
                status="Supported",
                rationale="",
                primary_evidence="",
                signals_ref=[],
                confidence={},
            ),
        ]

        filtered_text, _, _ = filter.filter_answer(answer, claims, decisions)
        assert filtered_text.startswith("[CLAIM REMOVED: Contradictory] It remains uncertain.")

    def test_no_substitution_when_both_claims_removed(self, sample_config):
        filter = ClaimFilter(sample_config)
        answer = "Istanbul is Turkey's largest city. It is also a major hub."
        claims = [
            Claim(
                claim_id="c1",
                answer_id="a1",
                text="Istanbul is Turkey's largest city.",
                answer_char_span=[0, 33],
            ),
            Claim(
                claim_id="c2",
                answer_id="a1",
                text="It is also a major hub.",
                answer_char_span=[34, len(answer)],
            ),
        ]
        decisions = [
            ClaimDecision(
                claim_id="c1",
                status="Contradictory",
                rationale="",
                primary_evidence="",
                signals_ref=[],
                confidence={},
            ),
            ClaimDecision(
                claim_id="c2",
                status="Contradictory",
                rationale="",
                primary_evidence="",
                signals_ref=[],
                confidence={},
            ),
        ]

        filtered_text, removed_count, _ = filter.filter_answer(answer, claims, decisions)
        assert removed_count == 2
        assert "CLAIM REMOVED is" not in filtered_text

    def test_pronoun_substitution_disabled(self):
        config = MagicMock(spec=Config)
        config.get.return_value = {
            'filter': {
                'enabled': True,
                'placeholder': '[CLAIM REMOVED: Contradictory]',
                'pronoun_substitution_enabled': False,
            }
        }

        filter = ClaimFilter(config)
        answer = "Istanbul is Turkey's largest city. It is also a major hub."
        claims = [
            Claim(
                claim_id="c1",
                answer_id="a1",
                text="Istanbul is Turkey's largest city.",
                answer_char_span=[0, 33],
            ),
            Claim(
                claim_id="c2",
                answer_id="a1",
                text="It is also a major hub.",
                answer_char_span=[34, len(answer)],
            ),
        ]
        decisions = [
            ClaimDecision(
                claim_id="c1",
                status="Contradictory",
                rationale="",
                primary_evidence="",
                signals_ref=[],
                confidence={},
            ),
            ClaimDecision(
                claim_id="c2",
                status="Supported",
                rationale="",
                primary_evidence="",
                signals_ref=[],
                confidence={},
            ),
        ]

        filtered_text, _, _ = filter.filter_answer(answer, claims, decisions)
        assert filtered_text.startswith("[CLAIM REMOVED: Contradictory]. It is also")
