"""
Unit tests for RePrompter module.

Tests cover:
- Threshold logic for triggering re-prompting
- Prompt construction strategies (full and claim-specific)
- Integration with GeneratorWrapper
- Re-verification flow
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import List, Dict, Any

from src.mitigation.reprompt import RePrompter
from src.utils.data_structures import ClaimDecision, EvidenceChunk, Claim
from src.utils.config import Config


class TestRePrompterInitialization:
    """Test RePrompter initialization and configuration."""
    
    def test_init_with_defaults(self):
        """Test initialization with default configuration."""
        config = Config()
        config_dict = {
            'mitigation': {
                'reprompt': {
                    'enabled': False,
                    'threshold': 0.5,
                    'max_iterations': 2,
                    'strategy': 'full'
                }
            }
        }
        config.get = lambda *args, **kwargs: config_dict.get(args[0], {})
        
        generator = Mock()
        repromptr = RePrompter(config, generator)
        
        assert repromptr.threshold == 0.5
        assert repromptr.max_iterations == 2
        assert repromptr.strategy == 'full'
        assert repromptr.enabled == False
    
    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        config = Config()
        config_dict = {
            'mitigation': {
                'reprompt': {
                    'enabled': True,
                    'threshold': 0.7,
                    'max_iterations': 3,
                    'strategy': 'claim-specific'
                }
            }
        }
        config.get = lambda *args, **kwargs: config_dict.get(args[0], {})
        
        generator = Mock()
        repromptr = RePrompter(config, generator)
        
        assert repromptr.threshold == 0.7
        assert repromptr.max_iterations == 3
        assert repromptr.strategy == 'claim-specific'
        assert repromptr.enabled == True
    
    def test_invalid_threshold_raises_error(self):
        """Test that invalid threshold raises ValueError."""
        config = Config()
        config_dict = {
            'mitigation': {
                'reprompt': {
                    'threshold': 1.5  # Invalid: > 1.0
                }
            }
        }
        config.get = lambda *args, **kwargs: config_dict.get(args[0], {})
        
        generator = Mock()
        
        with pytest.raises(ValueError, match="Threshold must be in"):
            RePrompter(config, generator)
    
    def test_invalid_max_iterations_raises_error(self):
        """Test that invalid max_iterations raises ValueError."""
        config = Config()
        config_dict = {
            'mitigation': {
                'reprompt': {
                    'threshold': 0.5,
                    'max_iterations': 0  # Invalid: < 1
                }
            }
        }
        config.get = lambda *args, **kwargs: config_dict.get(args[0], {})
        
        generator = Mock()
        
        with pytest.raises(ValueError, match="max_iterations must be"):
            RePrompter(config, generator)


class TestShouldReprompt:
    """Test hallucination rate analysis and threshold logic."""
    
    @pytest.fixture
    def repromptr(self):
        """Create a RePrompter instance for testing."""
        config = Config()
        config_dict = {
            'mitigation': {
                'reprompt': {
                    'enabled': True,
                    'threshold': 0.5,
                    'max_iterations': 2,
                    'strategy': 'full'
                }
            }
        }
        config.get = lambda *args, **kwargs: config_dict.get(args[0], {})
        
        generator = Mock()
        return RePrompter(config, generator)
    
    def test_no_contradictions_no_reprompt(self, repromptr):
        """Test that no contradictions means no re-prompting."""
        decisions = [
            ClaimDecision(
                claim_id="claim_1",
                status="Supported",
                rationale="Well grounded",
                primary_evidence="doc_1#sent_1",
                signals_ref=["signal_1"],
                confidence={"overall_confidence": 0.9}
            ),
            ClaimDecision(
                claim_id="claim_2",
                status="Supported",
                rationale="Well grounded",
                primary_evidence="doc_1#sent_2",
                signals_ref=["signal_2"],
                confidence={"overall_confidence": 0.85}
            )
        ]
        
        should_retry, rate = repromptr.should_reprompt(decisions)
        
        assert should_retry == False
        assert rate == 0.0
    
    def test_high_contradictions_triggers_reprompt(self, repromptr):
        """Test that high contradiction rate triggers re-prompting."""
        decisions = [
            ClaimDecision(
                claim_id="claim_1",
                status="Contradictory",
                rationale="NLI contradiction",
                primary_evidence="doc_1#sent_1",
                signals_ref=["signal_1"],
                confidence={"overall_confidence": 0.2}
            ),
            ClaimDecision(
                claim_id="claim_2",
                status="Contradictory",
                rationale="Coverage mismatch",
                primary_evidence="doc_1#sent_2",
                signals_ref=["signal_2"],
                confidence={"overall_confidence": 0.1}
            ),
            ClaimDecision(
                claim_id="claim_3",
                status="Supported",
                rationale="Well grounded",
                primary_evidence="doc_1#sent_3",
                signals_ref=["signal_3"],
                confidence={"overall_confidence": 0.9}
            )
        ]
        
        should_retry, rate = repromptr.should_reprompt(decisions)
        
        assert should_retry == True  # 2/3 = 0.67 > 0.5
        assert rate == pytest.approx(0.667, abs=0.01)
    
    def test_exactly_at_threshold_no_reprompt(self, repromptr):
        """Test that exactly at threshold does not trigger re-prompting."""
        decisions = [
            ClaimDecision(
                claim_id="claim_1",
                status="Contradictory",
                rationale="NLI contradiction",
                primary_evidence="doc_1#sent_1",
                signals_ref=["signal_1"],
                confidence={"overall_confidence": 0.2}
            ),
            ClaimDecision(
                claim_id="claim_2",
                status="Supported",
                rationale="Well grounded",
                primary_evidence="doc_1#sent_2",
                signals_ref=["signal_2"],
                confidence={"overall_confidence": 0.9}
            )
        ]
        
        should_retry, rate = repromptr.should_reprompt(decisions)
        
        assert should_retry == False  # 1/2 = 0.5 == threshold (not >)
        assert rate == 0.5
    
    def test_empty_decisions_no_reprompt(self, repromptr):
        """Test that empty decisions list does not trigger re-prompting."""
        decisions = []
        
        should_retry, rate = repromptr.should_reprompt(decisions)
        
        assert should_retry == False
        assert rate == 0.0


class TestPromptConstruction:
    """Test feedback prompt construction for different strategies."""
    
    @pytest.fixture
    def repromptr_full(self):
        """Create a RePrompter with 'full' strategy."""
        config = Config()
        config_dict = {
            'mitigation': {
                'reprompt': {
                    'enabled': True,
                    'threshold': 0.5,
                    'max_iterations': 2,
                    'strategy': 'full'
                }
            }
        }
        config.get = lambda *args, **kwargs: config_dict.get(args[0], {})
        
        generator = Mock()
        return RePrompter(config, generator)
    
    @pytest.fixture
    def repromptr_claim_specific(self):
        """Create a RePrompter with 'claim-specific' strategy."""
        config = Config()
        config_dict = {
            'mitigation': {
                'reprompt': {
                    'enabled': True,
                    'threshold': 0.5,
                    'max_iterations': 2,
                    'strategy': 'claim-specific'
                }
            }
        }
        config.get = lambda *args, **kwargs: config_dict.get(args[0], {})
        
        generator = Mock()
        return RePrompter(config, generator)
    
    def test_full_strategy_prompt_format(self, repromptr_full):
        """Test that 'full' strategy includes verification feedback."""
        query = "What is the capital of France?"
        answer = "The capital of France is Berlin."
        
        decisions = [
            ClaimDecision(
                claim_id="claim_1",
                status="Contradictory",
                rationale="NLI contradiction with evidence",
                primary_evidence="doc_1#sent_1",
                signals_ref=["signal_1"],
                confidence={"overall_confidence": 0.2}
            )
        ]
        
        claims = [
            Claim(
                claim_id="claim_1",
                claim_text="The capital of France is Berlin.",
                answer_char_span=[0, 35]
            )
        ]
        
        evidence = [
            EvidenceChunk(
                chunk_id="chunk_1",
                doc_id="doc_1",
                sent_id="sent_1",
                text="Paris is the capital of France.",
                score_dense=0.95,
                score_bm25=None,
                score_hybrid=None
            )
        ]
        
        prompt = repromptr_full.construct_feedback_prompt(
            original_query=query,
            original_answer=answer,
            decisions=decisions,
            evidence=evidence,
            claims=claims
        )
        
        # Verify key components are present
        assert "Context:" in prompt
        assert "Paris is the capital of France" in prompt
        assert "Question:" in prompt
        assert "What is the capital of France?" in prompt
        assert "Previous Answer:" in prompt
        assert "The capital of France is Berlin" in prompt
        assert "Verification Feedback:" in prompt
        assert "Contradictory" in prompt
        assert "Revised Answer:" in prompt
    
    def test_claim_specific_strategy_prompt_format(self, repromptr_claim_specific):
        """Test that 'claim-specific' strategy includes verification questions."""
        query = "What is the capital of France?"
        answer = "The capital of France is Berlin."
        
        decisions = [
            ClaimDecision(
                claim_id="claim_1",
                status="Contradictory",
                rationale="NLI contradiction with evidence",
                primary_evidence="doc_1#sent_1",
                signals_ref=["signal_1"],
                confidence={"overall_confidence": 0.2}
            )
        ]
        
        claims = [
            Claim(
                claim_id="claim_1",
                claim_text="The capital of France is Berlin.",
                answer_char_span=[0, 35]
            )
        ]
        
        evidence = [
            EvidenceChunk(
                chunk_id="chunk_1",
                doc_id="doc_1",
                sent_id="sent_1",
                text="Paris is the capital of France.",
                score_dense=0.95,
                score_bm25=None,
                score_hybrid=None
            )
        ]
        
        prompt = repromptr_claim_specific.construct_feedback_prompt(
            original_query=query,
            original_answer=answer,
            decisions=decisions,
            evidence=evidence,
            claims=claims
        )
        
        # Verify key components are present
        assert "Context:" in prompt
        assert "Question:" in prompt
        assert "Verification Questions:" in prompt
        assert "Is the following statement supported by the context" in prompt
        assert "Corrected Answer:" in prompt


class TestRepromptExecution:
    """Test the complete re-prompting execution flow."""
    
    @pytest.fixture
    def repromptr_with_mock_generator(self):
        """Create a RePrompter with a mocked generator."""
        config = Config()
        config_dict = {
            'mitigation': {
                'reprompt': {
                    'enabled': True,
                    'threshold': 0.5,
                    'max_iterations': 2,
                    'strategy': 'full'
                }
            }
        }
        config.get = lambda *args, **kwargs: config_dict.get(args[0], {})
        
        # Mock generator
        generator = Mock()
        generator.generate_with_metadata = Mock(return_value={
            'text': 'The capital of France is Paris.',
            'tokens': ['The', 'capital', 'of', 'France', 'is', 'Paris', '.'],
            'scores': [0.9] * 7
        })
        
        return RePrompter(config, generator)
    
    def test_reprompt_not_triggered_below_threshold(self, repromptr_with_mock_generator):
        """Test that re-prompting is not triggered when below threshold."""
        query = "What is the capital of France?"
        answer = "The capital of France is Paris."
        
        decisions = [
            ClaimDecision(
                claim_id="claim_1",
                status="Supported",
                rationale="Well grounded",
                primary_evidence="doc_1#sent_1",
                signals_ref=["signal_1"],
                confidence={"overall_confidence": 0.9}
            )
        ]
        
        claims = [
            Claim(
                claim_id="claim_1",
                claim_text="The capital of France is Paris.",
                answer_char_span=[0, 33]
            )
        ]
        
        evidence = [
            EvidenceChunk(
                chunk_id="chunk_1",
                doc_id="doc_1",
                sent_id="sent_1",
                text="Paris is the capital of France.",
                score_dense=0.95,
                score_bm25=None,
                score_hybrid=None
            )
        ]
        
        result = repromptr_with_mock_generator.reprompt(
            query=query,
            answer=answer,
            decisions=decisions,
            evidence=evidence,
            claims=claims
        )
        
        assert result['improved'] == False
        assert result['iterations'] == 0
        assert result['final_answer'] == answer
        assert result['hallucination_rate_before'] == 0.0
    
    def test_reprompt_triggered_above_threshold(self, repromptr_with_mock_generator):
        """Test that re-prompting is triggered when above threshold."""
        query = "What is the capital of France?"
        answer = "The capital of France is Berlin."
        
        decisions = [
            ClaimDecision(
                claim_id="claim_1",
                status="Contradictory",
                rationale="NLI contradiction",
                primary_evidence="doc_1#sent_1",
                signals_ref=["signal_1"],
                confidence={"overall_confidence": 0.2}
            ),
            ClaimDecision(
                claim_id="claim_2",
                status="Contradictory",
                rationale="Coverage mismatch",
                primary_evidence="doc_1#sent_2",
                signals_ref=["signal_2"],
                confidence={"overall_confidence": 0.1}
            )
        ]
        
        claims = [
            Claim(
                claim_id="claim_1",
                claim_text="The capital of France is Berlin.",
                answer_char_span=[0, 33]
            ),
            Claim(
                claim_id="claim_2",
                claim_text="Berlin is very large.",
                answer_char_span=[34, 55]
            )
        ]
        
        evidence = [
            EvidenceChunk(
                chunk_id="chunk_1",
                doc_id="doc_1",
                sent_id="sent_1",
                text="Paris is the capital of France.",
                score_dense=0.95,
                score_bm25=None,
                score_hybrid=None
            )
        ]
        
        result = repromptr_with_mock_generator.reprompt(
            query=query,
            answer=answer,
            decisions=decisions,
            evidence=evidence,
            claims=claims
        )
        
        assert result['improved'] == True
        assert result['iterations'] == 1
        assert result['final_answer'] == 'The capital of France is Paris.'
        assert result['hallucination_rate_before'] == 1.0  # 2/2
        assert result['feedback_prompt'] is not None
        
        # Verify generator was called with lower temperature
        repromptr_with_mock_generator.generator.generate_with_metadata.assert_called_once()
        call_kwargs = repromptr_with_mock_generator.generator.generate_with_metadata.call_args[1]
        assert call_kwargs['temperature'] == 0.3
        assert call_kwargs['do_sample'] == True


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_evidence_list(self):
        """Test handling of empty evidence list."""
        config = Config()
        config_dict = {
            'mitigation': {
                'reprompt': {
                    'enabled': True,
                    'threshold': 0.5,
                    'max_iterations': 2,
                    'strategy': 'full'
                }
            }
        }
        config.get = lambda *args, **kwargs: config_dict.get(args[0], {})
        
        generator = Mock()
        repromptr = RePrompter(config, generator)
        
        # Format evidence should handle empty list gracefully
        result = repromptr._format_evidence([])
        
        assert result == "No evidence provided."
    
    def test_unknown_strategy_defaults_to_full(self):
        """Test that unknown strategy defaults to 'full' with warning."""
        config = Config()
        config_dict = {
            'mitigation': {
                'reprompt': {
                    'enabled': True,
                    'threshold': 0.5,
                    'max_iterations': 2,
                    'strategy': 'unknown_strategy'
                }
            }
        }
        config.get = lambda *args, **kwargs: config_dict.get(args[0], {})
        
        generator = Mock()
        repromptr = RePrompter(config, generator)
        
        # Should default to 'full'
        assert repromptr.strategy == 'full'
