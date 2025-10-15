"""
Unit tests for the Generation module.

Tests the GeneratorWrapper and claim extraction functionality.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation import (
    GeneratorWrapper,
    extract_claims,
    extract_claims_spacy,
    extract_claims_regex,
    validate_claim_spans
)
from src.utils.data_structures import EvidenceChunk, Claim


class TestClaimExtractor:
    """Test suite for claim extraction functions."""
    
    def test_extract_claims_spacy_basic(self):
        """Test basic spaCy claim extraction."""
        text = "Machine learning is a subset of AI. It uses data to learn patterns."
        claims = extract_claims_spacy(text)
        
        assert len(claims) == 2
        assert all(isinstance(c, Claim) for c in claims)
        assert claims[0].text == "Machine learning is a subset of AI."
        assert claims[1].text == "It uses data to learn patterns."
    
    def test_extract_claims_regex_basic(self):
        """Test basic regex claim extraction."""
        text = "AI is growing fast. It has many applications!"
        claims = extract_claims_regex(text)
        
        assert len(claims) >= 2
        assert all(isinstance(c, Claim) for c in claims)
    
    def test_extract_claims_auto_method(self):
        """Test auto method selection."""
        text = "This is sentence one. This is sentence two."
        claims = extract_claims(text, method='auto')
        
        assert len(claims) == 2
        assert all(c.extraction_method in ['spacy_sent_v1', 'rule_sentence_split_v1'] for c in claims)
    
    def test_extract_claims_empty_text(self):
        """Test extraction with empty text."""
        claims = extract_claims("")
        assert len(claims) == 0
        
        claims = extract_claims("   ")
        assert len(claims) == 0
    
    def test_extract_claims_single_sentence(self):
        """Test extraction with single sentence."""
        text = "This is a single sentence."
        claims = extract_claims(text)
        
        assert len(claims) == 1
        assert claims[0].text == text
    
    def test_claim_fields(self):
        """Test that claims have all required fields."""
        text = "Deep learning uses neural networks."
        claims = extract_claims(text)
        
        for claim in claims:
            assert hasattr(claim, 'claim_id')
            assert hasattr(claim, 'answer_id')
            assert hasattr(claim, 'text')
            assert hasattr(claim, 'answer_char_span')
            assert hasattr(claim, 'extraction_method')
            
            # Check types
            assert isinstance(claim.claim_id, str)
            assert isinstance(claim.answer_id, str)
            assert isinstance(claim.text, str)
            assert isinstance(claim.answer_char_span, list)
            assert len(claim.answer_char_span) == 2
            assert isinstance(claim.extraction_method, str)
    
    def test_claim_char_spans(self):
        """Test that char spans are correct."""
        text = "First sentence. Second sentence."
        claims = extract_claims_spacy(text)
        
        for claim in claims:
            start, end = claim.answer_char_span
            extracted_text = text[start:end].strip()
            assert extracted_text == claim.text.strip()
    
    def test_validate_claim_spans(self):
        """Test claim span validation."""
        text = "This is a test. It works well."
        claims = extract_claims(text)
        
        # Should validate successfully
        assert validate_claim_spans(claims, text)
    
    def test_extract_claims_with_answer_id(self):
        """Test extraction with provided answer_id."""
        text = "Test sentence."
        answer_id = "test_answer_123"
        claims = extract_claims(text, answer_id=answer_id)
        
        assert all(c.answer_id == answer_id for c in claims)
    
    def test_unique_claim_ids(self):
        """Test that each claim gets a unique ID."""
        text = "Sentence one. Sentence two. Sentence three."
        claims = extract_claims(text)
        
        claim_ids = [c.claim_id for c in claims]
        assert len(claim_ids) == len(set(claim_ids))  # All unique


class TestGeneratorWrapper:
    """Test suite for GeneratorWrapper class."""
    
    @pytest.fixture(scope="class")
    def generator(self):
        """Create a GeneratorWrapper instance (loaded once for all tests)."""
        # Use FLAN-T5-base for testing
        generator = GeneratorWrapper(
            model_name='google/flan-t5-base',
            device='cuda'
        )
        return generator
    
    def test_initialization(self, generator):
        """Test generator initialization."""
        assert generator.model is not None
        assert generator.tokenizer is not None
        assert generator.model_name == 'google/flan-t5-base'
    
    def test_format_prompt_no_evidence(self, generator):
        """Test prompt formatting without evidence."""
        prompt = "What is AI?"
        formatted = generator._format_prompt(prompt, [])
        
        assert "Question: What is AI?" in formatted
        assert "Answer:" in formatted
        assert "Context:" not in formatted
    
    def test_format_prompt_with_evidence(self, generator):
        """Test prompt formatting with evidence."""
        prompt = "What is machine learning?"
        evidence = [
            EvidenceChunk(
                doc_id='doc1',
                sent_id=0,
                text='ML is a subset of AI.',
                char_start=0,
                char_end=23,
                score_dense=0.95,
                rank=0,
                source='wikipedia',
                version='v1'
            ),
            EvidenceChunk(
                doc_id='doc1',
                sent_id=1,
                text='It uses data to learn.',
                char_start=24,
                char_end=46,
                score_dense=0.90,
                rank=1,
                source='wikipedia',
                version='v1'
            )
        ]
        
        formatted = generator._format_prompt(prompt, evidence)
        
        assert "Context:" in formatted
        assert "ML is a subset of AI" in formatted
        assert "It uses data to learn" in formatted
        assert "Question: What is machine learning?" in formatted
        assert "Answer:" in formatted
    
    def test_generate_with_metadata_basic(self, generator):
        """Test basic generation with metadata capture."""
        prompt = "What is 2 + 2?"
        result = generator.generate_with_metadata(
            prompt=prompt,
            max_new_tokens=50
        )
        
        # Check all required fields
        assert 'text' in result
        assert 'prompt_text' in result
        assert 'tokens' in result
        assert 'token_ids' in result
        assert 'logits' in result
        assert 'scores' in result
        assert 'evidence_used' in result
        assert 'generation_config' in result
        
        # Check types
        assert isinstance(result['text'], str)
        assert isinstance(result['tokens'], list)
        assert isinstance(result['token_ids'], list)
        assert isinstance(result['logits'], list)
        assert isinstance(result['scores'], list)
        assert isinstance(result['evidence_used'], list)
        assert isinstance(result['generation_config'], dict)
    
    def test_generate_text_nonempty(self, generator):
        """Test that generated text is non-empty."""
        prompt = "What is the capital of France?"
        result = generator.generate_with_metadata(prompt=prompt, max_new_tokens=20)
        
        assert len(result['text']) > 0
        assert result['text'].strip() != ""
    
    def test_generate_with_evidence(self, generator):
        """Test generation with evidence chunks."""
        prompt = "What is the purpose of photosynthesis?"
        evidence = [
            EvidenceChunk(
                doc_id='bio_1',
                sent_id=0,
                text='Photosynthesis converts light energy into chemical energy.',
                char_start=0,
                char_end=58,
                score_dense=0.98,
                rank=0,
                source='textbook',
                version='v1'
            )
        ]
        
        result = generator.generate_with_metadata(
            prompt=prompt,
            evidence_chunks=evidence,
            max_new_tokens=50
        )
        
        # Check evidence usage
        assert 'bio_1' in result['evidence_used']
        assert 'Context:' in result['prompt_text']
        assert 'Photosynthesis' in result['prompt_text']
    
    def test_generate_tokens_match_text(self, generator):
        """Test that tokens can be joined to form text."""
        prompt = "Say hello"
        result = generator.generate_with_metadata(prompt=prompt, max_new_tokens=10)
        
        # Should have tokens
        assert len(result['tokens']) > 0
        assert len(result['token_ids']) == len(result['tokens'])
    
    def test_generate_logits_captured(self, generator):
        """Test that logits are captured correctly."""
        prompt = "What is 1 + 1?"
        result = generator.generate_with_metadata(prompt=prompt, max_new_tokens=10)
        
        # Should have logits for each generated token
        assert len(result['logits']) > 0
        # Each logit should be a numpy array
        for logit in result['logits']:
            assert hasattr(logit, 'shape')  # numpy array
    
    def test_generate_scores_captured(self, generator):
        """Test that scores are captured correctly."""
        prompt = "Count to three"
        result = generator.generate_with_metadata(prompt=prompt, max_new_tokens=20)
        
        # Should have scores
        assert len(result['scores']) > 0
        # Scores should be probabilities (0-1)
        for score in result['scores']:
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
    
    def test_generation_config_captured(self, generator):
        """Test that generation config is captured."""
        prompt = "Test prompt"
        result = generator.generate_with_metadata(
            prompt=prompt,
            max_new_tokens=30,
            temperature=0.8,
            top_p=0.9
        )
        
        config = result['generation_config']
        assert config['max_new_tokens'] == 30
        assert config['temperature'] == 0.8
        assert config['top_p'] == 0.9
        assert config['model_name'] == 'google/flan-t5-base'
    
    def test_generate_different_temperatures(self, generator):
        """Test generation with different temperature settings."""
        prompt = "Explain AI"
        
        # Low temperature (more deterministic)
        result1 = generator.generate_with_metadata(
            prompt=prompt,
            max_new_tokens=30,
            temperature=0.1
        )
        
        # Higher temperature (more random)
        result2 = generator.generate_with_metadata(
            prompt=prompt,
            max_new_tokens=30,
            temperature=1.5
        )
        
        # Both should generate text
        assert len(result1['text']) > 0
        assert len(result2['text']) > 0
    
    def test_generate_batch(self, generator):
        """Test batch generation."""
        prompts = ["What is AI?", "What is ML?"]
        evidence_list = [[], []]
        
        results = generator.generate_batch(
            prompts=prompts,
            evidence_chunks_list=evidence_list,
            max_new_tokens=30
        )
        
        assert len(results) == 2
        for result in results:
            assert 'text' in result
            assert len(result['text']) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
