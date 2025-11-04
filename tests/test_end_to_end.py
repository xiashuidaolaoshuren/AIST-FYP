"""
End-to-end integration tests for Month 2 Baseline RAG System.

These tests validate the complete pipeline and ensure the output format
matches the expected interface for Month 3 verifier integration.
"""

import pytest
import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_structures import EvidenceChunk, Claim, ClaimEvidencePair
from src.generation import GeneratorWrapper, extract_claims
from src.utils.config import Config


@pytest.mark.integration
class TestEndToEndPipeline:
    """
    Integration tests for the complete RAG pipeline.
    
    These tests validate that all components work together correctly
    and that the output format matches Month 3 verifier expectations.
    """
    
    def test_claim_extraction_integration(self):
        """Test that claim extraction works with generated text."""
        # Generate some sample text
        sample_text = "Machine learning is a subset of AI. It uses data to learn patterns."
        
        # Extract claims
        claims = extract_claims(sample_text, method='auto')
        
        # Verify claims were extracted
        assert len(claims) > 0
        
        # Verify claim structure
        for claim in claims:
            assert isinstance(claim, Claim)
            assert hasattr(claim, 'claim_id')
            assert hasattr(claim, 'answer_id')
            assert hasattr(claim, 'text')
            assert hasattr(claim, 'answer_char_span')
            assert hasattr(claim, 'extraction_method')
            
            # Verify char span is valid
            assert len(claim.answer_char_span) == 2
            start, end = claim.answer_char_span
            assert 0 <= start < end <= len(sample_text)
    
    def test_claim_evidence_pair_structure(self):
        """Test that ClaimEvidencePair matches Month 3 interface."""
        # Create sample evidence
        evidence_chunks = [
            EvidenceChunk(
                doc_id="enwiki_1",
                sent_id=0,
                text="Machine learning is a field of AI.",
                char_start=0,
                char_end=35,
                score_dense=0.95,
                rank=0,
                source="wikipedia",
                version="wiki_sent_v1"
            ),
            EvidenceChunk(
                doc_id="enwiki_1",
                sent_id=1,
                text="It uses data to improve performance.",
                char_start=36,
                char_end=72,
                score_dense=0.87,
                rank=1,
                source="wikipedia",
                version="wiki_sent_v1"
            )
        ]
        
        # Create sample claims
        sample_text = "Machine learning is a subset of AI."
        claims = extract_claims(sample_text)
        
        # Create claim-evidence pairs
        pairs = []
        for claim in claims:
            pair = ClaimEvidencePair(
                claim_id=claim.claim_id,
                evidence_candidates=[f"{e.doc_id}#{e.sent_id}" for e in evidence_chunks],
                top_evidence=f"{evidence_chunks[0].doc_id}#{evidence_chunks[0].sent_id}",
                evidence_spans=[e.to_dict() for e in evidence_chunks]
            )
            pairs.append(pair)
        
        # Verify pair structure
        assert len(pairs) > 0
        
        for pair in pairs:
            # Verify all required fields exist
            pair_dict = pair.to_dict()
            assert 'claim_id' in pair_dict
            assert 'evidence_candidates' in pair_dict
            assert 'top_evidence' in pair_dict
            assert 'evidence_spans' in pair_dict
            
            # Verify evidence format
            assert len(pair_dict['evidence_candidates']) > 0
            assert '#' in pair_dict['top_evidence']  # Format: "doc_id#sent_id"
            
            # Verify evidence spans contain EvidenceChunk data
            assert len(pair_dict['evidence_spans']) > 0
            for span in pair_dict['evidence_spans']:
                assert 'doc_id' in span
                assert 'sent_id' in span
                assert 'text' in span
                assert 'score_dense' in span
                assert 'rank' in span
    
    def test_generator_metadata_structure(self):
        """Test that generator metadata contains all required fields for Month 3."""
        # Initialize generator (using CPU for testing)
        generator = GeneratorWrapper(
            model_name='google/flan-t5-base',
            device='cpu'
        )
        
        # Generate with metadata
        prompt = "What is 2 + 2?"
        result = generator.generate_with_metadata(
            prompt=prompt,
            evidence_chunks=[],
            max_new_tokens=20
        )
        
        # Verify all required metadata fields exist
        required_fields = [
            'text',
            'prompt_text',
            'tokens',
            'token_ids',
            'logits',
            'scores',
            'evidence_used',
            'generation_config'
        ]
        
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"
        
        # Verify logits structure (needed for Month 3 entropy calculation)
        assert isinstance(result['logits'], list)
        assert len(result['logits']) > 0
        
        # Each logit should be a numpy array
        for logit in result['logits']:
            assert hasattr(logit, 'shape')  # numpy array
            assert len(logit.shape) == 1  # 1D array (vocabulary size)
        
        # Verify scores structure (needed for Month 3 confidence)
        assert isinstance(result['scores'], list)
        assert len(result['scores']) > 0
        
        # Each score should be a probability [0, 1]
        for score in result['scores']:
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
        
        # Verify tokens match token_ids
        assert len(result['tokens']) == len(result['token_ids'])
    
    def test_pipeline_output_json_serializable(self):
        """Test that pipeline output can be serialized to JSON (except logits)."""
        # Create sample pipeline output
        output = {
            'query': 'What is machine learning?',
            'draft_response': 'Machine learning is a field of AI.',
            'claim_evidence_pairs': [
                {
                    'claim_id': 'c_001',
                    'evidence_candidates': ['enwiki_1#0', 'enwiki_1#1'],
                    'top_evidence': 'enwiki_1#0',
                    'evidence_spans': [
                        {
                            'doc_id': 'enwiki_1',
                            'sent_id': 0,
                            'text': 'ML is AI.',
                            'char_start': 0,
                            'char_end': 9,
                            'score_dense': 0.95,
                            'rank': 0,
                            'source': 'wikipedia',
                            'version': 'wiki_sent_v1',
                            'score_bm25': None,
                            'score_hybrid': None
                        }
                    ]
                }
            ],
            'retrieval_metadata': {
                'top_k': 3,
                'num_retrieved': 2,
                'top_score': 0.95,
                'evidence_doc_ids': ['enwiki_1']
            }
        }
        
        # Should serialize without error (without logits)
        json_str = json.dumps(output)
        assert len(json_str) > 0
        
        # Should deserialize correctly
        loaded = json.loads(json_str)
        assert loaded['query'] == output['query']
        assert len(loaded['claim_evidence_pairs']) == 1
    
    def test_month3_interface_compatibility(self):
        """
        Test that Month 2 output is compatible with Month 3 verifier input.
        
        Month 3 verifier expects:
        - List of ClaimEvidencePair objects
        - Each pair has: claim_id, evidence_candidates, top_evidence, evidence_spans
        - Evidence spans contain all EvidenceChunk fields
        - Generator metadata with logits and scores for entropy calculation
        """
        # Simulate Month 2 output
        evidence = EvidenceChunk(
            doc_id="enwiki_test",
            sent_id=0,
            text="Test evidence text.",
            char_start=0,
            char_end=19,
            score_dense=0.9,
            rank=0
        )
        
        claim = Claim(
            claim_id="c_test",
            answer_id="ans_test",
            text="Test claim.",
            answer_char_span=[0, 11]
        )
        
        pair = ClaimEvidencePair(
            claim_id=claim.claim_id,
            evidence_candidates=[f"{evidence.doc_id}#{evidence.sent_id}"],
            top_evidence=f"{evidence.doc_id}#{evidence.sent_id}",
            evidence_spans=[evidence.to_dict()]
        )
        
        # Verify Month 3 can access all required fields
        pair_dict = pair.to_dict()
        
        # Month 3 verifier needs these fields
        assert pair_dict['claim_id'] == claim.claim_id
        assert len(pair_dict['evidence_candidates']) > 0
        assert pair_dict['top_evidence'] in pair_dict['evidence_candidates']
        assert len(pair_dict['evidence_spans']) > 0
        
        # Verify evidence span has all EvidenceChunk fields
        span = pair_dict['evidence_spans'][0]
        required_evidence_fields = [
            'doc_id', 'sent_id', 'text', 'char_start', 'char_end',
            'score_dense', 'rank', 'source', 'version'
        ]
        for field in required_evidence_fields:
            assert field in span, f"Missing evidence field: {field}"
    
    @pytest.mark.requires_data
    def test_full_pipeline_with_real_components(self):
        """
        Test complete pipeline with actual components (requires processed data).
        
        This test is marked as requires_data and will be skipped if the
        FAISS index doesn't exist.
        """
        from src.pipelines import BaselineRAGPipeline
        
        # Try to load pipeline
        try:
            pipeline = BaselineRAGPipeline.from_config(
                config_path="config.yaml",
                strategy="development"
            )
        except FileNotFoundError:
            pytest.skip("FAISS index not found - skipping full pipeline test")
        
        # Run a simple query
        result = pipeline.run("What is machine learning?", top_k=3)
        
        # Verify output structure
        assert 'query' in result
        assert 'draft_response' in result
        assert 'claim_evidence_pairs' in result
        assert 'generator_metadata' in result
        assert 'retrieval_metadata' in result
        
        # Verify claim-evidence pairs structure
        for pair in result['claim_evidence_pairs']:
            assert 'claim_id' in pair
            assert 'evidence_candidates' in pair
            assert 'top_evidence' in pair
            assert 'evidence_spans' in pair
        
        # Verify generator metadata for Month 3
        gen_meta = result['generator_metadata']
        assert 'logits' in gen_meta
        assert 'scores' in gen_meta
        assert 'tokens' in gen_meta
        assert len(gen_meta['logits']) > 0
        assert len(gen_meta['scores']) > 0


@pytest.mark.unit
class TestDataStructureInteroperability:
    """Test that data structures work together correctly."""
    
    def test_evidence_chunk_serialization(self):
        """Test EvidenceChunk can be serialized and deserialized."""
        chunk = EvidenceChunk(
            doc_id="test_doc",
            sent_id=0,
            text="Test text",
            char_start=0,
            char_end=9,
            score_dense=0.85,
            rank=0
        )
        
        # Serialize to dict
        chunk_dict = chunk.to_dict()
        
        # Verify all fields present
        assert chunk_dict['doc_id'] == "test_doc"
        assert chunk_dict['sent_id'] == 0
        assert chunk_dict['text'] == "Test text"
        
        # Serialize to JSON
        json_str = json.dumps(chunk_dict)
        loaded_dict = json.loads(json_str)
        
        assert loaded_dict == chunk_dict
    
    def test_claim_serialization(self):
        """Test Claim can be serialized and deserialized."""
        claim = Claim(
            claim_id="c_001",
            answer_id="ans_001",
            text="Test claim",
            answer_char_span=[0, 10]
        )
        
        claim_dict = claim.to_dict()
        json_str = json.dumps(claim_dict)
        loaded_dict = json.loads(json_str)
        
        assert loaded_dict == claim_dict
        assert loaded_dict['claim_id'] == "c_001"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'integration'])
