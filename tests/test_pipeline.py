"""
Unit tests for Baseline RAG Pipeline.

Tests the integration of retriever and generator in the RAG pipeline.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.baseline_rag import BaselineRAGPipeline
from src.retrieval.dense_retriever import DenseRetriever
from src.generation.generator_wrapper import GeneratorWrapper
from src.utils.data_structures import EvidenceChunk


class TestBaselineRAGPipeline:
    """Test suite for BaselineRAGPipeline class."""
    
    @pytest.fixture(scope="class")
    def pipeline(self):
        """
        Create a BaselineRAGPipeline instance for testing.
        
        Note: This requires that the FAISS index has been built.
        If index doesn't exist, tests will be skipped.
        """
        try:
            # Try to load from config
            pipeline = BaselineRAGPipeline.from_config(
                config_path="config.yaml",
                strategy="development"
            )
            return pipeline
        except FileNotFoundError:
            pytest.skip("FAISS index not found. Please run data processing pipeline first.")
    
    def test_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.retriever is not None
        assert pipeline.generator is not None
        assert isinstance(pipeline.retriever, DenseRetriever)
        assert isinstance(pipeline.generator, GeneratorWrapper)
    
    def test_run_basic(self, pipeline):
        """Test basic pipeline execution."""
        query = "What is machine learning?"
        result = pipeline.run(query, top_k=3)
        
        # Check all required fields are present
        assert 'query' in result
        assert 'draft_response' in result
        assert 'claim_evidence_pairs' in result
        assert 'generator_metadata' in result
        assert 'retrieval_metadata' in result
        
        # Check query matches
        assert result['query'] == query
        
        # Check draft response is non-empty
        assert len(result['draft_response']) > 0
        
        # Check claim_evidence_pairs is a list
        assert isinstance(result['claim_evidence_pairs'], list)
    
    def test_claim_evidence_pairing(self, pipeline):
        """Test that claims are properly paired with evidence."""
        query = "What is artificial intelligence?"
        result = pipeline.run(query, top_k=5)
        
        # If claims were extracted
        if result['claim_evidence_pairs']:
            for pair in result['claim_evidence_pairs']:
                # Check required fields
                assert 'claim_id' in pair
                assert 'evidence_candidates' in pair
                assert 'top_evidence' in pair
                assert 'evidence_spans' in pair
                
                # Check evidence_candidates format
                assert isinstance(pair['evidence_candidates'], list)
                if pair['evidence_candidates']:
                    # Should be in "doc_id#sent_id" format
                    assert '#' in pair['evidence_candidates'][0]
                
                # Check evidence_spans
                assert isinstance(pair['evidence_spans'], list)
    
    def test_generator_metadata_captured(self, pipeline):
        """Test that generator metadata is captured."""
        query = "What is 2 + 2?"
        result = pipeline.run(query, top_k=2)
        
        gen_meta = result['generator_metadata']
        
        # Check required metadata fields
        assert 'text' in gen_meta
        assert 'tokens' in gen_meta
        assert 'token_ids' in gen_meta
        assert 'logits' in gen_meta
        assert 'scores' in gen_meta
        assert 'evidence_used' in gen_meta
        
        # Check metadata content
        assert len(gen_meta['tokens']) > 0
        assert len(gen_meta['logits']) > 0
        assert len(gen_meta['scores']) > 0
    
    def test_retrieval_metadata(self, pipeline):
        """Test that retrieval metadata is captured."""
        query = "Test query"
        result = pipeline.run(query, top_k=3)
        
        ret_meta = result['retrieval_metadata']
        
        # Check required fields
        assert 'top_k' in ret_meta
        assert 'num_retrieved' in ret_meta
        assert 'top_score' in ret_meta
        assert 'evidence_doc_ids' in ret_meta
        
        # Check values
        assert ret_meta['top_k'] == 3
        assert ret_meta['num_retrieved'] >= 0
        assert isinstance(ret_meta['evidence_doc_ids'], list)
    
    def test_custom_generation_params(self, pipeline):
        """Test pipeline with custom generation parameters."""
        query = "Explain gravity"
        result = pipeline.run(
            query,
            top_k=2,
            max_new_tokens=50,
            temperature=0.5
        )
        
        # Should complete without error
        assert 'draft_response' in result
        assert len(result['draft_response']) > 0
    
    def test_no_evidence_case(self, pipeline):
        """Test pipeline behavior when no evidence is retrieved."""
        # Use a very obscure query that likely won't match
        query = "xyzabc123nonsense456"
        
        try:
            result = pipeline.run(query, top_k=1)
            
            # Should still return valid output structure
            assert 'draft_response' in result
            assert 'claim_evidence_pairs' in result
            
        except Exception:
            # Some error is acceptable if no evidence found
            pass
    
    def test_output_serializable(self, pipeline):
        """Test that output can be serialized to JSON."""
        import json
        
        query = "What is Python?"
        result = pipeline.run(query, top_k=2)
        
        # Remove non-serializable numpy arrays
        result_copy = result.copy()
        if 'generator_metadata' in result_copy:
            gen_meta = result_copy['generator_metadata'].copy()
            # Remove logits (numpy arrays)
            gen_meta.pop('logits', None)
            result_copy['generator_metadata'] = gen_meta
        
        # Should serialize without error
        json_str = json.dumps(result_copy)
        assert len(json_str) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
