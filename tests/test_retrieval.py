"""
Unit tests for the Dense Retriever.

Tests the DenseRetriever class for query encoding, FAISS search,
and EvidenceChunk object creation with proper ranking and scoring.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import DenseRetriever
from src.utils.data_structures import EvidenceChunk


class TestDenseRetriever:
    """Test suite for DenseRetriever class."""
    
    @pytest.fixture
    def retriever(self):
        """Create a DenseRetriever instance with development index."""
        project_root = Path(__file__).parent.parent
        index_dir = project_root / 'data' / 'indexes' / 'development'
        
        retriever = DenseRetriever(
            index_path=str(index_dir / 'faiss.index'),
            metadata_path=str(index_dir / 'metadata.pkl'),
            encoder_model='sentence-transformers/all-MiniLM-L6-v2',
            device='cuda'
        )
        return retriever
    
    def test_initialization(self, retriever):
        """Test retriever initialization."""
        assert retriever.index is not None
        assert retriever.metadata is not None
        assert retriever.encoder is not None
        assert len(retriever.metadata) == retriever.index.ntotal
        assert retriever.index.ntotal > 0
    
    def test_initialization_missing_index(self):
        """Test initialization with missing index file."""
        with pytest.raises(FileNotFoundError, match="FAISS index not found"):
            DenseRetriever(
                index_path='nonexistent/faiss.index',
                metadata_path='nonexistent/metadata.pkl',
                encoder_model='sentence-transformers/all-MiniLM-L6-v2'
            )
    
    def test_initialization_missing_metadata(self):
        """Test initialization with missing metadata file."""
        project_root = Path(__file__).parent.parent
        index_dir = project_root / 'data' / 'indexes' / 'development'
        
        with pytest.raises(FileNotFoundError, match="Metadata not found"):
            DenseRetriever(
                index_path=str(index_dir / 'faiss.index'),
                metadata_path='nonexistent/metadata.pkl',
                encoder_model='sentence-transformers/all-MiniLM-L6-v2'
            )
    
    def test_retrieve_basic(self, retriever):
        """Test basic retrieval functionality."""
        query = "What is artificial intelligence?"
        results = retriever.retrieve(query, top_k=5)
        
        # Check results
        assert len(results) <= 5
        assert len(results) > 0
        assert all(isinstance(chunk, EvidenceChunk) for chunk in results)
    
    def test_retrieve_evidence_chunk_fields(self, retriever):
        """Test that retrieved EvidenceChunks have all required fields."""
        query = "machine learning algorithms"
        results = retriever.retrieve(query, top_k=3)
        
        for chunk in results:
            # Check all required fields exist
            assert hasattr(chunk, 'doc_id')
            assert hasattr(chunk, 'sent_id')
            assert hasattr(chunk, 'text')
            assert hasattr(chunk, 'char_start')
            assert hasattr(chunk, 'char_end')
            assert hasattr(chunk, 'source')
            assert hasattr(chunk, 'version')
            assert hasattr(chunk, 'score_dense')
            assert hasattr(chunk, 'rank')
            
            # Check field types
            assert isinstance(chunk.doc_id, str)
            assert isinstance(chunk.sent_id, int)
            assert isinstance(chunk.text, str)
            assert isinstance(chunk.char_start, int)
            assert isinstance(chunk.char_end, int)
            assert isinstance(chunk.source, str)
            assert isinstance(chunk.version, str)
            assert isinstance(chunk.score_dense, float)
            assert isinstance(chunk.rank, int)
            
            # Check field values
            assert len(chunk.text) > 0
            assert chunk.char_start >= 0
            assert chunk.char_end > chunk.char_start
            assert chunk.source == 'wikipedia'
    
    def test_retrieve_ranking_order(self, retriever):
        """Test that results are ranked correctly (1, 2, 3, ...)."""
        query = "deep learning neural networks"
        results = retriever.retrieve(query, top_k=5)
        
        # Check ranking
        expected_ranks = list(range(1, len(results) + 1))
        actual_ranks = [chunk.rank for chunk in results]
        assert actual_ranks == expected_ranks
        
        # Check scores are in descending order
        scores = [chunk.score_dense for chunk in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_retrieve_score_range(self, retriever):
        """Test that scores are in reasonable range."""
        query = "computer vision image recognition"
        results = retriever.retrieve(query, top_k=5)
        
        for chunk in results:
            # For normalized vectors with inner product, scores should be in [-1, 1]
            # But typically positive for similar items
            assert -1.0 <= chunk.score_dense <= 1.0
            # For relevant results, scores should be positive
            assert chunk.score_dense > 0
    
    def test_retrieve_empty_query(self, retriever):
        """Test retrieval with empty query."""
        results = retriever.retrieve("", top_k=5)
        assert len(results) == 0
        
        results = retriever.retrieve("   ", top_k=5)
        assert len(results) == 0
    
    def test_retrieve_different_top_k(self, retriever):
        """Test retrieval with different top_k values."""
        query = "natural language processing"
        
        # Test top_k=1
        results = retriever.retrieve(query, top_k=1)
        assert len(results) == 1
        assert results[0].rank == 1
        
        # Test top_k=3
        results = retriever.retrieve(query, top_k=3)
        assert len(results) <= 3
        
        # Test top_k=10
        results = retriever.retrieve(query, top_k=10)
        assert len(results) <= 10
    
    def test_retrieve_specific_query(self, retriever):
        """Test retrieval with a specific query about AI."""
        query = "artificial intelligence machines"
        results = retriever.retrieve(query, top_k=3)
        
        # Should return relevant results
        assert len(results) > 0
        
        # Top result should mention AI or related terms
        top_text = results[0].text.lower()
        assert any(
            term in top_text
            for term in ['artificial', 'intelligence', 'ai', 'machine', 'learning']
        )
    
    def test_batch_retrieve_basic(self, retriever):
        """Test batch retrieval with multiple queries."""
        queries = [
            "What is machine learning?",
            "deep neural networks",
            "natural language processing"
        ]
        
        all_results = retriever.batch_retrieve(queries, top_k=3)
        
        assert len(all_results) == len(queries)
        for results in all_results:
            assert len(results) <= 3
            assert all(isinstance(chunk, EvidenceChunk) for chunk in results)
    
    def test_batch_retrieve_empty_queries(self, retriever):
        """Test batch retrieval with empty queries."""
        queries = []
        all_results = retriever.batch_retrieve(queries, top_k=5)
        assert len(all_results) == 0
        
        queries = ["", "  ", ""]
        all_results = retriever.batch_retrieve(queries, top_k=5)
        assert len(all_results) == len(queries)
        assert all(len(r) == 0 for r in all_results)
    
    def test_batch_retrieve_ranking(self, retriever):
        """Test that each query's results are ranked independently."""
        queries = ["machine learning", "deep learning"]
        all_results = retriever.batch_retrieve(queries, top_k=5)
        
        for results in all_results:
            # Each query should have independent ranking starting from 1
            if len(results) > 0:
                expected_ranks = list(range(1, len(results) + 1))
                actual_ranks = [chunk.rank for chunk in results]
                assert actual_ranks == expected_ranks
    
    def test_retrieve_consistency(self, retriever):
        """Test that same query returns same results."""
        query = "computer science algorithms"
        
        results1 = retriever.retrieve(query, top_k=5)
        results2 = retriever.retrieve(query, top_k=5)
        
        # Should return same results
        assert len(results1) == len(results2)
        for chunk1, chunk2 in zip(results1, results2):
            assert chunk1.doc_id == chunk2.doc_id
            assert chunk1.sent_id == chunk2.sent_id
            assert chunk1.text == chunk2.text
            assert abs(chunk1.score_dense - chunk2.score_dense) < 1e-6
            assert chunk1.rank == chunk2.rank


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
