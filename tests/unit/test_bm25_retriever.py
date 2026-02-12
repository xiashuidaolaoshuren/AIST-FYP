"""
Unit tests for BM25Retriever.

Tests BM25 retriever functionality including tokenization, scoring,
index building, caching, and batch retrieval.
"""

import pytest
import json
import tempfile
from pathlib import Path
from src.retrieval.bm25_retriever import BM25Retriever
from src.utils.data_structures import EvidenceChunk


@pytest.fixture
def sample_corpus_file(tmp_path):
    """Create a temporary corpus file for testing."""
    corpus_path = tmp_path / "test_corpus.jsonl"
    
    # Sample chunks
    chunks = [
        {
            "doc_id": "doc1",
            "sent_id": 0,
            "text": "Machine learning is a subset of artificial intelligence.",
            "char_start": 0,
            "char_end": 56,
            "source": "wikipedia",
            "version": "test_v1"
        },
        {
            "doc_id": "doc1",
            "sent_id": 1,
            "text": "Deep learning uses neural networks with multiple layers.",
            "char_start": 57,
            "char_end": 113,
            "source": "wikipedia",
            "version": "test_v1"
        },
        {
            "doc_id": "doc2",
            "sent_id": 0,
            "text": "Python is a popular programming language for machine learning.",
            "char_start": 0,
            "char_end": 62,
            "source": "wikipedia",
            "version": "test_v1"
        },
        {
            "doc_id": "doc3",
            "sent_id": 0,
            "text": "The capital of France is Paris.",
            "char_start": 0,
            "char_end": 31,
            "source": "wikipedia",
            "version": "test_v1"
        }
    ]
    
    # Write to file
    with open(corpus_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + '\n')
    
    return corpus_path


class TestBM25Retriever:
    """Test suite for BM25Retriever."""
    
    def test_initialization_without_cache(self, sample_corpus_file):
        """Test initializing retriever without cached index."""
        retriever = BM25Retriever(
            corpus_path=str(sample_corpus_file),
            index_path=None,
            k1=1.5,
            b=0.75
        )
        
        assert retriever is not None
        assert len(retriever.chunks) == 4
        assert retriever.bm25 is not None
        assert retriever.k1 == 1.5
        assert retriever.b == 0.75
    
    def test_initialization_with_cache(self, sample_corpus_file, tmp_path):
        """Test initializing retriever with cached index."""
        index_path = tmp_path / "bm25_index.pkl"
        
        # First: build and cache
        retriever1 = BM25Retriever(
            corpus_path=str(sample_corpus_file),
            index_path=str(index_path),
            k1=1.5,
            b=0.75
        )
        
        assert index_path.exists()
        
        # Second: load from cache
        retriever2 = BM25Retriever(
            corpus_path=str(sample_corpus_file),
            index_path=str(index_path),
            k1=1.5,
            b=0.75
        )
        
        assert retriever2 is not None
        assert len(retriever2.chunks) == 4
    
    def test_tokenization(self, sample_corpus_file):
        """Test spaCy tokenization."""
        retriever = BM25Retriever(
            corpus_path=str(sample_corpus_file),
            index_path=None
        )
        
        tokens = retriever._tokenize("Machine learning is awesome!")
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)
        assert all(t.islower() for t in tokens if t.isalpha())
    
    def test_retrieve_basic(self, sample_corpus_file):
        """Test basic retrieval functionality."""
        retriever = BM25Retriever(
            corpus_path=str(sample_corpus_file),
            index_path=None
        )
        
        results = retriever.retrieve("machine learning", top_k=3)
        
        assert isinstance(results, list)
        assert len(results) <= 3
        assert all(isinstance(r, EvidenceChunk) for r in results)
        
        # Check that top result contains relevant term
        if results:
            top_result = results[0]
            assert "machine learning" in top_result.text.lower() or \
                   "machine" in top_result.text.lower()
    
    def test_retrieve_score_ordering(self, sample_corpus_file):
        """Test that results are ordered by BM25 score (descending)."""
        retriever = BM25Retriever(
            corpus_path=str(sample_corpus_file),
            index_path=None
        )
        
        results = retriever.retrieve("machine learning", top_k=4)
        
        # Check scores are in descending order
        scores = [r.score_bm25 for r in results]
        assert scores == sorted(scores, reverse=True)
        
        # Check ranks are sequential
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, len(results) + 1))
    
    def test_retrieve_score_populated(self, sample_corpus_file):
        """Test that score_bm25 is populated correctly."""
        retriever = BM25Retriever(
            corpus_path=str(sample_corpus_file),
            index_path=None
        )
        
        results = retriever.retrieve("machine learning", top_k=3)
        
        for result in results:
            assert result.score_bm25 is not None
            assert result.score_bm25 >= 0
            assert result.score_dense == 0.0  # Should be zero for BM25-only
    
    def test_retrieve_metadata_preservation(self, sample_corpus_file):
        """Test that chunk metadata is preserved."""
        retriever = BM25Retriever(
            corpus_path=str(sample_corpus_file),
            index_path=None
        )
        
        results = retriever.retrieve("machine learning", top_k=1)
        
        assert len(results) > 0
        chunk = results[0]
        
        assert chunk.doc_id is not None
        assert chunk.sent_id >= 0
        assert chunk.text is not None
        assert chunk.char_start >= 0
        assert chunk.char_end > chunk.char_start
        assert chunk.source == "wikipedia"
        assert chunk.version == "test_v1"
    
    def test_retrieve_empty_query(self, sample_corpus_file):
        """Test retrieval with empty query."""
        retriever = BM25Retriever(
            corpus_path=str(sample_corpus_file),
            index_path=None
        )
        
        results = retriever.retrieve("", top_k=3)
        
        # Empty query should return empty results
        assert isinstance(results, list)
        assert len(results) == 0
    
    def test_retrieve_no_match(self, sample_corpus_file):
        """Test retrieval with query that has no strong matches."""
        retriever = BM25Retriever(
            corpus_path=str(sample_corpus_file),
            index_path=None
        )
        
        results = retriever.retrieve("quantum cryptography blockchain", top_k=3)
        
        # Should still return results (even with low scores)
        assert isinstance(results, list)
        # BM25 will return results, but scores should be relatively low
    
    def test_batch_retrieve(self, sample_corpus_file):
        """Test batch retrieval."""
        retriever = BM25Retriever(
            corpus_path=str(sample_corpus_file),
            index_path=None
        )
        
        queries = [
            "machine learning",
            "France Paris",
            "neural networks"
        ]
        
        results = retriever.batch_retrieve(queries, top_k=2)
        
        assert isinstance(results, list)
        assert len(results) == len(queries)
        
        for query_results in results:
            assert isinstance(query_results, list)
            assert len(query_results) <= 2
            assert all(isinstance(r, EvidenceChunk) for r in query_results)
    
    def test_top_k_limits(self, sample_corpus_file):
        """Test that top_k is respected even if greater than corpus size."""
        retriever = BM25Retriever(
            corpus_path=str(sample_corpus_file),
            index_path=None
        )
        
        # Request more results than corpus size
        results = retriever.retrieve("test", top_k=100)
        
        # Should return at most corpus size
        assert len(results) <= len(retriever.chunks)
    
    def test_parameter_warnings(self, sample_corpus_file, tmp_path):
        """Test warnings when cached index parameters differ from requested."""
        index_path = tmp_path / "bm25_index.pkl"
        
        # Build with specific parameters
        retriever1 = BM25Retriever(
            corpus_path=str(sample_corpus_file),
            index_path=str(index_path),
            k1=1.2,
            b=0.8
        )
        
        # Load with different parameters (should warn but still work)
        retriever2 = BM25Retriever(
            corpus_path=str(sample_corpus_file),
            index_path=str(index_path),
            k1=1.5,
            b=0.75
        )
        
        # Should still load successfully
        assert retriever2 is not None
        assert len(retriever2.chunks) == 4
