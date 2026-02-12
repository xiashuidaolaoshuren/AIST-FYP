"""
Unit tests for HybridRetriever.

Tests hybrid retrieval functionality including linear fusion, RRF fusion,
score normalization, and zero-filling for vocabulary mismatches.
"""

import pytest
from unittest.mock import Mock, MagicMock
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.data_structures import EvidenceChunk


@pytest.fixture
def mock_dense_results():
    """Mock results from dense retriever."""
    return [
        EvidenceChunk(
            doc_id="doc1", sent_id=0, text="Text 1", char_start=0, char_end=10,
            score_dense=0.9, rank=1
        ),
        EvidenceChunk(
            doc_id="doc2", sent_id=0, text="Text 2", char_start=0, char_end=10,
            score_dense=0.7, rank=2
        ),
        EvidenceChunk(
            doc_id="doc3", sent_id=0, text="Text 3", char_start=0, char_end=10,
            score_dense=0.5, rank=3
        )
    ]


@pytest.fixture
def mock_bm25_results():
    """Mock results from BM25 retriever."""
    return [
        EvidenceChunk(
            doc_id="doc2", sent_id=0, text="Text 2", char_start=0, char_end=10,
            score_dense=0.0, score_bm25=15.0, rank=1
        ),
        EvidenceChunk(
            doc_id="doc4", sent_id=0, text="Text 4", char_start=0, char_end=10,
            score_dense=0.0, score_bm25=12.0, rank=2
        ),
        EvidenceChunk(
            doc_id="doc1", sent_id=0, text="Text 1", char_start=0, char_end=10,
            score_dense=0.0, score_bm25=8.0, rank=3
        )
    ]


@pytest.fixture
def mock_dense_retriever(mock_dense_results):
    """Mock DenseRetriever."""
    retriever = Mock()
    retriever.retrieve = Mock(return_value=mock_dense_results)
    return retriever


@pytest.fixture
def mock_bm25_retriever(mock_bm25_results):
    """Mock BM25Retriever."""
    retriever = Mock()
    retriever.retrieve = Mock(return_value=mock_bm25_results)
    return retriever


class TestHybridRetriever:
    """Test suite for HybridRetriever."""
    
    def test_initialization_linear(self, mock_dense_retriever, mock_bm25_retriever):
        """Test initialization with linear fusion."""
        retriever = HybridRetriever(
            dense_retriever=mock_dense_retriever,
            bm25_retriever=mock_bm25_retriever,
            alpha=0.6,
            fusion_method='linear'
        )
        
        assert retriever.alpha == 0.6
        assert retriever.fusion_method == 'linear'
        assert retriever.dense_retriever == mock_dense_retriever
        assert retriever.bm25_retriever == mock_bm25_retriever
    
    def test_initialization_rrf(self, mock_dense_retriever, mock_bm25_retriever):
        """Test initialization with RRF fusion."""
        retriever = HybridRetriever(
            dense_retriever=mock_dense_retriever,
            bm25_retriever=mock_bm25_retriever,
            alpha=0.5,
            fusion_method='rrf',
            rrf_k=60
        )
        
        assert retriever.fusion_method == 'rrf'
        assert retriever.rrf_k == 60
    
    def test_initialization_invalid_alpha(self, mock_dense_retriever, mock_bm25_retriever):
        """Test that invalid alpha values raise error."""
        with pytest.raises(ValueError, match="alpha must be in"):
            HybridRetriever(
                dense_retriever=mock_dense_retriever,
                bm25_retriever=mock_bm25_retriever,
                alpha=1.5
            )
        
        with pytest.raises(ValueError, match="alpha must be in"):
            HybridRetriever(
                dense_retriever=mock_dense_retriever,
                bm25_retriever=mock_bm25_retriever,
                alpha=-0.1
            )
    
    def test_initialization_invalid_fusion_method(self, mock_dense_retriever, mock_bm25_retriever):
        """Test that invalid fusion method raises error."""
        with pytest.raises(ValueError, match="fusion_method must be"):
            HybridRetriever(
                dense_retriever=mock_dense_retriever,
                bm25_retriever=mock_bm25_retriever,
                fusion_method='invalid'
            )
    
    def test_normalize_scores(self, mock_dense_retriever, mock_bm25_retriever):
        """Test score normalization."""
        retriever = HybridRetriever(
            dense_retriever=mock_dense_retriever,
            bm25_retriever=mock_bm25_retriever
        )
        
        chunks = [
            EvidenceChunk(
                doc_id="doc1", sent_id=0, text="Text", char_start=0, char_end=10,
                score_dense=10.0, rank=1
            ),
            EvidenceChunk(
                doc_id="doc2", sent_id=0, text="Text", char_start=0, char_end=10,
                score_dense=5.0, rank=2
            ),
            EvidenceChunk(
                doc_id="doc3", sent_id=0, text="Text", char_start=0, char_end=10,
                score_dense=0.0, rank=3
            )
        ]
        
        normalized = retriever._normalize_scores(chunks, 'score_dense')
        
        # Scores should be in [0, 1] range
        assert all(0 <= c.score_dense <= 1 for c in normalized)
        
        # Min should be 0, max should be 1
        scores = [c.score_dense for c in normalized]
        assert min(scores) == 0.0
        assert max(scores) == 1.0
    
    def test_linear_fusion_basic(self, mock_dense_retriever, mock_bm25_retriever,
                                  mock_dense_results, mock_bm25_results):
        """Test basic linear fusion."""
        retriever = HybridRetriever(
            dense_retriever=mock_dense_retriever,
            bm25_retriever=mock_bm25_retriever,
            alpha=0.5,
            fusion_method='linear'
        )
        
        results = retriever.retrieve("test query", top_k=5)
        
        # Should have results
        assert len(results) > 0
        assert all(isinstance(r, EvidenceChunk) for r in results)
        
        # All three scores should be populated
        for result in results:
            assert result.score_dense is not None
            assert result.score_bm25 is not None
            assert result.score_hybrid is not None
    
    def test_linear_fusion_zero_filling(self, mock_dense_retriever, mock_bm25_retriever,
                                        mock_dense_results, mock_bm25_results):
        """Test zero-filling for vocabulary mismatches in linear fusion."""
        retriever = HybridRetriever(
            dense_retriever=mock_dense_retriever,
            bm25_retriever=mock_bm25_retriever,
            alpha=0.5,
            fusion_method='linear'
        )
        
        results = retriever.retrieve("test query", top_k=10)
        
        # Find doc4 which is only in BM25 results
        doc4_results = [r for r in results if r.doc_id == "doc4"]
        if doc4_results:
            doc4 = doc4_results[0]
            # Dense score should be zero-filled
            assert doc4.score_dense == 0.0
            # BM25 score should be present (normalized)
            assert doc4.score_bm25 > 0
            # Hybrid score should be computed
            assert doc4.score_hybrid is not None
        
        # Find doc3 which is only in dense results
        doc3_results = [r for r in results if r.doc_id == "doc3"]
        if doc3_results:
            doc3 = doc3_results[0]
            # BM25 score should be zero-filled
            assert doc3.score_bm25 == 0.0
            # Dense score should be present (normalized)
            # Note: after normalization it may be 0.0 if it's the minimum
            assert doc3.score_dense >= 0.0
            # Hybrid score should be computed
            assert doc3.score_hybrid is not None
    
    def test_rrf_fusion_basic(self, mock_dense_retriever, mock_bm25_retriever):
        """Test basic RRF fusion."""
        retriever = HybridRetriever(
            dense_retriever=mock_dense_retriever,
            bm25_retriever=mock_bm25_retriever,
            fusion_method='rrf',
            rrf_k=60
        )
        
        results = retriever.retrieve("test query", top_k=5)
        
        # Should have results
        assert len(results) > 0
        assert all(isinstance(r, EvidenceChunk) for r in results)
        
        # All scores should be populated
        for result in results:
            assert result.score_hybrid is not None
    
    def test_rrf_fusion_score_calculation(self, mock_dense_retriever, mock_bm25_retriever,
                                          mock_dense_results, mock_bm25_results):
        """Test RRF score calculation formula."""
        retriever = HybridRetriever(
            dense_retriever=mock_dense_retriever,
            bm25_retriever=mock_bm25_retriever,
            fusion_method='rrf',
            rrf_k=60
        )
        
        results = retriever.retrieve("test query", top_k=10)
        
        # Find doc1 which appears in both results
        doc1_results = [r for r in results if r.doc_id == "doc1"]
        if doc1_results:
            doc1 = doc1_results[0]
            
            # Doc1 is rank 1 in dense, rank 3 in BM25
            # RRF score should be: 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 ≈ 0.0323
            expected_rrf = 1.0/(60+1) + 1.0/(60+3)
            assert abs(doc1.score_hybrid - expected_rrf) < 0.001
    
    def test_rrf_fusion_zero_filling(self, mock_dense_retriever, mock_bm25_retriever):
        """Test zero-filling for vocabulary mismatches in RRF fusion."""
        retriever = HybridRetriever(
            dense_retriever=mock_dense_retriever,
            bm25_retriever=mock_bm25_retriever,
            fusion_method='rrf',
            rrf_k=60
        )
        
        results = retriever.retrieve("test query", top_k=10)
        
        # Find doc4 which is only in BM25
        doc4_results = [r for r in results if r.doc_id == "doc4"]
        if doc4_results:
            doc4 = doc4_results[0]
            # Dense score should be zero-filled
            assert doc4.score_dense == 0.0
            # RRF score should only include BM25 contribution
            # Doc4 is rank 2 in BM25
            expected_rrf = 1.0/(60+2)
            assert abs(doc4.score_hybrid - expected_rrf) < 0.001
    
    def test_retrieve_top_k_limit(self, mock_dense_retriever, mock_bm25_retriever):
        """Test that top_k is respected."""
        retriever = HybridRetriever(
            dense_retriever=mock_dense_retriever,
            bm25_retriever=mock_bm25_retriever
        )
        
        results = retriever.retrieve("test query", top_k=2)
        
        assert len(results) == 2
    
    def test_retrieve_rank_assignment(self, mock_dense_retriever, mock_bm25_retriever):
        """Test that ranks are assigned correctly after fusion."""
        retriever = HybridRetriever(
            dense_retriever=mock_dense_retriever,
            bm25_retriever=mock_bm25_retriever
        )
        
        results = retriever.retrieve("test query", top_k=5)
        
        # Ranks should be sequential starting from 1
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, len(results) + 1))
    
    def test_retrieve_score_ordering(self, mock_dense_retriever, mock_bm25_retriever):
        """Test that results are ordered by hybrid score (descending)."""
        retriever = HybridRetriever(
            dense_retriever=mock_dense_retriever,
            bm25_retriever=mock_bm25_retriever
        )
        
        results = retriever.retrieve("test query", top_k=5)
        
        # Hybrid scores should be in descending order
        scores = [r.score_hybrid for r in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_batch_retrieve(self, mock_dense_retriever, mock_bm25_retriever):
        """Test batch retrieval."""
        # Update mocks to support batch calls
        mock_dense_retriever.retrieve = Mock(side_effect=[
            [EvidenceChunk(doc_id="doc1", sent_id=0, text="T", char_start=0, 
                          char_end=1, score_dense=0.9, rank=1)],
            [EvidenceChunk(doc_id="doc2", sent_id=0, text="T", char_start=0, 
                          char_end=1, score_dense=0.8, rank=1)]
        ])
        
        mock_bm25_retriever.retrieve = Mock(side_effect=[
            [EvidenceChunk(doc_id="doc1", sent_id=0, text="T", char_start=0, 
                          char_end=1, score_dense=0.0, score_bm25=10.0, rank=1)],
            [EvidenceChunk(doc_id="doc3", sent_id=0, text="T", char_start=0, 
                          char_end=1, score_dense=0.0, score_bm25=8.0, rank=1)]
        ])
        
        retriever = HybridRetriever(
            dense_retriever=mock_dense_retriever,
            bm25_retriever=mock_bm25_retriever
        )
        
        queries = ["query1", "query2"]
        results = retriever.batch_retrieve(queries, top_k=3)
        
        assert isinstance(results, list)
        assert len(results) == len(queries)
        
        for query_results in results:
            assert isinstance(query_results, list)
            assert all(isinstance(r, EvidenceChunk) for r in query_results)
    
    def test_alpha_edge_cases(self, mock_dense_retriever, mock_bm25_retriever):
        """Test alpha=0 (BM25 only) and alpha=1 (dense only) edge cases."""
        # Alpha = 1.0 (dense only)
        retriever_dense_only = HybridRetriever(
            dense_retriever=mock_dense_retriever,
            bm25_retriever=mock_bm25_retriever,
            alpha=1.0,
            fusion_method='linear'
        )
        
        results = retriever_dense_only.retrieve("test", top_k=3)
        assert len(results) > 0
        
        # Alpha = 0.0 (BM25 only)
        retriever_bm25_only = HybridRetriever(
            dense_retriever=mock_dense_retriever,
            bm25_retriever=mock_bm25_retriever,
            alpha=0.0,
            fusion_method='linear'
        )
        
        results = retriever_bm25_only.retrieve("test", top_k=3)
        assert len(results) > 0
