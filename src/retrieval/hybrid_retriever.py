"""
Hybrid retriever combining dense and sparse (BM25) retrieval.

This module provides the HybridRetriever class that combines results from
DenseRetriever and BM25Retriever using configurable fusion methods.
"""

import numpy as np
from typing import List, Dict, Literal
from collections import defaultdict

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.utils.data_structures import EvidenceChunk
from src.utils.logger import setup_logger


class HybridRetriever:
    """
    Hybrid retriever combining dense and sparse retrieval.
    
    Fetches top-k results from both DenseRetriever and BM25Retriever,
    then fuses their scores using either linear combination or reciprocal
    rank fusion (RRF). Handles vocabulary mismatches by zero-filling
    missing scores.
    
    Attributes:
        dense_retriever: DenseRetriever instance
        bm25_retriever: BM25Retriever instance
        alpha: Weight for dense retrieval in linear fusion (0-1)
        fusion_method: Fusion method ('linear' or 'rrf')
        rrf_k: RRF constant (default: 60)
        logger: Logger instance
    
    Example:
        >>> dense = DenseRetriever(...)
        >>> bm25 = BM25Retriever(...)
        >>> hybrid = HybridRetriever(
        ...     dense_retriever=dense,
        ...     bm25_retriever=bm25,
        ...     alpha=0.5,
        ...     fusion_method='rrf'
        ... )
        >>> results = hybrid.retrieve("What is machine learning?", top_k=5)
        >>> print(f"Hybrid score: {results[0].score_hybrid}")
    """
    
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        alpha: float = 0.5,
        fusion_method: Literal['linear', 'rrf'] = 'rrf',
        rrf_k: int = 60
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            dense_retriever: Initialized DenseRetriever instance
            bm25_retriever: Initialized BM25Retriever instance
            alpha: Weight for dense scores in linear fusion (0-1). 
                   alpha=1.0 uses only dense, alpha=0.0 uses only BM25.
            fusion_method: Score fusion method ('linear' or 'rrf')
            rrf_k: Constant for reciprocal rank fusion (default: 60)
        """
        self.logger = setup_logger(self.__class__.__name__)
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        
        if fusion_method not in ['linear', 'rrf']:
            raise ValueError(f"fusion_method must be 'linear' or 'rrf', got {fusion_method}")
        
        self.logger.info(
            f"Hybrid retriever initialized: alpha={alpha}, "
            f"fusion_method={fusion_method}, rrf_k={rrf_k}"
        )
    
    def _normalize_scores(self, chunks: List[EvidenceChunk], score_field: str) -> List[EvidenceChunk]:
        """
        Normalize scores to [0, 1] range using min-max normalization.
        
        Args:
            chunks: List of EvidenceChunk objects
            score_field: Field name to normalize ('score_dense' or 'score_bm25')
            
        Returns:
            List of chunks with normalized scores
        """
        if not chunks:
            return chunks
        
        scores = [getattr(chunk, score_field) for chunk in chunks]
        min_score = min(scores)
        max_score = max(scores)
        
        # Avoid division by zero
        if max_score - min_score == 0:
            for chunk in chunks:
                setattr(chunk, score_field, 1.0)
            return chunks
        
        # Normalize
        for chunk in chunks:
            old_score = getattr(chunk, score_field)
            new_score = (old_score - min_score) / (max_score - min_score)
            setattr(chunk, score_field, new_score)
        
        return chunks
    
    def _linear_fusion(
        self,
        dense_chunks: List[EvidenceChunk],
        bm25_chunks: List[EvidenceChunk]
    ) -> List[EvidenceChunk]:
        """
        Fuse results using linear combination of normalized scores.
        
        Formula: score_hybrid = alpha * score_dense + (1 - alpha) * score_bm25
        
        Args:
            dense_chunks: Results from dense retriever
            bm25_chunks: Results from BM25 retriever
            
        Returns:
            Fused and re-ranked chunks with score_hybrid populated
        """
        # Normalize scores
        dense_chunks = self._normalize_scores(dense_chunks, 'score_dense')
        bm25_chunks = self._normalize_scores(bm25_chunks, 'score_bm25')
        
        # Build lookup dictionaries by (doc_id, sent_id)
        dense_map = {(c.doc_id, c.sent_id): c for c in dense_chunks}
        bm25_map = {(c.doc_id, c.sent_id): c for c in bm25_chunks}
        
        # Combine all unique chunks
        all_keys = set(dense_map.keys()) | set(bm25_map.keys())
        
        fused_chunks = []
        for key in all_keys:
            # Get chunks from both retrievers (None if not present)
            dense_chunk = dense_map.get(key)
            bm25_chunk = bm25_map.get(key)
            
            # Zero-fill missing scores
            score_dense = dense_chunk.score_dense if dense_chunk else 0.0
            score_bm25 = bm25_chunk.score_bm25 if bm25_chunk else 0.0
            
            # Compute hybrid score
            score_hybrid = self.alpha * score_dense + (1 - self.alpha) * score_bm25
            
            # Use the chunk that exists (prefer dense if both exist)
            base_chunk = dense_chunk if dense_chunk else bm25_chunk
            
            # Create new chunk with all scores populated
            chunk = EvidenceChunk(
                doc_id=base_chunk.doc_id,
                sent_id=base_chunk.sent_id,
                text=base_chunk.text,
                char_start=base_chunk.char_start,
                char_end=base_chunk.char_end,
                score_dense=score_dense,
                score_bm25=score_bm25,
                score_hybrid=score_hybrid,
                rank=0,  # Will be updated after sorting
                source=base_chunk.source,
                version=base_chunk.version
            )
            fused_chunks.append(chunk)
        
        # Sort by hybrid score (descending)
        fused_chunks.sort(key=lambda c: c.score_hybrid, reverse=True)
        
        # Update ranks
        for rank, chunk in enumerate(fused_chunks, start=1):
            chunk.rank = rank
        
        return fused_chunks
    
    def _rrf_fusion(
        self,
        dense_chunks: List[EvidenceChunk],
        bm25_chunks: List[EvidenceChunk]
    ) -> List[EvidenceChunk]:
        """
        Fuse results using Reciprocal Rank Fusion (RRF).
        
        Formula: RRF(chunk) = sum_over_retrievers(1 / (k + rank))
        
        Args:
            dense_chunks: Results from dense retriever
            bm25_chunks: Results from BM25 retriever
            
        Returns:
            Fused and re-ranked chunks with score_hybrid populated
        """
        # Build lookup dictionaries by (doc_id, sent_id)
        dense_map = {(c.doc_id, c.sent_id): c for c in dense_chunks}
        bm25_map = {(c.doc_id, c.sent_id): c for c in bm25_chunks}
        
        # Build rank dictionaries (1-indexed)
        dense_ranks = {(c.doc_id, c.sent_id): c.rank for c in dense_chunks}
        bm25_ranks = {(c.doc_id, c.sent_id): c.rank for c in bm25_chunks}
        
        # Combine all unique chunks
        all_keys = set(dense_map.keys()) | set(bm25_map.keys())
        
        # Compute RRF scores
        rrf_scores = {}
        for key in all_keys:
            rrf_score = 0.0
            
            # Add contribution from dense retriever
            if key in dense_ranks:
                rrf_score += 1.0 / (self.rrf_k + dense_ranks[key])
            
            # Add contribution from BM25 retriever
            if key in bm25_ranks:
                rrf_score += 1.0 / (self.rrf_k + bm25_ranks[key])
            
            rrf_scores[key] = rrf_score
        
        # Build fused chunks
        fused_chunks = []
        for key in all_keys:
            # Get chunks from both retrievers (None if not present)
            dense_chunk = dense_map.get(key)
            bm25_chunk = bm25_map.get(key)
            
            # Zero-fill missing scores
            score_dense = dense_chunk.score_dense if dense_chunk else 0.0
            score_bm25 = bm25_chunk.score_bm25 if bm25_chunk else 0.0
            
            # Use the chunk that exists (prefer dense if both exist)
            base_chunk = dense_chunk if dense_chunk else bm25_chunk
            
            # Create new chunk with all scores populated
            chunk = EvidenceChunk(
                doc_id=base_chunk.doc_id,
                sent_id=base_chunk.sent_id,
                text=base_chunk.text,
                char_start=base_chunk.char_start,
                char_end=base_chunk.char_end,
                score_dense=score_dense,
                score_bm25=score_bm25,
                score_hybrid=rrf_scores[key],
                rank=0,  # Will be updated after sorting
                source=base_chunk.source,
                version=base_chunk.version
            )
            fused_chunks.append(chunk)
        
        # Sort by RRF score (descending)
        fused_chunks.sort(key=lambda c: c.score_hybrid, reverse=True)
        
        # Update ranks
        for rank, chunk in enumerate(fused_chunks, start=1):
            chunk.rank = rank
        
        return fused_chunks
    
    def retrieve(self, query: str, top_k: int = 5) -> List[EvidenceChunk]:
        """
        Retrieve top-k most relevant chunks using hybrid fusion.
        
        Fetches top-k from both dense and BM25 retrievers (2k candidates total),
        fuses scores, and returns top-k from the fused results.
        
        Args:
            query: Query string
            top_k: Number of top results to return
            
        Returns:
            List of EvidenceChunk objects ranked by hybrid score (highest first)
        """
        # Retrieve from both retrievers
        self.logger.debug(f"Retrieving from dense retriever (top_k={top_k})")
        dense_chunks = self.dense_retriever.retrieve(query, top_k=top_k)
        
        self.logger.debug(f"Retrieving from BM25 retriever (top_k={top_k})")
        bm25_chunks = self.bm25_retriever.retrieve(query, top_k=top_k)
        
        # Fuse results
        if self.fusion_method == 'linear':
            self.logger.debug("Fusing with linear combination")
            fused_chunks = self._linear_fusion(dense_chunks, bm25_chunks)
        else:  # 'rrf'
            self.logger.debug("Fusing with reciprocal rank fusion")
            fused_chunks = self._rrf_fusion(dense_chunks, bm25_chunks)
        
        # Return top-k from fused results
        results = fused_chunks[:top_k]
        
        self.logger.debug(
            f"Retrieved {len(results)} chunks for query: {query[:50]}..."
        )
        
        return results
    
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 5
    ) -> List[List[EvidenceChunk]]:
        """
        Retrieve top-k chunks for multiple queries using hybrid fusion.
        
        Args:
            queries: List of query strings
            top_k: Number of top results per query
            
        Returns:
            List of lists of EvidenceChunk objects, one list per query
        """
        self.logger.info(f"Batch hybrid retrieving for {len(queries)} queries")
        
        results = []
        for i, query in enumerate(queries):
            if (i + 1) % 100 == 0:
                self.logger.info(f"Processed {i + 1}/{len(queries)} queries")
            results.append(self.retrieve(query, top_k))
        
        return results
