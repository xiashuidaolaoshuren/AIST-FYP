"""
Dense retriever for approximate nearest neighbor search.

This module provides the DenseRetriever class that encodes queries using
sentence-transformers, searches a FAISS index, and returns ranked EvidenceChunk
objects with relevance scores.
"""

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List
from sentence_transformers import SentenceTransformer

from src.utils.data_structures import EvidenceChunk
from src.utils.logger import setup_logger


class DenseRetriever:
    """
    Dense retriever for query-based evidence retrieval using FAISS.
    
    Encodes queries with the same sentence-transformer model used for
    embedding generation, searches a FAISS index using cosine similarity
    (via inner product on L2-normalized vectors), and returns ranked
    EvidenceChunk objects with relevance scores.
    
    Attributes:
        index: Loaded FAISS index
        metadata: List of chunk metadata dictionaries
        encoder: SentenceTransformer model for query encoding
        device: Device for encoding ('cuda' or 'cpu')
        logger: Logger instance
    
    Example:
        >>> retriever = DenseRetriever(
        ...     index_path='data/indexes/dev/faiss.index',
        ...     metadata_path='data/indexes/dev/metadata.pkl',
        ...     encoder_model='sentence-transformers/all-MiniLM-L6-v2'
        ... )
        >>> results = retriever.retrieve("What is machine learning?", top_k=5)
        >>> print(f"Top result: {results[0].text}")
    """
    
    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        encoder_model: str,
        device: str = 'cuda'
    ):
        """
        Initialize the dense retriever.
        
        Loads the FAISS index, metadata, and sentence-transformer encoder.
        Verifies that index size matches metadata length.
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata pickle file
            encoder_model: Name of sentence-transformer model (must match embedding model)
            device: Device to run encoder on ('cuda' or 'cpu')
        
        Raises:
            FileNotFoundError: If index or metadata files don't exist
            ValueError: If index size doesn't match metadata length
        """
        self.device = device
        self.logger = setup_logger(__name__)
        
        # Load FAISS index
        index_path = Path(index_path)
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        
        self.logger.info(f"Loading FAISS index from {index_path}")
        self.index = faiss.read_index(str(index_path))
        self.logger.info(f"FAISS index loaded: {self.index.ntotal:,} vectors")
        
        # Load metadata
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        self.logger.info(f"Loading metadata from {metadata_path}")
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        self.logger.info(f"Metadata loaded: {len(self.metadata):,} entries")
        
        # Verify consistency
        if len(self.metadata) != self.index.ntotal:
            raise ValueError(
                f"Metadata length ({len(self.metadata)}) doesn't match "
                f"index size ({self.index.ntotal})"
            )
        
        # Load encoder model
        self.logger.info(f"Loading encoder model: {encoder_model}")
        self.encoder = SentenceTransformer(encoder_model, device=device)
        self.logger.info(f"Encoder loaded on device: {self.encoder.device}")
        
        self.logger.info("DenseRetriever initialization complete")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[EvidenceChunk]:
        """
        Retrieve top-k most relevant evidence chunks for a query.
        
        Encodes the query with L2 normalization, searches the FAISS index
        using inner product (equivalent to cosine similarity for normalized
        vectors), and returns ranked EvidenceChunk objects.
        
        Args:
            query: Query text to search for
            top_k: Number of results to return (default: 5)
        
        Returns:
            List of EvidenceChunk objects sorted by rank (1 to top_k),
            with score_dense field populated with similarity scores
        
        Example:
            >>> results = retriever.retrieve("What is AI?", top_k=3)
            >>> for chunk in results:
            ...     print(f"Rank {chunk.rank}: {chunk.text[:50]}...")
        """
        if not query or not query.strip():
            self.logger.warning("Empty query provided, returning empty list")
            return []
        
        # Encode query with L2 normalization
        self.logger.debug(f"Encoding query: {query[:50]}...")
        query_embedding = self.encoder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        # Verify embedding shape
        expected_dim = self.index.d
        if query_embedding.shape != (1, expected_dim):
            raise ValueError(
                f"Query embedding shape {query_embedding.shape} doesn't match "
                f"expected shape (1, {expected_dim})"
            )
        
        # Ensure float32 for FAISS
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        
        # Search FAISS index
        self.logger.debug(f"Searching index for top-{top_k} results")
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Convert to 1D arrays (remove batch dimension)
        scores = scores[0]  # Shape: (top_k,)
        indices = indices[0]  # Shape: (top_k,)
        
        # Create EvidenceChunk objects
        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
            # Convert numpy types to Python types
            idx = int(idx)
            score = float(score)
            
            # Check for invalid indices (can happen if top_k > index size)
            if idx < 0 or idx >= len(self.metadata):
                self.logger.warning(
                    f"Invalid index {idx} returned by FAISS search, skipping"
                )
                continue
            
            # Get metadata for this chunk
            metadata = self.metadata[idx]
            
            # Create EvidenceChunk with all required fields
            chunk = EvidenceChunk(
                doc_id=metadata['doc_id'],
                sent_id=metadata['sent_id'],
                text=metadata['text'],
                char_start=metadata['char_start'],
                char_end=metadata['char_end'],
                source=metadata['source'],
                version=metadata['version'],
                score_dense=score,
                rank=rank
            )
            
            results.append(chunk)
        
        self.logger.info(
            f"Retrieved {len(results)} results for query. "
            f"Top score: {results[0].score_dense:.4f}" if results else "No results"
        )
        
        return results
    
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 5
    ) -> List[List[EvidenceChunk]]:
        """
        Retrieve evidence chunks for multiple queries in batch.
        
        More efficient than calling retrieve() multiple times as it
        encodes all queries together and performs batch FAISS search.
        
        Args:
            queries: List of query texts
            top_k: Number of results per query (default: 5)
        
        Returns:
            List of result lists, one per query, each containing
            EvidenceChunk objects sorted by rank
        
        Example:
            >>> queries = ["What is AI?", "Define machine learning"]
            >>> all_results = retriever.batch_retrieve(queries, top_k=3)
            >>> for i, results in enumerate(all_results):
            ...     print(f"Query {i+1}: {len(results)} results")
        """
        if not queries:
            return []
        
        # Filter out empty queries
        valid_queries = [q for q in queries if q and q.strip()]
        if not valid_queries:
            self.logger.warning("All queries are empty, returning empty lists")
            return [[] for _ in queries]
        
        # Encode all queries at once
        self.logger.debug(f"Encoding {len(valid_queries)} queries")
        query_embeddings = self.encoder.encode(
            valid_queries,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        # Ensure float32
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype(np.float32)
        
        # Batch search
        self.logger.debug(f"Batch searching index for top-{top_k} results per query")
        scores, indices = self.index.search(query_embeddings, top_k)
        
        # Process results for each query
        all_results = []
        for query_idx in range(len(valid_queries)):
            query_scores = scores[query_idx]
            query_indices = indices[query_idx]
            
            # Create EvidenceChunk objects for this query
            query_results = []
            for rank, (idx, score) in enumerate(zip(query_indices, query_scores), start=1):
                idx = int(idx)
                score = float(score)
                
                if idx < 0 or idx >= len(self.metadata):
                    continue
                
                metadata = self.metadata[idx]
                chunk = EvidenceChunk(
                    doc_id=metadata['doc_id'],
                    sent_id=metadata['sent_id'],
                    text=metadata['text'],
                    char_start=metadata['char_start'],
                    char_end=metadata['char_end'],
                    source=metadata['source'],
                    version=metadata['version'],
                    score_dense=score,
                    rank=rank
                )
                query_results.append(chunk)
            
            all_results.append(query_results)
        
        self.logger.info(
            f"Batch retrieved for {len(queries)} queries, "
            f"average {sum(len(r) for r in all_results) / len(queries):.1f} results per query"
        )
        
        return all_results
