"""
Retrieval module for the Hallucination Detection System.

This module provides components for building and querying FAISS indices
for approximate nearest neighbor search over embeddings, and for retrieving
relevant evidence chunks based on queries.
"""

from .faiss_index_manager import FAISSIndexManager
from .dense_retriever import DenseRetriever

__all__ = ['FAISSIndexManager', 'DenseRetriever']
