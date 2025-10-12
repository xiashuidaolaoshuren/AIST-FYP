"""
Retrieval module for the Hallucination Detection System.

This module provides components for building and querying FAISS indices
for approximate nearest neighbor search over embeddings.
"""

from .faiss_index_manager import FAISSIndexManager

__all__ = ['FAISSIndexManager']
