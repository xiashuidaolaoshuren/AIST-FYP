"""
Data processing package for Wikipedia parsing and text chunking.

This package provides utilities for processing Wikipedia dumps,
creating sentence-level chunks, and generating embeddings for retrieval.
"""

from .wiki_parser import WikipediaParser
from .text_chunker import TextChunker
from .embedding_generator import EmbeddingGenerator

__all__ = [
    'WikipediaParser',
    'TextChunker',
    'EmbeddingGenerator',
]
