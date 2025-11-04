"""
Pipelines package for RAG system.

This package contains pipeline implementations that integrate multiple
components (retrieval, generation, verification) into end-to-end workflows.
"""

from src.pipelines.baseline_rag import BaselineRAGPipeline

__all__ = ['BaselineRAGPipeline']
