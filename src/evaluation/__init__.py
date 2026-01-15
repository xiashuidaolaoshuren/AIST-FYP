"""
Evaluation module for RAG system assessment.

This module provides wrappers for external evaluation frameworks (Ragas) and
custom benchmark evaluation harnesses (RAGTruth, CiteBench).

Modules:
    ragas_evaluator: Wrapper for Ragas framework evaluation metrics
"""

from .ragas_evaluator import RagasEvaluator

__all__ = ['RagasEvaluator']
