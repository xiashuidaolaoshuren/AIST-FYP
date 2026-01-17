"""
Evaluation module for RAG system assessment.

This module provides wrappers for external evaluation frameworks (Ragas) and
custom benchmark evaluation harnesses (RAGTruth, CiteBench).

Modules:
    ragas_evaluator: Wrapper for Ragas framework evaluation metrics
    ragtruth_evaluator: Evaluation harness for RAGTruth hallucination benchmark
"""

from .ragas_evaluator import RagasEvaluator
from .ragtruth_evaluator import RAGTruthEvaluator

__all__ = ['RagasEvaluator', 'RAGTruthEvaluator']
