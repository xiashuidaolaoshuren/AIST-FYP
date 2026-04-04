"""
Evaluation module for RAG system assessment.

This module provides evaluation harnesses for project benchmarks
(RAGTruth, CiteBench).

Modules:
    ragtruth_evaluator: Evaluation harness for RAGTruth hallucination benchmark
"""

from .ragtruth_evaluator import RAGTruthEvaluator
from .composite_scorer import CompositeScorer

class RagasEvaluator:
    """Compatibility stub for removed Ragas integration."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Ragas integration has been removed from this repository. "
            "Use RAGTruth/CiteBench evaluation workflows instead."
        )


__all__ = ['RagasEvaluator', 'RAGTruthEvaluator', 'CompositeScorer']
