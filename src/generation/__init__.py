"""
Generation module for the Hallucination Detection System.

This module provides components for LLM text generation with metadata capture
and claim extraction from generated responses.
"""

from .generator_wrapper import GeneratorWrapper
from .claim_extractor import extract_claims, extract_claims_spacy, extract_claims_regex, validate_claim_spans

__all__ = [
    'GeneratorWrapper',
    'extract_claims',
    'extract_claims_spacy',
    'extract_claims_regex',
    'validate_claim_spans'
]
