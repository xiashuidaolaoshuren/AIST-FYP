"""
Utilities package for the Baseline RAG Module.

This package exports data structures, configuration loader, and logging utilities
used throughout the Month 2 implementation.
"""

from .data_structures import (
    EvidenceChunk,
    Claim,
    ClaimEvidencePair,
    VerifierSignal,
    ClaimDecision,
    AnnotatedAnswer
)
from .config import Config
from .logger import setup_logger, get_logger, set_log_level

__all__ = [
    # Data structures
    'EvidenceChunk',
    'Claim',
    'ClaimEvidencePair',
    'VerifierSignal',
    'ClaimDecision',
    'AnnotatedAnswer',
    # Configuration
    'Config',
    # Logging
    'setup_logger',
    'get_logger',
    'set_log_level',
]
