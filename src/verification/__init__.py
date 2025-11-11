"""
Month 3 Verification Module.

This module implements trainless hallucination detectors for the AIST-FYP project.
It provides intrinsic uncertainty detection and retrieval-grounded verification.

Available detectors:
- IntrinsicUncertaintyDetector: Measures model confidence via token entropy
- RetrievalGroundedDetector: Measures evidence coverage and citation integrity
"""

from src.verification.intrinsic_uncertainty import IntrinsicUncertaintyDetector

__all__ = ['IntrinsicUncertaintyDetector']
