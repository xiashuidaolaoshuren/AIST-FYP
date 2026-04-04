"""
Month 3-5 Verification Module.

This module implements trainless hallucination detectors and aggregation components
for the AIST-FYP project. It provides multiple detection strategies and a unified
interface for hallucination verification.

Available detectors:
- IntrinsicUncertaintyDetector: Measures model confidence via token-level entropy
- RetrievalGroundedDetector: Measures evidence coverage and citation integrity
- NLIDetector: Zero-shot Natural Language Inference for entailment/contradiction
- SelfAgreementDetector: Measures consistency across stochastic samples

Aggregation components:
- VerifierHub: Central orchestrator for all detectors
- SignalNormalizer: Normalizes heterogeneous signals to [0,1] confidence scale
- RuleBasedAggregator: Applies hierarchical rules to classify claims
"""

from src.verification.intrinsic_uncertainty import IntrinsicUncertaintyDetector
from src.verification.retrieval_grounded import RetrievalGroundedDetector
from src.verification.nli_detector import NLIDetector
from src.verification.lettuce_detector import LettuceDetectDetector
from src.verification.self_agreement import SelfAgreementDetector
from src.verification.verifier_hub import VerifierHub
from src.verification.rule_based_aggregator import SignalNormalizer, RuleBasedAggregator

__all__ = [
    'IntrinsicUncertaintyDetector',
    'RetrievalGroundedDetector',
    'NLIDetector',
    'LettuceDetectDetector',
    'SelfAgreementDetector',
    'VerifierHub',
    'SignalNormalizer',
    'RuleBasedAggregator',
]
