"""Unit tests for VerifierHub self-agreement batch preparation path."""

from unittest.mock import Mock

from src.utils.data_structures import Claim, EvidenceChunk
from src.verification.verifier_hub import VerifierHub


def _build_claim(claim_id: str, text: str) -> Claim:
    return Claim(
        claim_id=claim_id,
        answer_id=f"a_{claim_id}",
        text=text,
        answer_char_span=[0, len(text)],
        extraction_method="auto",
    )


def _build_evidence(doc_id: str, sent_id: int, text: str) -> EvidenceChunk:
    return EvidenceChunk(
        doc_id=doc_id,
        sent_id=sent_id,
        text=text,
        char_start=0,
        char_end=len(text),
        score_dense=0.9,
        rank=1,
    )


def _build_minimal_hub() -> VerifierHub:
    # Build a minimal test double instance without expensive detector init.
    hub = object.__new__(VerifierHub)
    hub.enabled = True
    hub.verify_all_evidence = False
    hub.strict_logits = False
    hub.logger = Mock()
    hub.uncertainty_detector = None
    hub.grounded_detector = None
    hub.nli_detector = object()
    hub.self_agreement_detector = None
    return hub


def test_prepare_verification_collect_nli_uses_sa_detect_batch():
    hub = _build_minimal_hub()

    sa_detector = Mock()
    sa_detector.detect_batch.return_value = [
        {"variance": 0.01, "score": 0.8, "samples_generated": 2},
        {"variance": 0.02, "score": 0.7, "samples_generated": 2},
    ]
    hub.self_agreement_detector = sa_detector

    claim_records = [
        {
            "claim": _build_claim("c1", "Claim one."),
            "evidence": _build_evidence("d1", 0, "Evidence one."),
            "metadata": {"original_query": "Q1"},
        },
        {
            "claim": _build_claim("c2", "Claim two."),
            "evidence": _build_evidence("d2", 1, "Evidence two."),
            "metadata": {"original_query": "Q2"},
        },
    ]

    prepared = hub.prepare_verification_collect_nli(claim_records)

    sa_detector.detect_batch.assert_called_once()
    assert len(prepared.prepared_items) == 2
    assert len(prepared.nli_pending) == 2
    assert prepared.prepared_items[0]["consistency"]["score"] == 0.8
    assert prepared.prepared_items[1]["consistency"]["score"] == 0.7


def test_prepare_verification_collect_nli_falls_back_without_sa_batch_api():
    hub = _build_minimal_hub()

    class LegacySADetector:
        def __init__(self):
            self.detect = Mock(return_value={"variance": 0.03, "score": 0.6, "samples_generated": 2})

    legacy = LegacySADetector()
    hub.self_agreement_detector = legacy

    claim_records = [
        {
            "claim": _build_claim("c1", "Claim one."),
            "evidence": _build_evidence("d1", 0, "Evidence one."),
            "metadata": {"original_query": "Q1"},
        }
    ]

    prepared = hub.prepare_verification_collect_nli(claim_records)

    legacy.detect.assert_called_once()
    assert len(prepared.prepared_items) == 1
    assert prepared.prepared_items[0]["consistency"]["score"] == 0.6
