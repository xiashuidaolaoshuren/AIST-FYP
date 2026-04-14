import logging
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

if "sentence_transformers" not in sys.modules:
    sentence_transformers_stub = types.ModuleType("sentence_transformers")

    class _SentenceTransformerStub:
        pass

    sentence_transformers_stub.SentenceTransformer = _SentenceTransformerStub
    sys.modules["sentence_transformers"] = sentence_transformers_stub

from src.ui.controlled_ui import ControlledPipelineUI
from src.utils.data_structures import Claim, ClaimDecision, EvidenceChunk


def _make_ui() -> ControlledPipelineUI:
    ui = ControlledPipelineUI.__new__(ControlledPipelineUI)
    ui.logger = logging.getLogger("test_controlled_ui_state")
    return ui


def test_user_context_evidence_matches_current_context():
    ui = _make_ui()
    context = (
        "Ankara is the capital of Turkey. "
        "Istanbul is Turkey's largest and most populous city."
    )
    evidence_chunks = [
        EvidenceChunk(
            doc_id="user_context",
            sent_id=0,
            text="Ankara is the capital of Turkey.",
            char_start=0,
            char_end=33,
            score_dense=0.9,
            rank=1,
            source="user_context",
            version="user_context_v1",
        ),
        EvidenceChunk(
            doc_id="user_context",
            sent_id=1,
            text="Istanbul is Turkey's largest and most populous city.",
            char_start=34,
            char_end=87,
            score_dense=0.8,
            rank=2,
            source="user_context",
            version="user_context_v1",
        ),
    ]

    assert ui._user_context_evidence_matches_context(evidence_chunks, context)


def test_user_context_evidence_detects_stale_bundle_content():
    ui = _make_ui()
    current_context = "Ankara is the capital of Turkey. Istanbul is Turkey's largest city."
    stale_chunks = [
        EvidenceChunk(
            doc_id="user_context",
            sent_id=0,
            text="Istanbul is the capital of Turkey, and has been since the founding of the modern state.",
            char_start=0,
            char_end=87,
            score_dense=0.7,
            rank=1,
            source="user_context",
            version="user_context_v1",
        )
    ]

    assert not ui._user_context_evidence_matches_context(stale_chunks, current_context)


def test_should_rebuild_verify_bundle_when_context_changes():
    ui = _make_ui()
    bundle = {
        "query": "What is the capital of Turkey?",
        "user_context_text": "Old context sentence.",
        "source_mode": "user_context",
        "evidence_chunks": [
            EvidenceChunk(
                doc_id="user_context",
                sent_id=0,
                text="Old context sentence.",
                char_start=0,
                char_end=21,
                score_dense=0.9,
                rank=1,
                source="user_context",
                version="user_context_v1",
            )
        ],
    }

    assert ui._should_rebuild_verify_bundle(
        bundle,
        query_input="What is the capital of Turkey?",
        user_context_input="New context sentence.",
    )


def test_should_rebuild_verify_bundle_when_evidence_not_in_current_context():
    ui = _make_ui()
    bundle = {
        "query": "What is the capital of Turkey?",
        "user_context_text": "Ankara is the capital of Turkey.",
        "source_mode": "user_context",
        "evidence_chunks": [
            EvidenceChunk(
                doc_id="user_context",
                sent_id=0,
                text="Istanbul is the capital of Turkey.",
                char_start=0,
                char_end=35,
                score_dense=0.9,
                rank=1,
                source="user_context",
                version="user_context_v1",
            )
        ],
    }

    assert ui._should_rebuild_verify_bundle(
        bundle,
        query_input="What is the capital of Turkey?",
        user_context_input="Ankara is the capital of Turkey.",
    )


def test_should_not_rebuild_when_bundle_matches_current_inputs():
    ui = _make_ui()
    context = "Ankara is the capital of Turkey. Istanbul is Turkey's largest city."
    bundle = {
        "query": "What is the capital of Turkey?",
        "user_context_text": context,
        "source_mode": "user_context",
        "evidence_chunks": [
            EvidenceChunk(
                doc_id="user_context",
                sent_id=0,
                text="Ankara is the capital of Turkey.",
                char_start=0,
                char_end=33,
                score_dense=0.9,
                rank=1,
                source="user_context",
                version="user_context_v1",
            )
        ],
    }

    assert not ui._should_rebuild_verify_bundle(
        bundle,
        query_input="What is the capital of Turkey?",
        user_context_input=context,
    )


def test_view_text_preserves_char_positions_for_placeholder_extraction():
    """view_text substitution must produce same-length whitespace so claim spans are valid."""
    ui = _make_ui()
    from unittest.mock import MagicMock
    from src.mitigation.claim_filter import ClaimFilter
    from src.utils.config import Config

    config = MagicMock(spec=Config)
    config.get.return_value = {'filter': {'enabled': True, 'placeholder': '[CLAIM REMOVED: Contradictory]'}}
    cf = ClaimFilter(config)

    working_answer = "[CLAIM REMOVED: Contradictory] Istanbul is Turkey's largest city."
    view_text = working_answer
    for ph in [cf.placeholder, cf.lc_placeholder]:
        if ph:
            view_text = view_text.replace(ph, ' ' * len(ph))

    assert len(view_text) == len(working_answer)
    assert view_text[31:].startswith("Istanbul")
    assert view_text[:30].strip() == ""


def test_carryover_decisions_preserves_verdict_after_pronoun_substitution():
    ui = _make_ui()

    original_claims = [
        Claim(
            claim_id="c1",
            answer_id="a1",
            text="Istanbul is the capital of Turkey, and has been since the founding of the modern state.",
            answer_char_span=[0, 87],
        ),
        Claim(
            claim_id="c2",
            answer_id="a1",
            text="It is Turkey's largest city.",
            answer_char_span=[88, 115],
        ),
        Claim(
            claim_id="c3",
            answer_id="a1",
            text="Ankara, while a well-known city in Anatolia, is merely the country's second-largest city and plays a primarily regional role.",
            answer_char_span=[116, 233],
        ),
    ]

    original_decisions = [
        ClaimDecision(
            claim_id="c1",
            status="Contradictory",
            rationale="",
            primary_evidence="doc#0",
            signals_ref=[],
            confidence={},
        ),
        ClaimDecision(
            claim_id="c2",
            status="Supported",
            rationale="",
            primary_evidence="doc#1",
            signals_ref=[],
            confidence={},
        ),
        ClaimDecision(
            claim_id="c3",
            status="Low Confidence",
            rationale="",
            primary_evidence="doc#2",
            signals_ref=[],
            confidence={},
        ),
    ]

    filtered_claims = [
        Claim(
            claim_id="f1",
            answer_id="a1",
            text="Istanbul is Turkey's largest city.",
            answer_char_span=[31, 63],
        ),
        Claim(
            claim_id="f2",
            answer_id="a1",
            text="Ankara, while a well-known city in Anatolia, is merely the country's second-largest city and plays a primarily regional role.",
            answer_char_span=[64, 181],
        ),
    ]

    carried = ui._carryover_decisions_after_filter(
        original_claims=original_claims,
        original_decisions=original_decisions,
        filtered_claims=filtered_claims,
    )

    assert len(carried) == 2
    status_by_id = {decision.claim_id: decision.status for decision in carried}
    assert status_by_id["f1"] == "Supported"
    assert status_by_id["f2"] == "Low Confidence"


def test_carryover_decisions_unmatched_claim_falls_back_to_low_confidence():
    ui = _make_ui()

    original_claims = [
        Claim(
            claim_id="c1",
            answer_id="a1",
            text="Ankara is the capital of Turkey.",
            answer_char_span=[0, 32],
        )
    ]
    original_decisions = [
        ClaimDecision(
            claim_id="c1",
            status="Supported",
            rationale="",
            primary_evidence="doc#0",
            signals_ref=[],
            confidence={},
        )
    ]
    filtered_claims = [
        Claim(
            claim_id="f1",
            answer_id="a1",
            text="Completely unrelated sentence.",
            answer_char_span=[0, 30],
        )
    ]

    carried = ui._carryover_decisions_after_filter(
        original_claims=original_claims,
        original_decisions=original_decisions,
        filtered_claims=filtered_claims,
    )

    assert len(carried) == 1
    assert carried[0].claim_id == "f1"
    assert carried[0].status == "Low Confidence"
    assert "fallback" in carried[0].rationale.lower()
