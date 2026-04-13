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
from src.utils.data_structures import EvidenceChunk


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
