"""
Controlled Gradio UI for step-by-step full pipeline demo workflow.

This UI separates generation and verification into explicit stages:
1) Generate draft answer (from Wikipedia retrieval or user-provided context)
2) Edit draft answer manually
3) Verify answer and visualize claim-level confidence
4) Optionally apply mitigation strategies in a mitigation lab panel
"""

from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Any, Dict, List, Optional, Tuple

import faiss
import gradio as gr
import numpy as np
import pandas as pd

from src.generation.claim_extractor import extract_claims
from src.mitigation.claim_filter import ClaimFilter
from src.mitigation.re_ranker import EvidenceReRanker
from src.mitigation.reprompt import RePrompter
from src.pipelines.baseline_rag import BaselineRAGPipeline
from src.ui.confidence_ui import ConfidenceUI
from src.utils.data_structures import Claim, ClaimDecision, ClaimEvidencePair, EvidenceChunk, VerifierSignal
from src.verification.rule_based_aggregator import RuleBasedAggregator
from src.verification.verifier_hub import VerifierHub


class ControlledPipelineUI(ConfidenceUI):
    """Stepwise Gradio UI for controlled generation/verification demos."""

    def __init__(
        self,
        rag_pipeline: BaselineRAGPipeline,
        verifier_hub: VerifierHub,
        aggregator: RuleBasedAggregator,
        repromptr: Optional[RePrompter] = None,
    ):
        super().__init__(rag_pipeline, verifier_hub, aggregator, repromptr=repromptr)

        self.claim_filter: Optional[ClaimFilter] = None
        self.evidence_reranker: Optional[EvidenceReRanker] = None
        if rag_pipeline.config is not None:
            try:
                self.claim_filter = ClaimFilter(rag_pipeline.config)
            except Exception as exc:
                self.logger.warning("ClaimFilter initialization failed in controlled UI: %s", exc)
            try:
                self.evidence_reranker = EvidenceReRanker(rag_pipeline.config)
            except Exception as exc:
                self.logger.warning("EvidenceReRanker initialization failed in controlled UI: %s", exc)

    def _split_user_context_sentences(self, context_text: str) -> List[Tuple[str, int, int]]:
        """Split free-form context into sentence-like chunks with char spans."""
        matches = list(re.finditer(r"[^.!?\n]+[.!?]?", context_text))
        sentences: List[Tuple[str, int, int]] = []
        for m in matches:
            raw = m.group(0)
            text = raw.strip()
            if not text:
                continue
            start = m.start() + (len(raw) - len(raw.lstrip()))
            end = start + len(text)
            sentences.append((text, start, end))
        return sentences

    def _get_user_context_encoder(self):
        """Resolve a sentence encoder for user-context retrieval across retriever modes."""
        retriever = self.rag_pipeline.retriever

        # Dense mode exposes encoder directly.
        encoder = getattr(retriever, "encoder", None)
        if encoder is not None:
            return encoder

        # Hybrid mode stores dense retriever separately.
        dense_retriever = getattr(retriever, "dense_retriever", None)
        encoder = getattr(dense_retriever, "encoder", None)
        if encoder is not None:
            return encoder

        raise AttributeError(
            "Retriever does not expose a sentence encoder for user-context retrieval. "
            f"Retriever type: {type(retriever).__name__}"
        )

    def _build_user_context_evidence(self, query: str, context_text: str, top_k: int = 5) -> List[EvidenceChunk]:
        """
        Build an in-memory dense index for user context and retrieve top-k chunks.

        This explicitly performs embeddings + FAISS indexing for user-provided
        context, then retrieves the highest-scoring context sentences for the query.
        """
        sentences = self._split_user_context_sentences(context_text)
        if not sentences:
            return []

        encoder = self._get_user_context_encoder()
        sentence_texts = [item[0] for item in sentences]

        sentence_embeddings = encoder.encode(
            sentence_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        sentence_embeddings = sentence_embeddings.astype(np.float32)

        index = faiss.IndexFlatIP(sentence_embeddings.shape[1])
        index.add(sentence_embeddings)

        query_embedding = encoder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        actual_k = min(max(1, top_k), len(sentence_texts))
        scores, indices = index.search(query_embedding, actual_k)

        retrieved: List[EvidenceChunk] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            sent_text, char_start, char_end = sentences[int(idx)]
            retrieved.append(
                EvidenceChunk(
                    doc_id="user_context",
                    sent_id=int(idx),
                    text=sent_text,
                    char_start=int(char_start),
                    char_end=int(char_end),
                    score_dense=float(score),
                    rank=rank,
                    source="user_context",
                    version="user_context_v1",
                )
            )

        return retrieved

    def _coerce_evidence_chunks(self, evidence_chunks: Any) -> List[EvidenceChunk]:
        """Coerce serialized or in-memory evidence chunks into dataclass instances."""
        if not evidence_chunks:
            return []

        coerced: List[EvidenceChunk] = []
        for chunk in evidence_chunks:
            if isinstance(chunk, EvidenceChunk):
                coerced.append(chunk)
            elif isinstance(chunk, dict):
                coerced.append(EvidenceChunk(**chunk))
        return coerced

    @staticmethod
    def _normalize_ui_text(value: Optional[str]) -> str:
        return (value or "").strip()

    def _user_context_evidence_matches_context(
        self,
        evidence_chunks: List[EvidenceChunk],
        context_text: str,
    ) -> bool:
        """Return True when all user-context evidence texts are substrings of the current context."""
        normalized_context = self._normalize_ui_text(context_text)
        if not normalized_context:
            return not any(chunk.doc_id == "user_context" for chunk in evidence_chunks)

        for chunk in evidence_chunks:
            if chunk.doc_id != "user_context":
                continue
            if self._normalize_ui_text(chunk.text) not in normalized_context:
                return False
        return True

    def _build_verify_bundle(
        self,
        query: str,
        answer_text: str,
        user_context: str,
    ) -> Dict[str, Any]:
        """Build fresh verification inputs from the current UI state."""
        query_clean = self._normalize_ui_text(query)
        context_clean = self._normalize_ui_text(user_context)

        if context_clean:
            evidence_chunks = self._build_user_context_evidence(query_clean, context_clean, top_k=5)
            source_mode = "user_context"
        else:
            retrieval_result = self.rag_pipeline.run(query_clean, top_k=5)
            evidence_chunks = self._extract_evidence_chunks(retrieval_result.get("claim_evidence_pairs", []))
            source_mode = "wikipedia"

        claims = extract_claims(text=answer_text, method="auto")
        metadata = self.rag_pipeline.generator.score_target_with_metadata(
            prompt=query_clean,
            target_text=answer_text,
            evidence_chunks=evidence_chunks,
        )
        metadata.update({
            "text": answer_text,
            "original_query": query_clean,
            "source_mode": source_mode,
        })
        sub_answers = [{
            "text": answer_text,
            "char_span": [0, len(answer_text)],
            "sub_answer_id": 0,
            "sub_query": query_clean,
        }]
        claims_by_sub_answer = [{
            "sub_answer_id": 0,
            "sub_text": answer_text,
            "sub_query": query_clean,
            "claims": claims,
        }]
        claim_evidence_pairs = self._build_pairs_for_claims(claims, evidence_chunks)

        return {
            "query": query_clean,
            "user_context_text": context_clean,
            "source_mode": source_mode,
            "generated_text": answer_text,
            "claims": claims,
            "evidence_chunks": evidence_chunks,
            "claim_evidence_pairs": claim_evidence_pairs,
            "generator_metadata": metadata,
            "sub_answers": sub_answers,
            "claims_by_sub_answer": claims_by_sub_answer,
        }

    def _should_rebuild_verify_bundle(
        self,
        bundle: Dict[str, Any],
        query_input: str,
        user_context_input: str,
    ) -> bool:
        """Determine whether verify must rebuild state from current UI inputs."""
        if not bundle or bundle.get("error"):
            return True

        bundle_query = self._normalize_ui_text(bundle.get("query"))
        current_query = self._normalize_ui_text(query_input)
        if current_query and current_query != bundle_query:
            return True

        bundle_context = self._normalize_ui_text(bundle.get("user_context_text"))
        current_context = self._normalize_ui_text(user_context_input)
        bundle_source_mode = bundle.get("source_mode", "wikipedia")

        if current_context:
            if bundle_source_mode != "user_context":
                return True
            if current_context != bundle_context:
                return True
            evidence_chunks = self._coerce_evidence_chunks(bundle.get("evidence_chunks", []))
            if not self._user_context_evidence_matches_context(evidence_chunks, current_context):
                self.logger.warning("User-context evidence no longer matches current textbox content; rebuilding verify bundle")
                return True
            return False

        return bundle_source_mode == "user_context" and bool(bundle_context)

    def _flatten_claims(self, claims_by_sub_answer: List[Dict[str, Any]]) -> List[Claim]:
        claims: List[Claim] = []
        for item in claims_by_sub_answer:
            claims.extend(item.get("claims", []))
        return claims

    def _extract_evidence_chunks(self, claim_evidence_pairs: List[Dict[str, Any]]) -> List[EvidenceChunk]:
        if not claim_evidence_pairs:
            return []
        evidence_spans = claim_evidence_pairs[0].get("evidence_spans", [])
        return [EvidenceChunk(**span) for span in evidence_spans]

    def _build_pairs_for_claims(
        self,
        claims: List[Claim],
        evidence_chunks: List[EvidenceChunk],
    ) -> List[Dict[str, Any]]:
        evidence_spans = [chunk.to_dict() for chunk in evidence_chunks]
        evidence_candidates = [f"{chunk.doc_id}#{chunk.sent_id}" for chunk in evidence_chunks]
        top_evidence = evidence_candidates[0] if evidence_candidates else ""
        pairs: List[Dict[str, Any]] = []
        for claim in claims:
            pairs.append(
                ClaimEvidencePair(
                    claim_id=claim.claim_id,
                    evidence_candidates=evidence_candidates,
                    top_evidence=top_evidence,
                    evidence_spans=evidence_spans,
                ).to_dict()
            )
        return pairs

    def _verify_from_bundle(
        self,
        query: str,
        answer_text: str,
        claims: List[Claim],
        evidence_chunks: List[EvidenceChunk],
        metadata: Dict[str, Any],
    ) -> Tuple[List[VerifierSignal], List[ClaimDecision]]:
        batch_records = [
            {
                "claim": claim,
                "evidence": evidence_chunks,
                "metadata": metadata,
            }
            for claim in claims
        ]

        signals: List[VerifierSignal] = []
        decisions: List[ClaimDecision] = []
        batch_signals = self.verifier_hub.verify_claims_batch(batch_records)
        for signal in batch_signals:
            if signal is None:
                continue
            signals.append(signal)
            decisions.append(self.aggregator.aggregate(signal))

        self.logger.info(
            "Controlled verify complete: query=%s claims=%d decisions=%d",
            query[:80],
            len(claims),
            len(decisions),
        )
        return signals, decisions

    @staticmethod
    def _decision_stats(decisions: List[ClaimDecision]) -> Dict[str, Any]:
        total = len(decisions)
        contradictory = len([d for d in decisions if d.status == "Contradictory"])
        low_conf = len([d for d in decisions if d.status == "Low Confidence"])
        supported = len([d for d in decisions if d.status == "Supported"])
        rate = (contradictory / total) if total else 0.0
        return {
            "total": total,
            "supported": supported,
            "low_conf": low_conf,
            "contradictory": contradictory,
            "hallucination_rate": rate,
        }

    @staticmethod
    def _normalize_claim_text_for_match(text: str) -> str:
        normalized = (text or "").strip().lower()
        normalized = re.sub(r"[^\w\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @staticmethod
    def _drop_leading_token(text: str) -> str:
        parts = text.split(" ", 1)
        return parts[1] if len(parts) > 1 else ""

    def _claim_match_score(self, source_claim: Claim, target_claim: Claim) -> float:
        source_text = self._normalize_claim_text_for_match(source_claim.text)
        target_text = self._normalize_claim_text_for_match(target_claim.text)
        if not source_text or not target_text:
            return 0.0

        full_ratio = SequenceMatcher(None, source_text, target_text).ratio()
        source_tail = self._drop_leading_token(source_text)
        target_tail = self._drop_leading_token(target_text)
        tail_ratio = 0.0
        if source_tail and target_tail:
            tail_ratio = SequenceMatcher(None, source_tail, target_tail).ratio()

        source_start, source_end = source_claim.answer_char_span
        target_start, target_end = target_claim.answer_char_span
        overlap = max(0, min(source_end, target_end) - max(source_start, target_start))
        source_len = max(1, source_end - source_start)
        target_len = max(1, target_end - target_start)
        overlap_ratio = overlap / max(source_len, target_len)

        text_similarity = max(full_ratio, tail_ratio)
        # Keep overlap as a weak tie-breaker only; do not let span alignment
        # alone map semantically unrelated claims.
        return (0.8 * text_similarity) + (0.2 * overlap_ratio)

    def _carryover_decisions_after_filter(
        self,
        original_claims: List[Claim],
        original_decisions: List[ClaimDecision],
        filtered_claims: List[Claim],
    ) -> List[ClaimDecision]:
        """Map post-filter claims to pre-filter claims and carry verdicts forward."""
        decision_map = {decision.claim_id: decision for decision in original_decisions}
        eligible_original_claims = [
            claim
            for claim in original_claims
            if claim.claim_id in decision_map
            and decision_map[claim.claim_id].status != "Contradictory"
        ]

        used_source_ids: set[str] = set()
        carried: List[ClaimDecision] = []
        mapped_count = 0
        unmatched_count = 0

        for filtered_claim in filtered_claims:
            best_claim: Optional[Claim] = None
            best_score = 0.0

            for source_claim in eligible_original_claims:
                if source_claim.claim_id in used_source_ids:
                    continue
                score = self._claim_match_score(source_claim, filtered_claim)
                if score > best_score:
                    best_score = score
                    best_claim = source_claim

            if best_claim is not None and best_score >= 0.55:
                source_decision = decision_map[best_claim.claim_id]
                carried.append(
                    ClaimDecision(
                        claim_id=filtered_claim.claim_id,
                        status=source_decision.status,
                        rationale=source_decision.rationale,
                        primary_evidence=source_decision.primary_evidence,
                        signals_ref=list(source_decision.signals_ref),
                        confidence=dict(source_decision.confidence),
                    )
                )
                used_source_ids.add(best_claim.claim_id)
                mapped_count += 1
                continue

            unmatched_count += 1
            carried.append(
                ClaimDecision(
                    claim_id=filtered_claim.claim_id,
                    status="Low Confidence",
                    rationale="Unmapped claim after mitigation rewrite; verdict fallback applied.",
                    primary_evidence="",
                    signals_ref=[],
                    confidence={
                        "overall_confidence": 50.0,
                        "band": "Medium",
                        "support_prob": 0.0,
                        "contradict_prob": 0.0,
                        "coverage_score": 0.0,
                        "entropy_conf": 0.0,
                    },
                )
            )

        self.logger.info(
            "[mitigate] carried verdicts: mapped=%d unmatched=%d total=%d",
            mapped_count,
            unmatched_count,
            len(filtered_claims),
        )
        return carried

    def create_interface(self) -> gr.Blocks:
        """Create a staged Blocks UI for generate -> verify -> mitigate."""

        def handle_generate(query: str, user_context: str):
            if not (query or "").strip():
                return (
                    "",
                    {"error": "Query is required."},
                    [],
                    pd.DataFrame(),
                )

            try:
                query_clean = query.strip()
                context_clean = (user_context or "").strip()

                if context_clean:
                    evidence_chunks = self._build_user_context_evidence(query_clean, context_clean, top_k=5)
                    result = self.rag_pipeline.generate_from_evidence(
                        query=query_clean,
                        evidence_chunks=evidence_chunks,
                        top_k=min(5, max(1, len(evidence_chunks))),
                    )
                    source_mode = "user_context"
                else:
                    result = self.rag_pipeline.run(query_clean, top_k=5)
                    evidence_chunks = self._extract_evidence_chunks(result.get("claim_evidence_pairs", []))
                    source_mode = "wikipedia"

                draft_text = result.get("draft_response", "")
                claims_by_sub_answer = result.get("claims_by_sub_answer", [])
                claims = self._flatten_claims(claims_by_sub_answer)
                claim_evidence_pairs = result.get("claim_evidence_pairs", [])
                if not claim_evidence_pairs:
                    claim_evidence_pairs = self._build_pairs_for_claims(claims, evidence_chunks)

                bundle = {
                    "query": query_clean,
                    "user_context_text": context_clean,
                    "source_mode": source_mode,
                    "generated_text": draft_text,
                    "claims": claims,
                    "evidence_chunks": evidence_chunks,
                    "claim_evidence_pairs": claim_evidence_pairs,
                    "generator_metadata": {
                        **result.get("generator_metadata", {}),
                        "source_mode": source_mode,
                    },
                    "sub_answers": result.get("sub_answers", [{"text": draft_text, "char_span": [0, len(draft_text)], "sub_answer_id": 0, "sub_query": query_clean}]),
                    "claims_by_sub_answer": claims_by_sub_answer or [{
                        "sub_answer_id": 0,
                        "sub_text": draft_text,
                        "sub_query": query_clean,
                        "claims": claims,
                    }],
                }

                evidence_df = self._build_evidence_dataframe(claims, claim_evidence_pairs)
                status = f"Generated draft from {source_mode} with {len(evidence_chunks)} evidence chunks and {len(claims)} claims."
                return draft_text, bundle, status, evidence_df
            except Exception as exc:
                self.logger.error("Generate stage failed: %s", exc, exc_info=True)
                return "", {"error": str(exc)}, f"Generate failed: {exc}", pd.DataFrame(columns=self.evidence_columns)

        def handle_verify(
            draft_text: str,
            bundle: Dict[str, Any],
            query_input: str,
            user_context_input: str,
        ):
            empty_highlight: List[Tuple[str, Optional[str]]] = []
            text = (draft_text or "").strip()
            if not text:
                return empty_highlight, pd.DataFrame(), pd.DataFrame(), {}, "Draft answer is empty."

            try:
                print("[verify] callback start", flush=True)
                bundle = bundle if isinstance(bundle, dict) else {}
                if self._should_rebuild_verify_bundle(bundle, query_input, user_context_input):
                    query_clean = self._normalize_ui_text(query_input) or self._normalize_ui_text(bundle.get("query"))
                    context_clean = self._normalize_ui_text(user_context_input)
                    if not query_clean:
                        msg = bundle.get("error", "Query is required before verification.") if bundle else "Query is required before verification."
                        return empty_highlight, pd.DataFrame(), pd.DataFrame(), {}, msg
                    bundle = self._build_verify_bundle(query=query_clean, answer_text=text, user_context=context_clean)

                query = self._normalize_ui_text(query_input) or bundle.get("query", "")
                generated_text = bundle.get("generated_text", "")
                evidence_chunks = self._coerce_evidence_chunks(bundle.get("evidence_chunks", []))
                metadata = dict(bundle.get("generator_metadata", {}))
                metadata.setdefault("source_mode", bundle.get("source_mode", "wikipedia"))

                edited = text != generated_text
                if edited:
                    claims = extract_claims(text=text, method="auto")
                    score_meta = self.rag_pipeline.generator.score_target_with_metadata(
                        prompt=query,
                        target_text=text,
                        evidence_chunks=evidence_chunks,
                    )
                    metadata = {
                        **metadata,
                        **score_meta,
                        "text": text,
                        "original_query": query,
                        "source_mode": bundle.get("source_mode", metadata.get("source_mode", "wikipedia")),
                    }
                    sub_answers = [{
                        "text": text,
                        "char_span": [0, len(text)],
                        "sub_answer_id": 0,
                        "sub_query": query,
                    }]
                    claims_by_sub_answer = [{
                        "sub_answer_id": 0,
                        "sub_text": text,
                        "sub_query": query,
                        "claims": claims,
                    }]
                    claim_evidence_pairs = self._build_pairs_for_claims(claims, evidence_chunks)
                else:
                    claims = bundle.get("claims", [])
                    sub_answers = bundle.get("sub_answers", [])
                    claims_by_sub_answer = bundle.get("claims_by_sub_answer", [])
                    claim_evidence_pairs = bundle.get("claim_evidence_pairs", [])
                    metadata.setdefault("text", text)
                    metadata.setdefault("original_query", query)

                if metadata.get("source_mode") == "user_context":
                    current_context = self._normalize_ui_text(user_context_input)
                    if not self._user_context_evidence_matches_context(evidence_chunks, current_context):
                        self.logger.warning("Detected stale user-context evidence at verify time; rebuilding from current context")
                        bundle = self._build_verify_bundle(query=query, answer_text=text, user_context=current_context)
                        evidence_chunks = self._coerce_evidence_chunks(bundle.get("evidence_chunks", []))
                        claims = bundle.get("claims", [])
                        sub_answers = bundle.get("sub_answers", [])
                        claims_by_sub_answer = bundle.get("claims_by_sub_answer", [])
                        claim_evidence_pairs = bundle.get("claim_evidence_pairs", [])
                        metadata = dict(bundle.get("generator_metadata", {}))

                signals, decisions = self._verify_from_bundle(
                    query=query,
                    answer_text=text,
                    claims=claims,
                    evidence_chunks=evidence_chunks,
                    metadata=metadata,
                )

                if not decisions:
                    return [(text, None)], pd.DataFrame(), self._build_evidence_dataframe(claims, claim_evidence_pairs), {}, "No verification decisions produced."

                highlighted = self._build_highlighted_output_with_headers(
                    answer_text=text,
                    sub_answers=sub_answers,
                    claims_by_sub_answer=claims_by_sub_answer,
                    decisions=decisions,
                )
                details_df = self._build_details_table(claims, decisions)
                evidence_df = self._build_evidence_dataframe(claims, claim_evidence_pairs)

                verify_state = {
                    "query": query,
                    "answer_text": text,
                    "claims": claims,
                    "evidence_chunks": evidence_chunks,
                    "claim_evidence_pairs": claim_evidence_pairs,
                    "signals": signals,
                    "decisions": decisions,
                    "sub_answers": sub_answers,
                    "claims_by_sub_answer": claims_by_sub_answer,
                    "metadata": metadata,
                }
                stats = self._decision_stats(decisions)
                status = (
                    f"Verified {stats['total']} claims: "
                    f"Supported={stats['supported']}, "
                    f"Low Confidence={stats['low_conf']}, "
                    f"Contradictory={stats['contradictory']}."
                )
                self.logger.info("[verify] %s", status)
                print(f"[verify] {status}", flush=True)
                return highlighted, details_df, evidence_df, verify_state, status
            except Exception as exc:
                self.logger.error("Verify stage failed: %s", exc, exc_info=True)
                print(f"[verify] failed: {exc}", flush=True)
                return empty_highlight, pd.DataFrame(), pd.DataFrame(), {}, f"Verify failed: {exc}"

        def handle_mitigate(
            enable_filter: bool,
            enable_reprompt: bool,
            enable_rerank: bool,
            verify_state: Dict[str, Any],
        ):
            empty = []
            if not verify_state or not verify_state.get("decisions"):
                return empty, empty, "Run verification before mitigation.", pd.DataFrame(), {}

            try:
                print("[mitigate] callback start", flush=True)
                query = verify_state.get("query", "")
                base_answer = verify_state.get("answer_text", "")
                claims: List[Claim] = verify_state.get("claims", [])
                decisions: List[ClaimDecision] = verify_state.get("decisions", [])
                evidence_chunks: List[EvidenceChunk] = verify_state.get("evidence_chunks", [])
                signals: List[VerifierSignal] = verify_state.get("signals", [])

                before_highlight = self._build_highlighted_output(base_answer, claims, decisions)
                before_stats = self._decision_stats(decisions)

                working_answer = base_answer
                working_claims = list(claims)
                working_evidence = list(evidence_chunks)
                working_decisions = list(decisions)
                working_metadata = dict(verify_state.get("metadata", {}))
                working_metadata.setdefault("text", working_answer)
                working_metadata.setdefault("original_query", query)

                if enable_rerank and self.evidence_reranker and signals:
                    signal_map = {
                        f"{signal.doc_id}#{signal.sent_id}": signal
                        for signal in signals
                    }
                    working_evidence = self.evidence_reranker.rerank(working_evidence, signal_map)

                if enable_filter and self.claim_filter:
                    filtered_text, _, _ = self.claim_filter.filter_answer(
                        answer_text=working_answer,
                        claims=working_claims,
                        decisions=working_decisions,
                    )
                    working_answer = filtered_text
                    pre_filter_claims = list(working_claims)
                    pre_filter_decisions = list(working_decisions)
                    working_claims = extract_claims(text=working_answer, method="auto")
                    working_claims = self.claim_filter.filter_placeholder_claims(
                        working_claims,
                        working_answer,
                    )
                    working_metadata = self.rag_pipeline.generator.score_target_with_metadata(
                        prompt=query,
                        target_text=working_answer,
                        evidence_chunks=working_evidence,
                    )
                    working_metadata["text"] = working_answer
                    working_metadata["original_query"] = query
                    working_decisions = self._carryover_decisions_after_filter(
                        original_claims=pre_filter_claims,
                        original_decisions=pre_filter_decisions,
                        filtered_claims=working_claims,
                    )

                if enable_reprompt and self.repromptr and self.repromptr.enabled:
                    reprompt_result = self.repromptr.reprompt(
                        query=query,
                        answer=working_answer,
                        decisions=working_decisions,
                        evidence=working_evidence,
                        claims=working_claims,
                    )
                    if reprompt_result.get("improved"):
                        working_answer = reprompt_result.get("final_answer", working_answer)
                        working_claims = extract_claims(text=working_answer, method="auto")
                        if self.claim_filter:
                            working_claims = self.claim_filter.filter_placeholder_claims(
                                working_claims,
                                working_answer,
                            )
                        working_metadata = self.rag_pipeline.generator.score_target_with_metadata(
                            prompt=query,
                            target_text=working_answer,
                            evidence_chunks=working_evidence,
                        )
                        working_metadata["text"] = working_answer
                        working_metadata["original_query"] = query
                        _, working_decisions = self._verify_from_bundle(
                            query=query,
                            answer_text=working_answer,
                            claims=working_claims,
                            evidence_chunks=working_evidence,
                            metadata=working_metadata,
                        )

                after_highlight = self._build_highlighted_output(working_answer, working_claims, working_decisions)
                after_stats = self._decision_stats(working_decisions)
                stats_md = (
                    "Before/After Mitigation\n\n"
                    f"- Contradictory: {before_stats['contradictory']} -> {after_stats['contradictory']}\n"
                    f"- Low Confidence: {before_stats['low_conf']} -> {after_stats['low_conf']}\n"
                    f"- Hallucination Rate: {before_stats['hallucination_rate']:.1%} -> {after_stats['hallucination_rate']:.1%}"
                )

                claim_evidence_pairs = self._build_pairs_for_claims(working_claims, working_evidence)
                evidence_df = self._build_evidence_dataframe(working_claims, claim_evidence_pairs)

                updated_state = {
                    "query": query,
                    "answer_text": working_answer,
                    "claims": working_claims,
                    "evidence_chunks": working_evidence,
                    "claim_evidence_pairs": claim_evidence_pairs,
                    "signals": [],
                    "decisions": working_decisions,
                    "metadata": working_metadata,
                }
                self.logger.info(
                    "[mitigate] before: Supported=%d Low=%d Contradictory=%d | after: Supported=%d Low=%d Contradictory=%d",
                    before_stats['supported'],
                    before_stats['low_conf'],
                    before_stats['contradictory'],
                    after_stats['supported'],
                    after_stats['low_conf'],
                    after_stats['contradictory'],
                )
                print(
                    "[mitigate] "
                    f"before(S={before_stats['supported']},L={before_stats['low_conf']},C={before_stats['contradictory']}) "
                    f"after(S={after_stats['supported']},L={after_stats['low_conf']},C={after_stats['contradictory']})",
                    flush=True,
                )
                return before_highlight, after_highlight, stats_md, evidence_df, updated_state
            except Exception as exc:
                self.logger.error("Mitigation stage failed: %s", exc, exc_info=True)
                print(f"[mitigate] failed: {exc}", flush=True)
                return empty, empty, f"Mitigation failed: {exc}", pd.DataFrame(), verify_state

        with gr.Blocks(title="Controlled Hallucination Detection Demo") as demo:
            gr.Markdown(
                "# Controlled Full Pipeline Demo\n"
                "Stage 1: Generate draft answer. Stage 2: Edit draft. Stage 3: Verify and inspect signals."
            )

            generator = getattr(self.rag_pipeline, "generator", None)
            if generator is not None and getattr(generator, "model_family", None) == "seq2seq":
                model_name = getattr(generator, "model_name", "unknown")
                gr.Markdown(
                    (
                        "<div style='border:1px solid #f59e0b; background:#fff7ed; color:#9a3412; "
                        "padding:10px 12px; border-radius:8px; margin-bottom:8px;'>"
                        "<strong>Generator Warning:</strong> "
                        f"Current model <code>{model_name}</code> is seq2seq. "
                        "It may ignore chat-style system instructions and produce very short answers "
                        "(for example single tokens). For reprompt testing, use an instruction-tuned "
                        "causal model such as Qwen3."
                        "</div>"
                    )
                )

            generation_state = gr.State({})
            verification_state = gr.State({})

            with gr.Row():
                with gr.Column(scale=1):
                    query_box = gr.Textbox(
                        label="Query",
                        placeholder="Ask a question...",
                        lines=2,
                    )
                    context_box = gr.Textbox(
                        label="Optional User Context for Verification",
                        placeholder="Paste custom context here. Leave empty to use Wikipedia corpus.",
                        lines=8,
                    )
                    generate_btn = gr.Button("Generate Draft Answer", variant="primary")
                    draft_box = gr.Textbox(
                        label="Generated Answer Draft (Editable)",
                        placeholder="Generated answer will appear here. You can edit before verification.",
                        lines=8,
                        interactive=True,
                    )
                    verify_btn = gr.Button("Verify Draft Answer", variant="secondary")
                    status_box = gr.Textbox(label="Status", interactive=False)

                with gr.Column(scale=1):
                    highlighted_output = gr.HighlightedText(
                        label="Answer with Confidence Highlighting",
                        combine_adjacent=False,
                        show_legend=True,
                        color_map=self.color_map,
                    )
                    details_df = gr.Dataframe(label="Claim-Level Details", wrap=True)
                    evidence_df = gr.Dataframe(label="Evidence (Per-Claim Grouped View)", wrap=True)

            with gr.Accordion("Mitigation Lab", open=False):
                with gr.Row():
                    cb_filter = gr.Checkbox(label="Apply Claim Filter", value=True)
                    cb_reprompt = gr.Checkbox(label="Apply Re-prompt", value=False)
                    cb_rerank = gr.Checkbox(label="Apply Evidence Re-rank", value=True)
                    mitigate_btn = gr.Button("Apply Mitigation")

                with gr.Row():
                    before_highlighted = gr.HighlightedText(
                        label="Before Mitigation",
                        combine_adjacent=False,
                        show_legend=True,
                        color_map=self.color_map,
                    )
                    after_highlighted = gr.HighlightedText(
                        label="After Mitigation",
                        combine_adjacent=False,
                        show_legend=True,
                        color_map=self.color_map,
                    )

                mitigation_stats = gr.Markdown("Run verification first, then apply mitigation.")

            generate_btn.click(
                fn=handle_generate,
                inputs=[query_box, context_box],
                outputs=[draft_box, generation_state, status_box, evidence_df],
            )

            verify_btn.click(
                fn=handle_verify,
                inputs=[draft_box, generation_state, query_box, context_box],
                outputs=[highlighted_output, details_df, evidence_df, verification_state, status_box],
            )

            mitigate_btn.click(
                fn=handle_mitigate,
                inputs=[cb_filter, cb_reprompt, cb_rerank, verification_state],
                outputs=[before_highlighted, after_highlighted, mitigation_stats, evidence_df, verification_state],
            )

        return demo
