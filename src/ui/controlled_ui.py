"""
Controlled Gradio UI for step-by-step full pipeline demo workflow.

This UI separates generation and verification into explicit stages:
1) Generate draft answer (from Wikipedia retrieval or user-provided context)
2) Edit draft answer manually
3) Verify answer and visualize claim-level confidence
4) Optionally apply mitigation strategies in a mitigation lab panel
"""

from __future__ import annotations

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
                    "source_mode": source_mode,
                    "generated_text": draft_text,
                    "claims": claims,
                    "evidence_chunks": evidence_chunks,
                    "claim_evidence_pairs": claim_evidence_pairs,
                    "generator_metadata": result.get("generator_metadata", {}),
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
                bundle = bundle if isinstance(bundle, dict) else {}
                if not bundle or bundle.get("error"):
                    query_clean = (query_input or "").strip()
                    context_clean = (user_context_input or "").strip()
                    if not query_clean:
                        msg = bundle.get("error", "Query is required before verification.") if bundle else "Query is required before verification."
                        return empty_highlight, pd.DataFrame(), pd.DataFrame(), {}, msg

                    if context_clean:
                        evidence_chunks = self._build_user_context_evidence(query_clean, context_clean, top_k=5)
                        source_mode = "user_context"
                    else:
                        retrieval_result = self.rag_pipeline.run(query_clean, top_k=5)
                        evidence_chunks = self._extract_evidence_chunks(retrieval_result.get("claim_evidence_pairs", []))
                        source_mode = "wikipedia"

                    claims = extract_claims(text=text, method="auto")
                    metadata = self.rag_pipeline.generator.score_target_with_metadata(
                        prompt=query_clean,
                        target_text=text,
                        evidence_chunks=evidence_chunks,
                    )
                    metadata.update({
                        "text": text,
                        "original_query": query_clean,
                    })
                    sub_answers = [{
                        "text": text,
                        "char_span": [0, len(text)],
                        "sub_answer_id": 0,
                        "sub_query": query_clean,
                    }]
                    claims_by_sub_answer = [{
                        "sub_answer_id": 0,
                        "sub_text": text,
                        "sub_query": query_clean,
                        "claims": claims,
                    }]
                    claim_evidence_pairs = self._build_pairs_for_claims(claims, evidence_chunks)

                    bundle = {
                        "query": query_clean,
                        "source_mode": source_mode,
                        "generated_text": text,
                        "claims": claims,
                        "evidence_chunks": evidence_chunks,
                        "claim_evidence_pairs": claim_evidence_pairs,
                        "generator_metadata": metadata,
                        "sub_answers": sub_answers,
                        "claims_by_sub_answer": claims_by_sub_answer,
                    }

                query = bundle.get("query", "")
                generated_text = bundle.get("generated_text", "")
                evidence_chunks: List[EvidenceChunk] = bundle.get("evidence_chunks", [])
                metadata = dict(bundle.get("generator_metadata", {}))

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
                return highlighted, details_df, evidence_df, verify_state, status
            except Exception as exc:
                self.logger.error("Verify stage failed: %s", exc, exc_info=True)
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
                    working_claims = extract_claims(text=working_answer, method="auto")
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
                return before_highlight, after_highlight, stats_md, evidence_df, updated_state
            except Exception as exc:
                self.logger.error("Mitigation stage failed: %s", exc, exc_info=True)
                return empty, empty, f"Mitigation failed: {exc}", pd.DataFrame(), verify_state

        with gr.Blocks(title="Controlled Hallucination Detection Demo") as demo:
            gr.Markdown(
                "# Controlled Full Pipeline Demo\n"
                "Stage 1: Generate draft answer. Stage 2: Edit draft. Stage 3: Verify and inspect signals."
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
