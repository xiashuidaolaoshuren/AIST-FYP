"""
Core mitigation orchestrator for rerank/reprompt/filter workflows.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.generation.claim_extractor import extract_claims
from src.mitigation.claim_filter import ClaimFilter
from src.mitigation.policy_router import MitigationPolicyRouter
from src.mitigation.re_ranker import EvidenceReRanker
from src.mitigation.reprompt import RePrompter
from src.utils.data_structures import ClaimDecision, EvidenceChunk
from src.utils.logger import setup_logger


class MitigationOrchestrator:
    """Applies objective-aware mitigation as a unified runtime flow."""

    def __init__(self, config, verifier_hub, aggregator, generator=None):
        self.config = config
        self.verifier_hub = verifier_hub
        self.aggregator = aggregator
        self.logger = setup_logger(__name__)

        mitigation_cfg = config.get("mitigation", {})
        if not isinstance(mitigation_cfg, dict):
            mitigation_cfg = {}

        self.enabled = bool(mitigation_cfg.get("enabled", False))
        self.router = MitigationPolicyRouter.from_config(config)

        self.claim_filter = None
        self.evidence_reranker = None
        self.reprompter = None

        if self.enabled:
            try:
                self.claim_filter = ClaimFilter(config)
            except Exception as exc:
                self.logger.warning("Failed to initialize ClaimFilter: %s", exc)
            try:
                self.evidence_reranker = EvidenceReRanker(config)
            except Exception as exc:
                self.logger.warning("Failed to initialize EvidenceReRanker: %s", exc)
            try:
                if generator is not None:
                    self.reprompter = RePrompter(config, generator)
            except Exception as exc:
                self.logger.warning("Failed to initialize RePrompter: %s", exc)

    def apply(
        self,
        *,
        query: str,
        answer_text: str,
        claim_records: List[Dict[str, Any]],
        objective_override: str | None = None,
    ) -> Dict[str, Any]:
        if not self.enabled or not claim_records:
            return {
                "final_answer": answer_text,
                "claim_records": claim_records,
                "decisions": [],
                "signals": [],
                "actions": [],
                "filtered_claim_count": 0,
            }

        signals, decisions = self._verify_and_decide(claim_records)
        actions: List[str] = []

        planned_actions = self.router.resolve_actions(decisions, objective_override)

        if (
            "rerank" in planned_actions
            and self.evidence_reranker
            and self.evidence_reranker.enabled
            and signals
        ):
            reranked_records = []
            reranked_any = False
            for record, signal in zip(claim_records, signals):
                evidence = record.get("evidence") or []
                if not evidence:
                    reranked_records.append(record)
                    continue
                signal_map = self._build_rerank_signal_map(signal, evidence)
                reranked_evidence = self.evidence_reranker.rerank(evidence, signal_map)
                if [f"{c.doc_id}#{c.sent_id}" for c in reranked_evidence] != [
                    f"{c.doc_id}#{c.sent_id}" for c in evidence
                ]:
                    reranked_any = True
                reranked_records.append(
                    {
                        "claim": record["claim"],
                        "evidence": reranked_evidence,
                        "metadata": record.get("metadata") or {},
                    }
                )

            claim_records = reranked_records
            if reranked_any:
                actions.append("rerank")
            signals, decisions = self._verify_and_decide(claim_records)
            planned_actions = self.router.resolve_actions(decisions, objective_override)

        if (
            "reprompt" in planned_actions
            and self.reprompter
            and self.reprompter.enabled
            and decisions
        ):
            pooled_evidence = self._pool_evidence(claim_records)
            reprompt_result = self.reprompter.reprompt(
                query=query,
                answer=answer_text,
                decisions=decisions,
                evidence=pooled_evidence,
                claims=[r["claim"] for r in claim_records],
            )
            if reprompt_result.get("improved", False):
                actions.append("reprompt")
                answer_text = reprompt_result.get("final_answer", answer_text)
                corrected_claims = extract_claims(text=answer_text, method="auto")
                shared_metadata = claim_records[0].get("metadata") if claim_records else {}
                default_evidence = pooled_evidence[:5]
                claim_records = [
                    {
                        "claim": claim,
                        "evidence": default_evidence,
                        "metadata": shared_metadata or {},
                    }
                    for claim in corrected_claims
                    if default_evidence
                ]
                signals, decisions = self._verify_and_decide(claim_records)
                planned_actions = self.router.resolve_actions(decisions, objective_override)

        filtered_claim_count = 0
        if (
            "filter" in planned_actions
            and self.claim_filter
            and self.claim_filter.enabled
            and decisions
            and claim_records
        ):
            answer_text, filtered_claim_count = self.claim_filter.filter_answer(
                answer_text=answer_text,
                claims=[r["claim"] for r in claim_records],
                decisions=decisions,
            )
            if filtered_claim_count > 0:
                actions.append("filter")

        return {
            "final_answer": answer_text,
            "claim_records": claim_records,
            "decisions": decisions,
            "signals": signals,
            "actions": actions,
            "filtered_claim_count": filtered_claim_count,
        }

    def _verify_and_decide(self, claim_records: List[Dict[str, Any]]):
        signals = []
        decisions = []

        if not self.verifier_hub or not getattr(self.verifier_hub, "enabled", False):
            return signals, decisions

        for record in claim_records:
            claim = record.get("claim")
            evidence = record.get("evidence") or []
            metadata = record.get("metadata") or {}
            if claim is None or not evidence:
                continue
            signal = self.verifier_hub.verify_claim(claim, evidence, metadata)
            if signal is None:
                continue
            signals.append(signal)
            if self.aggregator is not None:
                decisions.append(self.aggregator.aggregate(signal))

        return signals, decisions

    def _build_rerank_signal_map(self, signal: Any, evidence_items: List[EvidenceChunk]):
        if signal is None:
            return {}

        signal_map = {}
        per_chunk_signals = getattr(signal, "per_chunk_signals", None) or []

        for item in per_chunk_signals:
            if not isinstance(item, dict):
                continue
            doc_id = item.get("doc_id")
            sent_id = item.get("sent_id")
            if doc_id is None or sent_id is None:
                continue
            nli = item.get("nli", {}) or {}
            coverage = item.get("coverage", {}) or {}
            if "entailment" not in nli and "entail" in nli:
                nli = {**nli, "entailment": nli.get("entail", 0.0)}
            signal_map[f"{doc_id}#{sent_id}"] = type(
                "RerankSignal",
                (),
                {"nli": nli, "coverage": coverage}
            )()

        if signal_map:
            return signal_map

        base_nli = getattr(signal, "nli", {}) or {}
        if "entailment" not in base_nli and "entail" in base_nli:
            base_nli = {**base_nli, "entailment": base_nli.get("entail", 0.0)}
        base_coverage = getattr(signal, "coverage", {}) or {}

        for evidence in evidence_items:
            key = f"{evidence.doc_id}#{evidence.sent_id}"
            signal_map[key] = type(
                "RerankSignal",
                (),
                {"nli": base_nli, "coverage": base_coverage}
            )()

        return signal_map

    def _pool_evidence(self, claim_records: List[Dict[str, Any]]) -> List[EvidenceChunk]:
        pooled: List[EvidenceChunk] = []
        seen = set()
        for record in claim_records:
            for chunk in record.get("evidence") or []:
                key = f"{chunk.doc_id}#{chunk.sent_id}"
                if key in seen:
                    continue
                seen.add(key)
                pooled.append(chunk)
        return pooled
