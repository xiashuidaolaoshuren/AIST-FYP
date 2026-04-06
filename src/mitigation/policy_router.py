"""
Mitigation policy routing for objective-aware action gating.

This module centralizes rule-based mitigation action selection so evaluators,
scripts, and runtime pipeline share one source of truth.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from src.utils.data_structures import ClaimDecision


@dataclass
class RouterThresholds:
    rerank_low_confidence_ratio: float = 0.25
    reprompt_contradiction_ratio: float = 0.35
    filter_contradiction_ratio: float = 0.5
    filter_min_contradictory_claims: int = 1
    reprompt_lc_avg_contradict_prob_threshold: float = 0.25
    reprompt_lc_avg_contradict_ratio_threshold: float = 0.30


class MitigationPolicyRouter:
    """Rule-based router for mitigation actions."""

    VALID_OBJECTIVES = {"balanced", "ragtruth", "citation"}

    def __init__(self, *, enabled: bool, objective: str, thresholds: RouterThresholds):
        self.enabled = bool(enabled)
        self.objective = self._normalize_objective(objective)
        self.thresholds = thresholds

    @classmethod
    def from_config(cls, config):
        mitigation_cfg = config.get("mitigation", {})
        if not isinstance(mitigation_cfg, dict):
            mitigation_cfg = {}

        router_cfg = mitigation_cfg.get("router", {})
        if not isinstance(router_cfg, dict):
            router_cfg = {}

        thresholds = RouterThresholds(
            rerank_low_confidence_ratio=float(router_cfg.get("rerank_low_confidence_ratio", 0.25)),
            reprompt_contradiction_ratio=float(router_cfg.get("reprompt_contradiction_ratio", 0.35)),
            filter_contradiction_ratio=float(router_cfg.get("filter_contradiction_ratio", 0.5)),
            filter_min_contradictory_claims=int(router_cfg.get("filter_min_contradictory_claims", 1)),
            reprompt_lc_avg_contradict_prob_threshold=float(
                router_cfg.get("reprompt_lc_avg_contradict_prob_threshold", 0.25)
            ),
            reprompt_lc_avg_contradict_ratio_threshold=float(
                router_cfg.get("reprompt_lc_avg_contradict_ratio_threshold", 0.30)
            ),
        )

        return cls(
            enabled=bool(router_cfg.get("enabled", False)),
            objective=str(router_cfg.get("objective", "balanced")),
            thresholds=thresholds,
        )

    def resolve_actions(
        self,
        decisions: List[ClaimDecision],
        objective_override: str | None = None,
    ) -> Set[str]:
        allowed, _ = self._resolve_with_diagnostics(decisions, objective_override)
        return allowed

    def explain_actions(
        self,
        decisions: List[ClaimDecision],
        objective_override: str | None = None,
    ) -> Dict[str, Any]:
        """Return both allowed actions and a diagnostics dict explaining the routing decision.

        Diagnostics keys:
            objective, total_claims, contradictory_count, low_confidence_count,
            contradiction_ratio, low_confidence_ratio, lc_avg_contradict_signal,
            filter_fired, filter_blocked_reason, reprompt_fired, rerank_fired,
            filter_threshold (effective), filter_min_claims (effective).
        """
        allowed, diagnostics = self._resolve_with_diagnostics(decisions, objective_override)
        diagnostics["allowed_actions"] = sorted(allowed)
        return diagnostics

    def _resolve_with_diagnostics(
        self,
        decisions: List[ClaimDecision],
        objective_override: str | None = None,
    ):
        default_actions = {"rerank", "reprompt", "filter"}
        if not self.enabled:
            return default_actions, {"router_enabled": False}

        if not decisions:
            return set(), {"router_enabled": True, "total_claims": 0}

        objective = self._normalize_objective(objective_override or self.objective)
        total = len(decisions)
        contradictory_count = len([d for d in decisions if d.status == "Contradictory"])
        low_confidence_count = len([d for d in decisions if d.status == "Low Confidence"])
        contradiction_ratio = contradictory_count / total if total else 0.0
        low_confidence_ratio = low_confidence_count / total if total else 0.0
        low_confidence_decisions = [d for d in decisions if d.status == "Low Confidence"]
        avg_contradict_prob_low_conf = (
            sum(float(d.confidence.get('contradict_prob', 0.0)) for d in low_confidence_decisions)
            / len(low_confidence_decisions)
            if low_confidence_decisions else 0.0
        )
        lc_avg_contradict_signal = (
            contradictory_count == 0
            and low_confidence_ratio >= self.thresholds.reprompt_lc_avg_contradict_ratio_threshold
            and avg_contradict_prob_low_conf >= self.thresholds.reprompt_lc_avg_contradict_prob_threshold
        )

        allowed: Set[str] = set()

        if (
            low_confidence_ratio >= self.thresholds.rerank_low_confidence_ratio
            or contradictory_count > 0
        ):
            allowed.add("rerank")

        if objective == "ragtruth":
            if (
                contradictory_count > 0
                or contradiction_ratio >= self.thresholds.reprompt_contradiction_ratio
                or lc_avg_contradict_signal
            ):
                allowed.add("reprompt")
            if contradictory_count >= 1 or lc_avg_contradict_signal:
                allowed.add("filter")
        elif objective == "citation":
            if contradiction_ratio >= max(self.thresholds.reprompt_contradiction_ratio, 0.5):
                allowed.add("reprompt")
            if (
                contradiction_ratio >= max(self.thresholds.filter_contradiction_ratio, 0.6)
                and contradictory_count >= max(self.thresholds.filter_min_contradictory_claims, 2)
            ):
                allowed.add("filter")
        else:
            if contradiction_ratio >= self.thresholds.reprompt_contradiction_ratio:
                allowed.add("reprompt")
            if (
                contradiction_ratio >= self.thresholds.filter_contradiction_ratio
                and contradictory_count >= self.thresholds.filter_min_contradictory_claims
            ):
                allowed.add("filter")

        # Build diagnostics for callers that want to audit routing decisions.
        effective_filter_threshold = self.thresholds.filter_contradiction_ratio
        if objective == "citation":
            effective_filter_threshold = max(effective_filter_threshold, 0.6)

        filter_fired = "filter" in allowed
        filter_blocked_reason: str | None = None
        if not filter_fired and contradictory_count >= self.thresholds.filter_min_contradictory_claims:
            if objective == "ragtruth":
                filter_blocked_reason = None  # ragtruth always fires when contradictions > 0
            else:
                filter_blocked_reason = (
                    f"ratio {contradiction_ratio:.3f} < threshold {effective_filter_threshold:.3f}"
                )
        elif not filter_fired and contradictory_count == 0:
            filter_blocked_reason = "no_contradictory_claims"

        diagnostics: Dict[str, Any] = {
            "router_enabled": True,
            "objective": objective,
            "total_claims": total,
            "contradictory_count": contradictory_count,
            "low_confidence_count": low_confidence_count,
            "contradiction_ratio": round(contradiction_ratio, 4),
            "low_confidence_ratio": round(low_confidence_ratio, 4),
            "lc_avg_contradict_signal": lc_avg_contradict_signal,
            "avg_contradict_prob_low_conf": round(avg_contradict_prob_low_conf, 4),
            "effective_filter_threshold": effective_filter_threshold,
            "filter_min_claims": self.thresholds.filter_min_contradictory_claims,
            "filter_fired": filter_fired,
            "filter_blocked_reason": filter_blocked_reason,
            "reprompt_fired": "reprompt" in allowed,
            "rerank_fired": "rerank" in allowed,
        }
        return allowed, diagnostics

    def _normalize_objective(self, objective: str) -> str:
        value = str(objective).strip().lower()
        if value not in self.VALID_OBJECTIVES:
            return "balanced"
        return value
