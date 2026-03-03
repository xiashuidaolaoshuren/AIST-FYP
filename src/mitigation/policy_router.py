"""
Mitigation policy routing for objective-aware action gating.

This module centralizes rule-based mitigation action selection so evaluators,
scripts, and runtime pipeline share one source of truth.
"""

from dataclasses import dataclass
from typing import List, Set

from src.utils.data_structures import ClaimDecision


@dataclass
class RouterThresholds:
    rerank_low_confidence_ratio: float = 0.25
    reprompt_contradiction_ratio: float = 0.35
    filter_contradiction_ratio: float = 0.5
    filter_min_contradictory_claims: int = 1


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
        default_actions = {"rerank", "reprompt", "filter"}
        if not self.enabled:
            return default_actions

        if not decisions:
            return set()

        objective = self._normalize_objective(objective_override or self.objective)
        total = len(decisions)
        contradictory_count = len([d for d in decisions if d.status == "Contradictory"])
        low_confidence_count = len([d for d in decisions if d.status == "Low Confidence"])
        contradiction_ratio = contradictory_count / total if total else 0.0
        low_confidence_ratio = low_confidence_count / total if total else 0.0

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
            ):
                allowed.add("reprompt")
            if contradictory_count >= 1:
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

        return allowed

    def _normalize_objective(self, objective: str) -> str:
        value = str(objective).strip().lower()
        if value not in self.VALID_OBJECTIVES:
            return "balanced"
        return value
