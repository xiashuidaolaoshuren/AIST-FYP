from src.mitigation.policy_router import MitigationPolicyRouter, RouterThresholds
from src.utils.data_structures import ClaimDecision


def _decision(status: str, idx: int) -> ClaimDecision:
    return ClaimDecision(
        claim_id=f"c{idx}",
        status=status,
        rationale="r",
        primary_evidence=f"d#{idx}",
        signals_ref=[],
        confidence={},
    )


def test_router_disabled_returns_default_actions():
    router = MitigationPolicyRouter(
        enabled=False,
        objective="balanced",
        thresholds=RouterThresholds(),
    )

    actions = router.resolve_actions([_decision("Supported", 1)])

    assert actions == {"rerank", "reprompt", "filter"}


def test_router_citation_is_conservative_for_filtering():
    router = MitigationPolicyRouter(
        enabled=True,
        objective="citation",
        thresholds=RouterThresholds(),
    )

    decisions = [
        _decision("Contradictory", 1),
        _decision("Low Confidence", 2),
        _decision("Supported", 3),
    ]

    actions = router.resolve_actions(decisions)

    assert "rerank" in actions
    assert "reprompt" not in actions
    assert "filter" not in actions
