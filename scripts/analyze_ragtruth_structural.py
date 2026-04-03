"""
Structural analysis of RAGTruth evaluation at scale.

Compares the original 480-sample subset (shared) against the 1020 new-only
samples to identify WHY performance degrades, focusing on:
1. Gold label distribution shift
2. Per-task trigger path effectiveness
3. Guard activation impact (TP blocked vs FP blocked)
4. Numeric signal distributions for misclassified samples
5. Top structural bottlenecks

Usage:
    python scripts/analyze_ragtruth_structural.py \
        --baseline <eval22_or_23.json> \
        --candidate <eval24.json> \
        --output <path_prefix>
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

KNOWN_GUARD_FIELDS = [
    "qa_pure_lc_block",
    "qa_single_contra_exception_trigger",
    "data2txt_contradictory_override_block",
    "data2txt_contradictory_structural_guard_block",
    "all_sentences_summary_contradictory_override_block",
    "all_sentences_summary_contradictory_structural_guard_block",
    "all_sentences_summary_lc_guard_block",
    "all_sentences_summary_lc_residual_guard_block",
    "summary_lc_after_single_contra_guard_block",
    "data2txt_low_confidence_structural_guard_block",
    "summary_single_contra_block",
    "summary_single_contra_low_cov_guard_block",
    "lc_avg_contradict_task_block",
    "detected_pre_qa_block",
]

NUMERIC_SIGNAL_FIELDS = [
    "contradictory_count",
    "low_confidence_count",
    "low_confidence_ratio",
    "low_coverage_count",
    "low_coverage_ratio",
    "avg_coverage_score_all",
    "avg_support_prob_low_conf",
    "avg_contradict_prob_low_conf",
    "max_contradict_prob",
    "max_contradict_coverage",
    "total_sentences",
    "num_claims",
]


def load_samples(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return [s for s in payload["sample_results"] if isinstance(s, dict)]


def sample_key(s: dict) -> str:
    return f"{s.get('task_type', '?')}::{s.get('sample_id', s.get('task_id', '?'))}"


def outcome(s: dict) -> str:
    det = bool(s.get("detected_hallucination", False))
    gold = bool(s.get("gold_has_hallucination", False))
    if det and gold:
        return "TP"
    if det and not gold:
        return "FP"
    if not det and gold:
        return "FN"
    return "TN"


def safe_div(n: float, d: float) -> float:
    return 0.0 if d == 0 else n / d


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict:
    total = tp + fp + fn + tn
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    f1 = safe_div(2 * prec * rec, prec + rec)
    return {
        "n": total,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "accuracy": round(safe_div(tp + tn, total), 4),
        "gold_positive_rate": round(safe_div(tp + fn, total), 4),
    }


def compute_metrics(samples: list[dict]) -> dict:
    c = Counter(outcome(s) for s in samples)
    return metrics_from_counts(c["TP"], c["FP"], c["FN"], c["TN"])


def group_by_task(samples: list[dict]) -> dict[str, list[dict]]:
    g: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        g[str(s.get("task_type", "unknown"))].append(s)
    return dict(sorted(g.items()))


def trigger_breakdown(samples: list[dict]) -> dict[str, dict[str, int]]:
    """Trigger path counts grouped by outcome."""
    result: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for s in samples:
        oc = outcome(s)
        trigger = str(s.get("detection_trigger_path", "none"))
        result[oc][trigger] += 1
    return {oc: dict(sorted(triggers.items())) for oc, triggers in sorted(result.items())}


def guard_impact(samples: list[dict]) -> list[dict]:
    """For each guard, compute how many samples it fires on and the outcome distribution."""
    results = []
    for field in KNOWN_GUARD_FIELDS:
        fired = [s for s in samples if bool(s.get(field, False))]
        if not fired:
            continue
        oc_counts = Counter(outcome(s) for s in fired)
        # How many gold-positive samples are suppressed (guard fires + not detected)?
        suppressed_tp = sum(
            1 for s in fired
            if bool(s.get("gold_has_hallucination", False))
            and not bool(s.get("detected_hallucination", False))
        )
        # How many gold-negative samples are correctly suppressed?
        suppressed_fp = sum(
            1 for s in fired
            if not bool(s.get("gold_has_hallucination", False))
            and not bool(s.get("detected_hallucination", False))
        )
        results.append({
            "guard": field,
            "total_fired": len(fired),
            "outcome_dist": dict(oc_counts),
            "suppressed_tp_as_fn": suppressed_tp,
            "suppressed_fp_as_tn": suppressed_fp,
            "net_benefit": suppressed_fp - suppressed_tp,
        })
    results.sort(key=lambda x: -abs(x["net_benefit"]))
    return results


def signal_stats(samples: list[dict], field: str) -> dict | None:
    """Compute distribution stats for a numeric field."""
    values = []
    for s in samples:
        v = s.get(field)
        if v is not None:
            try:
                values.append(float(v))
            except (TypeError, ValueError):
                pass
    if len(values) < 2:
        return None
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "stdev": round(statistics.stdev(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "p25": round(sorted(values)[len(values) // 4], 4),
        "p75": round(sorted(values)[3 * len(values) // 4], 4),
    }


def signal_comparison_by_outcome(samples: list[dict]) -> dict:
    """For each outcome bucket, compute signal distributions."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        buckets[outcome(s)].append(s)

    result = {}
    for oc in ["TP", "FP", "FN", "TN"]:
        bucket = buckets.get(oc, [])
        if not bucket:
            continue
        signals = {}
        for field in NUMERIC_SIGNAL_FIELDS:
            stats = signal_stats(bucket, field)
            if stats:
                signals[field] = stats
        result[oc] = {"count": len(bucket), "signals": signals}
    return result


def analyze_task_block(
    task: str,
    shared_samples: list[dict],
    new_samples: list[dict],
) -> dict:
    """Deep analysis of one task type comparing shared vs new samples."""
    return {
        "shared": {
            "metrics": compute_metrics(shared_samples),
            "triggers": trigger_breakdown(shared_samples),
            "guard_impact": guard_impact(shared_samples),
        },
        "new_only": {
            "metrics": compute_metrics(new_samples),
            "triggers": trigger_breakdown(new_samples),
            "guard_impact": guard_impact(new_samples),
            "signal_by_outcome": signal_comparison_by_outcome(new_samples),
        },
    }


def identify_bottlenecks(all_new: list[dict]) -> dict:
    """Identify the top structural problems."""
    tasks = group_by_task(all_new)
    bottlenecks = {}

    for task_name, task_samples in tasks.items():
        issues = []
        m = compute_metrics(task_samples)

        # Issue: high FP rate
        fp_rate = safe_div(m["FP"], m["FP"] + m["TN"])
        if fp_rate > 0.2:
            fp_samples = [s for s in task_samples if outcome(s) == "FP"]
            fp_triggers = Counter(str(s.get("detection_trigger_path", "none")) for s in fp_samples)
            issues.append({
                "type": "high_fp_rate",
                "fp_rate": round(fp_rate, 4),
                "fp_count": m["FP"],
                "top_fp_triggers": dict(fp_triggers.most_common(5)),
            })

        # Issue: high FN rate (missed real hallucinations)
        fn_rate = safe_div(m["FN"], m["FN"] + m["TP"])
        if fn_rate > 0.2:
            fn_samples = [s for s in task_samples if outcome(s) == "FN"]
            # What guards are blocking these?
            fn_guard_blocks = {}
            for field in KNOWN_GUARD_FIELDS:
                blocked = sum(1 for s in fn_samples if bool(s.get(field, False)))
                if blocked > 0:
                    fn_guard_blocks[field] = blocked
            # What trigger paths do FN samples have?
            fn_triggers = Counter(str(s.get("detection_trigger_path", "none")) for s in fn_samples)
            issues.append({
                "type": "high_fn_rate",
                "fn_rate": round(fn_rate, 4),
                "fn_count": m["FN"],
                "fn_trigger_paths": dict(fn_triggers.most_common(5)),
                "fn_guard_blocks": fn_guard_blocks,
            })

        # Issue: guards blocking too many TP
        gi = guard_impact(task_samples)
        for g in gi:
            if g["suppressed_tp_as_fn"] >= 5:
                issues.append({
                    "type": "guard_suppresses_tp",
                    "guard": g["guard"],
                    "suppressed_tp": g["suppressed_tp_as_fn"],
                    "suppressed_fp": g["suppressed_fp_as_tn"],
                    "net_benefit": g["net_benefit"],
                })

        bottlenecks[task_name] = issues

    return bottlenecks


def format_report(analysis: dict) -> str:
    lines = []
    lines.append("=" * 90)
    lines.append("RAGTRUTH STRUCTURAL ANALYSIS: WHY PERFORMANCE DROPS AT SCALE")
    lines.append("=" * 90)

    # Section 1: Overall comparison
    lines.append("\n## 1. OVERALL: SHARED (480) vs NEW-ONLY (1020)")
    lines.append("-" * 90)
    shared_m = analysis["overall"]["shared_metrics"]
    new_m = analysis["overall"]["new_only_metrics"]
    lines.append(f"{'Subset':<15} {'N':>5} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}  {'Prec':>7} {'Rec':>7} {'F1':>7} {'Gold+%':>7}")
    lines.append(
        f"{'Shared':<15} {shared_m['n']:>5} {shared_m['TP']:>5} {shared_m['FP']:>5} "
        f"{shared_m['FN']:>5} {shared_m['TN']:>5}  {shared_m['precision']:>7.4f} "
        f"{shared_m['recall']:>7.4f} {shared_m['f1']:>7.4f} {shared_m['gold_positive_rate']:>7.4f}"
    )
    lines.append(
        f"{'New-only':<15} {new_m['n']:>5} {new_m['TP']:>5} {new_m['FP']:>5} "
        f"{new_m['FN']:>5} {new_m['TN']:>5}  {new_m['precision']:>7.4f} "
        f"{new_m['recall']:>7.4f} {new_m['f1']:>7.4f} {new_m['gold_positive_rate']:>7.4f}"
    )

    # Section 2: Gold label distribution shift
    lines.append("\n## 2. GOLD LABEL DISTRIBUTION SHIFT")
    lines.append("-" * 90)
    for task_name in ["Data2txt", "QA", "Summary"]:
        task_data = analysis["per_task"].get(task_name, {})
        if not task_data:
            continue
        sm = task_data["shared"]["metrics"]
        nm = task_data["new_only"]["metrics"]
        lines.append(f"\n  {task_name}:")
        lines.append(f"    Shared:   n={sm['n']:>4}  gold+={sm['TP']+sm['FN']:>3} ({sm['gold_positive_rate']:.2%})  gold-={sm['TN']+sm['FP']:>3}")
        lines.append(f"    New-only: n={nm['n']:>4}  gold+={nm['TP']+nm['FN']:>3} ({nm['gold_positive_rate']:.2%})  gold-={nm['TN']+nm['FP']:>3}")

    # Section 3: Per-task metrics comparison
    lines.append("\n## 3. PER-TASK METRICS: SHARED vs NEW-ONLY")
    lines.append("-" * 90)
    lines.append(f"{'Task':<12} {'Subset':<10} {'N':>5} {'Prec':>7} {'Rec':>7} {'F1':>7}  {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}")
    for task_name in ["Data2txt", "QA", "Summary"]:
        task_data = analysis["per_task"].get(task_name, {})
        if not task_data:
            continue
        sm = task_data["shared"]["metrics"]
        nm = task_data["new_only"]["metrics"]
        lines.append(
            f"{task_name:<12} {'Shared':<10} {sm['n']:>5} {sm['precision']:>7.4f} "
            f"{sm['recall']:>7.4f} {sm['f1']:>7.4f}  {sm['TP']:>4} {sm['FP']:>4} {sm['FN']:>4} {sm['TN']:>4}"
        )
        lines.append(
            f"{'':<12} {'New-only':<10} {nm['n']:>5} {nm['precision']:>7.4f} "
            f"{nm['recall']:>7.4f} {nm['f1']:>7.4f}  {nm['TP']:>4} {nm['FP']:>4} {nm['FN']:>4} {nm['TN']:>4}"
        )

    # Section 4: Trigger path comparison
    lines.append("\n## 4. TRIGGER PATH EFFECTIVENESS: NEW-ONLY SAMPLES")
    lines.append("-" * 90)
    for task_name in ["Data2txt", "QA", "Summary"]:
        task_data = analysis["per_task"].get(task_name, {})
        if not task_data:
            continue
        lines.append(f"\n  {task_name} — Shared triggers:")
        for oc in ["TP", "FP", "FN", "TN"]:
            triggers = task_data["shared"]["triggers"].get(oc, {})
            if triggers:
                t_str = ", ".join(f"{k}={v}" for k, v in triggers.items())
                lines.append(f"    {oc}: {t_str}")

        lines.append(f"  {task_name} — New-only triggers:")
        for oc in ["TP", "FP", "FN", "TN"]:
            triggers = task_data["new_only"]["triggers"].get(oc, {})
            if triggers:
                t_str = ", ".join(f"{k}={v}" for k, v in triggers.items())
                lines.append(f"    {oc}: {t_str}")

    # Section 5: Guard impact
    lines.append("\n## 5. GUARD IMPACT ON NEW-ONLY SAMPLES")
    lines.append("-" * 90)
    lines.append(f"{'Guard':<55} {'Fired':>6} {'TP→FN':>6} {'FP→TN':>6} {'Net':>6}")
    for task_name in ["Data2txt", "QA", "Summary"]:
        task_data = analysis["per_task"].get(task_name, {})
        if not task_data:
            continue
        lines.append(f"\n  [{task_name}]")
        for g in task_data["new_only"]["guard_impact"]:
            lines.append(
                f"  {g['guard']:<53} {g['total_fired']:>6} {g['suppressed_tp_as_fn']:>6} "
                f"{g['suppressed_fp_as_tn']:>6} {g['net_benefit']:>+6}"
            )

    # Section 6: Bottleneck summary
    lines.append("\n## 6. STRUCTURAL BOTTLENECKS")
    lines.append("-" * 90)
    for task_name, issues in analysis["bottlenecks"].items():
        if not issues:
            lines.append(f"\n  {task_name}: No major bottlenecks identified.")
            continue
        lines.append(f"\n  {task_name}:")
        for issue in issues:
            if issue["type"] == "high_fp_rate":
                lines.append(f"    ⚠ HIGH FP RATE: {issue['fp_rate']:.1%} ({issue['fp_count']} FP)")
                lines.append(f"      Top triggers: {issue['top_fp_triggers']}")
            elif issue["type"] == "high_fn_rate":
                lines.append(f"    ⚠ HIGH FN RATE: {issue['fn_rate']:.1%} ({issue['fn_count']} FN)")
                lines.append(f"      FN trigger paths: {issue['fn_trigger_paths']}")
                if issue["fn_guard_blocks"]:
                    lines.append(f"      Guards blocking FN: {issue['fn_guard_blocks']}")
            elif issue["type"] == "guard_suppresses_tp":
                lines.append(
                    f"    ⚠ GUARD [{issue['guard']}] suppresses {issue['suppressed_tp']} TP "
                    f"(saves {issue['suppressed_fp']} FP, net={issue['net_benefit']:+d})"
                )

    # Section 7: Key signal distributions for FP vs TN (new-only)
    lines.append("\n## 7. SIGNAL DISTRIBUTIONS FOR MISCLASSIFIED SAMPLES (NEW-ONLY)")
    lines.append("-" * 90)
    key_signals = [
        "max_contradict_prob",
        "avg_coverage_score_all",
        "low_confidence_ratio",
        "contradictory_count",
    ]
    for task_name in ["Data2txt", "QA", "Summary"]:
        task_data = analysis["per_task"].get(task_name, {})
        if not task_data:
            continue
        sig_data = task_data["new_only"].get("signal_by_outcome", {})
        lines.append(f"\n  {task_name}:")
        for signal in key_signals:
            lines.append(f"    {signal}:")
            for oc in ["TP", "FP", "FN", "TN"]:
                oc_data = sig_data.get(oc, {})
                stats = oc_data.get("signals", {}).get(signal)
                if stats:
                    lines.append(
                        f"      {oc} (n={stats['count']:>3}): "
                        f"mean={stats['mean']:.4f} median={stats['median']:.4f} "
                        f"p25={stats['p25']:.4f} p75={stats['p75']:.4f} "
                        f"min={stats['min']:.4f} max={stats['max']:.4f}"
                    )

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Structural analysis of RAGTruth at scale")
    parser.add_argument("--baseline", required=True, help="Path to 480-sample baseline JSON (Eval 22 or 23)")
    parser.add_argument("--candidate", required=True, help="Path to 1500-sample candidate JSON (Eval 24)")
    parser.add_argument("--output", required=True, help="Output prefix (without extension)")
    args = parser.parse_args()

    baseline_path = Path(args.baseline).resolve()
    candidate_path = Path(args.candidate).resolve()
    output_prefix = Path(args.output).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    baseline_samples = load_samples(baseline_path)
    candidate_samples = load_samples(candidate_path)

    # Build index for shared/new split
    baseline_keys = {sample_key(s) for s in baseline_samples}
    candidate_index = {sample_key(s): s for s in candidate_samples}

    shared = [candidate_index[k] for k in sorted(candidate_index) if k in baseline_keys]
    new_only = [candidate_index[k] for k in sorted(candidate_index) if k not in baseline_keys]

    print(f"Total candidate: {len(candidate_samples)}")
    print(f"Shared with baseline: {len(shared)}")
    print(f"New-only: {len(new_only)}")

    # Overall
    analysis: dict[str, Any] = {
        "baseline_path": str(baseline_path),
        "candidate_path": str(candidate_path),
        "overall": {
            "shared_metrics": compute_metrics(shared),
            "new_only_metrics": compute_metrics(new_only),
            "combined_metrics": compute_metrics(candidate_samples),
        },
    }

    # Per-task deep dive
    shared_tasks = group_by_task(shared)
    new_tasks = group_by_task(new_only)
    all_task_names = sorted(set(list(shared_tasks.keys()) + list(new_tasks.keys())))

    per_task = {}
    for task_name in all_task_names:
        s_samples = shared_tasks.get(task_name, [])
        n_samples = new_tasks.get(task_name, [])
        per_task[task_name] = analyze_task_block(task_name, s_samples, n_samples)

    analysis["per_task"] = per_task

    # Bottlenecks
    analysis["bottlenecks"] = identify_bottlenecks(new_only)

    # Generate report
    text_report = format_report(analysis)

    json_path = output_prefix.with_suffix(".json")
    txt_path = output_prefix.with_suffix(".txt")

    json_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    txt_path.write_text(text_report, encoding="utf-8")

    print(text_report)
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved TXT:  {txt_path}")


if __name__ == "__main__":
    main()
