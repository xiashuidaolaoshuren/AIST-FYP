import argparse
import json
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


def load_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object at top level: {path}")
    if not isinstance(payload.get("sample_results"), list):
        raise ValueError(f"Missing sample_results list: {path}")
    return payload


def sample_key(sample: dict[str, Any]) -> str:
    return f"{sample.get('task_type', 'unknown')}::{sample.get('sample_id', sample.get('task_id', 'unknown'))}"


def outcome(sample: dict[str, Any]) -> str:
    detected = bool(sample.get("detected_hallucination", False))
    gold = bool(sample.get("gold_has_hallucination", False))
    if detected and gold:
        return "TP"
    if detected and not gold:
        return "FP"
    if not detected and gold:
        return "FN"
    return "TN"


def confusion(samples: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(outcome(sample) for sample in samples)
    return {
        "TP": counts.get("TP", 0),
        "FP": counts.get("FP", 0),
        "FN": counts.get("FN", 0),
        "TN": counts.get("TN", 0),
    }


def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def derive_metrics(samples: list[dict[str, Any]]) -> dict[str, float | int]:
    cm = confusion(samples)
    tp = cm["TP"]
    fp = cm["FP"]
    fn = cm["FN"]
    tn = cm["TN"]
    total = tp + fp + fn + tn
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    accuracy = safe_div(tp + tn, total)
    return {
        "num_samples": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }


def per_task_confusion(samples: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample.get("task_type", "unknown"))].append(sample)
    return {task: confusion(task_samples) for task, task_samples in sorted(grouped.items())}


def trigger_counts(
    samples: list[dict[str, Any]],
    *,
    wanted_outcomes: set[str] | None = None,
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        current_outcome = outcome(sample)
        if wanted_outcomes is not None and current_outcome not in wanted_outcomes:
            continue
        task = str(sample.get("task_type", "unknown"))
        trigger = str(sample.get("detection_trigger_path", "none"))
        counts[task][trigger] += 1
    return {task: dict(counter) for task, counter in sorted(counts.items())}


def guard_counts(samples: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for field in KNOWN_GUARD_FIELDS:
        true_samples = [sample for sample in samples if bool(sample.get(field, False))]
        if not true_samples:
            continue
        summary[field] = {
            "total": len(true_samples),
            "gold_positive": sum(bool(sample.get("gold_has_hallucination", False)) for sample in true_samples),
            "gold_negative": sum(not bool(sample.get("gold_has_hallucination", False)) for sample in true_samples),
            "detected_positive": sum(bool(sample.get("detected_hallucination", False)) for sample in true_samples),
            "detected_negative": sum(not bool(sample.get("detected_hallucination", False)) for sample in true_samples),
        }
    return summary


def metric_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    keys = ["accuracy", "precision", "recall", "f1"]
    return {
        key: float(candidate["metrics"]["overall"][key]) - float(baseline["metrics"]["overall"][key])
        for key in keys
    }


def task_metric_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, dict[str, float]]:
    keys = ["accuracy", "precision", "recall", "f1"]
    deltas: dict[str, dict[str, float]] = {}
    for task in ["Data2txt", "QA", "Summary"]:
        cand_task = candidate["metrics"]["per_task"].get(task, {})
        base_task = baseline["metrics"]["per_task"].get(task, {})
        deltas[task] = {
            key: float(cand_task.get(key, 0.0)) - float(base_task.get(key, 0.0))
            for key in keys
        }
    return deltas


def extract_sample_brief(sample: dict[str, Any]) -> dict[str, Any]:
    result = {
        "sample_key": sample_key(sample),
        "sample_id": sample.get("sample_id"),
        "task_type": sample.get("task_type"),
        "gold_has_hallucination": sample.get("gold_has_hallucination"),
        "detected_hallucination": sample.get("detected_hallucination"),
        "classification": outcome(sample),
        "detection_trigger_path": sample.get("detection_trigger_path"),
        "contradictory_count": sample.get("contradictory_count"),
        "low_confidence_count": sample.get("low_confidence_count"),
        "low_confidence_ratio": sample.get("low_confidence_ratio"),
        "low_coverage_count": sample.get("low_coverage_count"),
        "low_coverage_ratio": sample.get("low_coverage_ratio"),
        "avg_coverage_score_all": sample.get("avg_coverage_score_all"),
        "avg_support_prob_low_conf": sample.get("avg_support_prob_low_conf"),
        "avg_contradict_prob_low_conf": sample.get("avg_contradict_prob_low_conf"),
        "max_contradict_prob": sample.get("max_contradict_prob"),
        "max_contradict_coverage": sample.get("max_contradict_coverage"),
    }
    guard_values = {field: bool(sample.get(field, False)) for field in KNOWN_GUARD_FIELDS if field in sample}
    result["guards"] = guard_values
    return result


def compare_shared_samples(
    baseline_samples: list[dict[str, Any]],
    candidate_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_index = {sample_key(sample): sample for sample in baseline_samples}
    candidate_index = {sample_key(sample): sample for sample in candidate_samples}

    baseline_keys = set(baseline_index)
    candidate_keys = set(candidate_index)
    shared_keys = sorted(baseline_keys & candidate_keys)
    baseline_only_keys = sorted(baseline_keys - candidate_keys)
    candidate_only_keys = sorted(candidate_keys - baseline_keys)

    baseline_shared = [baseline_index[key] for key in shared_keys]
    candidate_shared = [candidate_index[key] for key in shared_keys]
    candidate_only = [candidate_index[key] for key in candidate_only_keys]

    changed: list[dict[str, Any]] = []
    change_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key in shared_keys:
        before = baseline_index[key]
        after = candidate_index[key]
        before_outcome = outcome(before)
        after_outcome = outcome(after)
        if before_outcome == after_outcome:
            continue
        entry = {
            "sample_key": key,
            "before": extract_sample_brief(before),
            "after": extract_sample_brief(after),
            "transition": f"{before_outcome}->{after_outcome}",
        }
        changed.append(entry)
        change_buckets[entry["transition"]].append(entry)

    return {
        "shared_count": len(shared_keys),
        "baseline_only_count": len(baseline_only_keys),
        "candidate_only_count": len(candidate_only_keys),
        "baseline_shared_metrics": derive_metrics(baseline_shared),
        "candidate_shared_metrics": derive_metrics(candidate_shared),
        "candidate_only_metrics": derive_metrics(candidate_only),
        "candidate_only_per_task_confusion": per_task_confusion(candidate_only),
        "candidate_only_fp_triggers": trigger_counts(candidate_only, wanted_outcomes={"FP"}),
        "candidate_only_tp_triggers": trigger_counts(candidate_only, wanted_outcomes={"TP"}),
        "candidate_only_guard_counts": guard_counts(candidate_only),
        "shared_change_counts": {transition: len(entries) for transition, entries in sorted(change_buckets.items())},
        "shared_changed_samples": changed,
        "candidate_only_examples": [extract_sample_brief(sample) for sample in candidate_only[:50]],
        "baseline_only_keys": baseline_only_keys,
        "candidate_only_keys": candidate_only_keys,
    }


def build_text_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    baseline_name = Path(report["baseline_path"]).name
    candidate_name = Path(report["candidate_path"]).name
    lines.append(f"RAGTruth Eval Comparison: {candidate_name} vs {baseline_name}")
    lines.append("=" * 80)
    lines.append("")
    lines.append("1. Full-run metric delta (different sample sets if counts differ)")
    lines.append("-" * 80)
    full_delta = report["full_metric_delta"]
    full_candidate = report["candidate_full_metrics"]
    full_baseline = report["baseline_full_metrics"]
    lines.append(
        f"Baseline samples={full_baseline['num_samples']} | Candidate samples={full_candidate['num_samples']}"
    )
    for key in ["accuracy", "precision", "recall", "f1"]:
        lines.append(
            f"{key:10} baseline={full_baseline[key]:.6f} candidate={full_candidate[key]:.6f} delta={full_delta[key]:+.6f}"
        )
    lines.append("")
    lines.append("2. Shared-sample overlap check")
    lines.append("-" * 80)
    shared = report["shared_sample_analysis"]
    lines.append(
        f"Shared={shared['shared_count']} | Candidate-only={shared['candidate_only_count']} | Baseline-only={shared['baseline_only_count']}"
    )
    base_shared = shared["baseline_shared_metrics"]
    cand_shared = shared["candidate_shared_metrics"]
    lines.append(
        "Shared baseline confusion: "
        f"TN={base_shared['TN']} FP={base_shared['FP']} FN={base_shared['FN']} TP={base_shared['TP']}"
    )
    lines.append(
        "Shared candidate confusion: "
        f"TN={cand_shared['TN']} FP={cand_shared['FP']} FN={cand_shared['FN']} TP={cand_shared['TP']}"
    )
    lines.append(
        f"Shared sample classification changes: {shared['shared_change_counts'] or 'none'}"
    )
    lines.append("")
    lines.append("3. Candidate-only sample difficulty profile")
    lines.append("-" * 80)
    only_metrics = shared["candidate_only_metrics"]
    lines.append(
        f"Candidate-only confusion: TN={only_metrics['TN']} FP={only_metrics['FP']} FN={only_metrics['FN']} TP={only_metrics['TP']}"
    )
    lines.append(
        f"Candidate-only metrics: acc={only_metrics['accuracy']:.6f} prec={only_metrics['precision']:.6f} rec={only_metrics['recall']:.6f} f1={only_metrics['f1']:.6f}"
    )
    lines.append("Per-task candidate-only confusion:")
    for task, cm in shared["candidate_only_per_task_confusion"].items():
        lines.append(f"  {task}: TN={cm['TN']} FP={cm['FP']} FN={cm['FN']} TP={cm['TP']}")
    lines.append("Candidate-only FP triggers:")
    for task, triggers in shared["candidate_only_fp_triggers"].items():
        trigger_str = ", ".join(f"{name}={count}" for name, count in sorted(triggers.items()))
        lines.append(f"  {task}: {trigger_str}")
    lines.append("Candidate-only TP triggers:")
    for task, triggers in shared["candidate_only_tp_triggers"].items():
        trigger_str = ", ".join(f"{name}={count}" for name, count in sorted(triggers.items()))
        lines.append(f"  {task}: {trigger_str}")
    lines.append("")
    lines.append("4. Candidate-only guard activations")
    lines.append("-" * 80)
    if shared["candidate_only_guard_counts"]:
        for guard_name, counts in shared["candidate_only_guard_counts"].items():
            lines.append(
                f"{guard_name}: total={counts['total']} gold+={counts['gold_positive']} gold-={counts['gold_negative']} detected+={counts['detected_positive']} detected-={counts['detected_negative']}"
            )
    else:
        lines.append("No guard activations recorded in candidate-only samples.")
    lines.append("")
    lines.append("5. Per-task full-run metric delta")
    lines.append("-" * 80)
    for task, deltas in report["per_task_metric_delta"].items():
        lines.append(
            f"{task}: acc={deltas['accuracy']:+.6f} prec={deltas['precision']:+.6f} rec={deltas['recall']:+.6f} f1={deltas['f1']:+.6f}"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two RAGTruth evaluation outputs and save structured analysis.")
    parser.add_argument("--baseline", required=True, help="Path to baseline evaluation JSON")
    parser.add_argument("--candidate", required=True, help="Path to candidate evaluation JSON")
    parser.add_argument("--output-prefix", required=True, help="Output prefix without extension")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    baseline_path = Path(args.baseline).resolve()
    candidate_path = Path(args.candidate).resolve()
    output_prefix = Path(args.output_prefix).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    baseline = load_payload(baseline_path)
    candidate = load_payload(candidate_path)

    baseline_samples = [sample for sample in baseline["sample_results"] if isinstance(sample, dict)]
    candidate_samples = [sample for sample in candidate["sample_results"] if isinstance(sample, dict)]

    report = {
        "baseline_path": str(baseline_path),
        "candidate_path": str(candidate_path),
        "baseline_full_metrics": {
            **baseline["metrics"]["overall"],
            **{
                "TN": baseline["metrics"]["confusion_matrix"]["true_negatives"],
                "FP": baseline["metrics"]["confusion_matrix"]["false_positives"],
                "FN": baseline["metrics"]["confusion_matrix"]["false_negatives"],
                "TP": baseline["metrics"]["confusion_matrix"]["true_positives"],
            },
        },
        "candidate_full_metrics": {
            **candidate["metrics"]["overall"],
            **{
                "TN": candidate["metrics"]["confusion_matrix"]["true_negatives"],
                "FP": candidate["metrics"]["confusion_matrix"]["false_positives"],
                "FN": candidate["metrics"]["confusion_matrix"]["false_negatives"],
                "TP": candidate["metrics"]["confusion_matrix"]["true_positives"],
            },
        },
        "full_metric_delta": metric_delta(candidate, baseline),
        "per_task_metric_delta": task_metric_delta(candidate, baseline),
        "baseline_fp_triggers": trigger_counts(baseline_samples, wanted_outcomes={"FP"}),
        "candidate_fp_triggers": trigger_counts(candidate_samples, wanted_outcomes={"FP"}),
        "baseline_tp_triggers": trigger_counts(baseline_samples, wanted_outcomes={"TP"}),
        "candidate_tp_triggers": trigger_counts(candidate_samples, wanted_outcomes={"TP"}),
        "shared_sample_analysis": compare_shared_samples(baseline_samples, candidate_samples),
    }

    text_report = build_text_report(report)
    json_path = output_prefix.with_suffix(".json")
    txt_path = output_prefix.with_suffix(".txt")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    txt_path.write_text(text_report, encoding="utf-8")

    print(text_report)
    print(f"Saved JSON report: {json_path}")
    print(f"Saved text report: {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())