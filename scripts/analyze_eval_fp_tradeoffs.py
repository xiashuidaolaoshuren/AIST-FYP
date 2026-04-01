import argparse
import json
from pathlib import Path
from typing import Any


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return None


def _parse_float_list(raw: str) -> list[float]:
    if not raw.strip():
        return []
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _load_samples(input_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("sample_results"), list):
            return [item for item in payload["sample_results"] if isinstance(item, dict)]
        if isinstance(payload.get("samples"), list):
            return [item for item in payload["samples"] if isinstance(item, dict)]
    raise ValueError("Unsupported input JSON structure; expected list or object with sample_results/samples")


def _get_gold(sample: dict[str, Any]) -> bool | None:
    for key in ("gold_has_hallucination", "gold_label", "label", "gold"):
        if key in sample:
            return _to_bool(sample.get(key))
    return None


def _get_pred(sample: dict[str, Any]) -> bool | None:
    for key in ("detected_hallucination", "prediction", "predicted", "pred"):
        if key in sample:
            return _to_bool(sample.get(key))
    return None


def _safe_float(sample: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = sample.get(key, default)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(sample: dict[str, Any], key: str, default: int = 0) -> int:
    value = sample.get(key, default)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def compute_confusion(samples: list[dict[str, Any]]) -> dict[str, int]:
    tp = fp = fn = tn = 0
    for sample in samples:
        gold = _get_gold(sample)
        pred = _get_pred(sample)
        if gold is None or pred is None:
            continue
        if gold and pred:
            tp += 1
        elif (not gold) and pred:
            fp += 1
        elif gold and (not pred):
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def analyze_data2txt_contradictory_threshold(
    samples: list[dict[str, Any]],
    old_threshold: float,
    candidates: list[float],
    cp_limit: float,
    avg_cov_floor: float,
) -> list[dict[str, Any]]:
    pool: list[tuple[bool | None, float, float, float, str]] = []
    for sample in samples:
        if sample.get("task_type") != "Data2txt":
            continue
        if _get_pred(sample) is not True:
            continue
        if sample.get("detection_trigger_path") != "contradictory":
            continue
        pool.append(
            (
                _get_gold(sample),
                _safe_float(sample, "max_contradict_prob"),
                _safe_float(sample, "avg_coverage_score_all"),
                _safe_float(sample, "avg_contradict_prob_low_conf"),
                str(sample.get("sample_id") or sample.get("id") or sample.get("task_id") or "unknown"),
            )
        )

    results: list[dict[str, Any]] = []
    for threshold in sorted(candidates):
        impacted_ids: list[str] = []
        tp_lost = 0
        fp_saved = 0
        for gold, max_cp, avg_cov, avg_cp_lc, sample_id in pool:
            newly_blocked = (
                max_cp < cp_limit
                and avg_cov >= avg_cov_floor
                and avg_cp_lc <= threshold
                and avg_cp_lc > old_threshold
            )
            if not newly_blocked:
                continue
            impacted_ids.append(sample_id)
            if gold is True:
                tp_lost += 1
            elif gold is False:
                fp_saved += 1
        results.append(
            {
                "threshold": threshold,
                "fp_saved": fp_saved,
                "tp_lost": tp_lost,
                "net": fp_saved - tp_lost,
                "impacted_count": len(impacted_ids),
                "impacted_ids": impacted_ids,
            }
        )
    return results


def analyze_qa_single_contra_exception(
    samples: list[dict[str, Any]],
    base_cp: float,
    base_cov: float,
    cp_candidates: list[float],
    cov_candidates: list[float],
) -> list[dict[str, Any]]:
    qa_samples = [sample for sample in samples if sample.get("task_type") == "QA"]
    results: list[dict[str, Any]] = []

    for cp_threshold in sorted(cp_candidates, reverse=True):
        for cov_threshold in sorted(cov_candidates):
            tp_gain = 0
            fp_gain = 0
            impacted_ids: list[str] = []
            for sample in qa_samples:
                if _get_pred(sample) is not False:
                    continue
                contradictory_count = _safe_int(sample, "contradictory_count")
                max_cp = _safe_float(sample, "max_contradict_prob")
                max_cov = _safe_float(sample, "max_contradict_coverage")
                new_hit = contradictory_count == 1 and max_cp >= cp_threshold and max_cov <= cov_threshold
                base_hit = contradictory_count == 1 and max_cp >= base_cp and max_cov <= base_cov
                if not new_hit or base_hit:
                    continue
                impacted_ids.append(str(sample.get("sample_id") or sample.get("id") or sample.get("task_id") or "unknown"))
                gold = _get_gold(sample)
                if gold is True:
                    tp_gain += 1
                elif gold is False:
                    fp_gain += 1
            results.append(
                {
                    "min_contradict_prob": cp_threshold,
                    "max_contradict_coverage": cov_threshold,
                    "tp_gain": tp_gain,
                    "fp_gain": fp_gain,
                    "net": tp_gain - fp_gain,
                    "impacted_count": len(impacted_ids),
                    "impacted_ids": impacted_ids,
                }
            )
    return results


def analyze_summary_contradictory_lower_coverage_guard(
    samples: list[dict[str, Any]],
    cp_candidates: list[float],
    avg_cov_candidates: list[float],
) -> list[dict[str, Any]]:
    pool: list[tuple[bool | None, int, float, float, str]] = []
    for sample in samples:
        if sample.get("task_type") != "Summary":
            continue
        if _get_pred(sample) is not True:
            continue
        if sample.get("detection_trigger_path") != "contradictory":
            continue
        pool.append(
            (
                _get_gold(sample),
                _safe_int(sample, "contradictory_count"),
                _safe_float(sample, "max_contradict_prob"),
                _safe_float(sample, "avg_coverage_score_all"),
                str(sample.get("sample_id") or sample.get("id") or sample.get("task_id") or "unknown"),
            )
        )

    results: list[dict[str, Any]] = []
    for cp_threshold in sorted(cp_candidates):
        for avg_cov_threshold in sorted(avg_cov_candidates):
            tp_lost = 0
            fp_saved = 0
            impacted_ids: list[str] = []
            for gold, contradictory_count, max_cp, avg_cov, sample_id in pool:
                # Candidate second-tier guard shape: only single-contra Summary,
                # where contradiction confidence and overall coverage are both moderate.
                newly_blocked = (
                    contradictory_count == 1
                    and max_cp <= cp_threshold
                    and avg_cov <= avg_cov_threshold
                )
                if not newly_blocked:
                    continue
                impacted_ids.append(sample_id)
                if gold is True:
                    tp_lost += 1
                elif gold is False:
                    fp_saved += 1
            results.append(
                {
                    "max_contradict_prob": cp_threshold,
                    "max_avg_coverage_score_all": avg_cov_threshold,
                    "fp_saved": fp_saved,
                    "tp_lost": tp_lost,
                    "net": fp_saved - tp_lost,
                    "impacted_count": len(impacted_ids),
                    "impacted_ids": impacted_ids,
                }
            )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze Eval output JSON and sweep guard threshold tradeoffs for FP/FN changes."
    )
    parser.add_argument("--input", required=True, help="Path to evaluation output JSON")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    parser.add_argument(
        "--data2txt-candidates",
        default="0.03,0.05,0.07,0.10,0.12,0.15,0.18,0.20",
        help="Comma-separated candidate thresholds for data2txt contradictory avg_cp_lc guard",
    )
    parser.add_argument(
        "--old-data2txt-threshold",
        type=float,
        default=0.0105,
        help="Current configured data2txt contradictory avg_cp_lc threshold",
    )
    parser.add_argument(
        "--data2txt-cp-limit",
        type=float,
        default=0.9980,
        help="Max contradict prob condition used by the existing data2txt contradictory structural guard",
    )
    parser.add_argument(
        "--data2txt-avg-cov-floor",
        type=float,
        default=0.7777,
        help="Min avg coverage condition used by the existing data2txt contradictory structural guard",
    )
    parser.add_argument(
        "--qa-cp-candidates",
        default="0.98,0.97,0.95,0.93,0.90",
        help="Comma-separated candidate thresholds for QA single-contra min contradict_prob",
    )
    parser.add_argument(
        "--qa-cov-candidates",
        default="0.40,0.45,0.50,0.60",
        help="Comma-separated candidate thresholds for QA single-contra max contradict_coverage",
    )
    parser.add_argument(
        "--qa-base-cp",
        type=float,
        default=0.99,
        help="Current configured QA single-contra min contradict_prob",
    )
    parser.add_argument(
        "--qa-base-cov",
        type=float,
        default=0.40,
        help="Current configured QA single-contra max contradict_coverage",
    )
    parser.add_argument(
        "--summary-cp-candidates",
        default="0.90,0.93,0.95,0.97,0.99",
        help="Comma-separated candidate max contradict_prob values for a Summary lower-coverage contradictory guard",
    )
    parser.add_argument(
        "--summary-avg-cov-candidates",
        default="0.70,0.74,0.78,0.80",
        help="Comma-separated candidate max avg_coverage_score_all values for a Summary lower-coverage contradictory guard",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    samples = _load_samples(Path(args.input))

    confusion = compute_confusion(samples)
    data2txt_results = analyze_data2txt_contradictory_threshold(
        samples=samples,
        old_threshold=args.old_data2txt_threshold,
        candidates=_parse_float_list(args.data2txt_candidates),
        cp_limit=args.data2txt_cp_limit,
        avg_cov_floor=args.data2txt_avg_cov_floor,
    )
    qa_results = analyze_qa_single_contra_exception(
        samples=samples,
        base_cp=args.qa_base_cp,
        base_cov=args.qa_base_cov,
        cp_candidates=_parse_float_list(args.qa_cp_candidates),
        cov_candidates=_parse_float_list(args.qa_cov_candidates),
    )
    summary_results = analyze_summary_contradictory_lower_coverage_guard(
        samples=samples,
        cp_candidates=_parse_float_list(args.summary_cp_candidates),
        avg_cov_candidates=_parse_float_list(args.summary_avg_cov_candidates),
    )

    report = {
        "input": str(Path(args.input).resolve()),
        "sample_count": len(samples),
        "confusion": confusion,
        "data2txt_contradictory_threshold_sweep": data2txt_results,
        "qa_single_contra_exception_sweep": qa_results,
        "summary_contradictory_lower_coverage_sweep": summary_results,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    best_data2txt = max(data2txt_results, key=lambda x: (x["net"], x["fp_saved"], -x["tp_lost"])) if data2txt_results else None
    best_qa = max(qa_results, key=lambda x: (x["net"], x["tp_gain"], -x["fp_gain"])) if qa_results else None
    best_summary = max(summary_results, key=lambda x: (x["net"], x["fp_saved"], -x["tp_lost"])) if summary_results else None

    print(f"samples={report['sample_count']} confusion={confusion}")
    if best_data2txt:
        print(
            "best_data2txt: "
            f"threshold={best_data2txt['threshold']:.4f} "
            f"fp_saved={best_data2txt['fp_saved']} tp_lost={best_data2txt['tp_lost']} net={best_data2txt['net']}"
        )
    if best_qa:
        print(
            "best_qa: "
            f"cp>={best_qa['min_contradict_prob']:.4f} cov<={best_qa['max_contradict_coverage']:.4f} "
            f"tp_gain={best_qa['tp_gain']} fp_gain={best_qa['fp_gain']} net={best_qa['net']}"
        )
    if best_summary:
        print(
            "best_summary: "
            f"cp<={best_summary['max_contradict_prob']:.4f} avg_cov<={best_summary['max_avg_coverage_score_all']:.4f} "
            f"fp_saved={best_summary['fp_saved']} tp_lost={best_summary['tp_lost']} net={best_summary['net']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
