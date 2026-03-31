#!/usr/bin/env python
"""Analyze eval JSON to quantify FP/TP trade-offs by task and trigger path.

Usage:
  python scripts/analyze_eval_fp_tradeoffs.py \
    --input "C:\\Users\\admin\\Desktop\\eval_temp\\verification\\ragtruth_full_verifier(18).json" \
    --output outputs/eval16_fp_tradeoffs.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Tuple


SIGNAL_FIELDS = [
    "low_confidence_ratio",
    "low_coverage_ratio",
    "avg_coverage_score_all",
    "avg_coverage_score_low_conf",
    "avg_support_prob_low_conf",
    "avg_contradict_prob_low_conf",
    "max_contradict_prob",
    "max_contradict_coverage",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze RAGTruth eval outputs and generate FP/TP threshold sweep tables."
    )
    parser.add_argument("--input", required=True, help="Path to ragtruth_full_verifier JSON")
    parser.add_argument(
        "--output",
        default="outputs/eval_fp_tradeoffs.json",
        help="Output JSON path for analysis summary",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Decimal precision for floating stats",
    )
    return parser.parse_args()


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stats(values: Iterable[float], precision: int) -> Dict[str, Any]:
    data = [v for v in values if v is not None]
    if not data:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(data),
        "min": round(min(data), precision),
        "max": round(max(data), precision),
        "mean": round(mean(data), precision),
        "median": round(median(data), precision),
    }


def _extract_rows(samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    fp: List[Dict[str, Any]] = []
    fn: List[Dict[str, Any]] = []
    tp: List[Dict[str, Any]] = []
    tn: List[Dict[str, Any]] = []

    for s in samples:
        row = {
            "sample_id": str(s.get("sample_id", "")),
            "task_type": s.get("task_type", "Unknown"),
            "detection_trigger_path": s.get("detection_trigger_path", "none"),
            "detected_hallucination": bool(s.get("detected_hallucination", False)),
            "gold_has_hallucination": bool(s.get("gold_has_hallucination", False)),
            "num_claims": s.get("num_claims"),
            "contradictory_count": s.get("contradictory_count"),
            "low_confidence_count": s.get("low_confidence_count"),
        }
        for field in SIGNAL_FIELDS:
            row[field] = _safe_float(s.get(field))

        if row["detected_hallucination"] and not row["gold_has_hallucination"]:
            fp.append(row)
        elif (not row["detected_hallucination"]) and row["gold_has_hallucination"]:
            fn.append(row)
        elif row["detected_hallucination"] and row["gold_has_hallucination"]:
            tp.append(row)
        else:
            tn.append(row)

    return fp, fn, tp, tn


def _group_by_task_path(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["task_type"], r["detection_trigger_path"])].append(r)
    return grouped


def _threshold_sweep(
    fp_rows: List[Dict[str, Any]],
    tp_rows: List[Dict[str, Any]],
    field: str,
    operator: str,
    precision: int,
) -> List[Dict[str, Any]]:
    all_values = []
    for row in fp_rows + tp_rows:
        val = row.get(field)
        if val is not None:
            all_values.append(val)

    if not all_values:
        return []

    candidates = sorted({round(v, precision) for v in all_values})
    sweeps: List[Dict[str, Any]] = []

    for threshold in candidates:
        if operator == "ge":
            fp_saved = sum(1 for r in fp_rows if (r.get(field) is not None and r[field] < threshold))
            tp_lost = sum(1 for r in tp_rows if (r.get(field) is not None and r[field] < threshold))
        else:
            fp_saved = sum(1 for r in fp_rows if (r.get(field) is not None and r[field] > threshold))
            tp_lost = sum(1 for r in tp_rows if (r.get(field) is not None and r[field] > threshold))

        sweeps.append(
            {
                "threshold": threshold,
                "fp_saved": fp_saved,
                "tp_lost": tp_lost,
                "net_gain": fp_saved - tp_lost,
            }
        )

    return sweeps


def build_analysis(data: Dict[str, Any], precision: int) -> Dict[str, Any]:
    samples = data.get("sample_results", [])
    fp_rows, fn_rows, tp_rows, tn_rows = _extract_rows(samples)

    fp_grouped = _group_by_task_path(fp_rows)
    tp_grouped = _group_by_task_path(tp_rows)

    grouped_summary: Dict[str, Any] = {}
    for key in sorted(set(fp_grouped.keys()) | set(tp_grouped.keys())):
        task, path = key
        fp_group = fp_grouped.get(key, [])
        tp_group = tp_grouped.get(key, [])

        group_key = f"{task}|{path}"
        grouped_summary[group_key] = {
            "task_type": task,
            "trigger_path": path,
            "fp_count": len(fp_group),
            "tp_count": len(tp_group),
            "path_precision": round(len(tp_group) / (len(tp_group) + len(fp_group)), precision)
            if (len(tp_group) + len(fp_group)) > 0
            else None,
            "fp_signal_stats": {
                field: _stats((r.get(field) for r in fp_group), precision)
                for field in SIGNAL_FIELDS
            },
            "tp_signal_stats": {
                field: _stats((r.get(field) for r in tp_group), precision)
                for field in SIGNAL_FIELDS
            },
            "threshold_sweeps": {
                "low_confidence_ratio_min": _threshold_sweep(fp_group, tp_group, "low_confidence_ratio", "ge", precision),
                "low_coverage_ratio_min": _threshold_sweep(fp_group, tp_group, "low_coverage_ratio", "ge", precision),
                "max_contradict_prob_min": _threshold_sweep(fp_group, tp_group, "max_contradict_prob", "ge", precision),
                "max_contradict_coverage_min": _threshold_sweep(fp_group, tp_group, "max_contradict_coverage", "ge", precision),
                "avg_contradict_prob_low_conf_min": _threshold_sweep(fp_group, tp_group, "avg_contradict_prob_low_conf", "ge", precision),
                "avg_coverage_score_all_max": _threshold_sweep(fp_group, tp_group, "avg_coverage_score_all", "le", precision),
                "avg_support_prob_low_conf_max": _threshold_sweep(fp_group, tp_group, "avg_support_prob_low_conf", "le", precision),
            },
        }

    return {
        "summary_counts": {
            "fp": len(fp_rows),
            "fn": len(fn_rows),
            "tp": len(tp_rows),
            "tn": len(tn_rows),
            "num_samples": len(samples),
        },
        "grouped_tradeoffs": grouped_summary,
        "top_level_metrics": data.get("metrics", {}).get("overall", {}),
    }


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    analysis = build_analysis(data, precision=max(1, args.precision))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    counts = analysis["summary_counts"]
    print(
        "Saved analysis to {} (samples={}, FP={}, FN={}, TP={}, TN={})".format(
            output_path,
            counts["num_samples"],
            counts["fp"],
            counts["fn"],
            counts["tp"],
            counts["tn"],
        )
    )


if __name__ == "__main__":
    main()
