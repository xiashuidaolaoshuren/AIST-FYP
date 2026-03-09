"""Convert official RAGTruth baseline predictions to CiteEval system-input JSON.

The official baseline evaluator writes JSONL rows (e.g., prediction.jsonl) that are
not directly compatible with CiteEval system-track input. This adapter aligns the
baseline rows to a CiteBench source file and emits records in this schema:

[
  {
    "id": "...",
    "query": "...",
    "passages": [{"text": "...", "title": "..."}, ...],
    "pred": "..."
  }
]

Typical usage:
    python scripts/convert_ragtruth_baseline_to_citeeval.py \
      --prediction-jsonl benchmark/RAGTruth/baseline/prediction.jsonl \
      --system-source benchmark/CiteEval/data/system_eval/system_eval_examples.json \
      --output benchmark/CiteEval/data/system_eval/ragtruth_official.json \
      --match-by query
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


_CITATION_RE = re.compile(r"\[\d+\]")


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            item = json.loads(payload)
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object on line {idx} in {path}")
            rows.append(item)
    if not rows:
        raise ValueError(f"No rows found in prediction file: {path}")
    return rows


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list JSON in {path}, got {type(payload).__name__}")
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Expected object at index {idx} in {path}")
        rows.append(item)
    if not rows:
        raise ValueError(f"No rows found in system source: {path}")
    return rows


def _build_prediction_index(pred_rows: list[dict[str, Any]], match_by: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in pred_rows:
        if match_by == "id":
            key = str(row.get("id", "")).strip()
        else:
            key = _normalize_text(str(row.get("question", "")))

        if not key:
            continue
        if key in indexed:
            continue
        indexed[key] = row
    return indexed


def _ensure_citations(text: str, passages: list[dict[str, Any]], policy: str, max_citations: int) -> str:
    text = text.strip()
    if not text:
        return text

    if policy == "preserve":
        return text

    has_citation = bool(_CITATION_RE.search(text))
    if has_citation:
        return text

    if not passages:
        return text

    if policy == "append_first":
        return f"{text} [1]"

    cap = max(1, min(max_citations, len(passages)))
    suffix = "".join(f"[{i}]" for i in range(1, cap + 1))
    return f"{text} {suffix}"


def _build_output_row(
    source_row: dict[str, Any],
    pred_row: dict[str, Any],
    *,
    citation_policy: str,
    max_citations: int,
) -> dict[str, Any]:
    sample_id = str(source_row.get("id", "")).strip()
    query = str(source_row.get("query", "")).strip()
    passages = source_row.get("passages", [])

    if not sample_id:
        raise ValueError(f"Source row missing 'id': {source_row}")
    if not query:
        raise ValueError(f"Source row missing 'query' for id={sample_id}")
    if not isinstance(passages, list):
        raise ValueError(f"Source row has non-list 'passages' for id={sample_id}")

    response_text = str(pred_row.get("response", "")).strip()
    if not response_text:
        # Fallback to empty JSON-like output content if response is missing.
        # This keeps schema valid while making failures explicit in output quality.
        response_text = ""

    response_text = _ensure_citations(
        response_text,
        passages,
        policy=citation_policy,
        max_citations=max_citations,
    )

    return {
        "id": sample_id,
        "query": query,
        "passages": passages,
        "pred": response_text,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert official RAGTruth baseline predictions to CiteEval system input.")
    parser.add_argument("--prediction-jsonl", type=str, required=True, help="Path to official baseline prediction JSONL")
    parser.add_argument("--system-source", type=str, required=True, help="Path to source CiteBench JSON list with id/query/passages")
    parser.add_argument("--output", type=str, required=True, help="Output path for CiteEval system-input JSON")
    parser.add_argument("--match-by", choices=["id", "query"], default="query", help="How to align baseline predictions to source rows")
    parser.add_argument("--citation-policy", choices=["preserve", "append_first", "append_all"], default="append_first")
    parser.add_argument("--max-citations", type=int, default=3, help="Used only when --citation-policy append_all")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit converted rows for smoke runs")
    parser.add_argument("--allow-missing", action="store_true", help="Skip unmatched source rows instead of failing")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    pred_path = Path(args.prediction_jsonl).resolve()
    source_path = Path(args.system_source).resolve()
    output_path = Path(args.output).resolve()

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"System source file not found: {source_path}")

    pred_rows = _load_jsonl(pred_path)
    source_rows = _load_json_list(source_path)

    if args.max_samples is not None:
        source_rows = source_rows[: args.max_samples]

    pred_index = _build_prediction_index(pred_rows, args.match_by)

    output_rows: list[dict[str, Any]] = []
    unmatched = 0

    for src in source_rows:
        if args.match_by == "id":
            key = str(src.get("id", "")).strip()
        else:
            key = _normalize_text(str(src.get("query", "")))

        pred = pred_index.get(key)
        if pred is None:
            unmatched += 1
            if args.allow_missing:
                continue
            raise ValueError(
                f"No prediction match for source row id={src.get('id')} using match-by={args.match_by}. "
                "Use --allow-missing to skip unmatched rows."
            )

        out_row = _build_output_row(
            src,
            pred,
            citation_policy=args.citation_policy,
            max_citations=args.max_citations,
        )
        output_rows.append(out_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Conversion completed.")
    print(f"- Prediction rows: {len(pred_rows)}")
    print(f"- Source rows considered: {len(source_rows)}")
    print(f"- Output rows: {len(output_rows)}")
    print(f"- Unmatched source rows: {unmatched}")
    print(f"- Output file: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
