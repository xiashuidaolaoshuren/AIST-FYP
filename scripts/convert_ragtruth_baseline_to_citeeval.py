"""Convert RAGTruth baseline outputs to CiteEval/CiteBench system-eval JSON.

This adapter supports two common sources:
1) baseline prediction JSONL from benchmark/RAGTruth/baseline/predict_and_evaluate.py
2) local evaluation JSON with sample_results (e.g., outputs/ragtruth_eval/*.json)

Output schema per item:
{
  "id": "...",
  "query": "...",
  "passages": [{"id": "1", "title": "...", "text": "..."}],
  "pred": "..."
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle, start=1):
                payload = line.strip()
                if not payload:
                    continue
                row = json.loads(payload)
                if not isinstance(row, dict):
                    raise ValueError(f"Line {idx} in {path} is not a JSON object")
                rows.append(row)
        return rows

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("sample_results"), list):
        return [item for item in payload["sample_results"] if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Unsupported input shape in {path}")


def _load_source_info_map(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}

    source_map: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            row = json.loads(payload)
            if not isinstance(row, dict):
                continue
            source_id = str(row.get("source_id", "")).strip()
            source_text = str(row.get("source_info", "")).strip()
            if source_id and source_text:
                source_map[source_id] = source_text
    return source_map


def _iter_claim_dicts(sample: dict[str, Any]) -> list[dict[str, Any]]:
    claim_results = sample.get("claim_results", [])
    if not isinstance(claim_results, list):
        return []
    return [claim for claim in claim_results if isinstance(claim, dict)]


def _iter_evidence_dicts(claim: dict[str, Any]) -> list[dict[str, Any]]:
    evidences = claim.get("top_k_evidences", [])
    if not isinstance(evidences, list):
        return []
    return [ev for ev in evidences if isinstance(ev, dict)]


def _collect_claim_passages(sample: dict[str, Any]) -> list[dict[str, str]]:
    passages: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for claim in _iter_claim_dicts(sample):
        for ev in _iter_evidence_dicts(claim):
            text = str(ev.get("text", "")).strip()
            if not text:
                continue
            title = str(ev.get("doc_id", "ragtruth_context")).strip() or "ragtruth_context"
            key = (title, text)
            if key in seen:
                continue
            seen.add(key)
            passages.append({"title": title, "text": text})

    return passages


def _fallback_source_passage(sample: dict[str, Any], source_map: dict[str, str]) -> list[dict[str, str]]:
    source_id = str(sample.get("source_id", "")).strip()
    if source_id and source_id in source_map:
        return [{"title": f"ragtruth_source_{source_id}", "text": source_map[source_id]}]
    return []


def _to_citeeval_passages(passages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {"id": str(i + 1), "title": p["title"], "text": p["text"]}
        for i, p in enumerate(passages)
    ]


def _normalize_passages(sample: dict[str, Any], source_map: dict[str, str]) -> list[dict[str, str]]:
    passages = _collect_claim_passages(sample)
    if not passages:
        passages = _fallback_source_passage(sample, source_map)
    return _to_citeeval_passages(passages)


def _choose_pred(sample: dict[str, Any]) -> str:
    for key in ("generated_response", "response", "pred"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            for nested in ("response", "answer", "text", "output"):
                nested_value = value.get(nested)
                if isinstance(nested_value, str) and nested_value.strip():
                    return nested_value.strip()
            return json.dumps(value, ensure_ascii=False)
    return ""


def _choose_query(sample: dict[str, Any]) -> str:
    for key in ("question", "query", "prompt"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _choose_id(sample: dict[str, Any], index: int) -> str:
    for key in ("sample_id", "id", "answer_id", "source_id"):
        value = sample.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return f"sample_{index + 1}"


def convert(samples: list[dict[str, Any]], source_map: dict[str, str], strict: bool) -> tuple[list[dict[str, Any]], dict[str, int]]:
    converted: list[dict[str, Any]] = []
    skipped_empty_pred = 0
    skipped_empty_passages = 0

    for idx, sample in enumerate(samples):
        sample_id = _choose_id(sample, idx)
        query = _choose_query(sample)
        pred = _choose_pred(sample)
        passages = _normalize_passages(sample, source_map)

        if not pred:
            skipped_empty_pred += 1
            if strict:
                continue
        if not passages:
            skipped_empty_passages += 1
            if strict:
                continue

        converted.append(
            {
                "id": sample_id,
                "query": query,
                "passages": passages,
                "pred": pred,
            }
        )

    stats = {
        "total_input": len(samples),
        "total_output": len(converted),
        "skipped_empty_pred": skipped_empty_pred,
        "skipped_empty_passages": skipped_empty_passages,
    }
    return converted, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert RAGTruth baseline outputs to CiteEval system-eval JSON")
    parser.add_argument("--input", required=True, help="Input JSON/JSONL file")
    parser.add_argument("--output", required=True, help="Output CiteEval system-eval JSON file")
    parser.add_argument(
        "--source-info",
        default="benchmark/RAGTruth/dataset/source_info.jsonl",
        help="Optional source_info.jsonl path used to backfill passages",
    )
    parser.add_argument("--strict", action="store_true", help="Drop rows with empty pred or empty passages")
    parser.add_argument("--report-json", default=None, help="Optional output path for conversion statistics")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    source_info_path = Path(args.source_info).resolve() if args.source_info else None

    samples = _load_json_or_jsonl(input_path)
    source_map = _load_source_info_map(source_info_path)
    converted, stats = convert(samples, source_map, strict=args.strict)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(converted, handle, indent=2, ensure_ascii=False)

    if args.report_json:
        report_path = Path(args.report_json).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2, ensure_ascii=False)

    print(json.dumps({"output": str(output_path), **stats}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
