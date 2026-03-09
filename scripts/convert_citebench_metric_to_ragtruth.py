"""Convert CiteBench metric_eval JSON to RAGTruth baseline-style JSONL.

This converter prepares CiteBench samples for baseline-compatible processing.
By design in this project:
- task_type is fixed to QA
- response is prefilled from CiteBench prediction
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list JSON in {path}")
    return [row for row in payload if isinstance(row, dict)]


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def _normalize_passages(row: dict[str, Any]) -> list[dict[str, str]]:
    passages = row.get("passages", [])
    if not isinstance(passages, list):
        return []

    normalized: list[dict[str, str]] = []
    for idx, item in enumerate(passages, start=1):
        if not isinstance(item, dict):
            continue
        text = _safe_text(item.get("text"))
        if not text:
            continue
        title = _safe_text(item.get("title"))
        pid = _safe_text(item.get("id")) or str(idx)
        normalized.append({"id": pid, "title": title, "text": text})
    return normalized


def _reference_from_passages(passages: list[dict[str, str]]) -> str:
    chunks: list[str] = []
    for item in passages:
        title = item.get("title", "")
        text = item.get("text", "")
        if title:
            chunks.append(f"[{title}] {text}")
        else:
            chunks.append(text)
    return "\n\n".join(chunks)


def _build_output_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    sample_id = _safe_text(row.get("sample_idx"))
    query = _safe_text(row.get("query"))
    prediction = _safe_text(row.get("prediction"))
    passages = _normalize_passages(row)

    source_id = f"citebench_metric_{sample_id}" if sample_id else ""
    return {
        "id": sample_id,
        "source_id": source_id,
        "task_type": "QA",
        "question": query,
        "response": prediction,
        "reference": _reference_from_passages(passages),
        "labels": [],
        "fold": -1,
        "split": split,
        "quality": "good",
        "passages": passages,
    }


def _validate_required(output_row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not _safe_text(output_row.get("id")):
        errors.append("missing_id")
    if not _safe_text(output_row.get("question")):
        errors.append("missing_question")
    if not _safe_text(output_row.get("response")):
        errors.append("missing_response")
    passages = output_row.get("passages", [])
    if not isinstance(passages, list) or len(passages) == 0:
        errors.append("missing_passages")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CiteBench metric_eval JSON to RAGTruth baseline-style JSONL")
    parser.add_argument("--input", required=True, help="Input citebench.metric_* JSON file")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--strict", action="store_true", help="Drop rows failing required-field checks")
    parser.add_argument("--report-json", default=None, help="Optional conversion report JSON path")
    parser.add_argument("--aligned-ids-output", default=None, help="Optional aligned sample ID list JSON path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    rows = _load_rows(input_path)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    converted: list[dict[str, Any]] = []
    aligned_ids: list[str] = []
    dropped_reason_counts: dict[str, int] = {}

    for row in rows:
        output_row = _build_output_row(row, split=args.split)
        row_errors = _validate_required(output_row)

        if row_errors and args.strict:
            for reason in row_errors:
                dropped_reason_counts[reason] = dropped_reason_counts.get(reason, 0) + 1
            continue

        if row_errors:
            output_row["conversion_warnings"] = row_errors

        row_id = _safe_text(output_row.get("id"))
        if row_id:
            aligned_ids.append(row_id)
        converted.append(output_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in converted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.aligned_ids_output:
        ids_path = Path(args.aligned_ids_output).resolve()
        ids_path.parent.mkdir(parents=True, exist_ok=True)
        with ids_path.open("w", encoding="utf-8") as handle:
            json.dump(aligned_ids, handle, indent=2, ensure_ascii=False)

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "split": args.split,
        "strict": args.strict,
        "total_input": len(rows),
        "total_output": len(converted),
        "dropped": len(rows) - len(converted),
        "dropped_reason_counts": dropped_reason_counts,
        "aligned_id_count": len(aligned_ids),
    }

    if args.report_json:
        report_path = Path(args.report_json).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)

    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
