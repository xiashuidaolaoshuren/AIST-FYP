"""Convert CiteEval system JSON (oracle-track) to LettuceDetect inference input.

This converter prepares a stable, schema-checked input file for a pretrained
LettuceDetect pipeline. It preserves sample IDs and passages for aligned runs.

Expected input rows (from CiteEval system format):
- id
- query
- passages: [{id?, title?, text}, ...]
- pred (response text, often containing [1][2] style citations)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


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


def _strip_bracket_citations(text: str) -> str:
    if not text:
        return ""
    return _CITATION_PATTERN.sub("", text).strip()


def _normalize_passages(raw_passages: Any) -> list[dict[str, str]]:
    if not isinstance(raw_passages, list):
        return []

    passages: list[dict[str, str]] = []
    for idx, item in enumerate(raw_passages, start=1):
        if not isinstance(item, dict):
            continue
        text = _safe_text(item.get("text"))
        if not text:
            continue
        pid = _safe_text(item.get("id")) or str(idx)
        title = _safe_text(item.get("title"))
        passages.append({"id": pid, "title": title, "text": text})
    return passages


def _flatten_context(passages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for passage in passages:
        title = passage.get("title", "")
        text = passage.get("text", "")
        if title:
            parts.append(f"[{passage['id']}] {title}: {text}")
        else:
            parts.append(f"[{passage['id']}] {text}")
    return "\n\n".join(parts)


def _build_row(row: dict[str, Any], include_flat_context: bool) -> dict[str, Any]:
    sample_id = _safe_text(row.get("id"))
    query = _safe_text(row.get("query"))
    response = _strip_bracket_citations(_safe_text(row.get("pred")))
    passages = _normalize_passages(row.get("passages"))

    output: dict[str, Any] = {
        "id": sample_id,
        "query": query,
        "response": response,
        "passages": passages,
        "prediction_sentences_and_citations": row.get("prediction_sentences_and_citations", []),
    }
    if include_flat_context:
        output["context"] = _flatten_context(passages)
    return output


def _validate_required(output_row: dict[str, Any], include_flat_context: bool) -> list[str]:
    errors: list[str] = []
    if not _safe_text(output_row.get("id")):
        errors.append("missing_id")
    if not _safe_text(output_row.get("query")):
        errors.append("missing_query")
    if not _safe_text(output_row.get("response")):
        errors.append("missing_response")
    passages = output_row.get("passages")
    if not isinstance(passages, list) or len(passages) == 0:
        errors.append("missing_passages")
    if include_flat_context and not _safe_text(output_row.get("context")):
        errors.append("missing_context")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CiteEval oracle-track system JSON to LettuceDetect input format"
    )
    parser.add_argument("--input", required=True, help="Input CiteEval system JSON file")
    parser.add_argument("--output", required=True, help="Output JSON/JSONL path")
    parser.add_argument("--output-format", choices=["json", "jsonl"], default="json")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--strict", action="store_true", help="Drop rows failing required-field checks")
    parser.add_argument("--include-flat-context", action="store_true", help="Add a flattened `context` field")
    parser.add_argument("--report-json", default=None, help="Optional conversion report JSON path")
    parser.add_argument("--aligned-ids-output", default=None, help="Optional aligned sample ID list JSON path")
    return parser.parse_args()


def _record_drop_reasons(dropped_reason_counts: dict[str, int], reasons: list[str]) -> None:
    for reason in reasons:
        dropped_reason_counts[reason] = dropped_reason_counts.get(reason, 0) + 1


def _maybe_add_warning(row: dict[str, Any], reasons: list[str]) -> None:
    if reasons:
        row["conversion_warnings"] = reasons


def _convert_rows(
    rows: list[dict[str, Any]],
    strict: bool,
    include_flat_context: bool,
) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    converted: list[dict[str, Any]] = []
    aligned_ids: list[str] = []
    dropped_reason_counts: dict[str, int] = {}

    for row in rows:
        output_row = _build_row(row, include_flat_context=include_flat_context)
        row_errors = _validate_required(output_row, include_flat_context=include_flat_context)

        if row_errors and strict:
            _record_drop_reasons(dropped_reason_counts, row_errors)
            continue

        _maybe_add_warning(output_row, row_errors)

        row_id = _safe_text(output_row.get("id"))
        if row_id:
            aligned_ids.append(row_id)
        converted.append(output_row)

    return converted, aligned_ids, dropped_reason_counts


def _write_output_rows(output_path: Path, output_format: str, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "jsonl":
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    rows = _load_rows(input_path)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    converted, aligned_ids, dropped_reason_counts = _convert_rows(
        rows=rows,
        strict=args.strict,
        include_flat_context=args.include_flat_context,
    )
    _write_output_rows(output_path=output_path, output_format=args.output_format, rows=converted)

    if args.aligned_ids_output:
        ids_path = Path(args.aligned_ids_output).resolve()
        ids_path.parent.mkdir(parents=True, exist_ok=True)
        with ids_path.open("w", encoding="utf-8") as handle:
            json.dump(aligned_ids, handle, indent=2, ensure_ascii=False)

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "output_format": args.output_format,
        "strict": args.strict,
        "include_flat_context": args.include_flat_context,
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
