"""Convert pretrained LettuceDetect outputs to CiteEval/CiteBench system-eval JSON.

Because LettuceDetect output schema may vary by repo/runtime, this adapter is
field-configurable with dotted key paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nltk.tokenize import sent_tokenize


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle, start=1):
                payload = line.strip()
                if not payload:
                    continue
                row = json.loads(payload)
                if not isinstance(row, dict):
                    raise ValueError(f"Line {idx} in {path} is not an object")
                rows.append(row)
        return rows

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("items"), list):
            return [item for item in payload["items"] if isinstance(item, dict)]
        return [payload]
    raise ValueError(f"Unsupported input content in {path}")


def _get_path(data: dict[str, Any], dotted: str) -> Any:
    current: Any = data
    for key in dotted.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        for key in ("text", "answer", "response", "output"):
            nested = value.get(key)
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
        return json.dumps(value, ensure_ascii=False)
    return ""


def _normalize_passages(value: Any, fallback_title: str = "lettucedetect_context") -> list[dict[str, str]]:
    passages: list[dict[str, str]] = []
    if isinstance(value, list):
        for idx, item in enumerate(value):
            if not isinstance(item, dict):
                continue
            text = _as_text(item.get("text"))
            if not text:
                continue
            title = _as_text(item.get("title") or item.get("doc_id")) or fallback_title
            passages.append({"id": str(idx + 1), "title": title, "text": text})
    elif isinstance(value, str) and value.strip():
        passages.append({"id": "1", "title": fallback_title, "text": value.strip()})
    return passages


def _normalize_spans(value: Any, confidence_threshold: float) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    if not isinstance(value, list):
        return spans

    for item in value:
        if not isinstance(item, dict):
            continue
        start = item.get("start")
        end = item.get("end")
        confidence = item.get("confidence", item.get("hallucination_score", 1.0))

        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if end <= start:
            continue
        if isinstance(confidence, (int, float)) and float(confidence) < confidence_threshold:
            continue
        spans.append((start, end))

    return spans


def _sentence_spans(text: str) -> list[tuple[str, int, int]]:
    if not text:
        return []

    sentences: list[str]
    try:
        sentences = sent_tokenize(text)
    except Exception:
        # Fallback when punkt resources are unavailable in ad-hoc local runs.
        sentences = [s for s in text.split(".") if s.strip()]
        sentences = [f"{s.strip()}." for s in sentences]

    spans: list[tuple[str, int, int]] = []
    cursor = 0
    for sentence in sentences:
        candidate = sentence.strip()
        if not candidate:
            continue
        start = text.find(candidate, cursor)
        if start < 0:
            start = text.find(candidate)
            if start < 0:
                continue
        end = start + len(candidate)
        spans.append((candidate, start, end))
        cursor = end
    return spans


def _has_overlap(sent_start: int, sent_end: int, spans: list[tuple[int, int]]) -> bool:
    for span_start, span_end in spans:
        if sent_start < span_end and sent_end > span_start:
            return True
    return False


def _build_citation_suffix(passages: list[dict[str, str]]) -> str:
    refs: list[str] = []
    for idx, passage in enumerate(passages, start=1):
        pid = _as_text(passage.get("id")) or str(idx)
        refs.append(f"[{pid}]")
    return "".join(refs)


def _inject_span_citations(
    response: str,
    spans_raw: Any,
    passages: list[dict[str, str]],
    confidence_threshold: float,
) -> str:
    if not response:
        return response

    sentence_units = _sentence_spans(response)
    if not sentence_units:
        return response

    hallucinated_spans = _normalize_spans(spans_raw, confidence_threshold=confidence_threshold)
    citation_suffix = _build_citation_suffix(passages)
    if not citation_suffix:
        return response

    rewritten: list[str] = []
    for sentence, sent_start, sent_end in sentence_units:
        sentence_text = sentence.strip()
        if not sentence_text:
            continue
        hallucinated = _has_overlap(sent_start, sent_end, hallucinated_spans)
        if hallucinated:
            rewritten.append(sentence_text)
        else:
            rewritten.append(f"{sentence_text} {citation_suffix}")
    return " ".join(rewritten)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LettuceDetect output to CiteEval system-eval JSON")
    parser.add_argument("--input", required=True, help="LettuceDetect raw output (.json/.jsonl)")
    parser.add_argument("--output", required=True, help="CiteEval system-eval JSON output path")
    parser.add_argument("--id-key", default="id", help="Dotted key path for sample id")
    parser.add_argument("--query-key", default="query", help="Dotted key path for query/question")
    parser.add_argument("--pred-key", default="response", help="Dotted key path for predicted answer text")
    parser.add_argument("--spans-key", default="spans", help="Dotted key path for hallucination spans")
    parser.add_argument("--passages-key", default="passages", help="Dotted key path for passage list/text")
    parser.add_argument(
        "--fallback-context-key",
        default="source_info",
        help="Dotted key path for source text if passages-key is empty",
    )
    parser.add_argument("--strict", action="store_true", help="Drop rows with empty pred or passages")
    parser.add_argument(
        "--use-span-citations",
        action="store_true",
        help="Inject citations for non-hallucinated sentences based on detector spans",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum span confidence to mark a sentence as hallucinated",
    )
    parser.add_argument("--report-json", default=None, help="Optional path to write conversion stats JSON")
    return parser.parse_args()


def _convert_row(
    row: dict[str, Any],
    idx: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], bool, bool]:
    row_id = _as_text(_get_path(row, args.id_key)) or f"sample_{idx + 1}"
    query = _as_text(_get_path(row, args.query_key))
    pred = _as_text(_get_path(row, args.pred_key))
    passages = _normalize_passages(_get_path(row, args.passages_key))
    spans = _get_path(row, args.spans_key)
    if not passages:
        passages = _normalize_passages(_get_path(row, args.fallback_context_key))

    if args.use_span_citations and pred:
        pred = _inject_span_citations(
            response=pred,
            spans_raw=spans,
            passages=passages,
            confidence_threshold=args.confidence_threshold,
        )

    missing_pred = not bool(pred)
    missing_passages = not bool(passages)
    converted = {"id": row_id, "query": query, "passages": passages, "pred": pred}
    return converted, missing_pred, missing_passages


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    rows = _load_rows(input_path)
    converted: list[dict[str, Any]] = []
    skipped_empty_pred = 0
    skipped_empty_passages = 0

    for idx, row in enumerate(rows):
        converted_row, missing_pred, missing_passages = _convert_row(row=row, idx=idx, args=args)

        if missing_pred:
            skipped_empty_pred += 1
            if args.strict:
                continue
        if missing_passages:
            skipped_empty_passages += 1
            if args.strict:
                continue

        converted.append(converted_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(converted, handle, indent=2, ensure_ascii=False)

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "citation_mode": "span_injected" if args.use_span_citations else "original",
        "confidence_threshold": args.confidence_threshold,
        "total_input": len(rows),
        "total_output": len(converted),
        "skipped_empty_pred": skipped_empty_pred,
        "skipped_empty_passages": skipped_empty_passages,
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
