"""Run LettuceDetect pipeline with converter integration.

Pipeline stages:
1) Optional upstream conversion: CiteBench metric_eval -> LettuceDetect input JSON
2) LettuceDetect pretrained inference (span output)
3) Downstream conversion: LettuceDetect raw output -> CiteEval system input
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_command(command: list[str], cwd: Path, step: str, dry_run: bool) -> subprocess.CompletedProcess[str] | None:
    printable = " ".join(command)
    print(f"[{step}] {printable}")
    if dry_run:
        return None

    proc = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Step '{step}' failed with exit code {proc.returncode}.\n"
            f"Command: {printable}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )
    if proc.stdout.strip():
        print(proc.stdout.strip())
    return proc


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = line.strip()
                if not payload:
                    continue
                row = json.loads(payload)
                if isinstance(row, dict):
                    rows.append(row)
        return rows

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list JSON at {path}")
    return [item for item in payload if isinstance(item, dict)]


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return _to_jsonable(value.dict())
    if hasattr(value, "__dict__"):
        return _to_jsonable(vars(value))
    return str(value)


def _extract_contexts(row: dict[str, Any]) -> list[str]:
    passages = row.get("passages", [])
    contexts: list[str] = []
    if isinstance(passages, list):
        for item in passages:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if text:
                contexts.append(text)
    if contexts:
        return contexts

    context = str(row.get("context", "")).strip()
    if context:
        return [context]
    return []


def _build_detector(model_path: str, lang: str | None, trust_remote_code: bool) -> Any:
    try:
        from lettucedetect.models.inference import HallucinationDetector  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            "lettucedetect is not available. Install with 'pip install lettucedetect -U' "
            "or add it to your environment before running this script."
        ) from exc

    init_kwargs: dict[str, Any] = {
        "method": "transformer",
        "model_path": model_path,
    }
    if lang:
        init_kwargs["lang"] = lang
    if trust_remote_code:
        init_kwargs["trust_remote_code"] = True
    return HallucinationDetector(**init_kwargs)


def _build_missing_input_record(sample_id: str, query: str, answer: str, passages: Any) -> dict[str, Any]:
    return {
        "id": sample_id,
        "query": query,
        "response": answer,
        "passages": passages if isinstance(passages, list) else [],
        "spans": [],
        "error": "missing_query_or_answer_or_context",
    }


def _predict_row(detector: Any, sample_id: str, query: str, answer: str, passages: Any, contexts: list[str], model_path: str) -> dict[str, Any]:
    predictions = detector.predict(
        context=contexts,
        question=query,
        answer=answer,
        output_format="spans",
    )
    spans = _to_jsonable(predictions)
    return {
        "id": sample_id,
        "query": query,
        "response": answer,
        "passages": passages if isinstance(passages, list) else [],
        "spans": spans,
        "model_path": model_path,
    }


def _build_error_record(sample_id: str, query: str, answer: str, passages: Any, error: Exception) -> dict[str, Any]:
    return {
        "id": sample_id,
        "query": query,
        "response": answer,
        "passages": passages if isinstance(passages, list) else [],
        "spans": [],
        "error": str(error),
    }


def _run_inference(
    input_file: Path,
    raw_output_file: Path,
    model_path: str,
    lang: str | None,
    trust_remote_code: bool,
    max_samples: int | None,
    dry_run: bool,
) -> dict[str, Any]:
    if dry_run:
        print(f"[inference] dry-run: would read {input_file} and write {raw_output_file}")
        return {"processed": 0, "errors": 0}

    rows = _load_rows(input_file)
    if max_samples is not None:
        rows = rows[:max_samples]

    detector = _build_detector(model_path=model_path, lang=lang, trust_remote_code=trust_remote_code)

    outputs: list[dict[str, Any]] = []
    errors = 0

    for idx, row in enumerate(rows, start=1):
        sample_id = str(row.get("id", f"sample_{idx}")).strip() or f"sample_{idx}"
        query = str(row.get("query", "")).strip()
        answer = str(row.get("response", "")).strip()
        passages = row.get("passages", [])
        contexts = _extract_contexts(row)

        if not query or not answer or not contexts:
            outputs.append(_build_missing_input_record(sample_id, query, answer, passages))
            errors += 1
            continue

        try:
            outputs.append(
                _predict_row(
                    detector=detector,
                    sample_id=sample_id,
                    query=query,
                    answer=answer,
                    passages=passages,
                    contexts=contexts,
                    model_path=model_path,
                )
            )
        except Exception as exc:
            outputs.append(_build_error_record(sample_id, query, answer, passages, exc))
            errors += 1

    raw_output_file.parent.mkdir(parents=True, exist_ok=True)
    with raw_output_file.open("w", encoding="utf-8") as handle:
        json.dump(outputs, handle, indent=2, ensure_ascii=False)

    return {"processed": len(outputs), "errors": errors}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LettuceDetect pipeline with converter integration")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source-metric-file", help="Path to citebench.metric_dev/test source file")
    source_group.add_argument("--preconverted-input", help="Path to preconverted LettuceDetect input JSON/JSONL")

    parser.add_argument("--metric-split", choices=["dev", "test"], default="test")
    parser.add_argument("--model-path", default="KRLabsOrg/lettucedect-base-modernbert-en-v1")
    parser.add_argument("--lang", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--include-flat-context", action="store_true")
    parser.add_argument("--output-dir", default=None)

    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve_output_dir(output_dir_arg: str | None) -> Path:
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir_arg:
        return Path(output_dir_arg).resolve()
    return (PROJECT_ROOT / "outputs" / "lettucedetect_pipeline" / run_stamp).resolve()


def _build_artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "lettuce_input": output_dir / "lettucedetect_input.json",
        "upstream_report": output_dir / "upstream_conversion_report.json",
        "aligned_ids": output_dir / "aligned_ids.json",
        "raw_predictions": output_dir / "lettucedetect_raw_predictions.json",
        "system_eval_output": output_dir / "lettucedetect_system_eval.json",
        "downstream_report": output_dir / "downstream_conversion_report.json",
    }


def _run_upstream_conversion(args: argparse.Namespace, artifact_paths: dict[str, Path]) -> Path:
    if not args.source_metric_file:
        return Path(args.preconverted_input).resolve()

    convert_cmd = [
        sys.executable,
        "scripts/convert_citebench_metric_to_lettucedetect.py",
        "--input",
        str(Path(args.source_metric_file).resolve()),
        "--output",
        str(artifact_paths["lettuce_input"]),
        "--output-format",
        "json",
        "--report-json",
        str(artifact_paths["upstream_report"]),
        "--aligned-ids-output",
        str(artifact_paths["aligned_ids"]),
    ]
    if args.max_samples is not None:
        convert_cmd.extend(["--max-samples", str(args.max_samples)])
    if args.strict:
        convert_cmd.append("--strict")
    if args.include_flat_context:
        convert_cmd.append("--include-flat-context")

    _run_command(convert_cmd, cwd=PROJECT_ROOT, step="upstream.convert_metric_to_lettucedetect", dry_run=args.dry_run)
    return artifact_paths["lettuce_input"]


def _run_downstream_conversion(args: argparse.Namespace, raw_predictions: Path, artifact_paths: dict[str, Path]) -> None:
    convert_downstream_cmd = [
        sys.executable,
        "scripts/convert_lettucedetect_to_citeeval.py",
        "--input",
        str(raw_predictions),
        "--output",
        str(artifact_paths["system_eval_output"]),
        "--id-key",
        "id",
        "--query-key",
        "query",
        "--pred-key",
        "response",
        "--passages-key",
        "passages",
        "--report-json",
        str(artifact_paths["downstream_report"]),
    ]
    if args.strict:
        convert_downstream_cmd.append("--strict")

    _run_command(convert_downstream_cmd, cwd=PROJECT_ROOT, step="downstream.convert_to_citeeval", dry_run=args.dry_run)


def _write_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    lettuce_input: Path,
    artifact_paths: dict[str, Path],
    inference_stats: dict[str, Any],
) -> None:
    manifest = {
        "output_dir": str(output_dir),
        "source_metric_file": str(Path(args.source_metric_file).resolve()) if args.source_metric_file else None,
        "preconverted_input": str(Path(args.preconverted_input).resolve()) if args.preconverted_input else None,
        "lettucedetect_input": str(lettuce_input),
        "raw_predictions": str(artifact_paths["raw_predictions"]),
        "system_eval_output": str(artifact_paths["system_eval_output"]),
        "upstream_report": str(artifact_paths["upstream_report"]),
        "downstream_report": str(artifact_paths["downstream_report"]),
        "model_path": args.model_path,
        "lang": args.lang,
        "strict": args.strict,
        "dry_run": args.dry_run,
        "inference_stats": inference_stats,
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


def main() -> int:
    args = parse_args()

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths = _build_artifact_paths(output_dir)
    lettuce_input = _run_upstream_conversion(args, artifact_paths)

    inference_stats = _run_inference(
        input_file=lettuce_input,
        raw_output_file=artifact_paths["raw_predictions"],
        model_path=args.model_path,
        lang=args.lang,
        trust_remote_code=args.trust_remote_code,
        max_samples=args.max_samples,
        dry_run=args.dry_run,
    )

    _run_downstream_conversion(args=args, raw_predictions=artifact_paths["raw_predictions"], artifact_paths=artifact_paths)

    _write_manifest(
        output_dir=output_dir,
        args=args,
        lettuce_input=lettuce_input,
        artifact_paths=artifact_paths,
        inference_stats=inference_stats,
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "system_eval_output": str(artifact_paths["system_eval_output"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
