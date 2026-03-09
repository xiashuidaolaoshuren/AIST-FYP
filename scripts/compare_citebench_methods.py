"""Run controlled CiteBench system-track comparison for two methods.

Expected inputs are two CiteEval system-eval JSON files (same schema):
- RAGTruth baseline adapted output
- LettuceDetect adapted output

The script aligns sample IDs, evaluates both with identical settings, and writes:
- aligned input files
- per-method artifact summaries
- comparison summary JSON with deltas
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_OUTPUT_DIR = PROJECT_ROOT / "benchmark" / "CiteEval" / "data" / "system_eval_outputs"


def _load_list_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list JSON at {path}")
    return [item for item in payload if isinstance(item, dict)]


def _index_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("id", "")).strip()
        if not row_id:
            continue
        indexed[row_id] = row
    return indexed


def _align_rows(
    ragtruth_rows: list[dict[str, Any]],
    lettuce_rows: list[dict[str, Any]],
    max_samples: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    ragtruth_map = _index_by_id(ragtruth_rows)
    lettuce_map = _index_by_id(lettuce_rows)

    common_ids = sorted(set(ragtruth_map.keys()) & set(lettuce_map.keys()))
    if max_samples is not None:
        common_ids = common_ids[:max_samples]

    aligned_ragtruth = [ragtruth_map[row_id] for row_id in common_ids]
    aligned_lettuce = [lettuce_map[row_id] for row_id in common_ids]
    return aligned_ragtruth, aligned_lettuce, common_ids


def _run_eval(
    input_file: Path,
    provider: str,
    model_name: str,
    modules: str,
    version: str,
    context_source: str,
    dry_run: bool,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "scripts/run_citebench_eval.py",
        "--track",
        "system",
        "--system-input",
        str(input_file),
        "--provider",
        provider,
        "--model-name",
        model_name,
        "--modules",
        modules,
        "--version",
        version,
        "--context-source",
        context_source,
    ]
    if dry_run:
        command.append("--dry-run")

    return subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )


def _read_metric_rows(out_file: Path) -> list[dict[str, Any]]:
    with out_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def _collect_answer_ratings(rows: list[dict[str, Any]]) -> list[float]:
    ratings: list[float] = []
    for row in rows:
        answer_rating = row.get("answer_rating")
        if isinstance(answer_rating, (int, float)):
            ratings.append(float(answer_rating))
    return ratings


def _collect_sentence_ratings(rows: list[dict[str, Any]]) -> list[float]:
    ratings: list[float] = []
    for row in rows:
        sent_map = row.get("sent_id2rating")
        if not isinstance(sent_map, dict):
            continue
        for value in sent_map.values():
            if isinstance(value, (int, float)):
                ratings.append(float(value))
    return ratings


def _extract_metric_scores(out_file: Path) -> dict[str, Any]:
    rows = _read_metric_rows(out_file)
    answer_ratings = _collect_answer_ratings(rows)
    sentence_ratings = _collect_sentence_ratings(rows)

    result: dict[str, Any] = {
        "num_rows": len(rows),
        "num_answer_ratings": len(answer_ratings),
        "num_sentence_ratings": len(sentence_ratings),
    }
    if answer_ratings:
        result["mean_answer_rating"] = mean(answer_ratings)
    if sentence_ratings:
        result["mean_sentence_rating"] = mean(sentence_ratings)
    return result


def _collect_method_metrics(input_file: Path, modules: str, version: str, model_name: str) -> dict[str, Any]:
    citeeval_name = input_file.with_suffix(".citeeval").name
    metrics: dict[str, Any] = {}

    for module in [mod.strip() for mod in modules.split(",") if mod.strip()]:
        out_path = SYSTEM_OUTPUT_DIR / f"{citeeval_name}.{version}.{module}.{model_name}.out"
        if out_path.exists():
            metrics[module] = {
                "out_file": str(out_path),
                "summary": _extract_metric_scores(out_path),
            }
        else:
            metrics[module] = {
                "out_file": str(out_path),
                "summary": None,
            }
    return metrics


def _compute_deltas(ragtruth_metrics: dict[str, Any], lettuce_metrics: dict[str, Any]) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    shared_modules = sorted(set(ragtruth_metrics.keys()) & set(lettuce_metrics.keys()))
    for module in shared_modules:
        rag = ragtruth_metrics[module].get("summary") or {}
        let = lettuce_metrics[module].get("summary") or {}
        module_delta: dict[str, Any] = {}
        for key in ("mean_answer_rating", "mean_sentence_rating"):
            if key in rag and key in let:
                module_delta[key] = let[key] - rag[key]
        deltas[module] = module_delta
    return deltas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare RAGTruth baseline vs LettuceDetect on CiteBench system track")
    parser.add_argument("--ragtruth-input", required=True, help="CiteEval-formatted JSON for RAGTruth baseline")
    parser.add_argument("--lettuce-input", required=True, help="CiteEval-formatted JSON for LettuceDetect")
    parser.add_argument("--output-dir", default=None, help="Output directory for aligned inputs and summaries")
    parser.add_argument("--provider", choices=["openai", "deepseek"], default="deepseek")
    parser.add_argument("--model-name", default="deepseek-chat")
    parser.add_argument("--modules", default="ca,ce,cr_itercoe,cr_editdist")
    parser.add_argument("--version", default="citeeval-auto-12272024")
    parser.add_argument("--context-source", choices=["retrieval", "oracle"], default="oracle")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (PROJECT_ROOT / "outputs" / "method_comparison" / run_stamp).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    ragtruth_rows = _load_list_json(Path(args.ragtruth_input).resolve())
    lettuce_rows = _load_list_json(Path(args.lettuce_input).resolve())
    aligned_ragtruth, aligned_lettuce, common_ids = _align_rows(
        ragtruth_rows=ragtruth_rows,
        lettuce_rows=lettuce_rows,
        max_samples=args.max_samples,
    )

    if not common_ids:
        raise ValueError("No overlapping IDs between ragtruth-input and lettuce-input")

    aligned_ragtruth_path = output_dir / "ragtruth_aligned.json"
    aligned_lettuce_path = output_dir / "lettucedetect_aligned.json"
    aligned_ids_path = output_dir / "aligned_ids.json"

    with aligned_ragtruth_path.open("w", encoding="utf-8") as handle:
        json.dump(aligned_ragtruth, handle, indent=2, ensure_ascii=False)
    with aligned_lettuce_path.open("w", encoding="utf-8") as handle:
        json.dump(aligned_lettuce, handle, indent=2, ensure_ascii=False)
    with aligned_ids_path.open("w", encoding="utf-8") as handle:
        json.dump(common_ids, handle, indent=2, ensure_ascii=False)

    ragtruth_proc = _run_eval(
        input_file=aligned_ragtruth_path,
        provider=args.provider,
        model_name=args.model_name,
        modules=args.modules,
        version=args.version,
        context_source=args.context_source,
        dry_run=args.dry_run,
    )
    lettuce_proc = _run_eval(
        input_file=aligned_lettuce_path,
        provider=args.provider,
        model_name=args.model_name,
        modules=args.modules,
        version=args.version,
        context_source=args.context_source,
        dry_run=args.dry_run,
    )

    ragtruth_metrics = _collect_method_metrics(
        input_file=aligned_ragtruth_path,
        modules=args.modules,
        version=args.version,
        model_name=args.model_name,
    )
    lettuce_metrics = _collect_method_metrics(
        input_file=aligned_lettuce_path,
        modules=args.modules,
        version=args.version,
        model_name=args.model_name,
    )

    summary = {
        "run": {
            "output_dir": str(output_dir),
            "provider": args.provider,
            "model_name": args.model_name,
            "modules": args.modules,
            "version": args.version,
            "context_source": args.context_source,
            "max_samples": args.max_samples,
            "aligned_count": len(common_ids),
            "dry_run": args.dry_run,
        },
        "commands": {
            "ragtruth_returncode": ragtruth_proc.returncode,
            "lettucedetect_returncode": lettuce_proc.returncode,
        },
        "method_metrics": {
            "ragtruth": ragtruth_metrics,
            "lettucedetect": lettuce_metrics,
        },
        "delta": _compute_deltas(ragtruth_metrics, lettuce_metrics),
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    with (output_dir / "run_logs.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "ragtruth_stdout": ragtruth_proc.stdout,
                "ragtruth_stderr": ragtruth_proc.stderr,
                "lettucedetect_stdout": lettuce_proc.stdout,
                "lettucedetect_stderr": lettuce_proc.stderr,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(json.dumps({"output_dir": str(output_dir), "aligned_count": len(common_ids)}, ensure_ascii=False))

    if ragtruth_proc.returncode != 0 or lettuce_proc.returncode != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
