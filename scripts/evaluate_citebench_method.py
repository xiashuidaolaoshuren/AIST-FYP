"""Evaluate a single method on CiteBench system track and write method-only metrics.

This script runs CiteEval system evaluation for one CiteEval-formatted system input,
then summarizes module outputs (no cross-method alignment or delta computation).
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


def _ensure_eval_runtime_dependencies() -> str | None:
    """Return a human-readable error message if runtime deps are unavailable."""
    try:
        import configobj  # noqa: F401
    except Exception as exc:
        return f"missing_dependency: configobj ({exc})"

    try:
        import nltk
        from nltk.data import find
    except Exception as exc:
        return f"missing_dependency: nltk ({exc})"

    try:
        find("tokenizers/punkt_tab/english/")
    except LookupError:
        # Try to self-heal in non-notebook CLI runs.
        try:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            find("tokenizers/punkt_tab/english/")
        except Exception as exc:
            return f"missing_nltk_resource: punkt_tab ({exc})"

    return None


def _extract_error_hint(stderr: str) -> str | None:
    lowered = (stderr or "").lower()
    if "punkt_tab" in lowered:
        return "missing_nltk_punkt_tab"
    if "configobj" in lowered:
        return "missing_configobj"
    return None


def _run_eval(
    system_input: Path,
    provider: str,
    model_name: str,
    modules: str,
    version: str,
    context_source: str,
    max_samples: int | None,
    dry_run: bool,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "scripts/run_citebench_eval.py",
        "--track",
        "system",
        "--system-input",
        str(system_input),
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
    if max_samples is not None:
        command.extend(["--max-examples", str(max_samples)])
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


def _extract_metric_scores(out_file: Path) -> dict[str, Any]:
    rows = _read_metric_rows(out_file)
    answer_ratings: list[float] = []
    sentence_ratings: list[float] = []

    for row in rows:
        answer_rating = row.get("answer_rating")
        if isinstance(answer_rating, (int, float)):
            answer_ratings.append(float(answer_rating))

        sent_map = row.get("sent_id2rating")
        if isinstance(sent_map, dict):
            for value in sent_map.values():
                if isinstance(value, (int, float)):
                    sentence_ratings.append(float(value))

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


def _collect_method_metrics(system_input: Path, modules: str, version: str, model_name: str) -> tuple[dict[str, Any], int | None]:
    citeeval_name = system_input.with_suffix(".citeeval").name
    metrics: dict[str, Any] = {}
    row_counts: list[int] = []

    for module in [mod.strip() for mod in modules.split(",") if mod.strip()]:
        out_path = SYSTEM_OUTPUT_DIR / f"{citeeval_name}.{version}.{module}.{model_name}.out"
        if out_path.exists():
            summary = _extract_metric_scores(out_path)
            metrics[module] = {
                "out_file": str(out_path),
                "summary": summary,
            }
            row_counts.append(int(summary.get("num_rows", 0)))
        else:
            metrics[module] = {
                "out_file": str(out_path),
                "summary": None,
            }

    evaluated_rows = max(row_counts) if row_counts else None
    return metrics, evaluated_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one method on CiteBench system track")
    parser.add_argument("--method-name", required=True, help="Method label for output metadata, e.g. ragtruth_baseline")
    parser.add_argument("--system-input", required=True, help="CiteEval-formatted JSON input for system track")
    parser.add_argument("--output-dir", default=None, help="Directory for summary and logs")
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
        else (PROJECT_ROOT / "outputs" / "method_eval" / args.method_name / run_stamp).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    system_input = Path(args.system_input).resolve()
    preflight_error = _ensure_eval_runtime_dependencies()
    if preflight_error:
        proc = subprocess.CompletedProcess(
            args=["preflight"],
            returncode=2,
            stdout="",
            stderr=f"CiteBench evaluation preflight failed: {preflight_error}",
        )
    else:
        proc = _run_eval(
            system_input=system_input,
            provider=args.provider,
            model_name=args.model_name,
            modules=args.modules,
            version=args.version,
            context_source=args.context_source,
            max_samples=args.max_samples,
            dry_run=args.dry_run,
        )

    error_hint = _extract_error_hint(proc.stderr)

    module_metrics, evaluated_rows = _collect_method_metrics(
        system_input=system_input,
        modules=args.modules,
        version=args.version,
        model_name=args.model_name,
    )

    summary = {
        "run": {
            "method": args.method_name,
            "output_dir": str(output_dir),
            "system_input": str(system_input),
            "provider": args.provider,
            "model_name": args.model_name,
            "modules": args.modules,
            "version": args.version,
            "context_source": args.context_source,
            "max_samples": args.max_samples,
            "dry_run": args.dry_run,
            "evaluated_rows": evaluated_rows,
        },
        "command_returncode": proc.returncode,
        "error_hint": error_hint,
        "module_metrics": module_metrics,
    }

    summary_path = output_dir / "summary.json"
    logs_path = output_dir / "run_logs.json"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    with logs_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(
        json.dumps(
            {
                "method": args.method_name,
                "summary_path": str(summary_path),
                "output_dir": str(output_dir),
                "evaluated_rows": evaluated_rows,
                "error_hint": error_hint,
            },
            ensure_ascii=False,
        )
    )

    if proc.returncode != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
