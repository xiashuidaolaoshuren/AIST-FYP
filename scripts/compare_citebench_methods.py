"""Compare two CiteBench system outputs under a controlled protocol.

This script aligns two method outputs by sample id, optionally truncates to the
same first N aligned samples, runs CiteEval system-track evaluation for each,
and writes a side-by-side summary.

Intended use for controlled (gold/oracle-context) comparison, e.g.:
- official RAGTruth baseline adapted output
- LettuceDetect adapted output

Expected input schema per method file:
[
  {
    "id": "sample_id",
    "query": "...",
    "passages": [{"text": "...", "title": "..."}, ...],
    "pred": "answer with citations"
  }
]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MethodSpec:
    name: str
    input_path: Path


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list JSON in {path}, got {type(payload).__name__}")
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Expected object at index {idx} in {path}")
        out.append(item)
    return out


def _index_by_id(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("id", "")).strip()
        if not sample_id:
            raise ValueError(f"{label} row missing non-empty 'id': {row}")
        if sample_id in indexed:
            raise ValueError(f"Duplicate id in {label}: {sample_id}")
        indexed[sample_id] = row
    return indexed


def _aligned_rows(
    left_rows: list[dict[str, Any]],
    right_index: dict[str, dict[str, Any]],
    *,
    max_samples: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    aligned_left: list[dict[str, Any]] = []
    aligned_right: list[dict[str, Any]] = []
    aligned_ids: list[str] = []

    for row in left_rows:
        sample_id = str(row["id"])
        other = right_index.get(sample_id)
        if other is None:
            continue
        aligned_left.append(row)
        aligned_right.append(other)
        aligned_ids.append(sample_id)
        if max_samples is not None and len(aligned_left) >= max_samples:
            break

    return aligned_left, aligned_right, aligned_ids


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _run(command: list[str], cwd: Path, env: dict[str, str]) -> tuple[str, str]:
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(command)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )
    return proc.stdout, proc.stderr


def _module_output_file(output_dir: Path, response_output_file: Path, version: str, module: str, model_name: str) -> Path:
    return output_dir / f"{response_output_file.name}.{version}.{module}.{model_name}.out"


def _parse_evaluate_system_stdout(stdout: str) -> dict[str, float]:
    keys = ("statement_rating", "response_rating", "length", "density")
    out: dict[str, float] = {}
    for key in keys:
        match = re.search(rf"{key}:\s*([0-9]+(?:\.[0-9]+)?)", stdout)
        if not match:
            raise ValueError(f"Could not parse '{key}' from evaluate_system output")
        out[key] = float(match.group(1))
    return out


def _evaluate_method(
    *,
    project_root: Path,
    citeeval_src: Path,
    method_name: str,
    system_input: Path,
    context_source: str,
    provider: str,
    model_name: str,
    version: str,
    modules: str,
    n_threads: int,
    cited_only: bool,
) -> dict[str, float]:
    command = [
        sys.executable,
        "scripts/run_citebench_eval.py",
        "--evaluation-role",
        "mitigation",
        "--track",
        "system",
        "--system-input",
        str(system_input),
        "--context-source",
        context_source,
        "--provider",
        provider,
        "--model-name",
        model_name,
        "--version",
        version,
        "--modules",
        modules,
        "--n-threads",
        str(n_threads),
    ]
    if cited_only:
        command.append("--cited-only")

    env = os.environ.copy()
    _run(command, cwd=project_root, env=env)

    citeeval_input = system_input.with_suffix(".citeeval")
    if not citeeval_input.exists():
        raise FileNotFoundError(f"Converted .citeeval file not found for {method_name}: {citeeval_input}")

    output_root = project_root / "benchmark" / "CiteEval" / "data" / "system_eval_outputs"
    cr_iter = _module_output_file(output_root, citeeval_input, version, "cr_itercoe", model_name)
    cr_edit = _module_output_file(output_root, citeeval_input, version, "cr_editdist", model_name)

    for expected in (cr_iter, cr_edit):
        if not expected.exists():
            raise FileNotFoundError(f"Expected CiteEval output missing for {method_name}: {expected}")

    eval_cmd = [
        sys.executable,
        "-m",
        "scripts.evaluate_system",
        "--system_output",
        str(citeeval_input),
        "--metric_output",
        f"{cr_iter},{cr_edit}",
    ]
    if cited_only:
        eval_cmd.append("--cited")

    env_eval = os.environ.copy()
    citeeval_root = project_root / "benchmark" / "CiteEval"
    extra_pythonpath = os.pathsep.join([str(citeeval_root), str(citeeval_src)])
    existing = env_eval.get("PYTHONPATH", "")
    env_eval["PYTHONPATH"] = f"{existing}{os.pathsep}{extra_pythonpath}" if existing else extra_pythonpath

    stdout, _ = _run(eval_cmd, cwd=citeeval_src, env=env_eval)
    return _parse_evaluate_system_stdout(stdout)


def _write_summary_md(path: Path, metrics: dict[str, dict[str, float]], left_name: str, right_name: str) -> None:
    left = metrics[left_name]
    right = metrics[right_name]

    lines = [
        "# Controlled CiteBench Comparison Summary",
        "",
        "| Method | Statement Rating | Response Rating | Length | Density |",
        "|---|---:|---:|---:|---:|",
        f"| {left_name} | {left['statement_rating']:.4f} | {left['response_rating']:.4f} | {left['length']:.2f} | {left['density']:.4f} |",
        f"| {right_name} | {right['statement_rating']:.4f} | {right['response_rating']:.4f} | {right['length']:.2f} | {right['density']:.4f} |",
        "",
        "## Delta",
        "",
        f"- `statement_rating`: {right['statement_rating'] - left['statement_rating']:+.4f} ({right_name} - {left_name})",
        f"- `response_rating`: {right['response_rating'] - left['response_rating']:+.4f} ({right_name} - {left_name})",
        f"- `length`: {right['length'] - left['length']:+.4f} ({right_name} - {left_name})",
        f"- `density`: {right['density'] - left['density']:+.4f} ({right_name} - {left_name})",
    ]
    _write_json(path.with_suffix(".json"), {"metrics": metrics})
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two method outputs on controlled CiteBench evaluation.")
    parser.add_argument("--left-name", type=str, default="ragtruth_official")
    parser.add_argument("--left-input", type=str, required=True, help="Path to left method CiteEval system-input JSON")
    parser.add_argument("--right-name", type=str, default="lettucedetect")
    parser.add_argument("--right-input", type=str, required=True, help="Path to right method CiteEval system-input JSON")
    parser.add_argument("--context-source", type=str, choices=["oracle", "retrieval"], default="oracle")
    parser.add_argument("--provider", type=str, choices=["openai", "deepseek"], default="deepseek")
    parser.add_argument("--model-name", type=str, default="deepseek-chat")
    parser.add_argument("--version", type=str, default="citeeval-auto-12272024")
    parser.add_argument("--modules", type=str, default="ca,ce,cr_itercoe,cr_editdist")
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--cited-only", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to outputs/citebench_controlled_compare/<timestamp>",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    project_root = Path(__file__).resolve().parents[1]
    citeeval_src = project_root / "benchmark" / "CiteEval" / "src"

    left_spec = MethodSpec(args.left_name, Path(args.left_input).resolve())
    right_spec = MethodSpec(args.right_name, Path(args.right_input).resolve())

    for spec in (left_spec, right_spec):
        if not spec.input_path.exists():
            raise FileNotFoundError(f"Input file for {spec.name} not found: {spec.input_path}")

    left_rows = _load_json_list(left_spec.input_path)
    right_rows = _load_json_list(right_spec.input_path)

    right_index = _index_by_id(right_rows, right_spec.name)
    aligned_left, aligned_right, aligned_ids = _aligned_rows(
        left_rows,
        right_index,
        max_samples=args.max_samples,
    )

    if not aligned_ids:
        raise ValueError("No overlapping sample IDs between the two method input files")

    missing_from_right = len(left_rows) - len(aligned_left)
    missing_from_left = len(right_rows) - len(aligned_right)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else (project_root / "outputs" / "citebench_controlled_compare" / timestamp)
    inputs_dir = output_dir / "system_inputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    left_aligned_path = inputs_dir / f"{left_spec.name}.json"
    right_aligned_path = inputs_dir / f"{right_spec.name}.json"
    _write_json(left_aligned_path, aligned_left)
    _write_json(right_aligned_path, aligned_right)

    run_metadata = {
        "timestamp": timestamp,
        "context_source": args.context_source,
        "provider": args.provider,
        "model_name": args.model_name,
        "version": args.version,
        "modules": args.modules,
        "n_threads": args.n_threads,
        "cited_only": args.cited_only,
        "max_samples": args.max_samples,
        "left": {"name": left_spec.name, "source": str(left_spec.input_path), "aligned_input": str(left_aligned_path)},
        "right": {"name": right_spec.name, "source": str(right_spec.input_path), "aligned_input": str(right_aligned_path)},
        "alignment": {
            "aligned_count": len(aligned_ids),
            "missing_from_right": missing_from_right,
            "missing_from_left": missing_from_left,
            "aligned_ids_file": str((output_dir / "aligned_ids.json")),
        },
    }
    _write_json(output_dir / "aligned_ids.json", aligned_ids)
    _write_json(output_dir / "run_metadata.json", run_metadata)

    left_metrics = _evaluate_method(
        project_root=project_root,
        citeeval_src=citeeval_src,
        method_name=left_spec.name,
        system_input=left_aligned_path,
        context_source=args.context_source,
        provider=args.provider,
        model_name=args.model_name,
        version=args.version,
        modules=args.modules,
        n_threads=args.n_threads,
        cited_only=args.cited_only,
    )
    right_metrics = _evaluate_method(
        project_root=project_root,
        citeeval_src=citeeval_src,
        method_name=right_spec.name,
        system_input=right_aligned_path,
        context_source=args.context_source,
        provider=args.provider,
        model_name=args.model_name,
        version=args.version,
        modules=args.modules,
        n_threads=args.n_threads,
        cited_only=args.cited_only,
    )

    metrics_payload = {
        left_spec.name: left_metrics,
        right_spec.name: right_metrics,
    }
    _write_json(output_dir / "summary.json", {"metadata": run_metadata, "metrics": metrics_payload})
    _write_summary_md(output_dir / "summary.md", metrics_payload, left_spec.name, right_spec.name)

    print("Controlled comparison completed.")
    print(f"Output directory: {output_dir}")
    print(f"Aligned samples: {len(aligned_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
