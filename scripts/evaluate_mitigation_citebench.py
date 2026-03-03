"""
Run module-level evaluation variants on CiteBench system track.

This script automates a fair comparison by:
1. Creating temporary config variants (verifier-only, mitigation-only, full pipeline)
2. Generating CiteEval system-input JSON for each variant from identical queries
3. Running CiteBench/CiteEval system evaluation for each variant
4. Writing a delta summary report

Usage examples:
    # Smoke test on first 10 system-eval queries
    python scripts/evaluate_mitigation_citebench.py --max-samples 10

    # Custom output directory and model
    python scripts/evaluate_mitigation_citebench.py --output-dir outputs/mitigation_eval_citebench/run_01 --model-name deepseek-chat
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.citation.citation_formatter import CitationFormatter
from src.pipelines.baseline_rag import BaselineRAGPipeline
from src.utils.config import Config
from src.utils.data_structures import Claim, EvidenceChunk


@dataclass
class VariantRuntime:
    config: Config
    pipeline: BaselineRAGPipeline
    citation_formatter: CitationFormatter
    mitigation_enabled: bool


def _deep_update(target: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            target[key] = _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _variant_patch(name: str) -> dict[str, Any]:
    all_verifiers_enabled = {
        "verification": {
            "enabled": True,
            "modules": {
                "intrinsic": True,
                "grounded": True,
                "nli": True,
                "self_agreement": True,
            },
        }
    }

    all_mitigation_enabled = {
        "mitigation": {
            "enabled": True,
            "reranker": {"enabled": True},
            "filter": {"enabled": True},
            "reprompt": {"enabled": True},
        }
    }

    all_mitigation_disabled = {
        "mitigation": {
            "enabled": False,
            "reranker": {"enabled": False},
            "filter": {"enabled": False},
            "reprompt": {"enabled": False},
        }
    }

    if name == "baseline":
        return _deep_update(deepcopy(all_verifiers_enabled), deepcopy(all_mitigation_disabled))

    if name in {"full_pipeline", "mitigation_all"}:
        return _deep_update(deepcopy(all_verifiers_enabled), deepcopy(all_mitigation_enabled))

    if name == "verifier_intrinsic_only":
        return {
            "verification": {
                "enabled": True,
                "modules": {
                    "intrinsic": True,
                    "grounded": False,
                    "nli": False,
                    "self_agreement": False,
                },
            },
            **deepcopy(all_mitigation_disabled),
        }

    if name == "verifier_grounded_only":
        return {
            "verification": {
                "enabled": True,
                "modules": {
                    "intrinsic": False,
                    "grounded": True,
                    "nli": False,
                    "self_agreement": False,
                },
            },
            **deepcopy(all_mitigation_disabled),
        }

    if name == "verifier_nli_only":
        return {
            "verification": {
                "enabled": True,
                "modules": {
                    "intrinsic": False,
                    "grounded": False,
                    "nli": True,
                    "self_agreement": False,
                },
            },
            **deepcopy(all_mitigation_disabled),
        }

    if name == "verifier_self_agreement_only":
        return {
            "verification": {
                "enabled": True,
                "modules": {
                    "intrinsic": False,
                    "grounded": False,
                    "nli": False,
                    "self_agreement": True,
                },
            },
            **deepcopy(all_mitigation_disabled),
        }

    if name in {"mitigation_filter_only", "filter_only"}:
        return {
            **deepcopy(all_verifiers_enabled),
            "mitigation": {
                "enabled": True,
                "reranker": {"enabled": False},
                "filter": {"enabled": True},
                "reprompt": {"enabled": False},
            },
        }

    if name in {"mitigation_rerank_only", "rerank_only"}:
        return {
            **deepcopy(all_verifiers_enabled),
            "mitigation": {
                "enabled": True,
                "reranker": {"enabled": True},
                "filter": {"enabled": False},
                "reprompt": {"enabled": False},
            },
        }

    if name in {"mitigation_reprompt_only", "reprompt_only"}:
        return {
            **deepcopy(all_verifiers_enabled),
            "mitigation": {
                "enabled": True,
                "reranker": {"enabled": False},
                "filter": {"enabled": False},
                "reprompt": {"enabled": True},
            },
        }

    raise ValueError(f"Unknown variant: {name}")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} is not a YAML object.")
    return data


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _flatten_claims(claims_by_sub_answer: list[dict[str, Any]]) -> list[Claim]:
    claims: list[Claim] = []
    for group in claims_by_sub_answer:
        for claim in group.get("claims", []):
            if isinstance(claim, Claim):
                claims.append(claim)
            elif isinstance(claim, dict):
                claims.append(Claim(**claim))
    return claims


def _build_evidence_map(claim_evidence_pairs: list[dict[str, Any]]) -> dict[str, list[EvidenceChunk]]:
    evidence_map: dict[str, list[EvidenceChunk]] = {}
    for pair in claim_evidence_pairs:
        claim_id = pair.get("claim_id")
        evidence_spans = pair.get("evidence_spans", [])
        chunks: list[EvidenceChunk] = []
        for item in evidence_spans:
            if isinstance(item, EvidenceChunk):
                chunks.append(item)
            elif isinstance(item, dict):
                chunks.append(EvidenceChunk(**item))
        if claim_id:
            evidence_map[claim_id] = chunks
    return evidence_map


def _build_runtime(config_path: Path, strategy: str) -> VariantRuntime:
    config = Config(str(config_path))
    pipeline = BaselineRAGPipeline.from_config(config_path=str(config_path), strategy=strategy)
    citation_formatter = CitationFormatter(config)

    mitigation_cfg = config.get("mitigation", {})
    mitigation_enabled = bool(mitigation_cfg.get("enabled", False))

    return VariantRuntime(
        config=config,
        pipeline=pipeline,
        citation_formatter=citation_formatter,
        mitigation_enabled=mitigation_enabled,
    )


def _generate_system_input(
    *,
    runtime: VariantRuntime,
    source_queries: list[dict[str, Any]],
    output_path: Path,
) -> None:
    records: list[dict[str, Any]] = []

    for row in source_queries:
        sample_id = str(row.get("id", f"sample_{len(records) + 1}"))
        query = str(row.get("query", "")).strip()
        if not query:
            continue

        pipeline_output = runtime.pipeline.run(query)
        answer_text = (
            pipeline_output.get("response_after_mitigation", "")
            if runtime.mitigation_enabled
            else pipeline_output.get("draft_response", "")
        )
        if not answer_text:
            answer_text = pipeline_output.get("draft_response", "")

        mitigation_claims = pipeline_output.get("mitigation_claims", [])
        if mitigation_claims:
            claims = [Claim(**item) for item in mitigation_claims]
        else:
            claims = _flatten_claims(pipeline_output.get("claims_by_sub_answer", []))

        mitigation_evidence_map = pipeline_output.get("mitigation_evidence_map", {})
        if mitigation_evidence_map:
            evidence_map = {
                claim_id: [
                    item if isinstance(item, EvidenceChunk) else EvidenceChunk(**item)
                    for item in chunks
                ]
                for claim_id, chunks in mitigation_evidence_map.items()
            }
        else:
            evidence_map = _build_evidence_map(pipeline_output.get("claim_evidence_pairs", []))

        formatted_output = runtime.citation_formatter.format_with_citations(
            answer_text=answer_text,
            claims=claims,
            evidence_map=evidence_map,
        )
        citeeval_sample = runtime.citation_formatter.export_citeeval_format(
            query=query,
            formatted_output=formatted_output,
            answer_id=sample_id,
        )
        records.append(citeeval_sample)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def _run_system_eval(
    *,
    project_root: Path,
    system_input: Path,
    provider: str,
    model_name: str,
    version: str,
    modules: str,
    n_threads: int,
    cited_only: bool,
) -> None:
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
        "--version",
        version,
        "--modules",
        modules,
        "--n-threads",
        str(n_threads),
    ]
    if cited_only:
        command.append("--cited-only")

    proc = subprocess.run(
        command,
        cwd=str(project_root),
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "CiteBench system evaluation failed.\n"
            f"Command: {' '.join(command)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )


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


def _evaluate_system_summary(
    *,
    citeeval_root: Path,
    citeeval_src: Path,
    citeeval_input: Path,
    cr_iter_out: Path,
    cr_edit_out: Path,
    cited_only: bool,
) -> dict[str, float]:
    command = [
        sys.executable,
        "-m",
        "scripts.evaluate_system",
        "--system_output",
        str(citeeval_input),
        "--metric_output",
        f"{cr_iter_out},{cr_edit_out}",
    ]
    if cited_only:
        command.append("--cited")

    env = os.environ.copy()
    env["CITEEVAL_ROOT"] = str(citeeval_root)
    existing_pythonpath = env.get("PYTHONPATH", "")
    extra = os.pathsep.join([str(citeeval_root), str(citeeval_src)])
    env["PYTHONPATH"] = f"{existing_pythonpath}{os.pathsep}{extra}" if existing_pythonpath else extra

    proc = subprocess.run(
        command,
        cwd=str(citeeval_src),
        env=env,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to summarize CiteEval system outputs.\n"
            f"Command: {' '.join(command)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )
    return _parse_evaluate_system_stdout(proc.stdout)


def _write_summary(summary_path: Path, metrics: dict[str, dict[str, float]], baseline_name: str = "baseline") -> None:
    baseline = metrics[baseline_name]
    lines = [
        "# CiteBench Module Evaluation Summary",
        "",
        "| Variant | Statement Rating | Response Rating | Length | Density | ΔStatement vs Baseline | ΔResponse vs Baseline |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in metrics.items():
        lines.append(
            f"| {name} | {row['statement_rating']:.4f} | {row['response_rating']:.4f} | {row['length']:.2f} | {row['density']:.4f} | "
            f"{(row['statement_rating'] - baseline['statement_rating']):+.4f} | {(row['response_rating'] - baseline['response_rating']):+.4f} |"
        )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run verifier/mitigation/full-pipeline CiteBench system evaluation variants and summarize deltas."
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Base config file path")
    parser.add_argument("--strategy", type=str, default="validation", choices=["development", "validation", "production"])
    parser.add_argument("--system-source", type=str, default="benchmark/CiteEval/data/system_eval/system_eval_examples.json")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of source queries for smoke testing")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "baseline",
            "full_pipeline",
            "mitigation_filter_only",
            "mitigation_rerank_only",
            "mitigation_reprompt_only",
        ],
        choices=[
            "baseline",
            "full_pipeline",
            "mitigation_all",
            "verifier_intrinsic_only",
            "verifier_grounded_only",
            "verifier_nli_only",
            "verifier_self_agreement_only",
            "mitigation_filter_only",
            "mitigation_rerank_only",
            "mitigation_reprompt_only",
            "filter_only",
            "rerank_only",
            "reprompt_only",
        ],
    )
    parser.add_argument("--provider", type=str, default="deepseek", choices=["openai", "deepseek"])
    parser.add_argument("--model-name", type=str, default="deepseek-chat")
    parser.add_argument("--version", type=str, default="citeeval-auto-12272024")
    parser.add_argument("--modules", type=str, default="ca,ce,cr_itercoe,cr_editdist")
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--cited-only", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    project_root = Path(__file__).resolve().parents[1]
    base_config_path = (project_root / args.config).resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    source_path = (project_root / args.system_source).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"System source not found: {source_path}")

    source_rows = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError(f"Expected non-empty list JSON at {source_path}")
    if args.max_samples is not None:
        source_rows = source_rows[: args.max_samples]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else (project_root / "outputs" / "mitigation_eval_citebench" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_dir = output_dir / "configs"
    system_input_dir = output_dir / "system_inputs"

    base_config = _load_yaml(base_config_path)
    summary_metrics: dict[str, dict[str, float]] = {}

    citeeval_output_root = project_root / "benchmark" / "CiteEval" / "data" / "system_eval_outputs"
    citeeval_root = project_root / "benchmark" / "CiteEval"
    citeeval_src = project_root / "benchmark" / "CiteEval" / "src"

    for variant in args.variants:
        config_payload = _deep_update(deepcopy(base_config), _variant_patch(variant))
        variant_config_path = config_dir / f"config_{variant}.yaml"
        _write_yaml(variant_config_path, config_payload)

        runtime = _build_runtime(variant_config_path, args.strategy)

        system_input_json = system_input_dir / f"system_eval_{variant}.json"
        _generate_system_input(
            runtime=runtime,
            source_queries=source_rows,
            output_path=system_input_json,
        )

        _run_system_eval(
            project_root=project_root,
            system_input=system_input_json,
            provider=args.provider,
            model_name=args.model_name,
            version=args.version,
            modules=args.modules,
            n_threads=args.n_threads,
            cited_only=args.cited_only,
        )

        citeeval_input = system_input_json.with_suffix(".citeeval")
        cr_iter_out = _module_output_file(citeeval_output_root, citeeval_input, args.version, "cr_itercoe", args.model_name)
        cr_edit_out = _module_output_file(citeeval_output_root, citeeval_input, args.version, "cr_editdist", args.model_name)

        summary_metrics[variant] = _evaluate_system_summary(
            citeeval_root=citeeval_root,
            citeeval_src=citeeval_src,
            citeeval_input=citeeval_input,
            cr_iter_out=cr_iter_out,
            cr_edit_out=cr_edit_out,
            cited_only=args.cited_only,
        )

    if "baseline" not in summary_metrics:
        raise ValueError("`baseline` must be included in --variants for delta computation.")

    payload = {
        "metadata": {
            "timestamp": timestamp,
            "strategy": args.strategy,
            "system_source": str(source_path),
            "num_queries": len(source_rows),
            "provider": args.provider,
            "model_name": args.model_name,
            "version": args.version,
            "modules": args.modules,
            "n_threads": args.n_threads,
            "cited_only": args.cited_only,
            "variants": args.variants,
        },
        "metrics": summary_metrics,
    }

    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_md = output_dir / "summary.md"
    _write_summary(summary_md, summary_metrics)

    print("\nCiteBench mitigation evaluation completed.")
    print(f"Summary JSON: {summary_json}")
    print(f"Summary Markdown: {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
