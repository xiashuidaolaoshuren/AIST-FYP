"""
Run paired baseline-vs-mitigation evaluation on CiteBench system track.

This script automates a fair comparison by:
1. Creating temporary config variants (baseline, mitigation_all)
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
from src.generation.claim_extractor import extract_claims
from src.mitigation.claim_filter import ClaimFilter
from src.mitigation.re_ranker import EvidenceReRanker
from src.mitigation.reprompt import RePrompter
from src.pipelines.baseline_rag import BaselineRAGPipeline
from src.utils.config import Config
from src.utils.data_structures import Claim, ClaimDecision, EvidenceChunk, VerifierSignal
from src.verification.rule_based_aggregator import RuleBasedAggregator
from src.verification.verifier_hub import VerifierHub


@dataclass
class VariantRuntime:
    config: Config
    pipeline: BaselineRAGPipeline
    verifier_hub: VerifierHub
    aggregator: RuleBasedAggregator
    citation_formatter: CitationFormatter
    claim_filter: ClaimFilter | None
    reranker: EvidenceReRanker | None
    reprompter: RePrompter | None
    mitigation_enabled: bool


def _deep_update(target: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            target[key] = _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _variant_patch(name: str) -> dict[str, Any]:
    if name == "baseline":
        return {
            "mitigation": {
                "enabled": False,
                "reranker": {"enabled": False},
                "filter": {"enabled": False},
                "reprompt": {"enabled": False},
            }
        }

    if name == "mitigation_all":
        return {
            "mitigation": {
                "enabled": True,
                "reranker": {"enabled": True},
                "filter": {"enabled": True},
                "reprompt": {"enabled": True},
            }
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


def _to_signal_map(signals: list[VerifierSignal]) -> dict[str, VerifierSignal]:
    signal_map: dict[str, VerifierSignal] = {}
    for signal in signals:
        signal_map[f"{signal.doc_id}#{signal.sent_id}"] = signal
    return signal_map


def _verify_and_decide(
    *,
    verifier_hub: VerifierHub,
    aggregator: RuleBasedAggregator,
    claims: list[Claim],
    evidence_map: dict[str, list[EvidenceChunk]],
    generation_metadata: dict[str, Any],
) -> tuple[list[VerifierSignal], list[ClaimDecision]]:
    signals: list[VerifierSignal] = []
    decisions: list[ClaimDecision] = []

    for claim in claims:
        evidence = evidence_map.get(claim.claim_id, [])
        if not evidence:
            continue

        signal = verifier_hub.verify_claim(claim, evidence, generation_metadata)
        if signal is None:
            continue

        signals.append(signal)
        decisions.append(aggregator.aggregate(signal))

    return signals, decisions


def _apply_mitigation(
    *,
    runtime: VariantRuntime,
    query: str,
    answer_text: str,
    claims: list[Claim],
    evidence_map: dict[str, list[EvidenceChunk]],
    generation_metadata: dict[str, Any],
) -> tuple[str, list[Claim], dict[str, list[EvidenceChunk]]]:
    signals, decisions = _verify_and_decide(
        verifier_hub=runtime.verifier_hub,
        aggregator=runtime.aggregator,
        claims=claims,
        evidence_map=evidence_map,
        generation_metadata=generation_metadata,
    )

    if runtime.reranker and runtime.reranker.enabled and signals:
        signal_map = _to_signal_map(signals)
        reranked_map: dict[str, list[EvidenceChunk]] = {}
        for claim in claims:
            chunks = evidence_map.get(claim.claim_id, [])
            if chunks:
                reranked_map[claim.claim_id] = runtime.reranker.rerank(chunks, signal_map)
        evidence_map = reranked_map or evidence_map
        signals, decisions = _verify_and_decide(
            verifier_hub=runtime.verifier_hub,
            aggregator=runtime.aggregator,
            claims=claims,
            evidence_map=evidence_map,
            generation_metadata=generation_metadata,
        )

    if runtime.reprompter and runtime.reprompter.enabled and decisions and claims:
        pooled_evidence: list[EvidenceChunk] = []
        for chunk_list in evidence_map.values():
            pooled_evidence.extend(chunk_list)
        if pooled_evidence:
            reprompt_result = runtime.reprompter.reprompt(
                query=query,
                answer=answer_text,
                decisions=decisions,
                evidence=pooled_evidence,
                claims=claims,
            )
            if reprompt_result.get("improved"):
                answer_text = reprompt_result.get("final_answer", answer_text)
                claims = extract_claims(answer_text, method="auto")
                default_evidence = pooled_evidence[:5]
                evidence_map = {
                    claim.claim_id: default_evidence
                    for claim in claims
                }
                signals, decisions = _verify_and_decide(
                    verifier_hub=runtime.verifier_hub,
                    aggregator=runtime.aggregator,
                    claims=claims,
                    evidence_map=evidence_map,
                    generation_metadata=generation_metadata,
                )

    if runtime.claim_filter and runtime.claim_filter.enabled and decisions and claims:
        answer_text, _ = runtime.claim_filter.filter_answer(
            answer_text=answer_text,
            claims=claims,
            decisions=decisions,
        )

    return answer_text, claims, evidence_map


def _build_runtime(config_path: Path, strategy: str) -> VariantRuntime:
    config = Config(str(config_path))
    pipeline = BaselineRAGPipeline.from_config(config_path=str(config_path), strategy=strategy)
    verifier_hub = VerifierHub(config, pipeline.generator)
    aggregator = RuleBasedAggregator(config)
    citation_formatter = CitationFormatter(config)

    mitigation_cfg = config.get("mitigation", {})
    mitigation_enabled = bool(mitigation_cfg.get("enabled", False))

    claim_filter = ClaimFilter(config) if mitigation_enabled and mitigation_cfg.get("filter", {}).get("enabled", False) else None
    reranker = EvidenceReRanker(config) if mitigation_enabled and mitigation_cfg.get("reranker", {}).get("enabled", False) else None
    reprompter = RePrompter(config, pipeline.generator) if mitigation_enabled and mitigation_cfg.get("reprompt", {}).get("enabled", False) else None

    return VariantRuntime(
        config=config,
        pipeline=pipeline,
        verifier_hub=verifier_hub,
        aggregator=aggregator,
        citation_formatter=citation_formatter,
        claim_filter=claim_filter,
        reranker=reranker,
        reprompter=reprompter,
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
        answer_text = pipeline_output.get("draft_response", "")
        claims = _flatten_claims(pipeline_output.get("claims_by_sub_answer", []))
        evidence_map = _build_evidence_map(pipeline_output.get("claim_evidence_pairs", []))
        generation_metadata = pipeline_output.get("generator_metadata", {})

        if runtime.mitigation_enabled and claims and evidence_map:
            answer_text, claims, evidence_map = _apply_mitigation(
                runtime=runtime,
                query=query,
                answer_text=answer_text,
                claims=claims,
                evidence_map=evidence_map,
                generation_metadata=generation_metadata,
            )

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
        "# CiteBench Mitigation Evaluation Summary",
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
        description="Run paired baseline-vs-mitigation CiteBench system evaluation and summarize deltas."
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Base config file path")
    parser.add_argument("--strategy", type=str, default="validation", choices=["development", "validation", "production"])
    parser.add_argument("--system-source", type=str, default="benchmark/CiteEval/data/system_eval/system_eval_examples.json")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of source queries for smoke testing")
    parser.add_argument("--variants", nargs="+", default=["baseline", "mitigation_all"], choices=["baseline", "mitigation_all"])
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
