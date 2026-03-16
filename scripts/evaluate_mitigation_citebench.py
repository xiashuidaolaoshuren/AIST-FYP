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
from src.generation.claim_extractor import extract_claims
from src.pipelines.baseline_rag import BaselineRAGPipeline
from src.utils.config import Config
from src.utils.data_structures import Claim, EvidenceChunk


ORACLE_DATASET_PRESETS = {
    "asqa": "benchmark/CiteEval/data/dev/asqa_oracle.dev.jsonl",
    "eli5": "benchmark/CiteEval/data/dev/eli5_oracle.dev.jsonl",
    "msmarco": "benchmark/CiteEval/data/dev/msmarco_oracle.dev.jsonl",
}


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


def _resolve_oracle_source(project_root: Path, oracle_source: str | None, oracle_dataset: str) -> Path:
    if oracle_source:
        resolved = (project_root / oracle_source).resolve()
    else:
        resolved = (project_root / ORACLE_DATASET_PRESETS[oracle_dataset]).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Oracle source not found: {resolved}")
    return resolved


def _load_oracle_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            row = json.loads(payload)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object on line {idx} in {path}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Expected non-empty JSONL oracle file at {path}")
    return rows


def _generation_params(runtime: VariantRuntime) -> dict[str, Any]:
    return {
        "max_new_tokens": getattr(runtime.config.generation, "max_new_tokens", 256),
        "temperature": getattr(runtime.config.generation, "temperature", 0.7),
        "top_p": getattr(runtime.config.generation, "top_p", 0.9),
        "do_sample": getattr(runtime.config.generation, "do_sample", True),
    }


def _build_oracle_evidence_chunks(row: dict[str, Any]) -> list[EvidenceChunk]:
    passages = row.get("passages", [])
    chunks: list[EvidenceChunk] = []
    for idx, passage in enumerate(passages):
        if not isinstance(passage, dict):
            continue
        text = str(passage.get("text", "")).strip()
        if not text:
            continue
        doc_id = str(passage.get("id", f"oracle_doc_{idx}"))
        score_dense = passage.get("score", 1.0)
        try:
            score_dense = float(score_dense)
        except (TypeError, ValueError):
            score_dense = 1.0

        chunks.append(
            EvidenceChunk(
                doc_id=doc_id,
                sent_id=idx,
                text=text,
                char_start=0,
                char_end=len(text),
                score_dense=score_dense,
                rank=idx,
                source="oracle",
                version="citebench_dev",
            )
        )
    return chunks


def _run_with_oracle_context(runtime: VariantRuntime, query: str, row: dict[str, Any]) -> dict[str, Any]:
    evidence_chunks = _build_oracle_evidence_chunks(row)
    if not evidence_chunks:
        raise ValueError("Oracle row has no usable passages")

    split_enabled = bool(getattr(getattr(runtime.config.processing, "query_split", None), "enabled", True))
    if split_enabled and hasattr(runtime.pipeline, "_split_query_by_questions"):
        sub_queries = runtime.pipeline._split_query_by_questions(query)
    else:
        sub_queries = [{"text": query, "sub_query_id": 0}]

    if not sub_queries:
        sub_queries = [{"text": query, "sub_query_id": 0}]

    combined_response_parts: list[str] = []
    all_claims: list[Claim] = []
    claims_by_sub_answer: list[dict[str, Any]] = []
    sub_answer_metadata: list[dict[str, Any]] = []
    generation_params = _generation_params(runtime)

    for sub_query_data in sub_queries:
        sub_query_text = str(sub_query_data.get("text", "")).strip()
        sub_query_id = int(sub_query_data.get("sub_query_id", 0))

        generation_output = runtime.pipeline.generator.generate_with_metadata(
            prompt=sub_query_text,
            evidence_chunks=evidence_chunks,
            **generation_params,
        )
        generation_output["original_query"] = sub_query_text
        generated_text = generation_output.get("text", "")

        sub_claims = extract_claims(text=generated_text, method="auto")

        char_start = len(" ".join(combined_response_parts) + (" " if combined_response_parts else ""))
        combined_response_parts.append(generated_text)
        char_end = len(" ".join(combined_response_parts))

        for claim in sub_claims:
            original_span = claim.answer_char_span
            claim.answer_char_span = [
                original_span[0] + char_start,
                original_span[1] + char_start,
            ]

        claims_by_sub_answer.append(
            {
                "sub_answer_id": sub_query_id,
                "sub_text": generated_text,
                "sub_query": sub_query_text,
                "claims": sub_claims,
            }
        )
        sub_answer_metadata.append(
            {
                "sub_answer_id": sub_query_id,
                "char_span": [char_start, char_end],
                "sub_query": sub_query_text,
                "metadata": generation_output,
            }
        )
        all_claims.extend(sub_claims)

    draft_response = " ".join(combined_response_parts)
    mitigation_result = {
        "final_answer": draft_response,
        "actions": [],
        "filtered_claim_count": 0,
        "claim_records": [],
    }

    if runtime.pipeline.mitigation_orchestrator and runtime.pipeline.mitigation_orchestrator.enabled:
        claim_records = []
        for claim in all_claims:
            verification_metadata = None
            for entry in sub_answer_metadata:
                span = entry["char_span"]
                if claim.answer_char_span[0] >= span[0] and claim.answer_char_span[1] <= span[1]:
                    verification_metadata = dict(entry["metadata"])
                    verification_metadata.setdefault("original_query", entry["sub_query"])
                    break
            if verification_metadata is None:
                verification_metadata = {
                    "text": draft_response,
                    "original_query": query,
                    "tokens": [],
                    "scores": [],
                }
            claim_records.append(
                {
                    "claim": claim,
                    "evidence": evidence_chunks,
                    "metadata": verification_metadata,
                }
            )

        mitigation_result = runtime.pipeline.mitigation_orchestrator.apply(
            query=query,
            answer_text=draft_response,
            claim_records=claim_records,
        )

    claim_evidence_pairs = [
        {
            "claim_id": claim.claim_id,
            "evidence_spans": [chunk.to_dict() for chunk in evidence_chunks],
        }
        for claim in all_claims
    ]

    output = {
        "query": query,
        "draft_response": draft_response,
        "response_after_mitigation": mitigation_result["final_answer"],
        "mitigation_actions": mitigation_result["actions"],
        "filtered_claim_count": mitigation_result["filtered_claim_count"],
        "claims_by_sub_answer": claims_by_sub_answer,
        "claim_evidence_pairs": claim_evidence_pairs,
        "generator_metadata": {
            "sub_answer_metadata": sub_answer_metadata,
            "original_query": query,
            "num_sub_questions": len(sub_queries),
        },
        "retrieval_metadata": {
            "context_source": "oracle",
            "num_retrieved": len(evidence_chunks),
            "evidence_doc_ids": [chunk.doc_id for chunk in evidence_chunks[:10]],
        },
    }

    if mitigation_result.get("claim_records"):
        output["mitigation_claims"] = [record["claim"].to_dict() for record in mitigation_result["claim_records"]]
        output["mitigation_evidence_map"] = {
            record["claim"].claim_id: [chunk.to_dict() for chunk in record.get("evidence", [])]
            for record in mitigation_result["claim_records"]
        }

    return output


def _generate_system_input(
    *,
    runtime: VariantRuntime,
    source_queries: list[dict[str, Any]],
    context_source: str,
    output_path: Path,
    resume: bool,
) -> dict[str, int]:
    records: list[dict[str, Any]] = []
    processed_ids: set[str] = set()
    skipped_missing_query = 0
    skipped_missing_passages = 0

    if resume and output_path.exists():
        existing_payload = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(existing_payload, list):
            raise ValueError(f"Resume mismatch: expected list JSON at {output_path}")
        for item in existing_payload:
            if not isinstance(item, dict):
                raise ValueError(f"Resume mismatch: invalid sample entry in {output_path}")
            sample_id = str(item.get("id", "")).strip()
            if not sample_id:
                raise ValueError(f"Resume mismatch: sample without id in {output_path}")
            processed_ids.add(sample_id)
        records.extend(existing_payload)
        print(f"[resume] loaded {len(records)} existing samples from {output_path}")

    for row_index, row in enumerate(source_queries):
        sample_id = str(row.get("id", f"sample_{row_index + 1}"))
        if sample_id in processed_ids:
            continue
        query = str(row.get("query", "")).strip()
        if not query:
            skipped_missing_query += 1
            continue

        if context_source == "oracle":
            if not row.get("passages"):
                skipped_missing_passages += 1
                continue
            pipeline_output = _run_with_oracle_context(runtime, query, row)
        else:
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
        processed_ids.add(sample_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "total_rows": len(source_queries),
        "generated_rows": len(records),
        "skipped_missing_query": skipped_missing_query,
        "skipped_missing_passages": skipped_missing_passages,
    }


def _run_system_eval(
    *,
    project_root: Path,
    system_input: Path,
    context_source: str,
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


def _write_summary(
    summary_path: Path,
    metrics: dict[str, dict[str, float]],
    *,
    context_source: str,
    baseline_name: str = "baseline",
) -> None:
    baseline = metrics[baseline_name]
    comparable = "yes" if context_source == "retrieval" else "no (diagnostic/oracle)"
    lines = [
        "# CiteBench Module Evaluation Summary",
        "",
        f"Context source: `{context_source}`",
        f"Benchmark comparable: `{comparable}`",
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
    parser.add_argument(
        "--dataset-role",
        type=str,
        default="mitigation",
        choices=["mitigation"],
        help="Dataset role for this script. Only mitigation role is supported here.",
    )
    parser.add_argument("--strategy", type=str, default="validation", choices=["development", "validation", "production"])
    parser.add_argument("--system-source", type=str, default="benchmark/CiteEval/data/system_eval/system_eval_examples.json")
    parser.add_argument(
        "--context-source",
        type=str,
        default="retrieval",
        choices=["retrieval", "oracle"],
        help="Use retrieval pipeline input source or oracle dev passages as generation context",
    )
    parser.add_argument(
        "--oracle-source",
        type=str,
        default=None,
        help="Optional explicit oracle JSONL source path (overrides --oracle-dataset preset)",
    )
    parser.add_argument(
        "--oracle-dataset",
        type=str,
        default="asqa",
        choices=["asqa", "eli5", "msmarco"],
        help="Oracle dataset preset when --context-source=oracle and --oracle-source is not set",
    )
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
    parser.add_argument("--resume", action="store_true", help="Resume incomplete variant outputs in-place")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    project_root = Path(__file__).resolve().parents[1]
    base_config_path = (project_root / args.config).resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    if args.context_source == "oracle":
        source_path = _resolve_oracle_source(project_root, args.oracle_source, args.oracle_dataset)
        source_rows = _load_oracle_rows(source_path)
    else:
        source_path = (project_root / args.system_source).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"System source not found: {source_path}")
        source_rows = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(source_rows, list) or not source_rows:
            raise ValueError(f"Expected non-empty list JSON at {source_path}")

    if args.max_samples is not None:
        source_rows = source_rows[: args.max_samples]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_output_dir = Path(args.output_dir) if args.output_dir else (project_root / "outputs" / "mitigation_eval_citebench" / timestamp)
    output_dir = root_output_dir / args.context_source
    output_dir.mkdir(parents=True, exist_ok=True)

    config_dir = output_dir / "configs"
    system_input_dir = output_dir / "system_inputs"

    base_config = _load_yaml(base_config_path)
    summary_metrics: dict[str, dict[str, float]] = {}
    generation_stats: dict[str, dict[str, int]] = {}

    citeeval_output_root = project_root / "benchmark" / "CiteEval" / "data" / "system_eval_outputs"
    citeeval_root = project_root / "benchmark" / "CiteEval"
    citeeval_src = project_root / "benchmark" / "CiteEval" / "src"

    for variant in args.variants:
        config_payload = _deep_update(deepcopy(base_config), _variant_patch(variant))
        variant_config_path = config_dir / f"config_{variant}.yaml"
        _write_yaml(variant_config_path, config_payload)

        runtime = _build_runtime(variant_config_path, args.strategy)

        system_input_json = system_input_dir / f"system_eval_{variant}.json"
        if args.resume and system_input_json.exists() and args.max_samples is not None:
            existing_payload = json.loads(system_input_json.read_text(encoding="utf-8"))
            if not isinstance(existing_payload, list):
                raise ValueError(f"Resume mismatch: expected list JSON at {system_input_json}")
            if len(existing_payload) > len(source_rows):
                raise ValueError(
                    f"Resume mismatch for variant '{variant}': existing samples {len(existing_payload)} exceed target {len(source_rows)}."
                )

        generation_stats[variant] = _generate_system_input(
            runtime=runtime,
            source_queries=source_rows,
            context_source=args.context_source,
            output_path=system_input_json,
            resume=args.resume,
        )

        _run_system_eval(
            project_root=project_root,
            system_input=system_input_json,
            context_source=args.context_source,
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
            "dataset_role": args.dataset_role,
            "context_source": args.context_source,
            "baseline_comparable": args.context_source == "retrieval",
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
        "generation_stats": generation_stats,
    }

    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_md = output_dir / "summary.md"
    _write_summary(summary_md, summary_metrics, context_source=args.context_source)

    print("\nCiteBench mitigation evaluation completed.")
    print(f"Summary JSON: {summary_json}")
    print(f"Summary Markdown: {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
