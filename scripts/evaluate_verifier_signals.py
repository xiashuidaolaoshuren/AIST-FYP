"""
Run verifier signal ablation evaluation variants on RAGTruth.

This script automates a fair comparison by:
1. Creating temporary verifier config variants (all-signal and single-signal)
2. Running `scripts/demo_ragtruth_eval.py` with identical dataset settings
3. Saving per-variant metrics and a delta summary report

Usage examples:
    # Quick check
    python scripts/evaluate_verifier_signals.py --max-samples 30

    # Full verifier signal matrix
    python scripts/evaluate_verifier_signals.py --variants full_verifier verifier_intrinsic_only verifier_grounded_only verifier_nli_only verifier_self_agreement_only

    # Save into custom output directory
    python scripts/evaluate_verifier_signals.py --output-dir outputs/verifier_eval/run_01
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from tqdm.auto import tqdm


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

    all_mitigation_disabled = {
        "mitigation": {
            "enabled": False,
            "reranker": {"enabled": False},
            "filter": {"enabled": False},
            "reprompt": {"enabled": False},
        }
    }

    if name in {"baseline", "full_verifier"}:
        return _deep_update(deepcopy(all_verifiers_enabled), deepcopy(all_mitigation_disabled))

    if name == "full_verifier_lettuce":
        patch = _deep_update(deepcopy(all_verifiers_enabled), deepcopy(all_mitigation_disabled))
        nli_patch = patch.setdefault("verification", {}).setdefault("nli", {})
        nli_patch["backend"] = "lettucedetect"
        nli_patch["model_name"] = "KRLabsOrg/lettucedect-base-modernbert-en-v1"
        return patch

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
                "contradiction_first_fusion": True,
                "modules": {
                    "intrinsic": False,
                    "grounded": False,
                    "nli": True,
                    "self_agreement": False,
                },
            },
            **deepcopy(all_mitigation_disabled),
        }

    if name == "verifier_nli_only_lettuce":
        patch = {
            "verification": {
                "enabled": True,
                "contradiction_first_fusion": True,
                "modules": {
                    "intrinsic": False,
                    "grounded": False,
                    "nli": True,
                    "self_agreement": False,
                },
            },
            **deepcopy(all_mitigation_disabled),
        }
        nli_patch = patch.setdefault("verification", {}).setdefault("nli", {})
        nli_patch["backend"] = "lettucedetect"
        nli_patch["model_name"] = "KRLabsOrg/lettucedect-base-modernbert-en-v1"
        return patch

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


def _run_variant(
    *,
    project_root: Path,
    variant: str,
    config_path: Path,
    split: str,
    max_samples: int | None,
    samples_per_task: int | None,
    max_saved_samples: int | None,
    batch_size: int,
    strategy: str,
    ragtruth_eval_mode: str,
    output_path: Path,
    resume: bool,
) -> None:
    command = [
        sys.executable,
        "scripts/demo_ragtruth_eval.py",
        "--config",
        str(config_path),
        "--split",
        split,
        "--batch-size",
        str(batch_size),
        "--strategy",
        strategy,
        "--ragtruth-eval-mode",
        ragtruth_eval_mode,
        "--save-results",
        "--output-path",
        str(output_path),
    ]

    if max_samples is not None:
        command.extend(["--max-samples", str(max_samples)])
    if samples_per_task is not None:
        command.extend(["--samples-per-task", str(samples_per_task)])
    if max_saved_samples is not None:
        command.extend(["--max-saved-samples", str(max_saved_samples)])
    if resume:
        command.append("--resume")

    print(f"\n[run:{variant}] {' '.join(command)}")

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(project_root),
    )
    assert process.stdout is not None
    out_lines: list[str] = []
    for line in process.stdout:
        print(line, end="", flush=True)
        out_lines.append(line)
    process.wait()

    if process.returncode != 0:
        raise RuntimeError(
            f"Variant '{variant}' failed with exit code {process.returncode}.\n"
            f"OUTPUT:\n{''.join(out_lines)}"
        )


def _load_metrics(output_json: Path) -> dict[str, Any]:
    with output_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    metrics = payload.get("metrics", {})
    overall = metrics.get("overall", {})
    confusion = metrics.get("confusion_matrix", {})
    statistics = metrics.get("statistics", {})

    return {
        "accuracy": float(overall.get("accuracy", 0.0)),
        "precision": float(overall.get("precision", 0.0)),
        "recall": float(overall.get("recall", 0.0)),
        "f1": float(overall.get("f1", 0.0)),
        "num_samples": int(overall.get("num_samples", 0)),
        "tp": int(confusion.get("true_positives", 0)),
        "tn": int(confusion.get("true_negatives", 0)),
        "fp": int(confusion.get("false_positives", 0)),
        "fn": int(confusion.get("false_negatives", 0)),
        "sample_hallucinations": int(statistics.get("detected_hallucinations", 0)),
        "claim_hallucinations": int(statistics.get("detected_claim_hallucinations", 0)),
        "total_claims": int(statistics.get("total_claims", 0)),
        "avg_claim_hallucinations_per_sample": float(statistics.get("avg_claim_hallucinations_per_sample", 0.0)),
    }


def _load_existing_num_samples(output_json: Path) -> int | None:
    if not output_json.exists():
        return None
    with output_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    sample_results = payload.get("sample_results")
    if isinstance(sample_results, list):
        return len(sample_results)
    return None


def _load_existing_metadata(output_json: Path) -> dict[str, Any] | None:
    if not output_json.exists():
        return None
    with output_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    return None


def _build_expected_resume_fingerprint(
    *,
    config_payload: dict[str, Any],
    split: str,
    max_samples: int | None,
    samples_per_task: int | None,
    ragtruth_eval_mode: str,
    dataset_path: str,
) -> dict[str, Any]:
    verification_cfg = config_payload.get("verification", {})
    if not isinstance(verification_cfg, dict):
        verification_cfg = {}

    verification_modules = verification_cfg.get("modules", {})
    if not isinstance(verification_modules, dict):
        verification_modules = {}

    mitigation_cfg = config_payload.get("mitigation", {})
    if not isinstance(mitigation_cfg, dict):
        mitigation_cfg = {}

    def module_flag(raw_value: Any) -> bool:
        if isinstance(raw_value, dict):
            return bool(raw_value.get("enabled", False))
        return bool(raw_value)

    return {
        "split": split,
        "max_samples": max_samples,
        "samples_per_task": samples_per_task,
        "ragtruth_eval_mode": ragtruth_eval_mode,
        "dataset_path": dataset_path,
        "verification_enabled": bool(verification_cfg.get("enabled", True)),
        "nli_backend": str(verification_cfg.get("nli", {}).get("backend", "deberta")),
        "nli_model_name": str(verification_cfg.get("nli", {}).get("model_name", "")),
        "verification_modules": {
            "intrinsic": module_flag(verification_modules.get("intrinsic", False)),
            "grounded": module_flag(verification_modules.get("grounded", False)),
            "nli": module_flag(verification_modules.get("nli", False)),
            "self_agreement": module_flag(verification_modules.get("self_agreement", False)),
        },
        "mitigation_enabled": bool(mitigation_cfg.get("enabled", False)),
        "mitigation_modules": {
            "reranker": module_flag(mitigation_cfg.get("reranker", False)),
            "filter": module_flag(mitigation_cfg.get("filter", False)),
            "reprompt": module_flag(mitigation_cfg.get("reprompt", False)),
        },
    }


def _delta(base: float, current: float) -> float:
    return current - base


def _drop_abs(base: int, current: int) -> int:
    return base - current


def _drop_pct(base: int, current: int) -> float | None:
    if base <= 0:
        return None
    return ((base - current) / base) * 100.0


def _format_overall_deltas(
    *,
    baseline: dict[str, Any] | None,
    metric: dict[str, Any],
) -> tuple[str, str, str]:
    if baseline is None:
        return ("N/A", "N/A", "N/A")
    return (
        f"{_delta(baseline['f1'], metric['f1']):+.4f}",
        f"{_delta(baseline['recall'], metric['recall']):+.4f}",
        f"{_delta(baseline['precision'], metric['precision']):+.4f}",
    )


def _format_hallucination_deltas(
    *,
    baseline: dict[str, Any] | None,
    metric: dict[str, Any],
) -> tuple[str, str, str, str]:
    if baseline is None:
        return ("N/A", "N/A", "N/A", "N/A")

    sample_drop_abs = _drop_abs(baseline['sample_hallucinations'], metric['sample_hallucinations'])
    claim_drop_abs = _drop_abs(baseline['claim_hallucinations'], metric['claim_hallucinations'])
    sample_drop_pct = _drop_pct(baseline['sample_hallucinations'], metric['sample_hallucinations'])
    claim_drop_pct = _drop_pct(baseline['claim_hallucinations'], metric['claim_hallucinations'])
    return (
        f"{sample_drop_abs:+d}",
        "N/A" if sample_drop_pct is None else f"{sample_drop_pct:+.2f}%",
        f"{claim_drop_abs:+d}",
        "N/A" if claim_drop_pct is None else f"{claim_drop_pct:+.2f}%",
    )


def _write_summary(
    summary_path: Path,
    *,
    baseline_name: str,
    variant_metrics: dict[str, dict[str, Any]],
) -> None:
    baseline = variant_metrics.get(baseline_name)
    has_baseline = baseline is not None
    baseline_label = baseline_name if has_baseline else "N/A"

    lines = [
        "# Module Evaluation Summary (RAGTruth)",
        "",
        "## Overall Metrics",
        "",
        f"| Variant | Samples | Accuracy | Precision | Recall | F1 | ΔF1 vs {baseline_label} | ΔRecall vs {baseline_label} | ΔPrecision vs {baseline_label} |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for name, metric in variant_metrics.items():
        f1_delta, recall_delta, precision_delta = _format_overall_deltas(
            baseline=baseline,
            metric=metric,
        )
        lines.append(
            "| "
            f"{name} | {metric['num_samples']} | {metric['accuracy']:.4f} | {metric['precision']:.4f} | "
            f"{metric['recall']:.4f} | {metric['f1']:.4f} | {f1_delta} | {recall_delta} | {precision_delta} |"
        )

    lines.extend([
        "",
        f"## Hallucination Reduction vs {baseline_label}",
        "",
        "| Variant | Hallucinated Samples | Sample Drop (Abs) | Sample Drop (%) | Hallucinated Claims | Claim Drop (Abs) | Claim Drop (%) | Avg Claim Hallucinations / Sample |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for name, metric in variant_metrics.items():
        (
            sample_drop_abs_display,
            sample_drop_pct_display,
            claim_drop_abs_display,
            claim_drop_pct_display,
        ) = _format_hallucination_deltas(
            baseline=baseline,
            metric=metric,
        )
        lines.append(
            "| "
            f"{name} | {metric['sample_hallucinations']} | {sample_drop_abs_display} | {sample_drop_pct_display} | "
            f"{metric['claim_hallucinations']} | {claim_drop_abs_display} | {claim_drop_pct_display} | "
            f"{metric['avg_claim_hallucinations_per_sample']:.4f} |"
        )

    lines.extend([
        "",
        "## Confusion Matrix Counts",
        "",
        "| Variant | TP | TN | FP | FN |",
        "|---|---:|---:|---:|---:|",
    ])

    for name, metric in variant_metrics.items():
        lines.append(
            f"| {name} | {metric['tp']} | {metric['tn']} | {metric['fp']} | {metric['fn']} |"
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run verifier signal ablation RAGTruth evaluation variants and summarize deltas."
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Base config file path")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--max-samples", type=int, default=None, help="Limit sample count for quick checks")
    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=None,
        help="Limit samples per task type (overrides --max-samples when set)",
    )
    parser.add_argument(
        "--max-saved-samples",
        type=int,
        default=None,
        help="Limit persisted sample_results entries per variant JSON",
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--strategy", type=str, default="validation", choices=["development", "validation", "production"])
    parser.add_argument(
        "--ragtruth-eval-mode",
        type=str,
        default="gold_context_generation",
        choices=["ragtruth_eval", "normal", "gold_context_generation"],
        help=(
            "ragtruth_eval uses benchmark responses; "
            "normal uses pipeline generation with local retrieval; "
            "gold_context_generation uses benchmark contexts for generation"
        )
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "full_verifier",
            "full_verifier_lettuce",
            "verifier_intrinsic_only",
            "verifier_grounded_only",
            "verifier_nli_only",
            "verifier_nli_only_lettuce",
            "verifier_self_agreement_only",
        ],
        choices=[
            "baseline",
            "full_verifier",
            "full_verifier_lettuce",
            "verifier_intrinsic_only",
            "verifier_grounded_only",
            "verifier_nli_only",
            "verifier_nli_only_lettuce",
            "verifier_self_agreement_only",
        ],
        help="Evaluation variants to run"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/verifier_eval/<timestamp>)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume incomplete variant outputs in-place"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.max_saved_samples is not None and args.max_saved_samples <= 0:
        raise ValueError("--max-saved-samples must be a positive integer when provided")

    project_root = Path(__file__).resolve().parents[1]
    base_config_path = (project_root / args.config).resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else (project_root / "outputs" / "verifier_eval" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_dir = output_dir / "configs"
    run_dir = output_dir / "ragtruth"
    run_dir.mkdir(parents=True, exist_ok=True)

    base_config = _load_yaml(base_config_path)
    base_dataset_path = (
        base_config.get("evaluation", {})
        .get("benchmarks", {})
        .get("ragtruth", {})
        .get("dataset_path", "benchmark/RAGTruth/dataset")
    )
    resolved_dataset_path = str((project_root / str(base_dataset_path)).resolve())

    variant_metrics: dict[str, dict[str, Any]] = {}

    variant_bar = tqdm(args.variants, desc="Variants", unit="variant", position=0)
    for variant in variant_bar:
        variant_bar.set_postfix_str(variant)
        config_payload = _deep_update(deepcopy(base_config), _variant_patch(variant))
        variant_config_path = config_dir / f"config_{variant}.yaml"
        _write_yaml(variant_config_path, config_payload)

        result_path = run_dir / f"ragtruth_{variant}.json"
        if args.resume and result_path.exists():
            existing_metadata = _load_existing_metadata(result_path)
            if not isinstance(existing_metadata, dict):
                raise ValueError(
                    f"Resume mismatch for variant '{variant}': missing metadata in {result_path}."
                )
            existing_fingerprint = existing_metadata.get("selection_fingerprint")
            expected_fingerprint = _build_expected_resume_fingerprint(
                config_payload=config_payload,
                split=args.split,
                max_samples=args.max_samples,
                samples_per_task=args.samples_per_task,
                ragtruth_eval_mode=args.ragtruth_eval_mode,
                dataset_path=resolved_dataset_path,
            )
            if not isinstance(existing_fingerprint, dict) or existing_fingerprint != expected_fingerprint:
                raise ValueError(
                    "Resume mismatch for variant "
                    f"'{variant}': selection fingerprint differs. "
                    f"existing={existing_fingerprint}, expected={expected_fingerprint}"
                )

        _run_variant(
            project_root=project_root,
            variant=variant,
            config_path=variant_config_path,
            split=args.split,
            max_samples=args.max_samples,
            samples_per_task=args.samples_per_task,
            max_saved_samples=args.max_saved_samples,
            batch_size=args.batch_size,
            strategy=args.strategy,
            ragtruth_eval_mode=args.ragtruth_eval_mode,
            output_path=result_path,
            resume=args.resume,
        )

        variant_metrics[variant] = _load_metrics(result_path)
        variant_bar.set_postfix_str(f"{variant} ✓")

    baseline_variant = None
    if "full_verifier" in variant_metrics:
        baseline_variant = "full_verifier"
    elif "baseline" in variant_metrics:
        baseline_variant = "baseline"

    summary_payload = {
        "metadata": {
            "timestamp": timestamp,
            "split": args.split,
            "max_samples": args.max_samples,
            "samples_per_task": args.samples_per_task,
            "max_saved_samples": args.max_saved_samples,
            "batch_size": args.batch_size,
            "strategy": args.strategy,
            "ragtruth_eval_mode": args.ragtruth_eval_mode,
            "variants": args.variants,
            "hallucination_drop_definition": {
                "sample": "baseline_detected_hallucinated_samples - variant_detected_hallucinated_samples",
                "claim": "baseline_detected_contradictory_claims - variant_detected_contradictory_claims",
                "percent": "(drop_abs / baseline_count) * 100, or null when baseline_count == 0"
            }
        },
        "metrics": variant_metrics,
        "baseline_variant": baseline_variant,
        "deltas_vs_baseline": {},
    }

    if baseline_variant:
        baseline_metrics = variant_metrics[baseline_variant]
        for variant_name, metric in variant_metrics.items():
            summary_payload["deltas_vs_baseline"][variant_name] = {
                "f1_delta": _delta(baseline_metrics["f1"], metric["f1"]),
                "recall_delta": _delta(baseline_metrics["recall"], metric["recall"]),
                "precision_delta": _delta(baseline_metrics["precision"], metric["precision"]),
                "sample_hallucination_drop_abs": _drop_abs(
                    baseline_metrics["sample_hallucinations"], metric["sample_hallucinations"]
                ),
                "sample_hallucination_drop_pct": _drop_pct(
                    baseline_metrics["sample_hallucinations"], metric["sample_hallucinations"]
                ),
                "claim_hallucination_drop_abs": _drop_abs(
                    baseline_metrics["claim_hallucinations"], metric["claim_hallucinations"]
                ),
                "claim_hallucination_drop_pct": _drop_pct(
                    baseline_metrics["claim_hallucinations"], metric["claim_hallucinations"]
                ),
            }

    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_md = output_dir / "summary.md"
    _write_summary(
        summary_md,
        baseline_name=baseline_variant or "full_verifier",
        variant_metrics=variant_metrics,
    )

    print("\nVerifier evaluation completed.")
    print(f"Summary JSON: {summary_json}")
    print(f"Summary Markdown: {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
