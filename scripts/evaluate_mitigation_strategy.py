"""
Run paired baseline-vs-mitigation evaluation on RAGTruth.

This script automates a fair comparison by:
1. Creating temporary config variants (baseline, mitigation_all, ablations)
2. Running `scripts/demo_ragtruth_eval.py` with identical dataset settings
3. Saving per-variant metrics and a delta summary report

Usage examples:
    # Quick paired check (baseline vs full mitigation)
    python scripts/evaluate_mitigation_strategy.py --max-samples 30

    # Full ablation matrix
    python scripts/evaluate_mitigation_strategy.py --variants baseline mitigation_all filter_only rerank_only reprompt_only

    # Save into custom output directory
    python scripts/evaluate_mitigation_strategy.py --output-dir outputs/mitigation_eval/run_01
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

    if name == "filter_only":
        return {
            "mitigation": {
                "enabled": True,
                "reranker": {"enabled": False},
                "filter": {"enabled": True},
                "reprompt": {"enabled": False},
            }
        }

    if name == "rerank_only":
        return {
            "mitigation": {
                "enabled": True,
                "reranker": {"enabled": True},
                "filter": {"enabled": False},
                "reprompt": {"enabled": False},
            }
        }

    if name == "reprompt_only":
        return {
            "mitigation": {
                "enabled": True,
                "reranker": {"enabled": False},
                "filter": {"enabled": False},
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


def _run_variant(
    *,
    project_root: Path,
    variant: str,
    config_path: Path,
    split: str,
    max_samples: int | None,
    batch_size: int,
    strategy: str,
    ragtruth_eval_mode: str,
    output_path: Path,
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

    print(f"\n[run:{variant}] {' '.join(command)}")

    process = subprocess.run(
        command,
        text=True,
        capture_output=True,
        cwd=str(project_root),
    )

    if process.returncode != 0:
        raise RuntimeError(
            f"Variant '{variant}' failed with exit code {process.returncode}.\n"
            f"STDOUT:\n{process.stdout}\n"
            f"STDERR:\n{process.stderr}"
        )

    if process.stdout.strip():
        print(process.stdout.strip())


def _load_metrics(output_json: Path) -> dict[str, Any]:
    with output_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    metrics = payload.get("metrics", {})
    overall = metrics.get("overall", {})
    confusion = metrics.get("confusion_matrix", {})

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
    }


def _delta(base: float, current: float) -> float:
    return current - base


def _write_summary(
    summary_path: Path,
    *,
    baseline_name: str,
    variant_metrics: dict[str, dict[str, Any]],
) -> None:
    baseline = variant_metrics[baseline_name]

    lines = [
        "# Mitigation Evaluation Summary",
        "",
        "## Overall Metrics",
        "",
        "| Variant | Samples | Accuracy | Precision | Recall | F1 | ΔF1 vs Baseline | ΔRecall vs Baseline | ΔPrecision vs Baseline |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for name, metric in variant_metrics.items():
        lines.append(
            "| "
            f"{name} | {metric['num_samples']} | {metric['accuracy']:.4f} | {metric['precision']:.4f} | "
            f"{metric['recall']:.4f} | {metric['f1']:.4f} | "
            f"{_delta(baseline['f1'], metric['f1']):+.4f} | "
            f"{_delta(baseline['recall'], metric['recall']):+.4f} | "
            f"{_delta(baseline['precision'], metric['precision']):+.4f} |"
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
        description="Run paired baseline-vs-mitigation RAGTruth evaluations and summarize deltas."
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Base config file path")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--max-samples", type=int, default=None, help="Limit sample count for quick checks")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--strategy", type=str, default="validation", choices=["development", "validation", "production"])
    parser.add_argument(
        "--ragtruth-eval-mode",
        type=str,
        default="ragtruth_eval",
        choices=["ragtruth_eval", "normal"],
        help="ragtruth_eval uses benchmark responses; normal uses pipeline generation"
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "mitigation_all"],
        choices=["baseline", "mitigation_all", "filter_only", "rerank_only", "reprompt_only"],
        help="Mitigation variants to evaluate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/mitigation_eval/<timestamp>)"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    project_root = Path(__file__).resolve().parents[1]
    base_config_path = (project_root / args.config).resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else (project_root / "outputs" / "mitigation_eval" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_dir = output_dir / "configs"
    run_dir = output_dir / "ragtruth"
    run_dir.mkdir(parents=True, exist_ok=True)

    base_config = _load_yaml(base_config_path)

    variant_metrics: dict[str, dict[str, Any]] = {}

    for variant in args.variants:
        config_payload = _deep_update(deepcopy(base_config), _variant_patch(variant))
        variant_config_path = config_dir / f"config_{variant}.yaml"
        _write_yaml(variant_config_path, config_payload)

        result_path = run_dir / f"ragtruth_{variant}.json"
        _run_variant(
            project_root=project_root,
            variant=variant,
            config_path=variant_config_path,
            split=args.split,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            strategy=args.strategy,
            ragtruth_eval_mode=args.ragtruth_eval_mode,
            output_path=result_path,
        )

        variant_metrics[variant] = _load_metrics(result_path)

    if "baseline" not in variant_metrics:
        raise ValueError("`baseline` must be included in --variants for paired delta computation.")

    summary_payload = {
        "metadata": {
            "timestamp": timestamp,
            "split": args.split,
            "max_samples": args.max_samples,
            "batch_size": args.batch_size,
            "strategy": args.strategy,
            "ragtruth_eval_mode": args.ragtruth_eval_mode,
            "variants": args.variants,
        },
        "metrics": variant_metrics,
    }

    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_md = output_dir / "summary.md"
    _write_summary(summary_md, baseline_name="baseline", variant_metrics=variant_metrics)

    print("\nMitigation evaluation completed.")
    print(f"Summary JSON: {summary_json}")
    print(f"Summary Markdown: {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
