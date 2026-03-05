"""
RAGTruth baseline runner.

This script automates the baseline workflow located in
`benchmark/RAGTruth/baseline`:
1) prepare dataset
2) train baseline detector
3) evaluate with a served TGI endpoint

Examples:
    python scripts/run_ragtruth_baseline.py prepare
    python scripts/run_ragtruth_baseline.py train --profile single-gpu
    python scripts/run_ragtruth_baseline.py serve-cmd --model-subdir baseline
    python scripts/run_ragtruth_baseline.py evaluate --model-name baseline
    python scripts/run_ragtruth_baseline.py all --profile single-gpu
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE_DIR = PROJECT_ROOT / "benchmark" / "RAGTruth" / "baseline"
DEFAULT_MODEL_PATH = "meta-llama/Llama-2-13b-hf"


def _run(cmd: List[str], cwd: Path, env: Dict[str, str] | None = None, dry_run: bool = False) -> int:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"[run] cwd={cwd}")
    print(f"[run] {printable}")

    if dry_run:
        return 0

    completed = subprocess.run(cmd, cwd=str(cwd), env=env)
    return completed.returncode


def _build_train_command(args: argparse.Namespace) -> List[str]:
    output_dir = f"./exp/{args.model_name}"

    if args.profile == "exact":
        cmd = [
            "torchrun",
            "--nnodes",
            "1",
            "--nproc_per_node",
            str(args.nproc_per_node),
            "train.py",
            "--model_name_or_path",
            args.model_path,
            "--output_dir",
            output_dir,
            "--do_train",
            "--num_train_epochs",
            str(args.num_train_epochs),
            "--learning_rate",
            str(args.learning_rate),
            "--drop_neg_ratio",
            "-1",
            "--train_file",
            "./train.jsonl",
            "--eval_file",
            "./dev.jsonl",
            "--bf16",
            "True",
            "--tf32",
            "True",
            "--use_flashatt_2",
            "True",
            "--per_device_train_batch_size",
            str(args.train_batch_size),
            "--per_device_eval_batch_size",
            str(args.eval_batch_size),
            "--gradient_accumulation_steps",
            str(args.grad_accum_steps),
            "--model_max_length",
            str(args.model_max_length),
            "--logging_steps",
            str(args.logging_steps),
            "--run_name",
            args.model_name,
            "--lr_scheduler_type",
            "cosine",
            "--warmup_ratio",
            str(args.warmup_ratio),
            "--save_steps",
            str(args.save_steps),
            "--save_total_limit",
            "2",
            "--overwrite_output_dir",
            "--eval_strategy",
            "steps",
            "--eval_steps",
            str(args.eval_steps),
            "--fsdp",
            "shard_grad_op auto_wrap",
            "--fsdp_config",
            "./configs/fsdp.json",
        ]
        if args.report_to_wandb:
            cmd.extend(["--report_to", "wandb"])
        return cmd

    cmd = [
        sys.executable,
        "train.py",
        "--model_name_or_path",
        args.model_path,
        "--output_dir",
        output_dir,
        "--do_train",
        "--num_train_epochs",
        str(args.num_train_epochs),
        "--learning_rate",
        str(args.learning_rate),
        "--drop_neg_ratio",
        "-1",
        "--train_file",
        "./train.jsonl",
        "--eval_file",
        "./dev.jsonl",
        "--bf16",
        "True",
        "--tf32",
        "True",
        "--use_flashatt_2",
        "False",
        "--per_device_train_batch_size",
        str(args.train_batch_size),
        "--per_device_eval_batch_size",
        str(args.eval_batch_size),
        "--gradient_accumulation_steps",
        str(args.grad_accum_steps),
        "--model_max_length",
        str(args.model_max_length),
        "--logging_steps",
        str(args.logging_steps),
        "--run_name",
        args.model_name,
        "--lr_scheduler_type",
        "cosine",
        "--warmup_ratio",
        str(args.warmup_ratio),
        "--save_steps",
        str(args.save_steps),
        "--save_total_limit",
        "2",
        "--overwrite_output_dir",
        "--eval_strategy",
        "steps",
        "--eval_steps",
        str(args.eval_steps),
    ]
    if args.report_to_wandb:
        cmd.extend(["--report_to", "wandb"])
    return cmd


def _build_docker_command(args: argparse.Namespace, baseline_dir: Path) -> str:
    model_target = f"/data/exp/{args.model_subdir}"
    image = "ghcr.io/huggingface/text-generation-inference:2.0.1"

    if os.name == "nt":
        volume_expr = f'"{baseline_dir}:/data"'
    else:
        volume_expr = '"$PWD:/data"'

    return (
        f"docker run -d --name {args.container_name} --gpus '\"device={args.gpu_device}\"' "
        f"-v {volume_expr} --shm-size 1g -p {args.port}:80 {image} "
        f"--model-id {model_target} --dtype bfloat16 --max-total-tokens 8000 "
        "--sharded false --max-input-length 4095"
    )


def command_prepare(args: argparse.Namespace) -> int:
    baseline_dir = Path(args.baseline_dir).resolve()
    return _run([sys.executable, "prepare_dataset.py"], cwd=baseline_dir, dry_run=args.dry_run)


def command_train(args: argparse.Namespace) -> int:
    baseline_dir = Path(args.baseline_dir).resolve()
    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.wandb_api_key:
        env["WANDB_API_KEY"] = args.wandb_api_key
    if args.wandb_project:
        env["WANDB_PROJECT"] = args.wandb_project

    cmd = _build_train_command(args)
    return _run(cmd, cwd=baseline_dir, env=env, dry_run=args.dry_run)


def command_evaluate(args: argparse.Namespace) -> int:
    baseline_dir = Path(args.baseline_dir).resolve()
    cmd = [
        sys.executable,
        "predict_and_evaluate.py",
        "--raw_dataset",
        "./test.jsonl",
        "--output_file",
        args.output_file,
        "--metrics_output",
        args.metrics_file,
        "--model_name",
        args.model_name,
        "--tokenizer",
        args.tokenizer,
    ]
    return _run(cmd, cwd=baseline_dir, dry_run=args.dry_run)


def command_serve_cmd(args: argparse.Namespace) -> int:
    baseline_dir = Path(args.baseline_dir).resolve()
    cmd = _build_docker_command(args, baseline_dir)
    print("Run this command in your shell from baseline directory:")
    print(cmd)
    return 0


def command_all(args: argparse.Namespace) -> int:
    rc = command_prepare(args)
    if rc != 0:
        return rc

    rc = command_train(args)
    if rc != 0:
        return rc

    if args.run_evaluate:
        return command_evaluate(args)

    return 0


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--baseline-dir",
        default=str(DEFAULT_BASELINE_DIR),
        help="Path to benchmark/RAGTruth/baseline",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")


def main() -> int:
    parser = argparse.ArgumentParser(description="Automate RAGTruth baseline workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Create train/dev/test JSONL files")
    _common_parser(prepare_parser)
    prepare_parser.set_defaults(func=command_prepare)

    train_parser = subparsers.add_parser("train", help="Train baseline model")
    _common_parser(train_parser)
    train_parser.add_argument("--profile", choices=["exact", "single-gpu"], default="single-gpu")
    train_parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    train_parser.add_argument("--model-name", default="baseline")
    train_parser.add_argument("--num-train-epochs", type=float, default=1)
    train_parser.add_argument("--learning-rate", type=float, default=2e-5)
    train_parser.add_argument("--model-max-length", type=int, default=4096)
    train_parser.add_argument("--train-batch-size", type=int, default=1)
    train_parser.add_argument("--eval-batch-size", type=int, default=1)
    train_parser.add_argument("--grad-accum-steps", type=int, default=16)
    train_parser.add_argument("--logging-steps", type=int, default=1)
    train_parser.add_argument("--eval-steps", type=int, default=80)
    train_parser.add_argument("--save-steps", type=int, default=10000)
    train_parser.add_argument("--warmup-ratio", type=float, default=0.1)
    train_parser.add_argument("--nproc-per-node", type=int, default=4)
    train_parser.add_argument("--cuda-visible-devices", default=None)
    train_parser.add_argument("--report-to-wandb", action="store_true")
    train_parser.add_argument("--wandb-api-key", default=None)
    train_parser.add_argument("--wandb-project", default=None)
    train_parser.set_defaults(func=command_train)

    eval_parser = subparsers.add_parser("evaluate", help="Run baseline evaluator against TGI endpoint")
    _common_parser(eval_parser)
    eval_parser.add_argument("--model-name", default="baseline")
    eval_parser.add_argument("--tokenizer", default=DEFAULT_MODEL_PATH)
    eval_parser.add_argument("--output-file", default="./prediction.jsonl")
    eval_parser.add_argument("--metrics-file", default="./baseline_eval_metrics.json")
    eval_parser.set_defaults(func=command_evaluate)

    serve_parser = subparsers.add_parser("serve-cmd", help="Print recommended TGI docker command")
    _common_parser(serve_parser)
    serve_parser.add_argument("--container-name", default="baseline")
    serve_parser.add_argument("--model-subdir", default="baseline")
    serve_parser.add_argument("--gpu-device", default="0")
    serve_parser.add_argument("--port", type=int, default=8300)
    serve_parser.set_defaults(func=command_serve_cmd)

    all_parser = subparsers.add_parser("all", help="Run prepare + train (+ optional evaluate)")
    _common_parser(all_parser)
    all_parser.add_argument("--profile", choices=["exact", "single-gpu"], default="single-gpu")
    all_parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    all_parser.add_argument("--model-name", default="baseline")
    all_parser.add_argument("--num-train-epochs", type=float, default=1)
    all_parser.add_argument("--learning-rate", type=float, default=2e-5)
    all_parser.add_argument("--model-max-length", type=int, default=4096)
    all_parser.add_argument("--train-batch-size", type=int, default=1)
    all_parser.add_argument("--eval-batch-size", type=int, default=1)
    all_parser.add_argument("--grad-accum-steps", type=int, default=16)
    all_parser.add_argument("--logging-steps", type=int, default=1)
    all_parser.add_argument("--eval-steps", type=int, default=80)
    all_parser.add_argument("--save-steps", type=int, default=10000)
    all_parser.add_argument("--warmup-ratio", type=float, default=0.1)
    all_parser.add_argument("--nproc-per-node", type=int, default=4)
    all_parser.add_argument("--cuda-visible-devices", default=None)
    all_parser.add_argument("--report-to-wandb", action="store_true")
    all_parser.add_argument("--wandb-api-key", default=None)
    all_parser.add_argument("--wandb-project", default=None)
    all_parser.add_argument("--run-evaluate", action="store_true")
    all_parser.add_argument("--tokenizer", default=DEFAULT_MODEL_PATH)
    all_parser.add_argument("--output-file", default="./prediction.jsonl")
    all_parser.add_argument("--metrics-file", default="./baseline_eval_metrics.json")
    all_parser.set_defaults(func=command_all)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
