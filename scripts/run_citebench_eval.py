"""Unified entrypoint for CiteBench/CiteEval evaluation workflows.

This script provides a Windows-friendly orchestration layer over CiteEval
commands with explicit preflight validation and clearer failure messages.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalPaths:
    project_root: Path
    citeeval_root: Path
    citeeval_src: Path
    data_root: Path
    metric_root: Path
    metric_output_root: Path
    system_eval_root: Path
    system_eval_output_root: Path
    temp_root: Path


def resolve_metric_root(data_root: Path) -> Path:
    preferred = data_root / "citebench" / "metric_eval"
    fallback = data_root / "metric_eval"
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        "CiteBench metric data not found. Expected either "
        f"'{preferred}' or '{fallback}'."
    )


def resolve_paths(project_root: Path) -> EvalPaths:
    citeeval_root = project_root / "benchmark" / "CiteEval"
    citeeval_src = citeeval_root / "src"
    data_root = citeeval_root / "data"
    metric_root = resolve_metric_root(data_root)

    return EvalPaths(
        project_root=project_root,
        citeeval_root=citeeval_root,
        citeeval_src=citeeval_src,
        data_root=data_root,
        metric_root=metric_root,
        metric_output_root=data_root / "metric_eval_outputs",
        system_eval_root=data_root / "system_eval",
        system_eval_output_root=data_root / "system_eval_outputs",
        temp_root=data_root / "tmp",
    )


def _to_posix(path: Path) -> str:
    return path.as_posix()


def _build_env(paths: EvalPaths) -> dict[str, str]:
    dotenv_path = paths.project_root / ".env"
    try:
        from dotenv import load_dotenv
        if dotenv_path.exists():
            load_dotenv(dotenv_path=dotenv_path)
    except ImportError:
        pass

    env = os.environ.copy()
    env["CITEEVAL_ROOT"] = _to_posix(paths.citeeval_root)

    existing_pythonpath = env.get("PYTHONPATH", "")
    extra = os.pathsep.join([_to_posix(paths.citeeval_root), _to_posix(paths.citeeval_src)])
    env["PYTHONPATH"] = f"{existing_pythonpath}{os.pathsep}{extra}" if existing_pythonpath else extra
    return env


def _resolve_provider_and_model(
    provider_override: str | None,
    model_override: str | None,
    env: dict[str, str],
    check_credentials: bool,
) -> tuple[str, str]:
    provider = (provider_override or env.get("CITEEVAL_PROVIDER", "openai")).strip().lower()
    if provider not in {"openai", "deepseek"}:
        raise ValueError("CITEEVAL_PROVIDER must be either 'openai' or 'deepseek'.")

    if model_override:
        model_name = model_override
    elif provider == "deepseek":
        model_name = "deepseek-chat"
    else:
        model_name = "gpt-4o"

    if provider == "deepseek":
        _validate_deepseek_model_and_credentials(model_name, env, check_credentials)
    else:
        _validate_openai_model_and_credentials(model_name, env, check_credentials)

    env["CITEEVAL_PROVIDER"] = provider
    return provider, model_name


def _validate_deepseek_model_and_credentials(model_name: str, env: dict[str, str], check_credentials: bool) -> None:
    if not model_name.startswith("deepseek"):
        raise ValueError(
            f"Model '{model_name}' is incompatible with provider 'deepseek'. "
            "Use a deepseek-* model such as deepseek-chat."
        )
    if check_credentials and not env.get("DEEPSEEK_API_KEY"):
        raise KeyError("DEEPSEEK_API_KEY is required when CITEEVAL_PROVIDER=deepseek")


def _validate_openai_model_and_credentials(model_name: str, env: dict[str, str], check_credentials: bool) -> None:
    if model_name != "chatgpt" and not model_name.startswith("gpt"):
        raise ValueError(
            f"Model '{model_name}' is incompatible with provider 'openai'. "
            "Use gpt-* or chatgpt."
        )
    if check_credentials and not env.get("OPENAI_API_KEY"):
        raise KeyError("OPENAI_API_KEY is required when CITEEVAL_PROVIDER=openai")


def _run_command(command: list[str], cwd: Path, env: dict[str, str], dry_run: bool, step: str) -> None:
    printable = " ".join(command)
    print(f"[{step}] {printable}")
    if dry_run:
        return

    proc = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
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


def _require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def preflight(paths: EvalPaths, track: str, metric_split: str, system_input: Path | None) -> None:
    _require_path(paths.citeeval_root, "CiteEval root")
    _require_path(paths.citeeval_src, "CiteEval src directory")
    _require_path(paths.data_root, "CiteEval data directory")

    if track in {"metric", "both"}:
        metric_file = paths.metric_root / f"metric_{metric_split}" / f"citebench.metric_{metric_split}"
        human_file = paths.metric_root / f"metric_{metric_split}" / f"citebench.metric_{metric_split}.human.out"
        _require_path(metric_file, f"metric_{metric_split} input file")
        _require_path(human_file, f"metric_{metric_split} human annotation file")

    if track in {"system", "both"}:
        if system_input is None:
            raise ValueError("system_input is required for system/both track")
        _require_path(system_input, "system evaluation input file")


def _module_output_file(output_dir: Path, response_output_file: Path, version: str, module: str, model_name: str) -> Path:
    return output_dir / f"{response_output_file.name}.{version}.{module}.{model_name}.out"


def _build_subset_file_name(source_file: Path, max_examples: int) -> str:
    stem = source_file.stem
    suffix = source_file.suffix
    return f"{stem}.subset_{max_examples}{suffix}"


def _write_subset_json(source_file: Path, destination_file: Path, max_examples: int) -> Path:
    with open(source_file, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError(f"Subset sampling requires list JSON. Got {type(payload).__name__} in {source_file}")

    subset = payload[:max_examples]
    destination_file.parent.mkdir(parents=True, exist_ok=True)
    with open(destination_file, "w", encoding="utf-8") as handle:
        json.dump(subset, handle, indent=2, ensure_ascii=False)

    print(f"[sampling] {source_file.name}: using {len(subset)} / {len(payload)} examples")
    return destination_file


def _maybe_subset_file(paths: EvalPaths, source_file: Path, max_examples: int | None, scope: str) -> Path:
    if not max_examples:
        return source_file

    sampled_dir = paths.temp_root / "sampling"
    sampled_file = sampled_dir / f"{scope}.{_build_subset_file_name(source_file, max_examples)}"
    return _write_subset_json(source_file=source_file, destination_file=sampled_file, max_examples=max_examples)


def run_metric_track(paths: EvalPaths, env: dict[str, str], args: argparse.Namespace) -> None:
    metric_file = paths.metric_root / f"metric_{args.metric_split}" / f"citebench.metric_{args.metric_split}"
    metric_file = _maybe_subset_file(paths, metric_file, args.max_examples, f"metric_{args.metric_split}")
    paths.metric_output_root.mkdir(parents=True, exist_ok=True)

    _run_command(
        [
            sys.executable,
            "-m",
            "scripts.run_citeeval",
            "--response_output_file",
            _to_posix(metric_file),
            "--eval_output_dir",
            _to_posix(paths.metric_output_root),
            "--modules",
            args.modules,
            "--version",
            args.version,
            "--model_name",
            args.model_name,
            "--n_threads",
            str(args.n_threads),
        ],
        cwd=paths.citeeval_src,
        env=env,
        dry_run=args.dry_run,
        step="metric.run_citeeval",
    )

    ca_out = _module_output_file(paths.metric_output_root, metric_file, args.version, "ca", args.model_name)
    cr_iter_out = _module_output_file(paths.metric_output_root, metric_file, args.version, "cr_itercoe", args.model_name)
    cr_edit_out = _module_output_file(paths.metric_output_root, metric_file, args.version, "cr_editdist", args.model_name)

    _run_command(
        [
            sys.executable,
            "-m",
            "scripts.evaluate_metric",
            "--metric",
            f"{args.version}.ca",
            "--metric_output",
            _to_posix(cr_edit_out),
            "--split",
            args.metric_split,
        ],
        cwd=paths.citeeval_src,
        env=env,
        dry_run=args.dry_run,
        step="metric.evaluate_ca",
    )

    _run_command(
        [
            sys.executable,
            "-m",
            "scripts.evaluate_metric",
            "--metric",
            f"{args.version}.cr",
            "--metric_output",
            f"{_to_posix(cr_iter_out)},{_to_posix(cr_edit_out)}",
            "--split",
            args.metric_split,
        ],
        cwd=paths.citeeval_src,
        env=env,
        dry_run=args.dry_run,
        step="metric.evaluate_cr",
    )


def _system_citeeval_input(system_input: Path) -> Path:
    if system_input.suffix == ".citeeval":
        return system_input
    if system_input.suffix != ".json":
        raise ValueError(
            f"Unsupported system input extension '{system_input.suffix}'. "
            "Use .json or .citeeval"
        )
    return system_input.with_suffix(".citeeval")


def run_system_track(paths: EvalPaths, env: dict[str, str], args: argparse.Namespace) -> None:
    system_input = Path(args.system_input)
    system_input = _maybe_subset_file(paths, system_input, args.max_examples, "system")
    citeeval_input = _system_citeeval_input(system_input)
    paths.system_eval_output_root.mkdir(parents=True, exist_ok=True)

    if system_input.suffix == ".json" and not args.skip_convert:
        _run_command(
            [
                sys.executable,
                "-m",
                "data.convert_to_citeeval_format",
                "--system_output_file",
                _to_posix(system_input),
            ],
            cwd=paths.citeeval_src,
            env=env,
            dry_run=args.dry_run,
            step="system.convert_to_citeeval",
        )

    if not args.dry_run:
        _require_path(citeeval_input, "converted .citeeval input file")

    _run_command(
        [
            sys.executable,
            "-m",
            "scripts.run_citeeval",
            "--response_output_file",
            _to_posix(citeeval_input),
            "--eval_output_dir",
            _to_posix(paths.system_eval_output_root),
            "--modules",
            args.modules,
            "--version",
            args.version,
            "--model_name",
            args.model_name,
            "--n_threads",
            str(args.n_threads),
        ],
        cwd=paths.citeeval_src,
        env=env,
        dry_run=args.dry_run,
        step="system.run_citeeval",
    )

    cr_iter_out = _module_output_file(paths.system_eval_output_root, citeeval_input, args.version, "cr_itercoe", args.model_name)
    cr_edit_out = _module_output_file(paths.system_eval_output_root, citeeval_input, args.version, "cr_editdist", args.model_name)

    command = [
        sys.executable,
        "-m",
        "scripts.evaluate_system",
        "--system_output",
        _to_posix(citeeval_input),
        "--metric_output",
        f"{_to_posix(cr_iter_out)},{_to_posix(cr_edit_out)}",
    ]
    if args.cited_only:
        command.append("--cited")

    _run_command(
        command,
        cwd=paths.citeeval_src,
        env=env,
        dry_run=args.dry_run,
        step="system.evaluate",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CiteBench/CiteEval metric and system evaluation with preflight checks.")
    parser.add_argument("--track", choices=["metric", "system", "both"], default="both")
    parser.add_argument("--metric-split", choices=["dev", "test"], default="test")
    parser.add_argument(
        "--system-input",
        type=str,
        default="benchmark/CiteEval/data/system_eval/system_eval_examples.json",
        help="Path to system evaluation input (.json or .citeeval)",
    )
    parser.add_argument("--modules", type=str, default="ca,ce,cr_itercoe,cr_editdist")
    parser.add_argument("--version", type=str, default="citeeval-auto-12272024")
    parser.add_argument("--provider", choices=["openai", "deepseek"], default=None, help="Override CITEEVAL_PROVIDER from environment")
    parser.add_argument("--model-name", type=str, default=None, help="LLM model. If omitted, defaults to deepseek-chat for deepseek provider, otherwise gpt-4o")
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--max-examples", type=int, default=None, help="Limit evaluation to first N examples for quick testing")
    parser.add_argument("--cited-only", action="store_true", help="Use cited-only scenario for system evaluation summary")
    parser.add_argument("--skip-convert", action="store_true", help="Skip .json -> .citeeval conversion in system track")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]

    try:
        paths = resolve_paths(project_root)
        system_input = Path(args.system_input).resolve() if args.track in {"system", "both"} else None
        preflight(paths, args.track, args.metric_split, system_input)

        env = _build_env(paths)
        provider, resolved_model_name = _resolve_provider_and_model(
            provider_override=args.provider,
            model_override=args.model_name,
            env=env,
            check_credentials=not args.dry_run,
        )
        args.model_name = resolved_model_name
        print(f"[config] provider={provider}, model={args.model_name}, dry_run={args.dry_run}")

        if args.track in {"metric", "both"}:
            run_metric_track(paths, env, args)
        if args.track in {"system", "both"}:
            run_system_track(paths, env, args)

        print("CiteBench evaluation workflow completed successfully.")
        return 0
    except Exception as exc:
        print(f"CiteBench evaluation workflow failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
