from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_ragtruth_baseline as baseline


class _Completed:
    def __init__(self, returncode: int):
        self.returncode = returncode


def _base_args(**overrides):
    data = {
        "baseline_dir": ".",
        "dry_run": False,
        "profile": "single-gpu",
        "model_path": baseline.DEFAULT_MODEL_PATH,
        "model_name": "baseline",
        "num_train_epochs": 1,
        "learning_rate": 2e-5,
        "model_max_length": 4096,
        "train_batch_size": 1,
        "eval_batch_size": 1,
        "grad_accum_steps": 16,
        "logging_steps": 1,
        "eval_steps": 80,
        "save_steps": 10000,
        "warmup_ratio": 0.1,
        "nproc_per_node": 4,
        "cuda_visible_devices": None,
        "report_to_wandb": False,
        "wandb_api_key": None,
        "wandb_project": None,
        "tokenizer": baseline.DEFAULT_MODEL_PATH,
        "output_file": "./prediction.jsonl",
        "metrics_file": "./baseline_eval_metrics.json",
        "run_evaluate": False,
        "container_name": "baseline",
        "model_subdir": "baseline",
        "gpu_device": "0",
        "port": 8300,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_build_train_command_exact_profile_includes_torchrun_fsdp_flashatt():
    args = _base_args(profile="exact", train_batch_size=8, eval_batch_size=8)

    cmd = baseline._build_train_command(args)

    assert cmd[0] == "torchrun"
    assert "--nproc_per_node" in cmd
    assert "--fsdp" in cmd
    assert "--fsdp_config" in cmd
    assert "./configs/fsdp.json" in cmd
    assert "--use_flashatt_2" in cmd
    assert "True" in cmd
    assert f"./exp/{args.model_name}" in cmd


def test_build_train_command_single_gpu_uses_python_train_and_disables_flashatt():
    args = _base_args(profile="single-gpu")

    cmd = baseline._build_train_command(args)

    assert cmd[0] == baseline.sys.executable
    assert cmd[1] == "train.py"
    assert "--fsdp" not in cmd
    assert "--fsdp_config" not in cmd
    idx = cmd.index("--use_flashatt_2")
    assert cmd[idx + 1] == "False"


@pytest.mark.parametrize("enabled", [True, False])
def test_build_train_command_report_to_wandb_toggle(enabled: bool):
    args = _base_args(profile="single-gpu", report_to_wandb=enabled)

    cmd = baseline._build_train_command(args)

    if enabled:
        assert "--report_to" in cmd
        idx = cmd.index("--report_to")
        assert cmd[idx + 1] == "wandb"
    else:
        assert "--report_to" not in cmd


def test_run_dry_run_skips_subprocess_and_returns_zero(monkeypatch):
    called = {"value": False}

    def _fake_subprocess_run(*_args, **_kwargs):
        called["value"] = True
        return _Completed(1)

    monkeypatch.setattr(baseline.subprocess, "run", _fake_subprocess_run)

    rc = baseline._run(["python", "x.py"], cwd=Path("."), dry_run=True)

    assert rc == 0
    assert called["value"] is False


def test_run_executes_subprocess_and_returns_returncode(monkeypatch):
    received = {}

    def _fake_subprocess_run(cmd, cwd=None, env=None):
        received["cmd"] = cmd
        received["cwd"] = cwd
        received["env"] = env
        return _Completed(7)

    monkeypatch.setattr(baseline.subprocess, "run", _fake_subprocess_run)

    rc = baseline._run(["python", "x.py"], cwd=Path("abc"), env={"K": "V"}, dry_run=False)

    assert rc == 7
    assert received["cmd"] == ["python", "x.py"]
    assert received["cwd"].endswith("abc")
    assert received["env"]["K"] == "V"


def test_command_prepare_builds_expected_command_and_cwd(monkeypatch, tmp_path: Path):
    args = _base_args(baseline_dir=str(tmp_path), dry_run=True)
    captured = {}

    def _fake_run(cmd, cwd, env=None, dry_run=False):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env
        captured["dry_run"] = dry_run
        return 0

    monkeypatch.setattr(baseline, "_run", _fake_run)

    rc = baseline.command_prepare(args)

    assert rc == 0
    assert captured["cmd"] == [baseline.sys.executable, "prepare_dataset.py"]
    assert captured["cwd"] == tmp_path.resolve()
    assert captured["dry_run"] is True


def test_command_evaluate_builds_expected_command(monkeypatch, tmp_path: Path):
    args = _base_args(
        baseline_dir=str(tmp_path),
        model_name="m1",
        tokenizer="tok",
        output_file="./pred.jsonl",
        metrics_file="./metrics.json",
        dry_run=True,
    )
    captured = {}

    def _fake_run(cmd, cwd, env=None, dry_run=False):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["dry_run"] = dry_run
        return 0

    monkeypatch.setattr(baseline, "_run", _fake_run)

    rc = baseline.command_evaluate(args)

    assert rc == 0
    assert captured["cmd"] == [
        baseline.sys.executable,
        "predict_and_evaluate.py",
        "--raw_dataset",
        "./test.jsonl",
        "--output_file",
        "./pred.jsonl",
        "--metrics_output",
        "./metrics.json",
        "--model_name",
        "m1",
        "--tokenizer",
        "tok",
    ]
    assert captured["cwd"] == tmp_path.resolve()
    assert captured["dry_run"] is True


def test_command_train_sets_selected_env_vars_only(monkeypatch, tmp_path: Path):
    args = _base_args(
        baseline_dir=str(tmp_path),
        cuda_visible_devices="0",
        wandb_api_key="wk",
        wandb_project="wp",
        dry_run=True,
    )
    captured = {}

    monkeypatch.setattr(baseline, "_build_train_command", lambda _a: ["traincmd"])

    def _fake_run(cmd, cwd, env=None, dry_run=False):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env
        captured["dry_run"] = dry_run
        return 0

    monkeypatch.setattr(baseline, "_run", _fake_run)

    rc = baseline.command_train(args)

    assert rc == 0
    assert captured["cmd"] == ["traincmd"]
    assert captured["cwd"] == tmp_path.resolve()
    assert captured["dry_run"] is True
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert captured["env"]["WANDB_API_KEY"] == "wk"
    assert captured["env"]["WANDB_PROJECT"] == "wp"


def test_command_all_short_circuits_on_prepare_failure(monkeypatch):
    args = _base_args()
    called = {"train": 0, "eval": 0}

    monkeypatch.setattr(baseline, "command_prepare", lambda _a: 2)
    monkeypatch.setattr(baseline, "command_train", lambda _a: called.__setitem__("train", 1) or 0)
    monkeypatch.setattr(baseline, "command_evaluate", lambda _a: called.__setitem__("eval", 1) or 0)

    rc = baseline.command_all(args)

    assert rc == 2
    assert called["train"] == 0
    assert called["eval"] == 0


def test_command_all_runs_evaluate_only_when_enabled(monkeypatch):
    args = _base_args(run_evaluate=False)
    called = {"eval": 0}

    monkeypatch.setattr(baseline, "command_prepare", lambda _a: 0)
    monkeypatch.setattr(baseline, "command_train", lambda _a: 0)
    monkeypatch.setattr(baseline, "command_evaluate", lambda _a: called.__setitem__("eval", 1) or 9)

    rc = baseline.command_all(args)
    assert rc == 0
    assert called["eval"] == 0

    args2 = _base_args(run_evaluate=True)
    rc2 = baseline.command_all(args2)
    assert rc2 == 9
    assert called["eval"] == 1


def test_build_docker_command_windows_and_posix(monkeypatch):
    args = _base_args(container_name="c1", model_subdir="m1", gpu_device="7", port=8310)
    base_dir = Path("/tmp/baseline")

    monkeypatch.setattr(baseline.os, "name", "nt", raising=False)
    cmd_win = baseline._build_docker_command(args, base_dir)
    assert "--name c1" in cmd_win
    assert "device=7" in cmd_win
    assert "-p 8310:80" in cmd_win
    assert "--model-id /data/exp/m1" in cmd_win
    assert f'"{base_dir}:/data"' in cmd_win

    monkeypatch.setattr(baseline.os, "name", "posix", raising=False)
    cmd_posix = baseline._build_docker_command(args, base_dir)
    assert '"$PWD:/data"' in cmd_posix


def test_main_routes_prepare_subcommand(monkeypatch):
    called = {"value": False}

    def _fake_prepare(args):
        called["value"] = True
        assert isinstance(args, argparse.Namespace)
        return 0

    monkeypatch.setattr(baseline, "command_prepare", _fake_prepare)
    monkeypatch.setattr(baseline.sys, "argv", ["prog", "prepare"])

    rc = baseline.main()

    assert rc == 0
    assert called["value"] is True


def test_main_requires_subcommand(monkeypatch):
    monkeypatch.setattr(baseline.sys, "argv", ["prog"])

    with pytest.raises(SystemExit):
        baseline.main()
