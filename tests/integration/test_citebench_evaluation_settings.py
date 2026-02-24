"""Integration-level settings tests for CiteBench/CiteEval evaluation workflows.

These tests validate local dataset layout, official split semantics, and
configuration contracts that should hold before running expensive evaluations.
"""

import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CITEEVAL_ROOT = PROJECT_ROOT / "benchmark" / "CiteEval"
DATA_ROOT = CITEEVAL_ROOT / "data"

pytestmark = [pytest.mark.integration, pytest.mark.requires_data]


def _metric_eval_root() -> Path:
    preferred = DATA_ROOT / "citebench" / "metric_eval"
    fallback = DATA_ROOT / "metric_eval"
    if preferred.exists():
        return preferred
    return fallback


def test_official_dataset_split_directories_exist():
    metric_root = _metric_eval_root()

    assert CITEEVAL_ROOT.exists(), "CiteEval repository directory is missing"
    assert DATA_ROOT.exists(), "CiteEval data directory is missing"
    assert metric_root.exists(), "CiteBench metric_eval directory is missing"
    assert (DATA_ROOT / "dev").exists(), "Full dev split directory is missing"
    assert (DATA_ROOT / "test").exists(), "Full test split directory is missing"
    assert (DATA_ROOT / "system_eval").exists(), "System eval directory is missing"


def test_metric_eval_files_and_human_annotations_exist():
    metric_root = _metric_eval_root()

    for split in ("dev", "test"):
        split_dir = metric_root / f"metric_{split}"
        eval_file = split_dir / f"citebench.metric_{split}"
        human_file = split_dir / f"citebench.metric_{split}.human.out"

        assert split_dir.exists(), f"Missing metric split directory: {split_dir}"
        assert eval_file.exists(), f"Missing metric eval file: {eval_file}"
        assert human_file.exists(), f"Missing human annotation file: {human_file}"


def test_full_dev_test_splits_have_no_human_annotation_files():
    for split_dir in (DATA_ROOT / "dev", DATA_ROOT / "test"):
        human_files = list(split_dir.rglob("*.human.out"))
        assert not human_files, f"Unexpected human annotation files in {split_dir}"


def test_system_eval_example_matches_expected_input_schema():
    system_eval_file = DATA_ROOT / "system_eval" / "system_eval_examples.json"
    assert system_eval_file.exists(), "Missing system eval example file"

    with open(system_eval_file, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert isinstance(payload, list) and payload, "System eval example must be a non-empty list"
    first = payload[0]

    required_keys = {"id", "query", "passages", "pred"}
    assert required_keys.issubset(first.keys()), "System eval records must include id/query/passages/pred"
    assert isinstance(first["passages"], list), "passages must be a list"


def test_run_citeeval_cli_contract_uses_current_flag_names():
    run_script = CITEEVAL_ROOT / "src" / "scripts" / "run_citeeval.py"
    content = run_script.read_text(encoding="utf-8")

    assert "--response_output_file" in content
    assert "--eval_output_dir" in content
    assert "--input_file" not in content
    assert "--output_dir" not in content


def test_project_guide_matches_run_citeeval_flag_names():
    guide = PROJECT_ROOT / "docs" / "citeeval_evaluation_guide.md"
    content = guide.read_text(encoding="utf-8")

    assert "--response_output_file" in content
    assert "--eval_output_dir" in content
    assert "--input_file" not in content
    assert "--output_dir" not in content


def test_metric_eval_script_supports_both_metric_layouts():
    script = CITEEVAL_ROOT / "src" / "run_metric_eval.sh"
    content = script.read_text(encoding="utf-8")

    assert "data/citebench/metric_eval" in content
    assert "data/metric_eval" in content
