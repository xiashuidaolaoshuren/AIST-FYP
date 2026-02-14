from pathlib import Path

import pytest

from scripts.run_citebench_eval import (
    _maybe_subset_file,
    _module_output_file,
    _resolve_provider_and_model,
    _system_citeeval_input,
    build_parser,
    preflight,
    resolve_metric_root,
    resolve_paths,
)


def test_resolve_metric_root_prefers_citebench_layout(tmp_path: Path):
    data_root = tmp_path / "data"
    (data_root / "citebench" / "metric_eval").mkdir(parents=True)
    (data_root / "metric_eval").mkdir(parents=True)

    resolved = resolve_metric_root(data_root)
    assert resolved == data_root / "citebench" / "metric_eval"


def test_resolve_metric_root_falls_back_to_flat_layout(tmp_path: Path):
    data_root = tmp_path / "data"
    (data_root / "metric_eval").mkdir(parents=True)

    resolved = resolve_metric_root(data_root)
    assert resolved == data_root / "metric_eval"


def test_system_citeeval_input_conversion_rules():
    assert _system_citeeval_input(Path("a.json")) == Path("a.citeeval")
    assert _system_citeeval_input(Path("a.citeeval")) == Path("a.citeeval")

    with pytest.raises(ValueError):
        _system_citeeval_input(Path("a.txt"))


def test_module_output_file_naming_contract():
    output = _module_output_file(
        output_dir=Path("out"),
        response_output_file=Path("citebench.metric_test"),
        version="citeeval-auto-12272024",
        module="ca",
        model_name="gpt-4o",
    )
    assert output == Path("out") / "citebench.metric_test.citeeval-auto-12272024.ca.gpt-4o.out"


def test_parser_defaults():
    parser = build_parser()
    args = parser.parse_args([])

    assert args.track == "both"
    assert args.metric_split == "test"
    assert args.version == "citeeval-auto-12272024"
    assert args.provider is None
    assert args.model_name is None
    assert args.max_examples is None


def test_resolve_provider_and_model_defaults_to_deepseek_chat():
    env = {"CITEEVAL_PROVIDER": "deepseek", "DEEPSEEK_API_KEY": "test-key"}
    provider, model = _resolve_provider_and_model(
        provider_override=None,
        model_override=None,
        env=env,
        check_credentials=True,
    )
    assert provider == "deepseek"
    assert model == "deepseek-chat"


def test_resolve_provider_and_model_rejects_mismatch():
    env = {"CITEEVAL_PROVIDER": "deepseek", "DEEPSEEK_API_KEY": "test-key"}
    with pytest.raises(ValueError):
        _resolve_provider_and_model(
            provider_override=None,
            model_override="gpt-4o",
            env=env,
            check_credentials=True,
        )


def test_subset_sampling_writes_first_n_records(tmp_path: Path):
    source = tmp_path / "input.json"
    source.write_text('[{"id":1},{"id":2},{"id":3}]', encoding="utf-8")

    paths = resolve_paths(_make_project_layout(tmp_path))
    sampled = _maybe_subset_file(paths, source, max_examples=2, scope="system")

    assert sampled.exists()
    payload = sampled.read_text(encoding="utf-8")
    assert '"id": 1' in payload
    assert '"id": 2' in payload
    assert '"id": 3' not in payload


def test_subset_sampling_skips_when_disabled(tmp_path: Path):
    source = tmp_path / "input.json"
    source.write_text('[]', encoding="utf-8")

    paths = resolve_paths(_make_project_layout(tmp_path))
    resolved = _maybe_subset_file(paths, source, max_examples=None, scope="metric_test")
    assert resolved == source


def _make_project_layout(tmp_path: Path) -> Path:
    project_root = tmp_path / "proj"
    (project_root / "benchmark" / "CiteEval" / "src").mkdir(parents=True)
    (project_root / "benchmark" / "CiteEval" / "data" / "metric_eval").mkdir(parents=True)
    return project_root


def test_preflight_checks_metric_and_system_contracts(tmp_path: Path):
    project_root = tmp_path / "proj"
    citeeval_src = project_root / "benchmark" / "CiteEval" / "src"
    data_root = project_root / "benchmark" / "CiteEval" / "data"
    citeeval_src.mkdir(parents=True)
    (data_root / "metric_eval" / "metric_test").mkdir(parents=True)
    (data_root / "dev").mkdir(parents=True)
    (data_root / "test").mkdir(parents=True)
    (data_root / "system_eval").mkdir(parents=True)

    metric_file = data_root / "metric_eval" / "metric_test" / "citebench.metric_test"
    human_file = data_root / "metric_eval" / "metric_test" / "citebench.metric_test.human.out"
    system_file = data_root / "system_eval" / "system_eval_examples.json"

    metric_file.write_text("[]", encoding="utf-8")
    human_file.write_text("[]", encoding="utf-8")
    system_file.write_text("[]", encoding="utf-8")

    paths = resolve_paths(project_root)
    preflight(paths, track="both", metric_split="test", system_input=system_file)
