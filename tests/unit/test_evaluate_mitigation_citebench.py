from pathlib import Path

import pytest

from scripts.evaluate_mitigation_citebench import (
    _parse_evaluate_system_stdout,
    _variant_patch,
    _write_summary,
)


def test_variant_patch_baseline_disables_all_mitigation_branches():
    patch = _variant_patch("baseline")
    assert patch["mitigation"]["enabled"] is False
    assert patch["mitigation"]["reranker"]["enabled"] is False
    assert patch["mitigation"]["filter"]["enabled"] is False
    assert patch["mitigation"]["reprompt"]["enabled"] is False


def test_variant_patch_mitigation_all_enables_all_branches():
    patch = _variant_patch("mitigation_all")
    assert patch["mitigation"]["enabled"] is True
    assert patch["mitigation"]["reranker"]["enabled"] is True
    assert patch["mitigation"]["filter"]["enabled"] is True
    assert patch["mitigation"]["reprompt"]["enabled"] is True


def test_parse_evaluate_system_stdout_extracts_numeric_metrics():
    stdout = """
    Some header
    statement_rating: 0.7125
    response_rating: 0.6875
    length: 4.20
    density: 0.3300
    """
    parsed = _parse_evaluate_system_stdout(stdout)
    assert parsed == {
        "statement_rating": 0.7125,
        "response_rating": 0.6875,
        "length": 4.2,
        "density": 0.33,
    }


def test_parse_evaluate_system_stdout_raises_on_missing_field():
    stdout = "statement_rating: 0.5\nresponse_rating: 0.5\nlength: 3.0\n"
    with pytest.raises(ValueError):
        _parse_evaluate_system_stdout(stdout)


def test_write_summary_contains_delta_columns(tmp_path: Path):
    summary_file = tmp_path / "summary.md"
    metrics = {
        "baseline": {
            "statement_rating": 0.5,
            "response_rating": 0.4,
            "length": 3.0,
            "density": 0.2,
        },
        "mitigation_all": {
            "statement_rating": 0.7,
            "response_rating": 0.45,
            "length": 3.2,
            "density": 0.25,
        },
    }

    _write_summary(summary_file, metrics)
    content = summary_file.read_text(encoding="utf-8")

    assert "ΔStatement vs Baseline" in content
    assert "| mitigation_all | 0.7000 | 0.4500 | 3.20 | 0.2500 | +0.2000 | +0.0500 |" in content
