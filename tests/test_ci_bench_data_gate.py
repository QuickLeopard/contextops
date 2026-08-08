"""Tests for the CI bench data quality gate (`scripts/ci_bench_data_gate.py`).

Only `check_summary_file` is unit-tested here — it's the pure, file-in
file-out logic. `changed_summary_files` is a thin `git diff` subprocess
wrapper (same style as the existing `scripts/ci_bench_gate.py`, which also
has no unit tests for its subprocess-driving `main()`) and is exercised by
the workflow itself, not by pytest.
"""

from __future__ import annotations

import json

from scripts.ci_bench_data_gate import check_summary_file


def _write_summary(tmp_path, name: str, data: dict):
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def _summary(*, n=25, errors=0, ci_low=-0.002, ci_high=-0.001, error_breakdown=None):
    return {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "optimized": {"n": n, "errors": errors, "error_breakdown": error_breakdown or {}},
        "baseline": {"n": n, "errors": errors, "error_breakdown": error_breakdown or {}},
        "delta": {"cost_delta_ci_low_usd": ci_low, "cost_delta_ci_high_usd": ci_high},
    }


def test_check_summary_file_passes_verified_run(tmp_path):
    path = _write_summary(tmp_path, "cloud_openai_gpt-4o-mini.summary.json", _summary())
    ok, reasons = check_summary_file(path)
    assert ok is True
    assert reasons == []


def test_check_summary_file_fails_low_n(tmp_path):
    path = _write_summary(
        tmp_path, "cloud_openai_gpt-4o-mini.summary.json", _summary(n=5)
    )
    ok, reasons = check_summary_file(path)
    assert ok is False
    assert any("n=5" in r for r in reasons)


def test_check_summary_file_fails_auth_errors(tmp_path):
    path = _write_summary(
        tmp_path, "cloud_openai_gpt-4o-mini.summary.json",
        _summary(errors=1, error_breakdown={"auth": 1}),
    )
    ok, reasons = check_summary_file(path)
    assert ok is False
    assert any("auth" in r.lower() for r in reasons)


def test_check_summary_file_fails_insignificant_ci(tmp_path):
    path = _write_summary(
        tmp_path, "cloud_openai_gpt-4o-mini.summary.json",
        _summary(ci_low=-0.001, ci_high=0.001),
    )
    ok, reasons = check_summary_file(path)
    assert ok is False
    assert any("includes zero" in r for r in reasons)


def test_check_summary_file_fails_on_missing_file(tmp_path):
    ok, reasons = check_summary_file(tmp_path / "does_not_exist.summary.json")
    assert ok is False
    assert reasons and "does not exist" in reasons[0]


def test_check_summary_file_fails_on_malformed_json(tmp_path):
    path = tmp_path / "broken.summary.json"
    path.write_text("{not valid json")
    ok, reasons = check_summary_file(path)
    assert ok is False
    assert reasons and "not valid JSON" in reasons[0]


def test_check_summary_file_fails_on_non_dict_json(tmp_path):
    path = tmp_path / "list.summary.json"
    path.write_text(json.dumps([1, 2, 3]))
    ok, reasons = check_summary_file(path)
    assert ok is False
    assert reasons and "JSON object" in reasons[0]


def test_check_summary_file_fails_cache_marker_dropped_path(tmp_path):
    """A summary.json routed through OpenRouter for an Anthropic model is a
    known cache-marker-drop path — must fail even with a clean CI/error rate.
    """
    data = _summary()
    data["provider"] = "openrouter"
    data["model"] = "anthropic/claude-3-5-sonnet"
    path = _write_summary(tmp_path, "cloud_anthropic_claude-3-5-sonnet.summary.json", data)
    ok, reasons = check_summary_file(path)
    assert ok is False
    assert any("cache_hit_rate is not measurable" in r for r in reasons)
