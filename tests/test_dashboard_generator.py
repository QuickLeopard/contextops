"""Tests for the static benchmark dashboard generator."""

from __future__ import annotations

import json
from pathlib import Path


from scripts.generate_dashboard import _build_datasets, _parse_filename, _summary_stats


def test_parse_filename_extracts_provider_and_model():
    assert _parse_filename(Path("cloud_openai_gpt-4o-mini.summary.json")) == (
        "openai",
        "gpt-4o-mini",
    )
    assert _parse_filename(Path("mac_zen_n5.summary.json")) == ("zen", "n5")
    assert _parse_filename(Path("local_ollama.summary.json")) == ("ollama", "ollama")


def test_build_datasets_computes_expected_shapes():
    runs = [
        {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "data": {
                "optimized": {"cache_hit_rate_mean": 0.78, "cost_usd_per_call": 0.001},
                "baseline": {"cache_hit_rate_mean": 0.05, "cost_usd_per_call": 0.005},
                "delta": {
                    "cache_hit_rate_delta": 0.73,
                    "cost_per_call_delta_usd": -0.004,
                },
            },
        }
    ]
    datasets = _build_datasets(runs)
    assert datasets["labels"] == ["openai · gpt-4o-mini"]
    assert datasets["optimized_hits"] == [0.78]
    assert datasets["baseline_hits"] == [0.05]
    assert datasets["optimized_costs"] == [0.001]
    assert datasets["baseline_costs"] == [0.005]
    assert datasets["deltas"] == [0.73]


def test_summary_stats_aggregates_runs():
    runs = [
        {
            "provider": "a",
            "model": "m1",
            "data": {
                "optimized": {"cache_hit_rate_mean": 0.50},
                "baseline": {"cache_hit_rate_mean": 0.0},
                "delta": {"cache_hit_rate_delta": 0.50, "cost_per_call_delta_usd": -0.01},
            },
        },
        {
            "provider": "b",
            "model": "m2",
            "data": {
                "optimized": {"cache_hit_rate_mean": 0.75},
                "baseline": {"cache_hit_rate_mean": 0.0},
                "delta": {"cache_hit_rate_delta": 0.75, "cost_per_call_delta_usd": -0.02},
            },
        },
    ]
    stats = _summary_stats(runs)
    assert stats["runs"] == 2
    assert stats["wins"] == 2
    assert stats["best_hit"] == 0.75
    assert stats["best_cost_delta"] == 0.02
    assert stats["avg_delta"] == 0.625


def test_generate_dashboard_script_produces_html(tmp_path, monkeypatch):
    """End-to-end test: run the generator module against a fake bench directory."""
    results_dir = tmp_path / "bench" / "results"
    results_dir.mkdir(parents=True)
    summary = {
        "optimized": {
            "n": 10,
            "cache_hit_rate_mean": 0.78,
            "cost_usd_per_call": 0.001,
            "latency_ms_p50": 100.0,
        },
        "baseline": {
            "n": 10,
            "cache_hit_rate_mean": 0.05,
            "cost_usd_per_call": 0.005,
            "latency_ms_p50": 120.0,
        },
        "delta": {
            "cache_hit_rate_delta": 0.73,
            "cost_per_call_delta_usd": -0.004,
            "prompt_tokens_delta_mean": -2.0,
        },
    }
    (results_dir / "cloud_openai_gpt-4o-mini.summary.json").write_text(
        json.dumps(summary)
    )

    output_dir = tmp_path / "docs" / "dashboard"

    import scripts.generate_dashboard as gen

    monkeypatch.setattr(gen, "ROOT", tmp_path)
    monkeypatch.setattr(gen, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(gen, "OUTPUT_DIR", output_dir)
    gen.main()

    output = output_dir / "index.html"
    assert output.exists()
    html = output.read_text()
    assert "ContextOps Benchmark Dashboard" in html
    assert "gpt-4o-mini" in html
    assert "74.9%" not in html  # only real file has this; our fixture is 78%
    assert "78.0%" in html or "0.78" in html or "78%" in html
    assert "chart.js" in html
    assert "tailwindcss" in html
