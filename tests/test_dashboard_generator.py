"""Tests for the static benchmark dashboard generator."""

from __future__ import annotations

import json
from pathlib import Path


from scripts.generate_dashboard import (
    _build_datasets,
    _load_curator_runs,
    _load_runs,
    _parse_filename,
    _render_curator_section,
    _run_sort_key,
    _summary_stats,
)


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
            "quality": {"verified": True, "reasons": []},
        },
        {
            "provider": "b",
            "model": "m2",
            "data": {
                "optimized": {"cache_hit_rate_mean": 0.75},
                "baseline": {"cache_hit_rate_mean": 0.0},
                "delta": {"cache_hit_rate_delta": 0.75, "cost_per_call_delta_usd": -0.02},
            },
            "quality": {"verified": True, "reasons": []},
        },
    ]
    stats = _summary_stats(runs)
    assert stats["runs"] == 2
    assert stats["verified"] == 2
    assert stats["wins"] == 2
    assert stats["best_hit"] == 0.75
    assert stats["best_cost_delta"] == 0.02
    assert stats["avg_delta"] == 0.625


def test_summary_stats_excludes_unverified_from_wins():
    """A run with a favorable delta but a failed quality gate isn't a 'win'."""
    runs = [
        {
            "provider": "a",
            "model": "m1",
            "data": {
                "optimized": {"cache_hit_rate_mean": 0.50},
                "baseline": {"cache_hit_rate_mean": 0.0},
                "delta": {"cache_hit_rate_delta": 0.50, "cost_per_call_delta_usd": -0.01},
            },
            "quality": {"verified": False, "reasons": ["n=5 < min_n=20"]},
        },
    ]
    stats = _summary_stats(runs)
    assert stats["verified"] == 0
    assert stats["wins"] == 0


def _curator_summary_fixture() -> dict:
    return {
        "provider": "fake",
        "model": "fake-model",
        "judge": "self",
        "n": 10,
        "raw": {
            "n": 10, "prompt_tokens_mean": 120.0, "completion_tokens_mean": 4.0,
            "cost_usd_per_call": 0.001, "cost_usd_total": 0.01,
            "latency_ms_p50": 50.0, "context_precision_mean": 0.2,
        },
        "curated": {
            "n": 10, "prompt_tokens_mean": 60.0, "completion_tokens_mean": 4.0,
            "cost_usd_per_call": 0.0005, "cost_usd_total": 0.005,
            "latency_ms_p50": 45.0, "context_precision_mean": 0.8,
        },
        "delta": {
            "prompt_tokens_delta_mean": -60.0,
            "cost_delta_usd_per_call": -0.0005,
            "context_precision_delta": 0.6,
        },
        "quality": {
            "relevance": {"baseline_mean": 0.7, "optimized_mean": 0.75, "delta": 0.05,
                          "n_baseline": 10, "n_optimized": 10, "n": 10,
                          "ci_low": 0.01, "ci_high": 0.09, "significant": True,
                          "effect_size_pct": 7.1},
        },
        "quality_gate": {
            "n": 10, "min_n": 20, "low_n": True, "verified": False,
            "significant_metrics": ["relevance"],
            "reasons": ["n=10 < min_n=20"],
            "per_metric": {"relevance": {"ci_low": 0.01, "ci_high": 0.09,
                                          "significant": True, "effect_size_pct": 7.1}},
        },
        "curation": {"mean_drop_rate": 0.5, "total_dedup_drops": 2},
    }


def _fake_summary(
    *, n: int = 25, errors: int = 0, ci_low: float = -0.001, ci_high: float = -0.0005,
    error_breakdown: dict | None = None,
) -> dict:
    """Minimal summary.json shape sufficient to drive `evaluate_quality_gate`
    deterministically for sort-order tests.
    """
    return {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "optimized": {"n": n, "errors": errors, "error_breakdown": error_breakdown or {}},
        "baseline": {"n": n, "errors": errors, "error_breakdown": error_breakdown or {}},
        "delta": {"cost_delta_ci_low_usd": ci_low, "cost_delta_ci_high_usd": ci_high},
    }


def test_run_sort_key_verified_before_unverified():
    verified_run = {"quality": {"verified": True, "confidence": "high"}}
    unverified_run = {"quality": {"verified": False, "confidence": "high"}}
    assert _run_sort_key(verified_run) < _run_sort_key(unverified_run)


def test_run_sort_key_orders_confidence_high_to_low():
    high = {"quality": {"verified": False, "confidence": "high"}}
    medium = {"quality": {"verified": False, "confidence": "medium"}}
    low = {"quality": {"verified": False, "confidence": "low"}}
    invalid = {"quality": {"verified": False, "confidence": "invalid"}}
    assert _run_sort_key(high) < _run_sort_key(medium) < _run_sort_key(low) < _run_sort_key(invalid)


def test_load_runs_sorts_verified_first_then_by_confidence(tmp_path, monkeypatch):
    results_dir = tmp_path / "bench" / "results"
    results_dir.mkdir(parents=True)
    # "z_" prefix so filename-alphabetical order would put it LAST if sorting
    # were still by filename — proves the new sort overrides that.
    (results_dir / "z_verified.summary.json").write_text(
        json.dumps(_fake_summary(n=25, errors=0, ci_low=-0.002, ci_high=-0.001))
    )
    (results_dir / "a_unverified_low_n.summary.json").write_text(
        json.dumps(_fake_summary(n=5, errors=0, ci_low=-0.002, ci_high=-0.001))
    )
    (results_dir / "m_unverified_invalid.summary.json").write_text(
        json.dumps(_fake_summary(n=25, errors=1, error_breakdown={"auth": 1}))
    )

    import scripts.generate_dashboard as gen

    monkeypatch.setattr(gen, "ROOT", tmp_path)
    monkeypatch.setattr(gen, "RESULTS_DIR", results_dir)
    runs = _load_runs()

    assert [r["path"].split("/")[-1] for r in runs] == [
        "z_verified.summary.json",
        "a_unverified_low_n.summary.json",
        "m_unverified_invalid.summary.json",
    ]
    assert runs[0]["quality"]["verified"] is True
    assert runs[-1]["quality"]["confidence"] == "invalid"


def test_load_curator_runs_reads_provider_model_from_json(tmp_path, monkeypatch):
    results_dir = tmp_path / "bench" / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "curator_fake.curator_summary.json").write_text(
        json.dumps(_curator_summary_fixture())
    )

    import scripts.generate_dashboard as gen

    monkeypatch.setattr(gen, "ROOT", tmp_path)
    monkeypatch.setattr(gen, "RESULTS_DIR", results_dir)
    runs = _load_curator_runs()
    assert len(runs) == 1
    assert runs[0]["provider"] == "fake"
    assert runs[0]["model"] == "fake-model"
    assert runs[0]["data"]["n"] == 10


def test_load_curator_runs_empty_dir_returns_empty_list(tmp_path, monkeypatch):
    results_dir = tmp_path / "bench" / "results"
    results_dir.mkdir(parents=True)

    import scripts.generate_dashboard as gen

    monkeypatch.setattr(gen, "ROOT", tmp_path)
    monkeypatch.setattr(gen, "RESULTS_DIR", results_dir)
    assert _load_curator_runs() == []


def test_render_curator_section_empty_returns_empty_string():
    assert _render_curator_section([]) == ""


def test_render_curator_section_renders_provider_and_metrics():
    runs = [{"provider": "fake", "model": "fake-model", "data": _curator_summary_fixture()}]
    html = _render_curator_section(runs)
    assert "RAG Curator Bench" in html
    assert "fake" in html
    assert "fake-model" in html
    assert "relevance" in html


def test_render_curator_section_shows_unverified_badge_for_low_n():
    """Fixture's quality_gate has verified=False (n < min_n) — must render
    the amber "unverified" badge, not "verified".
    """
    runs = [{"provider": "fake", "model": "fake-model", "data": _curator_summary_fixture()}]
    html = _render_curator_section(runs)
    assert "unverified" in html
    assert "Quality gate" in html


def test_render_curator_section_shows_verified_badge_when_gate_passes():
    data = _curator_summary_fixture()
    data["quality_gate"] = {
        "n": 25, "min_n": 20, "low_n": False, "verified": True,
        "significant_metrics": ["relevance"], "reasons": [], "per_metric": {},
    }
    runs = [{"provider": "fake", "model": "fake-model", "data": data}]
    html = _render_curator_section(runs)
    assert ">verified<" in html


def test_render_curator_section_missing_quality_gate_renders_na_badge():
    """Older result files (from before quality_gate existed) shouldn't crash
    the dashboard — should degrade to an "n/a" badge.
    """
    data = _curator_summary_fixture()
    del data["quality_gate"]
    runs = [{"provider": "fake", "model": "fake-model", "data": data}]
    html = _render_curator_section(runs)
    assert ">n/a<" in html


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


def test_generate_dashboard_script_with_only_curator_runs(tmp_path, monkeypatch):
    """Regression test: the dashboard must not early-return when there are
    zero *.summary.json files but curator bench results exist.
    """
    results_dir = tmp_path / "bench" / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "curator_fake.curator_summary.json").write_text(
        json.dumps(_curator_summary_fixture())
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
    assert "RAG Curator Bench" in html
    assert "fake-model" in html


def test_generate_dashboard_script_with_both_run_types(tmp_path, monkeypatch):
    """Both *.summary.json and *.curator_summary.json files render their
    respective sections in the same dashboard.
    """
    results_dir = tmp_path / "bench" / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "cloud_openai_gpt-4o-mini.summary.json").write_text(
        json.dumps({
            "optimized": {"n": 10, "cache_hit_rate_mean": 0.78, "cost_usd_per_call": 0.001,
                          "latency_ms_p50": 100.0},
            "baseline": {"n": 10, "cache_hit_rate_mean": 0.05, "cost_usd_per_call": 0.005,
                         "latency_ms_p50": 120.0},
            "delta": {"cache_hit_rate_delta": 0.73, "cost_per_call_delta_usd": -0.004,
                      "prompt_tokens_delta_mean": -2.0},
        })
    )
    (results_dir / "curator_fake.curator_summary.json").write_text(
        json.dumps(_curator_summary_fixture())
    )

    output_dir = tmp_path / "docs" / "dashboard"

    import scripts.generate_dashboard as gen

    monkeypatch.setattr(gen, "ROOT", tmp_path)
    monkeypatch.setattr(gen, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(gen, "OUTPUT_DIR", output_dir)
    gen.main()

    html = (output_dir / "index.html").read_text()
    assert "gpt-4o-mini" in html
    assert "RAG Curator Bench" in html
    assert "fake-model" in html
