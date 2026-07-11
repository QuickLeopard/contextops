"""Smoke tests for the library CLI via click.testing.CliRunner.

The CLI previously had zero test coverage. These exercise the command wiring
and happy paths offline (echo judge / echo run_fn) without hitting a real API.
"""

import json

from click.testing import CliRunner

from contextops.cli import main


def test_version():
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.3.0" in result.output


def test_optimize_from_flags():
    """`contextops optimize` with inline flags exits 0 and renders a table."""
    result = CliRunner().invoke(main, [
        "optimize", "--system", "You are helpful", "--query", "What is 2+2?",
        "--model", "gpt-4o",
    ])
    assert result.exit_code == 0
    assert "ContextOps Optimization" in result.output
    assert "Cache hit rate" in result.output


def test_optimize_from_json(tmp_path):
    p = tmp_path / "prompt.json"
    p.write_text(json.dumps({
        "system": "sys", "query": "q", "model": "gpt-4o-mini", "goal": "cache_friendly",
    }))
    result = CliRunner().invoke(main, ["optimize", "--from-json", str(p)])
    assert result.exit_code == 0
    assert "Tokens" in result.output


def test_stats_on_empty_db(tmp_path):
    """`contextops stats` against a fresh DB should not crash."""
    db = tmp_path / "empty.db"
    result = CliRunner().invoke(main, ["stats", "--db", str(db)])
    assert result.exit_code == 0


def test_recent_on_empty_db(tmp_path):
    db = tmp_path / "empty.db"
    result = CliRunner().invoke(main, ["recent", "--db", str(db)])
    assert result.exit_code == 0


def test_compare(tmp_path):
    """`contextops compare` with one JSON auto-optimizes the baseline."""
    p = tmp_path / "baseline.json"
    p.write_text(json.dumps({"system": "s", "query": "q", "model": "gpt-4o"}))
    result = CliRunner().invoke(main, ["compare", str(p)])
    assert result.exit_code == 0


def test_eval_echo_offline(tmp_path):
    """`contextops eval --echo` runs the full A/B pipeline offline."""
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"system": "s", "query": "q", "model": "gpt-4o-mini"}))
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(json.dumps({"query": "q", "expected": "4"}) + "\n")
    result = CliRunner().invoke(main, [
        "eval", "--baseline", str(baseline), "--dataset", str(dataset), "--echo",
    ])
    assert result.exit_code == 0
    assert "A/B Eval Report" in result.output
