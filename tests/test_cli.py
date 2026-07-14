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


# --- doctor (offline via mock client) ---

def test_run_doctor_detects_cache_activation():
    """run_doctor reports activated=True when calls 2+ have cached_tokens > 0."""
    from contextops.cli import run_doctor

    class FakeClient:
        PROVIDER = "fake"
        DEFAULT_MODEL = "x"
        _call = 0

        def complete(self, *, model, messages, temperature=0.0, max_tokens=64, system=None):
            self._call += 1
            from contextops_bench.types import CompletionResponse
            # First call cold; subsequent calls warm.
            cached = 0 if self._call == 1 else 900
            return CompletionResponse(
                text="ok", prompt_tokens=1000, completion_tokens=4,
                cached_tokens=cached, cost_usd=0.0, model=model, raw={},
            )

    result = run_doctor(FakeClient(), system_content="STABLE PREFIX", n_calls=3)
    assert result["activated"] is True
    assert result["calls"][0]["cached_tokens"] == 0  # cold
    assert result["calls"][1]["cached_tokens"] == 900  # warm
    assert "cache active" in result["summary"]


def test_run_doctor_reports_no_activation():
    """run_doctor reports activated=False when cache never engages."""
    from contextops.cli import run_doctor

    class ColdClient:
        PROVIDER = "fake"
        DEFAULT_MODEL = "x"

        def complete(self, *, model, messages, temperature=0.0, max_tokens=64, system=None):
            from contextops_bench.types import CompletionResponse
            return CompletionResponse(
                text="ok", prompt_tokens=1000, completion_tokens=4,
                cached_tokens=0, cost_usd=0.0, model=model, raw={},
            )

    result = run_doctor(ColdClient(), system_content="STABLE PREFIX", n_calls=3)
    assert result["activated"] is False
    assert "NOT active" in result["summary"]
