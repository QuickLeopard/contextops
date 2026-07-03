"""Tests for v0.2 eval pipeline. Uses EchoJudge + stub run_fn for offline CI."""

from __future__ import annotations

from contextops.clients import EchoJudge
from contextops.dataset import DatasetItem, load as load_dataset
from contextops.eval import compare, evaluate, evaluate_ab
from contextops.judge import list_metrics, score_many, score_one
from contextops.models import Prompt
from contextops.report import a_b_compare, aggregate, render_table
from contextops.clients import CallableJudge


SAMPLE_DATASET = [
    DatasetItem(query="What is 2+2?", expected="4", context="Basic arithmetic."),
    DatasetItem(query="Capital of Japan?", expected="Tokyo", context="Japan's capital is Tokyo."),
    DatasetItem(query="Color of the sky?", expected="blue", context="The sky appears blue."),
]


def _stub_run_fn(prompt_str: str) -> str:
    """Pretend LLM that returns 'echo of query' — judge will rate it low on faithfulness
    but high on relevance when query is the only thing."""
    if "2+2" in prompt_str:
        return "4"
    if "Japan" in prompt_str:
        return "Tokyo is the capital."
    if "sky" in prompt_str:
        return "blue"
    return "I don't know."


def _perfect_run_fn(prompt_str: str) -> str:
    if "2+2" in prompt_str:
        return "4"
    if "Japan" in prompt_str:
        return "Tokyo"
    if "sky" in prompt_str:
        return "The sky appears blue due to Rayleigh scattering."
    return "I don't know."


def test_list_metrics():
    metrics = list_metrics()
    assert "faithfulness" in metrics
    assert "relevance" in metrics
    assert "completeness" in metrics


def test_score_one_returns_valid_dict():
    judge = EchoJudge(score=0.9)
    result = score_one(
        "relevance",
        "Paris is the capital.",
        judge=judge,
        query="Capital of France?",
    )
    assert result["metric"] == "relevance"
    assert 0.0 <= result["score"] <= 1.0
    assert result["score"] == 0.9


def test_score_one_unknown_metric_raises():
    judge = EchoJudge()
    try:
        score_one("bogus", "text", judge=judge)
    except ValueError:
        return
    raise AssertionError("should have raised")


def test_score_many_aggregates_correctly():
    judge = EchoJudge(score=0.8)
    scores = score_many(
        responses=["a", "b", "c"],
        metrics=["relevance", "completeness"],
        judge=judge,
        queries=["q1", "q2", "q3"],
        expecteds=["e1", "e2", "e3"],
    )
    # 3 responses x 2 metrics = 6 entries
    assert len(scores) == 6
    metrics_seen = {s["metric"] for s in scores}
    assert metrics_seen == {"relevance", "completeness"}


def test_score_one_with_callable_judge():
    def fake_judge(*, model, messages, temperature=0.0):
        return '{"score": 0.42, "reason": "custom"}'

    result = score_one("relevance", "x", judge=CallableJudge(fake_judge), query="y")
    assert result["score"] == 0.42
    assert result["reason"] == "custom"


def test_score_one_handles_non_json_gracefully():
    def junk_judge(*, model, messages, temperature=0.0):
        return "this is not json at all, sorry"

    result = score_one("relevance", "x", judge=CallableJudge(junk_judge), query="y")
    # Falls back to default score 0.5
    assert result["score"] == 0.5
    assert "non-JSON" in result["reason"]


def test_aggregate_basic():
    scores = [
        {"metric": "relevance", "score": 0.8},
        {"metric": "relevance", "score": 0.6},
        {"metric": "completeness", "score": 0.9},
    ]
    summary = aggregate(scores)
    assert summary["relevance"]["mean"] == 0.7
    assert summary["relevance"]["count"] == 2
    assert summary["completeness"]["mean"] == 0.9


def test_render_table_no_crash():
    summary = aggregate([{"metric": "relevance", "score": 0.75}])
    text = render_table(summary)
    assert "relevance" in text
    assert "Mean" in text


def test_a_b_compare():
    baseline = [{"metric": "relevance", "score": 0.5}]
    optimized = [{"metric": "relevance", "score": 0.7}]
    deltas = a_b_compare(baseline, optimized)
    assert deltas["relevance"]["delta"] == 0.2


def test_evaluate_runs_pipeline():
    p = Prompt(
        system="You are helpful.",
        documents="context here",
        query="",
        model="gpt-4o-mini",
    )
    judge = EchoJudge(score=0.8)
    report = evaluate(
        p,
        run_fn=_stub_run_fn,
        dataset=SAMPLE_DATASET,
        metrics=["relevance", "completeness"],
        judge=judge,
    )
    assert report["dataset_size"] == 3
    assert "relevance" in report["aggregate"]
    assert report["aggregate"]["relevance"]["mean"] == 0.8


def test_evaluate_ab_returns_full_report():
    baseline = Prompt(
        query="",
        documents="ctx",
        system="sys",
        model="gpt-4o-mini",
    )
    optimized = Prompt(
        system="sys",
        documents="ctx",
        query="",
        model="gpt-4o-mini",
    )
    judge = EchoJudge(score=0.9)
    report = evaluate_ab(
        baseline,
        optimized,
        run_fn=_perfect_run_fn,
        dataset=SAMPLE_DATASET,
        metrics=["relevance"],
        judge=judge,
    )
    assert "structural" in report
    assert "quality" in report
    assert "relevance" in report["quality"]
    assert report["quality"]["relevance"]["delta"] == 0.0  # both got 0.9


def test_load_dataset_jsonl(tmp_path):
    from contextops.dataset import to_jsonl
    p = tmp_path / "data.jsonl"
    to_jsonl(SAMPLE_DATASET, p)
    loaded = load_dataset(p)
    assert len(loaded) == 3
    assert loaded[0].query == "What is 2+2?"


def test_load_dataset_json_list():
    import json
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump([{"query": "q", "expected": "e"}], f)
        path = f.name
    loaded = load_dataset(path)
    assert len(loaded) == 1
    assert loaded[0].query == "q"


def test_load_dataset_csv(tmp_path):
    import csv
    p = tmp_path / "data.csv"
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "expected", "context"])
        writer.writeheader()
        writer.writerow({"query": "q", "expected": "e", "context": "c"})
    loaded = load_dataset(p)
    assert len(loaded) == 1
    assert loaded[0].context == "c"


def test_compare_v01_still_works():
    """v0.1 compare() must not break in v0.2."""
    p = Prompt(system="hi", query="q", model="gpt-4o")
    report = compare(p)
    assert "delta" in report
    assert "baseline" in report