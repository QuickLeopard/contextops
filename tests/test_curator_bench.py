"""Tests for contextops_bench.curator_bench — the RAG Curator bench module.

All tests run fully offline: `generate_curation_dataset()` is deterministic
and pure, and `run_curator_bench()` is exercised with a small local stub
client (NOT `contextops_bench.clients.EchoClient`, which always returns a
fixed `"echo"` response regardless of input — that would make
`context_precision` meaningless/identical for both arms). The stub below
simulates a competent LLM: it picks the chunk with the most word overlap
with the query and returns it verbatim, ignoring noise chunks — the same
behavior a real LLM should approximate.
"""

from __future__ import annotations

from contextops.clients import EchoJudge
from contextops.judge import score_many
from contextops_bench.clients import CompletionResponse
from contextops_bench.curator_bench import (
    CurationBenchItem,
    _ClientAsJudge,
    evaluate_curator_quality_gate,
    generate_curation_dataset,
    run_curator_bench,
    render_curator_summary,
)


class _FakeAnswerClient:
    """Deterministic stub client: answers with whichever "chunk" (paragraph
    in the rendered prompt, chunks are joined with blank lines) shares the
    most words with the query — the last paragraph. Ignores irrelevant
    (noise) chunks, same as a competent real LLM would.
    """

    PROVIDER = "fake"

    def complete(self, *, model: str, messages: list[dict],
                 temperature: float = 0.0, max_tokens: int = 64) -> CompletionResponse:
        prompt_str = messages[-1]["content"] if messages else ""
        parts = prompt_str.split("\n\n")
        if len(parts) < 2:
            text = parts[0] if parts else ""
        else:
            query = parts[-1]
            query_words = set(query.lower().split())
            best, best_overlap = "", -1
            for part in parts[:-1]:
                overlap = len(query_words & set(part.lower().split()))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best = part
            text = best or "I don't know."
        return CompletionResponse(
            text=text,
            prompt_tokens=max(1, len(prompt_str) // 4),
            completion_tokens=max(1, len(text) // 4),
            cached_tokens=0,
            cost_usd=0.0,
            model=model,
            raw={},
        )


def test_generate_curation_dataset_item_count():
    items = generate_curation_dataset(20, seed=1)
    assert len(items) == 20
    assert all(isinstance(it, CurationBenchItem) for it in items)
    assert all(it.chunks for it in items)


def test_generate_curation_dataset_deterministic():
    a = generate_curation_dataset(10, seed=7)
    b = generate_curation_dataset(10, seed=7)
    assert [it.query for it in a] == [it.query for it in b]
    assert [[c.text for c in it.chunks] for it in a] == [[c.text for c in it.chunks] for it in b]


def test_generate_curation_dataset_different_seeds_differ():
    a = generate_curation_dataset(10, seed=1)
    b = generate_curation_dataset(10, seed=2)
    assert [it.query for it in a] != [it.query for it in b]


def test_generate_curation_dataset_noise_ratio_respected():
    items = generate_curation_dataset(50, noise_ratio=0.8, chunks_per_item=10, dup_rate=0.0, seed=3)
    for it in items:
        noise_count = sum(1 for c in it.chunks if c.similarity < 0.6)
        # noise_ratio=0.8 of 10 chunks = 8 noise chunks (before any dup injection,
        # which is disabled here via dup_rate=0.0).
        assert noise_count == 8


def test_generate_curation_dataset_injects_duplicates():
    items = generate_curation_dataset(200, dup_rate=1.0, chunks_per_item=4, seed=5)
    # With dup_rate=1.0, every item gets an extra near-duplicate chunk appended.
    assert all(len(it.chunks) == 5 for it in items)


def test_generate_curation_dataset_zero_noise():
    items = generate_curation_dataset(10, noise_ratio=0.0, dup_rate=0.0, chunks_per_item=4, seed=9)
    for it in items:
        assert all(c.similarity >= 0.6 for c in it.chunks)


def test_run_curator_bench_summary_shape():
    items = generate_curation_dataset(8, seed=11)
    client = _FakeAnswerClient()
    judge = EchoJudge(score=0.85)
    summary = run_curator_bench(
        items, client=client, model="fake-model", judge=judge,
        metrics=["relevance"], judge_model="echo",
    )
    assert summary["provider"] == "fake"
    assert summary["model"] == "fake-model"
    assert summary["n"] == 8
    assert "raw" in summary and "curated" in summary
    assert summary["raw"]["n"] == 8
    assert summary["curated"]["n"] == 8
    assert "delta" in summary
    assert "quality" in summary
    assert "relevance" in summary["quality"]
    assert "curation" in summary
    assert 0.0 <= summary["curation"]["mean_drop_rate"] <= 1.0


def test_run_curator_bench_curated_uses_fewer_tokens():
    """The whole point of curation: fewer tokens sent for the same query."""
    items = generate_curation_dataset(10, noise_ratio=0.7, chunks_per_item=6, seed=13)
    client = _FakeAnswerClient()
    judge = EchoJudge()
    summary = run_curator_bench(
        items, client=client, model="fake-model", judge=judge,
        metrics=["relevance"], judge_model="echo",
    )
    assert summary["curated"]["prompt_tokens_mean"] < summary["raw"]["prompt_tokens_mean"]
    assert summary["delta"]["prompt_tokens_delta_mean"] < 0


def test_run_curator_bench_curated_context_precision_higher():
    """Curated context precision should be >= raw's: raw sends lots of noise
    the (simulated) LLM never uses, curated sends mostly-relevant chunks that
    largely ARE reflected in the answer.
    """
    items = generate_curation_dataset(15, noise_ratio=0.75, chunks_per_item=8, seed=17)
    client = _FakeAnswerClient()
    judge = EchoJudge()
    summary = run_curator_bench(
        items, client=client, model="fake-model", judge=judge,
        metrics=["relevance"], judge_model="echo",
    )
    assert summary["curated"]["context_precision_mean"] >= summary["raw"]["context_precision_mean"]
    assert summary["delta"]["context_precision_delta"] >= 0.0


def test_render_curator_summary_no_crash():
    items = generate_curation_dataset(5, seed=21)
    client = _FakeAnswerClient()
    judge = EchoJudge()
    summary = run_curator_bench(
        items, client=client, model="fake-model", judge=judge,
        metrics=["relevance", "completeness"], judge_model="echo",
    )
    text = render_curator_summary(summary, label="test_run")
    assert "test_run" in text
    assert "RAW" in text
    assert "CURATED" in text
    assert "DELTA" in text
    assert "QUALITY" in text


def test_render_curator_summary_empty_summary_no_crash():
    """Degenerate case: empty items list -> empty raw/curated stats."""
    text = render_curator_summary({"n": 0, "raw": {}, "curated": {}}, label="empty")
    assert "empty" in text


def test_run_curator_bench_max_tokens_plumbed_to_client():
    """`max_tokens` must reach the LLM-under-test's `client.complete()` calls
    (default was too short — 64 — starving the judge/context_precision of
    signal), but NOT necessarily the judge's own calls (handled separately).
    """
    seen_max_tokens: list[int] = []

    class _RecordingClient(_FakeAnswerClient):
        def complete(self, *, model, messages, temperature=0.0, max_tokens=64):
            seen_max_tokens.append(max_tokens)
            return super().complete(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
            )

    items = generate_curation_dataset(4, seed=31)
    run_curator_bench(
        items, client=_RecordingClient(), model="fake-model", judge=EchoJudge(),
        metrics=["relevance"], judge_model="echo", max_tokens=321,
    )
    assert seen_max_tokens
    assert all(mt == 321 for mt in seen_max_tokens)


class _EchoingJudgeClient:
    """Stub bench client for `_ClientAsJudge` tests: always answers with a
    fixed JSON score payload, distinguishable from any real answer text.
    """

    PROVIDER = "fake-judge"

    def complete(self, *, model: str, messages: list[dict],
                 temperature: float = 0.0, max_tokens: int = 64) -> CompletionResponse:
        return CompletionResponse(
            text='{"score": 0.9, "reason": "stub judge verdict"}',
            prompt_tokens=10, completion_tokens=5, cached_tokens=0,
            cost_usd=0.0, model=model, raw={},
        )


def test_client_as_judge_returns_text_from_response():
    judge = _ClientAsJudge(_EchoingJudgeClient())
    result = judge.complete(model="fake-model", messages=[{"role": "user", "content": "hi"}])
    assert result == '{"score": 0.9, "reason": "stub judge verdict"}'


def test_client_as_judge_satisfies_judge_client_protocol_for_score_many():
    """`_ClientAsJudge` must be directly usable wherever a `JudgeClient` is
    expected — e.g. `score_many()`, which only duck-types `.complete()`.
    """
    judge = _ClientAsJudge(_EchoingJudgeClient())
    results = score_many(
        ["some response"], metrics=["relevance"], judge=judge, model="fake-model",
    )
    assert len(results) == 1
    assert results[0]["score"] == 0.9


def test_client_as_judge_passes_through_max_tokens_not_model_under_test_value():
    """The judge's own `max_tokens` is independent of the LLM-under-test's
    (should stay small — the judge only needs to emit a short JSON verdict).
    """
    seen: list[int] = []

    class _RecordingJudgeClient(_EchoingJudgeClient):
        def complete(self, *, model, messages, temperature=0.0, max_tokens=64):
            seen.append(max_tokens)
            return super().complete(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)

    judge = _ClientAsJudge(_RecordingJudgeClient(), max_tokens=150)
    judge.complete(model="fake-model", messages=[{"role": "user", "content": "hi"}])
    assert seen == [150]


def _scores(metric: str, values: list[float]) -> list[dict]:
    return [{"index": i, "metric": metric, "score": v, "reason": "", "raw": ""} for i, v in enumerate(values)]


def test_evaluate_curator_quality_gate_significant_improvement():
    """A clear, consistent, non-zero delta across n>=min_n items should be
    flagged as verified with the metric in significant_metrics.
    """
    raw = _scores("relevance", [0.5] * 25)
    curated = _scores("relevance", [0.9] * 25)
    gate = evaluate_curator_quality_gate(raw, curated, min_n=20)
    assert gate["n"] == 25
    assert gate["low_n"] is False
    assert gate["verified"] is True
    assert "relevance" in gate["significant_metrics"]
    assert gate["per_metric"]["relevance"]["significant"] is True
    assert gate["per_metric"]["relevance"]["effect_size_pct"] > 0


def test_evaluate_curator_quality_gate_identical_scores_not_significant():
    """Identical scores both arms -> zero-width CI at zero -> not significant."""
    raw = _scores("relevance", [0.7] * 25)
    curated = _scores("relevance", [0.7] * 25)
    gate = evaluate_curator_quality_gate(raw, curated, min_n=20)
    assert gate["per_metric"]["relevance"]["significant"] is False
    assert gate["verified"] is False
    assert any("CI" in r for r in gate["reasons"])


def test_evaluate_curator_quality_gate_low_n():
    raw = _scores("relevance", [0.5] * 5)
    curated = _scores("relevance", [0.9] * 5)
    gate = evaluate_curator_quality_gate(raw, curated, min_n=20)
    assert gate["low_n"] is True
    assert gate["verified"] is False
    assert any("min_n" in r for r in gate["reasons"])


def test_evaluate_curator_quality_gate_multi_metric_pairing_by_index():
    """Metrics are paired by (metric, index), not just position — mixing two
    metrics in the input lists must not cross-contaminate their diffs.
    """
    raw = _scores("relevance", [0.5] * 20) + _scores("completeness", [0.8] * 20)
    curated = _scores("relevance", [0.9] * 20) + _scores("completeness", [0.8] * 20)
    gate = evaluate_curator_quality_gate(raw, curated, min_n=20)
    assert gate["per_metric"]["relevance"]["significant"] is True
    assert gate["per_metric"]["completeness"]["significant"] is False


def test_run_curator_bench_summary_has_quality_gate_and_judge_label():
    items = generate_curation_dataset(8, seed=41)
    summary = run_curator_bench(
        items, client=_FakeAnswerClient(), model="fake-model", judge=EchoJudge(),
        metrics=["relevance"], judge_model="echo", judge_label="echo",
    )
    assert summary["judge"] == "echo"
    assert "quality_gate" in summary
    assert "n" in summary["quality_gate"]
    assert "relevance" in summary["quality"]
    assert "ci_low" in summary["quality"]["relevance"]


def test_render_curator_summary_includes_judge_and_quality_gate_sections():
    items = generate_curation_dataset(5, seed=51)
    summary = run_curator_bench(
        items, client=_FakeAnswerClient(), model="fake-model", judge=EchoJudge(),
        metrics=["relevance"], judge_model="echo", judge_label="self",
    )
    text = render_curator_summary(summary, label="test_run")
    assert "judge=self" in text
    assert "self-judged" in text
    assert "QUALITY GATE" in text
