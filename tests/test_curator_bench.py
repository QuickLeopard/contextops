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
from contextops_bench.clients import CompletionResponse
from contextops_bench.curator_bench import (
    CurationBenchItem,
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
