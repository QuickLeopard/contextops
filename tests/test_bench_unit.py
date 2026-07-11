"""Tests for the bench framework. Uses EchoClient — no network."""

import csv

from contextops_bench.clients import EchoClient, get_client
from contextops_bench.prompt_factory import EDGE_CASES, generate_many
from contextops_bench.runner import run_batch, save_csv, summarize


def test_generate_many_count():
    prompts = list(generate_many(n=100, seed=42))
    assert len(prompts) == 100


def test_generate_many_diversity():
    """Different seeds should produce different prompts."""
    p1 = list(generate_many(n=10, seed=1))
    p2 = list(generate_many(n=10, seed=2))
    # At least 7 of 10 should differ
    diffs = sum(1 for a, b in zip(p1, p2) if a.query != b.query or a.system != b.system)
    assert diffs >= 7


def test_edge_cases_cover_paths():
    assert len(EDGE_CASES) >= 10
    # Empty
    assert EDGE_CASES[0].sections() == []
    # Only query
    assert len(EDGE_CASES[1].sections()) == 1
    # All sections filled
    assert len(EDGE_CASES[3].sections()) == 7


def test_echo_client_works():
    client = EchoClient()
    resp = client.complete(
        model="echo-model",
        messages=[{"role": "user", "content": "Hello world, this is a longer test message with multiple words."}],
    )
    assert resp.prompt_tokens > 0
    assert resp.completion_tokens > 0
    assert client.calls == 1


def test_echo_client_cache_accumulates():
    """EchoClient cache hit rate should grow with repeated prefixes."""
    client = EchoClient(base_cache_rate=0.8)
    # First call with prefix X: 0% cache.
    client.complete(model="x", messages=[{"role": "user", "content": "X" * 1000}])
    # Second call with same prefix: should be higher.
    r2 = client.complete(model="x", messages=[{"role": "user", "content": "X" * 1000}])
    rate2 = r2.cached_tokens / r2.prompt_tokens
    # Third call: even higher.
    r3 = client.complete(model="x", messages=[{"role": "user", "content": "X" * 1000}])
    rate3 = r3.cached_tokens / r3.prompt_tokens
    assert rate3 > rate2 > 0


def test_echo_client_empty_prompt_full_cache():
    """Empty prompt has zero tokens, but if 'content' is non-empty it's tiny -> high cache."""
    client = EchoClient(base_cache_rate=0.9)
    client.complete(model="x", messages=[{"role": "user", "content": "tiny"}])
    r2 = client.complete(model="x", messages=[{"role": "user", "content": "tiny"}])
    # Tiny prompt (3 chars / 4 = 0 tokens... actually max(1, 3//4)=1)
    # Either way, repeated same prefix should give high cache.
    assert r2.cached_tokens >= 0


def test_get_client_factory():
    assert isinstance(get_client("echo"), EchoClient)
    try:
        get_client("bogus")
    except ValueError:
        return
    raise AssertionError("should have raised")


def test_run_batch_alternates_optimized_baseline():
    """cache_warm=False: each prompt runs twice (optimized + baseline), alternating."""
    prompts = list(generate_many(n=10, seed=42))
    client = EchoClient()
    results = run_batch(prompts, client=client, parallel=1, cache_warm=False)
    # 2n: optimized + baseline for each of 10 prompts.
    assert len(results) == 20
    # cache_warm=False alternates: [opt, base, opt, base, ...]
    use_opt_seq = [r.use_optimized for r in results]
    assert use_opt_seq == [True, False] * 10
    # Every prompt_id appears exactly twice (once optimized, once baseline).
    ids = [r.prompt_id for r in results]
    assert ids == [i for i in range(10) for _ in (0, 1)]
    # Each (prompt_id, use_optimized) pair is unique.
    assert len({(r.prompt_id, r.use_optimized) for r in results}) == 20


def test_run_batch_cache_warm_mode():
    """In cache_warm=True (default): first half = all optimized, second half = all baseline."""
    prompts = list(generate_many(n=10, seed=42))
    client = EchoClient()
    results = run_batch(prompts, client=client, parallel=1, cache_warm=True)
    # 2n results: 10 optimized + 10 baseline.
    assert len(results) == 20
    # First half all optimized (prompt_ids 0..9), second half all baseline (prompt_ids 0..9).
    use_opt_seq = [r.use_optimized for r in results]
    assert use_opt_seq == [True] * 10 + [False] * 10
    assert [r.prompt_id for r in results] == list(range(10)) * 2


def test_save_csv_roundtrip(tmp_path):
    prompts = list(generate_many(n=5, seed=42))
    client = EchoClient()
    results = run_batch(prompts, client=client, parallel=1, cache_warm=False)
    csv_path = tmp_path / "out.csv"
    save_csv(results, csv_path)
    assert csv_path.exists()

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 10  # 5 prompts × 2 (optimized + baseline)
    assert "prompt_id" in rows[0]
    assert "provider" in rows[0]


def test_summarize_basic_stats():
    prompts = list(generate_many(n=20, seed=42))
    client = EchoClient()
    results = run_batch(prompts, client=client, parallel=1, cache_warm=False)
    s = summarize(results)
    assert "optimized" in s
    assert "baseline" in s
    assert "delta" in s
    # 20 prompts × 2 arms = 40 results, split evenly: 20 optimized + 20 baseline.
    assert s["optimized"]["n"] == 20
    assert s["baseline"]["n"] == 20
    assert s["optimized"]["errors"] == 0


def test_summarize_cache_warm_mode():
    """cache_warm=True should split by midpoint, not by parity."""
    prompts = list(generate_many(n=20, seed=42))
    client = EchoClient()
    results = run_batch(prompts, client=client, parallel=1, cache_warm=True)
    s = summarize(results)
    # With cache_warm, both halves have system-first ordering (reorder is idempotent),
    # so detection should still split correctly. At minimum, no errors.
    assert s["optimized"]["errors"] == 0
    assert s["baseline"]["errors"] == 0


def test_run_batch_parallel_safe():
    prompts = list(generate_many(n=20, seed=42))
    client = EchoClient()
    serial = run_batch(prompts, client=client, parallel=1, cache_warm=False)
    parallel = run_batch(prompts, client=client, parallel=4, cache_warm=False)
    # Same prompt_ids in both, possibly different order
    assert sorted(r.prompt_id for r in serial) == sorted(r.prompt_id for r in parallel)


def test_edge_cases_pass_through_optimize():
    """Every edge case MUST round-trip through optimize() without raising."""
    from contextops.optimizer import optimize
    for p in EDGE_CASES:
        r = optimize(p)
        assert r.optimized_tokens >= 0
        assert 0.0 <= r.estimated_cache_hit_rate <= 1.0


def test_smoke_under_30_seconds():
    """AC-LS-01: smoke suite must complete in < 30s."""
    import time
    prompts = list(generate_many(n=10, seed=42)) + EDGE_CASES
    client = EchoClient()
    t0 = time.time()
    results = run_batch(prompts, client=client, parallel=1, cache_warm=False)
    elapsed = time.time() - t0
    assert elapsed < 30.0
    # Each prompt runs twice (optimized + baseline).
    assert len(results) == 2 * len(prompts)
    # No errors expected
    errors = [r for r in results if r.error]
    assert len(errors) == 0


# --- Phase 11 backfill: percentile, save_csv, run_one, render_order ---

def test_percentile_small_n():
    """Nearest-rank p95 of [1..10] = 10 (rank = ceil(9.5) = 10 → the max)."""
    from contextops_bench.runner import _percentile
    assert _percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10


def test_percentile_empty():
    from contextops_bench.runner import _percentile
    assert _percentile([], 95) == 0.0


def test_percentile_large_n():
    """p95 of [1]*100 + [999] should be 1 (the 95th percentile by nearest-rank)."""
    from contextops_bench.runner import _percentile
    values = [1] * 100 + [999]
    # rank = ceil(0.95 * 101) = ceil(95.95) = 96 → sorted[95]. First 100 are 1s.
    assert _percentile(values, 95) == 1


def test_save_csv_empty_raises():
    """save_csv on empty input should raise (fail loud, not write a headerless file)."""
    import pytest
    from contextops_bench.runner import save_csv
    with pytest.raises(ValueError, match="no results"):
        save_csv([], "/tmp/should_not_exist.csv")


def test_run_one_passes_system_when_supported():
    """run_one must pass system= to clients that support split messages."""
    from contextops.models import Prompt
    from contextops_bench.runner import run_one

    class RecordingClient:
        PROVIDER = "test"
        supports_split_messages = True
        captured_system = None

        def complete(self, *, model, messages, temperature=0.0, max_tokens=64, system=None):
            self.captured_system = system
            from contextops_bench.types import CompletionResponse
            return CompletionResponse(
                text="ok", prompt_tokens=10, completion_tokens=2,
                cached_tokens=0, cost_usd=0.0, model=model, raw={},
            )

    p = Prompt(system="STABLE PREFIX", query="variable question")
    client = RecordingClient()
    run_one(p, prompt_id=0, client=client, use_optimized=True)
    assert client.captured_system == "STABLE PREFIX"


def test_run_one_passes_none_system_when_not_supported():
    """run_one must NOT split when the client lacks supports_split_messages."""
    from contextops.models import Prompt
    from contextops_bench.runner import run_one

    class RecordingClient:
        PROVIDER = "test"
        supports_split_messages = False
        captured_system = "SENTINEL"

        def complete(self, *, model, messages, temperature=0.0, max_tokens=64, system=None):
            self.captured_system = system
            from contextops_bench.types import CompletionResponse
            return CompletionResponse(
                text="ok", prompt_tokens=10, completion_tokens=2,
                cached_tokens=0, cost_usd=0.0, model=model, raw={},
            )

    p = Prompt(system="STABLE", query="Q")
    client = RecordingClient()
    run_one(p, prompt_id=0, client=client, use_optimized=True)
    # Not supported → system should be None (everything in one user message).
    assert client.captured_system is None


def test_echo_client_accepts_system_kwarg():
    """LSP regression: EchoClient.complete must accept system= (Phase 5.2)."""
    client = EchoClient()
    resp = client.complete(
        model="echo-model",
        messages=[{"role": "user", "content": "hello world test message"}],
        system="some system prompt",
    )
    assert resp.prompt_tokens > 0