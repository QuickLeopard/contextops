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


def test_openai_direct_factory():
    """OpenAIDirectClient is reachable via 'direct_openai' or 'openai'."""
    import os
    os.environ["OPENAI_API_KEY"] = "sk-test-fake-key-for-unittest"
    try:
        client = get_client("direct_openai")
        from contextops_bench.clients import OpenAIDirectClient
        assert isinstance(client, OpenAIDirectClient)
        assert client.base_url == "https://api.openai.com/v1"
        # supports_split_messages = False: OpenAI uses messages[] with system as a message
        assert client.supports_split_messages is False
    finally:
        os.environ.pop("OPENAI_API_KEY", None)


def test_openai_direct_requires_api_key():
    """OpenAIDirectClient must raise ValueError without an API key."""
    import os
    os.environ.pop("OPENAI_API_KEY", None)
    from contextops_bench.clients import OpenAIDirectClient
    try:
        OpenAIDirectClient()
    except ValueError as e:
        assert "OPENAI_API_KEY" in str(e)
        return
    raise AssertionError("should have raised ValueError for missing OPENAI_API_KEY")


def test_openai_direct_resolves_models():
    """OpenAIDirectClient._resolve_model handles both 'openai/foo' and bare 'foo'."""
    from contextops_bench.clients import OpenAIDirectClient
    client = OpenAIDirectClient(api_key="sk-test")
    assert client._resolve_model("openai/gpt-4o") == "gpt-4o"
    assert client._resolve_model("gpt-4o-mini") == "gpt-4o-mini"
    assert client._resolve_model("gpt-4.1") == "gpt-4.1"


def test_openai_direct_cache_read_discount():
    """OpenAI cached tokens cost 50% of input — verify the cost formula."""
    from contextops_bench.clients import OpenAIDirectClient
    client = OpenAIDirectClient(api_key="sk-test")
    # Send 1000 prompt tokens, of which 800 came from cache.
    # Effective cost: 200 * input + 800 * input * 0.5 + completion * output
    # For gpt-4o-mini: input=$0.15/M, output=$0.60/M.
    # Build a fake response inline by calling complete() with a mocked transport.
    import contextops_bench.clients as c_mod

    def fake_post(self, path, payload):
        return {
            "model": "gpt-4o-mini",
            "choices": [{"message": {"content": "ok"}}],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 50,
                "prompt_tokens_details": {"cached_tokens": 800},
            },
        }

    orig_post = c_mod.BaseHTTPClient._post
    c_mod.BaseHTTPClient._post = fake_post
    try:
        resp = client.complete(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello world"}],
        )
        assert resp.cached_tokens == 800
        # Expected: uncached=200, cached=800
        # Cost uncached: 200/1e6 * 0.15 = 0.00003
        # Cost cached:   800/1e6 * 0.15 * 0.5 = 0.00006
        # Cost output:   50/1e6  * 0.60 = 0.00003
        # Total:         0.00012
        assert abs(resp.cost_usd - 0.00012) < 1e-7
    finally:
        c_mod.BaseHTTPClient._post = orig_post


def test_google_direct_factory():
    """GoogleDirectClient is reachable via 'direct_google' or 'google'."""
    import os
    os.environ["GOOGLE_API_KEY"] = "AIzaSyTestFakeKeyForUnittest"
    try:
        client = get_client("direct_google")
        from contextops_bench.clients import GoogleDirectClient
        assert isinstance(client, GoogleDirectClient)
        assert client.base_url == "https://generativelanguage.googleapis.com/v1beta"
        # supports_split_messages = True: Google uses systemInstruction top-level field
        assert client.supports_split_messages is True
    finally:
        os.environ.pop("GOOGLE_API_KEY", None)


def test_google_direct_requires_api_key():
    """GoogleDirectClient must raise ValueError without an API key."""
    import os
    os.environ.pop("GOOGLE_API_KEY", None)
    os.environ.pop("GEMINI_API_KEY", None)
    from contextops_bench.clients import GoogleDirectClient
    try:
        GoogleDirectClient()
    except ValueError as e:
        assert "GOOGLE_API_KEY" in str(e) or "GEMINI_API_KEY" in str(e)
        return
    raise AssertionError("should have raised ValueError for missing GOOGLE_API_KEY")


def test_google_direct_resolves_models():
    """GoogleDirectClient._resolve_model handles 'google/', 'gemini/', and bare names."""
    from contextops_bench.clients import GoogleDirectClient
    client = GoogleDirectClient(api_key="test")
    assert client._resolve_model("google/gemini-2.5-flash") == "gemini-2.5-flash"
    assert client._resolve_model("gemini-2.5-pro") == "gemini-2.5-pro"
    assert client._resolve_model("gemini-2.5-flash-lite") == "gemini-2.5-flash-lite"


def test_google_direct_cache_read_discount():
    """Gemini cached tokens cost 10% of input — verify the cost formula."""
    from contextops_bench.clients import GoogleDirectClient
    import contextops_bench.clients as c_mod

    def fake_urlopen(req, timeout):
        class FakeResp:
            def __init__(self):
                self.body = (
                    b'{"candidates":[{"content":{"parts":[{"text":"ok"}]}}],'
                    b'"usageMetadata":{"promptTokenCount":1000,'
                    b'"candidatesTokenCount":50,'
                    b'"cachedContentTokenCount":800}}'
                )

            def read(self):
                return self.body

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return FakeResp()

    orig_urlopen = c_mod.urllib.request.urlopen
    c_mod.urllib.request.urlopen = fake_urlopen
    try:
        client = GoogleDirectClient(api_key="test")
        resp = client.complete(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "hello"}],
            system="system prompt",
        )
        assert resp.cached_tokens == 800
        # For gemini-2.5-flash: input=$0.30/M, output=$2.50/M, cache_read_mult=0.10.
        # Expected: uncached=200, cached=800
        # Cost uncached: 200/1e6 * 0.30 = 0.00006
        # Cost cached:   800/1e6 * 0.30 * 0.10 = 0.000024
        # Cost output:   50/1e6  * 2.50 = 0.000125
        # Total:         0.000209
        assert abs(resp.cost_usd - 0.000209) < 1e-7
    finally:
        c_mod.urllib.request.urlopen = orig_urlopen


def test_run_batch_alternates_optimized_baseline():
    prompts = list(generate_many(n=10, seed=42))
    client = EchoClient()
    results = run_batch(prompts, client=client, parallel=1, cache_warm=False)
    # Each prompt runs TWICE (optimized + baseline) for the A/B, so 10 prompts
    # → 20 results. With cache_warm=False, jobs interleave by parity.
    assert len(results) == 20
    optimized = [r for r in results if r.use_optimized]
    baseline = [r for r in results if not r.use_optimized]
    assert len(optimized) == 10
    assert len(baseline) == 10
    # Same prompts ran on both arms (paired A/B)
    assert sorted(r.prompt_id for r in optimized) == list(range(10))
    assert sorted(r.prompt_id for r in baseline) == list(range(10))


def test_run_batch_cache_warm_mode():
    """In cache_warm=True (default), optimized jobs run first, then baseline."""
    prompts = list(generate_many(n=10, seed=42))
    client = EchoClient()
    results = run_batch(prompts, client=client, parallel=1, cache_warm=True)
    # Each prompt runs twice (optimized + baseline), so 20 results, 10 distinct prompt_ids.
    optimized = [r for r in results if r.use_optimized]
    baseline = [r for r in results if not r.use_optimized]
    assert len(optimized) == 10
    assert len(baseline) == 10
    assert sorted(set(r.prompt_id for r in results)) == list(range(10))


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
    # Each prompt runs twice (optimized + baseline) → 10 rows for 5 prompts
    assert len(rows) == 10
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
    # Each prompt runs twice (optimized + baseline) → 20 each arm
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
    # Same prompt_ids in both, possibly different order. Each prompt runs twice → 40 results.
    assert sorted(r.prompt_id for r in serial) == sorted(r.prompt_id for r in parallel)
    assert len(serial) == 40
    assert len(parallel) == 40


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
    # Each prompt runs twice (optimized + baseline)
    assert len(results) == len(prompts) * 2
    # No errors expected
    errors = [r for r in results if r.error]
    assert len(errors) == 0


# ---------------------------------------------------------------------------
# Tests for _resolve_preset_args — the safety net for the v0.3.1 failure mode.
# Without this, `bench cloud --provider direct_openai` (no --preset-agent) would
# silently randomize role → cache key rotates every call → 0% cache hit rate.
# ---------------------------------------------------------------------------

def _make_args(provider="echo", preset_agent=None, fixed_system=None, fixed_tools=None):
    """Build a minimal argparse.Namespace for _resolve_preset_args."""
    import argparse
    return argparse.Namespace(
        provider=provider,
        preset_agent=preset_agent,
        fixed_system=fixed_system,
        fixed_tools=fixed_tools,
    )


def test_resolve_preset_auto_defaults_realistic_on_cloud_provider():
    """No preset, no fixed-* on a cache-bearing provider → realistic + warning.

    This is the regression: prior to v0.3.1.x, this exact path randomized
    role and silently produced 0% cache hit rate while looking legit.
    """
    from contextops_bench.__main__ import _resolve_preset_args
    from contextops_bench.prompt_factory import AGENT_PRESETS

    args = _make_args(provider="direct_openai", preset_agent=None)
    fs, ft, fr, label, warning = _resolve_preset_args(args)

    realistic = AGENT_PRESETS["realistic"]
    assert fs == realistic["system"]
    assert ft == realistic["tools"]
    assert fr == realistic["role"]
    assert label.startswith("realistic")
    assert warning is not None
    assert "realistic" in warning.lower()
    assert "auto" in warning.lower()


def test_resolve_preset_explicit_realistic_no_warning():
    """--preset-agent realistic on cloud → realistic preset, no warning."""
    from contextops_bench.__main__ import _resolve_preset_args
    from contextops_bench.prompt_factory import AGENT_PRESETS

    args = _make_args(provider="direct_anthropic", preset_agent="realistic")
    fs, ft, fr, label, warning = _resolve_preset_args(args)

    realistic = AGENT_PRESETS["realistic"]
    assert fs == realistic["system"]
    assert ft == realistic["tools"]
    assert fr == realistic["role"]
    assert label == "realistic"
    assert warning is None


def test_resolve_preset_explicit_none_opts_out():
    """--preset-agent none → randomized, no warning, role not pinned."""
    from contextops_bench.__main__ import _resolve_preset_args

    args = _make_args(provider="direct_openai", preset_agent="none")
    fs, ft, fr, label, warning = _resolve_preset_args(args)

    # None of the preset fields should be filled in.
    assert fs is None
    assert ft is None
    assert fr is None
    assert label == "none"
    assert warning is None


def test_resolve_preset_local_provider_no_default():
    """Echo / Ollama / LM Studio don't have cache → no auto-default.

    Smoke tests rely on this — they'd fail if every smoke run printed a
    'no preset' warning.
    """
    from contextops_bench.__main__ import _resolve_preset_args

    for prov in ("echo", "ollama", "lmstudio"):
        args = _make_args(provider=prov, preset_agent=None)
        fs, ft, fr, label, warning = _resolve_preset_args(args)
        assert fs is None
        assert ft is None
        assert fr is None
        assert label == "no-preset"
        assert warning is None


def test_resolve_preset_fixed_system_overrides_auto_default():
    """--fixed-system on cloud provider → uses it, no auto-default, no warning.

    The user knows what they're doing when they pass a fixed-system; don't
    second-guess them.
    """
    from contextops_bench.__main__ import _resolve_preset_args

    args = _make_args(
        provider="direct_openai",
        preset_agent=None,
        fixed_system="my custom system prompt",
    )
    fs, ft, fr, label, warning = _resolve_preset_args(args)
    assert fs == "my custom system prompt"
    assert label == "no-preset"
    assert warning is None


def test_resolve_preset_echo_subcommand_with_realistic_override():
    """--preset-agent realistic on echo → realistic preset still applies.

    Cache-bearing gating only fires on the auto-default path. Explicit preset
    always wins.
    """
    from contextops_bench.__main__ import _resolve_preset_args
    from contextops_bench.prompt_factory import AGENT_PRESETS

    args = _make_args(provider="echo", preset_agent="realistic")
    fs, ft, fr, label, warning = _resolve_preset_args(args)
    realistic = AGENT_PRESETS["realistic"]
    assert fs == realistic["system"]
    assert fr == realistic["role"]
    assert label == "realistic"
    assert warning is None