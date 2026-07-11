"""Network-gated integration test for Anthropic prompt caching.

Skipped unless `ANTHROPIC_API_KEY` is set. This is the durable artifact of what
the `scripts/diag/diag_pinned_v2.py` probe investigated: that the
AnthropicDirectClient surfaces `cache_read_input_tokens > 0` on the 2nd+ call
when the stable prefix is sent as a `system` field with `cache_control`.

Run locally with:
    export ANTHROPIC_API_KEY=sk-ant-...
    python -m pytest tests/test_anthropic_integration.py -v -s
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping live Anthropic integration test",
)

# Import after the skip guard so a missing key doesn't fail at collection time
# on machines without the bench's optional deps.
from contextops_bench.clients import AnthropicDirectClient  # noqa: E402
from contextops_bench.prompt_factory import AGENT_PRESETS  # noqa: E402

MODEL = "anthropic/claude-haiku-4.5"
N_CALLS = 5


def test_anthropic_cache_activates_on_second_call():
    """5 sequential calls with the same stable prefix → call 2+ has cache_read > 0.

    This validates the core premise of ContextOps: that reordering stable sections
    to the front and sending them as a `system` field with `cache_control` causes
    Anthropic to cache the prefix and serve subsequent calls from cache.
    """
    client = AnthropicDirectClient()
    preset = AGENT_PRESETS["realistic"]
    system_content = preset["system"] + "\n\n" + preset["tools"]

    cache_reads: list[int] = []
    for i in range(N_CALLS):
        resp = client.complete(
            model=MODEL,
            messages=[{"role": "user", "content": f"Question {i}: what is 2+2?"}],
            temperature=0.0,
            max_tokens=32,
            system=system_content,
        )
        cache_reads.append(resp.cached_tokens)

    # First call: no cache (cold). Cache_creation should be > 0 instead.
    assert cache_reads[0] == 0, f"First call should have no cache reads, got {cache_reads[0]}"

    # Second call onward: cache should be warm. At least one of calls 2..N must
    # show cache_read > 0 (we don't require all, since provider-side cache can
    # occasionally miss due to eviction/timing).
    warm_reads = [c for c in cache_reads[1:] if c > 0]
    assert len(warm_reads) >= 1, (
        f"Expected cache_read > 0 on at least one of calls 2-{N_CALLS}, "
        f"got cache_reads={cache_reads}. Cache may not be activating."
    )
