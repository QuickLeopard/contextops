"""Property-based fuzz tests for contextops/curator.py's pure functions.

Hypothesis generates adversarial inputs (extreme floats, unicode, empty
strings, huge weight ratios) to check invariants that the hand-written
examples in tests/test_curator.py don't systematically cover. Everything
here is pure/offline — no network, no LLM.
"""

from __future__ import annotations

import math

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from contextops.curator import (
    CuratorConfig,
    DocumentChunk,
    context_precision,
    curate,
    score_chunk,
)

# Keep generated collections/text small — these are pure functions with
# O(n^2) dedup, and hypothesis runs each test hundreds of times.
_SIMILARITY = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_SHORT_TEXT = st.text(min_size=0, max_size=40)
_POSITIVE_WEIGHT = st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False)

_settings = settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _chunk_strategy(similarity=_SIMILARITY, text=_SHORT_TEXT):
    return st.builds(DocumentChunk, text=text, similarity=similarity)


@_settings
@given(
    similarity=_SIMILARITY,
    trust=_SIMILARITY,
    age_days=st.floats(min_value=0.0, max_value=100_000, allow_nan=False, allow_infinity=False),
    half_life_days=st.floats(min_value=0.01, max_value=10_000, allow_nan=False, allow_infinity=False),
    w_sim=_POSITIVE_WEIGHT,
    w_rec=_POSITIVE_WEIGHT,
    w_trust=_POSITIVE_WEIGHT,
)
def test_score_chunk_always_in_unit_interval(similarity, trust, age_days, half_life_days,
                                              w_sim, w_rec, w_trust):
    config = CuratorConfig(weights={"similarity": w_sim, "recency": w_rec, "trust": w_trust},
                            half_life_days=half_life_days)
    now = 1_700_000_000.0
    chunk = DocumentChunk(text="x", similarity=similarity, updated_at=now - age_days * 86400,
                           trust_score=trust)
    score = score_chunk(chunk, config, now=now)
    assert -1e-9 <= score <= 1.0 + 1e-9


@_settings
@given(
    similarity=_SIMILARITY,
    w_sim=_POSITIVE_WEIGHT,
    w_rec=_POSITIVE_WEIGHT,
    w_trust=_POSITIVE_WEIGHT,
)
def test_score_chunk_deterministic(similarity, w_sim, w_rec, w_trust):
    config = CuratorConfig(weights={"similarity": w_sim, "recency": w_rec, "trust": w_trust})
    chunk = DocumentChunk(text="x", similarity=similarity)
    now = 1_700_000_000.0
    a = score_chunk(chunk, config, now=now)
    b = score_chunk(chunk, config, now=now)
    assert a == b


@_settings
@given(
    similarity=_SIMILARITY,
    trust=_SIMILARITY,
)
def test_score_chunk_missing_recency_and_trust_treated_neutrally(similarity, trust):
    """Unknown recency/trust default to 1.0 (neutral), never penalized to 0."""
    config = CuratorConfig(weights={"similarity": 0.0, "recency": 1.0, "trust": 0.0})
    chunk_no_recency = DocumentChunk(text="x", similarity=similarity)
    assert score_chunk(chunk_no_recency, config) == 1.0

    config2 = CuratorConfig(weights={"similarity": 0.0, "recency": 0.0, "trust": 1.0})
    chunk_no_trust = DocumentChunk(text="x", similarity=similarity)
    assert score_chunk(chunk_no_trust, config2) == 1.0


@_settings
@given(weights=st.fixed_dictionaries({
    "similarity": st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
    "recency": st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
    "trust": st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
}))
def test_score_chunk_zero_or_negative_weight_sum_raises_cleanly(weights):
    """Weights summing to <= 0 must raise ValueError, never silently produce
    NaN/garbage scores."""
    assume(sum(weights.values()) <= 0)
    config = CuratorConfig(weights=weights)
    chunk = DocumentChunk(text="x", similarity=0.5)
    try:
        score_chunk(chunk, config)
        raised = False
    except ValueError:
        raised = True
    assert raised


@_settings
@given(chunks=st.lists(_chunk_strategy(), min_size=0, max_size=15),
       threshold=_SIMILARITY,
       dedup_threshold=_SIMILARITY)
def test_curate_partition_invariant(chunks, threshold, dedup_threshold):
    """Every input chunk ends up in exactly one of kept/dropped."""
    config = CuratorConfig(weights={"similarity": 1.0, "recency": 0.0, "trust": 0.0},
                            threshold=threshold, dedup_threshold=dedup_threshold)
    result = curate(chunks, config)
    assert len(result.kept) + len(result.dropped) == len(chunks)


@_settings
@given(chunks=st.lists(_chunk_strategy(), min_size=0, max_size=15),
       threshold=_SIMILARITY)
def test_curate_kept_scores_meet_threshold(chunks, threshold):
    config = CuratorConfig(weights={"similarity": 1.0, "recency": 0.0, "trust": 0.0},
                            threshold=threshold)
    result = curate(chunks, config)
    assert all(s >= threshold - 1e-9 for s in result.scores)
    assert len(result.scores) == len(result.kept)


@_settings
@given(chunks=st.lists(_chunk_strategy(), min_size=0, max_size=15),
       max_chunks=st.integers(min_value=0, max_value=20))
def test_curate_max_chunks_never_exceeded(chunks, max_chunks):
    config = CuratorConfig(weights={"similarity": 1.0, "recency": 0.0, "trust": 0.0},
                            threshold=0.0, max_chunks=max_chunks)
    result = curate(chunks, config)
    assert len(result.kept) <= max_chunks


@_settings
@given(chunks=st.lists(_chunk_strategy(), min_size=0, max_size=15),
       threshold=_SIMILARITY, dedup_threshold=_SIMILARITY)
def test_curate_deterministic(chunks, threshold, dedup_threshold):
    config = CuratorConfig(weights={"similarity": 1.0, "recency": 0.0, "trust": 0.0},
                            threshold=threshold, dedup_threshold=dedup_threshold)
    now = 1_700_000_000.0
    a = curate(chunks, config, now=now)
    b = curate(chunks, config, now=now)
    assert [c.text for c in a.kept] == [c.text for c in b.kept]
    assert a.scores == b.scores
    assert len(a.dropped) == len(b.dropped)


@_settings
@given(chunks=st.lists(_chunk_strategy(text=st.text(min_size=0, max_size=40)),
                        min_size=0, max_size=15),
       threshold=_SIMILARITY, dedup_threshold=_SIMILARITY, max_chunks=st.one_of(
           st.none(), st.integers(min_value=0, max_value=10)))
def test_curate_never_raises_on_arbitrary_unicode_and_empty_text(
        chunks, threshold, dedup_threshold, max_chunks):
    """No crash on empty strings, unicode/emoji, or any similarity/threshold combo."""
    config = CuratorConfig(weights={"similarity": 1.0, "recency": 0.0, "trust": 0.0},
                            threshold=threshold, dedup_threshold=dedup_threshold,
                            max_chunks=max_chunks)
    curate(chunks, config)  # must not raise


def test_curate_never_raises_on_very_long_text():
    """Explicit (non-hypothesis) case: a single very long chunk (10k+ words)."""
    long_text = " ".join(f"word{i}" for i in range(12_000))
    chunks = [
        DocumentChunk(text=long_text, similarity=0.9),
        DocumentChunk(text="short chunk", similarity=0.5),
    ]
    config = CuratorConfig(weights={"similarity": 1.0, "recency": 0.0, "trust": 0.0},
                            threshold=0.4)
    result = curate(chunks, config)
    assert len(result.kept) == 2


def test_curate_all_identical_chunks_dedup_to_one():
    """All-identical text should collapse to exactly one kept chunk under any
    dedup_threshold <= 1.0 (Jaccard of identical texts is always 1.0)."""
    chunks = [DocumentChunk(text="the same exact content", similarity=0.9) for _ in range(5)]
    config = CuratorConfig(weights={"similarity": 1.0, "recency": 0.0, "trust": 0.0},
                            threshold=0.0, dedup_threshold=0.99)
    result = curate(chunks, config)
    assert len(result.kept) == 1
    assert len(result.dropped) == 4


def test_curate_word_reorder_paraphrase_is_deduped():
    """Jaccard dedup is bag-of-words: reordering words must still dedup
    (documents a known limitation, not testing a bug: synonym swaps would
    NOT be caught, since curate() explicitly does token-overlap, not
    semantic similarity)."""
    chunks = [
        DocumentChunk(text="the quick brown fox jumps over the lazy dog", similarity=0.9),
        DocumentChunk(text="dog lazy the over jumps fox brown quick the", similarity=0.85),
    ]
    config = CuratorConfig(weights={"similarity": 1.0, "recency": 0.0, "trust": 0.0},
                            threshold=0.0, dedup_threshold=0.99)
    result = curate(chunks, config)
    assert len(result.kept) == 1


def test_curate_synonym_swap_is_not_deduped():
    """Documents the flip side of the above: token-overlap dedup can't
    catch semantically-identical-but-lexically-different duplicates."""
    chunks = [
        DocumentChunk(text="the automobile is red", similarity=0.9),
        DocumentChunk(text="the car is crimson", similarity=0.85),
    ]
    config = CuratorConfig(weights={"similarity": 1.0, "recency": 0.0, "trust": 0.0},
                            threshold=0.0, dedup_threshold=0.5)
    result = curate(chunks, config)
    assert len(result.kept) == 2


@_settings
@given(chunks=st.lists(_chunk_strategy(), min_size=0, max_size=10),
       response=_SHORT_TEXT)
def test_context_precision_bounds(chunks, response):
    result = context_precision(chunks, response)
    assert 0.0 <= result["precision"] <= 1.0
    assert len(result["per_chunk_used"]) == len(chunks)


def test_context_precision_empty_kept_is_vacuously_perfect_fuzz():
    for response in ["", "some text", "🎉 emoji response"]:
        result = context_precision([], response)
        assert result["precision"] == 1.0


@_settings
@given(similarity=_SIMILARITY)
def test_curate_boundary_score_exactly_at_threshold_is_kept(similarity):
    """score == threshold is kept (strict '<' drop condition), for any
    similarity used as both the chunk's score and the threshold."""
    threshold = similarity
    config = CuratorConfig(weights={"similarity": 1.0, "recency": 0.0, "trust": 0.0},
                            threshold=threshold)
    chunk = DocumentChunk(text="boundary chunk", similarity=similarity)
    result = curate([chunk], config)
    assert len(result.kept) == 1


def test_score_chunk_handles_unicode_and_emoji_text():
    """Text content itself never affects score_chunk (only similarity/trust/
    recency do) — sanity check that unicode text doesn't crash scoring."""
    config = CuratorConfig(weights={"similarity": 1.0, "recency": 0.0, "trust": 0.0})
    chunk = DocumentChunk(text="日本語のテキスト 🎌 emoji café naïve", similarity=0.7)
    assert math.isclose(score_chunk(chunk, config), 0.7)
