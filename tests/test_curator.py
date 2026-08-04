"""Tests for the RAG Curator (contextops/curator.py). Pure functions — no
network/LLM calls needed."""

from __future__ import annotations

import time

from contextops.curator import (
    CuratorConfig,
    DocumentChunk,
    build_documents,
    context_precision,
    curate,
    score_chunk,
)
from contextops.models import Prompt


def test_score_chunk_pure_similarity_when_no_recency_or_trust():
    """Missing recency/trust are treated neutrally (1.0), not penalized."""
    config = CuratorConfig(weights={"similarity": 1.0, "recency": 0.0, "trust": 0.0})
    chunk = DocumentChunk(text="hello", similarity=0.8)
    assert score_chunk(chunk, config) == 0.8


def test_score_chunk_combines_all_signals():
    config = CuratorConfig(weights={"similarity": 0.5, "recency": 0.3, "trust": 0.2})
    now = time.time()
    chunk = DocumentChunk(text="hello", similarity=1.0, updated_at=now, trust_score=1.0)
    # recency_score for age=0 is exp(0) = 1.0
    score = score_chunk(chunk, config, now=now)
    assert abs(score - 1.0) < 1e-9


def test_score_chunk_recency_decays_with_age():
    config = CuratorConfig(weights={"similarity": 0.0, "recency": 1.0, "trust": 0.0},
                            half_life_days=10.0)
    now = time.time()
    fresh = DocumentChunk(text="a", similarity=0.0, updated_at=now)
    old = DocumentChunk(text="b", similarity=0.0, updated_at=now - 10 * 86400)
    fresh_score = score_chunk(fresh, config, now=now)
    old_score = score_chunk(old, config, now=now)
    assert fresh_score > old_score
    assert abs(old_score - 0.5) < 0.01  # one half-life elapsed


def test_score_chunk_weights_renormalize():
    """Weights that don't sum to 1 are re-normalized, not left raw."""
    config = CuratorConfig(weights={"similarity": 2.0, "recency": 0.0, "trust": 0.0})
    chunk = DocumentChunk(text="x", similarity=0.5)
    assert score_chunk(chunk, config) == 0.5


def test_curate_keeps_above_threshold_drops_below():
    config = CuratorConfig(weights={"similarity": 1.0, "recency": 0.0, "trust": 0.0},
                            threshold=0.5)
    chunks = [
        DocumentChunk(text="high relevance chunk", similarity=0.9),
        DocumentChunk(text="low relevance chunk", similarity=0.1),
    ]
    result = curate(chunks, config)
    assert [c.text for c in result.kept] == ["high relevance chunk"]
    assert len(result.dropped) == 1
    assert "below threshold" in result.dropped[0][1]


def test_curate_exactly_at_threshold_is_kept():
    config = CuratorConfig(weights={"similarity": 1.0, "recency": 0.0, "trust": 0.0},
                            threshold=0.5)
    chunks = [DocumentChunk(text="borderline", similarity=0.5)]
    result = curate(chunks, config)
    assert len(result.kept) == 1


def test_curate_all_above_threshold():
    config = CuratorConfig(weights={"similarity": 1.0, "recency": 0.0, "trust": 0.0},
                            threshold=0.1)
    chunks = [DocumentChunk(text=f"chunk {i}", similarity=0.9) for i in range(3)]
    result = curate(chunks, config)
    assert len(result.kept) == 3
    assert result.dropped == []


def test_curate_all_below_threshold():
    config = CuratorConfig(weights={"similarity": 1.0, "recency": 0.0, "trust": 0.0},
                            threshold=0.9)
    chunks = [DocumentChunk(text=f"chunk {i}", similarity=0.1) for i in range(3)]
    result = curate(chunks, config)
    assert result.kept == []
    assert len(result.dropped) == 3


def test_curate_dedup_drops_lower_scored_duplicate():
    config = CuratorConfig(weights={"similarity": 1.0, "recency": 0.0, "trust": 0.0},
                            threshold=0.0, dedup_threshold=0.8)
    chunks = [
        DocumentChunk(text="the quick brown fox jumps", similarity=0.5),
        DocumentChunk(text="the quick brown fox jumps", similarity=0.9),  # near-identical, higher score
    ]
    result = curate(chunks, config)
    assert len(result.kept) == 1
    assert result.kept[0].similarity == 0.9  # higher-scored duplicate wins
    assert len(result.dropped) == 1
    assert "duplicate of chunk" in result.dropped[0][1]


def test_curate_dedup_does_not_drop_distinct_chunks():
    config = CuratorConfig(weights={"similarity": 1.0, "recency": 0.0, "trust": 0.0},
                            threshold=0.0, dedup_threshold=0.9)
    chunks = [
        DocumentChunk(text="completely different content here", similarity=0.5),
        DocumentChunk(text="another unrelated topic entirely", similarity=0.9),
    ]
    result = curate(chunks, config)
    assert len(result.kept) == 2


def test_curate_max_chunks_cap():
    config = CuratorConfig(weights={"similarity": 1.0, "recency": 0.0, "trust": 0.0},
                            threshold=0.0, max_chunks=2)
    chunks = [
        DocumentChunk(text="a distinct chunk one", similarity=0.9),
        DocumentChunk(text="a distinct chunk two", similarity=0.8),
        DocumentChunk(text="a distinct chunk three", similarity=0.7),
    ]
    result = curate(chunks, config)
    assert len(result.kept) == 2
    assert [c.similarity for c in result.kept] == [0.9, 0.8]  # highest scores kept
    reasons = [r for _, r in result.dropped]
    assert "exceeds max_chunks cap" in reasons


def test_curate_scores_aligned_with_kept():
    config = CuratorConfig(weights={"similarity": 1.0, "recency": 0.0, "trust": 0.0},
                            threshold=0.0)
    chunks = [
        DocumentChunk(text="alpha distinct content", similarity=0.3),
        DocumentChunk(text="beta distinct content", similarity=0.7),
    ]
    result = curate(chunks, config)
    assert len(result.scores) == len(result.kept)
    assert result.scores == sorted(result.scores, reverse=True)


def test_build_documents_joins_kept_chunks():
    config = CuratorConfig(weights={"similarity": 1.0, "recency": 0.0, "trust": 0.0},
                            threshold=0.5)
    chunks = [
        DocumentChunk(text="chunk one content", similarity=0.9),
        DocumentChunk(text="chunk two content", similarity=0.1),
    ]
    documents, result = build_documents(chunks, config)
    assert documents == "chunk one content"
    assert len(result.kept) == 1


def test_context_precision_detects_used_chunk():
    chunk = DocumentChunk(text="the mitochondria is the powerhouse of the cell", similarity=1.0)
    response = "As noted, the mitochondria is the powerhouse of the cell in eukaryotes."
    result = context_precision([chunk], response)
    assert result["precision"] == 1.0
    assert result["per_chunk_used"] == [True]


def test_context_precision_detects_unused_chunk():
    chunk = DocumentChunk(text="quantum entanglement violates local realism", similarity=1.0)
    response = "The weather today is sunny with a light breeze."
    result = context_precision([chunk], response)
    assert result["precision"] == 0.0
    assert result["per_chunk_used"] == [False]


def test_context_precision_empty_kept_is_vacuously_perfect():
    result = context_precision([], "any response")
    assert result["precision"] == 1.0
    assert result["per_chunk_used"] == []


def test_context_precision_mixed_usage():
    used = DocumentChunk(text="the mitochondria is the powerhouse of the cell", similarity=1.0)
    unused = DocumentChunk(text="mount everest is the tallest mountain on earth", similarity=1.0)
    response = "As noted, the mitochondria is the powerhouse of the cell in eukaryotes."
    result = context_precision([used, unused], response)
    assert result["precision"] == 0.5
    assert result["per_chunk_used"] == [True, False]


def test_prompt_from_chunks_curates_and_joins():
    config = CuratorConfig(weights={"similarity": 1.0, "recency": 0.0, "trust": 0.0},
                            threshold=0.5)
    chunks = [
        DocumentChunk(text="relevant chunk content", similarity=0.9),
        DocumentChunk(text="irrelevant chunk content", similarity=0.1),
    ]
    prompt = Prompt.from_chunks(chunks, config, query="What is relevant?")
    assert prompt.documents == "relevant chunk content"
    assert prompt.query == "What is relevant?"


def test_prompt_from_chunks_default_config():
    """Omitting config falls back to CuratorConfig() defaults."""
    chunks = [DocumentChunk(text="some retrieved content here", similarity=0.95)]
    prompt = Prompt.from_chunks(chunks)
    assert prompt.documents == "some retrieved content here"


def test_cli_curate_end_to_end(tmp_path):
    """`contextops curate` runs end-to-end on a sample JSON file."""
    import json as jsonlib

    from click.testing import CliRunner

    from contextops.cli import main

    chunks_file = tmp_path / "chunks.json"
    chunks_file.write_text(jsonlib.dumps([
        {"text": "highly relevant retrieved chunk", "similarity": 0.95},
        {"text": "totally irrelevant noise chunk", "similarity": 0.05},
    ]))

    runner = CliRunner()
    result = runner.invoke(main, [
        "curate",
        "--chunks", str(chunks_file),
        "--weights", "similarity=1.0,recency=0.0,trust=0.0",
        "--threshold", "0.5",
    ])
    assert result.exit_code == 0, result.output
    assert "Kept (1)" in result.output
    assert "Dropped (1)" in result.output
