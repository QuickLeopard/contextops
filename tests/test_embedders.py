"""Tests for contextops_bench.embedders — pluggable similarity for the
curator bench's synthetic dataset. TfidfEmbedder is pure/offline; OpenAI-
Embedder is tested only via a mocked HTTP call (no real network/API key)."""

from __future__ import annotations

import json as jsonlib
from unittest.mock import MagicMock, patch

import pytest

from contextops_bench.embedders import (
    OpenAIEmbedder,
    TfidfEmbedder,
    cosine_similarity,
    get_embedder,
)


def test_cosine_similarity_identical_vectors_is_one():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors_clamped_to_zero():
    """Raw cosine of opposite vectors is -1; clamped to 0 since RAG
    similarity scores are conventionally non-negative."""
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == 0.0


def test_cosine_similarity_empty_or_mismatched_vectors_is_zero():
    assert cosine_similarity([], []) == 0.0
    assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0


def test_cosine_similarity_zero_vector_is_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_tfidf_embedder_identical_text_has_similarity_one():
    embedder = TfidfEmbedder()
    vectors = embedder.embed(["the quick brown fox", "the quick brown fox"])
    assert cosine_similarity(vectors[0], vectors[1]) == pytest.approx(1.0)


def test_tfidf_embedder_unrelated_text_has_lower_similarity():
    embedder = TfidfEmbedder()
    vectors = embedder.embed([
        "What is the capital of France?",
        "Paris is the capital and most populous city of France.",
        "The recipe calls for two cups of flour and a teaspoon of salt.",
    ])
    query_vec, relevant_vec, noise_vec = vectors
    relevant_sim = cosine_similarity(query_vec, relevant_vec)
    noise_sim = cosine_similarity(query_vec, noise_vec)
    assert relevant_sim > noise_sim


def test_tfidf_embedder_deterministic():
    embedder = TfidfEmbedder()
    texts = ["alpha beta gamma", "beta gamma delta", "totally unrelated text"]
    a = embedder.embed(texts)
    b = embedder.embed(texts)
    assert a == b


def test_tfidf_embedder_empty_input():
    assert TfidfEmbedder().embed([]) == []


def test_tfidf_embedder_empty_string_text_does_not_crash():
    embedder = TfidfEmbedder()
    vectors = embedder.embed(["", "some text", ""])
    assert len(vectors) == 3


def test_tfidf_embedder_vectors_same_length_as_vocab():
    embedder = TfidfEmbedder()
    vectors = embedder.embed(["a b c", "b c d"])
    vocab_size = len({"a", "b", "c", "d"})
    assert all(len(v) == vocab_size for v in vectors)


def test_get_embedder_tfidf():
    assert isinstance(get_embedder("tfidf"), TfidfEmbedder)
    assert isinstance(get_embedder("TFIDF"), TfidfEmbedder)  # case-insensitive


def test_get_embedder_unknown_raises():
    with pytest.raises(ValueError, match="Unknown embedder"):
        get_embedder("nonexistent")


def test_openai_embedder_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIEmbedder()


def test_openai_embedder_uses_explicit_api_key_arg(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    embedder = OpenAIEmbedder(api_key="sk-test-key")
    assert embedder.api_key == "sk-test-key"


def test_get_embedder_openai_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        get_embedder("openai")


def test_openai_embedder_embed_parses_response_preserving_order(monkeypatch):
    """Mock the HTTP call — no real network/API key needed."""
    embedder = OpenAIEmbedder(api_key="sk-test-key")

    fake_response_body = jsonlib.dumps({
        "data": [
            {"embedding": [0.1, 0.2], "index": 0},
            {"embedding": [0.3, 0.4], "index": 1},
        ]
    }).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.read.return_value = fake_response_body
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        vectors = embedder.embed(["hello", "world"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert mock_urlopen.called


def test_openai_embedder_embed_empty_input_skips_http_call():
    embedder = OpenAIEmbedder(api_key="sk-test-key")
    with patch("urllib.request.urlopen") as mock_urlopen:
        assert embedder.embed([]) == []
    mock_urlopen.assert_not_called()
