"""Pluggable embedders for the RAG curator bench's synthetic dataset.

`contextops_bench.curator_bench.generate_curation_dataset()` defaults to
synthetic `random.uniform(...)` similarity ranges — fast and deterministic,
but not what a real retriever would report. Passing an `Embedder` here
computes actual cosine similarity between the query and each chunk's text,
closer to production RAG traffic.

Two implementations, both zero heavy-dependency (no torch/sentence-
transformers, per project convention — see `contextops_bench/clients.py`'s
urllib-only HTTP pattern):

- `TfidfEmbedder`: pure stdlib, offline, free. Builds a per-call TF-IDF
  vector for each text against the corpus of texts passed to `embed()` in
  the same call (query + its own chunks) — good enough to separate
  "on-topic" from "off-topic" text without any network dependency.
- `OpenAIEmbedder`: calls OpenAI's `/v1/embeddings` endpoint (same
  `urllib`-based HTTP pattern as `contextops_bench/clients.py`'s direct
  provider clients). Requires `OPENAI_API_KEY` and costs a small amount
  per call — opt-in only.
"""

from __future__ import annotations

import json as jsonlib
import math
import os
import urllib.error
import urllib.request
from collections import Counter
from typing import Protocol


class Embedder(Protocol):
    """Minimal interface: embed a batch of texts into equal-length vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity clamped to [0, 1] (RAG similarity scores are
    conventionally non-negative; raw cosine can be negative for
    near-orthogonal/opposite sparse vectors, which we treat as "0 similarity"
    rather than propagating a negative score into `curate()`)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class TfidfEmbedder:
    """Zero-dependency TF-IDF embedder, scoped to one `embed()` call's batch.

    Each call to `embed()` builds its own corpus (the texts passed in that
    call) — there is no persistent global vocabulary across calls. This
    matches how `generate_curation_dataset()` uses it: one `embed([query,
    *chunk_texts])` call per dataset item, so TF-IDF is computed within that
    item's own query+chunks, not polluted by unrelated items' vocabulary.
    """

    PROVIDER = "tfidf"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        tokenized = [_tokenize(t) for t in texts]
        doc_count = len(tokenized)

        # Document frequency per term, for IDF.
        df: Counter = Counter()
        for tokens in tokenized:
            for term in set(tokens):
                df[term] += 1

        vocab = sorted(df.keys())
        idf = {
            term: math.log((1 + doc_count) / (1 + df[term])) + 1.0
            for term in vocab
        }

        vectors: list[list[float]] = []
        for tokens in tokenized:
            tf = Counter(tokens)
            total = len(tokens) or 1
            vectors.append([(tf.get(term, 0) / total) * idf[term] for term in vocab])
        return vectors


class OpenAIEmbedder:
    """Calls OpenAI's `/v1/embeddings` endpoint. Requires `OPENAI_API_KEY`.

    Costs a small amount per call (text-embedding-3-small is ~$0.02 / 1M
    tokens as of writing) — opt-in only via `--embedder openai`.
    """

    PROVIDER = "openai"
    MODEL = "text-embedding-3-small"
    BASE_URL = "https://api.openai.com/v1"

    def __init__(self, api_key: str | None = None, timeout: float = 60.0):
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OpenAIEmbedder requires OPENAI_API_KEY env var or api_key arg. "
                "Get a key at https://platform.openai.com/api-keys"
            )
        self.api_key = api_key
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {"model": self.MODEL, "input": texts}
        data = jsonlib.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.BASE_URL}/embeddings",
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = jsonlib.loads(resp.read().decode("utf-8"))
        # OpenAI's embeddings response preserves input order in "data".
        return [item["embedding"] for item in raw["data"]]


def get_embedder(name: str) -> Embedder:
    """Factory: `"tfidf"` (default, offline) or `"openai"` (opt-in, costs $)."""
    key = name.lower()
    if key == "tfidf":
        return TfidfEmbedder()
    if key == "openai":
        return OpenAIEmbedder()
    raise ValueError(f"Unknown embedder: {name!r}. Use 'tfidf' or 'openai'.")
