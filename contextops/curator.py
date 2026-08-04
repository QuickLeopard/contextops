"""RAG Curator — multi-signal filtering of retrieved chunks before they ever
reach the optimizer.

ContextOps' optimizer answers "in what order should I send my prompt
sections?". The curator answers the question that comes before it: "should
this retrieved chunk be in the prompt at all?" Retrieval (similarity search)
is a proxy for relevance, not ground truth — some retrieved chunks are
irrelevant, stale, or near-duplicates of chunks already selected. More
context is not free: it costs tokens and can increase hallucination risk
("lost in the middle" / distractor effect).

Everything here is a pure function — no network calls, no LLM calls. The
caller is responsible for supplying `similarity` (we don't want to force a
specific vector DB or embedding model).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DocumentChunk:
    """One retrieved chunk, plus optional signals beyond similarity."""

    text: str
    similarity: float                      # 0..1, required — caller-computed
    updated_at: Optional[float] = None     # unix timestamp, optional
    trust_score: Optional[float] = None    # 0..1, optional, caller-provided
    metadata: Optional[dict] = None


@dataclass
class CuratorConfig:
    """Scoring weights + cutoffs for `curate()`."""

    weights: dict[str, float] = field(
        default_factory=lambda: {"similarity": 0.6, "recency": 0.2, "trust": 0.2}
    )
    threshold: float = 0.6         # strict cutoff — chunks below this are dropped
    dedup_threshold: float = 0.9   # chunks more similar to each other than this are duplicates
    half_life_days: float = 30.0   # recency decay half-life
    max_chunks: Optional[int] = None  # optional hard cap even after filtering


@dataclass
class CurationResult:
    """Output of `curate()`."""

    kept: list[DocumentChunk]
    dropped: list[tuple[DocumentChunk, str]]  # (chunk, human-readable reason)
    scores: list[float]                        # combined score per kept chunk, same order as `kept`


def _normalized_weights(weights: dict[str, float]) -> dict[str, float]:
    """Re-normalize weights to sum to 1.0 so partial/odd weight dicts don't
    silently produce scores outside the expected 0..1 range."""
    total = sum(weights.values())
    if total <= 0:
        raise ValueError(f"CuratorConfig.weights must sum to > 0, got {weights}")
    return {k: v / total for k, v in weights.items()}


def _recency_score(updated_at: Optional[float], half_life_days: float, now: float) -> float:
    """Exponential half-life decay: a doc from today scores ~1.0, a doc aged
    exactly `half_life_days` scores 0.5, older docs decay further. Unknown
    recency is treated neutrally (1.0) — we don't want to punish chunks just
    because the caller didn't supply a timestamp.
    """
    if updated_at is None:
        return 1.0
    age_days = max(0.0, (now - updated_at) / 86400.0)
    return math.pow(0.5, age_days / half_life_days) if half_life_days > 0 else 1.0


def _trust_score(trust_score: Optional[float]) -> float:
    """Unknown trust is treated neutrally (1.0), same rationale as recency."""
    return 1.0 if trust_score is None else trust_score


def score_chunk(chunk: DocumentChunk, config: CuratorConfig, *, now: Optional[float] = None) -> float:
    """Combined 0..1 score for one chunk. Pure function — deterministic given
    `now` (defaults to current time only for the `updated_at` decay term)."""
    now = time.time() if now is None else now
    weights = _normalized_weights(config.weights)
    recency = _recency_score(chunk.updated_at, config.half_life_days, now)
    trust = _trust_score(chunk.trust_score)
    return (
        weights.get("similarity", 0.0) * chunk.similarity
        + weights.get("recency", 0.0) * recency
        + weights.get("trust", 0.0) * trust
    )


def _tokenize(text: str) -> set[str]:
    return set(text.lower().split())


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def curate(
    chunks: list[DocumentChunk],
    config: Optional[CuratorConfig] = None,
    *,
    now: Optional[float] = None,
) -> CurationResult:
    """Filter `chunks` down to the ones worth paying tokens for.

    Pipeline: score every chunk -> drop below `threshold` -> dedup remaining
    (keeping the higher-scored of any near-duplicate pair) -> apply
    `max_chunks` cap if set. Every dropped chunk carries a human-readable
    reason string for debugging bad RAG answers later.
    """
    config = config or CuratorConfig()
    now = time.time() if now is None else now

    # Score everything, remembering original index for reason strings.
    scored: list[tuple[int, DocumentChunk, float]] = [
        (i, c, score_chunk(c, config, now=now)) for i, c in enumerate(chunks)
    ]
    # Sort by score descending; ties broken by original order for determinism.
    scored.sort(key=lambda t: (-t[2], t[0]))

    dropped: list[tuple[DocumentChunk, str]] = []
    survivors: list[tuple[int, DocumentChunk, float]] = []
    for i, chunk, s in scored:
        if s < config.threshold:
            dropped.append((chunk, f"below threshold ({s:.2f} < {config.threshold:.2f})"))
        else:
            survivors.append((i, chunk, s))

    # Dedup: iterate best-score-first; a candidate is dropped if it's too
    # similar to a chunk already accepted (which is necessarily >= scored,
    # since we're iterating in descending score order).
    kept: list[tuple[int, DocumentChunk, float]] = []
    kept_tokens: list[set[str]] = []
    for i, chunk, s in survivors:
        candidate_tokens = _tokenize(chunk.text)
        duplicate_of: Optional[int] = None
        for kept_idx, (orig_i, _, _) in enumerate(kept):
            if _jaccard(candidate_tokens, kept_tokens[kept_idx]) >= config.dedup_threshold:
                duplicate_of = orig_i
                break
        if duplicate_of is not None:
            dropped.append((chunk, f"duplicate of chunk #{duplicate_of}"))
        else:
            kept.append((i, chunk, s))
            kept_tokens.append(candidate_tokens)

    # max_chunks cap — applied last, on the already-deduped, score-sorted list.
    if config.max_chunks is not None and len(kept) > config.max_chunks:
        overflow = kept[config.max_chunks:]
        kept = kept[: config.max_chunks]
        for _, chunk, _ in overflow:
            dropped.append((chunk, "exceeds max_chunks cap"))

    return CurationResult(
        kept=[c for _, c, _ in kept],
        dropped=dropped,
        scores=[s for _, _, s in kept],
    )


def build_documents(chunks: list[DocumentChunk], config: Optional[CuratorConfig] = None) -> tuple[str, CurationResult]:
    """Curate `chunks` and join the survivors into a single string suitable
    for `Prompt.documents`. Returns (documents_str, curation_result) so
    callers can inspect what was kept/dropped and why.
    """
    result = curate(chunks, config)
    documents = "\n\n".join(c.text for c in result.kept)
    return documents, result


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = text.lower().split()
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def context_precision(
    kept: list[DocumentChunk],
    response: str,
    *,
    ngram_size: int = 3,
    use_threshold: float = 0.05,
) -> dict:
    """Deterministic (no LLM call) estimate of how many `kept` chunks were
    actually drawn on in `response`, via word n-gram overlap.

    A chunk counts as "used" if the fraction of its n-grams that also appear
    in the response's n-grams exceeds `use_threshold`. This is a coarse
    heuristic (paraphrased content won't be detected), traded off against
    zero cost/latency — see ROADMAP.md Track A for the LLM-judge alternative.

    Returns {"precision": float, "per_chunk_used": list[bool]} — precision
    is `used_count / len(kept)`, or 1.0 for an empty `kept` list (vacuously
    true: nothing was provided, so nothing was wasted).
    """
    if not kept:
        return {"precision": 1.0, "per_chunk_used": []}

    response_ngrams = _ngrams(response, ngram_size)
    per_chunk_used: list[bool] = []
    for chunk in kept:
        chunk_ngrams = _ngrams(chunk.text, ngram_size)
        if not chunk_ngrams:
            per_chunk_used.append(False)
            continue
        overlap = len(chunk_ngrams & response_ngrams) / len(chunk_ngrams)
        per_chunk_used.append(overlap >= use_threshold)

    used_count = sum(1 for u in per_chunk_used if u)
    return {
        "precision": round(used_count / len(kept), 3),
        "per_chunk_used": per_chunk_used,
    }
