"""Shared bench data types.

Lives in its own module so the client Protocol (`client_protocol.py`) can
reference `CompletionResponse` without importing `clients.py` (which would
create a cycle, since `clients.py` needs the Protocol).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BenchResult:
    """One benchmark observation."""

    prompt_id: int
    model: str
    provider: str
    use_optimized: bool
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    error: str = ""
    section_order: list[str] = field(default_factory=list)


@dataclass
class CompletionResponse:
    """Normalised response from any provider."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    cost_usd: float
    model: str
    raw: dict
