"""Pydantic models — the prompt is a structured dict, not a string."""

from __future__ import annotations

from typing import Any, Literal, Optional


from pydantic import BaseModel, Field, field_validator

# Canonical section order. Higher stability = higher position.
# Each section has a "stability" score from 0 (variable) to 1 (stable across calls).
# The reordering logic uses this to maximize cache hits.
Section = Literal["system", "tools", "role", "context", "history", "documents", "query"]


class HistoryMessage(BaseModel):
    """A single chat history message."""

    role: str = "user"
    content: str = ""

    @classmethod
    def coerce(cls, item: Any) -> "HistoryMessage":
        if isinstance(item, HistoryMessage):
            return item
        if isinstance(item, dict):
            return cls(**item)
        if isinstance(item, str):
            return cls(role="user", content=item)
        raise ValueError(f"Unsupported history entry: {type(item)}")


class Prompt(BaseModel):
    """Structured prompt. Sections are ordered by stability, but you can give them in any order."""

    system: str = ""
    tools: str = ""
    role: str = ""
    context: str = ""
    documents: str = ""
    history: list[Any] = Field(default_factory=list)
    query: str = ""

    @field_validator("history", mode="before")
    @classmethod
    def _coerce_history(cls, v):
        """Accept dicts, strings, or HistoryMessage — normalize before validation."""
        if not v:
            return []
        return [HistoryMessage.coerce(item) for item in v]

    model: str = "gpt-4o"
    goal: Literal["cache_friendly", "balanced", "quality"] = "cache_friendly"

    def sections(self) -> list[tuple[Section, str]]:
        """All non-empty sections as (name, content) tuples. Order doesn't matter here."""
        out: list[tuple[Section, str]] = []
        if self.system:
            out.append(("system", self.system))
        if self.tools:
            out.append(("tools", self.tools))
        if self.role:
            out.append(("role", self.role))
        if self.context:
            out.append(("context", self.context))
        if self.documents:
            out.append(("documents", self.documents))
        if self.history:
            # Normalize every entry to a HistoryMessage, then render.
            msgs = [HistoryMessage.coerce(m) for m in self.history]
            rendered = "\n".join(f"{m.role}: {m.content}" for m in msgs)
            out.append(("history", rendered))
        if self.query:
            out.append(("query", self.query))
        return out


class OptimizationResult(BaseModel):
    """Result of running `optimize()` or `reorder()` on a Prompt."""

    original_sections: list[tuple[Section, str]]
    optimized_sections: list[tuple[Section, str]]
    original_tokens: int
    optimized_tokens: int
    estimated_cache_hit_rate: float  # 0..1
    estimated_cost_savings_usd: float  # per 1000 calls (rough)
    model: str
    notes: list[str] = Field(default_factory=list)

    def diff(self) -> str:
        """Human-readable diff of section order changes."""
        original_order = [s[0] for s in self.original_sections]
        new_order = [s[0] for s in self.optimized_sections]
        if original_order == new_order:
            return "no reordering needed"
        return f"{' → '.join(original_order)} → {' → '.join(new_order)}"


class CallLog(BaseModel):
    """A single LLM call record. Written to local SQLite."""

    timestamp: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: Optional[float] = None
    prompt_hash: str = ""
    section_order: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)