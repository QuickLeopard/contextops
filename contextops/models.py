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
    # When set, `sections()` yields in this order instead of declaration order.
    # `reorder()` and the bench's `_reverse_prompt()` set this so callers see the
    # new order without a private-attr monkeypatch. None = declaration order.
    render_order: Optional[list[Section]] = None

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
        if self.render_order is not None:
            by_name = dict(out)
            return [(name, by_name[name]) for name in self.render_order if name in by_name]
        return out

    # --- Importers for existing message lists ------------------------------
    # Most teams already have a `messages = [{"role": ..., "content": ...}]`
    # list. These classmethods build a `Prompt` from it without requiring a
    # rewrite of their prompt-construction code.

    @classmethod
    def from_openai_messages(cls, messages: list[dict]) -> "Prompt":
        """Build a Prompt from an OpenAI-style messages list.

        Mapping:
        - All `system` messages are concatenated into `system`.
        - The final `user` message becomes `query`.
        - Any prior `user`/`assistant` turns become `history` (in order).
        - `tool` messages are ignored (they carry tool-call results, not prompt text).

        This is a lossy import for the purpose of cache-aware reordering: it
        preserves the text content in the right slots, not call/function metadata.
        """
        system_parts: list[str] = []
        turns: list[dict] = []  # user/assistant turns in conversation order
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            # OpenAI content can be a list of parts (vision/tools); coerce to text.
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            if role == "system":
                if content:
                    system_parts.append(content)
            elif role in ("user", "assistant"):
                if content:
                    turns.append({"role": role, "content": content})
            # tool/function messages are dropped — not prompt text.

        # The last user turn becomes the query; everything before it is history.
        query = ""
        history = turns
        if turns and turns[-1]["role"] == "user":
            history = turns[:-1]
            query = turns[-1]["content"]
        elif turns:
            # No trailing user turn — keep all turns as history, leave query empty.
            query = ""
        return cls(
            system="\n\n".join(system_parts),
            history=history,
            query=query,
        )

    @classmethod
    def from_anthropic_messages(
        cls, messages: list[dict], system: str | list | None = None,
    ) -> "Prompt":
        """Build a Prompt from an Anthropic-style messages list.

        Anthropic keeps `system` as a separate top-level parameter (not in the
        messages array). It may be a string or a list of content blocks.

        Mapping is the same as `from_openai_messages` for the `messages` array;
        the top-level `system` is concatenated into the `system` field.
        """
        prompt = cls.from_openai_messages(messages)
        if isinstance(system, list):
            sys_text = " ".join(
                b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            sys_text = system or ""
        # Prepend the top-level system so it sorts to the front regardless.
        if sys_text:
            prompt.system = f"{sys_text}\n\n{prompt.system}".strip() if prompt.system else sys_text
        return prompt


class OptimizationResult(BaseModel):
    """Result of running `optimize()` or `reorder()` on a Prompt."""

    original_sections: list[tuple[Section, str]]
    optimized_sections: list[tuple[Section, str]]
    original_tokens: int
    optimized_tokens: int
    original_cache_hit_rate: float   # 0..1, the un-optimized ordering
    estimated_cache_hit_rate: float  # 0..1, the optimized ordering
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