"""Access-aware context filtering.

Pure functions for redacting prompt sections based on a principal's roles.
No network calls, no LLM calls. See `docs/PLAN_v1.0.md` and `ROADMAP.md`
Track B for design rationale.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Protocol

from contextops.models import HistoryMessage, Prompt


@dataclass(frozen=True)
class Principal:
    """Who is making the request."""

    id: str
    roles: set[str] = field(default_factory=set)
    attributes: dict[str, str] | None = None


@dataclass(frozen=True)
class AccessPolicy:
    """A single rule: content tagged with `required_roles` is visible only to
    principals whose roles overlap with that set. An empty set means public.
    """

    required_roles: set[str] = field(default_factory=set)
    section: str | None = None  # None = applies to all sections


class PolicyEngine(Protocol):
    """Pluggable access decision engine."""

    def is_allowed(self, principal: Principal, content: TaggedContent) -> bool: ...


@dataclass(frozen=True)
class TaggedContent:
    """A piece of content with an access tag."""

    text: str
    section: str
    required_roles: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class AccessDecision:
    """One inclusion/redaction decision, ready for the audit log.

    `content_hash` is SHA-256 of the original content so the audit trail can
    correlate without duplicating sensitive text.
    """

    principal_id: str
    section: str
    action: str  # "included" | "redacted"
    reason: str
    content_hash: str


@dataclass
class RedactionResult:
    """Output of `apply_access_policy()`."""

    prompt: Prompt
    included: list[tuple[TaggedContent, str]]
    redacted: list[tuple[TaggedContent, str]]
    decisions: list[AccessDecision]


class RoleBasedPolicyEngine:
    """Default engine: allow if the principal has any of the required roles
    (or the content has no required roles).
    """

    def is_allowed(self, principal: Principal, content: TaggedContent) -> bool:
        if not content.required_roles:
            return True
        return bool(principal.roles & content.required_roles)


_SCALAR_SECTIONS = frozenset(
    {"system", "tools", "role", "context", "documents", "query"}
)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_prompt(sections: dict[str, list[str]]) -> Prompt:
    """Reconstruct a Prompt from filtered section content."""

    kwargs: Dict[str, Any] = {}
    for section_name, contents in sections.items():
        if section_name == "history":
            kwargs["history"] = [HistoryMessage(role="user", content=c) for c in contents]
        elif section_name in _SCALAR_SECTIONS:
            kwargs[section_name] = "\n\n".join(contents)
        else:
            # Unknown sections land in `context` so data isn't silently lost.
            kwargs["context"] = kwargs.get("context", "") + "\n\n".join(contents)
    return Prompt(**kwargs)


def apply_access_policy(
    contents: list[TaggedContent],
    principal: Principal,
    engine: PolicyEngine | None = None,
) -> RedactionResult:
    """Filter tagged content by principal permissions.

    Returns a new `Prompt` containing only allowed content, plus lists of
    included/redacted items and a decision trail suitable for `Logger.log_access()`.
    """

    engine = engine or RoleBasedPolicyEngine()
    sections: dict[str, list[str]] = {}
    included: list[tuple[TaggedContent, str]] = []
    redacted: list[tuple[TaggedContent, str]] = []
    decisions: list[AccessDecision] = []

    for content in contents:
        allowed = engine.is_allowed(principal, content)
        if allowed:
            reason = "role allowed"
            action = "included"
            included.append((content, reason))
            sections.setdefault(content.section, []).append(content.text)
        else:
            missing = ", ".join(sorted(content.required_roles - principal.roles)) or "no matching role"
            reason = f"missing required roles: {missing}"
            action = "redacted"
            redacted.append((content, reason))

        decisions.append(
            AccessDecision(
                principal_id=principal.id,
                section=content.section,
                action=action,
                reason=reason,
                content_hash=_hash(content.text),
            )
        )

    prompt = _build_prompt(sections)
    return RedactionResult(
        prompt=prompt,
        included=included,
        redacted=redacted,
        decisions=decisions,
    )
