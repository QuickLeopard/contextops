"""The contract every bench client satisfies.

Defines the `complete(...)` signature all clients share, plus the capability
flags `run_one` consults. Made explicit (rather than left implicit/duck-typed)
so the type-checker can verify substitutability and new clients conform.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from contextops_bench.types import CompletionResponse


@runtime_checkable
class BenchClient(Protocol):
    """Every bench client provides these."""

    PROVIDER: str
    supports_split_messages: bool

    def list_models(self) -> list[str]: ...

    def complete(
        self,
        *,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 64,
        system: str | None = None,
    ) -> CompletionResponse: ...
