"""Optional integration: log calls automatically with a `litellm` callback.

This is intentionally opt-in. Install with:
    pip install "contextops[integrations]"
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contextops.models import CallLog

try:
    import litellm  # type: ignore
except ImportError:  # pragma: no cover - optional dep
    litellm = None  # type: ignore


def install_callback(db_path: str | None = None) -> None:
    """Register a litellm success callback that logs every call to contextops."""
    if litellm is None:
        raise RuntimeError(
            "litellm not installed. Run: pip install 'contextops[integrations]'"
        )

    from contextops.logger import Logger

    logger = Logger(Path(db_path) if db_path else None)

    def _callback(
        kwargs: dict[str, Any],
        completion_response: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        try:
            usage = getattr(completion_response, "usage", None) or {}
            model = getattr(completion_response, "model", kwargs.get("model", "unknown"))

            entry = CallLog(
                timestamp=datetime.now(timezone.utc).isoformat(),
                model=model,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                cached_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                cost_usd=getattr(completion_response, "_hidden_params", {})
                .get("response_cost", 0.0)
                or 0.0,
                latency_ms=(end_time - start_time) * 1000,
                prompt_hash="",
                section_order=[],
                metadata={"via": "litellm"},
            )
            logger.log(entry)
        except Exception:
            # Never let logging break the user's app.
            pass

    litellm.success_callback = [_callback]