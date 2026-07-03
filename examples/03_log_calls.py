"""Manually log a few LLM calls and view stats."""

from datetime import datetime, timezone

from contextops.logger import Logger
from contextops.models import CallLog


def main() -> None:
    logger = Logger()  # writes to ~/.contextops/calls.db

    # Pretend we just made 4 calls.
    sample = [
        CallLog(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model="gpt-4o",
            prompt_tokens=2400,
            completion_tokens=180,
            cached_tokens=2100,  # ~87% cache hit
            cost_usd=0.0017,
            latency_ms=820,
            prompt_hash="abc123",
            section_order=["system", "tools", "documents", "query"],
        ),
        CallLog(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model="gpt-4o",
            prompt_tokens=2400,
            completion_tokens=210,
            cached_tokens=2050,
            cost_usd=0.0019,
            latency_ms=910,
            prompt_hash="def456",
            section_order=["system", "tools", "documents", "query"],
        ),
        CallLog(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model="claude-sonnet-4.6",
            prompt_tokens=3100,
            completion_tokens=240,
            cached_tokens=0,
            cost_usd=0.0108,
            latency_ms=1340,
            prompt_hash="ghi789",
            section_order=["system", "history", "query"],  # bad order
        ),
    ]

    for entry in sample:
        logger.log(entry)

    print("Stats:")
    import json

    print(json.dumps(logger.stats(), indent=2))


if __name__ == "__main__":
    main()