"""Tests for the local SQLite logger."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from contextops.logger import Logger
from contextops.models import CallLog


def _make_entry(model="gpt-4o", tokens=100, cost=0.001):
    return CallLog(
        timestamp=datetime.now(timezone.utc).isoformat(),
        model=model,
        prompt_tokens=tokens,
        completion_tokens=50,
        cached_tokens=80,
        cost_usd=cost,
        latency_ms=500,
    )


def test_log_and_stats():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "test.db"
        logger = Logger(db)
        for _ in range(3):
            logger.log(_make_entry())
        s = logger.stats()
        assert s["total_calls"] == 3
        assert s["total_prompt_tokens"] == 300
        assert s["cache_hit_rate"] > 0


def test_recent_ordering():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "test.db"
        logger = Logger(db)
        logger.log(_make_entry(model="gpt-4o"))
        logger.log(_make_entry(model="claude-sonnet-4.6"))
        recent = logger.recent(limit=10)
        assert len(recent) == 2
        # Most recent first
        assert recent[0]["model"] == "claude-sonnet-4.6"


def test_by_model_aggregation():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "test.db"
        logger = Logger(db)
        for _ in range(2):
            logger.log(_make_entry(model="gpt-4o"))
        for _ in range(3):
            logger.log(_make_entry(model="claude-haiku-4.5"))
        s = logger.stats()
        models = {row["model"]: row["n"] for row in s["by_model"]}
        assert models["gpt-4o"] == 2
        assert models["claude-haiku-4.5"] == 3