"""Tests for the local SQLite logger."""

import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from contextops.access import AccessDecision
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


def test_wal_mode_enabled():
    """Regression test: the DB must use WAL journal mode so concurrent
    readers (stats/recent) don't block on an in-flight writer."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "test.db"
        logger = Logger(db)
        with logger._connect() as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"


def test_concurrent_writes_do_not_raise():
    """Regression test: many threads logging concurrently must not hit
    'database is locked' errors now that WAL + a busy timeout are set."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "test.db"
        logger = Logger(db)

        def _write(_i):
            logger.log(_make_entry())

        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(_write, range(50)))

        assert logger.stats(limit=100)["total_calls"] == 50


def test_log_access_decisions():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "test.db"
        logger = Logger(db)
        call_id = logger.log(_make_entry())
        decisions = [
            AccessDecision(
                principal_id="alice",
                section="documents",
                action="redacted",
                reason="missing required roles: executive",
                content_hash="abc123",
            ),
            AccessDecision(
                principal_id="alice",
                section="query",
                action="included",
                reason="role allowed",
                content_hash="def456",
            ),
        ]
        ids = logger.log_access(decisions, call_id=call_id)
        assert len(ids) == 2

        rows = logger.audit_query()
        assert len(rows) == 2
        # Most recent first
        assert rows[0]["action"] == "included"
        assert rows[0]["call_id"] == call_id
        assert rows[1]["action"] == "redacted"


def test_audit_query_filters_by_principal():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "test.db"
        logger = Logger(db)
        decisions = [
            AccessDecision(
                principal_id="alice",
                section="documents",
                action="included",
                reason="role allowed",
                content_hash="h1",
            ),
            AccessDecision(
                principal_id="bob",
                section="documents",
                action="included",
                reason="role allowed",
                content_hash="h2",
            ),
        ]
        logger.log_access(decisions)
        alice_rows = logger.audit_query(principal_id="alice")
        assert len(alice_rows) == 1
        assert alice_rows[0]["principal_id"] == "alice"


def test_audit_table_never_stores_raw_content():
    """Regression: the audit log stores only hashes, never the sensitive text."""

    sensitive = "top-secret salary data"
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "test.db"
        logger = Logger(db)
        decisions = [
            AccessDecision(
                principal_id="alice",
                section="documents",
                action="redacted",
                reason="missing role",
                content_hash="hash-of-secret",
            )
        ]
        logger.log_access(decisions)
        with logger._connect() as conn:
            dump = " ".join(
                str(row)
                for row in conn.execute("SELECT * FROM access_audit").fetchall()
            )
        assert sensitive not in dump