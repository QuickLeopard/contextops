"""Local SQLite logger for LLM calls. No cloud, no SDK — just append."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from contextops.access import AccessDecision
from contextops.models import CallLog

import datetime as _datetime

DEFAULT_DB_PATH = Path.home() / ".contextops" / "calls.db"


class Logger:
    """Append-only local logger. Threadsafe enough for single-process dev use."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        # `timeout` makes SQLite retry (instead of immediately raising
        # "database is locked") when another connection briefly holds the
        # write lock. WAL mode lets readers (stats/recent) proceed while a
        # write is in flight — important for `install_callback`, which may
        # log concurrently from multiple in-flight requests in a real app.
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    cached_tokens INTEGER NOT NULL DEFAULT 0,
                    cost_usd REAL NOT NULL DEFAULT 0.0,
                    latency_ms REAL,
                    prompt_hash TEXT,
                    section_order TEXT,
                    metadata TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_calls_timestamp ON calls(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_calls_model ON calls(model)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS access_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    principal_id TEXT NOT NULL,
                    call_id INTEGER,
                    section TEXT NOT NULL,
                    action TEXT NOT NULL CHECK(action IN ('included','redacted')),
                    reason TEXT,
                    content_hash TEXT NOT NULL,
                    FOREIGN KEY (call_id) REFERENCES calls(id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_access_audit_call ON access_audit(call_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_access_audit_principal ON access_audit(principal_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_access_audit_timestamp ON access_audit(timestamp)"
            )

    def log(self, entry: CallLog) -> int:
        """Append one call. Returns the row id."""
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO calls
                    (timestamp, model, prompt_tokens, completion_tokens,
                     cached_tokens, cost_usd, latency_ms, prompt_hash,
                     section_order, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.timestamp,
                    entry.model,
                    entry.prompt_tokens,
                    entry.completion_tokens,
                    entry.cached_tokens,
                    entry.cost_usd,
                    entry.latency_ms,
                    entry.prompt_hash,
                    json.dumps(entry.section_order),
                    json.dumps(entry.metadata),
                ),
            )
            row_id = cur.lastrowid
            return row_id if row_id is not None else -1

    def log_access(
        self,
        decisions: list[AccessDecision],
        *,
        call_id: Optional[int] = None,
        timestamp: Optional[str] = None,
    ) -> list[int]:
        """Append access decisions for one prompt. Returns inserted row ids."""

        ts = timestamp or _datetime.datetime.utcnow().isoformat()
        row_ids: list[int] = []
        with self._connect() as conn:
            for decision in decisions:
                cur = conn.execute(
                    """
                    INSERT INTO access_audit
                        (timestamp, principal_id, call_id, section,
                         action, reason, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ts,
                        decision.principal_id,
                        call_id,
                        decision.section,
                        decision.action,
                        decision.reason,
                        decision.content_hash,
                    ),
                )
                row_id = cur.lastrowid
                row_ids.append(row_id if row_id is not None else -1)
        return row_ids

    def audit_query(
        self,
        *,
        principal_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Return recent access-audit rows, optionally filtered by principal."""

        with self._connect() as conn:
            sql = """
                SELECT
                    a.id,
                    a.timestamp,
                    a.principal_id,
                    a.call_id,
                    a.section,
                    a.action,
                    a.reason,
                    a.content_hash
                FROM access_audit a
            """
            params: list[object] = []
            if principal_id is not None:
                sql += " WHERE a.principal_id = ?"
                params.append(principal_id)
            sql += " ORDER BY a.id DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def stats(self, limit: int = 100) -> dict:
        """Return aggregate stats over recent calls."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(prompt_tokens) AS total_prompt_tokens,
                    SUM(completion_tokens) AS total_completion_tokens,
                    SUM(cached_tokens) AS total_cached_tokens,
                    SUM(cost_usd) AS total_cost_usd,
                    AVG(latency_ms) AS avg_latency_ms
                FROM (SELECT * FROM calls ORDER BY id DESC LIMIT ?)
                """,
                (limit,),
            ).fetchone()
            by_model = conn.execute(
                """
                SELECT model, COUNT(*) AS n, SUM(cost_usd) AS cost
                FROM (SELECT * FROM calls ORDER BY id DESC LIMIT ?)
                GROUP BY model
                ORDER BY n DESC
                """,
                (limit,),
            ).fetchall()

            total = rows["total"] or 0
            cached = rows["total_cached_tokens"] or 0
            prompt = rows["total_prompt_tokens"] or 0
            cache_hit_rate = (cached / prompt) if prompt > 0 else 0.0

            return {
                "limit": limit,
                "total_calls": total,
                "total_prompt_tokens": prompt,
                "total_completion_tokens": rows["total_completion_tokens"] or 0,
                "total_cached_tokens": cached,
                "cache_hit_rate": round(cache_hit_rate, 3),
                "total_cost_usd": round(rows["total_cost_usd"] or 0.0, 6),
                "avg_latency_ms": round(rows["avg_latency_ms"] or 0.0, 2),
                "by_model": [dict(r) for r in by_model],
            }

    def recent(self, limit: int = 20) -> list[dict]:
        """Return the most recent N calls."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM calls ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["section_order"] = json.loads(d.pop("section_order") or "[]")
                d["metadata"] = json.loads(d.pop("metadata") or "{}")
                out.append(d)
            return out


@contextmanager
def run(db_path: Optional[Path] = None) -> Iterator[Logger]:
    """Context manager for ad-hoc logging. Mostly here for future expansion."""
    logger = Logger(db_path)
    try:
        yield logger
    finally:
        pass