"""Local SQLite logger for LLM calls. No cloud, no SDK — just append."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from contextops.models import CallLog

DEFAULT_DB_PATH = Path.home() / ".contextops" / "calls.db"


class Logger:
    """Append-only local logger. Threadsafe enough for single-process dev use."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
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
            return cur.lastrowid

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