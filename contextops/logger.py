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
            # Budget thresholds (key/value) — v1 is CLI-only, no webhooks.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS budgets (
                    key TEXT PRIMARY KEY,
                    value_usd REAL NOT NULL
                )
                """
            )

    def log(self, entry: CallLog) -> int:
        """Append one call. Returns the row id.

        Also checks against any configured daily/monthly budget and emits a
        warning to stderr if the new spend crosses a threshold. The row is
        always written (logging never blocks on budget).
        """
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

    # --- Budgets ----------------------------------------------------------

    def set_budget(self, *, daily_usd: float | None = None,
                   monthly_usd: float | None = None) -> None:
        """Persist (or clear, if None) daily/monthly spend budgets."""
        with self._connect() as conn:
            if daily_usd is not None:
                conn.execute(
                    "INSERT OR REPLACE INTO budgets(key, value_usd) VALUES (?, ?)",
                    ("daily", float(daily_usd)),
                )
            if monthly_usd is not None:
                conn.execute(
                    "INSERT OR REPLACE INTO budgets(key, value_usd) VALUES (?, ?)",
                    ("monthly", float(monthly_usd)),
                )

    def get_budgets(self) -> dict[str, float | None]:
        """Return {'daily': float|None, 'monthly': float|None}."""
        with self._connect() as conn:
            rows = {r["key"]: r["value_usd"] for r in conn.execute(
                "SELECT key, value_usd FROM budgets"
            ).fetchall()}
        return {"daily": rows.get("daily"), "monthly": rows.get("monthly")}

    def spend(self, *, window: str) -> float:
        """Sum cost_usd over the given window ('daily' or 'monthly')."""
        days = 1 if window == "daily" else 30
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0.0) AS s FROM calls "
                "WHERE timestamp >= datetime('now', ?)",
                (f"-{days} days",),
            ).fetchone()
        return round(row["s"], 6)

    def budget_status(self) -> dict:
        """Return current spend vs configured budgets + a warning if crossed."""
        budgets = self.get_budgets()
        out: dict = {"daily": None, "monthly": None}
        for k in ("daily", "monthly"):
            limit = budgets[k]
            if limit is not None:
                spent = self.spend(window=k)
                out[k] = {
                    "spent": spent,
                    "limit": limit,
                    "pct": round(spent / limit, 3) if limit > 0 else 0.0,
                    "over": spent >= limit,
                }
        return out

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

    def trend(self, days: int = 30, *, by_model: bool = False) -> list[dict]:
        """Per-day (or per-day-per-model) trend over the last `days` days.

        Returns a list of dicts ordered by date ascending, each with:
            date, model (only if by_model), calls, prompt_tokens, cached_tokens,
            cost_usd, cache_hit_rate, avg_latency_ms.

        The cache_hit_rate is the day's cached_tokens / prompt_tokens (0 if no
        prompt tokens). This is the "did our reorder ship and stick?" curve.
        """
        with self._connect() as conn:
            group = "date(timestamp), model" if by_model else "date(timestamp)"
            model_select = "model," if by_model else ""
            rows = conn.execute(
                f"""
                SELECT
                    date(timestamp) AS day,
                    {model_select}
                    COUNT(*)                       AS calls,
                    SUM(prompt_tokens)             AS prompt_tokens,
                    SUM(cached_tokens)             AS cached_tokens,
                    SUM(cost_usd)                  AS cost_usd,
                    AVG(latency_ms)                AS avg_latency_ms
                FROM calls
                WHERE timestamp >= datetime('now', ?)
                GROUP BY {group}
                ORDER BY day ASC
                """,
                (f"-{days} days",),
            ).fetchall()
            out = []
            for r in rows:
                d = {
                    "date": r["day"],
                    "calls": r["calls"],
                    "prompt_tokens": r["prompt_tokens"] or 0,
                    "cached_tokens": r["cached_tokens"] or 0,
                    "cost_usd": round(r["cost_usd"] or 0.0, 6),
                    "avg_latency_ms": round(r["avg_latency_ms"] or 0.0, 2),
                }
                if by_model:
                    d["model"] = r["model"]
                p = d["prompt_tokens"]
                d["cache_hit_rate"] = round(d["cached_tokens"] / p, 3) if p > 0 else 0.0
                out.append(d)
            return out

    def export(self, *, fmt: str = "csv", path: Path, days: int | None = None) -> Path:
        """Export logged calls to a CSV or JSONL file.

        `days=None` exports all rows; otherwise limits to the last N days.
        Metadata and section_order are serialized as JSON strings in the output.
        Returns the written path.
        """
        import csv as csvlib
        with self._connect() as conn:
            if days is not None:
                rows = conn.execute(
                    "SELECT * FROM calls WHERE timestamp >= datetime('now', ?) ORDER BY id ASC",
                    (f"-{days} days",),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM calls ORDER BY id ASC").fetchall()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["id", "timestamp", "model", "prompt_tokens", "completion_tokens",
                      "cached_tokens", "cost_usd", "latency_ms", "prompt_hash",
                      "section_order", "metadata"]

        if fmt == "jsonl":
            import json as json_mod
            with path.open("w") as f:
                for r in rows:
                    d = dict(r)
                    d["section_order"] = json.loads(d.get("section_order") or "[]")
                    d["metadata"] = json.loads(d.get("metadata") or "{}")
                    f.write(json_mod.dumps(d) + "\n")
        else:  # csv
            with path.open("w", newline="") as f:
                writer = csvlib.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in rows:
                    writer.writerow(dict(r))
        return path


@contextmanager
def run(db_path: Optional[Path] = None) -> Iterator[Logger]:
    """Context manager for ad-hoc logging. Mostly here for future expansion."""
    logger = Logger(db_path)
    try:
        yield logger
    finally:
        pass