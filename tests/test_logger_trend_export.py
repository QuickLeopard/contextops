"""Tests for Logger.trend() and Logger.export()."""

import csv
import json
from datetime import datetime, timezone

from contextops.logger import Logger
from contextops.models import CallLog


def _entry(*, model="gpt-4o", prompt=1000, cached=0, cost=0.01, latency=200.0, minutes_ago=0):
    from datetime import timedelta
    ts = (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()
    return CallLog(
        timestamp=ts, model=model,
        prompt_tokens=prompt, completion_tokens=10,
        cached_tokens=cached, cost_usd=cost, latency_ms=latency,
        prompt_hash="", section_order=["system", "query"], metadata={},
    )


def test_trend_groups_by_day(tmp_path):
    log = Logger(tmp_path / "t.db")
    # Two calls today with different cache ratios.
    log.log(_entry(prompt=1000, cached=100, cost=0.02))
    log.log(_entry(prompt=1000, cached=500, cost=0.01))
    rows = log.trend(days=1)
    assert len(rows) == 1  # both today → one day bucket
    r = rows[0]
    assert r["calls"] == 2
    assert r["prompt_tokens"] == 2000
    assert r["cached_tokens"] == 600
    assert r["cache_hit_rate"] == round(600 / 2000, 3)  # 0.3
    assert r["cost_usd"] == round(0.03, 6)


def test_trend_by_model(tmp_path):
    log = Logger(tmp_path / "t.db")
    log.log(_entry(model="gpt-4o", prompt=100, cached=10))
    log.log(_entry(model="claude-haiku-4.5", prompt=200, cached=20))
    rows = log.trend(days=1, by_model=True)
    assert len(rows) == 2  # one row per (day, model)
    models = {r["model"] for r in rows}
    assert models == {"gpt-4o", "claude-haiku-4.5"}


def test_trend_empty_window(tmp_path):
    log = Logger(tmp_path / "t.db")
    rows = log.trend(days=7)
    assert rows == []


def test_trend_cache_hit_rate_zero_when_no_prompt_tokens(tmp_path):
    log = Logger(tmp_path / "t.db")
    log.log(_entry(prompt=0, cached=0))  # edge: zero prompt tokens
    rows = log.trend(days=1)
    assert rows[0]["cache_hit_rate"] == 0.0


def test_export_csv(tmp_path):
    log = Logger(tmp_path / "t.db")
    log.log(_entry(prompt=500, cached=50, cost=0.005))
    out = tmp_path / "calls.csv"
    log.export(fmt="csv", path=out)
    assert out.exists()
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["prompt_tokens"] == "500"
    assert rows[0]["model"] == "gpt-4o"


def test_export_jsonl(tmp_path):
    log = Logger(tmp_path / "t.db")
    log.log(_entry(prompt=500, cached=50, cost=0.005))
    out = tmp_path / "calls.jsonl"
    log.export(fmt="jsonl", path=out)
    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    d = json.loads(lines[0])
    assert d["prompt_tokens"] == 500
    # metadata/section_order are parsed back into structured form in JSONL.
    assert isinstance(d["metadata"], dict)
    assert isinstance(d["section_order"], list)


def test_export_since_days_filters(tmp_path):
    log = Logger(tmp_path / "t.db")
    # An old call (well outside the 7-day window).
    log.log(_entry(prompt=999, minutes_ago=60 * 24 * 30))  # 30 days ago
    log.log(_entry(prompt=100, minutes_ago=10))  # recent
    out = tmp_path / "calls.csv"
    log.export(fmt="csv", path=out, days=7)
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1  # only the recent call
    assert rows[0]["prompt_tokens"] == "100"
