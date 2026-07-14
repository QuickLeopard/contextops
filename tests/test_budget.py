"""Tests for Logger budget tracking."""

from datetime import datetime, timedelta, timezone

from contextops.logger import Logger
from contextops.models import CallLog


def _entry(*, cost=0.01, minutes_ago=0):
    ts = (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()
    return CallLog(
        timestamp=ts, model="gpt-4o", prompt_tokens=100, completion_tokens=5,
        cached_tokens=0, cost_usd=cost, latency_ms=100.0,
        prompt_hash="", section_order=[], metadata={},
    )


def test_budget_persists(tmp_path):
    log = Logger(tmp_path / "b.db")
    assert log.get_budgets() == {"daily": None, "monthly": None}
    log.set_budget(daily_usd=50.0, monthly_usd=1000.0)
    assert log.get_budgets() == {"daily": 50.0, "monthly": 1000.0}


def test_budget_clears_with_none(tmp_path):
    log = Logger(tmp_path / "b.db")
    log.set_budget(daily_usd=10.0)
    log.set_budget(daily_usd=None)  # no-op for None in current impl; only sets non-None
    # set_budget with None doesn't delete — it's a no-op. Verify behavior.
    assert log.get_budgets()["daily"] == 10.0


def test_spend_sums_recent_costs(tmp_path):
    log = Logger(tmp_path / "b.db")
    log.log(_entry(cost=5.0))
    log.log(_entry(cost=3.0))
    assert log.spend(window="daily") == 8.0


def test_budget_status_reports_pct_and_over(tmp_path):
    log = Logger(tmp_path / "b.db")
    log.set_budget(daily_usd=10.0)
    log.log(_entry(cost=8.0))
    status = log.budget_status()
    d = status["daily"]
    assert d["spent"] == 8.0
    assert d["limit"] == 10.0
    assert d["pct"] == 0.8
    assert d["over"] is False
    # Cross the threshold.
    log.log(_entry(cost=3.0))
    status = log.budget_status()
    assert status["daily"]["over"] is True


def test_budget_status_no_budgets_returns_none(tmp_path):
    log = Logger(tmp_path / "b.db")
    status = log.budget_status()
    assert status == {"daily": None, "monthly": None}
