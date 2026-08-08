"""Tests for the contextops CLI."""

import tempfile
from pathlib import Path

from click.testing import CliRunner

from contextops.access import AccessDecision
from contextops.cli import main
from contextops.logger import Logger
import contextops.logger as logger_module


def _patch_db(tmp: str):
    db_path = Path(tmp) / "calls.db"
    original = logger_module.DEFAULT_DB_PATH
    logger_module.DEFAULT_DB_PATH = db_path
    return original, db_path


def test_optimize_redacts_missing_role():
    with tempfile.TemporaryDirectory() as tmp:
        original, db_path = _patch_db(tmp)
        try:
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "optimize",
                    "--context",
                    "public context",
                    "--documents",
                    "secret executive documents",
                    "--access-tags",
                    '{"documents": ["executive"]}',
                    "--principal-role",
                    "support",
                ],
            )
            assert result.exit_code == 0
            assert "Redacted" in result.output

            logger = Logger(db_path)
            rows = logger.audit_query(principal_id="anonymous")
            assert len(rows) == 2  # context included, documents redacted
            actions = {r["section"]: r["action"] for r in rows}
            assert actions["context"] == "included"
            assert actions["documents"] == "redacted"
        finally:
            logger_module.DEFAULT_DB_PATH = original


def test_optimize_allows_matching_role():
    with tempfile.TemporaryDirectory() as tmp:
        original, db_path = _patch_db(tmp)
        try:
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "optimize",
                    "--context",
                    "public context",
                    "--documents",
                    "secret executive documents",
                    "--access-tags",
                    '{"documents": ["executive"]}',
                    "--principal-role",
                    "executive,support",
                ],
            )
            assert result.exit_code == 0
            assert "Redacted" not in result.output

            logger = Logger(db_path)
            rows = logger.audit_query()
            assert all(r["action"] == "included" for r in rows)
        finally:
            logger_module.DEFAULT_DB_PATH = original


def test_audit_renders_decisions():
    with tempfile.TemporaryDirectory() as tmp:
        original, db_path = _patch_db(tmp)
        try:
            logger = Logger(db_path)
            logger.log_access(
                [
                    AccessDecision(
                        principal_id="alice",
                        section="documents",
                        action="redacted",
                        reason="missing required roles: executive",
                        content_hash="deadbeef" * 4,
                    ),
                    AccessDecision(
                        principal_id="bob",
                        section="query",
                        action="included",
                        reason="role allowed",
                        content_hash="cafebabe" * 4,
                    ),
                ]
            )

            runner = CliRunner()
            result = runner.invoke(main, ["audit"])
            assert result.exit_code == 0
            assert "redacted" in result.output
            assert "included" in result.output
            assert "alice" in result.output
            assert "bob" in result.output

            filtered = runner.invoke(main, ["audit", "--principal", "alice"])
            assert filtered.exit_code == 0
            assert "bob" not in filtered.output
            assert "alice" in filtered.output
        finally:
            logger_module.DEFAULT_DB_PATH = original
