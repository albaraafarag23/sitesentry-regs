"""Shared test helpers."""

import psycopg
import pytest

from app import db

_database_problem: str | None = None
_checked = False


@pytest.fixture
def ingested_database():
    """Skip the test (instead of failing) when the database is not running or is empty."""
    global _database_problem, _checked
    if not _checked:
        _checked = True
        try:
            db.ensure_schema()
            with db.connect() as conn:
                count = conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
            if count == 0:
                _database_problem = "no chunks in the database (run: python -m app.ingest)"
        except psycopg.Error:
            _database_problem = "database not reachable (run: docker compose up -d)"
    if _database_problem:
        pytest.skip(_database_problem)
