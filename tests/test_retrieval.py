"""Tests for retrieval. These need the database running and the PDFs ingested:

    docker compose up -d
    python -m app.ingest

If the database is not reachable, the tests are skipped instead of failing.
"""

import psycopg
import pytest

from app import db
from app.retrieval import retrieve

_skip_reason = ""


@pytest.fixture(scope="module", autouse=True)
def require_ingested_database():
    global _skip_reason
    if _skip_reason:
        pytest.skip(_skip_reason)   # already checked once; do not wait again
    try:
        with db.connect() as conn:
            count = conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
    except psycopg.Error:
        _skip_reason = "database not reachable (run: docker compose up -d)"
        pytest.skip(_skip_reason)
    if count == 0:
        pytest.skip("no chunks in the database (run: python -m app.ingest)")


def clauses(results: list[dict]) -> list[tuple[str, str]]:
    return [(r["document"], r["clause"]) for r in results]


def test_returns_top_k_results_best_first():
    results = retrieve("ladder inspection", top_k=3)
    assert len(results) == 3
    similarities = [r["similarity"] for r in results]
    assert similarities == sorted(similarities, reverse=True)


def test_english_question_finds_the_ladder_clause():
    results = retrieve("How should a portable ladder be secured?", top_k=5)
    assert ("OSHJ-CoP-04", "5.4.1") in clauses(results)


def test_question_about_guard_rails_finds_the_guard_rail_clause():
    results = retrieve("What are the requirements for guard rails?", top_k=5)
    assert ("OSHJ-CoP-04", "5.5.3") in clauses(results)


def test_arabic_question_finds_the_same_clause():
    # "What are the requirements for using ladders?"
    results = retrieve("ما هي متطلبات استخدام السلالم؟", top_k=5)
    assert ("OSHJ-CoP-04", "5.4.1") in clauses(results) or ("OSHJ-CoP-04", "5.4.2") in clauses(results)


def test_results_carry_everything_needed_for_a_citation():
    result = retrieve("site induction for new workers", top_k=1)[0]
    for field in ("document", "doc_title", "language", "clause", "heading", "page", "content"):
        assert result[field]
