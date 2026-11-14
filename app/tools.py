"""The tools: plain Python functions that look things up in the regulations.

They are written once and used in two places:
  - app/agent.py      gives them to the LLM, which decides when to call them
  - app/mcp_server.py exposes the same functions to any MCP client (Claude Desktop, IDEs, ...)

Each tool takes simple arguments and returns plain dicts/lists, so the result
can be turned into JSON for either caller.
"""

from app import db
from app.config import TOP_K
from app.embeddings import embed_query

MAX_TOP_K = 10


def _clean(row: dict) -> dict:
    """Make a database row JSON-friendly."""
    row = dict(row)
    if "similarity" in row:
        row["similarity"] = round(float(row["similarity"]), 4)
    return row


def search_regulations(
    query: str,
    language: str | None = None,
    document: str | None = None,
    top_k: int = TOP_K,
) -> list[dict]:
    """Find the clauses most relevant to a question, by meaning (vector search).

    query:    what to look for, in Arabic or English.
    language: "en" or "ar" to search only that language; leave empty for both.
    document: a document code such as "OSHJ-CoP-04" to search only that document.
    top_k:    how many clauses to return (1-10).
    """
    top_k = max(1, min(int(top_k), MAX_TOP_K))
    with db.connect() as conn:
        rows = db.search(conn, embed_query(query), top_k, language or None, document or None)
    return [_clean(row) for row in rows]


def get_clause(document: str, clause: str, language: str = "en") -> list[dict]:
    """Fetch the full text of one clause by its number, for example OSHJ-CoP-04 clause 5.4.1.

    Returns a list because a long clause is stored as several consecutive pieces.
    An empty list means there is no such clause.
    """
    with db.connect() as conn:
        return [_clean(row) for row in db.get_clause(conn, document.strip(), clause.strip(), language)]


def list_documents() -> list[dict]:
    """List the regulation documents that are available, with their languages."""
    with db.connect() as conn:
        return [_clean(row) for row in db.list_documents(conn)]
