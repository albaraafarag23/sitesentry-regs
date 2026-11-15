"""Database access: store chunks, search them by vector similarity, keep chat history."""

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from app.config import DATABASE_URL

# Columns returned for a chunk everywhere (everything except the embedding itself).
CHUNK_COLUMNS = "id, document, doc_title, language, clause, heading, page, content"

# The whole schema. Every statement is "IF NOT EXISTS", so it is safe to run at every start-up
# and works the same in Docker Compose, in CI and on a cloud database.
SCHEMA = """
-- pgvector adds the "vector" column type and the distance operators.
CREATE EXTENSION IF NOT EXISTS vector;

-- One row = one chunk of a regulation, with its clause number and its embedding.
CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    document    TEXT NOT NULL,          -- e.g. "OSHJ-CoP-04"
    doc_title   TEXT NOT NULL,
    language    TEXT NOT NULL,          -- "en" or "ar"
    clause      TEXT NOT NULL,          -- e.g. "5.4.1"
    heading     TEXT NOT NULL,
    page        INT  NOT NULL,          -- page where the clause starts
    content     TEXT NOT NULL,
    embedding   vector(384) NOT NULL    -- 384 = output size of multilingual-e5-small
);
-- HNSW index: finds nearest vectors fast without comparing against every row.
-- vector_cosine_ops = built for cosine distance (the <=> operator).
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);

-- Chat memory: one row per message of a conversation.
CREATE TABLE IF NOT EXISTS messages (
    id          BIGSERIAL PRIMARY KEY,
    session_id  TEXT NOT NULL,          -- one conversation
    role        TEXT NOT NULL,          -- "user" or "assistant"
    content     TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS messages_session_idx ON messages (session_id, id);
"""


def connect() -> psycopg.Connection:
    # connect_timeout: fail after 5 seconds if the database is not running, instead of hanging.
    conn = psycopg.connect(DATABASE_URL, row_factory=dict_row, connect_timeout=5)
    register_vector(conn)  # teaches psycopg how to send/receive the "vector" type
    return conn


def ensure_schema() -> None:
    """Create the extension and tables if they do not exist yet."""
    # A plain connection: register_vector() needs the extension, which may not exist yet.
    with psycopg.connect(DATABASE_URL, connect_timeout=5) as conn:
        conn.execute(SCHEMA)


# ---------------------------------------------------------------- chunks

def replace_document(conn: psycopg.Connection, document: str, language: str, rows: list[dict]) -> None:
    """Delete any old chunks for this document + language, then insert the new ones.

    This makes ingestion safe to run again: you never get duplicates.
    """
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM chunks WHERE document = %s AND language = %s",
            (document, language),
        )
        cur.executemany(
            """
            INSERT INTO chunks (document, doc_title, language, clause, heading, page, content, embedding)
            VALUES (%(document)s, %(doc_title)s, %(language)s, %(clause)s,
                    %(heading)s, %(page)s, %(content)s, %(embedding)s)
            """,
            [{**row, "embedding": Vector(row["embedding"])} for row in rows],
        )
    conn.commit()


def search(
    conn: psycopg.Connection,
    query_embedding: list[float],
    top_k: int,
    language: str | None = None,
    document: str | None = None,
) -> list[dict]:
    """Return the top_k chunks closest in meaning to the query.

    `<=>` is pgvector's cosine *distance* operator: 0 = same direction, 2 = opposite.
    We report similarity = 1 - distance, where higher is better.
    The optional filters are ordinary SQL: vectors and normal columns live in one table.
    """
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT {CHUNK_COLUMNS}, 1 - (embedding <=> %(q)s) AS similarity
            FROM chunks
            WHERE (%(language)s::text IS NULL OR language = %(language)s)
              AND (%(document)s::text IS NULL OR document = %(document)s)
            ORDER BY embedding <=> %(q)s
            LIMIT %(k)s
            """,
            {"q": Vector(query_embedding), "k": top_k, "language": language, "document": document},
        )
        return cur.fetchall()


def get_clause(conn: psycopg.Connection, document: str, clause: str, language: str) -> list[dict]:
    """Return every chunk of one clause, in reading order (a long clause has several)."""
    return conn.execute(
        f"SELECT {CHUNK_COLUMNS} FROM chunks "
        "WHERE document = %s AND clause = %s AND language = %s ORDER BY id",
        (document, clause, language),
    ).fetchall()


def list_documents(conn: psycopg.Connection) -> list[dict]:
    return conn.execute(
        "SELECT document, doc_title, language, count(*) AS chunks "
        "FROM chunks GROUP BY document, doc_title, language ORDER BY document, language"
    ).fetchall()


# ---------------------------------------------------------------- chat memory

def count_questions(conn: psycopg.Connection, seconds: int) -> int:
    """How many questions (user messages) all visitors asked in the last `seconds`."""
    return conn.execute(
        "SELECT count(*) FROM messages WHERE role = 'user' AND created_at > now() - make_interval(secs => %s)",
        (seconds,),
    ).fetchone()[0]


def add_message(conn: psycopg.Connection, session_id: str, role: str, content: str) -> None:
    conn.execute(
        "INSERT INTO messages (session_id, role, content) VALUES (%s, %s, %s)",
        (session_id, role, content),
    )


def get_history(conn: psycopg.Connection, session_id: str, limit: int) -> list[dict]:
    """Return the last `limit` messages of a conversation, oldest first."""
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id = %s ORDER BY id DESC LIMIT %s",
        (session_id, limit),
    ).fetchall()
    return rows[::-1]
