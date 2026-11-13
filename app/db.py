"""Database access: store chunks and search them by vector similarity."""

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from app.config import DATABASE_URL


def connect() -> psycopg.Connection:
    # connect_timeout: fail after 5 seconds if the database is not running, instead of hanging.
    conn = psycopg.connect(DATABASE_URL, row_factory=dict_row, connect_timeout=5)
    register_vector(conn)  # teaches psycopg how to send/receive the "vector" type
    return conn


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


def search(conn: psycopg.Connection, query_embedding: list[float], top_k: int) -> list[dict]:
    """Return the top_k chunks closest in meaning to the query.

    `<=>` is pgvector's cosine *distance* operator: 0 = same direction, 2 = opposite.
    We report similarity = 1 - distance, where higher is better.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT document, doc_title, language, clause, heading, page, content,
                   1 - (embedding <=> %(q)s) AS similarity
            FROM chunks
            ORDER BY embedding <=> %(q)s
            LIMIT %(k)s
            """,
            {"q": Vector(query_embedding), "k": top_k},
        )
        return cur.fetchall()
