"""The "retrieval" half of RAG: question -> vector -> nearest chunks."""

from app import db
from app.config import TOP_K
from app.embeddings import embed_query


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """Return the top_k chunks most similar in meaning to the question."""
    query_embedding = embed_query(question)
    with db.connect() as conn:
        return db.search(conn, query_embedding, top_k)
