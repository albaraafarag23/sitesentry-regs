"""The "retrieval" half of RAG: question -> vector -> nearest chunks."""

from app.config import TOP_K
from app.tools import search_regulations


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """Return the top_k chunks most similar in meaning to the question."""
    return search_regulations(question, top_k=top_k)
