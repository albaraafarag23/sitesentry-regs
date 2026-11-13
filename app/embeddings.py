"""Turn text into vectors with a Hugging Face multilingual model.

An embedding is a list of numbers (here 384 of them) that represents the
*meaning* of a text. Texts with similar meaning get vectors that point in a
similar direction, even across languages. That is what lets an Arabic
question find an English clause.

The E5 models were trained with a prefix on every input:
  "query: ..."   for the question being asked
  "passage: ..." for the documents being searched
Leaving the prefixes out makes retrieval noticeably worse, so we add them here
and nowhere else.
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the model once and reuse it (loading takes a few seconds)."""
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed document chunks for storage."""
    vectors = get_model().encode(
        [f"passage: {t}" for t in texts],
        normalize_embeddings=True,  # length 1, so cosine similarity is well behaved
        batch_size=16,
    )
    return vectors.tolist()


def embed_query(question: str) -> list[float]:
    """Embed a user question for searching."""
    vector = get_model().encode(f"query: {question}", normalize_embeddings=True)
    return vector.tolist()
