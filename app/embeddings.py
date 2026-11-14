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

Two ways to run the same model (EMBEDDING_BACKEND in .env):
  torch  sentence-transformers + PyTorch. Full precision. Used for ingestion.
  onnx   ONNX Runtime with the 8-bit quantized model. Small and light; used by
         the deployed API, which only has to embed one question at a time.
"""

from functools import lru_cache

from app.config import EMBEDDING_BACKEND, EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_model():
    """Load the PyTorch model once and reuse it (loading takes a few seconds)."""
    from sentence_transformers import SentenceTransformer   # imported here: not installed on the slim server

    return SentenceTransformer(EMBEDDING_MODEL)


def _embed(texts: list[str]) -> list[list[float]]:
    if EMBEDDING_BACKEND == "onnx":
        from app.embeddings_onnx import embed

        return embed(texts).tolist()
    vectors = get_model().encode(
        texts,
        normalize_embeddings=True,  # length 1, so cosine similarity is well behaved
        batch_size=16,
    )
    return vectors.tolist()


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed document chunks for storage."""
    return _embed([f"passage: {t}" for t in texts])


def embed_query(question: str) -> list[float]:
    """Embed a user question for searching."""
    return _embed([f"query: {question}"])[0]
