"""Tests for the ONNX embedding path.

The first test needs nothing. The others download the models on first run
(about 600 MB), so CI runs them only in the integration job.
"""

import numpy as np

from app import embeddings_onnx
from app.embeddings_onnx import BOS, EOS, MAX_TOKENS, UNK, token_ids


class FakeSentencePiece:
    """Stands in for the real tokenizer: returns the piece ids it was given."""

    def __init__(self, pieces):
        self.pieces = pieces

    def encode(self, text):
        return self.pieces


def test_token_ids_shift_by_one_and_add_start_and_end_tokens():
    # SentencePiece id 0 means "unknown"; every other id is shifted by one in the model's vocabulary.
    assert token_ids(FakeSentencePiece([5, 0, 9]), "any text") == [BOS, 6, UNK, 10, EOS]


def test_token_ids_are_cut_to_the_model_limit():
    ids = token_ids(FakeSentencePiece(list(range(1, 2000))), "a very long text")
    assert len(ids) == MAX_TOKENS
    assert ids[0] == BOS and ids[-1] == EOS


def test_onnx_vectors_are_normalized_and_do_not_depend_on_the_batch():
    short = "query: guard rail height"
    long = "query: what must be inspected before using work at height equipment for the first time"
    alone = embeddings_onnx.embed([short])[0]
    in_batch = embeddings_onnx.embed([short, long])[0]      # same text, embedded next to a longer one
    assert alone.shape == (384,)
    assert abs(np.linalg.norm(alone) - 1) < 1e-4
    assert float(alone @ in_batch) > 0.99999


def test_onnx_model_agrees_with_the_pytorch_model():
    from app.embeddings import get_model

    texts = ["query: How high must a guard rail be?", "query: ما هي قواعد استخدام السلالم؟"]
    onnx_vectors = embeddings_onnx.embed(texts)
    torch_vectors = get_model().encode(texts, normalize_embeddings=True)
    for onnx_vector, torch_vector in zip(onnx_vectors, torch_vectors):
        assert float(onnx_vector @ torch_vector) > 0.98     # quantized model: nearly the same direction
