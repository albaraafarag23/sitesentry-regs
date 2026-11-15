"""The same embedding model, run with ONNX Runtime instead of PyTorch.

Why: PyTorch needs about 1 GB of memory and a multi-GB install, too much for a
free cloud server (512 MB). ONNX Runtime runs the exported, 8-bit quantized
version of the same model (118 MB file) in a fraction of that.

What sentence-transformers normally does for us, done by hand here:
  1. tokenize   text -> token ids (the model's vocabulary numbers)
  2. run model  token ids -> one vector per token
  3. mean-pool  average the token vectors (ignoring padding) -> one vector per text
  4. normalize  scale to length 1, so cosine similarity is well behaved

Tokenizer note: the usual `tokenizers` library needs about 270 MB of memory for
this model's 250,000-entry vocabulary. The model's native SentencePiece file
produces identical token ids in about 50 MB, so we use that directly.
"""

from functools import lru_cache

import numpy as np
import onnxruntime
import sentencepiece
from huggingface_hub import hf_hub_download

from app.config import EMBEDDING_MODEL, ONNX_MODEL_FILE, ONNX_THREADS

MAX_TOKENS = 512                 # the model's limit, including the two special tokens
BOS, EOS, UNK = 0, 2, 3  # special token ids of XLM-RoBERTa, the model E5 is built on


@lru_cache(maxsize=1)
def _load() -> tuple[sentencepiece.SentencePieceProcessor, onnxruntime.InferenceSession]:
    """Download (once, then cached on disk) and load the tokenizer and the ONNX model."""
    tokenizer = sentencepiece.SentencePieceProcessor(
        model_file=hf_hub_download(EMBEDDING_MODEL, "sentencepiece.bpe.model")
    )
    options = onnxruntime.SessionOptions()
    options.enable_cpu_mem_arena = False     # do not keep a large memory pool around
    if ONNX_THREADS:                         # leave the other processor cores to whatever shares the machine
        options.intra_op_num_threads = ONNX_THREADS
    session = onnxruntime.InferenceSession(
        hf_hub_download(EMBEDDING_MODEL, ONNX_MODEL_FILE),
        options,
        providers=["CPUExecutionProvider"],
    )
    return tokenizer, session


def token_ids(tokenizer: sentencepiece.SentencePieceProcessor, text: str) -> list[int]:
    """Text -> model token ids: <s> pieces... </s>.

    The model's vocabulary is the SentencePiece vocabulary shifted by one
    (and SentencePiece's "unknown" id 0 maps to the model's <unk> id 3).
    """
    pieces = [piece + 1 if piece else UNK for piece in tokenizer.encode(text)]
    return [BOS] + pieces[: MAX_TOKENS - 2] + [EOS]


def embed(texts: list[str]) -> np.ndarray:
    """Return one normalized vector per text, shape (len(texts), 384).

    Each text is run on its own. Batching would need padding, and the quantized
    model's output shifts slightly (about 0.3%) when padding is present; one
    text at a time keeps the result identical however the function is called.
    """
    tokenizer, session = _load()
    needs_token_types = any(i.name == "token_type_ids" for i in session.get_inputs())
    vectors = []
    for text in texts:
        ids = np.array([token_ids(tokenizer, text)], dtype=np.int64)        # shape (1, tokens)
        inputs = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
        if needs_token_types:
            inputs["token_type_ids"] = np.zeros_like(ids)
        token_vectors = session.run(None, inputs)[0][0]                     # (tokens, 384)
        pooled = token_vectors.mean(axis=0)                                 # mean over the tokens
        vectors.append(pooled / np.linalg.norm(pooled))
    return np.array(vectors)
