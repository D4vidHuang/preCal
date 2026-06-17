"""Abstract Embedder interface shared by every engine.

Contract (spec decision `model`): Qwen3-Embedding family, last-token pooling,
L2-normalized vectors, cosine via inner product. The QUERY side gets the
Instruct/Query wrapper; documents/code are encoded RAW (no prefix).

    Instruct: {task_description}\nQuery: {query}

Every concrete engine (TEI, Infinity, vLLM, sentence-transformers) implements
``embed_documents`` and ``embed_queries`` and returns an ``np.ndarray`` of shape
``[N, embed_dim]`` in float32 (the canonical disk dtype), already pooled +
normalized per the model config. The base class provides the shared query-wrap
and L2-normalize helpers so engines don't reimplement them inconsistently.
"""

from __future__ import annotations

import abc
from typing import List, Sequence

import numpy as np


def build_query(task_description: str, query: str) -> str:
    """Apply the Qwen3-Embedding Instruct/Query wrapper to a single query."""
    return f"Instruct: {task_description}\nQuery: {query}"


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize (safe against zero rows)."""
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class Embedder(abc.ABC):
    """Abstract embedding engine."""

    def __init__(
        self,
        model_id: str,
        *,
        embed_dim: int,
        dtype: str,
        pooling: str = "last_token",
        normalize: bool = True,
        query_instruction: str = "",
        batch_size: int = 32,
    ) -> None:
        self.model_id = model_id
        self.embed_dim = embed_dim
        self.dtype = dtype
        self.pooling = pooling
        self.normalize = normalize
        self.query_instruction = query_instruction
        self.batch_size = batch_size

    # ----- subclasses implement these two -------------------------------- #
    @abc.abstractmethod
    def _encode(self, texts: Sequence[str], is_query: bool) -> np.ndarray:
        """Encode a batch of texts -> [N, embed_dim] float32 (pooled).

        Implementations should NOT apply the query wrapper themselves; the base
        ``embed_queries`` already wraps. Implementations MAY normalize; the base
        re-normalizes to enforce the contract regardless.
        """
        raise NotImplementedError

    # ----- public API ----------------------------------------------------- #
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed code/documents RAW (no instruction prefix)."""
        vecs = self._encode(list(texts), is_query=False)
        return self._postprocess(vecs)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Embed NL queries with the Instruct/Query wrapper applied."""
        wrapped = [build_query(self.query_instruction, q) for q in texts]
        vecs = self._encode(wrapped, is_query=True)
        return self._postprocess(vecs)

    # ----- shared post-processing ----------------------------------------- #
    def _postprocess(self, vecs: np.ndarray) -> np.ndarray:
        """Apply MRL truncation (if embed_dim < native) + L2 normalize.

        Matryoshka (MRL) truncation = take the first ``embed_dim`` dims then
        renormalize, which is how Qwen3-Embedding supports sub-dim outputs.
        """
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim != 2:
            raise ValueError(f"Engine returned {vecs.ndim}-d array; expected 2-d.")
        if vecs.shape[1] > self.embed_dim:
            vecs = vecs[:, : self.embed_dim]
        if self.normalize:
            vecs = l2_normalize(vecs)
        return np.ascontiguousarray(vecs, dtype=np.float32)

    def close(self) -> None:
        """Release any resources (HTTP session, model handle). Override as needed."""
        return None
