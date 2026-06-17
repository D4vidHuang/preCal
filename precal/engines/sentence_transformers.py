"""Reference engine: sentence-transformers.

REFERENCE ONLY (spec decision `engine`): used to compute eval ground-truth
deltas / validate pooling against the model card example, never the production
path. It loads Qwen3-Embedding via SentenceTransformer, which applies the
correct last-token pooling and normalization for the model.

This engine is registered in the factory so `engine.name=sentence_transformers`
works for debugging and for recomputing self-consistent reference vectors. All
heavy imports are lazy.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from precal.engines.base import Embedder
from precal.utils import get_logger

logger = get_logger("precal.engines.st")


class STEmbedder(Embedder):
    """SentenceTransformer-backed reference embedder."""

    def __init__(self, model_id: str, **kwargs) -> None:
        super().__init__(model_id, **kwargs)
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            import torch  # lazy
            from sentence_transformers import SentenceTransformer  # lazy
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "engine.name=sentence_transformers requires "
                "`sentence-transformers` + `torch`."
            ) from exc

        torch_dtype = torch.bfloat16 if self.dtype == "bfloat16" else torch.float16
        self._model = SentenceTransformer(
            self.model_id,
            model_kwargs={"torch_dtype": torch_dtype},
        )

    def _encode(self, texts: Sequence[str], is_query: bool) -> np.ndarray:
        self._ensure_model()
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.embed_dim), dtype=np.float32)
        # We already wrapped queries in the base class, so pass raw here and let
        # the base class handle normalization for a single source of truth.
        vecs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=False,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)
