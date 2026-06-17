"""Fallback engine: michaelfeil/infinity embedded AsyncEmbeddingEngine.

Used when TEI is unavailable or when bf16 vectors are wanted (Infinity runs the
model in-process and supports bf16, plus gives per-batch control useful for the
checkpoint cadence). It loads the model strictly from HF_HOME on compute nodes
(offline). Pooling/normalization are enforced by the base class.

All heavy imports (infinity_emb, torch) are lazy so the module imports clean.
"""

from __future__ import annotations

import asyncio
from typing import Sequence

import numpy as np

from precal.engines.base import Embedder
from precal.utils import get_logger

logger = get_logger("precal.engines.infinity")


class InfinityEmbedder(Embedder):
    """In-process embedded engine using infinity_emb's AsyncEmbeddingEngine."""

    def __init__(
        self,
        model_id: str,
        *,
        embed_dim: int,
        dtype: str = "bfloat16",
        pooling: str = "last_token",
        normalize: bool = True,
        query_instruction: str = "",
        batch_size: int = 32,
        device: str = "cuda",
    ) -> None:
        super().__init__(
            model_id,
            embed_dim=embed_dim,
            dtype=dtype,
            pooling=pooling,
            normalize=normalize,
            query_instruction=query_instruction,
            batch_size=batch_size,
        )
        self.device = device
        self._engine = None
        self._loop = None

    def _ensure_engine(self):
        if self._engine is not None:
            return
        try:
            from infinity_emb import AsyncEngineArray, EngineArgs  # lazy
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "engine.name=infinity requires `infinity-emb` "
                "(pip install 'infinity-emb[all]')."
            ) from exc

        # Map our dtype string to infinity's expected value.
        dtype = "bfloat16" if self.dtype == "bfloat16" else "float16"
        args = EngineArgs(
            model_name_or_path=self.model_id,
            engine="torch",
            dtype=dtype,
            device=self.device,
            batch_size=self.batch_size,
            pooling_method="mean" if self.pooling != "last_token" else "auto",
            # infinity reads the pooling from the model config when "auto".
        )
        self._array = AsyncEngineArray.from_args([args])
        self._engine = self._array[0]
        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(self._engine.astart())
        logger.info("Infinity engine started for %s (dtype=%s)", self.model_id, dtype)

    def _encode(self, texts: Sequence[str], is_query: bool) -> np.ndarray:
        self._ensure_engine()
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.embed_dim), dtype=np.float32)

        async def _run():
            embeddings, _usage = await self._engine.embed(sentences=texts)
            return embeddings

        result = self._loop.run_until_complete(_run())
        return np.asarray(result, dtype=np.float32)

    def close(self) -> None:
        if self._engine is not None and self._loop is not None:
            try:
                self._loop.run_until_complete(self._engine.astop())
            except Exception:
                pass
            self._loop.close()
            self._engine = None
            self._loop = None
