"""Fallback engine: vLLM in pooling/embedding mode (LLM.embed).

Used as a high-throughput alternative when a vLLM build with sm_120 kernels is
available. Optional fp8 (E4M3) is gated behind a validate-first flag
(``allow_fp8``) per the spec's `dtype` decision: fp8 must pass the bf16-vs-fp8
nDCG delta check on the held-out split before any full run, so this engine
refuses fp8 unless explicitly allowed.

vLLM / torch imports are lazy. The query wrapper is applied by the base class.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from precal.engines.base import Embedder
from precal.utils import get_logger

logger = get_logger("precal.engines.vllm")


class VLLMEmbedder(Embedder):
    """vLLM pooling-runner embedding engine."""

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
        allow_fp8: bool = False,
        max_model_len: int = 8192,
    ) -> None:
        if dtype == "fp8" and not allow_fp8:
            raise ValueError(
                "fp8 requested but allow_fp8 is False. fp8 must first pass the "
                "bf16-vs-fp8 nDCG delta check on the held-out split (block if "
                ">0.3-0.5pt). Set allow_fp8=True only after validation."
            )
        super().__init__(
            model_id,
            embed_dim=embed_dim,
            dtype=dtype,
            pooling=pooling,
            normalize=normalize,
            query_instruction=query_instruction,
            batch_size=batch_size,
        )
        self.max_model_len = max_model_len
        self._llm = None

    def _ensure_llm(self):
        if self._llm is not None:
            return
        try:
            from vllm import LLM  # lazy
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "engine.name=vllm requires `vllm` with a pooling-capable build."
            ) from exc

        # vLLM dtype strings: 'bfloat16'|'float16'|'fp8'.
        self._llm = LLM(
            model=self.model_id,
            runner="pooling",
            dtype="bfloat16" if self.dtype == "bfloat16" else self.dtype,
            max_model_len=self.max_model_len,
            enforce_eager=False,
        )
        logger.info("vLLM pooling engine ready for %s (dtype=%s)", self.model_id, self.dtype)

    def _encode(self, texts: Sequence[str], is_query: bool) -> np.ndarray:
        self._ensure_llm()
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.embed_dim), dtype=np.float32)
        outputs = self._llm.embed(texts)
        vecs = [o.outputs.embedding for o in outputs]
        return np.asarray(vecs, dtype=np.float32)
