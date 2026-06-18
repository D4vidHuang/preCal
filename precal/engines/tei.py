"""TEI (text-embeddings-inference) HTTP client engine.

Production engine for the Blackwell (sm_120) GPUs via the prebuilt
``ghcr.io/huggingface/text-embeddings-inference:120-1.9`` image. TEI replicas
are launched out-of-band (scripts/launch_tei_replicas.sh, owner=ops) and listen
on consecutive ports starting at ``base_port`` (default 7997); this client
round-robins requests across the discovered replicas for throughput.

Key TEI facts encoded here (spec decision `dtype`):
  * TEI exposes only float16|float32 -> vectors are fp16 when produced by TEI;
    the embed stage records dtype="float16" in the row.
  * TEI applies pooling server-side per the model card (last_token for Qwen3),
    so the client just POSTs raw text to ``/embed`` and receives pooled vectors.
  * The query wrapper is applied client-side (base class) so documents stay raw.

``requests`` is imported lazily; the engine constructs cleanly without it and
fails loudly only when an embed call is actually made.
"""

from __future__ import annotations

import itertools
import os
from typing import List, Optional, Sequence

import numpy as np

from precal.engines.base import Embedder
from precal.utils import get_logger

logger = get_logger("precal.engines.tei")


class TEIEmbedder(Embedder):
    """HTTP client for one or more TEI replicas."""

    def __init__(
        self,
        model_id: str,
        *,
        embed_dim: int,
        dtype: str = "float16",
        pooling: str = "last_token",
        normalize: bool = True,
        query_instruction: str = "",
        batch_size: int = 32,
        base_port: int = 7997,
        replicas: int = 1,
        host: str = "127.0.0.1",
        timeout: float = 120.0,
        endpoints: Optional[List[str]] = None,
    ) -> None:
        # TEI only supports fp16/fp32; coerce the recorded dtype accordingly.
        if dtype not in ("float16", "float32"):
            logger.info("TEI supports only float16|float32; coercing dtype -> float16.")
            dtype = "float16"
        super().__init__(
            model_id,
            embed_dim=embed_dim,
            dtype=dtype,
            pooling=pooling,
            normalize=normalize,
            query_instruction=query_instruction,
            batch_size=batch_size,
        )
        # Discover replica endpoints. Explicit `endpoints` override port math;
        # an env override (PRECAL_TEI_ENDPOINTS, comma-separated) is also honored.
        env_eps = os.environ.get("PRECAL_TEI_ENDPOINTS")
        if endpoints:
            self.endpoints = list(endpoints)
        elif env_eps:
            self.endpoints = [e.strip() for e in env_eps.split(",") if e.strip()]
        else:
            self.endpoints = [
                f"http://{host}:{base_port + i}" for i in range(max(1, replicas))
            ]
        self._rr = itertools.cycle(self.endpoints)
        self.timeout = timeout
        self._session = None
        logger.info("TEI client targeting %d replica(s): %s", len(self.endpoints), self.endpoints)

    def _get_session(self):
        if self._session is None:
            import requests  # lazy

            self._session = requests.Session()
        return self._session

    def _post_embed(self, texts: List[str]) -> np.ndarray:
        """POST one batch to the next replica's /embed endpoint."""
        import requests  # lazy (for exception types)

        endpoint = next(self._rr)
        url = f"{endpoint}/embed"
        # TEI rejects empty/whitespace inputs with 400; substitute a single space so
        # a degenerate item still yields a (throwaway) vector and the batch stays 1:1
        # with its rows. Callers should avoid empties; this is cheap insurance.
        texts = [t if (t and t.strip()) else " " for t in texts]
        # TEI accepts {"inputs": [...], "normalize": false}; we normalize
        # ourselves in the base class for a single source of truth.
        payload = {"inputs": texts, "normalize": False, "truncate": True}
        resp = self._get_session().post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        # TEI returns a list of vectors (list[list[float]]).
        return np.asarray(data, dtype=np.float32)

    def _encode(self, texts: Sequence[str], is_query: bool) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.embed_dim), dtype=np.float32)
        out: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            out.append(self._post_embed(batch))
        return np.vstack(out)

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
