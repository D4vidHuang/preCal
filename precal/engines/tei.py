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

    def _post_embed(self, start_idx: int, texts: List[str]) -> np.ndarray:
        """POST one batch, starting at replica start_idx, FAILING OVER to the other
        replicas with backoff if one is down (a replica can OOM/crash mid-run)."""
        import time
        from requests.exceptions import RequestException
        # TEI rejects empty/whitespace inputs with 400; substitute a single space so
        # a degenerate item still yields a (throwaway) vector and the batch stays 1:1.
        texts = [t if (t and t.strip()) else " " for t in texts]
        payload = {"inputs": texts, "normalize": False, "truncate": True}
        n = len(self.endpoints)
        attempts = max(4, 2 * n)
        last = None
        for k in range(attempts):
            ep = self.endpoints[(start_idx + k) % n]   # round to a (hopefully live) replica
            try:
                resp = self._get_session().post(f"{ep}/embed", json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return np.asarray(resp.json(), dtype=np.float32)
            except RequestException as e:
                last = e
                time.sleep(min(2 ** k, 8))
        raise RuntimeError(f"TEI /embed failed after {attempts} attempts across {n} replica(s): {last}")

    def _encode(self, texts: Sequence[str], is_query: bool) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.embed_dim), dtype=np.float32)
        bs = self.batch_size
        batches = [texts[i : i + bs] for i in range(0, len(texts), bs)]
        n_ep = len(self.endpoints)
        # Fan sub-batches across ALL replicas CONCURRENTLY (one in-flight POST per
        # replica), keeping every GPU replica busy. Endpoint chosen by index so the
        # threads don't race on a shared round-robin. Order preserved by index.
        def _work(item):
            idx, batch = item
            return self._post_embed(idx % n_ep, batch)   # start replica; fails over internally
        if n_ep > 1 and len(batches) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=n_ep) as ex:
                out = list(ex.map(_work, enumerate(batches)))
        else:
            out = [_work((i, b)) for i, b in enumerate(batches)]
        return np.vstack(out)

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
