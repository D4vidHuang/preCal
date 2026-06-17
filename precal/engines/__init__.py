"""Engine registry + factory keyed on ``engine.name``.

Engines are constructed lazily so importing this package never pulls torch /
vLLM / TEI-client deps. ``build_engine(cfg, ...)`` returns a concrete
:class:`precal.engines.base.Embedder` for the configured engine.

Supported engine.name values (spec decision `engine`):
  * "tei"                  -> TEIEmbedder (production, fp16, Blackwell sm_120)
  * "infinity"             -> InfinityEmbedder (fallback, bf16, in-process)
  * "vllm"                 -> VLLMEmbedder (fallback, optional fp8)
  * "sentence_transformers"-> STEmbedder (reference only, for eval ground-truth)
"""

from __future__ import annotations

from typing import Optional

from precal.config import Config
from precal.engines.base import Embedder
from precal.utils import get_logger

logger = get_logger("precal.engines")

# Public re-exports.
from precal.engines.base import build_query, l2_normalize  # noqa: E402,F401

__all__ = ["Embedder", "build_engine", "build_query", "l2_normalize"]


def build_engine(
    cfg: Config,
    *,
    base_port: int = 7997,
    tei_replicas: Optional[int] = None,
    allow_fp8: bool = False,
) -> Embedder:
    """Instantiate the engine selected by ``cfg.engine.name``."""
    name = cfg.engine.name
    common = dict(
        embed_dim=cfg.model.embed_dim,
        dtype=cfg.model.dtype,
        pooling=cfg.model.pooling,
        normalize=cfg.model.normalize,
        query_instruction=cfg.model.query_instruction,
        batch_size=cfg.engine.batch_size,
    )

    if name == "tei":
        from precal.engines.tei import TEIEmbedder

        return TEIEmbedder(
            cfg.model.id,
            base_port=base_port,
            replicas=tei_replicas or cfg.engine.replicas_per_gpu,
            **common,
        )
    if name == "infinity":
        from precal.engines.infinity import InfinityEmbedder

        return InfinityEmbedder(cfg.model.id, **common)
    if name == "vllm":
        from precal.engines.vllm_pooling import VLLMEmbedder

        return VLLMEmbedder(cfg.model.id, allow_fp8=allow_fp8, **common)
    if name == "sentence_transformers":
        from precal.engines.sentence_transformers import STEmbedder

        return STEmbedder(cfg.model.id, **common)

    raise ValueError(
        f"Unknown engine.name={name!r}. "
        f"Expected one of: tei | infinity | vllm | sentence_transformers."
    )
