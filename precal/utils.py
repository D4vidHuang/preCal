"""Shared low-level helpers: logging, the chunk_id hash, tokenizer loading,
GPU pinning, and a couple of small filesystem/normalization utilities.

Heavy imports (transformers, torch) are done lazily inside the functions that
need them so this module imports clean on a login node / in CI.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sys
from functools import lru_cache
from typing import List, Optional

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "precal", level: Optional[str] = None) -> logging.Logger:
    """Return a configured logger.

    Level resolves from the ``PRECAL_LOG_LEVEL`` env var (default INFO) unless
    overridden by the ``level`` argument. Logs go to stderr so stdout stays
    clean for any machine-readable output.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
    resolved = (level or os.environ.get("PRECAL_LOG_LEVEL", "INFO")).upper()
    logger.setLevel(getattr(logging, resolved, logging.INFO))
    return logger


# --------------------------------------------------------------------------- #
# chunk_id hashing (blake2b)
# --------------------------------------------------------------------------- #
# We normalize whitespace before hashing so that purely cosmetic reformatting of
# the *same* span does not produce a different id. We DO include repo+path+span
# so identical code in two files yields distinct ids (provenance-bound key).
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Collapse runs of whitespace and strip ends. Model-independent."""
    return _WS_RE.sub(" ", text).strip()


def chunk_id(text: str, repo_name: str, path: str, start_line: int, end_line: int) -> str:
    """Stable primary key = blake2b hex of normalized text + provenance + span.

    16-byte (32 hex char) digest: collision-safe at tens of billions of chunks
    while keeping the id compact for parquet + FAISS id mapping.
    """
    h = hashlib.blake2b(digest_size=16)
    norm = normalize_text(text)
    payload = f"{norm}\x00{repo_name}\x00{path}\x00{start_line}\x00{end_line}"
    h.update(payload.encode("utf-8", errors="replace"))
    return h.hexdigest()


def chunk_id_to_int64(cid: str) -> int:
    """Map a chunk_id to a non-negative int64 FAISS id (for IDMap).

    chunk_ids produced by :func:`chunk_id` are blake2b hex, so we take the top
    64 bits of the hex digest and clear the sign bit (FAISS ids must be
    non-negative). For any non-hex id (defensive: e.g. externally supplied ids)
    we fall back to a blake2b of the raw string so the mapping never crashes and
    stays deterministic. Collisions across tens of millions of vectors are
    negligible at 63 bits.
    """
    try:
        val = int(cid[:16], 16)  # first 64 bits of a hex digest
    except (ValueError, TypeError):
        val = int.from_bytes(
            hashlib.blake2b(str(cid).encode("utf-8"), digest_size=8).digest(), "big"
        )
    return val & 0x7FFFFFFFFFFFFFFF  # clear sign bit -> non-negative int64


# --------------------------------------------------------------------------- #
# Tokenizer loading (lazy, cached)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4)
def load_tokenizer(model_id: str, revision: str = "main"):
    """Load a HF tokenizer for token counting / windowing.

    Respects HF offline mode if the env vars are set. Cached so repeated calls
    in the chunk stage don't re-load. Returns the tokenizer object.
    """
    import os as _os
    from transformers import AutoTokenizer  # lazy

    # A local staged dir has no git revision; pass revision only for a repo id so
    # offline loads from a path don't trigger a hub lookup (HF_HUB_OFFLINE).
    kwargs = {"trust_remote_code": False}
    if not _os.path.isdir(model_id):
        kwargs["revision"] = revision
    return AutoTokenizer.from_pretrained(model_id, **kwargs)


def count_tokens(tokenizer, text: str) -> int:
    """Number of tokens for ``text`` under ``tokenizer`` (no special tokens)."""
    return len(tokenizer.encode(text, add_special_tokens=False))


# --------------------------------------------------------------------------- #
# GPU pinning
# --------------------------------------------------------------------------- #
def pin_gpu(gpu_id: Optional[int]) -> None:
    """Pin the process to a single GPU via CUDA_VISIBLE_DEVICES.

    Called by the embed driver so each SLURM array task uses exactly one card.
    If ``gpu_id`` is None we leave the environment untouched (e.g. when SLURM's
    --gres already scoped the device).
    """
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def set_offline(offline: bool) -> None:
    """Set HF offline env vars on compute nodes (hf.offline=true)."""
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def set_hf_home(hf_home: str) -> None:
    """Point HF caches at the staged scratch snapshot dir."""
    os.environ.setdefault("HF_HOME", hf_home)


# --------------------------------------------------------------------------- #
# Misc filesystem helpers
# --------------------------------------------------------------------------- #
def ensure_dir(path: str) -> str:
    """mkdir -p and return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def human_int(n: int) -> str:
    """Format an int with thousands separators for log readability."""
    return f"{n:,}"


def sliding_windows(token_ids: List[int], window: int, overlap: int) -> List[range]:
    """Yield (start, end) index ranges for a sliding window over a token list.

    Used by the oversized-symbol / non-parseable fallbacks. ``overlap`` tokens
    are shared between consecutive windows. Returns ranges into ``token_ids``.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    step = max(1, window - max(0, overlap))
    out: List[range] = []
    i = 0
    n = len(token_ids)
    while i < n:
        out.append(range(i, min(i + window, n)))
        if i + window >= n:
            break
        i += step
    return out
