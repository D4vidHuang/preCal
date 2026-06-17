"""preCal: a SLURM-array code-embedding pipeline that precomputes a dual-use
RAG/retrieval research artifact (code embeddings + FAISS index + NL<->code
pairs + frozen eval splits) and publishes it to the Hugging Face Hub.

Data flow: corpus -> shard -> embed -> index -> publish -> eval.

Heavy / optional dependencies (faiss, torch, transformers, tree-sitter, the
TEI client, mteb, huggingface_hub) are imported lazily inside the modules /
functions that need them so that importing this package (and running its
config + schema + manifest logic) stays clean on a machine without a GPU
stack. Only the lightweight config loader is re-exported at the top level.
"""

from __future__ import annotations

# Single source of truth for the package version. Bump on schema-affecting
# changes; the version is recorded into the dataset card at publish time.
__version__ = "1.0.0"

# Re-export the config loader so callers can do `from precal import load_config`
# without pulling in any heavy modules.
from precal.config import Config, load_config  # noqa: E402

__all__ = ["__version__", "Config", "load_config"]
