"""Single source of truth for the preCal row schema + the .npy sidecar contract.

Every stage (chunk, pairs, shard, embed, index, publish) reads the column
names and types from here so they never drift. The columns and their semantics
mirror the BUILD SPEC's `dataSchema` *verbatim*.

Storage model (spec decision `storage_format`): HYBRID per shard.
  * Parquet holds metadata + text (the browsable side). By default the
    ``embedding`` list<float32> column is OMITTED to keep the HF dataset viewer
    fast (the <5GB / no-wide-list rule). It is written only when
    ``publish.emit_inline_embedding=true``.
  * The canonical vectors live in a sidecar float32 ``.npy`` array of shape
    ``[N, embed_dim]``, C-contiguous, so FAISS can train/add via a zero-copy
    memmap. ``(vector_shard, row_in_shard)`` links a parquet row to its .npy row.

This module has no heavy imports at module load: pyarrow is imported lazily so
the schema *names/types* can be inspected without pyarrow installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

# --------------------------------------------------------------------------- #
# Canonical .npy sidecar dtype: vectors are stored as float32 on disk even when
# computed in bf16/fp16, because FAISS trains/adds on float32 and float32 is the
# lossless superset that downstream consumers expect. The `dtype` *column*
# records the COMPUTE dtype (bfloat16|float16) separately.
# --------------------------------------------------------------------------- #
VECTOR_DISK_DTYPE = np.float32


@dataclass(frozen=True)
class ColumnSpec:
    """Describes one parquet column: name, an Arrow type *string*, nullability,
    and a one-line description. The Arrow type string is resolved to a real
    ``pyarrow.DataType`` lazily in :func:`arrow_schema`."""

    name: str
    arrow_type: str  # e.g. "string", "int32", "bool", "list<float32>"
    description: str
    nullable: bool = True


# Order matters: this is the canonical column order written to parquet.
COLUMNS: List[ColumnSpec] = [
    ColumnSpec(
        "chunk_id",
        "string",
        "Stable primary key = blake2b hex of normalized chunk text + repo + "
        "path + span. Idempotency key for resume; FAISS id source; RAG corpus _id.",
        nullable=False,
    ),
    ColumnSpec("repo_name", "string", "Source repository (owner/name) for provenance + split partitioning."),
    ColumnSpec("path", "string", "File path within the repo (Stack path) for provenance + dedup."),
    ColumnSpec("language", "string", "Lowercase language id from the v1 slice (python|java|javascript|php|go|ruby)."),
    ColumnSpec("license", "string", "SPDX license id from Stack detected_licenses; gates text republishing."),
    ColumnSpec("text_publishable", "bool", "True if license on allowlist AND text may be redistributed."),
    ColumnSpec("symbol_kind", "string", "Chunk granularity: function|method|class|module|window|whole_file."),
    ColumnSpec("symbol_name", "string", "Extracted function/class/method name (empty for window/whole_file)."),
    ColumnSpec("start_line", "int32", "1-based start line of the chunk span."),
    ColumnSpec("end_line", "int32", "1-based end line of the chunk span."),
    ColumnSpec("n_tokens", "int32", "Token count under the model tokenizer."),
    ColumnSpec("truncated", "bool", "True if the chunk exceeded chunk_max_tokens and was truncated/windowed."),
    ColumnSpec("text", "string", "The chunk source code (DOCUMENT side, encoded RAW). Empty when not publishable."),
    ColumnSpec("query_text", "string", "NL query paired to this chunk (docstring/CSN/commit), empty if none."),
    ColumnSpec("query_source", "string", "Provenance of query_text: docstring|codesearchnet|coir|commit|none."),
    ColumnSpec("eval_split", "string", "Frozen split flag: index_only|eval_test|eval_valid (repo-level partition)."),
    ColumnSpec("vector_shard", "string", "Filename of the sidecar .npy holding this row's float32 vector."),
    ColumnSpec("row_in_shard", "int32", "0-based row index of this chunk within vector_shard (.npy row)."),
    ColumnSpec(
        "embedding",
        "list<float32>",
        "OPTIONAL inline vector (dim = embed_dim). Omitted unless emit_inline_embedding=true.",
    ),
    ColumnSpec("model_id", "string", "Producing model HF id (Qwen/Qwen3-Embedding-4B)."),
    ColumnSpec("model_revision", "string", "HF commit hash of the model snapshot used."),
    ColumnSpec("embed_dim", "int32", "Embedding dimensionality after any MRL truncation (e.g. 2560)."),
    ColumnSpec("pooling", "string", "Pooling method used (last_token)."),
    ColumnSpec("normalized", "bool", "True = L2-normalized (cosine via inner product). Always true in v1."),
    ColumnSpec("dtype", "string", "Compute dtype of the produced vector (bfloat16|float16)."),
    ColumnSpec("corpus_snapshot", "string", "Source corpus + snapshot id (bigcode/the-stack-dedup@<revision>)."),
]

# Convenience lookups.
COLUMN_NAMES: List[str] = [c.name for c in COLUMNS]
_COLUMN_BY_NAME: Dict[str, ColumnSpec] = {c.name: c for c in COLUMNS}

# Columns produced by the CHUNK stage (before embedding). The embed stage fills
# the remaining model/vector columns. We keep `embedding` out of the chunk-stage
# parquet and only ever add it (optionally) at publish time.
CHUNK_STAGE_COLUMNS: List[str] = [
    "chunk_id",
    "repo_name",
    "path",
    "language",
    "license",
    "text_publishable",
    "symbol_kind",
    "symbol_name",
    "start_line",
    "end_line",
    "n_tokens",
    "truncated",
    "text",
    "query_text",
    "query_source",
    "eval_split",
    "corpus_snapshot",
]

# Columns the EMBED stage adds / fills when it writes the embedded parquet.
EMBED_STAGE_COLUMNS: List[str] = [
    "vector_shard",
    "row_in_shard",
    "model_id",
    "model_revision",
    "embed_dim",
    "pooling",
    "normalized",
    "dtype",
    # `embedding` is appended only when emit_inline_embedding=true.
]


def _resolve_arrow_type(type_str: str):
    """Map a type string to a concrete ``pyarrow.DataType`` (lazy import)."""
    import pyarrow as pa  # local import keeps module import-clean without pyarrow

    mapping = {
        "string": pa.string(),
        "bool": pa.bool_(),
        "int32": pa.int32(),
        "int64": pa.int64(),
        "float32": pa.float32(),
        "list<float32>": pa.list_(pa.float32()),
    }
    if type_str not in mapping:
        raise ValueError(f"Unknown arrow type string: {type_str!r}")
    return mapping[type_str]


def arrow_schema(include_embedding: bool = False, subset: "List[str] | None" = None):
    """Build the canonical ``pyarrow.Schema`` for preCal parquet files.

    Parameters
    ----------
    include_embedding:
        Include the inline ``embedding`` list<float32> column. Default False
        (kept out for viewer performance, per the storage_format decision).
    subset:
        If given, only emit these columns (preserving canonical order). Used by
        the chunk stage which doesn't yet have vectors.
    """
    import pyarrow as pa

    selected: List[ColumnSpec] = []
    for col in COLUMNS:
        if col.name == "embedding" and not include_embedding:
            continue
        if subset is not None and col.name not in subset:
            continue
        selected.append(col)

    arrow_fields = [
        pa.field(c.name, _resolve_arrow_type(c.arrow_type), nullable=c.nullable)
        for c in selected
    ]
    return pa.schema(arrow_fields)


def empty_row(corpus_snapshot: str = "") -> Dict[str, Any]:
    """Return a dict with every column key set to a type-appropriate default.

    Stages fill in what they know and leave the rest at defaults; the schema
    keeps these coherent so a partially-populated row still casts cleanly to the
    Arrow schema.
    """
    defaults: Dict[str, Any] = {}
    for c in COLUMNS:
        if c.name == "embedding":
            defaults[c.name] = None
        elif c.arrow_type == "string":
            defaults[c.name] = ""
        elif c.arrow_type == "bool":
            defaults[c.name] = False
        elif c.arrow_type in ("int32", "int64"):
            defaults[c.name] = 0
        else:
            defaults[c.name] = None
    defaults["corpus_snapshot"] = corpus_snapshot
    # Sensible v1 constants.
    defaults["normalized"] = True
    defaults["pooling"] = "last_token"
    defaults["query_source"] = "none"
    defaults["eval_split"] = "index_only"
    defaults["symbol_kind"] = "whole_file"
    return defaults


def describe() -> str:
    """Human-readable schema dump (used by the dataset card + `--describe`)."""
    lines = ["preCal parquet schema (canonical column order):", ""]
    for c in COLUMNS:
        opt = " [optional/omitted by default]" if c.name == "embedding" else ""
        lines.append(f"  {c.name:<16} {c.arrow_type:<14} {c.description}{opt}")
    return "\n".join(lines)
