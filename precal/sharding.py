"""Turn the chunked corpus into deterministic shards + a shard manifest.

Two-level sharding (spec `shardingScheme`):
  Level 1 = language (corpus.languages).
  Level 2 = fixed-size chunk shards of ~shard.target_chunks within each language.

The shard stage:
  * reads each language's chunk parquet (produced by the `chunk` stage),
  * (optionally) runs pair/eval-split assignment (pairs.py) unless --no-pairs,
  * groups rows into shards of ~target_chunks, assigning GLOBAL contiguous
    shard_ids (0..N-1) across all languages so shard_id == SLURM_ARRAY_TASK_ID,
  * writes per-shard chunk parquet (the embed input) under shards/, and
  * writes shards/manifest.jsonl (one ShardEntry per shard).

Determinism: rows within a language are ordered by chunk_id (stable) before
being cut into shards, so the same inputs + seed always yield the same shards.
The stage is idempotent: re-running overwrites the manifest and shard parquet
atomically and is safe to repeat.

pyarrow/datasets are imported lazily so this module imports clean.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional

from precal import pairs
from precal.config import Config
from precal.manifest import ShardEntry, write_manifest
from precal.schema import CHUNK_STAGE_COLUMNS, arrow_schema
from precal.utils import ensure_dir, get_logger, human_int

logger = get_logger("precal.sharding")


def _chunk_parquet_glob(cfg: Config, language: str) -> str:
    """Glob for the chunk-stage parquet of a given language."""
    return os.path.join(cfg.chunks_dir, f"lang={language}", "*.parquet")


def _shard_out_parquet(cfg: Config, language: str, shard_id: int) -> str:
    return os.path.join(
        cfg.shards_dir, f"lang={language}", f"shard-{shard_id:05d}.parquet"
    )


def _shard_out_npy(cfg: Config, language: str, shard_id: int) -> str:
    # Mirrors the published vectors/ layout: vectors/lang=<l>/part-<id>.npy
    return os.path.join(
        cfg.vectors_dir, f"lang={language}", f"part-{shard_id:05d}.npy"
    )


def _dedupe_by_chunk_id(table):
    """Drop rows with duplicate chunk_id, keeping the FIRST occurrence.

    chunk_id is the primary key (resume idempotency key + FAISS id source), so a
    repeated chunk_id within a language would collide on the int64 FAISS id and
    break the positional parquet-row<->.npy-row contract. We compute a boolean
    keep-mask in one pass over the (already-sorted) chunk_id column and filter,
    which preserves the existing row order of the surviving rows.
    """
    import pyarrow as pa

    chunk_ids = table.column("chunk_id").to_pylist()
    seen: set = set()
    keep: List[bool] = []
    for cid in chunk_ids:
        if cid in seen:
            keep.append(False)
        else:
            seen.add(cid)
            keep.append(True)
    n_dups = len(keep) - sum(keep)
    if n_dups:
        logger.warning(
            "Dropped %s duplicate chunk_id row(s) while sharding (kept first).",
            human_int(n_dups),
        )
        table = table.filter(pa.array(keep, type=pa.bool_()))
    return table


def build_shards(
    cfg: Config,
    languages: Optional[List[str]] = None,
    target_chunks: Optional[int] = None,
    do_pairs: bool = True,
) -> List[ShardEntry]:
    """Build shards for the requested languages and write the global manifest.

    Parameters
    ----------
    languages:
        Subset of cfg.corpus.languages to shard (default: all).
    target_chunks:
        Override cfg.shard.target_chunks (e.g. CLI --target-chunks).
    do_pairs:
        Run pair/eval-split assignment while sharding (default True; --no-pairs
        skips it, leaving query_text/eval_split at chunk-stage defaults).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    languages = languages or list(cfg.corpus.languages)
    target = target_chunks or cfg.shard.target_chunks
    schema = arrow_schema(include_embedding=False, subset=CHUNK_STAGE_COLUMNS)

    entries: List[ShardEntry] = []
    next_shard_id = 0  # GLOBAL contiguous across all languages

    for language in languages:
        files = sorted(glob.glob(_chunk_parquet_glob(cfg, language)))
        if not files:
            logger.warning("No chunk parquet for language=%s (skipping)", language)
            continue

        # Load the eval-repo set once per language for the leakage-safe split.
        eval_repos = pairs.load_eval_repos(
            os.path.join(cfg.corpus_dir, "coir-csn"), language
        ) if do_pairs else set()

        # Read all rows for this language, ordered by chunk_id for determinism.
        # Read each file explicitly and concat so Hive-style directory layout
        # (lang=<l>/) does NOT inject a partition column into the table.
        # Code `text` across many parts can exceed Arrow's 2GB `string` (int32
        # offset) cap when concatenated -> promote string columns to large_string
        # for the concat/sort. Each per-shard slice (<=target rows) is narrowed
        # back to `string` by .cast(schema) at write time (always < 2GB).
        def _to_large_string(t):
            f2 = [
                pa.field(f.name, pa.large_string()) if pa.types.is_string(f.type) else f
                for f in t.schema
            ]
            return t.cast(pa.schema(f2))

        tables = [_to_large_string(pq.read_table(f)) for f in files]
        table = pa.concat_tables(tables, promote_options="default")
        # Keep only canonical chunk-stage columns in canonical order (drops any
        # stray columns such as an inferred partition field).
        keep = [c for c in CHUNK_STAGE_COLUMNS if c in table.column_names]
        table = table.select(keep)
        table = table.sort_by([("chunk_id", "ascending")])
        # chunk_id is the TRUE primary key (idempotency key / FAISS id source).
        # Dedupe WITHIN the language so each shard's n == its unique row count and
        # positional embed/index (parquet row i <-> .npy row i) never collides.
        # Keep the FIRST occurrence, drop later dups. Sorted-by-chunk_id above
        # makes dups adjacent; we still mask generically to be order-independent.
        table = _dedupe_by_chunk_id(table)
        n_rows = table.num_rows
        logger.info(
            "language=%s: %s chunks across %d file(s)",
            language,
            human_int(n_rows),
            len(files),
        )

        # Cut into shards of ~target rows.
        ensure_dir(os.path.join(cfg.shards_dir, f"lang={language}"))
        offset = 0
        while offset < n_rows:
            length = min(target, n_rows - offset)
            # combine_chunks() materializes JUST this slice into fresh contiguous
            # arrays; without it, the slice still points at the >2GB large_string
            # parent buffer and .cast(schema) (large_string->string) fails with
            # "input array too large". Each <=target-row slice is small (<2GB).
            shard_table = table.slice(offset, length).combine_chunks()

            if do_pairs:
                shard_table = _apply_pairs(
                    shard_table, language, eval_repos, cfg.run.seed, schema
                )

            shard_id = next_shard_id
            out_parquet = _shard_out_parquet(cfg, language, shard_id)
            out_npy = _shard_out_npy(cfg, language, shard_id)

            # Atomic-ish parquet write: write to tmp then rename.
            tmp = out_parquet + ".tmp"
            pq.write_table(
                shard_table.cast(schema) if shard_table.schema != schema else shard_table,
                tmp,
                row_group_size=20000,  # within the spec's 10k-50k guidance
                compression="zstd",
            )
            os.replace(tmp, out_parquet)

            entries.append(
                ShardEntry(
                    shard_id=shard_id,
                    language=language,
                    input_files=[out_parquet],
                    approx_chunks=shard_table.num_rows,
                    out_parquet=out_parquet,
                    out_npy=out_npy,
                )
            )
            logger.info(
                "  shard %d (lang=%s): %s chunks -> %s",
                shard_id,
                language,
                human_int(shard_table.num_rows),
                out_parquet,
            )
            next_shard_id += 1
            offset += length

    if not entries:
        raise RuntimeError(
            "No shards produced. Did you run `precal.cli chunk` for the languages first?"
        )

    ensure_dir(cfg.shards_dir)
    write_manifest(cfg.manifest_path, entries)
    logger.info(
        "Sharding complete: %d shards across %d language(s); manifest -> %s",
        len(entries),
        len(set(e.language for e in entries)),
        cfg.manifest_path,
    )
    return entries


def _apply_pairs(table, language: str, eval_repos, seed: int, schema):
    """Vectorized-ish application of query/eval_split assignment to a table.

    Reads the needed columns to Python lists (shards are bounded by
    target_chunks so this stays in memory), computes query_text/query_source and
    eval_split per row, and rebuilds the columns.
    """
    import pyarrow as pa

    texts = table.column("text").to_pylist()
    repos = table.column("repo_name").to_pylist()
    # Preserve any query already attached upstream (e.g. CSN-joined) if present.
    existing_q = (
        table.column("query_text").to_pylist()
        if "query_text" in table.column_names
        else [""] * len(texts)
    )

    query_text: List[str] = []
    query_source: List[str] = []
    eval_split: List[str] = []
    for txt, repo, eq in zip(texts, repos, existing_q):
        if eq:
            q, src = eq, "codesearchnet"
        else:
            q, src = pairs.assign_query(text=txt or "", language=language)
        query_text.append(q)
        query_source.append(src)
        eval_split.append(
            pairs.assign_eval_split(repo or "", eval_repos=eval_repos, seed=seed)
        )

    out = table
    for col_name, values in (
        ("query_text", query_text),
        ("query_source", query_source),
        ("eval_split", eval_split),
    ):
        arr = pa.array(values, type=pa.string())
        idx = out.schema.get_field_index(col_name)
        if idx >= 0:
            out = out.set_column(idx, col_name, arr)
        else:
            out = out.append_column(col_name, arr)
    return out
