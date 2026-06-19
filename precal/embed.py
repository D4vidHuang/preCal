"""Per-shard embed driver: read a shard, embed it, write .npy + parquet idempotently.

One call == one shard (shard_id == SLURM_ARRAY_TASK_ID). Resume-safety is the
core requirement (spec requeueStrategy) and is POSITIONAL (LOCKED CONTRACT D4):

  * The shard .npy is preallocated ONCE as a ``[n, embed_dim]`` float32 memmap
    (n == the shard's unique row count; sharding dedupes chunk_ids so chunk_id is
    a true primary key). Row i of the .npy is row i of the shard parquet.
  * On start, read the durable vector count V = number of rows already written
    (the ``.committed`` advisory log records how far we got; the .npy length is
    authoritative). We then embed EXACTLY shard rows ``[V:n]`` in order and write
    them into their fixed row slices -- never by chunk_id-set membership.
  * Every embed.checkpoint_every chunks we flush the memmap and fsync, THEN
    append the just-written chunk_ids to the advisory committed log and fsync.
    Vectors-before-ids ordering means the advisory log never gets ahead of the
    durable vectors; if they disagree we trust the .npy and resume from V.
  * On completion, hard-assert ``vecs.shape[0] == parquet_rows == n`` before
    writing the embedded parquet / any inline embedding and before flipping the
    shard status to ``done``; on success also drop the ops ``embed-<id>.done``
    marker (M.mark_done).

The .npy is the canonical vector store; parquet by default does NOT carry the
inline embedding column (kept for the viewer-fast layout).

Heavy imports (pyarrow, numpy, the engine) are scoped to call sites.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from precal import manifest as M
from precal.config import Config
from precal.engines import build_engine
from precal.schema import VECTOR_DISK_DTYPE, arrow_schema
from precal.utils import (
    chunk_id_to_int64,  # noqa: F401  (exposed for index id-mapping reuse)
    ensure_dir,
    get_logger,
    human_int,
    pin_gpu,
    set_hf_home,
    set_offline,
)

logger = get_logger("precal.embed")


def _load_shard_rows(out_parquet_input: str):
    """Read the shard's chunk parquet (the embed INPUT) into a pyarrow Table."""
    import pyarrow.parquet as pq

    return pq.read_table(out_parquet_input)


def _npy_shape(npy_path: str):
    """Return the (rows, dim) of an existing shard .npy, or None if absent/bad.

    Reads only the .npy header via mmap (no full load).
    """
    if not os.path.exists(npy_path):
        return None
    try:
        arr = np.load(npy_path, mmap_mode="r")
        if arr.ndim != 2:
            return None
        return int(arr.shape[0]), int(arr.shape[1])
    except Exception:
        return None


def _existing_vector_count(npy_path: str) -> int:
    """How many vectors are already durably written for this shard (0 if absent)."""
    shp = _npy_shape(npy_path)
    return shp[0] if shp is not None else 0


def _open_shard_memmap(npy_path: str, n: int, dim: int, valid_rows: int, dtype=None):
    """Open (or create) the shard .npy as a writeable ``[n, dim]`` float32 memmap.

    The .npy is preallocated ONCE at full shard size so batches write into fixed
    row slices ``[start:stop]`` -- O(N) total, no O(N^2) re-read/rewrite.

    ``valid_rows`` (= the positional resume cursor V, already clamped to n) tells
    us how many leading rows of an existing file are durable and must be carried
    over when we have to (re)allocate the full-size file. We only ever read rows
    ``[0:valid_rows]`` from a prior file; rows ``[valid_rows:n]`` are (re)written
    by the embed loop, so their prior contents are irrelevant.

    Reuse rules:
      * Existing file already has shape ``(n, dim)`` -> reuse in place (r+).
      * Existing file has the right ``dim`` but a different row count (e.g. an
        older append-style file of length V<n) -> migrate its first
        ``valid_rows`` rows into a fresh full-size memmap, drop the rest.
      * Otherwise -> fresh zero-initialized full-size memmap.
    """
    dtype = np.dtype(dtype if dtype is not None else VECTOR_DISK_DTYPE)
    shp = _npy_shape(npy_path)
    if shp == (n, dim):
        return np.lib.format.open_memmap(npy_path, mode="r+")

    # Preserve the durable prefix from an existing (differently-shaped) .npy.
    prefix = None
    if valid_rows > 0 and shp is not None and shp[1] == dim and shp[0] >= valid_rows:
        try:
            prefix = np.load(npy_path)[:valid_rows]  # bounded shard fits in RAM
        except Exception:
            prefix = None

    ensure_dir(os.path.dirname(os.path.abspath(npy_path)))
    tmp = npy_path + ".alloc.tmp"
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=dtype, shape=(n, dim))
    if prefix is not None and prefix.shape[0] > 0:
        mm[: prefix.shape[0]] = np.ascontiguousarray(prefix, dtype=dtype)
    mm.flush()
    del mm  # close the tmp memmap before rename
    os.replace(tmp, npy_path)
    return np.lib.format.open_memmap(npy_path, mode="r+")


def embed_shard(
    cfg: Config,
    shard_id: int,
    *,
    resume: bool = True,
    gpu_id: Optional[int] = None,
    base_port: int = 7997,
    engine_override: Optional[str] = None,
) -> None:
    """Embed a single shard, resumable and idempotent.

    Parameters
    ----------
    shard_id:
        Global shard id == SLURM_ARRAY_TASK_ID.
    resume:
        If True (default) resume positionally from the durable .npy row count V
        (embed only rows ``[V:n]``); if False, start over (the shard's .npy +
        committed log are cleared first).
    gpu_id:
        Optional CUDA device to pin (CUDA_VISIBLE_DEVICES). For TEI, the device
        is owned by the replica processes, so this is mainly for in-process
        engines (infinity/vllm).
    base_port:
        TEI replica base port (replicas listen on base_port..+replicas-1).
    engine_override:
        Override cfg.engine.name (CLI --engine).
    """
    import pyarrow.parquet as pq

    # Environment setup for compute nodes.
    set_offline(cfg.hf.offline)
    set_hf_home(cfg.paths.hf_home)
    pin_gpu(gpu_id)
    if engine_override:
        cfg.engine.name = engine_override

    entry = M.get_shard(cfg.manifest_path, shard_id)
    manifest_dir = cfg.paths.manifest_dir
    ensure_dir(manifest_dir)
    M.write_status(manifest_dir, shard_id, M.STATUS_RUNNING, language=entry.language)

    npy_path = entry.out_npy
    embedded_parquet = entry.out_parquet.replace(".parquet", ".embedded.parquet")
    ensure_dir(os.path.dirname(npy_path))

    # --- resume bookkeeping (POSITIONAL, D4) ----------------------------- #
    if not resume:
        for p in (npy_path, M.committed_log_path(manifest_dir, shard_id)):
            if os.path.exists(p):
                os.remove(p)

    table = _load_shard_rows(entry.out_parquet)
    chunk_ids = table.column("chunk_id").to_pylist()
    # The DOCUMENT side is encoded raw; queries (if any) are NOT what we index.
    texts = table.column("text").to_pylist()
    n = len(chunk_ids)
    dim = cfg.model.embed_dim

    # Positional resume cursor V = durable .npy row count, clamped to n. We embed
    # rows [V:n] only. The committed-id log is ADVISORY: if it disagrees with the
    # .npy we trust the .npy. A preallocated [n,dim] memmap has exactly n rows, so
    # for a partially-finished preallocated file the advisory log tells us how
    # many leading rows are actually written.
    npy_rows = _existing_vector_count(npy_path)
    committed = M.scan_committed_ids(manifest_dir, shard_id)
    n_committed = len(committed)
    shp = _npy_shape(npy_path)
    if shp == (n, dim):
        # Preallocated full-size file: trust the advisory log for how far we got.
        V = min(n_committed, n)
        if n_committed != V:
            logger.warning(
                "shard %d: committed log (%d) exceeds shard size (%d); clamping V=%d.",
                shard_id, n_committed, n, V,
            )
    else:
        # Legacy/append-style or absent file: the row count IS the durable count.
        V = min(npy_rows, n)
        if n_committed > V:
            logger.warning(
                "shard %d: committed log (%d) exceeds .npy rows (%d); truncating "
                "to .npy length V=%d (vectors-before-ids: .npy is authoritative).",
                shard_id, n_committed, npy_rows, V,
            )
    logger.info(
        "shard %d (lang=%s): %s chunks, resuming positionally from row %s",
        shard_id,
        entry.language,
        human_int(n),
        human_int(V),
    )

    _vdtype = np.dtype(cfg.model.vector_dtype)  # .npy storage dtype (fp32|fp16)
    if V >= n:
        logger.info("shard %d: nothing to embed (already complete).", shard_id)
        # Still (re)open to guarantee the canonical [n,dim] file exists on disk.
        mm = _open_shard_memmap(npy_path, n, dim, V, dtype=_vdtype)
        mm.flush()
        del mm
    else:
        mm = _open_shard_memmap(npy_path, n, dim, V, dtype=_vdtype)
        engine = build_engine(cfg, base_port=base_port)
        try:
            cursor = V
            since_ckpt = 0
            ckpt = max(1, cfg.embed.checkpoint_every)
            pending_ids: List[str] = []
            # Hand the engine a large BLOCK per call so the TEI client can fan many
            # per-POST batches (engine.batch_size each) across all replicas at once.
            block = max(cfg.engine.batch_size, 1024)

            for start in range(V, n, block):
                stop = min(start + block, n)
                batch_texts = [texts[i] or "" for i in range(start, stop)]
                vecs = engine.embed_documents(batch_texts)  # [b, dim] float32
                vecs = np.ascontiguousarray(vecs, dtype=_vdtype)
                if vecs.shape[0] != (stop - start):
                    raise RuntimeError(
                        f"shard {shard_id}: engine returned {vecs.shape[0]} vectors "
                        f"for a {stop - start}-row batch."
                    )
                if vecs.shape[1] != dim:
                    raise RuntimeError(
                        f"shard {shard_id}: engine returned dim {vecs.shape[1]} != "
                        f"model.embed_dim {dim}."
                    )
                mm[start:stop] = vecs  # fixed positional row slice
                cursor = stop
                pending_ids.extend(chunk_ids[start:stop])
                since_ckpt += (stop - start)

                if since_ckpt >= ckpt:
                    _flush_checkpoint(mm, manifest_dir, shard_id, pending_ids, cursor)
                    pending_ids, since_ckpt = [], 0

            # Final flush of any tail past the last checkpoint.
            if pending_ids:
                _flush_checkpoint(mm, manifest_dir, shard_id, pending_ids, cursor)
        finally:
            engine.close()
            del mm  # release the memmap handle

    # --- finalize: hard-assert counts, write parquet + status=done ------- #
    final_rows = _existing_vector_count(npy_path)
    assert final_rows == n, (
        f"shard {shard_id}: .npy rows ({final_rows}) != shard rows n ({n}) "
        f"after embed; refusing to finalize."
    )

    _write_embedded_parquet(cfg, entry, table, npy_path, embedded_parquet)

    parquet_rows = pq.read_metadata(embedded_parquet).num_rows
    # Hard invariant (D4): vectors == parquet rows == shard size, all equal to n.
    assert final_rows == parquet_rows == n, (
        f"shard {shard_id}: row mismatch (npy={final_rows}, parquet={parquet_rows}, "
        f"n={n}); refusing to mark done."
    )
    M.write_status(
        manifest_dir,
        shard_id,
        M.STATUS_DONE,
        language=entry.language,
        rows=final_rows,
    )
    # Drop the cheap ops marker the slurm guards stat (embed-<raw_id>.done).
    M.mark_done(manifest_dir, "embed", shard_id)
    logger.info("shard %d: DONE (%s vectors).", shard_id, human_int(final_rows))


def _flush_checkpoint(
    mm,
    manifest_dir: str,
    shard_id: int,
    pending_ids: List[str],
    cursor: int,
) -> None:
    """Durably flush the shard memmap, THEN append the just-written ids.

    Ordering is load-bearing: the vector rows hit disk (memmap.flush + fsync of
    the backing file) BEFORE their chunk_ids are appended to the advisory
    committed log (also fsync). A crash between the two simply re-embeds those
    rows on resume (positional, idempotent), never the reverse -- so the advisory
    log can never get ahead of the durable .npy.
    """
    mm.flush()
    # fsync the backing file so the row slices are durable, not just in page cache.
    try:
        fd = os.open(mm.filename, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except (AttributeError, OSError):
        # mm.filename may be absent on exotic numpy builds; flush() still ran.
        pass
    M.append_committed_ids(manifest_dir, shard_id, pending_ids)  # then ids
    logger.info(
        "shard %d: checkpoint flushed (rows durable up to %d).",
        shard_id,
        cursor,
    )


def _write_embedded_parquet(cfg: Config, entry, table, npy_path, out_parquet) -> None:
    """Write the embed-stage parquet with model/vector pointer columns filled.

    Adds vector_shard/row_in_shard pointers + model_id/revision/embed_dim/
    pooling/normalized/dtype. Optionally inlines the embedding column when
    publish.emit_inline_embedding=true (small variants only).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = table.num_rows
    vector_shard_name = os.path.basename(npy_path)
    # Reproduce the published relative path for vector_shard, e.g.
    # vectors/lang=python/part-00007.npy.
    rel_vector_shard = os.path.join(
        "vectors", f"lang={entry.language}", vector_shard_name
    )

    add_cols: Dict[str, "pa.Array"] = {
        "vector_shard": pa.array([rel_vector_shard] * n, pa.string()),
        "row_in_shard": pa.array(list(range(n)), pa.int32()),
        "model_id": pa.array([cfg.model.id] * n, pa.string()),
        "model_revision": pa.array([cfg.model.revision] * n, pa.string()),
        "embed_dim": pa.array([cfg.model.embed_dim] * n, pa.int32()),
        "pooling": pa.array([cfg.model.pooling] * n, pa.string()),
        "normalized": pa.array([cfg.model.normalize] * n, pa.bool_()),
        "dtype": pa.array([cfg.model.dtype] * n, pa.string()),
    }

    out = table
    for name, arr in add_cols.items():
        idx = out.schema.get_field_index(name)
        if idx >= 0:
            out = out.set_column(idx, name, arr)
        else:
            out = out.append_column(name, arr)

    include_embedding = cfg.publish.emit_inline_embedding
    if include_embedding and os.path.exists(npy_path):
        vecs = np.load(npy_path)
        # Hard invariant (D4): never inline a partial/misaligned vector set.
        assert vecs.shape[0] == n, (
            f"shard {entry.shard_id}: inline embedding requested but .npy rows "
            f"({vecs.shape[0]}) != parquet rows ({n})."
        )
        emb = pa.array(
            [row.tolist() for row in vecs], type=pa.list_(pa.float32())
        )
        idx = out.schema.get_field_index("embedding")
        if idx >= 0:
            out = out.set_column(idx, "embedding", emb)
        else:
            out = out.append_column("embedding", emb)

    schema = arrow_schema(include_embedding=include_embedding)
    # Reorder/cast to the canonical schema (only columns present).
    cols_present = [f.name for f in schema if f.name in out.column_names]
    out = out.select(cols_present)

    ensure_dir(os.path.dirname(out_parquet))
    tmp = out_parquet + ".tmp"
    pq.write_table(out, tmp, row_group_size=20000, compression="zstd")
    os.replace(tmp, out_parquet)
    logger.info("shard %d: wrote embedded parquet -> %s", entry.shard_id, out_parquet)
