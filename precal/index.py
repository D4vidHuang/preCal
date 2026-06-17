"""FAISS index build from the .npy sidecars.

Per the spec, indexes are built from the immutable float32 .npy vectors (zero
Arrow round-trip). LOCKED CONTRACT (D5) splits the work into two clearly
separated tiers so the expensive index is built exactly once:

  * Per shard (``build_shard_index``): ONLY an exact ``Flat`` index on
    METRIC_INNER_PRODUCT (cosine because vectors are L2-normalized). This is
    cheap, gives the exact-recall ceiling, and powers smoke runs and the
    optional scatter-gather (``sharded``) search layout. IVF/OPQ/PQ factories
    are REFUSED at the per-shard level — they belong to the merge path.

  * At merge (``merge_indexes``): the ONE full
    ``OPQ64_256,IVF65536_HNSW32,PQ64`` index (1M-50M tier) on inner product is
    built once. The IVF coarse quantizer is trained on ``index.train_sample``
    (~2M) random vectors sampled across ALL shard .npy memmaps (seeded by
    ``run.seed``); then ``add_with_ids(all vectors, chunk-derived int64 ids)``
    is called with the ids carried on the IVF ITSELF (via
    ``faiss.extract_index_ivf``), NOT an outer ``IndexIDMap2``, so the ids
    survive serialization. ``merge_ondisk`` is dropped: at v1 scale, building
    one global index directly from the memmaps is correct and simpler.

Build modes (``merge_indexes``):
  * ``ondisk`` (default): build the ONE full global IVF-PQ index from all shard
    memmaps -> ``faiss/merged/merged.faiss``.
  * ``sharded``: leave the cheap per-shard Flat indexes as-is and write a small
    JSON manifest for scatter-gather search.

FAISS is imported lazily so the module imports clean without it (faiss-cpu is
not installed on the macOS dev box); a clear error is raised when an index call
is actually made.
"""

from __future__ import annotations

import glob
import os
from typing import List, Optional, Tuple

import numpy as np

from precal import manifest as M
from precal.config import Config
from precal.utils import (
    chunk_id_to_int64,
    ensure_dir,
    get_logger,
    human_int,
)

logger = get_logger("precal.index")


def _import_faiss():
    try:
        import faiss  # lazy

        return faiss
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "FAISS is required for index/merge. Install faiss-cpu "
            "(`pip install faiss-cpu`) or a faiss-gpu build."
        ) from exc


def _metric(faiss, metric: str):
    if metric == "inner_product":
        return faiss.METRIC_INNER_PRODUCT
    if metric in ("l2", "euclidean"):
        return faiss.METRIC_L2
    raise ValueError(f"Unsupported index.metric={metric!r}")


def _shard_faiss_path(cfg: Config, shard_id: int, language: str) -> str:
    return os.path.join(
        cfg.faiss_dir, f"lang={language}", f"shard-{shard_id:05d}.faiss"
    )


# Factory tokens that require IVF training / coarse quantization. These are
# REFUSED at the per-shard level (LOCKED CONTRACT D5): the heavy IVF/OPQ/PQ
# index is built exactly once in the merge path, never per shard.
_IVF_FACTORY_TOKENS = ("IVF", "OPQ", "PQ", "HNSW", "LSH", "RCQ", "IMI")


def _is_ivf_factory(factory: str) -> bool:
    """True if ``factory`` describes anything beyond an exact Flat index."""
    upper = factory.upper()
    if upper in ("FLAT", "IDMAP,FLAT", "IDMAP2,FLAT"):
        return False
    return any(tok in upper for tok in _IVF_FACTORY_TOKENS)


def _load_ids_for_shard(cfg: Config, entry) -> np.ndarray:
    """Read chunk_ids for a shard and map them to int64 FAISS ids.

    Prefers the embedded parquet (has the final row order); falls back to the
    chunk-stage parquet (same order). The FAISS id is derived from chunk_id so
    search results can be joined back to the corpus rows.
    """
    import pyarrow.parquet as pq

    embedded = entry.out_parquet.replace(".parquet", ".embedded.parquet")
    src = embedded if os.path.exists(embedded) else entry.out_parquet
    ids = pq.read_table(src, columns=["chunk_id"]).column("chunk_id").to_pylist()
    return np.asarray([chunk_id_to_int64(c) for c in ids], dtype=np.int64)


def build_shard_index(
    cfg: Config,
    shard_id: int,
    *,
    factory_override: Optional[str] = None,
) -> str:
    """Build the per-shard exact ``Flat`` FAISS index from its .npy sidecar.

    LOCKED CONTRACT (D5): the per-shard stage builds ONLY a cheap exact Flat
    (inner product) index — never an IVF/OPQ/PQ factory. The heavy global index
    is built once in :func:`merge_indexes`. A ``--factory`` override is accepted
    ONLY if it is still an exact Flat (e.g. to force ``Flat`` explicitly); any
    IVF-family factory is REFUSED here with a clear error pointing at merge.

    The Flat index is wrapped in an ``IndexIDMap2`` so search returns
    chunk-derived int64 ids (joinable back to the corpus rows). On success the
    ``index`` done marker is written via ``M.mark_done(manifest_dir, "index",
    shard_id)``. Atomic write of the final index file.

    Returns the path to the written .faiss file.
    """
    faiss = _import_faiss()
    entry = M.get_shard(cfg.manifest_path, shard_id)

    if not os.path.exists(entry.out_npy):
        raise FileNotFoundError(
            f"shard {shard_id}: vectors not found at {entry.out_npy}; embed first."
        )

    # Per-shard factory is ALWAYS exact Flat. An override is allowed only if it
    # is itself a Flat factory; refuse IVF-family factories at the shard level.
    factory = factory_override or cfg.index.factory_smoke
    if _is_ivf_factory(factory):
        raise ValueError(
            f"shard {shard_id}: per-shard index must be an exact Flat factory, "
            f"got {factory!r}. IVF/OPQ/PQ factories (e.g. {cfg.index.factory_full!r}) "
            f"are built ONCE in the merge path (`precal.cli merge-index`), not per "
            f"shard. Pass --factory Flat or omit --factory."
        )

    vecs = np.load(entry.out_npy, mmap_mode="r")
    n, d = vecs.shape
    if d != cfg.model.embed_dim:
        logger.warning(
            "shard %d: .npy dim %d != model.embed_dim %d (using .npy dim).",
            shard_id,
            d,
            cfg.model.embed_dim,
        )

    metric = _metric(faiss, cfg.index.metric)
    logger.info(
        "shard %d: building exact Flat FAISS '%s' over %s x %d vectors (metric=%s)",
        shard_id,
        factory,
        human_int(n),
        d,
        cfg.index.metric,
    )

    index = faiss.index_factory(d, factory, metric)
    # Flat is always trained; assert so a mislabeled override can't slip through.
    assert index.is_trained, f"shard {shard_id}: expected a trained Flat index for {factory!r}"

    # Wrap with IDMap so we can attach chunk-derived ids (Flat has no native ids).
    ids = _load_ids_for_shard(cfg, entry)
    if ids.shape[0] != n:
        raise ValueError(
            f"shard {shard_id}: id count {ids.shape[0]} != vector count {n}; "
            f"parquet/.npy disagree (embed must finish before index)."
        )
    id_index = faiss.IndexIDMap2(index)

    # Add in blocks to bound peak memory.
    BLOCK = 100000
    for start in range(0, n, BLOCK):
        block = np.ascontiguousarray(vecs[start : start + BLOCK], dtype=np.float32)
        id_index.add_with_ids(block, ids[start : start + block.shape[0]])

    out_path = _shard_faiss_path(cfg, shard_id, entry.language)
    ensure_dir(os.path.dirname(out_path))
    tmp = out_path + ".tmp"
    faiss.write_index(id_index, tmp)
    os.replace(tmp, out_path)
    logger.info("shard %d: wrote exact Flat index -> %s", shard_id, out_path)

    # LOCKED CONTRACT (D2): write the per-stage done marker on success so the ops
    # guards (slurm/index.sbatch, scripts/resubmit_pending.sh) can skip finished
    # shards. RAW shard_id, no zero-pad: f"{manifest_dir}/index-{shard_id}.done".
    M.mark_done(cfg.paths.manifest_dir, "index", shard_id)
    return out_path


def build_all_indexes(cfg: Config, factory_override: Optional[str] = None) -> List[str]:
    """Build every shard index listed in the manifest (single process)."""
    outputs = []
    for entry in M.read_manifest(cfg.manifest_path):
        outputs.append(build_shard_index(cfg, entry.shard_id, factory_override=factory_override))
    return outputs


def _shard_npy_entries(cfg: Config):
    """All manifest entries whose .npy vectors exist, in shard-id order.

    Raises if none are present (nothing to merge / sample from).
    """
    entries = [e for e in M.read_manifest(cfg.manifest_path) if os.path.exists(e.out_npy)]
    if not entries:
        raise FileNotFoundError(
            f"No shard .npy vectors found (manifest {cfg.manifest_path}); embed first."
        )
    return entries


def _open_shard_memmaps(entries) -> "List[Tuple[object, np.ndarray]]":
    """Open each shard .npy as a read-only memmap; return [(entry, memmap), ...]."""
    out: List[Tuple[object, np.ndarray]] = []
    for e in entries:
        out.append((e, np.load(e.out_npy, mmap_mode="r")))
    return out


def _train_sample_across_shards(
    cfg: Config, memmaps, total_n: int, d: int
) -> np.ndarray:
    """Draw ``index.train_sample`` random vectors across ALL shard memmaps.

    The global sample is a deterministic (seeded by ``run.seed``) random subset
    of the concatenated [0, total_n) index space, then resolved per shard so we
    only ever materialize the sampled rows (memmaps stay on disk otherwise).
    """
    sample_n = min(int(cfg.index.train_sample), total_n)
    rng = np.random.default_rng(cfg.run.seed)
    global_idx = rng.choice(total_n, size=sample_n, replace=False)
    global_idx.sort()

    train = np.empty((sample_n, d), dtype=np.float32)
    write = 0
    offset = 0
    for _entry, mm in memmaps:
        n = mm.shape[0]
        lo = np.searchsorted(global_idx, offset, side="left")
        hi = np.searchsorted(global_idx, offset + n, side="left")
        if hi > lo:
            local = global_idx[lo:hi] - offset
            rows = np.ascontiguousarray(mm[local], dtype=np.float32)
            train[write : write + rows.shape[0]] = rows
            write += rows.shape[0]
        offset += n
    assert write == sample_n, f"train sample fill {write} != {sample_n}"
    return train


def merge_indexes(cfg: Config, mode: Optional[str] = None) -> str:
    """Build the ONE published index (LOCKED CONTRACT D5), or finalize sharded.

    Modes (``index.merge``):

    * ``ondisk`` (default): build the single global
      ``index.factory_full`` (``OPQ64_256,IVF65536_HNSW32,PQ64``) index from all
      shard .npy memmaps. The IVF coarse quantizer is trained on
      ``index.train_sample`` (~2M) random vectors sampled across ALL shards
      (seeded by ``run.seed``); then ``add_with_ids(all vectors, chunk-derived
      int64 ids)`` is called with the ids on the IVF ITSELF (via
      ``faiss.extract_index_ivf``), NOT an outer ``IndexIDMap2`` — so ids
      survive ``write_index``. ``merge_ondisk`` is dropped: one global index
      built directly from the memmaps is correct and simpler at v1 scale.
    * ``sharded``: leave the cheap per-shard Flat indexes as-is and write a small
      JSON manifest (``sharded_index.json``) listing them for scatter-gather.

    Returns the path to the merged index (ondisk) or the shard-list (sharded).
    """
    faiss = _import_faiss()
    mode = mode or cfg.index.merge

    if mode == "sharded":
        import json

        shard_paths = sorted(glob.glob(os.path.join(cfg.faiss_dir, "lang=*", "*.faiss")))
        if not shard_paths:
            raise FileNotFoundError(
                f"No per-shard indexes under {cfg.faiss_dir}; run `index` first."
            )
        out = os.path.join(cfg.faiss_dir, "sharded_index.json")
        ensure_dir(cfg.faiss_dir)
        payload = {"shards": [os.path.relpath(p, cfg.faiss_dir) for p in shard_paths]}
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("Finalized sharded layout (%d shards) -> %s", len(shard_paths), out)
        return out

    if mode != "ondisk":
        raise ValueError(f"Unsupported index.merge mode {mode!r} (expected ondisk|sharded).")

    # ----- the ONE global OPQ-IVF-PQ build straight from the .npy memmaps ----- #
    entries = _shard_npy_entries(cfg)
    memmaps = _open_shard_memmaps(entries)
    d = int(memmaps[0][1].shape[1])
    for entry, mm in memmaps:
        if int(mm.shape[1]) != d:
            raise ValueError(
                f"shard {entry.shard_id}: .npy dim {mm.shape[1]} != {d} (dim mismatch across shards)."
            )
    if d != cfg.model.embed_dim:
        logger.warning(
            "merge: .npy dim %d != model.embed_dim %d (using .npy dim).",
            d,
            cfg.model.embed_dim,
        )
    total_n = int(sum(int(mm.shape[0]) for _e, mm in memmaps))

    factory = cfg.index.factory_full
    metric = _metric(faiss, cfg.index.metric)
    logger.info(
        "merge: building the ONE global FAISS '%s' over %s x %d vectors from %d shards (metric=%s)",
        factory,
        human_int(total_n),
        d,
        len(memmaps),
        cfg.index.metric,
    )
    index = faiss.index_factory(d, factory, metric)

    # Train the coarse quantizer (+ OPQ/PQ codebooks) on a global random sample.
    if not index.is_trained:
        train = _train_sample_across_shards(cfg, memmaps, total_n, d)
        logger.info("merge: training quantizer on %s sampled vectors", human_int(train.shape[0]))
        index.train(train)
        del train

    # Attach chunk-derived int64 ids to the IVF ITSELF (NOT an outer IDMap2), so
    # ids survive serialization. extract_index_ivf reaches through the OPQ
    # pre-transform to the underlying IndexIVF; add_with_ids on the top-level
    # index then routes through the transform while keeping ids on the IVF.
    ivf = faiss.extract_index_ivf(index)
    logger.info("merge: ids carried on the IVF itself (nlist=%d)", ivf.nlist)

    BLOCK = 100000
    added = 0
    for entry, mm in memmaps:
        ids = _load_ids_for_shard(cfg, entry)
        n = int(mm.shape[0])
        if ids.shape[0] != n:
            raise ValueError(
                f"shard {entry.shard_id}: id count {ids.shape[0]} != vector count {n}; "
                f"parquet/.npy disagree."
            )
        for start in range(0, n, BLOCK):
            block = np.ascontiguousarray(mm[start : start + BLOCK], dtype=np.float32)
            index.add_with_ids(block, ids[start : start + block.shape[0]])
        added += n
        logger.info("merge: added shard %d (%s vectors; running total %s)",
                    entry.shard_id, human_int(n), human_int(added))
    assert added == total_n, f"merge: added {added} != total {total_n}"

    ivf_dir = ensure_dir(os.path.join(cfg.faiss_dir, "merged"))
    merged_index_path = os.path.join(ivf_dir, "merged.faiss")
    tmp = merged_index_path + ".tmp"
    faiss.write_index(index, tmp)
    os.replace(tmp, merged_index_path)
    logger.info(
        "merge: built ONE global index (%s vectors, %d shards) -> %s",
        human_int(added),
        len(memmaps),
        merged_index_path,
    )
    return merged_index_path


def load_search_index(cfg: Config):
    """Load the search index for eval and set nprobe.

    Prefers the ONE global merged index (``faiss/merged/merged.faiss``, the
    OPQ-IVF-PQ build whose ids live on the IVF); falls back to the first
    per-shard exact Flat index (IDMap2-wrapped, so it also returns chunk ids)
    when no merged index exists yet. Returns the FAISS index ready for search;
    ``index.search`` yields chunk-derived int64 ids in both cases. Used by the
    eval stage's ANN path.
    """
    faiss = _import_faiss()
    merged = os.path.join(cfg.faiss_dir, "merged", "merged.faiss")
    if os.path.exists(merged):
        index = faiss.read_index(merged)
    else:
        shard_paths = sorted(glob.glob(os.path.join(cfg.faiss_dir, "lang=*", "*.faiss")))
        if not shard_paths:
            raise FileNotFoundError("No FAISS index found; build/merge first.")
        index = faiss.read_index(shard_paths[0])
    # Set nprobe where applicable (IVF indexes).
    try:
        faiss.ParameterSpace().set_index_parameter(index, "nprobe", cfg.index.nprobe)
    except Exception:
        pass
    return index
