"""INTERNAL docstring->code retrieval eval (held-out repo-split).

HONEST LABEL (D6): this is NOT mteb / CoIR-CSN / CodeSearchNet. It is an
internal docstring->code retrieval probe on a leakage-safe held-out repo split:

  * Queries  = the NL docstrings we extracted from the corpus (the rows in the
    ``eval_test``/``eval_valid`` split that carry a ``query_text``). The relevant
    document for each query is its OWN chunk (its chunk_id).
  * Corpus   = every embedded chunk (the searchable pool).
  * Leakage control: the document body that gets EMBEDDED has its LEADING
    docstring / header comment STRIPPED (precal.chunking.strip_leading_doc) so
    the query (the docstring) is not a verbatim substring of its positive. The
    full original text stays in the parquet ``text`` column; only the embedded
    body is stripped. The split is repo-level (a repo is entirely in one split)
    so no eval positive leaks into the index_only pool.

Two retrieval paths:
  * exact -- brute-force inner product over the corpus, where the corpus
    documents are RE-EMBEDDED here from their STRIPPED bodies. This is the
    leakage-controlled metric and the headline number for the v1 gate.
  * ann   -- search the built FAISS index. NOTE: that index is built from the
    SHIPPED full-text document vectors (the artifact consumers actually query),
    so ANN reflects the real shipped index + PQ compression cost; it is NOT
    docstring-stripped. Reported for visibility, clearly labeled.

Metrics (pure numpy so the module imports clean without mteb): recall@k,
MRR@10, nDCG@10. A clearly-marked TODO stub (``run_mteb_coir``) is left for
wiring the REAL mteb CoIR-Retrieval / CodeSearchNet benchmark later; it imports
mteb lazily and does not claim to run yet.

Outputs a JSON report under cfg.eval_dir.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from precal import manifest as M
from precal.chunking import strip_leading_doc
from precal.config import Config
from precal.engines import build_engine
from precal.utils import (
    chunk_id_to_int64,
    ensure_dir,
    get_logger,
    human_int,
    set_hf_home,
    set_offline,
)

logger = get_logger("precal.eval")

# Honest metric label used in reports / logs / dataset card (D6).
EVAL_LABEL = "internal docstring->code retrieval (held-out repo-split)"


# --------------------------------------------------------------------------- #
# Metric primitives (per-query, single relevant doc)
# --------------------------------------------------------------------------- #
def _dcg_at_k(rank: Optional[int], k: int) -> float:
    """DCG with a single relevant doc at 0-based ``rank`` (None if not in top-k)."""
    if rank is None or rank >= k:
        return 0.0
    return 1.0 / math.log2(rank + 2)  # rel=1


def compute_metrics(
    ranks: List[Optional[int]], ks: List[int]
) -> Dict[str, float]:
    """Aggregate retrieval metrics from per-query ranks of the single relevant doc.

    ``ranks[i]`` = 0-based position of query i's relevant doc in its ranked list,
    or None if it was not retrieved at all. With one relevant doc per query the
    ideal DCG is 1.0, so nDCG@k == DCG@k here.
    """
    n = len(ranks)
    out: Dict[str, float] = {}
    if n == 0:
        return out

    for k in ks:
        hits = sum(1 for r in ranks if r is not None and r < k)
        out[f"recall@{k}"] = hits / n

    # MRR@10
    rr = [1.0 / (r + 1) if (r is not None and r < 10) else 0.0 for r in ranks]
    out["mrr@10"] = sum(rr) / n

    # nDCG@10
    ndcg = [_dcg_at_k(r, 10) for r in ranks]  # idcg = 1.0
    out["ndcg@10"] = sum(ndcg) / n
    return out


# --------------------------------------------------------------------------- #
# Build the eval corpus + queries from the embedded shards
# --------------------------------------------------------------------------- #
def _gather_eval_data(
    cfg: Config, split: str
) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    """Collect (corpus_doc_texts, corpus_chunk_ids, corpus_langs, query_texts,
    query_rel_ids).

    The corpus = every embedded chunk (the searchable pool). For each corpus
    chunk we return its FULL document ``text`` and ``language`` so the exact
    path can re-embed the docstring-STRIPPED body (D6 leakage control). Queries
    = chunks in the requested split (eval_test/eval_valid) that have a non-empty
    query_text; each query's single relevant doc is its own chunk_id.

    Note we no longer load the prebuilt ``.npy`` here for the exact path: those
    vectors were embedded from the FULL text (with the docstring inline) and
    would leak. The ANN path uses the prebuilt FAISS index separately and is
    labeled as the shipped full-text index.
    """
    import pyarrow.parquet as pq

    corpus_texts: List[str] = []
    corpus_ids: List[str] = []
    corpus_langs: List[str] = []
    query_texts: List[str] = []
    query_rel_ids: List[str] = []

    split_key = "eval_" + split if not split.startswith("eval_") else split

    for entry in M.read_manifest(cfg.manifest_path):
        embedded = entry.out_parquet.replace(".parquet", ".embedded.parquet")
        src = embedded if os.path.exists(embedded) else entry.out_parquet
        if not (os.path.exists(src) and os.path.exists(entry.out_npy)):
            continue
        table = pq.read_table(
            src, columns=["chunk_id", "text", "language", "query_text", "eval_split"]
        )
        ids = table.column("chunk_id").to_pylist()
        texts = table.column("text").to_pylist()
        langs = table.column("language").to_pylist()
        qtexts = table.column("query_text").to_pylist()
        splits = table.column("eval_split").to_pylist()

        for i, cid in enumerate(ids):
            corpus_ids.append(cid)
            corpus_texts.append(texts[i] or "")
            corpus_langs.append(langs[i] or "")
            if splits[i] == split_key and qtexts[i]:
                query_texts.append(qtexts[i])
                query_rel_ids.append(cid)

    return corpus_texts, corpus_ids, corpus_langs, query_texts, query_rel_ids


def _embed_stripped_corpus(
    cfg: Config,
    engine,
    corpus_texts: List[str],
    corpus_langs: List[str],
) -> np.ndarray:
    """Re-embed the corpus DOCUMENT bodies with the leading docstring stripped.

    D6 leakage control: the query is a docstring; if it stayed inline in the
    embedded document body it would be a verbatim substring of its positive.
    We strip it (precal.chunking.strip_leading_doc) for the embedded body only;
    parquet ``text`` keeps the full original. Returns [N, embed_dim] float32.
    """
    stripped = [
        strip_leading_doc(t or "", lang or "")
        for t, lang in zip(corpus_texts, corpus_langs)
    ]
    out: List[np.ndarray] = []
    bs = max(1, cfg.engine.batch_size)
    for start in range(0, len(stripped), bs):
        batch = stripped[start : start + bs]
        out.append(engine.embed_documents(batch))
    if out:
        return np.ascontiguousarray(np.vstack(out), dtype=np.float32)
    return np.zeros((0, cfg.model.embed_dim), dtype=np.float32)


def _rank_of_relevant(
    scores_topk_ids: np.ndarray, rel_id_int: int
) -> Optional[int]:
    """0-based rank of rel_id within an ordered id array, else None."""
    where = np.where(scores_topk_ids == rel_id_int)[0]
    return int(where[0]) if where.size else None


def evaluate(
    cfg: Config,
    split: str = "test",
    index_mode: str = "both",
    base_port: int = 7997,
) -> Dict[str, Dict[str, float]]:
    """Run exact and/or ANN retrieval eval and write a JSON report.

    Parameters
    ----------
    split:
        "test" -> eval_test rows, "valid" -> eval_valid rows.
    index_mode:
        "exact" | "ann" | "both".
    """
    set_offline(cfg.hf.offline)
    set_hf_home(cfg.paths.hf_home)

    (
        corpus_texts,
        corpus_ids,
        corpus_langs,
        query_texts,
        query_rel_ids,
    ) = _gather_eval_data(cfg, split)
    n_corpus = len(corpus_texts)
    n_queries = len(query_texts)
    logger.info(
        "eval (%s) split=%s: %s corpus docs, %s queries",
        EVAL_LABEL,
        split,
        human_int(n_corpus),
        human_int(n_queries),
    )
    if n_queries == 0 or n_corpus == 0:
        raise RuntimeError(
            f"No eval data for split={split!r}. Ensure pairs/eval_split were "
            f"assigned during sharding and the shards are embedded."
        )

    # Embed the queries (QUERY wrapper) and, for the exact path, RE-embed the
    # corpus from docstring-STRIPPED bodies (D6 leakage control). Same engine so
    # query and document vectors are self-consistent.
    engine = build_engine(cfg, base_port=base_port)
    try:
        q_vecs = engine.embed_queries(query_texts)  # [Q, d], normalized
        corpus = (
            _embed_stripped_corpus(cfg, engine, corpus_texts, corpus_langs)
            if index_mode in ("exact", "both")
            else np.zeros((0, cfg.model.embed_dim), dtype=np.float32)
        )
    finally:
        engine.close()

    rel_ints = np.asarray([chunk_id_to_int64(c) for c in query_rel_ids], dtype=np.int64)
    corpus_int_ids = np.asarray([chunk_id_to_int64(c) for c in corpus_ids], dtype=np.int64)
    max_k = max(cfg.eval.ks + [10])

    report: Dict[str, Dict[str, float]] = {}

    # ---- exact (brute-force inner product) -------------------------------- #
    if index_mode in ("exact", "both"):
        exact_ranks: List[Optional[int]] = []
        # Block over queries to bound memory: scores = q_vecs @ corpus.T
        BLOCK = 256
        for start in range(0, n_queries, BLOCK):
            qb = q_vecs[start : start + BLOCK]
            sims = qb @ corpus.T  # [b, n_corpus] inner product (cosine)
            topk = np.argpartition(-sims, kth=min(max_k, n_corpus - 1), axis=1)[:, :max_k]
            for bi in range(qb.shape[0]):
                row = topk[bi]
                ordered = row[np.argsort(-sims[bi, row])]
                ranked_ids = corpus_int_ids[ordered]
                exact_ranks.append(
                    _rank_of_relevant(ranked_ids, int(rel_ints[start + bi]))
                )
        report["exact"] = compute_metrics(exact_ranks, cfg.eval.ks)
        logger.info("exact metrics: %s", report["exact"])

    # ---- ANN (the built FAISS IVF-PQ index) ------------------------------- #
    if index_mode in ("ann", "both"):
        try:
            from precal.index import load_search_index

            # NOTE: the prebuilt index is built from the SHIPPED full-text
            # document vectors (NOT docstring-stripped), so ANN reflects the
            # real shipped index + PQ compression cost, not the leakage
            # controlled exact metric above.
            index = load_search_index(cfg)
            ann_ranks: List[Optional[int]] = []
            _, ids = index.search(np.ascontiguousarray(q_vecs, dtype=np.float32), max_k)
            for qi in range(n_queries):
                ann_ranks.append(_rank_of_relevant(ids[qi], int(rel_ints[qi])))
            report["ann"] = compute_metrics(ann_ranks, cfg.eval.ks)
            logger.info("ann metrics (shipped full-text index): %s", report["ann"])
        except Exception as exc:
            logger.warning("ANN eval skipped (%s); run `index` first for ANN metrics.", exc)

    # ---- write report ----------------------------------------------------- #
    ensure_dir(cfg.eval_dir)
    out = os.path.join(cfg.eval_dir, f"report_{split}.json")
    payload = {
        "run": cfg.run.name,
        "eval_kind": EVAL_LABEL,
        # `benchmark` is the configured eval.benchmark string; v1 does NOT run a
        # real mteb/CoIR benchmark (see run_mteb_coir TODO) — it is metadata
        # only, retained for the future real-mteb path.
        "benchmark": cfg.eval.benchmark,
        "is_mteb_coir": False,
        "exact_corpus": "docstring-stripped document bodies (leakage-controlled)",
        "ann_corpus": "shipped full-text document vectors (not stripped)",
        "split": split,
        "model_id": cfg.model.id,
        "model_revision": cfg.model.revision,
        "embed_dim": cfg.model.embed_dim,
        "dtype": cfg.model.dtype,
        "n_corpus": n_corpus,
        "n_queries": n_queries,
        "ks": cfg.eval.ks,
        "metrics": report,
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Wrote eval report -> %s", out)
    return report


# --------------------------------------------------------------------------- #
# TODO (NOT IMPLEMENTED): real mteb CoIR-Retrieval / CodeSearchNet
# --------------------------------------------------------------------------- #
def run_mteb_coir(
    cfg: Config,
    base_port: int = 7997,
) -> Dict[str, Dict[str, float]]:
    """STUB: wire the REAL mteb CoIR-Retrieval / CodeSearchNet benchmark.

    This does NOT run yet and intentionally raises. v1 ships only the internal
    ``evaluate`` probe above (internal docstring->code retrieval, held-out
    repo-split) and makes NO claim about mteb / CoIR-CSN numbers.

    Implementation sketch for the future real path:
      * ``import mteb`` lazily (kept out of module import so eval stays
        import-clean without mteb installed).
      * Load the official task, e.g.
        ``mteb.get_tasks(tasks=["CodeSearchNetRetrieval"])`` (CoIR-Retrieval /
        CodeSearchNet), which brings its OWN frozen corpus / queries / qrels —
        do not reuse our internal docstring split.
      * Wrap our engine as an ``mteb`` encoder (encode -> embed_documents /
        embed_queries with the Qwen3 Instruct/Query wrapper on the query side),
        run ``mteb.MTEB(tasks).run(model)`` and record nDCG@10 / recall@k.
      * The CoIR qrels are already staged under
        ``cfg.corpus_dir/coir-csn/<lang>/`` (see precal.staging /
        precal.pairs.load_eval_repos) for this path.
    """
    raise NotImplementedError(
        "Real mteb CoIR-Retrieval/CodeSearchNet eval is not wired yet (TODO). "
        "v1 reports only the internal docstring->code retrieval probe via "
        "evaluate(); it does NOT produce mteb/CoIR-CSN numbers."
    )
