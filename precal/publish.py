"""Assemble the Hive-sharded publish layout, write the dataset card, and upload.

Repo layout (spec decision `hf_repo_layout`) under publish.repo_id
(D4vidHuang/precal-code-embeddings, a *dataset* repo):

    corpus/lang=<l>/part-*.parquet      # browsable code chunks (+ provenance)
    queries/lang=<l>/part-*.parquet     # NL queries (eval + dual-use)
    qrels/lang=<l>/part-*.parquet       # frozen relevance judgments
    vectors/lang=<l>/part-*.npy         # canonical float32 vectors (sidecars)
    faiss/...                           # per-shard + merged index files
    README.md                           # YAML configs: + dataset card

YAML ``configs:`` expose corpus/queries/qrels as selectable viewer splits;
the dataset card records model_id/revision, embed_dim, pooling=last_token,
normalized, dtype, corpus_snapshot, and the license/redistribution policy.

Upload uses ``HfApi().upload_large_folder`` (resumable, parallel, Xet) per the
spec. ``--dry-run`` assembles + validates the layout without uploading.

huggingface_hub is imported lazily.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
from typing import Dict, List, Optional

from precal import manifest as M
from precal.config import Config, to_dict
from precal.schema import describe as schema_describe
from precal.utils import ensure_dir, get_logger

logger = get_logger("precal.publish")


def _link_or_copy(src: str, dst: str) -> None:
    """Hardlink src->dst when possible (same fs, cheap), else copy."""
    ensure_dir(os.path.dirname(dst))
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def assemble_layout(cfg: Config) -> str:
    """Build the Hive-sharded folder under cfg.publish_dir and return its path.

    Splits each shard's embedded parquet into the corpus/queries/qrels views and
    stages the vectors/faiss files into the published layout via hardlinks.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = ensure_dir(cfg.publish_dir)
    entries = M.read_manifest(cfg.manifest_path)
    # Prefix every published filename with the run name so MULTIPLE runs/batches
    # (each a distinct run.name) can coexist in ONE HF repo without clobbering
    # (corpus/lang=python/part-00000.parquet would otherwise collide across runs).
    rn = cfg.run.name

    for entry in entries:
        lang = entry.language
        embedded = entry.out_parquet.replace(".parquet", ".embedded.parquet")
        src_parquet = embedded if os.path.exists(embedded) else entry.out_parquet
        if not os.path.exists(src_parquet):
            logger.warning("shard %d: no parquet to publish (skip).", entry.shard_id)
            continue

        table = pq.read_table(src_parquet)

        # ---- corpus view: the code chunks (the retrieval documents) -------- #
        corpus_dir = ensure_dir(os.path.join(root, "corpus", f"lang={lang}"))
        corpus_out = os.path.join(corpus_dir, f"{rn}-part-{entry.shard_id:05d}.parquet")
        pq.write_table(table, corpus_out + ".tmp", compression="zstd", row_group_size=20000)
        os.replace(corpus_out + ".tmp", corpus_out)

        # ---- queries view: rows that carry an NL query -------------------- #
        if "query_text" in table.column_names:
            mask = pa.compute.not_equal(table.column("query_text"), pa.scalar(""))
            q_table = table.filter(mask)
            if q_table.num_rows:
                q_cols = [
                    c
                    for c in ("chunk_id", "query_text", "query_source", "eval_split", "language")
                    if c in q_table.column_names
                ]
                queries_dir = ensure_dir(os.path.join(root, "queries", f"lang={lang}"))
                q_out = os.path.join(queries_dir, f"{rn}-part-{entry.shard_id:05d}.parquet")
                pq.write_table(q_table.select(q_cols), q_out + ".tmp", compression="zstd")
                os.replace(q_out + ".tmp", q_out)

        # ---- qrels view: query<->chunk relevance for eval rows ------------ #
        if "eval_split" in table.column_names:
            mask = pa.compute.not_equal(table.column("eval_split"), pa.scalar("index_only"))
            ev = table.filter(mask)
            if ev.num_rows:
                qrels = pa.table(
                    {
                        "query_id": ev.column("chunk_id"),
                        "corpus_id": ev.column("chunk_id"),
                        "score": pa.array([1] * ev.num_rows, pa.int32()),
                        "split": ev.column("eval_split"),
                    }
                )
                qrels_dir = ensure_dir(os.path.join(root, "qrels", f"lang={lang}"))
                qr_out = os.path.join(qrels_dir, f"{rn}-part-{entry.shard_id:05d}.parquet")
                pq.write_table(qrels, qr_out + ".tmp", compression="zstd")
                os.replace(qr_out + ".tmp", qr_out)

        # ---- vectors sidecar ---------------------------------------------- #
        if os.path.exists(entry.out_npy):
            v_dst = os.path.join(
                root, "vectors", f"lang={lang}", f"{rn}-{os.path.basename(entry.out_npy)}"
            )
            _link_or_copy(entry.out_npy, v_dst)

    # ---- faiss indexes ---------------------------------------------------- #
    for fpath in glob.glob(os.path.join(cfg.faiss_dir, "**", "*.faiss"), recursive=True):
        rel = os.path.relpath(fpath, cfg.faiss_dir)
        _link_or_copy(fpath, os.path.join(root, "faiss", rn, rel))
    for fpath in glob.glob(os.path.join(cfg.faiss_dir, "**", "*.ivfdata"), recursive=True):
        rel = os.path.relpath(fpath, cfg.faiss_dir)
        _link_or_copy(fpath, os.path.join(root, "faiss", rn, rel))

    # ---- dataset card ----------------------------------------------------- #
    write_dataset_card(cfg, root)
    logger.info("Assembled publish layout -> %s", root)
    return root


def _yaml_configs_block(cfg: Config) -> str:
    """YAML `configs:` block exposing corpus/queries/qrels as viewer splits."""
    return (
        "configs:\n"
        "  - config_name: corpus\n"
        "    data_files:\n"
        '      - split: train\n'
        '        path: "corpus/lang=*/*.parquet"\n'
        "  - config_name: queries\n"
        "    data_files:\n"
        '      - split: train\n'
        '        path: "queries/lang=*/*.parquet"\n'
        "  - config_name: qrels\n"
        "    data_files:\n"
        '      - split: test\n'
        '        path: "qrels/lang=*/*.parquet"\n'
    )


def write_dataset_card(cfg: Config, root: str) -> str:
    """Write README.md with YAML front matter + the dataset card body."""
    from precal import __version__

    corpus_snapshot = f"{cfg.corpus.dataset_id}@{cfg.corpus.revision}"
    front_matter = (
        "---\n"
        "license: other\n"
        "pretty_name: preCal Code Embeddings\n"
        "task_categories:\n"
        "  - feature-extraction\n"
        "  - sentence-similarity\n"
        "tags:\n"
        "  - code\n"
        "  - retrieval\n"
        "  - rag\n"
        "  - embeddings\n"
        "  - faiss\n"
        f"{_yaml_configs_block(cfg)}"
        "---\n"
    )

    body = f"""
# preCal Code Embeddings

Precomputed, reusable code embeddings + FAISS index + NL<->code pairs and
frozen retrieval eval splits, produced by **preCal v{__version__}**.

## Producing model (vectors are model-bound)

| field | value |
|---|---|
| `model_id` | `{cfg.model.id}` |
| `model_revision` | `{cfg.model.revision}` |
| `embed_dim` | `{cfg.model.embed_dim}` |
| `pooling` | `{cfg.model.pooling}` |
| `normalized` | `{cfg.model.normalize}` (L2; cosine via inner product) |
| `dtype` | `{cfg.model.dtype}` |
| `corpus_snapshot` | `{corpus_snapshot}` |
| `languages` | `{', '.join(cfg.corpus.languages)}` |

Queries are encoded with the Qwen3 Instruct/Query wrapper
(`Instruct: {cfg.model.query_instruction}\\nQuery: ...`); code/documents are
encoded RAW. FAISS metric is inner product on L2-normalized vectors.

## Layout

```
corpus/lang=<l>/part-*.parquet   # code chunks (the retrieval documents)
queries/lang=<l>/part-*.parquet  # natural-language queries (dual-use)
qrels/lang=<l>/part-*.parquet    # frozen relevance judgments
vectors/lang=<l>/part-*.npy      # canonical float32 vectors (sidecars)
faiss/...                        # per-shard + merged index files
```

The browsable text lives in parquet; the canonical vectors live in zero-copy
`.npy` sidecars linked from each row via `(vector_shard, row_in_shard)`. The
inline `embedding` column is emitted only for small variants
(`publish.emit_inline_embedding=true`).

## FAISS index

- smoke tier: `{cfg.index.factory_smoke}` (exact, inner product)
- full tier:  `{cfg.index.factory_full}` (IVF-PQ; `nprobe={cfg.index.nprobe}`)

## Eval

v1 ships an **internal docstring->code retrieval (held-out repo-split)** eval.
This is **NOT** `mteb` / CoIR-CSN / CodeSearchNet — those benchmark numbers are
not produced here (real-mteb wiring is a tracked TODO in `precal/eval.py`).

Frozen `eval_split` (index_only | eval_test | eval_valid) is assigned by a
**deterministic repo-level hash partition** (salted by `run.seed`): every chunk
of a repo lands in exactly one split, so no eval positive leaks into the
index-only pool. Queries are docstrings extracted from the corpus; the corpus
is all chunks. To avoid a trivial leak the leading docstring / header comment is
**stripped from the embedded document body** for the exact metric (the full
text is kept here in the `text` column). Reported metrics: recall@{{1,5,10,100}},
MRR@10, nDCG@10 — exact (Flat, leakage-controlled) and ANN (IVF-PQ over the
shipped full-text index).

## License / redistribution policy

Source TEXT is republished ONLY for files whose SPDX license is on the
attribution-light permissive allowlist
(`{', '.join(cfg.corpus.license_allowlist)}`). For all other files
`text_publishable=false` and the published row stores empty text, keeping only
the vector + provenance pointer (reference-only). Vectors derive from
`{corpus_snapshot}`; data-removal/refresh obligations are honored via the
`corpus_snapshot` field.

## Schema

```
{schema_describe()}
```

## Reproduction config

```json
{json.dumps(to_dict(cfg), indent=2)}
```
"""
    out = os.path.join(root, "README.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(front_matter + body)
    logger.info("Wrote dataset card -> %s", out)
    return out


def publish(
    cfg: Config,
    repo_id: Optional[str] = None,
    dry_run: bool = False,
) -> str:
    """Assemble the layout and upload it via upload_large_folder.

    ``--dry-run`` assembles + validates the layout without uploading.
    """
    repo_id = repo_id or cfg.publish.repo_id
    root = assemble_layout(cfg)

    # Basic guardrails from the spec (<100k files, <=10k files/folder).
    n_files = sum(len(files) for _, _, files in os.walk(root))
    logger.info("Publish layout has %d files under %s", n_files, root)
    if n_files > 100000:
        logger.warning("Layout has >100k files; HF recommends staying under 100k.")

    if dry_run:
        logger.info("--dry-run: layout assembled + validated; skipping upload.")
        return root

    try:
        from huggingface_hub import HfApi
    except Exception as exc:  # pragma: no cover
        raise ImportError("publish needs `huggingface_hub`.") from exc

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    logger.info("Uploading %s -> %s (num_workers=%d)", root, repo_id, cfg.publish.num_workers)
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=root,
        num_workers=cfg.publish.num_workers,
    )
    logger.info("Upload complete: https://huggingface.co/datasets/%s", repo_id)
    return root
