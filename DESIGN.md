# preCal — Design

preCal v1 is a SLURM-array code-embedding pipeline producing a **dual-use
RAG/retrieval research artifact**. This document records the architecture, every
locked decision, the published data schema, the sharding/requeue/staging
strategy, the licensing policy, and the open questions that must be resolved on
the live DAIC cluster.

> Authored locally on macOS in `/Users/davidhuang/Desktop/PreCal`; the scaffold
> targets **Linux + CUDA + SLURM** execution on DAIC.

---

## 1. Architecture

```
        ┌────────── online (login / staging node) ───────────┐
corpus  │ accept gated terms → stream the-stack-dedup parquet │
        │ + CoIR-CSN + CodeSearchNet → snapshot_download model│
        │ → apptainer pull TEI sm_120 SIF → stage-complete    │
        └─────────────────────────────────────────────────────┘
                              │  (scratch is the handoff)
        ┌──────────────── offline (compute nodes) ────────────┐
shard   │ CPU array, 1 task/language: tree-sitter symbol      │
        │ chunks + pairs + eval-split → shards/manifest.jsonl │
embed   │ GPU array, 1 task/shard: TEI replicas on 1 card →   │
        │ .npy sidecar + parquet, atomic checkpoint, requeue  │
index   │ CPU array, 1 task/shard: cheap exact Flat per shard │
merge   │ fat-RAM single job: build the ONE full OPQ-IVF-PQ   │
        │ index from all shard .npy memmaps (no merge_ondisk) │
        └─────────────────────────────────────────────────────┘
        ┌──────────────── online (login node) ────────────────┐
publish │ assemble Hive layout + card + YAML → upload_large_folder
eval    │ 1 GPU: INTERNAL docstring→code retrieval (repo-split│
        │ held-out), exact (Flat) + ANN (IVF-PQ); NOT mteb    │
        └─────────────────────────────────────────────────────┘
```

**Container-vs-conda split.** The GPU embed step runs inside the **TEI sm_120
container** (`120-1.9`) — the only prebuilt image with Blackwell kernels — so the
conda env (`env/environment.yml`) deliberately does *not* pin a fragile sm_120
torch wheel. The conda env covers the CPU/host stages (chunking, sharding, FAISS,
eval, publish) and the fallback engines.

**Engine ownership.** `scripts/launch_tei_replicas.sh` starts N TEI replicas on
one card (ports `base..base+N-1`); the core `precal.engines.tei` client discovers
that port range (`PRECAL_TEI_BASE_PORT` / `PRECAL_TEI_REPLICAS`) and round-robins.

---

## 2. Locked decisions

| key | value | why (verified) |
|-----|-------|----------------|
| **model** | `Qwen/Qwen3-Embedding-4B` | Apache-2.0 (republishable, no NC taint); MTEB(Code,v1)=80.06 (~0.6 pt of the 8B); 2560-d MRL-truncatable; 32K ctx; ~8 GB bf16 → 4–8 replicas per 96 GB card. HF: 4021.8M params, license apache-2.0, `text-embeddings-inference` tag, updated 2025-06-20. |
| **fallback_model** | `Qwen3-Embedding-0.6B` (smoke + light variant); `Qwen3-Embedding-8B` (scale-up) | 0.6B = 595.8M params, 1024-d, Apache-2.0, **identical** instruction template → drop-in swap, no schema change. 8B = one-flag scale-up. SFR-Embedding-Code **rejected** for shipped vectors (cc-by-nc-4.0 contaminates republish); eval baseline only. |
| **corpus** | `bigcode/the-stack-dedup` (v1.2, permissive subset, `content` inline) | Largest gated **permissive-only** corpus shipping actual file content, so text+vector+index publish together. v1.1+ excludes MPL/EPL/LGPL; 193 permissive licenses. `the-stack-v2-dedup` is pointer-only (needs SWH S3 + INRIA agreement + egress) → deferred to a v2 path. Gated (viewer 404 without accepted terms). |
| **v1_languages** | python, java, javascript, php, go, ruby | **Exactly** the 6 languages in `CoIR-Retrieval/CodeSearchNet`. Chosen so the bulk slice already matches the eval languages for the **future** real-`mteb` CoIR-CSN path (v1 itself reports only the internal docstring→code eval — see `eval_v1` below). The 11-lang slice is dropped (ts/rust/c/cpp/csharp have no matching CoIR-CSN eval). |
| **chunk_unit** | tree-sitter function/class symbol (`tree-sitter-language-pack==1.8.1`, MIT, py≥3.10, 305 langs) | Symbol-granular chunks match how code retrievers are trained, give clean NL-query↔symbol positives, and are self-contained RAG units. Fallbacks: oversized symbol → sliding window (`chunk.max_tokens` / `chunk.overlap_tokens`); non-parseable → whole-file window. Parsing is **CPU-side**, never on the GPU node. |
| **engine** | TEI, image `ghcr.io/huggingface/text-embeddings-inference:120-1.9` | **Only** prebuilt sm_120 (Blackwell) image as of mid-2026 → avoids the #1 risk (stock torch/flash-attn lack sm_120 kernels). Supports Qwen3-Embedding + last-token pooling + token-based dynamic batching. Fallbacks: `michaelfeil/infinity` (bf16, per-batch checkpoint), vLLM `--runner pooling`. sentence-transformers = reference only. |
| **dtype** | bf16 canonical; **fp16 when served by TEI** | Qwen3 is bf16-trained; Blackwell is bf16-native. TEI exposes **only** `float16|float32` (no bf16, no fp8). So when TEI is the production engine, vectors are fp16 and eval-reference vectors **must be recomputed with TEI itself** for self-consistent recall@k. fp8 is gated behind a validate-first flag (block if >0.3–0.5 pt nDCG drop). |
| **storage_format** | **hybrid per shard**: parquet (metadata+text, no embedding col by default) + sidecar `.npy` float32 `[N,d]` C-contiguous | HF viewer only sorts/filters the first 5 GB and chokes on wide `list<float32>` columns; keeping browsable text in parquet and canonical vectors in a zero-copy memmap `.npy` gives FAISS fast train/add without an Arrow round-trip. `(vector_shard, row_in_shard)` links parquet→npy. row_group 10k–50k. |
| **faiss_index_smoke** | `Flat`, `METRIC_INNER_PRODUCT` (exact) | Smoke ~50–150k chunks (<1M) → exact Flat with inner product on L2-normalized vectors (= cosine). Gives the exact-recall ceiling to validate ANN against. |
| **faiss_index_full** | ONE global `OPQ64_256,IVF65536_HNSW32,PQ64` (1M–50M tier) on inner product, built at **merge** from all shard `.npy` memmaps (D5); per-shard stage builds only cheap exact `Flat` | v1 target ~15–40M chunks → 1M–50M tier. The full index is **not** built per shard: at merge we train the IVF coarse quantizer on `index.train_sample` (~2M) random vectors drawn across all shard memmaps, then `add_with_ids(all vectors, chunk-derived int64 ids)` with ids on the **IVF itself** (no outer `IndexIDMap2`, so ids survive). `merge_ondisk` is **dropped** — at v1 scale one global index built directly from the memmaps is correct and simpler. OPQ+PQ64 compresses 2560-d float32 (~10 KB) to 64 bytes (~160×) so the full index fits in host RAM. Auto-promote to IVF262144/IVF1048576 via config beyond v1. |
| **hf_repo_layout** | `D4vidHuang/precal-code-embeddings` (dataset), Hive-sharded corpus/queries/qrels/vectors/faiss + README YAML configs + Croissant | `upload_large_folder` (resumable/parallel/Xet) is the documented path for many prebuilt files; keep <100k files total, ≤10k files/folder, split files <200 GB. YAML `configs:` exposes corpus/queries/qrels as viewer splits. Authenticated as D4vidHuang. |
| **eval_v1** (honest) | **INTERNAL docstring→code retrieval** on a held-out **repo-level** split; recall@{1,5,10,100} / MRR@10 / nDCG@10, exact (Flat) + ANN (IVF-PQ) | This is **NOT** `mteb` / CoIR-CSN / CodeSearchNet. Queries = docstrings we extract from the corpus; corpus = all chunks. To avoid a trivial leak, the leading docstring / header comment is **stripped from the embedded document body** (the full text is still kept in the parquet `text` column) so a query is not a verbatim substring of its positive. Eval queries come from the `eval_test` split only; the corpus is all chunks; the split is repo-level so no leakage. **TODO (stubbed in `precal/eval.py`):** wire real `mteb` CoIR-Retrieval / CodeSearchNet using the qrels already staged in `corpus/coir-csn/<lang>/`. |

---

## 3. Published data schema

One parquet row per chunk; the canonical vector lives in the sidecar `.npy`
addressed by `(vector_shard, row_in_shard)`. **Bold** = dual-use RAG/retrieval
fields.

| column | type | description |
|--------|------|-------------|
| `chunk_id` | string | **PK** = blake2b hex of normalized chunk text + repo + path + span. Idempotency key for resume; FAISS id source; doubles as corpus `_id` for RAG. |
| `repo_name` | string | Source repo (owner/name) — provenance + repo-level split partitioning. |
| `path` | string | File path within the repo — provenance + dedup. |
| `language` | string | `python\|java\|javascript\|php\|go\|ruby` — drives per-language sharding + FAISS shards. |
| `license` | string | SPDX id from Stack `detected_licenses` — republish gate for TEXT. |
| `text_publishable` | bool | True if `license` ∈ allowlist; when False the published parquet stores empty text and keeps only vector+pointer+provenance (reference-only). |
| `symbol_kind` | string | `function\|method\|class\|module\|window\|whole_file` — extraction path / fallback. |
| `symbol_name` | string | Extracted name (empty for window/whole_file). |
| `start_line` / `end_line` | int32 | 1-based span in the source file. |
| `n_tokens` | int32 | Token count under the model tokenizer. |
| `truncated` | bool | True if the chunk exceeded `chunk.max_tokens` and was windowed/truncated. |
| **`text`** | string | The chunk source code (DOCUMENT side; encoded **raw**, no prefix). Empty when `text_publishable=False`. The RAG retrieval unit. |
| **`query_text`** | string | NL query paired to this chunk (docstring / CSN doc / commit subject). QUERY side; encoded with the Instruct/Query wrapper. |
| **`query_source`** | string | `docstring\|codesearchnet\|coir\|commit\|none` — lets consumers filter noisy queries. |
| **`eval_split`** | string | `index_only\|eval_test\|eval_valid`. **Required** dual-use field; assigned by a **deterministic repo-level hash partition** (salted by `run.seed`), so every chunk of a repo lands in exactly one split and no eval positive can leak into `index_only`. A real CoIR `eval_repos.txt` is **never produced in v1**, so v1 does **not** use CoIR test qrels to drive the split (the qrels are merely staged for the future real-`mteb` path); see `precal/pairs.py`. |
| `vector_shard` | string | Filename of the `.npy` holding this row's vector (e.g. `vectors/lang=python/part-00007.npy`). |
| `row_in_shard` | int32 | 0-based row index within `vector_shard`. |
| `embedding` | list<float32> | **Optional** inline vector (dim = `embed_dim`); omitted by default (viewer <5 GB), written only when `publish.emit_inline_embedding=true`. |
| `model_id` | string | Producing model HF id. Vectors are model-bound. |
| `model_revision` | string | HF commit hash of the model snapshot. |
| `embed_dim` | int32 | Dimensionality after any MRL truncation (2560 / 1024 / 768). |
| `pooling` | string | `last_token`. |
| `normalized` | bool | True (L2-normalized → cosine via inner product). Always true in v1. |
| `dtype` | string | `bfloat16\|float16` (float16 when produced by TEI). |
| `corpus_snapshot` | string | Source corpus + snapshot (e.g. `bigcode/the-stack-dedup@<revision>`) for data-removal/refresh obligations. |

---

## 4. Sharding scheme

**Two-level.** Level 1 = **language** (`corpus.languages`). Level 2 = fixed-size
chunk shards of ~`shard.target_chunks` (default 250k) within each language, so a
busy language (python) yields many shards, a small one (ruby) few.

The `shard` stage writes `shards/manifest.jsonl`, one JSON line per shard:

```json
{"shard_id": 7, "language": "python", "input_files": ["..."],
 "approx_chunks": 250000, "out_parquet": "...", "out_npy": "...", "status": "pending"}
```

- `shard_id` is **global, contiguous** (0..N-1) across all languages.
- `SLURM_ARRAY_TASK_ID` maps **directly** to `shard_id`:
  `sbatch --array=0-$((N-1))%8 slurm/embed.sbatch` → each task runs
  `precal.cli embed --shard-id $SLURM_ARRAY_TASK_ID`.
- **One task per GPU**; intra-task parallelism is `engine.replicas_per_gpu`
  internal TEI replicas (NOT multiple array tasks per card).
- The manifest is the single source of truth; a `manifest.lock` + per-shard
  status file (`pending/running/done` in `paths.manifest_dir`) lets the controller
  (`scripts/resubmit_pending.sh`) skip done shards on resubmit. **Index uses the
  identical mapping.**

v1: ~3–5M files → ~15–40M chunks → ~60–160 shards at 250k/shard — matches an 8+
GPU array with `%8` and requeue. (N is provisional; recompute after smoke.)

---

## 5. Requeue / idempotency strategy

- sbatch use `#SBATCH --requeue` + `#SBATCH --signal=B:USR1@120`. In
  `slurm/embed.sbatch` a `trap on_usr1 USR1` forwards `TERM` to the python driver
  so it **flushes the current batch** before exit, then returns non-zero so Slurm
  requeues the task.
- **Idempotency is per `chunk_id`, not per shard.** `precal.manifest` keeps a
  per-shard committed-ids log (`paths.manifest_dir/shard-<id>.committed`). The
  embed loop:
  1. on start, scans the committed log + existing `.npy` length and **skips**
     already-embedded `chunk_id`s,
  2. embeds the remainder in batches of `engine.batch_size`,
  3. every `embed.checkpoint_every` chunks writes vectors to a tmp `.npy`,
     **fsync**s, atomically renames/appends, then appends the new `chunk_id`s to
     the committed log and fsyncs.
  **Vectors-before-ids ordering** guarantees no id is marked committed without
  its vector on disk → on requeue the task resumes exactly where it stopped, with
  no duplicate or missing vectors.
- A shard goes `pending → running → done`; `done` is written **only** after the
  `.npy` row count reconciles with `approx_chunks` (within the dropped-`min_tokens`
  tolerance) and parquet/npy row counts match.
- `make embed` / `make index` (via `resubmit_pending.sh`) submit an array
  restricted to shards with `status != done`, so the pipeline is re-runnable until
  all shards are done.
- **FAISS index build is fully regenerable** from the immutable `.npy`, so index
  tasks need no mid-build checkpoint — just an atomic write of `shard.faiss` then
  `status=done`.

---

## 6. Staging strategy (offline compute)

Compute nodes are assumed **offline**; all internet I/O is confined to
login/staging nodes.

- **DOWNLOAD** (`slurm/stage.sbatch`, online): set `HF_HOME=paths.hf_home`,
  authenticate (token already authed as D4vidHuang), accept the
  `the-stack-dedup` gated terms once via the Hub. `precal.staging` streams
  per-language the-stack-dedup parquet (`content` inline) + CoIR-CSN
  (corpus/queries/qrels) + `code-search-net/code_search_net` to
  `paths.scratch`, and `snapshot_download`s `Qwen/Qwen3-Embedding-4B` (+0.6B for
  smoke) into a **materialized** local dir `${HF_HOME}/staged/<model_id>` (real
  files, `local_dir_use_symlinks=False` — the model_id keeps its slash, e.g.
  `staged/Qwen/Qwen3-Embedding-4B`). `scripts/pull_image.sh` pulls the TEI SIF.
  A **stage-complete marker** gates downstream jobs.
- **COMPUTE** (shard/embed/index): `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`
  (`hf.offline=true`); engines load the model strictly from `HF_HOME`; TEI runs
  from the local SIF with `--model-id ${HF_HOME}/staged/<model_id>` pointing at
  that materialized snapshot dir.
- **UPLOAD** (`slurm/publish.sbatch`, online): after merge,
  `HfApi().upload_large_folder(repo_id, repo_type='dataset', num_workers=16)` from
  the assembled scratch layout — resumable/Xet, so a dropped connection resumes.
  **No compute node ever needs egress.**

---

## 7. Licensing / redistribution policy

- **Vectors** are derived artifacts of permissively licensed code; the corpus is
  the permissive-only `the-stack-dedup`, and the model is Apache-2.0 → no NC taint.
- **Text redistribution is gated.** `corpus.license_allowlist`
  (`MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, 0BSD, Unlicense, CC0-1.0`)
  decides `text_publishable`. When False, the published parquet stores **empty
  text** and keeps only vector + pointer + provenance (reference-only) — the
  vector is still useful for retrieval research without redistributing the source.
- `corpus_snapshot` (corpus id @ revision) is stored per row to honor
  data-removal / refresh obligations.
- SFR-Embedding-Code (cc-by-nc-4.0) is **never** used to produce shipped vectors —
  eval baseline only.

---

## 8. Open questions (resolve before / during first run)

1. **DAIC specifics unconfirmed.** Exact scratch quota/path (`$TMPDIR` vs shared
   project scratch), apptainer vs singularity availability, the Blackwell
   driver/CUDA version on the embed partition, and the login-node egress policy.
   `scripts/daic_probe.sh` + `scripts/check_env.sh` must probe these **before the
   first array**. Public DAIC docs list only L40/A40/RTX2080Ti/V100 and use
   `--gres=gpu:<type>:<count>` + `gpumem*` features — the RTX Pro 6000 gres name
   and sm_120 constraint are unknown until probed. (Refs:
   <https://daic.tudelft.nl/docs/system/compute-nodes/>,
   <https://daic.tudelft.nl/docs/manual/job-submission/slurm-basics/>.)
2. **Stack v1.2 row schema** field names for repo/path/`detected_licenses` must be
   confirmed against the actual gated parquet once terms are accepted (viewer 404s
   without acceptance). `precal/staging.py` + schema mapping depend on it.
3. **Eval-language mismatch (confirmed):** CoIR-CSN covers only the 6 languages, no
   ts/rust/c/cpp/csharp → the v1 bulk slice was reduced to those 6. Revisit if a
   multi-language eval (full CoIR 10-subset) is wanted.
4. **Empirical symbol/file and token/chunk ratios** per language are unknown until
   smoke → the 15–40M chunk estimate and shard count N are provisional; recompute
   after smoke.
5. **fp8 (E4M3) go/no-go** for the full run pends the bf16-vs-fp8 nDCG delta on the
   held-out split (block if >0.3–0.5 pt).
6. **faiss-gpu sm_120 availability** on DAIC (conda-forge `faiss-gpu-raft` vs source
   build) is unconfirmed → v1 defaults to **faiss-cpu** for index build, with GPU as
   an optimization (`scripts/setup_env.sh --faiss-gpu`).
7. **5th research strand** (DAIC env specifics, delivered out-of-band) may contain
   login-node hostnames / partition / account names that would let us hardcode the
   sbatch headers — fold those into `~/.precal.env` and the `PLACEHOLDER` lines.

---

## 9. Smoke vs full at a glance

| | smoke (`configs/smoke.yaml`) | full v1 (`configs/full_v1.yaml`) |
|---|---|---|
| model | Qwen3-Embedding-0.6B | Qwen3-Embedding-4B |
| languages | python | python, java, javascript, php, go, ruby |
| files/lang | 5000 | 0 (or capped to ~3–5M total) |
| target_chunks | 75 000 | 250 000 |
| index | `Flat` (exact) | per-shard `Flat`, then ONE full `OPQ64_256,IVF65536_HNSW32,PQ64` built at merge from all `.npy` |
| replicas/GPU | 2 | 4 |
| scale | 1 GPU, minutes | 8+ GPUs, ~1–2 wall-days (calibrate on one real shard first) |
| gate | MRR@10/nDCG@10 on the internal docstring→code split clearly above random (sanity, not a CoIR-CSN number) | internal-eval quality + PQ compression cost acceptable before publish |
