# preCal

**preCal** uses spare RTX Pro 6000 (96 GB Blackwell) GPUs on TU Delft's **DAIC**
SLURM cluster to **pre**compute reusable, expensive artifacts in the
software-engineering + LLM space, then publishes them to the Hugging Face Hub
(namespace `D4vidHuang`).

**preCal v1 — Large-scale code embeddings.** Encode a large, permissively
licensed code corpus with a SOTA code-embedding model and publish the embedding
vectors + a FAISS index as a **dual-use RAG / retrieval research asset**: the
schema carries natural-language query ↔ code pairs and a reproducible,
leakage-safe **repo-level** eval split (recall@k / MRR@10 / nDCG@10), not just
raw vectors. (v1 ships an **internal** docstring→code retrieval eval; see the
Eval note below — real `mteb` CoIR-CSN wiring is a tracked TODO, not yet done.)

---

## Data flow

```
corpus ──▶ shard ──▶ embed ──▶ index ──▶ merge ──▶ publish
(login)    (CPU)     (GPU      (CPU/GPU) (fat-RAM) (login/online)
                      array)    array)
                                                    └──▶ eval (1 GPU)
```

| stage      | where it runs                | CLI / job                                  |
|------------|------------------------------|--------------------------------------------|
| **stage**  | login node (online)          | `slurm/stage.sbatch` → `cli stage` + `pull_image.sh` |
| **shard**  | CPU array (1 task/language)  | `slurm/shard.sbatch` → `cli chunk` + `cli shard` |
| **embed**  | GPU array (1 task/shard)     | `slurm/embed.sbatch` → TEI replicas + `cli embed` |
| **index**  | CPU/GPU array (1 task/shard) | `slurm/index.sbatch` → `cli index`         |
| **merge**  | fat-RAM single job           | `slurm/merge_index.sbatch` → `cli merge-index` |
| **eval**   | 1 GPU                        | `slurm/eval.sbatch` → `cli eval`           |
| **publish**| login node (online)          | `slurm/publish.sbatch` → `cli publish`     |

The **shard manifest** (`$PRECAL_SCRATCH/<run.name>/shards/manifest.jsonl`, one
JSON line per shard) is the single source of truth. `SLURM_ARRAY_TASK_ID` maps
**directly** to `shard_id`. Every stage is **idempotent on its shard**, so the
whole array is re-runnable until all shards report `status=done`.

---

## Why this design (key locked decisions)

- **Model:** [`Qwen/Qwen3-Embedding-4B`](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
  (Apache-2.0 → republishable, 4021.8M params, last-token pooling, L2-normalized,
  2560-d MRL-truncatable). Smoke/dry-run uses
  [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
  (identical instruction template → drop-in swap, no schema change).
- **Corpus:** [`bigcode/the-stack-dedup`](https://huggingface.co/datasets/bigcode/the-stack-dedup)
  (gated, permissive subset, `content` inline) for the 6 languages that match the
  eval pack: **python, java, javascript, php, go, ruby**.
- **Engine:** HF Text Embeddings Inference, image
  `ghcr.io/huggingface/text-embeddings-inference:120-1.9` — the **only prebuilt
  sm_120 (Blackwell) image** as of mid-2026. Run from a local apptainer SIF on
  offline compute nodes. Fallbacks: infinity (bf16), vLLM pooling.
- **Eval (v1, honest):** **internal docstring→code retrieval on a held-out
  repo-split** — recall@{1,5,10,100} / MRR@10 / nDCG@10 for **both** exact (Flat)
  and ANN (IVF-PQ). Queries are docstrings we extract from the corpus; the
  corpus is all chunks with the leading docstring **stripped from the embedded
  document body** so the query is not a verbatim substring of its positive. The
  split is **repo-level** (every chunk of a repo is entirely in one split), so no
  positive leaks into the index-only pool. This is **NOT** `mteb` / CoIR-CSN /
  CodeSearchNet — wiring the real
  [`CoIR-Retrieval/CodeSearchNet`](https://huggingface.co/datasets/CoIR-Retrieval/CodeSearchNet)
  benchmark is a tracked TODO (the eval pack is staged for it, but does not yet
  drive the reported numbers).

Full rationale, schema, licensing policy, and open questions live in
[`DESIGN.md`](DESIGN.md).

---

## DAIC offline workflow (important)

Compute nodes are assumed **offline**. All internet I/O is confined to
login/staging nodes:

1. **DOWNLOAD** (online): `slurm/stage.sbatch` accepts the gated `the-stack-dedup`
   terms (once, via the Hub), streams corpus + eval packs to `$PRECAL_SCRATCH`,
   `snapshot_download`s the model into `$HF_HOME`, and `pull_image.sh` fetches the
   TEI SIF.
2. **COMPUTE** (offline): shard / embed / index run with `HF_HUB_OFFLINE=1` and
   `TRANSFORMERS_OFFLINE=1`; TEI runs from the local SIF pointed at the staged
   model snapshot.
3. **UPLOAD** (online): `slurm/publish.sbatch` runs `upload_large_folder`
   (resumable / parallel / Xet) — a dropped connection just resumes.

> **DAIC specifics are unverified.** The public DAIC docs list only
> L40/A40/RTX2080Ti/V100 GPUs and request them via `--gres=gpu:<type>:<count>`
> with large-VRAM nodes tagged by a `gpumem*` **feature** (`--constraint`). The
> RTX Pro 6000 (sm_120) gres name, the right partition, scratch path/quota, the
> container runtime, and the egress policy must be confirmed **before the first
> run** with `scripts/daic_probe.sh`. The sbatch headers carry clearly-marked
> `PLACEHOLDER` lines pointing at that probe.
> (Refs: <https://daic.tudelft.nl/docs/system/compute-nodes/>,
> <https://daic.tudelft.nl/docs/manual/job-submission/slurm-basics/>.)

---

## Quickstart

### 0. One-time setup (login node)

```bash
# Per-user, machine-specific overrides (scratch path, account, GPU gres) go here:
cat > ~/.precal.env <<'EOF'
export PRECAL_SCRATCH=/tudelft.net/staff-umbrella/<your-project>/precal   # confirm with daic_probe.sh
# export PRECAL_ENV_NAME=precal
EOF

bash scripts/daic_probe.sh        # discover partitions / gres / quotas / modules
bash scripts/setup_env.sh         # create the uv venv (.venv) + install requirements
source scripts/activate_env.sh    # sets PRECAL_*, loads CUDA/container modules, activates .venv
```

Then edit the `PLACEHOLDER` `#SBATCH --partition/--gres/--constraint` lines in
`slurm/embed.sbatch` and `slurm/eval.sbatch` using what `daic_probe.sh` reported,
and set `--account` if DAIC requires it.

### 1. Smoke test (one GPU, minutes)

```bash
# On a node with internet for the stage step, then offline for the rest:
make smoke                        # or: bash scripts/smoke.sh 0
```

This runs stage → chunk → shard → embed → index → eval with `configs/smoke.yaml`
(0.6B model, python only, ~5k files, Flat index). It also **kills+reruns** the
embed step to prove idempotent resume. **Pass/fail gate:** the printed
MRR@10 / nDCG@10 on the **internal docstring→code retrieval** split should be
clearly non-trivial (well above a random baseline) for 0.6B; this is a sanity
gate, not a benchmark number comparable to published CoIR-CSN scores.

### 2. Full v1 run (8+ GPUs)

```bash
CFG=configs/full_v1.yaml

# 1) stage (login/online)
sbatch slurm/stage.sbatch $CFG           # wait for the stage-complete marker

# 2) shard (CPU array, one task per language) + global consolidation
make shard CONFIG=$CFG                    # inline, or:
sbatch --array=0-5 slurm/shard.sbatch $CFG
#   then: python -m precal.cli shard --config $CFG   (global shard_id stamping)

# 3) embed (GPU array; resubmits only pending shards, %8 concurrency, --requeue)
make embed CONFIG=$CFG CONC=%8

# 4) index (per-shard FAISS) then 5) merge on a fat-RAM node
make index CONFIG=$CFG CONC=%8
sbatch slurm/merge_index.sbatch $CFG

# 6) eval (exact + ANN) before publishing
sbatch slurm/eval.sbatch $CFG

# 7) publish (login/online)
sbatch slurm/publish.sbatch $CFG
```

Re-running `make embed` / `make index` is always safe: the controller
(`scripts/resubmit_pending.sh`) submits an array restricted to shards whose
`status != done`, and each task is idempotent.

---

## Make targets

```
make help          # list targets
make env           # create/update the uv venv (.venv)
make image         # pull the TEI sm_120 SIF (login node)
make probe         # DAIC discovery (scripts/daic_probe.sh)
make check         # pre-flight: sm_120 kernels, GPU, offline vars, scratch
make smoke         # full end-to-end smoke on one GPU
make stage         # stage corpus+eval+model+SIF (login node)
make shard         # chunk + shard all languages + consolidate manifest
make embed-local   # embed a single shard inline (SHARD=N)
make embed         # sbatch the GPU embed array over pending shards
make index         # sbatch the per-shard FAISS index array
make merge         # sbatch fat-RAM full-index build
make eval          # internal docstring->code retrieval eval (INDEX=exact|ann|both)
make publish       # assemble layout + upload_large_folder (== make upload)
make lint          # shellcheck + ruff (best-effort)
```

---

## HF output layout

Published to **[`D4vidHuang/precal-code-embeddings`](https://huggingface.co/datasets/D4vidHuang/precal-code-embeddings)**
(dataset), Hive-sharded so the viewer stays fast and `upload_large_folder` stays
within Hub limits (<100k files, ≤10k files/folder, files <200 GB):

```
precal-code-embeddings/
├── README.md                      # dataset card + YAML configs (corpus/queries/qrels splits)
├── corpus/   lang=<L>/part-*.parquet   # chunk text + provenance (NO embedding column by default)
├── queries/  lang=<L>/part-*.parquet   # NL query side (query_text, query_source)
├── qrels/    lang=<L>/split=<s>/*.parquet  # frozen eval relevance (index_only|eval_test|eval_valid)
├── vectors/  lang=<L>/part-*.npy        # canonical float32 [N,d] sidecars (memmap)
└── faiss/    <run>/index.faiss (+ ivf shards)   # merged or sharded FAISS index
```

The dataset card records `model_id` / `model_revision` / `embed_dim` /
`pooling=last_token` / `normalized=true` / `dtype` / `corpus_snapshot` and the
license redistribution policy. Parquet rows link to their vector via
`(vector_shard, row_in_shard)`; the `embedding` column is omitted by default
(viewer <5 GB rule) and only emitted for small variants when
`publish.emit_inline_embedding=true`.

---

## Repository layout (ops files)

```
slurm/         stage / shard / embed / index / merge_index / eval / publish sbatch jobs
scripts/       activate_env, setup_env, daic_probe, check_env, pull_image,
               launch_tei_replicas, resubmit_pending, smoke, cfg.py
env/           environment.yml (legacy conda fallback; default env is uv .venv)
configs/       default.yaml / smoke.yaml / full_v1.yaml      (owned by core)
precal/        the pipeline package + CLI                    (owned by core)
Makefile, README.md, DESIGN.md, .gitignore
```

Core CLI contract: `python -m precal.cli {stage,chunk,shard,embed,index,merge-index,publish,eval}`.
