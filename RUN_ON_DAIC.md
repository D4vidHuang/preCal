# Running preCal on DAIC

This is the **only page you need tonight.** preCal embeds a permissive code corpus
with `Qwen3-Embedding-4B` (via TEI on the RTX Pro 6000s) and publishes vectors + a
FAISS index + a query↔code retrieval dataset to `D4vidHuang/precal-code-embeddings`.

The flow respects DAIC's split: **online work on a login node** (download / upload),
**GPU work on offline compute nodes** (read staged files only).

---

## TL;DR — kick off the smoke run, then sleep

```bash
# on your laptop: push is already done; just clone on DAIC
ssh daic
git clone https://github.com/D4vidHuang/preCal.git ~/precal && cd ~/precal

# one-time: make sure this node can read the gated corpus (10s, browser):
#   https://huggingface.co/datasets/bigcode/the-stack-dedup  -> "Agree and access"
#   and, if you've never logged in on DAIC:  huggingface-cli login

# one command does the rest (env -> detect GPU -> stage -> submit smoke GPU job).
# run under tmux so it survives an SSH disconnect:
tmux new -s precal 'bash scripts/bootstrap_daic.sh 2>&1 | tee bootstrap.log; bash'
```

When it prints **“bootstrap done — you can sleep now 😴”**, the smoke job is queued.
In the morning:

```bash
squeue --me                                   # job state
cat $PRECAL_SCRATCH/precal-smoke/eval/*.json   # recall@k / MRR@10 / nDCG@10
```

The smoke run uses the small **0.6B** model on ~5k Python files with an exact `Flat`
index — it proves `stage → chunk → shard → embed → index → eval` (and the
checkpoint/resume machinery) end-to-end on **one** GPU before you spend the cluster.

---

## The one genuinely-unknown thing: the RTX Pro 6000 SLURM name

DAIC's public docs only list L40/A40/V100 — the Pro 6000 (Blackwell, sm_120) gres
name isn't documented. You do **not** edit any sbatch file by hand:

- `scripts/daic_autodetect.sh` greps live `sinfo`/`scontrol` for a Pro 6000 /
  Blackwell card + its `--constraint` feature + a GPU partition + your account, and
  writes them to `~/.precal.env`.
- `scripts/submit.sh` reads `~/.precal.env` and passes them as `sbatch` CLI options,
  which override the `#SBATCH` placeholders in the job files.

`bootstrap_daic.sh` runs autodetect for you. If it can't confidently pick the card it
says so and leaves the field blank (jobs then use the in-file placeholder). To check
or fix:

```bash
bash scripts/daic_autodetect.sh --print-only   # show what it detected
cat ~/.precal.env                              # edit PRECAL_GPU_TYPE / PRECAL_CONSTRAINT / PRECAL_PARTITION if needed
# e.g. if `sinfo` shows the card as gpu:rtx_pro_6000:
#   export PRECAL_GRES="gpu:rtx_pro_6000:1"
```

If you'd rather eyeball the raw cluster info first: `make probe` (or
`bash scripts/daic_probe.sh`) prints everything (partitions, gres, features,
quotas, modules, egress).

---

## Full run (after the smoke looks good)

8+ GPUs, the 4B model, all 6 languages, IVF-PQ index, publish to HF. Each step is a
SLURM job; the GPU embed array is requeue/checkpoint-safe and only re-runs shards
that aren't `done`.

```bash
cd ~/precal && source scripts/activate_env.sh

# 1) STAGE the full corpus + 4B model + SIF (login node, online)
bash scripts/submit.sh slurm/stage.sbatch configs/full_v1.yaml      # or: python -m precal.cli stage --config configs/full_v1.yaml

# 2) SHARD: chunk every language, then stamp the global manifest (CPU)
make shard CONFIG=configs/full_v1.yaml

# 3) EMBED: GPU array, one task per shard, %8 concurrent, resumable
make embed CONFIG=configs/full_v1.yaml CONC=%8        # -> scripts/submit.sh --gpu ... slurm/embed.sbatch
#   re-run the SAME command after preemption: it skips shards already 'done'.

# 4) INDEX (per-shard Flat) then MERGE (the ONE full OPQ-IVF-PQ index, fat-RAM node)
make index CONFIG=configs/full_v1.yaml CONC=%8
make merge CONFIG=configs/full_v1.yaml

# 5) EVAL + PUBLISH
make slurm-eval    CONFIG=configs/full_v1.yaml
make slurm-publish CONFIG=configs/full_v1.yaml         # upload_large_folder -> D4vidHuang/precal-code-embeddings
```

`N` (shard count) is read from the manifest automatically by `make embed`/`make index`.

---

## Where things live

| What | Path |
|---|---|
| Per-user DAIC settings (auto-written) | `~/.precal.env` |
| Scratch root (corpus, shards, vectors, index) | `$PRECAL_SCRATCH/<run.name>/` |
| Staged model for TEi | `$PRECAL_HF_HOME/staged/<model_id>` |
| TEI image (SIF) | `$PRECAL_TEI_SIF` |
| Eval report | `$PRECAL_SCRATCH/<run.name>/eval/*.json` |

Architecture and every locked decision: [`DESIGN.md`](DESIGN.md). Overview: [`README.md`](README.md).

---

## If something looks off in the morning

- **Job stuck `PD` (pending) forever** → the GPU gres/partition guess is wrong. Run
  `make probe`, set `PRECAL_GRES` / `PRECAL_PARTITION` / `PRECAL_CONSTRAINT` in
  `~/.precal.env`, resubmit.
- **`stage` 403 / GatedRepoError** → accept the-stack-dedup terms / `huggingface-cli login`.
- **TEI fails to load the model** → confirm `$PRECAL_HF_HOME/staged/<model_id>` exists
  (staging writes it); check `precal-smoke-*.err`.
- **Embed array dies on preemption** → expected; just re-run `make embed …`, it resumes.

> Real CoIR/mteb benchmarking is a tracked TODO — the v1 `eval` is an honest internal
> docstring→code retrieval metric on a leakage-safe repo-level split (see `DESIGN.md`).
