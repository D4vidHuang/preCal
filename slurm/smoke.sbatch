#!/bin/bash
#SBATCH --job-name=precal-smoke
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# ---------------------------------------------------------------------------
# preCal :: SMOKE as a 1-GPU BATCH job (compute-only; stage already done online)
# ---------------------------------------------------------------------------
# Submit via the wrapper so the GPU placeholders below get overridden from
# ~/.precal.env (scripts/daic_autodetect.sh):
#
#   bash scripts/submit.sh --gpu slurm/smoke.sbatch
#
# Assumes STAGE already ran on a login node (corpus sample + 0.6B model + TEI SIF
# on scratch) — scripts/bootstrap_daic.sh does that for you. This job then runs
# chunk -> shard -> embed (+resume test) -> index (Flat) -> eval, all OFFLINE.
# ---------------------------------------------------------------------------
#SBATCH --partition=general          # PLACEHOLDER — overridden by submit.sh (PRECAL_PARTITION)
#SBATCH --gres=gpu:1                 # PLACEHOLDER — overridden by submit.sh (PRECAL_GPU_TYPE)
#SBATCH --constraint=blackwell       # PLACEHOLDER — overridden by submit.sh (PRECAL_CONSTRAINT)
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail

REPO_ROOT="${PRECAL_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"
echo "[smoke.sbatch] host=$(hostname) job=${SLURM_JOB_ID:-NA}"
nvidia-smi -L || echo "[smoke.sbatch] WARN: nvidia-smi -L failed"

# Staging already done on the login node; this compute node is offline.
export PRECAL_SKIP_STAGE=1
bash "${REPO_ROOT}/scripts/smoke.sh" 0
