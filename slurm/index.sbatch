#!/bin/bash
#SBATCH --job-name=precal-index
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
# ---------------------------------------------------------------------------
# preCal :: INDEX  (per-shard FAISS build; identical shard_id<->array mapping)
# ---------------------------------------------------------------------------
# One task per shard, same mapping as embed:
#   N=$(wc -l < "$PRECAL_SCRATCH/precal-v1/shards/manifest.jsonl")
#   sbatch --array=0-$((N-1))%8 slurm/index.sbatch configs/full_v1.yaml
#
# FAISS build is fully REGENERABLE from the immutable .npy sidecars, so index
# tasks need no mid-build checkpoint — just an atomic write of shard.faiss then
# status=done. v1 defaults to faiss-cpu (see openQuestions re: faiss-gpu sm_120);
# GPU is requested optionally so a faiss-gpu build can use it if available.
# ---------------------------------------------------------------------------

# ------------------------------ SLURM RESOURCES ----------------------------
#SBATCH --partition=compute          # PLACEHOLDER: CPU partition fine for faiss-cpu (daic_probe.sh)
#SBATCH --cpus-per-task=16           # IVF training + add() parallelism
#SBATCH --mem=128G                   # IVF train sample + PQ codebooks held in RAM
#SBATCH --time=06:00:00
#SBATCH --requeue
# To use faiss-gpu (if a sm_120 build is available), add: #SBATCH --gres=gpu:1
# and submit to the gpu partition; the CLI auto-detects a usable GPU.

set -euo pipefail

CONFIG="${1:-configs/full_v1.yaml}"
REPO_ROOT="${PRECAL_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/activate_env.sh"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SHARD_ID="${SLURM_ARRAY_TASK_ID:?index.sbatch must run as a job array (SLURM_ARRAY_TASK_ID unset)}"
echo "[index] host=$(hostname) task=${SHARD_ID} config=${CONFIG}"

# LOCKED CONTRACT (D1/D2): manifests under ${PRECAL_SCRATCH}/${RUN_NAME}/manifests;
# done marker "index-${SHARD_ID}.done" (RAW id) == precal.manifest.mark_done(..., "index", sid).
RUN_NAME="$(python "${REPO_ROOT}/scripts/cfg.py" "$CONFIG" run.name precal-v1)"
STATUS_DIR="${PRECAL_MANIFEST_DIR:-${PRECAL_SCRATCH}/${RUN_NAME}/manifests}"
if [[ -f "${STATUS_DIR}/index-${SHARD_ID}.done" ]]; then
  echo "[index] shard ${SHARD_ID} index already status=done -> nothing to do."; exit 0
fi

# factory defaults to index.factory_smoke/full by run mode (handled inside the CLI).
python -m precal.cli index --config "${CONFIG}" --shard-id "${SHARD_ID}"

echo "[index] shard ${SHARD_ID} FAISS build done."
