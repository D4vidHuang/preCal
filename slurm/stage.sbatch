#!/bin/bash
#SBATCH --job-name=precal-stage
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# ---------------------------------------------------------------------------
# preCal :: STAGE  (corpus + eval packs + model weights + TEI image -> scratch)
# ---------------------------------------------------------------------------
# MUST run on an INTERNET-CAPABLE node (login / dedicated staging partition).
# Compute nodes are assumed OFFLINE, so ALL egress is confined here.
#
# Launch:
#   sbatch slurm/stage.sbatch configs/full_v1.yaml
#   # or run directly on a login node WITHOUT slurm:
#   bash slurm/stage.sbatch configs/full_v1.yaml
#
# Produces (under $PRECAL_SCRATCH):
#   the-stack-dedup per-language parquet, CoIR-CSN + CodeSearchNet eval packs,
#   snapshot_download'd model weights in $HF_HOME, tei_120-1.9.sif, and a
#   stage-complete marker that gates every downstream job.
# ---------------------------------------------------------------------------

# ------------------------------ SLURM RESOURCES ----------------------------
#SBATCH --partition=general          # PLACEHOLDER: confirm an internet-capable partition (daic_probe.sh)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
# No GPU, no --requeue: staging is download-bound and resumes via HF cache + idempotent CLI.

set -euo pipefail

CONFIG="${1:-configs/full_v1.yaml}"
REPO_ROOT="${PRECAL_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/activate_env.sh"

# Staging node is ONLINE — make sure offline flags are OFF here.
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HOME="${PRECAL_HF_HOME}"

echo "[stage] host=$(hostname) config=${CONFIG}"
echo "[stage] HF_HOME=${HF_HOME} SCRATCH=${PRECAL_SCRATCH}"

# Confirm HF auth (token already authed as D4vidHuang). Non-fatal warn if CLI shape differs.
if command -v hf >/dev/null 2>&1; then
  hf auth whoami || echo "[stage] WARN: 'hf auth whoami' failed — ensure token is set (HF_TOKEN / hf auth login)"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli whoami || echo "[stage] WARN: huggingface-cli whoami failed — ensure token is set"
fi
echo "[stage] NOTE: gated bigcode/the-stack-dedup terms must be accepted ONCE for D4vidHuang"
echo "[stage]       at https://huggingface.co/datasets/bigcode/the-stack-dedup before first run."

# 1) Corpus + eval packs + model weights (idempotent; re-running skips present files).
python -m precal.cli stage --config "${CONFIG}"

# 2) TEI Blackwell SIF (apptainer pull on this online node).
bash "${REPO_ROOT}/scripts/pull_image.sh" --config "${CONFIG}"

echo "[stage] DONE. stage-complete marker should now exist under \$PRECAL_SCRATCH (written by precal.cli stage)."
