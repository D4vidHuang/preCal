#!/bin/bash
#SBATCH --job-name=precal-publish
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# ---------------------------------------------------------------------------
# preCal :: PUBLISH  (login-node: upload_large_folder -> HF dataset)
# ---------------------------------------------------------------------------
# MUST run on an INTERNET-CAPABLE node (login / staging partition) — uploads to HF.
#
#   sbatch slurm/publish.sbatch configs/full_v1.yaml
#   bash   slurm/publish.sbatch configs/full_v1.yaml --dry-run   # validate layout, no upload
#
# Assembles the Hive-sharded corpus/queries/qrels/vectors/faiss layout + README
# YAML configs + dataset card, then runs HfApi().upload_large_folder
# (resumable / parallel / Xet) to D4vidHuang/precal-code-embeddings.
# A dropped connection just resumes — re-run is safe.
# ---------------------------------------------------------------------------

# ------------------------------ SLURM RESOURCES ----------------------------
#SBATCH --partition=general           # PLACEHOLDER: confirm an internet-capable partition
#SBATCH --cpus-per-task=16            # upload_large_folder parallelism (publish.num_workers)
#SBATCH --mem=32G
#SBATCH --time=24:00:00
# No GPU; resumable upload so no --requeue trap needed.

set -euo pipefail

CONFIG="${1:-configs/full_v1.yaml}"
shift || true
EXTRA_ARGS=("$@")   # e.g. --dry-run, --repo-id ...
REPO_ROOT="${PRECAL_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/activate_env.sh"

# Publish node is ONLINE.
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HOME="${PRECAL_HF_HOME}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

echo "[publish] host=$(hostname) config=${CONFIG} extra=${EXTRA_ARGS[*]:-<none>}"
if command -v hf >/dev/null 2>&1; then
  hf auth whoami || echo "[publish] WARN: not authenticated — run 'hf auth login' (D4vidHuang)"
fi

python -m precal.cli publish --config "${CONFIG}" "${EXTRA_ARGS[@]}"

echo "[publish] done."
