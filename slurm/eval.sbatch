#!/bin/bash
#SBATCH --job-name=precal-eval
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# ---------------------------------------------------------------------------
# preCal :: EVAL  (1-GPU: INTERNAL docstring->code retrieval, exact Flat + ANN IVF-PQ)
# ---------------------------------------------------------------------------
#   sbatch slurm/eval.sbatch configs/full_v1.yaml            # both exact + ann
#   sbatch slurm/eval.sbatch configs/full_v1.yaml exact      # exact only
#   sbatch slurm/eval.sbatch configs/full_v1.yaml ann test   # index + split
#
# HONEST LABEL (D6): this is an INTERNAL docstring->code retrieval eval on a
# held-out repo-split (queries = our own extracted docstrings; corpus = all
# chunks with the leading docstring stripped from the embedded document body).
# It is NOT mteb / CoIR-CSN / CodeSearchNet — that real benchmark wiring is a
# TODO (see precal/eval.py). Reports recall@{1,5,10,100}, MRR@10, nDCG@10 for
# BOTH exact (Flat) and ANN (IVF-PQ) so the PQ compression cost is visible
# BEFORE publish. Needs a GPU to embed the held-out QUERIES with the same
# engine/dtype.
# ---------------------------------------------------------------------------

# ------------------------------ SLURM RESOURCES ----------------------------
# See slurm/embed.sbatch for the DAIC gres/constraint discovery note (daic_probe.sh).
#SBATCH --partition=general           # PLACEHOLDER: confirm the GPU partition (daic_probe.sh)
#SBATCH --gres=gpu:1                  # PLACEHOLDER: pin the card type, e.g. gpu:rtx_pro_6000:1
#SBATCH --constraint=blackwell        # PLACEHOLDER: RTX Pro 6000 feature, e.g. blackwell / gpumem96 / sm_120
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue

set -euo pipefail

CONFIG="${1:-configs/full_v1.yaml}"
INDEX_KIND="${2:-both}"     # exact | ann | both
SPLIT="${3:-test}"          # test | valid
REPO_ROOT="${PRECAL_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/activate_env.sh"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${PRECAL_HF_HOME:-${PRECAL_SCRATCH}/hf}"

echo "[eval] internal docstring->code retrieval (held-out repo-split) — NOT mteb/CoIR-CSN"
echo "[eval] host=$(hostname) config=${CONFIG} index=${INDEX_KIND} split=${SPLIT}"
nvidia-smi -L || echo "[eval] WARN: nvidia-smi -L failed"

bash "${REPO_ROOT}/scripts/check_env.sh" --quick || echo "[eval] WARN: check_env quick checks failed"

# If TEI is the production engine, eval-reference query vectors MUST be produced by
# TEI too (fp16) for self-consistent recall@k. Launch a single replica if needed.
ENGINE="$(python "${REPO_ROOT}/scripts/cfg.py" "$CONFIG" engine.name)"
TEI_PIDS=()
cleanup() { if [[ ${#TEI_PIDS[@]} -gt 0 ]]; then kill "${TEI_PIDS[@]}" 2>/dev/null || true; fi; }
trap cleanup EXIT
if [[ "${ENGINE}" == "tei" ]]; then
  MODEL_ID="$(python "${REPO_ROOT}/scripts/cfg.py" "$CONFIG" model.id)"
  TEI_PID_FILE="$(mktemp)"
  TEI_PID_FILE="$TEI_PID_FILE" \
    bash "${REPO_ROOT}/scripts/launch_tei_replicas.sh" \
      --model-id "${HF_HOME}/staged/${MODEL_ID}" --replicas 1 \
      --base-port "${PRECAL_TEI_BASE_PORT:-7997}" --sif "${PRECAL_TEI_SIF}"
  mapfile -t TEI_PIDS < "$TEI_PID_FILE"; rm -f "$TEI_PID_FILE"
  export PRECAL_TEI_BASE_PORT="${PRECAL_TEI_BASE_PORT:-7997}" PRECAL_TEI_REPLICAS=1
fi

python -m precal.cli eval --config "${CONFIG}" --split "${SPLIT}" --index "${INDEX_KIND}"

echo "[eval] done. report json written under \$PRECAL_SCRATCH (path printed by the CLI)."
