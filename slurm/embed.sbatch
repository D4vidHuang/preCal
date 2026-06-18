#!/bin/bash
#SBATCH --job-name=precal-embed
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
# ---------------------------------------------------------------------------
# preCal :: GPU EMBED ARRAY  (one SLURM_ARRAY_TASK_ID == one shard_id)
# ---------------------------------------------------------------------------
# Launch:
#   N=$(wc -l < "$PRECAL_SCRATCH/precal-v1/shards/manifest.jsonl")
#   sbatch --array=0-$((N-1))%8 slurm/embed.sbatch configs/full_v1.yaml
#
# Resubmit (controller skips shards already status=done):
#   bash scripts/resubmit_pending.sh embed configs/full_v1.yaml   # see Makefile `embed`
#
# Each task:
#   * pins exactly ONE GPU (Slurm --gres=gpu:1 -> CUDA_VISIBLE_DEVICES already set)
#   * launches engine.replicas_per_gpu TEI replicas from the staged SIF
#   * runs `python -m precal.cli embed --shard-id $SLURM_ARRAY_TASK_ID`
#   * is --requeue safe: traps USR1 (sent 120s before preemption) -> graceful flush
#   * is idempotent: committed-id manifest means a requeued task resumes mid-shard
# ---------------------------------------------------------------------------

# ------------------------------ SLURM RESOURCES ----------------------------
# DAIC convention (confirmed June 2026, https://daic.tudelft.nl/docs/system/compute-nodes/):
#   GPUs are requested as --gres=gpu:<type>:<count> (e.g. gpu:a40:1) and large-VRAM
#   nodes carry a memory FEATURE used with --constraint (e.g. gpumem32). Public docs
#   list ONLY l40/a40/turing/v100 — the RTX Pro 6000 (Blackwell, sm_120) gres + feature
#   are NOT documented and MUST be discovered with scripts/daic_probe.sh, then pasted here.
#SBATCH --partition=general              # PLACEHOLDER: confirm the GPU partition (daic_probe.sh: sinfo)
#SBATCH --gres=gpu:1                     # PLACEHOLDER: pin the card type, e.g. gpu:rtx_pro_6000:1
# (no --constraint: the gres gpu:nvidia_rtx_pro_6000 already pins the card; submit.sh injects --constraint only if PRECAL_CONSTRAINT is set)
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --requeue
# Send SIGUSR1 to the BATCH shell (B:) 120s before the time/preemption kill.
#SBATCH --signal=B:USR1@120
# NOTE: #SBATCH --account=<your-account>  # PLACEHOLDER: set if DAIC requires -A (sacctmgr show assoc)

set -euo pipefail

# ------------------------------ ARGS / CONFIG ------------------------------
CONFIG="${1:-configs/full_v1.yaml}"
REPO_ROOT="${PRECAL_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"

SHARD_ID="${SLURM_ARRAY_TASK_ID:?embed.sbatch must run as a job array (SLURM_ARRAY_TASK_ID unset)}"

echo "[embed] host=$(hostname) job=${SLURM_JOB_ID:-NA} array=${SLURM_ARRAY_JOB_ID:-NA} task=${SHARD_ID}"
echo "[embed] config=${CONFIG} repo_root=${REPO_ROOT}"
echo "[embed] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || echo "[embed] WARN: nvidia-smi -L failed"

# ------------------------------ ENVIRONMENT --------------------------------
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/activate_env.sh"

# Compute nodes are OFFLINE: load model strictly from staged HF_HOME (hf.offline=true).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${PRECAL_HF_HOME:-${PRECAL_SCRATCH}/hf}"
export TOKENIZERS_PARALLELISM=false

# Read engine + replica + model settings from the SAME YAML the core CLI uses.
# scripts/cfg.py is an ops-side read-only helper (PyYAML + ${ENV} interpolation);
# it never mutates config and never bypasses the core CLI for actual work.
ENGINE="$(python "${REPO_ROOT}/scripts/cfg.py" "$CONFIG" engine.name)"
REPLICAS="$(python "${REPO_ROOT}/scripts/cfg.py" "$CONFIG" engine.replicas_per_gpu)"
MODEL_ID="$(python "${REPO_ROOT}/scripts/cfg.py" "$CONFIG" model.id)"
RUN_NAME="$(python "${REPO_ROOT}/scripts/cfg.py" "$CONFIG" run.name precal-v1)"
# Unique base port PER ARRAY TASK: DAIC packs 2 Pro 6000 per node, so two array
# tasks share a node and would otherwise both bind 7997/7998 -> collision ->
# "connection refused" on a replica. Offset by task id (8 ports apart, > replicas).
BASE_PORT="${PRECAL_TEI_BASE_PORT:-$(( 7997 + ( ${SLURM_ARRAY_TASK_ID:-0} % 120 ) * 8 ))}"
SIF_PATH="${PRECAL_TEI_SIF:-${PRECAL_SCRATCH}/tei_120-1.9.sif}"

echo "[embed] engine=${ENGINE} replicas=${REPLICAS} model=${MODEL_ID}"

# ------------------------------ PRE-FLIGHT ---------------------------------
bash "${REPO_ROOT}/scripts/check_env.sh" --quick || {
  echo "[embed] FATAL: check_env.sh failed (sm_120 kernels / GPU / offline / scratch)"; exit 3; }

# Skip shards already finished (cheap guard; the CLI is idempotent regardless).
# LOCKED CONTRACT (D1/D2): manifests live under ${PRECAL_SCRATCH}/${RUN_NAME}/manifests
# and the done marker is "embed-${SHARD_ID}.done" (RAW id, no zero-pad) — matching
# precal.manifest.mark_done(manifest_dir, "embed", shard_id).
STATUS_DIR="${PRECAL_MANIFEST_DIR:-${PRECAL_SCRATCH}/${RUN_NAME}/manifests}"
if [[ -f "${STATUS_DIR}/embed-${SHARD_ID}.done" ]]; then
  echo "[embed] shard ${SHARD_ID} already status=done -> nothing to do."
  exit 0
fi

# ------------------------------ TEI REPLICAS -------------------------------
TEI_PIDS=()
cleanup_tei() {
  if [[ "${ENGINE}" == "tei" && ${#TEI_PIDS[@]} -gt 0 ]]; then
    echo "[embed] stopping ${#TEI_PIDS[@]} TEI replica(s): ${TEI_PIDS[*]}"
    kill "${TEI_PIDS[@]}" 2>/dev/null || true
    wait "${TEI_PIDS[@]}" 2>/dev/null || true
  fi
}

if [[ "${ENGINE}" == "tei" ]]; then
  echo "[embed] launching ${REPLICAS} TEI replica(s) from ${SIF_PATH} starting at port ${BASE_PORT}"
  # launch_tei_replicas.sh backgrounds the replicas and writes PIDs to $TEI_PID_FILE.
  TEI_PID_FILE="$(mktemp)"
  TEI_PID_FILE="$TEI_PID_FILE" \
    bash "${REPO_ROOT}/scripts/launch_tei_replicas.sh" \
      --model-id "${HF_HOME}/staged/${MODEL_ID}" \
      --replicas "${REPLICAS}" \
      --base-port "${BASE_PORT}" \
      --sif "${SIF_PATH}"
  mapfile -t TEI_PIDS < "$TEI_PID_FILE"
  rm -f "$TEI_PID_FILE"
  # Tell the driver the EXACT endpoints (client honors PRECAL_TEI_ENDPOINTS) so it
  # uses the SAME unique ports we launched on, never the default 7997 (which both
  # mismatches the offset launch AND collides when 2 tasks share a node).
  export PRECAL_TEI_BASE_PORT="${BASE_PORT}"
  export PRECAL_TEI_REPLICAS="${REPLICAS}"
  EPS=""; for ((p=BASE_PORT; p<BASE_PORT+REPLICAS; p++)); do EPS="${EPS:+$EPS,}http://127.0.0.1:$p"; done
  export PRECAL_TEI_ENDPOINTS="$EPS"
  echo "[embed] PRECAL_TEI_ENDPOINTS=${PRECAL_TEI_ENDPOINTS}"
fi

# ------------------------------ SIGNAL TRAP --------------------------------
# On USR1 (preemption warning), forward TERM to the python driver so it flushes the
# current batch (atomic tmp->fsync->rename + committed-id append), then let Slurm requeue.
DRIVER_PID=""
on_usr1() {
  echo "[embed] received USR1 (preemption warning) -> flushing driver pid=${DRIVER_PID}"
  if [[ -n "${DRIVER_PID}" ]]; then
    kill -TERM "${DRIVER_PID}" 2>/dev/null || true
    wait "${DRIVER_PID}" 2>/dev/null || true
  fi
  cleanup_tei
  echo "[embed] graceful flush complete; exiting for requeue."
  # Non-zero so Slurm requeues this array task (with --requeue).
  exit 142
}
trap on_usr1 USR1
trap cleanup_tei EXIT

# ------------------------------ EMBED DRIVER -------------------------------
# --resume (default true) makes preempted requeue safe.
python -m precal.cli embed \
  --config "${CONFIG}" \
  --shard-id "${SHARD_ID}" \
  --engine "${ENGINE}" \
  --resume &
DRIVER_PID=$!
echo "[embed] driver pid=${DRIVER_PID} for shard ${SHARD_ID}"
wait "${DRIVER_PID}"
RC=$?

echo "[embed] shard ${SHARD_ID} driver exited rc=${RC}"
exit "${RC}"
