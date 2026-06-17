#!/bin/bash
# ---------------------------------------------------------------------------
# preCal :: launch N TEI replicas on the allocated GPU(s)
# ---------------------------------------------------------------------------
# Usage (per cliContracts):
#   bash scripts/launch_tei_replicas.sh \
#       --model-id Qwen/Qwen3-Embedding-4B \
#       --replicas 4 --base-port 7997 --sif $PRECAL_SCRATCH/tei_120-1.9.sif
#
# Starts `--replicas` TEI containers from the sm_120 SIF, each on a distinct port
# base_port .. base_port+replicas-1. The embed driver discovers exactly that port
# range (PRECAL_TEI_BASE_PORT / PRECAL_TEI_REPLICAS) and round-robins requests.
#
# GPU pinning: the task already holds ONE card (Slurm sets CUDA_VISIBLE_DEVICES);
# all replicas share that single 96GB card (4-8 of the 4B fit). If CUDA_VISIBLE_DEVICES
# lists multiple GPUs, replicas are spread across them round-robin.
#
# Backgrounds each replica, waits until /health is ready, and (if TEI_PID_FILE is
# set in the environment) writes the replica PIDs there one-per-line so the caller
# can trap/cleanup them. Exits non-zero if any replica fails to become healthy.
# ---------------------------------------------------------------------------
set -euo pipefail

MODEL_ID=""
REPLICAS=4
BASE_PORT=7997
SIF="${PRECAL_TEI_SIF:-}"
MAX_BATCH_TOKENS="${PRECAL_TEI_MAX_BATCH_TOKENS:-16384}"   # engine.max_batch_tokens
DTYPE="${PRECAL_TEI_DTYPE:-float16}"                       # TEI exposes only float16|float32
HEALTH_TIMEOUT="${PRECAL_TEI_HEALTH_TIMEOUT:-600}"         # seconds to wait per replica

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-id) MODEL_ID="$2"; shift 2 ;;
    --replicas) REPLICAS="$2"; shift 2 ;;
    --base-port) BASE_PORT="$2"; shift 2 ;;
    --sif) SIF="$2"; shift 2 ;;
    --max-batch-tokens) MAX_BATCH_TOKENS="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    *) echo "launch_tei_replicas.sh: unknown arg '$1'" >&2; exit 2 ;;
  esac
done

[[ -n "${MODEL_ID}" ]] || { echo "launch_tei_replicas.sh: --model-id required" >&2; exit 2; }
[[ -n "${SIF}" && -f "${SIF}" ]] || { echo "launch_tei_replicas.sh: SIF not found: '${SIF}' (run scripts/pull_image.sh)" >&2; exit 3; }

RUNTIME=""
if command -v apptainer >/dev/null 2>&1; then RUNTIME=apptainer
elif command -v singularity >/dev/null 2>&1; then RUNTIME=singularity
else echo "launch_tei_replicas.sh: no apptainer/singularity (daic_probe.sh)" >&2; exit 3; fi

# Resolve the GPU list this task owns.
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
echo "[tei] replicas=${REPLICAS} base_port=${BASE_PORT} gpus=[${CUDA_VISIBLE_DEVICES:-0}] sif=${SIF}"
echo "[tei] model=${MODEL_ID} dtype=${DTYPE} max_batch_tokens=${MAX_BATCH_TOKENS}"

PIDS=()
PORTS=()
for ((i=0; i<REPLICAS; i++)); do
  PORT=$((BASE_PORT + i))
  GPU="${GPUS[$(( i % ${#GPUS[@]} ))]}"
  LOG="${PRECAL_SCRATCH:-/tmp}/tei-${SLURM_JOB_ID:-local}-r${i}-p${PORT}.log"
  echo "[tei] replica ${i}: port=${PORT} gpu=${GPU} log=${LOG}"

  # --nv exposes the NVIDIA stack; bind HF_HOME so the staged snapshot is visible offline.
  # TEI reads the model from --model-id (a LOCAL staged dir on compute nodes).
  CUDA_VISIBLE_DEVICES="${GPU}" \
  "${RUNTIME}" run --nv \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env HF_HOME="${HF_HOME:-${PRECAL_HF_HOME:-/tmp/hf}}" \
    --bind "${HF_HOME:-${PRECAL_HF_HOME:-/tmp/hf}}:${HF_HOME:-${PRECAL_HF_HOME:-/tmp/hf}}" \
    "${SIF}" \
    --model-id "${MODEL_ID}" \
    --port "${PORT}" \
    --dtype "${DTYPE}" \
    --max-batch-tokens "${MAX_BATCH_TOKENS}" \
    --pooling last-token \
    >"${LOG}" 2>&1 &
  PIDS+=($!)
  PORTS+=("${PORT}")
done

# Wait for health on every replica.
all_ok=1
for idx in "${!PORTS[@]}"; do
  PORT="${PORTS[$idx]}"
  deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
  ready=0
  while [[ $(date +%s) -lt ${deadline} ]]; do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then ready=1; break; fi
    # If the replica process died, stop waiting on it.
    if ! kill -0 "${PIDS[$idx]}" 2>/dev/null; then break; fi
    sleep 3
  done
  if [[ "${ready}" -eq 1 ]]; then
    echo "[tei] replica ${idx} healthy on port ${PORT}"
  else
    echo "[tei] ERROR: replica ${idx} (port ${PORT}, pid ${PIDS[$idx]}) not healthy in ${HEALTH_TIMEOUT}s" >&2
    all_ok=0
  fi
done

# Hand PIDs back to the caller for trap/cleanup.
if [[ -n "${TEI_PID_FILE:-}" ]]; then
  printf "%s\n" "${PIDS[@]}" > "${TEI_PID_FILE}"
fi

if [[ "${all_ok}" -ne 1 ]]; then
  echo "[tei] FATAL: one or more replicas failed to start; killing the rest." >&2
  kill "${PIDS[@]}" 2>/dev/null || true
  exit 4
fi

# Qwen3-Embedding REQUIRES last-token pooling. We pass `--pooling last-token`
# above, but if a future TEI build silently ignores/renames the flag (or reads a
# wrong 1_Pooling config off the model dir) the vectors would be quietly wrong.
# Fail loudly: TEI's /info reports the served pooling (model.pooling /
# pooling_mode). Accept only a last-token variant; if /info has no pooling field
# we WARN (older TEI builds omit it) rather than block.
INFO="$(curl -fsS "http://127.0.0.1:${PORTS[0]}/info" 2>/dev/null || true)"
SERVED_POOLING="$(printf '%s' "$INFO" \
  | tr ',{}' '\n\n\n' \
  | grep -iE '"(pooling|pooling_mode|model_pooling)"' \
  | head -n1 \
  | tr 'A-Z' 'a-z' | tr -d ' "_-')"
if [[ -z "${SERVED_POOLING}" ]]; then
  echo "[tei] WARN: TEI /info exposes no pooling field; cannot verify last-token pooling (proceeding)." >&2
elif [[ "${SERVED_POOLING}" != *lasttoken* ]]; then
  echo "[tei] FATAL: served pooling is '${SERVED_POOLING}' but Qwen3-Embedding requires last-token." >&2
  echo "[tei]        Vectors would be WRONG. Killing replicas." >&2
  kill "${PIDS[@]}" 2>/dev/null || true
  exit 5
else
  echo "[tei] verified served pooling = last-token"
fi

echo "[tei] all ${REPLICAS} replica(s) healthy on ports ${PORTS[*]}"
