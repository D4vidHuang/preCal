#!/bin/bash
# ---------------------------------------------------------------------------
# preCal :: SMOKE  (full pipeline end-to-end on ONE GPU, ~minutes)
# ---------------------------------------------------------------------------
#   bash scripts/smoke.sh [GPU_ID=0]
#
# Exercises stage -> chunk -> shard -> embed -> index -> eval with configs/smoke.yaml
# (Qwen3-Embedding-0.6B, python only, ~5k files, Flat index) plus the offline +
# checkpoint machinery. Run STAGE on a node with internet first; the rest is offline.
#
# Pass/fail gate: step (6) eval — expect 0.6B MRR@10/nDCG@10 in the published
# CoIR-CSN range. Step (5) also kills+reruns embed to prove idempotent resume.
# ---------------------------------------------------------------------------
set -euo pipefail

GPU_ID="${1:-0}"
CONFIG="configs/smoke.yaml"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/activate_env.sh"

echo "############ preCal smoke (GPU ${GPU_ID}, config ${CONFIG}) ############"

# 1) STAGE (needs internet). Skip with PRECAL_SKIP_STAGE=1 if already staged.
if [[ "${PRECAL_SKIP_STAGE:-0}" != "1" ]]; then
  echo "## [1/6] stage (corpus sample + CoIR-CSN python + 0.6B model + TEI SIF)"
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
  python -m precal.cli stage --config "${CONFIG}"
  bash "${REPO_ROOT}/scripts/pull_image.sh" --config "${CONFIG}"
else
  echo "## [1/6] stage SKIPPED (PRECAL_SKIP_STAGE=1)"
fi

# From here on: offline + single GPU.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
bash "${REPO_ROOT}/scripts/check_env.sh" --quick || echo "## WARN: check_env quick reported issues"

# 2) CHUNK (CPU, tree-sitter symbol extraction)
echo "## [2/6] chunk (python)"
python -m precal.cli chunk --config "${CONFIG}" --language python

# 3) SHARD (+ pairs + eval-split); then global consolidation -> manifest.jsonl
echo "## [3/6] shard"
python -m precal.cli shard --config "${CONFIG}" --language python
python -m precal.cli shard --config "${CONFIG}"   # global contiguous shard_id stamping

RUN_NAME="$(python "${REPO_ROOT}/scripts/cfg.py" "${CONFIG}" run.name precal-v1)"
MANIFEST="${PRECAL_SCRATCH}/${RUN_NAME}/shards/manifest.jsonl"
N="$(wc -l < "${MANIFEST}" | tr -d ' ')"
echo "## shard count N=${N} (manifest: ${MANIFEST})"

# 4) TEI replicas (if engine=tei) for the smoke 0.6B model
ENGINE="$(python "${REPO_ROOT}/scripts/cfg.py" "${CONFIG}" engine.name)"
TEI_PIDS=()
cleanup() { if [[ ${#TEI_PIDS[@]} -gt 0 ]]; then kill "${TEI_PIDS[@]}" 2>/dev/null || true; fi; }
trap cleanup EXIT
if [[ "${ENGINE}" == "tei" ]]; then
  echo "## [4/6] launch TEI replicas"
  MODEL_ID="$(python "${REPO_ROOT}/scripts/cfg.py" "${CONFIG}" model.id)"
  REPLICAS="$(python "${REPO_ROOT}/scripts/cfg.py" "${CONFIG}" engine.replicas_per_gpu 2)"
  TEI_PID_FILE="$(mktemp)"
  TEI_PID_FILE="$TEI_PID_FILE" \
    bash "${REPO_ROOT}/scripts/launch_tei_replicas.sh" \
      --model-id "${HF_HOME}/staged/${MODEL_ID}" --replicas "${REPLICAS}" \
      --base-port "${PRECAL_TEI_BASE_PORT:-7997}" --sif "${PRECAL_TEI_SIF}"
  mapfile -t TEI_PIDS < "$TEI_PID_FILE"; rm -f "$TEI_PID_FILE"
  export PRECAL_TEI_BASE_PORT="${PRECAL_TEI_BASE_PORT:-7997}" PRECAL_TEI_REPLICAS="${REPLICAS}"
else
  echo "## [4/6] engine=${ENGINE}; no TEI replicas needed"
fi

# 5) EMBED each shard + prove idempotent resume on shard 0
echo "## [5/6] embed shards 0..$((N-1)) (+ resume test on shard 0)"
# 5a) start shard 0, kill mid-run, then resume -> committed-id manifest must dedupe.
( python -m precal.cli embed --config "${CONFIG}" --shard-id 0 --resume ) & EPID=$!
sleep "${PRECAL_SMOKE_KILL_AFTER:-15}"
if kill -0 "${EPID}" 2>/dev/null; then
  echo "## resume-test: killing embed pid=${EPID} mid-shard"; kill -TERM "${EPID}" 2>/dev/null || true; wait "${EPID}" 2>/dev/null || true
fi
echo "## resume-test: re-running shard 0 (should skip committed chunk_ids)"
python -m precal.cli embed --config "${CONFIG}" --shard-id 0 --resume
# 5b) remaining shards
for ((s=1; s<N; s++)); do
  python -m precal.cli embed --config "${CONFIG}" --shard-id "${s}" --resume
done

# 6) INDEX (Flat) + EVAL (exact) — the end-to-end pass/fail gate
echo "## [6/6] index (Flat) + eval (exact)"
python -m precal.cli index --config "${CONFIG}" --all
python -m precal.cli eval  --config "${CONFIG}" --split test --index exact

echo "############ smoke complete ############"
echo "PASS/FAIL: confirm MRR@10 / nDCG@10 above are in the published CoIR-CSN-python range for 0.6B."
