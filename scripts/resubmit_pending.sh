#!/bin/bash
# ---------------------------------------------------------------------------
# preCal :: resubmit only NOT-done array tasks (embed | index)
# ---------------------------------------------------------------------------
#   bash scripts/resubmit_pending.sh embed configs/full_v1.yaml [%CONC]
#   bash scripts/resubmit_pending.sh index configs/full_v1.yaml [%8]
#
# The shard manifest is the single source of truth. Each shard transitions
# pending->running->done; `done` is stamped as $PRECAL_MANIFEST_DIR/<stage>-<id>.done
# only after the per-shard row counts reconcile (see requeueStrategy). This
# controller reads manifest.jsonl, computes the set of shard_ids whose status != done,
# and submits a job array restricted to EXACTLY those ids (Slurm accepts a
# comma/range list in --array), so the whole pipeline is re-runnable until all
# shards are done. Idempotency guarantees a stray re-run of a done shard is a no-op,
# but skipping them saves scheduler slots.
# ---------------------------------------------------------------------------
set -euo pipefail

STAGE="${1:?usage: resubmit_pending.sh <embed|index> <config> [%CONCURRENCY]}"
CONFIG="${2:?config required}"
CONC="${3:-%8}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/activate_env.sh"

# LOCKED CONTRACT (D2): done markers are "<stage>-<rawid>.done" — for embed that
# is "embed-<id>.done" (NOT "shard-..."), matching precal.manifest.mark_done.
case "${STAGE}" in
  embed) SBATCH="slurm/embed.sbatch"; DONE_PREFIX="embed" ;;
  index) SBATCH="slurm/index.sbatch"; DONE_PREFIX="index" ;;
  *) echo "resubmit_pending.sh: stage must be embed|index" >&2; exit 2 ;;
esac

RUN_NAME="$(python "${REPO_ROOT}/scripts/cfg.py" "${CONFIG}" run.name precal-v1)"
MANIFEST="${PRECAL_SCRATCH}/${RUN_NAME}/shards/manifest.jsonl"
# LOCKED CONTRACT (D1): manifests live under ${PRECAL_SCRATCH}/${RUN_NAME}/manifests.
STATUS_DIR="${PRECAL_MANIFEST_DIR:-${PRECAL_SCRATCH}/${RUN_NAME}/manifests}"

[[ -f "${MANIFEST}" ]] || { echo "resubmit_pending.sh: manifest not found: ${MANIFEST} (run shard first)" >&2; exit 3; }

# All shard_ids from the manifest (one JSON line per shard).
ALL_IDS="$(python - "$MANIFEST" <<'PY'
import json,sys
ids=[]
with open(sys.argv[1]) as f:
    for line in f:
        line=line.strip()
        if line:
            ids.append(int(json.loads(line)["shard_id"]))
print(",".join(str(i) for i in sorted(ids)))
PY
)"

# Filter out ids that already have a <stage>-<id>.done marker.
PENDING=()
IFS=',' read -r -a IDS <<< "${ALL_IDS}"
for id in "${IDS[@]}"; do
  [[ -f "${STATUS_DIR}/${DONE_PREFIX}-${id}.done" ]] || PENDING+=("${id}")
done

if [[ ${#PENDING[@]} -eq 0 ]]; then
  echo "resubmit_pending.sh: all ${#IDS[@]} ${STAGE} shards already done — nothing to submit."
  exit 0
fi

ARRAY_SPEC="$(IFS=,; echo "${PENDING[*]}")${CONC}"
# embed needs a GPU (TEI); index build is faiss-cpu. Route through submit.sh so the
# DAIC partition/gres/constraint from ~/.precal.env override the #SBATCH placeholders.
GPU_FLAG=""; [[ "${STAGE}" == "embed" ]] && GPU_FLAG="--gpu"
echo "resubmit_pending.sh: submitting ${#PENDING[@]}/${#IDS[@]} pending ${STAGE} shards"
echo "  submit.sh ${GPU_FLAG} --array=${ARRAY_SPEC} ${SBATCH} ${CONFIG}"
bash "${REPO_ROOT}/scripts/submit.sh" ${GPU_FLAG} --array="${ARRAY_SPEC}" "${SBATCH}" "${CONFIG}"
