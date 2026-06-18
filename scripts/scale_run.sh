#!/bin/bash
# ---------------------------------------------------------------------------
# preCal :: batched scale-up driver  (run on a LOGIN node, under nohup/tmux)
# ---------------------------------------------------------------------------
# Scales the published HF dataset PAST local disk by processing one batch at a
# time. Each batch == its own run.name == an isolated scratch subtree (clean
# rm -rf) and a unique HF filename prefix (publish.py prefixes by run.name).
#
#   nohup bash scripts/scale_run.sh --config configs/scale_4b.yaml \
#         --batches batches.txt --conc %32 [--free] [--repo R] > scale.log 2>&1 &
#
# batches file: one line per batch  ->  "<run_name> <comma-langs>"
#   precal-4b-b0  python,javascript
#   precal-4b-b1  java,php
#
# Per batch: stage -> chunk -> shard -> embed (SLURM array, block) -> index -->
# merge -> publish -> VERIFY (remote files exist) -> [FREE: rm -rf the subtree].
# Resumable: cross-step markers live OUTSIDE the rundir ($STATE) so they survive
# the rm; re-running skips finished batches and resumes the first incomplete one.
# --free is OPT-IN (default OFF) — only enable once the loop is validated, since
# the prune is irreversible (the HF copy is the only backup).
# ---------------------------------------------------------------------------
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO_ROOT"

CONFIG=""; BATCHES=""; CONC="%32"; FREE=0; REPO_ID="D4vidHuang/precal-code-embeddings"
HEADROOM_G=40            # abort a batch if scratch free < this (shared 475G volume)
while [[ $# -gt 0 ]]; do case "$1" in
  --config) CONFIG="$2"; shift 2;;
  --batches) BATCHES="$2"; shift 2;;
  --conc) CONC="$2"; shift 2;;
  --free) FREE=1; shift;;
  --repo) REPO_ID="$2"; shift 2;;
  --headroom) HEADROOM_G="$2"; shift 2;;
  *) echo "scale_run.sh: unknown arg '$1'" >&2; exit 2;;
esac; done
[[ -f "$CONFIG" && -f "$BATCHES" ]] || { echo "need --config <f> and --batches <f>" >&2; exit 2; }

# shellcheck source=/dev/null
source scripts/activate_env.sh
PY="python -m precal.cli"
STATE="${PRECAL_SCRATCH}/.scale_state"; mkdir -p "$STATE"
df_free_g(){ df -BG --output=avail "$PRECAL_SCRATCH" 2>/dev/null | tail -1 | tr -dc '0-9'; }

while read -r RN LANGS _rest; do
  [[ -z "${RN:-}" || "$RN" == \#* ]] && continue
  export PRECAL_RUN_NAME="$RN"
  RUNDIR="${PRECAL_SCRATCH}/${RN}"
  MAN="${RUNDIR}/manifests"
  LANGS_SP="${LANGS//,/ }"
  echo "============ BATCH ${RN}  langs=${LANGS} ============ $(date -u +%FT%TZ)"
  [[ -f "${STATE}/${RN}.freed" ]] && { echo "[$RN] freed already; skip"; continue; }
  [[ -f "${STATE}/${RN}.done"  ]] && { echo "[$RN] done already; skip"; continue; }

  FG=$(df_free_g); echo "[$RN] scratch free=${FG:-?}G"
  if [[ -n "${FG:-}" && "$FG" -lt "$HEADROOM_G" ]]; then
    echo "[$RN] ABORT: free ${FG}G < headroom ${HEADROOM_G}G"; exit 3; fi

  # 1) STAGE (online). Corpus for this batch's langs + (shared) model/eval packs.
  if [[ ! -f "${STATE}/${RN}.staged" ]]; then
    echo "[$RN] stage"; $PY stage --config "$CONFIG" --languages "$LANGS" || exit 4
    touch "${STATE}/${RN}.staged"
  fi

  # 2) CHUNK (parallel CPU jobs, one per language — NOT inline on the login node)
  #    then a single global shard consolidation into this run's manifest.
  if [[ ! -f "${STATE}/${RN}.sharded" ]]; then
    CJIDS=""
    for L in $LANGS_SP; do
      jid=$(sbatch --parsable --partition=all --account=testusers --cpus-per-task=2 --mem=16G --time=3:00:00 \
        --job-name="chunk-${RN}-${L}" --output="${PRECAL_SCRATCH}/logs/chunk-${RN}-${L}.out" \
        --error="${PRECAL_SCRATCH}/logs/chunk-${RN}-${L}.err" \
        --wrap="cd ${REPO_ROOT} && source scripts/activate_env.sh && export PRECAL_RUN_NAME='${RN}' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && python -m precal.cli chunk --config ${CONFIG} --language ${L}")
      CJIDS="${CJIDS:+${CJIDS},}${jid}"
      echo "[$RN] chunk ${L} -> job ${jid}"
    done
    while squeue --me -h -j "$CJIDS" 2>/dev/null | grep -q .; do sleep 45; done
    for L in $LANGS_SP; do
      ls "${RUNDIR}/chunks/lang=${L}"/*.parquet >/dev/null 2>&1 || { echo "[$RN] CHUNK FAILED for ${L} (see chunk-${RN}-${L}.err)"; exit 5; }
    done
    $PY shard --config "$CONFIG" || exit 5     # global consolidation -> manifest.jsonl
    touch "${STATE}/${RN}.sharded"
  fi

  # 3) EMBED (GPU SLURM array; block until every shard has a done-marker).
  MANIFEST="${RUNDIR}/shards/manifest.jsonl"
  N=$(wc -l < "$MANIFEST" | tr -d ' ')
  echo "[$RN] embed: $N shards @ ${CONC}"
  STALL=0
  while :; do
    DONE=$(ls "${MAN}"/embed-*.done 2>/dev/null | wc -l | tr -d ' ')
    [[ "$DONE" -ge "$N" ]] && break
    bash scripts/resubmit_pending.sh embed "$CONFIG" "$CONC" || true
    sleep 30
    while squeue --me -h -n precal-embed 2>/dev/null | grep -q .; do sleep 60; done
    NOW=$(ls "${MAN}"/embed-*.done 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$NOW" -le "$DONE" ]]; then STALL=$((STALL+1)); else STALL=0; fi
    [[ "$STALL" -ge 2 ]] && { echo "[$RN] EMBED STALLED at ${NOW}/${N} — check precal-embed-*.err"; exit 6; }
  done
  echo "[$RN] embed complete ${N}/${N}"

  # 4) INDEX (per-shard Flat) + MERGE (per-batch OPQ-IVF-PQ; sharded mode).
  if [[ ! -f "${STATE}/${RN}.merged" ]]; then
    $PY index --config "$CONFIG" --all || exit 7
    $PY merge-index --config "$CONFIG" || exit 7
    touch "${STATE}/${RN}.merged"
  fi

  # 5) PUBLISH (online) + 6) VERIFY remote files exist for this run (delete-gate).
  if [[ ! -f "${STATE}/${RN}.published" ]]; then
    echo "[$RN] publish -> ${REPO_ID}"; $PY publish --config "$CONFIG" --repo-id "$REPO_ID" || exit 8
    python - "$REPO_ID" "$RN" <<'PYV' || { echo "[$RN] VERIFY FAILED — not marking published"; exit 9; }
import sys
from huggingface_hub import HfApi
repo, rn = sys.argv[1], sys.argv[2]
fs=[f for f in HfApi().list_repo_files(repo, repo_type="dataset") if (rn+"-") in f or ("/"+rn+"/" in f)]
print(f"[verify] {len(fs)} remote files for run {rn}")
sys.exit(0 if len(fs) >= 3 else 1)
PYV
    touch "${STATE}/${RN}.published"
  fi

  # 7) FREE (opt-in, irreversible): prune the per-batch subtree, HF is the copy.
  if [[ "$FREE" -eq 1 ]]; then
    echo "[$RN] FREE rm -rf ${RUNDIR}"; rm -rf "$RUNDIR" && touch "${STATE}/${RN}.freed"
  fi
  touch "${STATE}/${RN}.done"
  echo "[$RN] BATCH DONE $(date -u +%FT%TZ)"
done < "$BATCHES"
echo "==== ALL BATCHES DONE ===="
