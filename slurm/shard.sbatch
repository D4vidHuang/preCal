#!/bin/bash
#SBATCH --job-name=precal-shard
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
# ---------------------------------------------------------------------------
# preCal :: SHARD  (CPU array: chunk -> pairs/eval-split -> deterministic shards)
# ---------------------------------------------------------------------------
# Level-1 sharding is by LANGUAGE. This array runs ONE language per task, so set
# --array to the number of languages in corpus.languages (default 6):
#
#   sbatch --array=0-5 slurm/shard.sbatch configs/full_v1.yaml
#
# Each task:
#   * maps SLURM_ARRAY_TASK_ID -> the Nth language in corpus.languages
#   * runs `precal.cli chunk` (tree-sitter symbol extraction, CPU only)
#   * runs `precal.cli shard` for that language (writes its slice of the manifest)
#
# After ALL language tasks finish, a final consolidation `precal.cli shard`
# (no --language) stamps the GLOBAL contiguous shard_id ordering into
# shards/manifest.jsonl. The Makefile `shard` target chains this.
# Deterministic given run.seed; idempotent on re-run.
# ---------------------------------------------------------------------------

# ------------------------------ SLURM RESOURCES ----------------------------
#SBATCH --partition=compute           # PLACEHOLDER: confirm a CPU partition (daic_probe.sh)
#SBATCH --cpus-per-task=16            # tree-sitter parsing is CPU-parallel within a language
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue

set -euo pipefail

CONFIG="${1:-configs/full_v1.yaml}"
REPO_ROOT="${PRECAL_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/activate_env.sh"

# Parsing reads only staged parquet from scratch — run offline.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Map array index -> language from the SAME config the CLI reads.
LANGS_CSV="$(python "${REPO_ROOT}/scripts/cfg.py" "$CONFIG" corpus.languages)"
IFS=',' read -r -a LANGS <<< "${LANGS_CSV}"
IDX="${SLURM_ARRAY_TASK_ID:?shard.sbatch must run as a job array (one task per language)}"

if (( IDX >= ${#LANGS[@]} )); then
  echo "[shard] task ${IDX} >= #languages (${#LANGS[@]}); nothing to do."; exit 0
fi
LANG="${LANGS[$IDX]}"
echo "[shard] host=$(hostname) task=${IDX} language=${LANG} config=${CONFIG}"

# 1) Chunk (tree-sitter symbol extraction + window/whole-file fallbacks).
python -m precal.cli chunk --config "${CONFIG}" --language "${LANG}"

# 2) Per-language sharding (also assigns query_text/query_source + eval_split
#    via repo-level partition against frozen CoIR qrels unless --no-pairs).
python -m precal.cli shard --config "${CONFIG}" --language "${LANG}"

echo "[shard] language ${LANG} done."
echo "[shard] NOTE: after ALL language tasks finish, run the GLOBAL consolidation:"
echo "[shard]       python -m precal.cli shard --config ${CONFIG}"
echo "[shard]       (the Makefile 'shard' target chains this automatically)."
