#!/bin/bash
#SBATCH --job-name=precal-merge-index
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# ---------------------------------------------------------------------------
# preCal :: MERGE INDEX  (fat-RAM single job: build the ONE full published index)
# ---------------------------------------------------------------------------
# LOCKED CONTRACT (D5): the heavy OPQ64_256,IVF65536_HNSW32,PQ64 index is built
# ONCE here, NOT per shard. The per-shard `index` stage only writes cheap exact
# Flat indexes (smoke + optional scatter-gather). This job trains the coarse
# quantizer on index.train_sample (~2M) random vectors drawn across ALL shard
# .npy memmaps, then add_with_ids(all vectors, chunk-derived int64 ids) with the
# ids carried on the IVF itself. (merge_ondisk is dropped: at v1 scale, building
# one global index directly from the memmaps is correct AND simpler.) The actual
# build lives in precal/index.py; this shell just drives it on a fat-RAM node.
#
#   sbatch slurm/merge_index.sbatch configs/full_v1.yaml
#   sbatch slurm/merge_index.sbatch configs/full_v1.yaml sharded   # mode override
#
# Request a fat-RAM node: PQ64 keeps the 2560-d float32 (~10KB) compressed to
# ~64 bytes (~160x), so the full index fits in host RAM.
# ---------------------------------------------------------------------------

# ------------------------------ SLURM RESOURCES ----------------------------
#SBATCH --partition=memory            # PLACEHOLDER: confirm a fat-RAM/highmem partition (daic_probe.sh)
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G                    # PLACEHOLDER: size to the merged index footprint
#SBATCH --time=12:00:00
#SBATCH --requeue

set -euo pipefail

CONFIG="${1:-configs/full_v1.yaml}"
MODE_OVERRIDE="${2:-}"
REPO_ROOT="${PRECAL_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/activate_env.sh"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "[merge-index] building the ONE full OPQ-IVF-PQ index from all shard .npy memmaps (D5)"
echo "[merge-index] host=$(hostname) config=${CONFIG} mode_override=${MODE_OVERRIDE:-<from config>}"

if [[ -n "${MODE_OVERRIDE}" ]]; then
  python -m precal.cli merge-index --config "${CONFIG}" --mode "${MODE_OVERRIDE}"
else
  python -m precal.cli merge-index --config "${CONFIG}"
fi

echo "[merge-index] done."
