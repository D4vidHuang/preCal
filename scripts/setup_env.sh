#!/bin/bash
# ---------------------------------------------------------------------------
# preCal :: create the conda/mamba environment
# ---------------------------------------------------------------------------
#   bash scripts/setup_env.sh                 # create env 'precal' from env/environment.yml
#   bash scripts/setup_env.sh --update        # update an existing env
#   bash scripts/setup_env.sh --faiss-gpu     # attempt conda-forge faiss-gpu (sm_120 unconfirmed)
#   bash scripts/setup_env.sh --name myenv     # custom env name
#
# The PRODUCTION embed engine is the TEI sm_120 CONTAINER (pulled by
# scripts/pull_image.sh), NOT this conda env — so this env intentionally avoids
# pinning torch to a fragile sm_120 wheel. It covers chunking / sharding / FAISS /
# eval / publish. See env/environment.yml for the pins and DESIGN.md for the
# apptainer-vs-conda split.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/env/environment.yml"
ENV_NAME="${PRECAL_ENV_NAME:-precal}"
DO_UPDATE=0
DO_FAISS_GPU=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --update) DO_UPDATE=1; shift ;;
    --faiss-gpu) DO_FAISS_GPU=1; shift ;;
    --name) ENV_NAME="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "setup_env.sh: unknown arg '$1'" >&2; exit 2 ;;
  esac
done

# Load a conda/mamba provider via modules if present (DAIC), else assume on PATH.
if command -v module >/dev/null 2>&1; then
  module load 2>/dev/null miniconda3 || module load 2>/dev/null anaconda3 || true
fi

CONDA_BIN=""
if command -v mamba >/dev/null 2>&1; then CONDA_BIN=mamba
elif command -v conda >/dev/null 2>&1; then CONDA_BIN=conda
else
  echo "setup_env.sh: no conda/mamba found. Load a module (scripts/daic_probe.sh) or install miniforge." >&2
  exit 3
fi
echo "[setup_env] using ${CONDA_BIN}; env name='${ENV_NAME}'; file=${ENV_FILE}"

# Create or update.
if "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  if [[ "${DO_UPDATE}" -eq 1 ]]; then
    echo "[setup_env] updating existing env '${ENV_NAME}'"
    "${CONDA_BIN}" env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
  else
    echo "[setup_env] env '${ENV_NAME}' already exists (use --update to refresh)."
  fi
else
  echo "[setup_env] creating env '${ENV_NAME}'"
  "${CONDA_BIN}" env create -n "${ENV_NAME}" -f "${ENV_FILE}"
fi

# Optional faiss-gpu attempt (UNCONFIRMED on DAIC sm_120 — see open questions).
if [[ "${DO_FAISS_GPU}" -eq 1 ]]; then
  echo "[setup_env] attempting faiss-gpu (conda-forge). sm_120 support is UNCONFIRMED; falls back to faiss-cpu if it fails."
  "${CONDA_BIN}" install -n "${ENV_NAME}" -c conda-forge -c pytorch -c nvidia faiss-gpu-raft 2>/dev/null \
    || "${CONDA_BIN}" install -n "${ENV_NAME}" -c pytorch -c nvidia faiss-gpu 2>/dev/null \
    || echo "[setup_env] WARN: faiss-gpu install failed; staying on faiss-cpu (index build still works)."
fi

echo "[setup_env] done."
echo "[setup_env] Activate with: source scripts/activate_env.sh   (sets PRECAL_* + activates '${ENV_NAME}')"
echo "[setup_env] APPTAINER note: the GPU embed engine runs from the TEI SIF, not this env."
echo "[setup_env]   pull it on a login node:  bash scripts/pull_image.sh"
