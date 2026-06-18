#!/bin/bash
# ---------------------------------------------------------------------------
# preCal :: create the Python environment  (uv-first; conda fallback)
# ---------------------------------------------------------------------------
#   bash scripts/setup_env.sh             # uv venv (.venv) + install requirements.txt
#   bash scripts/setup_env.sh --update    # re-sync deps into the existing env
#   bash scripts/setup_env.sh --python 3.11
#   bash scripts/setup_env.sh --conda     # force the legacy conda path (env/environment.yml)
#
# DAIC has no conda -> uv is the default. uv manages its own Python, so NO module
# system is required. The GPU EMBED step runs inside the TEI sm_120 CONTAINER, not
# this env; this env covers chunking / sharding / FAISS / eval / publish (+ the
# reference engines). venv lives at $PRECAL_VENV (default <repo>/.venv) on the
# shared filesystem, so compute nodes can activate it too.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REQ="${REPO_ROOT}/requirements.txt"
PYVER="3.11"
DO_UPDATE=0
FORCE_CONDA=0
DO_FAISS_GPU=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --update)     DO_UPDATE=1; shift ;;
    --conda)      FORCE_CONDA=1; shift ;;
    --faiss-gpu)  DO_FAISS_GPU=1; shift ;;
    --python)     PYVER="$2"; shift 2 ;;
    --name)       PRECAL_ENV_NAME="$2"; shift 2 ;;   # conda path only
    -h|--help)    sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "setup_env.sh: unknown arg '$1'" >&2; exit 2 ;;
  esac
done

# Honor user overrides, then resolve PRECAL_SCRATCH + redirect ALL uv/pip/HF
# caches onto the big shared disk (DAIC $HOME has a tiny quota). _paths.sh sets
# PRECAL_SCRATCH, PRECAL_VENV, UV_INSTALL_DIR, UV_CACHE_DIR, UV_PYTHON_INSTALL_DIR, …
[[ -f "${HOME}/.precal.env" ]] && source "${HOME}/.precal.env"
_PRECAL_REPO="${REPO_ROOT}"
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/_paths.sh"

# --------------------------------------------------------------------------- uv
ensure_uv() {
  command -v uv >/dev/null 2>&1 && return 0
  echo "[setup_env] uv not found — installing to ${UV_INSTALL_DIR} (needs internet; login node)…"
  # UV_INSTALL_DIR (scratch) + UV_NO_MODIFY_PATH keep the install OFF the $HOME quota.
  curl -LsSf https://astral.sh/uv/install.sh | sh || return 1
  command -v uv >/dev/null 2>&1
}

if [[ "${FORCE_CONDA}" -eq 0 ]] && ensure_uv; then
  echo "[setup_env] uv $(uv --version 2>/dev/null | awk '{print $2}')  venv=${PRECAL_VENV}  python=${PYVER}"
  mkdir -p "$(dirname "${PRECAL_VENV}")"
  if [[ ! -d "${PRECAL_VENV}" ]]; then
    uv venv "${PRECAL_VENV}" --python "${PYVER}"
  else
    echo "[setup_env] venv exists (use --update to re-sync deps)."
  fi
  [[ -d "${PRECAL_VENV}" ]] || { echo "[setup_env] FATAL: venv create failed"; exit 3; }

  STAMP="${PRECAL_VENV}/.precal-deps-ok"
  if [[ ! -f "${STAMP}" || "${DO_UPDATE}" -eq 1 ]]; then
    echo "[setup_env] installing requirements via uv (torch-free core; ~1-2 min)…"
    # hf_transfer accelerates staged download/upload on login nodes.
    # Explicit --cache-dir is bulletproof even if the shell pre-exports UV_CACHE_DIR to $HOME.
    uv pip install --cache-dir "${UV_CACHE_DIR}" --python "${PRECAL_VENV}/bin/python" -r "${REQ}" hf_transfer
    touch "${STAMP}"
  else
    echo "[setup_env] deps already installed (stamp ${STAMP}); pass --update to refresh."
  fi
  [[ "${DO_FAISS_GPU}" -eq 1 ]] && \
    echo "[setup_env] NOTE: faiss-gpu is conda-only; the uv path stays on faiss-cpu (index build still works)."
  echo "[setup_env] done."
  echo "[setup_env] activate with:  source scripts/activate_env.sh   (sources ${PRECAL_VENV})"
  echo "[setup_env] APPTAINER note: the GPU embed engine runs from the TEI SIF, not this venv:"
  echo "[setup_env]   bash scripts/pull_image.sh"
  exit 0
fi

# ------------------------------------------------------------------ conda fallback
echo "[setup_env] uv unavailable or --conda forced → legacy conda/mamba path."
ENV_FILE="${REPO_ROOT}/env/environment.yml"
ENV_NAME="${PRECAL_ENV_NAME:-precal}"
if command -v module >/dev/null 2>&1; then
  module load 2>/dev/null miniconda3 || module load 2>/dev/null anaconda3 || true
fi
CONDA_BIN=""
if command -v mamba >/dev/null 2>&1; then CONDA_BIN=mamba
elif command -v conda >/dev/null 2>&1; then CONDA_BIN=conda
else
  echo "setup_env.sh: neither uv nor conda/mamba available." >&2
  echo "  Install uv:  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 3
fi
echo "[setup_env] using ${CONDA_BIN}; env='${ENV_NAME}'; file=${ENV_FILE}"
if "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  [[ "${DO_UPDATE}" -eq 1 ]] && "${CONDA_BIN}" env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune \
    || echo "[setup_env] env '${ENV_NAME}' exists (use --update to refresh)."
else
  "${CONDA_BIN}" env create -n "${ENV_NAME}" -f "${ENV_FILE}"
fi
if [[ "${DO_FAISS_GPU}" -eq 1 ]]; then
  "${CONDA_BIN}" install -n "${ENV_NAME}" -c conda-forge -c pytorch -c nvidia faiss-gpu-raft 2>/dev/null \
    || "${CONDA_BIN}" install -n "${ENV_NAME}" -c pytorch -c nvidia faiss-gpu 2>/dev/null \
    || echo "[setup_env] WARN: faiss-gpu install failed; staying on faiss-cpu."
fi
echo "[setup_env] done. Activate: source scripts/activate_env.sh"
