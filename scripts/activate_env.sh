#!/bin/bash
# ---------------------------------------------------------------------------
# preCal :: env activation (sourced by every sbatch and helper script)
# ---------------------------------------------------------------------------
# Sets PRECAL_* path vars, loads DAIC modules, and activates the conda/mamba env
# created by setup_env.sh. Source this — do NOT execute it.
#
#   source scripts/activate_env.sh
#
# Override any of these by exporting before sourcing, or in ~/.precal.env.
# ---------------------------------------------------------------------------

# ------------------------------ USER OVERRIDES -----------------------------
# Optional per-user overrides (hostnames, account, scratch path) live here so
# nothing DAIC-specific has to be hardcoded in version control.
if [[ -f "${HOME}/.precal.env" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/.precal.env"
fi

# ------------------------------ PATHS + CACHES -----------------------------
# Resolve PRECAL_SCRATCH (off the tiny $HOME quota) and redirect every uv/pip/HF
# cache onto the big shared disk. Single source of truth: scripts/_paths.sh.
_PRECAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${_PRECAL_REPO}/scripts/_paths.sh"

# LOCKED CONTRACT (D1): the manifests dir is namespaced by run.name
# (${PRECAL_SCRATCH}/<run.name>/manifests, matching precal.config's
# scratch_subdir("manifests")). activate_env.sh does NOT know run.name, so it
# must NOT bake an un-namespaced default into PRECAL_MANIFEST_DIR — doing so
# would pre-empt the ops scripts' run-name fallback
# (${PRECAL_MANIFEST_DIR:-${PRECAL_SCRATCH}/${RUN_NAME}/manifests}) and make the
# done-marker guards stat the wrong (un-namespaced) path. We export it ONLY when
# the user has explicitly set it (an intentional, run-aware override).
[[ -n "${PRECAL_MANIFEST_DIR:-}" ]] && export PRECAL_MANIFEST_DIR

mkdir -p "${PRECAL_SCRATCH}" "${PRECAL_HF_HOME}" 2>/dev/null || true
[[ -n "${PRECAL_MANIFEST_DIR:-}" ]] && mkdir -p "${PRECAL_MANIFEST_DIR}" 2>/dev/null || true

# ------------------------------ MODULES ------------------------------------
# DAIC uses Lmod/environment-modules. Names are PLACEHOLDERS — confirm with
# `module avail` (scripts/daic_probe.sh). Failures are non-fatal so local/dev
# (macOS, no module system) still works. NOTE: the Python env is uv (no conda),
# so we only try CUDA + container modules here.
if command -v module >/dev/null 2>&1; then
  module purge 2>/dev/null || true
  # CUDA toolkit for any non-container GPU step (TEI brings its own runtime).
  module load 2>/dev/null cuda/12.6 || module load 2>/dev/null cuda || true
  # Container runtime for the TEI SIF.
  module load 2>/dev/null apptainer || module load 2>/dev/null singularity || true
fi

# ------------------------- PYTHON ENV (uv venv; conda fallback) -------------
# PRECAL_VENV (default <repo>/.venv on the shared FS) is set by scripts/_paths.sh.
if [[ -f "${PRECAL_VENV}/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${PRECAL_VENV}/bin/activate"
  echo "[activate_env] uv venv active: ${PRECAL_VENV} (python: $(command -v python))"
else
  # Legacy conda/mamba fallback (only if someone built that instead).
  PRECAL_ENV_NAME="${PRECAL_ENV_NAME:-precal}"
  _conda_base=""
  if command -v conda >/dev/null 2>&1; then _conda_base="$(conda info --base 2>/dev/null || true)"
  elif command -v mamba >/dev/null 2>&1; then _conda_base="$(mamba info --base 2>/dev/null || true)"; fi
  if [[ -n "${_conda_base}" && -f "${_conda_base}/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${_conda_base}/etc/profile.d/conda.sh"
    conda activate "${PRECAL_ENV_NAME}" 2>/dev/null \
      && echo "[activate_env] conda env '${PRECAL_ENV_NAME}' active (python: $(command -v python))" \
      || echo "[activate_env] WARN: could not activate '${PRECAL_ENV_NAME}' — run scripts/setup_env.sh"
  else
    echo "[activate_env] WARN: no uv venv at ${PRECAL_VENV} and no conda — run scripts/setup_env.sh (python: $(command -v python || echo none))"
  fi
fi

# ------------------------------ HF / RUNTIME -------------------------------
# Egress policy differs per node; callers (stage/publish) clear these explicitly.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"

echo "[activate_env] PRECAL_SCRATCH=${PRECAL_SCRATCH}"
echo "[activate_env] HF_HOME=${HF_HOME}"
echo "[activate_env] PRECAL_MANIFEST_DIR=${PRECAL_MANIFEST_DIR:-<unset: ops use \${PRECAL_SCRATCH}/<run.name>/manifests>}"
