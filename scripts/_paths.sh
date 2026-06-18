# ---------------------------------------------------------------------------
# preCal :: shared path + cache resolution   (sourced, not executed)
# ---------------------------------------------------------------------------
# DAIC home dirs have a TINY quota. This file pins PRECAL_SCRATCH off $HOME and
# redirects EVERY package / Python / build cache onto that big shared disk, so
# `uv`, pip, HF, and torch never write to $HOME. Sourced by setup_env.sh and
# activate_env.sh. The caller may set _PRECAL_REPO first; otherwise we derive it.
# ---------------------------------------------------------------------------

# Repo root (abs).
if [[ -z "${_PRECAL_REPO:-}" ]]; then
  _PRECAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

# --- PRECAL_SCRATCH: default to a sibling of the repo when the repo lives on a
#     big shared disk (/tudelft.net/... or /scratch/...); never default to $HOME.
if [[ -z "${PRECAL_SCRATCH:-}" ]]; then
  _precal_parent="$(dirname "${_PRECAL_REPO}")"
  case "${_PRECAL_REPO}" in
    /tudelft.net/*|/scratch/*|/projects/*) PRECAL_SCRATCH="${_precal_parent}/precal-scratch" ;;
    *)                                     PRECAL_SCRATCH="${HOME}/precal-scratch" ;;   # local/dev
  esac
fi
export PRECAL_SCRATCH

export PRECAL_HF_HOME="${PRECAL_HF_HOME:-${PRECAL_SCRATCH}/hf}"
export HF_HOME="${PRECAL_HF_HOME}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export PRECAL_TEI_SIF="${PRECAL_TEI_SIF:-${PRECAL_SCRATCH}/tei_120-1.9.sif}"

# --- FORCE all caches/installs onto scratch (NOT ${VAR:-default}) -------------
# DAIC's default profile pre-exports some of these to $HOME (confirmed:
# XDG_CACHE_HOME=$HOME/.cache, UV_CACHE_DIR=$HOME/.cache/uv), so a :-default would
# INHERIT the $HOME value and re-hit the tiny home quota. Override unconditionally.
# Node-local, chmod-capable scratch (the staff-umbrella network FS forbids chmod).
_PRECAL_LOCAL_TMP="${TMPDIR:-/tmp}/${USER:-precal}"
export XDG_DATA_HOME="${PRECAL_SCRATCH}/share"          # uv managed CPython -> $XDG_DATA_HOME/uv/python
# XDG_CACHE_HOME must be node-local: tree-sitter-language-pack extracts its .so
# parser libs here and chmods them 755, which the network FS rejects (falls back to
# whole-file chunking). /tmp allows chmod. uv/pip caches stay on scratch (no chmod).
export XDG_CACHE_HOME="${_PRECAL_LOCAL_TMP}/xdg-cache"
export XDG_CONFIG_HOME="${PRECAL_SCRATCH}/config"
export UV_INSTALL_DIR="${PRECAL_SCRATCH}/bin"           # the uv binary
export UV_CACHE_DIR="${PRECAL_SCRATCH}/cache/uv"        # wheels/build cache (torch etc.)
export UV_PYTHON_INSTALL_DIR="${PRECAL_SCRATCH}/share/uv/python"
export PIP_CACHE_DIR="${PRECAL_SCRATCH}/cache/pip"
export UV_NO_MODIFY_PATH=1                              # don't touch ~/.bashrc
# apptainer cache/build also need chmod -> node-local (_PRECAL_LOCAL_TMP, set above).
# The finished SIF is copied to scratch separately (a plain file).
export APPTAINER_CACHEDIR="${_PRECAL_LOCAL_TMP}/apptainer-cache"
export APPTAINER_TMPDIR="${_PRECAL_LOCAL_TMP}/apptainer-tmp"
export SINGULARITY_CACHEDIR="${APPTAINER_CACHEDIR}"
export SINGULARITY_TMPDIR="${APPTAINER_TMPDIR}"
# Repo-local venv (repo is on the big disk too); override with PRECAL_VENV.
export PRECAL_VENV="${PRECAL_VENV:-${_PRECAL_REPO}/.venv}"

# Make a scratch-installed uv (and any ~/.local/.cargo uv) reachable.
export PATH="${PRECAL_SCRATCH}/bin:${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"

mkdir -p "${PRECAL_SCRATCH}" "${PRECAL_HF_HOME}" "${UV_INSTALL_DIR}" \
         "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}" "${XDG_DATA_HOME}" \
         "${XDG_CONFIG_HOME}" "${PIP_CACHE_DIR}" 2>/dev/null || true
mkdir -p "${XDG_CACHE_HOME}" "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}" 2>/dev/null || true  # node-local (chmod-capable)
