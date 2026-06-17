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

# --- keep ALL caches/installs off the small $HOME quota -----------------------
# XDG_DATA_HOME is uv's data ROOT: managed CPython lands in $XDG_DATA_HOME/uv/python.
# Setting it is the reliable redirect (uv ignored UV_PYTHON_INSTALL_DIR in 0.11.x).
export XDG_DATA_HOME="${XDG_DATA_HOME:-${PRECAL_SCRATCH}/share}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PRECAL_SCRATCH}/cache/xdg}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${PRECAL_SCRATCH}/config}"
export UV_INSTALL_DIR="${UV_INSTALL_DIR:-${PRECAL_SCRATCH}/bin}"           # the uv binary
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PRECAL_SCRATCH}/cache/uv}"           # wheels/build cache
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${XDG_DATA_HOME}/uv/python}"  # managed CPython
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PRECAL_SCRATCH}/cache/pip}"
export UV_NO_MODIFY_PATH=1                                                 # don't touch ~/.bashrc
# Repo-local venv (repo is on the big disk too); override with PRECAL_VENV.
export PRECAL_VENV="${PRECAL_VENV:-${_PRECAL_REPO}/.venv}"

# Make a scratch-installed uv (and any ~/.local/.cargo uv) reachable.
export PATH="${PRECAL_SCRATCH}/bin:${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"

mkdir -p "${PRECAL_SCRATCH}" "${PRECAL_HF_HOME}" "${UV_INSTALL_DIR}" \
         "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}" "${XDG_DATA_HOME}" \
         "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${PIP_CACHE_DIR}" 2>/dev/null || true
