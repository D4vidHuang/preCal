#!/bin/bash
# ---------------------------------------------------------------------------
# preCal :: pull the TEI Blackwell (sm_120) image to a scratch SIF
# ---------------------------------------------------------------------------
# Run on an INTERNET-CAPABLE login/stage node (compute nodes are offline).
#
#   bash scripts/pull_image.sh [--config configs/full_v1.yaml] [--image IMG] [--sif PATH] [--force]
#
# Pulls ghcr.io/huggingface/text-embeddings-inference:120-1.9 (the ONLY prebuilt
# sm_120 image as of mid-2026) into $PRECAL_TEI_SIF so embed/eval can run it
# offline via apptainer/singularity. Idempotent: skips if the SIF already exists.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/activate_env.sh"

CONFIG="configs/full_v1.yaml"
IMAGE=""
SIF="${PRECAL_TEI_SIF}"
FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --image)  IMAGE="$2";  shift 2 ;;
    --sif)    SIF="$2";    shift 2 ;;
    --force)  FORCE=1;     shift ;;
    *) echo "pull_image.sh: unknown arg '$1'" >&2; exit 2 ;;
  esac
done

if [[ -z "${IMAGE}" ]]; then
  IMAGE="$(python "${REPO_ROOT}/scripts/cfg.py" "${CONFIG}" engine.image \
            ghcr.io/huggingface/text-embeddings-inference:120-1.9)"
fi

echo "[pull_image] image=${IMAGE}"
echo "[pull_image] sif=${SIF}"

if [[ -f "${SIF}" && "${FORCE}" -ne 1 ]]; then
  echo "[pull_image] SIF already present -> skip (use --force to overwrite)."
  exit 0
fi

mkdir -p "$(dirname "${SIF}")"

# Prefer apptainer; fall back to singularity (DAIC availability unconfirmed — daic_probe.sh).
RUNTIME=""
if command -v apptainer >/dev/null 2>&1; then RUNTIME=apptainer
elif command -v singularity >/dev/null 2>&1; then RUNTIME=singularity
else
  echo "[pull_image] FATAL: neither apptainer nor singularity found. Load a container module (daic_probe.sh)." >&2
  exit 3
fi

# Build on a node-local, chmod-capable disk (APPTAINER_TMPDIR, set by _paths.sh to
# /tmp/$USER/...). The staff-umbrella network FS forbids chmod, which breaks
# apptainer's cache/build. --disable-cache avoids a persistent OCI cache entirely.
# Then copy the finished SIF (a plain file) onto scratch.
mkdir -p "${APPTAINER_TMPDIR}"
LOCAL_SIF="${APPTAINER_TMPDIR}/$(basename "${SIF}").build.$$"
echo "[pull_image] using ${RUNTIME} --disable-cache -> building (node-local) ${LOCAL_SIF}"
"${RUNTIME}" pull --disable-cache --force "${LOCAL_SIF}" "docker://${IMAGE}"
echo "[pull_image] copying SIF onto scratch: ${SIF}"
cp -f "${LOCAL_SIF}" "${SIF}.tmp.$$" && mv -f "${SIF}.tmp.$$" "${SIF}"   # rename = no chmod
rm -f "${LOCAL_SIF}"
echo "[pull_image] done: ${SIF}"
