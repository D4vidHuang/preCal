#!/bin/bash
# ---------------------------------------------------------------------------
# preCal :: sbatch submit wrapper
# ---------------------------------------------------------------------------
# Injects DAIC partition / account / GPU gres / constraint as sbatch CLI options
# (which OVERRIDE the in-file #SBATCH placeholders) from values in ~/.precal.env,
# so you NEVER hand-edit the sbatch headers. Populate ~/.precal.env once with
# scripts/daic_autodetect.sh.
#
#   bash scripts/submit.sh [--gpu] [--gpu-count N] <sbatch opts...> <script> [args]
#
# Examples:
#   bash scripts/submit.sh --gpu slurm/smoke.sbatch
#   bash scripts/submit.sh --gpu --array=0-63%8 slurm/embed.sbatch configs/full_v1.yaml
#   bash scripts/submit.sh slurm/stage.sbatch configs/full_v1.yaml      # CPU/login job
#
# Env it reads (set by daic_autodetect.sh or by hand in ~/.precal.env):
#   PRECAL_PARTITION   -> --partition=...      (applied to every job if set)
#   PRECAL_ACCOUNT     -> --account=...        (applied to every job if set)
#   PRECAL_GRES        -> --gres=...           (full string, e.g. gpu:rtx_pro_6000:1; --gpu only)
#   PRECAL_GPU_TYPE    -> --gres=gpu:<type>:N  (used if PRECAL_GRES unset; --gpu only)
#   PRECAL_CONSTRAINT  -> --constraint=...     (--gpu only)
# Anything left unset falls back to the script's in-file #SBATCH directive.
# ---------------------------------------------------------------------------
set -euo pipefail

[[ -f "${HOME}/.precal.env" ]] && source "${HOME}/.precal.env"

WANT_GPU=0
GPU_COUNT="${PRECAL_GPU_COUNT:-1}"
PASS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)        WANT_GPU=1; shift ;;
    --gpu-count)  GPU_COUNT="$2"; shift 2 ;;
    *)            PASS+=("$1"); shift ;;
  esac
done

OVR=()
[[ -n "${PRECAL_PARTITION:-}" ]] && OVR+=("--partition=${PRECAL_PARTITION}")
[[ -n "${PRECAL_ACCOUNT:-}"   ]] && OVR+=("--account=${PRECAL_ACCOUNT}")
if [[ "${WANT_GPU}" -eq 1 ]]; then
  if   [[ -n "${PRECAL_GRES:-}"     ]]; then OVR+=("--gres=${PRECAL_GRES}")
  elif [[ -n "${PRECAL_GPU_TYPE:-}" ]]; then OVR+=("--gres=gpu:${PRECAL_GPU_TYPE}:${GPU_COUNT}")
  fi
  [[ -n "${PRECAL_CONSTRAINT:-}" ]] && OVR+=("--constraint=${PRECAL_CONSTRAINT}")
fi

if [[ ${#OVR[@]} -gt 0 ]]; then
  echo "[submit] overrides: ${OVR[*]}" >&2
else
  echo "[submit] no ~/.precal.env overrides set — using in-file #SBATCH placeholders (run scripts/daic_autodetect.sh)" >&2
fi
echo "[submit] sbatch ${OVR[*]-} ${PASS[*]}" >&2
exec sbatch ${OVR[@]+"${OVR[@]}"} "${PASS[@]}"
