#!/bin/bash
# ---------------------------------------------------------------------------
# preCal :: pre-flight environment validation
# ---------------------------------------------------------------------------
#   bash scripts/check_env.sh           # full report (exit 0 if all hard checks pass)
#   bash scripts/check_env.sh --quick   # fast subset for inside sbatch (GPU+offline+scratch)
#
# Validates the things that silently break a Blackwell embedding run:
#   * a GPU is visible (CUDA_VISIBLE_DEVICES / nvidia-smi)
#   * the GPU is sm_120 (RTX Pro 6000 Blackwell) — the #1 risk is stock kernels
#     lacking sm_120 support
#   * HF offline flags are set on compute nodes (hf.offline=true)
#   * scratch exists and is writable; reports free space / quota
#   * the TEI SIF is present (for engine=tei)
# Hard failures -> non-zero exit; advisories -> warnings only.
# ---------------------------------------------------------------------------
set -uo pipefail

QUICK=0
[[ "${1:-}" == "--quick" ]] && QUICK=1

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${PRECAL_SCRATCH:=${HOME}/precal-scratch}"
: "${PRECAL_HF_HOME:=${PRECAL_SCRATCH}/hf}"
: "${PRECAL_TEI_SIF:=${PRECAL_SCRATCH}/tei_120-1.9.sif}"

fail=0
warn=0
ok()   { echo "  [OK]   $*"; }
note() { echo "  [..]   $*"; }
wrn()  { echo "  [WARN] $*"; warn=$((warn+1)); }
err()  { echo "  [FAIL] $*"; fail=$((fail+1)); }

echo "== preCal check_env ($([[ $QUICK -eq 1 ]] && echo quick || echo full)) on $(hostname) =="

# --------------------------- GPU visibility --------------------------------
echo "-- GPU --"
if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi -L >/dev/null 2>&1; then
    nvidia-smi -L | sed 's/^/  /'
    ok "nvidia-smi sees $(nvidia-smi -L | wc -l | tr -d ' ') GPU(s)"
  else
    err "nvidia-smi present but lists no GPUs (no allocation / driver issue)"
  fi
  # sm_120 / compute capability 12.0 check (Blackwell RTX Pro 6000).
  CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')"
  if [[ -n "${CC}" ]]; then
    note "compute capability: ${CC}"
    if [[ "${CC}" == "12.0" ]]; then ok "Blackwell sm_120 confirmed"
    else wrn "compute_cap=${CC} (expected 12.0 for RTX Pro 6000); ensure sm_120 kernels"; fi
  else
    wrn "could not read compute_cap (old nvidia-smi?); verify sm_120 manually"
  fi
else
  err "nvidia-smi not found — not on a GPU node, or no CUDA driver"
fi

# --------------------------- sm_120 kernel probe ---------------------------
# Confirm torch (if importable) actually has sm_120 in its kernel list. This is the
# concrete #1-risk check: stock wheels often omit sm_120 -> kernel-launch failure.
if [[ $QUICK -eq 0 ]] && command -v python >/dev/null 2>&1; then
  echo "-- torch sm_120 --"
  python - <<'PY' || wrn "torch sm_120 probe failed (TEI container ships its own kernels, so this is advisory)"
import sys
try:
    import torch
except Exception as e:
    print(f"  [..]   torch not importable in this env: {e}")
    sys.exit(0)
archs = torch.cuda.get_arch_list() if torch.cuda.is_available() or hasattr(torch.cuda, "get_arch_list") else []
print(f"  [..]   torch {torch.__version__} arch_list={archs}")
if any("120" in a for a in archs):
    print("  [OK]   torch built with sm_120 kernels")
else:
    print("  [WARN] torch arch_list lacks sm_120 — prefer the TEI container engine")
PY
fi

# --------------------------- HF offline flags ------------------------------
echo "-- HF offline --"
if [[ "${HF_HUB_OFFLINE:-0}" == "1" && "${TRANSFORMERS_OFFLINE:-0}" == "1" ]]; then
  ok "HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 (compute-node offline mode)"
else
  wrn "HF offline flags not both set (HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-unset}, TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-unset}) — fine on stage/publish nodes, REQUIRED on compute"
fi
if [[ -d "${PRECAL_HF_HOME}" ]]; then ok "HF_HOME exists: ${PRECAL_HF_HOME}"
else wrn "HF_HOME missing: ${PRECAL_HF_HOME} (run stage first)"; fi

# --------------------------- scratch ---------------------------------------
echo "-- scratch --"
if [[ -d "${PRECAL_SCRATCH}" ]]; then
  if [[ -w "${PRECAL_SCRATCH}" ]]; then ok "scratch writable: ${PRECAL_SCRATCH}"
  else err "scratch NOT writable: ${PRECAL_SCRATCH}"; fi
  df -h "${PRECAL_SCRATCH}" 2>/dev/null | sed 's/^/  /' || true
else
  err "scratch dir missing: ${PRECAL_SCRATCH} (source scripts/activate_env.sh or set PRECAL_SCRATCH)"
fi

# --------------------------- TEI SIF ---------------------------------------
if [[ $QUICK -eq 0 ]]; then
  echo "-- TEI image --"
  if [[ -f "${PRECAL_TEI_SIF}" ]]; then ok "TEI SIF present: ${PRECAL_TEI_SIF}"
  else wrn "TEI SIF missing: ${PRECAL_TEI_SIF} (run scripts/pull_image.sh on a login node; not needed for engine!=tei)"; fi
fi

# --------------------------- summary ---------------------------------------
echo "== summary: ${fail} failure(s), ${warn} warning(s) =="
[[ ${fail} -eq 0 ]] || { echo "check_env: HARD FAILURES present"; exit 1; }
exit 0
