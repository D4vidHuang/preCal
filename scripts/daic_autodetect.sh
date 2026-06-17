#!/bin/bash
# ---------------------------------------------------------------------------
# preCal :: DAIC GPU auto-detect  ->  ~/.precal.env
# ---------------------------------------------------------------------------
# Run on a DAIC LOGIN node. Greps live sinfo/scontrol for the RTX Pro 6000
# (Blackwell, sm_120) gres + constraint feature + a GPU partition + your account,
# and writes them to ~/.precal.env so scripts/submit.sh can inject them as sbatch
# overrides. Best-effort: it PREFERS a Pro 6000 / Blackwell card but prints every
# candidate so you can sanity-check (or override) in the morning if it guessed.
#
#   bash scripts/daic_autodetect.sh              # detect + write ~/.precal.env
#   bash scripts/daic_autodetect.sh --print-only # just show what it would write
#
# Override anything by editing ~/.precal.env afterwards (it wins over detection
# on the next run only if you pass --keep; otherwise re-running re-detects).
# ---------------------------------------------------------------------------
set -uo pipefail

PRINT_ONLY=0
[[ "${1:-}" == "--print-only" ]] && PRINT_ONLY=1
ENVFILE="${HOME}/.precal.env"

note() { printf '\033[1;36m[autodetect]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[autodetect] WARN:\033[0m %s\n' "$*"; }

if ! command -v sinfo >/dev/null 2>&1; then
  warn "sinfo not found — are you on a DAIC login node? Nothing detected."
  warn "Set values by hand in ${ENVFILE} (see RUN_ON_DAIC.md)."
  exit 1
fi

# --- 1. all GPU gres TYPES on the cluster (gpu:<type>) -----------------------
mapfile -t GPU_TYPES < <(sinfo --all -h -o '%G' 2>/dev/null \
  | tr ',' '\n' | grep -oiE 'gpu:[a-z0-9_.]+' | sed -E 's/^gpu://I' | sort -u)
note "GPU types seen on cluster: ${GPU_TYPES[*]:-<none>}"

# Prefer a Pro 6000 / Blackwell card; else leave unset (fall back to in-file gres).
GPU_TYPE=""
for pat in 'pro.?6000' 'rtx_?pro' 'blackwell' 'b200' 'gb200' '6000'; do
  for t in "${GPU_TYPES[@]:-}"; do
    if [[ "$t" =~ $pat ]]; then GPU_TYPE="$t"; break 2; fi
  done
done
if [[ -n "$GPU_TYPE" ]]; then note "selected GPU type: gpu:${GPU_TYPE} (matched Pro6000/Blackwell)"
else warn "no Pro6000/Blackwell gres token found among [${GPU_TYPES[*]:-}]; leaving PRECAL_GPU_TYPE empty (will use sbatch in-file --gres). If a card above IS the Pro 6000, set PRECAL_GPU_TYPE in ${ENVFILE}."; fi

# --- 2. partitions that actually host GPUs -----------------------------------
mapfile -t GPU_PARTS < <(sinfo --all -h -o '%P %G' 2>/dev/null \
  | awk 'tolower($2) ~ /gpu:/{gsub(/\*/,"",$1); print $1}' | sort -u)
note "GPU partitions: ${GPU_PARTS[*]:-<none>}"
PARTITION=""
if [[ -n "$GPU_TYPE" ]]; then
  # partition of nodes that actually carry the chosen card
  PARTITION="$(sinfo --all -h -o '%P %G' 2>/dev/null \
    | awk -v t="gpu:${GPU_TYPE}" 'tolower($2) ~ tolower(t){gsub(/\*/,"",$1); print $1; exit}')"
fi
[[ -z "$PARTITION" && ${#GPU_PARTS[@]} -eq 1 ]] && PARTITION="${GPU_PARTS[0]}"
if [[ -n "$PARTITION" ]]; then note "selected partition: ${PARTITION}"
else warn "could not pick a single GPU partition (candidates: ${GPU_PARTS[*]:-none}); leaving PRECAL_PARTITION empty (sbatch in-file --partition is 'general'). Set it in ${ENVFILE} if jobs pend forever."; fi

# --- 3. constraint feature matching the card (sm_120 / blackwell / gpumem) ----
CONSTRAINT=""
if command -v scontrol >/dev/null 2>&1; then
  FEATS="$(scontrol show node 2>/dev/null | grep -ioE '(Active|Available)Features=[^ ]*' | sed -E 's/.*=//' | tr ',' '\n' | sort -u)"
  for pat in 'sm_?120' 'blackwell' 'gpumem9[0-9]' 'gpumem96' '96g'; do
    m="$(echo "$FEATS" | grep -iE "$pat" | head -1 || true)"
    [[ -n "$m" ]] && { CONSTRAINT="$m"; break; }
  done
fi
[[ -n "$CONSTRAINT" ]] && note "selected constraint: ${CONSTRAINT}" || warn "no sm_120/blackwell/gpumem feature found; leaving PRECAL_CONSTRAINT empty (drop --constraint if jobs never start). Features seen: $(echo "${FEATS:-}" | tr '\n' ' ')"

# --- 4. account (if associations require -A) ---------------------------------
ACCOUNT=""
if command -v sacctmgr >/dev/null 2>&1; then
  ACCOUNT="$(sacctmgr -nP show assoc user="${USER}" format=Account 2>/dev/null | sed '/^$/d' | sort -u | head -1)"
  [[ -n "$ACCOUNT" ]] && note "account: ${ACCOUNT}" || note "no explicit account association (likely not required)"
fi

# --- 5. scratch root (derived off the repo location, never $HOME) -------------
_PRECAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${_PRECAL_REPO}/scripts/_paths.sh"
SCRATCH="${PRECAL_SCRATCH}"
note "scratch root: ${SCRATCH}  (on the shared disk, off the \$HOME quota; override in ${ENVFILE})"

# --- 6. emit ------------------------------------------------------------------
BLOCK="$(cat <<EOF
# ~/.precal.env  — written by scripts/daic_autodetect.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ)
# Edit freely; scripts/submit.sh sources this to override the sbatch #SBATCH headers.
export PRECAL_SCRATCH="${SCRATCH}"
export PRECAL_PARTITION="${PARTITION}"
export PRECAL_ACCOUNT="${ACCOUNT}"
export PRECAL_GPU_TYPE="${GPU_TYPE}"      # -> --gres=gpu:\${PRECAL_GPU_TYPE}:<count>
export PRECAL_GRES=""                      # set this for a FULL override, e.g. gpu:rtx_pro_6000:1
export PRECAL_CONSTRAINT="${CONSTRAINT}"
EOF
)"

echo "-------------------------------------------------------------------"
echo "$BLOCK"
echo "-------------------------------------------------------------------"

if [[ "$PRINT_ONLY" -eq 1 ]]; then
  note "--print-only: not writing. Re-run without it to save to ${ENVFILE}."
  exit 0
fi
[[ -f "$ENVFILE" ]] && { cp -f "$ENVFILE" "${ENVFILE}.bak"; note "backed up existing -> ${ENVFILE}.bak"; }
printf '%s\n' "$BLOCK" > "$ENVFILE"
note "wrote ${ENVFILE}. Sanity-check it, then: bash scripts/bootstrap_daic.sh"
