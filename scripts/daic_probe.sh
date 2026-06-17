#!/bin/bash
# ---------------------------------------------------------------------------
# preCal :: DAIC discovery probe
# ---------------------------------------------------------------------------
# Run on a DAIC LOGIN node to confirm the UNVERIFIED assumptions baked into the
# sbatch headers (partition names, GPU gres/constraint for the RTX Pro 6000,
# scratch path/quota, container runtime, module names, egress policy).
#
#   bash scripts/daic_probe.sh                 # human-readable report
#   bash scripts/daic_probe.sh > daic.txt      # capture for sharing
#
# CONTEXT (from research, June 2026):
#   * DAIC requests GPUs via --gres=gpu:<type>:<count> (e.g. gpu:a40:1).
#   * Large-VRAM nodes carry a memory FEATURE (e.g. gpumem32) used with --constraint.
#   * Partitions are research-group named (ewi-insy, me-cor, ...) plus a 'general'
#     partition; there is NO documented default partition.
#   * The PUBLIC DAIC docs list ONLY L40/A40/RTX2080Ti(turing)/V100 — NO Blackwell /
#     RTX Pro 6000. So the RTX Pro 6000 gres name + sm_120 constraint are UNKNOWN
#     until this probe is run on the live cluster (see DESIGN.md open questions).
#   Docs: https://daic.tudelft.nl/docs/system/compute-nodes/
#         https://daic.tudelft.nl/docs/manual/job-submission/slurm-basics/
#
# The grep lines below try to SURFACE any 'blackwell' / 'pro6000' / 'sm_120' /
# '96G' gres so you can copy the exact strings into the sbatch headers
# (replace the PLACEHOLDER --partition / --gres / --constraint lines).
# ---------------------------------------------------------------------------
set -uo pipefail

hr() { printf '\n========== %s ==========\n' "$*"; }

hr "host / user / date"
echo "host=$(hostname)  user=${USER}  date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

hr "PARTITIONS (sinfo)"
if command -v sinfo >/dev/null 2>&1; then
  echo "-- partitions, cpus, mem, gres per node --"
  sinfo --all -o '%P %a %l %c %m %G %D %N' | column -t 2>/dev/null || \
    sinfo --all -o '%P %a %l %c %m %G %D %N'
  echo
  echo "-- distinct GRES strings on the cluster (look for the RTX Pro 6000 / 96G entry) --"
  sinfo --all -h -o '%G' | tr ',' '\n' | sort -u
else
  echo "sinfo not found (are you on a login node?)"
fi

hr "NODE FEATURES / CONSTRAINTS (look for blackwell / sm_120 / gpumem* / 96)"
if command -v scontrol >/dev/null 2>&1; then
  scontrol show node 2>/dev/null \
    | grep -iE 'NodeName|Gres|ActiveFeatures|AvailableFeatures' \
    | grep -iE 'NodeName|Gres=|Features' \
    | sed 's/^[[:space:]]*//'
  echo
  echo "-- grep for Blackwell / Pro 6000 / sm120 / 96G hints --"
  scontrol show node 2>/dev/null | grep -iEo '[a-z0-9_]*(blackwell|pro6000|pro_6000|6000|sm_?120|96g|gpumem[0-9]+)[a-z0-9_]*' | sort -u \
    || echo "(no obvious Blackwell/Pro6000 token found — confirm gres name manually)"
else
  echo "scontrol not found"
fi

hr "GPU TYPES via scontrol Gres counts"
if command -v scontrol >/dev/null 2>&1; then
  scontrol show node 2>/dev/null | grep -oiE 'gpu:[a-z0-9_]+(:[0-9]+)?' | sort | uniq -c | sort -rn
fi

hr "ACCOUNT / ASSOCIATIONS (do you need -A / --account ?)"
if command -v sacctmgr >/dev/null 2>&1; then
  sacctmgr -nP show assoc user="${USER}" format=Account,Partition,QOS 2>/dev/null \
    || echo "sacctmgr show assoc returned nothing"
else
  echo "sacctmgr not found"
fi

hr "QOS limits"
if command -v sacctmgr >/dev/null 2>&1; then
  sacctmgr -nP show qos format=Name,MaxWall,MaxTRESPU,MaxJobsPU,Priority 2>/dev/null | head -40
fi

hr "MODULES (look for cuda / apptainer|singularity / miniconda|anaconda)"
if command -v module >/dev/null 2>&1; then
  echo "-- module avail (filtered) --"
  ( module avail 2>&1 || true ) | grep -iE 'cuda|cudnn|apptainer|singularity|conda|mamba|nccl' | sort -u
  echo
  echo "-- key tools on PATH --"
else
  echo "no 'module' command; checking PATH directly"
fi
for t in apptainer singularity conda mamba nvidia-smi python git-lfs curl rsync hf huggingface-cli; do
  if command -v "$t" >/dev/null 2>&1; then printf "  found: %-16s -> %s\n" "$t" "$(command -v "$t")"
  else printf "  MISSING: %s\n" "$t"; fi
done

hr "CONTAINER RUNTIME version"
if command -v apptainer >/dev/null 2>&1; then apptainer --version
elif command -v singularity >/dev/null 2>&1; then singularity --version
else echo "no apptainer/singularity"; fi

hr "SCRATCH / STORAGE & QUOTA"
echo "HOME=${HOME}"
echo "TMPDIR=${TMPDIR:-<unset>}"
echo "PRECAL_SCRATCH=${PRECAL_SCRATCH:-<unset>}"
echo "-- candidate shared scratch roots --"
for p in "/tudelft.net/staff-umbrella" "/scratch" "/scratch/${USER}" "${TMPDIR:-/tmp}" "${PRECAL_SCRATCH:-}"; do
  [[ -n "$p" && -e "$p" ]] && { echo "  exists: $p"; df -h "$p" 2>/dev/null | sed 's/^/    /'; }
done
echo "-- quota (if available) --"
quota -s 2>/dev/null | sed 's/^/  /' || echo "  'quota' not available"

hr "EGRESS POLICY (can THIS node reach the internet / HF?)"
echo "-- DNS + HTTPS reachability to huggingface.co (expect OK on login, FAIL on compute) --"
if command -v curl >/dev/null 2>&1; then
  if curl -fsS --max-time 8 -o /dev/null -w "  huggingface.co HTTP %{http_code}\n" https://huggingface.co; then
    echo "  -> egress AVAILABLE here (suitable for stage/publish)"
  else
    echo "  -> egress BLOCKED/unreachable here (treat as a compute/offline node)"
  fi
else
  echo "  curl missing; cannot test egress"
fi

hr "NEXT STEPS"
cat <<'EOF'
  1) Copy the real GPU gres line (e.g. gpu:rtx_pro_6000:1 or gpu:blackwell:1) into:
       slurm/embed.sbatch, slurm/eval.sbatch   (#SBATCH --gres=...)
  2) Copy the matching --constraint feature (sm_120 / blackwell / gpumem96) likewise.
  3) Set #SBATCH --partition=<gpu partition> and (if required) --account=<acct>
     in embed/eval/stage/shard/index/merge/publish sbatch headers.
  4) Set PRECAL_SCRATCH in ~/.precal.env to the confirmed shared scratch root.
  5) Re-run scripts/check_env.sh on a GPU node to confirm sm_120 + offline + scratch.
EOF
