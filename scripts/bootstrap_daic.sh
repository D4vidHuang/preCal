#!/bin/bash
# ---------------------------------------------------------------------------
# preCal :: DAIC one-command bootstrap  (run on an INTERNET-CAPABLE login node)
# ---------------------------------------------------------------------------
# Does the whole login-node side, then submits the smoke GPU job so you can sleep:
#   1) create the conda env            (online)
#   2) auto-detect GPU -> ~/.precal.env
#   3) check HF auth + the-stack-dedup access (gated)
#   4) pull the TEI sm_120 SIF         (online)
#   5) stage smoke corpus + 0.6B model (online)
#   6) submit the smoke GPU job        (runs offline on a GPU node)
#
# Run it so it survives an SSH disconnect:
#   tmux new -s precal 'bash scripts/bootstrap_daic.sh 2>&1 | tee bootstrap.log; bash'
#
# Re-runnable: every step is idempotent (env exists / SIF exists / staged files skip).
# ---------------------------------------------------------------------------
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO_ROOT"
say() { printf '\n\033[1;35m== %s ==\033[0m\n' "$*"; }

say "1/6  Python env (uv)"
bash scripts/setup_env.sh

say "2/6  auto-detect DAIC GPU/partition -> ~/.precal.env"
bash scripts/daic_autodetect.sh || echo "[bootstrap] WARN: autodetect incomplete — review ~/.precal.env before the full run"

# shellcheck source=/dev/null
source scripts/activate_env.sh

say "3/6  HF auth + gated-corpus access"
if ! python - <<'PY'
import sys
try:
    from huggingface_hub import HfApi, whoami
    print("[bootstrap] HF user:", whoami().get("name"))
    HfApi().dataset_info("bigcode/the-stack-dedup")
    print("[bootstrap] the-stack-dedup: ACCESS OK")
except Exception as e:
    print("[bootstrap] BLOCKED:", type(e).__name__, str(e)[:200]); sys.exit(7)
PY
then
  cat <<'EOF'

[bootstrap] STOP: cannot access bigcode/the-stack-dedup. Do ONE of these, then re-run:
   a) no token on this node?   ->  huggingface-cli login
   b) not accepted the terms?  ->  open https://huggingface.co/datasets/bigcode/the-stack-dedup
                                    and click "Agree and access repository" (~10s)
   re-run:  bash scripts/bootstrap_daic.sh
EOF
  exit 7
fi

say "4/6  pull TEI sm_120 image (SIF)"
bash scripts/pull_image.sh --config configs/smoke.yaml

say "5/6  stage smoke corpus + 0.6B model (online)"
python -m precal.cli stage --config configs/smoke.yaml

say "6/6  submit smoke GPU job (offline compute node)"
bash scripts/submit.sh --gpu slurm/smoke.sbatch

cat <<EOF

\033[1;32m== bootstrap done — you can sleep now 😴 ==\033[0m
  queue:    squeue --me
  live log: tail -f precal-smoke-*.out
  result:   cat \$PRECAL_SCRATCH/precal-smoke/eval/*.json     # in the morning
            (look for recall@k / MRR@10 / nDCG@10 — sanity-checks the 0.6B model end-to-end)

  Full 8+ GPU run on the-stack-dedup (after smoke looks good): see RUN_ON_DAIC.md  ->  "Full run".
EOF
