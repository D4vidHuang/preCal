# ===========================================================================
# preCal :: Makefile  (convenience wrappers over scripts/ + python -m precal.cli)
# ===========================================================================
# All real work goes through `python -m precal.cli <subcommand>` (cliContracts)
# or the scripts/ helpers; these targets just wire the common invocations.
#
# Config selection:  make <target> CONFIG=configs/full_v1.yaml   (default below)
# GPU concurrency:    make embed   CONC=%8
# ---------------------------------------------------------------------------

SHELL        := /bin/bash
CONFIG       ?= configs/full_v1.yaml
SMOKE_CONFIG ?= configs/smoke.yaml
CONC         ?= %8
GPU_ID       ?= 0
REPO_ID      ?= D4vidHuang/precal-code-embeddings
PY           := python -m precal.cli

# Resolve run name + manifest path lazily (needs the env sourced for PRECAL_SCRATCH).
RUN_NAME      = $(shell python scripts/cfg.py $(CONFIG) run.name precal-v1)
MANIFEST      = $(PRECAL_SCRATCH)/$(RUN_NAME)/shards/manifest.jsonl

.DEFAULT_GOAL := help

.PHONY: help env image probe autodetect bootstrap check smoke smoke-slurm \
        stage shard embed embed-local index index-local merge eval publish \
        upload slurm-stage slurm-shard slurm-eval slurm-publish lint clean

help:  ## Show this help
	@echo "preCal targets (CONFIG=$(CONFIG)):"
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}'

# --------------------------- setup -----------------------------------------
env:   ## Create/update the conda env (env/environment.yml)
	bash scripts/setup_env.sh

image: ## Pull the TEI sm_120 SIF (login node, online)
	bash scripts/pull_image.sh --config $(CONFIG)

probe: ## Discover DAIC partitions/gres/quotas/modules (login node)
	bash scripts/daic_probe.sh

autodetect: ## Auto-detect GPU/partition -> ~/.precal.env (login node)
	bash scripts/daic_autodetect.sh

bootstrap: ## ONE COMMAND: env+detect+stage+submit smoke GPU job (login node, online)
	bash scripts/bootstrap_daic.sh

check: ## Pre-flight env validation (GPU sm_120, offline, scratch)
	bash scripts/check_env.sh

# --------------------------- pipeline (local / inline) ---------------------
smoke: ## End-to-end smoke INLINE on this node's GPU (stage->...->eval, 0.6B)
	bash scripts/smoke.sh $(GPU_ID)

smoke-slurm: ## Submit the smoke as a 1-GPU batch job (stage must be done; uses ~/.precal.env)
	bash scripts/submit.sh --gpu slurm/smoke.sbatch

stage: ## Stage corpus+eval+model+SIF to scratch (login node, online)
	$(PY) stage --config $(CONFIG)
	bash scripts/pull_image.sh --config $(CONFIG)

shard: ## Chunk + shard all languages, then stamp global manifest
	@for L in $$(python scripts/cfg.py $(CONFIG) corpus.languages | tr ',' ' '); do \
	  echo "== chunk+shard $$L =="; \
	  $(PY) chunk --config $(CONFIG) --language $$L; \
	  $(PY) shard --config $(CONFIG) --language $$L; \
	done
	$(PY) shard --config $(CONFIG)   # global contiguous shard_id consolidation

embed-local: ## Embed a single shard inline (SHARD=N) on this node's GPU
	@test -n "$(SHARD)" || (echo "set SHARD=<id>, e.g. make embed-local SHARD=0"; exit 2)
	$(PY) embed --config $(CONFIG) --shard-id $(SHARD) --resume

index-local: ## Build one shard's FAISS inline (SHARD=N)
	@test -n "$(SHARD)" || (echo "set SHARD=<id>"; exit 2)
	$(PY) index --config $(CONFIG) --shard-id $(SHARD)

eval: ## internal docstring->code retrieval eval (exact + ANN)  [INDEX=exact|ann|both SPLIT=test|valid]
	$(PY) eval --config $(CONFIG) --split $(or $(SPLIT),test) --index $(or $(INDEX),both)

publish upload: ## Assemble Hive layout + upload_large_folder to HF (login node)
	$(PY) publish --config $(CONFIG) --repo-id $(REPO_ID)

# --------------------------- pipeline (SLURM, full scale) ------------------
slurm-stage: ## sbatch the stage job
	sbatch slurm/stage.sbatch $(CONFIG)

slurm-shard: ## sbatch the per-language CPU shard array (+ reminder to consolidate)
	@NL=$$(python scripts/cfg.py $(CONFIG) corpus.languages | tr ',' '\n' | grep -c .); \
	echo "submitting shard array 0-$$(($$NL-1))"; \
	sbatch --array=0-$$(($$NL-1)) slurm/shard.sbatch $(CONFIG)

embed: ## sbatch the GPU embed array over pending shards (CONC=%8)
	@test -f "$(MANIFEST)" || (echo "manifest missing: $(MANIFEST) — run 'make shard' first"; exit 3)
	bash scripts/resubmit_pending.sh embed $(CONFIG) $(CONC)

index: ## sbatch the per-shard FAISS index array over pending shards (CONC=%8)
	@test -f "$(MANIFEST)" || (echo "manifest missing: $(MANIFEST) — run 'make shard' first"; exit 3)
	bash scripts/resubmit_pending.sh index $(CONFIG) $(CONC)

merge: ## sbatch the fat-RAM job that builds the ONE full OPQ-IVF-PQ index
	bash scripts/submit.sh slurm/merge_index.sbatch $(CONFIG)

slurm-eval: ## sbatch the 1-GPU eval job (uses ~/.precal.env GPU overrides)
	bash scripts/submit.sh --gpu slurm/eval.sbatch $(CONFIG)

slurm-publish: ## sbatch the login-node publish job
	bash scripts/submit.sh slurm/publish.sbatch $(CONFIG)

# --------------------------- quality ---------------------------------------
lint: ## Shellcheck the ops scripts + yaml sanity (best-effort, non-fatal)
	@command -v shellcheck >/dev/null 2>&1 && \
	  shellcheck -x scripts/*.sh slurm/*.sbatch || echo "shellcheck not installed — skipping"
	@command -v ruff >/dev/null 2>&1 && ruff check scripts/cfg.py || echo "ruff not installed — skipping py lint"

clean: ## Remove local *.out/*.err slurm logs in the repo root
	rm -f *.out *.err
	@echo "Note: scratch artifacts under \$$PRECAL_SCRATCH are NOT touched."
