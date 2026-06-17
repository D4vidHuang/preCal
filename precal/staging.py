"""Login-node staging: bring everything onto shared scratch while online.

Compute nodes are assumed OFFLINE (spec stagingStrategy), so ALL internet I/O
lives here and runs on an internet-capable login/stage node. This stage:

  1. (one-time, manual/UI) accept the gated bigcode/the-stack-dedup terms.
  2. stream per-language the-stack-dedup parquet (content inline) to
     ``$PRECAL_SCRATCH/corpus/the-stack-dedup/lang=<l>/`` honoring
     corpus.max_files_per_lang,
  3. stage the CoIR-Retrieval/CodeSearchNet eval pack (corpus/queries/qrels) and
     code-search-net/code_search_net pairs to ``corpus/coir-csn`` / ``corpus/csn``,
  4. snapshot_download the embedding model(s) into HF_HOME, and
  5. write a stage-complete marker that gates downstream jobs.

Idempotent: re-running skips files already present. huggingface_hub / datasets
are imported lazily so the module imports clean without them; if the user runs
``stage`` without them installed, a clear error is raised at call time.
"""

from __future__ import annotations

import json
import os
import time
from typing import List, Optional

from precal.config import Config
from precal.utils import ensure_dir, get_logger, set_hf_home

logger = get_logger("precal.staging")

STAGE_MARKER = "STAGE_COMPLETE.json"

# CoIR-CSN per-language subset config names on the Hub.
_COIR_DATASET = "CoIR-Retrieval/CodeSearchNet"
_CSN_DATASET = "code-search-net/code_search_net"


def stage_marker_path(cfg: Config) -> str:
    return os.path.join(cfg.corpus_dir, STAGE_MARKER)


def is_staged(cfg: Config) -> bool:
    return os.path.exists(stage_marker_path(cfg))


def staged_model_dir(cfg: Config, model_id: Optional[str] = None) -> str:
    """Absolute path to a staged model dir: ``${HF_HOME}/staged/<model_id>``.

    LOCKED CONTRACT (D3): the model_id keeps its slash (e.g.
    ``staged/Qwen/Qwen3-Embedding-4B``). This is the EXACT directory the TEI
    launchers pass as ``--model-id`` (the ops shells build
    ``${HF_HOME}/staged/${MODEL_ID}``), so Python and ops cannot diverge.
    """
    mid = model_id or cfg.model.id
    return os.path.join(cfg.paths.hf_home, "staged", mid)


def _accept_gated_terms(dataset_id: str) -> None:
    """Best-effort: the gated terms for the-stack-dedup must be accepted once via
    the Hub UI/API by the authenticated user. We cannot click a checkbox here;
    we surface a clear instruction and let the download fail loudly if not done.
    """
    logger.info(
        "Ensure gated terms for %s are accepted by the authenticated HF user "
        "(one-time, via the dataset page). Downloads 403 otherwise.",
        dataset_id,
    )


def stage_corpus(cfg: Config, languages: Optional[List[str]] = None) -> None:
    """Stream the-stack-dedup per-language parquet (content inline) to scratch.

    Uses datasets streaming + the language config so we never materialize the
    whole corpus. Honors corpus.max_files_per_lang (0 = no cap). Writes one or
    more parquet parts per language under corpus/the-stack-dedup/lang=<l>/.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "stage_corpus needs `datasets` + `pyarrow` (pip install datasets pyarrow)."
        ) from exc

    languages = languages or list(cfg.corpus.languages)
    _accept_gated_terms(cfg.corpus.dataset_id)
    cap = cfg.corpus.max_files_per_lang

    for language in languages:
        out_dir = ensure_dir(
            os.path.join(cfg.corpus_dir, "the-stack-dedup", f"lang={language}")
        )
        done_marker = os.path.join(out_dir, "_STAGED")
        if os.path.exists(done_marker):
            logger.info("Corpus already staged for %s (skip).", language)
            continue

        logger.info("Streaming the-stack-dedup data/%s (cap=%s)...", language, cap or "none")
        # The Stack uses per-language config 'data/<lang>'. content is inline.
        ds = load_dataset(
            cfg.corpus.dataset_id,
            data_dir=f"data/{language}",
            revision=cfg.corpus.revision,
            split="train",
            streaming=True,
        )

        rows: List[dict] = []
        part = 0
        written = 0
        ROWS_PER_PART = 50000
        for i, row in enumerate(ds):
            if cap and i >= cap:
                break
            rows.append(_project_stack_row(row, language, cfg))
            if len(rows) >= ROWS_PER_PART:
                _write_corpus_part(out_dir, part, rows)
                written += len(rows)
                rows = []
                part += 1
        if rows:
            _write_corpus_part(out_dir, part, rows)
            written += len(rows)

        # LOCKED CONTRACT (staging non-empty assertion): a staged language MUST
        # produce at least one non-empty part whose rows carry content + repo +
        # license, else the offline embed/chunk stages would silently produce an
        # empty corpus. Fail the stage loudly here instead.
        _assert_corpus_parts_nonempty(out_dir, language, written)

        with open(done_marker, "w", encoding="utf-8") as fh:
            json.dump({"language": language, "files": written}, fh)
        logger.info("Staged %d %s files -> %s", written, language, out_dir)


def _project_stack_row(row: dict, language: str, cfg: Config) -> dict:
    """Project a raw Stack row to the minimal fields downstream needs.

    NOTE (openQuestion): the exact Stack v1.2 field names for repo/path/license
    must be confirmed against the gated parquet. We read several known aliases
    defensively so the chunk stage always gets repo_name/path/license/content.
    """
    content = row.get("content") or row.get("text") or ""
    repo = (
        row.get("max_stars_repo_name")
        or row.get("repository_name")
        or row.get("repo_name")
        or ""
    )
    path = (
        row.get("max_stars_repo_path")
        or row.get("path")
        or row.get("filename")
        or ""
    )
    licenses = (
        row.get("detected_licenses")
        or row.get("max_stars_repo_licenses")
        or row.get("license")
        or []
    )
    if isinstance(licenses, list):
        spdx = licenses[0] if licenses else ""
    else:
        spdx = str(licenses)
    return {
        "content": content,
        "repo_name": repo,
        "path": path,
        "language": language,
        "license": spdx,
    }


def _write_corpus_part(out_dir: str, part: int, rows: List[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(rows)
    out = os.path.join(out_dir, f"part-{part:05d}.parquet")
    tmp = out + ".tmp"
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, out)


def _assert_corpus_parts_nonempty(out_dir: str, language: str, written: int) -> None:
    """Fail the stage unless this language produced usable corpus parts.

    Checks (a) at least one part-*.parquet exists, (b) total rows > 0, and (c)
    the projected rows carry the fields downstream chunk/embed need
    (content + repo + license). The-stack-dedup's exact field names are an open
    question (see openQuestions); _project_stack_row reads aliases defensively,
    but if the projection still yields all-empty content/repo/license the
    pipeline would embed nothing — so we surface it here, not 3 stages later.
    """
    import pyarrow.parquet as pq

    parts = sorted(
        p for p in os.listdir(out_dir) if p.startswith("part-") and p.endswith(".parquet")
    )
    if not parts or written <= 0:
        raise RuntimeError(
            f"stage_corpus: language {language!r} produced no corpus rows "
            f"(parts={len(parts)}, written={written}) under {out_dir}. The Stack "
            f"per-language config/data_dir or gated-terms acceptance is likely "
            f"wrong; refusing to mark this language staged."
        )

    required = ("content", "repo_name", "license")
    first = os.path.join(out_dir, parts[0])
    table = pq.read_table(first)
    missing = [c for c in required if c not in table.column_names]
    if missing:
        raise RuntimeError(
            f"stage_corpus: language {language!r} parts are missing required "
            f"column(s) {missing} (have {table.column_names}). Fix "
            f"_project_stack_row's Stack field aliases before staging."
        )
    # Require non-empty CONTENT (a corpus of blank files is useless) and at least
    # some rows carrying repo + license provenance for the license allowlist.
    contents = table.column("content").to_pylist()
    if not any((c or "").strip() for c in contents):
        raise RuntimeError(
            f"stage_corpus: language {language!r} first part has all-empty "
            f"'content'; the Stack row->content projection is wrong."
        )
    repos = table.column("repo_name").to_pylist()
    licenses = table.column("license").to_pylist()
    if not any((r or "").strip() for r in repos):
        raise RuntimeError(
            f"stage_corpus: language {language!r} first part has all-empty "
            f"'repo_name'; repo-level eval split + provenance would break."
        )
    if not any((l or "").strip() for l in licenses):
        raise RuntimeError(
            f"stage_corpus: language {language!r} first part has all-empty "
            f"'license'; the permissive-license allowlist filter would drop "
            f"everything."
        )


def stage_eval(cfg: Config, languages: Optional[List[str]] = None) -> None:
    """Stage the CoIR-CSN eval pack + CodeSearchNet pairs to scratch."""
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover
        raise ImportError("stage_eval needs `datasets`.") from exc

    languages = languages or list(cfg.corpus.languages)
    for language in languages:
        coir_dir = ensure_dir(os.path.join(cfg.corpus_dir, "coir-csn", language))
        marker = os.path.join(coir_dir, "_STAGED")
        if os.path.exists(marker):
            logger.info("CoIR-CSN already staged for %s (skip).", language)
            continue
        logger.info("Staging CoIR-CSN (%s)...", language)
        # CoIR-CSN ships corpus/queries/qrels configs per language. We persist
        # them as parquet/json for the offline eval stage to consume.
        try:
            for cfg_name in ("corpus", "queries"):
                sub = load_dataset(_COIR_DATASET, f"{language}-{cfg_name}", split="train")
                sub.to_parquet(os.path.join(coir_dir, f"{cfg_name}.parquet"))
            qrels = load_dataset(_COIR_DATASET, f"{language}-qrels", split="test")
            qrels.to_parquet(os.path.join(coir_dir, "qrels_test.parquet"))
        except Exception as exc:
            logger.warning(
                "CoIR-CSN config layout for %s differs (%s); falling back to "
                "mteb-managed download at eval time.",
                language,
                exc,
            )
        # CodeSearchNet pairs (func docs) for query supervision.
        csn_dir = ensure_dir(os.path.join(cfg.corpus_dir, "csn", language))
        try:
            csn = load_dataset(_CSN_DATASET, language, split="train")
            csn.to_parquet(os.path.join(csn_dir, "train.parquet"))
        except Exception as exc:
            logger.warning("CodeSearchNet %s staging skipped: %s", language, exc)
        with open(marker, "w", encoding="utf-8") as fh:
            fh.write("ok")


def stage_model(cfg: Config, also_smoke: bool = True) -> None:
    """snapshot_download the embedding model(s) into HF_HOME on scratch."""
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover
        raise ImportError("stage_model needs `huggingface_hub`.") from exc

    set_hf_home(cfg.paths.hf_home)
    ensure_dir(cfg.paths.hf_home)

    model_ids = [cfg.model.id]
    if also_smoke and cfg.model.id != "Qwen/Qwen3-Embedding-0.6B":
        model_ids.append("Qwen/Qwen3-Embedding-0.6B")  # smoke / fallback

    for mid in model_ids:
        # LOCKED CONTRACT (D3): materialize each model at EXACTLY
        # ${HF_HOME}/staged/<model_id> (the model_id keeps its slash, e.g.
        # staged/Qwen/Qwen3-Embedding-4B). This is the literal path every TEI
        # launcher passes as --model-id, so the served model resolves offline
        # from a real local dir (NOT a symlinked blob cache, which apptainer
        # binds can't follow across the container boundary).
        staged_dir = staged_model_dir(cfg, mid)
        ensure_dir(staged_dir)
        logger.info("snapshot_download %s -> %s", mid, staged_dir)
        snapshot_download(
            repo_id=mid,
            revision=cfg.model.revision if mid == cfg.model.id else "main",
            local_dir=staged_dir,
            local_dir_use_symlinks=False,
        )


def write_stage_marker(cfg: Config, languages: List[str]) -> None:
    """Write the stage-complete marker that gates downstream jobs."""
    ensure_dir(cfg.corpus_dir)
    marker = {
        "run": cfg.run.name,
        "languages": languages,
        "corpus": f"{cfg.corpus.dataset_id}@{cfg.corpus.revision}",
        "model": f"{cfg.model.id}@{cfg.model.revision}",
        "timestamp": time.time(),
    }
    with open(stage_marker_path(cfg), "w", encoding="utf-8") as fh:
        json.dump(marker, fh, indent=2)
    logger.info("Wrote stage-complete marker -> %s", stage_marker_path(cfg))


def run_stage(
    cfg: Config,
    languages: Optional[List[str]] = None,
    skip_model: bool = False,
    skip_corpus: bool = False,
) -> None:
    """Top-level staging driver invoked by the CLI `stage` subcommand."""
    languages = languages or list(cfg.corpus.languages)
    set_hf_home(cfg.paths.hf_home)
    if not skip_corpus:
        stage_corpus(cfg, languages)
        stage_eval(cfg, languages)
    if not skip_model:
        stage_model(cfg)
    write_stage_marker(cfg, languages)
