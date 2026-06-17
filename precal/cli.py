"""argparse entrypoint dispatching the preCal subcommands.

Subcommands (the spec's cliContracts, verbatim usage):

  stage        python -m precal.cli stage --config configs/default.yaml
                 [--languages python,java] [--skip-model] [--skip-corpus]
  chunk        python -m precal.cli chunk --config configs/default.yaml
                 --language python [--input-glob '...'] [--out DIR]
  shard        python -m precal.cli shard --config configs/default.yaml
                 [--language python] [--target-chunks 250000] [--no-pairs]
  embed        python -m precal.cli embed --config configs/default.yaml
                 --shard-id ${SLURM_ARRAY_TASK_ID} [--engine tei]
                 [--resume/--no-resume]
  index        python -m precal.cli index --config configs/default.yaml
                 [--shard-id N | --all] [--factory OVERRIDE]
  merge-index  python -m precal.cli merge-index --config configs/default.yaml
                 [--mode ondisk|sharded]
  publish      python -m precal.cli publish --config configs/default.yaml
                 [--repo-id D4vidHuang/precal-code-embeddings] [--dry-run]
  eval         python -m precal.cli eval --config configs/default.yaml
                 [--split test] [--index exact|ann|both]

Configs are merged on top of configs/default.yaml automatically when a
non-default config is passed, so `--config configs/smoke.yaml` already inherits
every default. Heavy work is delegated to the stage modules.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from precal.config import Config, load_config
from precal.utils import get_logger

logger = get_logger("precal.cli")

# The canonical defaults file; non-default configs are layered on top of it so a
# partial override file (smoke/full) inherits all spec defaults.
_DEFAULT_CONFIG = os.path.join("configs", "default.yaml")


def _load(config_path: str, overrides: Optional[dict] = None) -> Config:
    """Load config, layering it over configs/default.yaml when distinct."""
    paths: List[str] = []
    if os.path.abspath(config_path) != os.path.abspath(_DEFAULT_CONFIG) and os.path.exists(
        _DEFAULT_CONFIG
    ):
        paths.append(_DEFAULT_CONFIG)
    paths.append(config_path)
    return load_config(paths, overrides=overrides)


def _csv(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


# --------------------------------------------------------------------------- #
# Subcommand handlers
# --------------------------------------------------------------------------- #
def cmd_stage(args: argparse.Namespace) -> int:
    from precal.staging import run_stage

    cfg = _load(args.config)
    langs = _csv(args.languages) if args.languages else None
    run_stage(cfg, languages=langs, skip_model=args.skip_model, skip_corpus=args.skip_corpus)
    return 0


def cmd_chunk(args: argparse.Namespace) -> int:
    """CPU chunk stage: read staged corpus parquet -> emit chunk parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import glob as _glob

    from precal.chunking import chunk_file
    from precal.schema import CHUNK_STAGE_COLUMNS, arrow_schema
    from precal.utils import ensure_dir, human_int, load_tokenizer, set_hf_home, set_offline

    cfg = _load(args.config)
    set_offline(cfg.hf.offline)
    set_hf_home(cfg.paths.hf_home)
    language = args.language

    in_glob = args.input_glob or os.path.join(
        cfg.corpus_dir, "the-stack-dedup", f"lang={language}", "*.parquet"
    )
    files = sorted(_glob.glob(in_glob))
    if not files:
        logger.error("No staged corpus parquet for %s (glob=%s).", language, in_glob)
        return 2

    out_dir = ensure_dir(args.out or os.path.join(cfg.chunks_dir, f"lang={language}"))
    tokenizer = load_tokenizer(cfg.model.id, cfg.model.revision)
    allow = set(cfg.corpus.license_allowlist)
    corpus_snapshot = f"{cfg.corpus.dataset_id}@{cfg.corpus.revision}"
    schema = arrow_schema(include_embedding=False, subset=CHUNK_STAGE_COLUMNS)

    total = 0
    for part_i, fpath in enumerate(files):
        table = pq.read_table(fpath)
        rows = table.to_pylist()
        out_records: List[dict] = []
        for row in rows:
            spdx = row.get("license", "") or ""
            recs = chunk_file(
                text=row.get("content", "") or "",
                language=language,
                repo_name=row.get("repo_name", "") or "",
                path=row.get("path", "") or "",
                license=spdx,
                text_publishable=spdx in allow,
                corpus_snapshot=corpus_snapshot,
                tokenizer=tokenizer,
                max_tokens=cfg.chunk.max_tokens,
                overlap_tokens=cfg.chunk.overlap_tokens,
                min_tokens=cfg.chunk.min_tokens,
                unit=cfg.chunk.unit,
            )
            out_records.extend(r.to_dict() for r in recs)

        if not out_records:
            continue
        out_table = pa.Table.from_pylist(out_records).select(
            [c for c in CHUNK_STAGE_COLUMNS if c in out_records[0]]
        )
        out_table = out_table.cast(schema) if out_table.schema != schema else out_table
        out_path = os.path.join(out_dir, f"chunks-{part_i:05d}.parquet")
        tmp = out_path + ".tmp"
        pq.write_table(out_table, tmp, compression="zstd", row_group_size=20000)
        os.replace(tmp, out_path)
        total += out_table.num_rows
        logger.info("chunked %s -> %s (%s chunks)", os.path.basename(fpath), out_path, human_int(out_table.num_rows))

    logger.info("chunk stage done for %s: %s chunks total", language, human_int(total))
    return 0


def cmd_shard(args: argparse.Namespace) -> int:
    from precal.sharding import build_shards

    overrides = {}
    cfg = _load(args.config, overrides=overrides)
    langs = [args.language] if args.language else None
    build_shards(
        cfg,
        languages=langs,
        target_chunks=args.target_chunks,
        do_pairs=not args.no_pairs,
    )
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    from precal.embed import embed_shard

    cfg = _load(args.config)
    embed_shard(
        cfg,
        shard_id=args.shard_id,
        resume=args.resume,
        gpu_id=args.gpu_id,
        base_port=args.base_port,
        engine_override=args.engine,
    )
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    from precal.index import build_all_indexes, build_shard_index

    cfg = _load(args.config)
    if args.all:
        build_all_indexes(cfg, factory_override=args.factory)
    elif args.shard_id is not None:
        build_shard_index(cfg, args.shard_id, factory_override=args.factory)
    else:
        logger.error("index requires either --all or --shard-id N")
        return 2
    return 0


def cmd_merge_index(args: argparse.Namespace) -> int:
    from precal.index import merge_indexes

    cfg = _load(args.config)
    merge_indexes(cfg, mode=args.mode)
    return 0


def cmd_publish(args: argparse.Namespace) -> int:
    from precal.publish import publish

    cfg = _load(args.config)
    publish(cfg, repo_id=args.repo_id, dry_run=args.dry_run)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from precal.eval import evaluate

    cfg = _load(args.config)
    evaluate(cfg, split=args.split, index_mode=args.index, base_port=args.base_port)
    return 0


# --------------------------------------------------------------------------- #
# Parser construction
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="precal", description="preCal code-embedding pipeline CLI."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_config(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", required=True, help="Path to a YAML config file.")

    # stage
    p_stage = sub.add_parser("stage", help="Login-node staging (corpus+eval+model).")
    add_config(p_stage)
    p_stage.add_argument("--languages", help="Comma-separated language subset.")
    p_stage.add_argument("--skip-model", action="store_true")
    p_stage.add_argument("--skip-corpus", action="store_true")
    p_stage.set_defaults(func=cmd_stage)

    # chunk
    p_chunk = sub.add_parser("chunk", help="CPU tree-sitter chunk stage.")
    add_config(p_chunk)
    p_chunk.add_argument("--language", required=True)
    p_chunk.add_argument("--input-glob", dest="input_glob", default=None)
    p_chunk.add_argument("--out", default=None, help="Output dir for chunk parquet.")
    p_chunk.set_defaults(func=cmd_chunk)

    # shard
    p_shard = sub.add_parser("shard", help="Build deterministic shards + manifest.")
    add_config(p_shard)
    p_shard.add_argument("--language", default=None)
    p_shard.add_argument("--target-chunks", dest="target_chunks", type=int, default=None)
    p_shard.add_argument("--no-pairs", dest="no_pairs", action="store_true")
    p_shard.set_defaults(func=cmd_shard)

    # embed
    p_embed = sub.add_parser("embed", help="Embed one shard (resumable).")
    add_config(p_embed)
    p_embed.add_argument("--shard-id", dest="shard_id", type=int, required=True)
    p_embed.add_argument("--engine", default=None, help="Override engine.name.")
    p_embed.add_argument("--gpu-id", dest="gpu_id", type=int, default=None)
    p_embed.add_argument("--base-port", dest="base_port", type=int, default=7997)
    # --resume / --no-resume (default resume=True)
    p_embed.add_argument("--resume", dest="resume", action="store_true", default=True)
    p_embed.add_argument("--no-resume", dest="resume", action="store_false")
    p_embed.set_defaults(func=cmd_embed)

    # index
    p_index = sub.add_parser("index", help="Build per-shard FAISS index.")
    add_config(p_index)
    g = p_index.add_mutually_exclusive_group()
    g.add_argument("--shard-id", dest="shard_id", type=int, default=None)
    g.add_argument("--all", action="store_true")
    p_index.add_argument("--factory", default=None, help="Override index_factory string.")
    p_index.set_defaults(func=cmd_index)

    # merge-index
    p_merge = sub.add_parser("merge-index", help="Merge per-shard indexes.")
    add_config(p_merge)
    p_merge.add_argument("--mode", choices=["ondisk", "sharded"], default=None)
    p_merge.set_defaults(func=cmd_merge_index)

    # publish
    p_pub = sub.add_parser("publish", help="Assemble layout + upload to HF.")
    add_config(p_pub)
    p_pub.add_argument("--repo-id", dest="repo_id", default=None)
    p_pub.add_argument("--dry-run", dest="dry_run", action="store_true")
    p_pub.set_defaults(func=cmd_publish)

    # eval
    p_eval = sub.add_parser("eval", help="Retrieval eval (recall@k/MRR@10/nDCG@10).")
    add_config(p_eval)
    p_eval.add_argument("--split", choices=["test", "valid"], default="test")
    p_eval.add_argument(
        "--index", dest="index", choices=["exact", "ann", "both"], default="both"
    )
    p_eval.add_argument("--base-port", dest="base_port", type=int, default=7997)
    p_eval.set_defaults(func=cmd_eval)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
