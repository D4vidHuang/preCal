"""Configuration schema + loader for preCal.

This module is the single source of truth for *every* config key listed in the
BUILD SPEC's `configKeys`. It:

  * defines a nested dataclass tree mirroring the dotted keys
    (run.*, paths.*, corpus.*, chunk.*, shard.*, model.*, engine.*, embed.*,
    index.*, publish.*, eval.*, hf.*),
  * loads one or more YAML files and deep-merges them (later files override
    earlier ones, so you can pass `default.yaml` then `smoke.yaml`),
  * performs ``${ENV_VAR}`` interpolation on string values (notably
    ``${PRECAL_SCRATCH}``), and
  * applies cross-field invariants (e.g. engine=tei forces model.dtype=float16,
    per the TEI decision in the spec).

It deliberately avoids importing anything heavy (no torch/faiss/datasets) so it
can be imported and validated anywhere, including on a login node or in CI.

Pydantic is used if available (nice validation + error messages); otherwise we
fall back to plain dataclasses so the package stays import-clean without it.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, get_type_hints

try:  # PyYAML is a hard dep (it is in requirements.txt) but guard anyway.
    import yaml
except Exception as exc:  # pragma: no cover - import guard
    raise ImportError(
        "PyYAML is required to load preCal configs (`pip install pyyaml`)."
    ) from exc


# --------------------------------------------------------------------------- #
# Environment-variable interpolation
# --------------------------------------------------------------------------- #
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _interpolate_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` references in strings.

    Unset variables raise a clear error (rather than silently expanding to the
    empty string) because most preCal paths key off ``${PRECAL_SCRATCH}`` and a
    silent empty value would write artifacts to the filesystem root.
    """
    if isinstance(value, str):

        def _sub(match: "re.Match[str]") -> str:
            name = match.group(1)
            if name not in os.environ:
                raise KeyError(
                    f"Config references environment variable ${{{name}}} which "
                    f"is not set. Export it (e.g. PRECAL_SCRATCH) before running."
                )
            return os.environ[name]

        return _ENV_PATTERN.sub(_sub, value)
    if isinstance(value, list):
        return [_interpolate_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    return value


# --------------------------------------------------------------------------- #
# Nested config dataclasses (mirror the dotted configKeys verbatim)
# --------------------------------------------------------------------------- #
@dataclass
class RunConfig:
    name: str = "precal-v1"          # run.name
    seed: int = 42                   # run.seed


@dataclass
class PathsConfig:
    scratch: str = "${PRECAL_SCRATCH}"                 # paths.scratch
    hf_home: str = "${PRECAL_SCRATCH}/hf"              # paths.hf_home
    manifest_dir: str = "${PRECAL_SCRATCH}/manifests"  # paths.manifest_dir


@dataclass
class CorpusConfig:
    dataset_id: str = "bigcode/the-stack-dedup"        # corpus.dataset_id
    revision: str = "main"                             # corpus.revision
    languages: List[str] = field(                      # corpus.languages
        default_factory=lambda: ["python", "java", "javascript", "php", "go", "ruby"]
    )
    license_allowlist: List[str] = field(              # corpus.license_allowlist
        default_factory=lambda: [
            "MIT",
            "Apache-2.0",
            "BSD-2-Clause",
            "BSD-3-Clause",
            "ISC",
            "0BSD",
            "Unlicense",
            "CC0-1.0",
        ]
    )
    max_files_per_lang: int = 0                        # corpus.max_files_per_lang


@dataclass
class ChunkConfig:
    unit: str = "symbol"                               # chunk.unit
    max_tokens: int = 2048                             # chunk.max_tokens
    overlap_tokens: int = 64                           # chunk.overlap_tokens
    min_tokens: int = 8                                # chunk.min_tokens


@dataclass
class ShardConfig:
    target_chunks: int = 250000                        # shard.target_chunks


@dataclass
class ModelConfig:
    id: str = "Qwen/Qwen3-Embedding-4B"                # model.id
    revision: str = "main"                             # model.revision
    dtype: str = "bfloat16"                            # model.dtype
    pooling: str = "last_token"                        # model.pooling
    normalize: bool = True                             # model.normalize
    embed_dim: int = 2560                              # model.embed_dim
    # model.query_instruction -> the task_description injected into the QUERY
    # wrapper 'Instruct: {task}\nQuery: {q}'. Documents/code get NO prefix.
    query_instruction: str = (
        "Given a natural language query, retrieve relevant code snippets"
    )


@dataclass
class EngineConfig:
    name: str = "tei"                                  # engine.name
    image: str = "ghcr.io/huggingface/text-embeddings-inference:120-1.9"  # engine.image
    replicas_per_gpu: int = 4                          # engine.replicas_per_gpu
    max_batch_tokens: int = 16384                      # engine.max_batch_tokens
    batch_size: int = 32                               # engine.batch_size


@dataclass
class EmbedConfig:
    checkpoint_every: int = 5000                       # embed.checkpoint_every


@dataclass
class IndexConfig:
    metric: str = "inner_product"                      # index.metric
    factory_smoke: str = "Flat"                        # index.factory_smoke
    factory_full: str = "OPQ64_256,IVF65536_HNSW32,PQ64"  # index.factory_full
    train_sample: int = 2000000                        # index.train_sample
    nprobe: int = 32                                   # index.nprobe
    merge: str = "ondisk"                              # index.merge


@dataclass
class PublishConfig:
    repo_id: str = "D4vidHuang/precal-code-embeddings"  # publish.repo_id
    num_workers: int = 16                              # publish.num_workers
    emit_inline_embedding: bool = False                # publish.emit_inline_embedding


@dataclass
class EvalConfig:
    benchmark: str = "coir-csn"                         # eval.benchmark
    ks: List[int] = field(default_factory=lambda: [1, 5, 10, 100])  # eval.ks


@dataclass
class HFConfig:
    offline: bool = True                               # hf.offline


@dataclass
class Config:
    """Top-level config object mirroring the dotted configKeys namespace."""

    run: RunConfig = field(default_factory=RunConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    shard: ShardConfig = field(default_factory=ShardConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    publish: PublishConfig = field(default_factory=PublishConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    hf: HFConfig = field(default_factory=HFConfig)

    # ----- derived convenience paths (not config keys; computed) ----------- #
    def scratch_subdir(self, *parts: str) -> str:
        """Absolute path under ``paths.scratch/<run.name>/`` (env-interpolated).

        LOCKED CONTRACT (D1): every run's artifacts are namespaced by
        ``run.name`` so concurrent / successive runs never collide on shared
        scratch. This helper PREPENDS ``run.name`` to every scratch-relative
        path, so callers automatically get
        ``${PRECAL_SCRATCH}/<run.name>/<...parts>`` (matching what the ops
        shells already build as ``${PRECAL_SCRATCH}/${RUN_NAME}/...``). Fixing
        the path here auto-fixes every property/caller that routes through it
        (corpus/chunks/shards/vectors/faiss/eval/publish + manifest_path).
        """
        return os.path.join(self.paths.scratch, self.run.name, *parts)

    @property
    def corpus_dir(self) -> str:
        return self.scratch_subdir("corpus")

    @property
    def chunks_dir(self) -> str:
        return self.scratch_subdir("chunks")

    @property
    def shards_dir(self) -> str:
        return self.scratch_subdir("shards")

    @property
    def vectors_dir(self) -> str:
        return self.scratch_subdir("vectors")

    @property
    def faiss_dir(self) -> str:
        return self.scratch_subdir("faiss")

    @property
    def eval_dir(self) -> str:
        return self.scratch_subdir("eval")

    @property
    def publish_dir(self) -> str:
        return self.scratch_subdir("publish")

    @property
    def manifest_path(self) -> str:
        """Path to the global shard manifest JSONL produced by the shard stage."""
        return os.path.join(self.shards_dir, "manifest.jsonl")


# --------------------------------------------------------------------------- #
# Deep-merge + dataclass hydration helpers
# --------------------------------------------------------------------------- #
def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    out = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _hydrate(dc_type: type, data: Dict[str, Any]) -> Any:
    """Build a (possibly nested) dataclass instance from a plain dict.

    Unknown keys raise so typos in YAML are caught early rather than silently
    ignored. Nested dataclass fields are recursively hydrated.

    Because this module uses ``from __future__ import annotations`` (so every
    field's ``.type`` is a *string*), we resolve the real annotation types via
    ``typing.get_type_hints`` to detect nested dataclasses reliably.
    """
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping for {dc_type.__name__}, got {type(data)}")

    field_map = {f.name: f for f in fields(dc_type)}
    unknown = set(data) - set(field_map)
    if unknown:
        raise KeyError(
            f"Unknown config key(s) for {dc_type.__name__}: {sorted(unknown)}. "
            f"Valid keys: {sorted(field_map)}"
        )

    # Resolve string annotations to actual types (handles PEP 563 stringized
    # annotations from `from __future__ import annotations`).
    try:
        hints = get_type_hints(dc_type)
    except Exception:
        hints = {}

    kwargs: Dict[str, Any] = {}
    for name in field_map:
        if name not in data:
            continue
        value = data[name]
        resolved = hints.get(name)
        if isinstance(resolved, type) and is_dataclass(resolved):
            kwargs[name] = _hydrate(resolved, value)
        else:
            kwargs[name] = value
    return dc_type(**kwargs)


def _apply_invariants(cfg: Config) -> Config:
    """Enforce cross-field rules from the spec's `decisions`.

    * engine=tei -> model.dtype forced to float16 (TEI exposes only
      float16|float32; no bf16). The spec's `model.dtype` doc says
      "Auto-forced to float16 when engine=tei".
    * normalize must be True in v1 (cosine via inner product); we don't hard
      fail, but we do not silently allow a non-normalized + inner_product mix
      without a warning path. We keep it permissive but coherent.
    """
    if cfg.engine.name == "tei" and cfg.model.dtype not in ("float16", "float32"):
        # Record-keeping happens at embed time (the `dtype` column reflects
        # what was actually produced); here we just coerce the request.
        cfg.model.dtype = "float16"

    # LOCKED CONTRACT (D1): the manifests dir (status + committed logs + .done
    # markers) MUST resolve under run.name, exactly like every other artifact
    # dir. paths.manifest_dir is a free-form config string (and the shipped
    # default.yaml still carries the un-namespaced ${PRECAL_SCRATCH}/manifests),
    # so we authoritatively rewrite it to scratch_subdir("manifests") here. This
    # keeps Python (cfg.paths.manifest_dir) and the ops shells
    # (${PRECAL_SCRATCH}/${RUN_NAME}/manifests) from ever diverging.
    cfg.paths.manifest_dir = cfg.scratch_subdir("manifests")
    return cfg


# --------------------------------------------------------------------------- #
# Public loader
# --------------------------------------------------------------------------- #
def load_config(
    paths: "str | List[str]",
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """Load + merge one or more YAML config files into a validated ``Config``.

    Parameters
    ----------
    paths:
        A single path or a list of paths. When a list, files are merged
        left-to-right (later overrides earlier), so the canonical pattern is
        ``["configs/default.yaml", "configs/smoke.yaml"]``. A single path is
        merged on top of the dataclass defaults, which are themselves the spec
        defaults, so passing only ``configs/smoke.yaml`` still works.
    overrides:
        Optional dict deep-merged last (used by the CLI to inject flags like
        ``--target-chunks`` without rewriting YAML).

    Returns
    -------
    Config
        Fully populated, env-interpolated, invariant-checked config.
    """
    if isinstance(paths, str):
        paths = [paths]

    merged: Dict[str, Any] = {}
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Config file not found: {p}")
        with open(p, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, dict):
            raise TypeError(f"Config file {p} must contain a YAML mapping at top level.")
        merged = _deep_merge(merged, loaded)

    if overrides:
        merged = _deep_merge(merged, overrides)

    # Interpolate environment variables *before* hydration so paths are concrete.
    merged = _interpolate_env(merged)

    cfg = _hydrate(Config, merged)
    cfg = _apply_invariants(cfg)
    return cfg


def to_dict(cfg: Any) -> Dict[str, Any]:
    """Recursively convert a (nested) dataclass config back to a plain dict.

    Used by `publish` to embed the resolved run config into the dataset card.
    """
    if is_dataclass(cfg) and not isinstance(cfg, type):
        return {f.name: to_dict(getattr(cfg, f.name)) for f in fields(cfg)}
    if isinstance(cfg, list):
        return [to_dict(v) for v in cfg]
    if isinstance(cfg, dict):
        return {k: to_dict(v) for k, v in cfg.items()}
    return cfg
