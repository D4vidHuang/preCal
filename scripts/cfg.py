#!/usr/bin/env python
"""
ops-side READ-ONLY config reader for shell scripts (sbatch / *.sh).

Usage:
    python scripts/cfg.py <config.yaml> <dotted.key> [default]

Examples:
    python scripts/cfg.py configs/full_v1.yaml engine.name           -> tei
    python scripts/cfg.py configs/full_v1.yaml engine.replicas_per_gpu-> 4
    python scripts/cfg.py configs/smoke.yaml   model.id               -> Qwen/Qwen3-Embedding-0.6B

Behaviour:
  * Loads the requested YAML and shallow-merges it OVER configs/default.yaml
    (overrides win), so smoke.yaml / full_v1.yaml only need to carry deltas —
    matching how precal.config layers the base + override files.
  * Interpolates ${ENV} / ${ENV:-fallback} style refs (e.g. ${PRECAL_SCRATCH}).
  * Prints the resolved scalar to stdout (lists -> comma-joined). No side effects.

This intentionally does NOT import precal.* so the ops layer can read config
before the core package is importable (e.g. very early in an sbatch). It is a
convenience for shell glue ONLY; all real work goes through `python -m precal.cli`.
"""
import os
import re
import sys

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.stderr.write("scripts/cfg.py: PyYAML not installed; activate the precal env first.\n")
    sys.exit(2)

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _interp(val):
    if isinstance(val, str):
        def sub(m):
            name, default = m.group(1), m.group(2)
            return os.environ.get(name, default if default is not None else "")
        return _ENV_RE.sub(sub, val)
    if isinstance(val, list):
        return [_interp(v) for v in val]
    if isinstance(val, dict):
        return {k: _interp(v) for k, v in val.items()}
    return val


def _deep_merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def main():
    if len(sys.argv) < 3:
        sys.stderr.write(__doc__)
        sys.exit(2)
    cfg_path, dotted = sys.argv[1], sys.argv[2]
    default = sys.argv[3] if len(sys.argv) > 3 else None

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_path = os.path.join(repo_root, "configs", "default.yaml")

    merged = {}
    if os.path.exists(base_path):
        merged = _load(base_path)
    if os.path.abspath(cfg_path) != os.path.abspath(base_path):
        merged = _deep_merge(merged, _load(cfg_path))

    node = merged
    for part in dotted.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            if default is not None:
                print(default)
                return
            sys.stderr.write(f"scripts/cfg.py: key '{dotted}' not found in {cfg_path}\n")
            sys.exit(1)

    node = _interp(node)
    if isinstance(node, list):
        print(",".join(str(x) for x in node))
    elif isinstance(node, bool):
        print("true" if node else "false")
    else:
        print(node)


if __name__ == "__main__":
    main()
