"""Shared manifest + checkpoint primitives.

Three responsibilities, all filesystem-backed (no DB) so they work on shared
SLURM scratch and survive preemption/requeue:

  1. **Atomic writes** -- ``atomic_write_bytes`` / ``atomic_write_text`` /
     ``atomic_save_npy`` implement the tmp -> fsync -> rename pattern the spec's
     requeueStrategy requires (a renamed file is either the old or new content,
     never a torn write).

  2. **Committed-id log** -- per shard, an append-only newline-delimited log of
     chunk_ids whose vectors are already on disk. The embed loop appends ids
     *after* the corresponding vectors are durably written
     (vectors-before-ids ordering), so on resume we can scan the log and skip
     work without risk of marking an id committed whose vector never landed.

  3. **Shard status state machine** -- pending -> running -> done, one tiny
     status file per shard in ``paths.manifest_dir``, so a controller can skip
     ``done`` shards when resubmitting the whole array.

The shard *manifest* itself (one JSON line per shard) is read/written here too:
the shard stage produces it; embed/index consume it; shard_id maps directly to
SLURM_ARRAY_TASK_ID.
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set

import numpy as np

from precal.utils import ensure_dir, get_logger

logger = get_logger("precal.manifest")

# Shard lifecycle states.
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"


# --------------------------------------------------------------------------- #
# Atomic write primitives
# --------------------------------------------------------------------------- #
def atomic_write_bytes(path: str, data: bytes) -> None:
    """Write bytes atomically: tmp file in same dir -> fsync -> os.replace."""
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    dir_name = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=dir_name, prefix=".tmp-", suffix=".swap")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)  # atomic on POSIX
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def atomic_write_text(path: str, text: str) -> None:
    """Atomic text write (UTF-8)."""
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_save_npy(path: str, array: np.ndarray) -> None:
    """Atomically persist a numpy array as ``.npy`` (tmp -> fsync -> rename)."""
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    dir_name = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=dir_name, prefix=".tmp-", suffix=".npy")
    os.close(fd)
    try:
        # np.save appends .npy if missing; pass allow_pickle=False for safety.
        with open(tmp, "wb") as fh:
            np.save(fh, np.ascontiguousarray(array), allow_pickle=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# Committed-id log (per shard)
# --------------------------------------------------------------------------- #
def committed_log_path(manifest_dir: str, shard_id: int) -> str:
    return os.path.join(manifest_dir, f"shard-{shard_id:05d}.committed")


def status_path(manifest_dir: str, shard_id: int) -> str:
    return os.path.join(manifest_dir, f"shard-{shard_id:05d}.status")


def scan_committed_ids(manifest_dir: str, shard_id: int) -> Set[str]:
    """Read the set of already-committed chunk_ids for a shard.

    Tolerant of a truncated final line (a crash mid-append): the last line is
    dropped if it doesn't end in a newline, because only fully-flushed lines are
    trusted. Returns an empty set if the log doesn't exist yet.
    """
    path = committed_log_path(manifest_dir, shard_id)
    if not os.path.exists(path):
        return set()
    with open(path, "rb") as fh:
        raw = fh.read()
    if not raw:
        return set()
    text = raw.decode("utf-8", errors="replace")
    lines = text.split("\n")
    # If the file doesn't end with a newline, the last token is a partial write.
    if not text.endswith("\n"):
        lines = lines[:-1]
    return {ln for ln in lines if ln}


def append_committed_ids(manifest_dir: str, shard_id: int, chunk_ids: Iterable[str]) -> None:
    """Append newly-committed chunk_ids and fsync.

    MUST be called only after the corresponding vectors are durably on disk
    (vectors-before-ids ordering) so a crash never leaves an id committed
    without its vector.
    """
    ensure_dir(manifest_dir)
    path = committed_log_path(manifest_dir, shard_id)
    payload = "".join(f"{cid}\n" for cid in chunk_ids)
    if not payload:
        return
    # Append mode + fsync. Append of a small buffer is effectively atomic for
    # our purposes; the scan tolerates a torn final line regardless.
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())


# --------------------------------------------------------------------------- #
# Shard status state machine
# --------------------------------------------------------------------------- #
def read_status(manifest_dir: str, shard_id: int) -> str:
    path = status_path(manifest_dir, shard_id)
    if not os.path.exists(path):
        return STATUS_PENDING
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh).get("status", STATUS_PENDING)
    except Exception:
        return STATUS_PENDING


def write_status(manifest_dir: str, shard_id: int, status: str, **extra) -> None:
    """Atomically record a shard's status (+ optional metadata)."""
    ensure_dir(manifest_dir)
    payload = {"shard_id": shard_id, "status": status, **extra}
    atomic_write_text(status_path(manifest_dir, shard_id), json.dumps(payload))


# --------------------------------------------------------------------------- #
# Per-stage .done markers (LOCKED CONTRACT D2)
# --------------------------------------------------------------------------- #
# A tiny empty marker file the ops guards (slurm/embed.sbatch, slurm/index.sbatch,
# scripts/resubmit_pending.sh) test for to skip already-finished work. The path
# uses the RAW shard_id (NO zero-pad) so it matches the shells'
#   f"{MANIFEST_DIR}/{stage}-${SHARD_ID}.done"
# literal exactly. ``stage`` is one of {"embed", "index"}. This is written in
# ADDITION to the richer .status JSON (write_status); the shells only stat the
# cheap marker and never parse JSON.
_DONE_STAGES = ("embed", "index")


def done_marker_path(manifest_dir: str, stage: str, shard_id: int) -> str:
    """Path to the per-stage done marker: f"{manifest_dir}/{stage}-{shard_id}.done"."""
    return os.path.join(manifest_dir, f"{stage}-{shard_id}.done")


def mark_done(manifest_dir: str, stage: str, shard_id: int) -> None:
    """Write the empty f"{manifest_dir}/{stage}-{shard_id}.done" marker.

    stage in {"embed", "index"}; shard_id is the RAW int (no zero-pad) so the
    file name matches the ops-shell guards verbatim. Atomic + fsync'd so a
    requeued task never sees a torn marker. Called by embed.py on embed success
    (alongside write_status(STATUS_DONE)) and by index.py on index success.
    """
    if stage not in _DONE_STAGES:
        raise ValueError(f"mark_done: stage must be one of {_DONE_STAGES}, got {stage!r}")
    ensure_dir(manifest_dir)
    atomic_write_bytes(done_marker_path(manifest_dir, stage, shard_id), b"")


def is_done(manifest_dir: str, stage: str, shard_id: int) -> bool:
    """True iff the f"{manifest_dir}/{stage}-{shard_id}.done" marker exists."""
    return os.path.exists(done_marker_path(manifest_dir, stage, shard_id))


# --------------------------------------------------------------------------- #
# Shard manifest (the global source of truth)
# --------------------------------------------------------------------------- #
@dataclass
class ShardEntry:
    """One line of shards/manifest.jsonl.

    shard_id is GLOBAL and contiguous (0..N-1) across all languages, so it maps
    directly to SLURM_ARRAY_TASK_ID.
    """

    shard_id: int
    language: str
    input_files: List[str] = field(default_factory=list)
    approx_chunks: int = 0
    out_parquet: str = ""
    out_npy: str = ""
    status: str = STATUS_PENDING

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def from_json(cls, line: str) -> "ShardEntry":
        d = json.loads(line)
        return cls(
            shard_id=int(d["shard_id"]),
            language=d.get("language", ""),
            input_files=list(d.get("input_files", [])),
            approx_chunks=int(d.get("approx_chunks", 0)),
            out_parquet=d.get("out_parquet", ""),
            out_npy=d.get("out_npy", ""),
            status=d.get("status", STATUS_PENDING),
        )


def write_manifest(manifest_path: str, entries: List[ShardEntry]) -> None:
    """Atomically write the JSONL shard manifest (sorted by shard_id)."""
    entries = sorted(entries, key=lambda e: e.shard_id)
    body = "\n".join(e.to_json() for e in entries) + "\n"
    atomic_write_text(manifest_path, body)
    logger.info("Wrote manifest with %d shards -> %s", len(entries), manifest_path)


def read_manifest(manifest_path: str) -> List[ShardEntry]:
    """Read the JSONL shard manifest into a list of ShardEntry, sorted by id."""
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Shard manifest not found: {manifest_path}. Run `precal.cli shard` first."
        )
    entries: List[ShardEntry] = []
    with open(manifest_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(ShardEntry.from_json(line))
    return sorted(entries, key=lambda e: e.shard_id)


def get_shard(manifest_path: str, shard_id: int) -> ShardEntry:
    """Look up a single shard entry by its (== SLURM_ARRAY_TASK_ID) id."""
    for entry in read_manifest(manifest_path):
        if entry.shard_id == shard_id:
            return entry
    raise KeyError(f"shard_id {shard_id} not present in manifest {manifest_path}")


def pending_shard_ids(manifest_path: str, manifest_dir: str) -> List[int]:
    """Shard ids whose status != done (what a controller should resubmit)."""
    out: List[int] = []
    for entry in read_manifest(manifest_path):
        if read_status(manifest_dir, entry.shard_id) != STATUS_DONE:
            out.append(entry.shard_id)
    return out
