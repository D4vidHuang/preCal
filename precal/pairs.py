"""Build NL query <-> code pairs and assign the frozen retrieval split.

Two jobs, both run in the CPU `shard` stage (after chunking, before sharding):

  1. **query_text / query_source** -- attach a natural-language query to each
     code chunk so the artifact is dual-use (RAG/retrieval), without a separate
     join at consume time. Priority of sources:
       docstring (extracted from the chunk itself)  -> query_source="docstring"
       CodeSearchNet func_documentation_string      -> query_source="codesearchnet"
       commit subject                               -> query_source="commit"
       (CoIR queries map in at eval time)           -> query_source="coir"
     If none is found, query_text="" and query_source="none".

  2. **eval_split** -- index_only | eval_test | eval_valid. Assigned by a
     DETERMINISTIC REPO-LEVEL HASH partition (leakage-safe). This is what v1
     ACTUALLY uses (D7): we hash each repo (salted by run.seed) into a bucket
     and carve fixed held-out fractions for eval_valid / eval_test. Because the
     unit is the repo (every chunk of a repo lands in exactly one split), no
     eval positive can leak into the index_only pool.

     IMPORTANT (honesty): a real CoIR ``eval_repos.txt`` is NEVER produced in
     v1, so v1 does NOT use CoIR test qrels to drive the split. The CoIR qrels
     are merely STAGED (read by load_eval_repos if present) for the FUTURE real
     mteb CoIR-Retrieval / CodeSearchNet path; they do not drive the v1 split.
     If an eval-repo list happens to be staged, those repos are additionally
     pinned to eval_test, but the leakage-safe hash partition is the mechanism.

This module is import-clean: it uses only stdlib + the docstring regexes; the
CoIR qrels are read from staged JSON/parquet on scratch if present, else (the
v1 default) the repo-level split is a deterministic hash-based held-out
fraction.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Dict, Iterable, List, Optional, Set

from precal.utils import get_logger

logger = get_logger("precal.pairs")

SPLIT_INDEX_ONLY = "index_only"
SPLIT_TEST = "eval_test"
SPLIT_VALID = "eval_valid"

# --------------------------------------------------------------------------- #
# Docstring extraction (best-effort, per language family)
# --------------------------------------------------------------------------- #
# Python: leading triple-quoted string inside a def/class body.
_PY_DOCSTRING = re.compile(
    r'(?:def|class)\s+\w+\s*\([^)]*\)\s*(?:->[^\:]+)?:\s*\n\s*(?:r|u|b)?("""|\'\'\')(.*?)(\1)',
    re.DOTALL,
)
# JS/Java/PHP/Go: a leading /** ... */ JSDoc/Javadoc block.
_BLOCK_DOC = re.compile(r"/\*\*(.*?)\*/", re.DOTALL)
# Go/Ruby line comments immediately above are harder to attach reliably; we use
# the block form where present and otherwise leave it to CSN/commit sources.


def _clean_doc(raw: str) -> str:
    """Normalize an extracted docstring/comment into a single-line-ish query.

    Strips comment markers, collapses whitespace, takes the first sentence/line
    as the query (docstrings often start with a one-line summary).
    """
    text = raw.strip()
    # Strip JSDoc/Javadoc leading-asterisk gutters.
    text = re.sub(r"^\s*\*\s?", "", text, flags=re.MULTILINE)
    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    # First sentence (up to '. ') or the whole thing if short.
    first = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]
    return first.strip()


def extract_docstring(text: str, language: str) -> str:
    """Return a best-effort NL summary extracted from the chunk's own doc text."""
    if language == "python":
        m = _PY_DOCSTRING.search(text)
        if m:
            return _clean_doc(m.group(2))
    m = _BLOCK_DOC.search(text)
    if m:
        return _clean_doc(m.group(1))
    return ""


def assign_query(
    *,
    text: str,
    language: str,
    csn_doc: Optional[str] = None,
    commit_subject: Optional[str] = None,
) -> "tuple[str, str]":
    """Pick the best available NL query for a chunk and its provenance tag.

    Returns (query_text, query_source). CSN docs (when joined upstream) are the
    cleanest, then in-chunk docstrings, then commit subjects.
    """
    if csn_doc:
        q = _clean_doc(csn_doc)
        if q:
            return q, "codesearchnet"
    doc = extract_docstring(text, language)
    if doc:
        return doc, "docstring"
    if commit_subject:
        q = _clean_doc(commit_subject)
        if q:
            return q, "commit"
    return "", "none"


# --------------------------------------------------------------------------- #
# Repo-level eval-split partition
# --------------------------------------------------------------------------- #
def _stable_bucket(repo_name: str, seed: int, n_buckets: int = 1000) -> int:
    """Deterministic [0, n_buckets) bucket for a repo, salted by seed."""
    h = hashlib.blake2b(f"{seed}:{repo_name}".encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "big") % n_buckets


def load_eval_repos(coir_qrels_dir: str, language: str) -> Set[str]:
    """Load any STAGED CoIR-CSN eval-repo list (FUTURE real-mteb path only).

    The staging stage may write CoIR-CSN corpus/queries/qrels to
    ``$PRECAL_SCRATCH/<run>/corpus/coir-csn/<language>/``. This helper looks for
    an optional precomputed ``eval_repos.txt`` there. In v1 this list is NEVER
    produced (see module docstring / D7), so this returns an EMPTY set and the
    deterministic repo-level hash partition (assign_eval_split) is what actually
    decides the split. The staged qrels are kept for the future real mteb
    CoIR-Retrieval path, not to drive the v1 split.
    """
    repos: Set[str] = set()
    base = os.path.join(coir_qrels_dir, language)
    candidates = [
        os.path.join(base, "eval_repos.txt"),  # optional precomputed list
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    r = line.strip()
                    if r:
                        repos.add(r)
            logger.info("Loaded %d eval repos for %s from %s", len(repos), language, path)
            return repos
    logger.warning(
        "No staged CoIR eval-repo list for %s under %s; falling back to "
        "deterministic hash-based held-out partition.",
        language,
        base,
    )
    return repos


def assign_eval_split(
    repo_name: str,
    *,
    eval_repos: Set[str],
    seed: int,
    valid_fraction: float = 0.05,
    test_fraction: float = 0.05,
) -> str:
    """Assign a chunk's eval_split by REPO via a deterministic hash (leakage-safe).

    v1 mechanism (D7): the assignment is driven by ``_stable_bucket`` — a
    deterministic blake2b hash of the repo name salted by ``seed``. Because the
    unit is the repo, every chunk of a repo lands in the same split and no eval
    positive leaks into index_only. ``eval_repos`` is normally EMPTY in v1 (no
    real CoIR ``eval_repos.txt`` is produced), so the hash partition alone
    decides the split:

    * bucket in the first ``valid_fraction`` of buckets -> eval_valid.
    * bucket in the next ``test_fraction`` of buckets   -> eval_test (so the
      internal docstring->code harness has held-out queries to score).
    * remainder                                          -> index_only.

    If a CoIR eval-repo list IS staged (future real-mteb path), repos in it are
    additionally pinned to eval_test, but that list does NOT drive v1.
    """
    if repo_name in eval_repos:
        return SPLIT_TEST
    bucket = _stable_bucket(repo_name, seed)
    valid_cutoff = int(valid_fraction * 1000)
    if bucket < valid_cutoff:
        return SPLIT_VALID
    # When no real CoIR list is available, also synthesize a held-out test slice
    # so the eval harness has something to score against in smoke runs.
    if not eval_repos:
        test_cutoff = valid_cutoff + int(test_fraction * 1000)
        if valid_cutoff <= bucket < test_cutoff:
            return SPLIT_TEST
    return SPLIT_INDEX_ONLY
