"""Code chunking: tree-sitter function/class symbol extraction with fallbacks.

Chosen unit (spec decision `chunk_unit`): tree-sitter function/class/method
symbols via ``tree-sitter-language-pack``. Fallbacks, in order:

  * oversized symbol (n_tokens > chunk.max_tokens)  -> sliding token window
    (chunk.max_tokens with chunk.overlap_tokens), each window a `window` chunk,
    truncated=True.
  * non-parseable file / language with no parser    -> whole-file, windowed if
    it exceeds max_tokens (symbol_kind=whole_file/window).

Degrades gracefully if tree-sitter-language-pack is missing: every file falls
back to the whole-file/window path, so the chunk stage still runs (just at
file granularity) and the pipeline stays import-clean.

This stage is CPU-only and never touches the GPU. It emits chunk *records*
(dicts keyed by the schema CHUNK_STAGE_COLUMNS) which `sharding`/`embed`
consume; it does not write parquet itself (the CLI/sharding layer does).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from precal.utils import (
    chunk_id as make_chunk_id,
    count_tokens,
    get_logger,
    sliding_windows,
)

logger = get_logger("precal.chunking")

# --------------------------------------------------------------------------- #
# tree-sitter availability probe (lazy, cached)
# --------------------------------------------------------------------------- #
_TS_AVAILABLE: Optional[bool] = None


def _tree_sitter_available() -> bool:
    global _TS_AVAILABLE
    if _TS_AVAILABLE is None:
        try:
            import tree_sitter_language_pack  # noqa: F401

            _TS_AVAILABLE = True
        except Exception:
            logger.warning(
                "tree-sitter-language-pack not importable; falling back to "
                "whole-file/window chunking for all files."
            )
            _TS_AVAILABLE = False
    return _TS_AVAILABLE


# Map our lowercase language ids (the v1 slice) to tree-sitter-language-pack
# language names. The pack covers 305 languages; these are the 6 v1 languages.
_TS_LANG_NAME: Dict[str, str] = {
    "python": "python",
    "java": "java",
    "javascript": "javascript",
    "php": "php",
    "go": "go",
    "ruby": "ruby",
}

# Tree-sitter node types we treat as extractable symbols, per language. We map
# each to a coarse symbol_kind (function|method|class). Node type names follow
# the official grammars used by tree-sitter-language-pack.
_SYMBOL_NODE_TYPES: Dict[str, Dict[str, str]] = {
    "python": {
        "function_definition": "function",
        "class_definition": "class",
    },
    "java": {
        "method_declaration": "method",
        "constructor_declaration": "method",
        "class_declaration": "class",
        "interface_declaration": "class",
        "enum_declaration": "class",
    },
    "javascript": {
        "function_declaration": "function",
        "function": "function",
        "method_definition": "method",
        "class_declaration": "class",
        "generator_function_declaration": "function",
    },
    "php": {
        "function_definition": "function",
        "method_declaration": "method",
        "class_declaration": "class",
        "interface_declaration": "class",
        "trait_declaration": "class",
    },
    "go": {
        "function_declaration": "function",
        "method_declaration": "method",
        "type_declaration": "class",  # structs/interfaces via type_spec
    },
    "ruby": {
        "method": "method",
        "singleton_method": "method",
        "class": "class",
        "module": "class",
    },
}

# Node types whose child node carries the symbol's name (field name "name").
# We read the identifier text where available; otherwise symbol_name="".
_NAME_FIELD = "name"


# --------------------------------------------------------------------------- #
# Leading docstring / header-comment stripping (D6)
# --------------------------------------------------------------------------- #
# The eval query for a chunk is its OWN docstring (extracted by pairs.py). If we
# embed the document body with that docstring still inline, the query is a
# verbatim substring of its positive and retrieval is trivially leaky. So for
# the DOCUMENT body that gets EMBEDDED we strip the LEADING docstring / header
# comment. The FULL original text is always kept in the parquet `text` column;
# only the embedded body is stripped (see precal/eval.py).
#
# Language families:
#   * python              -> a leading triple-quoted string at the top of the
#     function/class body (after the `def/class ...:` header line).
#   * c-like (java/js/php/go) -> a leading `/* ... */` block comment, or a run
#     of leading `//` line comments, possibly preceded by the symbol header.
#   * ruby (and any '#'-comment language) -> a leading run of `#` line comments.
# Stripping is conservative: if no leading doc is found we return the text
# unchanged, so worst case we just don't strip (never corrupt the body).

# Python: a `def`/`class` header line followed by a triple-quoted docstring as
# the FIRST statement of the body. We keep the header and drop the docstring.
_PY_HEADER_DOCSTRING = re.compile(
    r"^(?P<head>\s*(?:async\s+)?(?:def|class)\b[^\n]*:\s*\n)"
    r"(?P<indent>[ \t]+)(?:[rRuUbB]{1,2})?(?P<quote>\"\"\"|''')",
    re.DOTALL,
)
# Module-level leading triple-quoted docstring (whole_file chunks etc.).
_PY_MODULE_DOCSTRING = re.compile(
    r"^(?P<lead>\s*)(?:[rRuUbB]{1,2})?(?P<quote>\"\"\"|''')",
    re.DOTALL,
)

# c-like leading block comment: optional non-comment prefix lines are NOT
# skipped; we only strip a comment that begins the (whitespace-stripped) text.
_BLOCK_COMMENT = re.compile(r"^\s*/\*.*?\*/[ \t]*\n?", re.DOTALL)

_LANG_LINE_COMMENT: Dict[str, str] = {
    "java": "//",
    "javascript": "//",
    "php": "//",
    "go": "//",
    "ruby": "#",
}


def _strip_leading_line_comments(text: str, marker: str) -> str:
    """Drop a leading run of single-line comments beginning with ``marker``.

    Only comments at the very top of ``text`` (ignoring blank lines before the
    first comment) are removed; the first non-comment, non-blank line ends the
    run. Returns ``text`` unchanged if it does not start with a comment.
    """
    lines = text.split("\n")
    i = 0
    saw_comment = False
    # Allow leading blank lines before the comment block.
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped == "":
            i += 1
            continue
        if stripped.startswith(marker):
            saw_comment = True
            i += 1
            continue
        break
    if not saw_comment:
        return text
    return "\n".join(lines[i:])


def strip_leading_doc(text: str, language: str) -> str:
    """Return ``text`` with its LEADING docstring / header comment removed.

    Language-aware (D6). Used ONLY for the document body that gets embedded so
    the NL query (the docstring itself) is not a verbatim substring of its
    positive. The full original text is preserved elsewhere (parquet `text`).

    * python: drops a function/class-body triple-quoted docstring (keeps the
      `def`/`class` header), else a module-level leading triple-quoted string.
    * java/javascript/php/go: drops a leading ``/* ... */`` block (e.g. a
      JSDoc/Javadoc ``/** ... */``) or a leading run of ``//`` line comments.
    * ruby: drops a leading run of ``#`` line comments.

    Conservative: if no recognizable leading doc is found, ``text`` is returned
    unchanged.
    """
    if not text:
        return text

    if language == "python":
        m = _PY_HEADER_DOCSTRING.search(text)
        if m:
            quote = m.group("quote")
            body_start = m.end("quote")
            close = text.find(quote, body_start)
            if close != -1:
                return text[: m.end("head")] + text[close + len(quote):]
            return text  # unterminated -> leave untouched
        m = _PY_MODULE_DOCSTRING.search(text)
        if m:
            quote = m.group("quote")
            body_start = m.end("quote")
            close = text.find(quote, body_start)
            if close != -1:
                return text[: m.start("quote")] + text[close + len(quote):]
        return text

    # c-like + ruby: try a leading block comment first, then line comments.
    if language in ("java", "javascript", "php", "go"):
        m = _BLOCK_COMMENT.match(text)
        if m:
            return text[m.end():]
    marker = _LANG_LINE_COMMENT.get(language)
    if marker:
        return _strip_leading_line_comments(text, marker)
    return text


@dataclass
class ChunkRecord:
    """A single chunk before embedding. Field names match schema columns."""

    chunk_id: str
    repo_name: str
    path: str
    language: str
    license: str
    text_publishable: bool
    symbol_kind: str
    symbol_name: str
    start_line: int
    end_line: int
    n_tokens: int
    truncated: bool
    text: str
    # query_* / eval_split / corpus_snapshot are filled by pairs.py / sharding.
    query_text: str = ""
    query_source: str = "none"
    eval_split: str = "index_only"
    corpus_snapshot: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "repo_name": self.repo_name,
            "path": self.path,
            "language": self.language,
            "license": self.license,
            "text_publishable": self.text_publishable,
            "symbol_kind": self.symbol_kind,
            "symbol_name": self.symbol_name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "n_tokens": self.n_tokens,
            "truncated": self.truncated,
            "text": self.text,
            "query_text": self.query_text,
            "query_source": self.query_source,
            "eval_split": self.eval_split,
            "corpus_snapshot": self.corpus_snapshot,
        }


_PARSER_CACHE: Dict[str, object] = {}


def _get_parser(language: str):
    """Return a cached tree-sitter parser for ``language`` or None if unavailable."""
    if not _tree_sitter_available():
        return None
    ts_name = _TS_LANG_NAME.get(language)
    if ts_name is None:
        return None
    if ts_name in _PARSER_CACHE:
        return _PARSER_CACHE[ts_name]
    try:
        from tree_sitter_language_pack import get_parser

        parser = get_parser(ts_name)
        _PARSER_CACHE[ts_name] = parser
        return parser
    except Exception as exc:  # pragma: no cover - depends on installed grammars
        logger.warning("Could not load tree-sitter parser for %s: %s", language, exc)
        _PARSER_CACHE[ts_name] = None
        return None


def _node_name(node, source_bytes: bytes) -> str:
    """Best-effort extraction of a symbol's name node text."""
    try:
        name_node = node.child_by_field_name(_NAME_FIELD)
        if name_node is not None:
            return source_bytes[name_node.start_byte:name_node.end_byte].decode(
                "utf-8", errors="replace"
            )
    except Exception:
        pass
    # Fallback: scan immediate children for an identifier-ish node.
    try:
        for child in node.children:
            if "identifier" in child.type or child.type == "name":
                return source_bytes[child.start_byte:child.end_byte].decode(
                    "utf-8", errors="replace"
                )
    except Exception:
        pass
    return ""


def _walk_symbols(root, source_bytes: bytes, kinds: Dict[str, str]):
    """Iteratively walk the tree and yield (node, symbol_kind, symbol_name).

    We do NOT recurse into a matched symbol's body to avoid emitting both a
    class and its methods as overlapping chunks for the same span twice; instead
    we collect *top-level* matching symbols. We still descend through
    non-matching containers (modules, namespaces) so nested classes/functions at
    file scope are captured. Methods inside classes ARE emitted because we treat
    class bodies as containers worth descending into.
    """
    results: List[Tuple[object, str, str]] = []
    # Stack of nodes to visit.
    stack = [root]
    while stack:
        node = stack.pop()
        for child in node.children:
            kind = kinds.get(child.type)
            if kind is not None:
                results.append((child, kind, _node_name(child, source_bytes)))
                # Descend into class/module bodies to also capture their methods,
                # but do not re-capture the container itself.
                if kind == "class":
                    stack.append(child)
            else:
                stack.append(child)
    return results


def chunk_file(
    *,
    text: str,
    language: str,
    repo_name: str,
    path: str,
    license: str,
    text_publishable: bool,
    corpus_snapshot: str,
    tokenizer,
    max_tokens: int,
    overlap_tokens: int,
    min_tokens: int,
    unit: str = "symbol",
) -> List[ChunkRecord]:
    """Chunk a single source file into ChunkRecords.

    Strategy:
      * unit="symbol" and a parser is available -> extract function/class/method
        symbols; oversized symbols are windowed; if no symbols are found, fall
        back to whole-file/window.
      * unit="window" OR no parser OR parse failure -> whole-file/window.

    ``tokenizer`` is used for token counts and windowing boundaries.
    """
    records: List[ChunkRecord] = []
    lines = text.split("\n")

    def _emit_window(
        span_text: str,
        kind: str,
        name: str,
        base_start_line: int,
    ) -> None:
        """Emit one-or-more chunks for ``span_text`` honoring max_tokens.

        If the span fits within max_tokens it becomes a single chunk; otherwise
        it is split into sliding token windows (truncated=True on each piece).
        ``base_start_line`` is the 1-based start line of the span within the file.
        """
        # FAST PATH: char-based token BOUNDS skip the slow per-symbol tokenizer
        # call for the common (small) symbol — the chunk-throughput bottleneck at
        # scale. For code ~2-5 chars/token, so lo=len/5 lower-bounds and hi=len/2
        # upper-bounds the true token count. hi<=max_tokens => it GENUINELY fits
        # (never under-windows); only straddling cases call the exact tokenizer.
        _clen = len(span_text)
        _lo, _hi = _clen // 5, max(1, _clen // 2)
        if _hi < min_tokens:
            return  # genuinely below the noise floor
        if _hi <= max_tokens and _lo >= min_tokens:
            n_tok = max(min_tokens, _clen // 3)          # approx; definitely one chunk
        else:
            n_tok = count_tokens(tokenizer, span_text)   # exact: near boundary / oversized
            if n_tok < min_tokens:
                return  # noise filter
        if n_tok <= max_tokens:
            end_line = base_start_line + span_text.count("\n")
            cid = make_chunk_id(span_text, repo_name, path, base_start_line, end_line)
            records.append(
                ChunkRecord(
                    chunk_id=cid,
                    repo_name=repo_name,
                    path=path,
                    language=language,
                    license=license,
                    text_publishable=text_publishable,
                    symbol_kind=kind,
                    symbol_name=name,
                    start_line=base_start_line,
                    end_line=end_line,
                    n_tokens=n_tok,
                    truncated=False,
                    text=span_text,
                    corpus_snapshot=corpus_snapshot,
                )
            )
            return
        # Oversized -> sliding window over tokens.
        token_ids = tokenizer.encode(span_text, add_special_tokens=False)
        for win in sliding_windows(token_ids, max_tokens, overlap_tokens):
            piece_ids = token_ids[win.start:win.stop]
            piece_text = tokenizer.decode(piece_ids)
            if count_tokens(tokenizer, piece_text) < min_tokens:
                continue
            # Window line spans are approximate (token<->line is not exact); we
            # record the symbol's span and mark truncated so consumers know.
            cid = make_chunk_id(
                piece_text, repo_name, path, base_start_line, base_start_line
            )
            records.append(
                ChunkRecord(
                    chunk_id=cid,
                    repo_name=repo_name,
                    path=path,
                    language=language,
                    license=license,
                    text_publishable=text_publishable,
                    symbol_kind="window",
                    symbol_name=name,
                    start_line=base_start_line,
                    end_line=base_start_line + span_text.count("\n"),
                    n_tokens=len(piece_ids),
                    truncated=True,
                    text=piece_text,
                    corpus_snapshot=corpus_snapshot,
                )
            )

    parser = _get_parser(language) if unit == "symbol" else None

    if parser is not None:
        try:
            source_bytes = text.encode("utf-8", errors="replace")
            tree = parser.parse(source_bytes)
            kinds = _SYMBOL_NODE_TYPES.get(language, {})
            symbols = _walk_symbols(tree.root_node, source_bytes, kinds)
            if symbols:
                for node, kind, name in symbols:
                    span_text = source_bytes[node.start_byte:node.end_byte].decode(
                        "utf-8", errors="replace"
                    )
                    start_line = node.start_point[0] + 1  # tree-sitter is 0-based
                    _emit_window(span_text, kind, name, start_line)
                return records
            # No symbols found (e.g. a script with only top-level statements):
            # fall through to whole-file handling below.
        except Exception as exc:
            logger.debug("Parse failed for %s (%s): %s; whole-file fallback", path, language, exc)

    # Whole-file / window fallback.
    _emit_window(text, "whole_file", "", 1)
    return records
