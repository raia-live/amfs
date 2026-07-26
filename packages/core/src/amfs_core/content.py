"""Content classification for memory entries.

Distinguishes *knowledge* (durable facts, beliefs, experiences written in prose)
from *artifacts* (stored source code, markup, config, and other working files).
The two need different treatment in retrieval: an agent that dumps a 10 KB React
component into memory as a ``fact`` should not have that file outrank a genuine
preference on a query like "what do I prefer".

This module is the single source of truth for that distinction. It is used to:

* set ``MemoryEntry.is_artifact`` at persistence time (adapter write chokepoint),
* choose what text to embed (a clean descriptor for *file-keyed* artifacts, so
  the vector represents what the file is rather than its noisy, truncated
  body), and
* demote artifacts in the retrieve/search/briefing read paths.

The heuristics are intentionally narrow (precision over recall): a genuine
natural-language fact that happens to contain a word like ``return`` — or a path
glob, a quoted tag, or an arrow — must not be misread as code. Prose *about*
code is the common case in an agent's memory, so markers only count at the start
of a line, and the weak ones need corroboration.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Ranking prior applied to artifacts in the read paths. A demoted artifact is
# still returned (multiplicative, not a hard filter), so a genuinely
# code-relevant query can still surface it via high similarity. Shared by the
# OSS blend and mirrored by the Pro MultiStrategyRetriever.
ARTIFACT_PENALTY = 0.5

# Only scan the head of a value: enough to detect code markers cheaply and to
# build a descriptor, without paying to regex a multi-megabyte blob on every
# write or every candidate in a read-time fallback.
_SCAN_LIMIT = 4096

# File extensions that mark an entry key as a stored *artifact* (source code,
# markup, config, docs) rather than a knowledge-fact identifier. Kept in sync
# with amfs_security's poisoning detector (which now imports from here).
_ARTIFACT_EXTENSIONS = {
    "tsx", "ts", "js", "jsx", "mjs", "cjs", "py", "go", "rs", "java", "rb", "php",
    "c", "cc", "cpp", "h", "hpp", "cs", "css", "scss", "less", "html", "htm",
    "vue", "svelte", "json", "yaml", "yml", "toml", "ini", "xml", "md", "mdx",
    "rst", "sql", "sh", "bash", "zsh", "swift", "kt", "kts", "scala", "dart",
    "lua", "r", "pl", "proto", "gradle",
}

# Human-readable language labels for the most common source extensions, used to
# build the embedding descriptor.
_LANGUAGE_BY_EXT = {
    "tsx": "TypeScript React", "ts": "TypeScript", "js": "JavaScript",
    "jsx": "JavaScript React", "mjs": "JavaScript", "cjs": "JavaScript",
    "py": "Python", "go": "Go", "rs": "Rust", "java": "Java", "rb": "Ruby",
    "php": "PHP", "c": "C", "cc": "C++", "cpp": "C++", "h": "C header",
    "hpp": "C++ header", "cs": "C#", "css": "CSS", "scss": "SCSS",
    "less": "LESS", "html": "HTML", "htm": "HTML", "vue": "Vue", "svelte": "Svelte",
    "json": "JSON", "yaml": "YAML", "yml": "YAML", "toml": "TOML", "ini": "INI",
    "xml": "XML", "md": "Markdown", "mdx": "MDX", "rst": "reStructuredText",
    "sql": "SQL", "sh": "shell", "bash": "shell", "zsh": "shell", "swift": "Swift",
    "kt": "Kotlin", "kts": "Kotlin", "scala": "Scala", "dart": "Dart", "lua": "Lua",
    "r": "R", "pl": "Perl", "proto": "Protobuf", "gradle": "Gradle",
}

# Declarations that open a line. One is enough: prose does not begin a line with
# `def foo(` or `export default function`.
_STRONG_CODE_MARKERS = re.compile(
    r"(?m)^\s*(?:import\s|from\s+\S+\s+import\b|export\s+(?:default\s+|const\s+|function\b)|"
    r"def\s+\w+\s*\(|class\s+\w+|function\s+\w*\s*\(|#include\b|package\s+\w|@\w+\s*$)"
)

# Weaker hints, which turn up constantly in prose *about* code: a path glob
# (`packages/*/src`), a quoted tag (`<span>x</span>`), an arrow meaning "leads
# to", a mentioned `console.error(...)`. Matched as bare substrings these filed
# genuine knowledge as code — every flagged row in production was a technical
# note, not a file. So they count only when they open a line, and two separate
# lines must carry one before content alone decides.
_WEAK_CODE_MARKERS = re.compile(
    r"(?m)^\s*(?:/\*|\*/|</?[A-Za-z][\w.-]*(?:\s|/?>)|(?:const|let|var)\s+\w+\s*=|"
    r"console\.\w+\s*\(|require\s*\(|\)\s*=>|[})\]];?\s*$)"
)
_WEAK_MARKERS_REQUIRED = 2

# Top-level symbol declarations, used to summarise an artifact in its descriptor.
_SYMBOL_RE = re.compile(
    r"(?m)^\s*(?:export\s+)?(?:default\s+)?"
    r"(?:async\s+)?(?:def|class|function|const|let|var|interface|type|struct|enum|func)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)"
)


def _stringify(value: Any) -> str:
    """Best-effort string form of an arbitrary memory value."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def _basename(key: str) -> str:
    return key.strip().rsplit("/", 1)[-1]


def _extension(key: str) -> str | None:
    base = _basename(key).lower()
    if "." not in base:
        return None
    return base.rsplit(".", 1)[-1]


def is_artifact_key(key: str) -> bool:
    """True when the entry key looks like a stored file path (e.g.
    ``src/pages/InsightsPage.tsx``) rather than a knowledge-fact identifier."""
    ext = _extension(key)
    return ext is not None and ext in _ARTIFACT_EXTENSIONS


def is_code_like(value: Any) -> bool:
    """True when a value's text looks like source code/markup, not prose."""
    text = _stringify(value).strip()
    if len(text) < 24:
        return False
    head = text[:_SCAN_LIMIT]
    if _STRONG_CODE_MARKERS.search(head):
        return True
    return len(_WEAK_CODE_MARKERS.findall(head)) >= _WEAK_MARKERS_REQUIRED


def classify_artifact(key: str, value: Any) -> bool:
    """An entry is an artifact if its key is a file path or its content is code."""
    return is_artifact_key(key) or is_code_like(value)


def artifact_descriptor(key: str, value: Any) -> str:
    """A clean, embeddable summary of *what an artifact is*.

    bge-small caps at 512 tokens with no chunking, so embedding a large source
    file captures only a noisy, prefix-biased slice of it. Instead we embed a
    short descriptor built from the filename, detected language, and top-level
    symbol names — signal that actually distinguishes one file from another and
    keeps code blobs from spuriously matching generic prose queries.
    """
    base = _basename(key) or key.strip()
    ext = _extension(key)
    language = _LANGUAGE_BY_EXT.get(ext or "", "source")

    text = _stringify(value)[:_SCAN_LIMIT]
    symbols: list[str] = []
    seen: set[str] = set()
    for match in _SYMBOL_RE.finditer(text):
        name = match.group(1)
        if name not in seen:
            seen.add(name)
            symbols.append(name)
        if len(symbols) >= 8:
            break

    parts = [f"{language} file {base}." if base else f"{language} file."]
    if symbols:
        parts.append(f"Defines: {', '.join(symbols)}.")
    return " ".join(parts)


def embedding_input(key: str, value: Any) -> tuple[bool, str]:
    """Return ``(is_artifact, text_to_embed)`` for an entry's key/value.

    Every embed site (SDK write, HTTP write handler, adapter fallback, backfill,
    Pro retriever) routes through this so the classification and the embedded
    text never diverge.

    A descriptor replaces the value only when the key actually names a file. It
    is built from the filename, language and symbols, so under a prose-style key
    it collapses to "source file <key>." and the entry's meaning never reaches
    its vector — the entry stays listed and readable by exact key while becoming
    unfindable by meaning. Content-only classification therefore still demotes,
    but no longer decides what gets embedded.
    """
    if not classify_artifact(key, value):
        return False, _stringify(value)
    if not is_artifact_key(key):
        return True, _stringify(value)
    return True, artifact_descriptor(key, value)
