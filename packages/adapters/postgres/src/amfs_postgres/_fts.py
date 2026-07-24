"""Full-text query helpers shared by the sync and async Postgres adapters."""
from __future__ import annotations

import re

# Word tokens only: drop punctuation/underscores so each term is safe to feed
# to plainto_tsquery individually.
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)

# Bound the number of OR terms so a pathologically long query can't explode the
# SQL / param list. 20 covers any realistic natural-language recall query.
_MAX_TERMS = 20


def or_tsquery(query: str) -> tuple[str, list[str]]:
    """Build an OR-combined tsquery expression + its bound params.

    ``plainto_tsquery`` and ``websearch_to_tsquery`` both combine adjacent
    unquoted words with AND (``&``), so a noisy multi-term query such as
    ``"browser plugin extension delivered yesterday"`` only matches documents
    that contain *every* term — keyword recall silently collapses to zero when
    any single term is absent (this is why ``amfs_search`` returned empty).

    We instead OR each term's ``plainto_tsquery`` via the tsquery ``||``
    operator so a document matching ANY term becomes a candidate. ``ts_rank``
    evaluated over the same expression still ranks documents that match more
    (and rarer) terms higher, so precision is preserved at the top of the list
    while recall no longer depends on an exact full-term match.

    Returns ``(sql_expr, params)`` where ``sql_expr`` contains ``%s``
    placeholders and ``params`` are the per-term strings, in order. Single-term
    or empty input falls back to a plain ``plainto_tsquery`` (identical to the
    previous behaviour). The caller extends its param list with ``params`` once
    per occurrence of ``sql_expr`` in the final statement (e.g. WHERE + ORDER).
    """
    # Preserve order, drop duplicates, cap length.
    terms = list(dict.fromkeys(_WORD_RE.findall(query or "")))[:_MAX_TERMS]
    if len(terms) <= 1:
        return "plainto_tsquery('english', %s)", [query or ""]
    expr = "(" + " || ".join(["plainto_tsquery('english', %s)"] * len(terms)) + ")"
    return expr, terms
