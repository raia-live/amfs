"""Getting the account-wide vector search onto the HNSW index.

``semantic_search`` orders by ``embedding <=> query`` under a WHERE clause
(namespace, branch, live rows, confidence, and in hosted deployments the RLS
tenant predicate). Given that filter the planner abandons ``idx_entries_embedding``
and walks a b-tree instead, detoasting every embedding in the namespace to
compute every distance and top-N-sorting the lot: on a 100K-entry account that
is ~850K buffer hits and ~600 ms per query variant, hot (measured on production
2026-09-19; ``/retrieve`` issues one such query per rewriter variant). It avoids
the index for a reason: without an iterative scan the index hands back at most
``hnsw.ef_search`` (default 40) rows *before* the filter runs, so a LIMIT of
150 could come back with 30 rows.

pgvector 0.8 fixes the recall half with ``hnsw.iterative_scan``: the index
scan keeps walking until the LIMIT is satisfied after filtering (bounded by
``hnsw.max_scan_tuples``). The plan-choice half is fixed by taking the sort
off the table for the one statement: with ``enable_sort`` off, the only
unpenalised way to produce ``ORDER BY <=> LIMIT`` is the index, which is the
nudge pgvector's own docs recommend for filtered searches. Same query, same
WHERE, 4 ms instead of 588 ms, 145 of the exact top-150 recovered and the
exact top-20 intact.

The settings are ``SET LOCAL`` inside a transaction so nothing leaks to the
pooled connection. They are applied only when

* the read is account-wide (``entity_path is None``) — a path-scoped read is
  a small b-tree range and stays exact;
* pgvector is at least 0.8, because ``hnsw.iterative_scan`` does not exist
  before that and setting it errors.

An under-filled result (fewer rows than the LIMIT) means the iterative scan
gave up at ``max_scan_tuples`` before the filter let enough rows through —
a small tenant inside a large shared table, or an unusually narrow
confidence window. The caller then re-runs the exact scan, which is what it
did before and is cheap for exactly the tenants that under-fill.
"""

from __future__ import annotations

import re

__all__ = [
    "HNSW_ITERATIVE_MIN_VERSION",
    "hnsw_scan_settings",
    "parse_pgvector_version",
    "supports_iterative_scan",
    "use_hnsw_scan",
]

#: ``hnsw.iterative_scan`` arrived in pgvector 0.8.0.
HNSW_ITERATIVE_MIN_VERSION: tuple[int, ...] = (0, 8, 0)

#: Bounds for ``hnsw.ef_search``. pgvector requires 1..1000; below the LIMIT
#: the first pass of the scan cannot fill the result and the iterative scan
#: has to widen, so the floor tracks the LIMIT.
_EF_SEARCH_MIN = 40
_EF_SEARCH_MAX = 1000

_VERSION_RE = re.compile(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?")


def parse_pgvector_version(text: str | None) -> tuple[int, ...] | None:
    """``'0.8.1'`` -> ``(0, 8, 1)``; ``None`` when the extension is absent or
    the string is not a version."""
    if not text:
        return None
    m = _VERSION_RE.search(str(text))
    if m is None:
        return None
    return tuple(int(part) for part in m.groups() if part is not None)


def supports_iterative_scan(version: tuple[int, ...] | None) -> bool:
    """Whether this pgvector has ``hnsw.iterative_scan``."""
    return version is not None and tuple(version) >= HNSW_ITERATIVE_MIN_VERSION


def use_hnsw_scan(*, entity_path: str | None, version: tuple[int, ...] | None) -> bool:
    """The routing rule: account-wide read on a pgvector that can iterate."""
    return entity_path is None and supports_iterative_scan(version)


def hnsw_scan_settings(limit: int) -> str:
    """The ``SET LOCAL`` statements that put one statement on the HNSW index,
    as a single multi-statement string (one round trip; it carries no
    parameters, so psycopg sends it as-is).

    Literal values only — these are GUCs, not parameters, and psycopg cannot
    bind them. ``limit`` is clamped into pgvector's accepted range.
    """
    ef_search = max(_EF_SEARCH_MIN, min(int(limit), _EF_SEARCH_MAX))
    return "; ".join(
        [
            "SET LOCAL hnsw.iterative_scan = 'relaxed_order'",
            f"SET LOCAL hnsw.ef_search = {ef_search}",
            "SET LOCAL enable_sort = off",
        ]
    )
