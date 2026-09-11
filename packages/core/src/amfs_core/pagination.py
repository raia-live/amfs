"""Keyset pagination primitives shared by adapters and the HTTP server.

A cursor names the last row a caller has already seen — its timestamp plus a
tiebreaker that makes the ordering total — so the next page is "everything
strictly older than this", which an index can answer directly. ``OFFSET`` has
to walk and discard every skipped row first, which is why it degrades linearly
with page depth; a keyset page costs the same at row 100 as at row 100,000.

Cursors are opaque to callers: base64url over a small JSON document. Nothing
in the encoding is a secret and nothing about it is stable API — a client that
decodes one and depends on its shape has been warned here.

Also home to the scan ceiling that replaced the hard-coded ``limit=10000``
reads. Aggregates that cannot yet run in SQL still read a bounded number of
rows; the bound is configurable and every caller reports when it was hit.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Generic, TypeVar

T = TypeVar("T")

#: Default ceiling for bounded scans. The historical hard-coded value, kept as
#: the default so nothing changes for deployments that do not set the variable.
DEFAULT_MAX_SCAN_ROWS = 10_000

#: Hard upper bound on any single page, whatever the caller asks for.
MAX_PAGE_SIZE = 1_000


def max_scan_rows() -> int:
    """Rows a bounded scan may read before it stops and reports truncation.

    Read from ``AMFS_MAX_SCAN_ROWS`` on each call, not cached, so tests and
    operators can change it without restarting. Anything unparseable or
    non-positive falls back to the default rather than raising: a typo in an
    environment variable should not take the server down.
    """
    raw = os.environ.get("AMFS_MAX_SCAN_ROWS", "").strip()
    if not raw:
        return DEFAULT_MAX_SCAN_ROWS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_SCAN_ROWS
    return value if value > 0 else DEFAULT_MAX_SCAN_ROWS


def clamp_limit(limit: int | None, *, default: int = 100) -> int:
    """A page size that is at least 1 and at most :data:`MAX_PAGE_SIZE`."""
    if limit is None:
        return default
    return max(1, min(int(limit), MAX_PAGE_SIZE))


class InvalidCursorError(ValueError):
    """The cursor was not produced by :func:`encode_cursor`, or was corrupted."""


def encode_cursor(timestamp: datetime, tiebreak: Any) -> str:
    """Encode a ``(timestamp, tiebreak)`` position as an opaque string.

    *tiebreak* is whatever makes the ordering total for the row set in
    question — a row id for traces and events, a ``[entity_path, key, version]``
    triple for entries. It must be JSON-serialisable and must compare the same
    way in Python as in SQL for the adapter that issued it.
    """
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    payload = json.dumps(
        {"t": timestamp.isoformat(), "k": tiebreak},
        separators=(",", ":"),
        default=str,
    )
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")


def decode_cursor(cursor: str) -> tuple[datetime, Any]:
    """Invert :func:`encode_cursor`.

    Raises :class:`InvalidCursorError` for anything that is not one of ours.
    The HTTP layer turns that into a 400; adapters let it propagate.
    """
    if not isinstance(cursor, str) or not cursor:
        raise InvalidCursorError("empty cursor")
    padded = cursor + "=" * (-len(cursor) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
        doc = json.loads(raw.decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, UnicodeEncodeError, json.JSONDecodeError) as exc:
        raise InvalidCursorError("malformed cursor") from exc
    if not isinstance(doc, dict) or "t" not in doc or "k" not in doc:
        raise InvalidCursorError("malformed cursor")
    try:
        ts = datetime.fromisoformat(doc["t"])
    except (TypeError, ValueError) as exc:
        raise InvalidCursorError("malformed cursor timestamp") from exc
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts, doc["k"]


@dataclass
class Page(Generic[T]):
    """One page of results plus what a caller needs to ask for the next."""

    items: list[T]
    next_cursor: str | None
    has_more: bool
    #: True when a bounded scan stopped before seeing every candidate row, so
    #: the page (or aggregate built from it) may be incomplete.
    truncated: bool = False


def _as_key(ts: datetime, tiebreak: Any) -> tuple[datetime, str]:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts, json.dumps(tiebreak, default=str, sort_keys=True)


def paginate_desc(
    items: Iterable[T],
    *,
    timestamp: Callable[[T], datetime],
    tiebreak: Callable[[T], Any],
    limit: int,
    cursor: str | None = None,
    offset: int = 0,
) -> Page[T]:
    """Keyset-paginate an in-memory iterable, newest first.

    The reference implementation for adapters that cannot push the predicate
    into a query — the filesystem adapter, the ABC defaults, tests. SQL
    adapters do the same thing with ``(created_at, id) < (%s, %s)`` and an
    ``ORDER BY created_at DESC, id DESC``; this is what they must agree with.

    *offset* is honoured only when no cursor is given, for callers still on
    the old ``limit``/``offset`` contract.
    """
    # Not clamped: callers overfetch by one to learn ``has_more``, and the
    # clamp belongs at the HTTP boundary, before that extra row is added.
    limit = max(1, int(limit))
    keyed = [(_as_key(timestamp(x), tiebreak(x)), x) for x in items]
    keyed.sort(key=lambda kv: kv[0], reverse=True)

    if cursor:
        cursor_key = _as_key(*decode_cursor(cursor))
        keyed = [kv for kv in keyed if kv[0] < cursor_key]
    elif offset > 0:
        keyed = keyed[offset:]

    window = keyed[: limit + 1]
    has_more = len(window) > limit
    window = window[:limit]
    next_cursor = None
    if window and has_more:
        last = window[-1][1]
        next_cursor = encode_cursor(timestamp(last), tiebreak(last))
    return Page(items=[x for _, x in window], next_cursor=next_cursor, has_more=has_more)


def page_from_overfetch(
    rows: list[T],
    *,
    limit: int,
    timestamp: Callable[[T], datetime],
    tiebreak: Callable[[T], Any],
) -> Page[T]:
    """Build a page from a query that fetched ``limit + 1`` rows.

    The extra row is how ``has_more`` is known without a second COUNT query;
    it is dropped from the page. The cursor points at the last row *returned*,
    so a caller that filters the page afterwards (visibility, for instance)
    still resumes from the right place.
    """
    has_more = len(rows) > limit
    window = rows[:limit]
    next_cursor = None
    if window and has_more:
        last = window[-1]
        next_cursor = encode_cursor(timestamp(last), tiebreak(last))
    return Page(items=window, next_cursor=next_cursor, has_more=has_more)


def entry_tiebreak(entry: Any) -> list[Any]:
    """Tiebreaker for ``MemoryEntry`` rows, which carry no row id."""
    return [entry.entity_path, entry.key, entry.version]
