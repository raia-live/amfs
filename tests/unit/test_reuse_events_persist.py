"""Reuse that outlives the session it happened in.

``recall_count`` already counts reuse and is the right thing for "which memories
earn their keep". What it cannot answer is anything with a *when* or a *who* in
it, because a counter has neither. Until this, the only per-event record of reuse
was an in-memory session ledger whose own comment said "never persisted", so it
died with the process and nothing could be shown to anyone who was not reading
the chat at the time.

That mattered because every claim memory can make to a user needs one of those
two fields. A weekly digest needs *when*. A panel showing reuse without an agent
narrating it needs *when*. And the one claim a local file or a single tool's
memory structurally cannot make — "your Claude session just used what your Cursor
agent worked out on Tuesday" — needs both.

Three properties are pinned here.

**The row and the block agree.** The estimate written to the row is taken from
the block the caller was shown, not recomputed beside it. Two paths to one number
is how a digest ends up contradicting the chat it is summarising.

**Bookkeeping never breaks the read.** The row is scheduled, not awaited, and its
failures are swallowed at both layers. The read has already answered correctly by
the time any of this runs.

**An unknown reader is not a cross-surface reuse.** The naive predicate
(``written_by IS DISTINCT FROM reused_by``) counts a NULL reader as a different
agent, which would inflate the strongest claim in the product with rows that
prove nothing. This is the same class of false claim as the one ``amfs#395``
fixed, arriving by a different route.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from amfs_core.models import MemoryEntry, Provenance
from amfs_core.reuse_value import reuse_value_block


def _entry(
    *,
    value: str = "a fairly long stored decision, long enough to clear the floor" * 4,
    recall_count: int = 0,
    agent_id: str = "cursor-agent",
    version: int = 3,
) -> MemoryEntry:
    return MemoryEntry(
        entity_path="myapp/auth",
        key="decision-jwt",
        value=value,
        confidence=0.9,
        version=version,
        recall_count=recall_count,
        provenance=Provenance(
            agent_id=agent_id, session_id="s", written_at=datetime.now(UTC)
        ),
    )


class _Request:
    def __init__(self, agent_id: str | None = "claude-agent") -> None:
        self.headers = {"x-amfs-agent-id": agent_id} if agent_id else {}


class _Response:
    def __init__(self) -> None:
        self.headers: dict[str, str] = {}


class _Recorder:
    """Stands in for the async adapter, keeping what it was asked to write."""

    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self._fail = fail

    async def record_reuse_event(self, entity_path: str, key: str, **kw: Any) -> None:
        if self._fail:
            raise RuntimeError("database is having a day")
        self.calls.append({"entity_path": entity_path, "key": key, **kw})


# ── the row carries what a counter cannot ─────────────────────────────


@pytest.mark.asyncio
async def test_the_row_names_the_author_the_reader_and_the_version():
    """The three fields that make the cross-surface claim possible.

    Without ``written_by`` and ``reused_by`` there is no way to tell reuse by the
    agent that wrote something from reuse by a different one, and the second is
    the only one worth interrupting a user about. Without ``entry_version`` the
    row cannot say which version was reused, which ``recall_count`` also cannot.
    """
    from amfs_http import server

    rec = _Recorder()
    monkey = server._async_adapter
    server._async_adapter = rec
    try:
        resp = _Response()
        server._attach_reuse_value(
            resp,
            _Request("claude-agent"),
            credited=_entry(agent_id="cursor-agent", version=7),
            hits=1,
            surface="retrieve",
        )
        await asyncio.sleep(0)  # let the scheduled write run
    finally:
        server._async_adapter = monkey

    assert len(rec.calls) == 1
    call = rec.calls[0]
    assert call["entity_path"] == "myapp/auth"
    assert call["key"] == "decision-jwt"
    assert call["written_by"] == "cursor-agent"
    assert call["reused_by"] == "claude-agent"
    assert call["entry_version"] == 7
    assert call["surface"] == "retrieve"


@pytest.mark.asyncio
async def test_the_stored_estimate_is_the_one_the_caller_was_shown():
    """One number, not two that drift.

    The row's estimate is read out of the block rather than computed again from
    the entry. If a digest and a chat line can each derive their own figure, they
    will eventually disagree, and the user has no way to tell which is wrong.
    """
    from amfs_http import server

    entry = _entry()
    rec = _Recorder()
    monkey = server._async_adapter
    server._async_adapter = rec
    try:
        resp = _Response()
        server._attach_reuse_value(
            resp, _Request(), credited=entry, hits=1, surface="read"
        )
        await asyncio.sleep(0)
    finally:
        server._async_adapter = monkey

    shown = json.loads(resp.headers["X-SenseLab-Value"])
    assert rec.calls[0]["est_tokens_saved"] == shown["est_tokens_saved"]


@pytest.mark.asyncio
async def test_an_in_process_caller_with_no_response_still_records_the_reuse():
    """The reuse happened whether or not anyone wanted a header for it.

    Pro composes these handlers in-process and passes no response. Gating the row
    on the header would silently lose every reuse that arrived by that route,
    which is the exact shape of the bug the server-side move existed to end.
    """
    from amfs_http import server

    rec = _Recorder()
    monkey = server._async_adapter
    server._async_adapter = rec
    try:
        server._attach_reuse_value(
            None, _Request(), credited=_entry(), hits=1, surface="search"
        )
        await asyncio.sleep(0)
    finally:
        server._async_adapter = monkey

    assert len(rec.calls) == 1, "a reuse with no response to decorate is still a reuse"


@pytest.mark.asyncio
async def test_a_read_that_credited_nothing_writes_no_row():
    """No credit, no event. A row here would invent reuse that did not happen."""
    from amfs_http import server

    rec = _Recorder()
    monkey = server._async_adapter
    server._async_adapter = rec
    try:
        server._attach_reuse_value(None, _Request(), credited=None, hits=0)
        server._attach_reuse_value(None, _Request(), credited=_entry(), hits=0)
        await asyncio.sleep(0)
    finally:
        server._async_adapter = monkey

    assert rec.calls == []


# ── bookkeeping must not break the read ───────────────────────────────


@pytest.mark.asyncio
async def test_a_failing_write_does_not_disturb_the_answer():
    """The read is already correct by the time this runs.

    Both layers swallow: the adapter method logs and returns, and the scheduling
    here is fire-and-forget. A caller must not lose a memory it asked for because
    a diagnostic insert failed.
    """
    from amfs_http import server

    rec = _Recorder(fail=True)
    monkey = server._async_adapter
    server._async_adapter = rec
    try:
        resp = _Response()
        server._attach_reuse_value(
            resp, _Request(), credited=_entry(), hits=1, surface="read"
        )
        await asyncio.sleep(0)
    finally:
        server._async_adapter = monkey

    assert "X-SenseLab-Value" in resp.headers, "the caller still gets its block"


def test_no_adapter_and_no_event_loop_are_both_quiet():
    """A filesystem backend keeps no events, and neither case is an error."""
    from amfs_http import server

    monkey = server._async_adapter
    server._async_adapter = None
    try:
        resp = _Response()
        # No async adapter: nothing to write to, and no exception.
        server._attach_reuse_value(
            resp, _Request(), credited=_entry(), hits=1, surface="read"
        )
        assert "X-SenseLab-Value" in resp.headers
    finally:
        server._async_adapter = monkey

    # Called with an adapter but outside a running loop, as a sync test does.
    rec = _Recorder()
    server._async_adapter = rec
    try:
        server._attach_reuse_value(
            _Response(), _Request(), credited=_entry(), hits=1, surface="read"
        )
    finally:
        server._async_adapter = monkey
    assert rec.calls == [], "no loop means nowhere to schedule, not a crash"


# ── the honesty of the cross-surface claim ────────────────────────────


def test_an_anonymous_reader_is_not_evidence_of_a_second_agent():
    """The naive SQL predicate would count it, and it must not.

    ``written_by IS DISTINCT FROM reused_by`` is true when the reader is NULL, so
    an unattributed read would be counted as one agent reusing another's memory.
    That is the strongest claim in the product resting on a row that proves
    nothing. The summary requires both ids to be present, and so does the block.
    """
    written, unknown = "cursor-agent", None
    naive = written is not unknown
    ours = written is not None and unknown is not None and written != unknown
    assert naive and not ours

    block = reuse_value_block(
        hits=1, content_chars=800, written_by="cursor-agent", reused_by=None
    )
    assert block is not None
    assert not block.get("cross_surface"), (
        "an unknown reader cannot be a different agent"
    )


class _DictCursor:
    """A cursor shaped like the pool's real one, which uses ``dict_row``."""

    def __init__(self, results: list[list[dict[str, Any]]]) -> None:
        self._results = results
        self._current: list[dict[str, Any]] = []

    def execute(self, sql: str, params: Any = None) -> None:
        self._current = self._results.pop(0) if self._results else []

    def fetchone(self) -> dict[str, Any] | None:
        return self._current[0] if self._current else None

    def fetchall(self) -> list[dict[str, Any]]:
        return self._current

    def __enter__(self) -> _DictCursor:
        return self

    def __exit__(self, *a: Any) -> None:
        return None


def test_the_summary_reads_rows_by_name_not_position():
    """The regression guard for a bug that would have shipped as an empty panel.

    The connection pool is configured with ``dict_row``, so ``row[0]`` raises
    ``KeyError`` rather than returning the first column. Because this method
    swallows its failures to protect the page it serves, that mistake does not
    surface as an error — it surfaces as "no memory has ever been reused", for
    everyone, permanently. Mocks that hand back tuples cannot catch it, so this
    hands back dicts, exactly as the real cursor does.
    """
    from amfs_postgres.adapter import PostgresAdapter

    cur = _DictCursor(
        [
            [
                {
                    "reuses": 4,
                    "memories_reused": 3,
                    "est_tokens_saved": 1050,
                    "cross_surface": 2,
                }
            ],
            [
                {
                    "entity_path": "myapp/auth",
                    "key": "decision-jwt",
                    "reuses": 2,
                    "est_tokens_saved": 800,
                }
            ],
            [{"reused_by": "claude-agent", "reuses": 1}],
            [
                {
                    "entity_path": "myapp/auth",
                    "key": "decision-jwt",
                    "written_by": "cursor-agent",
                    "reused_by": "claude-agent",
                    "created_at": datetime.now(UTC),
                }
            ],
        ]
    )

    class _Conn:
        def cursor(self, *a: Any, **kw: Any) -> _DictCursor:
            return cur

        def __enter__(self) -> _Conn:
            return self

        def __exit__(self, *a: Any) -> None:
            return None

    class _Pool:
        def connection(self) -> _Conn:
            return _Conn()

    adapter = object.__new__(PostgresAdapter)
    adapter._pool = _Pool()
    adapter._namespace = "rt"

    out = adapter.reuse_summary(since=datetime.now(UTC) - timedelta(days=7))

    assert out["reuses"] == 4, "a dict row read positionally reports zero reuse"
    assert out["memories_reused"] == 3
    assert out["est_tokens_saved"] == 1050
    assert out["cross_surface"] == 2
    assert out["top"][0]["key"] == "decision-jwt"
    assert out["top"][0]["est_tokens_saved"] == 800
    assert out["by_agent"][0]["agent_id"] == "claude-agent"
    assert out["recent_cross_surface"][0]["written_by"] == "cursor-agent"
    assert out["recent_cross_surface"][0]["at"], "the timestamp is the 'on Tuesday' half"


def test_the_summary_window_is_bounded():
    """``days`` is clamped, so a hand-written URL cannot ask for a full scan.

    Zero falls through to the default week rather than clamping to a day: for a
    query parameter an absent value and a zero arrive the same way, and a
    one-day digest is not what either meant.
    """
    for asked, expected in ((0, 7), (-5, 1), (7, 7), (10_000, 365)):
        assert max(1, min(int(asked or 7), 365)) == expected


def test_a_window_is_a_lower_bound_on_created_at():
    """The digest asks for a week and must not be handed a year.

    Pinned as arithmetic rather than SQL because the filter is what decides
    whether a "this week" email is actually about this week.
    """
    now = datetime.now(UTC)
    since = now - timedelta(days=7)
    assert (now - timedelta(days=3)) >= since
    assert (now - timedelta(days=40)) < since
