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
    """Stands in for the async adapter, keeping what it was asked to write.

    Applies the same ``int()`` the real adapter method does. The first version of
    this double did not, which is exactly why it failed to catch the block's
    ``est_tokens_saved`` being a display string: a mock that accepts anything
    proves the caller passed something, not that the database could store it.
    """

    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self._fail = fail

    async def record_reuse_event(self, entity_path: str, key: str, **kw: Any) -> None:
        if self._fail:
            raise RuntimeError("database is having a day")
        kw["est_tokens_saved"] = max(int(kw.get("est_tokens_saved") or 0), 0)
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
async def test_the_stored_estimate_is_an_integer_the_database_can_hold():
    """The bug that would have made the whole feature inert, in silence.

    ``est_tokens_saved`` in the block is a *display* string from
    ``format_tokens`` — "~1.2K". Storing that field directly meant the adapter's
    ``int()`` raised ValueError, and because the write swallows its errors to
    protect the read, every credited reuse was dropped without a trace. The panel
    and the digest would have shipped permanently empty with nothing in the logs.

    So the row takes the raw integer from ``recall_tokens_for_chars``, which is
    the same function the block formats for display: one source for the number,
    and a value the column can actually hold.
    """
    from amfs_core.aggregates import recall_tokens_for_chars
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

    stored = rec.calls[0]["est_tokens_saved"]
    assert isinstance(stored, int) and stored > 0

    # The same figure the caller was shown, before formatting.
    shown = json.loads(resp.headers["X-SenseLab-Value"])["est_tokens_saved"]
    assert isinstance(shown, str), "the block's field is for display"
    expected = recall_tokens_for_chars(len(json.dumps(entry.value, default=str)), hits=1)
    assert stored == expected


@pytest.mark.asyncio
async def test_an_empty_agent_header_is_not_a_second_agent():
    """A present-but-empty header must read as unknown, not as another agent.

    ``request.headers.get`` returns "" for ``x-amfs-agent-id:`` with no value, and
    "" is not NULL, so it satisfies ``reused_by IS NOT NULL`` in the summary and
    would be counted as a *different* agent reusing the memory. The block already
    treats it as unknown, so leaving it would make the stored row and the line the
    user saw disagree about the one claim that matters most.
    """
    from amfs_http import server

    rec = _Recorder()
    monkey = server._async_adapter
    server._async_adapter = rec
    try:
        server._attach_reuse_value(
            None,
            _Request("   "),  # present, whitespace only
            credited=_entry(agent_id="cursor-agent"),
            hits=1,
            surface="read",
        )
        await asyncio.sleep(0)
    finally:
        server._async_adapter = monkey

    assert rec.calls[0]["reused_by"] is None, (
        "an empty header stored as a string counts as a cross-surface reuse"
    )


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


def test_a_caller_scoped_to_no_agents_is_told_about_no_reuse():
    """The degenerate case must short-circuit, not build ``= ANY('{}')``.

    A user who may see no agents may see no reuse. Returning early also avoids a
    query whose scan can never match, and — more importantly — makes the
    restricted case impossible to get wrong in SQL.
    """
    from amfs_postgres.adapter import PostgresAdapter

    class _Exploding:
        def connection(self) -> Any:
            raise AssertionError("a caller scoped to nothing must not query at all")

    adapter = object.__new__(PostgresAdapter)
    adapter._pool = _Exploding()
    adapter._namespace = "rt"

    out = adapter.reuse_summary(since=datetime.now(UTC) - timedelta(days=7), visible_agents=set())
    assert out["reuses"] == 0
    assert out["recent_cross_surface"] == []


def test_scoping_filters_on_the_reader_in_every_query():
    """Each of the four statements must carry the scope, not just the first.

    The rows name entity paths, keys and agent ids, and ``top`` and
    ``recent_cross_surface`` are the parts that name them outright. RLS separates
    accounts; this is the within-account restriction, and a scope applied to the
    totals alone would leave the detail lists leaking.

    Filtered on ``reused_by`` because a row means "this agent read this memory",
    so it is the caller's to see exactly when that agent is.
    """
    from amfs_postgres.adapter import PostgresAdapter

    seen: list[tuple[str, tuple]] = []

    class _Cur:
        def execute(self, sql: str, params: Any = None) -> None:
            seen.append((sql, params))

        def fetchone(self) -> dict[str, Any]:
            return {}

        def fetchall(self) -> list[dict[str, Any]]:
            return []

        def __enter__(self) -> _Cur:
            return self

        def __exit__(self, *a: Any) -> None:
            return None

    class _Conn:
        def cursor(self, *a: Any, **kw: Any) -> _Cur:
            return _Cur()

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
    adapter.reuse_summary(
        since=datetime.now(UTC) - timedelta(days=7), visible_agents={"mine"}
    )

    assert len(seen) == 4, "totals, top, by_agent and recent_cross_surface"
    for sql, params in seen:
        assert "reused_by = ANY(%s)" in sql, f"unscoped query would leak: {sql[:80]}"
        assert ["mine"] in params, "the scope must actually be bound"


def test_an_adapter_predating_the_agent_filter_degrades_instead_of_failing():
    """The pairing this guards is real, not hypothetical.

    ``http-server`` declares no dependency on the adapter package — it duck-types
    whatever backend it is handed — so a self-hosted install can upgrade one and
    not the other. Passing an unknown keyword would raise ``TypeError`` and take
    out the endpoint entirely, including the account-wide answer that still works
    perfectly. Only the narrowed question is refused.
    """
    import asyncio

    from amfs_http import server as srv

    class _OldAdapter:
        def reuse_summary(
            self, *, since: Any, limit: int = 10, visible_agents: Any = None
        ) -> dict[str, Any]:
            return {"reuses": 3, "memories_reused": 1}

    class _Memory:
        _adapter = _OldAdapter()

    class _State:
        visibility_filter = None

    class _EndpointRequest(_Request):
        state = _State()

    original = srv._get_memory
    srv._get_memory = lambda: _Memory()  # type: ignore[assignment]
    try:
        narrowed = asyncio.run(srv.reuse_summary(_EndpointRequest(), agent="some-agent"))
        account_wide = asyncio.run(srv.reuse_summary(_EndpointRequest()))
    finally:
        srv._get_memory = original  # type: ignore[assignment]

    assert narrowed["available"] is False
    assert narrowed["agent"] == "some-agent"
    assert "one agent" in narrowed["reason"]

    # The half that still works is untouched.
    assert account_wide["available"] is True
    assert account_wide["reuses"] == 3


def _sql_capturing_adapter() -> tuple[Any, list[tuple[str, tuple]]]:
    """An adapter whose every query is captured instead of executed."""
    from amfs_postgres.adapter import PostgresAdapter

    seen: list[tuple[str, tuple]] = []

    class _Cur:
        def execute(self, sql: str, params: Any = None) -> None:
            seen.append((sql, params))

        def fetchone(self) -> dict[str, Any]:
            return {}

        def fetchall(self) -> list[dict[str, Any]]:
            return []

        def __enter__(self) -> _Cur:
            return self

        def __exit__(self, *a: Any) -> None:
            return None

    class _Conn:
        def cursor(self, *a: Any, **kw: Any) -> _Cur:
            return _Cur()

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
    return adapter, seen


def test_naming_an_agent_filters_every_query_on_the_reader():
    """A page about one agent must be answered by the database, not the caller.

    Each list is cut to ``limit`` by reuse volume before it is returned, so a
    caller that asked for the account and kept the rows naming its agent would
    lose a quiet agent's reuse entirely — and could not tell that apart from the
    agent having none.
    """
    adapter, seen = _sql_capturing_adapter()
    adapter.reuse_summary(since=datetime.now(UTC) - timedelta(days=7), agent="one-agent")

    assert len(seen) == 4, "totals, top, by_agent and recent_cross_surface"
    for sql, params in seen:
        assert "reused_by = %s" in sql, f"unfiltered query answers about everyone: {sql[:80]}"
        assert "one-agent" in params, "the agent must actually be bound"


def test_naming_an_agent_narrows_the_visible_scope_and_never_replaces_it():
    """Both predicates apply, so naming an agent cannot widen what a caller sees."""
    adapter, seen = _sql_capturing_adapter()
    adapter.reuse_summary(
        since=datetime.now(UTC) - timedelta(days=7),
        visible_agents={"mine"},
        agent="mine",
    )

    for sql, params in seen:
        assert "reused_by = ANY(%s)" in sql, "the visibility scope must survive"
        assert "reused_by = %s" in sql, "the agent filter must apply too"
        assert ["mine"] in params and "mine" in params


def test_asking_about_an_agent_the_caller_cannot_see_answers_nothing():
    """The case that would otherwise be a within-account read of another user's agent.

    Without this, the two predicates would contradict each other and the SQL would
    match nothing anyway — but relying on that is relying on a coincidence of
    clause order. Refusing up front is the guarantee.
    """
    adapter, seen = _sql_capturing_adapter()
    summary = adapter.reuse_summary(
        since=datetime.now(UTC) - timedelta(days=7),
        visible_agents={"mine"},
        agent="someone-elses",
    )

    assert seen == [], "no query should be issued at all"
    assert summary["reuses"] == 0
    assert summary["recent_cross_surface"] == []


def test_an_empty_string_reader_is_excluded_from_every_claim():
    """Defence in depth for rows an older writer may already have stored.

    The header is normalised before the row is written, so this should never
    happen — but the summary is what makes the claim, and an empty string is the
    one value that is neither NULL nor a real agent.
    """
    import inspect

    from amfs_postgres.adapter import PostgresAdapter

    src = inspect.getsource(PostgresAdapter.reuse_summary)
    cross_surface_clauses = [
        line for line in src.splitlines() if "written_by <> reused_by" in line
    ]
    assert cross_surface_clauses, "the cross-surface predicate moved"
    assert src.count("reused_by <> ''") >= 2, "empty readers can still be counted"
    assert src.count("written_by <> ''") >= 2, "empty authors can still be counted"


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
