"""What a write hands back about the scope it just wrote into.

An agent writes far more than it reads. The cheapest thing that changes that is
telling it, at the moment it writes, that "six entries are already here and four
have never been read back" — a fact about its own store, with no instruction
attached, delivered as data in a tool result so it reaches every client rather
than only the ones that support hooks.

The reason this lives on the server is cost. The gateway used to ask separately,
which is a second GET /api/v1/entries and so a third billed op on every write —
enough that the feature shipped switched off and stayed off on all four services.
Computed inside the write request it is one aggregate in a request that was
already happening, so nothing new is metered and the block can simply be on.
"""

from __future__ import annotations

import asyncio
import types
from datetime import UTC, datetime

import pytest
from amfs_core.abc import AdapterABC
from amfs_core.models import MemoryEntry, Provenance


def _entry(
    key: str,
    *,
    recall_count: int = 0,
    entity_path: str = "repo/topic",
    agent_id: str = "a",
    shared: bool = True,
) -> MemoryEntry:
    return MemoryEntry(
        entity_path=entity_path,
        key=key,
        version=1,
        value=f"value of {key}",
        provenance=Provenance(
            agent_id=agent_id, session_id="s", written_at=datetime.now(UTC)
        ),
        confidence=1.0,
        recall_count=recall_count,
        shared=shared,
    )


class _ListOnlyAdapter(AdapterABC):
    """An adapter with no query language, exercising the Python form."""

    def __init__(self, entries: list[MemoryEntry]) -> None:
        self._entries = entries

    def list(self, entity_path=None, *, include_superseded=False, branch="main"):
        if entity_path is None:
            return list(self._entries)
        return [e for e in self._entries if e.entity_path == entity_path]

    # The rest of the ABC is unused here.
    def read(self, *a, **k): return None
    def write(self, entry): return entry
    def search(self, query, **k): return []
    def history(self, *a, **k): return []
    def watch(self, *a, **k): return None
    def commit_outcome(self, *a, **k): return None


class TestScopeCounts:
    def test_counts_neighbours_and_excludes_the_entry_just_written(self):
        a = _ListOnlyAdapter([
            _entry("just-written"), _entry("sibling-a"), _entry("sibling-b"),
        ])
        got = a.scope_counts("repo/topic", exclude_key="just-written")
        assert got["existing_entries"] == 2, "the write is not its own neighbour"
        assert "just-written" not in got["keys"]

    def test_never_read_counts_entries_at_recall_count_zero(self):
        a = _ListOnlyAdapter([
            _entry("read-once", recall_count=1),
            _entry("never-a"), _entry("never-b"), _entry("never-c"),
        ])
        got = a.scope_counts("repo/topic")
        assert got["existing_entries"] == 4
        assert got["never_read"] == 3, (
            "a stored recall_count of 0 is the whole signal: written, never read"
        )

    def test_key_sample_is_bounded(self):
        """A scope with hundreds of entries must not dominate the reply."""
        a = _ListOnlyAdapter([_entry(f"k{i:03d}") for i in range(300)])
        got = a.scope_counts("repo/topic", key_limit=8)
        assert got["existing_entries"] == 300
        assert len(got["keys"]) == 8

    def test_a_scope_holding_only_this_write_reports_nothing_to_report(self):
        a = _ListOnlyAdapter([_entry("only-me")])
        got = a.scope_counts("repo/topic", exclude_key="only-me")
        assert got["existing_entries"] == 0
        assert got["never_read"] == 0
        assert got["keys"] == []

    def test_other_entity_paths_are_not_counted(self):
        a = _ListOnlyAdapter([
            _entry("here"), _entry("elsewhere", entity_path="repo/other"),
        ])
        assert a.scope_counts("repo/topic")["existing_entries"] == 1


class TestAnotherAgentsPrivateEntriesStayPrivate:
    """The rule ``AgentMemory.list`` applies, which an aggregate has to repeat.

    ``list`` returns an entry only if it is shared or the caller wrote it, so the
    write tool's own ``list``-based scope sample was protected. Counting in SQL
    skips ``list`` and therefore skips its filter, which would put another
    agent's private key into a tool result — a leak, and one nothing else in the
    block's design would catch, since the per-user visibility filter is inactive
    on most deployments and only ever narrowed the sample, never the counts.
    """

    def test_a_private_entry_of_another_agent_is_neither_named_nor_counted(self):
        a = _ListOnlyAdapter([
            _entry("mine-shared"),
            _entry("theirs-private", agent_id="other", shared=False),
        ])
        got = a.scope_counts("repo/topic", agent_id="a")
        assert got["keys"] == ["mine-shared"]
        assert got["existing_entries"] == 1

    def test_my_own_private_entry_is_still_mine_to_see(self):
        a = _ListOnlyAdapter([
            _entry("mine-private", agent_id="a", shared=False),
            _entry("theirs-private", agent_id="other", shared=False),
        ])
        got = a.scope_counts("repo/topic", agent_id="a")
        assert got["keys"] == ["mine-private"]
        assert got["existing_entries"] == 1

    def test_a_shared_entry_of_another_agent_is_fine(self):
        a = _ListOnlyAdapter([_entry("theirs-shared", agent_id="other")])
        assert a.scope_counts("repo/topic", agent_id="a")["keys"] == ["theirs-shared"]

    def test_naming_no_agent_counts_only_shared(self):
        """The safe direction: an under-count is a quieter footnote, an
        over-count discloses. A caller that cannot say who it is gets the
        shared view."""
        a = _ListOnlyAdapter([
            _entry("shared-one"),
            _entry("private-one", agent_id="a", shared=False),
        ])
        got = a.scope_counts("repo/topic")
        assert got["keys"] == ["shared-one"]
        assert got["existing_entries"] == 1

    def test_never_read_counts_only_what_may_be_counted(self):
        """The leak was in the numbers too, not only the key sample."""
        a = _ListOnlyAdapter([
            _entry("shared-unread", recall_count=0),
            _entry("theirs-private-unread", agent_id="other", shared=False),
        ])
        got = a.scope_counts("repo/topic", agent_id="a")
        assert got["never_read"] == 1


class _DictCursor:
    """A cursor that returns mappings, which is what ``dict_row`` produces."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


class _DictConn:
    def __init__(self, responses: list[list[dict]]) -> None:
        self._responses = list(responses)
        self.queries: list[tuple] = []

    def execute(self, sql, params=None):
        self.queries.append((sql, params))
        return _DictCursor(self._responses.pop(0) if self._responses else [])

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _DictPool:
    def __init__(self, conn) -> None:
        self._conn = conn

    def connection(self):
        return self._conn


class _AsyncDictCursor:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    async def fetchone(self):
        return self._rows[0] if self._rows else None

    async def fetchall(self):
        return list(self._rows)


class _AsyncDictConn:
    def __init__(self, responses: list[list[dict]]) -> None:
        self._responses = list(responses)
        self.queries: list[tuple] = []

    async def execute(self, sql, params=None):
        self.queries.append((sql, params))
        return _AsyncDictCursor(self._responses.pop(0) if self._responses else [])

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _AsyncDictPool:
    def __init__(self, conn) -> None:
        self._conn = conn

    def connection(self):
        return self._conn


class TestPostgresReadsItsColumnsByName:
    """The SQL forms must read rows as mappings, not by position.

    Both pools set ``row_factory=dict_row``, so ``row[0]`` raises KeyError — and
    the caller swallows exceptions to protect the write it decorates, which turns
    that mistake into a scope block that silently never appears. Exactly the
    invisible failure this feature exists to end, so it is worth a test that
    drives the real method rather than a stand-in adapter. The first cut of this
    change had the bug, and the fakes above are the reason it was not caught:
    they never went near the SQL.

    The precedent is in the same file as the code — ``reuse_summary`` carries a
    comment about this having happened before.
    """

    _AGG = [{"total": 6, "never_read": 4}]
    _KEYS = [{"key": "alpha"}, {"key": "beta"}]

    def test_sync_adapter(self):
        pg = pytest.importorskip("amfs_postgres.adapter")
        a = object.__new__(pg.PostgresAdapter)
        a._pool = _DictPool(_DictConn([self._AGG, self._KEYS]))
        a._namespace = "default"
        got = a.scope_counts("repo/topic", exclude_key="just-written")
        assert got == {
            "entity_path": "repo/topic", "existing_entries": 6,
            "never_read": 4, "keys": ["alpha", "beta"],
        }

    def test_async_adapter(self):
        pg = pytest.importorskip("amfs_postgres.async_adapter")
        a = object.__new__(pg.AsyncPostgresAdapter)
        a._pool = _AsyncDictPool(_AsyncDictConn([self._AGG, self._KEYS]))
        a._namespace = "default"
        got = asyncio.run(a.scope_counts("repo/topic", exclude_key="just-written"))
        assert got["existing_entries"] == 6
        assert got["never_read"] == 4
        assert got["keys"] == ["alpha", "beta"]

    def test_sql_excludes_the_entry_just_written(self):
        """The exclusion has to be in the query, not applied afterwards."""
        pg = pytest.importorskip("amfs_postgres.adapter")
        conn = _DictConn([self._AGG, self._KEYS])
        a = object.__new__(pg.PostgresAdapter)
        a._pool = _DictPool(conn)
        a._namespace = "default"
        a.scope_counts("repo/topic", exclude_key="just-written")
        for _sql, params in conn.queries:
            assert "just-written" in params

    def test_the_key_sample_is_bounded_in_sql(self):
        """A LIMIT, so a three-hundred-entry scope costs what a small one does."""
        pg = pytest.importorskip("amfs_postgres.adapter")
        conn = _DictConn([self._AGG, self._KEYS])
        a = object.__new__(pg.PostgresAdapter)
        a._pool = _DictPool(conn)
        a._namespace = "default"
        a.scope_counts("repo/topic", key_limit=8)
        assert any("LIMIT" in sql.upper() for sql, _ in conn.queries)
        assert any(8 in (p or ()) for _, p in conn.queries)

    def test_both_sql_forms_carry_the_visibility_predicate(self):
        """Every query, not just the one naming keys.

        The counts leak as surely as the sample does: "six entries are already
        here" is wrong, and quietly discloses, if two of the six are another
        agent's private notes. So the predicate belongs on the aggregate and on
        the key query alike, with the asking agent bound to both.
        """
        pg = pytest.importorskip("amfs_postgres.adapter")
        conn = _DictConn([self._AGG, self._KEYS])
        a = object.__new__(pg.PostgresAdapter)
        a._pool = _DictPool(conn)
        a._namespace = "default"
        a.scope_counts("repo/topic", agent_id="asking-agent")
        assert len(conn.queries) == 2
        for sql, params in conn.queries:
            assert "shared" in sql
            assert "asking-agent" in params

    def test_the_async_form_carries_it_too(self):
        pg = pytest.importorskip("amfs_postgres.async_adapter")
        conn = _AsyncDictConn([self._AGG, self._KEYS])
        a = object.__new__(pg.AsyncPostgresAdapter)
        a._pool = _AsyncDictPool(conn)
        a._namespace = "default"
        asyncio.run(a.scope_counts("repo/topic", agent_id="asking-agent"))
        assert len(conn.queries) == 2
        for sql, params in conn.queries:
            assert "shared" in sql
            assert "asking-agent" in params


server = pytest.importorskip("amfs_http.server")


class _Adapter:
    def __init__(self, counts):
        self._counts = counts
        self.calls: list[tuple] = []

    def scope_counts(self, entity_path, *, exclude_key=None, branch="main",
                     key_limit=8, agent_id=None):
        self.calls.append((entity_path, exclude_key, branch, agent_id))
        return self._counts

    def list(self, entity_path=None, *, branch="main", **k):
        return []


def _request():
    return types.SimpleNamespace(state=types.SimpleNamespace(visibility_filter=None))


class TestScopeBlockOnTheWritePath:
    def _block(self, monkeypatch, counts):
        adapter = _Adapter(counts)
        monkeypatch.setattr(server, "_async_adapter", None, raising=False)
        monkeypatch.setattr(
            server, "_get_memory",
            lambda: types.SimpleNamespace(_adapter=adapter, namespace="default"),
        )
        out = asyncio.run(server._scope_block("repo/topic", "just-written"))
        return out, adapter

    def test_returns_the_counts_for_the_scope(self, monkeypatch):
        out, adapter = self._block(monkeypatch, {
            "entity_path": "repo/topic", "existing_entries": 6,
            "never_read": 4, "keys": ["a", "b"],
        })
        assert out["existing_entries"] == 6
        assert out["never_read"] == 4
        assert adapter.calls == [("repo/topic", "just-written", "main", None)], (
            "the entry just written must be excluded from its own neighbours"
        )

    def test_the_asking_agent_reaches_the_adapter(self, monkeypatch):
        """Otherwise the visibility rule is enforced against nobody.

        The adapter decides what may be counted from this argument, so a caller
        that never passes it gets the shared-only view and the private entries
        the writing agent legitimately owns go unmentioned — while a caller that
        passes it wrongly would disclose. Worth pinning at the call boundary.
        """
        adapter = _Adapter({
            "entity_path": "repo/topic", "existing_entries": 1,
            "never_read": 1, "keys": ["a"],
        })
        monkeypatch.setattr(server, "_async_adapter", None, raising=False)
        monkeypatch.setattr(
            server, "_get_memory",
            lambda: types.SimpleNamespace(_adapter=adapter, namespace="default"),
        )
        asyncio.run(server._scope_block("repo/topic", "k", agent_id="writer-1"))
        assert adapter.calls == [("repo/topic", "k", "main", "writer-1")]

    def test_says_nothing_when_the_scope_is_empty(self, monkeypatch):
        """An empty block would read as a finding; absence is the honest answer."""
        out, _ = self._block(monkeypatch, {
            "entity_path": "repo/topic", "existing_entries": 0,
            "never_read": 0, "keys": [],
        })
        assert out is None

    def test_a_failure_costs_the_footnote_not_the_write(self, monkeypatch):
        class _Broken:
            def scope_counts(self, *a, **k):
                raise RuntimeError("db down")

        monkeypatch.setattr(server, "_async_adapter", None, raising=False)
        monkeypatch.setattr(
            server, "_get_memory",
            lambda: types.SimpleNamespace(_adapter=_Broken(), namespace="default"),
        )
        assert asyncio.run(server._scope_block("repo/topic", "k")) is None

    def test_awaits_an_async_adapter(self, monkeypatch):
        """The hosted path is async; the same helper has to serve both."""
        class _Async:
            async def scope_counts(self, entity_path, *, exclude_key=None,
                                   branch="main", key_limit=8, agent_id=None):
                return {"entity_path": entity_path, "existing_entries": 2,
                        "never_read": 1, "keys": ["x"]}

        monkeypatch.setattr(server, "_async_adapter", _Async(), raising=False)
        out = asyncio.run(server._scope_block("repo/topic", "k"))
        assert out["existing_entries"] == 2
