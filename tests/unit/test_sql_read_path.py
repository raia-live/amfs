"""The dashboard read path when the database does the work.

Before this, ``GET /entries`` on an account-wide listing loaded every current
entry (100K on the largest account), filtered, sorted and sliced them in
Python on the event loop — for a caller asking for 50 rows. These tests pin
the replacement: visibility, order and page travel to the adapter as query
arguments, ``total`` comes from ``count_entries`` rather than ``len()``, the
aggregate handlers reach for the GROUP BY methods when the adapter has them,
and the HTTP client adapter pages transparently when a server answers with a
default-sized first page.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest
from amfs_core.models import MemoryEntry, Provenance
from amfs_core.scope import SqlScope

NOW = datetime.now(UTC)


def _entry(
    entity_path: str, key: str, agent_id: str = "agent-a", **kw: Any
) -> MemoryEntry:
    return MemoryEntry(
        entity_path=entity_path,
        key=key,
        value={"v": 1},
        provenance=Provenance(agent_id=agent_id, session_id="s1", written_at=NOW),
        confidence=0.8,
        **kw,
    )


# ---------------------------------------------------------------------------
# SqlScope.all_of
# ---------------------------------------------------------------------------


class TestSqlScopeAllOf:
    def test_nothing_to_conjoin_is_none(self) -> None:
        assert SqlScope.all_of() is None
        assert SqlScope.all_of(None, None) is None

    def test_single_scope_passes_through_unchanged(self) -> None:
        s = SqlScope("shared", ())
        assert SqlScope.all_of(None, s, None) is s

    def test_conjunction_keeps_params_in_clause_order(self) -> None:
        a = SqlScope("agent_id = %s", ("x",))
        b = SqlScope("entity_path = %s OR entity_path LIKE %s", ("p", "p/%"))
        combined = SqlScope.all_of(a, b)
        assert combined is not None
        assert combined.clause == "(agent_id = %s) AND (entity_path = %s OR entity_path LIKE %s)"
        assert combined.params == ("x", "p", "p/%")

    def test_apply_appends_parenthesised(self) -> None:
        conds: list[str] = ["namespace = %s"]
        params: list[Any] = ["ns"]
        SqlScope("agent_id = %s", ("a",)).apply(conds, params)
        assert conds == ["namespace = %s", "(agent_id = %s)"]
        assert params == ["ns", "a"]


# ---------------------------------------------------------------------------
# list_conditions — the WHERE shared by list() and count_entries()
# ---------------------------------------------------------------------------

pytest.importorskip("psycopg", reason="psycopg not installed")
from amfs_postgres.adapter import (  # noqa: E402
    _list_order_sql,
    _paginate_sql,
    list_conditions,
)


class TestListConditions:
    def _where(self, **kw: Any) -> tuple[str, list[Any]]:
        conds, params = list_conditions(
            "ns",
            kw.pop("entity_path", None),
            include_superseded=kw.pop("include_superseded", False),
            branch=kw.pop("branch", "main"),
            scope=kw.pop("scope", None),
            agent_id=kw.pop("agent_id", None),
        )
        assert not kw
        return " AND ".join(conds), params

    def test_scope_and_agent_become_predicates(self) -> None:
        where, params = self._where(
            scope=SqlScope("agent_id = ANY(%s)", (["a", "b"],)), agent_id="me"
        )
        assert "(agent_id = ANY(%s))" in where
        assert "agent_id = %s" in where
        assert ["a", "b"] in params
        assert "me" in params

    def test_unscoped_namespace_read_excludes_shared_paths(self) -> None:
        where, _ = self._where()
        assert "NOT LIKE" in where or "entity_path NOT" in where

    def test_entity_read_does_not_exclude_shared_paths(self) -> None:
        where, params = self._where(entity_path="repo/a")
        assert "repo/a" in params
        assert "NOT LIKE" not in where

    def test_superseded_filter_toggles(self) -> None:
        live, _ = self._where()
        all_, _ = self._where(include_superseded=True)
        assert "superseded_at IS NULL" in live
        assert "superseded_at IS NULL" not in all_


class TestOrderAndPage:
    def test_known_orders(self) -> None:
        assert "written_at DESC" in _list_order_sql("written_at")
        assert "recall_count DESC" in _list_order_sql("recall_count")
        assert _list_order_sql(None)

    def test_unknown_order_is_rejected_not_interpolated(self) -> None:
        with pytest.raises(ValueError):
            _list_order_sql("written_at; DROP TABLE x")

    def test_pagination_as_parameters(self) -> None:
        q, p = _paginate_sql("SELECT 1", ["a"], limit=50, offset=100)
        assert q.endswith(" LIMIT %s OFFSET %s")
        assert p == ["a", 50, 100]

    def test_no_limit_no_offset_leaves_query_alone(self) -> None:
        q, p = _paginate_sql("SELECT 1", ["a"], limit=None, offset=0)
        assert q == "SELECT 1"
        assert p == ["a"]


# ---------------------------------------------------------------------------
# HTTP handlers on a paging adapter
# ---------------------------------------------------------------------------

pytest.importorskip("fastapi", reason="fastapi not installed")
import amfs_http.server as server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


class _PagingAdapter:
    """A stand-in for the Postgres adapter: records the query arguments it is
    handed and answers from an in-memory list the way SQL would."""

    def __init__(self, entries: list[MemoryEntry]) -> None:
        self._entries = entries
        self.list_calls: list[dict[str, Any]] = []
        self.count_calls: list[dict[str, Any]] = []
        self.agent_summary_calls: list[dict[str, Any]] = []

    def _select(self, entity_path: str | None, kw: dict[str, Any]) -> list[MemoryEntry]:
        rows = [e for e in self._entries if entity_path is None or e.entity_path == entity_path]
        if kw.get("order_by") == "recall_count":
            rows.sort(key=lambda e: e.recall_count, reverse=True)
        return rows

    def list(self, entity_path: str | None = None, **kw: Any) -> list[MemoryEntry]:
        self.list_calls.append({"entity_path": entity_path, **kw})
        rows = self._select(entity_path, kw)
        off = kw.get("offset") or 0
        rows = rows[off:]
        if kw.get("limit") is not None:
            rows = rows[: kw["limit"]]
        return rows

    def count_entries(self, entity_path: str | None = None, **kw: Any) -> int:
        self.count_calls.append({"entity_path": entity_path, **kw})
        return len(self._select(entity_path, kw))

    def agent_summaries(self, **kw: Any) -> list[dict[str, Any]]:
        self.agent_summary_calls.append(kw)
        return [
            {
                "agent_id": "agent-a",
                "entries_written": 2,
                "entities_touched": 2,
                "first_seen": NOW,
                "last_active": NOW,
            }
        ]

    # The handlers probe for these; absent means "use the entry list".
    def agent_registration(self) -> dict[str, Any]:
        return {}

    def agent_descriptions(self) -> dict[str, str]:
        return {}


@pytest.fixture()
def paging(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, _PagingAdapter]:
    entries = [
        _entry("repo/a", "k1", recall_count=1),
        _entry("repo/a", "k2", recall_count=9),
        _entry("repo/b", "k1", recall_count=5),
        _entry("repo/c", "k1", recall_count=3),
    ]
    adapter = _PagingAdapter(entries)
    mem = MagicMock()
    mem.namespace = "test-ns"
    mem.agent_id = "me"
    mem._adapter = adapter
    mem.list.side_effect = lambda *a, **kw: adapter.list(*a)

    monkeypatch.setattr(server, "_memory", mem)
    monkeypatch.setattr(server, "_get_memory", lambda: mem)
    monkeypatch.setattr(server, "_async_adapter", None)
    monkeypatch.setattr(server, "_get_visibility_filter", lambda request: None)
    monkeypatch.setattr(server, "ENTRIES_DEFAULT_LIMIT", 0)
    return TestClient(server.app), adapter


class TestEntriesInSql:
    def test_page_and_order_reach_the_adapter(
        self, paging: tuple[TestClient, _PagingAdapter]
    ) -> None:
        client, adapter = paging
        res = client.get("/api/v1/entries?limit=2&offset=1&sort=recall_count")
        assert res.status_code == 200
        body = res.json()
        call = adapter.list_calls[-1]
        assert (call["limit"], call["offset"], call["order_by"]) == (2, 1, "recall_count")
        # The page came from the adapter; the total did not come from len(page).
        assert len(body["entries"]) == 2
        assert body["total"] == 4
        assert adapter.count_calls, "total must be a COUNT, not the page length"
        assert [e["recall_count"] for e in body["entries"]] == [5, 3]

    def test_unpaged_first_page_skips_the_count(
        self, paging: tuple[TestClient, _PagingAdapter]
    ) -> None:
        client, adapter = paging
        body = client.get("/api/v1/entries").json()
        assert body["total"] == 4 == len(body["entries"])
        assert adapter.count_calls == []

    def test_sync_path_hides_other_agents_private_entries_in_sql(
        self, paging: tuple[TestClient, _PagingAdapter]
    ) -> None:
        client, adapter = paging
        client.get("/api/v1/entries?limit=1")
        scope = adapter.list_calls[-1]["scope"]
        assert isinstance(scope, SqlScope)
        assert "shared OR agent_id = %s" in scope.clause
        assert "me" in scope.params
        # Page and total agree on which rows exist.
        assert adapter.count_calls[-1]["scope"] == scope

    def test_default_limit_applies_when_configured(
        self, paging: tuple[TestClient, _PagingAdapter], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, adapter = paging
        monkeypatch.setattr(server, "ENTRIES_DEFAULT_LIMIT", 3)
        body = client.get("/api/v1/entries").json()
        assert adapter.list_calls[-1]["limit"] == 3
        assert len(body["entries"]) == 3
        assert body["total"] == 4

    def test_visibility_predicate_is_conjoined(
        self, paging: tuple[TestClient, _PagingAdapter], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, adapter = paging
        vis = MagicMock()
        vis.should_filter.return_value = True
        vis.sql_predicate.return_value = SqlScope("agent_id = ANY(%s)", (["agent-a"],))
        monkeypatch.setattr(server, "_get_visibility_filter", lambda request: vis)
        client.get("/api/v1/entries?limit=2")
        scope = adapter.list_calls[-1]["scope"]
        assert "agent_id = ANY(%s)" in scope.clause
        assert "shared OR agent_id = %s" in scope.clause
        vis.filter_entries.assert_not_called()

    def test_python_only_filter_still_pages_after_filtering(
        self, paging: tuple[TestClient, _PagingAdapter], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, adapter = paging
        vis = MagicMock()
        vis.should_filter.return_value = True
        vis.sql_predicate.return_value = None
        vis.filter_entries.side_effect = lambda es: [e for e in es if e.entity_path != "repo/a"]
        monkeypatch.setattr(server, "_get_visibility_filter", lambda request: vis)
        body = client.get("/api/v1/entries?limit=1&offset=1").json()
        # Filtered to 2 visible rows, then paged: total reflects what the
        # caller may see, never the namespace.
        assert body["total"] == 2
        assert len(body["entries"]) == 1
        assert body["entries"][0]["entity_path"] == "repo/c"


class _LostContextAsyncAdapter:
    """An async adapter whose pool has lost its tenant context: every read
    and every count answers with nothing."""

    def __init__(self) -> None:
        self.count_calls = 0

    async def list(self, entity_path: str | None = None, **kw: Any) -> list[MemoryEntry]:
        return []

    async def count_entries(self, entity_path: str | None = None, **kw: Any) -> int:
        self.count_calls += 1
        return 0


class TestEntriesRecoveredFromSync:
    def test_total_follows_the_page_to_the_sync_adapter(
        self, paging: tuple[TestClient, _PagingAdapter], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When an empty async first page is replaced by the sync list, the
        total must come from the sync adapter too — the async pool that
        returned nothing would count nothing, and a client paging on ``total``
        would stop after page one. (Bugbot, raia-live/amfs#421.)"""
        client, adapter = paging
        lost = _LostContextAsyncAdapter()
        monkeypatch.setattr(server, "_async_adapter", lost)

        body = client.get("/api/v1/entries?limit=2").json()

        assert len(body["entries"]) == 2
        assert body["total"] == 4, "total came from the pool that produced the page"
        assert lost.count_calls == 0, "the lost-context pool must not be asked to count"
        # And the sync count was scoped exactly like the sync page.
        assert adapter.count_calls[-1]["scope"] == adapter.list_calls[-1]["scope"]

    def test_no_recovery_when_sync_is_empty_too(
        self, paging: tuple[TestClient, _PagingAdapter], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, adapter = paging
        adapter._entries.clear()
        lost = _LostContextAsyncAdapter()
        monkeypatch.setattr(server, "_async_adapter", lost)

        body = client.get("/api/v1/entries?limit=2").json()
        assert body["entries"] == []
        assert body["total"] == 0


class TestAgentsInSql:
    def test_agents_use_group_by_when_available(
        self, paging: tuple[TestClient, _PagingAdapter]
    ) -> None:
        client, adapter = paging
        res = client.get("/api/v1/agents")
        assert res.status_code == 200
        assert adapter.agent_summary_calls, "GET /agents must aggregate in SQL"
        agents = res.json()["agents"]
        assert [a["agentId"] for a in agents] == ["agent-a"]
        assert agents[0]["entriesWritten"] == 2
        assert agents[0]["entitiesTouched"] == 2


# ---------------------------------------------------------------------------
# HTTP client adapter pages through a default-limited server
# ---------------------------------------------------------------------------

pytest.importorskip("amfs_adapter_http", reason="http adapter not installed")
from amfs_adapter_http.adapter import HttpAdapter  # noqa: E402


def _wire_entry(entity_path: str, key: str) -> dict[str, Any]:
    return {
        "entity_path": entity_path,
        "key": key,
        "value": {"v": 1},
        "version": 1,
        "confidence": 0.8,
        "provenance": {
            "agent_id": "agent-a",
            "session_id": "s1",
            "written_at": (NOW - timedelta(minutes=1)).isoformat(),
        },
    }


class TestHttpAdapterPaging:
    def _adapter_with_pages(self, total: int, page: int) -> tuple[HttpAdapter, list[dict]]:
        rows = [_wire_entry("repo/a", f"k{i}") for i in range(total)]
        calls: list[dict[str, Any]] = []
        adapter = HttpAdapter.__new__(HttpAdapter)

        def fake_get(path: str, **params: Any) -> Any:
            calls.append(params)
            off = int(params.get("offset") or 0)
            lim = params.get("limit")
            chunk = rows[off:] if lim is None else rows[off : off + int(lim)]
            return {"entries": chunk[:page] if lim is None else chunk, "total": total}

        adapter._get = fake_get  # type: ignore[method-assign]
        return adapter, calls

    def test_follows_total_across_pages(self) -> None:
        adapter, calls = self._adapter_with_pages(total=7, page=3)
        out = adapter.list()
        assert [e.key for e in out] == [f"k{i}" for i in range(7)]
        # First call unpaged, then pages at the size the server chose.
        assert calls[0].get("limit") is None
        assert [(c["limit"], c["offset"]) for c in calls[1:]] == [(3, 3), (3, 6)]

    def test_single_page_makes_one_request(self) -> None:
        adapter, calls = self._adapter_with_pages(total=2, page=10)
        assert len(adapter.list()) == 2
        assert len(calls) == 1
