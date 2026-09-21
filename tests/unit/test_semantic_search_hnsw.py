"""The account-wide vector search takes the HNSW index, and only when it can.

Background in ``amfs_postgres.knn``: under ``/retrieve``'s WHERE clause the
planner walked a b-tree and computed every distance in the namespace (~600 ms
per query variant on a 100K-entry account). These tests pin the routing rule
and the fallback, with a fake connection that records what was sent.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
from typing import Any

import pytest
from amfs_core.models import MemoryEntry, Provenance, SemanticQuery
from amfs_postgres import knn
from amfs_postgres.async_adapter import AsyncPostgresAdapter

# ── knn helpers ───────────────────────────────────────────────────────


class TestVersionGate:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("0.8.1", (0, 8, 1)),
            ("0.8.0", (0, 8, 0)),
            ("0.7.4", (0, 7, 4)),
            ("1.0", (1, 0)),
            (None, None),
            ("", None),
            ("garbage", None),
        ],
    )
    def test_parse(self, text, expected) -> None:
        assert knn.parse_pgvector_version(text) == expected

    @pytest.mark.parametrize(
        "version,ok",
        [((0, 8, 1), True), ((0, 8, 0), True), ((1, 0), True), ((0, 7, 4), False), (None, False)],
    )
    def test_iterative_scan_needs_0_8(self, version, ok) -> None:
        assert knn.supports_iterative_scan(version) is ok

    def test_scoped_reads_stay_exact(self) -> None:
        """A path-scoped read is a small b-tree range; the index route is for
        the whole-account read only."""
        assert knn.use_hnsw_scan(entity_path=None, version=(0, 8, 1))
        assert not knn.use_hnsw_scan(entity_path="acme/billing", version=(0, 8, 1))
        assert not knn.use_hnsw_scan(entity_path=None, version=(0, 7, 4))
        assert not knn.use_hnsw_scan(entity_path=None, version=None)


class TestScanSettings:
    def test_are_local_and_cover_the_limit(self) -> None:
        sql = knn.hnsw_scan_settings(150)
        assert "SET LOCAL hnsw.iterative_scan = 'relaxed_order'" in sql
        assert "SET LOCAL hnsw.ef_search = 150" in sql
        assert "SET LOCAL enable_sort = off" in sql
        # SET LOCAL, never SET: the connection goes back to the pool.
        assert " SET hnsw" not in f" {sql}".replace("SET LOCAL", "")

    @pytest.mark.parametrize(
        "limit,ef", [(1, 40), (40, 40), (999, 999), (1000, 1000), (5000, 1000)]
    )
    def test_ef_search_is_clamped_to_pgvector_range(self, limit, ef) -> None:
        assert f"hnsw.ef_search = {ef}" in knn.hnsw_scan_settings(limit)

    def test_reset_undoes_every_setting_it_made(self) -> None:
        """Every GUC the settings touch is restored, by name, and locally."""
        settings = knn.hnsw_scan_settings(150)
        reset = knn.hnsw_scan_reset()
        set_gucs = {s.split()[2] for s in settings.split("; ")}
        reset_gucs = {s.split()[2] for s in reset.split("; ")}
        assert set_gucs == reset_gucs == {"hnsw.iterative_scan", "hnsw.ef_search", "enable_sort"}
        assert all(
            s.startswith("SET LOCAL ") and s.endswith(" TO DEFAULT") for s in reset.split("; ")
        )


# ── the adapter's routing, on a fake connection ───────────────────────


def _row(i: int) -> dict[str, Any]:
    return {"id": i, "similarity": 0.9 - i * 0.001}


def _entry(row: dict[str, Any]) -> MemoryEntry:
    return MemoryEntry(
        entity_path="acme/billing",
        key=f"k{row['id']}",
        value="v",
        provenance=Provenance(agent_id="a", session_id="s", written_at=datetime.now(UTC)),
    )


class _Cursor:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, sql: str, params: Any = None) -> None:
        self._conn.statements.append(sql)
        # While the HNSW settings are in force the fake returns however many
        # rows the test configured for the index route; otherwise the exact set.
        self._conn._pending = (
            self._conn.rows_hnsw if self._conn.hnsw_settings_active else self._conn.rows_exact
        )

    async def fetchall(self):
        return list(self._conn._pending)


class _Txn:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn

    async def __aenter__(self):
        self._conn.transactions += 1
        self._conn.depth += 1

    async def __aexit__(self, *exc):
        self._conn.depth -= 1
        # Postgres semantics: SET LOCAL lasts to the end of the *outermost*
        # transaction. Leaving a savepoint (depth still > 0) keeps it; leaving
        # a real transaction drops it.
        if self._conn.depth == 0:
            self._conn.hnsw_settings_active = False
        return False


class _Conn:
    """Records what was sent and applies Postgres' SET LOCAL lifetime rules.

    ``outer_transaction=True`` models the checkout the tenant RLS wrapper
    hands out: already inside a transaction, so the adapter's own
    ``transaction()`` is a savepoint.
    """

    def __init__(
        self,
        *,
        rows_hnsw: list[dict[str, Any]],
        rows_exact: list[dict[str, Any]],
        outer_transaction: bool = False,
    ) -> None:
        self.rows_hnsw = rows_hnsw
        self.rows_exact = rows_exact
        self.statements: list[str] = []
        self.transactions = 0
        self.depth = 1 if outer_transaction else 0
        self.hnsw_settings_active = False
        self._pending: list[dict[str, Any]] = []

    async def execute(self, sql: str) -> None:
        self.statements.append(sql)
        if sql.startswith("SET LOCAL"):
            if "TO DEFAULT" in sql:
                self.hnsw_settings_active = False
            elif self.depth > 0:  # SET LOCAL outside a transaction is a no-op
                self.hnsw_settings_active = True

    def cursor(self) -> _Cursor:
        return _Cursor(self)

    def transaction(self) -> _Txn:
        return _Txn(self)


class _Pool:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn

    @contextlib.asynccontextmanager
    async def connection(self):
        yield self._conn


def _adapter(conn: _Conn, *, version: tuple[int, ...] | None) -> AsyncPostgresAdapter:
    ad = object.__new__(AsyncPostgresAdapter)
    ad._namespace = "default"
    ad._has_embedding_col = True
    ad._has_is_artifact_col = True
    ad._pgvector_version = version
    ad._pool = _Pool(conn)
    ad._row_to_entry = _entry  # type: ignore[method-assign]
    return ad


def _search(ad: AsyncPostgresAdapter, **kw: Any):
    q = SemanticQuery(text="q", limit=kw.pop("limit", 150), embedding=[0.1, 0.2], **kw)
    return asyncio.run(ad.semantic_search(q, embedder=None))


class TestAccountWideReadUsesTheIndex:
    def test_settings_are_set_locally_around_the_one_statement(self) -> None:
        conn = _Conn(rows_hnsw=[_row(i) for i in range(150)], rows_exact=[])
        pairs = _search(_adapter(conn, version=(0, 8, 1)))

        assert len(pairs) == 150
        assert conn.transactions == 1
        sets = [s for s in conn.statements if s.startswith("SET LOCAL")]
        assert len(sets) == 2, "the settings, then their reset"
        assert "hnsw.iterative_scan" in sets[0] and "enable_sort = off" in sets[0]
        assert "TO DEFAULT" in sets[1]
        selects = [s for s in conn.statements if "FROM amfs_memory_entries" in s]
        assert len(selects) == 1, "a full result must not trigger the exact re-run"
        # Settings, statement, reset — in that order, so nothing after the
        # search on this connection runs under the index settings.
        order = [conn.statements.index(s) for s in (sets[0], selects[0], sets[1])]
        assert order == sorted(order)
        assert not conn.hnsw_settings_active

    def test_the_rows_are_projected_not_star(self) -> None:
        conn = _Conn(rows_hnsw=[_row(i) for i in range(150)], rows_exact=[])
        _search(_adapter(conn, version=(0, 8, 1)))
        select = next(s for s in conn.statements if "FROM amfs_memory_entries" in s)
        assert "SELECT *" not in select
        assert "AS similarity" in select
        # The row's own vector is not shipped back only to be dropped.
        select_list = select.split("FROM amfs_memory_entries")[0]
        assert " embedding," not in select_list and "embedding\n" not in select_list

    def test_an_underfilled_index_scan_falls_back_to_the_exact_scan(self) -> None:
        """The iterative scan hit ``max_scan_tuples`` before the filter let
        enough rows through — a small tenant in a large shared table. The
        exact scan is what the caller got before, and it is complete."""
        conn = _Conn(
            rows_hnsw=[_row(i) for i in range(30)], rows_exact=[_row(i) for i in range(150)]
        )
        pairs = _search(_adapter(conn, version=(0, 8, 1)))

        assert len(pairs) == 150
        selects = [s for s in conn.statements if "FROM amfs_memory_entries" in s]
        assert len(selects) == 2
        assert conn.transactions == 1
        # The re-run comes after the reset: no index setting applies to it.
        reset_at = next(i for i, s in enumerate(conn.statements) if "TO DEFAULT" in s)
        select_positions = [
            i for i, s in enumerate(conn.statements) if "FROM amfs_memory_entries" in s
        ]
        assert select_positions[0] < reset_at < select_positions[1]
        assert conn.statements[-1] == selects[-1]

    def test_fallback_is_exact_even_inside_the_rls_wrappers_transaction(self) -> None:
        """The tenant wrapper opens a transaction before the pool hands the
        connection out, so the adapter's ``transaction()`` is a savepoint and
        ``SET LOCAL`` survives its release. Without an explicit reset the
        "exact" re-run would be forced back onto the index and return the same
        under-filled 30 rows — for exactly the small tenant the fallback is
        for. (Bugbot, raia-live/amfs#421.)"""
        conn = _Conn(
            rows_hnsw=[_row(i) for i in range(30)],
            rows_exact=[_row(i) for i in range(150)],
            outer_transaction=True,
        )
        pairs = _search(_adapter(conn, version=(0, 8, 1)))

        assert len(pairs) == 150, "the fallback must not inherit the HNSW settings"
        assert not conn.hnsw_settings_active, (
            "nothing later in the outer transaction inherits them either"
        )


class TestWhenTheIndexRouteIsNotAvailable:
    def test_scoped_reads_do_not_touch_the_settings(self) -> None:
        conn = _Conn(rows_hnsw=[], rows_exact=[_row(i) for i in range(10)])
        pairs = _search(_adapter(conn, version=(0, 8, 1)), entity_path="acme/billing", limit=10)

        assert len(pairs) == 10
        assert conn.transactions == 0
        assert not any(s.startswith("SET LOCAL") for s in conn.statements)

    def test_old_pgvector_keeps_the_exact_path(self) -> None:
        """``hnsw.iterative_scan`` does not exist before 0.8; setting it would
        error, and without it the index returns at most ef_search rows before
        the filter — worse than the scan it replaces."""
        conn = _Conn(rows_hnsw=[], rows_exact=[_row(i) for i in range(150)])
        pairs = _search(_adapter(conn, version=(0, 7, 4)))

        assert len(pairs) == 150
        assert conn.transactions == 0
        assert not any(s.startswith("SET LOCAL") for s in conn.statements)

    def test_no_pgvector_column_means_no_search(self) -> None:
        conn = _Conn(rows_hnsw=[], rows_exact=[])
        ad = _adapter(conn, version=None)
        ad._has_embedding_col = False
        assert _search(ad) == []
        assert conn.statements == []
