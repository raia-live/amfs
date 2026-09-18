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
        # Under the HNSW settings the fake returns however many rows the test
        # configured for the index route; the plain path returns the exact set.
        self._conn._pending = self._conn.rows_hnsw if self._conn.in_txn else self._conn.rows_exact

    async def fetchall(self):
        return list(self._conn._pending)


class _Txn:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn

    async def __aenter__(self):
        self._conn.in_txn = True
        self._conn.transactions += 1

    async def __aexit__(self, *exc):
        self._conn.in_txn = False
        return False


class _Conn:
    def __init__(
        self, *, rows_hnsw: list[dict[str, Any]], rows_exact: list[dict[str, Any]]
    ) -> None:
        self.rows_hnsw = rows_hnsw
        self.rows_exact = rows_exact
        self.statements: list[str] = []
        self.transactions = 0
        self.in_txn = False
        self._pending: list[dict[str, Any]] = []

    async def execute(self, sql: str) -> None:
        self.statements.append(sql)

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
        assert len(sets) == 1
        assert "hnsw.iterative_scan" in sets[0] and "enable_sort = off" in sets[0]
        selects = [s for s in conn.statements if "FROM amfs_memory_entries" in s]
        assert len(selects) == 1, "a full result must not trigger the exact re-run"

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
        # The re-run is outside the transaction: no SET LOCAL applies to it.
        assert conn.statements[-1] == selects[-1]


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
