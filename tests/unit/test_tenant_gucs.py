"""The tenant settings module, and the transaction-scoped form of applying them.

The session-scoped path is covered by test_tenant_rls_connection.py. What is
tested here is the pooler-safe path and, mostly, the trap underneath it.

``set_config(name, value, true)`` lasts until the end of the current
transaction. Both adapters open their pools with ``autocommit=True``, so a
statement that is not inside an explicit transaction *is* its own transaction
and the setting is discarded before the next statement runs. The GUCs then read
empty, ``NULLIF`` in the policies turns empty into NULL, and every RLS-protected
read matches zero rows — without raising anything. An empty page that looks like
an empty account is the worst failure available here, so ``tenant_transaction``
checks it has a real transaction rather than trusting that it does.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from amfs_postgres import tenant_gucs


def _cursor(recorder: list) -> MagicMock:
    cur = MagicMock()
    cur.execute = MagicMock(
        side_effect=lambda sql, params=None: recorder.append((sql, params))
    )
    cur.__enter__ = MagicMock(return_value=cur)
    cur.__exit__ = MagicMock(return_value=None)
    return cur


class _AsyncCursor:
    """An async cursor, written out rather than mocked.

    MagicMock can be bent into supporting `async with` and an awaitable
    execute, but the bending is longer than the class and reads worse.
    """

    def __init__(self, recorder: list) -> None:
        self._recorder = recorder

    async def __aenter__(self) -> _AsyncCursor:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    async def execute(self, sql: str, params: tuple | None = None) -> None:
        self._recorder.append((sql, params))


def _connection(recorder: list, *, status: str = "INTRANS") -> MagicMock:
    """A connection whose ``transaction()`` reports ``status`` once entered."""
    conn = MagicMock()
    conn.cursor = MagicMock(return_value=_cursor(recorder))
    conn.info.transaction_status = MagicMock()
    conn.info.transaction_status.name = status

    txn = MagicMock()
    txn.__enter__ = MagicMock(return_value=None)
    txn.__exit__ = MagicMock(return_value=None)
    conn.transaction = MagicMock(return_value=txn)
    return conn


def _pool(conn: MagicMock) -> MagicMock:
    """An unwrapped pool.

    ``spec`` matters: ``tenant_transaction`` unwraps with
    ``getattr(pool, "_inner", pool)``, and a plain MagicMock would happily
    invent a ``_inner`` attribute, so the function would unwrap to a phantom
    mock instead of falling back to the pool. Restricting the spec makes the
    mock behave like the real pools, only one of which has ``_inner``.
    """
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=conn)
    ctx.__exit__ = MagicMock(return_value=None)
    pool = MagicMock(spec=["connection"])
    pool.connection = MagicMock(return_value=ctx)
    return pool


def _context(*, account="acct-1", team=None, admin=False, user=None):
    return (
        patch(
            "amfs_postgres.tenant_context.get_request_tenant_account_id",
            return_value=account,
        ),
        patch(
            "amfs_postgres.tenant_context.get_request_tenant_team_id",
            return_value=team,
        ),
        patch(
            "amfs_postgres.tenant_context.get_request_is_account_admin",
            return_value=admin,
        ),
        patch(
            "amfs_postgres.tenant_context.get_request_user_id", return_value=user
        ),
    )


class TestOneDefinitionOfTheTenant:
    def test_every_name_appears_in_both_statements(self):
        """The two adapters used to carry a copy each of this SQL. A fifth
        setting added to one and not the other is a silent policy hole, so the
        names drive the SQL rather than being written out beside it."""
        for name in tenant_gucs.TENANT_GUC_NAMES:
            assert f"set_config('{name}'" in tenant_gucs.SET_TENANT_GUCS_SESSION
            assert f"set_config('{name}'" in tenant_gucs.SET_TENANT_GUCS_LOCAL

    def test_the_two_statements_differ_only_in_scope(self):
        assert tenant_gucs.SET_TENANT_GUCS_SESSION.replace(
            ", false)", ", true)"
        ) == tenant_gucs.SET_TENANT_GUCS_LOCAL

    def test_one_round_trip(self):
        # This runs on every checkout, on every request.
        assert tenant_gucs.SET_TENANT_GUCS_LOCAL.count("SELECT") == 1

    def test_absent_values_become_empty_not_null(self):
        """Empty string is what NULLIF turns into NULL, so a missing tenant
        matches nothing. Saying it explicitly beats relying on how psycopg
        happens to render None."""
        with pytest.MonkeyPatch.context():
            for p in _context(account=None, team=None, admin=False, user=None):
                p.start()
            try:
                assert tenant_gucs.tenant_guc_values() == ("", "", "false", "")
            finally:
                patch.stopall()

    def test_values_are_ordered_to_match_the_statement(self):
        for p in _context(account="a", team="t", admin=True, user="u"):
            p.start()
        try:
            assert tenant_gucs.tenant_guc_values() == ("a", "t", "true", "u")
        finally:
            patch.stopall()


class TestTransactionScopedTenant:
    def test_sets_all_four_locally_inside_the_transaction(self):
        recorder: list = []
        conn = _connection(recorder)

        for p in _context(account="acct-9", team="team-3", admin=False, user="u-7"):
            p.start()
        try:
            with tenant_gucs.tenant_transaction(_pool(conn)) as got:
                assert got is conn
        finally:
            patch.stopall()

        assert conn.transaction.called, "must open a real transaction"
        assert len(recorder) == 1
        sql, params = recorder[0]
        assert sql == tenant_gucs.SET_TENANT_GUCS_LOCAL
        assert params == ("acct-9", "team-3", "false", "u-7")

    def test_refuses_to_run_without_an_open_transaction(self):
        """The whole point. Outside a transaction the local set_config is
        discarded immediately and every subsequent read silently returns
        nothing, so this has to be an exception rather than an empty result."""
        recorder: list = []
        conn = _connection(recorder, status="IDLE")

        for p in _context():
            p.start()
        try:
            with pytest.raises(RuntimeError, match="no open transaction"):
                with tenant_gucs.tenant_transaction(_pool(conn)):
                    pass
        finally:
            patch.stopall()

        assert recorder == [], "must not set anything it cannot scope"

    def test_unwraps_the_session_scoped_pool_wrapper(self):
        """Going through the wrapper would set the same four session-scoped at
        checkout, and through a transaction pooler those land on an arbitrary
        backend and stay there — the exact leak this function avoids."""
        recorder: list = []
        conn = _connection(recorder)
        inner = _pool(conn)

        wrapper = MagicMock()
        wrapper._inner = inner
        wrapper.connection = MagicMock(
            side_effect=AssertionError("must not check out through the wrapper")
        )

        for p in _context():
            p.start()
        try:
            with tenant_gucs.tenant_transaction(wrapper):
                pass
        finally:
            patch.stopall()

        assert inner.connection.called


class TestReturningAConnectionToThePool:
    def test_reset_blanks_all_four(self):
        recorder: list = []
        conn = MagicMock()
        conn.cursor = MagicMock(return_value=_cursor(recorder))

        tenant_gucs.reset_tenant_gucs(conn)

        sql, params = recorder[0]
        assert sql == tenant_gucs.SET_TENANT_GUCS_SESSION
        assert params == ("", "", "false", "")
        assert len(params) == len(tenant_gucs.TENANT_GUC_NAMES)

    async def test_the_async_reset_blanks_the_same_four(self):
        """Both pools need this and only one callback can be awaited, so the
        async twin exists. The request path is served by the async pool, which
        makes it the one where a connection left holding a real
        amfs.current_user_id matters most."""
        recorder: list = []
        conn = MagicMock()
        conn.cursor = MagicMock(return_value=_AsyncCursor(recorder))

        await tenant_gucs.areset_tenant_gucs(conn)

        sql, params = recorder[0]
        assert sql == tenant_gucs.SET_TENANT_GUCS_SESSION
        assert params == ("", "", "false", "")
