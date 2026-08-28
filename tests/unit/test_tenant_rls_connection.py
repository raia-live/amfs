"""The sync pooled-connection RLS wrapper.

Four session variables decide what a request can see. Three of them narrow:
account, team, and the admin flag. A stale value there fails closed — the
next borrower of the connection sees less than they should, which is a bug
but not a leak.

amfs.current_user_id is the one that widens. Permissive policies use it to
let a user reach rooms in accounts that are not theirs, so a connection
handed on with the previous borrower's user still set grants the next
request that person's room access. Hence: set on every checkout, never
conditionally.

Scope is the other half, and it is what most of this file is about. The four
used to be set session-scoped, which is sound with psycopg_pool alone and a
cross-tenant read behind a transaction-mode pooler, because there one client
connection maps to a different backend per transaction. So the checkout now
opens an explicit transaction and sets them inside it, and Postgres discards
them at commit. The tests below pin the parts of that which are easy to
regress by accident: that a transaction is opened *before* the settings are
written, that the statement is the local form, and that the two paths which
cannot be transactional write no tenant at all.
"""

from __future__ import annotations

import contextlib
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


class _Transaction:
    """A stand-in for ``conn.transaction()`` that records its own lifecycle.

    Written out rather than mocked because the ordering assertions need to know
    when it was entered relative to the ``set_config`` call, and a MagicMock
    would answer that question with another MagicMock.
    """

    def __init__(self, log: list) -> None:
        self._log = log
        self.entered = False
        self.exited_with: tuple | None = None

    def __enter__(self) -> _Transaction:
        self.entered = True
        self._log.append(("BEGIN", None))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.exited_with = (exc_type, exc, tb)
        self._log.append(("COMMIT" if exc_type is None else "ROLLBACK", None))
        return None


@pytest.fixture
def _make_conn():
    """An inner checkout context whose connection looks like it is in a txn.

    ``transaction_status.name`` has to be a real string: the wrapper asserts a
    transaction is genuinely open before it writes anything transaction-local,
    since a local ``set_config`` outside a transaction is discarded and turns
    every subsequent read into a silent empty result.
    """
    executed: list = []

    cur = MagicMock()
    cur.execute = MagicMock(
        side_effect=lambda sql, params=None: executed.append((sql, params))
    )
    cur.__enter__ = MagicMock(return_value=cur)
    cur.__exit__ = MagicMock(return_value=None)

    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cur)
    conn.info.transaction_status.name = "INTRANS"
    txn = _Transaction(executed)
    conn.transaction = MagicMock(return_value=txn)

    inner_ctx = MagicMock()
    inner_ctx.__enter__ = MagicMock(return_value=conn)
    inner_ctx.__exit__ = MagicMock(return_value=None)

    return inner_ctx, executed


def _sets(executed: list) -> list:
    """Just the set_config statements, dropping the BEGIN/COMMIT markers."""
    return [(sql, params) for sql, params in executed if "set_config" in str(sql)]


def _tenant(*, account="acct-1", team=None, admin=False, user=None):
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


def _checkout(
    inner_ctx,
    *,
    account="acct-1",
    team=None,
    admin=False,
    user=None,
    transactional=True,
):
    from amfs_postgres.adapter import _TenantRLSConnection

    with contextlib.ExitStack() as stack:
        for p in _tenant(account=account, team=team, admin=admin, user=user):
            stack.enter_context(p)
        wrapper = _TenantRLSConnection(inner_ctx, transactional=transactional)
        return wrapper, wrapper.__enter__()


class TestTheActingUserReachesTheDatabase:
    def test_the_user_guc_is_set_from_context(self, _make_conn):
        """Without this, the cross-account room policies match nothing and an
        invited user sees an empty room rather than its contents."""
        inner_ctx, executed = _make_conn

        _checkout(inner_ctx, account="acct-1", user="user-42")

        sql, params = _sets(executed)[0]
        assert "set_config('amfs.current_user_id'" in sql
        assert "user-42" in params

    def test_all_four_gucs_are_set_in_one_statement(self, _make_conn):
        # One round trip per checkout; this is on every request's hot path.
        inner_ctx, executed = _make_conn

        _checkout(inner_ctx, account="a", team="t", admin=True, user="u")

        assert len(_sets(executed)) == 1
        sql, params = _sets(executed)[0]
        for guc in (
            "amfs.current_account_id",
            "amfs.current_team_id",
            "amfs.is_account_admin",
            "amfs.current_user_id",
        ):
            assert f"set_config('{guc}'" in sql
        assert params == ("a", "t", "true", "u")


class TestAPooledConnectionCarriesNothingOver:
    def test_no_user_still_writes_an_empty_string(self, _make_conn):
        """Skipping the set when there is no user is what would make this
        dangerous: the connection keeps the last borrower's identity, and the
        permissive room policies would grant their access to whoever holds
        the connection next."""
        inner_ctx, executed = _make_conn

        _checkout(inner_ctx, account="acct-2", user=None)

        sql, params = _sets(executed)[0]
        assert "set_config('amfs.current_user_id'" in sql
        assert params[3] == "", "must clear, not leave the previous value"

    def test_an_unauthenticated_checkout_clears_everything(self, _make_conn):
        inner_ctx, executed = _make_conn

        _checkout(inner_ctx, account=None, team=None, admin=False, user=None)

        _sql, params = _sets(executed)[0]
        # NULLIF in the policies turns each empty string into NULL, which
        # matches no rows.
        assert params == ("", "", "false", "")


class TestTheTenantIsScopedToATransaction:
    """The pooler-safe property, and the three ways it is easy to lose."""

    def test_the_statement_is_the_transaction_local_form(self, _make_conn):
        from amfs_postgres import tenant_gucs

        inner_ctx, executed = _make_conn

        _checkout(inner_ctx, account="acct-1", user="u")

        sql, _params = _sets(executed)[0]
        assert sql == tenant_gucs.SET_TENANT_GUCS_LOCAL
        assert sql != tenant_gucs.SET_TENANT_GUCS_SESSION

    def test_a_transaction_is_opened_before_anything_is_set(self, _make_conn):
        """Order is the whole thing. set_config(..., true) applied outside a
        transaction is discarded before the next statement, and because the
        pools run autocommit that is exactly what would happen if the BEGIN
        came second — silently, with every read returning nothing."""
        inner_ctx, executed = _make_conn

        _checkout(inner_ctx)

        kinds = [sql if sql in ("BEGIN", "COMMIT") else "SET" for sql, _ in executed]
        assert kinds[0] == "BEGIN", f"expected BEGIN first, got {kinds}"
        assert kinds.index("BEGIN") < kinds.index("SET")

    def test_the_transaction_commits_and_the_connection_goes_back(self, _make_conn):
        inner_ctx, executed = _make_conn

        wrapper, _conn = _checkout(inner_ctx)
        wrapper.__exit__(None, None, None)

        assert [sql for sql, _ in executed if sql in ("BEGIN", "COMMIT")] == [
            "BEGIN",
            "COMMIT",
        ]
        assert inner_ctx.__exit__.called, "the checkout must be returned to the pool"

    def test_an_error_rolls_the_transaction_back(self, _make_conn):
        """Postgres would discard the settings either way. What matters is that
        the caller's failed work does not commit on the way out."""
        inner_ctx, executed = _make_conn

        wrapper, _conn = _checkout(inner_ctx)
        wrapper.__exit__(ValueError, ValueError("boom"), None)

        assert ("ROLLBACK", None) in executed
        assert inner_ctx.__exit__.called

    def test_it_refuses_to_set_anything_it_cannot_scope(self, _make_conn):
        """If the transaction did not actually open, the local settings would be
        thrown away and every read would quietly return zero rows. Raising is
        the only outcome that is not mistaken for an empty account."""
        inner_ctx, executed = _make_conn
        inner_ctx.__enter__.return_value.info.transaction_status.name = "IDLE"

        with pytest.raises(RuntimeError, match="no open transaction"):
            _checkout(inner_ctx)

        assert _sets(executed) == [], "must not write a tenant it cannot scope"

    def test_a_failure_while_setting_up_still_returns_the_connection(self, _make_conn):
        """A leaked checkout on this path would drain the pool one failed
        request at a time, which is a worse outage than the error itself."""
        inner_ctx, _executed = _make_conn
        inner_ctx.__enter__.return_value.info.transaction_status.name = "IDLE"

        with pytest.raises(RuntimeError):
            _checkout(inner_ctx)

        assert inner_ctx.__exit__.called, "checkout leaked on the failure path"


class TestTheMaintenancePath:
    """DDL and the backfills, which cannot run inside a caller's transaction.

    DDL because one transaction around schema.sql would hold every ACCESS
    EXCLUSIVE lock it takes until the last statement finished; the backfills
    because they call an embedder between statements and issue more SQL from
    inside an except handler. Both get the four blanked instead of set.
    """

    def test_no_transaction_is_opened(self, _make_conn):
        inner_ctx, executed = _make_conn

        _checkout(inner_ctx, transactional=False)

        assert [sql for sql, _ in executed if sql in ("BEGIN", "COMMIT")] == []

    def test_all_four_are_blanked_even_with_a_tenant_in_context(self, _make_conn):
        """The point of blanking rather than skipping: the connection cannot
        inherit the previous borrower's tenant, and cannot carry one of its own
        into a path that has no business reading tenant rows. Under FORCE ROW
        LEVEL SECURITY empty reads nothing, so this fails closed."""
        inner_ctx, executed = _make_conn

        _checkout(
            inner_ctx,
            account="acct-7",
            team="team-1",
            admin=True,
            user="user-9",
            transactional=False,
        )

        sql, params = _sets(executed)[0]
        assert params == ("", "", "false", "")
        assert "acct-7" not in params and "user-9" not in params
        assert len(params) == 4

    def test_the_blanking_statement_covers_every_named_guc(self, _make_conn):
        from amfs_postgres import tenant_gucs

        inner_ctx, executed = _make_conn

        _checkout(inner_ctx, transactional=False)

        sql, _params = _sets(executed)[0]
        for name in tenant_gucs.TENANT_GUC_NAMES:
            assert f"set_config('{name}'" in sql


class TestNoTenantValueIsEverSessionScoped:
    """The invariant the whole change exists to establish, asserted directly.

    Both scopes still exist as SQL, and the session-scoped one is still used —
    but only ever with empty values, to clear. If a tenant value is ever paired
    with the session-scoped statement again, the leak is back, so this checks
    the pairing rather than trusting the call sites to stay right.
    """

    @pytest.mark.parametrize("transactional", [True, False])
    def test_session_scoped_writes_carry_no_tenant(self, _make_conn, transactional):
        from amfs_postgres import tenant_gucs

        inner_ctx, executed = _make_conn

        _checkout(
            inner_ctx,
            account="acct-1",
            team="team-1",
            admin=True,
            user="user-1",
            transactional=transactional,
        )

        for sql, params in _sets(executed):
            if sql == tenant_gucs.SET_TENANT_GUCS_SESSION:
                assert params == tenant_gucs.CLEARED_TENANT_GUC_VALUES, (
                    "a session-scoped write with a real tenant in it is the "
                    "cross-tenant leak this change removed"
                )

    def test_the_scope_argument_has_no_default(self):
        """It used to default to the unsafe scope, which is the bug in one
        keyword: session scope was what a caller got for not thinking about
        it. Every caller now has to say which it means."""
        import inspect

        from amfs_postgres import tenant_gucs

        for fn in (tenant_gucs.set_tenant_gucs, tenant_gucs.aset_tenant_gucs):
            param = inspect.signature(fn).parameters["local"]
            assert param.default is inspect.Parameter.empty, (
                f"{fn.__name__}: 'local' must stay required"
            )
