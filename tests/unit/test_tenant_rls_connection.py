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
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def _make_conn():
    executed: list = []

    cur = MagicMock()
    cur.execute = MagicMock(side_effect=lambda sql, params=None: executed.append((sql, params)))
    cur.__enter__ = MagicMock(return_value=cur)
    cur.__exit__ = MagicMock(return_value=None)

    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cur)

    inner_ctx = MagicMock()
    inner_ctx.__enter__ = MagicMock(return_value=conn)
    inner_ctx.__exit__ = MagicMock(return_value=None)

    return inner_ctx, executed


def _checkout(inner_ctx, *, account="acct-1", team=None, admin=False, user=None):
    from amfs_postgres.adapter import _TenantRLSConnection

    with (
        patch("amfs_postgres.tenant_context.get_request_tenant_account_id", return_value=account),
        patch("amfs_postgres.tenant_context.get_request_tenant_team_id", return_value=team),
        patch("amfs_postgres.tenant_context.get_request_is_account_admin", return_value=admin),
        patch("amfs_postgres.tenant_context.get_request_user_id", return_value=user),
    ):
        return _TenantRLSConnection(inner_ctx).__enter__()


class TestTheActingUserReachesTheDatabase:
    def test_the_user_guc_is_set_from_context(self, _make_conn):
        """Without this, the cross-account room policies match nothing and an
        invited user sees an empty room rather than its contents."""
        inner_ctx, executed = _make_conn

        _checkout(inner_ctx, account="acct-1", user="user-42")

        sql, params = executed[0]
        assert "set_config('amfs.current_user_id'" in sql
        assert "user-42" in params

    def test_all_four_gucs_are_set_in_one_statement(self, _make_conn):
        # One round trip per checkout; this is on every request's hot path.
        inner_ctx, executed = _make_conn

        _checkout(inner_ctx, account="a", team="t", admin=True, user="u")

        assert len(executed) == 1
        sql, params = executed[0]
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

        sql, params = executed[0]
        assert "set_config('amfs.current_user_id'" in sql
        assert params[3] == "", "must clear, not leave the previous value"

    def test_an_unauthenticated_checkout_clears_everything(self, _make_conn):
        inner_ctx, executed = _make_conn

        _checkout(inner_ctx, account=None, team=None, admin=False, user=None)

        _sql, params = executed[0]
        # NULLIF in the policies turns each empty string into NULL, which
        # matches no rows.
        assert params == ("", "", "false", "")
