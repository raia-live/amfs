"""Tests for AsyncPostgresAdapter RLS context and basic wiring."""

from __future__ import annotations

import asyncio
from contextvars import copy_context
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestAsyncTenantRLSConnection:
    """Verify _AsyncTenantRLSConnection sets the correct GUCs."""

    @pytest.fixture
    def _make_conn(self):
        """Build a mock async connection with a cursor that records SQL."""
        executed = []

        mock_cur = AsyncMock()
        mock_cur.execute = AsyncMock(side_effect=lambda sql, params=None: executed.append((sql, params)))
        mock_cur.__aenter__ = AsyncMock(return_value=mock_cur)
        mock_cur.__aexit__ = AsyncMock(return_value=None)

        mock_conn = AsyncMock()
        mock_conn.cursor = MagicMock(return_value=mock_cur)

        mock_inner_ctx = AsyncMock()
        mock_inner_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_inner_ctx.__aexit__ = AsyncMock(return_value=None)

        return mock_inner_ctx, executed

    async def test_tenant_present(self, _make_conn):
        from amfs_postgres.async_adapter import _AsyncTenantRLSConnection

        inner_ctx, executed = _make_conn

        with (
            patch("amfs_postgres.tenant_context.get_request_tenant_account_id", return_value="acct-123"),
            patch("amfs_postgres.tenant_context.get_request_tenant_team_id", return_value="team-456"),
            patch("amfs_postgres.tenant_context.get_request_is_account_admin", return_value=True),
            patch("amfs_postgres.tenant_context.get_request_user_id", return_value="user-789"),
        ):
            rls = _AsyncTenantRLSConnection(inner_ctx)
            conn = await rls.__aenter__()

        assert len(executed) == 1
        sql, params = executed[0]
        assert "set_config('amfs.current_account_id'" in sql
        assert "set_config('amfs.current_team_id'" in sql
        assert "set_config('amfs.is_account_admin'" in sql
        assert "set_config('amfs.current_user_id'" in sql
        assert params == ("acct-123", "team-456", "true", "user-789")

    async def test_no_tenant_sets_empty_strings(self, _make_conn):
        from amfs_postgres.async_adapter import _AsyncTenantRLSConnection

        inner_ctx, executed = _make_conn

        with (
            patch("amfs_postgres.tenant_context.get_request_tenant_account_id", return_value=None),
            patch("amfs_postgres.tenant_context.get_request_tenant_team_id", return_value=None),
            patch("amfs_postgres.tenant_context.get_request_is_account_admin", return_value=False),
            patch("amfs_postgres.tenant_context.get_request_user_id", return_value=None),
        ):
            rls = _AsyncTenantRLSConnection(inner_ctx)
            await rls.__aenter__()

        assert len(executed) == 1
        _, params = executed[0]
        assert params == ("", "", "false", "")

    async def test_admin_false(self, _make_conn):
        from amfs_postgres.async_adapter import _AsyncTenantRLSConnection

        inner_ctx, executed = _make_conn

        with (
            patch("amfs_postgres.tenant_context.get_request_tenant_account_id", return_value="acct-x"),
            patch("amfs_postgres.tenant_context.get_request_tenant_team_id", return_value="team-y"),
            patch("amfs_postgres.tenant_context.get_request_is_account_admin", return_value=False),
            patch("amfs_postgres.tenant_context.get_request_user_id", return_value=None),
        ):
            rls = _AsyncTenantRLSConnection(inner_ctx)
            await rls.__aenter__()

        _, params = executed[0]
        assert params == ("acct-x", "team-y", "false", "")

    async def test_a_pooled_connection_never_inherits_the_last_users_identity(
        self, _make_conn
    ):
        """The GUC that grants cross-account access must be reset, not skipped.

        Every other GUC here narrows what a request can see, so leaving a
        stale one behind fails closed. current_user_id is the opposite: it is
        what lets an invited user reach a room in someone else's account. A
        connection handed on with the previous borrower's user still set
        would grant the next request their room access.
        """
        from amfs_postgres.async_adapter import _AsyncTenantRLSConnection

        inner_ctx, executed = _make_conn

        with (
            patch("amfs_postgres.tenant_context.get_request_tenant_account_id", return_value="acct-x"),
            patch("amfs_postgres.tenant_context.get_request_tenant_team_id", return_value=None),
            patch("amfs_postgres.tenant_context.get_request_is_account_admin", return_value=False),
            patch("amfs_postgres.tenant_context.get_request_user_id", return_value=None),
        ):
            await _AsyncTenantRLSConnection(inner_ctx).__aenter__()

        sql, params = executed[0]
        assert "set_config('amfs.current_user_id'" in sql, (
            "current_user_id is left untouched on checkout, so it keeps "
            "whatever the previous borrower of this pooled connection set"
        )
        assert params[3] == ""

    async def test_aexit_delegates(self, _make_conn):
        from amfs_postgres.async_adapter import _AsyncTenantRLSConnection

        inner_ctx, _ = _make_conn

        with (
            patch("amfs_postgres.tenant_context.get_request_tenant_account_id", return_value=None),
            patch("amfs_postgres.tenant_context.get_request_tenant_team_id", return_value=None),
            patch("amfs_postgres.tenant_context.get_request_is_account_admin", return_value=False),
            patch("amfs_postgres.tenant_context.get_request_user_id", return_value=None),
        ):
            rls = _AsyncTenantRLSConnection(inner_ctx)
            await rls.__aenter__()
            await rls.__aexit__(None, None, None)

        inner_ctx.__aexit__.assert_awaited_once()


class TestAsyncTenantRLSPoolWrapper:
    """Verify _AsyncTenantRLSPoolWrapper returns wrapped connections."""

    def test_connection_returns_rls_wrapper(self):
        from amfs_postgres.async_adapter import (
            _AsyncTenantRLSConnection,
            _AsyncTenantRLSPoolWrapper,
        )

        mock_pool = MagicMock()
        mock_pool.connection = MagicMock(return_value="inner_ctx")

        wrapper = _AsyncTenantRLSPoolWrapper(mock_pool)
        result = wrapper.connection()

        assert isinstance(result, _AsyncTenantRLSConnection)

    def test_getattr_delegates(self):
        from amfs_postgres.async_adapter import _AsyncTenantRLSPoolWrapper

        mock_pool = MagicMock()
        mock_pool.some_attribute = "test_value"

        wrapper = _AsyncTenantRLSPoolWrapper(mock_pool)
        assert wrapper.some_attribute == "test_value"


class TestContextVarsPropagation:
    """Verify that asyncio.create_task inherits contextvars."""

    async def test_create_task_inherits_context(self):
        from amfs_postgres.tenant_context import (
            get_request_tenant_account_id,
            set_tls_tenant_account_id,
            clear_tls_tenant_account_id,
        )

        set_tls_tenant_account_id("propagation-test-acct")

        captured = {}

        async def _bg_task():
            captured["account_id"] = get_request_tenant_account_id()

        task = asyncio.create_task(_bg_task())
        await task

        assert captured["account_id"] == "propagation-test-acct"
        clear_tls_tenant_account_id()


class TestAsyncAdapterInit:
    """Verify AsyncPostgresAdapter constructor and pool config."""

    def test_constructor_sets_fields(self):
        from amfs_postgres.async_adapter import AsyncPostgresAdapter

        adapter = AsyncPostgresAdapter(
            dsn="postgresql://test:test@localhost/test",
            namespace="test-ns",
            min_pool_size=1,
            max_pool_size=5,
        )

        assert adapter._namespace == "test-ns"
        assert adapter._has_embedding_col is False
        assert adapter._has_search_tsv is False
        assert adapter._pool is not None
