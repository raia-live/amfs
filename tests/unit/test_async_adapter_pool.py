"""What the async adapter's pool is configured with before it opens.

The async adapter serves the hot-path REST endpoints, so it holds the
statements a caller is actually waiting on. It went without a statement
ceiling while the sync adapter had one, which left the request path that
motivated the ceiling as the one place it did not apply.

The pool is built with open=False, so nothing here touches a database.
"""

from __future__ import annotations

from amfs_postgres.adapter import _DEFAULT_POOL_MAX
from amfs_postgres.async_adapter import AsyncPostgresAdapter

DSN = "postgresql://nobody@127.0.0.1:1/nothing"


def _pool(adapter: AsyncPostgresAdapter):
    return adapter._pool._inner


class TestTheStatementCeiling:
    def test_it_reaches_the_request_path(self, monkeypatch):
        monkeypatch.setenv("AMFS_POSTGRES_STATEMENT_TIMEOUT", "300s")
        kwargs = _pool(AsyncPostgresAdapter(dsn=DSN)).kwargs
        assert kwargs["options"] == "-c statement_timeout=300s"

    def test_nothing_is_passed_when_none_is_configured(self, monkeypatch):
        monkeypatch.delenv("AMFS_POSTGRES_STATEMENT_TIMEOUT", raising=False)
        kwargs = _pool(AsyncPostgresAdapter(dsn=DSN)).kwargs
        assert "options" not in kwargs

    def test_the_row_factory_survives_the_ceiling(self, monkeypatch):
        # The options string is added to the connect kwargs rather than
        # replacing them; dropping the row factory would break every read.
        monkeypatch.setenv("AMFS_POSTGRES_STATEMENT_TIMEOUT", "300s")
        kwargs = _pool(AsyncPostgresAdapter(dsn=DSN)).kwargs
        assert kwargs["autocommit"] is True
        assert kwargs["row_factory"] is not None


class TestTheTenantResetOnReturn:
    """The sync pool blanks the four tenant settings when a connection goes
    back. This pool serves the request path, where tenants change fastest, so
    going without it was the gap that mattered more of the two."""

    def test_the_pool_is_given_a_reset_callback(self):
        from amfs_postgres.tenant_gucs import areset_tenant_gucs

        assert _pool(AsyncPostgresAdapter(dsn=DSN))._reset is areset_tenant_gucs

    def test_the_callback_is_awaitable(self):
        # psycopg_pool awaits the async pool's reset, so handing it the sync
        # helper would raise on every connection return rather than at startup.
        import inspect

        from amfs_postgres.tenant_gucs import areset_tenant_gucs

        assert inspect.iscoroutinefunction(areset_tenant_gucs)


class TestThePoolSize:
    def test_it_follows_the_environment(self, monkeypatch):
        monkeypatch.setenv("AMFS_POSTGRES_POOL_MAX", "6")
        assert _pool(AsyncPostgresAdapter(dsn=DSN)).max_size == 6

    def test_an_argument_still_wins(self, monkeypatch):
        monkeypatch.setenv("AMFS_POSTGRES_POOL_MAX", "6")
        assert _pool(AsyncPostgresAdapter(dsn=DSN, max_pool_size=2)).max_size == 2

    def test_the_default_is_unchanged(self, monkeypatch):
        monkeypatch.delenv("AMFS_POSTGRES_POOL_MAX", raising=False)
        monkeypatch.delenv("AMFS_POSTGRES_POOL_MIN", raising=False)
        assert _pool(AsyncPostgresAdapter(dsn=DSN)).max_size == _DEFAULT_POOL_MAX
