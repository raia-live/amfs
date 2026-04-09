"""Per-request tenant account id for Postgres RLS (thread-local for HTTP server workers)."""

from __future__ import annotations

import threading

_tls = threading.local()


def set_tls_tenant_account_id(account_id: str | None) -> None:
    """Called from HTTP middleware before handling a request."""
    _tls.account_id = account_id


def clear_tls_tenant_account_id() -> None:
    """Called from HTTP middleware after a request."""
    if hasattr(_tls, "account_id"):
        delattr(_tls, "account_id")


def get_request_tenant_account_id() -> str | None:
    """Read by PostgresAdapter when checking out a connection."""
    return getattr(_tls, "account_id", None)
