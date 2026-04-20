"""Per-request tenant context for Postgres RLS (thread-local for HTTP server workers).

Stores account_id, team_id, and is_account_admin for two-layer tenancy:
  - account_id  -> amfs.current_account_id  (outer boundary)
  - team_id     -> amfs.current_team_id     (inner boundary)
  - is_admin    -> amfs.is_account_admin    (admin bypass)
"""

from __future__ import annotations

import threading

_tls = threading.local()


# -- account_id --

def set_tls_tenant_account_id(account_id: str | None) -> None:
    """Called from HTTP middleware before handling a request."""
    _tls.account_id = account_id or None


def clear_tls_tenant_account_id() -> None:
    """Called from HTTP middleware after a request."""
    if hasattr(_tls, "account_id"):
        delattr(_tls, "account_id")


def get_request_tenant_account_id() -> str | None:
    """Read by PostgresAdapter when checking out a connection."""
    return getattr(_tls, "account_id", None)


# -- team_id --

def set_tls_tenant_team_id(team_id: str | None) -> None:
    _tls.team_id = team_id or None


def clear_tls_tenant_team_id() -> None:
    if hasattr(_tls, "team_id"):
        delattr(_tls, "team_id")


def get_request_tenant_team_id() -> str | None:
    return getattr(_tls, "team_id", None)


# -- is_account_admin --

def set_tls_is_account_admin(is_admin: bool) -> None:
    _tls.is_account_admin = is_admin


def clear_tls_is_account_admin() -> None:
    if hasattr(_tls, "is_account_admin"):
        delattr(_tls, "is_account_admin")


def get_request_is_account_admin() -> bool:
    return getattr(_tls, "is_account_admin", False)
