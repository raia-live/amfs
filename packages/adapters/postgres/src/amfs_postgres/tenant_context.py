"""Per-request tenant context for Postgres RLS.

Uses ``contextvars.ContextVar`` so the tenant is properly isolated in
both threaded (sync endpoints) and async (event-loop) contexts.
Starlette/FastAPI creates a copy of the context for each request, so
concurrent requests never interfere with each other's tenant.

Stores account_id, team_id, and is_account_admin for two-layer tenancy:
  - account_id  -> amfs.current_account_id  (outer boundary)
  - team_id     -> amfs.current_team_id     (inner boundary)
  - is_admin    -> amfs.is_account_admin    (admin bypass)

and the acting user:
  - user_id     -> amfs.current_user_id     (cross-account grants)

The first three narrow what a request can see. user_id is the one that can
widen it: a user invited to a room in someone else's account reaches it
through permissive policies keyed on this value, and on nothing else. It is
therefore only ever set from an authenticated identity the server resolved
itself, never from a client-supplied header taken at face value.
"""

from __future__ import annotations

from contextvars import ContextVar

_account_id_var: ContextVar[str | None] = ContextVar("amfs_account_id", default=None)
_team_id_var: ContextVar[str | None] = ContextVar("amfs_team_id", default=None)
_is_admin_var: ContextVar[bool] = ContextVar("amfs_is_admin", default=False)
_user_id_var: ContextVar[str | None] = ContextVar("amfs_user_id", default=None)


# -- account_id --

def set_tls_tenant_account_id(account_id: str | None) -> None:
    """Called from HTTP middleware before handling a request."""
    _account_id_var.set(account_id or None)


def clear_tls_tenant_account_id() -> None:
    """Called from HTTP middleware after a request."""
    _account_id_var.set(None)


def get_request_tenant_account_id() -> str | None:
    """Read by PostgresAdapter when checking out a connection."""
    return _account_id_var.get()


# -- team_id --

def set_tls_tenant_team_id(team_id: str | None) -> None:
    _team_id_var.set(team_id or None)


def clear_tls_tenant_team_id() -> None:
    _team_id_var.set(None)


def get_request_tenant_team_id() -> str | None:
    return _team_id_var.get()


# -- is_account_admin --

def set_tls_is_account_admin(is_admin: bool) -> None:
    _is_admin_var.set(is_admin)


def clear_tls_is_account_admin() -> None:
    _is_admin_var.set(False)


def get_request_is_account_admin() -> bool:
    return _is_admin_var.get()


# -- user_id --

def set_tls_user_id(user_id: str | None) -> None:
    """Called from HTTP middleware once the acting user is authenticated.

    Only from a resolved identity — an API key's owner, or a dashboard header
    whose proxy secret validated. Unlike the account and team vars, this one
    grants access rather than restricting it.
    """
    _user_id_var.set(user_id or None)


def clear_tls_user_id() -> None:
    _user_id_var.set(None)


def get_request_user_id() -> str | None:
    return _user_id_var.get()
