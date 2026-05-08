"""Apply optional dashboard tenant headers for Postgres RLS."""

from __future__ import annotations

import logging
import os
from uuid import UUID

from starlette.requests import Request

logger = logging.getLogger(__name__)


def apply_tenant_headers_from_request(request: Request) -> bool:
    """If proxy secret + account id headers match env, set thread-local tenant for DB RLS.

    When the Pro scope-enforcement middleware (amfs_tenant) has already
    resolved a TenantContext it also sets the thread-local RLS variables
    to the API key's account — which is the account that actually owns
    the stored entries.  Overriding those values with the dashboard
    user's account/team headers would cause RLS to hide all entries
    whenever the service API key's account differs from the logged-in
    user's account (or when team IDs don't match).

    So: skip entirely when tenant_ctx is already present.

    Returns True if this function set TLS variables (caller must clear them).
    """
    if getattr(request.state, "tenant_ctx", None) is not None:
        return False

    try:
        from amfs_postgres.tenant_context import (
            set_tls_tenant_account_id,
            set_tls_tenant_team_id,
            set_tls_is_account_admin,
        )
    except ImportError:
        return False

    secret = os.environ.get("AMFS_DASHBOARD_PROXY_SECRET", "")
    if not secret:
        return False
    if request.headers.get("X-AMFS-Dashboard-Secret") != secret:
        return False
    raw = request.headers.get("X-AMFS-Dashboard-Account-Id")
    if not raw:
        return False
    try:
        UUID(raw)
    except ValueError:
        logger.warning("Invalid X-AMFS-Dashboard-Account-Id header")
        return False
    set_tls_tenant_account_id(raw)
    request.state.account_id = UUID(raw)

    # Never propagate team_id from the dashboard user's session.
    # The logged-in user's team can differ from the team that owns the
    # agents and entries in the shared account (e.g. invited users).
    # Setting a mismatched team_id causes RLS to hide all data.
    # Per-user scoping is handled by UserVisibilityFilter, not team RLS.
    set_tls_tenant_team_id(None)
    set_tls_is_account_admin(False)

    user_id_raw = request.headers.get("X-AMFS-Dashboard-User-Id")
    if user_id_raw:
        try:
            request.state.user_id = UUID(user_id_raw)
        except ValueError:
            logger.warning("Invalid X-AMFS-Dashboard-User-Id header")

    return True


def clear_tenant_headers() -> None:
    try:
        from amfs_postgres.tenant_context import (
            clear_tls_tenant_account_id,
            clear_tls_tenant_team_id,
            clear_tls_is_account_admin,
        )
    except ImportError:
        return
    clear_tls_tenant_account_id()
    clear_tls_tenant_team_id()
    clear_tls_is_account_admin()
