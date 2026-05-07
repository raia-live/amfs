"""Apply optional dashboard tenant headers for Postgres RLS."""

from __future__ import annotations

import logging
import os
from uuid import UUID

from starlette.requests import Request

logger = logging.getLogger(__name__)


def apply_tenant_headers_from_request(request: Request) -> None:
    """If proxy secret + account id headers match env, set thread-local tenant for DB RLS.

    When the Pro scope-enforcement middleware (amfs_tenant) has already
    resolved a TenantContext it also sets the thread-local RLS variables
    to the API key's account — which is the account that actually owns
    the stored entries.  Overriding those values with the dashboard
    user's account/team headers would cause RLS to hide all entries
    whenever the service API key's account differs from the logged-in
    user's account (or when team IDs don't match).

    So: skip entirely when tenant_ctx is already present.
    """
    if getattr(request.state, "tenant_ctx", None) is not None:
        return

    try:
        from amfs_postgres.tenant_context import (
            set_tls_tenant_account_id,
            set_tls_tenant_team_id,
            set_tls_is_account_admin,
        )
    except ImportError:
        return

    secret = os.environ.get("AMFS_DASHBOARD_PROXY_SECRET", "")
    if not secret:
        return
    if request.headers.get("X-AMFS-Dashboard-Secret") != secret:
        return
    raw = request.headers.get("X-AMFS-Dashboard-Account-Id")
    if not raw:
        return
    try:
        UUID(raw)
    except ValueError:
        logger.warning("Invalid X-AMFS-Dashboard-Account-Id header")
        return
    set_tls_tenant_account_id(raw)
    request.state.account_id = UUID(raw)

    team_raw = request.headers.get("X-AMFS-Dashboard-Team-Id")
    if team_raw:
        try:
            UUID(team_raw)
            set_tls_tenant_team_id(team_raw)
        except ValueError:
            logger.warning("Invalid X-AMFS-Dashboard-Team-Id header")

    is_admin = request.headers.get("X-AMFS-Dashboard-Is-Admin") == "true"
    set_tls_is_account_admin(is_admin)

    user_id_raw = request.headers.get("X-AMFS-Dashboard-User-Id")
    if user_id_raw:
        try:
            request.state.user_id = UUID(user_id_raw)
        except ValueError:
            logger.warning("Invalid X-AMFS-Dashboard-User-Id header")


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
