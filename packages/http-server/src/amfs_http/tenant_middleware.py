"""Apply optional dashboard tenant headers for Postgres RLS."""

from __future__ import annotations

import logging
import os
from uuid import UUID

from starlette.requests import Request

logger = logging.getLogger(__name__)


def apply_tenant_headers_from_request(request: Request) -> None:
    """If proxy secret + account id headers match env, set thread-local tenant for DB RLS.

    When valid dashboard proxy headers are present, they ALWAYS take
    precedence — even if a prior middleware (e.g. Pro scope) already set
    the thread-local tenant.  The dashboard proxy carries the actual
    end-user's account ID, which must override whatever account the
    service API key resolved to.
    """
    try:
        from amfs_postgres.tenant_context import (
            set_tls_tenant_account_id,
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

    user_id_raw = request.headers.get("X-AMFS-Dashboard-User-Id")
    if user_id_raw:
        try:
            request.state.user_id = UUID(user_id_raw)
        except ValueError:
            logger.warning("Invalid X-AMFS-Dashboard-User-Id header")


def clear_tenant_headers() -> None:
    try:
        from amfs_postgres.tenant_context import clear_tls_tenant_account_id
    except ImportError:
        return
    clear_tls_tenant_account_id()
