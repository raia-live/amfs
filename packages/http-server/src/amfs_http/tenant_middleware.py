"""Apply optional dashboard tenant headers for Postgres RLS."""

from __future__ import annotations

import logging
import os
from uuid import UUID

from starlette.requests import Request

logger = logging.getLogger(__name__)


def apply_tenant_headers_from_request(request: Request) -> None:
    """If proxy secret + account id headers match env, set thread-local tenant for DB RLS."""
    try:
        from amfs_postgres.tenant_context import (
            get_request_tenant_account_id,
            set_tls_tenant_account_id,
        )
    except ImportError:
        return

    if get_request_tenant_account_id() is not None:
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


def clear_tenant_headers() -> None:
    try:
        from amfs_postgres.tenant_context import clear_tls_tenant_account_id
    except ImportError:
        return
    clear_tls_tenant_account_id()
