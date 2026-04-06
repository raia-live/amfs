"""API key authentication and branch resolution for the AMFS HTTP server."""
from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass, field

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-AMFS-API-Key", auto_error=False)
BRANCH_HEADER = "X-AMFS-Branch"


def get_api_keys() -> set[str]:
    raw = os.environ.get("AMFS_API_KEYS", "")
    if not raw:
        return set()
    return {k.strip() for k in raw.split(",") if k.strip()}


@dataclass
class DBKeyInfo:
    """Metadata returned from a DB API key lookup."""
    id: str
    namespace: str
    default_branch: str | None = None


def _check_db_key(api_key: str) -> DBKeyInfo | None:
    """Check if an API key exists in amfs_api_keys table and return its metadata."""
    try:
        import psycopg
        from psycopg.rows import dict_row

        dsn = os.environ.get("AMFS_POSTGRES_DSN")
        if not dsn:
            return None

        with psycopg.connect(dsn, row_factory=dict_row) as conn:
            row = conn.execute(
                "SELECT id, namespace, default_branch FROM amfs_api_keys WHERE key_hash = %s AND active = TRUE LIMIT 1",
                (api_key,),
            ).fetchone()
            if row is None:
                return None
            return DBKeyInfo(
                id=str(row["id"]),
                namespace=row["namespace"],
                default_branch=row.get("default_branch"),
            )
    except Exception:
        logger.debug("DB API key check failed (table may not exist)", exc_info=True)
        return None


def generate_api_key() -> str:
    return f"amfs_{secrets.token_urlsafe(32)}"


async def verify_api_key(api_key: str | None = Security(API_KEY_HEADER)) -> str | None:
    env_keys = get_api_keys()
    if not env_keys:
        if not os.environ.get("AMFS_POSTGRES_DSN"):
            return None
        if api_key and _check_db_key(api_key) is not None:
            return api_key
        return None

    if api_key and api_key in env_keys:
        return api_key

    if api_key and _check_db_key(api_key) is not None:
        return api_key

    if api_key is None and not env_keys:
        return None

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key",
    )


@dataclass
class BranchContext:
    """Resolved branch context for an authenticated request."""

    branch: str = "main"
    api_key_id: str | None = None
    namespace: str = "default"
    is_owner: bool = True
    permission: str = "read_write"


async def resolve_branch_context(
    request: Request,
    api_key: str | None = Security(API_KEY_HEADER),
) -> BranchContext:
    """Resolve which branch this request targets and enforce access.

    Resolution order:
    1. X-AMFS-Branch header (explicit override)
    2. API key's default_branch (if DB key)
    3. Fall back to 'main'

    Access rules:
    - main: only owner's keys (same namespace) or env keys
    - branches: owner always allowed; external keys need a grant
    """
    env_keys = get_api_keys()

    if api_key and api_key in env_keys:
        header_branch = request.headers.get(BRANCH_HEADER)
        return BranchContext(
            branch=header_branch or "main",
            namespace=os.environ.get("AMFS_NAMESPACE", "default"),
            is_owner=True,
            permission="read_write",
        )

    db_key = _check_db_key(api_key) if api_key else None

    if db_key is None:
        if not env_keys and not os.environ.get("AMFS_POSTGRES_DSN"):
            header_branch = request.headers.get(BRANCH_HEADER)
            return BranchContext(branch=header_branch or "main")
        if env_keys or os.environ.get("AMFS_POSTGRES_DSN"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
            )
        return BranchContext()

    header_branch = request.headers.get(BRANCH_HEADER)
    resolved = header_branch or db_key.default_branch or "main"

    return BranchContext(
        branch=resolved,
        api_key_id=db_key.id,
        namespace=db_key.namespace,
        is_owner=True,
        permission="read_write",
    )
