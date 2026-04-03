"""API key authentication for the AMFS HTTP server."""
from __future__ import annotations

import logging
import os
import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-AMFS-API-Key", auto_error=False)


def get_api_keys() -> set[str]:
    raw = os.environ.get("AMFS_API_KEYS", "")
    if not raw:
        return set()
    return {k.strip() for k in raw.split(",") if k.strip()}


def _check_db_key(api_key: str) -> bool:
    """Check if an API key exists in amfs_api_keys table (active keys only)."""
    try:
        import psycopg
        from psycopg.rows import dict_row

        dsn = os.environ.get("AMFS_POSTGRES_DSN")
        if not dsn:
            return False

        with psycopg.connect(dsn, row_factory=dict_row) as conn:
            row = conn.execute(
                "SELECT id FROM amfs_api_keys WHERE key_hash = %s AND active = TRUE LIMIT 1",
                (api_key,),
            ).fetchone()
            return row is not None
    except Exception:
        logger.debug("DB API key check failed (table may not exist)", exc_info=True)
        return False


def generate_api_key() -> str:
    return f"amfs_{secrets.token_urlsafe(32)}"


async def verify_api_key(api_key: str | None = Security(API_KEY_HEADER)) -> str | None:
    env_keys = get_api_keys()
    if not env_keys:
        if not os.environ.get("AMFS_POSTGRES_DSN"):
            return None
        if api_key and _check_db_key(api_key):
            return api_key
        return None

    if api_key and api_key in env_keys:
        return api_key

    if api_key and _check_db_key(api_key):
        return api_key

    if api_key is None and not env_keys:
        return None

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key",
    )
