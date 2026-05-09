"""API key authentication for the AMFS HTTP server."""
from __future__ import annotations

import hashlib
import logging
import os
import secrets
import threading
import time

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-AMFS-API-Key", auto_error=False)

_AUTH_CACHE_TTL = 120.0
_auth_cache: dict[str, float] = {}
_auth_cache_lock = threading.Lock()


def get_api_keys() -> set[str]:
    raw = os.environ.get("AMFS_API_KEYS", "")
    if not raw:
        return set()
    return {k.strip() for k in raw.split(",") if k.strip()}


def _check_db_key(api_key: str) -> bool:
    """Check if an API key exists in amfs_api_keys table (active keys only).

    Keys are stored as SHA-256 hex digests, so we hash the incoming raw key
    before comparing.  Results are cached in-process for ``_AUTH_CACHE_TTL``
    seconds to avoid opening a new DB connection on every request.  Only
    positive results are cached; revoked/invalid keys always hit the DB.

    Uses SECURITY DEFINER functions to bypass RLS — this connection does not
    set amfs.current_account_id, so direct table queries would return nothing
    when FORCE ROW LEVEL SECURITY is enabled.
    """
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    now = time.monotonic()

    with _auth_cache_lock:
        cached_at = _auth_cache.get(key_hash)
        if cached_at is not None and (now - cached_at) < _AUTH_CACHE_TTL:
            return True

    try:
        import psycopg
        from psycopg.rows import dict_row

        dsn = os.environ.get("AMFS_POSTGRES_DSN")
        if not dsn:
            return False

        with psycopg.connect(dsn, row_factory=dict_row) as conn:
            row = conn.execute(
                "SELECT * FROM amfs_authenticate_api_key(%s)",
                (key_hash,),
            ).fetchone()
            if row:
                conn.execute(
                    "SELECT amfs_touch_api_key_usage(%s)",
                    (row["key_id"],),
                )
            if row is not None:
                with _auth_cache_lock:
                    _auth_cache[key_hash] = now
                return True
            return False
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
