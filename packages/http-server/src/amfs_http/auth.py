"""API key authentication for the AMFS HTTP server."""
from __future__ import annotations

import hashlib
import logging
import os
import secrets
import threading
import time

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-AMFS-API-Key", auto_error=False)

# Also accept the key as `Authorization: Bearer <amfs_key>`. Credential-brokering
# gateways (e.g. Fly.io Sprites connectors) store the key once and inject it as a
# bearer token, so they can't set our custom X-AMFS-API-Key header. auto_error is
# off so a missing/other-scheme Authorization header falls through to the API-key
# header rather than 403-ing before verify_api_key runs.
BEARER_SCHEME = HTTPBearer(auto_error=False)

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
    # SHA-256 (not bcrypt/argon2) is intentional and correct here: API keys are
    # 256-bit random tokens (generate_api_key -> secrets.token_urlsafe(32)), not
    # human passwords, so there is nothing to brute-force and a slow password
    # hash buys no security. A fast *deterministic* digest is also required — the
    # lookup below (amfs_authenticate_api_key) is an indexed equality match on
    # key_hash, which salted password hashers cannot support. This mirrors how
    # GitHub/Stripe/AWS hash API tokens. CodeQL py/weak-sensitive-data-hashing
    # flags this as password hashing; it is a false positive for random tokens.
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


async def verify_api_key(
    api_key: str | None = Security(API_KEY_HEADER),
    bearer: HTTPAuthorizationCredentials | None = Security(BEARER_SCHEME),
) -> str | None:
    # X-AMFS-API-Key is canonical; fall back to the bearer token so gateways that
    # can only inject `Authorization: Bearer <key>` authenticate the same way.
    if not api_key and bearer is not None:
        api_key = bearer.credentials

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
