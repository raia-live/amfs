"""API key authentication for the AMFS HTTP server."""
from __future__ import annotations

import os
import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-AMFS-API-Key", auto_error=False)


def get_api_keys() -> set[str]:
    raw = os.environ.get("AMFS_API_KEYS", "")
    if not raw:
        return set()
    return {k.strip() for k in raw.split(",") if k.strip()}


def generate_api_key() -> str:
    return f"amfs_{secrets.token_urlsafe(32)}"


async def verify_api_key(api_key: str | None = Security(API_KEY_HEADER)) -> str | None:
    valid_keys = get_api_keys()
    if not valid_keys:
        return None
    if api_key is None or api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return api_key
