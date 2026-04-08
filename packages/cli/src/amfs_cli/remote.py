"""Shared HTTP client helper for remote AMFS operations."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console

console = Console(stderr=True)

_CRED_DIR = Path.home() / ".config" / "amfs"
_CRED_FILE = _CRED_DIR / "credentials.json"


def _load_credentials() -> dict[str, str]:
    if _CRED_FILE.exists():
        return json.loads(_CRED_FILE.read_text(encoding="utf-8"))
    return {}


def save_credentials(url: str, api_key: str) -> Path:
    _CRED_DIR.mkdir(parents=True, exist_ok=True)
    _CRED_FILE.write_text(
        json.dumps({"url": url, "api_key": api_key}, indent=2),
        encoding="utf-8",
    )
    _CRED_FILE.chmod(0o600)
    return _CRED_FILE


def get_client() -> httpx.Client:
    """Build an httpx client from env vars or stored credentials."""
    url = os.environ.get("AMFS_HTTP_URL")
    api_key = os.environ.get("AMFS_API_KEY")

    if not url:
        creds = _load_credentials()
        url = creds.get("url")
        api_key = api_key or creds.get("api_key")

    if not url:
        console.print(
            "[red]No AMFS server configured.[/red]\n"
            "Run [bold]amfs login[/bold] or set AMFS_HTTP_URL.",
        )
        sys.exit(1)

    headers: dict[str, str] = {}
    if api_key:
        headers["X-AMFS-API-Key"] = api_key

    return httpx.Client(base_url=url.rstrip("/"), headers=headers, timeout=30.0)


def has_remote_config() -> bool:
    """Return True if remote credentials are available (env vars or stored)."""
    if os.environ.get("AMFS_HTTP_URL"):
        return True
    return bool(_load_credentials().get("url"))


def api_get(path: str, params: dict[str, Any] | None = None) -> Any:
    with get_client() as client:
        resp = client.get(path, params=params)
        resp.raise_for_status()
        return resp.json()


def api_post(path: str, payload: dict[str, Any]) -> Any:
    with get_client() as client:
        resp = client.post(path, json=payload)
        resp.raise_for_status()
        return resp.json()
