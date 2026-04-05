"""Pro SaaS proxy — forwards Pro feature requests to the hosted Pro API.

When AMFS_PRO_URL and AMFS_PRO_API_KEY are configured, this module:
1. Mounts proxy routes for hot-context and anticipation endpoints
2. Provides an async event forwarder for the CortexWorker
3. Falls back gracefully when the Pro API is unreachable

Customer flow:
  Dashboard → OSS HTTP Server (proxy) → Pro SaaS API
  CortexWorker → Pro SaaS API (event forwarding)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

PRO_URL = os.environ.get("AMFS_PRO_URL", "").rstrip("/")
PRO_KEY = os.environ.get("AMFS_PRO_API_KEY", "")


def is_pro_configured() -> bool:
    return bool(PRO_URL and PRO_KEY)


def _pro_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {PRO_KEY}",
        "Content-Type": "application/json",
    }


# ─────────────────────────────────────────────────────────────────────
# Route proxying — mount onto the FastAPI app
# ─────────────────────────────────────────────────────────────────────


def mount_pro_proxy(app) -> None:
    """Mount proxy routes that forward Pro requests to the SaaS API."""
    if not is_pro_configured():
        return

    import httpx
    from fastapi import Depends, Query

    _client = httpx.AsyncClient(timeout=10.0)

    logger.info("Pro SaaS proxy enabled → %s", PRO_URL)

    try:
        from amfs_http.auth import verify_api_key
    except ImportError:
        async def verify_api_key():
            return None

    @app.get("/api/v1/cortex/hot-context")
    async def proxy_hot_context(
        agent_id: str | None = Query(None),
        _auth: str | None = Depends(verify_api_key),
    ) -> dict[str, Any]:
        """Proxy hot-context request to Pro SaaS API."""
        params = {}
        if agent_id:
            params["agent_id"] = agent_id
        try:
            resp = await _client.get(
                f"{PRO_URL}/api/v1/pro/cortex/hot-context",
                params=params,
                headers=_pro_headers(),
            )
            return resp.json()
        except Exception:
            logger.debug("Pro API unreachable for hot-context", exc_info=True)
            return {"agent_id": agent_id, "status": "unavailable", "focus_entities": [], "last_read_entities": []}

    @app.get("/api/v1/cortex/anticipation")
    async def proxy_anticipation(
        entity_path: str | None = Query(None),
        limit: int = Query(default=20, le=100),
        _auth: str | None = Depends(verify_api_key),
    ) -> dict[str, Any]:
        """Fetch local digests, send to Pro SaaS for anticipation ranking."""
        try:
            from amfs_http.server import _cortex_worker
            if _cortex_worker:
                adapter = _cortex_worker._compiler._adapter
                all_digests = adapter.list_digests()
                digest_dicts = [d.model_dump(mode="json") for d in all_digests]
            else:
                digest_dicts = []

            resp = await _client.post(
                f"{PRO_URL}/api/v1/pro/cortex/rank",
                headers=_pro_headers(),
                json={
                    "digests": digest_dicts,
                    "entity_path": entity_path,
                },
            )
            data = resp.json()
            ranked = data.get("digests", [])[:limit]
            return {"digests": ranked, "total": data.get("total", 0)}
        except Exception:
            logger.debug("Pro API unreachable for anticipation", exc_info=True)
            return {"digests": [], "message": "Pro API unavailable"}

    # --- Trace Pro actions (verify, replay, diff) ---

    @app.post("/api/v1/pro/traces/{trace_id}/verify")
    async def proxy_verify_trace(
        trace_id: str,
        _auth: str | None = Depends(verify_api_key),
    ) -> dict[str, Any]:
        """Proxy trace verification to Pro SaaS API."""
        try:
            resp = await _client.post(
                f"{PRO_URL}/api/v1/pro/traces/{trace_id}/verify",
                headers=_pro_headers(),
            )
            if resp.status_code == 404:
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="Trace not found")
            return resp.json()
        except Exception as exc:
            if hasattr(exc, "status_code"):
                raise
            logger.debug("Pro API unreachable for trace verify", exc_info=True)
            return {"error": "Pro API unavailable"}

    @app.post("/api/v1/pro/traces/{trace_id}/replay")
    async def proxy_replay_trace(
        trace_id: str,
        _auth: str | None = Depends(verify_api_key),
    ) -> dict[str, Any]:
        """Proxy trace replay to Pro SaaS API."""
        try:
            resp = await _client.post(
                f"{PRO_URL}/api/v1/pro/traces/{trace_id}/replay",
                headers=_pro_headers(),
            )
            if resp.status_code == 404:
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="Trace not found")
            return resp.json()
        except Exception as exc:
            if hasattr(exc, "status_code"):
                raise
            logger.debug("Pro API unreachable for trace replay", exc_info=True)
            return {"error": "Pro API unavailable"}

    @app.post("/api/v1/pro/traces/{trace_id}/diff")
    async def proxy_diff_trace(
        trace_id: str,
        _auth: str | None = Depends(verify_api_key),
    ) -> dict[str, Any]:
        """Proxy trace diff to Pro SaaS API."""
        try:
            resp = await _client.post(
                f"{PRO_URL}/api/v1/pro/traces/{trace_id}/diff",
                headers=_pro_headers(),
            )
            if resp.status_code == 404:
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="Trace not found")
            return resp.json()
        except Exception as exc:
            if hasattr(exc, "status_code"):
                raise
            logger.debug("Pro API unreachable for trace diff", exc_info=True)
            return {"error": "Pro API unavailable"}

    logger.info("Pro proxy routes mounted: hot-context, anticipation, trace-verify, trace-replay, trace-diff")


# ─────────────────────────────────────────────────────────────────────
# Event forwarding — used by CortexWorker to push events to Pro SaaS
# ─────────────────────────────────────────────────────────────────────


class ProEventForwarder:
    """Async-safe event forwarder for the CortexWorker.

    Buffers events and sends them in batches to the Pro SaaS API.
    Runs a background thread with its own event loop so it works
    from the synchronous CortexWorker.
    """

    def __init__(self, batch_size: int = 10, flush_interval: float = 2.0) -> None:
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="pro-event-forwarder")
        self._thread.start()
        logger.info("Pro event forwarder started → %s", PRO_URL)

    def stop(self) -> None:
        self._stop.set()

    def enqueue(self, event: dict[str, Any]) -> None:
        batch = None
        with self._lock:
            self._buffer.append(event)
            if len(self._buffer) >= self._batch_size:
                batch = self._buffer[:]
                self._buffer.clear()
        if batch:
            self._send_batch(batch)

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(self._flush_interval)
            with self._lock:
                if not self._buffer:
                    continue
                batch = self._buffer[:]
                self._buffer.clear()
            self._send_batch(batch)

    def _send_batch(self, batch: list[dict[str, Any]]) -> None:
        if not batch:
            return
        try:
            import urllib.request
            req = urllib.request.Request(
                f"{PRO_URL}/api/v1/pro/cortex/ingest",
                data=json.dumps({"events": batch}).encode(),
                headers=_pro_headers(),
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                logger.debug("Forwarded %d events to Pro API (status=%d)", len(batch), resp.status)
        except Exception:
            logger.debug("Failed to forward %d events to Pro API", len(batch), exc_info=True)


def create_forwarder() -> ProEventForwarder | None:
    """Create and start a Pro event forwarder if Pro is configured."""
    if not is_pro_configured():
        return None
    forwarder = ProEventForwarder()
    forwarder.start()
    return forwarder
