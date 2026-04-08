"""HttpAdapter — AdapterABC implementation that proxies all operations through the AMFS HTTP API.

Every call carries the ``X-AMFS-API-Key`` header so that the server-side tenant
middleware can enforce row-level security.  This adapter is intentionally
**synchronous** (httpx sync client) because the MCP server tool functions are
synchronous.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable

import httpx

from amfs_core.abc import AdapterABC, WatchHandle
from amfs_core.models import (
    DecisionTrace,
    MemoryEntry,
    MemoryStats,
    OutcomeRecord,
    SearchQuery,
)

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


def _parse_entry(data: dict[str, Any]) -> MemoryEntry:
    """Reconstruct a MemoryEntry from a JSON dict returned by the API."""
    return MemoryEntry.model_validate(data)


class HttpAdapter(AdapterABC):
    """Storage adapter that delegates to the AMFS HTTP/REST API.

    Args:
        base_url: Root URL of the AMFS HTTP server (e.g. ``https://amfs-api-xxx.run.app``).
        api_key: Value sent in the ``X-AMFS-API-Key`` header for tenant scoping.
        timeout: Optional httpx Timeout override.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: httpx.Timeout | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._api_key = api_key
        self._client = httpx.Client(
            base_url=self._base,
            headers={"X-AMFS-API-Key": api_key},
            timeout=timeout or _TIMEOUT,
        )

    # ── helpers ────────────────────────────────────────────────────────

    def _get(self, path: str, **params: Any) -> Any:
        resp = self._client.get(path, params={k: v for k, v in params.items() if v is not None})
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict[str, Any] | None = None) -> Any:
        resp = self._client.post(path, json=body or {})
        resp.raise_for_status()
        return resp.json()

    # ── required abstract methods ─────────────────────────────────────

    def read(
        self,
        entity_path: str,
        key: str,
        *,
        min_confidence: float = 0.0,
    ) -> MemoryEntry | None:
        data = self._get(f"/api/v1/entries/{entity_path}/{key}")
        if data.get("status") == "not_found":
            return None
        entry = _parse_entry(data)
        if entry.confidence < min_confidence:
            return None
        return entry

    def write(self, entry: MemoryEntry) -> MemoryEntry:
        body = {
            "entity_path": entry.entity_path,
            "key": entry.key,
            "value": entry.value,
            "confidence": entry.confidence,
            "pattern_refs": entry.provenance.pattern_refs,
            "memory_type": entry.memory_type.value if hasattr(entry.memory_type, "value") else str(entry.memory_type),
            "shared": entry.shared,
            "branch": entry.branch,
        }
        data = self._post("/api/v1/entries", body)
        return _parse_entry(data)

    def list(
        self,
        entity_path: str | None = None,
        *,
        include_superseded: bool = False,
    ) -> list[MemoryEntry]:
        data = self._get("/api/v1/entries", entity_path=entity_path)
        return [_parse_entry(e) for e in data.get("entries", [])]

    def watch(
        self,
        entity_path: str,
        callback: Callable[[MemoryEntry], None],
    ) -> WatchHandle:
        logger.warning("HttpAdapter.watch() is a no-op over HTTP; use SSE streaming instead")
        return WatchHandle(cancel_fn=lambda: None)

    def commit_outcome(self, record: OutcomeRecord) -> list[MemoryEntry]:
        body = {
            "outcome_ref": record.outcome_ref,
            "outcome_type": record.outcome_type.value if hasattr(record.outcome_type, "value") else str(record.outcome_type),
            "causal_entry_keys": record.causal_entry_keys,
            "causal_confidence": record.causal_confidence,
        }
        data = self._post("/api/v1/outcomes", body)
        return [_parse_entry(e) for e in data.get("entries", [])]

    # ── optional overrides ────────────────────────────────────────────

    def search(self, query: SearchQuery) -> list[MemoryEntry]:
        body: dict[str, Any] = {
            "entity_path": query.entity_path,
            "min_confidence": query.min_confidence,
            "limit": query.limit,
            "sort_by": query.sort_by or "confidence",
        }
        if query.query:
            body["query"] = query.query
        if query.max_confidence is not None:
            body["max_confidence"] = query.max_confidence
        if query.agent_id:
            body["agent_id"] = query.agent_id
        if query.since:
            body["since"] = query.since.isoformat()
        if query.pattern_ref:
            body["pattern_ref"] = query.pattern_ref
        data = self._post("/api/v1/search", body)
        if isinstance(data, list):
            return [_parse_entry(e) for e in data]
        return [_parse_entry(e) for e in data.get("entries", data if isinstance(data, list) else [])]

    def stats(self) -> MemoryStats:
        data = self._get("/api/v1/stats")
        return MemoryStats.model_validate(data)

    def list_outcomes(
        self,
        *,
        entity_path: str | None = None,
        since: datetime | None = None,
        limit: int = 1000,
    ) -> list[OutcomeRecord]:
        params: dict[str, Any] = {"limit": limit}
        if entity_path:
            params["entity_path"] = entity_path
        if since:
            params["since"] = since.isoformat()
        data = self._get("/api/v1/outcomes", **params)
        return [OutcomeRecord.model_validate(o) for o in data.get("outcomes", [])]

    def list_traces(
        self,
        *,
        entity_path: str | None = None,
        agent_id: str | None = None,
        outcome_type: str | None = None,
        limit: int = 100,
    ) -> list[DecisionTrace]:
        data = self._get(
            "/api/v1/traces",
            entity_path=entity_path,
            agent_id=agent_id,
            outcome_type=outcome_type,
            limit=limit,
        )
        return [DecisionTrace.model_validate(t) for t in data.get("traces", [])]

    def get_trace(self, trace_id: str) -> DecisionTrace | None:
        try:
            data = self._get(f"/api/v1/traces/{trace_id}")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise
        if data.get("error"):
            return None
        return DecisionTrace.model_validate(data)
