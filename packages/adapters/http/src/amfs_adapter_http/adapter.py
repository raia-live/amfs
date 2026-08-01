"""HttpAdapter — AdapterABC implementation that proxies all operations through the AMFS HTTP API.

Every call carries the ``X-AMFS-API-Key`` header so that the server-side tenant
middleware can enforce row-level security.  This adapter is intentionally
**synchronous** (httpx sync client) because the MCP server tool functions are
synchronous.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Callable

import httpx

from amfs_core.abc import AdapterABC, Agent, WatchHandle
from amfs_core.models import (
    DecisionTrace,
    Digest,
    Event,
    EventType,
    GraphEdge,
    GraphNeighborQuery,
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


def _raise_with_detail(resp: httpx.Response) -> None:
    """raise_for_status, but with the server's `detail` in the message.

    httpx's default message is just "409 Conflict for url ..." — the API's
    actionable guidance (e.g. "Agent identity X is already registered to a
    different user... call amfs_set_identity...") lives in the JSON body and
    MUST reach the caller: MCP agents read the exception text to self-correct.
    """
    if not resp.is_error:
        return
    detail = None
    try:
        payload = resp.json()
        if isinstance(payload, dict):
            detail = payload.get("detail") or payload.get("error")
    except Exception:
        detail = None
    if detail:
        raise httpx.HTTPStatusError(
            f"{resp.status_code} {resp.reason_phrase} for "
            f"{resp.request.method} {resp.request.url.path}: {detail}",
            request=resp.request,
            response=resp,
        )
    resp.raise_for_status()


class HttpAdapter(AdapterABC):
    """Storage adapter that delegates to the AMFS HTTP/REST API.

    Args:
        base_url: Root URL of the AMFS HTTP server (e.g. ``https://amfs-api-xxx.run.app``).
        api_key: Value sent in the ``X-AMFS-API-Key`` header for tenant scoping.
        timeout: Optional httpx Timeout override.
    """

    # The server's write handler (POST /api/v1/entries) records the WRITE
    # timeline event, so the SDK must not log it again (avoids double events).
    server_side_write_events: bool = True

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

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        max_retries = 4
        for attempt in range(max_retries):
            resp = self._client.request(method, path, **kwargs)
            if resp.status_code == 429 and attempt < max_retries - 1:
                wait = float(resp.headers.get("Retry-After", 2 ** attempt))
                logger.debug("Rate limited on %s %s, retrying in %.1fs", method, path, wait)
                time.sleep(wait)
                continue
            _raise_with_detail(resp)
            return resp.json()

    def _get(self, path: str, **params: Any) -> Any:
        return self._request("GET", path, params={k: v for k, v in params.items() if v is not None})

    def _post(self, path: str, body: dict[str, Any] | None = None) -> Any:
        return self._request("POST", path, json=body or {})

    # ── required abstract methods ─────────────────────────────────────

    def read(
        self,
        entity_path: str,
        key: str,
        *,
        min_confidence: float = 0.0,
        branch: str = "main",
    ) -> MemoryEntry | None:
        params: dict[str, Any] = {}
        if branch and branch != "main":
            params["branch"] = branch
        data = self._get(f"/api/v1/entries/{entity_path}/{key}", **params)
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
            "agent_id": entry.provenance.agent_id,
            # Without this the server stamps its own session on the entry, so
            # every write served by one process shares an id and none of them
            # match the decision trace that recorded the write.
            "session_id": entry.provenance.session_id,
        }
        data = self._post("/api/v1/entries", body)
        return _parse_entry(data)

    def list(
        self,
        entity_path: str | None = None,
        *,
        include_superseded: bool = False,
        branch: str = "main",
    ) -> list[MemoryEntry]:
        params: dict[str, Any] = {}
        if entity_path:
            params["entity_path"] = entity_path
        if branch and branch != "main":
            params["branch"] = branch
        if include_superseded:
            params["include_superseded"] = "true"
        data = self._get("/api/v1/entries", **params)
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
            "agent_id": record.agent_id,
        }
        data = self._post("/api/v1/outcomes", body)
        return [_parse_entry(e) for e in data.get("entries", [])]

    # ── optional overrides ────────────────────────────────────────────

    def search(self, query: SearchQuery, **kwargs: Any) -> list[MemoryEntry]:
        body: dict[str, Any] = {
            "entity_path": query.entity_path,
            "min_confidence": query.min_confidence,
            "limit": query.limit,
            "sort_by": query.sort_by or "confidence",
            "depth": query.depth,
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
        # Without this the server applies its own default and every SaaS caller's
        # include_artifacts=False is silently ignored, so a single search can come
        # back carrying whole source files.
        if getattr(query, "include_artifacts", True) is False:
            body["include_artifacts"] = False
        branch = kwargs.get("branch")
        if branch:
            body["branch"] = branch
        data = self._post("/api/v1/search", body)
        if isinstance(data, list):
            return [_parse_entry(e) for e in data]
        return [_parse_entry(e) for e in data.get("entries", data if isinstance(data, list) else [])]

    def retrieve(
        self,
        query: str,
        *,
        entity_path: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 10,
        semantic_weight: float = 0.5,
        recency_weight: float = 0.3,
        confidence_weight: float = 0.2,
        branch: str = "main",
        include_artifacts: bool = True,
    ) -> list[tuple[MemoryEntry, float, dict[str, float]]]:
        """Server-side semantic retrieval via POST /api/v1/retrieve.

        The server does the embedding + pgvector similarity + blend, so this
        works even when the client has no embedder. Returns
        (entry, score, breakdown) tuples; breakdown is empty on the server's
        lexical fallback. Artifacts (stored source files) are demoted by the
        server; pass ``include_artifacts=False`` to exclude them entirely.
        """
        body: dict[str, Any] = {
            "query": query,
            "min_confidence": min_confidence,
            "limit": limit,
            "semantic_weight": semantic_weight,
            "recency_weight": recency_weight,
            "confidence_weight": confidence_weight,
            "branch": branch,
            "include_artifacts": include_artifacts,
        }
        if entity_path:
            body["entity_path"] = entity_path
        data = self._post("/api/v1/retrieve", body)
        rows = data if isinstance(data, list) else data.get("entries", [])
        out: list[tuple[MemoryEntry, float, dict[str, float]]] = []
        for e in rows:
            score = float(e.get("_score", 0.0)) if isinstance(e, dict) else 0.0
            breakdown = e.get("_breakdown", {}) if isinstance(e, dict) else {}
            out.append((_parse_entry(e), score, breakdown))
        return out

    def stats(self) -> MemoryStats:
        data = self._get("/api/v1/stats")
        return MemoryStats.model_validate(data)

    def stats_extended(self, *, agent_ids: list[str] | None = None) -> dict:
        # The server applies its own visibility scoping; a local agent_ids
        # restriction can't be forwarded, so fall back to the list() scan.
        if agent_ids is not None:
            return super().stats_extended(agent_ids=agent_ids)
        data = self._get("/api/v1/stats")
        if "total_recalls" not in data:
            # Older server without extended stats — compute locally.
            return super().stats_extended()
        return data

    def entity_summaries(self, *, agent_ids: list[str] | None = None) -> list[dict]:
        if agent_ids is not None:
            return super().entity_summaries(agent_ids=agent_ids)
        try:
            data = self._get("/api/v1/entities")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return super().entity_summaries()
            raise
        return data.get("entities", [])

    def share_stats(
        self,
        *,
        since: datetime | None = None,
        pair_limit: int = 20,
        agent_ids: list[str] | None = None,
    ) -> dict:
        if agent_ids is not None:
            return super().share_stats(
                since=since, pair_limit=pair_limit, agent_ids=agent_ids
            )
        params: dict[str, Any] = {"pair_limit": pair_limit}
        if since:
            params["since"] = since.isoformat()
        try:
            return self._get("/api/v1/traces/share-stats", **params)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                # Older server routes this to /traces/{trace_id} → 404.
                return super().share_stats(since=since, pair_limit=pair_limit)
            raise

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

    def save_trace(self, trace: DecisionTrace) -> DecisionTrace:
        data = self._post("/api/v1/traces", trace.model_dump(mode="json"))
        return DecisionTrace.model_validate(data)

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

    def log_event(self, event: Event) -> Event:
        body: dict[str, Any] = {
            "agent_id": event.agent_id,
            "event_type": event.event_type.value if isinstance(event.event_type, EventType) else str(event.event_type),
            "summary": event.summary,
            "details": event.details,
            "branch": event.branch,
        }
        if event.actor_agent_id:
            body["actor_agent_id"] = event.actor_agent_id
        data = self._post("/api/v1/timeline/events", body)
        return Event.model_validate(data)

    def list_events(
        self,
        agent_id: str,
        namespace: str = "default",
        *,
        branch: str | None = None,
        event_type: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[Event]:
        params: dict[str, Any] = {"limit": limit}
        if branch:
            params["branch"] = branch
        if event_type:
            params["event_type"] = event_type
        if since:
            params["since"] = since.isoformat()
        data = self._get(f"/api/v1/agents/{agent_id}/timeline", **params)
        return [Event.model_validate(e) for e in data.get("events", [])]

    def upsert_graph_edge(
        self,
        edge: GraphEdge,
        *,
        namespace: str = "default",
        branch: str = "main",
    ) -> None:
        body = {
            "source_entity": edge.source_entity,
            "source_type": edge.source_type,
            "relation": edge.relation,
            "target_entity": edge.target_entity,
            "target_type": edge.target_type,
            "provenance": edge.provenance,
            "branch": branch,
        }
        self._post("/api/v1/graph/edges", body)

    def graph_neighbors(
        self,
        query: GraphNeighborQuery,
        *,
        namespace: str = "default",
        branch: str = "main",
    ) -> list[GraphEdge]:
        data = self._get(
            "/api/v1/pro/graph/neighbors",
            entity=query.entity,
            relation=query.relation,
            direction=query.direction,
            min_confidence=query.min_confidence,
            depth=query.depth,
            limit=query.limit,
        )
        return [GraphEdge.model_validate(e) for e in data.get("edges", [])]

    def ensure_agent(self, agent_id: str, namespace: str = "default") -> Agent:
        # Server-side ensure_agent runs automatically during writes.
        # Return a stub so AgentMemory.__init__ doesn't hit the ABC no-op.
        return Agent(agent_id=agent_id, namespace=namespace)

    def update_agent_profile(
        self,
        agent_id: str,
        profile: Any,
        namespace: str = "default",
    ) -> Agent:
        from urllib.parse import quote
        resp = self._client.put(
            f"/api/v1/agents/{quote(agent_id, safe='')}/profile",
            json=profile.model_dump(),
        )
        _raise_with_detail(resp)
        return Agent.model_validate(resp.json())

    def list_digests(
        self,
        digest_type: Any = None,
        namespace: str = "default",
        branch: str = "main",
    ) -> list[Digest]:
        params: dict[str, Any] = {}
        if digest_type is not None:
            params["digest_type"] = digest_type.value if hasattr(digest_type, "value") else str(digest_type)
        data = self._get("/api/v1/cortex/digests", **params)
        return [Digest.model_validate(d) for d in data.get("digests", [])]

    def briefing(
        self,
        entity_path: str | None = None,
        agent_id: str | None = None,
        limit: int = 10,
    ) -> list[Digest]:
        """Proxy briefing to the HTTP server which has full Cortex access."""
        params: dict[str, Any] = {"limit": limit}
        if entity_path:
            params["entity_path"] = entity_path
        if agent_id:
            params["agent_id"] = agent_id
        data = self._get("/api/v1/briefing", **params)
        return [Digest.model_validate(d) for d in data.get("digests", [])]
