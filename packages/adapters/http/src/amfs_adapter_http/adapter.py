"""HttpAdapter — AdapterABC implementation that proxies all operations through the AMFS HTTP API.

Every call carries the ``X-AMFS-API-Key`` header so that the server-side tenant
middleware can enforce row-level security.  This adapter is intentionally
**synchronous** (httpx sync client) because the MCP server tool functions are
synchronous.
"""

from __future__ import annotations

import json
import logging
import time
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Callable

import httpx

from amfs_core.abc import AdapterABC, Agent, WatchHandle
from amfs_core.models import (
    Commit,
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
from amfs_core.reuse_value import REUSE_VALUE_HEADER

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(30.0, connect=10.0)

# Things the server computed that the ABC's return types have nowhere to put: the
# reuse block a read earns, and the gap report a commit earns. Both have to cross
# from the response to the caller somehow, and both used to do it as attributes on
# the adapter.
#
# Context variables rather than attributes because one adapter serves many calls.
# The gateway builds a separate adapter per session, so this was never a
# cross-tenant hazard, but a client may perfectly well have two reads in flight on
# one session — and the sequence "A responds, B responds, A reads the value" then
# hands A the block describing B's lookup. That is the staleness bug the adapter
# already guards against between sequential reads, in its concurrent form, and a
# block whose only job is to be believed cannot describe a different call.
#
# A ContextVar is the fix rather than a lock: it scopes the value to the logical
# call, which is the thing it actually belongs to. asyncio copies the context per
# task and anyio's to_thread copies it per worker, so both ways the gateway and
# the stdio servers reach a sync adapter land in the right place, and no caller
# has to hold anything.
_LAST_REUSE_HEADER: ContextVar[str | None] = ContextVar(
    "amfs_last_reuse_header", default=None
)
_LAST_REUSE_VALUE: ContextVar[dict[str, Any] | None] = ContextVar(
    "amfs_last_reuse_value", default=None
)
_LAST_MEMORY_GAP: ContextVar[dict[str, Any] | None] = ContextVar(
    "amfs_last_memory_gap", default=None
)
#: The scope facts the last write came back with, for the same reason as the two
#: above: ``write()`` has to return a MemoryEntry, so anything else the server
#: said travels beside it rather than in it.
_LAST_SCOPE: ContextVar[dict[str, Any] | None] = ContextVar(
    "amfs_last_scope", default=None
)


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

    #: Outcome learning happens on the server this adapter talks to: it runs
    #: the trigger and writes derived entries (contrast lessons) itself, under
    #: the caller's identity. The SDK checks this to avoid doing the same
    #: work a second time on the client.
    remote_learning = True

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
            # The reuse block travels as a header because /search and /retrieve
            # answer with a bare JSON array: there is no envelope to add a key to,
            # and wrapping the array would break every existing client to carry a
            # diagnostic. Taken off here and left in the calling context, so the
            # read methods below can pick it up without every one of them having
            # to touch the response object.
            _LAST_REUSE_HEADER.set(resp.headers.get(REUSE_VALUE_HEADER))
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
        if "/" in key:
            # The path form cannot express this. Its converter is greedy, so
            # everything up to the final segment becomes the entity path, and a
            # read of sweep/abc + active-negotiation/xyz asks the server for
            # sweep/abc/active-negotiation + xyz — coordinates nothing was
            # written at, which comes back as an ordinary "not found" rather
            # than as an error anyone would notice.
            #
            # Only slashed keys take the query route, so a server too old to
            # have it keeps serving every read it was already serving
            # correctly. The ones it would 404 on are the ones it could never
            # answer anyway.
            data = self._get(
                "/api/v1/entry", entity_path=entity_path, key=key, **params
            )
        else:
            data = self._get(f"/api/v1/entries/{entity_path}/{key}", **params)
        self._capture_reuse_value()
        if data.get("status") == "not_found":
            return None
        entry = _parse_entry(data)
        if entry.confidence < min_confidence:
            return None
        return entry

    #: Whether writes ask the server for the scope block. On by default, unlike
    #: the server-side flag: it costs the caller nothing extra now that it rides
    #: on the write request instead of a second one, and a client that never
    #: shows it loses only an aggregate. Set AMFS_WRITE_SCOPE=0 to opt out.
    @property
    def _want_scope(self) -> bool:
        import os

        return os.environ.get("AMFS_WRITE_SCOPE", "1").lower() not in (
            "0", "false", "off", "no",
        )

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
        if self._want_scope:
            body["include_scope"] = True
        data = self._post("/api/v1/entries", body)
        # Always assign, so a write to a fresh scope — or one served by a server
        # too old to compute it — clears the previous write's block instead of
        # leaving it to be reported against this one. Same rule as the reuse
        # block; the failure it avoids is reporting neighbours that belong to a
        # different entity path.
        _LAST_SCOPE.set(data.get("scope") if isinstance(data, dict) else None)
        return _parse_entry(data)

    @property
    def _last_scope(self) -> dict[str, Any] | None:
        """Scope facts from the write most recently made *in this context*."""
        return _LAST_SCOPE.get()

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
        body: dict[str, Any] = {
            "outcome_ref": record.outcome_ref,
            "outcome_type": record.outcome_type.value if hasattr(record.outcome_type, "value") else str(record.outcome_type),
            "causal_entry_keys": record.causal_entry_keys,
            "causal_confidence": record.causal_confidence,
            "agent_id": record.agent_id,
        }
        # The server seals its immutable trace from this call, so the capture has to
        # travel with it. Sending it only on the later save_trace left the sealed
        # copy — which is what Pro export and training read — with an empty prompt.
        if record.task_input:
            body["task_input"] = record.task_input
        if record.response_text:
            body["response_text"] = record.response_text
        if record.tool_calls:
            body["tool_calls"] = record.tool_calls
        # Same story for the session's attribute bag and LLM calls: the server
        # reads ``session_metadata.attributes`` / ``.llm_calls`` from this body
        # when it seals, so a bag sent only with the trace never reached the
        # sealed copy.
        if record.session_metadata:
            body["session_metadata"] = record.session_metadata
        # The counterpart of all of the above: everything this call carries is
        # carried so the server can seal from it, and none of it makes the server's
        # copy as good as the trace that follows on ``save_trace``. When one is
        # coming, say so, and the server seals once from the better of the two.
        if record.trace_follows:
            body["trace_follows"] = True
        # Failed attempts and the terminal action: the trigger applies each
        # attempt's outcome to its own entries server-side, so these must
        # arrive on the outcome, not only on the trace.
        if record.attempts:
            body["attempts"] = [a.model_dump(mode="json") for a in record.attempts]
        if record.final_action_index is not None:
            body["final_action_index"] = record.final_action_index
        data = self._post("/api/v1/outcomes", body)
        # The ABC returns entries, so anything else the server computed for this
        # commit has nowhere to go in the signature and would be dropped here.
        # The gap report is computed server-side on purpose — that is the only
        # place a retrieve is free and side-effect-free — so it needs a way
        # across. Stashed rather than returned, and read once by
        # ``AgentMemory.commit_outcome``: widening the ABC would oblige every
        # adapter, including the filesystem one that has no retrieval, to carry a
        # concept only the server can produce.
        _LAST_MEMORY_GAP.set(data.get("memory_gap"))
        return [_parse_entry(e) for e in data.get("entries", [])]

    @property
    def _last_reuse_value(self) -> dict[str, Any] | None:
        """Reuse block from the read most recently made *in this context*."""
        return _LAST_REUSE_VALUE.get()

    @property
    def _last_memory_gap(self) -> dict[str, Any] | None:
        """Gap report from the commit most recently made in this context."""
        return _LAST_MEMORY_GAP.get()

    def _capture_reuse_value(self) -> None:
        """Take the reuse block off the last response, or clear it.

        Always assigns, so a read that credited no reuse — a miss, or a
        filter-only search — leaves None behind rather than the previous read's
        block. A server too old to send the header is the same case as a read
        that credited nothing, and correctly reports nothing.
        """
        raw = _LAST_REUSE_HEADER.get()
        if not raw:
            _LAST_REUSE_VALUE.set(None)
            return
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            logger.debug("Unparseable reuse value header", exc_info=True)
            _LAST_REUSE_VALUE.set(None)
            return
        _LAST_REUSE_VALUE.set(parsed if isinstance(parsed, dict) else None)

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
        # Same reasoning as include_artifacts above: unsent, the server matches
        # the path exactly and a caller asking about a scope silently gets only
        # the entries written at the root of it, which is usually none of them.
        if getattr(query, "include_descendants", False):
            body["include_descendants"] = True
        branch = kwargs.get("branch")
        if branch:
            body["branch"] = branch
        data = self._post("/api/v1/search", body)
        self._capture_reuse_value()
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
        evidence_weight: float | None = None,
        include_discredited: bool | None = None,
        include_avoid: bool | None = None,
        adaptive_k: bool | None = None,
    ) -> list[tuple[MemoryEntry, float, dict[str, Any]]]:
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
        # Sent only when set, so an older server that does not know the fields
        # is not handed keys it would reject.
        if evidence_weight is not None:
            body["evidence_weight"] = evidence_weight
        if include_discredited is not None:
            body["include_discredited"] = include_discredited
        if include_avoid is not None:
            body["include_avoid"] = include_avoid
        if adaptive_k is not None:
            body["adaptive_k"] = adaptive_k
        data = self._post("/api/v1/retrieve", body)
        self._capture_reuse_value()
        rows = data if isinstance(data, list) else data.get("entries", [])
        out: list[tuple[MemoryEntry, float, dict[str, Any]]] = []
        for e in rows:
            score = float(e.get("_score", 0.0)) if isinstance(e, dict) else 0.0
            breakdown = dict(e.get("_breakdown", {}) or {}) if isinstance(e, dict) else {}
            if isinstance(e, dict) and e.get("_avoid"):
                # A discredited entry the server appended because the caller
                # asked what not to do. Carried in the breakdown so ScoredEntry
                # keeps it and the SDK can hand it back as an avoid list.
                breakdown["_avoid"] = True
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
        outcome_ref: str | None = None,
    ) -> list[OutcomeRecord]:
        params: dict[str, Any] = {"limit": limit}
        if entity_path:
            params["entity_path"] = entity_path
        if since:
            params["since"] = since.isoformat()
        if outcome_ref:
            params["outcome_ref"] = outcome_ref
        data = self._get("/api/v1/outcomes", **params)
        return [OutcomeRecord.model_validate(o) for o in data.get("outcomes", [])]

    def list_traces(
        self,
        *,
        entity_path: str | None = None,
        agent_id: str | None = None,
        outcome_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
        cursor: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[DecisionTrace]:
        """One page of traces. The server caps a page at 1,000 rows; callers
        that need more than that follow cursors (see :meth:`_iter_traces`)."""
        data = self.list_traces_page(
            entity_path=entity_path,
            agent_id=agent_id,
            outcome_type=outcome_type,
            limit=limit,
            offset=offset,
            cursor=cursor,
            since=since,
            until=until,
        )
        return data[0]

    def list_traces_page(
        self,
        *,
        entity_path: str | None = None,
        agent_id: str | None = None,
        outcome_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
        cursor: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> tuple[list[DecisionTrace], str | None, bool]:
        """``(traces, next_cursor, has_more)`` straight from the server's page."""
        params: dict[str, Any] = {
            "entity_path": entity_path,
            "agent_id": agent_id,
            "outcome_type": outcome_type,
            "limit": limit,
        }
        if cursor:
            params["cursor"] = cursor
        elif offset:
            params["offset"] = offset
        # The window is applied by the server's query, not to the page after
        # the fact: filtering a page here would empty it whenever the window
        # excludes the newest traces, and the cursor walk in _iter_traces
        # stops on an empty page — so older matches would never be reached.
        if since is not None:
            params["since"] = since.isoformat()
        if until is not None:
            params["until"] = until.isoformat()
        data = self._get("/api/v1/traces", **params)
        traces = [DecisionTrace.model_validate(t) for t in data.get("traces", [])]
        return traces, data.get("next_cursor"), bool(data.get("has_more", False))

    def _iter_traces(self, *, max_rows: int, **filters: Any) -> list[DecisionTrace]:
        """Follow cursors until *max_rows* traces have been read or the list ends."""
        out: list[DecisionTrace] = []
        cursor: str | None = None
        while len(out) < max_rows:
            page, cursor, has_more = self.list_traces_page(
                limit=min(1000, max_rows - len(out)), cursor=cursor, **filters
            )
            out.extend(page)
            if not has_more or not cursor or not page:
                break
        return out

    def count_traces(
        self,
        *,
        entity_path: str | None = None,
        agent_id: str | None = None,
        outcome_type: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> int:
        from amfs_core.pagination import max_scan_rows

        return len(
            self._iter_traces(
                max_rows=max_scan_rows(),
                entity_path=entity_path,
                agent_id=agent_id,
                outcome_type=outcome_type,
                since=since,
                until=until,
            )
        )

    def trace_read_counts(self, agent_id: str) -> dict[str, dict[str, int]]:
        from amfs_core.pagination import max_scan_rows

        counts: dict[str, dict[str, int]] = {}
        for t in self._iter_traces(max_rows=max_scan_rows(), agent_id=agent_id):
            for ce in t.causal_entries:
                per_entity = counts.setdefault(ce.entity_path, {})
                per_entity[ce.key] = per_entity.get(ce.key, 0) + 1
        return counts

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

    # ── commits ───────────────────────────────────────────────────────
    #
    # These inherited the base class's placeholders — ``None`` and ``[]`` — for
    # as long as this adapter has existed, which is silent and total failure for
    # everything built on them. Over HTTP, ``commit_log()`` returned an empty
    # history for every account, and ``common_ancestor()`` reported no common
    # ancestor for every pair of commits, including two that obviously share
    # one. Both read as an honest answer about an empty repository, so nothing
    # ever complained.
    #
    # ``save_commit`` remains the inherited no-op, and now deliberately rather
    # than as an unfinished thought. It was one because a client-side
    # transaction assembled a Commit here and had nowhere to send it: no
    # endpoint accepted a commit somebody else had minted, so the id went back
    # to the caller and the object was dropped, leaving an id ``get_commit``
    # could not resolve.
    #
    # The way out was not to add that endpoint, which would mean trusting a
    # client-assembled id and tree hash, but to stop assembling commits on this
    # side at all. ``commit_batch`` posts the writes and the server runs the
    # transaction, mints the commit and persists it. That is also what makes
    # the group atomic: one transaction on one connection, rather than n
    # separate requests that can half-succeed.
    #
    # What that leaves is a ``save_commit`` nobody should reach. It is not made
    # to raise, because ``TransactionBuffer.flush`` still calls it on the paths
    # that have not moved over, and turning those into errors would break
    # working code to make a point.
    #
    # One consequence of ``list_commits`` working: ``flush`` calls it to find a
    # parent, so a transaction over HTTP now costs one more request than before
    # and its commits form a real chain instead of every one being a root.

    def commit_batch(
        self,
        writes: list[dict[str, Any]],
        message: str = "",
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> Commit | None:
        """Write a group of entries as one commit, run server-side.

        Returns the commit the server minted, or ``None`` from a server old
        enough not to send one back. ``None`` means "committed, details
        unavailable" rather than "failed": by the time the response is read the
        transaction has already landed, and treating it as an error would have
        a caller retry writes that are already there.

        That reading only holds because a 200 now means something was written.
        An empty ``writes`` list never reached flush, so the server answered
        200 with no commit — the same answer an old server gives — and the
        caller could not tell "nothing happened" from "landed, silently". The
        server refuses an empty batch instead, which leaves ``None`` with one
        meaning rather than two.

        ``agent_id`` and ``session_id`` say who is committing, the same way
        ``write`` does. Without them the server transacts as itself, which
        credits the entries and the commit to the server process rather than to
        the caller — and a commit the caller does not appear to have authored
        is one they are not shown when they ask for their own log.
        """
        if not writes:
            raise ValueError("writes is empty — nothing to commit.")
        body: dict[str, Any] = {"writes": writes, "message": message}
        if agent_id:
            body["agent_id"] = agent_id
        if session_id:
            body["session_id"] = session_id
        data = self._post("/api/v1/commits", body)
        commit = data.get("commit")
        return Commit.model_validate(commit) if commit else None

    def get_commit(self, commit_id: str) -> Commit | None:
        try:
            data = self._get(f"/api/v1/commits/{commit_id}")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise
        return Commit.model_validate(data)

    def list_commits(
        self,
        *,
        branch: str = "main",
        limit: int = 50,
        namespace: str = "default",
    ) -> list[Commit]:
        """Commits, newest first.

        ``branch`` and ``namespace`` are sent, so the server filters before it
        applies the limit. Filtering here afterwards instead — which is what
        this did first — takes the newest N account-wide and then discards the
        ones the caller did not want, so a page full of another branch's
        commits comes back empty even though the requested branch has plenty.

        The case that makes it more than a paging nicety is
        ``TransactionBuffer.flush``, which asks for exactly one commit to use
        as the new commit's parent. One commit on the wrong branch means no
        parent, so every commit over HTTP is a root, no two share an ancestor,
        and ``common_ancestor`` answers "none" forever — indistinguishable from
        an honest answer about unrelated history.

        The local filter stays as a guard, not as the mechanism. A server old
        enough to ignore the query parameters returns an unfiltered page, and
        handing a caller another branch's commits because the server did not
        understand the question is worse than handing it too few. Against such
        a server a result can still be short, which is a completeness problem;
        without the guard it would be a correctness one.
        """
        data = self._get(
            "/api/v1/commits",
            limit=limit,
            # Empty means "do not filter" to the guard below, so it has to mean
            # the same to the server rather than arriving as branch="" and
            # matching nothing.
            branch=branch or None,
            namespace=namespace or None,
        )
        commits = [Commit.model_validate(c) for c in data.get("commits", [])]
        return [
            c for c in commits
            if (not branch or c.branch == branch)
            and (not namespace or c.namespace == namespace)
        ]

    def common_ancestor(self, commit_a_id: str, commit_b_id: str) -> str | None:
        """The latest commit both descend from, resolved by the server.

        An override rather than a missing method. The generic implementation
        walks the DAG breadth-first calling ``get_commit`` at every node, which
        is reasonable against a local store and wasteful against this one: each
        step becomes an HTTP round trip, and each round trip is a billable
        operation, so the cost of the answer scales with how much history the
        two commits happen to share. The server has the whole graph in one
        place and exposes exactly this question, so ask it once.

        Falls back to the inherited walk if the endpoint is missing, which is
        what an older server looks like from here.
        """
        try:
            data = self._post(
                "/api/v1/merge-base",
                {"commit_a": commit_a_id, "commit_b": commit_b_id},
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return super().common_ancestor(commit_a_id, commit_b_id)
            raise
        return data.get("ancestor_commit_id")

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
        offset: int = 0,
        cursor: str | None = None,
        until: datetime | None = None,
    ) -> list[Event]:
        params: dict[str, Any] = {"limit": limit}
        if branch:
            params["branch"] = branch
        if event_type:
            params["event_type"] = event_type
        if since:
            params["since"] = since.isoformat()
        if until:
            params["until"] = until.isoformat()
        if cursor:
            params["cursor"] = cursor
        elif offset:
            params["offset"] = offset
        data = self._get(f"/api/v1/agents/{agent_id}/timeline", **params)
        events = [Event.model_validate(e) for e in data.get("events", [])]
        if until is not None:
            # Older servers ignore the parameter; keep the bound exact locally.
            events = [e for e in events if e.created_at < until]
        return events

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
        credit_reuse: bool = False,
        compact: bool = False,
    ) -> list[Digest]:
        """Proxy briefing to the HTTP server which has full Cortex access."""
        params: dict[str, Any] = {"limit": limit}
        if entity_path:
            params["entity_path"] = entity_path
        if agent_id:
            params["agent_id"] = agent_id
        # Sent only when asked for. The server treats a briefing as a real read
        # of what it surfaces, and that is true for an agent about to act on it
        # and false for a panel rendering it for a human, so the caller decides.
        if credit_reuse:
            params["credit_reuse"] = "true"
        if compact:
            params["compact"] = "true"
        data = self._get("/api/v1/briefing", **params)
        # A credited briefing is a credited read, so it carries the reuse header
        # like any other. Unconditional: _capture_reuse_value always assigns, so
        # an uncredited briefing correctly clears any earlier read's block rather
        # than leaving it to be reported against this one.
        self._capture_reuse_value()
        return [Digest.model_validate(d) for d in data.get("digests", [])]
