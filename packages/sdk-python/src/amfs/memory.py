"""AgentMemory — the main SDK entry point for agents."""

from __future__ import annotations

import logging
import math
import threading
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

from amfs_core.abc import AdapterABC, WatchHandle
from amfs_core.capture import scan_captured_arguments, scan_captured_text
from amfs_core.content import embedding_input
from amfs_core.embedder import EmbedderABC
from amfs_core.engine import CausalTagger, CoWEngine, ReadTracker
from amfs_core.exceptions import StaleWriteError
from amfs_core.lifecycle import LifecycleManager
from amfs_core.models import (
    Commit,
    ConflictPolicy,
    DecisionTrace,
    ErrorEvent,
    Event,
    EventType,
    ExternalContext,
    GraphEdge,
    GraphNeighborQuery,
    MemoryEntry,
    MemoryStateDiff,
    MemoryStats,
    MemoryType,
    OutcomeType,
    QueryEvent,
    RecallConfig,
    ScopeInfo,
    ScoredEntry,
    SearchQuery,
    SemanticQuery,
    SessionMetadata,
    ToolCall,
    TraceEntry,
)
from amfs_core.outcome import OutcomeBackPropagator

from amfs.config import load_config_or_default
from amfs.factory import create_adapter_from_config

logger = logging.getLogger(__name__)

_sdk_bg_executor: ThreadPoolExecutor | None = None
_sdk_bg_lock = threading.Lock()


def _get_sdk_executor() -> ThreadPoolExecutor:
    global _sdk_bg_executor
    if _sdk_bg_executor is None:
        with _sdk_bg_lock:
            if _sdk_bg_executor is None:
                _sdk_bg_executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="amfs-sdk-bg"
                )
    return _sdk_bg_executor


# ---------------------------------------------------------------------------
# Session metadata extras: the attribute bag and the LLM-call list a session
# collects for its trace. Both travel in ``DecisionTrace.session_metadata``
# under the keys below, which is what the server lifts into the sealed trace.
# ---------------------------------------------------------------------------

SESSION_ATTRIBUTES_KEY = "attributes"
SESSION_LLM_CALLS_KEY = "llm_calls"

#: Attributes are dimensions to filter and group traces by (customer, task
#: type, ...). They are indexed server-side, so the bag is kept small and flat.
SESSION_ATTRIBUTES_MAX_KEYS = 20
SESSION_ATTRIBUTE_KEY_MAX_LEN = 64
SESSION_ATTRIBUTE_VALUE_MAX_LEN = 256

_ATTRIBUTE_SCALARS = (str, int, float, bool)


def _check_attribute_count(attributes: dict[str, Any], what: str = "session attributes") -> None:
    """``ValueError`` when *attributes* holds more than ``SESSION_ATTRIBUTES_MAX_KEYS``.

    Applied to each incoming bag and again to every merge (the session bag as
    it grows, and the trace's bag of identity metadata + session bag + commit
    attributes): a per-call check alone lets three bags of 20 become one of 60.
    """
    if len(attributes) > SESSION_ATTRIBUTES_MAX_KEYS:
        raise ValueError(
            f"at most {SESSION_ATTRIBUTES_MAX_KEYS} {what} are allowed "
            f"(got {len(attributes)})"
        )


def validate_session_attributes(attributes: Any) -> dict[str, str | int | float | bool]:
    """The validated form of a session attribute bag, or ``ValueError``.

    Keys are stripped and lowercased (the server indexes them that way, so two
    spellings of one dimension would otherwise never group together). Values
    must be ``str``, ``int``, ``float`` or ``bool``; at most
    ``SESSION_ATTRIBUTES_MAX_KEYS`` keys, each at most
    ``SESSION_ATTRIBUTE_KEY_MAX_LEN`` characters, string values at most
    ``SESSION_ATTRIBUTE_VALUE_MAX_LEN``. Rejected rather than trimmed: this is a
    developer-facing API, and a silently shortened customer id is worse than an
    error at the call site.
    """
    if attributes is None:
        return {}
    if not isinstance(attributes, dict):
        raise TypeError("attributes must be a dict of scalar values")
    _check_attribute_count(attributes)
    out: dict[str, str | int | float | bool] = {}
    for raw_key, value in attributes.items():
        key = str(raw_key).strip().lower()
        if not key:
            raise ValueError("attribute keys must be non-empty strings")
        if len(key) > SESSION_ATTRIBUTE_KEY_MAX_LEN:
            raise ValueError(
                f"attribute key {key[:20]!r}... exceeds {SESSION_ATTRIBUTE_KEY_MAX_LEN} characters"
            )
        if not isinstance(value, _ATTRIBUTE_SCALARS):
            raise TypeError(
                f"attribute {key!r} must be str, int, float or bool, "
                f"not {type(value).__name__}"
            )
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise ValueError(f"attribute {key!r} must be a finite number")
        if isinstance(value, str) and len(value) > SESSION_ATTRIBUTE_VALUE_MAX_LEN:
            raise ValueError(
                f"attribute {key!r} exceeds {SESSION_ATTRIBUTE_VALUE_MAX_LEN} characters"
            )
        out[key] = value
    return out


def _normalize_llm_call(call: Any) -> dict[str, Any] | None:
    """A JSON-safe LLM call record in the shape the trace pipeline reads
    (``call_id``, ``model``, ``provider``, ``input_tokens``, ``output_tokens``,
    ``cost_usd``, ``latency_ms``, ``started_at``), or ``None`` for an entry that
    is not a usable call. Unknown keys are kept, so a richer client record
    (cached tokens, finish reason, request id) travels intact."""
    if not isinstance(call, dict):
        return None
    try:
        input_tokens = int(call.get("input_tokens") or 0)
        output_tokens = int(call.get("output_tokens") or 0)
    except (TypeError, ValueError):
        return None
    model = call.get("model")
    if not model and not input_tokens and not output_tokens:
        return None
    started_at = call.get("started_at")
    if isinstance(started_at, datetime):
        started_at = started_at.isoformat()
    elif not isinstance(started_at, str) or not started_at:
        started_at = datetime.now(UTC).isoformat()

    def _num(value: Any) -> float | None:
        if value is None:
            return None
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if math.isfinite(out) else None

    normalized: dict[str, Any] = {
        **call,
        "call_id": str(call.get("call_id") or uuid.uuid4()),
        "model": str(model or ""),
        "provider": str(call.get("provider") or ""),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": _num(call.get("cost_usd")),
        "latency_ms": _num(call.get("latency_ms")),
        "started_at": started_at,
    }
    return normalized


def _metadata_to_dict(meta: Any) -> dict[str, Any]:
    """``session_metadata`` as a plain dict, extras included, whatever it is held as."""
    if meta is None:
        return {}
    if hasattr(meta, "model_dump"):
        return meta.model_dump(mode="json")
    if isinstance(meta, dict):
        return dict(meta)
    return {}


# ---------------------------------------------------------------------------
# Lightweight digest scoring — used when amfs_cortex is not installed but the
# adapter supports list_digests (e.g. PostgresAdapter used directly by the MCP
# server without the full Cortex package).
# ---------------------------------------------------------------------------

def _score_digests(
    digests: list,
    *,
    entity_path: str | None = None,
    agent_id: str | None = None,
    limit: int = 10,
) -> list:
    """Rank digests by relevance to entity_path / agent_id."""
    from amfs_core.models import DigestType

    now = datetime.now(timezone.utc)
    scored: list[tuple[float, Any]] = []

    for d in digests:
        score = 0.0

        if entity_path:
            if d.digest_type == DigestType.ENTITY and d.scope == entity_path:
                score += 100.0
            elif d.digest_type == DigestType.CONNECTION_MAP and d.scope == entity_path:
                score += 80.0
            elif d.digest_type == DigestType.SOURCE:
                if entity_path in d.summary.get("entities_touched", []):
                    score += 60.0
            elif d.digest_type == DigestType.AGENT_BRIEF:
                if entity_path in d.summary.get("entities_written", []):
                    score += 40.0

        if agent_id:
            if d.digest_type == DigestType.AGENT_BRIEF and d.scope == agent_id:
                score += 100.0
            elif d.digest_type == DigestType.ENTITY:
                if agent_id in d.summary.get("agents", []):
                    score += 50.0

        if score == 0:
            continue

        age_hours = max((now - d.compiled_at).total_seconds() / 3600, 0.01)
        score += min(10.0 / age_hours, 20.0)
        score += min(d.entry_count * 0.5, 15.0)
        score += d.anticipation_score * 30.0

        d.staleness_ms = int((now - d.compiled_at).total_seconds() * 1000)
        scored.append((score, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:limit]]


class AgentMemory:
    """High-level API for agents to read, write, and observe shared memory.

    Features:

    - **Auto-causal tracking**: every ``read()`` is logged. ``commit_outcome()``
      auto-links to everything this session read.
    - **Confidence decay**: stale entries lose effective confidence over time.
    - **Rich search**: filter by confidence, agent, recency, pattern refs.
    - **Semantic search**: find entries by meaning using pluggable embedders.
    - **Conflict detection**: detect when another agent modified an entry
      since your last read.
    - **Memory stats**: aggregate introspection for debugging and UIs.

    Usage::

        with AgentMemory(agent_id="review-agent") as mem:
            mem.write("checkout-service", "retry-pattern", {"max_retries": 3})
            entry = mem.read("checkout-service", "retry-pattern")
            mem.commit_outcome("INC-001", OutcomeType.P1_INCIDENT)
    """

    def __init__(
        self,
        agent_id: str,
        *,
        session_id: str | None = None,
        config_path: Path | None = None,
        adapter: AdapterABC | None = None,
        ttl_sweep_interval: float | None = None,
        decay_half_life_days: float | None = None,
        embedder: EmbedderABC | None = None,
        conflict_policy: ConflictPolicy = ConflictPolicy.LAST_WRITE_WINS,
        on_conflict: Callable[[MemoryEntry, MemoryEntry, Any], Any] | None = None,
        importance_evaluator: Any | None = None,
    ) -> None:
        self._config = load_config_or_default(config_path)

        if adapter is not None:
            self._adapter = adapter
        else:
            self._adapter = create_adapter_from_config(self._config)

        self._tagger = CausalTagger(agent_id, session_id)
        self._read_tracker = ReadTracker()
        self._engine = CoWEngine(self._adapter, self._tagger, self._read_tracker)
        self._propagator = OutcomeBackPropagator(self._adapter)
        self._decay_half_life_days = decay_half_life_days
        self._embedder = embedder
        self._conflict_policy = conflict_policy
        self._on_conflict = on_conflict
        self._importance_evaluator = importance_evaluator
        self._branch = "main"
        self._session_metadata: SessionMetadata | None = None
        # Buffered separately from ``_session_metadata`` — which callers (and the
        # MCP server) reassign wholesale — and merged into it at commit time.
        self._session_attributes: dict[str, str | int | float | bool] = {}
        self._session_llm_calls: list[dict[str, Any]] = []

        self._lifecycle: LifecycleManager | None = None
        if ttl_sweep_interval is not None:
            self._lifecycle = LifecycleManager(self._adapter, interval=ttl_sweep_interval)
            self._lifecycle.start()

        self._adapter.ensure_agent(self.agent_id, self._config.namespace)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def agent_id(self) -> str:
        return self._tagger.agent_id

    @property
    def session_id(self) -> str:
        return self._tagger.session_id

    @property
    def session_metadata(self) -> SessionMetadata | None:
        return self._session_metadata

    @session_metadata.setter
    def session_metadata(self, value: SessionMetadata | None) -> None:
        self._session_metadata = value

    @property
    def session_attributes(self) -> dict[str, str | int | float | bool]:
        """The attribute bag the next ``commit_outcome`` will stamp on its trace."""
        return dict(self._session_attributes)

    @property
    def session_llm_calls(self) -> list[dict[str, Any]]:
        """The LLM calls recorded since the last ``commit_outcome``."""
        return [dict(c) for c in self._session_llm_calls]

    @property
    def namespace(self) -> str:
        return self._config.namespace

    @property
    def adapter(self) -> AdapterABC:
        return self._adapter

    @property
    def read_log(self) -> list[str]:
        """Entry keys read during this session (for inspection/debugging)."""
        return self._read_tracker.causal_keys

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def read(
        self,
        entity_path: str,
        key: str,
        *,
        min_confidence: float = 0.0,
        branch: str | None = None,
    ) -> MemoryEntry | None:
        """Read the current version of a key.

        Automatically tracked for causal linking and conflict detection.
        If *decay_half_life_days* is set, applies confidence decay before
        the min_confidence check. Private entries from other agents are
        not visible — use ``recall()`` to access your own private entries.
        """
        import time

        effective_branch = branch or self._branch
        start = time.monotonic()
        try:
            if self._decay_half_life_days is not None:
                entry = self._engine.read(entity_path, key, min_confidence=0.0, branch=effective_branch)
                if entry is None:
                    return None
                if not entry.shared and entry.provenance.agent_id != self.agent_id:
                    return None
                effective = entry.effective_confidence(
                    decay_half_life_days=self._decay_half_life_days,
                )
                if effective < min_confidence:
                    return None
                self._log_read_event(entry, effective_branch)
                return entry
            entry = self._engine.read(entity_path, key, min_confidence=min_confidence, branch=effective_branch)
            if entry is not None and not entry.shared and entry.provenance.agent_id != self.agent_id:
                return None
            if entry is not None:
                self._log_read_event(entry, effective_branch)
            return entry
        except Exception as exc:
            self._read_tracker.record_error(
                "read", type(exc).__name__, str(exc),
            )
            raise

    def write(
        self,
        entity_path: str,
        key: str,
        value: Any,
        *,
        confidence: float = 1.0,
        ttl_at: datetime | None = None,
        pattern_refs: list[str] | None = None,
        memory_type: MemoryType = MemoryType.FACT,
        artifact_refs: list | None = None,
        shared: bool = True,
        branch: str | None = None,
    ) -> MemoryEntry:
        """Write a new version of a key with automatic provenance.

        If *conflict_policy* is ``RAISE``, checks whether the entry was
        modified by another agent since our last read and raises
        ``StaleWriteError`` if so. If an ``on_conflict`` callback is set,
        it is called with ``(our_last_read, current_entry, new_value)``
        and should return the merged value to write.
        """
        effective_branch = branch or self._branch
        entry_key = f"{entity_path}/{key}"
        read_version = self._read_tracker.read_version(entry_key)

        if read_version is not None:
            current = self._adapter.read(entity_path, key)
            if (
                current is not None
                and current.version > read_version
                and current.provenance.agent_id != self.agent_id
            ):
                if self._on_conflict is not None:
                    value = self._on_conflict(
                        current.model_copy(),
                        current,
                        value,
                    )
                    logger.info(
                        "Conflict on %s resolved by on_conflict callback",
                        entry_key,
                    )
                elif self._conflict_policy == ConflictPolicy.RAISE:
                    raise StaleWriteError(
                        entity_path,
                        key,
                        read_version,
                        current.version,
                        current.provenance.agent_id,
                    )

        # Artifacts embed a clean descriptor (filename + symbols) rather than the
        # raw, 512-token-truncated blob, so code files stop matching generic queries.
        # The adapter is the authoritative place the is_artifact flag is persisted.
        _, embed_text = embedding_input(key, value)
        embedding = None
        if self._embedder is not None:
            embedding = self._embedder.embed(embed_text)

        importance_score = None
        importance_dimensions = None
        if self._importance_evaluator is not None:
            try:
                importance_score, importance_dimensions = self._importance_evaluator.evaluate(
                    value,
                    entity_path=entity_path,
                    key=key,
                )
            except Exception:
                logger.debug("Importance evaluation failed, skipping", exc_info=True)

        entry = self._engine.write(
            entity_path,
            key,
            value,
            confidence=confidence,
            ttl_at=ttl_at,
            pattern_refs=pattern_refs,
            memory_type=memory_type,
            artifact_refs=artifact_refs,
            shared=shared,
            branch=effective_branch,
            embedding=embedding,
            importance_score=importance_score,
            importance_dimensions=importance_dimensions or None,
        )
        self._read_tracker.record_write(entity_path, key, entry.version, entry.version == 1)

        _adapter = self._adapter
        _ns = self.namespace
        _aid = self.agent_id
        _ev_branch = effective_branch
        _entry_version = entry.version
        _entry_confidence = entry.confidence
        _entry_mt = entry.memory_type.value if hasattr(entry.memory_type, "value") else str(entry.memory_type)
        _entry_shared = entry.shared
        _ep = entity_path
        _k = key
        _prefs = pattern_refs

        def _bg_log_and_edges() -> None:
            # Skip client-side WRITE logging when the adapter delegates to a
            # remote server that already records the event (HttpAdapter) —
            # otherwise a single write produces two identical timeline events.
            if not getattr(_adapter, "server_side_write_events", False):
                try:
                    _adapter.log_event(Event(
                        namespace=_ns,
                        agent_id=_aid,
                        branch=_ev_branch,
                        event_type=EventType.WRITE,
                        summary=f"Wrote {_ep}/{_k} v{_entry_version}",
                        details={
                            "entity_path": _ep,
                            "key": _k,
                            "version": _entry_version,
                            "confidence": _entry_confidence,
                            "memory_type": _entry_mt,
                            "shared": _entry_shared,
                        },
                    ))
                except Exception:
                    logger.debug("Failed to log write event", exc_info=True)
            if _prefs:
                try:
                    self._materialize_pattern_ref_edges(_ep, _k, _prefs, _ev_branch)
                except Exception:
                    logger.debug("Failed to materialize pattern ref edges", exc_info=True)

        _get_sdk_executor().submit(_bg_log_and_edges)

        return entry

    def _materialize_pattern_ref_edges(
        self,
        entity_path: str,
        key: str,
        pattern_refs: list[str],
        branch: str,
    ) -> None:
        """Best-effort: create graph edges from pattern_refs."""
        for ref in pattern_refs:
            try:
                self._adapter.upsert_graph_edge(
                    GraphEdge(
                        source_entity=f"{entity_path}/{key}",
                        source_type="entry",
                        relation="references",
                        target_entity=ref,
                        target_type="entry",
                        provenance={"agent_id": self.agent_id, "trigger": "write"},
                    ),
                    namespace=self.namespace,
                    branch=branch,
                )
            except Exception:
                logger.debug("Failed to materialize pattern_ref edge for %s", ref, exc_info=True)

    def list(
        self,
        entity_path: str | None = None,
        *,
        include_superseded: bool = False,
        branch: str | None = None,
    ) -> list[MemoryEntry]:
        """List current entries, optionally filtered to an entity path.

        Private entries from other agents are excluded.
        """
        import time
        effective_branch = branch or self._branch
        start = time.monotonic()
        results = self._engine.list(entity_path, include_superseded=include_superseded, branch=effective_branch)
        results = [
            e for e in results
            if e.shared or e.provenance.agent_id == self.agent_id
        ]
        duration = (time.monotonic() - start) * 1000
        self._read_tracker.record_query(
            "list",
            {"entity_path": entity_path, "include_superseded": include_superseded},
            len(results),
            duration,
        )
        return results

    def watch(
        self,
        entity_path: str,
        callback: Any,
    ) -> WatchHandle:
        """Watch for writes to any key under an entity path."""
        return self._adapter.watch(entity_path, callback)

    # ------------------------------------------------------------------
    # Search & Stats
    # ------------------------------------------------------------------

    def search(
        self,
        *,
        query: str | None = None,
        entity_path: str | None = None,
        entity_paths: list[str] | None = None,
        min_confidence: float = 0.0,
        max_confidence: float | None = None,
        agent_id: str | None = None,
        since: datetime | None = None,
        pattern_ref: str | None = None,
        limit: int = 100,
        sort_by: str = "confidence",
        recall_config: RecallConfig | None = None,
        depth: int = 3,
        include_artifacts: bool = True,
    ) -> list[MemoryEntry] | list[ScoredEntry]:
        """Search across all entities with rich filters.

        When *query* is provided the text is forwarded to the adapter for
        full-text search (Postgres tsvector) and, when *recall_config* is
        also set, used for cosine-similarity scoring against entry embeddings.

        When *entity_paths* is provided, runs a search for each path and merges
        the results.  *entity_path* (singular) is still supported for backwards
        compatibility; if both are given, *entity_paths* takes precedence.

        When *recall_config* is provided, returns ``ScoredEntry`` objects
        sorted by composite recall score instead.

        *depth* controls progressive retrieval across memory tiers:
          1 = HOT only, 2 = HOT + WARM, 3 = all tiers (default).
        """
        from amfs_core.embedder import cosine_similarity

        paths = entity_paths or ([entity_path] if entity_path else [None])

        seen_keys: set[str] = set()
        merged: list[MemoryEntry] = []
        for ep in paths:
            sq = SearchQuery(
                query=query,
                entity_path=ep,
                min_confidence=min_confidence,
                max_confidence=max_confidence,
                agent_id=agent_id,
                since=since,
                pattern_ref=pattern_ref,
                limit=limit,
                sort_by=sort_by,
                recall_config=recall_config,
                depth=depth,
                include_artifacts=include_artifacts,
            )
            for entry in self._adapter.search(sq):
                if entry.entry_key not in seen_keys:
                    if not entry.shared and entry.provenance.agent_id != self.agent_id:
                        continue
                    seen_keys.add(entry.entry_key)
                    merged.append(entry)

        if sort_by != "priority":
            sort_key: Callable[[MemoryEntry], Any]
            if sort_by == "recency":
                sort_key = lambda e: e.provenance.written_at
            elif sort_by == "version":
                sort_key = lambda e: e.version
            else:
                sort_key = lambda e: e.confidence
            merged.sort(key=sort_key, reverse=True)
        entries = merged[:limit]

        self._read_tracker.record_query(
            "search",
            {"entity_path": entity_path, "min_confidence": min_confidence, "limit": limit, "sort_by": sort_by},
            len(entries),
        )

        if recall_config is None:
            return entries

        query_vec: list[float] | None = None
        if query and self._embedder is not None:
            query_vec = self._embedder.embed(query)

        now = datetime.now(timezone.utc)
        scored: list[ScoredEntry] = []
        for entry in entries:
            age = now - entry.provenance.written_at
            age_days = age.total_seconds() / 86400.0
            half_life = recall_config.recency_half_life_days

            recency_score = math.exp(-math.log(2) * age_days / half_life) if half_life > 0 else 0.0
            confidence_score = max(0.0, min(1.0, entry.confidence))

            semantic_score = 0.0
            if query_vec is not None and entry.embedding is not None:
                semantic_score = max(0.0, cosine_similarity(query_vec, entry.embedding))

            composite = (
                recall_config.semantic_weight * semantic_score
                + recall_config.recency_weight * recency_score
                + recall_config.confidence_weight * confidence_score
            )
            scored.append(ScoredEntry(
                entry=entry,
                score=composite,
                breakdown={
                    "semantic": recall_config.semantic_weight * semantic_score,
                    "recency": recall_config.recency_weight * recency_score,
                    "confidence": recall_config.confidence_weight * confidence_score,
                },
            ))

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored

    def semantic_search(
        self,
        text: str,
        *,
        entity_path: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 10,
        min_similarity: float = 0.0,
    ) -> list[tuple[MemoryEntry, float]]:
        """Search entries by meaning. Requires an embedder to be configured.

        Returns ``(entry, similarity_score)`` tuples sorted by similarity.
        """
        if self._embedder is None:
            raise RuntimeError(
                "semantic_search() requires an embedder. "
                "Pass embedder= to AgentMemory()."
            )
        query = SemanticQuery(
            text=text,
            entity_path=entity_path,
            min_confidence=min_confidence,
            limit=limit,
            min_similarity=min_similarity,
        )
        return self._adapter.semantic_search(query, self._embedder)

    def retrieve(
        self,
        query: str,
        *,
        entity_path: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 10,
        recall_config: RecallConfig | None = None,
        include_artifacts: bool = True,
    ) -> list[ScoredEntry]:
        """Rank memories by meaning for a natural-language query.

        Prefers server-side retrieval when the adapter supports it (e.g. the
        HTTP adapter, where the server owns the embedder + pgvector index), so
        semantic recall works even when this client has no local embedder.
        Otherwise falls back to client-side composite scoring via ``search``.

        Artifacts (stored source files) are demoted by default; pass
        ``include_artifacts=False`` to exclude them entirely.
        """
        cfg = recall_config or RecallConfig()

        adapter_retrieve = getattr(self._adapter, "retrieve", None)
        if callable(adapter_retrieve):
            try:
                rows = adapter_retrieve(
                    query,
                    entity_path=entity_path,
                    min_confidence=min_confidence,
                    limit=limit,
                    semantic_weight=cfg.semantic_weight,
                    recency_weight=cfg.recency_weight,
                    confidence_weight=cfg.confidence_weight,
                    include_artifacts=include_artifacts,
                )
                scored = [
                    ScoredEntry(entry=entry, score=score, breakdown=breakdown or {})
                    for entry, score, breakdown in rows
                ]
                self._record_retrieval_reuse(query, entity_path, scored)
                return scored
            except Exception:  # noqa: BLE001 - fall back to local scoring
                logger.debug("Adapter server-side retrieve failed; using local scoring", exc_info=True)

        result = self.search(
            query=query,
            entity_path=entity_path,
            min_confidence=min_confidence,
            limit=limit,
            recall_config=cfg,
            include_artifacts=include_artifacts,
        )
        self._record_retrieval_reuse(query, entity_path, result)  # type: ignore[arg-type]
        return result  # type: ignore[return-value]

    def _record_retrieval_reuse(
        self,
        query: str,
        entity_path: str | None,
        results: list[ScoredEntry],
    ) -> None:
        """Link a successful retrieval into the session's causal chain.

        Retrieval is how agents mostly recall memory, but it never entered the
        read tracker — so ``commit_outcome`` saw an empty causal chain, traces
        recorded no reads, and no outcome ever reinforced the memory that
        actually drove it.

        Only the top-ranked entry becomes a causal read. Retrieval returns a
        ranked candidate list, and back-propagating an outcome onto every
        candidate would reinforce memories the agent never acted on and undo the
        bounded-causal-set assumption commit_outcome relies on for speed. The
        full result set is still preserved as a query event.

        This is deliberately separate from ``recall_count``, which the server
        credits to the top-K surfaced hits: reuse counts what was shown, causal
        linkage records what an outcome should reinforce.
        """
        self._read_tracker.record_query(
            "retrieve",
            {"query": query, "entity_path": entity_path},
            len(results),
        )
        if results:
            self._read_tracker.record(results[0].entry)

    def stats(self) -> MemoryStats:
        """Aggregate statistics about current memory state."""
        return self._adapter.stats()

    def verify(self, entity_path: str | None = None) -> dict:
        """Verify content integrity of stored entries.

        Checks that each entry's ``content_hash`` matches its value and
        that ``integrity_chain`` links are consistent across versions.
        Returns an IntegrityReport dict with ``total_checked``, ``valid``,
        ``corrupted``, and ``chain_breaks``.
        """
        return self._adapter.verify_integrity(entity_path, branch=self._branch)

    # ------------------------------------------------------------------
    # Atomic commits / transactions
    # ------------------------------------------------------------------

    def transaction(self, message: str = "") -> "TransactionContext":
        """Create an atomic transaction that groups multiple writes.

        Usage::

            with mem.transaction("update configs") as tx:
                tx.write("repo/svc", "key1", {"data": 1})
                tx.write("repo/svc", "key2", {"data": 2})
            # all writes committed atomically on exit
        """
        from amfs_core.transaction import TransactionBuffer

        buf = TransactionBuffer(
            agent_id=self.agent_id,
            session_id=self._tagger.session_id,
            branch=self._branch,
            namespace=self._config.namespace,
        )
        buf.set_message(message)
        return TransactionContext(buf, self._adapter, self._tagger)

    def commit_log(self, *, limit: int = 50) -> list[Commit]:
        """Retrieve the commit log for the current branch, newest first."""
        return self._adapter.list_commits(
            branch=self._branch,
            limit=limit,
            namespace=self._config.namespace,
        )

    def get_commit(self, commit_id: str) -> Commit | None:
        """Retrieve a single commit by ID."""
        return self._adapter.get_commit(commit_id)

    def common_ancestor(self, commit_a_id: str, commit_b_id: str) -> str | None:
        """Find the most recent common ancestor of two commits.

        Delegated to the adapter, which holds the same breadth-first walk as its
        default. The indirection is so that a remote store can answer in one
        call instead of one call per commit visited — see
        ``AdapterABC.common_ancestor``.

        The walk stays here as a fallback for an adapter that predates that
        method. Every adapter in this repository subclasses ``AdapterABC`` and
        so inherits it, but this is a published interface and an adapter written
        against the old one only had to provide ``get_commit``.
        """
        delegate = getattr(self._adapter, "common_ancestor", None)
        if delegate is not None:
            return delegate(commit_a_id, commit_b_id)

        from amfs_core.dag import find_common_ancestor

        return find_common_ancestor(
            commit_a_id,
            commit_b_id,
            self._adapter.get_commit,
        )

    # ------------------------------------------------------------------
    # Agent binding
    # ------------------------------------------------------------------

    def set_profile(
        self,
        description: str = "",
        *,
        default_branch: str | None = None,
        auto_context_paths: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Set this agent's profile (description, defaults, tags)."""
        from amfs_core.models import AgentProfile

        profile = AgentProfile(
            description=description,
            default_branch=default_branch or self._branch,
            auto_context_paths=auto_context_paths or [],
            tags=tags or [],
        )
        self._adapter.update_agent_profile(
            self.agent_id, profile, namespace=self._config.namespace,
        )

    def declare_capability(
        self,
        name: str,
        description: str = "",
        entity_paths: list[str] | None = None,
    ) -> None:
        """Declare a capability for this agent."""
        from amfs_core.models import AgentCapability

        agent = self._adapter.get_agent(self.agent_id, self._config.namespace)
        existing = list(agent.capabilities) if agent else []
        existing = [c for c in existing if c.name != name]
        existing.append(AgentCapability(
            name=name,
            description=description,
            entity_paths=entity_paths or [],
        ))
        self._adapter.update_agent_capabilities(
            self.agent_id, existing, namespace=self._config.namespace,
        )

    def set_contracts(self, contracts: list[dict]) -> None:
        """Set memory contracts for this agent."""
        from amfs_core.models import MemoryContract

        parsed = [MemoryContract.model_validate(c) for c in contracts]
        self._adapter.update_agent_contracts(
            self.agent_id, parsed, namespace=self._config.namespace,
        )

    def discover_agents(
        self,
        *,
        capability: str | None = None,
        entity_path: str | None = None,
    ) -> list:
        """Discover other agents by capability or entity path."""
        return self._adapter.discover_agents(
            capability=capability,
            entity_path=entity_path,
            namespace=self._config.namespace,
        )

    # ------------------------------------------------------------------
    # Diff & patch
    # ------------------------------------------------------------------

    def diff(
        self,
        entity_path: str,
        key: str,
        old_version: int | None = None,
    ) -> dict:
        """Compute a structural diff for a key.

        If old_version is specified, diffs between that version and current.
        Otherwise diffs between the two most recent versions.
        """
        from amfs_core.diff import diff_entries

        entries = [
            e for e in self._adapter.list(entity_path, include_superseded=True)
            if e.key == key
        ]
        entries.sort(key=lambda e: e.version)

        if len(entries) < 2:
            return {"diff_type": "no_diff", "reason": "fewer than 2 versions"}

        if old_version is not None:
            old = next((e for e in entries if e.version == old_version), None)
        else:
            old = entries[-2]
        new = entries[-1]

        if old is None:
            return {"diff_type": "no_diff", "reason": "old version not found"}

        diff = diff_entries(old, new)
        return diff.model_dump(mode="json")

    def create_patch(
        self,
        entity_path: str,
        key: str,
        source_version: int | None = None,
    ) -> dict:
        """Create a serializable patch between two versions."""
        from amfs_core.diff import create_patch

        entries = [
            e for e in self._adapter.list(entity_path, include_superseded=True)
            if e.key == key
        ]
        entries.sort(key=lambda e: e.version)

        if len(entries) < 2:
            return {"error": "fewer than 2 versions"}

        if source_version is not None:
            old = next((e for e in entries if e.version == source_version), None)
        else:
            old = entries[-2]
        new = entries[-1]

        if old is None:
            return {"error": "source version not found"}

        patch = create_patch(old, new)
        return patch.model_dump(mode="json")

    # ------------------------------------------------------------------
    # Knowledge graph
    # ------------------------------------------------------------------

    def graph_neighbors(
        self,
        entity: str,
        *,
        relation: str | None = None,
        direction: str = "both",
        min_confidence: float = 0.0,
        depth: int = 1,
        limit: int = 50,
    ) -> list[GraphEdge]:
        """Traverse the knowledge graph from an entity."""
        query = GraphNeighborQuery(
            entity=entity,
            relation=relation,
            direction=direction,
            min_confidence=min_confidence,
            depth=depth,
            limit=limit,
        )
        return self._adapter.graph_neighbors(
            query,
            namespace=self.namespace,
            branch=self._branch,
        )

    # ------------------------------------------------------------------
    # Outcomes
    # ------------------------------------------------------------------

    def _scanned_actions(
        self, explicit: list[dict[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        """Recorded actions, with arguments and results cleared for secrets.

        An action whose arguments cannot be cleared is left out entirely rather
        than included with a hole in it, since a tool call missing a parameter is
        still a plausible-looking training example and would teach the wrong call.
        A result that cannot be cleared only costs the result, which is context
        rather than the target.

        *explicit* is used verbatim when given, including when empty. The HTTP
        server relays a client's actions through this method on a shared ``mem``
        whose own tracker belongs to no particular caller, so falling back to the
        tracker on an empty list would attribute one session's actions to another.

        Each action is validated here and returned in JSON-safe form, so it is also
        the one place a malformed action can be turned away. Actions arriving over
        HTTP are free-form JSON from the caller, and one that failed validation
        further down would take the whole commit with it, losing the memories being
        written alongside it.
        """
        source = self._read_tracker.actions if explicit is None else explicit
        scanned: list[dict[str, Any]] = []
        for action in source:
            if not isinstance(action, dict) or not action.get("tool_name"):
                continue
            arguments = scan_captured_arguments(
                action.get("arguments"),
                adapter=self._adapter,
                agent_id=self.agent_id,
                session_id=self.session_id,
            )
            if arguments is None:
                continue
            # The result is caller text exactly as much as the arguments are — a
            # tool that mints a credential echoes it back here. Unlike a dropped
            # argument, a dropped result costs nothing: the training target is the
            # tool and its parameters, so the action is kept without its result
            # rather than discarded.
            summary = scan_captured_text(
                action.get("result_summary"),
                adapter=self._adapter,
                agent_id=self.agent_id,
                session_id=self.session_id,
            )
            fields = {
                **action,
                "arguments": arguments,
                "result_summary": summary or "",
            }
            # A JSON caller sending an explicit null means "not told", which for a
            # success flag is the default rather than a reason to drop the action.
            if fields.get("success") is None:
                fields.pop("success", None)
            try:
                validated = ToolCall(**fields)
            except ValidationError:
                continue
            scanned.append(validated.model_dump(mode="json"))
        return scanned

    def commit_outcome(
        self,
        outcome_ref: str,
        outcome_type: OutcomeType,
        causal_entry_keys: list[str] | None = None,
        *,
        causal_confidence: float = 1.0,
        decision_summary: str | None = None,
        task_input: str | None = None,
        response_text: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        attributes: dict[str, Any] | None = None,
        llm_calls: list[dict[str, Any]] | None = None,
    ) -> list[MemoryEntry]:
        """Record an outcome and back-propagate confidence changes.

        If *causal_entry_keys* is ``None``, automatically uses the session's
        read log — every entry this agent read becomes a causal link.

        *task_input* is the request that triggered the decision and
        *response_text* the agent's answer. Both are optional and only stored
        when supplied, and both are scanned for secrets before being persisted —
        a value the safety gate blocks is dropped rather than stored.

        *tool_calls* are the actions taken. Like *causal_entry_keys* it defaults
        to the session's own log, filled in by ``record_action``; pass a list to
        supply them directly, which is what the HTTP server does when relaying a
        remote client's actions.

        *attributes* are merged over the bag built by ``set_session_attributes``
        and stored in ``session_metadata["attributes"]`` — the dimensions the
        trace can later be filtered and grouped by. *llm_calls* are appended to
        the calls recorded with ``record_llm_call`` and stored in
        ``session_metadata["llm_calls"]``; like *tool_calls*, the explicit form
        exists for a server relaying a remote client's session. Both buffers are
        cleared once the trace is built.
        """
        # Validated first so a bad bag fails before anything is written.
        commit_attributes = validate_session_attributes(attributes)
        explicit_calls = [
            c for c in (_normalize_llm_call(lc) for lc in (llm_calls or [])) if c is not None
        ]
        # Built here, before the record leaves for the adapter: the server seals
        # its trace from the outcome call, so the bag has to travel on it. This
        # is also where the merged bag (session metadata + set_session_attributes
        # + *attributes*) is checked against the key cap, so an oversized merge
        # raises before anything is sent.
        session_metadata = self._session_metadata_for_trace(commit_attributes, explicit_calls)

        if causal_entry_keys is None:
            causal_entry_keys = self._read_tracker.causal_keys

        # Scanned here, before anything leaves the process, and reused for both the
        # outcome record and the trace below. Scanning only at trace construction
        # would send the raw text to the server on the SaaS path, since the record
        # goes over the wire first.
        task_input = scan_captured_text(
            task_input,
            adapter=self._adapter,
            agent_id=self.agent_id,
            session_id=self.session_id,
        )
        response_text = scan_captured_text(
            response_text,
            adapter=self._adapter,
            agent_id=self.agent_id,
            session_id=self.session_id,
        )
        # Scanned here for the same reason and at the same moment as the capture
        # above: the record leaves for the server before the trace is built, so a
        # scan deferred to trace construction would ship raw arguments over the
        # wire on the SaaS path.
        tool_calls = self._scanned_actions(tool_calls)

        record = OutcomeBackPropagator.make_record(
            outcome_ref=outcome_ref,
            outcome_type=outcome_type,
            causal_entry_keys=causal_entry_keys,
            agent_id=self.agent_id,
            causal_confidence=causal_confidence,
            # Carried on the record because the adapter's commit_outcome is what
            # reaches the server, and the server seals its immutable trace from
            # that call. Sending the capture only on the later save_trace left the
            # sealed copy — the one export and training read — without it.
            task_input=task_input,
            response_text=response_text,
            tool_calls=tool_calls,
            session_metadata=_metadata_to_dict(session_metadata) or None,
        )
        updated = self._propagator.propagate(record)

        causal_trace_entries: list[TraceEntry] = []
        for ek in causal_entry_keys:
            parts = ek.rsplit("/", 1)
            if len(parts) != 2:
                continue
            ep, k = parts
            snapshot = self._read_tracker.entry_snapshot(ek)
            if snapshot:
                causal_trace_entries.append(TraceEntry(
                    entity_path=ep, key=k,
                    version=self._read_tracker.read_version(ek) or snapshot.get("version", 1),
                    confidence=snapshot.get("confidence", 1.0),
                    value=snapshot.get("value"),
                    memory_type=snapshot.get("memory_type"),
                    written_by=snapshot.get("written_by"),
                    read_at=self._read_tracker._reads.get(ek),
                ))
            else:
                entry = self._adapter.read(ep, k)
                if entry:
                    causal_trace_entries.append(TraceEntry(
                        entity_path=ep, key=k,
                        version=self._read_tracker.read_version(ek) or entry.version,
                        confidence=entry.confidence,
                        value=entry.value,
                        memory_type=entry.memory_type.value if entry.memory_type else None,
                        written_by=entry.provenance.agent_id,
                        read_at=self._read_tracker._reads.get(ek),
                    ))

        ext_contexts = [
            ExternalContext(
                label=c.get("label", ""),
                summary=c.get("summary", ""),
                source=c.get("source"),
                recorded_at=datetime.fromisoformat(c["recorded_at"]) if c.get("recorded_at") else datetime.now(timezone.utc),
            )
            for c in self._read_tracker.external_contexts
        ]

        now = datetime.now(timezone.utc)
        query_events = [
            QueryEvent(
                operation=q.get("operation", ""),
                parameters=q.get("parameters", {}),
                result_count=q.get("result_count", 0),
                duration_ms=q.get("duration_ms"),
                occurred_at=datetime.fromisoformat(q["occurred_at"]) if q.get("occurred_at") else now,
            )
            for q in self._read_tracker.query_events
        ]

        session_started = self._read_tracker.session_started_at
        session_duration = (now - session_started).total_seconds() * 1000

        error_events = [
            ErrorEvent(
                operation=e.get("operation", ""),
                error_type=e.get("error_type", ""),
                message=e.get("message", ""),
                stack_trace=e.get("stack_trace"),
                occurred_at=datetime.fromisoformat(e["occurred_at"]) if e.get("occurred_at") else now,
            )
            for e in self._read_tracker.error_events
        ]

        writes = self._read_tracker.write_events
        state_diff = MemoryStateDiff(
            entries_created=sum(1 for w in writes if w.get("is_new")),
            entries_updated=sum(1 for w in writes if not w.get("is_new")),
        )

        trace = DecisionTrace(
            agent_id=self.agent_id,
            session_id=self.session_id,
            outcome_ref=outcome_ref,
            outcome_type=outcome_type.value,
            decision_summary=decision_summary,
            # Already scanned at the top of this method, before the record went to
            # the adapter. Scanning is centralised here rather than left to callers
            # because the MCP tool calls straight through and its documentation
            # promises the text has been redacted.
            task_input=task_input,
            response_text=response_text,
            tool_calls=[ToolCall(**tc) for tc in tool_calls],
            causal_entries=causal_trace_entries,
            external_contexts=ext_contexts,
            query_events=query_events,
            session_metadata=session_metadata,
            session_started_at=session_started,
            session_ended_at=now,
            session_duration_ms=session_duration,
            error_events=error_events,
            state_diff=state_diff,
        )
        try:
            trace = self._adapter.save_trace(trace)
        except Exception:
            logger.debug("Failed to persist decision trace", exc_info=True)

        executor = _get_sdk_executor()
        adapter = self._adapter
        agent_id = self.agent_id
        namespace = self.namespace
        branch = self._branch
        n_updated = len(updated)
        n_causal = len(causal_entry_keys)

        def _bg_log_and_edges() -> None:
            try:
                adapter.log_event(Event(
                    namespace=namespace,
                    agent_id=agent_id,
                    branch=branch,
                    event_type=EventType.OUTCOME,
                    summary=f"Committed outcome '{outcome_ref}' ({outcome_type.value})",
                    details={
                        "outcome_ref": outcome_ref,
                        "outcome_type": outcome_type.value,
                        "causal_entries": n_causal,
                        "entries_updated": n_updated,
                    },
                ))
            except Exception:
                logger.debug("Failed to log outcome event", exc_info=True)
            self._materialize_causal_edges(outcome_ref, outcome_type, causal_entry_keys)

        executor.submit(_bg_log_and_edges)

        self._last_trace = trace
        # Start a fresh causal window. The reads, contexts and queries just
        # snapshotted belong to this outcome; without this a second
        # commit_outcome in the same session re-links the first task's reads and
        # reinforces them again, and the trace's duration keeps growing from
        # process start rather than describing the work it covers.
        self._read_tracker.clear()
        self._session_attributes = {}
        self._session_llm_calls = []
        return updated

    def _session_metadata_for_trace(
        self,
        commit_attributes: dict[str, Any],
        explicit_calls: list[dict[str, Any]],
    ) -> SessionMetadata | None:
        """The metadata the trace carries: the session's own, with the attribute
        bag and LLM calls merged in under ``attributes`` / ``llm_calls``.

        Built from a dump of the current metadata so extras already on it — a
        span tree a Pro recorder placed there, say — are kept, and so the trace
        holds its own instance rather than the one the session keeps mutating.
        ``None`` stays ``None`` when there is nothing to add.

        Raises ``ValueError`` when the merged bag exceeds
        ``SESSION_ATTRIBUTES_MAX_KEYS``: each source is capped on its own, so
        only the merge can tell whether the trace's bag is within the limit.
        """
        data = _metadata_to_dict(self._session_metadata)
        existing_attrs = data.get(SESSION_ATTRIBUTES_KEY)
        attributes = {
            **(existing_attrs if isinstance(existing_attrs, dict) else {}),
            **self._session_attributes,
            **commit_attributes,
        }
        _check_attribute_count(attributes, "the merged session attributes")
        existing_calls = data.get(SESSION_LLM_CALLS_KEY)
        llm_calls = [
            *(existing_calls if isinstance(existing_calls, list) else []),
            *self._session_llm_calls,
            *explicit_calls,
        ]
        if attributes:
            data[SESSION_ATTRIBUTES_KEY] = attributes
        else:
            data.pop(SESSION_ATTRIBUTES_KEY, None)
        if llm_calls:
            data[SESSION_LLM_CALLS_KEY] = llm_calls
        else:
            data.pop(SESSION_LLM_CALLS_KEY, None)
        if not data:
            return None
        try:
            return SessionMetadata(**data)
        except (TypeError, ValidationError):
            logger.debug("session metadata could not be rebuilt; sending it as-is", exc_info=True)
            return self._session_metadata

    def set_session_attributes(
        self, attributes: dict[str, Any]
    ) -> dict[str, str | int | float | bool]:
        """Merge *attributes* into the bag the next ``commit_outcome`` stamps on
        its trace (``session_metadata["attributes"]``). Returns the bag as it
        now stands.

        Attributes are the dimensions a run is later filtered and grouped by —
        ``customer``, ``task_type``, ``environment``. Scalars only (``str``,
        ``int``, ``float``, ``bool``), at most 20 keys of up to 64 characters,
        string values up to 256 characters; anything else raises. Keys are
        lowercased. The bag is cleared when the trace is committed.

        The key cap applies to the bag as merged, not to each call: a merge that
        would take it past 20 keys raises ``ValueError`` and leaves the bag as
        it was.

        Example::

            mem.set_session_attributes({"customer": "acme", "task_type": "deploy"})
        """
        validated = validate_session_attributes(attributes)
        merged = {**self._session_attributes, **validated}
        _check_attribute_count(merged, "the session attribute bag")
        self._session_attributes = merged
        return dict(self._session_attributes)

    def record_llm_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        *,
        cost_usd: float | None = None,
        latency_ms: float | None = None,
        provider: str | None = None,
        call_id: str | None = None,
    ) -> dict[str, Any]:
        """Record one LLM call for the trace the next ``commit_outcome`` builds.

        Token counts and cost are only ever known if the agent reports them, so
        this is what makes a trace's duration, token and cost figures real
        rather than blank. Stored in ``session_metadata["llm_calls"]`` as::

            {"call_id", "model", "provider", "input_tokens", "output_tokens",
             "cost_usd", "latency_ms", "started_at"}

        Returns the record as stored. Cleared when the trace is committed.

        Example::

            resp = client.chat.completions.create(...)
            mem.record_llm_call(
                resp.model, resp.usage.prompt_tokens, resp.usage.completion_tokens,
                provider="openai", latency_ms=412.0,
            )
        """
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a non-empty string")
        try:
            in_tokens = int(input_tokens)
            out_tokens = int(output_tokens)
        except (TypeError, ValueError) as exc:
            raise TypeError("input_tokens and output_tokens must be integers") from exc
        if in_tokens < 0 or out_tokens < 0:
            raise ValueError("token counts must be non-negative")
        for label, value in (("cost_usd", cost_usd), ("latency_ms", latency_ms)):
            if value is not None and (
                not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(float(value)) or value < 0
            ):
                raise ValueError(f"{label} must be a non-negative number or None")
        record = _normalize_llm_call({
            "call_id": call_id,
            "model": model,
            "provider": provider or "",
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "cost_usd": cost_usd,
            "latency_ms": latency_ms,
            "started_at": datetime.now(UTC),
        })
        if record is None:  # pragma: no cover - a named model always normalises
            raise ValueError("could not build an LLM call record")
        self._session_llm_calls.append(record)
        return dict(record)

    def _materialize_causal_edges(
        self,
        outcome_ref: str,
        outcome_type: OutcomeType,
        causal_entry_keys: list[str],
    ) -> None:
        """Best-effort: create graph edges from the causal chain."""
        edge_conf = 1.0 if outcome_type in (OutcomeType.SUCCESS,) else 0.7
        branch = self._branch
        for ek in causal_entry_keys:
            try:
                self._adapter.upsert_graph_edge(
                    GraphEdge(
                        source_entity=ek,
                        source_type="entry",
                        relation="informed",
                        target_entity=outcome_ref,
                        target_type="outcome",
                        confidence=edge_conf,
                        provenance={"agent_id": self.agent_id, "trigger": "commit_outcome"},
                    ),
                    namespace=self.namespace,
                    branch=branch,
                )
                self._adapter.upsert_graph_edge(
                    GraphEdge(
                        source_entity=self.agent_id,
                        source_type="agent",
                        relation="read",
                        target_entity=ek,
                        target_type="entry",
                        confidence=edge_conf,
                        provenance={"agent_id": self.agent_id, "trigger": "commit_outcome"},
                    ),
                    namespace=self.namespace,
                    branch=branch,
                )
            except Exception:
                logger.debug("Failed to materialize causal edge for %s", ek, exc_info=True)

        for i, a in enumerate(causal_entry_keys):
            for b in causal_entry_keys[i + 1:]:
                try:
                    self._adapter.upsert_graph_edge(
                        GraphEdge(
                            source_entity=a,
                            source_type="entry",
                            relation="co_occurs_with",
                            target_entity=b,
                            target_type="entry",
                            confidence=edge_conf,
                            provenance={"agent_id": self.agent_id, "trigger": "commit_outcome"},
                        ),
                        namespace=self.namespace,
                        branch=branch,
                    )
                except Exception:
                    logger.debug("Failed to materialize co-occurrence edge", exc_info=True)

    # ------------------------------------------------------------------
    # Temporal & Explainability
    # ------------------------------------------------------------------

    def history(
        self,
        entity_path: str,
        key: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[MemoryEntry]:
        """Return the full version history of a key, ordered by version.

        Enables temporal queries like "how did this memory change over time?"
        Each entry in the returned list is a CoW snapshot with its confidence
        and provenance at the time it was written.
        """
        return self._engine.history(entity_path, key, since=since, until=until, branch=self._branch)

    def record_context(
        self,
        label: str,
        summary: str,
        *,
        source: str | None = None,
    ) -> None:
        """Record external context in the causal chain without writing to storage.

        Call this after consulting an external tool, API, or data source so
        that ``explain()`` returns a complete decision trace — not just which
        AMFS entries were read, but also which external inputs informed the
        agent's decisions.

        Example::

            mem.record_context(
                "pagerduty-incidents",
                "3 SEV-1 incidents in the last 24h for checkout-service",
                source="PagerDuty API",
            )
        """
        self._read_tracker.record_context(label, summary, source=source)

    def record_action(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        *,
        result: str = "",
        source: str | None = None,
        duration_ms: int = 0,
        success: bool = True,
    ) -> None:
        """Record an action taken during this session, sealed into the trace on commit.

        ``record_context`` captures what the agent learned; this captures what it
        did. Recorded by the caller because AMFS observes only its own tools — a
        call to your deploy or refund tool is invisible to it otherwise.

        Together with ``task_input`` on ``commit_outcome`` this forms a complete
        supervised example: the request in, the action out. Arguments are scanned
        for secrets when the trace is built, and an action whose arguments cannot
        be cleared is dropped whole.

        Example::

            mem.record_action(
                "deploy_rollback",
                {"service": "checkout", "to_version": "v41"},
                result='{"status": "rolled_back"}',
                duration_ms=1430,
            )
        """
        self._read_tracker.record_action(
            tool_name,
            arguments,
            result=result,
            source=source,
            duration_ms=duration_ms,
            success=success,
        )

    def explain(self, outcome_ref: str | None = None) -> dict[str, Any]:
        """Return the causal chain for the current session or a specific outcome.

        Shows which memories were read (and in what order) before the outcome
        was committed, plus any external contexts recorded via
        ``record_context()``.  This is production-grounded explainability:
        not what the LLM inferred, but which stored knowledge and external
        inputs actually drove the decision.
        """
        causal_keys = self._read_tracker.causal_keys
        entries: list[dict[str, Any]] = []
        for ek in causal_keys:
            parts = ek.rsplit("/", 1)
            if len(parts) != 2:
                continue
            ep, k = parts
            # Serve from the read-time snapshot. Re-reading through the adapter
            # would both misreport what the agent actually saw and, over HTTP,
            # bump recall_count — making this diagnostic inflate the reuse
            # metric it exists to explain. Mirrors commit_outcome's snapshot-first
            # path.
            snapshot = self._read_tracker.entry_snapshot(ek)
            if snapshot:
                entries.append({
                    "entity_path": ep,
                    "key": k,
                    "value": snapshot.get("value"),
                    "confidence": snapshot.get("confidence"),
                    "version": snapshot.get("version"),
                    "memory_type": snapshot.get("memory_type"),
                    "written_by": snapshot.get("written_by"),
                    "read_version": self._read_tracker.read_version(ek),
                })
                continue
            entry = self._adapter.read(ep, k)
            if entry:
                data = entry.model_dump(mode="json")
                data.pop("embedding", None)
                data["read_version"] = self._read_tracker.read_version(ek)
                entries.append(data)
        return {
            "outcome_ref": outcome_ref,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "causal_chain_length": len(causal_keys),
            "causal_entries": entries,
            "external_contexts": self._read_tracker.external_contexts,
        }

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
        """Browse persisted decision traces from past sessions, newest first.

        Each trace is the full causal chain a committed outcome snapshotted:
        the reads, external contexts, recorded actions, and the outcome itself.
        Use this to learn from past decisions before making a similar one.

        Args:
            entity_path: Only traces touching this entity.
            agent_id: Only traces committed by this agent.
            outcome_type: Filter by outcome (e.g. "success", "failure").
            limit: Maximum traces to return (default 100).
            offset: Rows to skip; honoured only when no cursor is given.
            cursor: Opaque keyset position from a previous page — pass
                ``amfs_core.pagination.encode_cursor(last.created_at, last.id)``
                (or the ``next_cursor`` an HTTP page returned) to get the
                traces strictly older than the last one seen.
            since: Only traces created at or after this time.
            until: Only traces created before this time.

        Returns an empty list on adapters without trace persistence (e.g. the
        default filesystem adapter) — traces live in Postgres or the hosted API.
        """
        return self._adapter.list_traces(
            entity_path=entity_path,
            agent_id=agent_id,
            outcome_type=outcome_type,
            limit=limit,
            offset=offset,
            cursor=cursor,
            since=since,
            until=until,
        )

    def get_trace(self, trace_id: str) -> DecisionTrace | None:
        """Retrieve a full decision trace by ID, or None if not found.

        Args:
            trace_id: The trace ID (from ``list_traces`` or ``commit_outcome``).
        """
        return self._adapter.get_trace(trace_id)

    # ------------------------------------------------------------------
    # Agent brain — scoped recall & cross-agent reads
    # ------------------------------------------------------------------

    def _log_read_event(self, entry: MemoryEntry, branch: str | None = None) -> None:
        """Log a READ event to the timeline (fire-and-forget)."""
        try:
            self._adapter.log_event(Event(
                namespace=self.namespace,
                agent_id=self.agent_id,
                branch=branch or self._branch,
                event_type=EventType.READ,
                summary=f"Read {entry.entity_path}/{entry.key}",
                details={
                    "entity_path": entry.entity_path,
                    "key": entry.key,
                    "version": entry.version,
                    "author_agent_id": entry.provenance.agent_id,
                },
            ))
        except Exception:
            logger.debug("Failed to log read event", exc_info=True)

    def recall(
        self,
        entity_path: str,
        key: str,
        *,
        min_confidence: float = 0.0,
    ) -> MemoryEntry | None:
        """Recall this agent's own memory for a key.

        Unlike ``read()``, which returns the latest version by any agent,
        ``recall()`` returns only entries written by this agent — what this
        brain actually knows from direct experience.  Falls back through
        history to find the most recent version authored by this agent.
        """
        entry = self._engine.read(entity_path, key, min_confidence=0.0)
        if entry is None:
            return None
        if entry.provenance.agent_id == self.agent_id:
            if entry.confidence >= min_confidence:
                self._log_read_event(entry)
                return entry
            return None
        versions = self._engine.history(entity_path, key)
        for v in reversed(versions):
            if v.provenance.agent_id == self.agent_id:
                if v.confidence >= min_confidence:
                    self._log_read_event(v)
                    return v
                return None
        return None

    def my_entries(
        self,
        entity_path: str | None = None,
    ) -> list[MemoryEntry]:
        """List entries written by this agent — the contents of this brain."""
        return self.search(entity_path=entity_path, agent_id=self.agent_id)

    def read_from(
        self,
        agent_id: str,
        entity_path: str,
        key: str,
    ) -> MemoryEntry | None:
        """Read a specific key from another agent's memory.

        Makes cross-agent knowledge transfer explicit and trackable.
        The read is logged in the causal chain for decision tracing.
        """
        entries = self.search(entity_path=entity_path, agent_id=agent_id)
        matching = [e for e in entries if e.key == key]
        if not matching:
            return None
        entry = matching[0]
        self._read_tracker.record(entry)

        try:
            self._adapter.log_event(Event(
                namespace=self.namespace,
                agent_id=self.agent_id,
                branch=self._branch,
                event_type=EventType.CROSS_AGENT_READ,
                summary=f"Read {entity_path}/{key} from agent '{agent_id}'",
                details={
                    "source_agent_id": agent_id,
                    "entity_path": entity_path,
                    "key": key,
                    "version": entry.version,
                },
            ))
        except Exception:
            logger.debug("Failed to log cross-agent read event", exc_info=True)

        try:
            self._adapter.log_event(Event(
                namespace=self.namespace,
                agent_id=agent_id,
                branch=self._branch,
                event_type=EventType.CROSS_AGENT_READ,
                summary=f"Memory {entity_path}/{key} was read by agent '{self.agent_id}'",
                actor_agent_id=self.agent_id,
                details={
                    "reader_agent_id": self.agent_id,
                    "entity_path": entity_path,
                    "key": key,
                    "version": entry.version,
                    "direction": "inbound",
                },
            ))
        except Exception:
            logger.debug("Failed to log inbound cross-agent read event", exc_info=True)

        try:
            self._adapter.upsert_graph_edge(
                GraphEdge(
                    source_entity=self.agent_id,
                    source_type="agent",
                    relation="learned_from",
                    target_entity=agent_id,
                    target_type="agent",
                    provenance={
                        "entity_path": entity_path,
                        "key": key,
                        "trigger": "read_from",
                    },
                ),
                namespace=self.namespace,
                branch=self._branch,
            )
        except Exception:
            logger.debug("Failed to materialize cross-agent graph edge", exc_info=True)

        return entry

    # ------------------------------------------------------------------
    # Cross-agent relationships
    # ------------------------------------------------------------------

    def cross_agent_reads(self) -> dict[str, list[dict[str, Any]]]:
        """Return which other agents' memory this agent has read.

        Examines the decision traces to find all entries read by this agent,
        then identifies which of those were written by other agents.

        Returns a dict mapping ``other_agent_id`` to a list of dicts with
        ``entity_path``, ``key``, and ``read_count``.

        Use this to answer questions like:
        - "Which agents have I talked to?"
        - "What memory did I get from agent X?"

        Example::

            reads = mem.cross_agent_reads()
            # {'deploy-agent': [{'entity_path': 'checkout-service', 'key': 'retry-pattern', 'read_count': 3}]}
        """
        entries = self.list()

        entry_authors: dict[str, str] = {}
        for e in entries:
            entry_authors[f"{e.entity_path}/{e.key}"] = e.provenance.agent_id

        # Aggregated by the adapter (in SQL on Postgres) instead of pulling up
        # to 10,000 full traces here to tally their causal_entries.
        read_entities = self._adapter.trace_read_counts(self.agent_id)

        cross_reads: dict[str, list[dict[str, Any]]] = {}
        for ep, keys in read_entities.items():
            for key, count in keys.items():
                author = entry_authors.get(f"{ep}/{key}")
                if author and author != self.agent_id:
                    if author not in cross_reads:
                        cross_reads[author] = []
                    cross_reads[author].append({
                        "entity_path": ep,
                        "key": key,
                        "read_count": count,
                    })

        return cross_reads

    def agents_i_read_from(self) -> list[str]:
        """Return a list of other agent IDs whose memory this agent has read.

        A convenience wrapper around :meth:`cross_agent_reads` that returns
        just the agent IDs.
        """
        return list(self.cross_agent_reads().keys())

    # ------------------------------------------------------------------
    # Timeline (git log)
    # ------------------------------------------------------------------

    def timeline(
        self,
        *,
        event_type: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[Event]:
        """Return this agent's event timeline (git commit log).

        Every write, outcome, and cross-agent read is automatically
        recorded as an event. Use this to see the history of what
        happened to this agent's memory.
        """
        return self._adapter.list_events(
            self.agent_id,
            self.namespace,
            event_type=event_type,
            since=since,
            limit=limit,
        )

    # ------------------------------------------------------------------
    # Briefing (Memory Cortex)
    # ------------------------------------------------------------------

    def briefing(
        self,
        entity_path: str | None = None,
        agent_id: str | None = None,
        limit: int = 10,
        branch: str | None = None,
    ) -> list:
        """Get a ranked briefing of compiled knowledge digests.

        Returns pre-compiled Digest objects from the Cortex, ranked by
        relevance to the given entity or agent context.

        Resolution order:
        1. Adapter-native ``briefing()`` (e.g. HttpAdapter proxy)
        2. ``amfs_cortex.BriefingService`` (full Cortex scoring)
        3. Inline scoring via adapter ``list_digests()`` (fallback when
           amfs_cortex is not installed but adapter has digest access)

        Args:
            entity_path: Focus on digests relevant to this entity.
            agent_id: Focus on digests relevant to this agent.
            limit: Maximum number of digests to return.
            branch: Branch to read digests from (defaults to active branch).

        Returns:
            List of Digest objects ranked by relevance.
        """
        # Prefer the adapter's native briefing when available (e.g. HttpAdapter
        # proxies to the server which has full Cortex + Postgres access).
        adapter_briefing = getattr(self._adapter, "briefing", None)
        if callable(adapter_briefing):
            return adapter_briefing(
                entity_path=entity_path,
                agent_id=agent_id or self.agent_id,
                limit=limit,
            )

        resolved_agent = agent_id or self.agent_id
        resolved_branch = branch or self._branch

        try:
            from amfs_cortex.briefing import BriefingService
        except ImportError:
            pass
        else:
            service = BriefingService(
                adapter=self._adapter,
                namespace=self.namespace,
            )
            return service.briefing(
                entity_path=entity_path,
                agent_id=resolved_agent,
                limit=limit,
                branch=resolved_branch,
            )

        # Fallback: amfs_cortex not installed but adapter supports list_digests
        # (e.g. PostgresAdapter used directly). Apply inline relevance scoring
        # so the MCP server doesn't silently return empty briefings.
        list_digests_fn = getattr(self._adapter, "list_digests", None)
        if not callable(list_digests_fn):
            return []

        return _score_digests(
            list_digests_fn(namespace=self.namespace, branch=resolved_branch),
            entity_path=entity_path,
            agent_id=resolved_agent,
            limit=limit,
        )

    # ------------------------------------------------------------------
    # Scoped access
    # ------------------------------------------------------------------

    def scope(self, entity_path: str, *, readonly: bool = False) -> MemoryScope:
        """Return a :class:`MemoryScope` bound to *entity_path*."""
        return MemoryScope(self, entity_path, readonly=readonly)

    def list_scopes(self) -> list[str]:
        """Return all unique entity paths that contain at least one entry."""
        entries = self.list()
        return sorted({e.entity_path for e in entries})

    def info(self, entity_path: str) -> ScopeInfo:
        """Return summary information about a single scope."""
        entries = [e for e in self.list() if e.entity_path == entity_path]
        if not entries:
            return ScopeInfo(path=entity_path, entry_count=0, avg_confidence=0.0)
        timestamps = [e.provenance.written_at for e in entries]
        return ScopeInfo(
            path=entity_path,
            entry_count=len(entries),
            avg_confidence=sum(e.confidence for e in entries) / len(entries),
            keys=[e.key for e in entries],
            oldest=min(timestamps),
            newest=max(timestamps),
        )

    def tree(self, max_depth: int = 3) -> str:
        """Render all entity paths as an indented tree with entry counts.

        Example output::

            myapp (5)
              auth (2)
              checkout-service (3)
        """
        entries = self.list()
        path_counts: dict[str, int] = defaultdict(int)
        for e in entries:
            path_counts[e.entity_path] += 1

        tree_nodes: dict[str, Any] = {}
        for path in sorted(path_counts):
            parts = path.split("/")[:max_depth]
            node = tree_nodes
            for part in parts:
                node = node.setdefault(part, {})

        lines: list[str] = []

        def _walk(node: dict[str, Any], prefix: str, depth: int) -> None:
            for name in sorted(node):
                current = f"{prefix}/{name}" if prefix else name
                count = sum(
                    c for p, c in path_counts.items()
                    if p == current or p.startswith(f"{current}/")
                )
                lines.append(f"{'  ' * depth}{name} ({count})")
                if depth < max_depth - 1:
                    _walk(node[name], current, depth + 1)

        _walk(tree_nodes, "", 0)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Read tracker management
    # ------------------------------------------------------------------

    def clear_read_log(self) -> None:
        """Reset the session read log (e.g. between sub-tasks)."""
        self._read_tracker.clear()

    # ------------------------------------------------------------------
    # Room operations (Pro — requires HTTP adapter with rooms backend)
    # ------------------------------------------------------------------

    def room_join(self, room_id: str) -> dict[str, Any]:
        """Join a room. The user owning this agent must have an accepted invite.

        Returns membership details and a briefing of room history.
        """
        return self._room_request("POST", f"/api/v1/rooms/{room_id}/agents", {
            "agent_id": self.agent_id,
        })

    def room_leave(self, room_id: str) -> dict[str, Any]:
        """Leave a room. Knowledge snapshots are preserved in private memory."""
        return self._room_request("DELETE", f"/api/v1/rooms/{room_id}/agents/{self.agent_id}")

    def my_rooms(self) -> list[dict[str, Any]]:
        """List rooms this agent's user owns or is invited to."""
        result = self._room_request("GET", "/api/v1/rooms")
        if isinstance(result, list):
            return result
        return result.get("rooms", []) if isinstance(result, dict) else []

    def room_info(self, room_id: str) -> dict[str, Any]:
        """Get room details including members, settings, and discussion status."""
        return self._room_request("GET", f"/api/v1/rooms/{room_id}")

    def room_updates(
        self, room_id: str, *, since: str | None = None
    ) -> dict[str, Any]:
        """Get recent room activity (writes, joins, discussions, negotiations).

        Args:
            room_id: UUID of the room
            since: ISO timestamp — only return events after this time
        """
        params = f"?since={since}" if since else ""
        return self._room_request("GET", f"/api/v1/rooms/{room_id}/activity{params}")

    # ------------------------------------------------------------------
    # Room: Discussions
    # ------------------------------------------------------------------

    def room_discuss(
        self,
        room_id: str,
        content: str,
        *,
        message_type: str = "message",
        addressed_to: str | None = None,
    ) -> dict[str, Any]:
        """Post a discussion message in a room.

        Args:
            room_id: UUID of the room
            content: Message text
            message_type: One of 'message', 'question', 'answer', 'proposal', 'summary'
            addressed_to: Optional agent_id to address the message to
        """
        return self._room_request("POST", f"/api/v1/rooms/{room_id}/discussions", {
            "agent_id": self.agent_id,
            "content": content,
            "message_type": message_type,
            "addressed_to": addressed_to,
        })

    def room_discussions(
        self,
        room_id: str,
        *,
        limit: int = 50,
        since: str | None = None,
    ) -> dict[str, Any]:
        """List discussion messages in a room.

        Args:
            room_id: UUID of the room
            limit: Max messages to return
            since: ISO timestamp — only messages after this time
        """
        params = f"?limit={limit}"
        if since:
            params += f"&since={since}"
        return self._room_request("GET", f"/api/v1/rooms/{room_id}/discussions{params}")

    # ------------------------------------------------------------------
    # Room: Negotiations
    # ------------------------------------------------------------------

    def negotiate_create(
        self,
        room_id: str,
        title: str,
        *,
        description: str | None = None,
        max_rounds: int = 10,
    ) -> dict[str, Any]:
        """Create a new negotiation session in a room.

        Args:
            room_id: UUID of the room
            title: Topic of negotiation
            description: Detailed description of what needs to be decided
            max_rounds: Maximum rounds before timeout
        """
        return self._room_request("POST", f"/api/v1/rooms/{room_id}/negotiations", {
            "title": title,
            "description": description,
            "agent_id": self.agent_id,
            "max_rounds": max_rounds,
        })

    def negotiate_propose(
        self,
        room_id: str,
        session_id: str,
        *,
        action: str = "propose",
        proposal: dict[str, Any] | None = None,
        rationale: str | None = None,
    ) -> dict[str, Any]:
        """Submit a proposal or response in a negotiation.

        Args:
            room_id: UUID of the room
            session_id: UUID of the negotiation session
            action: 'propose', 'counter', 'accept', 'reject', 'abstain'
            proposal: JSON-serializable dict with values for each issue
            rationale: Reasoning for this action
        """
        return self._room_request("POST", f"/api/v1/rooms/{room_id}/negotiations/{session_id}/propose", {
            "agent_id": self.agent_id,
            "action": action,
            "proposal": proposal or {},
            "rationale": rationale,
        })

    def negotiate_respond(
        self,
        room_id: str,
        session_id: str,
        *,
        action: str,
        rationale: str | None = None,
    ) -> dict[str, Any]:
        """Respond to the current proposal (accept/reject/counter).

        Args:
            room_id: UUID of the room
            session_id: UUID of the negotiation session
            action: 'accept', 'reject', 'counter', 'abstain'
            rationale: Your reasoning
        """
        return self.negotiate_propose(
            room_id, session_id, action=action, rationale=rationale,
        )

    def negotiate_status(
        self,
        room_id: str,
        session_id: str,
    ) -> dict[str, Any]:
        """Get the current state of a negotiation session.

        Args:
            room_id: UUID of the room
            session_id: UUID of the negotiation session
        """
        return self._room_request("GET", f"/api/v1/rooms/{room_id}/negotiations/{session_id}")

    def _room_request(self, method: str, path: str, body: dict | None = None) -> Any:
        """Send a request to the rooms API via the HTTP adapter."""
        adapter = self._adapter
        if hasattr(adapter, "_base_url") and hasattr(adapter, "_session"):
            import requests
            url = f"{adapter._base_url.rstrip('/')}{path}"
            headers = getattr(adapter, "_headers", {})
            if method == "GET":
                resp = adapter._session.get(url, headers=headers)
            elif method == "POST":
                resp = adapter._session.post(url, json=body or {}, headers=headers)
            elif method == "DELETE":
                resp = adapter._session.delete(url, headers=headers)
            else:
                resp = adapter._session.request(method, url, json=body, headers=headers)
            resp.raise_for_status()
            return resp.json()
        raise NotImplementedError(
            "Room operations require the HTTP adapter (set AMFS_HTTP_URL)"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Stop background threads and clean up resources."""
        if self._lifecycle is not None:
            self._lifecycle.stop()
        if hasattr(self._adapter, "close"):
            self._adapter.close()  # type: ignore[attr-defined]

    def __enter__(self) -> AgentMemory:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class MemoryScope:
    """A scoped view of :class:`AgentMemory` bound to a fixed entity path.

    All operations are automatically routed to the underlying memory with
    the configured *entity_path*.  When *readonly* is ``True``, writes
    raise :class:`PermissionError`.
    """

    def __init__(
        self,
        memory: AgentMemory,
        entity_path: str,
        *,
        readonly: bool = False,
    ) -> None:
        self._memory = memory
        self._entity_path = entity_path
        self._readonly = readonly

    @property
    def entity_path(self) -> str:
        return self._entity_path

    @property
    def readonly(self) -> bool:
        return self._readonly

    def read(self, key: str, **kwargs: Any) -> MemoryEntry | None:
        return self._memory.read(self._entity_path, key, **kwargs)

    def write(self, key: str, value: Any, **kwargs: Any) -> MemoryEntry:
        if self._readonly:
            raise PermissionError("Read-only scope")
        return self._memory.write(self._entity_path, key, value, **kwargs)

    def list(self, **kwargs: Any) -> list[MemoryEntry]:
        return self._memory.list(self._entity_path, **kwargs)

    def search(self, **kwargs: Any) -> list[MemoryEntry]:
        return self._memory.search(entity_path=self._entity_path, **kwargs)

    def history(self, key: str, **kwargs: Any) -> list[MemoryEntry]:
        return self._memory.history(self._entity_path, key, **kwargs)

    def info(self) -> ScopeInfo:
        return self._memory.info(self._entity_path)


class TransactionContext:
    """Context manager wrapping a TransactionBuffer for use with ``with`` statements."""

    def __init__(self, buffer: Any, adapter: Any, tagger: Any) -> None:
        self._buffer = buffer
        self._adapter = adapter
        self._tagger = tagger
        self._committed = False
        self.commit: Commit | None = None
        self.entries: list[MemoryEntry] = []

    def write(self, entity_path: str, key: str, value: Any, **kwargs: Any) -> None:
        self._buffer.write(entity_path, key, value, **kwargs)

    def set_message(self, message: str) -> None:
        self._buffer.set_message(message)

    @property
    def pending_count(self) -> int:
        return self._buffer.pending_count

    def __enter__(self) -> "TransactionContext":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is None and not self._committed and self._buffer.pending_count > 0:
            self.commit, self.entries = self._buffer.flush(self._adapter, self._tagger)
            self._committed = True
