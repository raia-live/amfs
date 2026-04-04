"""AgentMemory — the main SDK entry point for agents."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from amfs_core.abc import AdapterABC, WatchHandle
from amfs_core.embedder import EmbedderABC
from amfs_core.engine import CausalTagger, CoWEngine, ReadTracker
from amfs_core.exceptions import StaleWriteError
from amfs_core.lifecycle import LifecycleManager
from amfs_core.models import (
    ConflictPolicy,
    DecisionTrace,
    ErrorEvent,
    ExternalContext,
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
    TraceEntry,
)
from amfs_core.outcome import OutcomeBackPropagator

from amfs.config import load_config_or_default
from amfs.factory import create_adapter_from_config

logger = logging.getLogger(__name__)


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

        self._lifecycle: LifecycleManager | None = None
        if ttl_sweep_interval is not None:
            self._lifecycle = LifecycleManager(self._adapter, interval=ttl_sweep_interval)
            self._lifecycle.start()

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
    ) -> MemoryEntry | None:
        """Read the current version of a key.

        Automatically tracked for causal linking and conflict detection.
        If *decay_half_life_days* is set, applies confidence decay before
        the min_confidence check.
        """
        import time

        start = time.monotonic()
        try:
            if self._decay_half_life_days is not None:
                entry = self._engine.read(entity_path, key, min_confidence=0.0)
                if entry is None:
                    return None
                effective = entry.effective_confidence(
                    decay_half_life_days=self._decay_half_life_days,
                )
                if effective < min_confidence:
                    return None
                return entry
            return self._engine.read(entity_path, key, min_confidence=min_confidence)
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
    ) -> MemoryEntry:
        """Write a new version of a key with automatic provenance.

        If *conflict_policy* is ``RAISE``, checks whether the entry was
        modified by another agent since our last read and raises
        ``StaleWriteError`` if so. If an ``on_conflict`` callback is set,
        it is called with ``(our_last_read, current_entry, new_value)``
        and should return the merged value to write.
        """
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

        entry = self._engine.write(
            entity_path,
            key,
            value,
            confidence=confidence,
            ttl_at=ttl_at,
            pattern_refs=pattern_refs,
            memory_type=memory_type,
            artifact_refs=artifact_refs,
        )
        self._read_tracker.record_write(entity_path, key, entry.version, entry.version == 1)

        if self._embedder is not None:
            embedding = self._embedder.embed_value(value)
            entry = entry.model_copy(update={"embedding": embedding, "version": 1})
            entry = self._adapter.write(entry)

        return entry

    def list(
        self,
        entity_path: str | None = None,
        *,
        include_superseded: bool = False,
    ) -> list[MemoryEntry]:
        """List current entries, optionally filtered to an entity path."""
        import time
        start = time.monotonic()
        results = self._engine.list(entity_path, include_superseded=include_superseded)
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
    ) -> list[MemoryEntry] | list[ScoredEntry]:
        """Search across all entities with rich filters.

        When *entity_paths* is provided, runs a search for each path and merges
        the results.  *entity_path* (singular) is still supported for backwards
        compatibility; if both are given, *entity_paths* takes precedence.

        When *recall_config* is provided, returns ``ScoredEntry`` objects
        sorted by composite recall score instead.
        """
        paths = entity_paths or ([entity_path] if entity_path else [None])

        seen_keys: set[str] = set()
        merged: list[MemoryEntry] = []
        for ep in paths:
            query = SearchQuery(
                entity_path=ep,
                min_confidence=min_confidence,
                max_confidence=max_confidence,
                agent_id=agent_id,
                since=since,
                pattern_ref=pattern_ref,
                limit=limit,
                sort_by=sort_by,
                recall_config=recall_config,
            )
            for entry in self._adapter.search(query):
                if entry.entry_key not in seen_keys:
                    seen_keys.add(entry.entry_key)
                    merged.append(entry)

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

        now = datetime.now(timezone.utc)
        scored: list[ScoredEntry] = []
        for entry in entries:
            age = now - entry.provenance.written_at
            age_days = age.total_seconds() / 86400.0
            half_life = recall_config.recency_half_life_days

            recency_score = math.exp(-math.log(2) * age_days / half_life) if half_life > 0 else 0.0
            confidence_score = max(0.0, min(1.0, entry.confidence))

            composite = (
                recall_config.semantic_weight * 1.0
                + recall_config.recency_weight * recency_score
                + recall_config.confidence_weight * confidence_score
            )
            scored.append(ScoredEntry(
                entry=entry,
                score=composite,
                breakdown={
                    "semantic": recall_config.semantic_weight * 1.0,
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

    def stats(self) -> MemoryStats:
        """Aggregate statistics about current memory state."""
        return self._adapter.stats()

    # ------------------------------------------------------------------
    # Outcomes
    # ------------------------------------------------------------------

    def commit_outcome(
        self,
        outcome_ref: str,
        outcome_type: OutcomeType,
        causal_entry_keys: list[str] | None = None,
        *,
        causal_confidence: float = 1.0,
        decision_summary: str | None = None,
    ) -> list[MemoryEntry]:
        """Record an outcome and back-propagate confidence changes.

        If *causal_entry_keys* is ``None``, automatically uses the session's
        read log — every entry this agent read becomes a causal link.
        """
        if causal_entry_keys is None:
            causal_entry_keys = self._read_tracker.causal_keys
        record = OutcomeBackPropagator.make_record(
            outcome_ref=outcome_ref,
            outcome_type=outcome_type,
            causal_entry_keys=causal_entry_keys,
            agent_id=self.agent_id,
            causal_confidence=causal_confidence,
        )
        updated = self._propagator.propagate(record)

        causal_trace_entries: list[TraceEntry] = []
        for ek in causal_entry_keys:
            parts = ek.rsplit("/", 1)
            if len(parts) != 2:
                continue
            ep, k = parts
            entry = self._adapter.read(ep, k)
            if entry:
                snapshot = self._read_tracker.entry_snapshot(ek)
                causal_trace_entries.append(TraceEntry(
                    entity_path=ep, key=k,
                    version=self._read_tracker.read_version(ek) or entry.version,
                    confidence=entry.confidence,
                    value=snapshot.get("value") if snapshot else None,
                    memory_type=snapshot.get("memory_type") if snapshot else None,
                    written_by=snapshot.get("written_by") if snapshot else None,
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
            causal_entries=causal_trace_entries,
            external_contexts=ext_contexts,
            query_events=query_events,
            session_started_at=session_started,
            session_ended_at=now,
            session_duration_ms=session_duration,
            error_events=error_events,
            state_diff=state_diff,
        )
        try:
            self._adapter.save_trace(trace)
        except Exception:
            logger.debug("Failed to persist decision trace", exc_info=True)

        return updated

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
        return self._engine.history(entity_path, key, since=since, until=until)

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
        traces = self._adapter.list_traces(agent_id=self.agent_id, limit=10000)

        entry_authors: dict[str, str] = {}
        for e in entries:
            entry_authors[f"{e.entity_path}/{e.key}"] = e.provenance.agent_id

        read_entities: dict[str, dict[str, int]] = {}
        for t in traces:
            for ce in t.causal_entries:
                ep = ce.entity_path
                key = ce.key
                if ep not in read_entities:
                    read_entities[ep] = {}
                read_entities[ep][key] = read_entities[ep].get(key, 0) + 1

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
