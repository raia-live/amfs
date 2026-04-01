"""AgentMemory — the main SDK entry point for agents."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from amfs_core.abc import AdapterABC, WatchHandle
from amfs_core.embedder import EmbedderABC
from amfs_core.engine import CausalTagger, CoWEngine, ReadTracker
from amfs_core.exceptions import StaleWriteError
from amfs_core.lifecycle import LifecycleManager
from amfs_core.models import (
    ConflictPolicy,
    MemoryEntry,
    MemoryStats,
    MemoryType,
    OutcomeType,
    SearchQuery,
    SemanticQuery,
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
        )

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
        return self._engine.list(entity_path, include_superseded=include_superseded)

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
        min_confidence: float = 0.0,
        max_confidence: float | None = None,
        agent_id: str | None = None,
        since: datetime | None = None,
        pattern_ref: str | None = None,
        limit: int = 100,
        sort_by: str = "confidence",
    ) -> list[MemoryEntry]:
        """Search across all entities with rich filters.

        Returns entries matching ALL specified criteria, sorted and limited.
        """
        query = SearchQuery(
            entity_path=entity_path,
            min_confidence=min_confidence,
            max_confidence=max_confidence,
            agent_id=agent_id,
            since=since,
            pattern_ref=pattern_ref,
            limit=limit,
            sort_by=sort_by,
        )
        return self._adapter.search(query)

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
        return self._propagator.propagate(record)

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

    def explain(self, outcome_ref: str | None = None) -> dict[str, Any]:
        """Return the causal chain for the current session or a specific outcome.

        Shows which memories were read (and in what order) before the outcome
        was committed, enabling production-grounded explainability.
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
        }

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
