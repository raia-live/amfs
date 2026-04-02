"""Abstract base class for AMFS storage adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable

from amfs_core.embedder import EmbedderABC, cosine_similarity
from amfs_core.models import MemoryEntry, MemoryStats, OutcomeRecord, SearchQuery, SemanticQuery


class WatchHandle:
    """Handle returned by adapter.watch() — call .cancel() to stop watching."""

    def __init__(self, cancel_fn: Callable[[], None]) -> None:
        self._cancel_fn = cancel_fn
        self._cancelled = False

    def cancel(self) -> None:
        if not self._cancelled:
            self._cancel_fn()
            self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled


class AdapterABC(ABC):
    """Interface that every AMFS storage adapter must implement.

    Seven operations define the contract:
      - read: fetch the current version of a key
      - write: persist a new version of a key (CoW)
      - list: enumerate entries under an entity path
      - search: query entries with rich filters (confidence, agent, recency)
      - stats: aggregate statistics about memory state
      - watch: observe writes to an entity path in real time
      - commit_outcome: record an outcome and back-propagate to entries
    """

    @abstractmethod
    def read(
        self,
        entity_path: str,
        key: str,
        *,
        min_confidence: float = 0.0,
    ) -> MemoryEntry | None:
        """Read the current version of a key, or None if not found.

        If the entry's confidence is below *min_confidence*, return None.
        """

    @abstractmethod
    def write(self, entry: MemoryEntry) -> MemoryEntry:
        """Persist a memory entry. Returns the written entry (with final version)."""

    @abstractmethod
    def list(
        self,
        entity_path: str | None = None,
        *,
        include_superseded: bool = False,
    ) -> list[MemoryEntry]:
        """List current entries, optionally filtered to an entity path.

        If *include_superseded* is True, also include older versions.
        """

    @abstractmethod
    def watch(
        self,
        entity_path: str,
        callback: Callable[[MemoryEntry], None],
    ) -> WatchHandle:
        """Watch for writes to any key under *entity_path*.

        Returns a WatchHandle that can be cancelled.
        """

    @abstractmethod
    def commit_outcome(self, record: OutcomeRecord) -> list[MemoryEntry]:
        """Record an outcome and back-propagate confidence to causal entries.

        Returns the list of entries whose confidence was updated.
        """

    def list_outcomes(
        self,
        *,
        entity_path: str | None = None,
        since: datetime | None = None,
        limit: int = 1000,
    ) -> list[OutcomeRecord]:
        """Return historical outcome records.

        Used by the Pro ML layer for training ranking models and calibrating
        confidence multipliers.  Default implementation returns an empty list;
        adapters that persist outcomes (e.g. Postgres) should override.
        """
        return []

    def search(self, query: SearchQuery) -> list[MemoryEntry]:
        """Search entries with rich filters. Default: filter over list().

        Adapters may override with optimised implementations (e.g. SQL WHERE).
        """
        entries = self.list(query.entity_path)
        results: list[MemoryEntry] = []
        for entry in entries:
            if entry.confidence < query.min_confidence:
                continue
            if query.max_confidence is not None and entry.confidence > query.max_confidence:
                continue
            if query.agent_id is not None and entry.provenance.agent_id != query.agent_id:
                continue
            if query.since is not None and entry.provenance.written_at < query.since:
                continue
            if query.pattern_ref is not None and query.pattern_ref not in entry.provenance.pattern_refs:
                continue
            results.append(entry)

        if query.sort_by == "confidence":
            results.sort(key=lambda e: e.confidence, reverse=True)
        elif query.sort_by == "recency":
            results.sort(key=lambda e: e.provenance.written_at, reverse=True)
        elif query.sort_by == "version":
            results.sort(key=lambda e: e.version, reverse=True)

        return results[: query.limit]

    def stats(self) -> MemoryStats:
        """Compute aggregate statistics. Default: iterate over list().

        Adapters may override with optimised implementations (e.g. SQL aggregates).
        """
        entries = self.list()
        if not entries:
            return MemoryStats()

        agents: dict[str, int] = {}
        entities: dict[str, int] = {}
        confidences: list[float] = []
        outcome_linked = 0
        oldest = entries[0].provenance.written_at
        newest = entries[0].provenance.written_at

        for entry in entries:
            aid = entry.provenance.agent_id
            agents[aid] = agents.get(aid, 0) + 1
            entities[entry.entity_path] = entities.get(entry.entity_path, 0) + 1
            confidences.append(entry.confidence)
            if entry.outcome_count > 0:
                outcome_linked += 1
            if entry.provenance.written_at < oldest:
                oldest = entry.provenance.written_at
            if entry.provenance.written_at > newest:
                newest = entry.provenance.written_at

        return MemoryStats(
            total_entries=len(entries),
            total_entities=len(entities),
            total_agents=len(agents),
            agents=agents,
            entities=entities,
            confidence_avg=sum(confidences) / len(confidences),
            confidence_min=min(confidences),
            confidence_max=max(confidences),
            outcome_linked_count=outcome_linked,
            oldest_entry_at=oldest,
            newest_entry_at=newest,
        )

    def semantic_search(
        self, query: SemanticQuery, embedder: EmbedderABC
    ) -> list[tuple[MemoryEntry, float]]:
        """Search entries by semantic similarity. Default: brute-force cosine.

        Returns (entry, similarity) tuples sorted by similarity descending.
        Adapters may override with vector-index implementations (e.g. pgvector).
        """
        query_vec = embedder.embed(query.text)
        entries = self.list(query.entity_path)

        scored: list[tuple[MemoryEntry, float]] = []
        for entry in entries:
            if entry.confidence < query.min_confidence:
                continue
            if entry.embedding is None:
                continue
            sim = cosine_similarity(query_vec, entry.embedding)
            if sim >= query.min_similarity:
                scored.append((entry, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: query.limit]
