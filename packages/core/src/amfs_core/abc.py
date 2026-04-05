"""Abstract base class for AMFS storage adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable

from amfs_core.embedder import EmbedderABC, cosine_similarity
from amfs_core.models import (
    Agent,
    Branch,
    BranchAccess,
    DecisionTrace,
    DiffEntry,
    Event,
    MemoryEntry,
    MemoryStats,
    MergeResult,
    MergeStrategy,
    OutcomeRecord,
    SearchQuery,
    SemanticQuery,
    Tag,
)


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

    def read_at_version(
        self,
        entity_path: str,
        key: str,
        version: int,
    ) -> MemoryEntry | None:
        """Read a specific historical version of an entry.

        Returns the entry at exactly *version*, even if it has been
        superseded.  Default implementation scans via ``list()`` with
        ``include_superseded=True``; adapters with indexed storage
        (e.g. Postgres) should override for O(1) lookup.
        """
        all_versions = self.list(entity_path, include_superseded=True)
        for entry in all_versions:
            if entry.key == key and entry.version == version:
                return entry
        return None

    def get_trace(self, trace_id: str) -> DecisionTrace | None:
        """Return a single trace by ID, or None if not found.

        Default implementation scans ``list_traces()``; adapters with
        indexed storage (e.g. Postgres) should override for O(1) lookup.
        """
        for t in self.list_traces(limit=10_000):
            if t.id == trace_id:
                return t
        return None

    def save_trace(self, trace: DecisionTrace) -> DecisionTrace:
        """Persist a decision trace. Default is a no-op; adapters with
        persistent storage (e.g. Postgres) should override."""
        return trace

    def list_traces(
        self,
        *,
        entity_path: str | None = None,
        agent_id: str | None = None,
        outcome_type: str | None = None,
        limit: int = 100,
    ) -> list[DecisionTrace]:
        """Return persisted decision traces. Default returns an empty list."""
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

    # ── Agent registration (Pro) ────────────────────────────────────────

    def ensure_agent(self, agent_id: str, namespace: str = "default") -> Agent:
        """Auto-register an agent on first write. Returns the Agent record.

        Default is a no-op returning a stub. Postgres adapter implements
        INSERT ... ON CONFLICT DO UPDATE.
        """
        return Agent(agent_id=agent_id, namespace=namespace)

    def get_agent(self, agent_id: str, namespace: str = "default") -> Agent | None:
        """Return a registered agent or None."""
        return None

    def list_agents(self, namespace: str = "default") -> list[Agent]:
        """Return all registered agents in a namespace."""
        return []

    # ── Event log / timeline (Pro) ────────────────────────────────────

    def log_event(self, event: Event) -> Event:
        """Persist a timeline event. Default is a no-op."""
        return event

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
        """Return events on an agent's timeline. Default returns empty list."""
        return []

    # ── Branch management (Pro) ───────────────────────────────────────

    def create_branch(self, branch: Branch) -> Branch:
        """Create a new memory branch. Default raises NotImplementedError."""
        raise NotImplementedError("Branching requires the Postgres adapter")

    def get_branch(self, name: str, namespace: str = "default") -> Branch | None:
        """Return a branch by name, or None."""
        return None

    def list_branches(
        self, namespace: str = "default", *, status: str | None = None
    ) -> list[Branch]:
        """List all branches in a namespace."""
        return []

    def close_branch(self, name: str, namespace: str = "default") -> Branch:
        """Mark a branch as closed."""
        raise NotImplementedError("Branching requires the Postgres adapter")

    def diff_branch(self, name: str, namespace: str = "default") -> list[DiffEntry]:
        """Diff a branch against its parent."""
        return []

    def merge_branch(
        self,
        name: str,
        namespace: str = "default",
        *,
        strategy: MergeStrategy = MergeStrategy.FAST_FORWARD,
        resolve_conflicts: dict[str, str] | None = None,
    ) -> MergeResult:
        """Merge a branch into its parent."""
        raise NotImplementedError("Branching requires the Postgres adapter")

    # ── Branch access control (Pro) ───────────────────────────────────

    def grant_branch_access(self, access: BranchAccess) -> BranchAccess:
        """Grant external access to a branch."""
        raise NotImplementedError("Branch access requires the Postgres adapter")

    def revoke_branch_access(
        self, branch_name: str, grantee_type: str, grantee_id: str,
        namespace: str = "default",
    ) -> None:
        """Revoke access from a branch."""
        raise NotImplementedError("Branch access requires the Postgres adapter")

    def list_branch_access(
        self, branch_name: str, namespace: str = "default"
    ) -> list[BranchAccess]:
        """List access grants for a branch."""
        return []

    def check_branch_access(
        self, branch_name: str, api_key_id: str, namespace: str = "default"
    ) -> str | None:
        """Check if an API key has access to a branch. Returns permission or None."""
        return None

    # ── Tags / Snapshots (Pro) ────────────────────────────────────────

    def create_tag(self, tag: Tag) -> Tag:
        """Create a named point-in-time tag."""
        raise NotImplementedError("Tags require the Postgres adapter")

    def get_tag(self, name: str, namespace: str = "default") -> Tag | None:
        """Return a tag by name, or None."""
        return None

    def list_tags(
        self, namespace: str = "default", *, branch: str | None = None
    ) -> list[Tag]:
        """List tags, optionally filtered by branch."""
        return []

    def delete_tag(self, name: str, namespace: str = "default") -> None:
        """Delete a tag."""
        raise NotImplementedError("Tags require the Postgres adapter")

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
