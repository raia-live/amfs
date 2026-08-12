"""Abstract base class for AMFS storage adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable

from amfs_core.embedder import EmbedderABC, cosine_similarity
from amfs_core.models import (
    Agent,
    AgentCapability,
    AgentProfile,
    Branch,
    BranchAccess,
    Commit,
    DecisionTrace,
    DiffEntry,
    Event,
    GraphEdge,
    GraphNeighborQuery,
    MemoryContract,
    MemoryEntry,
    MemoryStats,
    MergeResult,
    MergeStrategy,
    OutcomeRecord,
    PRReview,
    PullRequest,
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

    def list_traces_for_entry(
        self,
        *,
        agent_id: str,
        session_id: str | None,
        written_at: datetime | None = None,
        limit: int = 5,
    ) -> list[DecisionTrace]:
        """Traces plausibly responsible for writing an entry, best match first.

        An entry records the session that wrote it and a trace records the
        session it was committed from, and that shared id is the only link
        between the two — ``causal_entries`` holds the reads a session made,
        never its writes. So resolution is: same agent, same session, and among
        those the trace committed soonest *after* the write.

        The match is exact when a session committed one outcome and a heuristic
        when it committed several, so callers must present the result as the
        likely committing trace rather than a certainty.

        Default returns an empty list; adapters with persisted traces override.
        """
        return []

    def search(self, query: SearchQuery) -> list[MemoryEntry]:
        """Search entries with rich filters. Default: filter over list().

        When ``query.query`` is set **and** ``recall_config`` is *not* set,
        performs case-insensitive substring matching against key, value
        (stringified), and entity_path **before** applying sort and limit —
        so text filtering is never silently truncated.

        When ``recall_config`` is set the query text is intended for semantic
        scoring in the SDK layer, so the adapter skips substring filtering
        to avoid pre-emptively discarding valid candidates.

        Adapters may override with optimised implementations (e.g. SQL WHERE
        with tsvector FTS).
        """
        entries = self.list(query.entity_path)
        use_text_filter = query.query and query.recall_config is None
        query_lower = query.query.lower() if use_text_filter else None
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
            if query_lower is not None and not (
                query_lower in entry.key.lower()
                or query_lower in str(entry.value).lower()
                or query_lower in entry.entity_path.lower()
            ):
                continue
            results.append(entry)

        if query.sort_by == "priority":
            from amfs_core.tiering import PriorityScorer
            scorer = PriorityScorer()
            scores = scorer.score_batch(results)
            results.sort(key=lambda e: scores.get(e.entry_key, 0.0), reverse=True)
        elif query.sort_by == "confidence":
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

    def entity_summaries(
        self,
        *,
        agent_ids: list[str] | None = None,
    ) -> list[dict]:
        """Per-entity aggregates (count, avg confidence, last write, agents).

        Returns a list of dicts sorted by most recently updated:
        ``{entity_path, entry_count, avg_confidence, last_updated,
        last_agent, agents, hashed_count, total_recalls}``.

        *agent_ids*, when given, restricts aggregation to entries written by
        those agents (used by per-user visibility layers).

        Default implementation groups over ``list()``; adapters with SQL
        storage should override with GROUP BY aggregates.
        """
        from amfs_core.aggregates import entity_summaries_from_entries

        entries = self.list()
        if agent_ids is not None:
            allowed = set(agent_ids)
            entries = [e for e in entries if e.provenance.agent_id in allowed]
        return entity_summaries_from_entries(entries)

    def agent_entity_stats(
        self,
        *,
        entity_path: str | None = None,
        agent_ids: list[str] | None = None,
    ) -> list[dict]:
        """Per-(agent, entity) aggregates — who has written where, and how much.

        Returns ``{agent_id, entity_path, entry_count, avg_confidence,
        last_written, first_written, total_recalls, outcome_linked_count}``,
        sorted by entity path then descending entry count.

        *entity_path*, when given, restricts to that entity and its descendants
        (``a/b`` matches ``a/b`` and ``a/b/c``, never ``a/bc``). *agent_ids*
        restricts to those writers, for per-user visibility layers.

        Default implementation groups over ``list()``; adapters with SQL
        storage should override with GROUP BY aggregates.
        """
        from amfs_core.aggregates import agent_entity_stats_from_entries

        entries = self.list()
        if agent_ids is not None:
            allowed = set(agent_ids)
            entries = [e for e in entries if e.provenance.agent_id in allowed]
        if entity_path:
            prefix = entity_path.rstrip("/")
            entries = [
                e
                for e in entries
                if e.entity_path == prefix or e.entity_path.startswith(prefix + "/")
            ]
        return agent_entity_stats_from_entries(entries)

    def stats_extended(
        self,
        *,
        agent_ids: list[str] | None = None,
    ) -> dict:
        """Aggregate stats with value/ROI extras beyond ``stats()``.

        Returns the ``MemoryStats`` fields plus ``total_recalls``,
        ``entries_this_week``, ``entries_last_week``, and
        ``memory_type_counts``. *agent_ids* restricts to those writers.

        Default implementation iterates over ``list()``; adapters with SQL
        storage should override with aggregates.
        """
        from amfs_core.aggregates import extended_stats_from_entries

        entries = self.list()
        if agent_ids is not None:
            allowed = set(agent_ids)
            entries = [e for e in entries if e.provenance.agent_id in allowed]
        return extended_stats_from_entries(entries)

    def share_stats(
        self,
        *,
        since: datetime | None = None,
        pair_limit: int = 20,
        agent_ids: list[str] | None = None,
    ) -> dict:
        """Cross-agent knowledge-share aggregate from decision traces.

        A "share" is a trace causal entry written by a different agent than
        the trace's own agent. Returns ``{total, pairs}`` where pairs are
        ``{reader, author, count}`` sorted by count (capped at *pair_limit*).

        Default implementation scans ``list_traces()``; SQL adapters should
        override with a JSONB aggregate so full traces never leave the DB.
        """
        from amfs_core.aggregates import share_stats_from_traces

        return share_stats_from_traces(
            self.list_traces(limit=10_000),
            since=since,
            pair_limit=pair_limit,
            agent_ids=agent_ids,
        )

    # ── Content integrity ────────────────────────────────────────────────

    def verify_integrity(
        self,
        entity_path: str | None = None,
        *,
        branch: str = "main",
    ) -> dict:
        """Verify content hashes and integrity chains for stored entries.

        Returns an IntegrityReport dict with total_checked, valid, corrupted,
        orphaned, and chain_breaks. Default implementation loads entries via
        list() and delegates to ``amfs_core.hashing.verify_entries``.
        """
        from amfs_core.hashing import verify_entries

        entries = self.list(entity_path, include_superseded=True)
        report = verify_entries(entries)
        return report.to_dict()

    # ── Atomic commits ────────────────────────────────────────────────

    def write_batch(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
        """Persist multiple entries atomically.

        Default implementation writes each entry sequentially — adapter
        subclasses should override with a true atomic batch when possible.
        """
        return [self.write(e) for e in entries]

    def save_commit(self, commit: "Commit") -> None:
        """Persist a commit object. Default is a no-op for adapters that
        don't support commit storage yet."""

    def get_commit(self, commit_id: str) -> "Commit | None":
        """Retrieve a commit by its ID. Returns None if not found."""
        return None

    def common_ancestor(self, commit_a_id: str, commit_b_id: str) -> str | None:
        """The most recent commit both given commits descend from.

        Lives here, rather than only in the caller, so that a store which can
        answer the question directly may say so. The default walks the DAG from
        both ends calling :meth:`get_commit` at every node, which is the right
        shape for a local store and the wrong shape for a remote one: there,
        each step is a round trip, and the cost of the answer grows with the
        history the two commits share. An adapter in that position overrides
        this and asks its server once.
        """
        from amfs_core.dag import find_common_ancestor

        return find_common_ancestor(commit_a_id, commit_b_id, self.get_commit)

    def list_commits(
        self,
        *,
        branch: str = "main",
        limit: int = 50,
        namespace: str = "default",
    ) -> "list[Commit]":
        """List commits, newest first."""
        return []

    # ── Recall tracking ─────────────────────────────────────────────────

    def increment_recall_count(
        self,
        entity_path: str,
        key: str,
        *,
        branch: str = "main",
    ) -> None:
        """Increment the recall_count of the current entry version in place.

        This is a mutable metadata update that does NOT create a new CoW
        version.  Adapters that do not support in-place updates can safely
        leave the default no-op.
        """

    # ── Tiered memory ─────────────────────────────────────────────────

    def update_tiers(
        self,
        tier_assignments: dict[str, int],
        scores: dict[str, float],
        *,
        branch: str = "main",
    ) -> int:
        """Batch-update tier and priority_score for entries in place.

        *tier_assignments* maps ``entry_key`` -> tier (1/2/3).
        *scores* maps ``entry_key`` -> priority_score.
        Returns count of entries updated.  Default no-op returns 0.
        """
        return 0


    # ── Agent registration ─────────────────────────────────────────────

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

    def update_agent_profile(
        self,
        agent_id: str,
        profile: "AgentProfile",
        namespace: str = "default",
    ) -> Agent:
        """Update an agent's profile. Default merges into stub."""
        agent = self.ensure_agent(agent_id, namespace)
        return agent.model_copy(update={"profile": profile})

    def update_agent_capabilities(
        self,
        agent_id: str,
        capabilities: "list[AgentCapability]",
        namespace: str = "default",
    ) -> Agent:
        """Update an agent's declared capabilities."""
        agent = self.ensure_agent(agent_id, namespace)
        return agent.model_copy(update={"capabilities": capabilities})

    def update_agent_contracts(
        self,
        agent_id: str,
        contracts: "list[MemoryContract]",
        namespace: str = "default",
    ) -> Agent:
        """Update an agent's memory contracts."""
        agent = self.ensure_agent(agent_id, namespace)
        return agent.model_copy(update={"contracts": contracts})

    def discover_agents(
        self,
        *,
        capability: str | None = None,
        entity_path: str | None = None,
        namespace: str = "default",
    ) -> list[Agent]:
        """Discover agents by capability or entity path.

        Filters the agent list by matching capability name or
        entity_path overlap. Default implementation does simple matching.
        """
        agents = self.list_agents(namespace)
        results = []
        for agent in agents:
            if capability:
                if not any(c.name == capability for c in agent.capabilities):
                    continue
            if entity_path:
                has_path = any(
                    entity_path.startswith(c_ep) or c_ep.startswith(entity_path)
                    for cap in agent.capabilities
                    for c_ep in cap.entity_paths
                )
                if not has_path and not any(
                    p.startswith(entity_path) or entity_path.startswith(p)
                    for p in (agent.profile.auto_context_paths if agent.profile else [])
                ):
                    continue
            results.append(agent)
        return results

    # --- Agent Groups ---

    def create_agent_group(
        self, group: "AgentGroup", namespace: str = "default"
    ) -> "AgentGroup":
        raise NotImplementedError

    def list_agent_groups(self, namespace: str = "default") -> list["AgentGroup"]:
        return []

    def get_agent_group(self, group_id: str, namespace: str = "default") -> "AgentGroup | None":
        return None

    def update_agent_group(
        self, group_id: str, *, name: str | None = None,
        description: str | None = None, color: str | None = None,
        icon: str | None = None, position: float | None = None,
        namespace: str = "default",
    ) -> "AgentGroup | None":
        raise NotImplementedError

    def delete_agent_group(self, group_id: str, namespace: str = "default") -> bool:
        raise NotImplementedError

    def add_agents_to_group(
        self, group_id: str, agent_ids: list[str],
        added_by: str = "user", namespace: str = "default",
    ) -> int:
        raise NotImplementedError

    def remove_agents_from_group(
        self, group_id: str, agent_ids: list[str], namespace: str = "default"
    ) -> int:
        raise NotImplementedError

    def reorder_agent_groups(
        self, positions: list[tuple[str, float]], namespace: str = "default"
    ) -> None:
        raise NotImplementedError

    # --- Enriched agent queries ---

    def list_agents_enriched(self, namespace: str = "default") -> list[dict]:
        return []

    def get_agent_activity_histogram(
        self, agent_id: str, days: int = 7, namespace: str = "default"
    ) -> list[int]:
        return [0] * days

    # --- Cluster suggestions ---

    def dismiss_cluster_suggestion(
        self, cluster_id: str, account_id: str
    ) -> None:
        raise NotImplementedError

    def list_dismissed_cluster_ids(self, account_id: str) -> list[str]:
        return []

    # ── Event log / timeline ───────────────────────────────────────────

    #: When True, this adapter delegates writes to a remote AMFS server that
    #: records the WRITE timeline event itself (in its write handler). The SDK
    #: must then NOT also emit a WRITE event, otherwise a single write produces
    #: two identical timeline events. Set True on HttpAdapter; False for direct
    #: adapters (filesystem/postgres) where the SDK is the only logger.
    server_side_write_events: bool = False

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

    def get_event(self, event_id: str, namespace: str = "default") -> Event | None:
        """Retrieve a single event by ID. Default returns None."""
        return None

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

    # ── Pull Requests (Pro) ───────────────────────────────────────────

    def create_pull_request(self, pr: PullRequest) -> PullRequest:
        """Create a pull request for branch merge review."""
        raise NotImplementedError("PRs require the Postgres adapter")

    def get_pull_request(self, pr_id: str, namespace: str = "default") -> PullRequest | None:
        return None

    def list_pull_requests(
        self, namespace: str = "default", *, status: str | None = None
    ) -> list[PullRequest]:
        return []

    def update_pull_request_status(
        self, pr_id: str, status: str, *, by: str = "", namespace: str = "default"
    ) -> PullRequest:
        raise NotImplementedError("PRs require the Postgres adapter")

    def add_pr_review(self, review: PRReview) -> PRReview:
        raise NotImplementedError("PRs require the Postgres adapter")

    def list_pr_reviews(self, pr_id: str, namespace: str = "default") -> list[PRReview]:
        return []

    # ── Rollback (Pro) ────────────────────────────────────────────────

    def rollback_to_timestamp(
        self,
        agent_id: str,
        branch: str,
        timestamp: datetime,
        namespace: str = "default",
    ) -> int:
        """Rollback memory to a point in time. Returns count of restored entries."""
        raise NotImplementedError("Rollback requires the Postgres adapter")

    # ── Fork (Pro) ────────────────────────────────────────────────────

    def fork_agent(
        self,
        source_agent_id: str,
        target_agent_id: str,
        *,
        namespace: str = "default",
        branch: str = "main",
    ) -> int:
        """Copy all entries from source_agent's branch into target_agent's main.

        Returns the number of entries copied.
        """
        raise NotImplementedError("Fork requires the Postgres adapter")

    # ── Knowledge graph ─────────────────────────────────────────────

    def upsert_graph_edge(
        self,
        edge: GraphEdge,
        *,
        namespace: str = "default",
        branch: str = "main",
    ) -> GraphEdge:
        """Insert or update a knowledge graph edge.

        On conflict (same source, relation, target within namespace+branch)
        bumps evidence_count, updates last_seen and takes max confidence.
        Default is a no-op returning the edge unchanged.
        """
        return edge

    def graph_neighbors(
        self,
        query: GraphNeighborQuery,
        *,
        namespace: str = "default",
        branch: str = "main",
    ) -> list[GraphEdge]:
        """Traverse the knowledge graph from an entity.

        Supports multi-hop via *query.depth*. Default returns empty list.
        """
        return []

    def list_graph_edges(
        self,
        *,
        entity: str | None = None,
        relation: str | None = None,
        min_confidence: float = 0.0,
        namespace: str = "default",
        branch: str = "main",
        limit: int = 500,
    ) -> list[GraphEdge]:
        """List graph edges with optional filters. Default returns empty list."""
        return []

    def graph_stats(
        self,
        *,
        namespace: str = "default",
        branch: str = "main",
    ) -> dict:
        """Return aggregate graph statistics. Default returns empty dict."""
        return {}

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
