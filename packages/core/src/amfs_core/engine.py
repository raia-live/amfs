"""CoWEngine, CausalTagger, and ReadTracker — core write logic for AMFS."""

from __future__ import annotations

import hashlib
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator

from amfs_core.abc import AdapterABC
from amfs_core.hashing import content_hash, integrity_chain_hash
from amfs_core.models import ArtifactRef, MemoryEntry, MemoryType, Provenance

ExternalContext = dict[str, Any]


class CausalTagger:
    """Stamps provenance metadata on every write.

    Parameters
    ----------
    agent_id:
        Identifier of the agent performing writes.
    session_id:
        Optional session identifier. Auto-generated if not provided.
    """

    def __init__(self, agent_id: str, session_id: str | None = None) -> None:
        self.agent_id = agent_id
        self.session_id = session_id or f"sess-{uuid.uuid4().hex[:8]}"

    def tag(self, *, pattern_refs: list[str] | None = None) -> Provenance:
        """Create a new Provenance with the current timestamp."""
        return Provenance(
            agent_id=self.agent_id,
            session_id=self.session_id,
            written_at=datetime.now(timezone.utc),
            pattern_refs=pattern_refs or [],
        )


#: An action's result is summarised, not stored. Long enough to be recognisable in
#: a trace timeline, short enough that a chatty tool cannot bloat every row.
_MAX_ACTION_RESULT_CHARS = 2000


@dataclass
class _TrackerState:
    """Everything a :class:`ReadTracker` accumulates over one session.

    Held apart from the tracker so that a server sharing one tracker across
    requests can give each request a session of its own — see
    :func:`read_tracker_scope`.
    """

    reads: dict[str, datetime] = field(default_factory=dict)
    versions: dict[str, int] = field(default_factory=dict)
    entries: dict[str, dict] = field(default_factory=dict)
    contexts: list[ExternalContext] = field(default_factory=list)
    queries: list[dict] = field(default_factory=list)
    session_started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    errors: list[dict] = field(default_factory=list)
    writes: list[dict] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)


#: The session in force for the current context, if any. Unset in a normal
#: single-agent process, where a tracker's own state is its session.
_TRACKER_SCOPE: ContextVar[_TrackerState | None] = ContextVar(
    "amfs_read_tracker_scope", default=None
)


@contextmanager
def read_tracker_scope() -> Iterator[_TrackerState]:
    """Give the enclosed block a session of its own on every shared tracker.

    A ``ReadTracker`` accumulates the reads, contexts, queries and writes of one
    session, which is exactly right for the process it was designed for: one
    agent, one memory handle, one session. A server is not that. It serves every
    caller from a single ``AgentMemory``, so one tracker holds them all, and what
    it accumulated for one caller is still in place for the next — leaving anything
    that reads it back, whether ``explain`` or a ``commit_outcome`` falling back to
    it for causal entries, describing a session that belongs to no one caller.

    The state is moved rather than the tracker replaced because the tracker
    instance is captured at construction, by ``CoWEngine`` among others; changing
    what a caller *sees* then needs no cooperation from anything holding a
    reference.

    Scoped per request, because that is the unit the state belongs to: within one
    request the reads and the commit go together, and across requests they do not,
    since a remote caller's own session lives in its own process and names its
    causal entries when it commits.
    """
    state = _TrackerState()
    token = _TRACKER_SCOPE.set(state)
    try:
        yield state
    finally:
        _TRACKER_SCOPE.reset(token)


class ReadTracker:
    """Automatically records every read within a session for causal linking
    and conflict detection.

    When an agent reads entries and later commits an outcome, the tracker
    provides the causal chain without the developer manually specifying
    which entries were involved.

    Also tracks the version at read time so the engine can detect stale
    writes (another agent modified the entry since we last read it).
    """

    def __init__(self) -> None:
        #: This tracker's own session, used whenever no scope is in force — which
        #: is every single-agent process. The fields below read through to it, or
        #: to the scope's state where one is set, so nothing that touches them by
        #: name has to know which it got.
        self._own_state = _TrackerState()

    @property
    def _state(self) -> _TrackerState:
        scoped = _TRACKER_SCOPE.get()
        return scoped if scoped is not None else self._own_state

    @property
    def _reads(self) -> dict[str, datetime]:
        return self._state.reads

    @property
    def _versions(self) -> dict[str, int]:
        return self._state.versions

    @property
    def _entries(self) -> dict[str, dict]:
        return self._state.entries

    @property
    def _contexts(self) -> list[ExternalContext]:
        return self._state.contexts

    @property
    def _queries(self) -> list[dict]:
        return self._state.queries

    @property
    def _errors(self) -> list[dict]:
        return self._state.errors

    @property
    def _writes(self) -> list[dict]:
        return self._state.writes

    @property
    def _actions(self) -> list[dict]:
        return self._state.actions

    @property
    def _session_started_at(self) -> datetime:
        return self._state.session_started_at

    @_session_started_at.setter
    def _session_started_at(self, when: datetime) -> None:
        # Assigned by ``clear()`` and by the Pro span recorder, which backdates the
        # window to the first tool call. Writable for that reason.
        self._state.session_started_at = when

    def record(self, entry: MemoryEntry) -> None:
        """Record that an entry was read during this session."""
        self._reads[entry.entry_key] = datetime.now(timezone.utc)
        self._versions[entry.entry_key] = entry.version
        self._entries[entry.entry_key] = {
            "value": entry.value,
            "confidence": entry.confidence,
            "version": entry.version,
            "memory_type": entry.memory_type.value if hasattr(entry.memory_type, 'value') else str(entry.memory_type),
            "written_by": entry.provenance.agent_id,
        }

    def record_surfaced(
        self,
        entity_path: str,
        key: str,
        *,
        version: int,
        value: str,
        confidence: float,
        memory_type: str | None = None,
        written_by: str | None = None,
    ) -> None:
        """Record a read of an entry that arrived already-materialised.

        ``record`` needs a ``MemoryEntry``, which a briefing does not return: it
        returns digests whose ``hot_context`` carries the entry's text and
        confidence but is not the entry object. Re-reading to get one would
        itself book a recall, and the lookup that reports a read must not be one
        — the distinction amfs#257 was opened over.

        Same fields as ``record`` writes, so a briefing-sourced causal entry is
        indistinguishable downstream from a directly-read one. Confidence is
        snapshotted here rather than resolved at commit time, because what a
        later reader needs to know is how sure the entry looked *when it was
        acted on*.
        """
        ek = f"{entity_path}/{key}"
        self._reads[ek] = datetime.now(timezone.utc)
        self._versions[ek] = version
        self._entries[ek] = {
            "value": value,
            "confidence": confidence,
            "version": version,
            "memory_type": memory_type or "fact",
            "written_by": written_by,
        }

    def record_context(
        self,
        label: str,
        summary: str,
        *,
        source: str | None = None,
    ) -> None:
        """Record external context that influenced decisions in this session.

        Unlike ``record()``, this doesn't correspond to an AMFS entry — it
        captures external tool calls, API responses, and other inputs that
        informed the agent's decisions.  These are included in the causal
        chain returned by ``explain()`` so decision traces are complete.
        """
        self._contexts.append({
            "label": label,
            "summary": summary,
            "source": source,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        })

    def record_action(
        self,
        tool_name: str,
        arguments: dict | None = None,
        *,
        result: str = "",
        source: str | None = None,
        duration_ms: int = 0,
        success: bool = True,
    ) -> None:
        """Record an action the agent took during this session.

        Distinct from ``record_context``, which captures what the agent *learned*
        from an external system. This captures what it *did* — the half of a
        decision that supervised training predicts. Both end up in the trace.

        The result is hashed in full and stored only as a prefix: a trace is a
        record of the decision, not a cache of everybody's API responses.
        """
        result_hash = hashlib.sha256(result.encode("utf-8")).hexdigest() if result else ""
        summary = result[:_MAX_ACTION_RESULT_CHARS]
        if len(result) > _MAX_ACTION_RESULT_CHARS:
            summary += "... (truncated)"
        self._actions.append({
            "tool_name": tool_name,
            "arguments": arguments or {},
            "result_summary": summary,
            "result_hash": result_hash,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "duration_ms": duration_ms,
            "source": source,
            "success": success,
        })

    @property
    def actions(self) -> list[dict]:
        """All actions recorded in this session, in the order they were taken."""
        return list(self._actions)

    @property
    def causal_keys(self) -> list[str]:
        """All entry keys read in this session, ordered by read time."""
        return [k for k, _ in sorted(self._reads.items(), key=lambda x: x[1])]

    @property
    def external_contexts(self) -> list[ExternalContext]:
        """All external contexts recorded in this session, in order."""
        return list(self._contexts)

    def record_error(
        self,
        operation: str,
        error_type: str,
        message: str,
        stack_trace: str | None = None,
    ) -> None:
        """Record an error that occurred during this session."""
        self._errors.append({
            "operation": operation,
            "error_type": error_type,
            "message": message,
            "stack_trace": stack_trace,
            "occurred_at": datetime.now(timezone.utc).isoformat(),
        })

    @property
    def error_events(self) -> list[dict]:
        return list(self._errors)

    def record_write(self, entity_path: str, key: str, version: int, is_new: bool) -> None:
        """Record a write operation for state diff computation."""
        self._writes.append({
            "entity_path": entity_path,
            "key": key,
            "version": version,
            "is_new": is_new,
        })

    @property
    def write_events(self) -> list[dict]:
        return list(self._writes)

    @property
    def read_count(self) -> int:
        return len(self._reads)

    def read_version(self, entry_key: str) -> int | None:
        """Return the version we last read for an entry, or None if never read."""
        return self._versions.get(entry_key)

    def record_query(
        self,
        operation: str,
        parameters: dict,
        result_count: int,
        duration_ms: float | None = None,
    ) -> None:
        """Record a search or list operation."""
        self._queries.append({
            "operation": operation,
            "parameters": parameters,
            "result_count": result_count,
            "duration_ms": duration_ms,
            "occurred_at": datetime.now(timezone.utc).isoformat(),
        })

    @property
    def query_events(self) -> list[dict]:
        """All query events recorded in this session."""
        return list(self._queries)

    @property
    def session_started_at(self) -> datetime:
        return self._session_started_at

    def entry_snapshot(self, entry_key: str) -> dict | None:
        """Return the snapshot captured at read time for an entry."""
        return self._entries.get(entry_key)

    def clear(self) -> None:
        """Reset the read log (e.g. between sub-tasks within a session)."""
        self._reads.clear()
        self._versions.clear()
        self._entries.clear()
        self._contexts.clear()
        self._queries.clear()
        self._errors.clear()
        self._writes.clear()
        self._actions.clear()
        # The window this tracker describes restarts here, so a trace committed
        # after a clear reports the duration of its own work rather than the
        # lifetime of the process.
        self._session_started_at = datetime.now(timezone.utc)

    def contains(self, entry_key: str) -> bool:
        return entry_key in self._reads


class CoWEngine:
    """Copy-on-Write engine that reads existing entries, increments versions,
    and delegates writes to the underlying adapter.

    Parameters
    ----------
    adapter:
        The storage adapter to use.
    tagger:
        A CausalTagger for stamping provenance.
    read_tracker:
        Optional ReadTracker for auto-causal linking. If provided, every
        successful read is logged for later use in commit_outcome().
    """

    def __init__(
        self,
        adapter: AdapterABC,
        tagger: CausalTagger,
        read_tracker: ReadTracker | None = None,
    ) -> None:
        self._adapter = adapter
        self._tagger = tagger
        self._read_tracker = read_tracker

    @property
    def adapter(self) -> AdapterABC:
        return self._adapter

    @property
    def tagger(self) -> CausalTagger:
        return self._tagger

    @property
    def read_tracker(self) -> ReadTracker | None:
        return self._read_tracker

    def read(
        self,
        entity_path: str,
        key: str,
        *,
        min_confidence: float = 0.0,
        branch: str = "main",
    ) -> MemoryEntry | None:
        """Read the current version of a key from the adapter.

        If a ReadTracker is attached, the read is automatically logged
        for causal linking.
        """
        entry = self._adapter.read(entity_path, key, min_confidence=min_confidence, branch=branch)
        if entry is not None:
            self._adapter.increment_recall_count(entity_path, key, branch=branch)
            if self._read_tracker is not None:
                self._read_tracker.record(entry)
        return entry

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
        artifact_refs: list[ArtifactRef] | None = None,
        shared: bool = True,
        branch: str = "main",
        embedding: list[float] | None = None,
        importance_score: float | None = None,
        importance_dimensions: dict[str, float] | None = None,
    ) -> MemoryEntry:
        """Write a new version of a key with CoW semantics.

        - Reads the current version (if any) to determine the next version number.
        - Stamps provenance via the CausalTagger.
        - Delegates the actual write to the adapter.
        """
        current = self._adapter.read(entity_path, key, branch=branch)
        next_version = (current.version + 1) if current else 1

        value_hash = content_hash(value)
        previous_chain = current.integrity_chain if current else None
        chain = integrity_chain_hash(value_hash, previous_chain)

        entry = MemoryEntry(
            entity_path=entity_path,
            key=key,
            version=next_version,
            value=value,
            provenance=self._tagger.tag(pattern_refs=pattern_refs),
            confidence=confidence,
            outcome_count=current.outcome_count if current else 0,
            recall_count=current.recall_count if current else 0,
            importance_score=importance_score,
            importance_dimensions=importance_dimensions,
            ttl_at=ttl_at,
            artifact_refs=artifact_refs or [],
            memory_type=memory_type,
            shared=shared,
            branch=branch,
            embedding=embedding,
            content_hash=value_hash,
            integrity_chain=chain,
        )

        return self._adapter.write(entry)

    def list(
        self,
        entity_path: str | None = None,
        *,
        include_superseded: bool = False,
        branch: str = "main",
    ) -> list[MemoryEntry]:
        """List entries from the adapter."""
        return self._adapter.list(entity_path, include_superseded=include_superseded, branch=branch)

    def history(
        self,
        entity_path: str,
        key: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        branch: str = "main",
    ) -> list[MemoryEntry]:
        """Return all versions of a key ordered by version ascending.

        Enables temporal queries like "how did this memory change over time?"
        Optionally filter to a time window using *since* and *until*.
        """
        all_versions = self._adapter.list(entity_path, include_superseded=True, branch=branch)
        versions = [e for e in all_versions if e.key == key]
        versions.sort(key=lambda e: e.version)

        if since is not None:
            versions = [e for e in versions if e.provenance.written_at >= since]
        if until is not None:
            versions = [e for e in versions if e.provenance.written_at <= until]

        return versions
