"""CoWEngine, CausalTagger, and ReadTracker — core write logic for AMFS."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

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
        self._reads: dict[str, datetime] = {}
        self._versions: dict[str, int] = {}
        self._entries: dict[str, dict] = {}
        self._contexts: list[ExternalContext] = []
        self._queries: list[dict] = []
        self._session_started_at: datetime = datetime.now(timezone.utc)
        self._errors: list[dict] = []
        self._writes: list[dict] = []

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
